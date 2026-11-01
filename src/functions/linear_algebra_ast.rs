//! AST-native linear algebra functions.
//!
//! Dot, Det, Inverse, Tr, IdentityMatrix, DiagonalMatrix, Cross, Eigenvalues, Fit.

use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::{
  expr_to_rational, gcd_i128, make_sqrt, rat_reduce, try_eval_to_f64,
};

/// Transpose a matrix (Vec<Vec<Expr>>)
fn transpose_matrix(m: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
  if m.is_empty() {
    return vec![];
  }
  let nrows = m.len();
  let ncols = m[0].len();
  let mut result = vec![vec![Expr::Integer(0); nrows]; ncols];
  for i in 0..nrows {
    for j in 0..ncols {
      result[j][i] = m[i][j].clone();
    }
  }
  result
}

/// Multiply two matrices
fn mat_mul(a: &[Vec<Expr>], b: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
  let arows = a.len();
  let acols = if arows > 0 { a[0].len() } else { 0 };
  let bcols = if b.is_empty() { 0 } else { b[0].len() };
  let mut result = vec![vec![Expr::Integer(0); bcols]; arows];
  for i in 0..arows {
    for j in 0..bcols {
      let mut sum = Expr::Integer(0);
      for k in 0..acols {
        sum = eval_add(&sum, &eval_mul(&a[i][k], &b[k][j]));
      }
      result[i][j] = sum;
    }
  }
  result
}

/// Try to invert a square matrix. Returns None if singular.
fn try_invert(matrix: &[Vec<Expr>]) -> Option<Vec<Vec<Expr>>> {
  // try_invert is a probe — a singular matrix is an expected, handled outcome
  // (PseudoInverse falls through to another formula). Detect singularity up
  // front via the determinant so we never call inverse_ast on a singular
  // matrix (which would emit and capture an Inverse::sing message).
  if is_zero_expr(&determinant(matrix)) {
    return None;
  }
  let expr = matrix_to_expr(matrix.to_vec());
  match inverse_ast(&[expr]) {
    Ok(result) => {
      let inv = expr_to_matrix(&result)?;
      // `try_invert` is only used on the numeric M^T·M / M·M^T products, so a
      // genuine inverse has finite numeric entries. A singular matrix instead
      // yields division-by-zero artifacts (`9./0.`, Indeterminate,
      // ComplexInfinity, …); treat any non-finite-numeric entry as
      // non-invertible so PseudoInverse falls through to the other formula.
      let is_finite_number = |e: &Expr| match e {
        Expr::Integer(_) | Expr::BigInteger(_) => true,
        Expr::Real(x) => x.is_finite(),
        Expr::FunctionCall { name, .. } => name == "Rational",
        _ => false,
      };
      if inv.iter().flatten().all(is_finite_number) {
        Some(inv)
      } else {
        None
      }
    }
    Err(_) => None,
  }
}

/// PseudoInverse[matrix] - Moore-Penrose pseudoinverse
pub fn pseudo_inverse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PseudoInverse expects exactly 1 argument".into(),
    ));
  }
  let Some(matrix) = expr_to_matrix(&args[0]) else {
    return Ok(unevaluated("PseudoInverse", args));
  };

  let nrows = matrix.len();
  if nrows == 0 {
    return Ok(Expr::List(vec![].into()));
  }
  let ncols = matrix[0].len();

  // Check if zero matrix
  let is_zero = matrix.iter().all(|row| {
    row.iter().all(|e| {
      matches!(e, Expr::Integer(0))
        || matches!(e, Expr::Real(x) if *x == 0.0)
        || matches!(e, Expr::BigInteger(n) if *n == BigInt::from(0))
    })
  });
  if is_zero {
    let result = vec![vec![Expr::Integer(0); nrows]; ncols];
    return Ok(matrix_to_expr(result));
  }

  // A machine-real matrix: from its singular value decomposition, keeping
  // the singular values above wolframscript's rank tolerance of
  // 100 $MachineEpsilon relative to the largest one.
  if let Some(a) = machine_real_matrix(&matrix)
    && let Some((u, sigma, v)) = jacobi_svd(&a)
  {
    let rank = nrows.min(ncols);
    let largest = (0..rank).map(|i| sigma[i][i].abs()).fold(0.0, f64::max);
    let cutoff = 100.0 * f64::EPSILON * largest;
    // A⁺ = V . Σ⁺ . U'
    let result: Vec<Vec<Expr>> = (0..ncols)
      .map(|i| {
        (0..nrows)
          .map(|j| {
            let mut sum = 0.0;
            for k in 0..rank {
              let s = sigma[k][k];
              if s > cutoff {
                sum += v[i][k] * u[j][k] / s;
              }
            }
            Expr::Real(sum)
          })
          .collect()
      })
      .collect();
    return Ok(matrix_to_expr(result));
  }

  // Square non-singular: just use the exact inverse. (try_invert returns None
  // for a singular matrix, so we fall through to the rank-deficient path
  // instead of emitting Inverse::sing.)
  if nrows == ncols
    && let Some(inv) = try_invert(&matrix)
  {
    return Ok(matrix_to_expr(inv));
  }

  // Complex matrices need the conjugate transpose throughout — with the
  // plain transpose the Gram products degenerate (C.Transpose[C] of the
  // row (1, I) is 1 + I² = 0) and rank-deficient complex matrices stayed
  // unevaluated.
  let mt = adjoint_matrix(&matrix);

  // Two one-sided formulas, each needing one matrix inverse:
  //   left:  (M^H M)^{-1} M^H   — invert the ncols×ncols product (full column rank)
  //   right: M^H (M M^H)^{-1}   — invert the nrows×nrows product (full row rank)
  // Invert whichever product is smaller first: for a wide matrix (ncols >
  // nrows) M^H M is the large, rank-deficient one, so inverting it is both
  // wrong and potentially very slow — prefer the right formula there, and
  // vice-versa for a tall matrix.
  let left = || {
    let mtm = mat_mul(&mt, &matrix);
    try_invert(&mtm).map(|mtm_inv| matrix_to_expr(mat_mul(&mtm_inv, &mt)))
  };
  let right = || {
    let mmt = mat_mul(&matrix, &mt);
    try_invert(&mmt).map(|mmt_inv| matrix_to_expr(mat_mul(&mt, &mmt_inv)))
  };
  let (first, second): (&dyn Fn() -> Option<Expr>, &dyn Fn() -> Option<Expr>) =
    if ncols <= nrows {
      (&left, &right)
    } else {
      (&right, &left)
    };
  if let Some(result) = first().or_else(second) {
    return Ok(result);
  }

  // Rank-deficient matrices: compute the Moore-Penrose pseudoinverse exactly
  // via a rank factorization A = B·C (B = pivot columns of A, C = the non-zero
  // RREF rows), giving A+ = Cᵀ (C Cᵀ)⁻¹ (Bᵀ B)⁻¹ Bᵀ.
  if let Some(result) = rank_factor_pseudoinverse(&matrix) {
    return Ok(result);
  }

  // Otherwise (e.g. symbolic entries we can't reduce), return unevaluated.
  Ok(unevaluated("PseudoInverse", args))
}

/// Moore-Penrose pseudoinverse of a rank-deficient matrix via rank
/// factorization. Returns None when the matrix can't be reduced to exact
/// pivots (e.g. symbolic entries).
fn rank_factor_pseudoinverse(matrix: &[Vec<Expr>]) -> Option<Expr> {
  let nrows = matrix.len();
  let rref = row_reduce_impl(matrix);
  // Pivot columns are the leading column of each non-zero RREF row; those rows
  // form C (full row rank), the corresponding columns of A form B (full column
  // rank).
  let mut pivot_cols: Vec<usize> = Vec::new();
  let mut c: Vec<Vec<Expr>> = Vec::new();
  for row in &rref {
    if let Some(lead) = row.iter().position(|e| !is_zero_expr(e)) {
      pivot_cols.push(lead);
      c.push(row.clone());
    }
  }
  if pivot_cols.is_empty() {
    return None;
  }
  let b: Vec<Vec<Expr>> = (0..nrows)
    .map(|i| {
      pivot_cols
        .iter()
        .map(|&col| matrix[i][col].clone())
        .collect()
    })
    .collect();
  let bt = adjoint_matrix(&b);
  let ct = adjoint_matrix(&c);
  let btb_inv = try_invert(&mat_mul(&bt, &b))?;
  let cct_inv = try_invert(&mat_mul(&c, &ct))?;
  // A+ = Cᴴ · (C Cᴴ)⁻¹ · (Bᴴ B)⁻¹ · Bᴴ
  let step1 = mat_mul(&ct, &cct_inv);
  let step2 = mat_mul(&step1, &btb_inv);
  Some(matrix_to_expr(mat_mul(&step2, &bt)))
}

/// Conjugate transpose for complex matrices; plain transpose otherwise
/// (so real and symbolic entries take the unchanged path).
fn adjoint_matrix(matrix: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
  let transposed = transpose_matrix(matrix);
  if !matrix
    .iter()
    .any(|row| row.iter().any(contains_imaginary_unit))
  {
    return transposed;
  }
  transposed
    .iter()
    .map(|row| {
      row
        .iter()
        .map(|e| {
          evaluate_expr_to_expr(&call1("Conjugate", e.clone()))
            .unwrap_or_else(|_| e.clone())
        })
        .collect()
    })
    .collect()
}

/// Orthogonalize[{v1, v2, ...}] — orthonormalize a set of vectors with the
/// (default) normalized Gram-Schmidt process and the standard dot-product
/// inner product. Linearly dependent vectors collapse to a zero vector in
/// their position, matching wolframscript. Coefficients are kept exact, so
/// the result may contain radicals (e.g. 1/Sqrt[2]).
pub fn orthogonalize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("Orthogonalize", args));
  // The standard inner product, with an optional `Method -> …`.
  let (vectors, opts) = match args {
    [v] => (v, &[][..]),
    [v, rest @ ..] if rest.iter().all(|o| matches!(o, Expr::Rule { .. })) => {
      (v, rest)
    }
    _ => return unevaluated(),
  };
  let mut method = "GramSchmidt".to_string();
  let mut method_given = false;
  for o in opts {
    if let Expr::Rule {
      pattern,
      replacement,
    } = o
      && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Method")
    {
      let name = match replacement.as_ref() {
        Expr::String(m) => m.clone(),
        Expr::Identifier(m) if m == "Automatic" => continue,
        other => crate::syntax::expr_to_string(other),
      };
      if !matches!(
        name.as_str(),
        "GramSchmidt"
          | "ModifiedGramSchmidt"
          | "Reorthogonalization"
          | "Householder"
      ) {
        crate::emit_message(&format!(
          "Orthogonalize::orbop: {name} is not a valid value for the Method option."
        ));
        return unevaluated();
      }
      method = name;
      method_given = true;
    }
  }
  let Some(matrix) = expr_to_matrix(vectors) else {
    return unevaluated();
  };
  // Machine vectors are orthogonalized numerically; Reorthogonalization is
  // then the default.
  let inexact = matrix
    .iter()
    .flatten()
    .any(crate::functions::math_ast::contains_inexact_real);
  if inexact
    && let Some(rows) = matrix
      .iter()
      .map(|r| {
        r.iter()
          .map(crate::functions::math_ast::try_eval_to_f64)
          .collect::<Option<Vec<f64>>>()
      })
      .collect::<Option<Vec<_>>>()
  {
    let method = if method_given {
      method.as_str()
    } else {
      "Reorthogonalization"
    };
    let q = machine_orthogonalize(&rows, method);
    return Ok(Expr::List(
      q.into_iter()
        .map(|r| Expr::List(r.into_iter().map(Expr::Real).collect()))
        .collect(),
    ));
  }
  if method != "GramSchmidt" && !inexact {
    crate::emit_message(&format!(
      "Orthogonalize::ornex: The option Method -> {method} cannot be used with exact precision vectors. It is replaced with Method -> GramSchmidt."
    ));
  }

  let simplify = |e: Expr| evaluate_expr_to_expr(&e).unwrap_or(e);
  // Normalized components can accumulate nested radicals (e.g. combinations of
  // 1/Sqrt[6] and Sqrt[2/3]) that plain evaluation leaves un-reduced; run the
  // Simplify builtin so they collapse to canonical radicals.
  let simplify_full = |e: Expr| -> Expr {
    let s = call1("Simplify", e);
    evaluate_expr_to_expr(&s).unwrap_or_else(|_| Expr::Integer(0))
  };
  let is_rational = |e: &Expr| {
    matches!(e, Expr::Integer(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };

  // Complex vectors use the Hermitian inner product Conjugate[e].v for
  // the projections and |w|² for the norm — the plain dot product made
  // Orthogonalize[{{1, I}, {1, -I}}] collapse to zero vectors (the
  // "norm" of (1, I) came out 0). Real/symbolic input keeps the plain
  // product.
  let complex = matrix
    .iter()
    .any(|row| row.iter().any(contains_imaginary_unit));
  let conj = |e: &Expr| -> Expr {
    if !complex {
      return e.clone();
    }
    evaluate_expr_to_expr(&call1("Conjugate", e.clone()))
      .unwrap_or_else(|_| e.clone())
  };

  let mut result: Vec<Vec<Expr>> = Vec::new();
  for v in &matrix {
    // Subtract the projections onto the previously produced orthonormal
    // vectors: w = v - sum_e <e, v> e with <e, v> = Conjugate[e].v.
    let mut w = v.clone();
    for e in &result {
      let mut dot = Expr::Integer(0);
      for i in 0..v.len() {
        dot = eval_add(&dot, &eval_mul(&conj(&e[i]), &v[i]));
      }
      let dot = simplify(dot);
      if matches!(dot, Expr::Integer(0)) {
        continue;
      }
      for i in 0..w.len() {
        w[i] = simplify(eval_sub(&w[i], &eval_mul(&dot, &e[i])));
      }
    }

    // normsq = <w, w>
    let mut normsq = Expr::Integer(0);
    for wi in &w {
      normsq = eval_add(&normsq, &eval_mul(&conj(wi), wi));
    }
    let normsq = simplify(normsq);

    if matches!(normsq, Expr::Integer(0)) {
      // Linearly dependent (or zero) vector → zero vector of the same length.
      result.push(vec![Expr::Integer(0); v.len()]);
    } else {
      // When the squared norm is rational, plain evaluation already yields the
      // canonical radical form (e.g. -(1/Sqrt[2])); only fall back to Simplify
      // for an irrational norm, where components carry nested radicals.
      let norm_rational = is_rational(&normsq);
      // e = w / Sqrt[normsq] = w * normsq^(-1/2).
      let inv_norm =
        pow(normsq, crate::functions::math_ast::make_rational(-1, 2));
      let e: Vec<Expr> = w
        .iter()
        .map(|wi| {
          let prod = eval_mul(wi, &inv_norm);
          if norm_rational {
            simplify(prod)
          } else {
            simplify_full(prod)
          }
        })
        .collect();
      result.push(e);
    }
  }

  Ok(matrix_to_expr(result))
}

/// Orthonormalize machine vectors by `method`: classical Gram–Schmidt
/// (`GramSchmidt`), the modified form, classical with a second projection
/// pass (`Reorthogonalization`), or Householder QR. A dependent vector
/// gives a zero vector.
fn machine_orthogonalize(rows: &[Vec<f64>], method: &str) -> Vec<Vec<f64>> {
  let dot =
    |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
  let n = rows.first().map_or(0, Vec::len);
  if method == "Householder" {
    // QR of the matrix whose columns are the vectors; the rows of the
    // result are Q's first columns.
    let m = rows.len();
    let mut a: Vec<Vec<f64>> = (0..n)
      .map(|i| rows.iter().map(|r| r[i]).collect())
      .collect();
    let mut reflectors: Vec<Vec<f64>> = Vec::new();
    for k in 0..m.min(n) {
      let x: Vec<f64> = (k..n).map(|i| a[i][k]).collect();
      let norm = dot(&x, &x).sqrt();
      let mut v = x.clone();
      // A one-element column is left alone (LAPACK's `dlarfg` sets
      // tau = 0 there), and so is a zero one.
      if norm == 0.0 || x.len() == 1 {
        reflectors.push(vec![0.0; n - k]);
        continue;
      }
      v[0] += if x[0] >= 0.0 { norm } else { -norm };
      let vv = dot(&v, &v);
      for j in k..m {
        let col: Vec<f64> = (k..n).map(|i| a[i][j]).collect();
        let f = 2.0 * dot(&v, &col) / vv;
        for i in k..n {
          a[i][j] -= f * v[i - k];
        }
      }
      reflectors.push(v);
    }
    let mut out = Vec::with_capacity(m);
    for j in 0..m {
      let mut e = vec![0.0; n];
      if j < n {
        e[j] = 1.0;
      }
      for (k, v) in reflectors.iter().enumerate().rev() {
        let vv = dot(v, v);
        if vv == 0.0 {
          continue;
        }
        let f = 2.0 * dot(v, &e[k..]) / vv;
        for i in k..n {
          e[i] -= f * v[i - k];
        }
      }
      out.push(e);
    }
    return out;
  }
  let passes = if method == "Reorthogonalization" {
    2
  } else {
    1
  };
  let mut q: Vec<Vec<f64>> = Vec::with_capacity(rows.len());
  for v in rows {
    let original = dot(v, v).sqrt();
    let mut w = v.clone();
    for _ in 0..passes {
      if method == "ModifiedGramSchmidt" {
        for e in &q {
          let c = dot(e, &w);
          for (wi, ei) in w.iter_mut().zip(e) {
            *wi -= c * ei;
          }
        }
      } else {
        let coeffs: Vec<f64> = q.iter().map(|e| dot(e, &w)).collect();
        for (c, e) in coeffs.iter().zip(&q) {
          for (wi, ei) in w.iter_mut().zip(e) {
            *wi -= c * ei;
          }
        }
      }
    }
    let norm = dot(&w, &w).sqrt();
    if norm <= 1e-12 * original.max(f64::MIN_POSITIVE) {
      q.push(vec![0.0; n]);
    } else {
      let inv = 1.0 / norm;
      q.push(w.iter().map(|x| x * inv).collect());
    }
  }
  q
}

/// Helper: extract a matrix (list of lists) from an Expr.
/// Returns None if it's not a valid matrix.
pub fn expr_to_matrix(expr: &Expr) -> Option<Vec<Vec<Expr>>> {
  if let Expr::List(rows) = expr {
    let mut matrix = Vec::new();
    let ncols = if let Expr::List(first) = rows.first()? {
      first.len()
    } else {
      return None;
    };
    for row in rows {
      if let Expr::List(cols) = row {
        if cols.len() != ncols {
          return None;
        }
        matrix.push(cols.to_vec());
      } else {
        return None;
      }
    }
    Some(matrix)
  } else {
    None
  }
}

/// Helper: extract a vector (flat list) from an Expr
fn expr_to_vector(expr: &Expr) -> Option<Vec<Expr>> {
  if let Expr::List(items) = expr {
    // Make sure it's not a matrix (list of lists)
    if items.iter().any(|i| matches!(i, Expr::List(_))) {
      return None;
    }
    Some(items.to_vec())
  } else {
    None
  }
}

/// Helper: convert matrix back to Expr
fn matrix_to_expr(matrix: Vec<Vec<Expr>>) -> Expr {
  Expr::List(matrix.into_iter().map(|v| Expr::List(v.into())).collect())
}

/// Helper: evaluate a simple arithmetic expression on Exprs
fn eval_add(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) => match x.checked_add(*y) {
      Some(s) => Expr::Integer(s),
      None => fallback_plus(a, b),
    },
    (Expr::Integer(x), Expr::Real(y)) | (Expr::Real(y), Expr::Integer(x)) => {
      Expr::Real(*x as f64 + y)
    }
    (Expr::Real(x), Expr::Real(y)) => Expr::Real(x + y),
    _ => fallback_plus(a, b),
  }
}

fn fallback_plus(a: &Expr, b: &Expr) -> Expr {
  match crate::functions::math_ast::plus_ast(&[a.clone(), b.clone()]) {
    Ok(r) => r,
    Err(_) => plus2(a.clone(), b.clone()),
  }
}

/// Sum every term in one `plus_ast` call instead of folding pairwise.
///
/// `plus_ast` flattens and re-collects *all* of its arguments on every
/// call, so folding `n` terms with `n - 1` two-argument calls (`sum =
/// eval_add(sum, term)`) re-flattens the ever-growing `sum` each time —
/// O(1) + O(2) + … + O(n) = O(n²) work to add up `n` terms. A dot product
/// of two length-`n` vectors needs exactly this sum, so that quadratic
/// blowup turned every `Dot` call in a Newton-solved PDE discretization
/// (each equation a dot product of a matrix row with hundreds of unknowns)
/// into the dominant cost of the whole solve. Passing every term to
/// `plus_ast` at once flattens and combines them in a single pass.
fn eval_add_many(terms: Vec<Expr>) -> Expr {
  match terms.len() {
    0 => Expr::Integer(0),
    1 => terms.into_iter().next().unwrap(),
    _ => match crate::functions::math_ast::plus_ast(&terms) {
      Ok(r) => r,
      Err(_) => plus(terms),
    },
  }
}

fn eval_mul(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) => match x.checked_mul(*y) {
      Some(p) => Expr::Integer(p),
      None => {
        match crate::functions::math_ast::times_ast(&[a.clone(), b.clone()]) {
          Ok(r) => r,
          Err(_) => times2(a.clone(), b.clone()),
        }
      }
    },
    (Expr::Integer(x), Expr::Real(y)) | (Expr::Real(y), Expr::Integer(x)) => {
      Expr::Real(*x as f64 * y)
    }
    (Expr::Real(x), Expr::Real(y)) => Expr::Real(x * y),
    _ => match crate::functions::math_ast::times_ast(&[a.clone(), b.clone()]) {
      Ok(r) => r,
      Err(_) => times2(a.clone(), b.clone()),
    },
  }
}

fn eval_sub(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) => match x.checked_sub(*y) {
      Some(s) => Expr::Integer(s),
      None => fallback_sub(a, b),
    },
    (Expr::Integer(x), Expr::Real(y)) => Expr::Real(*x as f64 - y),
    (Expr::Real(x), Expr::Integer(y)) => Expr::Real(x - *y as f64),
    (Expr::Real(x), Expr::Real(y)) => Expr::Real(x - y),
    _ => fallback_sub(a, b),
  }
}

fn fallback_sub(a: &Expr, b: &Expr) -> Expr {
  match crate::functions::math_ast::subtract_ast(&[a.clone(), b.clone()]) {
    Ok(r) => r,
    Err(_) => minus2(a.clone(), b.clone()),
  }
}

/// Dot[a, b] - dot product (vector) or matrix multiply
/// Expand a structured-matrix wrapper produced by LUDecomposition (and other
/// structured-matrix functions) to its dense nested-list form:
///   LowerTriangularMatrix[StructuredArray`StructuredData[{n, n}, rows]]
///   UpperTriangularMatrix[StructuredArray`StructuredData[{n, n}, rows]]
///   PermutationMatrix[StructuredArray`StructuredData[{n, n},
///                                                    {Cycles[…], _}]]
/// Returns None for anything else.
pub fn structured_matrix_to_dense(e: &Expr) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  if !matches!(
    name.as_str(),
    "LowerTriangularMatrix"
      | "UpperTriangularMatrix"
      | "PermutationMatrix"
      | "BlockDiagonalMatrix"
      | "CauchyMatrix"
  ) || args.len() != 1
  {
    return None;
  }
  let Expr::FunctionCall {
    name: sd_name,
    args: sd_args,
  } = &args[0]
  else {
    return None;
  };
  if sd_name != "StructuredArray`StructuredData" || sd_args.len() != 2 {
    return None;
  }
  let Expr::List(dims) = &sd_args[0] else {
    return None;
  };
  if name == "BlockDiagonalMatrix" {
    // Payload is {blockList, rowIndices, colIndices}; place each square block
    // along the diagonal of an n×n zero matrix.
    let n = match &dims[..] {
      [Expr::Integer(r), Expr::Integer(_c)] => usize::try_from(*r).ok()?,
      _ => return None,
    };
    let Expr::List(payload) = &sd_args[1] else {
      return None;
    };
    let Some(Expr::List(blocks)) = payload.first() else {
      return None;
    };
    let mut dense: Vec<Vec<Expr>> = vec![vec![Expr::Integer(0); n]; n];
    let mut offset = 0usize;
    for block in blocks {
      let Expr::List(rows) = block else {
        return None;
      };
      let bn = rows.len();
      for (i, row) in rows.iter().enumerate() {
        let Expr::List(cols) = row else {
          return None;
        };
        for (j, val) in cols.iter().enumerate() {
          dense[offset + i][offset + j] = val.clone();
        }
      }
      offset += bn;
    }
    return Some(Expr::List(
      dense
        .into_iter()
        .map(|r| Expr::List(r.into()))
        .collect::<Vec<_>>()
        .into(),
    ));
  }
  if name == "CauchyMatrix" {
    // Payload is {x, y}; entry (i, j) is 1/(x_i + y_j).
    let Expr::List(payload) = &sd_args[1] else {
      return None;
    };
    let [Expr::List(x), Expr::List(y)] = &payload[..] else {
      return None;
    };
    return cauchy_dense(x, y).ok();
  }
  if name == "PermutationMatrix" {
    // Payload is {Cycles[{{…}, …}], workingLength}; row i of the dense form
    // has its 1 in column π(i), where π is the cycle product.
    let n = match &dims[..] {
      [Expr::Integer(n), _] => usize::try_from(*n).ok()?,
      _ => return None,
    };
    let Expr::List(payload) = &sd_args[1] else {
      return None;
    };
    let Some(Expr::FunctionCall {
      name: cyc_name,
      args: cyc_args,
    }) = payload.first()
    else {
      return None;
    };
    if cyc_name != "Cycles" || cyc_args.len() != 1 {
      return None;
    }
    let Expr::List(cycles) = &cyc_args[0] else {
      return None;
    };
    let mut perm: Vec<usize> = (0..n).collect();
    for cycle in cycles {
      let Expr::List(members) = cycle else {
        return None;
      };
      let idx: Vec<usize> = members
        .iter()
        .map(|m| match m {
          Expr::Integer(i) if (1..=n as i128).contains(i) => {
            Some(*i as usize - 1)
          }
          _ => None,
        })
        .collect::<Option<_>>()?;
      for w in 0..idx.len() {
        perm[idx[w]] = idx[(w + 1) % idx.len()];
      }
    }
    let rows: Vec<Expr> = perm
      .iter()
      .map(|&j| {
        Expr::List(
          (0..n)
            .map(|c| Expr::Integer(i128::from(c == j)))
            .collect::<Vec<_>>()
            .into(),
        )
      })
      .collect();
    return Some(Expr::List(rows.into()));
  }
  // Lower/UpperTriangularMatrix store the dense rows directly.
  match &sd_args[1] {
    rows @ Expr::List(_) => Some(rows.clone()),
    _ => None,
  }
}

/// BlockDiagonalMatrix[{d1, d2, …}] assembles square matrices d1, d2, … as the
/// diagonal blocks of a larger matrix (zeros off-diagonal), represented as a
/// structured array
///   BlockDiagonalMatrix[StructuredArray`StructuredData[{n, n},
///                       {{d1, d2, …}, Range[n], Range[n]}]]
/// A single block returns that block directly; wolframscript leaves the empty
/// list, non-square blocks, or an already-structured argument unevaluated.
pub fn block_diagonal_matrix_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unchanged = || Ok(unevaluated("BlockDiagonalMatrix", args));

  // Already in the canonical StructuredData form — keep it as-is.
  if let Expr::FunctionCall { name, .. } = &args[0]
    && name == "StructuredArray`StructuredData"
  {
    return unchanged();
  }

  let Expr::List(blocks) = &args[0] else {
    return unchanged();
  };
  if blocks.is_empty() {
    return unchanged();
  }

  // Every block must be a square matrix (a list of equal-length rows whose
  // length equals the row count).
  let mut sizes: Vec<usize> = Vec::with_capacity(blocks.len());
  for b in blocks {
    let Expr::List(rows) = b else {
      return unchanged();
    };
    let nr = rows.len();
    let square = nr > 0
      && rows
        .iter()
        .all(|r| matches!(r, Expr::List(cols) if cols.len() == nr));
    if !square {
      return unchanged();
    }
    sizes.push(nr);
  }

  // A single block is returned as the plain matrix.
  if blocks.len() == 1 {
    return Ok(blocks[0].clone());
  }

  let n: usize = sizes.iter().sum();
  let range_n = Expr::List(
    (1..=n as i128)
      .map(Expr::Integer)
      .collect::<Vec<_>>()
      .into(),
  );
  let payload = Expr::List(
    vec![
      Expr::List(blocks.iter().cloned().collect::<Vec<_>>().into()),
      range_n.clone(),
      range_n,
    ]
    .into(),
  );
  let data = vec![
    Expr::List(vec![Expr::Integer(n as i128), Expr::Integer(n as i128)].into()),
    payload,
  ];
  let structured = call("StructuredArray`StructuredData", data);
  Ok(call1("BlockDiagonalMatrix", structured))
}

/// An explicit number: the only entries wolframscript accepts when recovering
/// the generating vectors of a Cauchy matrix (a symbolic matrix such as
/// `{{1/(a+c), 1/(a+d)}, {1/(b+c), 1/(b+d)}}` is rejected there).
fn is_explicit_number(e: &Expr) -> bool {
  match e {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _) => true,
    Expr::FunctionCall { name, args } => {
      (name == "Rational" || name == "Complex")
        && args.len() == 2
        && args.iter().all(is_explicit_number)
    }
    _ => false,
  }
}

/// Recover the generating vectors of an explicit Cauchy matrix `mat`: writing
/// s_ij = 1/mat_ij = x_i + y_j, the normalization y_1 == 0 gives x_i = s_i1 and
/// y_j = s_1j - s_11 — the same choice wolframscript makes. Returns `None` when
/// `mat` has a zero or non-numeric entry, or when the reciprocals are not
/// consistent with any pair of generating vectors. Inexact matrices are matched
/// to a relative tolerance, so a Cauchy matrix that has been through machine
/// arithmetic is still recognized.
fn cauchy_generating_vectors(
  mat: &[Vec<Expr>],
) -> Result<Option<(Vec<Expr>, Vec<Expr>)>, InterpreterError> {
  if mat.is_empty() || mat[0].is_empty() {
    return Ok(None);
  }
  let mut inexact = false;
  for row in mat {
    for e in row {
      if is_zero_expr(e) || !is_explicit_number(e) {
        return Ok(None);
      }
      inexact |= matches!(e, Expr::Real(_) | Expr::BigFloat(_, _));
    }
  }

  // s_ij = 1/mat_ij
  let mut s: Vec<Vec<Expr>> = Vec::with_capacity(mat.len());
  for row in mat {
    let mut sr = Vec::with_capacity(row.len());
    for e in row {
      sr.push(evaluate_expr_to_expr(&pow(e.clone(), Expr::Integer(-1)))?);
    }
    s.push(sr);
  }

  let sub = |a: &Expr, b: &Expr| -> Result<Expr, InterpreterError> {
    evaluate_expr_to_expr(&plus(vec![a.clone(), neg(b.clone())]))
  };

  let x: Vec<Expr> = s.iter().map(|r| r[0].clone()).collect();
  let mut y: Vec<Expr> = Vec::with_capacity(s[0].len());
  for j in 0..s[0].len() {
    y.push(sub(&s[0][j], &s[0][0])?);
  }

  for (i, row) in s.iter().enumerate() {
    for (j, sij) in row.iter().enumerate() {
      let want =
        evaluate_expr_to_expr(&plus(vec![x[i].clone(), y[j].clone()]))?;
      let diff = sub(sij, &want)?;
      if is_zero_expr(&diff) {
        continue;
      }
      let close = inexact
        && match (
          crate::functions::math_ast::expr_to_f64(&diff),
          crate::functions::math_ast::expr_to_f64(sij),
        ) {
          (Some(d), Some(v)) => d.abs() <= 1e-8 * v.abs().max(1.0),
          _ => false,
        };
      if !close {
        return Ok(None);
      }
    }
  }
  Ok(Some((x, y)))
}

/// `CauchyMatrix[x, y]` — the Cauchy matrix whose entry (i, j) is
/// 1/(x_i + y_j), kept as the structured array
///   CauchyMatrix[StructuredArray`StructuredData[{m, n}, {x, y}]]
/// `CauchyMatrix[x]` is `CauchyMatrix[x, x]`, and `CauchyMatrix[mat]` recovers
/// the generating vectors of an explicit Cauchy matrix. `TargetStructure` ->
/// "Dense" returns the dense matrix instead of the structured array; Automatic
/// and "Structured" keep the structured form. Generating vectors with
/// x_i + y_j == 0 for some pair have no Cauchy matrix (`::cmvecs`), and a
/// matrix that is not a Cauchy matrix reports `::cmat` — both like
/// wolframscript.
pub fn cauchy_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unchanged = || Ok(unevaluated("CauchyMatrix", args));
  if args.is_empty() {
    return unchanged();
  }

  // Already in the canonical StructuredData form — keep it as-is.
  if let Expr::FunctionCall { name, .. } = &args[0]
    && name == "StructuredArray`StructuredData"
  {
    return unchanged();
  }

  // A rule or a (possibly empty) list of rules is an option, not a generating
  // vector: `CauchyMatrix[x, {}]` is `CauchyMatrix[x]` with no options.
  let is_option = |e: &Expr| match e {
    Expr::List(items) => items.iter().all(|i| rule_parts(i).is_some()),
    other => rule_parts(other).is_some(),
  };
  let positional = if args.len() >= 2 && !is_option(&args[1]) {
    2
  } else {
    1
  };

  let mut dense = false;
  for extra in &args[positional..] {
    let rules: Vec<&Expr> = match extra {
      Expr::List(items) => items.iter().collect(),
      other => vec![other],
    };
    for item in rules {
      let Some((lhs, rhs)) = rule_parts(item) else {
        crate::emit_message(&format!(
          "CauchyMatrix::nonopt: Options expected (instead of {}) beyond position {} in {}. An option must be a rule or a list of rules.",
          expr_to_string(extra),
          positional,
          expr_to_string(&unevaluated("CauchyMatrix", args))
        ));
        return unchanged();
      };
      match lhs {
        Expr::Identifier(name) if name == "TargetStructure" => match rhs {
          Expr::String(s) if s == "Dense" => dense = true,
          Expr::String(s) if s == "Structured" => dense = false,
          Expr::Identifier(a) if a == "Automatic" => dense = false,
          other => {
            let shown = match other {
              Expr::String(s) => s.clone(),
              e => expr_to_string(e),
            };
            crate::emit_message(&format!(
              "CauchyMatrix::badts: {shown} is not a valid target structure."
            ));
            return unchanged();
          }
        },
        other => {
          crate::emit_message(&format!(
            "CauchyMatrix::optx: Unknown option {} in {}.",
            expr_to_string(other),
            expr_to_string(&unevaluated("CauchyMatrix", args))
          ));
          return unchanged();
        }
      }
    }
  }

  let cmat = |shown: &Expr| {
    crate::emit_message(&format!(
      "CauchyMatrix::cmat: A Cauchy matrix could not be constructed from {}.",
      expr_to_string(shown)
    ));
  };

  // Resolve the generating vectors.
  let (x, y) = if positional == 2 {
    match (
      expr_to_vector(&args[0]).filter(|v| !v.is_empty()),
      expr_to_vector(&args[1]).filter(|v| !v.is_empty()),
    ) {
      (Some(x), Some(y)) => (x, y),
      // An empty generating vector, or a matrix in the first position, stays
      // unevaluated without a message in wolframscript.
      (Some(_), None) if matches!(&args[1], Expr::List(_)) => {
        cmat(&unevaluated("CauchyMatrix", args));
        return unchanged();
      }
      (None, _) if is_explicit_number(&args[0]) => {
        cmat(&unevaluated("CauchyMatrix", args));
        return unchanged();
      }
      _ => return unchanged(),
    }
  } else if let Some(v) = expr_to_vector(&args[0]).filter(|v| !v.is_empty()) {
    (v.clone(), v)
  } else if let Some(mat) = expr_to_matrix(&args[0]) {
    if let Some(pair) = cauchy_generating_vectors(&mat)? {
      pair
    } else {
      cmat(&args[0]);
      return unchanged();
    }
  } else if matches!(&args[0], Expr::List(items) if items.is_empty()) {
    // CauchyMatrix[{}] stays unevaluated without a message.
    return unchanged();
  } else if matches!(&args[0], Expr::List(_)) || is_explicit_number(&args[0]) {
    // A ragged list or a bare number is reported against the whole call.
    cmat(&unevaluated("CauchyMatrix", args));
    return unchanged();
  } else {
    return unchanged();
  };

  // No entry may have a vanishing denominator.
  for xi in &x {
    for yj in &y {
      let sum = evaluate_expr_to_expr(&plus(vec![xi.clone(), yj.clone()]))?;
      if is_zero_expr(&sum) {
        crate::emit_message(&format!(
          "CauchyMatrix::cmvecs: A Cauchy matrix could not be constructed from the vectors {} and {}.",
          expr_to_string(&Expr::List(x.clone().into())),
          expr_to_string(&Expr::List(y.clone().into()))
        ));
        return unchanged();
      }
    }
  }

  if dense {
    return cauchy_dense(&x, &y);
  }
  let data = vec![
    Expr::List(
      vec![
        Expr::Integer(x.len() as i128),
        Expr::Integer(y.len() as i128),
      ]
      .into(),
    ),
    Expr::List(vec![Expr::List(x.into()), Expr::List(y.into())].into()),
  ];
  let structured = call("StructuredArray`StructuredData", data);
  Ok(call1("CauchyMatrix", structured))
}

/// The dense m x n matrix with entry (i, j) equal to 1/(x_i + y_j).
fn cauchy_dense(x: &[Expr], y: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut rows = Vec::with_capacity(x.len());
  for xi in x {
    let mut row = Vec::with_capacity(y.len());
    for yj in y {
      row.push(evaluate_expr_to_expr(&pow(
        plus(vec![xi.clone(), yj.clone()]),
        Expr::Integer(-1),
      ))?);
    }
    rows.push(Expr::List(row.into()));
  }
  Ok(Expr::List(rows.into()))
}

/// A matrix SparseArray with background 0 dotted with a vector (either
/// side), computed from the stored entries alone. The result is a plain
/// list, as in wolframscript; a row with no entries is `0.` where the
/// product is machine arithmetic and `0` otherwise.
/// The dimensions and stored `(position, value)` entries of a rank-2
/// `SparseArray[Automatic, {r, c}, 0, …]`.
fn sparse_matrix_parts(
  e: &Expr,
) -> Option<([usize; 2], Vec<(Vec<i128>, Expr)>)> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  let [Expr::Identifier(auto), Expr::List(dims), background, csr] =
    args.as_slice()
  else {
    return None;
  };
  if name != "SparseArray"
    || auto != "Automatic"
    || !matches!(background, Expr::Integer(0))
  {
    return None;
  }
  let [Expr::Integer(r), Expr::Integer(c)] = dims.as_slice() else {
    return None;
  };
  let dims = [usize::try_from(*r).ok()?, usize::try_from(*c).ok()?];
  let entries =
    crate::functions::list_helpers_ast::sparse_array_extract_rules(&dims, csr);
  Some((dims, entries))
}

fn sparse_matrix_vector_dot(
  a: &Expr,
  b: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let matrix_parts = sparse_matrix_parts;
  let vector = |e: &Expr| -> Option<Vec<Expr>> {
    match e {
      Expr::List(items)
        if items.iter().all(|x| !matches!(x, Expr::List(_))) =>
      {
        Some(items.to_vec())
      }
      _ => None,
    }
  };
  // (matrix, vector, matrix on the left)
  let (dims, entries, v, left) =
    if let (Some((d, en)), Some(v)) = (matrix_parts(a), vector(b)) {
      (d, en, v, true)
    } else if let (Some(v), Some((d, en))) = (vector(a), matrix_parts(b)) {
      (d, en, v, false)
    } else {
      return None;
    };
  let (out_len, inner) = if left {
    (dims[0], dims[1])
  } else {
    (dims[1], dims[0])
  };
  if v.len() != inner {
    return None;
  }
  // One machine real among the stored values makes them all machine reals
  // (a packed real array in wolframscript).
  let real_matrix = entries.iter().any(|(_, v)| matches!(v, Expr::Real(_)));
  let to_real = |v: Expr| -> Expr {
    if real_matrix
      && let Some(x) = crate::functions::math_ast::try_eval_to_f64(&v)
      && !matches!(v, Expr::Real(_))
      && crate::functions::predicate_ast::is_numeric_q(&v)
    {
      Expr::Real(x)
    } else {
      v
    }
  };
  let mut terms: Vec<Vec<Expr>> = vec![Vec::new(); out_len];
  for (pos, val) in entries {
    let val = to_real(val);
    let (row, col) = (pos[0] as usize - 1, pos[1] as usize - 1);
    let (out, k) = if left { (row, col) } else { (col, row) };
    terms[out]
      .push(crate::functions::math_ast::times_ast(&[val, v[k].clone()]).ok()?);
  }
  let numeric_vector = v
    .iter()
    .all(|x| matches!(x, Expr::Real(_) | Expr::Integer(_)));
  let zero = if numeric_vector
    && (real_matrix || v.iter().any(|x| matches!(x, Expr::Real(_))))
  {
    Expr::Real(0.0)
  } else {
    Expr::Integer(0)
  };
  let mut out = Vec::with_capacity(out_len);
  for t in terms {
    out.push(match t.len() {
      0 => zero.clone(),
      1 => t.into_iter().next().unwrap(),
      _ => match crate::functions::math_ast::plus_ast(&t) {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
      },
    });
  }
  Some(Ok(Expr::List(out.into())))
}

pub fn dot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Dot expects exactly 2 arguments".into(),
    ));
  }

  if let Some(product) = sparse_matrix_vector_dot(&args[0], &args[1]) {
    return product;
  }

  // Densify any SparseArray or structured-matrix operand to its Normal
  // (dense nested-list) form and retry, so `SparseArray[...] . vec` and
  // `LowerTriangularMatrix[...] . UpperTriangularMatrix[...]` behave like
  // the dense case.
  let is_sparse = |e: &Expr| matches!(e, Expr::FunctionCall { name, .. } if name == "SparseArray");
  if is_sparse(&args[0])
    || is_sparse(&args[1])
    || structured_matrix_to_dense(&args[0]).is_some()
    || structured_matrix_to_dense(&args[1]).is_some()
  {
    let densify = |e: &Expr| -> Expr {
      if let Expr::FunctionCall { name, args: sa } = e
        && name == "SparseArray"
      {
        crate::functions::list_helpers_ast::sparse_array_ast(sa)
          .unwrap_or_else(|_| e.clone())
      } else {
        structured_matrix_to_dense(e).unwrap_or_else(|| e.clone())
      }
    };
    let (a, b) = (densify(&args[0]), densify(&args[1]));
    // An operand that could not be densified (too large to expand) would
    // send this straight back here; leave the product unevaluated instead.
    if is_sparse(&a)
      || is_sparse(&b)
      || structured_matrix_to_dense(&a).is_some()
      || structured_matrix_to_dense(&b).is_some()
    {
      return Ok(unevaluated("Dot", args));
    }
    return dot_ast(&[a, b]);
  }

  // TransformationFunction[M1] . TransformationFunction[M2] →
  //   TransformationFunction[M1.M2]  (compose affine transforms)
  let extract_tf = |e: &Expr| -> Option<Expr> {
    if let Expr::FunctionCall { name, args } = e
      && name == "TransformationFunction"
      && args.len() == 1
    {
      return Some(args[0].clone());
    }
    None
  };
  if let (Some(ma), Some(mb)) = (extract_tf(&args[0]), extract_tf(&args[1])) {
    let product = dot_ast(&[ma, mb])?;
    return Ok(call1("TransformationFunction", product));
  }

  // Incompatible shapes emit ::dotsh and return the unevaluated form.
  let dotsh = || {
    crate::emit_message(&format!(
      "Dot::dotsh: Tensors {} and {} have incompatible shapes.",
      crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output),
      crate::syntax::format_expr(&args[1], crate::syntax::ExprForm::Output)
    ));
    Ok(unevaluated("Dot", args))
  };

  // A tensor of rank three or more contracts its last index with the
  // other operand's first, which is ArrayDot at depth 1. (The matrix code
  // below would take a rank-3 tensor for a matrix of vectors.)
  if let (Some(da), Some(db)) = (tensor_dims(&args[0]), tensor_dims(&args[1]))
    && (da.len() > 2 || db.len() > 2)
  {
    if da.last() != db.first() {
      return dotsh();
    }
    return array_dot_ast(&[
      args[0].clone(),
      args[1].clone(),
      Expr::Integer(1),
    ]);
  }

  if let Some(product) = machine_dot(&args[0], &args[1]) {
    return Ok(product);
  }
  if let Some(product) = machine_complex_dot(&args[0], &args[1]) {
    return Ok(product);
  }

  // Vector . Vector -> scalar
  if let (Some(va), Some(vb)) =
    (expr_to_vector(&args[0]), expr_to_vector(&args[1]))
  {
    if va.len() != vb.len() {
      return dotsh();
    }
    let terms: Vec<Expr> = va
      .iter()
      .zip(vb.iter())
      .map(|(a, b)| eval_mul(a, b))
      .collect();
    return Ok(eval_add_many(terms));
  }

  // Matrix . Vector → vector
  if let (Some(ma), Some(vb)) =
    (expr_to_matrix(&args[0]), expr_to_vector(&args[1]))
  {
    let ncols = ma.first().map_or(0, std::vec::Vec::len);
    if ncols != vb.len() {
      return dotsh();
    }
    let mut result = Vec::new();
    for row in &ma {
      let terms: Vec<Expr> = row
        .iter()
        .zip(vb.iter())
        .map(|(a, b)| eval_mul(a, b))
        .collect();
      result.push(eval_add_many(terms));
    }
    return Ok(Expr::List(result.into()));
  }

  // Vector . Matrix → vector (treat vector as a row)
  if let (Some(va), Some(mb)) =
    (expr_to_vector(&args[0]), expr_to_matrix(&args[1]))
  {
    let b_rows = mb.len();
    if va.len() != b_rows {
      return dotsh();
    }
    let b_cols = mb.first().map_or(0, std::vec::Vec::len);
    let mut result = Vec::with_capacity(b_cols);
    for j in 0..b_cols {
      let terms: Vec<Expr> =
        (0..b_rows).map(|i| eval_mul(&va[i], &mb[i][j])).collect();
      result.push(eval_add_many(terms));
    }
    return Ok(Expr::List(result.into()));
  }

  // Matrix . Matrix → matrix
  if let (Some(ma), Some(mb)) =
    (expr_to_matrix(&args[0]), expr_to_matrix(&args[1]))
  {
    let a_cols = ma.first().map_or(0, std::vec::Vec::len);
    let b_rows = mb.len();
    if a_cols != b_rows {
      return dotsh();
    }
    let b_cols = mb.first().map_or(0, std::vec::Vec::len);
    let mut result = Vec::new();
    for i in 0..ma.len() {
      let mut row = Vec::new();
      for j in 0..b_cols {
        let terms: Vec<Expr> = (0..a_cols)
          .map(|k| eval_mul(&ma[i][k], &mb[k][j]))
          .collect();
        row.push(eval_add_many(terms));
      }
      result.push(row);
    }
    return Ok(matrix_to_expr(result));
  }

  // Return unevaluated
  Ok(unevaluated("Dot", args))
}

/// The parts of a machine complex number (or machine real), read
/// structurally from its usual forms `re + im*I`, `im*I` and `I` before
/// falling back to the general reader.
fn machine_complex_parts(e: &Expr) -> Option<(f64, f64)> {
  let imaginary = |e: &Expr| -> Option<f64> {
    match e {
      Expr::Identifier(i) if i == "I" => Some(1.0),
      Expr::FunctionCall { name, args }
        if name == "Times" && args.len() == 2 =>
      {
        match (&args[0], &args[1]) {
          (Expr::Real(c), Expr::Identifier(i)) if i == "I" => Some(*c),
          (Expr::Integer(c), Expr::Identifier(i)) if i == "I" => {
            Some(*c as f64)
          }
          _ => None,
        }
      }
      _ => None,
    }
  };
  match e {
    Expr::Real(r) => return Some((*r, 0.0)),
    Expr::Integer(n) => return Some((*n as f64, 0.0)),
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() == 2 => {
      if let (Expr::Real(re), Some(im)) = (&args[0], imaginary(&args[1])) {
        return Some((*re, im));
      }
    }
    other => {
      if let Some(im) = imaginary(other) {
        return Some((0.0, im));
      }
    }
  }
  crate::functions::list_helpers_ast::sorting::expr_to_complex_parts(e)
}

/// `a . b` for vectors and matrices of machine complex numbers (at least
/// one entry inexact and one non-real), in complex machine arithmetic.
fn machine_complex_dot(a: &Expr, b: &Expr) -> Option<Expr> {
  type C = (f64, f64);
  fn vector(e: &Expr) -> Option<Vec<C>> {
    match e {
      Expr::List(items)
        if !items.is_empty()
          && !items.iter().any(|x| matches!(x, Expr::List(_))) =>
      {
        items.iter().map(machine_complex_parts).collect()
      }
      _ => None,
    }
  }
  fn matrix(e: &Expr) -> Option<Vec<Vec<C>>> {
    let Expr::List(rows) = e else { return None };
    let m: Vec<Vec<C>> = rows.iter().map(vector).collect::<Option<_>>()?;
    let width = m.first()?.len();
    m.iter().all(|r| r.len() == width).then_some(m)
  }
  // Cheap gate before parsing: some entry of either operand mentions I.
  fn mentions_i(e: &Expr) -> bool {
    match e {
      Expr::Identifier(i) => i == "I",
      Expr::List(items) => items.iter().any(mentions_i),
      Expr::FunctionCall { args, .. } => args.iter().any(mentions_i),
      _ => false,
    }
  }
  if !(mentions_i(a) || mentions_i(b)) {
    return None;
  }
  if !(crate::functions::math_ast::numeric_utils::contains_inexact_real(a)
    || crate::functions::math_ast::numeric_utils::contains_inexact_real(b))
  {
    return None;
  }
  let mul = |x: C, y: C| (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0);
  let dot = |pairs: &mut dyn Iterator<Item = (C, C)>| -> C {
    pairs.fold((0.0, 0.0), |acc, (x, y)| {
      let p = mul(x, y);
      (acc.0 + p.0, acc.1 + p.1)
    })
  };
  let out = |c: C| {
    crate::functions::math_ast::numeric_utils::machine_complex_result(c.0, c.1)
  };
  let list = |v: Vec<Expr>| Expr::List(v.into());
  if let (Some(u), Some(v)) = (vector(a), vector(b)) {
    if u.len() != v.len() {
      return None;
    }
    return Some(out(dot(&mut u.iter().copied().zip(v.iter().copied()))));
  }
  if let (Some(u), Some(mb)) = (vector(a), matrix(b)) {
    if mb.len() != u.len() {
      return None;
    }
    return Some(list(
      (0..mb[0].len())
        .map(|j| out(dot(&mut (0..u.len()).map(|i| (u[i], mb[i][j])))))
        .collect(),
    ));
  }
  let ma = matrix(a)?;
  if let Some(v) = vector(b) {
    if ma[0].len() != v.len() {
      return None;
    }
    return Some(list(
      ma.iter()
        .map(|row| out(dot(&mut row.iter().copied().zip(v.iter().copied()))))
        .collect(),
    ));
  }
  let mb = matrix(b)?;
  if ma[0].len() != mb.len() {
    return None;
  }
  Some(list(
    ma.iter()
      .map(|row| {
        list(
          (0..mb[0].len())
            .map(|j| out(dot(&mut (0..row.len()).map(|k| (row[k], mb[k][j])))))
            .collect(),
        )
      })
      .collect(),
  ))
}

/// `a . b` for vectors and matrices of machine reals, computed in `f64`
/// with each sum folded left in index order — the order `Plus` folds the
/// same products in. `None` when either operand holds anything but machine
/// reals, or the shapes do not fit.
fn machine_dot(a: &Expr, b: &Expr) -> Option<Expr> {
  // Exact numbers multiply in machine arithmetic once a machine real takes
  // part (`{1, Sqrt[2]} . {1.5, 2.}` is a machine number in wolframscript).
  fn has_real(e: &Expr) -> bool {
    match e {
      Expr::Real(_) => true,
      Expr::List(items) => items.iter().any(|x| match x {
        Expr::Real(_) => true,
        Expr::List(row) => row.iter().any(|y| matches!(y, Expr::Real(_))),
        _ => false,
      }),
      _ => false,
    }
  }
  let mixed = has_real(a) || has_real(b);
  let scalar = |x: &Expr| -> Option<f64> {
    match x {
      Expr::Real(r) => Some(*r),
      Expr::Integer(n) if mixed => Some(*n as f64),
      Expr::List(_) => None,
      other
        if mixed && crate::functions::predicate_ast::is_numeric_q(other) =>
      {
        // Compiled arithmetic first: the general numericizer is slow on the
        // large exact entries a normalized exact data matrix has.
        crate::functions::polynomial_ast::solve::compile_numeric(other, &[])
          .filter(|n| !n.has_fallback())
          .map(|n| n.eval(&[]))
          .or_else(|| crate::functions::math_ast::try_eval_to_f64(other))
          .filter(|v| v.is_finite())
      }
      _ => None,
    }
  };
  let reals = |e: &Expr| -> Option<Vec<f64>> {
    match e {
      Expr::List(items) => items.iter().map(scalar).collect(),
      _ => None,
    }
  };
  let matrix = |e: &Expr| -> Option<Vec<Vec<f64>>> {
    let Expr::List(rows) = e else { return None };
    let m: Vec<Vec<f64>> = rows.iter().map(reals).collect::<Option<_>>()?;
    let width = m.first()?.len();
    (width > 0 && m.iter().all(|r| r.len() == width)).then_some(m)
  };
  let dot = |x: &mut dyn Iterator<Item = f64>| -> f64 {
    let first = x.next().unwrap_or(0.0);
    x.fold(first, |acc, v| acc + v)
  };
  let reals_list = |v: Vec<f64>| {
    Expr::List(v.into_iter().map(Expr::Real).collect::<Vec<_>>().into())
  };
  match (reals(a), reals(b)) {
    (Some(u), Some(v)) if !u.is_empty() && u.len() == v.len() => {
      return Some(Expr::Real(dot(&mut u.iter().zip(&v).map(|(x, y)| x * y))));
    }
    (Some(u), None) => {
      let mb = matrix(b)?;
      if mb.len() != u.len() || u.is_empty() {
        return None;
      }
      return Some(reals_list(
        (0..mb[0].len())
          .map(|j| dot(&mut (0..u.len()).map(|i| u[i] * mb[i][j])))
          .collect(),
      ));
    }
    _ => {}
  }
  let ma = matrix(a)?;
  if let Some(v) = reals(b) {
    if ma[0].len() != v.len() {
      return None;
    }
    return Some(reals_list(
      ma.iter()
        .map(|row| dot(&mut row.iter().zip(&v).map(|(x, y)| x * y)))
        .collect(),
    ));
  }
  let mb = matrix(b)?;
  if ma[0].len() != mb.len() {
    return None;
  }
  let cols = mb[0].len();
  Some(Expr::List(
    ma.iter()
      .map(|row| {
        reals_list(
          (0..cols)
            .map(|j| dot(&mut row.iter().zip(&mb).map(|(x, brow)| x * brow[j])))
            .collect(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  ))
}

/// Dimensions of a rectangular tensor. Scalars have dimensions `[]`; a ragged
/// list returns `None`.
fn tensor_dims(e: &Expr) -> Option<Vec<usize>> {
  match e {
    Expr::List(items) => {
      if items.is_empty() {
        return Some(vec![0]);
      }
      let first = tensor_dims(&items[0])?;
      for it in items.iter().skip(1) {
        if tensor_dims(it)? != first {
          return None;
        }
      }
      let mut dims = Vec::with_capacity(first.len() + 1);
      dims.push(items.len());
      dims.extend(first);
      Some(dims)
    }
    _ => Some(Vec::new()),
  }
}

/// Flatten a rectangular tensor into its leaves in row-major order.
fn flatten_tensor(e: &Expr) -> Vec<Expr> {
  match e {
    Expr::List(items) => items.iter().flat_map(flatten_tensor).collect(),
    other => vec![other.clone()],
  }
}

/// Rebuild a tensor of the given dimensions from a row-major flat list.
/// Empty `dims` yields the single scalar leaf.
fn build_from_flat(flat: &[Expr], dims: &[usize]) -> Expr {
  if dims.is_empty() {
    return flat[0].clone();
  }
  let inner: usize = dims[1..].iter().product();
  let mut out = Vec::with_capacity(dims[0]);
  for i in 0..dims[0] {
    out.push(build_from_flat(
      &flat[i * inner..(i + 1) * inner],
      &dims[1..],
    ));
  }
  Expr::List(out.into())
}

/// ArrayDot[a, b, k] — contract the last `k` dimensions of `a` with the first
/// `k` dimensions of `b`, summing products over the shared block. `k == 0`
/// gives the outer product; full contraction yields a scalar.
pub fn array_dot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("ArrayDot", args));
  if args.len() != 3 {
    return unevaluated();
  }
  // The contraction depth must be a non-negative integer.
  let k = match &args[2] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => return unevaluated(),
  };
  let (a, b) = (&args[0], &args[1]);
  let (Some(da), Some(db)) = (tensor_dims(a), tensor_dims(b)) else {
    return unevaluated();
  };

  // k may not exceed the depth (rank) of either array.
  if k > da.len() || k > db.len() {
    let bad = if k > da.len() { a } else { b };
    crate::emit_message(&format!(
      "ArrayDot::kspec: {} is larger than the depth of array {}.",
      k,
      crate::syntax::format_expr(bad, crate::syntax::ExprForm::Output)
    ));
    return unevaluated();
  }

  // The contracted dimensions must line up.
  let a_contract = &da[da.len() - k..];
  let b_contract = &db[..k];
  if a_contract != b_contract {
    crate::emit_message(&format!(
      "Dot::dotsh: Tensors {} and {} have incompatible shapes.",
      crate::syntax::format_expr(a, crate::syntax::ExprForm::Output),
      crate::syntax::format_expr(b, crate::syntax::ExprForm::Output)
    ));
    return unevaluated();
  }

  let c: usize = a_contract.iter().product();
  let a_lead = &da[..da.len() - k];
  let b_trail = &db[k..];
  let pa: usize = a_lead.iter().product();
  let pb: usize = b_trail.iter().product();

  let fa = flatten_tensor(a);
  let fb = flatten_tensor(b);

  let mut res = Vec::with_capacity(pa * pb);
  for i in 0..pa {
    for j in 0..pb {
      let mut sum = Expr::Integer(0);
      for cc in 0..c {
        sum = eval_add(&sum, &eval_mul(&fa[i * c + cc], &fb[cc * pb + j]));
      }
      res.push(sum);
    }
  }

  let out_dims: Vec<usize> =
    a_lead.iter().chain(b_trail.iter()).copied().collect();
  Ok(build_from_flat(&res, &out_dims))
}

/// Whether a `Vec<Vec<Expr>>` from `expr_to_matrix` is a nonempty square
/// matrix (every row has length equal to the number of rows, and no entry
/// is itself a list).
pub fn is_nonempty_square(m: &[Vec<Expr>]) -> bool {
  !m.is_empty()
    && m.iter().all(|row| {
      row.len() == m.len() && row.iter().all(|e| !matches!(e, Expr::List(_)))
    })
}

/// Whether `arg` should trigger a `::matsq` message — a concrete non-matrix
/// argument (a list, or a numeric value such as 5, Pi, 3/4). Bare symbols and
/// symbolic expressions (`x`, `a + b`, `f[x]`) stay held with no message.
fn is_matsq_subject(arg: &Expr) -> bool {
  matches!(arg, Expr::List(_))
    || crate::functions::predicate_ast::is_numeric_q(arg)
}

/// Emit `<F>::matsq: Argument <m> at position 1 is not a nonempty square
/// matrix.` (for concrete arguments) and return the unevaluated call,
/// matching wolframscript for matrix functions given a non-square or
/// non-matrix argument.
pub fn matsq_unevaluated(name: &str, args: &[Expr]) -> Expr {
  if is_matsq_subject(&args[0]) {
    crate::emit_message(&format!(
      "{}::matsq: Argument {} at position 1 is not a nonempty square matrix.",
      name,
      crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
    ));
  }
  unevaluated(name, args)
}

/// Det[matrix] - determinant of a square matrix
pub fn det_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Det expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) if is_nonempty_square(&m) => m,
    _ => return Ok(matsq_unevaluated("Det", args)),
  };
  if let Some(m) = machine_real_matrix(&matrix) {
    return Ok(Expr::Real(machine_determinant(&m)));
  }
  if let Some(det) = det_by_one_symbolic_row(&matrix) {
    return det;
  }
  let det = determinant(&matrix);
  // Wolfram's determinant of a polynomial matrix is the expanded
  // polynomial (`Det[{{2, 4}, {6, 2 x}}]` is `-24 + 4 x`), not a factored
  // simplification of it.
  // With rational coefficients it is brought over one denominator
  // (`Det[{{1/2, x}, {x, 1}}]` is `(1 - 2 x^2)/2`).
  if is_polynomial_matrix(&matrix) {
    let combine = if matrix.iter().flatten().any(has_rational) {
      "Together"
    } else {
      "Expand"
    };
    return Ok(
      crate::evaluator::evaluate_function_call_ast(
        combine,
        std::slice::from_ref(&det),
      )
      .unwrap_or(det),
    );
  }
  let r = crate::functions::expand_and_combine(&det);
  Ok(simplify_expr(&r))
}

/// A machine-number matrix with a single row (or column) of symbolic
/// entries — the `Det[{p1, …, pn, {1, x, y, …}}]` that gives the curve
/// through n points — expanded along that row with machine-number minors.
/// Symbolic elimination of the whole matrix took minutes for 10 x 10.
fn det_by_one_symbolic_row(
  matrix: &[Vec<Expr>],
) -> Option<Result<Expr, InterpreterError>> {
  let n = matrix.len();
  if n < 3 {
    return None;
  }
  let numeric = |e: &Expr| {
    matches!(e, Expr::Real(_) | Expr::Integer(_))
      || matches!(e, Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2)
  };
  let inexact = matrix.iter().flatten().any(|e| matches!(e, Expr::Real(_)));
  if !inexact {
    return None;
  }
  let symbolic_rows: Vec<usize> =
    (0..n).filter(|&i| !matrix[i].iter().all(numeric)).collect();
  let symbolic_cols: Vec<usize> = (0..n)
    .filter(|&j| !(0..n).all(|i| numeric(&matrix[i][j])))
    .collect();
  // Expand along a row: transpose a column case into it.
  let (rows, row): (Vec<Vec<Expr>>, usize) =
    match (symbolic_rows.as_slice(), symbolic_cols.as_slice()) {
      ([i], _) => (matrix.to_vec(), *i),
      (_, [j]) => (
        (0..n)
          .map(|r| (0..n).map(|c| matrix[c][r].clone()).collect())
          .collect(),
        *j,
      ),
      _ => return None,
    };
  let mut terms = Vec::with_capacity(n);
  for j in 0..n {
    let minor: Option<Vec<Vec<f64>>> = (0..n)
      .filter(|&r| r != row)
      .map(|r| {
        (0..n)
          .filter(|&c| c != j)
          .map(|c| try_eval_to_f64(&rows[r][c]))
          .collect()
      })
      .collect();
    let minor = minor?;
    let sign = if (row + j) % 2 == 0 { 1.0 } else { -1.0 };
    // Small minors by their formula, which is exact where the products are.
    let value = if minor.len() == 2 {
      minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0]
    } else {
      machine_determinant(&minor)
    };
    terms.push(crate::helpers::times(vec![
      Expr::Real(sign * value),
      rows[row][j].clone(),
    ]));
  }
  Some(crate::evaluator::evaluate_function_call_ast(
    "Expand",
    &[crate::helpers::plus(terms)],
  ))
}

/// Whether `e` contains a `Rational` anywhere.
fn has_rational(e: &Expr) -> bool {
  match e {
    Expr::FunctionCall { name, args } => {
      name == "Rational" || args.iter().any(has_rational)
    }
    Expr::List(items) => items.iter().any(has_rational),
    Expr::BinaryOp { left, right, .. } => {
      has_rational(left) || has_rational(right)
    }
    Expr::UnaryOp { operand, .. } => has_rational(operand),
    _ => false,
  }
}

/// Whether every entry is a polynomial in its symbols with exact rational
/// coefficients (no division by a symbol, no radicals of symbols).
fn is_polynomial_matrix(matrix: &[Vec<Expr>]) -> bool {
  let mut symbols: Vec<Expr> = Vec::new();
  for e in matrix.iter().flatten() {
    if crate::functions::math_ast::numeric_utils::contains_inexact_real(e) {
      return false;
    }
    if let Ok(Expr::List(ref vs)) = crate::evaluator::evaluate_function_call_ast(
      "Variables",
      std::slice::from_ref(e),
    ) {
      for v in vs {
        if !symbols
          .iter()
          .any(|s| crate::evaluator::pattern_matching::expr_equal(s, v))
        {
          symbols.push(v.clone());
        }
      }
    } else {
      return false;
    }
  }
  let vars = Expr::List(symbols.into());
  matrix.iter().flatten().all(|e| {
    matches!(
      crate::evaluator::evaluate_function_call_ast(
        "PolynomialQ",
        &[e.clone(), vars.clone()]
      ),
      Ok(Expr::Identifier(ref t)) if t == "True"
    )
  })
}

/// The matrix as machine reals when it holds a machine real and every
/// entry is a real number — the matrices Wolfram inverts and factors
/// numerically (LU with partial pivoting) instead of symbolically.
fn machine_real_matrix(matrix: &[Vec<Expr>]) -> Option<Vec<Vec<f64>>> {
  // An inexact number anywhere in an entry (`0.5 Sqrt[2]`) makes the
  // matrix a machine-precision one.
  if !matrix
    .iter()
    .flatten()
    .any(crate::functions::math_ast::numeric_utils::contains_inexact_real)
  {
    return None;
  }
  matrix
    .iter()
    .map(|row| row.iter().map(try_eval_to_f64).collect::<Option<Vec<_>>>())
    .collect()
}

/// LU factorization with partial pivoting, in place: `a` becomes `L\U`
/// (unit lower triangle implied). Returns the row permutation and its sign,
/// or `None` when a pivot column is entirely zero (an exactly singular
/// matrix).
fn lu_partial_pivot(a: &mut [Vec<f64>]) -> Option<(Vec<usize>, f64)> {
  let n = a.len();
  let mut perm: Vec<usize> = (0..n).collect();
  let mut sign = 1.0;
  for k in 0..n {
    let p = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
    if a[p][k] == 0.0 {
      return None;
    }
    if p != k {
      a.swap(p, k);
      perm.swap(p, k);
      sign = -sign;
    }
    for i in k + 1..n {
      let factor = a[i][k] / a[k][k];
      a[i][k] = factor;
      for j in k + 1..n {
        a[i][j] -= factor * a[k][j];
      }
    }
  }
  Some((perm, sign))
}

/// `Det` of a machine-real matrix: the product of the LU pivots.
fn machine_determinant(m: &[Vec<f64>]) -> f64 {
  let mut a = m.to_vec();
  match lu_partial_pivot(&mut a) {
    Some((_, sign)) => (0..a.len()).fold(sign, |acc, i| acc * a[i][i]),
    None => 0.0,
  }
}

/// `Inverse` of a machine-real matrix by LU, or `None` when singular.
fn machine_inverse(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
  let n = m.len();
  let mut a = m.to_vec();
  let (perm, _) = lu_partial_pivot(&mut a)?;
  let mut inv = vec![vec![0.0; n]; n];
  for col in 0..n {
    // Solve L y = P e_col, then U x = y.
    let mut x: Vec<f64> = (0..n)
      .map(|i| if perm[i] == col { 1.0 } else { 0.0 })
      .collect();
    for i in 0..n {
      for k in 0..i {
        x[i] -= a[i][k] * x[k];
      }
    }
    for i in (0..n).rev() {
      for k in i + 1..n {
        x[i] -= a[i][k] * x[k];
      }
      x[i] /= a[i][i];
    }
    for i in 0..n {
      inv[i][col] = x[i];
    }
  }
  Some(inv)
}

/// True for an exact rational scalar (Integer, BigInteger, or literal
/// Rational) — the entries the fast Bareiss path handles exactly.
fn is_exact_rational_entry(e: &Expr) -> bool {
  matches!(e, Expr::Integer(_) | Expr::BigInteger(_))
    || matches!(e, Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2)
}

/// Bareiss fraction-free Gaussian elimination — an O(n^3) exact determinant
/// for integer/rational matrices. Every division is exact (the running
/// quotient is always the previous pivot), so the result stays exact while
/// avoiding the O(n!) blow-up of cofactor expansion.
fn determinant_bareiss(matrix: &[Vec<Expr>]) -> Expr {
  let n = matrix.len();
  let mut a: Vec<Vec<Expr>> = matrix.to_vec();
  let mut prev = Expr::Integer(1);
  let mut sign: i32 = 1;
  for k in 0..n - 1 {
    if is_zero_expr(&a[k][k]) {
      // Pivot is zero: swap in a row below with a nonzero entry in column k.
      let swap_row = (k + 1..n).find(|&i| !is_zero_expr(&a[i][k]));
      match swap_row {
        Some(i) => {
          a.swap(k, i);
          sign = -sign;
        }
        None => return Expr::Integer(0), // singular
      }
    }
    let pivot = a[k][k].clone();
    for i in k + 1..n {
      for j in k + 1..n {
        let num =
          eval_sub(&eval_mul(&a[i][j], &pivot), &eval_mul(&a[i][k], &a[k][j]));
        a[i][j] = eval_divide(&num, &prev);
      }
    }
    prev = pivot;
  }
  let det = a[n - 1][n - 1].clone();
  if sign < 0 {
    eval_sub(&Expr::Integer(0), &det)
  } else {
    det
  }
}

/// Compute determinant recursively via cofactor expansion
fn determinant(matrix: &[Vec<Expr>]) -> Expr {
  let n = matrix.len();
  if n == 0 {
    return Expr::Integer(1);
  }
  if n == 1 {
    return matrix[0][0].clone();
  }
  if n == 2 {
    return eval_sub(
      &eval_mul(&matrix[0][0], &matrix[1][1]),
      &eval_mul(&matrix[0][1], &matrix[1][0]),
    );
  }
  // Fast O(n^3) path for exact integer/rational matrices, where cofactor
  // expansion's O(n!) cost makes even a 20x20 determinant hang.
  if matrix
    .iter()
    .all(|row| row.iter().all(is_exact_rational_entry))
  {
    return determinant_bareiss(matrix);
  }
  // A single row (or column) holding all the symbols — an interpolation
  // matrix whose last row is the monomials — is expanded along that line,
  // every minor then being an exact numeric determinant.
  let exact_row = |row: &[Expr]| row.iter().all(is_exact_rational_entry);
  let symbolic_rows: Vec<usize> =
    (0..n).filter(|&i| !exact_row(&matrix[i])).collect();
  if symbolic_rows.len() == 1 && symbolic_rows[0] != 0 {
    let r = symbolic_rows[0];
    let mut reordered = matrix.to_vec();
    let row = reordered.remove(r);
    reordered.insert(0, row);
    // Moving row r to the top is r adjacent swaps.
    let det = determinant(&reordered);
    let det = if r % 2 == 1 {
      eval_sub(&Expr::Integer(0), &det)
    } else {
      det
    };
    return crate::evaluator::evaluate_function_call_ast(
      "Expand",
      std::slice::from_ref(&det),
    )
    .unwrap_or(det);
  }
  if symbolic_rows.len() > 1 {
    let transposed: Vec<Vec<Expr>> = (0..n)
      .map(|j| (0..n).map(|i| matrix[i][j].clone()).collect())
      .collect();
    let symbolic_cols = (0..n).filter(|&j| !exact_row(&transposed[j])).count();
    if symbolic_cols == 1 {
      let det = determinant(&transposed);
      return crate::evaluator::evaluate_function_call_ast(
        "Expand",
        std::slice::from_ref(&det),
      )
      .unwrap_or(det);
    }
  }
  let mut det = Expr::Integer(0);
  for j in 0..n {
    // A zero entry's cofactor contributes nothing.
    if is_zero_expr(&matrix[0][j]) {
      continue;
    }
    let mut minor = Vec::new();
    for i in 1..n {
      let mut row = Vec::new();
      for k in 0..n {
        if k != j {
          row.push(matrix[i][k].clone());
        }
      }
      minor.push(row);
    }
    let cofactor = eval_mul(&matrix[0][j], &determinant(&minor));
    if j % 2 == 0 {
      det = eval_add(&det, &cofactor);
    } else {
      det = eval_sub(&det, &cofactor);
    }
  }
  det
}

/// Permanent[matrix] — the matrix permanent, `Sum[Product[m[i, sigma(i)]]]`
/// over all permutations sigma. Like the determinant but without the
/// alternating signs. Works for numeric and symbolic square matrices.
pub fn permanent_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Permanent expects exactly 1 argument".into(),
    ));
  }
  let matsq = || {
    crate::emit_message(&format!(
      "Permanent::matsq: Argument {} at position 1 is not a nonempty square matrix.",
      crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
    ));
    Ok(unevaluated("Permanent", args))
  };
  let Some(matrix) = expr_to_matrix(&args[0]) else {
    return matsq();
  };
  let n = matrix.len();
  if n == 0 || matrix.iter().any(|row| row.len() != n) {
    return matsq();
  }

  // Fast O(2^n * n^2) Ryser path for exact integer/rational matrices, where
  // the O(n!) permutation sum below hangs (n=11 already takes ~10s). Capped
  // at n < 63 so the subset bitmask fits in u64 (larger permanents are
  // infeasible by any method).
  if n < 63
    && matrix
      .iter()
      .all(|row| row.iter().all(is_exact_rational_entry))
  {
    return Ok(crate::functions::expand_and_combine(&permanent_ryser(
      &matrix,
    )));
  }

  // Sum over all column permutations of the product of selected entries.
  fn accumulate(
    matrix: &[Vec<Expr>],
    used: &mut [bool],
    row: usize,
    partial: &Expr,
    total: &mut Expr,
  ) {
    let n = matrix.len();
    if row == n {
      *total = eval_add(total, partial);
      return;
    }
    for col in 0..n {
      if !used[col] {
        used[col] = true;
        let next = eval_mul(partial, &matrix[row][col]);
        accumulate(matrix, used, row + 1, &next, total);
        used[col] = false;
      }
    }
  }

  let mut used = vec![false; n];
  let mut total = Expr::Integer(0);
  accumulate(&matrix, &mut used, 0, &Expr::Integer(1), &mut total);
  Ok(crate::functions::expand_and_combine(&total))
}

/// Ryser's formula for the permanent, O(2^n * n^2):
///   perm(A) = sum over column subsets S of (-1)^(n-|S|) * prod_i (sum_{j in S} a_ij)
/// (the outer (-1)^n is folded into the per-subset sign). Exact for
/// integer/rational entries.
fn permanent_ryser(matrix: &[Vec<Expr>]) -> Expr {
  let n = matrix.len();
  let mut total = Expr::Integer(0);
  for s in 1u64..(1u64 << n) {
    let popcount = s.count_ones() as usize;
    let mut prod = Expr::Integer(1);
    for row in matrix {
      let mut rowsum = Expr::Integer(0);
      for (j, entry) in row.iter().enumerate() {
        if s & (1u64 << j) != 0 {
          rowsum = eval_add(&rowsum, entry);
        }
      }
      prod = eval_mul(&prod, &rowsum);
      if is_zero_expr(&prod) {
        break;
      }
    }
    if (n - popcount).is_multiple_of(2) {
      total = eval_add(&total, &prod);
    } else {
      total = eval_sub(&total, &prod);
    }
  }
  total
}

/// PfaffianDet[m] — the Pfaffian of an antisymmetric matrix `m`. Defined so
/// that `PfaffianDet[m]^2 == Det[m]`. Requires a nonempty square antisymmetric
/// matrix; an odd-order matrix has Pfaffian `0`. Non-matrix / non-square /
/// non-antisymmetric arguments emit the matching message and stay unevaluated,
/// as in wolframscript.
pub fn pfaffian_det_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PfaffianDet expects exactly 1 argument".into(),
    ));
  }
  let unevaluated = || unevaluated("PfaffianDet", args);
  let Some(matrix) = expr_to_matrix(&args[0]) else {
    // A concrete non-matrix argument (scalar or ragged list) → ::matrix.
    if is_matsq_subject(&args[0]) {
      crate::emit_message(&format!(
        "PfaffianDet::matrix: Argument {} at position 1 is not a nonempty rectangular matrix.",
        crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
      ));
    }
    return Ok(unevaluated());
  };
  if !is_nonempty_square(&matrix) {
    return Ok(matsq_unevaluated("PfaffianDet", args));
  }
  // The Pfaffian is only defined for antisymmetric matrices (A^T == -A, which
  // forces a zero diagonal). Bail with ::asymm otherwise.
  let n = matrix.len();
  let antisymmetric = (0..n).all(|i| {
    (i..n).all(|j| is_zero_expr(&eval_add(&matrix[i][j], &matrix[j][i])))
  });
  if !antisymmetric {
    // PfaffianDet ships in wolframscript only as a resource function whose
    // ::asymm message text is not registered, so wolframscript prints the
    // "-- Message text not found --" placeholder (and no trailing period).
    // Match that verbatim for conformance.
    crate::emit_message(&format!(
      "PfaffianDet::asymm: -- Message text not found -- ({})",
      crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
    ));
    return Ok(unevaluated());
  }
  Ok(crate::functions::expand_and_combine(&pfaffian(&matrix)))
}

/// Recursive Pfaffian via minor expansion along the first row:
///   Pf(A) = sum_{j>0} (-1)^(j-1) A[0][j] Pf(A with rows/cols 0 and j removed).
/// Pf of the empty (0x0) matrix is 1; an odd-order matrix has Pfaffian 0.
fn pfaffian(m: &[Vec<Expr>]) -> Expr {
  let n = m.len();
  if n == 0 {
    return Expr::Integer(1);
  }
  if n % 2 == 1 {
    return Expr::Integer(0);
  }
  let mut result = Expr::Integer(0);
  for j in 1..n {
    if is_zero_expr(&m[0][j]) {
      continue;
    }
    // Minor: keep rows and columns in {1, …, n-1} \ {j}.
    let minor: Vec<Vec<Expr>> = (1..n)
      .filter(|&i| i != j)
      .map(|i| {
        (1..n)
          .filter(|&k| k != j)
          .map(|k| m[i][k].clone())
          .collect()
      })
      .collect();
    let term = eval_mul(&m[0][j], &pfaffian(&minor));
    if (j - 1) % 2 == 0 {
      result = eval_add(&result, &term);
    } else {
      result = eval_sub(&result, &term);
    }
  }
  result
}

/// Inverse[matrix] - matrix inverse (integer matrices → rational entries)
pub fn inverse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Inverse expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) if is_nonempty_square(&m) => m,
    _ => return Ok(matsq_unevaluated("Inverse", args)),
  };
  let n = matrix.len();

  if let Some(m) = machine_real_matrix(&matrix) {
    return Ok(if let Some(inv) = machine_inverse(&m) {
      Expr::List(
        inv
          .into_iter()
          .map(|row| {
            Expr::List(
              row.into_iter().map(Expr::Real).collect::<Vec<_>>().into(),
            )
          })
          .collect::<Vec<_>>()
          .into(),
      )
    } else {
      crate::emit_message(&format!(
        "Inverse::sing: Matrix {} is singular.",
        expr_to_string(&args[0])
      ));
      unevaluated("Inverse", args)
    });
  }

  // A machine-complex matrix: Gauss–Jordan elimination with partial
  // pivoting in complex machine arithmetic.
  if matrix
    .iter()
    .flatten()
    .any(crate::functions::math_ast::numeric_utils::contains_inexact_real)
    && let Some(m) = matrix
      .iter()
      .map(|row| {
        row
          .iter()
          .map(machine_complex_parts)
          .collect::<Option<Vec<(f64, f64)>>>()
      })
      .collect::<Option<Vec<_>>>()
  {
    return Ok(if let Some(inv) = machine_complex_inverse(&m) {
      matrix_to_expr(
        inv
          .into_iter()
          .map(|row| {
            row
            .into_iter()
            .map(|(re, im)| {
              crate::functions::math_ast::numeric_utils::machine_complex_result(
                re, im,
              )
            })
            .collect()
          })
          .collect(),
      )
    } else {
      crate::emit_message(&format!(
        "Inverse::sing: Matrix {} is singular.",
        expr_to_string(&args[0])
      ));
      unevaluated("Inverse", args)
    });
  }

  // A matrix of exact numbers: Gauss–Jordan elimination on [A | I], which
  // is polynomial where the adjugate's n^2 determinants are not.
  let exact_number = |e: &Expr| {
    matches!(e, Expr::Integer(_) | Expr::BigInteger(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };
  if matrix.iter().flatten().all(exact_number) {
    let mut aug: Vec<Vec<Expr>> = matrix
      .iter()
      .enumerate()
      .map(|(i, row)| {
        let mut r = row.clone();
        r.extend((0..n).map(|j| Expr::Integer(i128::from(i == j))));
        r
      })
      .collect();
    for col in 0..n {
      let Some(pivot_row) = (col..n).find(|&r| !is_zero_expr(&aug[r][col]))
      else {
        crate::emit_message(&format!(
          "Inverse::sing: Matrix {} is singular.",
          expr_to_string(&args[0])
        ));
        return Ok(unevaluated("Inverse", args));
      };
      aug.swap(col, pivot_row);
      let pivot = aug[col][col].clone();
      for j in col..2 * n {
        aug[col][j] = eval_divide(&aug[col][j], &pivot);
      }
      for row in 0..n {
        if row == col || is_zero_expr(&aug[row][col]) {
          continue;
        }
        let factor = aug[row][col].clone();
        for j in col..2 * n {
          if is_zero_expr(&aug[col][j]) {
            continue;
          }
          let prod = eval_mul(&factor, &aug[col][j]);
          aug[row][j] = eval_sub(&aug[row][j], &prod);
        }
      }
    }
    return Ok(matrix_to_expr(
      aug.into_iter().map(|row| row[n..].to_vec()).collect(),
    ));
  }

  // A triangular polynomial matrix with exact-number diagonal: solve by
  // substitution, which stays polynomial where the adjugate's n^2
  // symbolic determinants blow up.
  if is_polynomial_matrix(&matrix)
    && let Some(inv) = triangular_polynomial_inverse(&matrix)
  {
    return Ok(matrix_to_expr(inv));
  }

  // Try integer matrix inverse via adjugate method
  let det = determinant(&matrix);
  let det_is_zero = matches!(&det, Expr::Integer(0))
    || matches!(&det, Expr::BigInteger(n) if n == &BigInt::from(0));
  if det_is_zero {
    // Matches wolframscript: emit Inverse::sing and return unevaluated.
    crate::emit_message(&format!(
      "Inverse::sing: Matrix {} is singular.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("Inverse", args));
  }

  // Compute cofactor matrix
  let mut cofactors = vec![vec![Expr::Integer(0); n]; n];
  for i in 0..n {
    for j in 0..n {
      let mut minor = Vec::new();
      for ii in 0..n {
        if ii == i {
          continue;
        }
        let mut row = Vec::new();
        for jj in 0..n {
          if jj == j {
            continue;
          }
          row.push(matrix[ii][jj].clone());
        }
        minor.push(row);
      }
      let minor_det = determinant(&minor);
      if (i + j) % 2 == 0 {
        cofactors[i][j] = minor_det;
      } else {
        cofactors[i][j] = eval_sub(&Expr::Integer(0), &minor_det);
      }
    }
  }

  // Transpose cofactor matrix (adjugate) and divide by det. Over a
  // polynomial matrix each entry is one reduced fraction of expanded
  // polynomials, as Wolfram gives it: the denominator is then `Det`'s own
  // form, so `Inverse[m] Det[m]` cancels to the adjugate.
  // With rational coefficients the fraction is brought together instead.
  let polynomial = is_polynomial_matrix(&matrix);
  let rational = polynomial && matrix.iter().flatten().any(has_rational);
  let call = |head: &str, e: Expr| {
    crate::evaluator::evaluate_function_call_ast(head, std::slice::from_ref(&e))
      .unwrap_or(e)
  };
  let det = if polynomial && !rational {
    call("Expand", det)
  } else {
    det
  };
  let mut result = vec![vec![Expr::Integer(0); n]; n];
  for i in 0..n {
    for j in 0..n {
      result[i][j] = if rational {
        call("Together", eval_divide(&cofactors[j][i], &det))
      } else if polynomial {
        eval_divide(&call("Expand", cofactors[j][i].clone()), &det)
      } else {
        eval_divide(&cofactors[j][i], &det)
      };
    }
  }

  Ok(matrix_to_expr(result))
}

/// The inverse of an upper- or lower-triangular matrix whose diagonal
/// entries are all nonzero exact numbers, by back substitution with every
/// entry expanded; `None` when the matrix has another shape.
fn triangular_polynomial_inverse(m: &[Vec<Expr>]) -> Option<Vec<Vec<Expr>>> {
  let n = m.len();
  let exact_number = |e: &Expr| {
    matches!(e, Expr::Integer(_) | Expr::BigInteger(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };
  if !(0..n).all(|i| exact_number(&m[i][i]) && !is_zero_expr(&m[i][i])) {
    return None;
  }
  let upper = (0..n).all(|i| (0..i).all(|j| is_zero_expr(&m[i][j])));
  let lower = (0..n).all(|i| (i + 1..n).all(|j| is_zero_expr(&m[i][j])));
  if !upper && !lower {
    return None;
  }
  // Work on the upper-triangular form; a lower one is solved transposed.
  let u: Vec<Vec<Expr>> = if upper {
    m.to_vec()
  } else {
    (0..n)
      .map(|i| (0..n).map(|j| m[j][i].clone()).collect())
      .collect()
  };
  let expand = |e: Expr| {
    crate::evaluator::evaluate_function_call_ast(
      "Expand",
      std::slice::from_ref(&e),
    )
    .unwrap_or(e)
  };
  let mut x = vec![vec![Expr::Integer(0); n]; n];
  for j in 0..n {
    x[j][j] = eval_divide(&Expr::Integer(1), &u[j][j]);
    for i in (0..j).rev() {
      let mut sum = Expr::Integer(0);
      for k in i + 1..=j {
        if is_zero_expr(&u[i][k]) || is_zero_expr(&x[k][j]) {
          continue;
        }
        sum = eval_add(&sum, &eval_mul(&u[i][k], &x[k][j]));
      }
      let neg = eval_sub(&Expr::Integer(0), &sum);
      x[i][j] = expand(eval_divide(&neg, &u[i][i]));
    }
  }
  if upper {
    Some(x)
  } else {
    Some(
      (0..n)
        .map(|i| (0..n).map(|j| x[j][i].clone()).collect())
        .collect(),
    )
  }
}

/// The inverse of a complex machine matrix by Gauss–Jordan elimination
/// with partial pivoting; `None` when it is singular.
fn machine_complex_inverse(
  m: &[Vec<(f64, f64)>],
) -> Option<Vec<Vec<(f64, f64)>>> {
  type C = (f64, f64);
  let mul = |a: C, b: C| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
  let div = |a: C, b: C| {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
  };
  let n = m.len();
  let mut aug: Vec<Vec<C>> = m
    .iter()
    .enumerate()
    .map(|(i, row)| {
      let mut r = row.clone();
      r.extend((0..n).map(|j| (if i == j { 1.0 } else { 0.0 }, 0.0)));
      r
    })
    .collect();
  let abs = |c: C| c.0.hypot(c.1);
  let scale = m.iter().flatten().fold(0.0f64, |acc, &c| acc.max(abs(c)));
  for col in 0..n {
    let pivot_row = (col..n)
      .max_by(|&a, &b| abs(aug[a][col]).total_cmp(&abs(aug[b][col])))?;
    if abs(aug[pivot_row][col]) <= scale * 1e-15 * n as f64 {
      return None;
    }
    aug.swap(col, pivot_row);
    let pivot = aug[col][col];
    for j in col..2 * n {
      aug[col][j] = div(aug[col][j], pivot);
    }
    for row in 0..n {
      if row == col {
        continue;
      }
      let factor = aug[row][col];
      if factor == (0.0, 0.0) {
        continue;
      }
      for j in col..2 * n {
        let p = mul(factor, aug[col][j]);
        aug[row][j] = (aug[row][j].0 - p.0, aug[row][j].1 - p.1);
      }
    }
  }
  Some(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// Helper: divide two expressions, producing Rational if needed
fn eval_divide(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) if *y != 0 => {
      if x % y == 0 {
        Expr::Integer(x / y)
      } else {
        // Simplify the fraction
        let (n, d) = rat_reduce(*x, *y);
        crate::functions::math_ast::make_rational(n, d)
      }
    }
    _ => {
      match crate::functions::math_ast::divide_ast(&[a.clone(), b.clone()]) {
        Ok(r) => r,
        Err(_) => div2(a.clone(), b.clone()),
      }
    }
  }
}

/// The rectangular shape of a nested-List array, found by descending through
/// the first element of each level. A non-list has an empty shape; `{a, b}`
/// gives `[2]`; `{{a, b}}` gives `[1, 2]`.
fn array_shape(expr: &Expr) -> Vec<usize> {
  let mut shape = Vec::new();
  let mut cur = expr;
  while let Expr::List(items) = cur {
    shape.push(items.len());
    match items.first() {
      Some(first) => cur = first,
      None => break,
    }
  }
  shape
}

/// The main-diagonal element `array[[i, i, ..., i]]` (index `i` repeated
/// `depth` times). Returns None if an index is out of range or the array is
/// too shallow (ragged input).
fn diagonal_element(expr: &Expr, i: usize, depth: usize) -> Option<Expr> {
  if depth == 0 {
    return Some(expr.clone());
  }
  match expr {
    Expr::List(items) => {
      items.get(i).and_then(|e| diagonal_element(e, i, depth - 1))
    }
    _ => None,
  }
}

/// Tr[array] - the trace of an array: the sum of its main-diagonal entries
/// `array[[i, i, ..., i]]` for i = 1..min(dimensions). For a rank-1 vector
/// this is the sum of all entries; for a matrix it is the usual trace. With a
/// second argument f, the combining head f is used in place of Plus.
pub fn tr_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Tr expects 1, 2, or 3 arguments".into(),
    ));
  }

  // A SparseArray argument is traced over its dense form.
  if let Expr::FunctionCall { name, args: sa } = &args[0]
    && name == "SparseArray"
  {
    let dense = crate::functions::list_helpers_ast::sparse_array_ast(sa)?;
    if !matches!(&dense, Expr::FunctionCall { name, .. } if name == "SparseArray")
    {
      let mut new_args = args.to_vec();
      new_args[0] = dense;
      return tr_ast(&new_args);
    }
  }

  let unevaluated = || unevaluated("Tr", args);

  // The combining function f (default: Plus) is applied to the whole
  // sequence of diagonal elements at once as f[d1, d2, ..., dn], not
  // pairwise — so Tr[m, List] returns a list of the diagonal.
  let combine_name: String = if args.len() >= 2 {
    match &args[1] {
      Expr::Identifier(name) => name.clone(),
      _ => return Ok(unevaluated()),
    }
  } else {
    "Plus".to_string()
  };

  // Optional level n: the generalized trace runs over the first n indices,
  // combining the resulting slices. n is clamped to the tensor rank; a zero
  // level is rejected with Tr::intnz, matching wolframscript.
  let level: Option<usize> = if args.len() == 3 {
    match expr_to_i128(&args[2]) {
      Some(n) if n >= 1 => Some(n as usize),
      Some(_) => {
        crate::emit_message(&format!(
          "Tr::intnz: Nonzero integer expected at position 3 in {}.",
          expr_to_string(&unevaluated())
        ));
        return Ok(unevaluated());
      }
      None => return Ok(unevaluated()),
    }
  } else {
    None
  };

  // Collect the diagonal entries (or, for a plain vector, all entries).
  let diag: Vec<Expr> = if let Some(vec) = expr_to_vector(&args[0]) {
    // A rank-1 vector: the diagonal over any (clamped) level is every entry.
    vec
  } else if matches!(&args[0], Expr::List(_)) {
    // Rank >= 2 array: the main diagonal a[[i, i, ..., i]] over the first
    // `n` levels (default: all levels), i = 1..min(first n dimensions). For
    // n < rank each diagonal entry is the remaining sub-array (slice). A
    // rank-2 matrix with the default level reduces to the usual trace.
    let shape = array_shape(&args[0]);
    if shape.len() < 2 {
      return Ok(unevaluated());
    }
    let rank = shape.len();
    let depth = level.unwrap_or(rank).min(rank);
    let min_dim = *shape[..depth].iter().min().unwrap();
    let mut d = Vec::with_capacity(min_dim);
    for i in 0..min_dim {
      match diagonal_element(&args[0], i, depth) {
        Some(e) => d.push(e),
        None => return Ok(unevaluated()),
      }
    }
    d
  } else {
    return Ok(unevaluated());
  };

  if diag.is_empty() {
    // Plus[] → 0, Times[] → 1, List[] → {} — let the evaluator decide.
    return crate::evaluator::evaluate_function_call_ast(&combine_name, &[]);
  }

  crate::evaluator::evaluate_function_call_ast(&combine_name, &diag)
}

/// IdentityMatrix[n] - n×n identity matrix.
/// IdentityMatrix[{m, n}] - m×n identity matrix (1s on leading diagonal).
pub fn identity_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "IdentityMatrix expects exactly 1 argument".into(),
    ));
  }
  fn extract_size(expr: &Expr) -> Option<usize> {
    match expr {
      Expr::Integer(n) if *n >= 0 => Some(*n as usize),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        use num_traits::Zero;
        if *n >= BigInt::zero() {
          n.to_usize()
        } else {
          None
        }
      }
      _ => None,
    }
  }
  let unevaluated = || unevaluated("IdentityMatrix", args);
  let (m, n) = match &args[0] {
    Expr::List(items) if items.len() == 2 => {
      match (extract_size(&items[0]), extract_size(&items[1])) {
        (Some(m), Some(n)) => (m, n),
        _ => return Ok(unevaluated()),
      }
    }
    other => match extract_size(other) {
      Some(n) => (n, n),
      None => return Ok(unevaluated()),
    },
  };
  let mut matrix = Vec::with_capacity(m);
  for i in 0..m {
    let mut row = vec![Expr::Integer(0); n];
    if i < n {
      row[i] = Expr::Integer(1);
    }
    matrix.push(Expr::List(row.into()));
  }
  Ok(Expr::List(matrix.into()))
}

/// PermutationMatrix[perm] - the permutation matrix of a permutation, given as
/// a permutation list (perm[i] = image of i) or a Cycles object. The result is
/// the n x n matrix with a 1 at (i, perm[i]) and 0 elsewhere. (wolframscript
/// returns a sparse StructuredArray here; Woxi returns the dense matrix, like
/// AdjacencyMatrix, so Normal[...] agrees with wolframscript.)
pub fn permutation_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("PermutationMatrix", args));
  // PermutationMatrix[perm, n] pads perm with fixed points to length n.
  let size = match args {
    [_] => None,
    [_, Expr::Integer(n)] if *n >= 0 => Some(*n as usize),
    _ => return unevaluated(),
  };
  // Accept a permutation list directly, or convert a Cycles object via
  // PermutationList.
  let mut perm_list = match &args[0] {
    Expr::List(items) => items.iter().cloned().collect::<Vec<_>>(),
    cycles @ Expr::FunctionCall { name, .. } if name == "Cycles" => {
      let converted = crate::evaluator::evaluate_expr_to_expr(&call(
        "PermutationList",
        vec![cycles.clone()],
      ))?;
      match &converted {
        Expr::List(items) => items.iter().cloned().collect(),
        _ => return unevaluated(),
      }
    }
    _ => return unevaluated(),
  };
  if let Some(size) = size {
    if size < perm_list.len() {
      return unevaluated();
    }
    for k in perm_list.len()..size {
      perm_list.push(Expr::Integer(k as i128 + 1));
    }
  }
  // Every entry must be an integer in 1..=n and no two the same, or the
  // matrix would have a repeated column. wolframscript reports nothing
  // for a list that is not a permutation, it just declines.
  let n = perm_list.len();
  let Some(perm) =
    crate::evaluator::dispatch::list_operations::permutation_list_indices(
      "PermutationMatrix",
      &Expr::List(perm_list.clone().into()),
      false,
    )
  else {
    return unevaluated();
  };
  let mut matrix = Vec::with_capacity(n);
  for &p in &perm {
    let mut row = vec![Expr::Integer(0); n];
    row[p] = Expr::Integer(1);
    matrix.push(Expr::List(row.into()));
  }
  Ok(Expr::List(matrix.into()))
}

/// Where a companion matrix keeps its negated coefficients.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CoefficientPlacement {
  /// First column, coefficients reversed; ones on the superdiagonal.
  Left,
  /// Last column (the default); ones on the subdiagonal.
  Right,
  /// First row, coefficients reversed; ones on the subdiagonal.
  Top,
  /// Last row; ones on the superdiagonal.
  Bottom,
}

impl CoefficientPlacement {
  fn from_expr(e: &Expr) -> Option<Self> {
    match e {
      Expr::Identifier(name) => match name.as_str() {
        "Left" => Some(Self::Left),
        "Right" => Some(Self::Right),
        "Top" => Some(Self::Top),
        "Bottom" => Some(Self::Bottom),
        _ => None,
      },
      _ => None,
    }
  }
}

/// CompanionMatrix[cvec] / CompanionMatrix[cvec, dir] /
/// CompanionMatrix[poly, var] / CompanionMatrix[poly, var, dir]
///
/// The n x n companion matrix of the monic polynomial
/// `x^n + c_{n-1} x^{n-1} + … + c_1 x + c_0`, whose coefficients come either
/// from the vector `{c0, …, c_{n-1}}` or from an explicit polynomial in `var`,
/// which is divided through by its leading coefficient first (so
/// `CompanionMatrix[2 x^2 + 3 x + 1, x]` is `{{0, -1/2}, {1, -3/2}}`).
///
/// `dir` says where the negated coefficients sit, the ones filling the
/// diagonal beside them. `Right` — the default — runs `-c` down the last
/// column with ones on the subdiagonal; `Bottom` is its transpose, `-c` along
/// the last row with ones on the superdiagonal; `Left` is `Right` turned
/// through half a turn, putting the reversed coefficients in the first column;
/// and `Top` is the transpose of `Left`.
pub fn companion_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("CompanionMatrix", args));
  let render =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);
  // A list first argument is a coefficient vector, so the only other argument
  // it takes is the placement; anything else is an explicit polynomial whose
  // variable follows it.
  let (variable, placement_arg) = match (&args[0], args.len()) {
    (Expr::List(_), 1) => (None, None),
    (Expr::List(_), 2) => (None, Some(&args[1])),
    (Expr::List(_), _) => return unevaluated(),
    (_, 2) => (Some(&args[1]), None),
    (_, 3) => (Some(&args[1]), Some(&args[2])),
    _ => return unevaluated(),
  };
  let placement = match placement_arg {
    None => CoefficientPlacement::Right,
    Some(e) => {
      if let Some(p) = CoefficientPlacement::from_expr(e) {
        p
      } else {
        crate::emit_message(&format!(
          "CompanionMatrix::plspecc: Specification {} for placement of \
         coefficients must be Top, Bottom, Left or Right.",
          render(e)
        ));
        return unevaluated();
      }
    }
  };

  let coeffs: Vec<Expr> = match variable {
    None => match &args[0] {
      Expr::List(c) => c.to_vec(),
      _ => return unevaluated(),
    },
    Some(var) => {
      let clorpoly = || {
        crate::emit_message(&format!(
          "CompanionMatrix::clorpoly: Argument {} is neither a non-empty list \
           of coefficients nor an explicit polynomial in a given variable.",
          render(&args[0])
        ));
      };
      let list = crate::functions::polynomial_ast::coefficient_list_ast(&[
        args[0].clone(),
        var.clone(),
      ])?;
      // A non-polynomial leaves CoefficientList unevaluated, and a constant
      // gives a single coefficient — neither has a companion matrix.
      let Expr::List(c) = &list else {
        clorpoly();
        return unevaluated();
      };
      if c.len() < 2 {
        clorpoly();
        return unevaluated();
      }
      // Divide through by the leading coefficient to make the polynomial
      // monic, then drop it: a degree-n polynomial has an n x n matrix.
      let lead = c.last().expect("checked non-empty");
      let mut monic = Vec::with_capacity(c.len() - 1);
      for coeff in &c[..c.len() - 1] {
        monic.push(evaluate_expr_to_expr(&div(coeff.clone(), lead.clone()))?);
      }
      monic
    }
  };

  // An empty coefficient vector gives the single empty row wolframscript
  // reports for `CompanionMatrix[{}]`, not an empty matrix.
  let n = coeffs.len();
  if n == 0 {
    return Ok(Expr::List(vec![Expr::List(vec![].into())].into()));
  }
  // Build the Right form, then derive the others from it.
  let mut matrix: Vec<Vec<Expr>> = Vec::with_capacity(n);
  for (i, c) in coeffs.iter().enumerate() {
    let mut row = vec![Expr::Integer(0); n];
    if i >= 1 {
      row[i - 1] = Expr::Integer(1); // subdiagonal
    }
    row[n - 1] = evaluate_expr_to_expr(&neg(c.clone()))?;
    matrix.push(row);
  }
  if matches!(
    placement,
    CoefficientPlacement::Left | CoefficientPlacement::Top
  ) {
    // Half a turn: reverse the rows and each row.
    matrix.reverse();
    for row in &mut matrix {
      row.reverse();
    }
  }
  if matches!(
    placement,
    CoefficientPlacement::Top | CoefficientPlacement::Bottom
  ) {
    matrix = transpose_matrix(&matrix);
  }
  Ok(Expr::List(
    matrix
      .into_iter()
      .map(|row| Expr::List(row.into()))
      .collect(),
  ))
}

/// VandermondeMatrix[{x1, ..., xn}] - the n x n Vandermonde matrix with entry
/// (i, j) equal to xi^(j-1). (wolframscript returns a sparse StructuredArray;
/// Woxi returns the dense matrix, so Normal[...] agrees with wolframscript.)
pub fn vandermonde_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("VandermondeMatrix", args));
  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }
  let points: Vec<Expr> = match &args[0] {
    Expr::List(p) if !p.is_empty() => p.iter().cloned().collect(),
    _ => return unevaluated(),
  };
  // VandermondeMatrix[v, m]: m columns (powers 0 .. m-1).
  let n = match args.get(1) {
    None => points.len(),
    Some(Expr::Integer(m)) if *m >= 1 => *m as usize,
    Some(_) => return unevaluated(),
  };
  let mut matrix = Vec::with_capacity(points.len());
  for x in &points {
    let mut row = Vec::with_capacity(n);
    for j in 0..n {
      let entry = match j {
        0 => Expr::Integer(1),
        1 => x.clone(),
        _ => crate::evaluator::evaluate_expr_to_expr(&pow(
          x.clone(),
          Expr::Integer(j as i128),
        ))?,
      };
      row.push(entry);
    }
    matrix.push(Expr::List(row.into()));
  }
  Ok(Expr::List(matrix.into()))
}

/// DiagonalMatrix[list] - diagonal matrix from a list
pub fn diagonal_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "DiagonalMatrix expects 1, 2 or 3 arguments".into(),
    ));
  }
  let diag = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(unevaluated("DiagonalMatrix", args));
    }
  };
  let unevaluated = || Ok(unevaluated("DiagonalMatrix", args));
  let k: i128 = if args.len() >= 2 {
    match &args[1] {
      Expr::Integer(v) => *v,
      _ => return unevaluated(),
    }
  } else {
    0
  };
  let n = diag.len();
  let k_abs = k.unsigned_abs() as usize;
  // DiagonalMatrix[list, k, m] forces an m×m square matrix instead of the
  // auto-sized (len + |k|); entries that fall outside the m×m grid are
  // dropped. wolframscript: DiagonalMatrix[{1,2,3}, 0, 4] pads to 4×4.
  let size = if args.len() == 3 {
    match &args[2] {
      Expr::Integer(m) if *m >= 0 => *m as usize,
      _ => return unevaluated(),
    }
  } else {
    n + k_abs
  };
  // An inexact diagonal makes the whole matrix inexact: the off-diagonal
  // fill is the machine real 0. rather than the exact Integer 0, and any
  // exact numeric diagonal entry is promoted to a Real too
  // (DiagonalMatrix[{1, 2.}] = {{1., 0.}, {0., 2.}}). Symbolic entries are
  // left untouched (DiagonalMatrix[{a, 2.}] = {{a, 0.}, {0., 2.}}).
  let inexact = diag.iter().any(|e| matches!(e, Expr::Real(_)));
  let zero = if inexact {
    Expr::Real(0.0)
  } else {
    Expr::Integer(0)
  };
  let is_exact_number = |e: &Expr| -> bool {
    match e {
      Expr::Integer(_) | Expr::BigInteger(_) => true,
      Expr::FunctionCall { name, .. } => name == "Rational",
      _ => false,
    }
  };
  let promote = |e: &Expr| -> Expr {
    if inexact
      && is_exact_number(e)
      && let Some(v) = crate::functions::math_ast::expr_to_f64(e)
    {
      return Expr::Real(v);
    }
    e.clone()
  };
  let mut matrix = Vec::with_capacity(size);
  for i in 0..size {
    let mut row = vec![zero.clone(); size];
    if k >= 0 {
      // Super-diagonal: element j goes at row j, column j + k. Bounds-check
      // the column so an explicit size that clips the diagonal is honoured.
      if i < n && i + k_abs < size {
        row[i + k_abs] = promote(&diag[i]);
      }
    } else {
      // Sub-diagonal: element j goes at row j + |k|, column j
      if i >= k_abs && i - k_abs < n {
        row[i - k_abs] = promote(&diag[i - k_abs]);
      }
    }
    matrix.push(Expr::List(row.into()));
  }
  Ok(Expr::List(matrix.into()))
}

/// DiamondMatrix[n] - creates (2n+1)x(2n+1) matrix where entry (i,j) is 1 if
/// the Manhattan distance from center <= n.
pub fn diamond_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(unevaluated("DiamondMatrix", args));
    }
  };
  let size = 2 * n + 1;
  let mut matrix = Vec::with_capacity(size);
  for i in 0..size {
    let mut row = Vec::with_capacity(size);
    for j in 0..size {
      let dist = (i as i128 - n as i128).unsigned_abs()
        + (j as i128 - n as i128).unsigned_abs();
      row.push(if dist <= n as u128 {
        Expr::Integer(1)
      } else {
        Expr::Integer(0)
      });
    }
    matrix.push(Expr::List(row.into()));
  }
  Ok(Expr::List(matrix.into()))
}

/// DiskMatrix[n] - creates (2n+1)x(2n+1) matrix where entry (i,j) is 1 if
/// the Euclidean distance from center <= n.
pub fn disk_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(unevaluated("DiskMatrix", args));
    }
  };
  let size = 2 * n + 1;
  // Threshold is (n + 0.5)^2 = n^2 + n + 0.25; for integer comparisons, n^2 + n
  let threshold = (n * n + n) as i128;
  let mut matrix = Vec::with_capacity(size);
  for i in 0..size {
    let mut row = Vec::with_capacity(size);
    for j in 0..size {
      let di = i as i128 - n as i128;
      let dj = j as i128 - n as i128;
      row.push(if di * di + dj * dj <= threshold {
        Expr::Integer(1)
      } else {
        Expr::Integer(0)
      });
    }
    matrix.push(Expr::List(row.into()));
  }
  Ok(Expr::List(matrix.into()))
}

/// Cross[{x, y}] - 2D cross product (perpendicular vector)
/// Cross[a, b] - cross product of two 3-vectors
pub fn cross_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Cross[{x, y}] = {-y, x}
    if let Expr::List(items) = &args[0] {
      if items.len() == 2 {
        let neg_y = eval_sub(&Expr::Integer(0), &items[1]);
        return Ok(Expr::List(vec![neg_y, items[0].clone()].into()));
      }
      // An explicit list of the wrong length emits ::nonn1.
      crate::emit_message(
        "Cross::nonn1: The arguments are expected to be vectors of equal length, and the number of arguments is expected to be one less than their length.",
      );
    }
    return Ok(unevaluated("Cross", args));
  }
  // Generalized cross product: k = n - 1 vectors of length n give the vector
  // orthogonal to all of them, with component i (1-indexed) equal to
  // (-1)^(n+i) Det[Mi], where Mi is the k×k matrix of the vectors as rows with
  // column i removed. (The 1- and 2-argument cases above are special cases of
  // this, kept separate to preserve their exact output form.)
  if args.len() >= 3 {
    // All arguments must be explicit lists of equal length n with k = n - 1;
    // otherwise emit ::nonn1 for all-list inputs, else stay unevaluated.
    let mut vecs: Vec<&[Expr]> = Vec::with_capacity(args.len());
    for a in args {
      match a {
        Expr::List(items) => vecs.push(items.as_slice()),
        _ => {
          return Ok(unevaluated("Cross", args));
        }
      }
    }
    let n = vecs[0].len();
    let valid = n == args.len() + 1 && vecs.iter().all(|v| v.len() == n);
    if !valid {
      crate::emit_message(
        "Cross::nonn1: The arguments are expected to be vectors of equal length, and the number of arguments is expected to be one less than their length.",
      );
      return Ok(unevaluated("Cross", args));
    }
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
      // Minor matrix: vectors as rows, with column i removed.
      let minor: Vec<Expr> = vecs
        .iter()
        .map(|v| {
          let row: Vec<Expr> = v
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, e)| e.clone())
            .collect();
          Expr::List(row.into())
        })
        .collect();
      let det = evaluate_expr_to_expr(&call1("Det", Expr::List(minor.into())))?;
      // Sign (-1)^(n + (i+1)).
      let component = if (n + i + 1).is_multiple_of(2) {
        det
      } else {
        evaluate_expr_to_expr(&neg(det))?
      };
      result.push(component);
    }
    return Ok(Expr::List(result.into()));
  }

  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Cross expects 1 or 2 arguments".into(),
    ));
  }
  // Explicit lists that are not two 3-vectors emit ::nonn1; symbolic
  // arguments stay silently unevaluated.
  let (va, vb) = match (&args[0], &args[1]) {
    (Expr::List(a), Expr::List(b)) if a.len() == 3 && b.len() == 3 => {
      (a.clone(), b.clone())
    }
    (Expr::List(_), Expr::List(_)) => {
      crate::emit_message(
        "Cross::nonn1: The arguments are expected to be vectors of equal length, and the number of arguments is expected to be one less than their length.",
      );
      return Ok(unevaluated("Cross", args));
    }
    _ => {
      return Ok(unevaluated("Cross", args));
    }
  };
  Ok(Expr::List(
    vec![
      eval_sub(&eval_mul(&va[1], &vb[2]), &eval_mul(&va[2], &vb[1])),
      eval_sub(&eval_mul(&va[2], &vb[0]), &eval_mul(&va[0], &vb[2])),
      eval_sub(&eval_mul(&va[0], &vb[1]), &eval_mul(&va[1], &vb[0])),
    ]
    .into(),
  ))
}

/// ConjugateTranspose[matrix] - transpose and conjugate each element
pub fn conjugate_transpose_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ConjugateTranspose expects exactly 1 argument".into(),
    ));
  }
  let Expr::List(rows) = &args[0] else {
    return Ok(unevaluated("ConjugateTranspose", args));
  };

  // Get dimensions
  let nrows = rows.len();
  if nrows == 0 {
    return Ok(Expr::List(vec![].into()));
  }
  let ncols = match &rows[0] {
    Expr::List(cols) => cols.len(),
    _ => {
      return Ok(unevaluated("ConjugateTranspose", args));
    }
  };

  // Transpose and conjugate
  let mut result = Vec::with_capacity(ncols);
  for j in 0..ncols {
    let mut row = Vec::with_capacity(nrows);
    for item in rows.iter().take(nrows) {
      if let Expr::List(cols) = item
        && j < cols.len()
      {
        // Apply Conjugate
        let conj = evaluate_expr_to_expr(&call1("Conjugate", cols[j].clone()))?;
        row.push(conj);
      }
    }
    result.push(Expr::List(row.into()));
  }
  Ok(Expr::List(result.into()))
}

/// Projection[u, v] - project vector u onto vector v.
/// Formula uses conjugate inner product (matches Wolfram):
///   (Conjugate[v].u / Conjugate[v].v) * v
pub fn projection_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Projection[u, v, f] uses a custom inner product f:
  //   (v * f[v, u]) / f[v, v]
  // (for a list v this threads componentwise). f[v, u] puts v first, matching
  // wolframscript.
  if args.len() == 3 {
    let u = args[0].clone();
    let v = args[1].clone();
    let f = &args[2];
    let apply = crate::evaluator::function_application::apply_curried_call;
    let fvu = apply(f, &[v.clone(), u])?;
    let fvv = apply(f, &[v.clone(), v.clone()])?;
    let scalar = div2(fvu, fvv);
    let result = times2(scalar, v);
    return evaluate_expr_to_expr(&result);
  }
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Projection expects 2 or 3 arguments".into(),
    ));
  }
  // Symbolic case (neither argument is a literal list): expand to the
  // Hermitian projection formula
  //   (Conjugate[v] . u / Conjugate[v] . v) * v
  // matching Wolfram, which prints it as `(v*Conjugate[v] . u)/Conjugate[v] . v`.
  if !matches!(&args[0], Expr::List(_)) && !matches!(&args[1], Expr::List(_)) {
    let u = args[0].clone();
    let v = args[1].clone();
    let conj_v = call1("Conjugate", v.clone());
    let dot_cv_u = call("Dot", vec![conj_v.clone(), u]);
    let dot_cv_v = call("Dot", vec![conj_v, v.clone()]);
    let scalar = div2(dot_cv_u, dot_cv_v);
    let result = times2(scalar, v);
    return evaluate_expr_to_expr(&result);
  }
  let u = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(unevaluated("Projection", args));
    }
  };
  let v = match &args[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(unevaluated("Projection", args));
    }
  };
  // Projection[u, {v1, v2, …}]: the orthogonal projection onto the span of
  // the vi, i.e. the sum of the projections onto a Gram-Schmidt basis of it
  // (vectors already in the span of the earlier ones are dropped).
  if !u.iter().any(|ui| matches!(ui, Expr::List(_)))
    && !v.is_empty()
    && v
      .iter()
      .all(|vi| matches!(vi, Expr::List(row) if row.len() == u.len()))
  {
    let is_zero_vector = |w: &Expr| match w {
      Expr::List(items) => items.iter().all(|x| {
        matches!(x, Expr::Integer(0))
          || matches!(x, Expr::Real(r) if *r == 0.0)
          || try_eval_to_f64(x) == Some(0.0)
      }),
      _ => false,
    };
    let mut basis: Vec<Expr> = Vec::new();
    for vi in &v {
      let mut w = vi.clone();
      for b in &basis {
        let p = projection_ast(&[vi.clone(), b.clone()])?;
        w = evaluate_expr_to_expr(&call("Subtract", vec![w, p]))?;
      }
      if !is_zero_vector(&w) {
        basis.push(w);
      }
    }
    let mut total = Expr::List(vec![Expr::Integer(0); u.len()].into());
    for b in &basis {
      let p = projection_ast(&[args[0].clone(), b.clone()])?;
      total = evaluate_expr_to_expr(&call("Plus", vec![total, p]))?;
    }
    return Ok(total);
  }
  if u.len() != v.len() {
    return Err(InterpreterError::EvaluationError(
      "Projection: vectors have incompatible lengths".into(),
    ));
  }
  let conj_v: Vec<Expr> = v
    .iter()
    .map(|vi| evaluate_expr_to_expr(&call1("Conjugate", vi.clone())))
    .collect::<Result<_, _>>()?;
  let mut cvu = Expr::Integer(0);
  for (cv, ui) in conj_v.iter().zip(u.iter()) {
    cvu = eval_add(&cvu, &eval_mul(cv, ui));
  }
  let mut cvv = Expr::Integer(0);
  for (cv, vi) in conj_v.iter().zip(v.iter()) {
    cvv = eval_add(&cvv, &eval_mul(cv, vi));
  }
  let scalar = eval_divide(&cvu, &cvv);
  let result: Vec<Expr> = v.iter().map(|vi| eval_mul(&scalar, vi)).collect();
  Ok(Expr::List(result.into()))
}

// ─── Fit (least-squares) ────────────────────────────────────────────────

/// Fit[data, funs, var] — least-squares fit of data to a linear combination
/// of the basis functions `funs` in the variable `var`.
///
/// `data` is either:
///   - `{y1, y2, …}` with implicit x = 1, 2, …, n
///   - `{{x1, y1}, {x2, y2}, …}` with explicit x-y pairs
///
/// Returns `c1*f1 + c2*f2 + … + cm*fm`.
pub fn fit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Fit expects exactly 3 arguments".into(),
    ));
  }

  // A variable need not be a symbol (`#`, `Subscript[x, 1]`): each other
  // one is fitted as a placeholder symbol and put back afterwards.
  let vars: Vec<Expr> = match &args[2] {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  if vars
    .iter()
    .any(|v| !matches!(v, Expr::Identifier(_)) && try_eval_to_f64(v).is_none())
  {
    let mut basis = args[1].clone();
    let mut placeholders = Vec::new();
    let new_vars: Vec<Expr> = vars
      .iter()
      .enumerate()
      .map(|(k, v)| {
        if matches!(v, Expr::Identifier(_)) {
          return v.clone();
        }
        let name = format!("Woxi`FitVariable{}", k + 1);
        basis = crate::functions::polynomial_ast::solve::substitute_expr(
          &basis,
          v,
          &Expr::Identifier(name.clone()),
        );
        placeholders.push((name.clone(), v.clone()));
        Expr::Identifier(name)
      })
      .collect();
    let var_arg = if matches!(&args[2], Expr::List(_)) {
      Expr::List(new_vars.into())
    } else {
      new_vars
        .into_iter()
        .next()
        .unwrap_or(Expr::List(vec![].into()))
    };
    let fitted = fit_ast(&[args[0].clone(), basis, var_arg])?;
    return Ok(placeholders.iter().fold(fitted, |e, (name, var)| {
      crate::syntax::substitute_variable(&e, name, var)
    }));
  }

  // Extract basis functions
  let basis = match &args[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Fit: second argument must be a list of basis functions".into(),
      ));
    }
  };

  // The variable, or a list of them for rows `{x1, …, xn, f}`.
  let var_names: Vec<String> = match &args[2] {
    Expr::Identifier(name) => vec![name.clone()],
    Expr::List(items)
      if !items.is_empty()
        && items.iter().all(|v| matches!(v, Expr::Identifier(_))) =>
    {
      items
        .iter()
        .filter_map(|v| match v {
          Expr::Identifier(name) => Some(name.clone()),
          _ => None,
        })
        .collect()
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Fit: third argument must be a variable".into(),
      ));
    }
  };

  // Extract data points: either {y1, y2, ...} or {{x1, y1}, {x2, y2}, ...}
  let data_list = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Fit: first argument must be a non-empty list".into(),
      ));
    }
  };

  // Responses that are not numbers (`a`, `0.5 ± 0.05`) are fitted as the
  // linear combinations of them the estimates are.
  if let Some(fitted) = fit_symbolic_responses(data_list, &basis, &var_names)? {
    return Ok(fitted);
  }

  let (x_vals, y_vals) =
    extract_fit_data_multi(data_list, var_names.len(), "Fit")?;
  let n = x_vals.len(); // number of data points
  let m = basis.len(); // number of basis functions

  // Build design matrix A (n×m) where A[i][j] = basis[j] evaluated at x = x_vals[i]
  let mut a_matrix = vec![vec![0.0f64; m]; n];
  for i in 0..n {
    for j in 0..m {
      let mut substituted = basis[j].clone();
      for (name, v) in var_names.iter().zip(&x_vals[i]) {
        substituted = crate::syntax::substitute_variable(
          &substituted,
          name,
          &Expr::Real(*v),
        );
      }
      let evaluated = evaluate_expr_to_expr(&substituted)?;
      match try_eval_to_f64(&evaluated) {
        Some(v) => a_matrix[i][j] = v,
        None => {
          return Err(InterpreterError::EvaluationError(format!(
            "Fit: could not evaluate basis function {:?} at x = {:?}",
            basis[j], x_vals[i]
          )));
        }
      }
    }
  }

  // Solve via QR decomposition (Householder reflections)
  let coeffs = solve_least_squares_qr(&a_matrix, &y_vals)?;

  // Build result: c[0]*basis[0] + c[1]*basis[1] + ...
  // We construct the expression using Times and Plus function calls
  // so the result matches Wolfram's output format.
  let mut terms = Vec::new();
  for (j, c) in coeffs.iter().enumerate() {
    let coeff_expr = Expr::Real(*c);
    // Check if basis function is just the constant 1
    if matches!(&basis[j], Expr::Integer(1)) {
      terms.push(coeff_expr);
    } else {
      terms.push(times(vec![coeff_expr, basis[j].clone()]));
    }
  }

  // Build Plus expression
  if terms.len() == 1 {
    Ok(terms.into_iter().next().unwrap())
  } else {
    Ok(plus(terms))
  }
}

/// `Fit` of data whose coordinates are numbers but whose responses are not
/// all numbers: each estimate is a linear combination of the responses,
/// written as wolframscript does with the design matrix's columns scaled
/// to unit length, `s_j (P_j1 y_1 + … + P_jn y_n)` where `s_j` is one over
/// the length of column `j` and `P` the pseudo-inverse of the scaled
/// matrix. `None` when every response is a number (or the coordinates are
/// not).
fn fit_symbolic_responses(
  data: &[Expr],
  basis: &[Expr],
  var_names: &[String],
) -> Result<Option<Expr>, InterpreterError> {
  let k = var_names.len();
  let mut coords: Vec<Vec<f64>> = Vec::with_capacity(data.len());
  let mut responses: Vec<Expr> = Vec::with_capacity(data.len());
  for (i, row) in data.iter().enumerate() {
    match row {
      Expr::List(items) if items.len() == k + 1 => {
        let Some(xs) = items[..k].iter().map(try_eval_to_f64).collect() else {
          return Ok(None);
        };
        coords.push(xs);
        responses.push(items[k].clone());
      }
      Expr::List(_) => return Ok(None),
      y if k == 1 => {
        coords.push(vec![(i + 1) as f64]);
        responses.push(y.clone());
      }
      _ => return Ok(None),
    }
  }
  if responses.iter().all(|y| try_eval_to_f64(y).is_some()) {
    return Ok(None);
  }
  let (n, m) = (coords.len(), basis.len());
  let mut design = vec![vec![0.0f64; m]; n];
  for i in 0..n {
    for j in 0..m {
      let at = var_names
        .iter()
        .zip(&coords[i])
        .fold(basis[j].clone(), |e, (v, x)| {
          crate::syntax::substitute_variable(&e, v, &Expr::Real(*x))
        });
      let Some(value) = try_eval_to_f64(&evaluate_expr_to_expr(&at)?) else {
        return Ok(None);
      };
      design[i][j] = value;
    }
  }
  let scales: Vec<f64> = (0..m)
    .map(|j| 1.0 / design.iter().map(|r| r[j] * r[j]).sum::<f64>().sqrt())
    .collect();
  let scaled = Expr::List(
    design
      .iter()
      .map(|r| {
        Expr::List((0..m).map(|j| Expr::Real(r[j] * scales[j])).collect())
      })
      .collect(),
  );
  let pinv = pseudo_inverse_ast(&[scaled])?;
  let Expr::List(pinv_rows) = &pinv else {
    return Ok(None);
  };
  let mut terms = Vec::with_capacity(m);
  for (j, b) in basis.iter().enumerate() {
    let Expr::List(row) = &pinv_rows[j] else {
      return Ok(None);
    };
    let combination = plus(
      row
        .iter()
        .zip(&responses)
        .map(|(c, y)| times(vec![c.clone(), y.clone()]))
        .collect(),
    );
    let mut factors = vec![Expr::Real(scales[j]), combination];
    if !matches!(b, Expr::Integer(1)) {
      factors.push(b.clone());
    }
    terms.push(times(factors));
  }
  Ok(Some(evaluate_expr_to_expr(&plus(terms))?))
}

/// The model-fitting data forms that name the response explicitly, as the
/// row form `{{x11, …, y1}, …}` the fitters read:
/// - `{x1 -> y1, …}` and `{{x11, …} -> y1, …}`,
/// - `{x1, …} -> {y1, …}` and `{{x11, …}, …} -> {y1, …}`,
/// - `{{v11, v12, …}, …} -> k`, whose `k`-th column is the response.
///
/// `None` when `data` is in none of these forms (it is then used as given).
fn rule_form_fit_data(data: &Expr) -> Option<Expr> {
  let row = |x: &Expr, y: &Expr| -> Expr {
    let mut items = match x {
      Expr::List(xs) => xs.to_vec(),
      other => vec![other.clone()],
    };
    items.push(y.clone());
    Expr::List(items.into())
  };
  match data {
    Expr::List(items)
      if !items.is_empty()
        && items.iter().all(|r| matches!(r, Expr::Rule { .. })) =>
    {
      Some(Expr::List(
        items
          .iter()
          .map(|r| match r {
            Expr::Rule {
              pattern,
              replacement,
            } => row(pattern, replacement),
            _ => unreachable!(),
          })
          .collect(),
      ))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => match (pattern.as_ref(), replacement.as_ref()) {
      (Expr::List(xs), Expr::List(ys))
        if !xs.is_empty() && xs.len() == ys.len() =>
      {
        Some(Expr::List(
          xs.iter().zip(ys.iter()).map(|(x, y)| row(x, y)).collect(),
        ))
      }
      (Expr::List(rows), Expr::Integer(k)) if !rows.is_empty() => {
        let width = match &rows[0] {
          Expr::List(r) => r.len() as i128,
          _ => return None,
        };
        let k = if *k < 0 { width + k } else { k - 1 };
        if !(0..width).contains(&k) {
          return None;
        }
        let k = k as usize;
        rows
          .iter()
          .map(|r| match r {
            Expr::List(items) if items.len() as i128 == width => {
              let mut out = items.to_vec();
              let y = out.remove(k);
              out.push(y);
              Some(Expr::List(out.into()))
            }
            _ => None,
          })
          .collect::<Option<Vec<_>>>()
          .map(|rows| Expr::List(rows.into()))
      }
      _ => None,
    },
    _ => None,
  }
}

/// `args` with its data argument (the first) rewritten by
/// [`rule_form_fit_data`], or `None` when it needs no rewriting.
fn with_row_fit_data(args: &[Expr]) -> Option<Vec<Expr>> {
  let rows = rule_form_fit_data(args.first()?)?;
  let mut out = args.to_vec();
  out[0] = rows;
  Some(out)
}

/// Extract x and y values from Fit data.
/// Handles both `{y1, y2, ...}` and `{{x1, y1}, {x2, y2}, ...}` forms.
/// Like `extract_fit_data` but for `n_vars` independent variables: each
/// row is `{x1, x2, …, xn_vars, y}` (length `n_vars + 1`). Returns the
/// independent vars as a `Vec<Vec<f64>>` (one row per data point) and the
/// y-column separately. Falls back to univariate `extract_fit_data` shape
/// when `n_vars == 1` so the {y} (implicit-x) case still works.
fn extract_fit_data_multi(
  data: &[Expr],
  n_vars: usize,
  caller: &str,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), InterpreterError> {
  if n_vars == 1 {
    let (xs, ys) = extract_fit_data(data, caller)?;
    let multi = xs.into_iter().map(|x| vec![x]).collect();
    return Ok((multi, ys));
  }
  let mut x_rows = Vec::with_capacity(data.len());
  let mut ys = Vec::with_capacity(data.len());
  for item in data {
    let Expr::List(row) = item else {
      return Err(InterpreterError::EvaluationError(format!(
        "{caller}: each data row must be a list of length n_vars + 1"
      )));
    };
    if row.len() != n_vars + 1 {
      return Err(InterpreterError::EvaluationError(format!(
        "{caller}: each data row must have {} entries (got {})",
        n_vars + 1,
        row.len()
      )));
    }
    let mut xs = Vec::with_capacity(n_vars);
    for x in &row[..n_vars] {
      let v = try_eval_to_f64(x).ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "{caller}: could not convert x coordinate to a number"
        ))
      })?;
      xs.push(v);
    }
    let y = try_eval_to_f64(&row[n_vars]).ok_or_else(|| {
      InterpreterError::EvaluationError(format!(
        "{caller}: could not convert y coordinate to a number"
      ))
    })?;
    x_rows.push(xs);
    ys.push(y);
  }
  Ok((x_rows, ys))
}

fn extract_fit_data(
  data: &[Expr],
  caller: &str,
) -> Result<(Vec<f64>, Vec<f64>), InterpreterError> {
  // Check if first element is a list (=> {x, y} pair form)
  if let Expr::List(first_pair) = &data[0]
    && first_pair.len() == 2
  {
    // {{x1, y1}, {x2, y2}, ...} form
    let mut xs = Vec::with_capacity(data.len());
    let mut ys = Vec::with_capacity(data.len());
    for item in data {
      if let Expr::List(pair) = item {
        if pair.len() != 2 {
          return Err(InterpreterError::EvaluationError(format!(
            "{caller}: data points must be {{x, y}} pairs of length 2"
          )));
        }
        let x = try_eval_to_f64(&pair[0]).ok_or_else(|| {
          InterpreterError::EvaluationError(format!(
            "{caller}: could not convert x coordinate to a number"
          ))
        })?;
        let y = try_eval_to_f64(&pair[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(format!(
            "{caller}: could not convert y coordinate to a number"
          ))
        })?;
        xs.push(x);
        ys.push(y);
      } else {
        return Err(InterpreterError::EvaluationError(format!(
          "{caller}: all data entries must be {{x, y}} pairs"
        )));
      }
    }
    return Ok((xs, ys));
  }

  // {y1, y2, ...} form with implicit x = 1, 2, ..., n
  let mut ys = Vec::with_capacity(data.len());
  for item in data {
    let y = try_eval_to_f64(item).ok_or_else(|| {
      InterpreterError::EvaluationError(format!(
        "{caller}: could not convert data value to a number"
      ))
    })?;
    ys.push(y);
  }
  let xs: Vec<f64> = (1..=data.len()).map(|i| i as f64).collect();
  Ok((xs, ys))
}

/// The minimum-norm least-squares solution of `A x = b` after scaling every
/// column of `A` to unit length, via a one-sided Jacobi SVD of the scaled
/// design matrix.
///
/// This is the answer wolframscript gives when the design matrix does not have
/// full column rank — fewer data points than basis functions, repeated abscissae,
/// a basis with a linear dependency. `Fit[{{1, 1}}, {1, x}, x]` is
/// `0.5 + 0.5 x`: of the infinitely many lines through the single point, the one
/// whose (scaled) coefficient vector is shortest. The scaling makes the answer
/// independent of the units of each basis function: `Fit[{{10, 1}, {20, 3}},
/// {1, x, x^2}, x]` is `Fit[{{1, 1}, {2, 3}}, {1, x, x^2}, x]` with `x` replaced
/// by `x/10`, and a high-degree polynomial basis over large abscissae stays
/// well conditioned. A Gauss–Newton step of a nonlinear fit leaves the columns
/// unscaled (`scale_columns` false): the shortest step in the parameters
/// themselves keeps an unidentifiable model's iteration from wandering off
/// along its flat directions.
///
/// One-sided Jacobi orthogonalizes the columns of `A` by plane rotations; the
/// resulting column norms are the singular values and the accumulated rotations
/// are `V`. Padding with zero rows to make `A` tall leaves the problem alone,
/// since the added rows contribute nothing to the residual.
fn solve_least_squares_min_norm(
  a: &[Vec<f64>],
  b: &[f64],
  scale_columns: bool,
) -> Vec<f64> {
  let m = a[0].len();
  let rows = a.len().max(m);
  // Columns of A, zero-padded to `rows` and scaled to unit length,
  // alongside b.
  let column_scale: Vec<f64> = (0..m)
    .map(|j| {
      let norm = a.iter().map(|r| r[j] * r[j]).sum::<f64>().sqrt();
      if !scale_columns {
        1.0
      } else if norm > 0.0 && norm.is_finite() {
        1.0 / norm
      } else {
        0.0
      }
    })
    .collect();
  let mut u: Vec<Vec<f64>> = (0..m)
    .map(|j| {
      (0..rows)
        .map(|i| a.get(i).map_or(0.0, |r| r[j] * column_scale[j]))
        .collect()
    })
    .collect();
  let rhs: Vec<f64> = (0..rows)
    .map(|i| b.get(i).copied().unwrap_or(0.0))
    .collect();
  let mut v: Vec<Vec<f64>> = (0..m)
    .map(|j| (0..m).map(|i| if i == j { 1.0 } else { 0.0 }).collect())
    .collect();

  let dot =
    |x: &[f64], y: &[f64]| -> f64 { x.iter().zip(y).map(|(p, q)| p * q).sum() };
  // Sweep until no pair of columns needs rotating. The pair count is m(m-1)/2
  // and each sweep reduces the off-diagonal norm quadratically, so a handful of
  // sweeps suffices; the cap only bounds a pathological case.
  for _ in 0..60 {
    let mut rotated = false;
    for p in 0..m {
      for q in (p + 1)..m {
        let alpha = dot(&u[p], &u[p]);
        let beta = dot(&u[q], &u[q]);
        let gamma = dot(&u[p], &u[q]);
        if gamma.abs() <= 1e-15 * (alpha * beta).sqrt().max(f64::MIN_POSITIVE) {
          continue;
        }
        rotated = true;
        // The rotation that zeroes the (p, q) entry of the Gram matrix.
        let zeta = (beta - alpha) / (2.0 * gamma);
        let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = c * t;
        for i in 0..rows {
          let (up, uq) = (u[p][i], u[q][i]);
          u[p][i] = c * up - s * uq;
          u[q][i] = s * up + c * uq;
        }
        for i in 0..m {
          let (vp, vq) = (v[p][i], v[q][i]);
          v[p][i] = c * vp - s * vq;
          v[q][i] = s * vp + c * vq;
        }
      }
    }
    if !rotated {
      break;
    }
  }

  // x = sum over the non-degenerate singular values of (u_i . b / sigma_i) v_i.
  let sigmas: Vec<f64> = u.iter().map(|col| dot(col, col).sqrt()).collect();
  let largest = sigmas.iter().copied().fold(0.0f64, f64::max);
  let cutoff = largest * (rows.max(m) as f64) * f64::EPSILON;
  let mut x = vec![0.0f64; m];
  for j in 0..m {
    if sigmas[j] <= cutoff {
      continue;
    }
    let coefficient = dot(&u[j], &rhs) / (sigmas[j] * sigmas[j]);
    for i in 0..m {
      x[i] += coefficient * v[j][i];
    }
  }
  for (xi, s) in x.iter_mut().zip(&column_scale) {
    *xi *= s;
  }
  x
}

/// Solve least-squares problem min ||Ax - b||_2 using Householder QR decomposition.
/// A is n×m (n rows, m cols) with n >= m.  Returns the m-element solution vector.
/// A design matrix without full column rank has no unique solution, so those
/// fall through to [`solve_least_squares_min_norm`] rather than failing — that
/// used to abort the whole evaluation with a hard error.
fn solve_least_squares_qr(
  a: &[Vec<f64>],
  b: &[f64],
) -> Result<Vec<f64>, InterpreterError> {
  least_squares_qr(a, b, true)
}

/// [`solve_least_squares_qr`], with the rank-deficient fallback's column
/// scaling chosen by the caller (see [`solve_least_squares_min_norm`]).
fn least_squares_qr(
  a: &[Vec<f64>],
  b: &[f64],
  scale_columns: bool,
) -> Result<Vec<f64>, InterpreterError> {
  let n = a.len();
  if n == 0 {
    return Err(InterpreterError::EvaluationError(
      "Fit: empty design matrix".into(),
    ));
  }
  let m = a[0].len();
  if m == 0 || b.len() != n {
    return Err(InterpreterError::EvaluationError(
      "Fit: dimension mismatch in least squares".into(),
    ));
  }
  // Fewer equations than unknowns is underdetermined by construction.
  if n < m {
    return Ok(solve_least_squares_min_norm(a, b, scale_columns));
  }

  // Copy A and b (we'll transform them in place)
  let mut r: Vec<Vec<f64>> = a.to_vec();
  let mut rhs: Vec<f64> = b.to_vec();

  // Householder QR: transform A into R, apply same reflections to rhs
  for j in 0..m {
    // Compute norm of column j below diagonal
    let mut sigma = 0.0f64;
    for i in j..n {
      sigma += r[i][j] * r[i][j];
    }
    let norm = sigma.sqrt();

    if norm < 1e-15 {
      // Column j is dependent on the earlier ones: no unique least-squares
      // solution, so take the minimum-norm one.
      return Ok(solve_least_squares_min_norm(a, b, scale_columns));
    }

    // Choose sign to maximize |v[j]| (avoid cancellation)
    let alpha = if r[j][j] >= 0.0 { -norm } else { norm };

    // Build Householder vector v (save a copy since column j gets overwritten)
    let mut v = vec![0.0f64; n];
    v[j] = r[j][j] - alpha;
    for i in (j + 1)..n {
      v[i] = r[i][j];
    }

    let vtv: f64 = v[j..].iter().map(|x| x * x).sum();
    if vtv < 1e-30 {
      r[j][j] = alpha;
      continue;
    }
    let tau = 2.0 / vtv;

    // Apply H = I - tau * v * v^T to columns of R
    for k in j..m {
      let dot: f64 = (j..n).map(|i| v[i] * r[i][k]).sum();
      let factor = tau * dot;
      for i in j..n {
        r[i][k] -= factor * v[i];
      }
    }

    // Apply H to rhs
    let dot: f64 = (j..n).map(|i| v[i] * rhs[i]).sum();
    let factor = tau * dot;
    for i in j..n {
      rhs[i] -= factor * v[i];
    }
  }

  // Back substitution: solve R[0:m, 0:m] * x = rhs[0:m]
  let mut x = vec![0.0f64; m];
  for j in (0..m).rev() {
    let mut sum = rhs[j];
    for k in (j + 1)..m {
      sum -= r[j][k] * x[k];
    }
    x[j] = sum / r[j][j];
  }

  Ok(x)
}

// ─── Eigenvalues ────────────────────────────────────────────────────────

/// Extract an integer from an Expr, or None if not an integer.
fn expr_to_i128(e: &Expr) -> Option<i128> {
  match e {
    Expr::Integer(n) => Some(*n),
    _ => None,
  }
}

/// Extract integer matrix entries, returning None if any entry is non-integer.
fn matrix_to_i128(matrix: &[Vec<Expr>]) -> Option<Vec<Vec<i128>>> {
  matrix
    .iter()
    .map(|row| row.iter().map(expr_to_i128).collect())
    .collect()
}

/// Faddeev-LeVerrier algorithm: compute integer coefficients of the
/// characteristic polynomial det(λI - A) = λ^n + c_{n-1}*λ^{n-1} + ... + c_0.
///
/// Returns coefficients [c_0, c_1, ..., c_{n-1}, 1] (ascending powers), or
/// `None` when an intermediate value overflows `i128`.
fn char_poly_coefficients(a: &[Vec<i128>]) -> Option<Vec<i128>> {
  let n = a.len();
  // M starts as A, c[n] = 1 (monic)
  let mut coeffs = vec![0i128; n + 1];
  coeffs[n] = 1;

  // M_k tracks A * (M_{k-1} + c_{n-k+1} * I)
  let mut m = a.to_vec();

  for k in 1..=n {
    // c_{n-k} = -Tr(M_k) / k
    let trace = (0..n).try_fold(0i128, |s, i| s.checked_add(m[i][i]))?;
    coeffs[n - k] = -trace / k as i128;

    if k < n {
      // M_{k+1} = A * (M_k + c_{n-k} * I)
      let c = coeffs[n - k];
      // First add c*I to M_k
      let mut temp = m.clone();
      for i in 0..n {
        temp[i][i] = temp[i][i].checked_add(c)?;
      }
      // Then multiply: M_{k+1} = A * temp
      let mut new_m = vec![vec![0i128; n]; n];
      for i in 0..n {
        for j in 0..n {
          for p in 0..n {
            new_m[i][j] =
              new_m[i][j].checked_add(a[i][p].checked_mul(temp[p][j])?)?;
          }
        }
      }
      m = new_m;
    }
  }

  Some(coeffs)
}

/// BigInt Faddeev–LeVerrier: ascending coefficients of the monic
/// characteristic polynomial det(x I - A). O(n^3), exact for any size.
fn char_poly_coefficients_big(a: &[Vec<BigInt>]) -> Vec<BigInt> {
  let n = a.len();
  let mut coeffs = vec![BigInt::from(0); n + 1];
  coeffs[n] = BigInt::from(1);
  let mut m = a.to_vec();
  for k in 1..=n {
    let trace: BigInt = (0..n).map(|i| m[i][i].clone()).sum();
    // Tr(M_k) is always divisible by k in Faddeev–LeVerrier.
    coeffs[n - k] = (-&trace) / BigInt::from(k as i128);
    if k < n {
      let c = coeffs[n - k].clone();
      let mut temp = m.clone();
      for (i, row) in temp.iter_mut().enumerate().take(n) {
        row[i] += &c;
      }
      // new_m = A * temp
      let mut new_m = vec![vec![BigInt::from(0); n]; n];
      for i in 0..n {
        for j in 0..n {
          let mut s = BigInt::from(0);
          for p in 0..n {
            s += &a[i][p] * &temp[p][j];
          }
          new_m[i][j] = s;
        }
      }
      m = new_m;
    }
  }
  coeffs
}

/// CharacteristicPolynomial[A, x] for an exact integer matrix, computed via
/// Faddeev–LeVerrier so it doesn't fall back to the O(n!) symbolic Det of
/// `A - x I`. det(A - x I) = (-1)^n det(x I - A). Returns `None` when any
/// entry is not an integer.
pub fn characteristic_polynomial_int(
  rows: &[Expr],
  var: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let n = rows.len();
  let mut a: Vec<Vec<BigInt>> = Vec::with_capacity(n);
  for r in rows {
    let Expr::List(cols) = r else {
      return None;
    };
    let mut row = Vec::with_capacity(n);
    for e in cols {
      match e {
        Expr::Integer(v) => row.push(BigInt::from(*v)),
        Expr::BigInteger(v) => row.push(v.clone()),
        _ => return None,
      }
    }
    a.push(row);
  }
  let coeffs = char_poly_coefficients_big(&a);
  let negate = n % 2 == 1; // det(A - xI) = (-1)^n det(xI - A)
  let zero = BigInt::from(0);
  let mut terms: Vec<Expr> = Vec::new();
  for (i, c) in coeffs.iter().enumerate() {
    if c == &zero {
      continue;
    }
    let coeff = if negate { -c.clone() } else { c.clone() };
    let coeff_expr = crate::functions::math_ast::bigint_to_expr(coeff);
    let term = if i == 0 {
      coeff_expr
    } else {
      let pow = if i == 1 {
        var.clone()
      } else {
        pow(var.clone(), Expr::Integer(i as i128))
      };
      times(vec![coeff_expr, pow])
    };
    terms.push(term);
  }
  if terms.is_empty() {
    return Some(Ok(Expr::Integer(0)));
  }
  let sum = plus(terms);
  Some(crate::evaluator::evaluate_expr_to_expr(&sum))
}

/// Get all divisors of |n| (including 1 and |n|). Returns empty if n == 0.
fn divisors(n: i128) -> Vec<i128> {
  if n == 0 {
    return vec![];
  }
  let n = n.abs();
  let mut divs = Vec::new();
  let mut i = 1i128;
  while i * i <= n {
    if n % i == 0 {
      divs.push(i);
      if i != n / i {
        divs.push(n / i);
      }
    }
    i += 1;
  }
  divs.sort_unstable();
  divs
}

/// Evaluate polynomial (ascending coefficient order) at integer x.
fn poly_eval(coeffs: &[i128], x: i128) -> i128 {
  let mut result = 0i128;
  let mut power = 1i128;
  for &c in coeffs {
    result = result
      .checked_add(c.checked_mul(power).unwrap_or(0))
      .unwrap_or(0);
    power = power.checked_mul(x).unwrap_or(0);
  }
  result
}

/// Synthetic division: divide polynomial by (x - root).
/// Returns quotient coefficients (ascending order).
fn synthetic_div(coeffs: &[i128], root: i128) -> Vec<i128> {
  let n = coeffs.len() - 1; // degree of dividend
  if n == 0 {
    return vec![];
  }
  // Work with descending coefficients for standard synthetic division
  let desc: Vec<i128> = coeffs.iter().rev().copied().collect();
  let mut result = Vec::with_capacity(n);
  result.push(desc[0]);
  for i in 1..n {
    let val = desc[i] + root * result[i - 1];
    result.push(val);
  }
  // Convert back to ascending order
  result.reverse();
  result
}

/// Sort eigenvalues: Wolfram sorts by absolute value descending,
/// then by value descending for ties.
/// True if every entry of the matrix evaluates to a real number.
fn matrix_all_numeric(matrix: &[Vec<Expr>]) -> bool {
  matrix
    .iter()
    .all(|row| row.iter().all(|e| try_eval_to_f64(e).is_some()))
}

/// True if the matrix is upper- or lower-triangular (a zero on every entry
/// strictly below, or strictly above, the main diagonal).
fn is_triangular_matrix(matrix: &[Vec<Expr>]) -> bool {
  let n = matrix.len();
  let is_zero = |e: &Expr| {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(f) if *f == 0.0)
  };
  let upper = (0..n).all(|i| (0..i).all(|j| is_zero(&matrix[i][j])));
  let lower = (0..n).all(|i| (i + 1..n).all(|j| is_zero(&matrix[i][j])));
  upper || lower
}

fn sort_eigenvalues(eigenvalues: &mut [Expr]) {
  // Wolfram's Eigenvalues sorts by decreasing magnitude; when the magnitudes
  // tie, by ascending real part and then descending imaginary part:
  // {-1, I, -I, 1}, {-1 + I, -1 - I, 1 + I, 1 - I}.
  eigenvalues.sort_by(|a, b| {
    eigenvalue_order(eigenvalue_sort_key(a), eigenvalue_sort_key(b))
  });
}

/// The order of two numeric eigenvalues `(re, im)`: decreasing magnitude,
/// then increasing real part, then decreasing imaginary part.
fn eigenvalue_order(
  (ra, ia): (f64, f64),
  (rb, ib): (f64, f64),
) -> std::cmp::Ordering {
  let cmp =
    |x: f64, y: f64| x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal);
  cmp(rb.hypot(ib), ra.hypot(ia))
    .then(cmp(ra, rb))
    .then(cmp(ib, ia))
}

/// True when `e` evaluates (under N) to a real number — i.e. not a complex
/// value and not a symbolic expression with free variables.
fn is_real_numeric_value(e: &Expr) -> bool {
  matches!(
    crate::functions::math_ast::n_ast(std::slice::from_ref(e)),
    Ok(Expr::Real(_) | Expr::Integer(_))
  )
}

/// The numeric value `(re, im)` an eigenvalue is sorted by; `(0, 0)` for a
/// symbolic one.
fn eigenvalue_sort_key(e: &Expr) -> (f64, f64) {
  match e {
    Expr::Integer(n) => (*n as f64, 0.0),
    Expr::Real(r) => (*r, 0.0),
    _ => evaluate_expr_to_expr(&call1("N", e.clone()))
      .ok()
      .and_then(|v| {
        crate::functions::math_ast::numeric_utils::try_extract_complex_float(&v)
          .or_else(|| try_eval_to_f64(&v).map(|re| (re, 0.0)))
      })
      .unwrap_or((0.0, 0.0)),
  }
}

/// Convert an exact rational matrix (with at least one non-integer entry)
/// into `scale * m` as an integer matrix, where `scale` is the LCM of the
/// entry denominators. Returns None for non-rational entries, all-integer
/// matrices, or on i128 overflow.
fn matrix_to_scaled_i128(
  matrix: &[Vec<Expr>],
) -> Option<(Vec<Vec<i128>>, i128)> {
  let mut entries: Vec<Vec<(i128, i128)>> = Vec::with_capacity(matrix.len());
  let mut scale: i128 = 1;
  let mut any_rational = false;
  for row in matrix {
    let mut out_row: Vec<(i128, i128)> = Vec::with_capacity(row.len());
    for cell in row {
      let (num, den) = match cell {
        Expr::Integer(n) => (*n, 1),
        Expr::FunctionCall { name, args }
          if name == "Rational" && args.len() == 2 =>
        {
          match (&args[0], &args[1]) {
            (Expr::Integer(n), Expr::Integer(d)) if *d != 0 => (*n, *d),
            _ => return None,
          }
        }
        _ => return None,
      };
      if den != 1 {
        any_rational = true;
      }
      scale = scale.checked_mul(den / gcd_i128(scale, den))?;
      out_row.push((num, den));
    }
    entries.push(out_row);
  }
  if !any_rational {
    return None;
  }
  let scale = scale.abs();
  let mut int_matrix: Vec<Vec<i128>> = Vec::with_capacity(entries.len());
  for row in entries {
    let mut out_row: Vec<i128> = Vec::with_capacity(row.len());
    for (num, den) in row {
      out_row.push(num.checked_mul(scale / den)?);
    }
    int_matrix.push(out_row);
  }
  Some((int_matrix, scale))
}

/// Divide an exact eigenvalue expression by a positive integer, keeping
/// wolframscript's forms: integers fold to Rationals, and a leading minus
/// stays outside the quotient so `-Sqrt[6]/6` prints as `-(1/Sqrt[6])`.
fn divide_exact_root(root: &Expr, d: i128) -> Option<Expr> {
  if d == 1 {
    return Some(root.clone());
  }
  match root {
    Expr::Integer(n) => Some(simplify_fraction(*n, d)),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let pos = divide_exact_root(operand, d)?;
      crate::evaluator::evaluate_expr_to_expr(&neg(pos)).ok()
    }
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(-1)) =>
    {
      let pos = divide_exact_root(&args[1], d)?;
      crate::evaluator::evaluate_expr_to_expr(&neg(pos)).ok()
    }
    _ => crate::evaluator::evaluate_expr_to_expr(&div2(
      root.clone(),
      Expr::Integer(d),
    ))
    .ok(),
  }
}

/// Build eigenvalue expression from quadratic formula components.
/// Given polynomial x^2 + bx + c = 0, eigenvalues are (-b ± sqrt(b²-4c))/2.
fn quadratic_eigenvalues(b_coeff: i128, c_coeff: i128) -> Vec<Expr> {
  let neg_b = -b_coeff; // This is the trace
  let discriminant = b_coeff * b_coeff - 4 * c_coeff;

  if discriminant == 0 {
    // Repeated root
    let root = simplify_fraction(neg_b, 2);
    return vec![root.clone(), root];
  }

  // Check if discriminant is a perfect square
  if discriminant > 0 {
    let sqrt_disc = (discriminant as f64).sqrt().round() as i128;
    if sqrt_disc * sqrt_disc == discriminant {
      // Rational roots
      let r1 = simplify_fraction(neg_b + sqrt_disc, 2);
      let r2 = simplify_fraction(neg_b - sqrt_disc, 2);
      return vec![r1, r2];
    }
  }

  // Simplify sqrt(|discriminant|): factor out perfect squares
  let (outer, inner) = simplify_sqrt(discriminant.unsigned_abs());

  // Negative discriminant: conjugate pair (neg_b ± I*outer*Sqrt[inner])/2,
  // plus branch (positive imaginary part) first, matching wolframscript's
  // {1 + I*Sqrt[2], 1 - I*Sqrt[2]} ordering.
  if discriminant < 0 {
    let imag_term = |coeff: i128| -> Expr {
      // Flat Times[coeff, I, Sqrt[inner]] so the printed factor order is
      // wolframscript's `2*I*Sqrt[5]`
      let mut factors: Vec<Expr> = Vec::new();
      if coeff != 1 {
        factors.push(Expr::Integer(coeff));
      }
      factors.push(id_expr("I"));
      if inner != 1 {
        factors.push(call1("Sqrt", Expr::Integer(inner as i128)));
      }
      if factors.len() == 1 {
        factors.remove(0)
      } else {
        times(factors)
      }
    };
    let with_real = |re: i128, imag: Expr| -> Expr {
      if re == 0 {
        imag
      } else {
        plus(vec![Expr::Integer(re), imag])
      }
    };
    let outer = outer as i128;
    if neg_b % 2 == 0 && outer % 2 == 0 {
      // Components fold to exact halves: re ± oc*I*Sqrt[inner]
      let (re, oc) = (neg_b / 2, outer / 2);
      return vec![with_real(re, imag_term(oc)), with_real(re, imag_term(-oc))];
    }
    // Otherwise keep the quotient form (3 + I*Sqrt[11])/2
    let halved = |num: Expr| div2(num, Expr::Integer(2));
    return vec![
      halved(with_real(neg_b, imag_term(outer))),
      halved(with_real(neg_b, imag_term(-outer))),
    ];
  }

  // Factor out GCD of neg_b and outer from the numerator:
  // (neg_b ± outer*Sqrt[inner]) / 2  =  common * (reduced_b ± reduced_outer*Sqrt[inner]) / 2
  let common = gcd_i128(neg_b, outer as i128);
  let (reduced_b, reduced_outer) = (neg_b / common, outer as i128 / common);

  let sqrt_expr = make_sqrt(Expr::Integer(inner as i128));

  let reduced_sqrt = if reduced_outer == 1 {
    sqrt_expr.clone()
  } else {
    eval_mul(&Expr::Integer(reduced_outer), &sqrt_expr)
  };

  // Build the inner sum/difference
  let inner_plus = if reduced_b == 0 {
    reduced_sqrt.clone()
  } else {
    eval_add(&Expr::Integer(reduced_b), &reduced_sqrt)
  };
  let inner_minus = if reduced_b == 0 {
    eval_sub(&Expr::Integer(0), &reduced_sqrt)
  } else {
    eval_sub(&Expr::Integer(reduced_b), &reduced_sqrt)
  };

  // Multiply by common factor if needed
  let numer_plus = if common == 1 {
    inner_plus
  } else {
    eval_mul(&Expr::Integer(common), &inner_plus)
  };
  let numer_minus = if common == 1 {
    inner_minus
  } else {
    eval_mul(&Expr::Integer(common), &inner_minus)
  };

  // Divide by 2
  let plus_expr = eval_divide(&numer_plus, &Expr::Integer(2));
  let minus_expr = eval_divide(&numer_minus, &Expr::Integer(2));

  vec![plus_expr, minus_expr]
}

/// Simplify fraction n/d, returning Expr::Integer or Expr::Rational.
fn simplify_fraction(n: i128, d: i128) -> Expr {
  let (n, d) = rat_reduce(n, d);
  if d == 1 {
    Expr::Integer(n)
  } else {
    crate::functions::math_ast::make_rational(n, d)
  }
}

/// Simplify sqrt(n) into outer * sqrt(inner) where inner is square-free.
/// E.g. sqrt(12) = 2*sqrt(3), so returns (2, 3).
fn simplify_sqrt(n: u128) -> (u128, u128) {
  if n == 0 {
    return (0, 1);
  }
  let mut outer = 1u128;
  let mut inner = n;
  let mut factor = 2u128;
  while factor * factor <= inner {
    while inner.is_multiple_of(factor * factor) {
      outer *= factor;
      inner /= factor * factor;
    }
    factor += 1;
  }
  (outer, inner)
}

/// Split a matrix with symbolic entries into `scale * B` where `B` is
/// purely numeric (`{{4c, -8c}, {5c, -9c}}` -> `c * {{4, -8}, {5, -9}}`).
/// The scale is the symbolic part of the first nonzero entry; `None` when the
/// matrix is already numeric or no such numeric `B` exists. Eigenvectors are
/// those of `B` and eigenvalues are `scale` times those of `B`.
fn split_symbolic_scale(
  matrix: &[Vec<Expr>],
) -> Option<(Expr, Vec<Vec<Expr>>)> {
  if matrix
    .iter()
    .flatten()
    .all(crate::functions::predicate_ast::is_numeric_q)
  {
    return None;
  }
  let first = matrix.iter().flatten().find(|e| {
    !matches!(e, Expr::Integer(0)) && !matches!(e, Expr::Real(z) if *z == 0.0)
  })?;
  let scale = match first {
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() >= 2
        && crate::functions::predicate_ast::is_numeric_q(&args[0]) =>
    {
      if args.len() == 2 {
        args[1].clone()
      } else {
        times(args[1..].to_vec())
      }
    }
    other => other.clone(),
  };
  let mut numeric = Vec::with_capacity(matrix.len());
  for row in matrix {
    let mut out = Vec::with_capacity(row.len());
    for e in row {
      let q = evaluate_expr_to_expr(&div(e.clone(), scale.clone())).ok()?;
      if !crate::functions::predicate_ast::is_numeric_q(&q) {
        return None;
      }
      out.push(q);
    }
    numeric.push(out);
  }
  Some((scale, numeric))
}

/// Eigenvalues[matrix] - eigenvalues of a square matrix, sorted descending.
/// A real machine 2×2 matrix lists a conjugate pair of eigenvalues upper
/// half first, like the QR route for larger matrices does:
/// `{1. + 2.45 I, 1. - 2.45 I}`. Exact and symbolic pairs keep their order.
fn machine_conjugate_pair_upper_first(matrix: &[Vec<Expr>], eigs: &mut [Expr]) {
  let machine = matrix
    .iter()
    .flatten()
    .all(|e| matches!(e, Expr::Real(_) | Expr::Integer(_)))
    && matrix.iter().flatten().any(|e| matches!(e, Expr::Real(_)));
  if machine
    && let [first, second] = eigs
    && let (Some((_, im0)), Some((_, im1))) =
      (numeric_re_im(first), numeric_re_im(second))
    && im0 < im1
  {
    eigs.swap(0, 1);
  }
}

/// `{m}` — a one-element list holding a square matrix — is that matrix to
/// the eigen functions (the one-matrix case of the `{m, a}` form).
fn unwrap_single_matrix_list(args: &[Expr]) -> Option<Vec<Expr>> {
  let Expr::List(outer) = args.first()? else {
    return None;
  };
  let [inner @ Expr::List(rows)] = outer.as_slice() else {
    return None;
  };
  if rows.is_empty()
    || !rows
      .iter()
      .all(|r| matches!(r, Expr::List(c) if c.len() == rows.len()))
  {
    return None;
  }
  let mut new_args = vec![inner.clone()];
  new_args.extend(args[1..].iter().cloned());
  Some(new_args)
}

pub fn eigenvalues_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let Some(inner) = unwrap_single_matrix_list(args) {
    return eigenvalues_ast(&inner);
  }
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Eigenvalues expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) if is_nonempty_square(&m) => m,
    _ => return Ok(matsq_unevaluated("Eigenvalues", args)),
  };
  let n = matrix.len();

  // 1x1
  if n == 1 {
    return Ok(Expr::List(vec![matrix[0][0].clone()].into()));
  }

  if let Some((scale, numeric)) = split_symbolic_scale(&matrix) {
    let Expr::List(ref vals) = eigenvalues_ast(&[matrix_to_expr(numeric)])?
    else {
      return Ok(unevaluated("Eigenvalues", args));
    };
    let scaled = vals
      .iter()
      .map(|l| evaluate_expr_to_expr(&times(vec![scale.clone(), l.clone()])))
      .collect::<Result<Vec<_>, _>>()?;
    return Ok(Expr::List(scaled.into()));
  }

  // Try integer matrix path
  if let Some(int_matrix) = matrix_to_i128(&matrix)
    && let Some(coeffs) = char_poly_coefficients(&int_matrix)
  {
    let mut eigenvalues = find_polynomial_roots(&coeffs);
    sort_eigenvalues(&mut eigenvalues);
    return Ok(Expr::List(eigenvalues.into()));
  }

  // Exact rational matrix: scale by the LCM of the denominators to an
  // integer matrix (eigenvalues scale linearly), find the scaled roots
  // exactly, and divide back — keeping 1/2 exact instead of leaking the
  // float 0.5 from the numeric 2×2 path.
  if let Some((int_matrix, scale)) = matrix_to_scaled_i128(&matrix)
    && let Some(coeffs) = char_poly_coefficients(&int_matrix)
  {
    let mut scaled_roots = find_polynomial_roots(&coeffs);
    // Root objects cannot be rescaled into wolframscript's form; fall
    // through to the generic paths instead.
    if !scaled_roots
      .iter()
      .any(|e| expr_to_string(e).contains("Root["))
    {
      // A positive scale preserves the magnitude order, so sorting the
      // scaled roots sorts the eigenvalues.
      sort_eigenvalues(&mut scaled_roots);
      let mut out: Vec<Expr> = Vec::with_capacity(scaled_roots.len());
      let mut ok = true;
      for r in &scaled_roots {
        if let Some(v) = divide_exact_root(r, scale) {
          out.push(v);
        } else {
          ok = false;
          break;
        }
      }
      if ok {
        return Ok(Expr::List(out.into()));
      }
    }
  }

  // Floating-point 2×2 path. wolframscript produces real-valued
  // eigenvalues for matrices with non-negative discriminant via the
  // closed-form `(tr ± Sqrt[tr² − 4·det]) / 2`. Sort descending so the
  // result matches `Eigenvalues[{{1.5, 0.5}, {0.5, 2.0}}]`
  // → `{2.3090169…, 1.1909830…}`.
  if n == 2
    && let (Some(a), Some(b), Some(c), Some(d)) = (
      try_eval_to_f64(&matrix[0][0]),
      try_eval_to_f64(&matrix[0][1]),
      try_eval_to_f64(&matrix[1][0]),
      try_eval_to_f64(&matrix[1][1]),
    )
  {
    let tr = a + d;
    let det = a * d - b * c;
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
      let s = disc.sqrt();
      let l1 = f64::midpoint(tr, s);
      let l2 = (tr - s) / 2.0;
      // Match Wolfram's ordering (largest-magnitude first).
      let (e1, e2) = if l1.abs() >= l2.abs() {
        (l1, l2)
      } else {
        (l2, l1)
      };
      return Ok(Expr::List(vec![Expr::Real(e1), Expr::Real(e2)].into()));
    }
  }

  // Triangular (incl. diagonal) matrices with symbolic entries: the
  // eigenvalues are the diagonal entries, in matrix order. (All-numeric
  // matrices are handled by the integer/float paths above, which also sort,
  // so this only fires when at least one entry is symbolic or complex —
  // otherwise the generic characteristic-polynomial path would print an
  // unsimplified `(a + b ± Sqrt[a² − 2ab + b²]) / 2` instead of `{a, b}`.)
  if !matrix_all_numeric(&matrix) && is_triangular_matrix(&matrix) {
    let mut diag: Vec<Expr> = (0..n).map(|i| matrix[i][i].clone()).collect();
    // Complex-numeric diagonals sort like wolframscript: descending
    // magnitude, ties ascending by (Re, Im) — Eigenvalues[{{I, 0}, {0, 2}}]
    // is {2, I} and {{2 I, 0}, {0, -2}} gives {-2, 2 I}. Genuinely symbolic
    // entries keep matrix order.
    let keys: Vec<Option<(f64, f64)>> =
      diag.iter().map(numeric_re_im).collect();
    if keys
      .iter()
      .all(|k| k.is_some_and(|(re, im)| re.is_finite() && im.is_finite()))
    {
      let mut pairs: Vec<((f64, f64), Expr)> = keys
        .into_iter()
        .map(Option::unwrap)
        .zip(diag.iter().cloned())
        .collect();
      pairs.sort_by(|a, b| eigenvalue_order(a.0, b.0));
      diag = pairs.into_iter().map(|(_, e)| e).collect();
    }
    return Ok(Expr::List(diag.into()));
  }

  // Symbolic 2×2 rotation-matrix shortcut: `[[a, b], [-b, a]]` has
  // eigenvalues `a ± I*b`. Wolframscript returns the simplified
  // imaginary form rather than `(2a ± Sqrt[-4 b²]) / 2`, so a generic
  // discriminant formula here would print `Sqrt[-Sin[t]^2]` instead of
  // `I*Sin[t]`. Detecting the pattern lets us hand back the closed form.
  if n == 2
    && let Some(mut eigs) = symbolic_rotation_eigenvalues(&matrix)
  {
    // When an imaginary off-diagonal makes the eigenvalues real (a Hermitian
    // matrix), order them like wolframscript — decreasing magnitude, ascending
    // tiebreak (e.g. {{0, I}, {-I, 0}} → {-1, 1}, {{1, -I}, {I, 1}} → {2, 0}).
    // Genuinely symbolic / complex pairs keep the `a - I*b, a + I*b` order.
    let is_real_num = |e: &Expr| {
      matches!(e, Expr::Integer(_) | Expr::Real(_))
        || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
    };
    if eigs.iter().all(is_real_num) {
      sort_eigenvalues(&mut eigs);
    } else {
      machine_conjugate_pair_upper_first(&matrix, &mut eigs);
    }
    return Ok(Expr::List(eigs.into()));
  }

  // Generic symbolic 2×2: characteristic polynomial λ^2 − (a+d)λ + (ad−bc)
  // gives λ = (a + d ± Sqrt[a² − 2 a d + d² + 4 b c]) / 2. wolframscript
  // returns the minus-branch first, then plus.
  if n == 2 {
    let a = matrix[0][0].clone();
    let b = matrix[0][1].clone();
    let c = matrix[1][0].clone();
    let d = matrix[1][1].clone();
    // discriminant: a² + 4 b c - 2 a d + d²
    let disc = plus(vec![
      pow(a.clone(), Expr::Integer(2)),
      times(vec![Expr::Integer(4), b.clone(), c.clone()]),
      times(vec![Expr::Integer(-2), a.clone(), d.clone()]),
      pow(d.clone(), Expr::Integer(2)),
    ]);
    let sqrt_disc = call1("Sqrt", disc);
    let trace = plus(vec![a, d]);
    let lambda_minus = div2(
      plus(vec![trace.clone(), neg(sqrt_disc.clone())]),
      Expr::Integer(2),
    );
    let lambda_plus = div2(plus(vec![trace, sqrt_disc]), Expr::Integer(2));
    let lm = crate::evaluator::evaluate_expr_to_expr(&lambda_minus)?;
    let lp = crate::evaluator::evaluate_expr_to_expr(&lambda_plus)?;
    let mut result = vec![lm, lp];
    // When both eigenvalues are real numbers (a complex Hermitian matrix whose
    // diagonal differs, e.g. {{4, I}, {-I, 1}}), order them like wolframscript
    // — decreasing magnitude. Symbolic matrices and genuinely complex
    // eigenvalues keep the closed `minus, plus` branch order.
    if result.iter().all(is_real_numeric_value) {
      sort_eigenvalues(&mut result);
    } else {
      machine_conjugate_pair_upper_first(&matrix, &mut result);
    }
    return Ok(Expr::List(result.into()));
  }

  // A machine matrix with complex entries (n ≥ 3): Hessenberg reduction and
  // shifted complex QR, the eigenvalues by decreasing absolute value.
  if n >= 3
    && matrix.iter().flatten().any(|c| {
      crate::functions::math_ast::try_extract_complex_float(c)
        .is_some_and(|(_, im)| im != 0.0)
    })
    && matrix
      .iter()
      .flatten()
      .any(crate::functions::math_ast::contains_inexact_real)
    && let Some(entries) = matrix
      .iter()
      .map(|row| {
        row
          .iter()
          .map(|c| {
            crate::functions::math_ast::try_extract_complex_float(c)
              .or_else(|| try_eval_to_f64(c).map(|re| (re, 0.0)))
          })
          .collect::<Option<Vec<_>>>()
      })
      .collect::<Option<Vec<_>>>()
    && let hermitian = (0..n).all(|i| {
      (0..n).all(|j| entries[i][j] == (entries[j][i].0, -entries[j][i].1))
    })
    && let Some(mut values) = complex_matrix_eigenvalues(entries)
  {
    values.sort_by(|a, b| b.0.hypot(b.1).total_cmp(&a.0.hypot(a.1)));
    return Ok(Expr::List(
      values
        .into_iter()
        .map(|(re, im)| {
          // A Hermitian matrix has real eigenvalues.
          if hermitian {
            Expr::Real(re)
          } else {
            crate::functions::math_ast::build_complex_float_expr_keep_real(
              re, im,
            )
          }
        })
        .collect::<Vec<_>>()
        .into(),
    ));
  }

  // Float-valued n×n matrix (n ≥ 3): compute eigenvalues numerically
  // via Faddeev–LeVerrier + QR on the companion matrix.
  if n >= 3 {
    let mut float_mat: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut all_floats = true;
    for row in &matrix {
      let mut frow: Vec<f64> = Vec::with_capacity(n);
      for cell in row {
        if let Some(v) = try_eval_to_f64(cell) {
          frow.push(v);
        } else {
          all_floats = false;
          break;
        }
      }
      if !all_floats {
        break;
      }
      float_mat.push(frow);
    }
    if all_floats {
      let any_real_literal = matrix
        .iter()
        .flat_map(|r| r.iter())
        .any(|c| matches!(c, Expr::Real(_)));
      if any_real_literal {
        let roots = numeric_eigenvalues(&float_mat, n);
        if roots.len() == n {
          let mut out: Vec<Expr> = roots.into_iter().map(Expr::Real).collect();
          out.sort_by(|a, b| {
            let av = if let Expr::Real(v) = a { *v } else { 0.0 };
            let bv = if let Expr::Real(v) = b { *v } else { 0.0 };
            bv.abs()
              .partial_cmp(&av.abs())
              .unwrap_or(std::cmp::Ordering::Equal)
          });
          return Ok(Expr::List(out.into()));
        }
        // Complex-conjugate pairs: Wolfram complexifies the whole result.
        let complex = numeric_eigenvalues_complex(&float_mat, n);
        if complex.len() == n {
          let out: Vec<Expr> = complex
            .into_iter()
            .map(|(re, im)| {
              crate::functions::math_ast::build_complex_float_expr_keep_real(
                re, im,
              )
            })
            .collect();
          return Ok(Expr::List(out.into()));
        }
      }
    }
  }

  // Block-diagonal 3×3: split into a 2×2 top-left block and the
  // bottom-right scalar, recurse, sort. Matches case 1337's
  // `Eigenvalues[{{Cos, Sin, 0}, {-Sin, Cos, 0}, {0, 0, 1}}]`.
  if n == 3
    && is_zero_expr(&matrix[0][2])
    && is_zero_expr(&matrix[1][2])
    && is_zero_expr(&matrix[2][0])
    && is_zero_expr(&matrix[2][1])
  {
    let block = vec![
      vec![matrix[0][0].clone(), matrix[0][1].clone()],
      vec![matrix[1][0].clone(), matrix[1][1].clone()],
    ];
    if let Ok(block_evs) = eigenvalues_ast(&[matrix_to_expr(block)])
      && let Expr::List(items_ref) = &block_evs
    {
      let mut items = items_ref.clone();
      items.push(matrix[2][2].clone());
      let mut items = items.to_vec();
      if items.iter().all(is_real_numeric_value) {
        sort_eigenvalues(&mut items);
      } else {
        items = sort_with_canonical(items);
      }
      return Ok(Expr::List(items.into()));
    }
  }

  // Matrices whose nonzero pattern splits into independent index groups
  // (a permuted block-diagonal matrix): the eigenvalues of each block
  // together.
  if n >= 3
    && let Some(groups) = independent_index_groups(&matrix)
  {
    let mut all: Vec<Expr> = Vec::with_capacity(n);
    for g in &groups {
      let block: Vec<Vec<Expr>> = g
        .iter()
        .map(|&i| g.iter().map(|&j| matrix[i][j].clone()).collect())
        .collect();
      match &eigenvalues_ast(&[matrix_to_expr(block)])? {
        Expr::List(vs) => all.extend(vs.iter().cloned()),
        _ => return Ok(unevaluated("Eigenvalues", args)),
      }
    }
    if all.iter().all(is_real_numeric_value) {
      sort_eigenvalues(&mut all);
    } else {
      all = sort_with_canonical(all);
    }
    return Ok(Expr::List(all.into()));
  }

  // Return unevaluated for matrices we can't yet handle (defective, complex
  // eigenvalues for floats, larger non-integer matrices, etc.).
  Ok(unevaluated("Eigenvalues", args))
}

/// Connected components of the graph whose edges are the nonzero entries of
/// a square matrix, as sorted index groups; `None` when it is connected.
fn independent_index_groups(matrix: &[Vec<Expr>]) -> Option<Vec<Vec<usize>>> {
  let n = matrix.len();
  let mut group: Vec<usize> = (0..n).collect();
  fn find(g: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while g[r] != r {
      r = g[r];
    }
    g[i] = r;
    r
  }
  for i in 0..n {
    for j in 0..n {
      if i != j && !is_zero_expr(&matrix[i][j]) {
        let (a, b) = (find(&mut group, i), find(&mut group, j));
        group[a] = b;
      }
    }
  }
  let mut groups: Vec<Vec<usize>> = Vec::new();
  let mut roots: Vec<usize> = Vec::new();
  for i in 0..n {
    let r = find(&mut group, i);
    if let Some(k) = roots.iter().position(|&x| x == r) {
      groups[k].push(i);
    } else {
      roots.push(r);
      groups.push(vec![i]);
    }
  }
  (groups.len() > 1).then_some(groups)
}

/// Detect a symbolic 2×2 rotation matrix `[[a, b], [-b, a]]` and return
/// its eigenvalues `{a - I*b, a + I*b}`. Returns `None` for any other
/// shape (so the caller falls back to the generic path).
fn symbolic_rotation_eigenvalues(matrix: &[Vec<Expr>]) -> Option<Vec<Expr>> {
  if matrix.len() != 2 || matrix[0].len() != 2 || matrix[1].len() != 2 {
    return None;
  }
  let a = &matrix[0][0];
  let b = &matrix[0][1];
  let c = &matrix[1][0];
  let d = &matrix[1][1];
  // bottom-right must equal top-left
  if expr_to_string(a) != expr_to_string(d) {
    return None;
  }
  // bottom-left must equal `-(top-right)`
  let neg_b = neg(b.clone());
  let neg_b_eval = crate::evaluator::evaluate_expr_to_expr(&neg_b).ok()?;
  if expr_to_string(c) != expr_to_string(&neg_b_eval) {
    return None;
  }
  // Build `a - I*b` and `a + I*b`, then evaluate so e.g. `Cos[t] - I*Sin[t]`
  // surfaces in canonical form. Evaluate `I*b` on its own first: when b is
  // itself imaginary (e.g. b = I), `Times[I, I]` collapses to a single
  // `Power[I, 2]` that a later combined Plus-evaluation leaves unreduced — but
  // reduced standalone it becomes -1, so the eigenvalues come out as plain
  // numbers (e.g. Eigenvalues[{{2, I}, {-I, 2}}] = {3, 1}, not {2 - I^2, …}).
  // Use the imaginary unit as a `Complex[0, 1]` literal, not the bare symbol
  // `Constant["I"]`: matrix entries are stored as `Complex[…]`, and
  // `Times[Constant["I"], Complex[0, 1]]` mixes the two spellings and collapses
  // to an unreduced `Power[I, 2]`, whereas `Complex × Complex` does proper
  // complex arithmetic (e.g. I*I -> -1). So Eigenvalues[{{2, I}, {-I, 2}}]
  // = {3, 1}, not {2 - I^2, 2 + I^2}.
  let i = call("Complex", vec![Expr::Integer(0), Expr::Integer(1)]);
  let i_b = times(vec![i, b.clone()]);
  let minus_ib = neg(i_b.clone());
  let lo = plus(vec![a.clone(), minus_ib]);
  let hi = plus(vec![a.clone(), i_b]);
  let lo = crate::evaluator::evaluate_expr_to_expr(&lo).ok()?;
  let hi = crate::evaluator::evaluate_expr_to_expr(&hi).ok()?;
  Some(vec![lo, hi])
}

fn sort_with_canonical(mut items: Vec<Expr>) -> Vec<Expr> {
  items.sort_by(crate::functions::list_helpers_ast::canonical_cmp);
  items
}

/// Find roots of a monic polynomial with integer coefficients.
/// coeffs are in ascending order: [c_0, c_1, ..., c_{n-1}, 1].
fn find_polynomial_roots(coeffs: &[i128]) -> Vec<Expr> {
  let degree = coeffs.len() - 1;
  if degree == 0 {
    return vec![];
  }
  if degree == 1 {
    // x + c_0 = 0 → x = -c_0
    return vec![Expr::Integer(-coeffs[0])];
  }
  if degree == 2 {
    // x^2 + c_1*x + c_0 = 0
    return quadratic_eigenvalues(coeffs[1], coeffs[0]);
  }

  // For degree >= 3: find integer roots via rational root theorem,
  // then reduce degree via synthetic division.
  let mut current = coeffs.to_vec();
  let mut roots = Vec::new();

  loop {
    let deg = current.len() - 1;
    if deg <= 2 {
      break;
    }

    // Try to find an integer root
    let c0 = current[0];
    let mut found = false;

    if c0 == 0 {
      // x = 0 is a root
      roots.push(Expr::Integer(0));
      // Divide out x: just shift coefficients
      current = current[1..].to_vec();
      found = true;
    } else {
      let candidates = divisors(c0);
      for &d in &candidates {
        for &sign in &[1i128, -1] {
          let candidate = d * sign;
          if poly_eval(&current, candidate) == 0 {
            roots.push(Expr::Integer(candidate));
            current = synthetic_div(&current, candidate);
            found = true;
            break;
          }
        }
        if found {
          break;
        }
      }
    }

    if !found {
      break;
    }
  }

  // Handle remaining polynomial
  let deg = current.len() - 1;
  match deg {
    0 => {}
    1 => {
      roots.push(Expr::Integer(-current[0]));
    }
    2 => {
      roots.extend(quadratic_eigenvalues(current[1], current[0]));
    }
    _ => {
      // Repeated irreducible factors: for p = ∏ qᵢ^{eᵢ} the gcd with the
      // derivative is g = ∏ qᵢ^{eᵢ−1}, so the root multiset of p is
      // roots(p/g) ⊎ roots(g) — recurse on both halves. This resolves e.g.
      // the characteristic polynomial (x² − 2)² to {±Sqrt[2], ±Sqrt[2]}.
      if let Some(g) = int_poly_gcd_with_derivative(&current)
        && g.len() > 1
        && g.len() < current.len()
        && let Some(quot) = int_poly_div_exact(&current, &g)
      {
        roots.extend(find_polynomial_roots(&quot));
        roots.extend(find_polynomial_roots(&g));
      } else {
        // Square-free irreducible remainder: represent the roots as
        // wolframscript-shaped Root[c0 + c1*#1 + … + #1^n & , k, 0] objects.
        roots.extend(make_root_exprs(&current));
      }
    }
  }

  roots
}

/// Derivative of an integer polynomial in ascending coefficient order.
fn int_poly_derivative(p: &[i128]) -> Vec<i128> {
  p.iter()
    .enumerate()
    .skip(1)
    .map(|(i, &c)| c.checked_mul(i as i128))
    .collect::<Option<Vec<_>>>()
    .unwrap_or_default()
}

/// Content (gcd of all coefficients) of a nonzero integer polynomial.
fn int_poly_content(p: &[i128]) -> i128 {
  p.iter().fold(0i128, |acc, &c| gcd_i128(acc, c)).max(1)
}

/// Primitive part: divide out the content and drop leading zeros.
fn int_poly_primitive(p: &[i128]) -> Vec<i128> {
  let mut out: Vec<i128> = p.to_vec();
  while out.last() == Some(&0) {
    out.pop();
  }
  if out.is_empty() {
    return out;
  }
  let content = int_poly_content(&out);
  for c in &mut out {
    *c /= content;
  }
  out
}

/// Pseudo-remainder of a by b (ascending coefficients, b nonzero); returns
/// None on i128 overflow so callers can fall back to Root objects.
fn int_poly_pseudo_rem(a: &[i128], b: &[i128]) -> Option<Vec<i128>> {
  let mut r: Vec<i128> = a.to_vec();
  let db = b.len() - 1;
  let lb = *b.last()?;
  while r.len() > db {
    let lr = *r.last()?;
    let shift = r.len() - 1 - db;
    for c in &mut r {
      *c = c.checked_mul(lb)?;
    }
    for (i, &bc) in b.iter().enumerate() {
      r[shift + i] = r[shift + i].checked_sub(lr.checked_mul(bc)?)?;
    }
    while r.last() == Some(&0) {
      r.pop();
    }
    if r.is_empty() {
      break;
    }
  }
  Some(r)
}

/// Primitive gcd of a monic integer polynomial and its derivative,
/// normalized to a positive leading coefficient. Returns None when the
/// computation overflows i128.
fn int_poly_gcd_with_derivative(p: &[i128]) -> Option<Vec<i128>> {
  let mut a = int_poly_primitive(p);
  let mut b = int_poly_primitive(&int_poly_derivative(p));
  while !b.is_empty() {
    let r = int_poly_pseudo_rem(&a, &b)?;
    a = b;
    b = int_poly_primitive(&r);
  }
  if *a.last()? < 0 {
    for c in &mut a {
      *c = -*c;
    }
  }
  Some(a)
}

/// Exact division of integer polynomials (ascending coefficients). The
/// divisor must divide evenly with an integer quotient; returns None
/// otherwise (or when the divisor's leading coefficient is not ±1).
fn int_poly_div_exact(p: &[i128], d: &[i128]) -> Option<Vec<i128>> {
  let ld = *d.last()?;
  if ld.abs() != 1 {
    return None;
  }
  let mut rem: Vec<i128> = p.to_vec();
  let dq = p.len().checked_sub(d.len())?;
  let mut quot = vec![0i128; dq + 1];
  for k in (0..=dq).rev() {
    let coeff = rem[k + d.len() - 1] / ld;
    quot[k] = coeff;
    for (i, &dc) in d.iter().enumerate() {
      rem[k + i] = rem[k + i].checked_sub(coeff.checked_mul(dc)?)?;
    }
  }
  if rem.iter().any(|&c| c != 0) {
    return None;
  }
  Some(quot)
}

/// Build `Root[c0 + c1*#1 + … + #1^n & , k, 0]` expressions for every root
/// of an integer polynomial given in ascending coefficient order.
fn make_root_exprs(coeffs: &[i128]) -> Vec<Expr> {
  let degree = coeffs.len() - 1;
  let slot = Expr::Slot(1);
  let mut terms: Vec<Expr> = Vec::new();
  for (i, &c) in coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let var_pow = match i {
      0 => None,
      1 => Some(slot.clone()),
      _ => Some(pow(slot.clone(), Expr::Integer(i as i128))),
    };
    terms.push(match (var_pow, c) {
      (None, c) => Expr::Integer(c),
      (Some(p), 1) => p,
      (Some(p), c) => times(vec![Expr::Integer(c), p]),
    });
  }
  let body = match terms.len() {
    0 => Expr::Integer(0),
    1 => terms.remove(0),
    _ => plus(terms),
  };
  let func = Expr::Function {
    body: Box::new(body),
  };
  (1..=degree)
    .map(|k| {
      call(
        "Root",
        vec![func.clone(), Expr::Integer(k as i128), Expr::Integer(0)],
      )
    })
    .collect()
}

/// The count argument of `Eigenvalues`, `Eigenvectors` and `Eigensystem`: a
/// Take-style sequence spec `n`, `-n`, `{n}`, `{-n}`, `{m, n}` or
/// `{m, n, s}` (integer-valued reals count too), as the written first and
/// last positions and the step.
fn eigen_seq_spec(spec: &Expr) -> Option<(i128, i128, i128)> {
  let int = |e: &Expr| match e {
    Expr::Integer(k) => Some(*k),
    Expr::Real(r) if r.fract() == 0.0 && r.abs() < 1e15 => Some(*r as i128),
    _ => None,
  };
  match spec {
    Expr::List(items) => match items.as_slice() {
      [k] => int(k).map(|k| (k, k, 1)),
      [m, n] => Some((int(m)?, int(n)?, 1)),
      [m, n, st] => Some((int(m)?, int(n)?, int(st)?)).filter(|t| t.2 != 0),
      _ => None,
    },
    _ => int(spec).map(|k| if k >= 0 { (1, k, 1) } else { (k, -1, 1) }),
  }
}

/// The checks `head[matrix, spec]` makes before computing anything: the
/// matrix (`::matsq`), then the spec (`::seqs`). Returns the arguments
/// (a `{m}` matrix unwrapped) and the spec, or the unevaluated call.
fn eigen_count_args(
  head: &str,
  args: &[Expr],
) -> Result<(Vec<Expr>, (i128, i128, i128)), Expr> {
  let args = unwrap_single_matrix_list(args).unwrap_or_else(|| args.to_vec());
  let square =
    matches!(expr_to_matrix(&args[0]), Some(m) if is_nonempty_square(&m));
  if !square && is_matsq_subject(&args[0]) {
    return Err(matsq_unevaluated(head, &args));
  }
  let Some(spec) = eigen_seq_spec(&args[1]) else {
    crate::emit_message(&format!(
      "{head}::seqs: Sequence specification (+n, -n, {{+n}}, {{-n}}, {{m, n}} \
       or {{m, n, s}}) expected at position 2 in {}.",
      crate::syntax::format_expr(
        &unevaluated(head, &args),
        crate::syntax::ExprForm::Output
      )
    ));
    return Err(unevaluated(head, &args));
  };
  Ok((args, spec))
}

/// The items a spec takes out of `total`, as Take would, or None (after
/// `head::take`, naming `what`) when it reaches past them.
fn eigen_take_indices(
  head: &str,
  what: &str,
  (lo, hi, step): (i128, i128, i128),
  total: usize,
) -> Option<Vec<usize>> {
  let n = total as i128;
  let resolve = |i: i128| if i < 0 { n + 1 + i } else { i };
  let (from, to) = (resolve(lo), resolve(hi));
  let valid = if step > 0 {
    (1..=n + 1).contains(&from) && (0..=n).contains(&to)
  } else {
    (0..=n).contains(&from) && (1..=n + 1).contains(&to)
  };
  if !valid {
    crate::emit_message(&format!(
      "{head}::take: Cannot take {what} {lo} through {hi} out of the total of {n} {what}."
    ));
    return None;
  }
  let mut out = Vec::new();
  let mut i = from;
  while (step > 0 && i <= to) || (step < 0 && i >= to) {
    out.push((i - 1) as usize);
    i += step;
  }
  Some(out)
}

fn pick(items: &[Expr], indices: &[usize]) -> Expr {
  Expr::List(indices.iter().map(|&i| items[i].clone()).collect())
}

/// Eigenvalues[matrix, spec] — Take[Eigenvalues[matrix], spec]; a spec past
/// the eigenvalues emits Eigenvalues::take and stays unevaluated.
pub fn eigenvalues_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (args, spec) = match eigen_count_args("Eigenvalues", args) {
    Ok(parts) => parts,
    Err(held) => return Ok(held),
  };
  let Expr::List(items) = &eigenvalues_ast(&args[0..1])? else {
    return Ok(unevaluated("Eigenvalues", &args));
  };
  Ok(
    match eigen_take_indices("Eigenvalues", "eigenvalues", spec, items.len()) {
      Some(indices) => pick(items, &indices),
      None => unevaluated("Eigenvalues", &args),
    },
  )
}

/// Eigenvectors[matrix, spec] — Take[Eigenvectors[matrix], spec].
pub fn eigenvectors_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (args, spec) = match eigen_count_args("Eigenvectors", args) {
    Ok(parts) => parts,
    Err(held) => return Ok(held),
  };
  let Expr::List(items) = &eigenvectors_ast(&args[0..1])? else {
    return Ok(unevaluated("Eigenvectors", &args));
  };
  Ok(
    match eigen_take_indices("Eigenvectors", "eigenvectors", spec, items.len())
    {
      Some(indices) => pick(items, &indices),
      None => unevaluated("Eigenvectors", &args),
    },
  )
}

/// Eigensystem[matrix, spec] — {values, vectors}, each taken by the spec.
pub fn eigensystem_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (args, spec) = match eigen_count_args("Eigensystem", args) {
    Ok(parts) => parts,
    Err(held) => return Ok(held),
  };
  // The first k only: the exact path computes no more vectors than asked.
  if let (1, k, 1) = spec
    && k > 0
    && let Ok(Expr::List(vals)) = &eigenvalues_ast(&args[0..1])
    && k as usize <= vals.len()
    && let Ok(Expr::List(vecs)) =
      &eigenvectors_impl(&args[0..1], Some(k as usize))
    && vecs.len() == k as usize
  {
    return Ok(Expr::List(
      vec![
        Expr::List(vals[..k as usize].to_vec().into()),
        Expr::List(vecs.clone()),
      ]
      .into(),
    ));
  }
  let all = eigensystem_ast(&args[0..1])?;
  let (vals, vecs) = match &all {
    Expr::List(parts) => match parts.as_slice() {
      [Expr::List(vl), Expr::List(vc)] => (vl, vc),
      _ => return Ok(unevaluated("Eigensystem", &args)),
    },
    _ => return Ok(unevaluated("Eigensystem", &args)),
  };
  Ok(
    match eigen_take_indices(
      "Eigensystem",
      "eigenvectors and eigenvalues",
      spec,
      vals.len(),
    ) {
      Some(indices) => {
        Expr::List(vec![pick(vals, &indices), pick(vecs, &indices)].into())
      }
      None => unevaluated("Eigensystem", &args),
    },
  )
}

// ─── Eigenvectors ───────────────────────────────────────────────────────

/// Eigenvectors[matrix] - eigenvectors of a square matrix
pub fn eigenvectors_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let Some(inner) = unwrap_single_matrix_list(args) {
    return eigenvectors_ast(&inner);
  }
  eigenvectors_impl(args, None)
}

/// Eigenvectors, or with `wanted` = Some(k) possibly just the first k of
/// them (only the exact algebraic path, the costly one, stops early).
fn eigenvectors_impl(
  args: &[Expr],
  wanted: Option<usize>,
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Eigenvectors expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) if is_nonempty_square(&m) => m,
    _ => return Ok(matsq_unevaluated("Eigenvectors", args)),
  };
  let n = matrix.len();

  // 1x1: the unit vector, a machine number for a machine matrix.
  if n == 1 {
    let one = if crate::functions::math_ast::contains_inexact_real(&args[0]) {
      Expr::Real(1.0)
    } else {
      Expr::Integer(1)
    };
    return Ok(Expr::List(vec![Expr::List(vec![one].into())].into()));
  }

  // A common symbolic factor doesn't change the eigenvectors.
  if let Some((_, numeric)) = split_symbolic_scale(&matrix) {
    let result = eigenvectors_ast(&[matrix_to_expr(numeric)])?;
    // An unresolved inner call must not leak the scale-stripped matrix.
    return Ok(match &result {
      Expr::FunctionCall { name, .. } if name == "Eigenvectors" => {
        unevaluated("Eigenvectors", args)
      }
      _ => result,
    });
  }

  // Check if matrix is numeric (at least one Real entry)
  let has_real = matrix
    .iter()
    .any(|row| row.iter().any(|e| matches!(e, Expr::Real(_))));
  // Exact rationals mixed with machine reals (e.g. a covariance matrix)
  // are inexact overall, so they are converted to reals.
  let matrix: Vec<Vec<Expr>> = if has_real {
    matrix
      .into_iter()
      .map(|row| {
        row
          .into_iter()
          .map(|e| match &e {
            Expr::FunctionCall { name, .. } if name == "Rational" => {
              try_eval_to_f64(&e).map_or(e, Expr::Real)
            }
            _ => e,
          })
          .collect()
      })
      .collect()
  } else {
    matrix
  };
  let all_numeric = matrix.iter().all(|row| {
    row
      .iter()
      .all(|e| matches!(e, Expr::Integer(_) | Expr::Real(_)))
  });

  if has_real && all_numeric {
    if n >= 3
      && let Some(result) = complex_numeric_eigenvectors(&matrix, n)
    {
      return Ok(result);
    }
    let real_vectors = numeric_eigenvectors(&matrix, n);
    // A real matrix with complex eigenvalues has complex eigenvectors,
    // found the way a complex matrix's are.
    if matches!(&real_vectors, Expr::List(v) if v.len() < n)
      && let Some(result) = complex_machine_eigenvectors(args, &matrix, true)
    {
      return Ok(result);
    }
    return Ok(real_vectors);
  }

  // Try integer matrix path
  if let Some(int_matrix) = matrix_to_i128(&matrix) {
    let result = integer_eigenvectors(&matrix, &int_matrix, n, wanted);
    // Real 2×2 matrices with complex eigenvalues (e.g. rotations) fall
    // through to the exact complex path below instead of staying
    // unevaluated.
    let unresolved = matches!(
      &result,
      Expr::FunctionCall { name, .. } if name == "Eigenvectors"
    );
    if !unresolved || n != 2 {
      return Ok(result);
    }
    if let Some(complex_result) = complex_2x2_eigenvectors(&matrix)? {
      return Ok(complex_result);
    }
    return Ok(result);
  }

  // Exact complex (Gaussian-rational) 2×2 matrices.
  if n == 2
    && matrix
      .iter()
      .all(|row| row.iter().all(|e| expr_to_gq(e).is_some()))
    && matrix
      .iter()
      .any(|row| row.iter().any(contains_imaginary_unit))
    && let Some(result) = complex_2x2_eigenvectors(&matrix)?
  {
    return Ok(result);
  }

  if let Some(result) = complex_machine_eigenvectors(args, &matrix, false) {
    return Ok(result);
  }

  if n >= 3
    && let Some(result) = exact_nullspace_eigenvectors(args, &matrix)?
  {
    return Ok(result);
  }

  Ok(unevaluated("Eigenvectors", args))
}

/// Eigenvectors of an exact matrix whose `Eigenvalues` resolve: for every
/// distinct eigenvalue `l` the null space of `A - l I`, zero-padded up to the
/// eigenvalue's multiplicity (defective eigenvalues get zero vectors, as in
/// the Wolfram Language). `None` when the eigenvalues stay unresolved.
fn exact_nullspace_eigenvectors(
  args: &[Expr],
  matrix: &[Vec<Expr>],
) -> Result<Option<Expr>, InterpreterError> {
  let n = matrix.len();
  if matrix.iter().flatten().any(|e| {
    crate::functions::math_ast::contains_inexact_real(e)
      || expr_to_gq(e).is_none()
  }) {
    return Ok(None);
  }
  let values = match &eigenvalues_ast(args)? {
    Expr::List(vs) if vs.len() == n => vs.clone(),
    _ => return Ok(None),
  };
  let mat_expr = matrix_to_expr(matrix.to_vec());
  let identity =
    evaluate_expr_to_expr(&call1("IdentityMatrix", Expr::Integer(n as i128)))?;
  let mut out: Vec<Expr> = Vec::with_capacity(n);
  let mut seen: Vec<String> = Vec::new();
  for l in &values {
    let key = expr_to_string(l);
    if seen.contains(&key) {
      continue;
    }
    seen.push(key.clone());
    let multiplicity =
      values.iter().filter(|v| expr_to_string(v) == key).count();
    let shifted = evaluate_expr_to_expr(&call(
      "Subtract",
      vec![
        mat_expr.clone(),
        call("Times", vec![l.clone(), identity.clone()]),
      ],
    ))?;
    let null = evaluate_expr_to_expr(&call1("NullSpace", shifted))?;
    let Expr::List(vecs) = &null else {
      return Ok(None);
    };
    if vecs.len() > multiplicity {
      return Ok(None);
    }
    out.extend(vecs.iter().cloned());
    for _ in vecs.len()..multiplicity {
      out.push(Expr::List(vec![Expr::Integer(0); n].into()));
    }
  }
  Ok(Some(Expr::List(out.into())))
}

/// Eigenvectors of a machine matrix with complex entries, one per eigenvalue
/// in `Eigenvalues`' order, found by inverse iteration on the real 2n×2n
/// embedding of `A - λI`. The vectors of a repeated eigenvalue are kept
/// orthogonal to each other, so a normal matrix gets a full eigenbasis. As
/// LAPACK's `zgeev` does, each vector has unit length and a real, positive
/// component of largest modulus.
fn complex_machine_eigenvectors(
  args: &[Expr],
  matrix: &[Vec<Expr>],
  real_entries_allowed: bool,
) -> Option<Expr> {
  type C = (f64, f64);
  let n = matrix.len();
  if !(real_entries_allowed
    || matrix.iter().flatten().any(|c| {
      crate::functions::math_ast::try_extract_complex_float(c)
        .is_some_and(|(_, im)| im != 0.0)
    }))
    || !matrix
      .iter()
      .flatten()
      .any(crate::functions::math_ast::contains_inexact_real)
  {
    return None;
  }
  let entries: Vec<Vec<C>> = matrix
    .iter()
    .map(|row| {
      row
        .iter()
        .map(|c| {
          crate::functions::math_ast::try_extract_complex_float(c)
            .or_else(|| try_eval_to_f64(c).map(|re| (re, 0.0)))
        })
        .collect::<Option<Vec<_>>>()
    })
    .collect::<Option<_>>()?;
  let values = eigenvalues_ast(args).ok()?;
  let Expr::List(values) = &values else {
    return None;
  };
  let values: Vec<C> = values
    .iter()
    .map(|v| {
      crate::functions::math_ast::try_extract_complex_float(v)
        .or_else(|| try_eval_to_f64(v).map(|re| (re, 0.0)))
    })
    .collect::<Option<_>>()?;
  if values.len() != n {
    return None;
  }
  let scale = entries
    .iter()
    .flatten()
    .fold(1.0f64, |acc, &(re, im)| acc.max(re.hypot(im)));
  let dot = |a: &[C], b: &[C]| -> C {
    // conj(a) . b
    a.iter()
      .zip(b)
      .fold((0.0, 0.0), |acc, (&(ar, ai), &(br, bi))| {
        (acc.0 + ar * br + ai * bi, acc.1 + ar * bi - ai * br)
      })
  };
  let normalize = |v: &mut Vec<C>| -> bool {
    let norm = v.iter().map(|(r, i)| r * r + i * i).sum::<f64>().sqrt();
    if norm <= 1e-300 || !norm.is_finite() {
      return false;
    }
    for c in v.iter_mut() {
      *c = (c.0 / norm, c.1 / norm);
    }
    true
  };
  let mut found: Vec<(C, Vec<C>)> = Vec::with_capacity(n);
  let mut out = Vec::with_capacity(n);
  for (k, &(lr, li)) in values.iter().enumerate() {
    let cluster: Vec<Vec<C>> = found
      .iter()
      .filter(|((r, i), _)| (r - lr).hypot(i - li) <= 1e-8 * scale)
      .map(|(_, v)| v.clone())
      .collect();
    let (sr, si) = (lr + scale * 1e-10, li);
    let mut m = vec![vec![0.0; 2 * n]; 2 * n];
    for i in 0..n {
      for j in 0..n {
        let (mut ar, mut ai) = entries[i][j];
        if i == j {
          ar -= sr;
          ai -= si;
        }
        m[i][j] = ar;
        m[i][n + j] = -ai;
        m[n + i][j] = ai;
        m[n + i][n + j] = ar;
      }
    }
    // A start vector of its own for each member of a cluster.
    let mut v: Vec<C> = (0..n)
      .map(|i| {
        let t = (i * (k + 1)) as f64;
        (
          1.0 + 0.5 * (1.3 * t + 0.7).sin(),
          0.5 * (0.9 * t + 0.3).cos(),
        )
      })
      .collect();
    let orthogonalize = |v: &mut Vec<C>| {
      for u in &cluster {
        let (pr, pi) = dot(u, v);
        for (c, &(ur, ui)) in v.iter_mut().zip(u) {
          *c = (c.0 - (pr * ur - pi * ui), c.1 - (pr * ui + pi * ur));
        }
      }
    };
    orthogonalize(&mut v);
    if !normalize(&mut v) {
      return None;
    }
    for _ in 0..4 {
      let rhs: Vec<f64> =
        v.iter().map(|c| c.0).chain(v.iter().map(|c| c.1)).collect();
      let y = solve_linear_system(&m, &rhs)?;
      v = (0..n).map(|i| (y[i], y[n + i])).collect();
      orthogonalize(&mut v);
      if !normalize(&mut v) {
        return None;
      }
    }
    // Rotate the phase so the largest-modulus component — the first of
    // several equally large ones, as LAPACK picks it — is real positive.
    let largest = v.iter().map(|c| c.0.hypot(c.1)).fold(0.0, f64::max);
    let bidx = v
      .iter()
      .position(|c| c.0.hypot(c.1) >= largest * (1.0 - 1e-12))?;
    let (br, bi) = v[bidx];
    let mag = br.hypot(bi);
    let (pr, pi) = (br / mag, -bi / mag);
    let rotated: Vec<C> = v
      .iter()
      .enumerate()
      .map(|(i, &(re, im))| {
        if i == bidx {
          (mag, 0.0)
        } else {
          (re * pr - im * pi, re * pi + im * pr)
        }
      })
      .collect();
    out.push(Expr::List(
      rotated
        .iter()
        .map(|&(re, im)| {
          // Rounding noise in a unit vector (a part ~1e-17) is zero.
          let chop = |x: f64| if x.abs() < 1e-14 { 0.0 } else { x };
          crate::functions::math_ast::build_complex_float_expr_keep_real(
            chop(re),
            chop(im),
          )
        })
        .collect(),
    ));
    found.push(((lr, li), rotated));
  }
  Some(Expr::List(out.into()))
}

/// Eigenvectors of an exact complex 2×2 matrix, following wolframscript's
/// conventions (the same ones the integer path uses):
/// - c ≠ 0: v = ((λ − d)/c, 1); an exact Gaussian-rational first component
///   has its denominator cleared, e.g. ((1+I)/2, 1) → (1 + I, 2)
/// - upper triangular: (1, 0) for λ = a, (b/(d − a), 1) for λ = d
/// - diagonal: permuted basis vectors; defective matrices append a zero
///   vector after the single eigenvector.
///   Returns Ok(None) when the eigenvalues don't come out as a clean pair.
fn complex_2x2_eigenvectors(
  matrix: &[Vec<Expr>],
) -> Result<Option<Expr>, InterpreterError> {
  let (a, b) = (&matrix[0][0], &matrix[0][1]);
  let (c, d) = (&matrix[1][0], &matrix[1][1]);
  let eigen = eigenvalues_ast(&[matrix_to_expr(matrix.to_vec())])?;
  let (l1, l2) = match &eigen {
    Expr::List(ls) if ls.len() == 2 => (ls[0].clone(), ls[1].clone()),
    _ => return Ok(None),
  };
  let b_zero = expr_to_gq(b).is_some_and(|v| v.is_zero());
  let c_zero = expr_to_gq(c).is_some_and(|v| v.is_zero());
  let repeated = expr_to_string(&l1) == expr_to_string(&l2);

  let zero_row = || vec![Expr::Integer(0), Expr::Integer(0)];
  let basis = |first: bool| {
    if first {
      vec![Expr::Integer(1), Expr::Integer(0)]
    } else {
      vec![Expr::Integer(0), Expr::Integer(1)]
    }
  };
  // ((λ − d)/c, 1) — or (b/(d − a), 1) — with an exact Gaussian-rational
  // component scaled to clear its denominator.
  let quotient_vector =
    |num_minuend: &Expr, num_subtrahend: &Expr, den: &Expr| {
      let x = evaluate_expr_to_expr(&div2(
        plus(vec![num_minuend.clone(), neg(num_subtrahend.clone())]),
        den.clone(),
      ))?;
      if let Some(gq) = expr_to_gq(&x) {
        let q = (gq.re.d / gcd_i128(gq.re.d, gq.im.d)).checked_mul(gq.im.d);
        if let Some(q) = q
          && q != 1
          && let (Some(re_n), Some(im_n)) = (
            gq.re.n.checked_mul(q / gq.re.d),
            gq.im.n.checked_mul(q / gq.im.d),
          )
        {
          let scaled = GQNum {
            re: QNum { n: re_n, d: 1 },
            im: QNum { n: im_n, d: 1 },
          };
          if let Some(p) = gq_to_expr(&scaled) {
            return Ok(vec![p, Expr::Integer(q)]);
          }
        }
      }
      Ok(vec![x, Expr::Integer(1)])
    };

  let vector_for = |lam: &Expr| -> Result<Vec<Expr>, InterpreterError> {
    if !c_zero {
      quotient_vector(lam, d, c)
    } else if b_zero {
      // Diagonal: basis vector for the matching slot.
      Ok(basis(expr_to_string(lam) == expr_to_string(a)))
    } else if expr_to_string(lam) == expr_to_string(a) {
      Ok(basis(true))
    } else {
      quotient_vector(b, &Expr::Integer(0), &{
        plus(vec![d.clone(), neg(a.clone())])
      })
    }
  };

  let rows: Vec<Vec<Expr>> = if repeated {
    if b_zero && c_zero {
      // Scalar matrix (c·Id): the eigenspace is all of C^2, so the vectors
      // come from NullSpace[0], which wolframscript returns in reversed-basis
      // order ({{0, 1}, {1, 0}}) — matching the real/integer scalar path.
      vec![basis(false), basis(true)]
    } else {
      vec![vector_for(&l1)?, zero_row()]
    }
  } else {
    vec![vector_for(&l1)?, vector_for(&l2)?]
  };
  Ok(Some(Expr::List(
    rows
      .into_iter()
      .map(|r| Expr::List(r.into()))
      .collect::<Vec<_>>()
      .into(),
  )))
}

/// Scale a vector of rational Exprs to integer entries (clear denominators, reduce GCD).
fn scale_to_integers(vec: &[Expr]) -> Vec<Expr> {
  let rationals: Vec<Option<(i128, i128)>> =
    vec.iter().map(expr_to_rational).collect();
  if rationals.iter().any(std::option::Option::is_none) {
    return vec.to_vec();
  }
  let rats: Vec<(i128, i128)> =
    rationals.into_iter().map(|r| r.unwrap()).collect();

  // Find LCM of denominators
  let lcm_den = rats.iter().fold(1i128, |acc, &(_, d)| {
    let g = gcd_i128(acc, d);
    if g == 0 { acc } else { (acc / g) * d.abs() }
  });

  // Scale all entries
  let scaled: Vec<i128> =
    rats.iter().map(|&(n, d)| n * (lcm_den / d)).collect();

  // Find GCD of all entries
  let g = scaled.iter().fold(0i128, |acc, &x| gcd_i128(acc, x));

  if g == 0 || g == 1 {
    scaled.iter().map(|&x| Expr::Integer(x)).collect()
  } else {
    scaled.iter().map(|&x| Expr::Integer(x / g)).collect()
  }
}

/// Compute eigenvectors for an integer matrix.
fn integer_eigenvectors(
  matrix: &[Vec<Expr>],
  int_matrix: &[Vec<i128>],
  n: usize,
  wanted: Option<usize>,
) -> Expr {
  let Some(coeffs) = char_poly_coefficients(int_matrix) else {
    return unevaluated("Eigenvectors", &[matrix_to_expr(matrix.to_vec())]);
  };
  let mut eigenvalues = find_polynomial_roots(&coeffs);
  sort_eigenvalues(&mut eigenvalues);

  // Check if all eigenvalues are integers
  let all_integer = eigenvalues.iter().all(|e| expr_to_i128(e).is_some());

  // Gaussian-integer eigenvalues (`-1 + I`): the null spaces of `m - λ I`,
  // a repeated eigenvalue taking its vectors in turn and a defective one
  // padded with zero vectors.
  if !all_integer
    && n >= 3
    && eigenvalues.iter().all(|e| expr_to_gq(e).is_some())
  {
    let mut eigenvectors: Vec<Expr> = Vec::new();
    let mut processed: Vec<(String, Vec<Expr>)> = Vec::new();
    for eigenvalue in &eigenvalues {
      let key = expr_to_string(eigenvalue);
      if !processed.iter().any(|(k, _)| *k == key) {
        let shifted = call(
          "Subtract",
          vec![
            matrix_to_expr(matrix.to_vec()),
            times2(
              eigenvalue.clone(),
              call1("IdentityMatrix", Expr::Integer(n as i128)),
            ),
          ],
        );
        let null_space = evaluate_expr_to_expr(&shifted).and_then(|m| {
          crate::evaluator::evaluate_function_call_ast("NullSpace", &[m])
        });
        let Ok(Expr::List(ref basis)) = null_space else {
          return call1("Eigenvectors", matrix_to_expr(matrix.to_vec()));
        };
        processed.push((key.clone(), basis.to_vec()));
      }
      let entry = processed.iter_mut().find(|(k, _)| *k == key).unwrap();
      eigenvectors.push(if entry.1.is_empty() {
        Expr::List(vec![Expr::Integer(0); n].into())
      } else {
        entry.1.remove(0)
      });
    }
    return Expr::List(eigenvectors.into());
  }

  if !all_integer && n == 2 {
    // 2x2 with symbolic eigenvalues: build eigenvectors directly
    let vecs = eigenvectors_2x2_symbolic(int_matrix);
    if !vecs.is_empty() {
      return Expr::List(
        vecs.into_iter().map(|v| Expr::List(v.into())).collect(),
      );
    }
    return call1("Eigenvectors", matrix_to_expr(matrix.to_vec()));
  }

  if !all_integer && n == 3 {
    let vecs = eigenvectors_3x3_symbolic(matrix, &eigenvalues);
    if !vecs.is_empty() {
      return Expr::List(
        vecs.into_iter().map(|v| Expr::List(v.into())).collect(),
      );
    }
    return call1("Eigenvectors", matrix_to_expr(matrix.to_vec()));
  }

  if !all_integer {
    return algebraic_eigenvectors(matrix, n, wanted).unwrap_or_else(|| {
      call1("Eigenvectors", matrix_to_expr(matrix.to_vec()))
    });
  }

  // All integer eigenvalues: compute null spaces
  let mut eigenvectors: Vec<Expr> = Vec::new();
  let mut processed: Vec<(i128, Vec<Vec<Expr>>)> = Vec::new();

  for eigenvalue in &eigenvalues {
    let lambda = expr_to_i128(eigenvalue).unwrap();

    let null_vecs =
      if let Some(entry) = processed.iter_mut().find(|(l, _)| *l == lambda) {
        &mut entry.1
      } else {
        let vecs =
          null_space_for_integer_eigenvalue(matrix, lambda, int_matrix, n);
        processed.push((lambda, vecs));
        &mut processed.last_mut().unwrap().1
      };

    if null_vecs.is_empty() {
      eigenvectors.push(Expr::List(vec![Expr::Integer(0); n].into()));
    } else {
      eigenvectors.push(Expr::List(null_vecs.remove(0).into()));
    }
  }

  Expr::List(eigenvectors.into())
}

/// Eigenvectors of an exact matrix whose eigenvalues are algebraic numbers
/// (radicals or `Root` objects): the null space of `m - λ I` for each
/// eigenvalue in `Eigenvalues` order, a repeated eigenvalue taking its
/// vectors in turn and a defective one padded with zero vectors.
fn algebraic_eigenvectors(
  matrix: &[Vec<Expr>],
  n: usize,
  wanted: Option<usize>,
) -> Option<Expr> {
  // The elimination canonicalizes algebraic numbers of degree up to n at
  // every step; past 5x5 computing all of them is too slow to be worth it
  // here (see conformance_gaps.md). One or two of them (Eigensystem[m, k])
  // stay affordable a little further.
  let limit = match wanted {
    Some(k) if k <= 2 => 8,
    _ => 5,
  };
  if n > limit {
    return None;
  }
  let eigen = eigenvalues_ast(&[matrix_to_expr(matrix.to_vec())]).ok()?;
  let Expr::List(eigenvalues) = &eigen else {
    return None;
  };
  if eigenvalues.len() != n {
    return None;
  }
  // Wolfram takes the eigenvalues as `Root` objects here, ordered by
  // decreasing absolute value, complex before real, then increasing real
  // part, then decreasing imaginary part (a conjugate pair lists `a + b I`
  // first).
  let mut keyed: Vec<((f64, f64), Expr)> = Vec::with_capacity(n);
  for lambda in eigenvalues {
    let value = evaluate_expr_to_expr(&call1("N", lambda.clone())).ok()?;
    let parts =
      crate::functions::math_ast::numeric_utils::try_extract_complex_float(
        &value,
      )
      .or_else(|| try_eval_to_f64(&value).map(|re| (re, 0.0)))?;
    keyed.push((parts, lambda.clone()));
  }
  // Radical eigenvalues (`(-1 + I Sqrt[3])/2`) keep `Eigenvalues`' order.
  let has_root = eigenvalues.iter().any(|l| {
    crate::functions::polynomial_ast::solve::expr_contains_expr(
      l,
      &Expr::Identifier("Root".into()),
    ) || expr_to_string(l).contains("Root[")
  });
  keyed.sort_by(|((ar, ai), _), ((br, bi), _)| {
    if !has_root {
      return std::cmp::Ordering::Equal;
    }
    let (abs_a, abs_b) = (ar.hypot(*ai), br.hypot(*bi));
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-10 * x.abs().max(1.0);
    let (real_a, real_b) = (ai.abs() <= 1e-12, bi.abs() <= 1e-12);
    if !close(abs_a, abs_b) {
      abs_b.total_cmp(&abs_a)
    } else if real_a != real_b {
      // Among equal magnitudes the complex eigenvalues come first.
      real_a.cmp(&real_b)
    } else if !close(*ar, *br) {
      ar.total_cmp(br)
    } else {
      bi.total_cmp(ai)
    }
  });
  let mut eigenvalues: Vec<Expr> = keyed.into_iter().map(|(_, l)| l).collect();
  if let Some(k) = wanted {
    eigenvalues.truncate(k);
  }
  let mut groups: Vec<(Expr, Vec<Expr>)> = Vec::new();
  let mut rows: Vec<Expr> = Vec::with_capacity(n);
  for lambda in &eigenvalues {
    let pos = if let Some(pos) = groups
      .iter()
      .position(|(l, _)| expr_to_string(l) == expr_to_string(lambda))
    {
      pos
    } else {
      let shifted: Vec<Vec<Expr>> = (0..n)
        .map(|i| {
          (0..n)
            .map(|j| {
              if i == j {
                simplify_expr(&plus(vec![
                  matrix[i][j].clone(),
                  neg(lambda.clone()),
                ]))
              } else {
                matrix[i][j].clone()
              }
            })
            .collect()
        })
        .collect();
      let space = null_space_ast(&[matrix_to_expr(shifted)]).ok()?;
      let Expr::List(space) = &space else {
        return None;
      };
      if space.is_empty() {
        return None;
      }
      // Each entry in canonical form: `-Root[f, 2]` is `Root[g, k]`.
      let space: Vec<Expr> = space
        .iter()
        .map(|v| match v {
          Expr::List(items) => {
            Expr::List(items.iter().map(simplify_expr).collect())
          }
          other => other.clone(),
        })
        .collect();
      groups.push((lambda.clone(), space));
      groups.len() - 1
    };
    let vecs = &mut groups[pos].1;
    rows.push(if vecs.is_empty() {
      Expr::List(vec![Expr::Integer(0); n].into())
    } else {
      vecs.remove(0)
    });
  }
  Some(Expr::List(rows.into()))
}

/// Compute null space vectors for a specific integer eigenvalue.
fn null_space_for_integer_eigenvalue(
  matrix: &[Vec<Expr>],
  lambda: i128,
  int_matrix: &[Vec<i128>],
  n: usize,
) -> Vec<Vec<Expr>> {
  let mut m: Vec<Vec<Expr>> = matrix.to_vec();
  for i in 0..n {
    m[i][i] = Expr::Integer(int_matrix[i][i] - lambda);
  }

  let rref = row_reduce_impl(&m);

  // Find pivot columns
  let mut pivot_cols = Vec::new();
  let mut pivot_row_for_col = vec![None; n];
  let mut current_row = 0;
  for col in 0..n {
    if current_row >= n {
      break;
    }
    if is_one_expr(&rref[current_row][col]) {
      pivot_cols.push(col);
      pivot_row_for_col[col] = Some(current_row);
      current_row += 1;
    }
  }

  let free_cols: Vec<usize> =
    (0..n).filter(|c| !pivot_cols.contains(c)).collect();

  // Build null space basis vectors (reverse order to match Wolfram convention)
  let mut basis = Vec::new();
  for &free_col in free_cols.iter().rev() {
    let mut vec = vec![Expr::Integer(0); n];
    vec[free_col] = Expr::Integer(1);
    for &pivot_col in &pivot_cols {
      if let Some(row) = pivot_row_for_col[pivot_col] {
        vec[pivot_col] = eval_sub(&Expr::Integer(0), &rref[row][free_col]);
      }
    }
    let scaled = scale_to_integers(&vec);
    basis.push(scaled);
  }
  basis
}

/// Build eigenvector expressions for 2x2 symbolic case (irrational eigenvalues).
/// Returns eigenvectors in the same order as eigenvalues (sorted by absolute value).
fn eigenvectors_2x2_symbolic(int_matrix: &[Vec<i128>]) -> Vec<Vec<Expr>> {
  let a = int_matrix[0][0];
  let b = int_matrix[0][1];
  let c = int_matrix[1][0];
  let d = int_matrix[1][1];

  // Discriminant: (a-d)^2 + 4bc
  let diff = a - d;
  let disc = diff * diff + 4 * b * c;

  if disc <= 0 {
    // Complex eigenvalues or zero discriminant (should be handled elsewhere)
    return vec![];
  }

  // Check if disc is a perfect square (should have been handled by integer path)
  let sqrt_disc_approx = (disc as f64).sqrt().round() as i128;
  if sqrt_disc_approx * sqrt_disc_approx == disc {
    return vec![]; // Integer eigenvalues, should be handled elsewhere
  }

  // Simplify sqrt(disc)
  let (outer, inner) = simplify_sqrt(disc.unsigned_abs());

  if c != 0 {
    // Eigenvector: {(diff ± outer*sqrt(inner)) / (2c), 1}
    let denom = 2 * c;

    // Build the two eigenvectors
    let build_v1 = |sign: i128| -> Expr {
      let signed_outer = sign * outer as i128;
      // Simplify: GCD of |diff|, |signed_outer|, |denom|
      let g = gcd_i128(gcd_i128(diff, signed_outer), denom);
      let rd = diff / g;
      let ro = (signed_outer / g).abs();
      let rden = denom / g;

      let sqrt_expr = make_sqrt(Expr::Integer(inner as i128));

      let sqrt_term = if ro == 1 {
        sqrt_expr
      } else {
        eval_mul(&Expr::Integer(ro), &sqrt_expr)
      };

      let numer = if rd == 0 {
        if sign > 0 {
          sqrt_term
        } else {
          eval_sub(&Expr::Integer(0), &sqrt_term)
        }
      } else if sign > 0 {
        eval_add(&Expr::Integer(rd), &sqrt_term)
      } else {
        eval_sub(&Expr::Integer(rd), &sqrt_term)
      };

      if rden == 1 {
        numer
      } else if rden == -1 {
        eval_sub(&Expr::Integer(0), &numer)
      } else {
        eval_divide(&numer, &Expr::Integer(rden))
      }
    };

    let v1_plus = build_v1(1);
    let v1_minus = build_v1(-1);

    // Determine which eigenvalue is larger by absolute value
    // λ = (a+d ± sqrt(disc)) / 2
    let trace = a + d;
    let sqrt_val = (disc as f64).sqrt();
    let l_plus = f64::midpoint(trace as f64, sqrt_val);
    let l_minus = (trace as f64 - sqrt_val) / 2.0;

    if l_plus.abs() >= l_minus.abs() {
      vec![
        vec![v1_plus, Expr::Integer(1)],
        vec![v1_minus, Expr::Integer(1)],
      ]
    } else {
      vec![
        vec![v1_minus, Expr::Integer(1)],
        vec![v1_plus, Expr::Integer(1)],
      ]
    }
  } else if b != 0 {
    // c = 0, use row 0: (a-λ)*v1 + b*v2 = 0 → v = {1, (λ-a)/b}
    // Eigenvector: {1, (λ-a)/b} = {1, (d-a ± sqrt(disc)) / (2b)}
    // But since c=0, eigenvalues are a and d (integers), handled by integer path
    vec![]
  } else {
    vec![]
  }
}

/// Compute eigenvectors for a 3×3 matrix with symbolic eigenvalues (e.g.
/// containing `Sqrt[d]`). For each eigenvalue λ, takes the cross product
/// of two rows of A-λI; if the result is non-zero it spans the (1-D) null
/// space. The last non-zero entry is normalized to 1.
fn eigenvectors_3x3_symbolic(
  matrix: &[Vec<Expr>],
  eigenvalues: &[Expr],
) -> Vec<Vec<Expr>> {
  let mut out = Vec::new();
  for lambda in eigenvalues {
    let r0 = vec![
      eval_sub(&matrix[0][0], lambda),
      matrix[0][1].clone(),
      matrix[0][2].clone(),
    ];
    let r1 = vec![
      matrix[1][0].clone(),
      eval_sub(&matrix[1][1], lambda),
      matrix[1][2].clone(),
    ];
    let r2 = vec![
      matrix[2][0].clone(),
      matrix[2][1].clone(),
      eval_sub(&matrix[2][2], lambda),
    ];
    // Try row pairs in the order wolframscript appears to prefer:
    // rows 1×2 first, then 0×2, then 0×1.
    let mut v = simplify_vec(cross_product_3d(&r1, &r2));
    if is_zero_expr_vec(&v) {
      v = simplify_vec(cross_product_3d(&r0, &r2));
    }
    if is_zero_expr_vec(&v) {
      v = simplify_vec(cross_product_3d(&r0, &r1));
    }
    if is_zero_expr_vec(&v) {
      return vec![];
    }
    out.push(normalize_symbolic_eigenvector(v));
  }
  out
}

fn cross_product_3d(a: &[Expr], b: &[Expr]) -> Vec<Expr> {
  vec![
    eval_sub(&eval_mul(&a[1], &b[2]), &eval_mul(&a[2], &b[1])),
    eval_sub(&eval_mul(&a[2], &b[0]), &eval_mul(&a[0], &b[2])),
    eval_sub(&eval_mul(&a[0], &b[1]), &eval_mul(&a[1], &b[0])),
  ]
}

fn simplify_vec(v: Vec<Expr>) -> Vec<Expr> {
  v.into_iter()
    .map(|e| {
      let evaluated = evaluate_expr_to_expr(&e).unwrap_or(e);
      simplify_via_full_simplify(&evaluated)
    })
    .collect()
}

fn simplify_via_full_simplify(e: &Expr) -> Expr {
  crate::functions::polynomial_ast::simplify_ast(std::slice::from_ref(e))
    .unwrap_or_else(|_| e.clone())
}

fn is_zero_symbolic(e: &Expr) -> bool {
  match e {
    Expr::Integer(0) => true,
    Expr::Real(x) => x.abs() < 1e-14,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!(&args[0], Expr::Integer(0))
    }
    _ => false,
  }
}

fn is_zero_expr_vec(v: &[Expr]) -> bool {
  v.iter().all(is_zero_symbolic)
}

/// Normalize a symbolic eigenvector. If all components are integers,
/// reduce by GCD with a sign chosen so the leading non-zero entry is
/// positive. Otherwise divide each entry by the last non-zero entry so
/// that the last coordinate becomes 1.
fn normalize_symbolic_eigenvector(v: Vec<Expr>) -> Vec<Expr> {
  let all_int = v.iter().all(|e| matches!(e, Expr::Integer(_)));
  if all_int {
    let ints: Vec<i128> = v
      .iter()
      .filter_map(|e| {
        if let Expr::Integer(x) = e {
          Some(*x)
        } else {
          None
        }
      })
      .collect();
    let g = ints.iter().fold(0i128, |acc, &x| gcd_i128(acc, x));
    if g == 0 {
      return v;
    }
    let leading_neg = ints.iter().find(|&&x| x != 0).is_some_and(|&x| x < 0);
    let sign: i128 = if leading_neg { -1 } else { 1 };
    return ints
      .into_iter()
      .map(|x| Expr::Integer(sign * (x / g)))
      .collect();
  }
  // Symbolic: divide by the last non-zero entry.
  let last_nz = (0..v.len()).rev().find(|&i| !is_zero_symbolic(&v[i]));
  if let Some(idx) = last_nz {
    let divisor = v[idx].clone();
    return v
      .iter()
      .enumerate()
      .map(|(i, e)| {
        if i == idx {
          Expr::Integer(1)
        } else {
          let q = eval_divide(e, &divisor);
          evaluate_expr_to_expr(&q).unwrap_or(q)
        }
      })
      .collect();
  }
  v
}

/// Eigenvectors of a real float matrix (n ≥ 3) that has complex-conjugate
/// eigenvalues; `None` when all eigenvalues are real (the real path handles
/// that) or the iteration fails. Vectors follow `Eigenvalues`' order, have
/// unit length, and — as LAPACK's `dgeev` does — a real largest component.
/// Eigenvectors of real eigenvalues stay real.
fn complex_numeric_eigenvectors(
  matrix: &[Vec<Expr>],
  n: usize,
) -> Option<Expr> {
  let f_matrix: Vec<Vec<f64>> = matrix
    .iter()
    .map(|row| row.iter().map(try_eval_to_f64).collect())
    .collect::<Option<_>>()?;
  let eigenvalues = numeric_eigenvalues_complex(&f_matrix, n);
  if eigenvalues.len() != n || eigenvalues.iter().all(|&(_, im)| im == 0.0) {
    return None;
  }
  let scale = f_matrix
    .iter()
    .flat_map(|row| row.iter())
    .fold(1.0f64, |acc, v| acc.max(v.abs()));
  let mut out = Vec::with_capacity(n);
  for &(lr, li) in &eigenvalues {
    // Inverse iteration on the real 2n×2n embedding of (A - λI) y = x.
    let (sr, si) = (lr + scale * 1e-10, li);
    let mut m = vec![vec![0.0; 2 * n]; 2 * n];
    for i in 0..n {
      for j in 0..n {
        let a = f_matrix[i][j] - if i == j { sr } else { 0.0 };
        m[i][j] = a;
        m[n + i][n + j] = a;
      }
      m[i][n + i] = si;
      m[n + i][i] = -si;
    }
    let mut x = vec![1.0; 2 * n];
    for _ in 0..4 {
      let y = solve_linear_system(&m, &x)?;
      let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
      if norm <= 1e-300 || !norm.is_finite() {
        return None;
      }
      x = y.iter().map(|v| v / norm).collect();
    }
    let comps: Vec<(f64, f64)> = (0..n).map(|i| (x[i], x[n + i])).collect();
    // Rotate the phase so the largest-magnitude component is real positive.
    let (bidx, &(br, bi)) = comps
      .iter()
      .enumerate()
      .max_by(|a, b| a.1.0.hypot(a.1.1).total_cmp(&b.1.0.hypot(b.1.1)))?;
    let mag = br.hypot(bi);
    let (pr, pi) = (br / mag, -bi / mag);
    let vec: Vec<Expr> = comps
      .iter()
      .enumerate()
      .map(|(k, &(re, im))| {
        let (re, im) = (re * pr - im * pi, re * pi + im * pr);
        // The pivot component is real by construction; drop rounding noise.
        let (re, im) = if k == bidx { (mag, 0.0) } else { (re, im) };
        if li == 0.0 {
          Expr::Real(re)
        } else {
          crate::functions::math_ast::build_complex_float_expr_keep_real(re, im)
        }
      })
      .collect();
    out.push(Expr::List(vec.into()));
  }
  Some(Expr::List(out.into()))
}

/// Compute eigenvectors for a numeric (f64) matrix.
fn numeric_eigenvectors(matrix: &[Vec<Expr>], n: usize) -> Expr {
  // Convert to f64 matrix
  let f_matrix: Vec<Vec<f64>> = matrix
    .iter()
    .map(|row| {
      row
        .iter()
        .map(|e| match e {
          Expr::Integer(v) => *v as f64,
          Expr::Real(v) => *v,
          _ => 0.0,
        })
        .collect()
    })
    .collect();

  // Compute eigenvalues numerically using characteristic polynomial
  let eigenvalues = numeric_eigenvalues(&f_matrix, n);

  let mut eigenvectors = Vec::new();
  // Normalized eigenvectors already emitted for the run of eigenvalues equal
  // to the one currently being processed (eigenvalues are sorted, so a
  // repeated eigenvalue's occurrences are adjacent). `(A - λI)` is identical
  // across the whole run, so its null space is too — without tracking what
  // has already been handed out, every occurrence would independently pick
  // the same first free-column vector out of that shared null space instead
  // of a genuinely different one spanning the rest of the eigenspace.
  let mut group_lambda: Option<f64> = None;
  let mut group_prev: Vec<Vec<f64>> = Vec::new();
  for &lambda in &eigenvalues {
    if group_lambda.is_none_or(|g| (g - lambda).abs() > 1e-9) {
      group_lambda = Some(lambda);
      group_prev.clear();
    }

    // Build (A - λI)
    let mut m = f_matrix.clone();
    for i in 0..n {
      m[i][i] -= lambda;
    }

    // Find a null space vector independent from any already used for this
    // same eigenvalue: Gram-Schmidt each free-column basis vector against
    // the ones already emitted in this group and take the first that still
    // has a non-negligible component left afterward.
    let mut vec = numeric_null_space_basis(&m, n)
      .into_iter()
      .map(|mut candidate| {
        for prev in &group_prev {
          let dot: f64 = candidate.iter().zip(prev).map(|(a, b)| a * b).sum();
          for (c, p) in candidate.iter_mut().zip(prev) {
            *c -= dot * p;
          }
        }
        candidate
      })
      .find(|candidate| {
        candidate.iter().map(|x| x * x).sum::<f64>().sqrt() > 1e-9
      })
      .unwrap_or_else(|| vec![0.0; n]);

    // The fixed pivot tolerance inside `numeric_null_space_basis` is tuned
    // for well-conditioned small matrices; for a larger matrix (or one with
    // several eigenvalues close enough together to leave only a small gap
    // between them) the residual along the true null direction can sit
    // above that tolerance, so no column gets recognized as free and the
    // search comes back with a zero vector instead of an eigenvector.
    // Recover it via inverse iteration instead: solving `(A - λI) y = x`
    // amplifies whatever component of `x` lies along the (near-)singular
    // direction every time, regardless of the matrix's scale, and a couple
    // of iterations are enough since the shift already sits at the
    // converged eigenvalue.
    if vec.iter().all(|x| x.abs() < 1e-15)
      && let Some(mut v) = inverse_iteration_eigenvector(&f_matrix, lambda, n)
    {
      for prev in &group_prev {
        let dot: f64 = v.iter().zip(prev).map(|(a, b)| a * b).sum();
        for (c, p) in v.iter_mut().zip(prev) {
          *c -= dot * p;
        }
      }
      // A defective matrix's repeated eigenvalue has a null space smaller
      // than its multiplicity, so inverse iteration — deterministic given
      // the same shifted matrix and starting vector — converges to the
      // same direction already recorded in `group_prev` for every
      // occurrence. Once that's subtracted out here, what's left is only
      // the tiny residual from `inverse_iteration_eigenvector`'s shift
      // regularization, not a genuine second eigenvector; require the same
      // significance the null-space search above does, so a defective
      // occurrence falls through to the zero-vector convention below
      // (matching wolframscript, e.g. `Eigenvectors[{{1, 1}, {0, 1}}]` ==
      // `{{1, 0}, {0, 0}}`) instead of fabricating an unrelated vector.
      let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
      if v_norm > 1e-9 {
        vec = v;
      }
    }

    // Normalize to unit length
    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
      // 2x2 sign convention: leading non-zero entry is non-positive,
      // matching wolframscript's GEEV-derived output. For larger
      // matrices the sign comes from LAPACK QR iterations and isn't
      // reproducible by Gaussian elimination, so we leave it alone.
      let mut normalized: Vec<f64> = vec.iter().map(|&x| x / norm).collect();
      if n == 2
        && let Some(&first) = normalized.iter().find(|&&v| v.abs() > 1e-12)
        && first > 0.0
      {
        for v in &mut normalized {
          *v = -*v;
        }
      }
      group_prev.push(normalized.clone());
      let normalized_exprs: Vec<Expr> =
        normalized.into_iter().map(Expr::Real).collect();
      eigenvectors.push(Expr::List(normalized_exprs.into()));
    } else {
      eigenvectors.push(Expr::List(vec![Expr::Real(0.0); n].into()));
    }
  }

  Expr::List(eigenvectors.into())
}

/// Compute eigenvalues of a f64 matrix numerically.
fn numeric_eigenvalues(matrix: &[Vec<f64>], n: usize) -> Vec<f64> {
  if n == 1 {
    return vec![matrix[0][0]];
  }
  if n == 2 {
    let tr = matrix[0][0] + matrix[1][1];
    let det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
      let sq = disc.sqrt();
      let l1 = f64::midpoint(tr, sq);
      let l2 = (tr - sq) / 2.0;
      // Sort by absolute value descending
      if l1.abs() >= l2.abs() {
        return vec![l1, l2];
      }
      return vec![l2, l1];
    }
    // Complex eigenvalues - return empty
    return vec![];
  }

  // For n >= 3: Householder reduction to Hessenberg form followed by a
  // Wilkinson-shifted QR iteration on the matrix itself. (The previous
  // route — floating-point Faddeev–LeVerrier coefficients + companion
  // matrix — loses half the significant digits at repeated roots, e.g.
  // a triangular {{2,1,0},{0,2,0},{0,0,1}} came out as {1.9995…, 1.0…}.)
  let mut h = matrix.to_vec();
  hessenberg_reduce(&mut h, n);
  let mut roots = qr_eigenvalues(&mut h, n);
  roots.sort_by(|a, b| {
    b.abs()
      .partial_cmp(&a.abs())
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  roots
}

/// Eigenvalues of a real n×n matrix (n ≥ 3) including complex-conjugate
/// pairs, as `(re, im)` sorted by decreasing magnitude with the `+im` member
/// of a conjugate pair first. Empty when the iteration does not converge.
fn numeric_eigenvalues_complex(
  matrix: &[Vec<f64>],
  n: usize,
) -> Vec<(f64, f64)> {
  let mut h = matrix.to_vec();
  hessenberg_reduce(&mut h, n);
  let mut roots = qr_eigenvalues_complex(&mut h, n);
  if roots.len() != n {
    return Vec::new();
  }
  roots.sort_by(|a, b| {
    b.0
      .hypot(b.1)
      .partial_cmp(&a.0.hypot(a.1))
      .unwrap_or(std::cmp::Ordering::Equal)
      .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
  });
  roots
}

/// Eigenvalues of a complex matrix: reduction to Hessenberg form by
/// stabilized elimination, then single-shift QR with Givens rotations and a
/// Wilkinson shift. `None` if the iteration does not converge.
fn complex_matrix_eigenvalues(
  mut h: Vec<Vec<(f64, f64)>>,
) -> Option<Vec<(f64, f64)>> {
  type C = (f64, f64);
  let add = |a: C, b: C| (a.0 + b.0, a.1 + b.1);
  let sub = |a: C, b: C| (a.0 - b.0, a.1 - b.1);
  let mul = |a: C, b: C| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
  let div = |a: C, b: C| {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
  };
  let abs = |a: C| a.0.hypot(a.1);
  let conj = |a: C| (a.0, -a.1);
  let sqrt = |a: C| {
    let r = abs(a);
    let re = f64::midpoint(r, a.0).max(0.0).sqrt();
    let im = ((r - a.0) / 2.0).max(0.0).sqrt();
    (re, if a.1 < 0.0 { -im } else { im })
  };
  let n = h.len();
  // Hessenberg form: eliminate below the subdiagonal, pivoting on the
  // largest entry of each column (a similarity transform throughout).
  for k in 0..n.saturating_sub(2) {
    let pivot =
      (k + 1..n).max_by(|&i, &j| abs(h[i][k]).total_cmp(&abs(h[j][k])))?;
    if abs(h[pivot][k]) == 0.0 {
      continue;
    }
    if pivot != k + 1 {
      h.swap(pivot, k + 1);
      for row in &mut h {
        row.swap(pivot, k + 1);
      }
    }
    for i in k + 2..n {
      let m = div(h[i][k], h[k + 1][k]);
      if m == (0.0, 0.0) {
        continue;
      }
      for j in 0..n {
        let t = mul(m, h[k + 1][j]);
        h[i][j] = sub(h[i][j], t);
      }
      for row in &mut h {
        let t = mul(m, row[i]);
        row[k + 1] = add(row[k + 1], t);
      }
    }
  }
  let mut values = Vec::with_capacity(n);
  let mut hi = n;
  let mut iterations = 0usize;
  while hi > 0 {
    if hi == 1 {
      values.push(h[0][0]);
      break;
    }
    // Deflate at the lowest negligible subdiagonal entry.
    let mut lo = hi - 1;
    while lo > 0 {
      let scale = abs(h[lo][lo]) + abs(h[lo - 1][lo - 1]);
      if abs(h[lo][lo - 1]) <= 1e-15 * scale.max(1e-300) {
        h[lo][lo - 1] = (0.0, 0.0);
        break;
      }
      lo -= 1;
    }
    if lo == hi - 1 {
      values.push(h[hi - 1][hi - 1]);
      hi -= 1;
      iterations = 0;
      continue;
    }
    iterations += 1;
    if iterations > 60 * n {
      return None;
    }
    // Wilkinson shift: the eigenvalue of the trailing 2x2 block nearer its
    // last diagonal entry (an exceptional shift now and then).
    let (a, b, c, d) = (
      h[hi - 2][hi - 2],
      h[hi - 2][hi - 1],
      h[hi - 1][hi - 2],
      h[hi - 1][hi - 1],
    );
    let shift = if iterations % 11 == 10 {
      add(d, (abs(c), 0.0))
    } else {
      let tr_half = mul(add(a, d), (0.5, 0.0));
      let det = sub(mul(a, d), mul(b, c));
      let disc = sqrt(sub(mul(tr_half, tr_half), det));
      let (l1, l2) = (add(tr_half, disc), sub(tr_half, disc));
      if abs(sub(l1, d)) <= abs(sub(l2, d)) {
        l1
      } else {
        l2
      }
    };
    for i in lo..hi {
      h[i][i] = sub(h[i][i], shift);
    }
    // QR of the active block by Givens rotations, then RQ.
    let mut rotations = Vec::with_capacity(hi - lo);
    for k in lo..hi - 1 {
      let (x, y) = (h[k][k], h[k + 1][k]);
      let r = abs(x).hypot(abs(y));
      if r == 0.0 {
        rotations.push(((1.0, 0.0), (0.0, 0.0)));
        continue;
      }
      let (cg, sg) = (div(x, (r, 0.0)), div(y, (r, 0.0)));
      for j in k..n {
        let (u, v) = (h[k][j], h[k + 1][j]);
        h[k][j] = add(mul(conj(cg), u), mul(conj(sg), v));
        h[k + 1][j] = sub(mul(cg, v), mul(sg, u));
      }
      rotations.push((cg, sg));
    }
    for (offset, &(cg, sg)) in rotations.iter().enumerate() {
      let k = lo + offset;
      for row in h.iter_mut().take((k + 2).min(hi)) {
        let (u, v) = (row[k], row[k + 1]);
        row[k] = add(mul(u, cg), mul(v, sg));
        row[k + 1] = sub(mul(v, conj(cg)), mul(u, conj(sg)));
      }
    }
    for i in lo..hi {
      h[i][i] = add(h[i][i], shift);
    }
  }
  values
    .iter()
    .all(|v| v.0.is_finite() && v.1.is_finite())
    .then_some(values)
}

/// In-place Householder reduction to upper Hessenberg form. Columns whose
/// below-subdiagonal part is exactly zero are skipped, so triangular and
/// block-triangular inputs pass through bit-for-bit unchanged.
fn hessenberg_reduce(a: &mut [Vec<f64>], n: usize) {
  for k in 0..n.saturating_sub(2) {
    let norm_x: f64 = (k + 1..n).map(|i| a[i][k] * a[i][k]).sum::<f64>().sqrt();
    if norm_x == 0.0 || (k + 2..n).all(|i| a[i][k] == 0.0) {
      continue;
    }
    let alpha = if a[k + 1][k] >= 0.0 { -norm_x } else { norm_x };
    let m = n - k - 1;
    let mut v: Vec<f64> = (k + 1..n).map(|i| a[i][k]).collect();
    v[0] -= alpha;
    let vnorm2: f64 = v.iter().map(|x| x * x).sum();
    if vnorm2 == 0.0 {
      continue;
    }
    // A ← H A with H = I − 2vvᵀ/‖v‖²
    for j in 0..n {
      let dot: f64 = (0..m).map(|t| v[t] * a[k + 1 + t][j]).sum();
      let f = 2.0 * dot / vnorm2;
      for t in 0..m {
        a[k + 1 + t][j] -= f * v[t];
      }
    }
    // A ← A H
    for row in a.iter_mut().take(n) {
      let dot: f64 = (0..m).map(|t| row[k + 1 + t] * v[t]).sum();
      let f = 2.0 * dot / vnorm2;
      for t in 0..m {
        row[k + 1 + t] -= f * v[t];
      }
    }
    a[k + 1][k] = alpha;
    for i in k + 2..n {
      a[i][k] = 0.0;
    }
  }
}

/// Real eigenvalues of an upper Hessenberg matrix via shifted QR
/// iteration with deflation. Trailing 1×1 and 2×2 blocks deflate in
/// closed form, so clean cases (triangular, block-diagonal, defective
/// with exact entries) produce exact values. Returns an empty vector when
/// a complex pair appears or the iteration fails to converge — the
/// callers then leave the expression unevaluated instead of fabricating
/// real "eigenvalues" (the old companion-matrix path returned garbage for
/// rotation-like matrices).
fn qr_eigenvalues(h: &mut [Vec<f64>], n: usize) -> Vec<f64> {
  let all = qr_eigenvalues_complex(h, n);
  if all.len() != n || all.iter().any(|&(_, im)| im != 0.0) {
    return Vec::new();
  }
  all.into_iter().map(|(re, _)| re).collect()
}

/// Like `qr_eigenvalues`, but complex-conjugate pairs are returned as
/// `(re, im)` entries (`im > 0` first) instead of aborting. Empty on
/// non-convergence.
fn qr_eigenvalues_complex(h: &mut [Vec<f64>], n: usize) -> Vec<(f64, f64)> {
  // Francis double-shift QR on the upper Hessenberg matrix (EISPACK's
  // `hqr`): a complex-conjugate pair converges as a real 2×2 block, so a
  // random real matrix takes a few sweeps per eigenvalue instead of the
  // hundreds a real single shift needs. Indices below are 1-based.
  let mut a = vec![vec![0.0f64; n + 1]; n + 1];
  for i in 0..n {
    for j in 0..n {
      a[i + 1][j + 1] = h[i][j];
    }
  }
  let sign = |a: f64, b: f64| if b >= 0.0 { a.abs() } else { -a.abs() };
  let mut wr = vec![0.0f64; n + 1];
  let mut wi = vec![0.0f64; n + 1];
  let mut anorm = 0.0;
  for i in 1..=n {
    for j in i.saturating_sub(1).max(1)..=n {
      anorm += a[i][j].abs();
    }
  }
  let mut nn = n;
  let mut t = 0.0;
  while nn >= 1 {
    let mut its = 0;
    loop {
      let mut l = nn;
      while l >= 2 {
        let mut s = a[l - 1][l - 1].abs() + a[l][l].abs();
        if s == 0.0 {
          s = anorm;
        }
        if a[l][l - 1].abs() + s == s {
          a[l][l - 1] = 0.0;
          break;
        }
        l -= 1;
      }
      let mut x = a[nn][nn];
      if l == nn {
        wr[nn] = x + t;
        wi[nn] = 0.0;
        nn -= 1;
        break;
      }
      let mut y = a[nn - 1][nn - 1];
      let mut w = a[nn][nn - 1] * a[nn - 1][nn];
      if l == nn - 1 {
        let p = 0.5 * (y - x);
        let q = p * p + w;
        let mut z = q.abs().sqrt();
        x += t;
        if q >= 0.0 {
          z = p + sign(z, p);
          wr[nn - 1] = x + z;
          wr[nn] = x + z;
          if z != 0.0 {
            wr[nn] = x - w / z;
          }
          wi[nn - 1] = 0.0;
          wi[nn] = 0.0;
        } else {
          wr[nn - 1] = x + p;
          wr[nn] = x + p;
          wi[nn - 1] = z;
          wi[nn] = -z;
        }
        nn -= 2;
        break;
      }
      if its == 60 {
        return Vec::new();
      }
      if its == 10 || its == 20 {
        // Exceptional shift.
        t += x;
        for i in 1..=nn {
          a[i][i] -= x;
        }
        let s = a[nn][nn - 1].abs() + a[nn - 1][nn - 2].abs();
        x = 0.75 * s;
        y = x;
        w = -0.4375 * s * s;
      }
      its += 1;
      let (mut p, mut q, mut r, mut z);
      let mut m = nn - 2;
      loop {
        z = a[m][m];
        r = x - z;
        let s0 = y - z;
        p = (r * s0 - w) / a[m + 1][m] + a[m][m + 1];
        q = a[m + 1][m + 1] - z - r - s0;
        r = a[m + 2][m + 1];
        let s = p.abs() + q.abs() + r.abs();
        p /= s;
        q /= s;
        r /= s;
        if m == l {
          break;
        }
        let u = a[m][m - 1].abs() * (q.abs() + r.abs());
        let v =
          p.abs() * (a[m - 1][m - 1].abs() + z.abs() + a[m + 1][m + 1].abs());
        if u + v == v {
          break;
        }
        m -= 1;
      }
      for i in m + 2..=nn {
        a[i][i - 2] = 0.0;
        if i != m + 2 {
          a[i][i - 3] = 0.0;
        }
      }
      let mut k = m;
      while k < nn {
        if k != m {
          p = a[k][k - 1];
          q = a[k + 1][k - 1];
          r = 0.0;
          if k != nn - 1 {
            r = a[k + 2][k - 1];
          }
          x = p.abs() + q.abs() + r.abs();
          if x != 0.0 {
            p /= x;
            q /= x;
            r /= x;
          }
        }
        let s = sign((p * p + q * q + r * r).sqrt(), p);
        if s != 0.0 {
          if k == m {
            if l != m {
              a[k][k - 1] = -a[k][k - 1];
            }
          } else {
            a[k][k - 1] = -s * x;
          }
          p += s;
          x = p / s;
          y = q / s;
          z = r / s;
          q /= p;
          r /= p;
          for j in k..=nn {
            let mut pj = a[k][j] + q * a[k + 1][j];
            if k != nn - 1 {
              pj += r * a[k + 2][j];
              a[k + 2][j] -= pj * z;
            }
            a[k + 1][j] -= pj * y;
            a[k][j] -= pj * x;
          }
          let mmin = if nn < k + 3 { nn } else { k + 3 };
          for i in l..=mmin {
            let mut pi = x * a[i][k] + y * a[i][k + 1];
            if k != nn - 1 {
              pi += z * a[i][k + 2];
              a[i][k + 2] -= pi * r;
            }
            a[i][k + 1] -= pi * q;
            a[i][k] -= pi;
          }
        }
        k += 1;
      }
    }
  }
  (1..=n).map(|i| (wr[i], wi[i])).collect()
}

/// Refine an eigenvalue's eigenvector via (shifted) inverse iteration:
/// repeatedly solving `(A - λI) y = x` for `y` amplifies whatever component
/// of `x` lies along `A`'s (near-)singular direction at shift `λ`, since
/// that direction's effective eigenvalue of `(A - λI)` is close to zero. A
/// tiny extra shift keeps the system from being *exactly* singular (which
/// would make its own pivoting degenerate) while staying far smaller than
/// the gap to any other eigenvalue, so it still converges to `λ`'s own
/// eigenvector in a couple of iterations — including when `λ` sits close to
/// another eigenvalue, which is exactly the case `numeric_null_space_basis`'s
/// fixed pivot tolerance can miss for larger or near-degenerate matrices.
fn inverse_iteration_eigenvector(
  a: &[Vec<f64>],
  lambda: f64,
  n: usize,
) -> Option<Vec<f64>> {
  let scale = a
    .iter()
    .flat_map(|row| row.iter())
    .fold(1.0f64, |acc, v| acc.max(v.abs()));
  let mut shifted = a.to_vec();
  for i in 0..n {
    shifted[i][i] -= lambda + scale * 1e-10;
  }
  let mut x = vec![1.0; n];
  let mut result = None;
  for _ in 0..3 {
    let y = solve_linear_system(&shifted, &x)?;
    let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-300 {
      return result;
    }
    x = y.iter().map(|v| v / norm).collect();
    result = Some(x.clone());
  }
  result
}

/// Every basis vector of a numeric matrix's null space, one per free
/// column, via Gaussian elimination with partial pivoting. A repeated
/// eigenvalue's shifted matrix `(A - λI)` has a null space of dimension
/// equal to the eigenvalue's multiplicity; returning every free-column
/// vector (instead of only the first) lets the caller hand each occurrence
/// of the repeated eigenvalue a genuinely independent eigenvector instead
/// of the same one every time.
fn numeric_null_space_basis(m: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
  let mut aug: Vec<Vec<f64>> = m.to_vec();
  let mut pivot_cols = Vec::new();
  let mut pivot_row = 0;

  for col in 0..n {
    if pivot_row >= n {
      break;
    }
    // Partial pivoting
    let mut max_row = pivot_row;
    let mut max_val = aug[pivot_row][col].abs();
    for row in pivot_row + 1..n {
      if aug[row][col].abs() > max_val {
        max_row = row;
        max_val = aug[row][col].abs();
      }
    }
    if max_val < 1e-12 {
      continue;
    }
    aug.swap(pivot_row, max_row);
    pivot_cols.push((pivot_row, col));

    let pivot = aug[pivot_row][col];
    for row in 0..n {
      if row == pivot_row {
        continue;
      }
      let factor = aug[row][col] / pivot;
      for j in col..n {
        aug[row][j] -= factor * aug[pivot_row][j];
      }
      aug[row][col] = 0.0;
    }
    // Normalize pivot row
    for j in col..n {
      aug[pivot_row][j] /= pivot;
    }
    pivot_row += 1;
  }

  // Find free columns
  let pivot_col_set: Vec<usize> = pivot_cols.iter().map(|&(_, c)| c).collect();
  let free_cols: Vec<usize> =
    (0..n).filter(|c| !pivot_col_set.contains(c)).collect();

  free_cols
    .into_iter()
    .map(|free_col| {
      let mut vec = vec![0.0; n];
      vec[free_col] = 1.0;
      for &(row, col) in &pivot_cols {
        vec[col] = -aug[row][free_col];
      }
      vec
    })
    .collect()
}

/// RowReduce[matrix] - Gaussian elimination to reduced row echelon form
pub fn row_reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RowReduce expects exactly 1 argument".into(),
    ));
  }
  let Some(matrix) = expr_to_matrix(&args[0]) else {
    return Ok(unevaluated("RowReduce", args));
  };
  Ok(matrix_to_expr(row_reduce_impl(&matrix)))
}

/// RankDecomposition[m] factors an m×n matrix of rank r into {C, F} where C is
/// m×r (the pivot columns of `m`) and F is r×n (the nonzero rows of the reduced
/// row echelon form), so that C.F == m. A rank-0 (all-zero) matrix has no rank
/// decomposition; a non-matrix argument is a usage error. Both cases emit the
/// matching wolframscript message and stay unevaluated.
pub fn rank_decomposition_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("RankDecomposition", args);
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) if !m.is_empty() && !m[0].is_empty() => m,
    _ => {
      crate::emit_message(&format!(
        "RankDecomposition::matrix: Argument {} at position 1 is not a \
         nonempty rectangular matrix.",
        expr_to_string(&args[0])
      ));
      return Ok(unevaluated());
    }
  };

  let rref = row_reduce_impl(&matrix);
  let is_zero = |e: &Expr| {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(f) if *f == 0.0)
  };
  // Each nonzero RREF row's leading entry marks a pivot column; those columns
  // of the original matrix form C, and the nonzero RREF rows form F.
  let mut pivot_cols: Vec<usize> = Vec::new();
  let mut f_rows: Vec<Vec<Expr>> = Vec::new();
  for row in &rref {
    if let Some(pc) = row.iter().position(|e| !is_zero(e)) {
      pivot_cols.push(pc);
      f_rows.push(row.clone());
    }
  }
  if pivot_cols.is_empty() {
    crate::emit_message(
      "RankDecomposition::rnkz: A rank decomposition does not exist for a \
       rank 0 matrix.",
    );
    return Ok(unevaluated());
  }

  let c: Vec<Vec<Expr>> = matrix
    .iter()
    .map(|row| pivot_cols.iter().map(|&j| row[j].clone()).collect())
    .collect();
  Ok(Expr::List(
    vec![matrix_to_expr(c), matrix_to_expr(f_rows)].into(),
  ))
}

/// DrazinInverse[m] - the Drazin inverse A^D of a square matrix, the unique
/// matrix with A^D A A^D = A^D, A A^D = A^D A, and A^{k+1} A^D = A^k where k is
/// the matrix index. Computed via the core–nilpotent split: with k the index
/// and W = m^k, the columns of m at the pivot positions of W span the core
/// (invertible) subspace and NullSpace[W] spans the nilpotent subspace; in that
/// basis m is block-diagonal diag(C, N) and A^D = P diag(C^{-1}, 0) P^{-1}. An
/// invertible matrix reduces to Inverse[m] and a nilpotent one to the zero
/// matrix. A non-square argument emits the wolframscript `matsq` message.
pub fn drazin_inverse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("DrazinInverse", args);
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) if !m.is_empty() && m.len() == m[0].len() => m,
    _ => {
      crate::emit_message(&format!(
        "DrazinInverse::matsq: Argument {} at position 1 is not a nonempty \
         square matrix.",
        expr_to_string(&args[0])
      ));
      return Ok(unevaluated());
    }
  };
  let n = matrix.len();
  let a_expr = matrix_to_expr(matrix.clone());

  let dot = |x: &Expr, y: &Expr| {
    evaluate_expr_to_expr(&call("Dot", vec![x.clone(), y.clone()]))
  };
  let unary =
    |name: &str, m: &Expr| evaluate_expr_to_expr(&call1(name, m.clone()));
  let rank = |m: &Expr| -> Option<usize> {
    match unary("MatrixRank", m) {
      Ok(Expr::Integer(r)) if r >= 0 => Some(r as usize),
      _ => None,
    }
  };

  // Matrix index k (≤ n) and W = mᵏ: smallest k with rank(mᵏ⁺¹) == rank(mᵏ).
  let identity = identity_matrix_ast(&[Expr::Integer(n as i128)])?;
  let mut powers = vec![identity]; // m⁰ = I
  let mut ranks = vec![n]; // rank(m⁰) = n
  let mut k = 0usize;
  loop {
    let next = dot(&powers[k], &a_expr)?;
    let Some(rnext) = rank(&next) else {
      return Ok(unevaluated());
    };
    powers.push(next);
    ranks.push(rnext);
    if rnext == ranks[k] {
      break;
    }
    k += 1;
    if k >= n {
      break;
    }
  }
  let w = powers[k].clone();
  let r = ranks[k];

  // Invertible → ordinary inverse; nilpotent → zero matrix.
  if r == n {
    return unary("Inverse", &a_expr);
  }
  if r == 0 {
    return Ok(matrix_to_expr(vec![vec![Expr::Integer(0); n]; n]));
  }

  // Core basis: the columns of W at the pivot positions of RowReduce[W].
  let Some(w_mat) = expr_to_matrix(&w) else {
    return Ok(unevaluated());
  };
  let rref = row_reduce_impl(&w_mat);
  let mut pivot_cols: Vec<usize> = Vec::new();
  for row in &rref {
    if let Some(pc) = row.iter().position(|e| !is_zero_expr(e)) {
      pivot_cols.push(pc);
    }
  }
  // Nilpotent basis: NullSpace[W].
  let null_expr = unary("NullSpace", &w)?;
  let Expr::List(null_vecs) = &null_expr else {
    return Ok(unevaluated());
  };
  if pivot_cols.len() + null_vecs.len() != n {
    return Ok(unevaluated());
  }

  // P = [core columns | nilpotent columns] (n × n).
  let mut p: Vec<Vec<Expr>> = Vec::with_capacity(n);
  for i in 0..n {
    let mut row = Vec::with_capacity(n);
    for &j in &pivot_cols {
      row.push(w_mat[i][j].clone());
    }
    for nv in null_vecs {
      let Expr::List(comp) = nv else {
        return Ok(unevaluated());
      };
      row.push(comp[i].clone());
    }
    p.push(row);
  }
  let p_expr = matrix_to_expr(p);
  let pinv = unary("Inverse", &p_expr)?;

  // B = P⁻¹ m P is block-diagonal diag(C, N); invert the r×r core block C.
  let b = dot(&dot(&pinv, &a_expr)?, &p_expr)?;
  let Some(b_mat) = expr_to_matrix(&b) else {
    return Ok(unevaluated());
  };
  let c: Vec<Vec<Expr>> = (0..r)
    .map(|i| (0..r).map(|j| b_mat[i][j].clone()).collect())
    .collect();
  let cinv = unary("Inverse", &matrix_to_expr(c))?;
  let Some(cinv_mat) = expr_to_matrix(&cinv) else {
    return Ok(unevaluated());
  };

  // D = block-diag(C⁻¹, 0); A^D = P D P⁻¹, entrywise simplified.
  let mut dblock: Vec<Vec<Expr>> = vec![vec![Expr::Integer(0); n]; n];
  for (i, crow) in cinv_mat.iter().enumerate() {
    for (j, val) in crow.iter().enumerate() {
      dblock[i][j] = val.clone();
    }
  }
  let ad = dot(&dot(&p_expr, &matrix_to_expr(dblock))?, &pinv)?;
  unary("Simplify", &ad)
}

/// Simplify expression via the evaluator (with Simplify for symbolic)
fn contains_root_object(e: &Expr) -> bool {
  match e {
    Expr::FunctionCall { name, args } => {
      name == "Root" || args.iter().any(contains_root_object)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_root_object(left) || contains_root_object(right)
    }
    Expr::UnaryOp { operand, .. } => contains_root_object(operand),
    _ => false,
  }
}

fn simplify_expr(e: &Expr) -> Expr {
  let Ok(evaluated) = evaluate_expr_to_expr(e) else {
    return e.clone();
  };
  // For symbolic expressions, also try Simplify
  match &evaluated {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::BigInteger(_)
    | Expr::Identifier(_) => evaluated,
    // An algebraic number written with `Root` objects (an eigenvalue of a
    // matrix with no radical roots) is zero when it vanishes to 50 digits
    // (a nonzero one this small would need an enormous minimal polynomial),
    // and otherwise canonicalized by RootReduce.
    _ if contains_root_object(&evaluated)
      && crate::functions::predicate_ast::is_numeric_q(&evaluated) =>
    {
      let vanishes = evaluate_expr_to_expr(&call(
        "Less",
        vec![
          call1("Abs", call("N", vec![evaluated.clone(), Expr::Integer(50)])),
          pow2(Expr::Integer(10), Expr::Integer(-40)),
        ],
      ));
      if matches!(vanishes, Ok(Expr::Identifier(ref t)) if t == "True") {
        return Expr::Integer(0);
      }
      evaluate_expr_to_expr(&call1("RootReduce", evaluated.clone()))
        .unwrap_or(evaluated)
    }
    _ => match evaluate_expr_to_expr(&call1("Simplify", evaluated.clone())) {
      Ok(r) => r,
      Err(_) => evaluated,
    },
  }
}

/// Internal row reduction (Gauss-Jordan elimination) that works symbolically.
/// Uses fraction-free approach: instead of dividing by pivot immediately,
/// we compute row[j] = row[j]*pivot - factor*pivot_row[j] to avoid symbolic division issues,
/// then normalize at the end.
fn row_reduce_impl(matrix: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
  let nrows = matrix.len();
  if nrows == 0 {
    return vec![];
  }
  let ncols = matrix[0].len();
  let mut m: Vec<Vec<Expr>> = matrix.to_vec();

  let mut pivot_row = 0;
  let mut pivot_cols = Vec::new();

  for col in 0..ncols {
    if pivot_row >= nrows {
      break;
    }
    // Find a non-zero pivot in this column
    let mut found = None;
    for row in pivot_row..nrows {
      if !is_zero_expr(&m[row][col]) {
        found = Some(row);
        break;
      }
    }
    let Some(swap_row) = found else { continue };
    if swap_row != pivot_row {
      m.swap(pivot_row, swap_row);
    }

    let pivot = m[pivot_row][col].clone();
    pivot_cols.push((pivot_row, col));

    // Eliminate all other rows using fraction-free approach
    for row in 0..nrows {
      if row == pivot_row {
        continue;
      }
      let factor = m[row][col].clone();
      if !is_zero_expr(&factor) {
        // new_row[j] = pivot * row[j] - factor * pivot_row[j]
        for j in 0..ncols {
          let term1 = eval_mul(&pivot, &m[row][j]);
          let term2 = eval_mul(&factor, &m[pivot_row][j]);
          let result = eval_sub(&term1, &term2);
          m[row][j] = simplify_expr(&result);
        }
      }
    }
    pivot_row += 1;
  }

  // Now normalize: scale each pivot row so pivot element is 1
  for &(row, col) in &pivot_cols {
    let pivot = m[row][col].clone();
    if !is_one_expr(&pivot) && !is_zero_expr(&pivot) {
      for j in 0..ncols {
        if !is_zero_expr(&m[row][j]) {
          m[row][j] = simplify_expr(&eval_divide(&m[row][j], &pivot));
        }
      }
    }
  }

  m
}

/// Check if an expression is zero
fn is_zero_expr(e: &Expr) -> bool {
  matches!(e, Expr::Integer(0))
    || matches!(e, Expr::Real(x) if *x == 0.0)
    || matches!(e, Expr::BigInteger(n) if n == &BigInt::from(0))
}

/// Check if an expression is one
fn is_one_expr(e: &Expr) -> bool {
  matches!(e, Expr::Integer(1))
    || matches!(e, Expr::Real(x) if *x == 1.0)
    || matches!(e, Expr::BigInteger(n) if n == &BigInt::from(1))
}

/// MatrixRank[matrix] - rank of a matrix
pub fn matrix_rank_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "MatrixRank expects exactly 1 argument".into(),
    ));
  }
  let Some(matrix) = expr_to_matrix(&args[0]) else {
    return Ok(unevaluated("MatrixRank", args));
  };
  let rref = row_reduce_impl(&matrix);
  // Count non-zero rows
  let rank = rref
    .iter()
    .filter(|row| row.iter().any(|e| !is_zero_expr(e)))
    .count();
  Ok(Expr::Integer(rank as i128))
}

/// NullSpace[matrix] - find null space basis vectors
pub fn null_space_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NullSpace expects exactly 1 argument".into(),
    ));
  }
  let Some(matrix) = expr_to_matrix(&args[0]) else {
    // Return empty list for non-matrix (symbolic matrix)
    return Ok(Expr::List(vec![].into()));
  };
  let nrows = matrix.len();
  if nrows == 0 {
    return Ok(Expr::List(vec![].into()));
  }
  let ncols = matrix[0].len();

  let rref = row_reduce_impl(&matrix);

  // Find pivot columns
  let mut pivot_cols = Vec::new();
  let mut pivot_row_for_col = vec![None; ncols];
  let mut current_row = 0;
  for col in 0..ncols {
    if current_row >= nrows {
      break;
    }
    if is_one_expr(&rref[current_row][col]) {
      pivot_cols.push(col);
      pivot_row_for_col[col] = Some(current_row);
      current_row += 1;
    }
  }

  // Free columns are non-pivot columns
  let free_cols: Vec<usize> =
    (0..ncols).filter(|c| !pivot_cols.contains(c)).collect();

  // Build null space basis vectors
  let mut basis = Vec::new();
  for &free_col in &free_cols {
    let mut vec = vec![Expr::Integer(0); ncols];
    vec[free_col] = Expr::Integer(1);
    for &pivot_col in &pivot_cols {
      if let Some(row) = pivot_row_for_col[pivot_col] {
        // Negate the entry from rref
        vec[pivot_col] = eval_sub(&Expr::Integer(0), &rref[row][free_col]);
      }
    }
    basis.push(Expr::List(vec.into()));
  }
  // Wolfram emits the basis in order of decreasing free-variable index, i.e.
  // the reverse of the ascending free-column order built above.
  basis.reverse();
  Ok(Expr::List(basis.into()))
}

/// Compute the sign of a permutation (Levi-Civita symbol).
/// Returns 1 for even permutations, -1 for odd, 0 if any indices repeat.
fn permutation_sign(perm: &[usize]) -> i128 {
  let n = perm.len();
  // Check for duplicates
  let mut seen = vec![false; n];
  for &p in perm {
    if p >= n || seen[p] {
      return 0;
    }
    seen[p] = true;
  }
  // Count inversions
  let mut inversions = 0;
  for i in 0..n {
    for j in i + 1..n {
      if perm[i] > perm[j] {
        inversions += 1;
      }
    }
  }
  if inversions % 2 == 0 { 1 } else { -1 }
}

/// LeviCivitaTensor[n] - produces a SparseArray containing only the
/// non-zero (permutation) entries of the d-dimensional Levi-Civita symbol.
pub fn levi_civita_tensor_sparse_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 1 => n as usize,
    _ => {
      return Ok(unevaluated("LeviCivitaTensor", args));
    }
  };

  // Generate all permutations of {1..n} in lexicographic order and emit
  // a rule for each with the permutation sign as the value.
  let mut perm: Vec<usize> = (1..=n).collect();
  let mut rules: Vec<Expr> = Vec::new();
  loop {
    // Convert perm to 0-based for sign calculation.
    let zero_based: Vec<usize> = perm.iter().map(|&x| x - 1).collect();
    let sign = permutation_sign(&zero_based);
    let key = Expr::List(
      perm
        .iter()
        .map(|&i| Expr::Integer(i as i128))
        .collect::<Vec<_>>()
        .into(),
    );
    rules.push(rule_expr(key, Expr::Integer(sign)));
    // Next permutation (Knuth's algorithm L).
    if n < 2 {
      break;
    }
    let mut i = n - 1;
    while i > 0 && perm[i - 1] >= perm[i] {
      i -= 1;
    }
    if i == 0 {
      break;
    }
    let mut j = n - 1;
    while perm[j] <= perm[i - 1] {
      j -= 1;
    }
    perm.swap(i - 1, j);
    perm[i..].reverse();
  }

  // Delegate to SparseArray[rules, {n, n, ..., n}] to get the same internal
  // representation that wolframscript emits.
  let dims = Expr::List(vec![Expr::Integer(n as i128); n].into());
  let sparse_call = call("SparseArray", vec![Expr::List(rules.into()), dims]);
  crate::evaluator::evaluate_expr_to_expr(&sparse_call)
}

/// LeviCivitaTensor[n, List] - produces an n-dimensional tensor with the Levi-Civita symbol values.
pub fn levi_civita_tensor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 1 => n as usize,
    _ => {
      return Ok(unevaluated("LeviCivitaTensor", args));
    }
  };

  // Build the n-dimensional tensor recursively
  fn build_tensor(n: usize, depth: usize, indices: &mut Vec<usize>) -> Expr {
    if depth == n {
      let sign = permutation_sign(indices);
      Expr::Integer(sign)
    } else {
      let mut items = Vec::new();
      for i in 0..n {
        indices.push(i);
        items.push(build_tensor(n, depth + 1, indices));
        indices.pop();
      }
      Expr::List(items.into())
    }
  }

  let mut indices = Vec::new();
  Ok(build_tensor(n, 0, &mut indices))
}

/// Solve a rectangular system whose entries are real numbers (at least one
/// inexact) by Gaussian elimination in machine arithmetic. Pivots and the
/// consistency of the surplus equations are judged against a tolerance
/// relative to the largest entry, so rounding noise in an over-determined
/// but consistent system does not make it "inconsistent". Free variables are
/// set to 0. `None` unless the data is purely real machine numbers; the inner
/// `None` means the system has no solution.
fn linear_solve_rectangular_machine(
  matrix: &[Vec<Expr>],
  b: &[Expr],
  ncols: usize,
) -> Option<Option<Vec<f64>>> {
  use crate::functions::math_ast::numeric_utils::try_extract_complex_f64;
  let mut inexact = false;
  let mut m: Vec<Vec<f64>> = Vec::with_capacity(matrix.len());
  for (row, bi) in matrix.iter().zip(b) {
    let mut r = Vec::with_capacity(ncols + 1);
    for e in row.iter().chain(std::iter::once(bi)) {
      inexact |= matches!(e, Expr::Real(_));
      let (re, im) = match e {
        Expr::Real(x) => (*x, 0.0),
        Expr::Integer(i) => (*i as f64, 0.0),
        other => try_extract_complex_f64(other)?,
      };
      if im != 0.0 || !re.is_finite() {
        return None;
      }
      r.push(re);
    }
    m.push(r);
  }
  if !inexact {
    return None;
  }
  let scale = m.iter().flatten().fold(0.0_f64, |acc, v| acc.max(v.abs()));
  let tol = scale * 1e-10;
  let nrows = m.len();
  let mut pivots: Vec<(usize, usize)> = Vec::new();
  let mut pivot_row = 0;
  for col in 0..ncols {
    if pivot_row >= nrows {
      break;
    }
    let best = (pivot_row..nrows)
      .max_by(|&r1, &r2| m[r1][col].abs().total_cmp(&m[r2][col].abs()))?;
    if m[best][col].abs() <= tol {
      continue;
    }
    m.swap(pivot_row, best);
    let pivot = m[pivot_row][col];
    for row in 0..nrows {
      if row == pivot_row {
        continue;
      }
      let factor = m[row][col] / pivot;
      if factor != 0.0 {
        for j in col..=ncols {
          let sub = factor * m[pivot_row][j];
          m[row][j] -= sub;
        }
      }
    }
    pivots.push((pivot_row, col));
    pivot_row += 1;
  }
  if m[pivot_row..].iter().any(|r| r[ncols].abs() > tol) {
    return Some(None);
  }
  let mut x = vec![0.0; ncols];
  for (row, col) in pivots {
    x[col] = m[row][ncols] / m[row][col];
  }
  Some(Some(x))
}

/// LinearSolve[m, b] — solves the matrix equation m.x = b for x.
/// Uses Gaussian elimination with exact rational arithmetic.
/// Solve a rectangular (non-square) linear system `A x = b` via the reduced
/// row echelon form of `[A | b]`, returning the particular solution with all
/// non-pivot (free) variables set to 0 — matching wolframscript. Emits
/// LinearSolve::nosol and stays unevaluated when the system is inconsistent.
fn linear_solve_rectangular(
  matrix: &[Vec<Expr>],
  b: &[Expr],
  ncols: usize,
  args: &[Expr],
) -> Expr {
  // Augment [A | b] and reduce.
  let aug: Vec<Vec<Expr>> = matrix
    .iter()
    .zip(b.iter())
    .map(|(row, bi)| {
      let mut r = row.clone();
      r.push(bi.clone());
      r
    })
    .collect();
  let rref = row_reduce_impl(&aug);

  let mut solution = vec![Expr::Integer(0); ncols];
  for row in &rref {
    // The pivot is the first non-zero column of the row.
    let lead = (0..=ncols).find(|&j| !is_zero_expr(&row[j]));
    match lead {
      Some(j) if j < ncols => {
        // RREF pivot coefficient is 1, so x_j = (augmented RHS) with the free
        // variables (which are 0) dropped.
        solution[j] = row[ncols].clone();
      }
      Some(_) => {
        // Leading entry is in the RHS column: 0 = nonzero → no solution.
        crate::emit_message(
          "LinearSolve::nosol: Linear equation encountered that has no solution.",
        );
        return unevaluated("LinearSolve", args);
      }
      None => {} // all-zero row
    }
  }
  Expr::List(solution.into())
}

/// Whether `LinearSolve`'s `Method` option (the first one among `opts`) can
/// be used on `matrix`. Woxi solves by elimination whatever the method, but
/// wolframscript refuses — leaving `LinearSolve[…]` unevaluated — when the
/// method is unknown (`rmeth`), when `"Cholesky"` gets a matrix that is not
/// Hermitian (`herm`) or not positive definite (`npdef`), and when
/// `"Banded"` gets entries that are not machine numbers (`bdnmt`). With
/// `emit`, the corresponding message is issued.
pub fn linear_solve_method_ok(
  matrix: &Expr,
  opts: &[Expr],
  emit: bool,
) -> bool {
  use crate::evaluator::evaluate_expr_to_expr as eval;
  let method = opts.iter().find_map(|o| match o {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Method") => {
      Some(replacement.as_ref())
    }
    _ => None,
  });
  let Some(method) = method else {
    return true;
  };
  let name = match method {
    Expr::List(items) => items.first(),
    other => Some(other),
  };
  let fail = |msg: String| {
    if emit {
      crate::emit_message(&msg);
    }
    false
  };
  const METHODS: [&str; 9] = [
    "Cholesky",
    "Multifrontal",
    "Krylov",
    "CofactorExpansion",
    "OneStepRowReduction",
    "DivisionFreeRowReduction",
    "Direct",
    "IterativeRefinement",
    "Banded",
  ];
  let name = match name {
    Some(Expr::Identifier(a)) if a == "Automatic" => return true,
    Some(Expr::String(s)) if METHODS.contains(&s.as_str()) => s.as_str(),
    _ => {
      let listed = METHODS
        .iter()
        .map(|m| format!("\"{m}\""))
        .collect::<Vec<_>>()
        .join(", ");
      let (listed, last) = listed.rsplit_once(", ").unwrap();
      return fail(format!(
        "LinearSolve::rmeth: The value of the option Method -> {} should be {listed}, {last} or Automatic.",
        crate::syntax::expr_to_output(method)
      ));
    }
  };
  let Some(rows) = expr_to_matrix(matrix) else {
    return true;
  };
  let shown = || crate::syntax::expr_to_output(matrix);
  match name {
    // A matrix of machine reals: symmetric, and positive definite exactly
    // when its Cholesky factorization runs to the end with positive pivots
    // (one O(n^3) pass instead of n determinants).
    "Cholesky"
      if rows
        .iter()
        .all(|r| r.iter().all(|e| matches!(e, Expr::Real(_)))) =>
    {
      let n = rows.len();
      let a: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
          r.iter()
            .map(|e| match e {
              Expr::Real(v) => *v,
              _ => f64::NAN,
            })
            .collect()
        })
        .collect();
      if a.iter().any(|r| r.len() != n)
        || (0..n).any(|i| (0..i).any(|j| a[i][j] != a[j][i]))
      {
        return fail(format!(
          "LinearSolve::herm: The matrix {} is not Hermitian or real and symmetric.",
          shown()
        ));
      }
      let mut l = vec![vec![0.0f64; n]; n];
      for j in 0..n {
        let d = a[j][j] - (0..j).map(|k| l[j][k] * l[j][k]).sum::<f64>();
        if d.is_nan() || d <= 0.0 {
          return fail(format!(
            "LinearSolve::npdef: The matrix {} is not positive definite.",
            shown()
          ));
        }
        l[j][j] = d.sqrt();
        for i in j + 1..n {
          let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
          l[i][j] = (a[i][j] - s) / l[j][j];
        }
      }
      true
    }
    "Cholesky" => {
      let n = rows.len();
      let hermitian = rows.iter().all(|r| r.len() == n)
        && (0..n).all(|i| {
          (0..n).all(|j| {
            let conj = eval(&call1("Conjugate", rows[j][i].clone()));
            conj.is_ok_and(|c| {
              crate::functions::predicate_ast::is_numeric_q(&rows[i][j])
                && expr_to_string(&c) == expr_to_string(&rows[i][j])
            })
          })
        });
      if !hermitian {
        return fail(format!(
          "LinearSolve::herm: The matrix {} is not Hermitian or real and symmetric.",
          shown()
        ));
      }
      // Sylvester's criterion: every leading principal minor is positive.
      let positive_definite = (1..=n).all(|k| {
        let sub = Expr::List(
          rows[..k]
            .iter()
            .map(|r| Expr::List(r[..k].to_vec().into()))
            .collect::<Vec<_>>()
            .into(),
        );
        let test = call("Positive", vec![call("Re", vec![call1("Det", sub)])]);
        matches!(eval(&test), Ok(Expr::Identifier(ref t)) if t == "True")
      });
      if !positive_definite {
        return fail(format!(
          "LinearSolve::npdef: The matrix {} is not positive definite.",
          shown()
        ));
      }
      true
    }
    "Banded" => {
      // A machine complex number keeps machine-real parts (`4. + 1.*I`).
      fn has_real(e: &Expr) -> bool {
        match e {
          Expr::Real(_) => true,
          Expr::BinaryOp { left, right, .. } => {
            has_real(left) || has_real(right)
          }
          Expr::UnaryOp { operand, .. } => has_real(operand),
          Expr::FunctionCall { args, .. } => args.iter().any(has_real),
          _ => false,
        }
      }
      let machine = |e: &Expr| {
        matches!(e, Expr::Real(_))
          || (crate::functions::predicate_ast::is_complex_number(e)
            && has_real(e))
      };
      if !rows.iter().flatten().all(machine) {
        return fail(
          "LinearSolve::bdnmt: The method \"Banded\" accepts only matrices with elements that are machine-real or machine-complex numbers.".to_string(),
        );
      }
      true
    }
    _ => true,
  }
}

/// Solve a square system whose entries are all machine numbers (real or
/// complex, with at least one inexact entry) by LU decomposition with partial
/// pivoting on plain `f64` pairs. Exact symbolic elimination is orders of
/// magnitude slower on the dense numeric systems that grid-based simulations
/// build. Returns `None` when the data is not purely numeric or the matrix is
/// (numerically) singular, so the general path can report it.
fn linear_solve_machine(matrix: &[Vec<Expr>], b: &[Expr]) -> Option<Expr> {
  // A matrix right-hand side solves for each of its columns at once.
  let columns: Vec<Vec<Expr>> = if b.iter().all(|e| matches!(e, Expr::List(_)))
  {
    let rows: Vec<&[Expr]> = b
      .iter()
      .map(|e| match e {
        Expr::List(items) => &items[..],
        _ => unreachable!(),
      })
      .collect();
    let k = rows.first()?.len();
    if k == 0 || rows.iter().any(|r| r.len() != k) {
      return None;
    }
    rows.iter().map(|r| r.to_vec()).collect()
  } else {
    b.iter().map(|e| vec![e.clone()]).collect()
  };
  let solution = linear_solve_machine_columns(matrix, &columns)?;
  let to_expr = |(re, im): (f64, f64)| {
    if im == 0.0 {
      Expr::Real(re)
    } else {
      crate::functions::math_ast::numeric_utils::build_complex_float_expr(
        re, im,
      )
    }
  };
  if b.iter().all(|e| matches!(e, Expr::List(_))) {
    Some(Expr::List(
      solution
        .into_iter()
        .map(|row| {
          Expr::List(row.into_iter().map(to_expr).collect::<Vec<_>>().into())
        })
        .collect::<Vec<_>>()
        .into(),
    ))
  } else {
    Some(Expr::List(
      solution
        .into_iter()
        .map(|row| to_expr(row[0]))
        .collect::<Vec<_>>()
        .into(),
    ))
  }
}

/// `matrix . x == rhs` for an n-by-k right-hand side, by Gaussian
/// elimination with partial pivoting in machine arithmetic (complex only
/// when some entry is). `None` unless some entry is inexact and all are
/// numbers, or when the matrix is singular.
fn linear_solve_machine_columns(
  matrix: &[Vec<Expr>],
  rhs: &[Vec<Expr>],
) -> Option<Vec<Vec<(f64, f64)>>> {
  use crate::functions::math_ast::numeric_utils::try_extract_complex_f64;
  fn has_real(e: &Expr) -> bool {
    match e {
      Expr::Real(_) => true,
      Expr::BinaryOp { left, right, .. } => has_real(left) || has_real(right),
      Expr::UnaryOp { operand, .. } => has_real(operand),
      Expr::FunctionCall { args, .. } => args.iter().any(has_real),
      _ => false,
    }
  }
  let n = matrix.len();
  let k = rhs.first()?.len();
  if rhs.len() != n
    || !matrix
      .iter()
      .flatten()
      .chain(rhs.iter().flatten())
      .any(has_real)
  {
    return None;
  }
  let mut a: Vec<Vec<(f64, f64)>> = Vec::with_capacity(n);
  for (row, r) in matrix.iter().zip(rhs) {
    let mut parts = Vec::with_capacity(n + k);
    for e in row.iter().chain(r.iter()) {
      let (re, im) = match e {
        Expr::Real(x) => (*x, 0.0),
        Expr::Integer(i) => (*i as f64, 0.0),
        other => try_extract_complex_f64(other)?,
      };
      if !re.is_finite() || !im.is_finite() {
        return None;
      }
      parts.push((re, im));
    }
    a.push(parts);
  }
  let width = n + k;
  let solution = if a.iter().flatten().all(|c| c.1 == 0.0) {
    // Real arithmetic.
    let mut m: Vec<Vec<f64>> =
      a.iter().map(|r| r.iter().map(|c| c.0).collect()).collect();
    for col in 0..n {
      let pivot_row = (col..n)
        .max_by(|&r1, &r2| m[r1][col].abs().total_cmp(&m[r2][col].abs()))?;
      if m[pivot_row][col] == 0.0 {
        return None;
      }
      m.swap(col, pivot_row);
      let (upper, lower) = m.split_at_mut(col + 1);
      let pivot_row = &upper[col];
      let pivot = pivot_row[col];
      for row in lower.iter_mut() {
        if row[col] == 0.0 {
          continue;
        }
        let factor = row[col] / pivot;
        for j in col..width {
          row[j] -= factor * pivot_row[j];
        }
      }
    }
    let mut x = vec![vec![0.0; k]; n];
    for c in 0..k {
      for i in (0..n).rev() {
        let mut sum = m[i][n + c];
        for j in (i + 1)..n {
          sum -= m[i][j] * x[j][c];
        }
        x[i][c] = sum / m[i][i];
        if !x[i][c].is_finite() {
          return None;
        }
      }
    }
    x.into_iter()
      .map(|r| r.into_iter().map(|v| (v, 0.0)).collect())
      .collect()
  } else {
    let mul = |x: (f64, f64), y: (f64, f64)| {
      (x.0 * y.0 - x.1 * y.1, x.0 * y.1 + x.1 * y.0)
    };
    let div = |x: (f64, f64), y: (f64, f64)| {
      let d = y.0 * y.0 + y.1 * y.1;
      ((x.0 * y.0 + x.1 * y.1) / d, (x.1 * y.0 - x.0 * y.1) / d)
    };
    let norm = |x: (f64, f64)| x.0.hypot(x.1);
    for col in 0..n {
      let pivot_row = (col..n)
        .max_by(|&r1, &r2| norm(a[r1][col]).total_cmp(&norm(a[r2][col])))?;
      if norm(a[pivot_row][col]) == 0.0 {
        return None;
      }
      a.swap(col, pivot_row);
      let pivot = a[col][col];
      for row in (col + 1)..n {
        if a[row][col] == (0.0, 0.0) {
          continue;
        }
        let factor = div(a[row][col], pivot);
        for j in col..width {
          let p = mul(factor, a[col][j]);
          a[row][j] = (a[row][j].0 - p.0, a[row][j].1 - p.1);
        }
      }
    }
    let mut x = vec![vec![(0.0, 0.0); k]; n];
    for c in 0..k {
      for i in (0..n).rev() {
        let mut sum = a[i][n + c];
        for j in (i + 1)..n {
          let p = mul(a[i][j], x[j][c]);
          sum = (sum.0 - p.0, sum.1 - p.1);
        }
        x[i][c] = div(sum, a[i][i]);
        if !x[i][c].0.is_finite() || !x[i][c].1.is_finite() {
          return None;
        }
      }
    }
    x
  };
  Some(solution)
}

/// `LinearSolve[sparse, b]` for a large machine-number sparse matrix,
/// without densifying it: the unknowns are renumbered by reverse
/// Cuthill-McKee to bring the nonzeros near the diagonal, and the banded
/// system is factored by Gaussian elimination with partial pivoting (the
/// LAPACK `gbsv` scheme). `None` for small, exact, symbolic or singular
/// systems and for ones whose band stays too wide, which the dense solver
/// handles.
pub(crate) fn sparse_linear_solve(a: &Expr, b: &Expr) -> Option<Expr> {
  // A machine sparse matrix has the background `0.`; it reads like `0`.
  let a = match a {
    Expr::FunctionCall { name, args }
      if name == "SparseArray"
        && args.len() == 4
        && matches!(&args[2], Expr::Real(z) if *z == 0.0) =>
    {
      let mut args = args.to_vec();
      args[2] = Expr::Integer(0);
      Expr::FunctionCall {
        name: name.clone(),
        args: args.into(),
      }
    }
    other => other.clone(),
  };
  let ([n, m], entries) = sparse_matrix_parts(&a)?;
  if n != m || n < 100 {
    return None;
  }
  // A sparse right-hand side is read as the vector it stands for.
  let dense_b;
  let b = if matches!(b, Expr::FunctionCall { name, .. } if name == "SparseArray")
  {
    dense_b =
      crate::evaluator::evaluate_expr_to_expr(&call1("Normal", b.clone()))
        .ok()?;
    &dense_b
  } else {
    b
  };
  let Expr::List(rhs) = b else { return None };
  if rhs.len() != n {
    return None;
  }
  let machine = entries.iter().any(|(_, v)| matches!(v, Expr::Real(_)))
    || rhs.iter().any(|v| matches!(v, Expr::Real(_)));
  if !machine {
    return None;
  }
  let number = |v: &Expr| match v {
    Expr::Real(x) => Some(*x),
    Expr::Integer(k) => Some(*k as f64),
    other
      if crate::functions::predicate_ast::is_numeric_q(other)
        && !contains_imaginary_unit(other) =>
    {
      crate::functions::math_ast::try_eval_to_f64(other)
    }
    _ => None,
  };
  let rhs: Vec<f64> = rhs.iter().map(number).collect::<Option<_>>()?;
  let mut triplets: Vec<(usize, usize, f64)> =
    Vec::with_capacity(entries.len());
  for (pos, v) in &entries {
    let x = number(v)?;
    if x != 0.0 {
      triplets.push((pos[0] as usize - 1, pos[1] as usize - 1, x));
    }
  }
  // Equations are first matched to unknowns (a maximum bipartite matching
  // by augmenting paths), so every unknown's own equation sits on the
  // diagonal; a system written in any order then has a pattern that a
  // symmetric renumbering can bring near the diagonal.
  let mut cols_of_row: Vec<Vec<usize>> = vec![Vec::new(); n];
  for &(i, j, _) in &triplets {
    cols_of_row[i].push(j);
  }
  let mut row_of_col: Vec<Option<usize>> = vec![None; n];
  let mut col_of_row: Vec<Option<usize>> = vec![None; n];
  // Cheap pass: a free column straight away.
  for i in 0..n {
    if let Some(&j) = cols_of_row[i].iter().find(|&&j| row_of_col[j].is_none())
    {
      row_of_col[j] = Some(i);
      col_of_row[i] = Some(j);
    }
  }
  for start in 0..n {
    if col_of_row[start].is_some() {
      continue;
    }
    // Depth-first search for an augmenting path, iteratively.
    let mut visited = vec![false; n];
    let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
    let mut path: Vec<(usize, usize)> = Vec::new();
    let mut found = false;
    while let Some(&mut (row, ref mut next)) = stack.last_mut() {
      if *next >= cols_of_row[row].len() {
        stack.pop();
        path.pop();
        continue;
      }
      let col = cols_of_row[row][*next];
      *next += 1;
      if visited[col] {
        continue;
      }
      visited[col] = true;
      path.push((row, col));
      match row_of_col[col] {
        None => {
          found = true;
          break;
        }
        Some(other) => stack.push((other, 0)),
      }
    }
    if !found {
      return None;
    }
    // `path` holds the edge out of every open frame, the last to a free
    // unknown: each row along it takes the next unknown.
    for &(row, col) in &path {
      row_of_col[col] = Some(row);
      col_of_row[row] = Some(col);
    }
  }
  // Row `i` of the system is moved to the place of its matched unknown.
  let triplets: Vec<(usize, usize, f64)> = triplets
    .into_iter()
    .map(|(i, j, x)| (col_of_row[i].unwrap(), j, x))
    .collect();
  let rhs: Vec<f64> = {
    let mut r = vec![0.0; n];
    for (i, v) in rhs.iter().enumerate() {
      r[col_of_row[i].unwrap()] = *v;
    }
    r
  };
  // Reverse Cuthill-McKee on the symmetrized pattern.
  let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
  for &(i, j, _) in &triplets {
    if i != j {
      adj[i].push(j);
      adj[j].push(i);
    }
  }
  for list in &mut adj {
    list.sort_unstable();
    list.dedup();
  }
  let mut order: Vec<usize> = Vec::with_capacity(n);
  let mut seen = vec![false; n];
  let mut by_degree: Vec<usize> = (0..n).collect();
  by_degree.sort_by_key(|&v| adj[v].len());
  for &start in &by_degree {
    if seen[start] {
      continue;
    }
    seen[start] = true;
    let mut head = order.len();
    order.push(start);
    while head < order.len() {
      let v = order[head];
      head += 1;
      let mut next: Vec<usize> =
        adj[v].iter().copied().filter(|&u| !seen[u]).collect();
      next.sort_by_key(|&u| adj[u].len());
      for u in next {
        seen[u] = true;
        order.push(u);
      }
    }
  }
  order.reverse();
  let mut position = vec![0usize; n];
  for (new, &old) in order.iter().enumerate() {
    position[old] = new;
  }
  let (mut kl, mut ku) = (0usize, 0usize);
  for &(i, j, _) in &triplets {
    let (pi, pj) = (position[i], position[j]);
    if pi > pj {
      kl = kl.max(pi - pj);
    } else {
      ku = ku.max(pj - pi);
    }
  }
  // Row interchanges widen the upper band by `kl`.
  let width = 2 * kl + ku + 1;
  if width.checked_mul(n)? > 40_000_000 {
    return None;
  }
  // Row `r` holds columns `r - kl ..= r + kl + ku` (absolute, by position).
  let mut band = vec![0.0f64; width * n];
  let at = |r: usize, c: usize| -> Option<usize> {
    let lo = r as isize - kl as isize;
    let off = c as isize - lo;
    (off >= 0 && (off as usize) < width).then(|| r * width + off as usize)
  };
  for &(i, j, x) in &triplets {
    let k = at(position[i], position[j])?;
    band[k] += x;
  }
  let mut y: Vec<f64> = vec![0.0; n];
  for (i, v) in rhs.iter().enumerate() {
    y[position[i]] = *v;
  }
  for k in 0..n {
    let last = (k + kl).min(n - 1);
    let p = (k..=last).max_by(|&r1, &r2| {
      band[at(r1, k).unwrap()]
        .abs()
        .total_cmp(&band[at(r2, k).unwrap()].abs())
    })?;
    let pivot = band[at(p, k).unwrap()];
    if pivot == 0.0 || !pivot.is_finite() {
      return None;
    }
    let hi = (k + kl + ku).min(n - 1);
    if p != k {
      for c in k..=hi {
        let (a1, a2) = (at(k, c).unwrap(), at(p, c)?);
        band.swap(a1, a2);
      }
      y.swap(k, p);
    }
    for r in k + 1..=last {
      let idx = at(r, k).unwrap();
      let factor = band[idx] / pivot;
      if factor == 0.0 {
        continue;
      }
      band[idx] = 0.0;
      for c in k + 1..=hi {
        let src = band[at(k, c).unwrap()];
        if src != 0.0 {
          band[at(r, c)?] -= factor * src;
        }
      }
      y[r] -= factor * y[k];
    }
  }
  let mut x = vec![0.0f64; n];
  for k in (0..n).rev() {
    let hi = (k + kl + ku).min(n - 1);
    let mut sum = y[k];
    for c in k + 1..=hi {
      sum -= band[at(k, c).unwrap()] * x[c];
    }
    x[k] = sum / band[at(k, k).unwrap()];
  }
  Some(Expr::List(
    (0..n).map(|i| Expr::Real(x[position[i]])).collect(),
  ))
}

pub fn linear_solve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LinearSolve expects exactly 2 arguments".into(),
    ));
  }

  let Some(matrix) = expr_to_matrix(&args[0]) else {
    return Ok(unevaluated("LinearSolve", args));
  };

  let b = match &args[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(unevaluated("LinearSolve", args));
    }
  };

  let n = matrix.len();
  if n == 0 {
    return Err(InterpreterError::EvaluationError(
      "LinearSolve: empty matrix".into(),
    ));
  }
  let ncols = matrix[0].len();
  if matrix.iter().any(|row| row.len() != ncols) {
    // Ragged matrix — leave unevaluated.
    return Ok(unevaluated("LinearSolve", args));
  }
  // Rectangular (non-square) systems: solve via the reduced row echelon form
  // of the augmented matrix, taking the particular solution with the
  // non-pivot (free) variables set to 0, matching wolframscript.
  if ncols != n {
    if b.len() != n {
      crate::emit_message(
        "LinearSolve::lslc: Coefficient matrix and target vector(s) or matrix do not have the same dimensions.",
      );
      return Ok(unevaluated("LinearSolve", args));
    }
    match linear_solve_rectangular_machine(&matrix, &b, ncols) {
      Some(Some(x)) => {
        return Ok(Expr::List(
          x.into_iter().map(Expr::Real).collect::<Vec<_>>().into(),
        ));
      }
      Some(None) => {
        crate::emit_message(
          "LinearSolve::nosol: Linear equation encountered that has no solution.",
        );
        return Ok(unevaluated("LinearSolve", args));
      }
      None => {}
    }
    return Ok(linear_solve_rectangular(&matrix, &b, ncols, args));
  }
  if b.len() != n {
    crate::emit_message(
      "LinearSolve::lslc: Coefficient matrix and target vector(s) or matrix do not have the same dimensions.",
    );
    return Ok(unevaluated("LinearSolve", args));
  }

  if let Some(solution) = linear_solve_machine(&matrix, &b) {
    return Ok(solution);
  }

  // Build augmented matrix [A | b]
  let mut aug: Vec<Vec<Expr>> = Vec::with_capacity(n);
  for i in 0..n {
    let mut row = matrix[i].clone();
    row.push(b[i].clone());
    aug.push(row);
  }

  // Forward elimination with partial pivoting. Track which columns were
  // used as pivots (the rest become free variables in the singular case).
  let mut pivot_col_of_row: Vec<Option<usize>> = vec![None; n];
  let mut next_row = 0usize;
  for col in 0..n {
    if next_row >= n {
      break;
    }
    // Find a non-zero pivot at or below next_row in column `col`.
    let mut pivot_row = None;
    for row in next_row..n {
      if !is_zero_expr(&aug[row][col]) {
        pivot_row = Some(row);
        break;
      }
    }
    let Some(pivot_row) = pivot_row else { continue };

    if pivot_row != next_row {
      aug.swap(next_row, pivot_row);
    }
    pivot_col_of_row[next_row] = Some(col);

    // Eliminate below the pivot.
    let pivot = aug[next_row][col].clone();
    for row in (next_row + 1)..n {
      if is_zero_expr(&aug[row][col]) {
        continue;
      }
      let factor = eval_divide(&aug[row][col], &pivot);
      aug[row][col] = Expr::Integer(0);
      for j in (col + 1)..=n {
        let prod = eval_mul(&factor, &aug[next_row][j]);
        aug[row][j] = eval_sub(&aug[row][j], &prod);
        aug[row][j] = simplify_expr(&aug[row][j]);
      }
    }
    next_row += 1;
  }

  // Consistency check: any fully-zero row in A must have zero b.
  for row in 0..n {
    if pivot_col_of_row[row].is_none() && !is_zero_expr(&aug[row][n]) {
      crate::emit_message(
        "LinearSolve::nosol: Linear equation encountered that has no solution.",
      );
      return Ok(unevaluated("LinearSolve", args));
    }
  }

  // Back substitution. Free columns (no pivot in any row) are set to 0,
  // giving a particular solution — matches wolframscript for singular
  // but consistent systems.
  let mut x = vec![Expr::Integer(0); n];
  for i in (0..n).rev() {
    let Some(pivot_col) = pivot_col_of_row[i] else {
      continue;
    };
    let mut sum = aug[i][n].clone();
    for j in (pivot_col + 1)..n {
      let prod = eval_mul(&aug[i][j], &x[j]);
      sum = eval_sub(&sum, &prod);
    }
    x[pivot_col] = eval_divide(&sum, &aug[i][pivot_col]);
    x[pivot_col] = simplify_expr(&x[pivot_col]);
  }

  Ok(Expr::List(x.into()))
}

/// Eigensystem[matrix] - returns {eigenvalues, eigenvectors}
pub fn eigensystem_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let Some(inner) = unwrap_single_matrix_list(args) {
    return eigensystem_ast(&inner);
  }
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Eigensystem expects exactly 1 argument".into(),
    ));
  }

  // Check if argument is a matrix (list of lists)
  if !matches!(&args[0], Expr::List(rows) if !rows.is_empty() && matches!(&rows[0], Expr::List(_)))
  {
    return Ok(unevaluated("Eigensystem", args));
  }

  let eigenvalues = eigenvalues_ast(args)?;
  let eigenvectors = eigenvectors_ast(args)?;

  Ok(Expr::List(vec![eigenvalues, eigenvectors].into()))
}

/// Minors[m] - gives the minors of a matrix (determinants of (n-1)×(n-1) submatrices)
/// Minors[m, k] - gives the k-th order minors
pub fn minors_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Minors expects 1, 2, or 3 arguments".into(),
    ));
  }

  let Some(matrix) = expr_to_matrix(&args[0]) else {
    return Ok(unevaluated("Minors", args));
  };

  let nrows = matrix.len();
  if nrows == 0 {
    return Ok(Expr::List(vec![].into()));
  }
  let ncols = matrix[0].len();

  // Determine the minor order k
  let k = if args.len() >= 2 {
    match &args[1] {
      Expr::Integer(n) if *n > 0 => *n as usize,
      _ => {
        return Ok(unevaluated("Minors", args));
      }
    }
  } else {
    // Default: n-1 for n×n matrix (standard minors). A matrix with an
    // empty row has no minors at all, and subtracting from zero used to
    // wrap round.
    match nrows.min(ncols).checked_sub(1) {
      Some(k) => k,
      None => return Ok(Expr::List(Vec::new().into())),
    }
  };

  if k > nrows || k > ncols {
    return Err(InterpreterError::EvaluationError(
      "Minors: minor order exceeds matrix dimensions".into(),
    ));
  }

  // Generate all combinations of k rows and k columns
  let row_combos = combinations(nrows, k);
  let col_combos = combinations(ncols, k);

  // Build the result matrix
  let mut result_rows = Vec::new();
  for row_combo in &row_combos {
    let mut result_row = Vec::new();
    for col_combo in &col_combos {
      // Extract submatrix
      let mut submatrix = Vec::new();
      for &r in row_combo {
        let mut row = Vec::new();
        for &c in col_combo {
          row.push(matrix[r][c].clone());
        }
        submatrix.push(row);
      }
      // Apply function (default is Det)
      if args.len() == 3 {
        // Minors[m, k, f] - apply f to the submatrix
        let sub_expr = Expr::List(
          submatrix
            .into_iter()
            .map(|v| Expr::List(v.into()))
            .collect(),
        );
        let f_call = call1(
          match &args[2] {
            Expr::Identifier(name) => name.as_str(),
            _ => "Det",
          },
          sub_expr,
        );
        result_row.push(crate::evaluator::evaluate_expr_to_expr(&f_call)?);
      } else {
        result_row.push(determinant(&submatrix));
      }
    }
    result_rows.push(Expr::List(result_row.into()));
  }

  Ok(Expr::List(result_rows.into()))
}

/// Generate all combinations of `k` items from `0..n`
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
  let mut result = Vec::new();
  let mut combo = Vec::new();
  combinations_helper(0, n, k, &mut combo, &mut result);
  result
}

fn combinations_helper(
  start: usize,
  n: usize,
  k: usize,
  current: &mut Vec<usize>,
  result: &mut Vec<Vec<usize>>,
) {
  if current.len() == k {
    result.push(current.clone());
    return;
  }
  for i in start..n {
    current.push(i);
    combinations_helper(i + 1, n, k, current, result);
    current.pop();
  }
}

/// LatticeReduce[{v1, v2, ...}] - LLL lattice basis reduction
/// Implements the Lenstra–Lenstra–Lovász algorithm with exact rational arithmetic.
pub fn lattice_reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(unevaluated("LatticeReduce", args));
  }

  // Extract the matrix of integer vectors
  let Expr::List(rows) = &args[0] else {
    return Ok(unevaluated("LatticeReduce", args));
  };

  if rows.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Parse into Vec<Vec<i128>>
  let mut basis: Vec<Vec<i128>> = Vec::new();
  for row in rows {
    match row {
      Expr::List(elems) => {
        let mut vec = Vec::new();
        for e in elems {
          match e {
            Expr::Integer(n) => vec.push(*n),
            _ => {
              return Ok(unevaluated("LatticeReduce", args));
            }
          }
        }
        basis.push(vec);
      }
      _ => {
        return Ok(unevaluated("LatticeReduce", args));
      }
    }
  }

  // Verify all vectors have the same dimension
  let dim = basis[0].len();
  if basis.iter().any(|v| v.len() != dim) {
    return Err(InterpreterError::EvaluationError(
      "LatticeReduce: all vectors must have the same dimension".into(),
    ));
  }

  // Run LLL algorithm
  let result = lll_reduce(&mut basis);

  // Convert back to Expr
  let result_rows: Vec<Expr> = result
    .iter()
    .map(|row| Expr::List(row.iter().map(|&n| Expr::Integer(n)).collect()))
    .collect();

  Ok(Expr::List(result_rows.into()))
}

fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
  a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// LLL lattice basis reduction algorithm (delta = 3/4)
/// Uses floating-point Gram-Schmidt with integer basis vectors
fn lll_reduce(basis: &mut [Vec<i128>]) -> Vec<Vec<i128>> {
  let n = basis.len();
  if n <= 1 {
    return basis
      .iter()
      .filter(|v| v.iter().any(|&x| x != 0))
      .cloned()
      .collect();
  }

  let dim = basis[0].len();
  let delta: f64 = 0.75;

  // Gram-Schmidt orthogonal vectors (as f64)
  let mut bstar: Vec<Vec<f64>> = vec![vec![0.0; dim]; n];
  // mu coefficients
  let mut mu: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
  // B[i] = ||b*_i||^2
  let mut big_b: Vec<f64> = vec![0.0; n];

  // Compute Gram-Schmidt
  let compute_gs = |basis: &[Vec<i128>],
                    bstar: &mut Vec<Vec<f64>>,
                    mu: &mut Vec<Vec<f64>>,
                    big_b: &mut Vec<f64>| {
    let n = basis.len();
    for i in 0..n {
      // b*_i = b_i
      for d in 0..dim {
        bstar[i][d] = basis[i][d] as f64;
      }
      // Subtract projections
      for j in 0..i {
        if big_b[j].abs() < 1e-30 {
          mu[i][j] = 0.0;
          continue;
        }
        let bi_f: Vec<f64> = basis[i].iter().map(|&x| x as f64).collect();
        mu[i][j] = dot_f64(&bi_f, &bstar[j]) / big_b[j];
        for d in 0..dim {
          bstar[i][d] -= mu[i][j] * bstar[j][d];
        }
      }
      big_b[i] = dot_f64(&bstar[i], &bstar[i]);
    }
  };

  compute_gs(basis, &mut bstar, &mut mu, &mut big_b);

  let max_iters = n * n * 200;
  let mut iters = 0;
  let mut k = 1;
  while k < n && iters < max_iters {
    iters += 1;

    // Size reduce basis[k] by basis[j] for j = k-1, k-2, ..., 0
    for j in (0..k).rev() {
      if mu[k][j].abs() > 0.5 {
        let r = mu[k][j].round() as i128;
        if r != 0 {
          for d in 0..dim {
            basis[k][d] -= r * basis[j][d];
          }
          // Update mu values
          compute_gs(basis, &mut bstar, &mut mu, &mut big_b);
        }
      }
    }

    // Check Lovász condition
    if big_b[k - 1].abs() < 1e-30 {
      if big_b[k].abs() >= 1e-30 {
        basis.swap(k, k - 1);
        compute_gs(basis, &mut bstar, &mut mu, &mut big_b);
      } else {
        k += 1;
      }
    } else {
      let lhs = big_b[k] + mu[k][k - 1] * mu[k][k - 1] * big_b[k - 1];
      if lhs >= delta * big_b[k - 1] {
        k += 1;
      } else {
        basis.swap(k, k - 1);
        compute_gs(basis, &mut bstar, &mut mu, &mut big_b);
        k = if k > 1 { k - 1 } else { 1 };
      }
    }
  }

  // Remove zero vectors
  basis
    .iter()
    .filter(|v| v.iter().any(|&x| x != 0))
    .cloned()
    .collect()
}

// ─── FindIntegerNullVector ───────────────────────────────────────────

/// FindIntegerNullVector[{x1, x2, ..., xn}] - finds integers {a1, ..., an}
/// such that a1*x1 + a2*x2 + ... + an*xn == 0.
/// Uses LLL lattice reduction on a specially constructed lattice.
pub fn find_integer_null_vector_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated("FindIntegerNullVector", args));
  }

  let Expr::List(items) = &args[0] else {
    return Ok(unevaluated("FindIntegerNullVector", args));
  };

  let n = items.len();
  if n < 2 {
    return Ok(unevaluated("FindIntegerNullVector", args));
  }

  // Evaluate all elements to f64
  let vals: Vec<f64> = items
    .iter()
    .map(|e| {
      let evaluated = crate::evaluator::evaluate_expr_to_expr(e)
        .unwrap_or_else(|_| e.clone());
      try_eval_to_f64(&evaluated)
    })
    .collect::<Option<Vec<f64>>>()
    .ok_or_else(|| {
      InterpreterError::EvaluationError(
        "FindIntegerNullVector: all elements must be numeric".into(),
      )
    })?;

  // Determine scaling factor based on desired precision
  // Use 10^(precision/2) as scaling factor
  let precision: f64 = 15.0; // default working precision
  let scale = 10f64.powi((precision * 0.6) as i32);

  // Construct the lattice: n rows of dimension n+1
  // Row i: e_i (standard basis) with last column = round(x_i * scale)
  let mut basis: Vec<Vec<i128>> = Vec::new();
  for i in 0..n {
    let mut row = vec![0i128; n + 1];
    row[i] = 1;
    row[n] = (vals[i] * scale).round() as i128;
    basis.push(row);
  }

  // Apply LLL reduction
  let reduced = lll_reduce(&mut basis);

  if reduced.is_empty() {
    return Ok(no_integer_null_vector(args));
  }

  // Find the vector with smallest |last component| (closest to null relation)
  // Among those, prefer the one with smallest norm of the integer coefficients
  let mut best: Option<&Vec<i128>> = None;
  let mut best_score = f64::MAX;

  for row in &reduced {
    let last = row[n].abs() as f64;
    let coeff_norm: f64 = row[..n]
      .iter()
      .map(|&c| (c as f64).powi(2))
      .sum::<f64>()
      .sqrt();
    if coeff_norm < 1e-10 {
      continue; // skip zero vector
    }
    // Score: prefer small last component relative to coefficient norm
    let score = last / (coeff_norm + 1.0);
    if score < best_score {
      best_score = score;
      best = Some(row);
    }
  }

  let result = match best {
    Some(row) => row[..n].to_vec(),
    None => {
      return Ok(no_integer_null_vector(args));
    }
  };

  // Verify the relation actually holds (dot product should be very small)
  let dot: f64 = result
    .iter()
    .zip(vals.iter())
    .map(|(&a, &x)| a as f64 * x)
    .sum();
  let max_val = vals.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
  if dot.abs() > max_val * 1e-6 {
    // Relation doesn't hold well enough
    return Ok(no_integer_null_vector(args));
  }

  // FindIntegerNullVector[{x1, …}, d]: a relation whose norm exceeds the
  // bound d does not count.
  if let Some(bound) = args.get(1).and_then(try_eval_to_f64) {
    let norm = result
      .iter()
      .map(|&c| (c as f64).powi(2))
      .sum::<f64>()
      .sqrt();
    if norm > bound {
      return Ok(no_integer_null_vector(args));
    }
  }

  // Normalize: make the first nonzero coefficient positive
  let mut coeffs = result;
  if let Some(&first_nonzero) = coeffs.iter().find(|&&c| c != 0)
    && first_nonzero < 0
  {
    for c in &mut coeffs {
      *c = -*c;
    }
  }

  Ok(Expr::List(coeffs.into_iter().map(Expr::Integer).collect()))
}

/// The result of FindIntegerNullVector when no relation was found: the call
/// stays unevaluated with FindIntegerNullVector::rnfu, or with
/// FindIntegerNullVector::norel when a norm bound d was given.
fn no_integer_null_vector(args: &[Expr]) -> Expr {
  let xs = crate::syntax::expr_to_string(&args[0]);
  match args.get(1) {
    Some(d) => crate::emit_message(&format!(
      "FindIntegerNullVector::norel: There is no integer null vector for {} \
       with norm less than or equal to {}.",
      xs,
      crate::syntax::expr_to_string(d)
    )),
    None => crate::emit_message(&format!(
      "FindIntegerNullVector::rnfu: FindIntegerNullVector has not found an \
       integer null vector for {xs}."
    )),
  }
  unevaluated("FindIntegerNullVector", args)
}

/// Find an integer relation among `vals` via LLL: integers `a` with
/// `sum(a_i * vals_i) ≈ 0`, returned together with the absolute residual. The
/// lattice is the same one `FindIntegerNullVector` uses. Returns `None` if the
/// reduction produces only the zero vector.
fn lll_integer_relation(vals: &[f64], scale: f64) -> Option<(Vec<i128>, f64)> {
  let n = vals.len();
  let mut basis: Vec<Vec<i128>> = Vec::with_capacity(n);
  for (i, &v) in vals.iter().enumerate() {
    let mut row = vec![0i128; n + 1];
    row[i] = 1;
    row[n] = (v * scale).round() as i128;
    basis.push(row);
  }
  let reduced = lll_reduce(&mut basis);
  // Pick the genuine relation: the reduced vector whose ACTUAL residual
  // `|sum(a_i * vals_i)|` (in f64, independent of the lattice scale) is
  // smallest. A minimal integer relation sits at machine-zero; spurious short
  // vectors have far larger residuals. Ties break toward the smaller-height
  // (smaller coefficient-norm) vector, which is the minimal polynomial.
  let mut best: Option<(Vec<i128>, f64)> = None;
  for row in &reduced {
    let coeff_norm_sq: i128 = row[..n].iter().map(|&c| c * c).sum();
    if coeff_norm_sq == 0 {
      continue;
    }
    let residual: f64 = row[..n]
      .iter()
      .zip(vals.iter())
      .map(|(&a, &x)| a as f64 * x)
      .sum::<f64>()
      .abs();
    let better = match &best {
      None => true,
      Some((cur, cur_res)) => {
        let cur_norm_sq: i128 = cur.iter().map(|&c| c * c).sum();
        residual < *cur_res * 0.999
          || ((residual - *cur_res).abs() <= *cur_res * 0.001
            && coeff_norm_sq < cur_norm_sq)
      }
    };
    if better {
      best = Some((row[..n].to_vec(), residual));
    }
  }
  best
}

/// RootApproximant[x] / RootApproximant[x, n] — recover the algebraic number of
/// least degree and height that `x` (a machine real) approximates to its
/// precision. Searches increasing degrees for a minimal integer polynomial that
/// `x` satisfies (via LLL), then returns `Root[poly, k, 0]` for the branch
/// closest to `x`; the evaluator's `Root` machinery renders low-degree cases as
/// radicals or rationals, matching wolframscript (`Sqrt[2]`, `(1 + Sqrt[5])/2`,
/// `Root[-2 + #1^3 &, 1, 0]`, `1/2`).
fn contains_inexact(e: &Expr) -> bool {
  match e {
    Expr::Real(_) | Expr::BigFloat(..) => true,
    Expr::FunctionCall { args, .. } => args.iter().any(contains_inexact),
    Expr::List(items) => items.iter().any(contains_inexact),
    Expr::BinaryOp { left, right, .. } => {
      contains_inexact(left) || contains_inexact(right)
    }
    Expr::UnaryOp { operand, .. } => contains_inexact(operand),
    _ => false,
  }
}

pub fn root_approximant_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated("RootApproximant", args));
  }

  // Exact numeric input is already an algebraic number: return it unchanged.
  if matches!(&args[0], Expr::Integer(_))
    || matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Rational")
  {
    return Ok(args[0].clone());
  }

  // Listable.
  if let Expr::List(items) = &args[0] {
    return items
      .iter()
      .map(|item| {
        let mut call_args = vec![item.clone()];
        call_args.extend(args[1..].iter().cloned());
        root_approximant_ast(&call_args)
      })
      .collect::<Result<Vec<_>, _>>()
      .map(|v| Expr::List(v.into()));
  }

  let x = match &args[0] {
    Expr::Real(v) => *v,
    other => {
      if let Some(v) = try_eval_to_f64(other) {
        v
      } else {
        // An exact complex algebraic number (`Root[1 + #^4 &, 1]`) is
        // already what it approximates, in canonical form; one whose
        // imaginary part cancels (`Root[…, 3] + Root[…, 4]`) is real.
        if !crate::functions::predicate_ast::is_numeric_q(other) {
          return Ok(unevaluated("RootApproximant", args));
        }
        let value =
          crate::evaluator::evaluate_expr_to_expr(&call1("N", other.clone()))?;
        match crate::functions::math_ast::numeric_utils::try_extract_complex_float(
        &value,
      ) {
        Some((re, im)) if im.abs() <= 1e-13 * re.abs().max(1.0) => re,
        Some(_) if !contains_inexact(other) => {
          return crate::evaluator::evaluate_expr_to_expr(&call1(
            "RootReduce",
            other.clone(),
          ));
        }
        _ => return Ok(unevaluated("RootApproximant", args)),
      }
      }
    }
  };
  if !x.is_finite() {
    return Ok(unevaluated("RootApproximant", args));
  }

  // Maximum degree to search: an explicit second argument, else a default that
  // is generous for machine precision without inviting spurious relations.
  let max_degree = match args.get(1) {
    Some(Expr::Integer(n)) if *n >= 1 => (*n as usize).min(16),
    Some(_) => return Ok(unevaluated("RootApproximant", args)),
    None => 8,
  };

  // Detect the integer relation whose residual is at machine-zero: that is the
  // polynomial x genuinely satisfies. A degree-d LLL fit to a number that is
  // NOT of degree d leaves a residual near SCALE^(-(d)/(d+1)) (e.g. ~1e-8 for a
  // spurious quadratic), whereas a genuine relation sits at ~1e-15. The
  // tolerance sits well below the spurious floor, scaled by |x|^d so large
  // arguments are handled.
  const SCALE: f64 = 1.0e12;

  for d in 1..=max_degree {
    let powers: Vec<f64> = (0..=d).map(|k| x.powi(k as i32)).collect();
    let Some((coeffs, residual)) = lll_integer_relation(&powers, SCALE) else {
      continue;
    };
    let res_tol = 1.0e-11 * x.abs().powi(d as i32).max(1.0);
    // The leading coefficient must be nonzero for a genuine degree-d relation,
    // and the fit must be tight (a spurious lower-degree fit is far looser).
    if coeffs.last() == Some(&0) || residual >= res_tol {
      continue;
    }
    if let Some(result) = build_root_from_coeffs(&coeffs, x) {
      return Ok(result);
    }
  }

  // No algebraic relation within the degree budget: fall back to the exact
  // rational the machine real represents, as wolframscript does.
  crate::evaluator::evaluate_expr_to_expr(&call(
    "Rationalize",
    vec![Expr::Real(x), Expr::Integer(0)],
  ))
}

/// Build `Root[poly, k, 0]` for the integer polynomial with the given ascending
/// coefficients, choosing the branch `k` whose numeric value is closest to `x`,
/// and let the evaluator simplify it (to a radical / rational when possible).
fn build_root_from_coeffs(coeffs: &[i128], x: f64) -> Option<Expr> {
  // Trim any trailing zero coefficients and normalise the sign so the leading
  // coefficient is positive (matching wolframscript's Root polynomials).
  let mut c = coeffs.to_vec();
  while c.len() > 1 && *c.last().unwrap() == 0 {
    c.pop();
  }
  let degree = c.len().checked_sub(1)?;
  if degree == 0 {
    return None;
  }
  if *c.last().unwrap() < 0 {
    for v in &mut c {
      *v = -*v;
    }
  }

  // Assemble the polynomial body in Slot[1], ascending, skipping zero terms.
  let slot = Expr::Slot(1);
  let mut terms: Vec<Expr> = Vec::new();
  for (i, &coeff) in c.iter().enumerate() {
    if coeff == 0 {
      continue;
    }
    let var_pow = match i {
      0 => None,
      1 => Some(slot.clone()),
      _ => Some(pow(slot.clone(), Expr::Integer(i as i128))),
    };
    let term = match (var_pow, coeff) {
      (None, coeff) => Expr::Integer(coeff),
      (Some(p), 1) => p,
      (Some(p), coeff) => times(vec![Expr::Integer(coeff), p]),
    };
    terms.push(term);
  }
  let body = match terms.len() {
    0 => return None,
    1 => terms.remove(0),
    _ => plus(terms),
  };
  let body = crate::evaluator::evaluate_expr_to_expr(&body).ok()?;
  let func = Expr::Function {
    body: Box::new(body),
  };

  // Pick the root index k (1-based, wolframscript's ordering) whose numeric
  // value is closest to x, by numericising each Root[func, k, 0].
  let make_root = |k: usize| {
    call(
      "Root",
      vec![func.clone(), Expr::Integer(k as i128), Expr::Integer(0)],
    )
  };
  let mut best_k = 0usize;
  let mut best_dist = f64::MAX;
  for k in 1..=degree {
    let n_expr = call1("N", make_root(k));
    if let Ok(v) = crate::evaluator::evaluate_expr_to_expr(&n_expr)
      && let Some(rv) = try_eval_to_f64(&v)
    {
      let dist = (rv - x).abs();
      if dist < best_dist {
        best_dist = dist;
        best_k = k;
      }
    }
  }
  if best_k == 0 || best_dist > 1e-6 {
    return None;
  }
  crate::evaluator::evaluate_expr_to_expr(&make_root(best_k)).ok()
}

// ─── FindFit ──────────────────────────────────────────────────────────

/// Split a model specification into the model and any constraints given
/// beside it: Wolfram writes a constrained fit as
/// `{model, cons₁, cons₂, …}`, e.g. `{a Sin[x + b], 50 < a < 60}`.
fn split_fit_model_spec(spec: &Expr) -> (Expr, Vec<Expr>) {
  match spec {
    // A bare list of two or more items is only a constrained model when
    // what follows the model reads as a condition; a `{f1, f2}` list of
    // basis functions (Fit's spelling) has no comparisons in it.
    Expr::List(items)
      if items.len() >= 2 && items[1..].iter().all(is_fit_constraint) =>
    {
      (items[0].clone(), items[1..].to_vec())
    }
    // `{form}` is a model with no constraints.
    Expr::List(items) if items.len() == 1 => (items[0].clone(), Vec::new()),
    other => (other.clone(), Vec::new()),
  }
}

/// Whether an expression is a comparison, i.e. a constraint rather than
/// another piece of the model.
fn is_fit_constraint(expr: &Expr) -> bool {
  match expr {
    Expr::Comparison { .. } => true,
    Expr::FunctionCall { name, .. } => matches!(
      name.as_str(),
      "Less" | "LessEqual" | "Greater" | "GreaterEqual" | "Inequality"
    ),
    _ => false,
  }
}

/// Box bounds `(low, high)` per parameter, read off the constraints. A
/// constraint that does not bound a single parameter by a number is left
/// out — those the fit simply does not enforce.
fn fit_parameter_bounds(
  constraints: &[Expr],
  param_names: &[String],
) -> Vec<(f64, f64)> {
  let mut bounds: Vec<(f64, f64)> =
    vec![(f64::NEG_INFINITY, f64::INFINITY); param_names.len()];
  // One `lhs op rhs` link of a comparison chain, with `op` reading left to
  // right ("<" means lhs < rhs).
  let mut apply = |lhs: &Expr, op: &str, rhs: &Expr| {
    let param_index = |e: &Expr| match e {
      Expr::Identifier(n) => param_names.iter().position(|p| p == n),
      _ => None,
    };
    let number = |e: &Expr| try_eval_to_f64(e);
    // `param op value` bounds the parameter from above for `<`/`<=`;
    // `value op param` bounds it from below.
    let (idx, value, upper) = match (param_index(lhs), number(rhs)) {
      (Some(i), Some(v)) => (i, v, matches!(op, "Less" | "LessEqual")),
      _ => match (number(lhs), param_index(rhs)) {
        (Some(v), Some(i)) => (i, v, matches!(op, "Greater" | "GreaterEqual")),
        _ => return,
      },
    };
    if upper {
      bounds[idx].1 = bounds[idx].1.min(value);
    } else {
      bounds[idx].0 = bounds[idx].0.max(value);
    }
  };
  for constraint in constraints {
    // A parsed chain is a `Comparison`; the same thing written as a call
    // (`Less[50, a, 60]`) is a `FunctionCall`. Both are read through the
    // one decomposition Wolfram's structural operations use.
    let owned;
    let (name, args) = match constraint {
      Expr::Comparison {
        operands,
        operators,
      } => {
        owned = crate::syntax::comparison_head_and_args(operands, operators);
        (owned.0.clone(), owned.1.clone())
      }
      Expr::FunctionCall { name, args } => {
        (name.clone(), args.iter().cloned().collect::<Vec<_>>())
      }
      _ => continue,
    };
    let args = &args[..];
    match name.as_str() {
      // `Inequality[a, op, b, op, c]` — the general chained form, walked
      // as (operand, op, operand, op, operand, …).
      "Inequality" => {
        let mut i = 0;
        while i + 2 < args.len() {
          if let Expr::Identifier(op) = &args[i + 1] {
            apply(&args[i], op, &args[i + 2]);
          }
          i += 2;
        }
      }
      // `Less[a, b, c]` is `a < b < c`: every neighbouring pair links.
      "Less" | "LessEqual" | "Greater" | "GreaterEqual" => {
        for pair in args.windows(2) {
          apply(&pair[0], &name, &pair[1]);
        }
      }
      _ => {}
    }
  }
  bounds
}

/// FindFit[data, model, params, var]
/// Finds parameter values that minimize the sum of squared residuals.
/// Returns {param1 -> val1, param2 -> val2, ...}.
pub fn find_fit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 4 {
    return Err(InterpreterError::EvaluationError(
      "FindFit expects at least 4 arguments: data, model, params, var".into(),
    ));
  }
  let (model, constraints) = split_fit_model_spec(&args[1]);
  // `NormFunction -> f` replaces the sum of squares by `f` of the residual
  // vector; the other options only steer how the minimum is sought.
  let mut norm: Option<Expr> = None;
  for opt in &args[4..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "NormFunction")
    {
      norm = Some(replacement.as_ref().clone());
    }
  }
  find_fit_weighted(
    &args[0],
    &model,
    &constraints,
    &args[2],
    &args[3],
    None,
    norm.as_ref(),
    "FindFit",
  )
}

/// Nelder-Mead minimization of `objective` from `start`, kept inside the
/// box `bounds`. The simplex is restarted around the best point until a
/// restart no longer improves it, which copes with the kinks of norms such
/// as the 1-norm. `start` is overwritten with the minimizer found.
fn minimize_nelder_mead(
  start: &mut [f64],
  bounds: &[(f64, f64)],
  objective: impl Fn(&[f64]) -> f64,
) {
  let n = start.len();
  if n == 0 {
    return;
  }
  let clamp = |p: &mut Vec<f64>| {
    for (v, &(lo, hi)) in p.iter_mut().zip(bounds) {
      *v = v.clamp(lo, hi);
    }
  };
  let mut best = start.to_vec();
  let mut best_val = objective(&best);
  for _ in 0..30 {
    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
    simplex.push((best.clone(), best_val));
    for j in 0..n {
      let mut p = best.clone();
      p[j] += 0.1 * p[j].abs().max(0.1);
      clamp(&mut p);
      let v = objective(&p);
      simplex.push((p, v));
    }
    for _ in 0..400 {
      simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
      let spread = simplex[n].1 - simplex[0].1;
      if spread.is_finite() && spread <= 1e-14 * simplex[0].1.abs().max(1e-300)
      {
        break;
      }
      let centroid: Vec<f64> = (0..n)
        .map(|j| simplex[..n].iter().map(|(p, _)| p[j]).sum::<f64>() / n as f64)
        .collect();
      let along = |t: f64| -> (Vec<f64>, f64) {
        let mut p: Vec<f64> = (0..n)
          .map(|j| centroid[j] + t * (simplex[n].0[j] - centroid[j]))
          .collect();
        clamp(&mut p);
        let v = objective(&p);
        (p, v)
      };
      let reflected = along(-1.0);
      if reflected.1 < simplex[0].1 {
        let expanded = along(-2.0);
        simplex[n] = if expanded.1 < reflected.1 {
          expanded
        } else {
          reflected
        };
      } else if reflected.1 < simplex[n - 1].1 {
        simplex[n] = reflected;
      } else {
        let contracted = if reflected.1 < simplex[n].1 {
          along(-0.5)
        } else {
          along(0.5)
        };
        if contracted.1 < simplex[n].1.min(reflected.1) {
          simplex[n] = contracted;
        } else {
          let anchor = simplex[0].0.clone();
          for item in simplex.iter_mut().skip(1) {
            let mut p: Vec<f64> = (0..n)
              .map(|j| anchor[j] + 0.5 * (item.0[j] - anchor[j]))
              .collect();
            clamp(&mut p);
            let v = objective(&p);
            *item = (p, v);
          }
        }
      }
    }
    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    if simplex[0].1 < best_val {
      best.clone_from(&simplex[0].0);
      best_val = simplex[0].1;
    } else {
      break;
    }
  }
  start.copy_from_slice(&best);
}

/// The fitting core: least squares by Gauss-Newton, optionally weighted
/// (each residual scaled by `sqrt(w)`, as `NonlinearModelFit`'s `Weights`
/// asks for) and optionally confined to the box the constraints describe.
fn find_fit_weighted(
  data_expr: &Expr,
  model: &Expr,
  constraints: &[Expr],
  params_expr: &Expr,
  var_expr: &Expr,
  weights: Option<&[f64]>,
  norm: Option<&Expr>,
  caller: &str,
) -> Result<Expr, InterpreterError> {
  // A parameter may be any expression (`sigma[1]`): each one that is not a
  // symbol stands in the model, the constraints and the parameter list as a
  // fresh symbol, and the fitted rules name it again at the end.
  let param_head = |item: &Expr| match item {
    Expr::List(inner) if !inner.is_empty() => inner[0].clone(),
    other => other.clone(),
  };
  let param_items: Vec<Expr> = match params_expr {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  let mut originals: Vec<(String, Expr)> = Vec::new();
  let mut model = model.clone();
  let mut constraints = constraints.to_vec();
  let mut renamed_items = Vec::with_capacity(param_items.len());
  for item in &param_items {
    let head = param_head(item);
    if matches!(head, Expr::Identifier(_))
      || crate::functions::predicate_ast::is_numeric_q(&head)
    {
      renamed_items.push(item.clone());
      continue;
    }
    let fresh = format!("FindFit$param{}", originals.len());
    let symbol = Expr::Identifier(fresh.clone());
    model =
      crate::functions::polynomial_ast::substitute_expr(&model, &head, &symbol);
    for c in &mut constraints {
      *c = crate::functions::polynomial_ast::substitute_expr(c, &head, &symbol);
    }
    renamed_items.push(match item {
      Expr::List(inner) => {
        let mut inner = inner.to_vec();
        inner[0] = symbol;
        Expr::List(inner.into())
      }
      _ => symbol,
    });
    originals.push((fresh, head));
  }
  let params_expr = match params_expr {
    Expr::List(_) => Expr::List(renamed_items.into()),
    _ => renamed_items.remove(0),
  };
  let constraints = &constraints[..];
  let args = [data_expr.clone(), model, params_expr, var_expr.clone()];
  let args = &args[..];
  let data_expr = &args[0];
  let model = &args[1];
  let params_expr = &args[2];
  let var_expr = &args[3];

  // The independent variable, or a list of them (`{x}`, `{x, y}`) for data
  // rows `{x1, …, xn, f}`. Any expression that is not a number can stand
  // for a variable — an `InterpolatingFunction` the model is written in,
  // say — and is replaced wherever it occurs.
  let vars: Vec<Expr> = match var_expr {
    Expr::List(items) if !items.is_empty() => items.to_vec(),
    other => vec![other.clone()],
  };
  if let Some(bad) = vars
    .iter()
    .find(|v| crate::functions::predicate_ast::is_numeric_q(v))
  {
    crate::emit_message(&format!(
      "General::ivar: {} is not a valid variable.",
      crate::syntax::format_expr(bad, crate::syntax::ExprForm::Output)
    ));
    return Ok(unevaluated(
      caller,
      &[
        data_expr.clone(),
        model.clone(),
        params_expr.clone(),
        var_expr.clone(),
      ],
    ));
  }
  let replace_var = |e: &Expr, var: &Expr, value: &Expr| -> Expr {
    match var {
      Expr::Identifier(name) => {
        crate::syntax::substitute_variable(e, name, value)
      }
      other => {
        crate::functions::polynomial_ast::substitute_expr(e, other, value)
      }
    }
  };

  // Extract parameter names
  let param_names: Vec<String> = match params_expr {
    Expr::List(items) => {
      let mut names = Vec::new();
      for item in items {
        match item {
          Expr::Identifier(name) => names.push(name.clone()),
          Expr::List(inner) if !inner.is_empty() => {
            // {param, init_value} form
            if let Expr::Identifier(name) = &inner[0] {
              names.push(name.clone());
            }
          }
          _ => {
            return Err(InterpreterError::EvaluationError(format!(
              "{caller}: parameters must be identifiers or {{name, init}} pairs"
            )));
          }
        }
      }
      names
    }
    Expr::Identifier(name) => vec![name.clone()],
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "{caller}: third argument must be a list of parameters"
      )));
    }
  };

  // Extract initial guesses
  let mut param_values: Vec<f64> = vec![1.0; param_names.len()];
  if let Expr::List(items) = params_expr {
    for (i, item) in items.iter().enumerate() {
      if let Expr::List(inner) = item
        && inner.len() >= 2
        && let Some(v) = try_eval_to_f64(&inner[1])
        && let Some(slot) = param_values.get_mut(i)
      {
        *slot = v;
      }
    }
  }

  // Extract data points
  let data_evaluated = evaluate_expr_to_expr(data_expr)?;
  let Expr::List(data_list) = &data_evaluated else {
    return Err(InterpreterError::EvaluationError(format!(
      "{caller}: first argument must be a list"
    )));
  };

  let (x_vals, y_vals) = extract_fit_data_multi(data_list, vars.len(), caller)?;
  let at_point = |x: &[f64]| -> Expr {
    let mut e = model.clone();
    for (var, v) in vars.iter().zip(x) {
      e = replace_var(&e, var, &Expr::Real(*v));
    }
    e
  };
  let n_data = x_vals.len();
  let n_params = param_names.len();
  let at_point_of = |e: &Expr, x: &[f64]| -> Expr {
    let mut e = e.clone();
    for (var, v) in vars.iter().zip(x) {
      e = replace_var(&e, var, &Expr::Real(*v));
    }
    e
  };
  let partials: Vec<Option<Expr>> = param_names
    .iter()
    .map(|p| {
      evaluate_expr_to_expr(&call(
        "D",
        vec![model.clone(), Expr::Identifier(p.clone())],
      ))
      .ok()
      // A derivative D could not take (a model defined only for numeric
      // arguments) is differenced numerically instead: substituting the
      // parameter values into an unevaluated `D[…, k]` makes `k` a number.
      .filter(|d| !crate::functions::math_ast::expr_contains_head(d, "D"))
    })
    .collect();

  // `Weights -> {w₁, …}` scales each residual by `sqrt(wᵢ)`, so a point
  // counts `wᵢ` times in the sum of squares.
  let scale: Vec<f64> = match weights {
    Some(w) if w.len() == n_data => {
      w.iter().map(|v| v.max(0.0).sqrt()).collect()
    }
    _ => vec![1.0; n_data],
  };
  // Constraints confine the parameters to a box, which the iteration
  // stays inside by projecting each step back into it.
  let bounds = fit_parameter_bounds(constraints, &param_names);
  let clamp = |values: &mut [f64]| {
    for (v, &(lo, hi)) in values.iter_mut().zip(bounds.iter()) {
      if lo.is_finite() || hi.is_finite() {
        *v = v.clamp(lo, hi);
      }
    }
  };
  clamp(&mut param_values);
  // A model linear in its parameters is a linear least-squares problem:
  // wolframscript solves it outright, so the starting values play no part
  // (they would offset the minimum-norm answer of an underdetermined one).
  let linear_model = bounds
    .iter()
    .all(|(lo, hi)| lo.is_infinite() && hi.is_infinite())
    && partials.iter().all(|d| {
      d.as_ref().is_some_and(|d| {
        param_names.iter().all(|p| {
          !crate::functions::polynomial_ast::solve::expr_contains_expr(
            d,
            &Expr::Identifier(p.clone()),
          )
        })
      })
    });
  if linear_model {
    param_values.fill(0.0);
  }

  // The model compiled over the data variables and the parameters: a fit
  // to thousands of points evaluates it, and its gradient in the
  // parameters, at every one of them on each iteration.
  let compiled_model: Option<crate::functions::polynomial_ast::solve::NumNode> =
    vars
      .iter()
      .map(|v| match v {
        Expr::Identifier(n) => Some(n.clone()),
        _ => None,
      })
      .collect::<Option<Vec<String>>>()
      .and_then(|names| {
        let all: Vec<String> = names
          .into_iter()
          .chain(param_names.iter().cloned())
          .collect();
        crate::functions::polynomial_ast::solve::compile_numeric(model, &all)
          .filter(|node| !node.has_fallback())
      });
  let n_vars = vars.len();
  let compiled_point = |i: usize, params: &[f64]| -> Vec<f64> {
    x_vals[i].iter().chain(params).copied().collect()
  };

  // Gauss-Newton iteration
  let max_iter = 100;
  let tol = 1e-12;

  for _ in 0..max_iter {
    // Evaluate model at each data point with current parameters
    let mut residuals = vec![0.0f64; n_data];
    let mut jacobian = vec![vec![0.0f64; n_params]; n_data];

    for i in 0..n_data {
      if let Some(node) = &compiled_model {
        let (val, grad) = node.eval_dual(&compiled_point(i, &param_values));
        residuals[i] = (y_vals[i] - val) * scale[i];
        for (k, d) in grad {
          if k >= n_vars {
            jacobian[i][k - n_vars] = d * scale[i];
          }
        }
        continue;
      }
      // Substitute variable and parameters
      let mut expr_at_x = at_point(&x_vals[i]);
      for (j, pname) in param_names.iter().enumerate() {
        expr_at_x = crate::syntax::substitute_variable(
          &expr_at_x,
          pname,
          &Expr::Real(param_values[j]),
        );
      }
      let val = match evaluate_expr_to_expr(&expr_at_x) {
        Ok(ref e) => try_eval_to_f64(e).unwrap_or(f64::NAN),
        Err(_) => f64::NAN,
      };
      residuals[i] = (y_vals[i] - val) * scale[i];

      // Jacobian from the symbolic partial derivatives, falling back to a
      // central difference where one does not evaluate to a number.
      let point = at_point(&x_vals[i]);
      let with_params = |e: &Expr, values: &[f64]| -> f64 {
        let mut e = e.clone();
        for (k, pname) in param_names.iter().enumerate() {
          e = crate::syntax::substitute_variable(
            &e,
            pname,
            &Expr::Real(values[k]),
          );
        }
        match evaluate_expr_to_expr(&e) {
          Ok(ref v) => try_eval_to_f64(v).unwrap_or(f64::NAN),
          Err(_) => f64::NAN,
        }
      };
      for j in 0..n_params {
        let exact = partials[j]
          .as_ref()
          .map(|d| with_params(&at_point_of(d, &x_vals[i]), &param_values))
          .filter(|v| v.is_finite());
        let slope = if let Some(v) = exact {
          v
        } else {
          let orig = param_values[j];
          let h = 1e-6 * orig.abs().max(1.0);
          param_values[j] = orig + h;
          let up = with_params(&point, &param_values);
          param_values[j] = orig - h;
          let down = with_params(&point, &param_values);
          param_values[j] = orig;
          (up - down) / (2.0 * h)
        };
        jacobian[i][j] = slope * scale[i];
      }
    }

    // Solve J^T J delta = J^T r using QR on J
    let Ok(mut delta) = least_squares_qr(&jacobian, &residuals, linear_model)
    else {
      break;
    };

    // A parameter the step would push out of its box is held at that
    // bound and the others re-fitted around it (an active set). Clamping
    // alone would leave the rest at whatever the unconstrained step
    // wanted, which is not the constrained best fit.
    let pinned: Vec<bool> = (0..n_params)
      .map(|j| {
        let (lo, hi) = bounds[j];
        (param_values[j] <= lo && delta[j] < 0.0)
          || (param_values[j] >= hi && delta[j] > 0.0)
      })
      .collect();
    if pinned.iter().any(|&p| p) && !pinned.iter().all(|&p| p) {
      let free: Vec<usize> = (0..n_params).filter(|&j| !pinned[j]).collect();
      let reduced: Vec<Vec<f64>> = jacobian
        .iter()
        .map(|row| free.iter().map(|&j| row[j]).collect())
        .collect();
      if let Ok(d) = least_squares_qr(&reduced, &residuals, linear_model) {
        delta = vec![0.0; n_params];
        for (k, &j) in free.iter().enumerate() {
          delta[j] = d[k];
        }
      }
    }

    // Update parameters, keeping them inside the constraint box. The
    // Gauss-Newton direction always lowers the sum of squares for a short
    // enough step, so the step is halved until it does: a full step from a
    // poor start (an exponential's rate, say) can overshoot into a region
    // the iteration never returns from.
    let before = param_values.clone();
    let current_ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let ssr_at = |values: &[f64]| -> f64 {
      let mut total = 0.0;
      for i in 0..n_data {
        if let Some(node) = &compiled_model {
          let r =
            (y_vals[i] - node.eval(&compiled_point(i, values))) * scale[i];
          total += r * r;
          continue;
        }
        let mut e = at_point(&x_vals[i]);
        for (k, pname) in param_names.iter().enumerate() {
          e = crate::syntax::substitute_variable(
            &e,
            pname,
            &Expr::Real(values[k]),
          );
        }
        let v = match evaluate_expr_to_expr(&e) {
          Ok(ref v) => try_eval_to_f64(v).unwrap_or(f64::NAN),
          Err(_) => f64::NAN,
        };
        let r = (y_vals[i] - v) * scale[i];
        total += r * r;
      }
      if total.is_nan() { f64::INFINITY } else { total }
    };
    let mut step = 1.0;
    for _ in 0..40 {
      for j in 0..n_params {
        param_values[j] = before[j] + step * delta[j];
      }
      clamp(&mut param_values);
      if !current_ssr.is_finite() || ssr_at(&param_values) <= current_ssr {
        break;
      }
      step *= 0.5;
    }
    let max_change = param_values
      .iter()
      .zip(before.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0f64, f64::max);

    if max_change < tol {
      break;
    }
  }

  // A `NormFunction` other than the 2-norm changes what is minimized, so
  // the least-squares solution only serves as the starting point of a
  // derivative-free search on the norm of the residual vector.
  if let Some(norm) = norm {
    let objective = |values: &[f64]| -> f64 {
      let mut residuals = Vec::with_capacity(n_data);
      for i in 0..n_data {
        let mut e = at_point(&x_vals[i]);
        for (k, pname) in param_names.iter().enumerate() {
          e = crate::syntax::substitute_variable(
            &e,
            pname,
            &Expr::Real(values[k]),
          );
        }
        let v = match evaluate_expr_to_expr(&e) {
          Ok(ref v) => try_eval_to_f64(v).unwrap_or(f64::NAN),
          Err(_) => f64::NAN,
        };
        residuals.push(Expr::Real((y_vals[i] - v) * scale[i]));
      }
      let applied =
        crate::evaluator::function_application::apply_function_to_arg(
          norm,
          &Expr::List(residuals.into()),
        );
      match applied.ok().as_ref().and_then(try_eval_to_f64) {
        Some(v) if v.is_finite() => v,
        _ => f64::INFINITY,
      }
    };
    minimize_nelder_mead(&mut param_values, &bounds, objective);
  }

  // Build result: {param1 -> val1, param2 -> val2, ...}
  let rules: Vec<Expr> = param_names
    .iter()
    .zip(param_values.iter())
    .map(|(name, val)| {
      let param = originals
        .iter()
        .find(|(fresh, _)| fresh == name)
        .map_or_else(|| Expr::Identifier(name.clone()), |(_, e)| e.clone());
      rule_expr(param, Expr::Real(*val))
    })
    .collect();

  Ok(Expr::List(rules.into()))
}

// ─── VectorAngle ────────────────────────────────────────────────────────

/// VectorAngle[u, v] — angle between two vectors.
/// Returns ArcCos[Dot[u,v] / (Norm[u] * Norm[v])].
pub fn vector_angle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("VectorAngle", args));
  }
  let u = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(unevaluated("VectorAngle", args));
    }
  };
  let v = match &args[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(unevaluated("VectorAngle", args));
    }
  };
  if u.len() != v.len() {
    return Ok(unevaluated("VectorAngle", args));
  }

  // Check for zero vectors — Norm[{0,...}] = 0 → Indeterminate
  let norm_u_expr = evaluate_expr_to_expr(&call1("Norm", args[0].clone()))?;
  let norm_v_expr = evaluate_expr_to_expr(&call1("Norm", args[1].clone()))?;
  if matches!(norm_u_expr, Expr::Integer(0))
    || matches!(norm_v_expr, Expr::Integer(0))
  {
    return Ok(id_expr("Indeterminate"));
  }

  // Build ArcCos[u.Conjugate[v] / (Norm[u] * Norm[v])] and evaluate. The
  // conjugate matters for complex vectors (VectorAngle[{1, I}, {1, -I}]
  // is Pi/2, not 0) and wolframscript keeps a genuinely complex cosine
  // symbolic (VectorAngle[{1, 0}, {I, 0}] = ArcCos[-I]). Real vectors
  // skip the wrapper so the symbolic path is unchanged.
  let complex =
    contains_imaginary_unit(&args[0]) || contains_imaginary_unit(&args[1]);
  let second = if complex {
    evaluate_expr_to_expr(&call1("Conjugate", args[1].clone()))?
  } else {
    args[1].clone()
  };
  let dot_expr = call("Dot", vec![args[0].clone(), second]);
  let norm_u = call1("Norm", args[0].clone());
  let norm_v = call1("Norm", args[1].clone());
  let denom = times(vec![norm_u, norm_v]);
  let ratio = times(vec![dot_expr, pow(denom, Expr::Integer(-1))]);
  let result = call1("ArcCos", ratio);
  evaluate_expr_to_expr(&result)
}

// ─── DihedralAngle ──────────────────────────────────────────────────────

/// DihedralAngle[{p1, p2}, {v1, v2}] — the dihedral angle along the edge
/// directed from p1 to p2, between the half-planes that contain the
/// directions v1 and v2.
///
/// It is the angle between the components of v1 and v2 perpendicular to the
/// edge direction e = p2 - p1, i.e. VectorAngle[v1 - Proj_e[v1], v2 - Proj_e[v2]].
/// Only defined for 3D vectors — Wolfram leaves all other dimensions
/// unevaluated.
pub fn dihedral_angle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DihedralAngle", args));
  if args.len() != 2 {
    return unevaluated();
  }
  // Each argument must be a pair {a, b}.
  let pair = |e: &Expr| -> Option<(Expr, Expr)> {
    match e {
      Expr::List(items) if items.len() == 2 => {
        Some((items[0].clone(), items[1].clone()))
      }
      _ => None,
    }
  };
  let Some((p1, p2)) = pair(&args[0]) else {
    return unevaluated();
  };
  let Some((v1, v2)) = pair(&args[1]) else {
    return unevaluated();
  };
  // All four entries must be length-3 vectors.
  let is_vec3 = |e: &Expr| matches!(e, Expr::List(items) if items.len() == 3);
  if ![&p1, &p2, &v1, &v2].iter().all(|e| is_vec3(e)) {
    return unevaluated();
  }
  // Edge direction e = p2 - p1.
  let edge =
    evaluate_expr_to_expr(&call("Subtract", vec![p2.clone(), p1.clone()]))?;
  // Component of v perpendicular to the edge: v - Projection[v, e].
  let perp = |v: &Expr| -> Result<Expr, InterpreterError> {
    let proj = call("Projection", vec![v.clone(), edge.clone()]);
    evaluate_expr_to_expr(&call("Subtract", vec![v.clone(), proj]))
  };
  let v1p = perp(&v1)?;
  let v2p = perp(&v2)?;
  vector_angle_ast(&[v1p, v2p])
}

/// SolidAngle[p, {u1, …, ud}] — the solid angle at point `p` spanned by the
/// vectors from `p` to the points `u_i`. In 2-D the two-vector form is the
/// planar angle (VectorAngle); in 3-D the three-vector form is the trihedral
/// solid angle via the Van Oosterom–Strackee formula
///   Ω = 2 ArcTan[den, |Det[{v1, v2, v3}]|]
/// with den = |v1||v2||v3| + (v1·v2)|v3| + (v1·v3)|v2| + (v2·v3)|v1|. A cone of
/// fewer vectors than the ambient dimension is degenerate (solid angle 0).
pub fn solid_angle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("SolidAngle", args));
  if args.len() != 2 {
    return unevaluated();
  }
  let Expr::List(p) = &args[0] else {
    return unevaluated();
  };
  let n = p.len();
  if n == 0 || !p.iter().all(crate::functions::predicate_ast::is_numeric_q) {
    return unevaluated();
  }
  let Expr::List(us) = &args[1] else {
    return unevaluated();
  };
  // Vectors from p to each u_i (each u_i an n-coordinate numeric point).
  let mut vecs: Vec<Expr> = Vec::with_capacity(us.len());
  for u in us {
    let Expr::List(uc) = u else {
      return unevaluated();
    };
    if uc.len() != n
      || !uc.iter().all(crate::functions::predicate_ast::is_numeric_q)
    {
      return unevaluated();
    }
    vecs.push(evaluate_expr_to_expr(&call(
      "Subtract",
      vec![u.clone(), args[0].clone()],
    ))?);
  }
  let d = vecs.len();
  // Full-dimensional cones only: a lower-dimensional wedge has zero measure.
  if d < n {
    return Ok(Expr::Integer(0));
  }
  if d > n {
    return unevaluated();
  }
  match n {
    // Planar angle between the two vectors.
    2 => vector_angle_ast(&[vecs[0].clone(), vecs[1].clone()]),
    // Trihedral solid angle (Van Oosterom–Strackee).
    3 => {
      let dot = |a: &Expr, b: &Expr| call("Dot", vec![a.clone(), b.clone()]);
      let norm = |a: &Expr| call1("Norm", a.clone());
      let (v1, v2, v3) = (&vecs[0], &vecs[1], &vecs[2]);
      let matrix = Expr::List(vec![v1.clone(), v2.clone(), v3.clone()].into());
      let num = call1("Abs", call1("Det", matrix));
      let den = plus(vec![
        times(vec![norm(v1), norm(v2), norm(v3)]),
        times(vec![dot(v1, v2), norm(v3)]),
        times(vec![dot(v1, v3), norm(v2)]),
        times(vec![dot(v2, v3), norm(v1)]),
      ]);
      evaluate_expr_to_expr(&times(vec![
        Expr::Integer(2),
        call("ArcTan", vec![den, num]),
      ]))
    }
    _ => unevaluated(),
  }
}

// ─── UpperTriangularize / LowerTriangularize ───────────────────────────

/// UpperTriangularize[m] or UpperTriangularize[m, k]
/// Zeros out elements below the k-th diagonal (default k=0).
pub fn upper_triangularize_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  Ok(triangularize_ast(args, true))
}

/// LowerTriangularize[m] or LowerTriangularize[m, k]
/// Zeros out elements above the k-th diagonal (default k=0).
pub fn lower_triangularize_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  Ok(triangularize_ast(args, false))
}

fn triangularize_ast(args: &[Expr], upper: bool) -> Expr {
  let name = if upper {
    "UpperTriangularize"
  } else {
    "LowerTriangularize"
  };

  if args.is_empty() || args.len() > 2 {
    return unevaluated(name, args);
  }

  let Expr::List(rows) = &args[0] else {
    return unevaluated(name, args);
  };

  let k = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(n) => *n,
      _ => {
        return unevaluated(name, args);
      }
    }
  } else {
    0
  };

  let nrows = rows.len();
  let mut result_rows = Vec::with_capacity(nrows);

  for (i, row) in rows.iter().enumerate() {
    match row {
      Expr::List(cols) => {
        let ncols = cols.len();
        let mut new_cols = Vec::with_capacity(ncols);
        for (j, elem) in cols.iter().enumerate() {
          let keep = if upper {
            // Keep if j >= i + k (at or above the k-th diagonal)
            (j as i128) >= (i as i128) + k
          } else {
            // Keep if j <= i + k (at or below the k-th diagonal)
            (j as i128) <= (i as i128) + k
          };
          if keep {
            new_cols.push(elem.clone());
          } else {
            new_cols.push(Expr::Integer(0));
          }
        }
        result_rows.push(Expr::List(new_cols.into()));
      }
      _ => {
        return unevaluated(name, args);
      }
    }
  }

  Expr::List(result_rows.into())
}

// ─── LinearModelFit ────────────────────────────────────────────────────

/// LinearModelFit[data, funs, var] — fits a linear model and returns a FittedModel object.
///
/// The FittedModel can be:
/// - Called as a function: `lm[x]` evaluates the model at x
/// - Queried for properties: `lm["BestFitParameters"]`, `lm["FitResiduals"]`, `lm["RSquared"]`
/// - Converted with Normal: `Normal[lm]` returns the fitted expression
pub fn linear_model_fit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() >= 3
    && let Some(rewritten) = with_row_fit_data(args)
  {
    let fitted = linear_model_fit_ast(&rewritten)?;
    return Ok(match fitted {
      Expr::FunctionCall { ref name, .. } if name == "LinearModelFit" => {
        unevaluated("LinearModelFit", args)
      }
      other => other,
    });
  }
  if args.len() < 3 {
    return linear_model_fit_core(args);
  }
  let unevaluated_with = |tag: &str, text: &str| {
    crate::emit_message(&format!("LinearModelFit::{tag}: {text}"));
    Ok(unevaluated("LinearModelFit", args))
  };
  let rows = match &args[0] {
    Expr::List(rows) => rows.as_ref(),
    _ => &[],
  };
  if rows.is_empty() {
    return unevaluated_with(
      "notdata",
      "The first argument is not a vector, a matrix or a list containing a \
       design matrix and response vector.",
    );
  }
  // A vector of responses is the matrix `{{1, y1}, {2, y2}, …}`; then every
  // row holding a `Missing[…]` is left out, as wolframscript does
  // (`{Missing[], 1, 2}` fits `{{2, 1}, {3, 2}}`).
  let is_missing = |e: &Expr| matches!(e, Expr::FunctionCall { name, .. } if name == "Missing");
  // One-element rows (`{{y1}, {y2}, …}`) are responses as well.
  let responses: Option<Vec<&Expr>> = rows
    .iter()
    .map(|r| match r {
      Expr::List(items) if items.len() == 1 => Some(&items[0]),
      Expr::List(_) => None,
      other => Some(other),
    })
    .collect();
  let rows: Vec<Expr> = if let Some(responses) = responses {
    responses
      .into_iter()
      .enumerate()
      .map(|(i, y)| {
        Expr::List(vec![Expr::Integer(i as i128 + 1), y.clone()].into())
      })
      .collect()
  } else {
    rows.to_vec()
  };
  let rows: Vec<Expr> = rows
    .into_iter()
    .filter(|row| match row {
      Expr::List(items) => !items.iter().any(is_missing),
      other => !is_missing(other),
    })
    .collect();
  let vars: Vec<Expr> = match &args[2] {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  if vars.iter().any(|v| try_eval_to_f64(v).is_some()) {
    return unevaluated_with(
      "ivar",
      &format!(
        "{} is not a valid variable.",
        crate::syntax::expr_to_message_form(&Expr::List(vars.clone().into()))
      ),
    );
  }
  if let Some(Expr::List(items)) = rows.iter().find(
    |row| matches!(row, Expr::List(items) if items.len() != vars.len() + 1),
  ) {
    return unevaluated_with(
      "fitc",
      &format!(
        "The number of coordinates ({}) is not equal to the number of \
         variables ({}).",
        items.len().saturating_sub(1),
        vars.len()
      ),
    );
  }
  let numeric = |e: &Expr| try_eval_to_f64(e).is_some();
  if rows.is_empty()
    || rows.iter().any(|row| match row {
      Expr::List(items) => !items.iter().all(numeric),
      other => !numeric(other),
    })
  {
    return unevaluated_with(
      "desmat",
      "Unable to construct a numeric design matrix. Nominal variables may \
       need to be specified, or non-numeric entries for numeric variables \
       may need to be replaced.",
    );
  }
  // A variable need not be a symbol (`Subscript[x, 1]`): each other one is
  // fitted as a placeholder symbol and put back into the model afterwards.
  let mut basis = args[1].clone();
  let mut placeholders: Vec<(String, Expr)> = Vec::new();
  let new_vars: Vec<Expr> = vars
    .iter()
    .enumerate()
    .map(|(k, v)| {
      if matches!(v, Expr::Identifier(_)) {
        return v.clone();
      }
      let name = format!("Woxi`LinearModelVariable{}", k + 1);
      basis = crate::functions::polynomial_ast::solve::substitute_expr(
        &basis,
        v,
        &Expr::Identifier(name.clone()),
      );
      placeholders.push((name.clone(), v.clone()));
      Expr::Identifier(name)
    })
    .collect();
  let mut core_args = vec![
    Expr::List(rows.into()),
    basis,
    if matches!(&args[2], Expr::List(_)) {
      Expr::List(new_vars.into())
    } else {
      new_vars
        .into_iter()
        .next()
        .unwrap_or(Expr::List(vec![].into()))
    },
  ];
  core_args.extend(args[3..].iter().cloned());
  let fitted = linear_model_fit_core(&core_args)?;
  if matches!(&fitted, Expr::FunctionCall { name, .. } if name == "LinearModelFit")
  {
    return Ok(unevaluated("LinearModelFit", args));
  }
  Ok(placeholders.iter().fold(fitted, |model, (name, var)| {
    crate::syntax::substitute_variable(&model, name, var)
  }))
}

fn linear_model_fit_core(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    return linear_model_fit_design_matrix_form(&args[0]);
  }
  if args.len() < 3 {
    return Err(InterpreterError::EvaluationError(
      "LinearModelFit expects 1 or 3 arguments".into(),
    ));
  }
  // Options after the three positional arguments. Only
  // `IncludeConstantBasis` changes the fit here; one that would change it in
  // a way not implemented (weights, nominal variables, …) leaves the call
  // unevaluated rather than fitting the wrong model.
  let mut include_constant = true;
  let mut weights: Option<(Vec<f64>, Expr)> = None;
  for opt in &args[3..] {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = opt
    else {
      return Ok(unevaluated("LinearModelFit", args));
    };
    match pattern.as_ref() {
      Expr::Identifier(n) if n == "IncludeConstantBasis" => {
        include_constant =
          !matches!(replacement.as_ref(), Expr::Identifier(b) if b == "False");
      }
      // `Weights -> {w1, …}`: weighted least squares, the error variance of
      // the i-th response taken to be σ²/wi. Anything but a list of
      // positive numbers (a weighting function, …) is left unevaluated.
      Expr::Identifier(n) if n == "Weights" => match replacement.as_ref() {
        Expr::Identifier(a) if a == "Automatic" => {}
        list @ Expr::List(items) => {
          match items
            .iter()
            .map(|w| try_eval_to_f64(w).filter(|w| *w > 0.0))
            .collect::<Option<Vec<f64>>>()
          {
            Some(w) => weights = Some((w, list.clone())),
            None => return Ok(unevaluated("LinearModelFit", args)),
          }
        }
        _ => return Ok(unevaluated("LinearModelFit", args)),
      },
      Expr::Identifier(n)
        if matches!(
          n.as_str(),
          "ConfidenceLevel"
            | "WorkingPrecision"
            | "Method"
            | "VarianceEstimatorFunction"
        ) => {}
      _ => return Ok(unevaluated("LinearModelFit", args)),
    }
  }
  let all_args = args;
  let args = &args[..3];

  // Extract basis functions. Wolfram's `LinearModelFit` prepends the
  // constant `1` (the `IncludeConstantBasis -> True` default), whether the
  // basis is given as a single expression (`x`) or a list
  // (`{Sin[x], Cos[y]}`); with `IncludeConstantBasis -> False` only an
  // explicit `1` in the basis is fitted.
  let user_basis: Vec<Expr> = match &args[1] {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  let mut basis: Vec<Expr> = if include_constant {
    vec![Expr::Integer(1)]
  } else {
    Vec::new()
  };
  for b in user_basis {
    if !(include_constant && matches!(b, Expr::Integer(1))) {
      basis.push(b);
    }
  }

  // Extract variable name(s). Accept either a single Identifier (uni­variate
  // case) or a List of Identifiers (multi-variate, e.g. `{x, y}`).
  let var_names: Vec<String> = match &args[2] {
    Expr::Identifier(name) => vec![name.clone()],
    Expr::List(items) => {
      let mut names = Vec::with_capacity(items.len());
      for item in items {
        match item {
          Expr::Identifier(n) => names.push(n.clone()),
          _ => {
            return Err(InterpreterError::EvaluationError(
              "LinearModelFit: variable list must contain only symbols".into(),
            ));
          }
        }
      }
      names
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "LinearModelFit: third argument must be a variable or list of variables"
          .into(),
      ));
    }
  };

  // Extract data points
  let data_list = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "LinearModelFit: first argument must be a non-empty list".into(),
      ));
    }
  };

  let (x_vals_multi, y_vals) =
    extract_fit_data_multi(data_list, var_names.len(), "LinearModelFit")?;
  let n = y_vals.len();
  let m = basis.len();
  // For univariate compatibility paths below, keep a flat x list.
  let x_vals: Vec<f64> = if var_names.len() == 1 {
    x_vals_multi.iter().map(|row| row[0]).collect()
  } else {
    Vec::new()
  };
  if weights.as_ref().is_some_and(|(w, _)| w.len() != n) {
    return Ok(unevaluated("LinearModelFit", all_args));
  }
  let weight_values: Vec<f64> = weights
    .as_ref()
    .map_or_else(|| vec![1.0; n], |(w, _)| w.clone());

  // Build design matrix A (n×m)
  let mut a_matrix = vec![vec![0.0f64; m]; n];
  for i in 0..n {
    for j in 0..m {
      let mut subst = basis[j].clone();
      for (k, name) in var_names.iter().enumerate() {
        subst = crate::syntax::substitute_variable(
          &subst,
          name,
          &Expr::Real(x_vals_multi[i][k]),
        );
      }
      let evaluated = evaluate_expr_to_expr(&subst)?;
      match try_eval_to_f64(&evaluated) {
        Some(v) => a_matrix[i][j] = v,
        None => {
          return Err(InterpreterError::EvaluationError(format!(
            "LinearModelFit: could not evaluate basis function {:?} at row {}",
            basis[j], i
          )));
        }
      }
    }
  }

  // Weighted least squares is ordinary least squares on the rows scaled
  // by the square roots of the weights; every statistic below (residual
  // and total sums of squares, parameter covariance, ANOVA) is computed on
  // the scaled problem, while the residuals and predictions are reported
  // on the original scale.
  let root_weights: Vec<f64> = weight_values.iter().map(|w| w.sqrt()).collect();
  let a_scaled: Vec<Vec<f64>> = a_matrix
    .iter()
    .zip(&root_weights)
    .map(|(row, s)| row.iter().map(|v| v * s).collect())
    .collect();
  let y_scaled: Vec<f64> = y_vals
    .iter()
    .zip(&root_weights)
    .map(|(y, s)| y * s)
    .collect();

  // Solve via QR decomposition
  let coeffs = solve_least_squares_qr(&a_scaled, &y_scaled)?;

  // Compute fitted values and residuals
  let mut fitted_values = Vec::with_capacity(n);
  let mut residuals = Vec::with_capacity(n);
  for i in 0..n {
    let y_hat: f64 = (0..m).map(|j| coeffs[j] * a_matrix[i][j]).sum();
    fitted_values.push(y_hat);
    residuals.push(y_vals[i] - y_hat);
  }

  // Compute R-squared (around the weighted mean of the responses)
  let weight_total: f64 = weight_values.iter().sum();
  let y_mean: f64 = y_vals
    .iter()
    .zip(&weight_values)
    .map(|(y, w)| w * y)
    .sum::<f64>()
    / weight_total;
  let ss_tot: f64 = y_vals
    .iter()
    .zip(&weight_values)
    .map(|(y, w)| w * (y - y_mean).powi(2))
    .sum();
  let ss_res: f64 = residuals
    .iter()
    .zip(&weight_values)
    .map(|(r, w)| w * r.powi(2))
    .sum();
  let r_squared = if ss_tot.abs() < 1e-30 {
    1.0
  } else {
    1.0 - ss_res / ss_tot
  };

  // Compute adjusted R-squared
  let adjusted_r_squared = if n > m && ss_tot.abs() > 1e-30 {
    1.0 - (ss_res / (n - m) as f64) / (ss_tot / (n - 1) as f64)
  } else {
    r_squared
  };

  // Per-parameter significance statistics ("ParameterTable"/
  // "ParameterTableEntries" and their constituent properties): the standard
  // OLS covariance `Cov[beta] = sigma^2 (X^T X)^{-1}`, whose diagonal gives
  // each coefficient's standard error, from which the t-statistic and its
  // two-tailed Student-t p-value follow. `None` when the degrees of freedom
  // are exhausted (`n <= m`) or `X^T X` is singular — the properties are
  // simply absent then, same as before this statistic existed.
  let parameter_stats: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = (n > m)
    .then(|| {
      let dof = (n - m) as f64;
      let sigma2 = ss_res / dof;
      let mut xtx = vec![vec![0.0f64; m]; m];
      for row in &a_scaled {
        for j in 0..m {
          for k in 0..m {
            xtx[j][k] += row[j] * row[k];
          }
        }
      }
      let xtx_inv = invert_square_matrix(&xtx)?;
      let mut errors = Vec::with_capacity(m);
      let mut t_stats = Vec::with_capacity(m);
      let mut p_values = Vec::with_capacity(m);
      for j in 0..m {
        let se = (sigma2 * xtx_inv[j][j]).max(0.0).sqrt();
        let t_stat = if se > 0.0 {
          coeffs[j] / se
        } else {
          f64::INFINITY
        };
        let p_value = student_t_two_tailed_p_value(dof, t_stat)?;
        errors.push(se);
        t_stats.push(t_stat);
        p_values.push(p_value);
      }
      Some((errors, t_stats, p_values))
    })
    .flatten();

  // ANOVA table ("ANOVATableEntries" and its constituent properties): one
  // row per basis function but the constant with its sequential sum of
  // squares (the drop in residual sum of squares as it joins the ones
  // before it) and F-test, then the error and total rows. A model with a
  // constant term is centered on the response mean (`n - 1` total degrees
  // of freedom); one without is uncentered. `None` when no error degrees
  // of freedom remain or a nested fit is singular.
  let anova: Option<AnovaTable> = (n > m)
    .then(|| {
      let constant = basis.iter().position(|b| matches!(b, Expr::Integer(1)));
      let (df_total, ss_total) = match constant {
        Some(_) => (n - 1, ss_tot),
        None => (n, y_scaled.iter().map(|y| y * y).sum::<f64>()),
      };
      let df_error = n - m;
      let ms_error = ss_res / df_error as f64;
      // Residual sum of squares of the least-squares fit on `columns`.
      let rss_of = |columns: &[usize]| -> Option<f64> {
        let xtx: Vec<Vec<f64>> = columns
          .iter()
          .map(|&i| {
            columns
              .iter()
              .map(|&j| a_scaled.iter().map(|r| r[i] * r[j]).sum())
              .collect()
          })
          .collect();
        let xty: Vec<f64> = columns
          .iter()
          .map(|&i| a_scaled.iter().zip(&y_scaled).map(|(r, y)| r[i] * y).sum())
          .collect();
        let beta = solve_linear_system(&xtx, &xty)?;
        Some(
          a_scaled
            .iter()
            .zip(&y_scaled)
            .map(|(r, y)| {
              let fit: f64 =
                columns.iter().zip(&beta).map(|(&c, b)| r[c] * b).sum();
              (y - fit).powi(2)
            })
            .sum(),
        )
      };
      let mut columns: Vec<usize> = constant.into_iter().collect();
      let mut previous_rss = ss_total;
      let mut terms = Vec::new();
      for j in (0..m).filter(|&j| Some(j) != constant) {
        columns.push(j);
        let rss = if columns.len() == m {
          ss_res
        } else {
          rss_of(&columns)?
        };
        let ss = previous_rss - rss;
        previous_rss = rss;
        let f_stat = ss / ms_error;
        let dist = call(
          "FRatioDistribution",
          vec![Expr::Integer(1), Expr::Integer(df_error as i128)],
        );
        let cdf =
          evaluate_expr_to_expr(&call("CDF", vec![dist, Expr::Real(f_stat)]))
            .ok()
            .and_then(|e| try_eval_to_f64(&e))?;
        terms.push(AnovaTerm {
          ss,
          f_stat,
          p_value: 1.0 - cdf,
        });
      }
      if terms.is_empty() {
        return None;
      }
      Some(AnovaTable {
        terms,
        df_error,
        ss_error: ss_res,
        ms_error,
        df_total,
        ss_total,
      })
    })
    .flatten();

  // Build the fitted expression: c[0]*basis[0] + c[1]*basis[1] + ...
  let mut terms = Vec::new();
  for (j, c) in coeffs.iter().enumerate() {
    let coeff_expr = Expr::Real(*c);
    if matches!(&basis[j], Expr::Integer(1)) {
      terms.push(coeff_expr);
    } else {
      terms.push(times(vec![coeff_expr, basis[j].clone()]));
    }
  }
  let fitted_expr = if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    // Run the assembled `Plus[c0, c1*basis1, c2*basis2, …]` through the
    // evaluator so it lands in canonical Plus order — wolframscript prints
    // `m["BestFit"]` and `m["Function"]` with the basis terms sorted by
    // primary function name (Cos before Sin, etc.), not in the input
    // basis order.
    let raw = plus(terms);
    evaluate_expr_to_expr(&raw).unwrap_or(raw)
  };

  // Build the pure-function form of the fitted expression: replace each
  // independent variable in turn with the matching Slot[k] and wrap in
  // `Function`. wolframscript exposes this as the `"Function"` property
  // (`0.186… + 0.779…*#1 & ` for the univariate case).
  let mut function_body = fitted_expr.clone();
  for (k, name) in var_names.iter().enumerate() {
    function_body = crate::syntax::substitute_variable(
      &function_body,
      name,
      &Expr::Slot(k + 1),
    );
  }
  let function_form = Expr::Function {
    body: Box::new(function_body),
  };

  // Build input data: each row is `{x1, …, xn, y}` (or `{x, y}` in the
  // univariate case).
  let input_data = Expr::List(
    x_vals_multi
      .iter()
      .zip(y_vals.iter())
      .map(|(xs, y)| {
        let mut row: Vec<Expr> = xs.iter().map(|v| Expr::Real(*v)).collect();
        row.push(Expr::Real(*y));
        Expr::List(row.into())
      })
      .collect(),
  );
  let _ = &x_vals; // silence "unused" warning in multi-var path

  // Build the response vector: the y-column of the input data, kept as
  // exact Integers when the input was integer-valued (matching
  // wolframscript's `m["Response"]` → `{1, 4, 3, 6}` for integer inputs).
  let response_vec = Expr::List(
    y_vals
      .iter()
      .map(|y| {
        if y.fract() == 0.0 && y.abs() < (i128::MAX as f64) {
          Expr::Integer(*y as i128)
        } else {
          Expr::Real(*y)
        }
      })
      .collect(),
  );

  // Build design matrix as Expr
  let design_matrix = Expr::List(
    a_matrix
      .iter()
      .map(|row| Expr::List(row.iter().map(|v| Expr::Real(*v)).collect()))
      .collect(),
  );

  // Build FittedModel association
  let mut assoc = vec![
    (str_expr("Type"), id_expr("Linear")),
    (str_expr("FittedExpression"), fitted_expr),
    (
      str_expr("BestFitParameters"),
      Expr::List(coeffs.iter().map(|c| Expr::Real(*c)).collect()),
    ),
    (
      str_expr("IndependentVariables"),
      Expr::List(
        var_names
          .iter()
          .map(|n| Expr::Identifier(n.clone()))
          .collect(),
      ),
    ),
    (str_expr("BasisFunctions"), Expr::List(basis.into())),
    (
      str_expr("FitResiduals"),
      Expr::List(residuals.iter().map(|r| Expr::Real(*r)).collect()),
    ),
    (
      str_expr("PredictedResponse"),
      Expr::List(fitted_values.iter().map(|v| Expr::Real(*v)).collect()),
    ),
    (str_expr("RSquared"), Expr::Real(r_squared)),
    (str_expr("AdjustedRSquared"), Expr::Real(adjusted_r_squared)),
    (str_expr("InputData"), input_data),
    (str_expr("DesignMatrix"), design_matrix),
    (str_expr("Function"), function_form),
    (str_expr("Response"), response_vec),
    (
      str_expr("VariableName"),
      Expr::String(var_names.first().cloned().unwrap_or_default()),
    ),
  ];
  if let Some((_, given)) = weights {
    assoc.push((str_expr("Weights"), given));
  }

  // No error degrees of freedom left: wolframscript divides the residual
  // sum of squares by zero. A nonzero one makes every standard error
  // ComplexInfinity, every t statistic 0. and every p-value 1.; an exactly
  // zero one makes them all Indeterminate.
  if n == m {
    let (se, t, p) = if ss_res == 0.0 {
      let indeterminate = id_expr("Indeterminate");
      (indeterminate.clone(), indeterminate.clone(), indeterminate)
    } else {
      (id_expr("ComplexInfinity"), Expr::Real(0.0), Expr::Real(1.0))
    };
    let column = |e: &Expr| Expr::List((0..m).map(|_| e.clone()).collect());
    let entries = Expr::List(
      coeffs
        .iter()
        .map(|c| {
          Expr::List(
            vec![Expr::Real(*c), se.clone(), t.clone(), p.clone()].into(),
          )
        })
        .collect(),
    );
    assoc.push((str_expr("ParameterErrors"), column(&se)));
    assoc.push((str_expr("ParameterTStatistics"), column(&t)));
    assoc.push((str_expr("ParameterPValues"), column(&p)));
    assoc.push((str_expr("ParameterTableEntries"), entries));
  }

  if let Some((errors, t_stats, p_values)) = parameter_stats {
    let entries = Expr::List(
      (0..m)
        .map(|j| {
          Expr::List(
            vec![
              Expr::Real(coeffs[j]),
              Expr::Real(errors[j]),
              Expr::Real(t_stats[j]),
              Expr::Real(p_values[j]),
            ]
            .into(),
          )
        })
        .collect(),
    );
    assoc.push((
      str_expr("ParameterErrors"),
      Expr::List(errors.iter().map(|e| Expr::Real(*e)).collect()),
    ));
    assoc.push((
      str_expr("ParameterTStatistics"),
      Expr::List(t_stats.iter().map(|t| Expr::Real(*t)).collect()),
    ));
    assoc.push((
      str_expr("ParameterPValues"),
      Expr::List(p_values.iter().map(|p| Expr::Real(*p)).collect()),
    ));
    assoc.push((str_expr("ParameterTableEntries"), entries));
  }

  if let Some(t) = anova {
    let int = |v: usize| Expr::Integer(v as i128);
    let reals =
      |vs: Vec<f64>| Expr::List(vs.into_iter().map(Expr::Real).collect());
    let row = |items: Vec<Expr>| Expr::List(items.into());
    let mut entries: Vec<Expr> = t
      .terms
      .iter()
      .map(|term| {
        row(vec![
          int(1),
          Expr::Real(term.ss),
          Expr::Real(term.ss),
          Expr::Real(term.f_stat),
          Expr::Real(term.p_value),
        ])
      })
      .collect();
    entries.push(row(vec![
      int(t.df_error),
      Expr::Real(t.ss_error),
      Expr::Real(t.ms_error),
    ]));
    entries.push(row(vec![int(t.df_total), Expr::Real(t.ss_total)]));
    let term_ss: Vec<f64> = t.terms.iter().map(|term| term.ss).collect();
    assoc.push((str_expr("ANOVATableEntries"), row(entries)));
    assoc.push((
      str_expr("ANOVATableDegreesOfFreedom"),
      row(
        t.terms
          .iter()
          .map(|_| int(1))
          .chain([int(t.df_error), int(t.df_total)])
          .collect(),
      ),
    ));
    assoc.push((
      str_expr("ANOVATableSumsOfSquares"),
      reals(
        term_ss
          .iter()
          .copied()
          .chain([t.ss_error, t.ss_total])
          .collect(),
      ),
    ));
    assoc.push((
      str_expr("ANOVATableMeanSquares"),
      reals(term_ss.iter().copied().chain([t.ms_error]).collect()),
    ));
    assoc.push((
      str_expr("ANOVATableFStatistics"),
      reals(t.terms.iter().map(|term| term.f_stat).collect()),
    ));
    assoc.push((
      str_expr("ANOVATablePValues"),
      reals(t.terms.iter().map(|term| term.p_value).collect()),
    ));
    assoc.push((str_expr("SequentialSumOfSquares"), reals(term_ss)));
  }

  Ok(call1("FittedModel", Expr::Association(assoc)))
}

/// The legacy `LinearRegression\`Regress[data, basis, vars]`: an ordinary
/// least-squares fit whose basis is taken literally (no implicit constant
/// term) and whose result is a list of `RegressionCommon\`` rules.
/// `ParameterTable` holds a `TableForm` of the estimate / standard error /
/// t-statistic / p-value rows labelled by the basis functions.
pub fn regress_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 {
    return Ok(unevaluated("LinearRegression`Regress", args));
  }
  let mut fit_args = args[..3].to_vec();
  fit_args.push(rule_expr(id_expr("IncludeConstantBasis"), id_expr("False")));
  let fit = linear_model_fit_ast(&fit_args)?;
  let Expr::FunctionCall {
    name,
    args: fit_parts,
  } = &fit
  else {
    return Ok(unevaluated("LinearRegression`Regress", args));
  };
  let (true, [Expr::Association(assoc)]) =
    (name == "FittedModel", &fit_parts[..])
  else {
    return Ok(unevaluated("LinearRegression`Regress", args));
  };
  let get = |key: &str| {
    assoc.iter().find_map(|(k, v)| match k {
      Expr::String(k) if k == key => Some(v.clone()),
      _ => None,
    })
  };
  let (Some(entries), Some(basis)) =
    (get("ParameterTableEntries"), get("BasisFunctions"))
  else {
    // Too few observations for parameter statistics.
    return Ok(unevaluated("LinearRegression`Regress", args));
  };
  let headings = Expr::List(
    ["Estimate", "SE", "TStat", "PValue"]
      .iter()
      .map(|h| str_expr(h))
      .collect(),
  );
  let table = call(
    "TableForm",
    vec![
      entries,
      rule_expr(
        id_expr("TableHeadings"),
        Expr::List(vec![basis, headings].into()),
      ),
    ],
  );
  let mut rules =
    vec![rule_expr(id_expr("RegressionCommon`ParameterTable"), table)];
  for (symbol, value) in [
    ("RSquared", get("RSquared")),
    ("AdjustedRSquared", get("AdjustedRSquared")),
    (
      "EstimatedVariance",
      derived_fit_property(assoc, "EstimatedVariance"),
    ),
  ] {
    if let Some(value) = value {
      rules.push(rule_expr(
        id_expr(&format!("RegressionCommon`{symbol}")),
        value,
      ));
    }
  }
  Ok(Expr::List(rules.into()))
}

/// The overall-significance ANOVA decomposition of a linear least-squares
/// fit: model/error/total degrees of freedom and sums of squares, the
/// model/error mean squares, and the F statistic with its p-value.
struct AnovaTable {
  terms: Vec<AnovaTerm>,
  df_error: usize,
  ss_error: f64,
  ms_error: f64,
  df_total: usize,
  ss_total: f64,
}

/// One basis function's row of a linear model's ANOVA table (one degree
/// of freedom, so its mean square is its sum of squares).
struct AnovaTerm {
  ss: f64,
  f_stat: f64,
  p_value: f64,
}

/// Invert a square matrix by solving `A x_j = e_j` for each standard basis
/// column via [`solve_linear_system`]. Returns `None` when `a` is singular.
fn invert_square_matrix(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
  let n = a.len();
  let mut cols = Vec::with_capacity(n);
  for j in 0..n {
    let mut e_j = vec![0.0f64; n];
    e_j[j] = 1.0;
    cols.push(solve_linear_system(a, &e_j)?);
  }
  // `cols[j]` is the inverse's j-th column; transpose into row-major form.
  Some(
    (0..n)
      .map(|i| (0..n).map(|j| cols[j][i]).collect())
      .collect(),
  )
}

/// Two-tailed p-value `2 (1 - CDF[StudentTDistribution[dof], |t|])` for a
/// t-statistic with `dof` degrees of freedom, delegating to the canonical
/// `StudentTDistribution`/`CDF` implementation rather than re-deriving the
/// incomplete-beta form here.
fn student_t_two_tailed_p_value(dof: f64, t_stat: f64) -> Option<f64> {
  let dist = call("StudentTDistribution", vec![Expr::Real(dof)]);
  let cdf_expr = call("CDF", vec![dist, Expr::Real(t_stat.abs())]);
  let evaluated = evaluate_expr_to_expr(&cdf_expr).ok()?;
  let cdf_val = try_eval_to_f64(&evaluated)?;
  Some(2.0 * (1.0 - cdf_val))
}

/// Asymptotic parameter statistics of a nonlinear least-squares fit:
/// `Cov = sigma^2 (J^T J)^-1` with `J` the model's Jacobian with respect to
/// the parameters (central differences) at the fitted values. Returns the
/// standard errors, t-statistics, two-tailed p-values and parameter values;
/// `None` when degrees of freedom are exhausted or `J^T J` is singular.
/// Standard errors, t statistics, p-values, estimates and the covariance
/// matrix of a nonlinear fit's parameters.
type NonlinearParameterStats =
  (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>);

fn nonlinear_parameter_stats(
  data: &Expr,
  model: &Expr,
  rules: &[Expr],
  var_name: &str,
) -> Option<NonlinearParameterStats> {
  let data_evaluated = evaluate_expr_to_expr(data).ok()?;
  let Expr::List(data_list) = &data_evaluated else {
    return None;
  };
  let (x_vals, y_vals) =
    extract_fit_data(data_list, "NonlinearModelFit").ok()?;
  let mut names = Vec::new();
  let mut values = Vec::new();
  for r in rules {
    let Expr::Rule {
      pattern,
      replacement,
    } = r
    else {
      return None;
    };
    let Expr::Identifier(n) = pattern.as_ref() else {
      return None;
    };
    names.push(n.clone());
    values.push(try_eval_to_f64(replacement)?);
  }
  let (n, m) = (x_vals.len(), values.len());
  if n <= m {
    return None;
  }
  let eval_at = |x: f64, vals: &[f64]| -> Option<f64> {
    let mut e =
      crate::syntax::substitute_variable(model, var_name, &Expr::Real(x));
    for (nm, v) in names.iter().zip(vals) {
      e = crate::syntax::substitute_variable(&e, nm, &Expr::Real(*v));
    }
    try_eval_to_f64(&evaluate_expr_to_expr(&e).ok()?)
  };
  // The Jacobian from the model's symbolic gradient at the estimate, or
  // by central differences when D leaves a derivative unevaluated.
  let rules_expr = Expr::List(rules.to_vec().into());
  let gradient: Option<Vec<Expr>> = names
    .iter()
    .map(|nm| {
      let d = evaluate_expr_to_expr(&call(
        "ReplaceAll",
        vec![
          call("D", vec![model.clone(), Expr::Identifier(nm.clone())]),
          rules_expr.clone(),
        ],
      ))
      .ok()?;
      (!expr_to_string(&d).contains("Derivative")).then_some(d)
    })
    .collect();
  let symbolic_jacobian = gradient.and_then(|g| {
    x_vals
      .iter()
      .map(|&x| {
        g.iter()
          .map(|d| {
            let at =
              crate::syntax::substitute_variable(d, var_name, &Expr::Real(x));
            try_eval_to_f64(&evaluate_expr_to_expr(&at).ok()?)
          })
          .collect::<Option<Vec<f64>>>()
      })
      .collect::<Option<Vec<Vec<f64>>>>()
  });
  let mut jac = vec![vec![0.0f64; m]; n];
  if let Some(symbolic) = symbolic_jacobian {
    jac = symbolic;
  } else {
    for j in 0..m {
      let h = 1e-6 * values[j].abs().max(1.0);
      let (mut up, mut down) = (values.clone(), values.clone());
      up[j] += h;
      down[j] -= h;
      for i in 0..n {
        jac[i][j] =
          (eval_at(x_vals[i], &up)? - eval_at(x_vals[i], &down)?) / (2.0 * h);
      }
    }
  }
  let mut ss_res = 0.0;
  for i in 0..n {
    ss_res += (y_vals[i] - eval_at(x_vals[i], &values)?).powi(2);
  }
  let dof = (n - m) as f64;
  let sigma2 = ss_res / dof;
  let mut jtj = vec![vec![0.0f64; m]; m];
  for row in &jac {
    for a in 0..m {
      for b in 0..m {
        jtj[a][b] += row[a] * row[b];
      }
    }
  }
  // A model whose parameters are not all identifiable (`Exp[a x/(b + c x)]`
  // only depends on the ratios) has a singular `Transpose[J] . J`;
  // wolframscript's covariance is then its pseudo-inverse. Rounding keeps
  // such a matrix invertible in name, so judge by the condition number.
  let max_abs = |m: &[Vec<f64>]| {
    m.iter()
      .flat_map(|row| row.iter())
      .fold(0.0f64, |acc, v| acc.max(v.abs()))
  };
  let inv = invert_square_matrix(&jtj)
    .filter(|inv| max_abs(&jtj) * max_abs(inv) < 1e12)
    .or_else(|| {
      let matrix = Expr::List(
        jtj
          .iter()
          .map(|row| Expr::List(row.iter().map(|v| Expr::Real(*v)).collect()))
          .collect(),
      );
      let pinv = pseudo_inverse_ast(&[matrix]).ok()?;
      let Expr::List(rows) = &pinv else {
        return None;
      };
      rows
        .iter()
        .map(|row| match row {
          Expr::List(items) => items.iter().map(try_eval_to_f64).collect(),
          _ => None,
        })
        .collect()
    })?;
  let covariance: Vec<Vec<f64>> = inv
    .iter()
    .map(|row| row.iter().map(|v| v * sigma2).collect())
    .collect();
  let (mut errors, mut ts, mut ps) = (vec![], vec![], vec![]);
  for j in 0..m {
    let se = (sigma2 * inv[j][j]).max(0.0).sqrt();
    let t = if se > 0.0 {
      values[j] / se
    } else {
      f64::INFINITY
    };
    ps.push(student_t_two_tailed_p_value(dof, t)?);
    errors.push(se);
    ts.push(t);
  }
  Some((errors, ts, ps, values, covariance))
}

/// NonlinearModelFit[data, model, params, var] — fits a (possibly nonlinear)
/// model and returns a FittedModel object. The parameters are fitted with
/// FindFit; the FittedModel then answers property queries ("BestFitParameters"
/// gives the fitted `{a -> …, b -> …}` rules, matching wolframscript) and
/// evaluates the fitted function at a point.
pub fn nonlinear_model_fit_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let uneval = || Ok(unevaluated("NonlinearModelFit", args));
  if args.len() < 4 {
    return uneval();
  }
  if let Some(rewritten) = with_row_fit_data(args) {
    let fitted = nonlinear_model_fit_ast(&rewritten)?;
    return match fitted {
      Expr::FunctionCall { ref name, .. } if name == "NonlinearModelFit" => {
        uneval()
      }
      other => Ok(other),
    };
  }
  let (data, spec, params, var) = (&args[0], &args[1], &args[2], &args[3]);
  // A constrained model is written `{model, cons…}`; the fitted function
  // that comes back is the model alone.
  let (model, constraints) = split_fit_model_spec(spec);
  let model = &model;

  // `Weights -> {w₁, …}` weights the data points; the other options are
  // accepted and ignored (they only steer how the minimum is sought).
  let mut weights: Option<Vec<f64>> = None;
  for opt in &args[4..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Weights")
      && let Ok(evaluated) = evaluate_expr_to_expr(replacement)
      && let Expr::List(items) = &evaluated
    {
      weights = items.iter().map(try_eval_to_f64).collect();
    }
  }

  // Fit the parameters. FindFit returns `{a -> v, b -> v}` (rules) on success.
  let param_rules = find_fit_weighted(
    data,
    model,
    &constraints,
    params,
    var,
    weights.as_deref(),
    None,
    "NonlinearModelFit",
  )?;
  let Expr::List(rules) = &param_rules else {
    return uneval();
  };
  if rules.is_empty() || !rules.iter().all(|r| matches!(r, Expr::Rule { .. })) {
    return uneval();
  }

  // The independent variable name (a bare symbol, or a single-element list).
  let var_name = match var {
    Expr::Identifier(n) => n.clone(),
    Expr::List(items) if items.len() == 1 => match &items[0] {
      Expr::Identifier(n) => n.clone(),
      _ => return uneval(),
    },
    _ => return uneval(),
  };

  // The fitted function: the model with the parameter rules substituted.
  let fitted_expr = evaluate_expr_to_expr(&call(
    "ReplaceAll",
    vec![model.clone(), param_rules.clone()],
  ))?;

  // "RSquared"/"AdjustedRSquared"/"FitResiduals"/"PredictedResponse" — the
  // same goodness-of-fit numbers `LinearModelFit` reports, computed here by
  // evaluating the fitted expression at each data point (nonlinear fits have
  // no design matrix to read residuals off directly).
  let data_evaluated = evaluate_expr_to_expr(data)?;
  let goodness_of_fit = match &data_evaluated {
    Expr::List(data_list) => extract_fit_data(data_list, "NonlinearModelFit")
      .ok()
      .map(|(x_vals, y_vals)| {
        let n = x_vals.len();
        let predicted: Vec<f64> = x_vals
          .iter()
          .map(|x| {
            let at_x = crate::syntax::substitute_variable(
              &fitted_expr,
              &var_name,
              &Expr::Real(*x),
            );
            evaluate_expr_to_expr(&at_x)
              .ok()
              .and_then(|e| try_eval_to_f64(&e))
              .unwrap_or(f64::NAN)
          })
          .collect();
        let residuals: Vec<f64> = y_vals
          .iter()
          .zip(&predicted)
          .map(|(y, y_hat)| y - y_hat)
          .collect();
        let y_mean: f64 = y_vals.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = y_vals.iter().map(|y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
        let r_squared = if ss_tot.abs() < 1e-30 {
          1.0
        } else {
          1.0 - ss_res / ss_tot
        };
        let n_params = rules.len();
        let adjusted_r_squared = if n > n_params && ss_tot.abs() > 1e-30 {
          1.0 - (ss_res / (n - n_params) as f64) / (ss_tot / (n - 1) as f64)
        } else {
          r_squared
        };
        (residuals, predicted, r_squared, adjusted_r_squared)
      }),
    _ => None,
  };

  let mut assoc = vec![
    (str_expr("Type"), id_expr("Nonlinear")),
    (str_expr("FittedExpression"), fitted_expr.clone()),
    (str_expr("BestFitParameters"), param_rules.clone()),
    (
      str_expr("IndependentVariables"),
      Expr::List(vec![Expr::Identifier(var_name.clone())].into()),
    ),
    (str_expr("VariableName"), Expr::String(var_name.clone())),
    (str_expr("BestFit"), fitted_expr),
    (str_expr("Model"), model.clone()),
  ];
  if let Some((residuals, predicted, r_squared, adjusted_r_squared)) =
    goodness_of_fit.clone()
  {
    assoc.push((
      str_expr("FitResiduals"),
      Expr::List(residuals.into_iter().map(Expr::Real).collect()),
    ));
    assoc.push((
      str_expr("PredictedResponse"),
      Expr::List(predicted.into_iter().map(Expr::Real).collect()),
    ));
    assoc.push((str_expr("RSquared"), Expr::Real(r_squared)));
    assoc.push((str_expr("AdjustedRSquared"), Expr::Real(adjusted_r_squared)));
  }

  // ANOVA table of a nonlinear fit: model/error/uncorrected-total/
  // corrected-total rows (no F-test, unlike the linear case). Needs error
  // degrees of freedom, as the parameter statistics below do.
  if let Some((residuals, predicted, ..)) = &goodness_of_fit {
    let (n, m) = (residuals.len(), rules.len());
    if n > m {
      let y: Vec<f64> = residuals
        .iter()
        .zip(predicted)
        .map(|(r, p)| r + p)
        .collect();
      let y_mean = y.iter().sum::<f64>() / n as f64;
      let ss_error: f64 = residuals.iter().map(|r| r * r).sum();
      let ss_uncorrected: f64 = y.iter().map(|v| v * v).sum();
      let ss_corrected: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
      let ss_model = ss_uncorrected - ss_error;
      let (ms_model, ms_error) =
        (ss_model / m as f64, ss_error / (n - m) as f64);
      let int = |v: usize| Expr::Integer(v as i128);
      let reals =
        |vs: &[f64]| Expr::List(vs.iter().map(|v| Expr::Real(*v)).collect());
      let row = |items: Vec<Expr>| Expr::List(items.into());
      assoc.push((
        str_expr("ANOVATableEntries"),
        row(vec![
          row(vec![int(m), Expr::Real(ss_model), Expr::Real(ms_model)]),
          row(vec![int(n - m), Expr::Real(ss_error), Expr::Real(ms_error)]),
          row(vec![int(n), Expr::Real(ss_uncorrected)]),
          row(vec![int(n - 1), Expr::Real(ss_corrected)]),
        ]),
      ));
      assoc.push((
        str_expr("ANOVATableDegreesOfFreedom"),
        row(vec![int(m), int(n - m), int(n), int(n - 1)]),
      ));
      assoc.push((
        str_expr("ANOVATableSumsOfSquares"),
        reals(&[ss_model, ss_error, ss_uncorrected, ss_corrected]),
      ));
      assoc.push((
        str_expr("ANOVATableMeanSquares"),
        reals(&[ms_model, ms_error]),
      ));
    }
  }

  if let Some((errors, t_stats, p_values, values, covariance)) =
    nonlinear_parameter_stats(data, model, rules, &var_name)
  {
    let entries = Expr::List(
      (0..values.len())
        .map(|j| {
          Expr::List(
            vec![
              Expr::Real(values[j]),
              Expr::Real(errors[j]),
              Expr::Real(t_stats[j]),
              Expr::Real(p_values[j]),
            ]
            .into(),
          )
        })
        .collect(),
    );
    let reals =
      |v: &[f64]| Expr::List(v.iter().map(|x| Expr::Real(*x)).collect());
    assoc.push((str_expr("ParameterErrors"), reals(&errors)));
    assoc.push((str_expr("ParameterTStatistics"), reals(&t_stats)));
    assoc.push((str_expr("ParameterPValues"), reals(&p_values)));
    assoc.push((str_expr("ParameterTableEntries"), entries));
    assoc.push((
      str_expr("CovarianceMatrix"),
      Expr::List(covariance.iter().map(|row| reals(row)).collect()),
    ));
  }

  Ok(call1("FittedModel", Expr::Association(assoc)))
}

/// `LinearModelFit[{X, y}]` — design-matrix form. Fits `y ≈ X · β` directly,
/// using each column of `X` as its own basis "function" (`Slot[1]`,
/// `Slot[2]`, …). Unlike the symbolic 3-arg form, no constant column is
/// auto-prepended — the caller is expected to include it in `X` already.
fn linear_model_fit_design_matrix_form(
  data: &Expr,
) -> Result<Expr, InterpreterError> {
  // Expect data = `{X, y}` where X is an n×k matrix and y is an n-vector.
  let (x_rows, y_items) = match data {
    Expr::List(items) if items.len() == 2 => (&items[0], &items[1]),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "LinearModelFit: 1-arg form expects {designMatrix, response}".into(),
      ));
    }
  };
  let x_matrix = match x_rows {
    Expr::List(rows) if !rows.is_empty() => rows,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "LinearModelFit: design matrix must be a non-empty list of rows".into(),
      ));
    }
  };
  let Expr::List(y_list) = y_items else {
    return Err(InterpreterError::EvaluationError(
      "LinearModelFit: response must be a list".into(),
    ));
  };
  if x_matrix.len() != y_list.len() {
    return Err(InterpreterError::EvaluationError(format!(
      "LinearModelFit: design matrix has {} rows but response has {} entries",
      x_matrix.len(),
      y_list.len()
    )));
  }
  // Convert X to f64 rows; require all rows to be lists with the same width.
  let n = x_matrix.len();
  let mut a_matrix: Vec<Vec<f64>> = Vec::with_capacity(n);
  let mut k: Option<usize> = None;
  for row in x_matrix {
    let Expr::List(row_items) = row else {
      return Err(InterpreterError::EvaluationError(
        "LinearModelFit: each design-matrix row must be a list".into(),
      ));
    };
    let width = row_items.len();
    match k {
      Some(prev) if prev != width => {
        return Err(InterpreterError::EvaluationError(
          "LinearModelFit: design-matrix rows must all have the same width"
            .into(),
        ));
      }
      None => k = Some(width),
      _ => {}
    }
    let mut row_f: Vec<f64> = Vec::with_capacity(width);
    for v in row_items {
      match try_eval_to_f64(v) {
        Some(x) => row_f.push(x),
        None => {
          return Err(InterpreterError::EvaluationError(
            "LinearModelFit: design-matrix entries must be numeric".into(),
          ));
        }
      }
    }
    a_matrix.push(row_f);
  }
  let m = k.unwrap_or(0);
  if m == 0 {
    return Err(InterpreterError::EvaluationError(
      "LinearModelFit: design matrix has zero columns".into(),
    ));
  }
  // Convert y to f64.
  let mut y_vals: Vec<f64> = Vec::with_capacity(n);
  for v in y_list {
    match try_eval_to_f64(v) {
      Some(y) => y_vals.push(y),
      None => {
        return Err(InterpreterError::EvaluationError(
          "LinearModelFit: response entries must be numeric".into(),
        ));
      }
    }
  }

  // Solve via QR (same path the 3-arg form uses).
  let coeffs = solve_least_squares_qr(&a_matrix, &y_vals)?;

  // Compute fitted values and residuals.
  let mut fitted_values = Vec::with_capacity(n);
  let mut residuals = Vec::with_capacity(n);
  for i in 0..n {
    let y_hat: f64 = (0..m).map(|j| coeffs[j] * a_matrix[i][j]).sum();
    fitted_values.push(y_hat);
    residuals.push(y_vals[i] - y_hat);
  }

  // R-squared / adjusted R-squared.
  let y_mean: f64 = y_vals.iter().sum::<f64>() / n as f64;
  let ss_tot: f64 = y_vals.iter().map(|y| (y - y_mean).powi(2)).sum();
  let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
  let r_squared = if ss_tot.abs() < 1e-30 {
    1.0
  } else {
    1.0 - ss_res / ss_tot
  };
  let adjusted_r_squared = if n > m && ss_tot.abs() > 1e-30 {
    1.0 - (ss_res / (n - m) as f64) / (ss_tot / (n - 1) as f64)
  } else {
    r_squared
  };

  // BasisFunctions = {#1, #2, …, #k} — one slot per column of X.
  let basis: Vec<Expr> = (1..=m).map(Expr::Slot).collect();

  // Build the fitted expression: c1*#1 + c2*#2 + … (no auto constant).
  let terms: Vec<Expr> = coeffs
    .iter()
    .enumerate()
    .map(|(j, c)| times(vec![Expr::Real(*c), basis[j].clone()]))
    .collect();
  let fitted_expr = if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    let raw = plus(terms);
    evaluate_expr_to_expr(&raw).unwrap_or(raw)
  };
  let function_form = Expr::Function {
    body: Box::new(fitted_expr.clone()),
  };

  // Reuse the design matrix and the y vector verbatim.
  let design_matrix = Expr::List(
    a_matrix
      .iter()
      .map(|row| Expr::List(row.iter().map(|v| Expr::Real(*v)).collect()))
      .collect(),
  );
  let response_vec = Expr::List(
    y_vals
      .iter()
      .map(|y| {
        if y.fract() == 0.0 && y.abs() < (i128::MAX as f64) {
          Expr::Integer(*y as i128)
        } else {
          Expr::Real(*y)
        }
      })
      .collect(),
  );
  let input_data =
    Expr::List(vec![design_matrix.clone(), response_vec.clone()].into());

  let assoc = Expr::Association(vec![
    (str_expr("Type"), id_expr("Linear")),
    (str_expr("FittedExpression"), fitted_expr),
    (
      str_expr("BestFitParameters"),
      Expr::List(coeffs.iter().map(|c| Expr::Real(*c)).collect()),
    ),
    (
      str_expr("IndependentVariables"),
      Expr::List(basis.clone().into()),
    ),
    (str_expr("BasisFunctions"), Expr::List(basis.into())),
    (
      str_expr("FitResiduals"),
      Expr::List(residuals.iter().map(|r| Expr::Real(*r)).collect()),
    ),
    (
      str_expr("PredictedResponse"),
      Expr::List(fitted_values.iter().map(|v| Expr::Real(*v)).collect()),
    ),
    (str_expr("RSquared"), Expr::Real(r_squared)),
    (str_expr("AdjustedRSquared"), Expr::Real(adjusted_r_squared)),
    (str_expr("InputData"), input_data),
    (str_expr("DesignMatrix"), design_matrix),
    (str_expr("Function"), function_form),
    (str_expr("Response"), response_vec),
    (str_expr("VariableName"), Expr::String(String::new())),
  ]);

  Ok(call1("FittedModel", assoc))
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
  let n = a.len();
  // Build augmented matrix
  let mut aug: Vec<Vec<f64>> = a
    .iter()
    .enumerate()
    .map(|(i, row)| {
      let mut r = row.clone();
      r.push(b[i]);
      r
    })
    .collect();

  for col in 0..n {
    // Partial pivoting
    let mut max_row = col;
    let mut max_val = aug[col][col].abs();
    for row in (col + 1)..n {
      if aug[row][col].abs() > max_val {
        max_val = aug[row][col].abs();
        max_row = row;
      }
    }
    if max_val < 1e-30 {
      return None;
    }
    aug.swap(col, max_row);

    let pivot = aug[col][col];
    for row in (col + 1)..n {
      let factor = aug[row][col] / pivot;
      for j in col..=n {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  // Back substitution
  let mut x = vec![0.0; n];
  for i in (0..n).rev() {
    let mut s = aug[i][n];
    for j in (i + 1)..n {
      s -= aug[i][j] * x[j];
    }
    x[i] = s / aug[i][i];
  }
  Some(x)
}

/// `fit["ParameterTable"]`: the estimates table as a styled `Grid` whose
/// first row holds the column headings and whose later rows hold a basis
/// function label followed by the entries of `"ParameterTableEntries"`.
fn fit_property<'a>(assoc: &'a [(Expr, Expr)], name: &str) -> Option<&'a Expr> {
  assoc.iter().find_map(|(k, v)| match k {
    Expr::String(k) if k == name => Some(v),
    _ => None,
  })
}

/// A fitted model's report table as wolframscript styles it: a heading
/// row, then one row per label holding that label and its entries.
fn fit_report_table(
  heading: &[&str],
  labels: Vec<Expr>,
  entries: &[Expr],
) -> Option<Expr> {
  if labels.len() != entries.len() {
    return None;
  }
  let mut rows = vec![Expr::List(
    heading
      .iter()
      .map(|h| Expr::String((*h).to_string()))
      .collect(),
  )];
  for (label, entry) in labels.into_iter().zip(entries) {
    let Expr::List(values) = entry else {
      return None;
    };
    let mut row = vec![label];
    row.extend(values.iter().cloned());
    rows.push(Expr::List(row.into()));
  }
  let rule = |name: &str, value: Expr| Expr::Rule {
    pattern: Box::new(Expr::Identifier(name.to_string())),
    replacement: Box::new(value),
  };
  let at_second = |value: Expr| {
    Expr::List(
      vec![Expr::Rule {
        pattern: Box::new(Expr::Integer(2)),
        replacement: Box::new(value),
      }]
      .into(),
    )
  };
  let gray = || call1("GrayLevel", Expr::Real(0.7));
  let grid = call(
    "Grid",
    vec![
      Expr::List(rows.into()),
      rule(
        "Alignment",
        Expr::List(vec![id_expr("Left"), id_expr("Automatic")].into()),
      ),
      rule(
        "Dividers",
        Expr::List(vec![at_second(gray()), at_second(gray())].into()),
      ),
      rule(
        "Spacings",
        Expr::List(
          vec![at_second(Expr::Integer(1)), at_second(Expr::Real(0.75))].into(),
        ),
      ),
    ],
  );
  Some(call(
    "Style",
    vec![grid, Expr::String("DialogStyle".to_string())],
  ))
}

/// The fitted parameter names: the basis functions of a linear model, the
/// parameter symbols of a nonlinear one.
fn fit_parameter_labels(assoc: &[(Expr, Expr)]) -> Option<Vec<Expr>> {
  if let Some(Expr::List(basis)) = fit_property(assoc, "BasisFunctions") {
    return Some(basis.to_vec());
  }
  match fit_property(assoc, "BestFitParameters")? {
    Expr::List(rules) => rules
      .iter()
      .map(|r| match r {
        Expr::Rule { pattern, .. } => Some(pattern.as_ref().clone()),
        _ => None,
      })
      .collect(),
    _ => None,
  }
}

/// The fitted parameter values, from either a list of values or rules.
fn fit_parameter_values(assoc: &[(Expr, Expr)]) -> Option<Vec<f64>> {
  match fit_property(assoc, "BestFitParameters")? {
    Expr::List(items) => items
      .iter()
      .map(|r| match r {
        Expr::Rule { replacement, .. } => try_eval_to_f64(replacement),
        other => try_eval_to_f64(other),
      })
      .collect(),
    _ => None,
  }
}

fn parameter_table(assoc: &[(Expr, Expr)]) -> Option<Expr> {
  let Expr::List(entries) = fit_property(assoc, "ParameterTableEntries")?
  else {
    return None;
  };
  let statistic = match fit_property(assoc, "Type") {
    Some(Expr::Identifier(t)) if t == "GeneralizedLinear" => {
      "z\u{2010}Statistic"
    }
    _ => "t\u{2010}Statistic",
  };
  fit_report_table(
    &[
      "",
      "Estimate",
      "Standard Error",
      statistic,
      "P\u{2010}Value",
    ],
    fit_parameter_labels(assoc)?,
    entries,
  )
}

/// `"ANOVATable"`: a linear model has one row per basis function but the
/// constant, then the error and total rows; a nonlinear one has model,
/// error and the uncorrected and corrected totals.
fn anova_table(assoc: &[(Expr, Expr)]) -> Option<Expr> {
  let Expr::List(entries) = fit_property(assoc, "ANOVATableEntries")? else {
    return None;
  };
  let word = |w: &str| Expr::String(w.to_string());
  match fit_property(assoc, "Type")? {
    Expr::Identifier(t) if t == "Nonlinear" => fit_report_table(
      &["", "DF", "SS", "MS"],
      vec![
        word("Model"),
        word("Error"),
        word("Uncorrected Total"),
        word("Corrected Total"),
      ],
      entries,
    ),
    Expr::Identifier(t) if t == "Linear" => {
      let Expr::List(basis) = fit_property(assoc, "BasisFunctions")? else {
        return None;
      };
      let mut labels: Vec<Expr> = basis
        .iter()
        .filter(|b| !matches!(b, Expr::Integer(1)))
        .cloned()
        .collect();
      labels.extend([word("Error"), word("Total")]);
      fit_report_table(
        &["", "DF", "SS", "MS", "F\u{2010}Statistic", "P\u{2010}Value"],
        labels,
        entries,
      )
    }
    _ => None,
  }
}

/// `"ParameterConfidenceRegion"`: the ellipsoid of parameter values whose
/// F statistic stays below its `level` quantile, with semi-axes along the
/// covariance matrix's eigenvectors.
fn parameter_confidence_region(
  assoc: &[(Expr, Expr)],
  level: f64,
) -> Option<Expr> {
  let center = fit_parameter_values(assoc)?;
  let covariance = fit_covariance_matrix(assoc)?;
  let p = center.len();
  let n = match fit_property(assoc, "FitResiduals")? {
    Expr::List(items) => items.len(),
    _ => return None,
  };
  if n <= p || covariance.len() != p {
    return None;
  }
  // The F(p, n - p) quantile for a least-squares fit; wolframscript takes
  // the ChiSquare(p) one in its place for a generalized linear model.
  let glm = matches!(
    fit_property(assoc, "Type"),
    Some(Expr::Identifier(t)) if t == "GeneralizedLinear"
  );
  let distribution = if glm {
    call1("ChiSquareDistribution", Expr::Integer(p as i128))
  } else {
    call(
      "FRatioDistribution",
      vec![Expr::Integer(p as i128), Expr::Integer((n - p) as i128)],
    )
  };
  let quantile = try_eval_to_f64(
    &evaluate_expr_to_expr(&call(
      "Quantile",
      vec![distribution, Expr::Real(level)],
    ))
    .ok()?,
  )?;
  let f_quantile = quantile;
  let matrix = Expr::List(
    covariance
      .iter()
      .map(|row| Expr::List(row.iter().map(|v| Expr::Real(*v)).collect()))
      .collect(),
  );
  let eigensystem =
    evaluate_expr_to_expr(&call1("Eigensystem", matrix)).ok()?;
  let Expr::List(system) = &eigensystem else {
    return None;
  };
  let [Expr::List(values), vectors @ Expr::List(_)] = system.as_slice() else {
    return None;
  };
  let radii: Vec<Expr> = values
    .iter()
    .map(|v| {
      try_eval_to_f64(v)
        .map(|v| Expr::Real((v.max(0.0) * p as f64 * f_quantile).sqrt()))
    })
    .collect::<Option<_>>()?;
  Some(call(
    "FittedModels`ParameterEllipsoid",
    vec![
      Expr::List(center.into_iter().map(Expr::Real).collect()),
      Expr::List(radii.into()),
      vectors.clone(),
    ],
  ))
}

/// The `ConfidenceLevel -> cl` option of a property query, 0.95 by default.
fn confidence_level(options: &[Expr]) -> f64 {
  options
    .iter()
    .find_map(|o| match o {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ConfidenceLevel") => {
        try_eval_to_f64(replacement)
      }
      _ => None,
    })
    .unwrap_or(0.95)
}

/// `"MeanPredictionBands"` / `"SinglePredictionBands"`: the fitted function
/// minus and plus the Student t quantile times the standard error of the
/// mean prediction, `Sqrt[g . C . g]` with `g` the model's gradient in the
/// parameters (the basis functions of a linear model) and `C` the
/// parameter covariance; a single prediction adds the estimated variance
/// under the root.
fn prediction_bands(
  assoc: &[(Expr, Expr)],
  single: bool,
  level: f64,
) -> Option<Expr> {
  let fitted = fit_property(assoc, "FittedExpression")?.clone();
  let covariance = fit_covariance_matrix(assoc)?;
  let p = covariance.len();
  let gradient: Vec<Expr> = match fit_property(assoc, "Type")? {
    Expr::Identifier(t) if t == "Linear" => {
      match fit_property(assoc, "BasisFunctions")? {
        Expr::List(basis) => basis.to_vec(),
        _ => return None,
      }
    }
    Expr::Identifier(t) if t == "Nonlinear" => {
      let model = fit_property(assoc, "Model")?;
      let Expr::List(rules) = fit_property(assoc, "BestFitParameters")? else {
        return None;
      };
      let rules_expr = Expr::List(rules.clone());
      rules
        .iter()
        .map(|r| {
          let Expr::Rule { pattern, .. } = r else {
            return None;
          };
          let derivative =
            call("D", vec![model.clone(), pattern.as_ref().clone()]);
          evaluate_expr_to_expr(&call(
            "ReplaceAll",
            vec![derivative, rules_expr.clone()],
          ))
          .ok()
        })
        .collect::<Option<_>>()?
    }
    _ => return None,
  };
  if gradient.len() != p {
    return None;
  }
  let n = match fit_property(assoc, "FitResiduals")? {
    Expr::List(items) => items.len(),
    _ => return None,
  };
  if n <= p {
    return None;
  }
  let residuals: Vec<f64> = match fit_property(assoc, "FitResiduals")? {
    Expr::List(items) => {
      items.iter().map(try_eval_to_f64).collect::<Option<_>>()?
    }
    _ => return None,
  };
  let variance =
    weighted_residual_sum_of_squares(assoc, &residuals)? / (n - p) as f64;
  let mut terms = Vec::new();
  if single {
    terms.push(Expr::Real(variance));
  }
  for i in 0..p {
    for j in i..p {
      let c = if i == j {
        covariance[i][i]
      } else {
        covariance[i][j] + covariance[j][i]
      };
      terms.push(times(vec![
        Expr::Real(c),
        gradient[i].clone(),
        gradient[j].clone(),
      ]));
    }
  }
  let t = try_eval_to_f64(
    &evaluate_expr_to_expr(&call(
      "Quantile",
      vec![
        call1("StudentTDistribution", Expr::Integer((n - p) as i128)),
        Expr::Real(f64::midpoint(1.0, level)),
      ],
    ))
    .ok()?,
  )?;
  let half_width = times(vec![Expr::Real(t), call1("Sqrt", plus(terms))]);
  evaluate_expr_to_expr(&Expr::List(
    vec![
      plus(vec![
        fitted.clone(),
        times(vec![Expr::Integer(-1), half_width.clone()]),
      ]),
      plus(vec![fitted, half_width]),
    ]
    .into(),
  ))
  .ok()
}

/// The `Weights` of a weighted fit with `n` data points, all 1 for an
/// unweighted one; `None` when they do not match the data.
fn fit_weights(assoc: &[(Expr, Expr)], n: usize) -> Option<Vec<f64>> {
  match fit_property(assoc, "Weights") {
    None => Some(vec![1.0; n]),
    Some(Expr::List(items)) if items.len() == n => {
      items.iter().map(try_eval_to_f64).collect()
    }
    Some(_) => None,
  }
}

/// The residual sum of squares of a fit, each squared residual times its
/// data point's weight.
fn weighted_residual_sum_of_squares(
  assoc: &[(Expr, Expr)],
  residuals: &[f64],
) -> Option<f64> {
  let weights = fit_weights(assoc, residuals.len())?;
  Some(residuals.iter().zip(&weights).map(|(r, w)| w * r * r).sum())
}

/// The parameter covariance matrix: stored by a nonlinear fit, the
/// estimated variance times `Inverse[Transpose[X] . W . X]` for a linear
/// one (`W` the diagonal matrix of the weights).
fn fit_covariance_matrix(assoc: &[(Expr, Expr)]) -> Option<Vec<Vec<f64>>> {
  let matrix_of = |e: &Expr| -> Option<Vec<Vec<f64>>> {
    let Expr::List(rows) = e else {
      return None;
    };
    rows
      .iter()
      .map(|row| match row {
        Expr::List(items) => items.iter().map(try_eval_to_f64).collect(),
        _ => None,
      })
      .collect()
  };
  if let Some(stored) = fit_property(assoc, "CovarianceMatrix") {
    return matrix_of(stored);
  }
  if !matches!(fit_property(assoc, "Type")?, Expr::Identifier(t) if t == "Linear")
  {
    return None;
  }
  let x = matrix_of(fit_property(assoc, "DesignMatrix")?)?;
  let residuals: Vec<f64> = match fit_property(assoc, "FitResiduals")? {
    Expr::List(items) => {
      items.iter().map(try_eval_to_f64).collect::<Option<_>>()?
    }
    _ => return None,
  };
  let (n, p) = (x.len(), x.first()?.len());
  if n <= p || residuals.len() != n {
    return None;
  }
  let weights = fit_weights(assoc, n)?;
  let variance =
    weighted_residual_sum_of_squares(assoc, &residuals)? / (n - p) as f64;
  let xtx: Vec<Vec<f64>> = (0..p)
    .map(|i| {
      (0..p)
        .map(|j| x.iter().zip(&weights).map(|(r, w)| w * r[i] * r[j]).sum())
        .collect()
    })
    .collect();
  let inverse = invert_square_matrix(&xtx)?;
  Some(
    inverse
      .into_iter()
      .map(|row| row.into_iter().map(|v| v * variance).collect())
      .collect(),
  )
}

/// Look up a single named property on a `FittedModel` association.
fn lookup_fitted_model_property(
  assoc: &[(Expr, Expr)],
  prop: &str,
  options: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unknown = || {
    InterpreterError::EvaluationError(format!(
      "FittedModel: unknown property \"{prop}\""
    ))
  };
  // The statistics of a fit without error degrees of freedom come from a
  // division by zero, which wolframscript reports on every query.
  if matches!(
    prop,
    "ParameterTable"
      | "ParameterTableEntries"
      | "ParameterErrors"
      | "ParameterTStatistics"
      | "ParameterPValues"
  ) && matches!(
    fit_property(assoc, "ParameterErrors"),
    Some(Expr::List(errors)) if errors.iter().any(|e| matches!(
      e,
      Expr::Identifier(n) if n == "ComplexInfinity" || n == "Indeterminate"
    ))
  ) && let Some(Expr::List(residuals)) = fit_property(assoc, "FitResiduals")
  {
    let rss: f64 = residuals
      .iter()
      .filter_map(try_eval_to_f64)
      .map(|r| r * r)
      .sum();
    let _ = evaluate_expr_to_expr(&call(
      "Times",
      vec![
        Expr::Real(rss),
        call("Power", vec![Expr::Integer(0), Expr::Integer(-1)]),
      ],
    ));
  }
  match prop {
    "ParameterTable" => return parameter_table(assoc).ok_or_else(unknown),
    "ANOVATable" => return anova_table(assoc).ok_or_else(unknown),
    "CovarianceMatrix" => {
      return fit_covariance_matrix(assoc)
        .map(|m| {
          Expr::List(
            m.into_iter()
              .map(|row| Expr::List(row.into_iter().map(Expr::Real).collect()))
              .collect(),
          )
        })
        .ok_or_else(unknown);
    }
    "MeanPredictionBands" | "SinglePredictionBands" => {
      let level = confidence_level(options);
      return prediction_bands(assoc, prop == "SinglePredictionBands", level)
        .ok_or_else(unknown);
    }
    "ParameterConfidenceRegion" => {
      let level = confidence_level(options);
      return parameter_confidence_region(assoc, level).ok_or_else(unknown);
    }
    _ => {}
  }
  // `"BestFit"` is wolframscript's name for the fitted expression itself
  // (mathematically: the polynomial of the basis terms with their best
  // coefficients). Internally we store that under `"FittedExpression"`.
  let lookup_key = if prop == "BestFit" {
    "FittedExpression"
  } else {
    prop
  };
  for (key, val) in assoc {
    if let Expr::String(k) = key
      && k == lookup_key
    {
      return Ok(val.clone());
    }
  }
  if let Some(value) = derived_fit_property(assoc, prop) {
    return Ok(value);
  }
  Err(InterpreterError::EvaluationError(format!(
    "FittedModel: unknown property \"{prop}\""
  )))
}

/// Diagnostics computed from the stored design matrix and residuals:
/// `EstimatedVariance` (residual sum of squares over the residual degrees
/// of freedom), `HatDiagonal` (leverages, the diagonal of
/// X (XᵀX)⁻¹ Xᵀ) and `CookDistances`.
fn derived_fit_property(assoc: &[(Expr, Expr)], prop: &str) -> Option<Expr> {
  let get = |name: &str| {
    assoc.iter().find_map(|(k, v)| match k {
      Expr::String(k) if k == name => Some(v),
      _ => None,
    })
  };
  let numbers = |e: &Expr| -> Option<Vec<f64>> {
    match e {
      Expr::List(items) => items
        .iter()
        .map(crate::functions::math_ast::try_eval_to_f64)
        .collect(),
      _ => None,
    }
  };
  let residuals = numbers(get("FitResiduals")?)?;
  let p = match get("BestFitParameters")? {
    Expr::List(items) => items.len(),
    _ => return None,
  };
  let n = residuals.len();
  if n <= p {
    return None;
  }
  let weights = fit_weights(assoc, n)?;
  let variance =
    weighted_residual_sum_of_squares(assoc, &residuals)? / (n - p) as f64;
  let leverages = || -> Option<Vec<f64>> {
    let Expr::List(rows) = get("DesignMatrix")? else {
      return None;
    };
    let x: Vec<Vec<f64>> = rows.iter().map(numbers).collect::<Option<_>>()?;
    if x.len() != n || x.iter().any(|row| row.len() != p) {
      return None;
    }
    // (XᵀWX)⁻¹ by Gauss–Jordan.
    let mut a: Vec<Vec<f64>> = (0..p)
      .map(|i| {
        let mut row: Vec<f64> = (0..p)
          .map(|j| x.iter().zip(&weights).map(|(r, w)| w * r[i] * r[j]).sum())
          .collect();
        row.extend((0..p).map(|j| if i == j { 1.0 } else { 0.0 }));
        row
      })
      .collect();
    for col in 0..p {
      let pivot = (col..p).max_by(|&i, &j| {
        a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap()
      })?;
      if a[pivot][col].abs() < 1e-300 {
        return None;
      }
      a.swap(col, pivot);
      let d = a[col][col];
      for v in &mut a[col] {
        *v /= d;
      }
      for r in 0..p {
        if r != col {
          let f = a[r][col];
          for c in 0..2 * p {
            a[r][c] -= f * a[col][c];
          }
        }
      }
    }
    Some(
      x.iter()
        .zip(&weights)
        .map(|(row, w)| {
          w * (0..p)
            .map(|i| (0..p).map(|j| row[i] * a[i][p + j] * row[j]).sum::<f64>())
            .sum::<f64>()
        })
        .collect(),
    )
  };
  let reals = |v: Vec<f64>| Expr::List(v.into_iter().map(Expr::Real).collect());
  match prop {
    "EstimatedVariance" => Some(Expr::Real(variance)),
    "HatDiagonal" => leverages().map(reals),
    "CookDistances" => {
      let h = leverages()?;
      Some(reals(
        residuals
          .iter()
          .zip(&h)
          .zip(&weights)
          .map(|((r, h), w)| {
            w * r * r / (p as f64 * variance) * h / ((1.0 - h) * (1.0 - h))
          })
          .collect(),
      ))
    }
    _ => None,
  }
}

/// Evaluate a FittedModel at a point or query a property.
/// Called when FittedModel[assoc][arg] is encountered.
pub fn evaluate_fitted_model(
  model_args: &[Expr],
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if model_args.len() != 1 || call_args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "FittedModel: invalid call".into(),
    ));
  }
  let stay_unevaluated = || {
    Ok(Expr::CurriedCall {
      func: Box::new(call("FittedModel", model_args.to_vec())),
      args: call_args.to_vec(),
    })
  };

  let Expr::Association(assoc) = &model_args[0] else {
    return Err(InterpreterError::EvaluationError(
      "FittedModel: expected association argument".into(),
    ));
  };

  // Check if the argument is a string (property query). A property the
  // model does not have is FittedModel::elmntav, as in wolframscript.
  if let Expr::String(prop) = &call_args[0] {
    return match lookup_fitted_model_property(assoc, prop, &call_args[1..]) {
      Err(InterpreterError::EvaluationError(msg))
        if msg.starts_with("FittedModel: unknown property") =>
      {
        crate::emit_message(&format!(
          "FittedModel::elmntav: \"{prop}\" is not an available property."
        ));
        Ok(Expr::CurriedCall {
          func: Box::new(call("FittedModel", model_args.to_vec())),
          args: call_args.to_vec(),
        })
      }
      other => other,
    };
  }

  // `fit[{prop1, prop2, ...}]` looks up each property and returns the list
  // of results, mirroring how a single string is handled above.
  if let Expr::List(props) = &call_args[0]
    && !props.is_empty()
    && props.iter().all(|p| matches!(p, Expr::String(_)))
  {
    let mut results = Vec::with_capacity(props.len());
    for p in props {
      let Expr::String(prop) = p else {
        unreachable!()
      };
      match lookup_fitted_model_property(assoc, prop, &call_args[1..]) {
        Err(InterpreterError::EvaluationError(msg))
          if msg.starts_with("FittedModel: unknown property") =>
        {
          crate::emit_message(&format!(
            "FittedModel::elmntav: \"{prop}\" is not an available property."
          ));
          return stay_unevaluated();
        }
        other => results.push(other?),
      }
    }
    return Ok(Expr::List(results.into()));
  }

  // Otherwise, evaluate the model at the given point: one value per
  // independent variable. A list of points is not threaded over, and a
  // wrong argument count is FittedModel::argrx, both left unevaluated as in
  // wolframscript.
  let mut fitted_expr = None;
  let mut variables: Option<Vec<Expr>> = None;
  for (key, val) in assoc {
    if let Expr::String(k) = key {
      match (k.as_str(), val) {
        ("FittedExpression", _) => fitted_expr = Some(val.clone()),
        ("IndependentVariables", Expr::List(vars)) => {
          variables = Some(vars.to_vec());
        }
        ("VariableName", Expr::String(v)) if variables.is_none() => {
          variables = Some(vec![Expr::Identifier(v.clone())]);
        }
        _ => {}
      }
    }
  }
  let fitted_expr = fitted_expr.ok_or_else(|| {
    InterpreterError::EvaluationError(
      "FittedModel: missing FittedExpression".into(),
    )
  })?;
  let variables = variables.ok_or_else(|| {
    InterpreterError::EvaluationError(
      "FittedModel: missing VariableName".into(),
    )
  })?;
  if call_args.iter().any(|a| matches!(a, Expr::List(_))) {
    return stay_unevaluated();
  }
  if call_args.len() != variables.len() {
    crate::emit_message(&format!(
      "FittedModel::argrx: FittedModel[\u{2026}] called with {} arguments; \
       {} arguments are expected.",
      call_args.len(),
      variables.len()
    ));
    return stay_unevaluated();
  }
  let substituted =
    variables
      .iter()
      .zip(call_args)
      .fold(fitted_expr, |e, (var, value)| match var {
        Expr::Identifier(name) => {
          crate::syntax::substitute_variable(&e, name, value)
        }
        other => crate::functions::polynomial_ast::solve::substitute_expr(
          &e, other, value,
        ),
      });
  evaluate_expr_to_expr(&substituted)
}

/// Extract the fitted expression from a FittedModel (for Normal[]).
pub fn fitted_model_normal(
  model_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if model_args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FittedModel: invalid structure".into(),
    ));
  }

  let Expr::Association(assoc) = &model_args[0] else {
    return Err(InterpreterError::EvaluationError(
      "FittedModel: expected association argument".into(),
    ));
  };

  for (key, val) in assoc {
    if let Expr::String(k) = key
      && k == "FittedExpression"
    {
      return Ok(val.clone());
    }
  }

  Err(InterpreterError::EvaluationError(
    "FittedModel: missing FittedExpression".into(),
  ))
}

/// QRDecomposition[m] - returns {Q^T, R} where m = Q^T . R
/// `CholeskyDecomposition[A]` — Cholesky factorization of a Hermitian
/// positive-definite matrix `A`, returning the upper-triangular `U` such
/// that `A == ConjugateTranspose[U].U`. Works symbolically over exact
/// numbers (radicals kept as `Sqrt[...]`) and numerically over reals.
pub fn cholesky_decomposition_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("CholeskyDecomposition", args);

  // Options: only TargetStructure is understood. For Cholesky, Automatic
  // means the plain dense matrix (unlike LDLDecomposition, whose default
  // is the structured form).
  #[derive(PartialEq)]
  enum CholeskyTarget {
    Dense,
    Structured,
    Sparse,
  }
  let mut target = CholeskyTarget::Dense;
  for extra in &args[1..] {
    let rules: Vec<&Expr> = match extra {
      Expr::List(items) => items.iter().collect(),
      other => vec![other],
    };
    for item in rules {
      let Some((lhs, rhs)) = rule_parts(item) else {
        crate::emit_message(&format!(
          "CholeskyDecomposition::nonopt: Options expected (instead of {}) beyond position 1 in {}. An option must be a rule or a list of rules.",
          expr_to_string(extra),
          expr_to_string(&unevaluated())
        ));
        return Ok(unevaluated());
      };
      match lhs {
        Expr::Identifier(name) if name == "TargetStructure" => {
          target = match rhs {
            Expr::String(s) if s == "Dense" => CholeskyTarget::Dense,
            Expr::String(s) if s == "Sparse" => CholeskyTarget::Sparse,
            Expr::String(s) if s == "Structured" => CholeskyTarget::Structured,
            Expr::Identifier(a) if a == "Automatic" => CholeskyTarget::Dense,
            other => {
              let shown = match other {
                Expr::String(s) => s.clone(),
                e => expr_to_string(e),
              };
              crate::emit_message(&format!(
                "CholeskyDecomposition::badts: {shown} is not a valid target structure."
              ));
              return Ok(unevaluated());
            }
          };
        }
        other => {
          crate::emit_message(&format!(
            "CholeskyDecomposition::optx: Unknown option {} in {}.",
            expr_to_string(other),
            expr_to_string(&unevaluated())
          ));
          return Ok(unevaluated());
        }
      }
    }
  }

  let Some(matrix) = expr_to_matrix(&args[0]) else {
    crate::emit_message(&format!(
      "CholeskyDecomposition::matsq: Argument {} at position 1 is not a nonempty square matrix.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  };

  let n = matrix.len();
  if n == 0 || matrix.iter().any(|row| row.len() != n) {
    crate::emit_message(&format!(
      "CholeskyDecomposition::matsq: Argument {} at position 1 is not a nonempty square matrix.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  }

  // U is upper triangular; A = U^* . U, so for j <= i:
  //   U[j][j] = Sqrt(A[j][j] - sum_{k<j} |U[k][j]|^2)
  //   U[j][i] = (A[j][i] - sum_{k<j} Conjugate[U[k][j]] U[k][i]) / U[j][j]
  // The conjugates matter for complex Hermitian input (previously U[k][j]^2
  // was used, giving Sqrt[5/2] instead of Sqrt[3/2] for {{2, I}, {-I, 2}});
  // for exact real entries Conjugate folds away, and for symbolic entries
  // wolframscript keeps the Conjugate wrapper too.
  // For a machine-precision (Real) matrix the strictly-lower zeros are the
  // machine zero `0.`, matching wolframscript; otherwise they are exact 0.
  let is_real = matrix
    .iter()
    .any(|row| row.iter().any(|e| matches!(e, Expr::Real(_))));
  let zero = if is_real {
    Expr::Real(0.0)
  } else {
    Expr::Integer(0)
  };
  let mut u = vec![vec![zero; n]; n];

  let conj = |e: &Expr| -> Result<Expr, InterpreterError> {
    eval_expr(&call1("Conjugate", e.clone()))
  };

  for j in 0..n {
    // Diagonal: radicand = A[j][j] - sum_{k<j} |U[k][j]|^2
    let mut radicand = matrix[j][j].clone();
    for k in 0..j {
      let sq = eval_mul(&conj(&u[k][j])?, &u[k][j]);
      radicand = eval_sub(&radicand, &sq);
    }
    let radicand = eval_expr(&radicand)?;

    // Reject non-positive-definite matrices: the diagonal radicand must
    // be a positive number.
    if let Some(v) = try_eval_to_f64(&radicand)
      && v <= 0.0
    {
      crate::emit_message(&format!(
        "CholeskyDecomposition::npdef: The matrix {} is not positive definite.",
        expr_to_string(&args[0])
      ));
      return Ok(unevaluated());
    }

    let diag = eval_expr(&make_sqrt(radicand))?;
    u[j][j] = diag.clone();

    // Off-diagonal entries in row j to the right of the diagonal.
    for i in (j + 1)..n {
      let mut numer = matrix[j][i].clone();
      for k in 0..j {
        let prod = eval_mul(&conj(&u[k][j])?, &u[k][i]);
        numer = eval_sub(&numer, &prod);
      }
      let entry = eval_expr(&div2(numer, diag.clone()))?;
      u[j][i] = entry;
    }
  }

  let rows: Vec<Vec<Expr>> = u
    .iter()
    .map(|row| row.iter().map(simplify_radical_factor).collect())
    .collect();
  Ok(match target {
    CholeskyTarget::Dense => Expr::List(
      rows
        .into_iter()
        .map(|r| Expr::List(r.into()))
        .collect::<Vec<_>>()
        .into(),
    ),
    CholeskyTarget::Structured => structured_matrix(
      "UpperTriangularMatrix",
      n,
      Expr::List(
        rows
          .into_iter()
          .map(|r| Expr::List(r.into()))
          .collect::<Vec<_>>()
          .into(),
      ),
    ),
    CholeskyTarget::Sparse => {
      let is_zero = |e: &Expr| {
        matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(v) if *v == 0.0)
      };
      let mut nonzeros: Vec<(usize, usize, Expr)> = Vec::new();
      for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
          if !is_zero(v) {
            nonzeros.push((i, j, v.clone()));
          }
        }
      }
      sparse_array_literal(n, &nonzeros)
    }
  })
}

/// Whether an expression mentions the imaginary unit in any spelling
/// (the constant/identifier `I` or a `Complex[...]` literal).
pub(crate) fn contains_imaginary_unit(e: &Expr) -> bool {
  match e {
    Expr::Constant(c) | Expr::Identifier(c) => c == "I",
    Expr::FunctionCall { name, args } => {
      name == "Complex" || args.iter().any(contains_imaginary_unit)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_imaginary_unit(left) || contains_imaginary_unit(right)
    }
    Expr::UnaryOp { operand, .. } => contains_imaginary_unit(operand),
    Expr::List(items) => items.iter().any(contains_imaginary_unit),
    _ => false,
  }
}

pub fn qr_decomposition_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Some(matrix) = expr_to_matrix(&args[0]) else {
    return Ok(unevaluated("QRDecomposition", args));
  };

  let n = matrix.len(); // rows
  if n == 0 {
    return Ok(Expr::List(
      vec![Expr::List(vec![].into()), Expr::List(vec![].into())].into(),
    ));
  }
  let m = matrix[0].len(); // cols
  // `QRDecomposition[m, Full]`: q square (rows × rows), r of m's shape.
  let full = args.len() == 2;

  if let Some(a) = machine_real_matrix(&matrix) {
    return Ok(householder_qr(a, full));
  }

  // Complex matrices need the Hermitian inner product Σ Conjugate[aₖ]·bₖ
  // (a plain dot product made ‖(1, I)‖² = 0 and produced ComplexInfinity
  // garbage) and the returned q rows are conjugated so that
  // ConjugateTranspose[q].r reconstructs the input. Real and symbolic
  // matrices keep the plain dot product (wolframscript's symbolic QR
  // carries Conjugate wrappers everywhere, which is out of scope here).
  let complex = matrix
    .iter()
    .any(|row| row.iter().any(contains_imaginary_unit));
  let conj = |e: &Expr| -> Result<Expr, InterpreterError> {
    eval_expr(&call1("Conjugate", e.clone()))
  };
  let dot = |a: &[Expr], b: &[Expr]| -> Result<Expr, InterpreterError> {
    if complex {
      let conj_a: Vec<Expr> =
        a.iter().map(&conj).collect::<Result<Vec<_>, _>>()?;
      eval_dot_product(&conj_a, b)
    } else {
      eval_dot_product(a, b)
    }
  };

  // Extract columns
  let mut cols: Vec<Vec<Expr>> = Vec::with_capacity(m);
  for j in 0..m {
    let mut col = Vec::with_capacity(n);
    for i in 0..n {
      col.push(matrix[i][j].clone());
    }
    cols.push(col);
  }

  // Modified Gram-Schmidt. A column in the span of the earlier ones
  // (orthogonalized to zero) adds no row to Q or R; an all-zero matrix
  // gives Q = {e1} and a zero row of R, as in wolframscript.
  let mut u: Vec<Vec<Expr>> = cols.clone();
  let mut q: Vec<Vec<Expr>> = Vec::with_capacity(n);
  let mut r_entries: Vec<Vec<Expr>> = Vec::with_capacity(n.min(m));

  for i in 0..m {
    if q.len() == n {
      break;
    }
    // Compute norm of u[i]
    let norm_sq = dot(&u[i], &u[i])?;
    if matches!(norm_sq, Expr::Integer(0)) {
      continue;
    }
    let norm = eval_expr(&make_sqrt(norm_sq.clone()))?;

    // R[row][i] = norm
    r_entries.push(vec![Expr::Integer(0); m]);
    let row = r_entries.len() - 1;
    r_entries[row][i] = norm.clone();

    // Q[i] = u[i] / norm
    let mut qi = Vec::with_capacity(n);
    for k in 0..n {
      qi.push(eval_expr(&div2(u[i][k].clone(), norm.clone()))?);
    }
    q.push(qi);

    // Orthogonalize the remaining columns against the UNNORMALIZED u[i]:
    // the projection coefficient (u_i·u_j)/(u_i·u_i) stays rational for
    // rational input, so every intermediate column stays rational and
    // radicals appear only in the final normalization. Projecting against
    // the normalized q_i instead mixes radical shapes (1/Sqrt[35] vs
    // Sqrt[5/7]) that neither woxi nor wolframscript re-combines, leaving
    // the later norms as unsimplified nested radicals.
    for j in (i + 1)..m {
      let unum = dot(&u[i], &u[j])?;
      // R[row][j] = q_i·u_j = (u_i·u_j)/‖u_i‖ — one exact division.
      r_entries[row][j] = eval_expr(&div2(unum.clone(), norm.clone()))?;
      let coeff = eval_expr(&div2(unum, norm_sq.clone()))?;
      for k in 0..n {
        u[j][k] = eval_expr(&minus2(
          u[j][k].clone(),
          times2(coeff.clone(), u[i][k].clone()),
        ))?;
      }
    }
  }

  if q.is_empty() {
    q.push((0..n).map(|k| Expr::Integer((k == 0) as i128)).collect());
    r_entries.push(vec![Expr::Integer(0); m]);
  }
  // Completing an exact q to a square one would pick a basis of the
  // orthogonal complement, which is not implemented: only a q that is
  // square already is the full decomposition.
  if full && q.len() != n {
    return Ok(unevaluated("QRDecomposition", args));
  }

  // Build Q^T (rows of Q^T = the qi vectors, conjugated for complex input
  // so that ConjugateTranspose[q].r == m).
  let q_rows: Vec<Vec<Expr>> = if complex {
    q.iter()
      .map(|row| row.iter().map(&conj).collect::<Result<Vec<_>, _>>())
      .collect::<Result<Vec<_>, _>>()?
  } else {
    q
  };
  let qt_expr = Expr::List(
    q_rows
      .iter()
      .map(|row| Expr::List(row.iter().map(simplify_radical_factor).collect()))
      .collect(),
  );

  // Build R
  let r_expr = Expr::List(
    r_entries
      .iter()
      .map(|row| Expr::List(row.iter().map(simplify_radical_factor).collect()))
      .collect(),
  );

  Ok(Expr::List(vec![qt_expr, r_expr].into()))
}

/// `{Q, R}` of a machine-real matrix by Householder reflections, LAPACK's
/// `dgeqr2` + `dorg2r`: each reflector maps its column onto
/// `-Sign[a_ii] ‖x‖ e_1` (none for an already-reduced column), and Q has
/// `Min[rows, cols]` orthonormal rows, rank-deficient matrices included.
/// `full` asks for all `rows` rows of Q (`dorg2r` with as many columns as
/// rows) and pads R with zero rows to the matrix's shape.
fn householder_qr(mut a: Vec<Vec<f64>>, full: bool) -> Expr {
  let rows = a.len();
  let cols = a[0].len();
  let k = rows.min(cols);
  // Columns of Q (rows of the returned q).
  let q_cols = if full { rows } else { k };
  let mut tau = vec![0.0; k];
  // Apply I - t v v' (v = (1, a[i+1..][i])) to a[i..][c0..cend].
  let reflect = |a: &mut Vec<Vec<f64>>,
                 v: &[f64],
                 t: f64,
                 i: usize,
                 c0: usize,
                 cend: usize| {
    if t == 0.0 {
      return;
    }
    for c in c0..cend {
      let w: f64 = (i..rows).map(|r| v[r - i] * a[r][c]).sum();
      for r in i..rows {
        a[r][c] -= t * v[r - i] * w;
      }
    }
  };
  for i in 0..k {
    let alpha = a[i][i];
    let xnorm = (i + 1..rows).map(|r| a[r][i] * a[r][i]).sum::<f64>().sqrt();
    if xnorm != 0.0 {
      let beta = -alpha.hypot(xnorm).copysign(alpha);
      tau[i] = (beta - alpha) / beta;
      let scale = 1.0 / (alpha - beta);
      for row in a.iter_mut().skip(i + 1) {
        row[i] *= scale;
      }
      a[i][i] = beta;
    }
    if i + 1 < cols {
      let v: Vec<f64> = std::iter::once(1.0)
        .chain((i + 1..rows).map(|r| a[r][i]))
        .collect();
      reflect(&mut a, &v, tau[i], i, i + 1, cols);
    }
  }
  let r: Vec<Vec<f64>> = (0..q_cols)
    .map(|i| {
      (0..cols)
        .map(|j| if i < k && j >= i { a[i][j] } else { 0.0 })
        .collect()
    })
    .collect();
  // dorg2r: accumulate Q (rows × q_cols) in place from the reflectors,
  // the columns past the k-th starting as unit vectors.
  let mut q: Vec<Vec<f64>> = (0..rows)
    .map(|r| {
      (0..q_cols)
        .map(|c| {
          if c >= k {
            (r == c) as u8 as f64
          } else if r > c {
            a[r][c]
          } else {
            0.0
          }
        })
        .collect()
    })
    .collect();
  for i in (0..k).rev() {
    if i + 1 < q_cols {
      q[i][i] = 1.0;
      let v: Vec<f64> = (i..rows).map(|r| q[r][i]).collect();
      reflect(&mut q, &v, tau[i], i, i + 1, q_cols);
    }
    for row in q.iter_mut().skip(i + 1) {
      row[i] *= -tau[i];
    }
    q[i][i] = 1.0 - tau[i];
    for row in q.iter_mut().take(i) {
      row[i] = 0.0;
    }
  }
  let to_expr = |m: Vec<Vec<f64>>| {
    Expr::List(
      m.into_iter()
        .map(|row| Expr::List(row.into_iter().map(Expr::Real).collect()))
        .collect(),
    )
  };
  let qt: Vec<Vec<f64>> = (0..q_cols)
    .map(|c| (0..rows).map(|r| q[r][c]).collect())
    .collect();
  Expr::List(vec![to_expr(qt), to_expr(r)].into())
}

/// Full singular value decomposition `{U, Σ, V}` (`A = U.Σ.V'`, U m×m,
/// Σ m×n, V n×n) of a machine-real matrix by one-sided Jacobi rotations,
/// singular values descending. Each right singular vector's first nonzero
/// component is made non-positive, as in the 2×2 case. `None` if it fails
/// to converge.
fn jacobi_svd(
  a: &[Vec<f64>],
) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
  // wolframscript's rank tolerance: 100 $MachineEpsilon relative to the
  // largest singular value, whatever the size.
  jacobi_svd_with_tolerance(a, 100.0 * f64::EPSILON)
}

/// `jacobi_svd` where singular values below `rel_tol` times the largest one
/// are set to zero.
fn jacobi_svd_with_tolerance(
  a: &[Vec<f64>],
  rel_tol: f64,
) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
  let m = a.len();
  let n = a.first()?.len();
  if n == 0 {
    return None;
  }
  if m < n {
    // A' = V Σ' U'.
    let at: Vec<Vec<f64>> =
      (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
    let (u, sigma, v) = jacobi_svd_with_tolerance(&at, rel_tol)?;
    let sigma_t = (0..m)
      .map(|i| (0..n).map(|j| sigma[j][i]).collect())
      .collect();
    return Some((v, sigma_t, u));
  }
  // Columns of `w` converge to U·Σ; `vt` accumulates the rotations.
  let mut w: Vec<Vec<f64>> =
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
  let mut vcols: Vec<Vec<f64>> = (0..n)
    .map(|j| (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect())
    .collect();
  let dot =
    |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
  let mut converged = false;
  // Two columns of rounding noise (what a rank-deficient matrix's extra
  // columns become) stay noise under rotation: their correlation is below
  // the noise floor of the largest column and is left alone.
  let noise_floor = {
    let largest = w.iter().map(|c| dot(c, c)).fold(0.0, f64::max);
    f64::EPSILON * f64::EPSILON * largest
  };
  for _sweep in 0..60 {
    let mut rotated = false;
    for p in 0..n {
      for q in p + 1..n {
        let alpha = dot(&w[p], &w[p]);
        let beta = dot(&w[q], &w[q]);
        let gamma = dot(&w[p], &w[q]);
        // Square roots taken apart: `alpha * beta` underflows to zero for
        // columns of ~1e-160 entries, and the pair then never converged.
        if gamma.abs() <= 1e-15 * alpha.sqrt() * beta.sqrt()
          || gamma.abs() <= noise_floor
        {
          continue;
        }
        rotated = true;
        let zeta = (beta - alpha) / (2.0 * gamma);
        let sign = if zeta >= 0.0 { 1.0 } else { -1.0 };
        let t = sign / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let sn = c * t;
        for k in 0..m {
          let (x, y) = (w[p][k], w[q][k]);
          w[p][k] = c * x - sn * y;
          w[q][k] = sn * x + c * y;
        }
        for k in 0..n {
          let (x, y) = (vcols[p][k], vcols[q][k]);
          vcols[p][k] = c * x - sn * y;
          vcols[q][k] = sn * x + c * y;
        }
      }
    }
    if !rotated {
      converged = true;
      break;
    }
  }
  if !converged {
    return None;
  }
  let mut order: Vec<usize> = (0..n).collect();
  let norms: Vec<f64> = w.iter().map(|c| dot(c, c).sqrt()).collect();
  order.sort_by(|&i, &j| norms[j].total_cmp(&norms[i]));
  let tol = norms.iter().copied().fold(0.0, f64::max) * rel_tol;
  let mut ucols: Vec<Vec<f64>> = Vec::with_capacity(m);
  let mut vout: Vec<Vec<f64>> = Vec::with_capacity(n);
  let mut sig: Vec<f64> = Vec::with_capacity(n);
  for &j in &order {
    let mut v = vcols[j].clone();
    let mut u = w[j].clone();
    let lead = v.iter().find(|x| x.abs() > 1e-15).copied().unwrap_or(0.0);
    if lead > 0.0 {
      for x in &mut v {
        *x = -*x;
      }
      for x in &mut u {
        *x = -*x;
      }
    }
    let s = norms[j];
    if s > tol {
      for x in &mut u {
        *x /= s;
      }
      ucols.push(u);
      sig.push(s);
    } else {
      sig.push(0.0);
    }
    vout.push(v);
  }
  // Complete U to an orthonormal basis of R^m (Gram-Schmidt on e_k).
  let mut k = 0;
  while ucols.len() < m && k < m {
    let mut e: Vec<f64> =
      (0..m).map(|i| if i == k { 1.0 } else { 0.0 }).collect();
    for _ in 0..2 {
      for u in &ucols {
        let d = dot(&e, u);
        e.iter_mut().zip(u).for_each(|(x, y)| *x -= d * y);
      }
    }
    let nrm = dot(&e, &e).sqrt();
    if nrm > 1e-8 {
      for x in &mut e {
        *x /= nrm;
      }
      let lead = e.iter().find(|x| x.abs() > 1e-15).copied().unwrap_or(0.0);
      if lead > 0.0 {
        for x in &mut e {
          *x = -*x;
        }
      }
      ucols.push(e);
    }
    k += 1;
  }
  let u = (0..m)
    .map(|i| ucols.iter().map(|c| c[i]).collect())
    .collect();
  let v = (0..n)
    .map(|i| vout.iter().map(|c| c[i]).collect())
    .collect();
  let sigma = (0..m)
    .map(|i| (0..n).map(|j| if i == j { sig[j] } else { 0.0 }).collect())
    .collect();
  Some((u, sigma, v))
}

/// `SingularValueDecomposition[A]` — closed-form 2×2 SVD for a numeric
/// matrix. Returns `{U, Σ, V}` with `A = U.Σ.V'`, matching wolframscript:
/// V's columns are the right singular vectors of A^T A, U's columns are
/// `A V / σ`, and each singular vector is normalised so its first
/// component is non-positive.
pub fn singular_value_decomposition_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(unevaluated("SingularValueDecomposition", args));
  }
  let Some(matrix) = expr_to_matrix(&args[0]) else {
    return Ok(unevaluated("SingularValueDecomposition", args));
  };
  let rows = matrix.len();
  {
    // A machine matrix is decomposed numerically (one-sided Jacobi);
    // exact matrices other than 2×2 stay unevaluated.
    if let Some(a) = machine_real_matrix(&matrix)
      && let Some((u, sigma, v)) = jacobi_svd(&a)
    {
      let to_expr = |m: Vec<Vec<f64>>| {
        Expr::List(
          m.into_iter()
            .map(|r| {
              Expr::List(
                r.into_iter().map(Expr::Real).collect::<Vec<_>>().into(),
              )
            })
            .collect::<Vec<_>>()
            .into(),
        )
      };
      return Ok(Expr::List(
        vec![to_expr(u), to_expr(sigma), to_expr(v)].into(),
      ));
    }
    if rows != 2 || matrix[0].len() != 2 {
      return Ok(unevaluated("SingularValueDecomposition", args));
    }
  }
  let to_f64 = |e: &Expr| -> Option<f64> {
    match e {
      Expr::Integer(n) => Some(*n as f64),
      Expr::Real(v) => Some(*v),
      _ => None,
    }
  };
  let a = to_f64(&matrix[0][0]);
  let b = to_f64(&matrix[0][1]);
  let c = to_f64(&matrix[1][0]);
  let d = to_f64(&matrix[1][1]);
  let (Some(a), Some(b), Some(c), Some(d)) = (a, b, c, d) else {
    return Ok(unevaluated("SingularValueDecomposition", args));
  };
  // A^T A = {{a^2 + c^2, ab + cd}, {ab + cd, b^2 + d^2}}.
  let m11 = a * a + c * c;
  let m12 = a * b + c * d;
  let m22 = b * b + d * d;
  let tr = m11 + m22;
  let det = m11 * m22 - m12 * m12;
  let disc = (tr * tr - 4.0 * det).max(0.0);
  let sq = disc.sqrt();
  let lambda1 = f64::midpoint(tr, sq);
  let lambda2 = ((tr - sq) / 2.0).max(0.0);
  let sigma1 = lambda1.sqrt();
  let sigma2 = lambda2.sqrt();
  // Right-singular vector for the larger eigenvalue. Solve
  // (m11 - λ) v_x + m12 v_y = 0.
  let (v1x_raw, v1y_raw) = if m12.abs() > 1e-15 {
    let r_unnorm = lambda1 - m11;
    let nrm = (m12 * m12 + r_unnorm * r_unnorm).sqrt();
    (m12 / nrm, r_unnorm / nrm)
  } else if m11 >= m22 {
    (1.0, 0.0)
  } else {
    (0.0, 1.0)
  };
  // The other singular vector is orthogonal to the first.
  let (v2x_raw, v2y_raw) = (-v1y_raw, v1x_raw);
  // Wolfram's convention: each singular vector's leading non-zero
  // component is non-positive. Negate when needed.
  let needs_flip = |x: f64, y: f64| -> bool {
    if x.abs() > 1e-15 { x > 0.0 } else { y > 0.0 }
  };
  let (v1x, v1y) = if needs_flip(v1x_raw, v1y_raw) {
    (-v1x_raw, -v1y_raw)
  } else {
    (v1x_raw, v1y_raw)
  };
  let (v2x, v2y) = if needs_flip(v2x_raw, v2y_raw) {
    (-v2x_raw, -v2y_raw)
  } else {
    (v2x_raw, v2y_raw)
  };
  // U_i = A v_i / σ_i. A (numerically) zero singular value carries no
  // direction, so that column of U is completed to an orthonormal basis.
  let tol = 1e-12 * sigma1;
  let (sigma1, sigma2) = (
    if sigma1 > tol.max(f64::MIN_POSITIVE) {
      sigma1
    } else {
      0.0
    },
    if sigma2 > tol { sigma2 } else { 0.0 },
  );
  let (u1x, u1y, u2x, u2y) = if sigma1 > 0.0 && sigma2 > 0.0 {
    (
      (a * v1x + b * v1y) / sigma1,
      (c * v1x + d * v1y) / sigma1,
      (a * v2x + b * v2y) / sigma2,
      (c * v2x + d * v2y) / sigma2,
    )
  } else if sigma1 > 0.0 {
    let (ux, uy) = ((a * v1x + b * v1y) / sigma1, (c * v1x + d * v1y) / sigma1);
    let (ox, oy) = (-uy, ux);
    let (ox, oy) = if needs_flip(ox, oy) {
      (-ox, -oy)
    } else {
      (ox, oy)
    };
    (ux, uy, ox, oy)
  } else {
    (1.0, 0.0, 0.0, 1.0)
  };
  let real = Expr::Real;
  // Build U with rows {U[1,1] U[1,2]} = {u1x, u2x} (columns are u1, u2).
  let u = Expr::List(
    vec![
      Expr::List(vec![real(u1x), real(u2x)].into()),
      Expr::List(vec![real(u1y), real(u2y)].into()),
    ]
    .into(),
  );
  let sigma = Expr::List(
    vec![
      Expr::List(vec![real(sigma1), real(0.0)].into()),
      Expr::List(vec![real(0.0), real(sigma2)].into()),
    ]
    .into(),
  );
  let v = Expr::List(
    vec![
      Expr::List(vec![real(v1x), real(v2x)].into()),
      Expr::List(vec![real(v1y), real(v2y)].into()),
    ]
    .into(),
  );
  Ok(Expr::List(vec![u, sigma, v].into()))
}

/// Simplify radical-bearing scalars produced by Gram-Schmidt so they
/// match Wolfram's canonical form:
///   `n / Sqrt[d]`             → `Sqrt[n^2/d]`           (when gcd(n^2, d) > 1)
///   `(n * Sqrt[a/b]) / d`     → `Sqrt[n^2*a / (d^2*b)]` (when reduction applies)
/// Both then run through `Sqrt[…]` so any leftover perfect-square factor is
/// extracted (e.g. `Sqrt[20/7]` → `2*Sqrt[5/7]`). Other expression shapes
/// pass through unchanged.
fn simplify_radical_factor(expr: &Expr) -> Expr {
  use crate::functions::math_ast::{make_rational, sqrt_ast, times_ast};
  let extract_int = |e: &Expr| -> Option<i128> {
    if let Expr::Integer(n) = e {
      Some(*n)
    } else {
      None
    }
  };
  // Pull factors out of Times (FunctionCall or BinaryOp) into a flat Vec.
  let mut factors: Vec<Expr> = Vec::new();
  let mut sign: i128 = 1;
  fn flatten(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args {
          flatten(a, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        flatten(left, out);
        flatten(right, out);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => {
        flatten(left, out);
        out.push(pow(right.as_ref().clone(), Expr::Integer(-1)));
      }
      _ => out.push(e.clone()),
    }
  }
  flatten(expr, &mut factors);
  if factors.is_empty() {
    return expr.clone();
  }
  // Collect numeric coefficient (Integer/Rational) and an optional Sqrt
  // argument (positive Integer or Rational). Reject the case if there
  // are non-numeric, non-radical factors left over.
  let mut coeff_num: i128 = 1;
  let mut coeff_den: i128 = 1;
  let mut sqrt_num: i128 = 1;
  let mut sqrt_den: i128 = 1;
  let mut had_sqrt = false;
  let mut other: Vec<Expr> = Vec::new();
  for f in &factors {
    if let Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } = f
    {
      sign = -sign;
      if let Some((n, d)) = expr_to_rational(operand) {
        coeff_num *= n;
        coeff_den *= d;
        continue;
      }
      other.push(*operand.clone());
      continue;
    }
    if let Some((n, d)) = expr_to_rational(f) {
      coeff_num *= n;
      coeff_den *= d;
      continue;
    }
    if let Expr::FunctionCall { name, args } = f
      && name == "Sqrt"
      && args.len() == 1
    {
      if let Some((n, d)) = expr_to_rational(&args[0])
        && n > 0
        && d > 0
      {
        if had_sqrt {
          return expr.clone();
        }
        had_sqrt = true;
        sqrt_num *= n;
        sqrt_den *= d;
        continue;
      }
      other.push(f.clone());
      continue;
    }
    // Generic Power[base, exp] handling — accepts both FunctionCall("Power", …)
    // and BinaryOp Power. Identifies base/exp uniformly.
    let power_parts: Option<(&Expr, &Expr)> = match f {
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        Some((&args[0], &args[1]))
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => Some((left.as_ref(), right.as_ref())),
      _ => None,
    };
    if let Some((base, exp)) = power_parts {
      let exp_kind: i32 = match exp {
        Expr::Integer(-1) => -2, // sentinel for "Power[base, -1]"
        Expr::Integer(_) => 0,
        Expr::FunctionCall { name: rn, args: ra }
          if rn == "Rational" && ra.len() == 2 =>
        {
          match (&ra[0], &ra[1]) {
            (Expr::Integer(1), Expr::Integer(2)) => 1, // Sqrt[base]
            (Expr::Integer(-1), Expr::Integer(2)) => -1, // 1/Sqrt[base]
            _ => 0,
          }
        }
        _ => 0,
      };
      // Power[base, ±1/2] with integer/Rational base.
      if (exp_kind == 1 || exp_kind == -1)
        && let Some((bn, bd)) = expr_to_rational(base)
        && bn > 0
        && bd > 0
      {
        if had_sqrt {
          return expr.clone();
        }
        had_sqrt = true;
        if exp_kind > 0 {
          sqrt_num *= bn;
          sqrt_den *= bd;
        } else {
          sqrt_num *= bd;
          sqrt_den *= bn;
        }
        continue;
      }
      // Power[base, -1]: either base is integer (1/n → folds into coeff_den)
      // or base is Sqrt[X]/Power[int, 1/2] (1/Sqrt[X] → swap into radical).
      if exp_kind == -2 {
        if let Some(b) = extract_int(base)
          && b > 0
        {
          coeff_den *= b;
          continue;
        }
        // 1/Sqrt[d] where Sqrt[d] is FunctionCall("Sqrt", [d])
        if let Expr::FunctionCall { name: sn, args: sa } = base
          && sn == "Sqrt"
          && sa.len() == 1
          && let Some((sn_n, sn_d)) = expr_to_rational(&sa[0])
          && sn_n > 0
          && sn_d > 0
        {
          if had_sqrt {
            return expr.clone();
          }
          had_sqrt = true;
          sqrt_num *= sn_d;
          sqrt_den *= sn_n;
          continue;
        }
        // 1/Power[base, 1/2] (Sqrt encoded as Power)
        let inner_power_parts: Option<(&Expr, &Expr)> = match base {
          Expr::FunctionCall { name, args }
            if name == "Power" && args.len() == 2 =>
          {
            Some((&args[0], &args[1]))
          }
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } => Some((left.as_ref(), right.as_ref())),
          _ => None,
        };
        if let Some((ibase, iexp)) = inner_power_parts
          && matches!(
            iexp,
            Expr::FunctionCall { name, args }
              if name == "Rational"
                && args.len() == 2
                && matches!((&args[0], &args[1]), (Expr::Integer(1), Expr::Integer(2)))
          )
          && let Some((bn, bd)) = expr_to_rational(ibase)
          && bn > 0
          && bd > 0
        {
          if had_sqrt {
            return expr.clone();
          }
          had_sqrt = true;
          sqrt_num *= bd;
          sqrt_den *= bn;
          continue;
        }
      }
    }
    other.push(f.clone());
  }
  if !other.is_empty() {
    return expr.clone();
  }
  // Combine coefficient and radical: result = sign * (coeff_num / coeff_den) * sqrt(sqrt_num/sqrt_den)
  // = sign * sqrt(coeff_num^2 * sqrt_num / (coeff_den^2 * sqrt_den))
  let Some(n2) = coeff_num.checked_mul(coeff_num) else {
    return expr.clone();
  };
  let Some(d2) = coeff_den.checked_mul(coeff_den) else {
    return expr.clone();
  };
  let Some(inner_num) = n2.checked_mul(sqrt_num) else {
    return expr.clone();
  };
  let Some(inner_den) = d2.checked_mul(sqrt_den) else {
    return expr.clone();
  };
  if inner_den == 0 {
    return expr.clone();
  }
  let g = gcd_i128(inner_num, inner_den);
  if g == 0 {
    return expr.clone();
  }
  // Only commit the rewrite when actual reduction happens — otherwise the
  // sqrt_ast pass would just re-extract the perfect-square factor back
  // into the coefficient (e.g. `4/Sqrt[35]` → `Sqrt[16/35]` → `4/Sqrt[35]`).
  if g <= 1 {
    return expr.clone();
  }
  let nn = inner_num / g;
  let dd = inner_den / g;
  let inner_expr = if dd == 1 {
    Expr::Integer(nn)
  } else {
    make_rational(nn, dd)
  };
  let Ok(simplified) = sqrt_ast(&[inner_expr]) else {
    return expr.clone();
  };
  if sign < 0 {
    match times_ast(&[Expr::Integer(-1), simplified]) {
      Ok(e) => e,
      Err(_) => expr.clone(),
    }
  } else {
    simplified
  }
}

/// Helper: evaluate a dot product of two vectors
fn eval_dot_product(a: &[Expr], b: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut sum = Expr::Integer(0);
  for (ai, bi) in a.iter().zip(b.iter()) {
    let prod = times2(ai.clone(), bi.clone());
    sum = plus2(sum, prod);
  }
  eval_expr(&sum)
}

/// Helper: evaluate an expression through the evaluator
fn eval_expr(e: &Expr) -> Result<Expr, InterpreterError> {
  evaluate_expr_to_expr(e)
}

// ─── PrincipalComponents ───────────────────────────────────────────────

/// PrincipalComponents[matrix] - principal component scores of a data matrix.
/// Each row is an observation, each column is a variable.
/// Returns centered data projected onto eigenvectors of the covariance matrix,
/// sorted by descending eigenvalue.
pub fn principal_components_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PrincipalComponents expects exactly 1 argument".into(),
    ));
  }
  let Some(data) = expr_to_matrix(&args[0]) else {
    return Ok(unevaluated("PrincipalComponents", args));
  };

  let nrows = data.len();
  if nrows == 0 {
    return Ok(Expr::List(vec![].into()));
  }
  let ncols = data[0].len();
  if ncols == 0 || data.iter().any(|r| r.len() != ncols) {
    return Err(InterpreterError::EvaluationError(
      "PrincipalComponents: argument must be a rectangular matrix".into(),
    ));
  }

  // Single row: centered data is all zeros
  if nrows == 1 {
    return Ok(Expr::List(
      vec![Expr::List(vec![Expr::Integer(0); ncols].into())].into(),
    ));
  }

  // 1. Compute column means
  let nrows_expr = Expr::Integer(nrows as i128);
  let mut col_means = Vec::with_capacity(ncols);
  for j in 0..ncols {
    let mut sum = Expr::Integer(0);
    for i in 0..nrows {
      sum = eval_add(&sum, &data[i][j]);
    }
    col_means.push(eval_divide(&sum, &nrows_expr));
  }

  // 2. Center the data
  let mut centered = vec![vec![Expr::Integer(0); ncols]; nrows];
  for i in 0..nrows {
    for j in 0..ncols {
      centered[i][j] = eval_sub(&data[i][j], &col_means[j]);
    }
  }

  // 3. Compute covariance matrix: Transpose[centered] . centered / (nrows - 1)
  let ct = transpose_matrix(&centered);
  let cov_unnorm = mat_mul(&ct, &centered);
  let denom = Expr::Integer((nrows as i128) - 1);
  let mut cov = vec![vec![Expr::Integer(0); ncols]; ncols];
  for i in 0..ncols {
    for j in 0..ncols {
      cov[i][j] = eval_divide(&cov_unnorm[i][j], &denom);
    }
  }

  // 4. Get eigenvectors of covariance matrix (sorted by descending eigenvalue)
  let cov_expr = matrix_to_expr(cov);
  let eigvecs = eigenvectors_ast(&[cov_expr])?;

  // 5. Normalize eigenvectors to unit length
  let Some(eigvec_matrix) = expr_to_matrix(&eigvecs) else {
    return Ok(unevaluated("PrincipalComponents", args));
  };

  let mut normalized_eigvecs = Vec::with_capacity(eigvec_matrix.len());
  for vec in &eigvec_matrix {
    // Compute norm = Sqrt[sum of squares]
    let mut norm_sq = Expr::Integer(0);
    for entry in vec {
      norm_sq = eval_add(&norm_sq, &eval_mul(entry, entry));
    }
    let norm_sq = simplify_expr(&norm_sq);
    // Normalize: divide each component by Sqrt[norm_sq]
    let norm = if matches!(&norm_sq, Expr::Integer(1)) {
      Expr::Integer(1)
    } else {
      eval_expr(&make_sqrt(norm_sq))?
    };
    let mut normalized: Vec<Expr> = vec
      .iter()
      .map(|e| simplify_expr(&eval_divide(e, &norm)))
      .collect();

    // Sign convention: negate eigenvector if dot product with first
    // centered observation is negative (matches Wolfram's convention)
    let mut dot = Expr::Integer(0);
    for (c, v) in centered[0].iter().zip(normalized.iter()) {
      dot = eval_add(&dot, &eval_mul(c, v));
    }
    let dot = simplify_expr(&dot);
    let should_negate = crate::functions::math_ast::try_eval_to_f64(&dot)
      .is_some_and(|v| v < 0.0);
    if should_negate {
      normalized = normalized
        .into_iter()
        .map(|e| eval_mul(&Expr::Integer(-1), &e))
        .collect();
    }

    normalized_eigvecs.push(normalized);
  }

  // 6. Project centered data onto normalized eigenvectors: centered . Transpose[eigvecs]
  let eigvecs_t = transpose_matrix(&normalized_eigvecs);
  let projected = mat_mul(&centered, &eigvecs_t);

  // Simplify each entry through the evaluator
  let mut result = Vec::with_capacity(nrows);
  for row in projected {
    let mut simplified_row = Vec::with_capacity(ncols);
    for entry in row {
      let simplified = eval_expr(&entry).unwrap_or(entry);
      simplified_row.push(simplified);
    }
    result.push(Expr::List(simplified_row.into()));
  }

  Ok(Expr::List(result.into()))
}

// ─── SmithDecomposition ────────────────────────────────────────────────

/// SmithDecomposition[matrix] - returns {U, S, V} where U.matrix.V = S
/// and S is the Smith normal form (diagonal with divisibility).
pub fn smith_decomposition_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::functions::math_ast::{bigint_to_expr, expr_to_bigint};
  use num_traits::{Signed, Zero};
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "SmithDecomposition expects exactly 1 argument".into(),
    ));
  }

  let Some(matrix) = expr_to_matrix(&args[0]) else {
    return Ok(unevaluated("SmithDecomposition", args));
  };

  // Entries are exact integers of any size: the elimination below runs on
  // BigInt because the transformation matrices grow quickly.
  let Some(int_matrix) = matrix
    .iter()
    .map(|row| row.iter().map(expr_to_bigint).collect::<Option<Vec<_>>>())
    .collect::<Option<Vec<Vec<BigInt>>>>()
  else {
    // Matches wolframscript: emit SmithDecomposition::latm when an
    // entry isn't a rational (integer-coefficient matrices satisfy
    // this trivially; non-integer rationals would otherwise need a
    // denominator-clearing step Woxi doesn't yet do).
    crate::emit_message(
      "SmithDecomposition::latm: Matrix contains an entry that is not rational.",
    );
    return Ok(unevaluated("SmithDecomposition", args));
  };

  let nrows = int_matrix.len();
  let ncols = if nrows > 0 { int_matrix[0].len() } else { 0 };
  if nrows == 0 || ncols == 0 {
    return Ok(unevaluated("SmithDecomposition", args));
  }

  let mut s: Vec<Vec<BigInt>> = int_matrix;
  let mut u: Vec<Vec<BigInt>> = identity_bigint(nrows);
  let mut v: Vec<Vec<BigInt>> = identity_bigint(ncols);

  let min_dim = nrows.min(ncols);

  for k in 0..min_dim {
    loop {
      // Find smallest nonzero absolute value in submatrix s[k..][k..]
      let mut pivot: Option<(usize, usize, BigInt)> = None;
      for (i, row) in s.iter().enumerate().skip(k) {
        for (j, x) in row.iter().enumerate().skip(k) {
          if !x.is_zero() && pivot.as_ref().is_none_or(|(_, _, p)| x.abs() < *p)
          {
            pivot = Some((i, j, x.abs()));
          }
        }
      }
      let Some((pivot_r, pivot_c, _)) = pivot else {
        break;
      };

      // Move pivot to (k, k)
      if pivot_r != k {
        s.swap(k, pivot_r);
        u.swap(k, pivot_r);
      }
      if pivot_c != k {
        for row in &mut s {
          row.swap(k, pivot_c);
        }
        for row in &mut v {
          row.swap(k, pivot_c);
        }
      }

      // Ensure positive
      if s[k][k].is_negative() {
        for x in &mut s[k] {
          *x = -&*x;
        }
        for x in &mut u[k] {
          *x = -&*x;
        }
      }

      // Eliminate column k
      let mut changed = false;
      for i in (k + 1)..nrows {
        if !s[i][k].is_zero() {
          let q = &s[i][k] / &s[k][k];
          for j in 0..ncols {
            let t = &q * &s[k][j];
            s[i][j] -= t;
          }
          for j in 0..nrows {
            let t = &q * &u[k][j];
            u[i][j] -= t;
          }
          if !s[i][k].is_zero() {
            changed = true;
          }
        }
      }

      // Eliminate row k
      for j in (k + 1)..ncols {
        if !s[k][j].is_zero() {
          let q = &s[k][j] / &s[k][k];
          for row in &mut s {
            let t = &q * &row[k];
            row[j] -= t;
          }
          for row in &mut v {
            let t = &q * &row[k];
            row[j] -= t;
          }
          if !s[k][j].is_zero() {
            changed = true;
          }
        }
      }

      if !changed {
        // Check divisibility
        let mut all_divide = true;
        'outer: for i in (k + 1)..nrows {
          for j in (k + 1)..ncols {
            if !s[i][j].is_zero() && !(&s[i][j] % &s[k][k]).is_zero() {
              all_divide = false;
              for jj in 0..ncols {
                let t = s[i][jj].clone();
                s[k][jj] += t;
              }
              for jj in 0..nrows {
                let t = u[i][jj].clone();
                u[k][jj] += t;
              }
              break 'outer;
            }
          }
        }
        if all_divide {
          break;
        }
      }
    }
  }

  let to_expr = |m: Vec<Vec<BigInt>>| {
    matrix_to_expr(
      m.into_iter()
        .map(|row| row.into_iter().map(bigint_to_expr).collect())
        .collect(),
    )
  };

  Ok(Expr::List(vec![to_expr(u), to_expr(s), to_expr(v)].into()))
}

fn identity_bigint(n: usize) -> Vec<Vec<BigInt>> {
  (0..n)
    .map(|i| (0..n).map(|j| BigInt::from(u8::from(i == j))).collect())
    .collect()
}

/// TensorWedge[v1, v2, ...] — exterior (wedge) product of vectors/tensors.
///
/// For k vectors of dimension n, produces a rank-k antisymmetric tensor.
/// TensorWedge[u, v]_{i,j} = u_i * v_j - u_j * v_i
pub fn tensor_wedge_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "TensorWedge expects at least 1 argument".into(),
    ));
  }

  // Single argument: TensorWedge[v] = v
  if args.len() == 1 {
    return Ok(args[0].clone());
  }

  // Extract vectors (each arg must be a List)
  let mut vectors: Vec<&[Expr]> = Vec::new();
  for arg in args {
    match arg {
      Expr::List(items) => vectors.push(items),
      _ => {
        return Ok(unevaluated("TensorWedge", args));
      }
    }
  }

  // All vectors must have the same dimension
  let n = vectors[0].len();
  for v in &vectors {
    if v.len() != n {
      return Err(InterpreterError::EvaluationError(
        "TensorWedge: all vectors must have the same dimension".into(),
      ));
    }
  }

  let k = vectors.len();
  if k > n {
    // Wedge product of more vectors than the dimension is zero
    return Ok(build_zero_tensor(n, k));
  }

  // Build the k-dimensional antisymmetric tensor
  // For indices (i1, i2, ..., ik), the value is:
  //   Sum over permutations σ of {0,...,k-1}: sign(σ) * v_{σ(0)}[i0] * v_{σ(1)}[i1] * ... * v_{σ(k-1)}[ik-1]
  let perms = permutations(k);
  build_wedge_tensor(&vectors, n, k, &perms, &[])
}

/// Recursively build the wedge product tensor.
fn build_wedge_tensor(
  vectors: &[&[Expr]],
  n: usize,
  k: usize,
  perms: &[(Vec<usize>, i32)],
  indices: &[usize],
) -> Result<Expr, InterpreterError> {
  if indices.len() == k {
    // Compute the element at this multi-index
    // Build Times[sign, v_{σ(0)}[i0], ..., v_{σ(k-1)}[ik-1]] for each perm
    // and sum them all using Plus[...]
    let mut terms = Vec::new();
    for (perm, sign) in perms {
      let mut factors: Vec<Expr> = vec![Expr::Integer(*sign as i128)];
      for (slot, &p_idx) in perm.iter().enumerate() {
        factors.push(vectors[p_idx][indices[slot]].clone());
      }
      let term = times(factors);
      terms.push(term);
    }
    let sum_expr = plus(terms);
    return evaluate_expr_to_expr(&sum_expr);
  }

  // Build the next level of the tensor
  let mut elements = Vec::with_capacity(n);
  for i in 0..n {
    let mut new_indices = indices.to_vec();
    new_indices.push(i);
    elements.push(build_wedge_tensor(vectors, n, k, perms, &new_indices)?);
  }
  Ok(Expr::List(elements.into()))
}

/// Build a zero tensor of shape n × n × ... × n (k times).
fn build_zero_tensor(n: usize, k: usize) -> Expr {
  if k == 0 {
    return Expr::Integer(0);
  }
  let inner = build_zero_tensor(n, k - 1);
  Expr::List(vec![inner; n].into())
}

/// Generate all permutations of {0, 1, ..., n-1} with their signs.
fn permutations(n: usize) -> Vec<(Vec<usize>, i32)> {
  let mut result = Vec::new();
  let mut perm: Vec<usize> = (0..n).collect();
  permutations_helper(&mut perm, 0, &mut result);
  result
}

fn permutations_helper(
  perm: &mut Vec<usize>,
  start: usize,
  result: &mut Vec<(Vec<usize>, i32)>,
) {
  let n = perm.len();
  if start == n {
    // Count inversions to determine sign
    let mut inversions = 0;
    for i in 0..n {
      for j in (i + 1)..n {
        if perm[i] > perm[j] {
          inversions += 1;
        }
      }
    }
    let sign = if inversions % 2 == 0 { 1 } else { -1 };
    result.push((perm.clone(), sign));
    return;
  }
  for i in start..n {
    perm.swap(start, i);
    permutations_helper(perm, start + 1, result);
    perm.swap(start, i);
  }
}

// ─── DesignMatrix ───────────────────────────────────────────────────

/// DesignMatrix[data, fns, vars] - constructs a design matrix
/// by evaluating basis functions at each data point.
/// Each row of data has predictor columns and a response (last column).
pub fn design_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "DesignMatrix expects 3 arguments".into(),
    ));
  }

  let Expr::List(data) = &args[0] else {
    return Err(InterpreterError::EvaluationError(
      "DesignMatrix: first argument must be a list of data points".into(),
    ));
  };

  // Parse basis functions - can be single function or list
  let fns: Vec<Expr> = match &args[1] {
    Expr::List(fs) => fs.to_vec(),
    single => vec![single.clone()],
  };

  // Parse variables - can be single variable or list
  let vars: Vec<Expr> = match &args[2] {
    Expr::List(vs) => vs.to_vec(),
    single => vec![single.clone()],
  };

  // Check if first function is already 1 (constant term)
  let has_constant = matches!(&fns[0], Expr::Integer(1));
  let basis_fns: Vec<Expr> = if has_constant {
    fns.clone()
  } else {
    let mut bfs = vec![Expr::Integer(1)];
    bfs.extend(fns.iter().cloned());
    bfs
  };

  let mut result_rows = Vec::new();
  for row_expr in data {
    let Expr::List(row) = row_expr else {
      return Err(InterpreterError::EvaluationError(
        "DesignMatrix: each data point must be a list".into(),
      ));
    };

    // Predictors are all columns except the last
    let predictors = &row[..row.len() - 1];

    let mut result_row = Vec::new();
    for func in &basis_fns {
      // Substitute each variable with its predictor value
      let mut expr = (*func).clone();
      for (var_idx, var) in vars.iter().enumerate() {
        if var_idx < predictors.len() {
          // Build: func /. var -> value and evaluate
          let rule = rule_expr(var.clone(), predictors[var_idx].clone());
          expr = call("ReplaceAll", vec![expr, rule]);
        }
      }
      let substituted = evaluate_expr_to_expr(&expr)?;
      let evaluated = evaluate_expr_to_expr(&substituted)?;
      result_row.push(evaluated);
    }
    result_rows.push(Expr::List(result_row.into()));
  }

  Ok(Expr::List(result_rows.into()))
}

/// SingularValueList[m] — the nonzero singular values of a numeric matrix,
/// sorted decreasing. SingularValueList[m, k] — the k largest (k < 0: the
/// |k| smallest). Exact input yields exact values (square roots of the
/// Gram-matrix eigenvalues); machine-precision input yields reals, with
/// near-zero values dropped relative to the largest singular value.
pub fn singular_value_list_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  singular_value_list_with_tolerance(args, None)
}

/// `SingularValueList[m]` / `SingularValueList[m, k]`, keeping the singular
/// values of a machine matrix down to `rel_tol` times the largest one (the
/// default tolerance when `None`).
pub fn singular_value_list_with_tolerance(
  args: &[Expr],
  rel_tol: Option<f64>,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("SingularValueList", args);
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated(args));
  }

  // Validate: rectangular, non-empty, numeric matrix
  let rows = match &args[0] {
    Expr::List(rows) if !rows.is_empty() => rows,
    _ => return Ok(unevaluated(args)),
  };
  let ncols = match &rows[0] {
    Expr::List(cols) if !cols.is_empty() => cols.len(),
    _ => return Ok(unevaluated(args)),
  };
  let mut has_real = false;
  for row in rows {
    let cols = match row {
      Expr::List(cols) if cols.len() == ncols => cols,
      _ => return Ok(unevaluated(args)),
    };
    for entry in cols {
      if try_eval_to_f64(entry).is_none() {
        return Ok(unevaluated(args));
      }
      if matches!(entry, Expr::Real(_)) {
        has_real = true;
      }
    }
  }
  let nrows = rows.len();

  // Optional second argument: how many singular values to keep
  let take = match args.get(1) {
    None => None,
    Some(k) => match crate::functions::math_ast::expr_to_i128(k) {
      Some(k) => Some(k),
      None => return Ok(unevaluated(args)),
    },
  };

  // The first k values, or the |k| smallest for a negative k.
  let take_values = |mut values: Vec<Expr>| match take {
    None => Expr::List(values.into()),
    Some(k) if k >= 0 => {
      values.truncate(k as usize);
      Expr::List(values.into())
    }
    Some(k) => {
      let keep = (-k) as usize;
      let skip = values.len().saturating_sub(keep);
      Expr::List(values.split_off(skip).into())
    }
  };

  // A machine matrix: the nonzero singular values of its SVD, which keeps
  // the small ones a Gram matrix's eigenvalues cannot resolve (`3.*^-14`
  // squares to below their rounding).
  let matrix: Option<Vec<Vec<Expr>>> = rows
    .iter()
    .map(|r| match r {
      Expr::List(cols) => Some(cols.to_vec()),
      _ => None,
    })
    .collect();
  if has_real
    && let Some(a) = matrix.as_deref().and_then(machine_real_matrix)
    && let Some((_, sigma, _)) =
      jacobi_svd_with_tolerance(&a, rel_tol.unwrap_or(100.0 * f64::EPSILON))
  {
    return Ok(take_values(
      (0..nrows.min(ncols))
        .map(|i| sigma[i][i])
        .filter(|s| *s > 0.0)
        .map(Expr::Real)
        .collect(),
    ));
  }

  // Eigenvalues of the smaller Gram matrix (m.m^H or m^H.m); both share
  // the nonzero spectrum
  let ct = call1("ConjugateTranspose", args[0].clone());
  let gram_args = if nrows <= ncols {
    vec![args[0].clone(), ct]
  } else {
    vec![ct, args[0].clone()]
  };
  let eigenvalues =
    evaluate_expr_to_expr(&call("Eigenvalues", vec![call("Dot", gram_args)]))?;
  let Expr::List(eig_items) = &eigenvalues else {
    return Ok(unevaluated(args));
  };

  // Pair each eigenvalue with its numeric value for sorting and zero-drop
  let mut pairs: Vec<(Expr, f64)> = Vec::with_capacity(eig_items.len());
  for e in eig_items {
    match try_eval_to_f64(e) {
      Some(v) => pairs.push((e.clone(), v)),
      None => return Ok(unevaluated(args)),
    }
  }
  let max_val = pairs.iter().map(|(_, v)| v.abs()).fold(0.0_f64, f64::max);

  // Exact input: drop only exact zeros. Machine input: drop eigenvalues
  // that are zero relative to the largest (Gram eigenvalues scale as the
  // squares of the singular values).
  let tol = if has_real { max_val * 1e-24 } else { 0.0 };
  pairs.retain(|(e, v)| !matches!(e, Expr::Integer(0)) && v.abs() > tol);

  // Largest first
  pairs
    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

  let mut values = Vec::with_capacity(pairs.len());
  for (e, _) in &pairs {
    // Use the Sqrt head (not make_sqrt's Power form): the Power
    // simplifier splits Sqrt[a*b/c] into radical products, while
    // wolframscript keeps the nested Sqrt[(a*b)/c] form
    values.push(evaluate_expr_to_expr(&call1("Sqrt", e.clone()))?);
  }

  Ok(take_values(values))
}

/// Shared MatrixExp / MatrixLog implementation. `head` is the matrix
/// function name (for unevaluated returns); `func` the scalar head
/// ("Exp" / "Log").
///
/// Diagonal matrices apply `func` entrywise to the diagonal. 2×2 matrices
/// with distinct eigenvalues use Sylvester's formula `f(M) = α·M + β·I`
/// (α = (f(λ₁) − f(λ₂))/(λ₁ − λ₂), β = (λ₁·f(λ₂) − λ₂·f(λ₁))/(λ₁ − λ₂)),
/// which keeps Wolfram-canonical factorings like `2*(-1 + E)` intact.
/// A repeated eigenvalue uses the Jordan form `f(M) = f'(λ)·M +
/// (f(λ) − λ·f'(λ))·I` — the same α·M + β·I assembly with α = f'(λ).
/// `f(A) = V.f(D).V^-1` for an exact n x n matrix with n independent
/// eigenvectors; `None` when the matrix is defective, an eigenvalue is
/// outside `f`'s domain (`Log` at 0), or the eigen-decomposition is not exact.
fn diagonalizable_matrix_function(
  mat: &Expr,
  n: usize,
  func: &str,
  make: &dyn Fn(&str, Vec<Expr>) -> Result<Expr, InterpreterError>,
) -> Result<Option<Expr>, InterpreterError> {
  let evals = match &make("Eigenvalues", vec![mat.clone()])? {
    Expr::List(xs) if xs.len() == n => xs.clone(),
    _ => return Ok(None),
  };
  let vecs = match &make("Eigenvectors", vec![mat.clone()])? {
    Expr::List(xs) if xs.len() == n => xs.clone(),
    _ => return Ok(None),
  };
  let is_zero = |e: &Expr| {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(z) if *z == 0.0)
  };
  if func == "Log" && evals.iter().any(is_zero) {
    return Ok(None);
  }
  // Defective matrices get zero-padded eigenvectors; the determinant of the
  // eigenvector matrix (vectors as columns) is nonzero exactly when they are
  // independent.
  let v = make("Transpose", vec![Expr::List(vecs)])?;
  let det = make("Simplify", vec![make("Det", vec![v.clone()])?])?;
  if is_zero(&det) {
    return Ok(None);
  }
  let mut f_evals = Vec::with_capacity(n);
  for l in &evals {
    f_evals.push(make(func, vec![l.clone()])?);
  }
  let d = make("DiagonalMatrix", vec![Expr::List(f_evals.into())])?;
  let vinv = make("Inverse", vec![v.clone()])?;
  let prod = make("Dot", vec![v, d, vinv])?;
  Ok(Some(make("Simplify", vec![prod])?))
}

pub fn matrix_function_ast(
  args: &[Expr],
  head: &str,
  func: &str,
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated(head, args);
  let mat = args[0].clone();

  // Square-matrix shape check
  let n = match &mat {
    Expr::List(rows)
      if !rows.is_empty()
        && rows.iter().all(
          |r| matches!(r, Expr::List(cols) if cols.len() == rows.len()),
        ) =>
    {
      rows.len()
    }
    _ => 0,
  };
  if n == 0 {
    // Not a nonempty square matrix: emit <head>::matsq for a concrete
    // argument, hold symbolic ones.
    return Ok(matsq_unevaluated(head, args));
  }

  let is_zero = |e: &Expr| {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(z) if *z == 0.0)
  };
  // Log is not defined at 0: emit fnand (matching wolframscript) and
  // return unevaluated
  let log_zero_bail = |args: &[Expr]| {
    crate::emit_message(&format!(
      "{head}::fnand: The function Log is not analytic or defined at 0.",
    ));
    Ok(unevaluated(args))
  };

  // Diagonal-matrix fast path: works for symbolic entries too
  if let Expr::List(rows) = &mat {
    let is_diag = rows.iter().enumerate().all(|(i, r)| {
      if let Expr::List(cs) = r {
        cs.iter().enumerate().all(|(j, c)| i == j || is_zero(c))
      } else {
        false
      }
    });
    if is_diag {
      let mut new_rows: Vec<Expr> = Vec::with_capacity(n);
      for (i, r) in rows.iter().enumerate() {
        if let Expr::List(cs) = r {
          let mut new_cs: Vec<Expr> = Vec::with_capacity(n);
          for (j, c) in cs.iter().enumerate() {
            if i == j {
              if func == "Log" && is_zero(c) {
                return log_zero_bail(args);
              }
              new_cs.push(evaluate_expr_to_expr(&call1(func, c.clone()))?);
            } else {
              new_cs.push(c.clone());
            }
          }
          new_rows.push(Expr::List(new_cs.into()));
        }
      }
      return Ok(Expr::List(new_rows.into()));
    }
  }
  // Machine-precision real matrix larger than 2x2: numeric exponential.
  if func == "Exp"
    && n > 2
    && let Some(result) = numeric_matrix_exp(&mat)
  {
    return Ok(result);
  }
  let make = |head: &str, args: Vec<Expr>| -> Result<Expr, InterpreterError> {
    evaluate_expr_to_expr(&call(head, args))
  };
  if n != 2 {
    // Larger exact matrices: f(A) = V.f(D).V^-1 whenever A has a full set of
    // independent eigenvectors. Defective matrices stay unevaluated.
    return Ok(
      diagonalizable_matrix_function(&mat, n, func, &make)?
        .unwrap_or_else(|| unevaluated(args)),
    );
  }

  let evals = make("Eigenvalues", vec![mat.clone()])?;
  let evals_list = match &evals {
    Expr::List(xs) if xs.len() == 2 => xs.clone(),
    _ => return Ok(unevaluated(args)),
  };
  let l1 = evals_list[0].clone();
  let l2 = evals_list[1].clone();
  if func == "Log" && (is_zero(&l1) || is_zero(&l2)) {
    return log_zero_bail(args);
  }

  // λ₁ − λ₂ decides between Sylvester (distinct) and Jordan (repeated)
  let diff = make("Plus", vec![l1.clone(), neg(l2.clone())])?;

  let (alpha, beta) = if is_zero(&diff) {
    // Repeated eigenvalue λ: α = f'(λ), β = f(λ) − λ·f'(λ)
    let f_l = make(func, vec![l1.clone()])?;
    let fp_l = match func {
      "Exp" => f_l.clone(),
      "Log" => make("Divide", vec![Expr::Integer(1), l1.clone()])?,
      _ => return Ok(unevaluated(args)),
    };
    let beta = make(
      "Plus",
      vec![
        f_l,
        times(vec![Expr::Integer(-1), l1.clone(), fp_l.clone()]),
      ],
    )?;
    (fp_l, beta)
  } else {
    let f_l1 = make(func, vec![l1.clone()])?;
    let f_l2 = make(func, vec![l2.clone()])?;
    // α = (f(λ₁) − f(λ₂)) / (λ₁ − λ₂)
    let alpha_num = make("Plus", vec![f_l1.clone(), neg(f_l2.clone())])?;
    let alpha = make("Divide", vec![alpha_num, diff.clone()])?;
    // β = (λ₁·f(λ₂) − λ₂·f(λ₁)) / (λ₁ − λ₂)
    let beta_num = make(
      "Plus",
      vec![
        times(vec![l1.clone(), f_l2.clone()]),
        times(vec![Expr::Integer(-1), l2.clone(), f_l1.clone()]),
      ],
    )?;
    let beta = make("Divide", vec![beta_num, diff])?;
    (alpha, beta)
  };

  // α·M + β·I
  let identity = crate::evaluator::evaluate_function_call_ast(
    "IdentityMatrix",
    &[Expr::Integer(2)],
  )?;
  let alpha_m = make("Times", vec![alpha, mat.clone()])?;
  let beta_i = make("Times", vec![beta, identity])?;
  let result = make("Plus", vec![alpha_m, beta_i])?;

  // Pure-imaginary eigenvalues (a zero-trace 2×2 matrix with complex
  // eigenvalues ±i·b) turn the α·M + β·I assembly into a tangle of complex
  // exponentials (e.g. (I/E^I + I*E^I)/(2*I)); convert these to the canonical
  // trig form, matching wolframscript:
  // MatrixExp[{{0,-1},{1,0}}] = {{Cos[1], -Sin[1]}, {Sin[1], Cos[1]}}.
  // General complex eigenvalues a±i·b (a ≠ 0) are left as-is: Simplify cannot
  // reduce E^(a±i·b) to E^a·(Cos b ± i·Sin b), so converting them would only
  // produce a differently-tangled Cosh/Sinh form.
  if func == "Exp" && mentions_imaginary_unit(&l1) {
    let trace = if let Expr::List(rows) = &mat {
      match (&rows[0], &rows[1]) {
        (Expr::List(r0), Expr::List(r1)) => {
          make("Plus", vec![r0[0].clone(), r1[1].clone()]).ok()
        }
        _ => None,
      }
    } else {
      None
    };
    if matches!(&trace, Some(t) if is_zero(t)) {
      let trig = make("ExpToTrig", vec![result])?;
      return make("Simplify", vec![trig]);
    }
  }
  Ok(result)
}

/// Matrix exponential of a square matrix whose entries are real numbers (at
/// least one of them inexact), via scaling and squaring of a Taylor series.
/// Returns None for any other kind of entry so exact input stays symbolic.
fn numeric_matrix_exp(mat: &Expr) -> Option<Expr> {
  let Expr::List(rows) = mat else { return None };
  let n = rows.len();
  let mut a = vec![vec![0.0f64; n]; n];
  let mut inexact = false;
  for (i, r) in rows.iter().enumerate() {
    let Expr::List(cs) = r else { return None };
    for (j, c) in cs.iter().enumerate() {
      a[i][j] = match c {
        Expr::Real(x) => {
          inexact = true;
          *x
        }
        Expr::Integer(k) => *k as f64,
        _ => return None,
      };
    }
  }
  if !inexact || a.iter().flatten().any(|x| !x.is_finite()) {
    return None;
  }
  let mul = |x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>| {
    let mut z = vec![vec![0.0f64; n]; n];
    for i in 0..n {
      for k in 0..n {
        let xik = x[i][k];
        for j in 0..n {
          z[i][j] += xik * y[k][j];
        }
      }
    }
    z
  };
  let norm = a
    .iter()
    .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
    .fold(0.0, f64::max);
  let squarings = if norm > 0.5 {
    (norm / 0.5).log2().ceil() as i32
  } else {
    0
  };
  let scale = 2f64.powi(-squarings);
  for x in a.iter_mut().flatten() {
    *x *= scale;
  }
  let mut result = vec![vec![0.0f64; n]; n];
  let mut term = vec![vec![0.0f64; n]; n];
  for i in 0..n {
    result[i][i] = 1.0;
    term[i][i] = 1.0;
  }
  for k in 1..=24 {
    term = mul(&term, &a);
    for x in term.iter_mut().flatten() {
      *x /= k as f64;
    }
    for i in 0..n {
      for j in 0..n {
        result[i][j] += term[i][j];
      }
    }
  }
  for _ in 0..squarings {
    result = mul(&result, &result);
  }
  Some(Expr::List(
    result
      .into_iter()
      .map(|r| Expr::List(r.into_iter().map(Expr::Real).collect()))
      .collect(),
  ))
}

/// True if the expression structurally contains the imaginary unit — the bare
/// symbol `I` or a `Complex[_, nonzero]` literal. Used to detect non-real
/// eigenvalues so MatrixExp can render them in trig form. (Eigenvalues print
/// `I` as the identifier `I` rather than the `Complex[0, 1]` literal, so a
/// direct `Im[...]` test does not resolve.)
fn mentions_imaginary_unit(e: &Expr) -> bool {
  match e {
    Expr::Identifier(s) | Expr::Constant(s) => s == "I",
    Expr::FunctionCall { name, args } => {
      (name == "Complex"
        && args.len() == 2
        && !matches!(&args[1], Expr::Integer(0))
        && !matches!(&args[1], Expr::Real(z) if *z == 0.0))
        || args.iter().any(mentions_imaginary_unit)
    }
    Expr::BinaryOp { left, right, .. } => {
      mentions_imaginary_unit(left) || mentions_imaginary_unit(right)
    }
    Expr::UnaryOp { operand, .. } => mentions_imaginary_unit(operand),
    _ => false,
  }
}

/// JordanDecomposition[m] for exact 2×2 matrices: returns {s, j} with
/// m == s.j.Inverse[s], following wolframscript's conventions:
/// - j is Diagonal/Jordan with the eigenvalues in Eigenvalues[m] order
/// - eigenvectors normalize to ((λ - d)/c, 1) when c ≠ 0; triangular
///   matrices use (1, 0) for λ = a and (b/(d - a), 1) for λ = d;
///   diagonal matrices use permuted basis vectors
/// - defective matrices append the generalized eigenvector with second
///   component 0 (or {0, 1/b} in the upper-triangular case)
pub fn jordan_decomposition_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("JordanDecomposition", args);
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }

  // 2×2 matrix with exact (Integer/Rational) entries
  let is_exact = |e: &Expr| {
    matches!(e, Expr::Integer(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };
  let rows = match &args[0] {
    Expr::List(rows) if rows.len() == 2 => rows,
    _ => return Ok(unevaluated(args)),
  };
  let (a, b, c, d) = match (&rows[0], &rows[1]) {
    (Expr::List(r0), Expr::List(r1)) if r0.len() == 2 && r1.len() == 2 => {
      (r0[0].clone(), r0[1].clone(), r1[0].clone(), r1[1].clone())
    }
    _ => return Ok(unevaluated(args)),
  };
  if ![&a, &b, &c, &d].into_iter().all(is_exact) {
    return Ok(unevaluated(args));
  }

  let eigen = evaluate_expr_to_expr(&call1("Eigenvalues", args[0].clone()))?;
  let (l1, l2) = match &eigen {
    Expr::List(ls) if ls.len() == 2 => (ls[0].clone(), ls[1].clone()),
    _ => return Ok(unevaluated(args)),
  };

  // Evaluate + double Simplify: one Simplify pass normalizes the
  // I*Sqrt[..] factor order but can leave an unfolded 3*2 denominator
  // in nested-radical quotients; the second pass folds it to match
  // wolframscript's (-3 + Sqrt[33])/6. The nested negation
  // Times[-1, Times[I, ...]] flattens so the printer gives -I*Sqrt[2]
  // instead of -(I*Sqrt[2]).
  let ev = |e: Expr| -> Result<Expr, InterpreterError> {
    let simplify = |x: Expr| evaluate_expr_to_expr(&call1("Simplify", x));
    let result = simplify(simplify(evaluate_expr_to_expr(&e)?)?)?;
    let inner = match &result {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => Some(operand.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Times"
          && args.len() == 2
          && matches!(&args[0], Expr::Integer(-1)) =>
      {
        Some(&args[1])
      }
      _ => None,
    };
    fn times_args(e: &Expr) -> Option<Vec<Expr>> {
      match e {
        Expr::FunctionCall { name, args } if name == "Times" => {
          Some(args.iter().cloned().collect())
        }
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left,
          right,
        } => {
          let mut out =
            times_args(left).unwrap_or_else(|| vec![(**left).clone()]);
          out.extend(
            times_args(right).unwrap_or_else(|| vec![(**right).clone()]),
          );
          Some(out)
        }
        _ => None,
      }
    }
    if let Some(e) = inner
      && let Some(factors) = times_args(e)
    {
      let mut flat = vec![Expr::Integer(-1)];
      flat.extend(factors);
      return Ok(times(flat));
    }
    Ok(result)
  };
  let same = |x: &Expr, y: &Expr| expr_to_string(x) == expr_to_string(y);
  let zero = Expr::Integer(0);
  let one = Expr::Integer(1);
  let b_zero = same(&b, &zero);
  let c_zero = same(&c, &zero);
  let repeated = same(&l1, &l2);

  let sub = |x: Expr, y: Expr| plus(vec![x, neg(y)]);
  let matrix = |c1: (Expr, Expr), c2: (Expr, Expr)| {
    Expr::List(
      vec![
        Expr::List(vec![c1.0, c2.0].into()),
        Expr::List(vec![c1.1, c2.1].into()),
      ]
      .into(),
    )
  };
  let diag = |x: Expr, y: Expr| {
    Expr::List(
      vec![
        Expr::List(vec![x, Expr::Integer(0)].into()),
        Expr::List(vec![Expr::Integer(0), y].into()),
      ]
      .into(),
    )
  };

  // Diagonal matrix: permuted basis vectors (λ1 prefers the d slot)
  if b_zero && c_zero {
    let (col1, col2) = if same(&l1, &d) {
      ((zero.clone(), one.clone()), (one.clone(), zero.clone()))
    } else {
      ((one.clone(), zero.clone()), (zero.clone(), one.clone()))
    };
    let s = matrix(col1, col2);
    return Ok(Expr::List(vec![s, diag(l1, l2)].into()));
  }

  if !repeated {
    let s = if c_zero {
      // Upper triangular: (1, 0) for λ = a, (b/(d - a), 1) for λ = d
      let vd = (ev(div2(b.clone(), sub(d.clone(), a.clone())))?, one.clone());
      let va = (one.clone(), zero.clone());
      let (col1, col2) = if same(&l1, &a) { (va, vd) } else { (vd, va) };
      matrix(col1, col2)
    } else {
      // ((λ - d)/c, 1) for both eigenvalues
      let x1 = ev(div2(sub(l1.clone(), d.clone()), c.clone()))?;
      let x2 = ev(div2(sub(l2.clone(), d.clone()), c.clone()))?;
      matrix((x1, one.clone()), (x2, one.clone()))
    };
    return Ok(Expr::List(vec![s, diag(l1, l2)].into()));
  }

  // Repeated eigenvalue, non-diagonal: defective Jordan block
  let lam = l1;
  let j = Expr::List(
    vec![
      Expr::List(vec![lam.clone(), Expr::Integer(1)].into()),
      Expr::List(vec![Expr::Integer(0), lam.clone()].into()),
    ]
    .into(),
  );
  let s = if c_zero {
    // Upper triangular defective: v = (1, 0), w = (0, 1/b)
    let w2 = ev(div2(one.clone(), b.clone()))?;
    matrix((one.clone(), zero.clone()), (zero.clone(), w2))
  } else {
    let v1 = ev(div2(sub(lam.clone(), d.clone()), c.clone()))?;
    let a_minus_lam = ev(sub(a.clone(), lam.clone()))?;
    let w1 = if same(&a_minus_lam, &zero) {
      ev(div2(one.clone(), c.clone()))?
    } else {
      ev(div2(v1.clone(), a_minus_lam))?
    };
    matrix((v1, one.clone()), (w1, zero.clone()))
  };
  Ok(Expr::List(vec![s, j].into()))
}

/// JordanReduce[m] — the Jordan canonical form of a square matrix (the
/// second element of JordanDecomposition, without the similarity matrix),
/// following wolframscript's block-ordering conventions:
/// - all eigenvalues distinct: a diagonal matrix in Eigenvalues[m] order
///   (decreasing magnitude);
/// - any repeated eigenvalue: blocks sorted ascending by eigenvalue
///   (real part, then imaginary part), and within one eigenvalue by
///   ascending block size.
///   Block sizes come from the ranks of (m − λI)^k. Matrices whose
///   eigenvalues need Root objects stay unevaluated (wolframscript's Root
///   ordering there is undecoded), as do inexact matrices of size ≥ 3 with
///   repeated eigenvalues (the float eigensolver cannot separate clusters).
pub fn jordan_reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("JordanReduce", args);
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) if is_nonempty_square(&m) => m,
    _ => return Ok(matsq_unevaluated("JordanReduce", args)),
  };
  let n = matrix.len();

  let eig = eigenvalues_ast(&args[0..1])?;
  let eigs: Vec<Expr> = match &eig {
    Expr::List(items) if items.len() == n => items.to_vec(),
    _ => return Ok(unevaluated()),
  };
  if eigs.iter().any(|e| expr_to_string(e).contains("Root[")) {
    return Ok(unevaluated());
  }

  // Group structurally equal eigenvalues, keeping first-seen order.
  let mut distinct: Vec<(Expr, usize)> = Vec::new();
  for e in &eigs {
    let key = expr_to_string(e);
    match distinct.iter_mut().find(|(d, _)| expr_to_string(d) == key) {
      Some(entry) => entry.1 += 1,
      None => distinct.push((e.clone(), 1)),
    }
  }

  // All eigenvalues distinct: diagonal in Eigenvalues order.
  if distinct.len() == n {
    let blocks: Vec<(Expr, usize)> =
      eigs.iter().map(|e| (e.clone(), 1)).collect();
    return Ok(build_jordan_matrix(&blocks, n));
  }

  // Sort distinct eigenvalues ascending by (Re, Im); with a single
  // distinct value (e.g. a purely symbolic defective matrix) no numeric
  // key is needed.
  if distinct.len() > 1 {
    let mut keyed: Vec<((f64, f64), Expr, usize)> = Vec::new();
    for (e, m) in &distinct {
      match numeric_re_im(e) {
        Some(key) if key.0.is_finite() && key.1.is_finite() => {
          keyed.push((key, e.clone(), *m));
        }
        _ => return Ok(unevaluated()),
      }
    }
    keyed.sort_by(|a, b| {
      a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    distinct = keyed.into_iter().map(|(_, e, m)| (e, m)).collect();
  }

  // Jordan block sizes for each eigenvalue λ of algebraic multiplicity m:
  // with r_k = rank((A − λI)^k) and ν_k = r_{k−1} − r_k (the number of
  // blocks of size ≥ k), there are ν_k − ν_{k+1} blocks of size exactly k.
  let mut blocks: Vec<(Expr, usize)> = Vec::new();
  for (lam, mult) in &distinct {
    if *mult == 1 {
      blocks.push((lam.clone(), 1));
      continue;
    }
    let Some(b) = subtract_scalar_from_diagonal(&matrix, lam) else {
      return Ok(unevaluated());
    };
    let mut ranks: Vec<usize> = vec![n];
    let mut p = b.clone();
    loop {
      let Some(r) = matrix_rank_of(&p) else {
        return Ok(unevaluated());
      };
      ranks.push(r);
      if r <= n - mult {
        break;
      }
      if ranks.len() > mult + 1 {
        return Ok(unevaluated());
      }
      p = match evaluated_matrix_product(&p, &b) {
        Some(next) => next,
        None => return Ok(unevaluated()),
      };
    }
    if *ranks.last().unwrap() != n - mult {
      return Ok(unevaluated());
    }
    let kmax = ranks.len() - 1;
    let nu = |k: usize| -> usize {
      if k > kmax { 0 } else { ranks[k - 1] - ranks[k] }
    };
    // Ascending block size within one eigenvalue.
    for k in 1..=kmax {
      let count = nu(k).saturating_sub(nu(k + 1));
      for _ in 0..count {
        blocks.push((lam.clone(), k));
      }
    }
  }

  if blocks.iter().map(|(_, s)| s).sum::<usize>() != n {
    return Ok(unevaluated());
  }
  Ok(build_jordan_matrix(&blocks, n))
}

/// Assemble a Jordan matrix from `(eigenvalue, block size)` pairs laid out
/// along the diagonal, with exact integer 0/1 filler entries.
fn build_jordan_matrix(blocks: &[(Expr, usize)], n: usize) -> Expr {
  let mut rows = vec![vec![Expr::Integer(0); n]; n];
  let mut pos = 0;
  for (lam, size) in blocks {
    for i in 0..*size {
      rows[pos + i][pos + i] = lam.clone();
      if i + 1 < *size {
        rows[pos + i][pos + i + 1] = Expr::Integer(1);
      }
    }
    pos += size;
  }
  Expr::List(
    rows
      .into_iter()
      .map(|r| Expr::List(r.into()))
      .collect::<Vec<_>>()
      .into(),
  )
}

/// Numeric (Re, Im) key of an eigenvalue expression via Re/Im + N, or None
/// when either part does not evaluate to a number.
fn numeric_re_im(e: &Expr) -> Option<(f64, f64)> {
  let component = |head: &str| -> Option<f64> {
    let call = call1(head, e.clone());
    let value = crate::evaluator::evaluate_expr_to_expr(&call).ok()?;
    try_eval_to_f64(&value)
  };
  Some((component("Re")?, component("Im")?))
}

/// m − λI with every entry evaluated, or None when evaluation fails.
/// The constant spelling `I` in λ is normalized to a `Complex[0, 1]`
/// literal first: `Times[I, I]` with the constant spelling survives as an
/// unreduced `I^2` in matrix products, whereas Complex × Complex does
/// proper complex arithmetic (see jordan_decomposition_ast).
fn subtract_scalar_from_diagonal(
  matrix: &[Vec<Expr>],
  lam: &Expr,
) -> Option<Vec<Vec<Expr>>> {
  let lam = replace_constant_i(lam);
  let mut out: Vec<Vec<Expr>> = Vec::with_capacity(matrix.len());
  for (i, row) in matrix.iter().enumerate() {
    let mut new_row: Vec<Expr> = Vec::with_capacity(row.len());
    for (j, cell) in row.iter().enumerate() {
      if i == j {
        let diff = plus(vec![cell.clone(), neg(lam.clone())]);
        new_row.push(crate::evaluator::evaluate_expr_to_expr(&diff).ok()?);
      } else {
        new_row.push(cell.clone());
      }
    }
    out.push(new_row);
  }
  Some(out)
}

/// Replace the constant/identifier spelling of the imaginary unit with a
/// `Complex[0, 1]` literal throughout an expression.
fn replace_constant_i(e: &Expr) -> Expr {
  let complex_i = || call("Complex", vec![Expr::Integer(0), Expr::Integer(1)]);
  match e {
    Expr::Constant(c) if c == "I" => complex_i(),
    Expr::Identifier(c) if c == "I" => complex_i(),
    Expr::FunctionCall { name, args } => {
      call(name.as_str(), args.iter().map(replace_constant_i).collect())
    }
    Expr::BinaryOp { op, left, right } => {
      binop(*op, replace_constant_i(left), replace_constant_i(right))
    }
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(replace_constant_i(operand)),
    },
    _ => e.clone(),
  }
}

/// Rank of a matrix via the evaluator's MatrixRank (handles exact, radical
/// and float entries alike), or None when it does not reduce to an integer.
fn matrix_rank_of(matrix: &[Vec<Expr>]) -> Option<usize> {
  let call = call1("MatrixRank", matrix_to_expr(matrix.to_vec()));
  match crate::evaluator::evaluate_expr_to_expr(&call).ok()? {
    Expr::Integer(r) if r >= 0 => Some(r as usize),
    _ => None,
  }
}

/// Evaluated (and simplified) matrix product a.b, or None when the result
/// is not a plain matrix.
fn evaluated_matrix_product(
  a: &[Vec<Expr>],
  b: &[Vec<Expr>],
) -> Option<Vec<Vec<Expr>>> {
  let dot = call(
    "Dot",
    vec![matrix_to_expr(a.to_vec()), matrix_to_expr(b.to_vec())],
  );
  let product = crate::evaluator::evaluate_expr_to_expr(&dot).ok()?;
  expr_to_matrix(&product)
}

// ─── FrobeniusReduce ────────────────────────────────────────────────────

/// Exact rational number n/d, kept normalized (d > 0, gcd(|n|, d) = 1).
/// All arithmetic is overflow-checked; None propagates to an unevaluated
/// FrobeniusReduce call.
#[derive(Clone, Copy, PartialEq, Eq)]
struct QNum {
  n: i128,
  d: i128,
}

impl QNum {
  fn new(n: i128, d: i128) -> Option<Self> {
    if d == 0 {
      return None;
    }
    let (n, d) = if d < 0 {
      (n.checked_neg()?, d.checked_neg()?)
    } else {
      (n, d)
    };
    let (n, d) = rat_reduce(n, d);
    Some(Self { n, d })
  }
  fn zero() -> Self {
    Self { n: 0, d: 1 }
  }
  fn is_zero(&self) -> bool {
    self.n == 0
  }
  fn add(&self, o: &Self) -> Option<Self> {
    Self::new(
      self
        .n
        .checked_mul(o.d)?
        .checked_add(o.n.checked_mul(self.d)?)?,
      self.d.checked_mul(o.d)?,
    )
  }
  fn sub(&self, o: &Self) -> Option<Self> {
    Self::new(
      self
        .n
        .checked_mul(o.d)?
        .checked_sub(o.n.checked_mul(self.d)?)?,
      self.d.checked_mul(o.d)?,
    )
  }
  fn mul(&self, o: &Self) -> Option<Self> {
    Self::new(self.n.checked_mul(o.n)?, self.d.checked_mul(o.d)?)
  }
}

/// Gaussian rational re + im*I — the exact scalar field FrobeniusReduce
/// works over (integer, rational and complex-rational matrix entries).
#[derive(Clone, Copy, PartialEq, Eq)]
struct GQNum {
  re: QNum,
  im: QNum,
}

impl GQNum {
  fn zero() -> Self {
    Self {
      re: QNum::zero(),
      im: QNum::zero(),
    }
  }
  fn one() -> Self {
    Self {
      re: QNum { n: 1, d: 1 },
      im: QNum::zero(),
    }
  }
  fn is_zero(&self) -> bool {
    self.re.is_zero() && self.im.is_zero()
  }
  fn add(&self, o: &Self) -> Option<Self> {
    Some(Self {
      re: self.re.add(&o.re)?,
      im: self.im.add(&o.im)?,
    })
  }
  fn sub(&self, o: &Self) -> Option<Self> {
    Some(Self {
      re: self.re.sub(&o.re)?,
      im: self.im.sub(&o.im)?,
    })
  }
  fn mul(&self, o: &Self) -> Option<Self> {
    Some(Self {
      re: self.re.mul(&o.re)?.sub(&self.im.mul(&o.im)?)?,
      im: self.re.mul(&o.im)?.add(&self.im.mul(&o.re)?)?,
    })
  }
  fn div(&self, o: &Self) -> Option<Self> {
    // (a + bI)/(c + dI) = (a + bI)(c − dI)/(c² + d²)
    let norm = o.re.mul(&o.re)?.add(&o.im.mul(&o.im)?)?;
    if norm.is_zero() {
      return None;
    }
    let num = self.mul(&o.conj())?;
    let inv_norm = QNum::new(norm.d, norm.n)?;
    Some(Self {
      re: num.re.mul(&inv_norm)?,
      im: num.im.mul(&inv_norm)?,
    })
  }
  fn conj(&self) -> Self {
    Self {
      re: self.re,
      im: QNum {
        n: -self.im.n,
        d: self.im.d,
      },
    }
  }
}

/// The coefficient-field operations the polynomial Smith reduction needs:
/// implemented by the Gaussian rationals (exact path) and the prime fields
/// GF(p) (Modulus option). All operations are fallible so i128 overflow
/// propagates to an unevaluated call.
trait PField: Copy + PartialEq {
  fn zero() -> Self;
  fn is_zero(&self) -> bool;
  fn add(&self, o: &Self) -> Option<Self>;
  fn sub(&self, o: &Self) -> Option<Self>;
  fn mul(&self, o: &Self) -> Option<Self>;
  fn div(&self, o: &Self) -> Option<Self>;
}

impl PField for GQNum {
  fn zero() -> Self {
    Self::zero()
  }
  fn is_zero(&self) -> bool {
    Self::is_zero(self)
  }
  fn add(&self, o: &Self) -> Option<Self> {
    Self::add(self, o)
  }
  fn sub(&self, o: &Self) -> Option<Self> {
    Self::sub(self, o)
  }
  fn mul(&self, o: &Self) -> Option<Self> {
    Self::mul(self, o)
  }
  fn div(&self, o: &Self) -> Option<Self> {
    Self::div(self, o)
  }
}

/// Element of the prime field GF(p), stored as a canonical representative
/// in 0..p−1. The modulus rides along in each value; mixed-modulus
/// arithmetic never happens because a single reduction uses one p.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FpNum {
  v: i128,
  p: i128,
}

impl FpNum {
  fn new(v: i128, p: i128) -> Self {
    Self {
      v: v.rem_euclid(p),
      p,
    }
  }
  /// Multiplicative inverse via the extended Euclidean algorithm; None
  /// for values not coprime to p.
  fn inverse(&self) -> Option<Self> {
    let (mut r0, mut r1) = (self.p, self.v);
    let (mut s0, mut s1) = (0i128, 1i128);
    while r1 != 0 {
      let q = r0 / r1;
      (r0, r1) = (r1, r0 - q * r1);
      (s0, s1) = (s1, s0 - q * s1);
    }
    if r0 != 1 {
      return None;
    }
    Some(Self::new(s0, self.p))
  }
}

impl PField for FpNum {
  // zero() carries no modulus: the polynomial helpers only synthesize it
  // for padding, and arithmetic picks up the real modulus from the other
  // operand (max(p, 1)), so the p = 0 sentinel never reaches rem_euclid.
  fn zero() -> Self {
    Self { v: 0, p: 0 }
  }
  fn is_zero(&self) -> bool {
    self.v == 0
  }
  fn add(&self, o: &Self) -> Option<Self> {
    Some(Self::new(self.v.checked_add(o.v)?, self.p.max(o.p).max(1)))
  }
  fn sub(&self, o: &Self) -> Option<Self> {
    Some(Self::new(self.v.checked_sub(o.v)?, self.p.max(o.p).max(1)))
  }
  fn mul(&self, o: &Self) -> Option<Self> {
    Some(Self::new(self.v.checked_mul(o.v)?, self.p.max(o.p).max(1)))
  }
  fn div(&self, o: &Self) -> Option<Self> {
    let p = self.p.max(o.p).max(1);
    let inv = Self { v: o.v, p }.inverse()?;
    self.mul(&inv)
  }
}

/// Polynomial over a coefficient field, ascending coefficients with no
/// trailing zeros (the zero polynomial is the empty vector).
type GPoly = Vec<GQNum>;

fn gp_trim<F: PField>(mut p: Vec<F>) -> Vec<F> {
  while p.last().is_some_and(PField::is_zero) {
    p.pop();
  }
  p
}

fn gp_sub<F: PField>(a: &[F], b: &[F]) -> Option<Vec<F>> {
  let mut out = vec![F::zero(); a.len().max(b.len())];
  out[..a.len()].copy_from_slice(a);
  for (i, c) in b.iter().enumerate() {
    out[i] = out[i].sub(c)?;
  }
  Some(gp_trim(out))
}

fn gp_mul<F: PField>(a: &[F], b: &[F]) -> Option<Vec<F>> {
  if a.is_empty() || b.is_empty() {
    return Some(Vec::new());
  }
  let mut out = vec![F::zero(); a.len() + b.len() - 1];
  for (i, x) in a.iter().enumerate() {
    for (j, y) in b.iter().enumerate() {
      out[i + j] = out[i + j].add(&x.mul(y)?)?;
    }
  }
  Some(gp_trim(out))
}

/// Quotient and remainder of a by nonzero b over the field.
fn gp_divmod<F: PField>(a: &[F], b: &[F]) -> Option<(Vec<F>, Vec<F>)> {
  let db = b.len().checked_sub(1)?;
  let lead = *b.last()?;
  let mut rem = a.to_vec();
  let mut quot = vec![F::zero(); a.len().saturating_sub(db)];
  while rem.len() > db {
    let k = rem.len() - 1 - db;
    let c = rem.last()?.div(&lead)?;
    quot[k] = c;
    let mut shifted = vec![F::zero(); k];
    shifted.extend(b.iter().copied());
    let scaled: Vec<F> = shifted
      .iter()
      .map(|x| x.mul(&c))
      .collect::<Option<Vec<_>>>()?;
    rem = gp_sub(&rem, &scaled)?;
  }
  Some((gp_trim(quot), rem))
}

/// Normalize to a monic polynomial (nonzero input).
fn gp_monic<F: PField>(p: &[F]) -> Option<Vec<F>> {
  let lead = *p.last()?;
  p.iter().map(|c| c.div(&lead)).collect()
}

/// Parse an exact scalar (Integer, Rational, complex with exact parts, or
/// the imaginary-unit spellings) into a Gaussian rational.
fn expr_to_gq(e: &Expr) -> Option<GQNum> {
  match e {
    Expr::Integer(n) => Some(GQNum {
      re: QNum::new(*n, 1)?,
      im: QNum::zero(),
    }),
    Expr::Constant(c) | Expr::Identifier(c) if c == "I" => Some(GQNum {
      re: QNum::zero(),
      im: QNum::new(1, 1)?,
    }),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(n), Expr::Integer(d)) => Some(GQNum {
          re: QNum::new(*n, *d)?,
          im: QNum::zero(),
        }),
        _ => None,
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      let re = expr_to_gq(&args[0])?;
      let im = expr_to_gq(&args[1])?;
      if !re.im.is_zero() || !im.im.is_zero() {
        return None;
      }
      Some(GQNum {
        re: re.re,
        im: im.re,
      })
    }
    // a + b*I and c*I forms that survive as unevaluated arithmetic
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut acc = GQNum::zero();
      for a in args {
        acc = acc.add(&expr_to_gq(a)?)?;
      }
      Some(acc)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut acc = GQNum::one();
      for a in args {
        acc = acc.mul(&expr_to_gq(a)?)?;
      }
      Some(acc)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => GQNum::zero().sub(&expr_to_gq(operand)?),
    // Complex-rational scalars can survive as unevaluated binary
    // arithmetic, e.g. 1/2 + I/3.
    Expr::BinaryOp { op, left, right } => {
      let l = expr_to_gq(left)?;
      let r = expr_to_gq(right)?;
      match op {
        BinaryOperator::Plus => l.add(&r),
        BinaryOperator::Minus => l.sub(&r),
        BinaryOperator::Times => l.mul(&r),
        BinaryOperator::Divide => l.div(&r),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Render a Gaussian rational as an exact expression in canonical form.
fn gq_to_expr(q: &GQNum) -> Option<Expr> {
  let q_part = |x: &QNum| -> Expr {
    if x.d == 1 {
      Expr::Integer(x.n)
    } else {
      crate::functions::math_ast::make_rational(x.n, x.d)
    }
  };
  if q.im.is_zero() {
    return Some(q_part(&q.re));
  }
  let complex = call("Complex", vec![q_part(&q.re), q_part(&q.im)]);
  crate::evaluator::evaluate_expr_to_expr(&complex).ok()
}

/// Smith normal form of a nonsingular polynomial matrix over the Gaussian
/// rationals, returning the monic diagonal (invariant factors, each
/// dividing the next). None on overflow.
fn gp_smith_diagonal<F: PField>(
  mut p: Vec<Vec<Vec<F>>>,
) -> Option<Vec<Vec<F>>> {
  let n = p.len();
  let mut diag: Vec<Vec<F>> = Vec::with_capacity(n);
  for t in 0..n {
    'reduce: loop {
      // Pivot: nonzero entry of minimal degree in the trailing submatrix.
      let mut pivot: Option<(usize, usize)> = None;
      for i in t..n {
        for j in t..n {
          if !p[i][j].is_empty()
            && pivot.is_none_or(|(pi, pj)| p[i][j].len() < p[pi][pj].len())
          {
            pivot = Some((i, j));
          }
        }
      }
      let (pi, pj) = pivot?; // xI − A is nonsingular, so a pivot exists
      p.swap(t, pi);
      for row in &mut p {
        row.swap(t, pj);
      }

      // Clear the pivot column and row by polynomial division.
      let mut dirty = false;
      for i in t + 1..n {
        if p[i][t].is_empty() {
          continue;
        }
        let (q, _) = gp_divmod(&p[i][t], &p[t][t])?;
        for j in t..n {
          let scaled = gp_mul(&q, &p[t][j])?;
          p[i][j] = gp_sub(&p[i][j], &scaled)?;
        }
        if !p[i][t].is_empty() {
          dirty = true; // remainder of smaller degree: re-pick the pivot
        }
      }
      for j in t + 1..n {
        if p[t][j].is_empty() {
          continue;
        }
        let (q, _) = gp_divmod(&p[t][j], &p[t][t])?;
        for i in t..n {
          let scaled = gp_mul(&q, &p[i][t])?;
          p[i][j] = gp_sub(&p[i][j], &scaled)?;
        }
        if !p[t][j].is_empty() {
          dirty = true;
        }
      }
      if dirty {
        continue 'reduce;
      }

      // Divisibility: the pivot must divide every remaining entry; when it
      // does not, mixing that row into row t reduces the pivot degree on
      // the next pass.
      for i in t + 1..n {
        for j in t + 1..n {
          let (_, r) = gp_divmod(&p[i][j], &p[t][t])?;
          if !r.is_empty() {
            let row_i = p[i].clone();
            for (dst, src) in p[t].iter_mut().zip(row_i.iter()) {
              let sum = gp_sub(dst, &gp_sub(&Vec::new(), src)?)?;
              *dst = sum;
            }
            continue 'reduce;
          }
        }
      }
      diag.push(gp_monic(&p[t][t])?);
      break;
    }
  }
  Some(diag)
}

/// The (lhs, rhs) of a rule expression in either representation.
fn rule_parts(e: &Expr) -> Option<(&Expr, &Expr)> {
  match e {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => Some((pattern, replacement)),
    Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2 =>
    {
      Some((&args[0], &args[1]))
    }
    _ => None,
  }
}

/// Assemble the companion-block matrix of a chain of monic invariant
/// factors, rendering each negated coefficient with `to_expr`.
fn companion_block_matrix<F: PField>(
  diag: &[Vec<F>],
  n: usize,
  to_expr: impl Fn(&F) -> Option<Expr>,
) -> Option<Expr> {
  let factors: Vec<&Vec<F>> = diag.iter().filter(|f| f.len() > 1).collect();
  if factors.iter().map(|f| f.len() - 1).sum::<usize>() != n {
    return None;
  }
  let mut rows = vec![vec![Expr::Integer(0); n]; n];
  let mut pos = 0;
  for f in factors {
    let m = f.len() - 1;
    for i in 0..m {
      if i > 0 {
        rows[pos + i][pos + i - 1] = Expr::Integer(1);
      }
      let neg_c = to_expr(&F::zero().sub(&f[i])?)?;
      rows[pos + i][pos + m - 1] = neg_c;
    }
    pos += m;
  }
  Some(Expr::List(
    rows
      .into_iter()
      .map(|r| Expr::List(r.into()))
      .collect::<Vec<_>>()
      .into(),
  ))
}

/// Reduce an exact scalar modulo m. `Ok` is the canonical representative;
/// `Err(true)` marks an entry that exists but is not reducible (its
/// denominator shares a factor with m — reported via ::nmod), `Err(false)`
/// an unsupported entry kind (complex, symbolic — silently unevaluated).
fn expr_mod_m(e: &Expr, m: i128) -> Result<i128, bool> {
  match e {
    Expr::Integer(n) => Ok(n.rem_euclid(m)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(a), Expr::Integer(b)) => {
          let inv = FpNum::new(*b, m).inverse().ok_or(true)?;
          FpNum::new(*a, m).mul(&inv).map(|v| v.v).ok_or(false)
        }
        _ => Err(false),
      }
    }
    // Inexact entries reduce through Rationalize (0.5 → 1/2 works mod 5,
    // 0.1 → 1/10 does not), matching wolframscript.
    Expr::Real(_) => {
      let rat = crate::evaluator::evaluate_expr_to_expr(&call(
        "Rationalize",
        vec![e.clone()],
      ))
      .map_err(|_| false)?;
      if matches!(rat, Expr::Real(_)) {
        return Err(true);
      }
      expr_mod_m(&rat, m)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => expr_mod_m(operand, m).map(|v| (-v).rem_euclid(m)),
    _ => Err(false),
  }
}

/// Emit `FrobeniusReduce::nmod` with the matrix displayed entrywise
/// reduced modulo m (irreducible entries wrapped in PolynomialMod).
fn emit_frobenius_nmod(matrix: &[Vec<Expr>], m: i128) {
  let display = Expr::List(
    matrix
      .iter()
      .map(|row| {
        Expr::List(
          row
            .iter()
            .map(|cell| match expr_mod_m(cell, m) {
              Ok(v) => Expr::Integer(v),
              Err(_) => {
                call("PolynomialMod", vec![cell.clone(), Expr::Integer(m)])
              }
            })
            .collect::<Vec<_>>()
            .into(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  );
  crate::emit_message(&format!(
    "FrobeniusReduce::nmod: {} is not valid modulo {}.",
    expr_to_string(&display),
    m
  ));
}

/// FrobeniusReduce[m] — the Frobenius (rational canonical) form: the block
/// diagonal of companion matrices of the invariant factors of xI − m, in
/// divisibility-chain order (each factor divides the next). Supported for
/// exact integer / rational / complex-rational matrices, plus
/// Modulus -> p over GF(p) for primes (composite moduli emit ::nmod, as
/// do entries whose denominators are not invertible). Inexact and
/// symbolic matrices stay unevaluated.
pub fn frobenius_reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("FrobeniusReduce", args);
  if args.is_empty() {
    return Ok(unevaluated());
  }
  // Arguments beyond the matrix must be options (rules or lists of
  // rules): the first non-rule emits ::nonopt. Unknown option names emit
  // OptionValue::nodef (with wolframscript's internal-symbol wording) and
  // the computation proceeds with defaults.
  let mut modulus: Option<Expr> = None;
  for extra in &args[1..] {
    let rules: Vec<&Expr> = match extra {
      Expr::List(items) => items.iter().collect(),
      other => vec![other],
    };
    for item in rules {
      let Some((lhs, rhs)) = rule_parts(item) else {
        crate::emit_message(&format!(
          "FrobeniusReduce::nonopt: Options expected (instead of {}) beyond position 1 in {}. An option must be a rule or a list of rules.",
          expr_to_string(extra),
          expr_to_string(&unevaluated())
        ));
        return Ok(unevaluated());
      };
      match lhs {
        Expr::Identifier(name) if name == "Modulus" => {
          modulus = Some(rhs.clone());
        }
        other => {
          crate::emit_message(&format!(
            "OptionValue::nodef: Unknown option {} for System`FrobeniusDecompositionDump`iFrobeniusDecomp.",
            expr_to_string(other)
          ));
        }
      }
    }
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) if is_nonempty_square(&m) => m,
    _ => return Ok(matsq_unevaluated("FrobeniusReduce", args)),
  };
  let n = matrix.len();

  // Modulus -> p: compute over GF(p). Modulus -> 0 means no modulus;
  // 1 and composite moduli are ::nmod; anything else (including negative
  // integers, where wolframscript crashes outright) stays unevaluated.
  match &modulus {
    None | Some(Expr::Integer(0)) => {}
    Some(Expr::Integer(m)) if *m >= 1 => {
      let m = *m;
      let is_prime = matches!(
        crate::evaluator::evaluate_expr_to_expr(&call1("PrimeQ", Expr::Integer(m))),
        Ok(Expr::Identifier(ref t)) if t == "True"
      );
      if !is_prime {
        emit_frobenius_nmod(&matrix, m);
        return Ok(unevaluated());
      }
      let mut entries: Vec<Vec<FpNum>> = Vec::with_capacity(n);
      for row in &matrix {
        let mut out_row = Vec::with_capacity(n);
        for cell in row {
          match expr_mod_m(cell, m) {
            Ok(v) => out_row.push(FpNum::new(v, m)),
            Err(true) => {
              emit_frobenius_nmod(&matrix, m);
              return Ok(unevaluated());
            }
            Err(false) => return Ok(unevaluated()),
          }
        }
        entries.push(out_row);
      }
      let p: Vec<Vec<Vec<FpNum>>> = (0..n)
        .map(|i| {
          (0..n)
            .map(|j| {
              let mut poly = vec![FpNum::new(-entries[i][j].v, m)];
              if i == j {
                poly.push(FpNum::new(1, m));
              }
              gp_trim(poly)
            })
            .collect()
        })
        .collect();
      let result = gp_smith_diagonal(p).and_then(|diag| {
        companion_block_matrix(&diag, n, |c: &FpNum| Some(Expr::Integer(c.v)))
      });
      return Ok(result.unwrap_or_else(unevaluated));
    }
    Some(_) => return Ok(unevaluated()),
  }

  // Exact entries only.
  let mut entries: Vec<Vec<GQNum>> = Vec::with_capacity(n);
  for row in &matrix {
    let mut out_row = Vec::with_capacity(n);
    for cell in row {
      match expr_to_gq(cell) {
        Some(v) => out_row.push(v),
        None => return Ok(unevaluated()),
      }
    }
    entries.push(out_row);
  }

  // xI − A as a polynomial matrix.
  let mut p: Vec<Vec<GPoly>> = Vec::with_capacity(n);
  for i in 0..n {
    let mut row: Vec<GPoly> = Vec::with_capacity(n);
    for j in 0..n {
      let Some(neg) = GQNum::zero().sub(&entries[i][j]) else {
        return Ok(unevaluated());
      };
      let mut poly = vec![neg];
      if i == j {
        poly.push(GQNum::one());
      }
      row.push(gp_trim(poly));
    }
    p.push(row);
  }

  let result = gp_smith_diagonal(p)
    .and_then(|diag| companion_block_matrix(&diag, n, gq_to_expr));
  Ok(result.unwrap_or_else(unevaluated))
}

/// Wrap a payload in `head[StructuredArray`StructuredData[{n, n}, payload]]`
/// (the structured-array form wolframscript uses for decomposition factors).
fn structured_matrix(head: &str, n: usize, payload: Expr) -> Expr {
  let dims =
    Expr::List(vec![Expr::Integer(n as i128), Expr::Integer(n as i128)].into());
  call(
    head,
    vec![call("StructuredArray`StructuredData", vec![dims, payload])],
  )
}

/// `SparseArray[Automatic, {n, n}, 0, {1, {rowptr, colidx}, vals}]` in the
/// CSR literal form wolframscript prints, from row-major nonzero triples.
fn sparse_array_literal(n: usize, nonzeros: &[(usize, usize, Expr)]) -> Expr {
  let mut rowptr: Vec<Expr> = Vec::with_capacity(n + 1);
  let mut colidx: Vec<Expr> = Vec::with_capacity(nonzeros.len());
  let mut vals: Vec<Expr> = Vec::with_capacity(nonzeros.len());
  let mut count = 0i128;
  rowptr.push(Expr::Integer(0));
  let mut iter = nonzeros.iter().peekable();
  for row in 0..n {
    while let Some((i, j, v)) = iter.peek() {
      if *i != row {
        break;
      }
      colidx.push(Expr::List(vec![Expr::Integer(*j as i128 + 1)].into()));
      vals.push(v.clone());
      count += 1;
      iter.next();
    }
    rowptr.push(Expr::Integer(count));
  }
  call(
    "SparseArray",
    vec![
      id_expr("Automatic"),
      Expr::List(
        vec![Expr::Integer(n as i128), Expr::Integer(n as i128)].into(),
      ),
      Expr::Integer(0),
      Expr::List(
        vec![
          Expr::Integer(1),
          Expr::List(
            vec![Expr::List(rowptr.into()), Expr::List(colidx.into())].into(),
          ),
          Expr::List(vals.into()),
        ]
        .into(),
      ),
    ],
  )
}

/// LDLDecomposition[m] — the L·D·Lᴴ factorization of an exact Hermitian
/// (or real symmetric) matrix: L unit lower triangular, D diagonal.
/// Returns wolframscript's structured form by default; TargetStructure
/// -> "Dense" gives plain matrices and -> "Sparse" gives CSR SparseArray
/// literals. Emits ::herm for non-Hermitian input, ::nopiv when a zero
/// pivot appears (wolframscript rejects those too, even in the last
/// position), and ::optx / ::badts / ::nonopt for bad options. Inexact
/// matrices stay unevaluated (wolframscript's float pivots carry
/// LAPACK-internal rounding that is not reproducible here).
pub fn ldl_decomposition_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("LDLDecomposition", args);
  if args.is_empty() {
    return Ok(unevaluated());
  }

  // Options: only TargetStructure is understood.
  #[derive(PartialEq)]
  enum Target {
    Structured,
    Dense,
    Sparse,
  }
  let mut target = Target::Structured;
  for extra in &args[1..] {
    let rules: Vec<&Expr> = match extra {
      Expr::List(items) => items.iter().collect(),
      other => vec![other],
    };
    for item in rules {
      let Some((lhs, rhs)) = rule_parts(item) else {
        crate::emit_message(&format!(
          "LDLDecomposition::nonopt: Options expected (instead of {}) beyond position 1 in {}. An option must be a rule or a list of rules.",
          expr_to_string(extra),
          expr_to_string(&unevaluated())
        ));
        return Ok(unevaluated());
      };
      match lhs {
        Expr::Identifier(name) if name == "TargetStructure" => {
          target = match rhs {
            Expr::String(s) if s == "Dense" => Target::Dense,
            Expr::String(s) if s == "Sparse" => Target::Sparse,
            Expr::String(s) if s == "Structured" => Target::Structured,
            Expr::Identifier(a) if a == "Automatic" => Target::Structured,
            other => {
              let shown = match other {
                Expr::String(s) => s.clone(),
                e => expr_to_string(e),
              };
              crate::emit_message(&format!(
                "LDLDecomposition::badts: {shown} is not a valid target structure."
              ));
              return Ok(unevaluated());
            }
          };
        }
        other => {
          crate::emit_message(&format!(
            "LDLDecomposition::optx: Unknown option {} in {}.",
            expr_to_string(other),
            expr_to_string(&unevaluated())
          ));
          return Ok(unevaluated());
        }
      }
    }
  }

  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) if is_nonempty_square(&m) => m,
    _ => return Ok(matsq_unevaluated("LDLDecomposition", args)),
  };
  let n = matrix.len();

  // Exact entries only; inexact matrices are a documented gap.
  let mut entries: Vec<Vec<GQNum>> = Vec::with_capacity(n);
  for row in &matrix {
    let mut out_row = Vec::with_capacity(n);
    for cell in row {
      if let Some(v) = expr_to_gq(cell) {
        out_row.push(v);
      } else {
        let has_real = matrix
          .iter()
          .any(|r| r.iter().any(|e| matches!(e, Expr::Real(_))));
        if !has_real {
          crate::emit_message(&format!(
            "LDLDecomposition::herm: The matrix {} is not Hermitian or real and symmetric.",
            expr_to_string(&args[0])
          ));
        }
        return Ok(unevaluated());
      }
    }
    entries.push(out_row);
  }

  // Hermitian check: a[i][j] must equal conj(a[j][i]) (real diagonal).
  let hermitian =
    (0..n).all(|i| (i..n).all(|j| entries[i][j] == entries[j][i].conj()));
  if !hermitian {
    crate::emit_message(&format!(
      "LDLDecomposition::herm: The matrix {} is not Hermitian or real and symmetric.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated());
  }

  // Factorize: d_j = a_jj − Σ_k |l_jk|²·d_k;  l_ij = (a_ij − Σ_k
  // l_ik·conj(l_jk)·d_k)/d_j. A zero pivot anywhere is ::nopiv.
  let factor = || -> Option<Result<(Vec<Vec<GQNum>>, Vec<GQNum>), ()>> {
    let mut l = vec![vec![GQNum::zero(); n]; n];
    let mut d = vec![GQNum::zero(); n];
    for j in 0..n {
      let mut dj = entries[j][j];
      for k in 0..j {
        dj = dj.sub(&l[j][k].mul(&l[j][k].conj())?.mul(&d[k])?)?;
      }
      if dj.is_zero() {
        return Some(Err(()));
      }
      d[j] = dj;
      l[j][j] = GQNum::one();
      for i in j + 1..n {
        let mut x = entries[i][j];
        for k in 0..j {
          x = x.sub(&l[i][k].mul(&l[j][k].conj())?.mul(&d[k])?)?;
        }
        l[i][j] = x.div(&dj)?;
      }
    }
    Some(Ok((l, d)))
  };
  let (l, d) = match factor() {
    Some(Ok(ld)) => ld,
    Some(Err(())) => {
      crate::emit_message(&format!(
        "LDLDecomposition::nopiv: Encountered a zero pivot while attempting LDLDecomposition on matrix {}.",
        expr_to_string(&args[0])
      ));
      return Ok(unevaluated());
    }
    None => return Ok(unevaluated()), // i128 overflow
  };

  // Render L as a full unit-lower-triangular matrix and D as values.
  let mut l_rows: Vec<Expr> = Vec::with_capacity(n);
  for i in 0..n {
    let mut row: Vec<Expr> = Vec::with_capacity(n);
    for j in 0..n {
      row.push(match j.cmp(&i) {
        std::cmp::Ordering::Less => match gq_to_expr(&l[i][j]) {
          Some(e) => e,
          None => return Ok(unevaluated()),
        },
        std::cmp::Ordering::Equal => Expr::Integer(1),
        std::cmp::Ordering::Greater => Expr::Integer(0),
      });
    }
    l_rows.push(Expr::List(row.into()));
  }
  let l_full = Expr::List(l_rows.into());
  let mut d_vals: Vec<Expr> = Vec::with_capacity(n);
  for v in &d {
    match gq_to_expr(v) {
      Some(e) => d_vals.push(e),
      None => return Ok(unevaluated()),
    }
  }

  Ok(match target {
    Target::Structured => Expr::List(
      vec![
        structured_matrix("LowerTriangularMatrix", n, l_full),
        structured_matrix(
          "DiagonalMatrix",
          n,
          Expr::List(vec![Expr::List(d_vals.into()), Expr::Integer(0)].into()),
        ),
      ]
      .into(),
    ),
    Target::Dense => {
      let d_rows: Vec<Expr> = (0..n)
        .map(|i| {
          Expr::List(
            (0..n)
              .map(|j| {
                if i == j {
                  d_vals[i].clone()
                } else {
                  Expr::Integer(0)
                }
              })
              .collect::<Vec<_>>()
              .into(),
          )
        })
        .collect();
      Expr::List(vec![l_full, Expr::List(d_rows.into())].into())
    }
    Target::Sparse => {
      let mut l_nz: Vec<(usize, usize, Expr)> = Vec::new();
      for i in 0..n {
        for j in 0..=i {
          let v = if i == j {
            Expr::Integer(1)
          } else {
            match gq_to_expr(&l[i][j]) {
              Some(e) => e,
              None => return Ok(unevaluated()),
            }
          };
          if !(l[i][j].is_zero() && i != j) {
            l_nz.push((i, j, v));
          }
        }
      }
      let d_nz: Vec<(usize, usize, Expr)> = d_vals
        .iter()
        .enumerate()
        .map(|(i, v)| (i, i, v.clone()))
        .collect();
      Expr::List(
        vec![
          sparse_array_literal(n, &l_nz),
          sparse_array_literal(n, &d_nz),
        ]
        .into(),
      )
    }
  })
}

/// CoordinateTransform[src -> dst, pt] for the named coordinate systems
/// Polar/Cartesian (2D) and Spherical/Cylindrical/Cartesian (3D), in
/// wolframscript's conventions (e.g. ArcTan[x, y] for the polar angle,
/// Spherical as {r, theta, phi} with theta from the z axis).
pub fn coordinate_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("CoordinateTransform", args);
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let (src, dst) = match &args[0] {
    Expr::Rule {
      pattern,
      replacement,
    } => match (pattern.as_ref(), replacement.as_ref()) {
      (Expr::String(s), Expr::String(d)) => (s.clone(), d.clone()),
      _ => return Ok(unevaluated(args)),
    },
    Expr::FunctionCall { name, args: ra }
      if name == "Rule" && ra.len() == 2 =>
    {
      match (&ra[0], &ra[1]) {
        (Expr::String(s), Expr::String(d)) => (s.clone(), d.clone()),
        _ => return Ok(unevaluated(args)),
      }
    }
    _ => return Ok(unevaluated(args)),
  };
  let Expr::List(pt) = &args[1] else {
    return Ok(unevaluated(args));
  };

  let sq = |e: &Expr| pow2(e.clone(), Expr::Integer(2));
  let norm = |coords: &[&Expr]| {
    call("Sqrt", vec![plus(coords.iter().map(|c| sq(c)).collect())])
  };

  let components: Vec<Expr> = match (src.as_str(), dst.as_str(), pt.len()) {
    ("Polar", "Cartesian", 2) => {
      let (r, t) = (&pt[0], &pt[1]);
      vec![
        times(vec![r.clone(), call1("Cos", t.clone())]),
        times(vec![r.clone(), call1("Sin", t.clone())]),
      ]
    }
    ("Cartesian", "Polar", 2) => {
      let (x, y) = (&pt[0], &pt[1]);
      vec![norm(&[x, y]), call("ArcTan", vec![x.clone(), y.clone()])]
    }
    ("Spherical", "Cartesian", 3) => {
      let (r, t, p) = (&pt[0], &pt[1], &pt[2]);
      vec![
        times(vec![
          r.clone(),
          call1("Cos", p.clone()),
          call1("Sin", t.clone()),
        ]),
        times(vec![
          r.clone(),
          call1("Sin", p.clone()),
          call1("Sin", t.clone()),
        ]),
        times(vec![r.clone(), call1("Cos", t.clone())]),
      ]
    }
    ("Cartesian", "Spherical", 3) => {
      let (x, y, z) = (&pt[0], &pt[1], &pt[2]);
      vec![
        norm(&[x, y, z]),
        call("ArcTan", vec![z.clone(), norm(&[x, y])]),
        call("ArcTan", vec![x.clone(), y.clone()]),
      ]
    }
    ("Cylindrical", "Cartesian", 3) => {
      let (r, t, z) = (&pt[0], &pt[1], &pt[2]);
      vec![
        times(vec![r.clone(), call1("Cos", t.clone())]),
        times(vec![r.clone(), call1("Sin", t.clone())]),
        z.clone(),
      ]
    }
    ("Cartesian", "Cylindrical", 3) => {
      let (x, y, z) = (&pt[0], &pt[1], &pt[2]);
      vec![
        norm(&[x, y]),
        call("ArcTan", vec![x.clone(), y.clone()]),
        z.clone(),
      ]
    }
    _ => return Ok(unevaluated(args)),
  };

  let mut evaluated = Vec::with_capacity(components.len());
  for c in components {
    evaluated.push(evaluate_expr_to_expr(&c)?);
  }
  Ok(Expr::List(evaluated.into()))
}

/// CoordinateTransformData[src -> dst, property] and
/// CoordinateTransformData[src -> dst, property, pt] for the coordinate
/// systems of `CoordinateTransform`: "Mapping" (the transform itself),
/// "MappingJacobian" and "MappingJacobianDeterminant". Without a point the
/// property is a pure function of the coordinate list.
pub fn coordinate_transform_data_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("CoordinateTransformData", args));
  if args.len() != 2 && args.len() != 3 {
    return unevaluated();
  }
  let (
    Expr::Rule {
      pattern,
      replacement,
    },
    Expr::String(property),
  ) = (&args[0], &args[1])
  else {
    return unevaluated();
  };
  let dimension = |system: &Expr| match system {
    Expr::String(s) if s == "Polar" => Some(2),
    Expr::String(s) if s == "Spherical" || s == "Cylindrical" => Some(3),
    _ => None,
  };
  let Some(n) = dimension(pattern).or_else(|| dimension(replacement)) else {
    return unevaluated();
  };
  // The coordinates the property is computed in: fresh symbols, bound to
  // the given point or to the slots of a pure function afterwards.
  let symbols: Vec<Expr> = (1..=n)
    .map(|i| Expr::Identifier(format!("CoordinateTransformData`x$${i}")))
    .collect();
  let point = Expr::List(symbols.clone().into());
  let mapping = coordinate_transform_ast(&[args[0].clone(), point.clone()])?;
  if !matches!(mapping, Expr::List(_)) {
    return unevaluated();
  }
  let jacobian = || {
    evaluate_expr_to_expr(&call(
      "D",
      vec![mapping.clone(), Expr::List(vec![point.clone()].into())],
    ))
  };
  let value = match property.as_str() {
    "Mapping" => mapping.clone(),
    "MappingJacobian" => jacobian()?,
    "MappingJacobianDeterminant" => evaluate_expr_to_expr(&call(
      "Simplify",
      vec![call("Det", vec![jacobian()?])],
    ))?,
    _ => return unevaluated(),
  };
  let bind = |values: &[Expr]| -> Result<Expr, InterpreterError> {
    let mut bound = value.clone();
    for (sym, v) in symbols.iter().zip(values) {
      bound = crate::functions::polynomial_ast::solve::substitute_expr(
        &bound, sym, v,
      );
    }
    evaluate_expr_to_expr(&bound)
  };
  match args.get(2) {
    Some(Expr::List(pt)) if pt.len() == n => bind(pt),
    Some(_) => unevaluated(),
    None => {
      let slots: Vec<Expr> = (1..=n)
        .map(|i| Expr::Part {
          expr: Box::new(Expr::Slot(1)),
          index: Box::new(Expr::Integer(i as i128)),
        })
        .collect();
      // wolframscript's stored functions keep each product's coordinate
      // factors last: `{Cos[#1[[2]]]*#1[[1]], Sin[#1[[2]]]*#1[[1]]} &`.
      fn coordinates_last(e: &Expr) -> Expr {
        let is_coordinate = |f: &Expr| matches!(f, Expr::Part { expr, .. } if matches!(expr.as_ref(), Expr::Slot(1)));
        match e {
          Expr::FunctionCall { name, args } if name == "Times" => {
            let mut factors: Vec<Expr> =
              args.iter().map(coordinates_last).collect();
            factors.sort_by_key(is_coordinate);
            Expr::FunctionCall {
              name: name.clone(),
              args: factors.into(),
            }
          }
          Expr::FunctionCall { name, args } => Expr::FunctionCall {
            name: name.clone(),
            args: args.iter().map(coordinates_last).collect(),
          },
          Expr::List(items) => {
            Expr::List(items.iter().map(coordinates_last).collect())
          }
          Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
            op: *op,
            left: Box::new(coordinates_last(left)),
            right: Box::new(coordinates_last(right)),
          },
          Expr::UnaryOp { op, operand } => Expr::UnaryOp {
            op: *op,
            operand: Box::new(coordinates_last(operand)),
          },
          _ => e.clone(),
        }
      }
      Ok(Expr::Function {
        body: Box::new(coordinates_last(&bind(&slots)?)),
      })
    }
  }
}

/// HermiteDecomposition[m] — {u, h} with u unimodular, u.m == h, and h in
/// row Hermite normal form (positive pivots, entries above each pivot
/// reduced into [0, pivot)). Integer matrices only; anything else stays
/// unevaluated.
///
/// h is canonical and always matches wolframscript. u is only unique up to
/// kernel-row mixing for rank-deficient tall matrices; in a minority of
/// those cases wolframscript's internal operation sequence produces a
/// different (equally valid) u than the Euclidean min-pivot reduction
/// used here.
pub fn hermite_decomposition_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::functions::math_ast::{bigint_to_expr, expr_to_bigint};
  use num_traits::{One, Signed, Zero};
  let unevaluated = || unevaluated("HermiteDecomposition", args);
  // Parse an integer matrix; the elimination runs on BigInt because the
  // transformation matrix grows quickly.
  let rows = match &args[0] {
    Expr::List(rows) if !rows.is_empty() => rows,
    _ => return Ok(unevaluated()),
  };
  let mut m: Vec<Vec<BigInt>> = Vec::with_capacity(rows.len());
  let mut width: Option<usize> = None;
  for r in rows {
    let Expr::List(cells) = r else {
      return Ok(unevaluated());
    };
    if *width.get_or_insert(cells.len()) != cells.len() {
      return Ok(unevaluated());
    }
    let mut row = Vec::with_capacity(cells.len());
    for c in cells {
      match c {
        Expr::Integer(_) | Expr::BigInteger(_) => {
          row.push(expr_to_bigint(c).unwrap());
        }
        _ => return Ok(unevaluated()),
      }
    }
    m.push(row);
  }
  let nrows = m.len();
  let ncols = width.unwrap_or(0);
  if ncols == 0 {
    return Ok(unevaluated());
  }

  // Euclidean quotient: a - q*p lies in [0, |p|)
  let div_euclid = |a: &BigInt, p: &BigInt| -> BigInt {
    let mut q = a / p;
    if (a - &q * p).is_negative() {
      if p.is_positive() {
        q -= BigInt::one();
      } else {
        q += BigInt::one();
      }
    }
    q
  };

  // u starts as the identity and tracks every row operation
  let mut u: Vec<Vec<BigInt>> = (0..nrows)
    .map(|i| (0..nrows).map(|j| BigInt::from(u8::from(i == j))).collect())
    .collect();
  let mut h = m;

  let mut pivot_row = 0usize;
  for col in 0..ncols {
    if pivot_row >= nrows {
      break;
    }
    // Reduce column `col` among rows >= pivot_row to a single nonzero
    // entry via Euclidean row operations
    loop {
      // Row with the smallest nonzero |entry| in this column
      let mut best: Option<usize> = None;
      for r in pivot_row..nrows {
        if !h[r][col].is_zero()
          && best.is_none_or(|b: usize| h[r][col].abs() < h[b][col].abs())
        {
          best = Some(r);
        }
      }
      let Some(b) = best else {
        break;
      };
      if b != pivot_row {
        h.swap(b, pivot_row);
        u.swap(b, pivot_row);
      }
      let p = h[pivot_row][col].clone();
      let mut done = true;
      for r in (pivot_row + 1)..nrows {
        if !h[r][col].is_zero() {
          let q = div_euclid(&h[r][col], &p);
          for c in 0..ncols {
            let t = &q * &h[pivot_row][c];
            h[r][c] -= t;
          }
          for c in 0..nrows {
            let t = &q * &u[pivot_row][c];
            u[r][c] -= t;
          }
          if !h[r][col].is_zero() {
            done = false;
          }
        }
      }
      if done {
        break;
      }
    }
    if h[pivot_row][col].is_zero() {
      continue;
    }
    // Positive pivot
    if h[pivot_row][col].is_negative() {
      for x in &mut h[pivot_row] {
        *x = -&*x;
      }
      for x in &mut u[pivot_row] {
        *x = -&*x;
      }
    }
    // Reduce the entries above the pivot into [0, pivot)
    let p = h[pivot_row][col].clone();
    for r in 0..pivot_row {
      let q = div_euclid(&h[r][col], &p);
      if !q.is_zero() {
        for c in 0..ncols {
          let t = &q * &h[pivot_row][c];
          h[r][c] -= t;
        }
        for c in 0..nrows {
          let t = &q * &u[pivot_row][c];
          u[r][c] -= t;
        }
      }
    }
    pivot_row += 1;
  }

  let to_matrix = |m: Vec<Vec<BigInt>>| {
    Expr::List(
      m.into_iter()
        .map(|row| {
          Expr::List(
            row
              .into_iter()
              .map(bigint_to_expr)
              .collect::<Vec<_>>()
              .into(),
          )
        })
        .collect::<Vec<_>>()
        .into(),
    )
  };
  Ok(Expr::List(vec![to_matrix(u), to_matrix(h)].into()))
}

// ─── Modulus -> m support for Det / Inverse / RowReduce / NullSpace /
//     MatrixRank / LinearSolve ────────────────────────────────────────

/// Extract `Modulus -> m` (m >= 0) from an option argument.
pub fn extract_modulus_option_la(opt: &Expr) -> Option<i128> {
  let value = |v: &Expr| crate::functions::math_ast::expr_to_i128(v);
  if let Expr::Rule {
    pattern,
    replacement,
  } = opt
    && let Expr::Identifier(s) = pattern.as_ref()
    && s == "Modulus"
  {
    return value(replacement).filter(|&m| m >= 0);
  }
  if let Expr::FunctionCall { name, args } = opt
    && (name == "Rule" || name == "RuleDelayed")
    && args.len() == 2
    && let Expr::Identifier(s) = &args[0]
    && s == "Modulus"
  {
    return value(&args[1]).filter(|&m| m >= 0);
  }
  None
}

/// Rectangular non-empty integer matrix, or None.
fn la_int_matrix(expr: &Expr) -> Option<Vec<Vec<i128>>> {
  use crate::functions::math_ast::expr_to_i128;
  let Expr::List(rows) = expr else {
    return None;
  };
  if rows.is_empty() {
    return None;
  }
  let mut out = Vec::with_capacity(rows.len());
  let mut width = None;
  for row in rows {
    let Expr::List(cells) = row else {
      return None;
    };
    if *width.get_or_insert(cells.len()) != cells.len() || cells.is_empty() {
      return None;
    }
    let r: Option<Vec<i128>> = cells.iter().map(expr_to_i128).collect();
    out.push(r?);
  }
  Some(out)
}

fn la_int_vector(expr: &Expr) -> Option<Vec<i128>> {
  use crate::functions::math_ast::expr_to_i128;
  let Expr::List(cells) = expr else {
    return None;
  };
  if cells.is_empty() || cells.iter().any(|c| matches!(c, Expr::List(_))) {
    return None;
  }
  cells.iter().map(expr_to_i128).collect()
}

fn la_is_prime(m: i128) -> bool {
  if m < 2 {
    return false;
  }
  let mut d = 2i128;
  while d * d <= m {
    if m % d == 0 {
      return false;
    }
    d += 1;
  }
  true
}

fn la_mod_inv(a: i128, m: i128) -> i128 {
  let (mut old_r, mut r) = (a.rem_euclid(m), m);
  let (mut old_s, mut s) = (1i128, 0i128);
  while r != 0 {
    let q = old_r / r;
    (old_r, r) = (r, old_r - q * r);
    (old_s, s) = (s, old_s - q * s);
  }
  old_s.rem_euclid(m)
}

/// `a * b mod m` for reduced operands, without i128 overflow.
fn la_mul_mod(a: i128, b: i128, m: i128) -> i128 {
  if let Some(p) = a.checked_mul(b) {
    p.rem_euclid(m)
  } else {
    use num_bigint::BigInt;
    use num_traits::ToPrimitive;
    let p = (BigInt::from(a) * BigInt::from(b)) % BigInt::from(m);
    p.to_i128().unwrap_or(0).rem_euclid(m)
  }
}

/// Determinant over Z/m (any modulus m >= 1) by Euclidean row reduction,
/// which needs no inverses and keeps every entry in [0, m).
fn la_det_mod(mat: &[Vec<i128>], modulus: i128) -> i128 {
  let n = mat.len();
  let mut m: Vec<Vec<i128>> = mat
    .iter()
    .map(|r| r.iter().map(|c| c.rem_euclid(modulus)).collect())
    .collect();
  let mut det = 1i128 % modulus;
  for k in 0..n {
    loop {
      // Smallest nonzero entry of column k (rows k..) becomes the pivot.
      let Some(pr) = (k..n).filter(|&r| m[r][k] != 0).min_by_key(|&r| m[r][k])
      else {
        return 0;
      };
      if pr != k {
        m.swap(k, pr);
        det = (modulus - det) % modulus;
      }
      let mut done = true;
      for i in (k + 1)..n {
        if m[i][k] != 0 {
          let q = m[i][k] / m[k][k];
          for j in k..n {
            let t = la_mul_mod(q, m[k][j], modulus);
            m[i][j] = (m[i][j] - t).rem_euclid(modulus);
          }
          if m[i][k] != 0 {
            done = false;
          }
        }
      }
      if done {
        break;
      }
    }
    det = la_mul_mod(det, m[k][k], modulus);
  }
  det
}

fn la_matrix_expr(rows: &[Vec<i128>]) -> Expr {
  Expr::List(
    rows
      .iter()
      .map(|r| {
        Expr::List(
          r.iter()
            .map(|&c| Expr::Integer(c))
            .collect::<Vec<_>>()
            .into(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  )
}

fn la_vector_expr(v: &[i128]) -> Expr {
  Expr::List(
    v.iter()
      .map(|&c| Expr::Integer(c))
      .collect::<Vec<_>>()
      .into(),
  )
}

/// Gauss-Jordan over Z/m with invertible pivots (always succeeds for
/// prime m); returns pivot columns or None when a column has nonzero
/// entries but no invertible one (possible only for composite m).
fn la_rref(mat: &mut [Vec<i128>], m: i128) -> Option<Vec<usize>> {
  let rows = mat.len();
  let cols = mat[0].len();
  for row in mat.iter_mut() {
    for c in row.iter_mut() {
      *c = c.rem_euclid(m);
    }
  }
  let mut pivots = Vec::new();
  let mut rank = 0usize;
  for col in 0..cols {
    if rank == rows {
      break;
    }
    let Some(pr) = (rank..rows).find(|&r| gcd_i128(mat[r][col], m) == 1) else {
      if (rank..rows).any(|r| mat[r][col] != 0) {
        return None;
      }
      continue;
    };
    mat.swap(rank, pr);
    let inv = la_mod_inv(mat[rank][col], m);
    for j in 0..cols {
      mat[rank][j] = la_mul_mod(mat[rank][j], inv, m);
    }
    for r in 0..rows {
      if r != rank && mat[r][col] != 0 {
        let f = mat[r][col];
        for j in 0..cols {
          mat[r][j] =
            (mat[r][j] - la_mul_mod(f, mat[rank][j], m)).rem_euclid(m);
        }
      }
    }
    pivots.push(col);
    rank += 1;
  }
  Some(pivots)
}

fn la_unevaluated(name: &str, args: &[Expr]) -> Expr {
  unevaluated(name, args)
}

/// Det[m, Modulus -> p]
pub fn det_modulus_ast(
  args: &[Expr],
  m: i128,
) -> Result<Expr, InterpreterError> {
  let Some(mat) = la_int_matrix(&args[0]) else {
    return Ok(la_unevaluated("Det", args));
  };
  if mat.len() != mat[0].len() {
    crate::emit_message(&format!(
      "Det::matsq: Argument {} at position 1 is not a nonempty square matrix.",
      expr_to_string(&args[0])
    ));
    return Ok(la_unevaluated("Det", args));
  }
  Ok(Expr::Integer(la_det_mod(&mat, m)))
}

/// Inverse[m, Modulus -> p] via the adjugate, which works for any
/// modulus coprime to the determinant.
pub fn inverse_modulus_ast(
  args: &[Expr],
  m: i128,
) -> Result<Expr, InterpreterError> {
  let Some(mat) = la_int_matrix(&args[0]) else {
    return Ok(la_unevaluated("Inverse", args));
  };
  let n = mat.len();
  if n != mat[0].len() {
    return Ok(la_unevaluated("Inverse", args));
  }
  let reduced: Vec<Vec<i128>> = mat
    .iter()
    .map(|r| r.iter().map(|c| c.rem_euclid(m)).collect())
    .collect();
  let det = la_det_mod(&reduced, m);
  let g = gcd_i128(det, m);
  if g != 1 {
    if la_is_prime(m) {
      crate::emit_message(&format!(
        "Inverse::sing: Matrix {} is singular.",
        expr_to_string(&la_matrix_expr(&reduced))
      ));
    } else {
      crate::emit_message(&format!(
        "Inverse::nmod: 1/{g} cannot be inverted in non-prime modulus {m}."
      ));
    }
    return Ok(la_unevaluated("Inverse", args));
  }
  let inv_det = la_mod_inv(det, m);
  let mut out = vec![vec![0i128; n]; n];
  for i in 0..n {
    for j in 0..n {
      // Cofactor C[i][j] lands at out[j][i] (adjugate transpose).
      let minor: Vec<Vec<i128>> = reduced
        .iter()
        .enumerate()
        .filter(|(r, _)| *r != i)
        .map(|(_, row)| {
          row
            .iter()
            .enumerate()
            .filter(|(c, _)| *c != j)
            .map(|(_, &v)| v)
            .collect()
        })
        .collect();
      let mut cof = if n == 1 { 1 } else { la_det_mod(&minor, m) };
      if (i + j) % 2 == 1 {
        cof = -cof;
      }
      out[j][i] = la_mul_mod(cof.rem_euclid(m), inv_det, m);
    }
  }
  Ok(la_matrix_expr(&out))
}

/// RowReduce[m, Modulus -> p]; wolframscript insists on a prime modulus.
pub fn row_reduce_modulus_ast(
  args: &[Expr],
  m: i128,
) -> Result<Expr, InterpreterError> {
  let Some(mut mat) = la_int_matrix(&args[0]) else {
    return Ok(la_unevaluated("RowReduce", args));
  };
  if !la_is_prime(m) {
    crate::emit_message(&format!(
      "RowReduce::nmod: {} cannot be reduced in non-prime modulus {m}.",
      expr_to_string(&args[0])
    ));
    return Ok(la_unevaluated("RowReduce", args));
  }
  la_rref(&mut mat, m);
  Ok(la_matrix_expr(&mat))
}

/// MatrixRank[m, Modulus -> p]; non-prime moduli emit `modp`.
pub fn matrix_rank_modulus_ast(
  args: &[Expr],
  m: i128,
) -> Result<Expr, InterpreterError> {
  let Some(mut mat) = la_int_matrix(&args[0]) else {
    return Ok(la_unevaluated("MatrixRank", args));
  };
  if !la_is_prime(m) {
    crate::emit_message(&format!(
      "MatrixRank::modp: The value of the option Modulus -> {m} should be \
       a prime number or zero."
    ));
    return Ok(la_unevaluated("MatrixRank", args));
  }
  let pivots = la_rref(&mut mat, m).expect("prime modulus always reduces");
  Ok(Expr::Integer(pivots.len() as i128))
}

/// NullSpace[m, Modulus -> p]. Basis vectors take the free columns in
/// reverse order with the free variable set to 1, like wolframscript.
pub fn null_space_modulus_ast(
  args: &[Expr],
  m: i128,
) -> Result<Expr, InterpreterError> {
  let Some(mut mat) = la_int_matrix(&args[0]) else {
    return Ok(la_unevaluated("NullSpace", args));
  };
  let cols = mat[0].len();
  let Some(pivots) = la_rref(&mut mat, m) else {
    return Ok(la_unevaluated("NullSpace", args));
  };
  let mut basis = Vec::new();
  for free in (0..cols).rev() {
    if pivots.contains(&free) {
      continue;
    }
    let mut v = vec![0i128; cols];
    v[free] = 1;
    for (r, &pc) in pivots.iter().enumerate() {
      v[pc] = (-mat[r][free]).rem_euclid(m);
    }
    basis.push(la_vector_expr(&v));
  }
  Ok(Expr::List(basis.into()))
}

/// LinearSolve[a, b, Modulus -> p] for a vector or matrix right-hand
/// side; inconsistent systems emit `nosol`, free variables become 0.
pub fn linear_solve_modulus_ast(
  args: &[Expr],
  m: i128,
) -> Result<Expr, InterpreterError> {
  let Some(a) = la_int_matrix(&args[0]) else {
    return Ok(la_unevaluated("LinearSolve", args));
  };
  if !la_is_prime(m) {
    return Ok(la_unevaluated("LinearSolve", args));
  }
  let (b_cols, matrix_rhs): (Vec<Vec<i128>>, bool) =
    if let Some(v) = la_int_vector(&args[1]) {
      (vec![v], false)
    } else if let Some(bm) = la_int_matrix(&args[1]) {
      // Columns of the right-hand matrix.
      let cols = bm[0].len();
      (
        (0..cols)
          .map(|c| bm.iter().map(|row| row[c]).collect())
          .collect(),
        true,
      )
    } else {
      return Ok(la_unevaluated("LinearSolve", args));
    };
  let rows = a.len();
  let cols = a[0].len();
  if b_cols.iter().any(|b| b.len() != rows) {
    return Ok(la_unevaluated("LinearSolve", args));
  }
  let mut solutions: Vec<Vec<i128>> = Vec::new();
  for b in &b_cols {
    let mut aug: Vec<Vec<i128>> = a
      .iter()
      .zip(b.iter())
      .map(|(row, &bv)| {
        let mut r = row.clone();
        r.push(bv);
        r
      })
      .collect();
    let pivots = la_rref(&mut aug, m).expect("prime modulus always reduces");
    if pivots.contains(&cols) {
      crate::emit_message(
        "LinearSolve::nosol: Linear equation encountered that has no \
         solution.",
      );
      return Ok(la_unevaluated("LinearSolve", args));
    }
    let mut x = vec![0i128; cols];
    for (r, &pc) in pivots.iter().enumerate() {
      x[pc] = aug[r][cols];
    }
    solutions.push(x);
  }
  if matrix_rhs {
    // Solutions are columns of the result.
    let n = solutions[0].len();
    let rows: Vec<Vec<i128>> = (0..n)
      .map(|i| solutions.iter().map(|s| s[i]).collect())
      .collect();
    Ok(la_matrix_expr(&rows))
  } else {
    Ok(la_vector_expr(&solutions[0]))
  }
}

/// Least-squares Savitzky-Golay smoothing kernel of radius `r` and polynomial
/// degree `k` over a window of `2 r + 1` points. Returns the machine-precision
/// weights that evaluate the fitted polynomial at the window centre.
fn sg_kernel(r: usize, k: usize) -> Option<Vec<f64>> {
  let m = 2 * r + 1;
  let ncols = (k + 1).min(m);
  let offs: Vec<f64> = (0..m).map(|i| i as f64 - r as f64).collect();
  // Normal-equations matrix B = A^T A with B[p][q] = sum_i offs_i^(p+q).
  let mut mat = vec![vec![0.0f64; ncols]; ncols];
  for (p, row) in mat.iter_mut().enumerate() {
    for (q, cell) in row.iter_mut().enumerate() {
      *cell = offs.iter().map(|&o| o.powi((p + q) as i32)).sum();
    }
  }
  // Solve B x = e_0 (the first row of B^-1) with partial pivoting.
  let mut b = vec![0.0f64; ncols];
  b[0] = 1.0;
  for col in 0..ncols {
    let mut piv = col;
    for row in (col + 1)..ncols {
      if mat[row][col].abs() > mat[piv][col].abs() {
        piv = row;
      }
    }
    if mat[piv][col].abs() < 1e-15 {
      return None;
    }
    mat.swap(col, piv);
    b.swap(col, piv);
    let d = mat[col][col];
    for row in (col + 1)..ncols {
      let f = mat[row][col] / d;
      for c in col..ncols {
        mat[row][c] -= f * mat[col][c];
      }
      b[row] -= f * b[col];
    }
  }
  let mut x = vec![0.0f64; ncols];
  for i in (0..ncols).rev() {
    let mut s = b[i];
    for j in (i + 1)..ncols {
      s -= mat[i][j] * x[j];
    }
    x[i] = s / mat[i][i];
  }
  // kernel[j] = sum_p x[p] * offs_j^p.
  Some(
    offs
      .iter()
      .map(|&o| (0..ncols).map(|p| x[p] * o.powi(p as i32)).sum())
      .collect(),
  )
}

fn sg_outer(rows: &[f64], cols: &[f64]) -> Expr {
  Expr::List(
    rows
      .iter()
      .map(|&a| {
        Expr::List(
          cols
            .iter()
            .map(|&b| Expr::Real(a * b))
            .collect::<Vec<_>>()
            .into(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  )
}

/// The exact 1D Savitzky–Golay smoothing kernel of radius `r` and degree `k`
/// (WorkingPrecision -> Infinity): the first row of (AᵀA)⁻¹Aᵀ for the
/// Vandermonde matrix A[i][p] = (i - r)^p, computed with exact rationals.
fn sg_kernel_exact(r: usize, k: usize) -> Result<Expr, InterpreterError> {
  use crate::evaluator::evaluate_function_call_ast as eval;
  let m = 2 * r + 1;
  let ncols = (k + 1).min(m);
  let offs: Vec<i128> = (0..m).map(|i| i as i128 - r as i128).collect();
  let int_list =
    |v: Vec<i128>| Expr::List(v.into_iter().map(Expr::Integer).collect());
  let vander = Expr::List(
    offs
      .iter()
      .map(|&o| int_list((0..ncols).map(|p| o.pow(p as u32)).collect()))
      .collect(),
  );
  let normal = Expr::List(
    (0..ncols)
      .map(|p| {
        int_list(
          (0..ncols)
            .map(|q| offs.iter().map(|&o| o.pow((p + q) as u32)).sum())
            .collect(),
        )
      })
      .collect(),
  );
  let e0 = int_list((0..ncols).map(|p| i128::from(p == 0)).collect());
  let coeffs = eval("LinearSolve", &[normal, e0])?;
  eval("Dot", &[vander, coeffs])
}

/// SavitzkyGolayMatrix[spec, k, WorkingPrecision -> Infinity] — the exact
/// kernel for the same specifications as savitzky_golay_matrix_ast.
pub fn savitzky_golay_matrix_exact_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("SavitzkyGolayMatrix", args));
  let as_int = |e: &Expr| -> Option<usize> {
    match e {
      Expr::Integer(n) if *n >= 0 => Some(*n as usize),
      _ => None,
    }
  };
  let outer = |a: Expr, b: Expr| {
    crate::evaluator::evaluate_function_call_ast(
      "Outer",
      &[Expr::Identifier("Times".to_string()), a, b],
    )
  };
  match (&args[0], &args[1]) {
    (Expr::List(rs), kexpr) if rs.len() == 1 => {
      let (Some(r), Some(k)) = (as_int(&rs[0]), as_int(kexpr)) else {
        return unevaluated();
      };
      sg_kernel_exact(r, k)
    }
    (Expr::List(rs), Expr::List(ks)) if rs.len() == 2 && ks.len() == 2 => {
      let (Some(r1), Some(r2), Some(k1), Some(k2)) = (
        as_int(&rs[0]),
        as_int(&rs[1]),
        as_int(&ks[0]),
        as_int(&ks[1]),
      ) else {
        return unevaluated();
      };
      outer(sg_kernel_exact(r1, k1)?, sg_kernel_exact(r2, k2)?)
    }
    (rexpr, kexpr) => {
      let (Some(r), Some(k)) = (as_int(rexpr), as_int(kexpr)) else {
        return unevaluated();
      };
      let ker = sg_kernel_exact(r, k)?;
      outer(ker.clone(), ker)
    }
  }
}

/// SavitzkyGolayMatrix[{r}, k] — the 1D smoothing kernel (a vector).
/// SavitzkyGolayMatrix[r, k] — the isotropic 2D kernel (outer product of the
/// 1D kernel with itself). SavitzkyGolayMatrix[{r1, r2}, {k1, k2}] — the
/// anisotropic 2D kernel.
pub fn savitzky_golay_matrix_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("SavitzkyGolayMatrix", args));
  if args.len() != 2 {
    return unevaluated();
  }
  let as_int = |e: &Expr| -> Option<usize> {
    match e {
      Expr::Integer(n) if *n >= 0 => Some(*n as usize),
      _ => None,
    }
  };
  match (&args[0], &args[1]) {
    // 1D kernel: {r}, k
    (Expr::List(rs), kexpr) if rs.len() == 1 => {
      let (Some(r), Some(k)) = (as_int(&rs[0]), as_int(kexpr)) else {
        return unevaluated();
      };
      match sg_kernel(r, k) {
        Some(ker) => Ok(Expr::List(
          ker.into_iter().map(Expr::Real).collect::<Vec<_>>().into(),
        )),
        None => unevaluated(),
      }
    }
    // Anisotropic 2D kernel: {r1, r2}, {k1, k2}
    (Expr::List(rs), Expr::List(ks)) if rs.len() == 2 && ks.len() == 2 => {
      let (Some(r1), Some(r2), Some(k1), Some(k2)) = (
        as_int(&rs[0]),
        as_int(&rs[1]),
        as_int(&ks[0]),
        as_int(&ks[1]),
      ) else {
        return unevaluated();
      };
      match (sg_kernel(r1, k1), sg_kernel(r2, k2)) {
        (Some(kr), Some(kc)) => Ok(sg_outer(&kr, &kc)),
        _ => unevaluated(),
      }
    }
    // Isotropic 2D kernel: r, k
    (rexpr, kexpr) => {
      let (Some(r), Some(k)) = (as_int(rexpr), as_int(kexpr)) else {
        return unevaluated();
      };
      match sg_kernel(r, k) {
        Some(ker) => Ok(sg_outer(&ker, &ker)),
        None => unevaluated(),
      }
    }
  }
}

// ── Schur decomposition ─────────────────────────────────────────────────

/// In-place Householder reduction to upper Hessenberg form, accumulating the
/// orthogonal transformation into `q` so that the original matrix is
/// `q · h · qᵀ`. Columns that are already reduced are skipped, so a
/// triangular input passes through untouched and keeps `q` the identity.
fn hessenberg_reduce_accumulating(
  a: &mut [Vec<f64>],
  q: &mut [Vec<f64>],
  n: usize,
) {
  for k in 0..n.saturating_sub(2) {
    let norm_x: f64 = (k + 1..n).map(|i| a[i][k] * a[i][k]).sum::<f64>().sqrt();
    if norm_x == 0.0 || (k + 2..n).all(|i| a[i][k] == 0.0) {
      continue;
    }
    let alpha = if a[k + 1][k] >= 0.0 { -norm_x } else { norm_x };
    let m = n - k - 1;
    let mut v: Vec<f64> = (k + 1..n).map(|i| a[i][k]).collect();
    v[0] -= alpha;
    let vnorm2: f64 = v.iter().map(|x| x * x).sum();
    if vnorm2 == 0.0 {
      continue;
    }
    // A ← H A
    for j in 0..n {
      let dot: f64 = (0..m).map(|t| v[t] * a[k + 1 + t][j]).sum();
      let f = 2.0 * dot / vnorm2;
      for t in 0..m {
        a[k + 1 + t][j] -= f * v[t];
      }
    }
    // A ← A H and Q ← Q H (the reflection is its own inverse)
    for row in a.iter_mut().take(n) {
      let dot: f64 = (0..m).map(|t| row[k + 1 + t] * v[t]).sum();
      let f = 2.0 * dot / vnorm2;
      for t in 0..m {
        row[k + 1 + t] -= f * v[t];
      }
    }
    for row in q.iter_mut().take(n) {
      let dot: f64 = (0..m).map(|t| row[k + 1 + t] * v[t]).sum();
      let f = 2.0 * dot / vnorm2;
      for t in 0..m {
        row[k + 1 + t] -= f * v[t];
      }
    }
    a[k + 1][k] = alpha;
    for i in k + 2..n {
      a[i][k] = 0.0;
    }
  }
}

/// Apply the Givens rotation `(cs, sn)` acting on rows/columns `i` and `i + 1`
/// to `h` (both sides) and to the accumulated `q` (right side only).
fn apply_givens(
  h: &mut [Vec<f64>],
  q: &mut [Vec<f64>],
  n: usize,
  i: usize,
  cs: f64,
  sn: f64,
) {
  // Rows i, i+1 of H ← Gᵀ H
  for col in 0..n {
    let (a, b) = (h[i][col], h[i + 1][col]);
    h[i][col] = cs * a + sn * b;
    h[i + 1][col] = -sn * a + cs * b;
  }
  // Columns i, i+1 of H ← H G, and the same on Q
  for row in h.iter_mut().take(n) {
    let (a, b) = (row[i], row[i + 1]);
    row[i] = cs * a + sn * b;
    row[i + 1] = -sn * a + cs * b;
  }
  for row in q.iter_mut().take(n) {
    let (a, b) = (row[i], row[i + 1]);
    row[i] = cs * a + sn * b;
    row[i + 1] = -sn * a + cs * b;
  }
}

/// Split a 2×2 block with real eigenvalues into an upper-triangular one by
/// rotating its first column onto an eigenvector.
fn split_two_by_two(
  h: &mut [Vec<f64>],
  q: &mut [Vec<f64>],
  n: usize,
  i: usize,
) -> bool {
  let (a, b, c, d) = (h[i][i], h[i][i + 1], h[i + 1][i], h[i + 1][i + 1]);
  let disc = (a - d) * (a - d) + 4.0 * b * c;
  if disc < 0.0 {
    // A complex pair stays as a real 2×2 block.
    return false;
  }
  let sq = disc.sqrt();
  // The eigenvalue nearer d keeps the block well conditioned.
  let lambda = if (a + d + sq) / 2.0 - d < d - (a + d - sq) / 2.0 {
    (a + d + sq) / 2.0
  } else {
    (a + d - sq) / 2.0
  };
  let (x, y) = if b.abs() > (lambda - a).abs().max(c.abs()) {
    (b, lambda - a)
  } else {
    (lambda - d, c)
  };
  let norm = (x * x + y * y).sqrt();
  if norm == 0.0 {
    return true;
  }
  apply_givens(h, q, n, i, x / norm, y / norm);
  h[i + 1][i] = 0.0;
  true
}

/// Real Schur decomposition `a = q · t · qᵀ`: `q` orthogonal and `t`
/// quasi-upper-triangular (1×1 blocks for real eigenvalues, 2×2 for a complex
/// pair). Returns None when the shifted QR iteration does not converge.
fn real_schur(a: &[Vec<f64>]) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
  let n = a.len();
  let mut h = a.to_vec();
  let mut q: Vec<Vec<f64>> = (0..n)
    .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
    .collect();
  hessenberg_reduce_accumulating(&mut h, &mut q, n);

  let mut hi = n;
  let mut iterations = 0usize;
  while hi > 1 {
    // Deflate every negligible subdiagonal entry, then find the start of the
    // active unreduced block.
    for i in 1..hi {
      if h[i][i - 1].abs()
        <= 1e-14 * (h[i][i].abs() + h[i - 1][i - 1].abs()).max(1e-300)
      {
        h[i][i - 1] = 0.0;
      }
    }
    let mut lo = hi - 1;
    while lo > 0 && h[lo][lo - 1] != 0.0 {
      lo -= 1;
    }
    if lo == hi - 1 {
      hi -= 1;
      iterations = 0;
      continue;
    }
    if lo == hi - 2 {
      // A trailing 2×2 either splits or stays as a real block.
      split_two_by_two(&mut h, &mut q, n, lo);
      hi -= 2;
      iterations = 0;
      continue;
    }
    iterations += 1;
    if iterations > 200 {
      return None;
    }
    // Wilkinson shift from the trailing 2×2 of the active block; a complex
    // pair contributes its real part, and every tenth step takes an
    // exceptional shift to break cycles.
    let (a1, b1, c1, d1) = (
      h[hi - 2][hi - 2],
      h[hi - 2][hi - 1],
      h[hi - 1][hi - 2],
      h[hi - 1][hi - 1],
    );
    let disc = (a1 - d1) * (a1 - d1) + 4.0 * b1 * c1;
    let mut mu = if disc >= 0.0 {
      let sq = disc.sqrt();
      let l1 = (a1 + d1 + sq) / 2.0;
      let l2 = (a1 + d1 - sq) / 2.0;
      if (l1 - d1).abs() < (l2 - d1).abs() {
        l1
      } else {
        l2
      }
    } else {
      f64::midpoint(a1, d1)
    };
    if iterations.is_multiple_of(10) {
      mu += h[hi - 1][hi - 2].abs();
    }
    // One explicit QR step on H − μI, rotation by rotation.
    for i in lo..hi {
      h[i][i] -= mu;
    }
    let mut rotations: Vec<(usize, f64, f64)> = Vec::new();
    for i in lo..hi - 1 {
      let (x, y) = (h[i][i], h[i + 1][i]);
      let norm = (x * x + y * y).sqrt();
      let (cs, sn) = if norm == 0.0 {
        (1.0, 0.0)
      } else {
        (x / norm, y / norm)
      };
      // Rows i, i+1 ← Gᵀ H
      for col in 0..n {
        let (p, r) = (h[i][col], h[i + 1][col]);
        h[i][col] = cs * p + sn * r;
        h[i + 1][col] = -sn * p + cs * r;
      }
      rotations.push((i, cs, sn));
    }
    // H ← R G, accumulating the same rotations into Q.
    for (i, cs, sn) in rotations {
      for row in h.iter_mut().take(n) {
        let (p, r) = (row[i], row[i + 1]);
        row[i] = cs * p + sn * r;
        row[i + 1] = -sn * p + cs * r;
      }
      for row in q.iter_mut().take(n) {
        let (p, r) = (row[i], row[i + 1]);
        row[i] = cs * p + sn * r;
        row[i + 1] = -sn * p + cs * r;
      }
    }
    for i in lo..hi {
      h[i][i] += mu;
    }
  }

  // Below the (quasi-)diagonal everything is zero by construction.
  for i in 0..n {
    for j in 0..i.saturating_sub(1) {
      h[i][j] = 0.0;
    }
  }
  Some((q, h))
}

/// The square machine-precision matrix a `head[m]` decomposition works on.
/// Emits `head::matsq` for a non-square argument and `head::schurf` for an
/// exact one, returning `None` in both cases.
fn machine_square_matrix(head: &str, arg: &Expr) -> Option<Vec<Vec<f64>>> {
  let matsq = || {
    crate::emit_message(&format!(
      "{head}::matsq: Argument {} at position 1 is not a nonempty square \
       matrix.",
      expr_to_output(arg)
    ));
    None
  };
  let Expr::List(rows) = arg else {
    return matsq();
  };
  let n = rows.len();
  if n == 0 {
    return matsq();
  }
  let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(n);
  let mut exact = true;
  for row in rows {
    let Expr::List(entries) = row else {
      return matsq();
    };
    if entries.len() != n {
      return matsq();
    }
    let mut values = Vec::with_capacity(n);
    for entry in entries {
      if matches!(entry, Expr::Real(_) | Expr::BigFloat(..)) {
        exact = false;
      }
      match crate::functions::math_ast::expr_to_num(entry) {
        Some(v) => values.push(v),
        None => return matsq(),
      }
    }
    matrix.push(values);
  }
  // An exact matrix has no machine-precision decomposition to compute.
  if exact {
    crate::emit_message(&format!(
      "{head}::schurf: {head} has received a matrix with infinite precision."
    ));
    return None;
  }
  Some(matrix)
}

fn real_matrix_expr(m: Vec<Vec<f64>>) -> Expr {
  Expr::List(
    m.into_iter()
      .map(|row| Expr::List(row.into_iter().map(Expr::Real).collect()))
      .collect(),
  )
}

/// `SchurDecomposition[m]` — the real Schur decomposition `{q, t}` of a
/// square machine-precision matrix, with `m == q · t · Transpose[q]`.
pub fn schur_decomposition_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("SchurDecomposition", args);
  if args.len() != 1 {
    return Ok(original());
  }
  let Some(matrix) = machine_square_matrix("SchurDecomposition", &args[0])
  else {
    return Ok(original());
  };
  let Some((q, t)) = real_schur(&matrix) else {
    return Ok(original());
  };
  Ok(Expr::List(
    vec![real_matrix_expr(q), real_matrix_expr(t)].into(),
  ))
}

/// `HessenbergDecomposition[m]` — `{p, h}` with `p` orthogonal, `h` upper
/// Hessenberg and `m == p · h · Transpose[p]`, by Householder reflections
/// (LAPACK's sign convention, as wolframscript uses).
pub fn hessenberg_decomposition_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("HessenbergDecomposition", args);
  if args.len() != 1 {
    return Ok(original());
  }
  let Some(mut h) = machine_square_matrix("HessenbergDecomposition", &args[0])
  else {
    return Ok(original());
  };
  let n = h.len();
  let mut p: Vec<Vec<f64>> = (0..n)
    .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
    .collect();
  hessenberg_reduce_accumulating(&mut h, &mut p, n);
  Ok(Expr::List(
    vec![real_matrix_expr(p), real_matrix_expr(h)].into(),
  ))
}

/// How the entries of a `SymmetrizedArray` transform under the permutations of
/// the tensor slots the symmetry names.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SymmetryKind {
  /// Every permutation of the slots leaves the entry unchanged.
  Symmetric,
  /// An odd permutation of the slots negates the entry.
  Antisymmetric,
  /// An odd permutation of the slots conjugates the entry.
  Hermitian,
  /// Every entry is zero.
  ZeroSymmetric,
}

impl SymmetryKind {
  fn head(self) -> &'static str {
    match self {
      Self::Symmetric => "Symmetric",
      Self::Antisymmetric => "Antisymmetric",
      Self::Hermitian => "Hermitian",
      Self::ZeroSymmetric => "ZeroSymmetric",
    }
  }

  fn from_head(head: &str) -> Option<Self> {
    match head {
      "Symmetric" => Some(Self::Symmetric),
      "Antisymmetric" => Some(Self::Antisymmetric),
      "Hermitian" => Some(Self::Hermitian),
      "ZeroSymmetric" => Some(Self::ZeroSymmetric),
      _ => None,
    }
  }
}

/// Why a symmetry specification was refused, and which message wolframscript
/// emits for it.
enum SymmetryError {
  /// `Symmetrize::symm` — not a symmetry at all, or a repeated slot.
  Invalid,
  /// `Symmetrize::symmrank` — the slot lies beyond the tensor's rank.
  BeyondRank(usize),
}

/// Parse a symmetry specification for a rank-`rank` tensor into its kind and
/// its 1-based slots. `None` is the `Symmetrize` default: full `Symmetric`
/// symmetry over every slot. The slots come back sorted, matching
/// wolframscript's canonicalisation of `Symmetric[{2, 1}]` to
/// `Symmetric[{1, 2}]`.
fn parse_symmetry(
  spec: Option<&Expr>,
  rank: usize,
) -> Result<(SymmetryKind, Vec<usize>), SymmetryError> {
  let Some(spec) = spec else {
    return Ok((SymmetryKind::Symmetric, (1..=rank).collect()));
  };
  let Expr::FunctionCall { name, args } = spec else {
    return Err(SymmetryError::Invalid);
  };
  let Some(kind) = SymmetryKind::from_head(name) else {
    return Err(SymmetryError::Invalid);
  };
  let [Expr::List(items)] = &args[..] else {
    return Err(SymmetryError::Invalid);
  };
  let mut slots = Vec::with_capacity(items.len());
  for item in items {
    let Expr::Integer(i) = item else {
      return Err(SymmetryError::Invalid);
    };
    let i = *i;
    if i < 1 {
      return Err(SymmetryError::Invalid);
    }
    let slot = i as usize;
    if slot > rank {
      return Err(SymmetryError::BeyondRank(slot));
    }
    if slots.contains(&slot) {
      return Err(SymmetryError::Invalid);
    }
    slots.push(slot);
  }
  if slots.is_empty() {
    return Err(SymmetryError::Invalid);
  }
  slots.sort_unstable();
  Ok((kind, slots))
}

/// The symmetry tag a `SymmetrizedArray` stores. A group over fewer than two
/// slots is trivial and `ZeroSymmetric` has no entries left to relate, so both
/// print as an empty list rather than as a symmetry head.
fn symmetry_tag(kind: SymmetryKind, slots: &[usize]) -> Expr {
  if slots.len() < 2 || kind == SymmetryKind::ZeroSymmetric {
    return Expr::List(vec![].into());
  }
  call1(
    kind.head(),
    Expr::List(slots.iter().map(|&s| Expr::Integer(s as i128)).collect()),
  )
}

/// Every permutation of `0..n` in lexicographic order, paired with whether it
/// is odd.
fn permutations_with_parity(n: usize) -> Vec<(Vec<usize>, bool)> {
  let mut out = Vec::new();
  let mut perm: Vec<usize> = (0..n).collect();
  loop {
    // The parity of a permutation is the parity of its number of inversions.
    let inversions = (0..n)
      .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
      .filter(|&(i, j)| perm[i] > perm[j])
      .count();
    out.push((perm.clone(), inversions % 2 == 1));
    // Next permutation in lexicographic order.
    let Some(i) = (0..n.saturating_sub(1))
      .rev()
      .find(|&i| perm[i] < perm[i + 1])
    else {
      return out;
    };
    let j = ((i + 1)..n).rev().find(|&j| perm[j] > perm[i]).unwrap();
    perm.swap(i, j);
    perm[i + 1..].reverse();
  }
}

/// The row-major linear index of `pos` (0-based components) in `dims`.
fn pos_to_linear(pos: &[usize], dims: &[usize]) -> usize {
  pos.iter().zip(dims).fold(0, |acc, (&p, &d)| acc * d + p)
}

/// The 0-based position `dims` assigns to row-major linear index `lin`.
fn linear_to_pos(mut lin: usize, dims: &[usize]) -> Vec<usize> {
  let mut pos = vec![0; dims.len()];
  for k in (0..dims.len()).rev() {
    pos[k] = lin % dims[k];
    lin /= dims[k];
  }
  pos
}

/// Wrap `dims`, the canonical `rules` and the symmetry `tag` in the
/// `SymmetrizedArray[StructuredArray`StructuredData[…]]` form wolframscript
/// prints.
fn symmetrized_array(dims: &[usize], rules: Vec<Expr>, tag: Expr) -> Expr {
  let data = vec![
    Expr::List(dims.iter().map(|&d| Expr::Integer(d as i128)).collect()),
    Expr::List(vec![Expr::List(rules.into()), tag].into()),
  ];
  let structured_data = call("StructuredArray`StructuredData", data);
  call1("SymmetrizedArray", structured_data)
}

/// Symmetrize[t] / Symmetrize[t, sym] — project the array `t` onto a symmetry,
/// as a `SymmetrizedArray`. Without `sym` the whole tensor is symmetrized.
///
/// Only the canonical positions are stored: those whose components at the
/// symmetry slots are non-decreasing, or *strictly* increasing under
/// `Antisymmetric`, where a repeated index forces the entry to zero. Each one
/// holds the group average `(1/|G|) Σ_σ ε(σ) t[σ·p]`, where an odd `σ` negates
/// the term under `Antisymmetric` and conjugates it under `Hermitian`. Entries
/// that come out zero are dropped, and `ZeroSymmetric` stores nothing at all.
pub fn symmetrize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("Symmetrize", args);
  let render =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);
  let Some(dims) = tensor_dims(&args[0]) else {
    return Ok(unevaluated());
  };
  // A scalar has no slots to permute.
  if dims.is_empty() {
    return Ok(args[0].clone());
  }
  let rank = dims.len();
  let (kind, slots) = match parse_symmetry(args.get(1), rank) {
    Ok(parsed) => parsed,
    Err(SymmetryError::Invalid) => {
      crate::emit_message(&format!(
        "Symmetrize::symm: Invalid symmetry specification {}.",
        render(args.get(1).unwrap_or(&args[0]))
      ));
      return Ok(unevaluated());
    }
    Err(SymmetryError::BeyondRank(slot)) => {
      crate::emit_message(&format!(
        "Symmetrize::symmrank: Symmetry specification moves index {slot} beyond \
         tensor rank {rank}."
      ));
      return Ok(unevaluated());
    }
  };
  // Permuting two slots only makes sense when they are the same length.
  if slots.iter().any(|&s| dims[s - 1] != dims[slots[0] - 1]) {
    crate::emit_message(&format!(
      "Symmetrize::symmcomp: Symmetry specification {} is incompatible with \
       expression {}.",
      render(&call1(
        kind.head(),
        Expr::List(slots.iter().map(|&s| Expr::Integer(s as i128)).collect()),
      )),
      render(&Expr::List(
        dims.iter().map(|&d| Expr::Integer(d as i128)).collect()
      ))
    ));
    return Ok(unevaluated());
  }

  let tag = symmetry_tag(kind, &slots);
  if kind == SymmetryKind::ZeroSymmetric {
    return Ok(symmetrized_array(&dims, Vec::new(), tag));
  }
  let flat = flatten_tensor(&args[0]);
  let perms = permutations_with_parity(slots.len());
  let mut rules: Vec<Expr> = Vec::new();
  for lin in 0..flat.len() {
    let pos = linear_to_pos(lin, &dims);
    let at_slots: Vec<usize> = slots.iter().map(|&s| pos[s - 1]).collect();
    let canonical = at_slots.windows(2).all(|w| match kind {
      SymmetryKind::Antisymmetric => w[0] < w[1],
      _ => w[0] <= w[1],
    });
    if !canonical {
      continue;
    }
    // The average runs over the *distinct* terms the group produces, not over
    // the group itself: a position whose orbit is smaller than the group (a
    // repeated index) contributes each orbit element once, which is why
    // wolframscript reports (a[1,1,2] + a[1,2,1] + a[2,1,1])/3 rather than the
    // same sum doubled over 6. Under `Hermitian` a term is only a duplicate of
    // another when its conjugation matches too.
    let mut seen: Vec<(Vec<usize>, bool)> = Vec::with_capacity(perms.len());
    let mut terms: Vec<Expr> = Vec::with_capacity(perms.len());
    for (perm, odd) in &perms {
      let mut permuted = pos.clone();
      for (k, &from) in perm.iter().enumerate() {
        permuted[slots[k] - 1] = at_slots[from];
      }
      let flips = *odd && kind != SymmetryKind::Symmetric;
      if seen.contains(&(permuted.clone(), flips)) {
        continue;
      }
      seen.push((permuted.clone(), flips));
      let mut term = flat[pos_to_linear(&permuted, &dims)].clone();
      if flips {
        term = match kind {
          SymmetryKind::Antisymmetric => neg(term),
          SymmetryKind::Hermitian => call1("Conjugate", term),
          _ => term,
        };
      }
      terms.push(term);
    }
    let count = terms.len() as i128;
    let sum = plus(terms);
    let average = times(vec![
      call("Rational", vec![Expr::Integer(1), Expr::Integer(count)]),
      sum,
    ]);
    let value = evaluate_expr_to_expr(&average)?;
    if matches!(&value, Expr::Integer(0)) {
      continue;
    }
    let rule = vec![
      Expr::List(
        pos
          .iter()
          .map(|&p| Expr::Integer((p + 1) as i128))
          .collect(),
      ),
      value,
    ];
    rules.push(call("Rule", rule));
  }
  Ok(symmetrized_array(&dims, rules, tag))
}

/// The dimensions, canonical entries and symmetry of a `SymmetrizedArray`.
/// The symmetry is `None` when the stored tag is empty — a trivial one-slot
/// group, or `ZeroSymmetric`. Returns `None` for anything that is not a
/// `SymmetrizedArray`.
type SymmetrizedParts = (
  Vec<usize>,
  Vec<(Vec<usize>, Expr)>,
  Option<(SymmetryKind, Vec<usize>)>,
);
fn parse_symmetrized_array(e: &Expr) -> Option<SymmetrizedParts> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  if name != "SymmetrizedArray" || args.len() != 1 {
    return None;
  }
  let Expr::FunctionCall {
    name: inner,
    args: data,
  } = &args[0]
  else {
    return None;
  };
  if inner != "StructuredArray`StructuredData" || data.len() != 2 {
    return None;
  }
  let (Expr::List(dim_items), Expr::List(payload)) = (&data[0], &data[1])
  else {
    return None;
  };
  let mut dims = Vec::with_capacity(dim_items.len());
  for d in dim_items {
    let Expr::Integer(d) = d else { return None };
    dims.push(*d as usize);
  }
  let [Expr::List(rule_items), tag] = &payload[..] else {
    return None;
  };
  let mut entries = Vec::with_capacity(rule_items.len());
  for rule in rule_items {
    let (pos, value) = match rule {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Rule" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      _ => return None,
    };
    let Expr::List(components) = pos else {
      return None;
    };
    let mut p = Vec::with_capacity(components.len());
    for c in components {
      let Expr::Integer(c) = c else { return None };
      p.push(*c as usize - 1);
    }
    entries.push((p, value.clone()));
  }
  let symmetry = parse_symmetry(Some(tag), dims.len()).ok();
  Some((dims, entries, symmetry))
}

/// Every non-zero position of a `SymmetrizedArray` and its value, in
/// lexicographic position order: each stored canonical entry spread over the
/// orbit of its position, negated or conjugated wherever the permutation that
/// reaches the position is odd.
fn symmetrized_array_entries(
  parts: &SymmetrizedParts,
) -> Result<Vec<(Vec<usize>, Expr)>, InterpreterError> {
  let (_, entries, symmetry) = parts;
  let mut dense: std::collections::BTreeMap<Vec<usize>, Expr> =
    std::collections::BTreeMap::new();
  let Some((kind, slots)) = symmetry else {
    // A trivial symmetry relates nothing: the stored entries are the whole
    // array.
    for (pos, value) in entries {
      dense.insert(pos.clone(), value.clone());
    }
    return Ok(dense.into_iter().collect());
  };
  for (pos, value) in entries {
    let at_slots: Vec<usize> = slots.iter().map(|&s| pos[s - 1]).collect();
    for (perm, odd) in permutations_with_parity(slots.len()) {
      let mut permuted = pos.clone();
      for (k, &from) in perm.iter().enumerate() {
        permuted[slots[k] - 1] = at_slots[from];
      }
      let mut entry = value.clone();
      if odd {
        entry = match kind {
          SymmetryKind::Antisymmetric => evaluate_expr_to_expr(&neg(entry))?,
          SymmetryKind::Hermitian => {
            evaluate_expr_to_expr(&call1("Conjugate", entry))?
          }
          _ => entry,
        };
      }
      dense.insert(permuted, entry);
    }
  }
  Ok(dense.into_iter().collect())
}

/// The dense array a `SymmetrizedArray` stands for, with every position its
/// symmetry leaves unconstrained filled in with 0. `Normal`, `Dimensions` and
/// `ArrayRules` all go through this so they share the generic list
/// implementations instead of each re-deriving the symmetry.
pub fn symmetrized_array_to_dense(
  e: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let parts = parse_symmetrized_array(e)?;
  let entries = match symmetrized_array_entries(&parts) {
    Ok(entries) => entries,
    Err(err) => return Some(Err(err)),
  };
  let (dims, _, _) = &parts;
  let total: usize = dims.iter().product();
  let mut flat = vec![Expr::Integer(0); total];
  for (pos, value) in entries {
    flat[pos_to_linear(&pos, dims)] = value;
  }
  Some(Ok(build_from_flat(&flat, dims)))
}

/// `DiscreteIndicator[x, x1, {u1, u2, …}]`: 1 when `x` is `x1` and 0 when it
/// is another of the `ui`; `None` (unevaluated) unless both `x` and `x1`
/// are among the `ui`. Membership is structural, so `1.` is not `1`.
pub fn discrete_indicator(args: &[Expr]) -> Option<Expr> {
  let Expr::List(levels) = &args[2] else {
    return None;
  };
  let member = |e: &Expr| {
    levels
      .iter()
      .any(|l| crate::evaluator::pattern_matching::expr_equal(l, e))
  };
  if !member(&args[0]) || !member(&args[1]) {
    return None;
  }
  Some(Expr::Integer(i128::from(
    crate::evaluator::pattern_matching::expr_equal(&args[0], &args[1]),
  )))
}

/// The link between a generalized linear model's mean `mu` and its linear
/// predictor `eta`.
#[derive(Clone, Copy, PartialEq)]
enum GlmLink {
  Identity,
  Log,
  Logit,
  /// `eta = 1/mu`, the Gamma family's canonical link.
  Reciprocal,
  /// `eta = 1/mu^2`, the inverse Gaussian family's canonical link.
  InverseSquare,
  /// `eta = InverseCDF[NormalDistribution[], mu]`.
  Probit,
}

impl GlmLink {
  fn mean(self, eta: f64) -> f64 {
    match self {
      GlmLink::Identity => eta,
      GlmLink::Log => eta.exp(),
      GlmLink::Logit => 1.0 / (1.0 + (-eta).exp()),
      GlmLink::Reciprocal => 1.0 / eta,
      GlmLink::InverseSquare => 1.0 / eta.sqrt(),
      GlmLink::Probit => {
        0.5
          * crate::functions::math_ast::numeric_utils::erfc_f64(
            -eta / std::f64::consts::SQRT_2,
          )
      }
    }
  }

  fn predictor(self, mu: f64) -> f64 {
    match self {
      GlmLink::Identity => mu,
      GlmLink::Log => mu.ln(),
      GlmLink::Logit => (mu / (1.0 - mu)).ln(),
      GlmLink::Reciprocal => 1.0 / mu,
      GlmLink::InverseSquare => 1.0 / (mu * mu),
      GlmLink::Probit => {
        std::f64::consts::SQRT_2
          * crate::functions::math_ast::numeric_utils::inverse_erf_f64(
            2.0 * mu - 1.0,
          )
      }
    }
  }

  /// `d mu / d eta`, written in terms of `mu` so that it stays usable
  /// where `eta` has run off to a saturated mean.
  fn mean_derivative(self, mu: f64) -> f64 {
    match self {
      GlmLink::Identity => 1.0,
      GlmLink::Log => mu,
      GlmLink::Logit => mu * (1.0 - mu),
      GlmLink::Reciprocal => -mu * mu,
      GlmLink::InverseSquare => -0.5 * mu * mu * mu,
      GlmLink::Probit => {
        let eta = self.predictor(mu);
        (-0.5 * eta * eta).exp() / (2.0 * std::f64::consts::PI).sqrt()
      }
    }
  }

  /// The mean as an expression in the linear predictor `eta`.
  fn mean_expr(self, eta: Expr) -> Expr {
    match self {
      GlmLink::Identity => eta,
      GlmLink::Log => pow(Expr::Constant("E".into()), eta),
      GlmLink::Logit => pow(
        plus(vec![
          Expr::Integer(1),
          pow(Expr::Constant("E".into()), neg(eta)),
        ]),
        Expr::Integer(-1),
      ),
      GlmLink::Reciprocal => pow(eta, Expr::Integer(-1)),
      GlmLink::InverseSquare => pow(call1("Sqrt", eta), Expr::Integer(-1)),
      // (1 + Erf[eta/Sqrt[2]])/2
      GlmLink::Probit => times(vec![
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
        },
        plus(vec![
          Expr::Integer(1),
          call1(
            "Erf",
            times(vec![
              eta,
              pow(call1("Sqrt", Expr::Integer(2)), Expr::Integer(-1)),
            ]),
          ),
        ]),
      ]),
    }
  }
}

/// The exponential families a generalized linear model can assume, with
/// their variance functions `V(mu)`.
#[derive(Clone, Copy, PartialEq)]
enum GlmFamily {
  Gaussian,
  Binomial,
  Poisson,
  Gamma,
  InverseGaussian,
}

impl GlmFamily {
  fn from_name(name: &str) -> Option<Self> {
    Some(match name {
      "Gaussian" | "Normal" => GlmFamily::Gaussian,
      "Binomial" => GlmFamily::Binomial,
      "Poisson" => GlmFamily::Poisson,
      "Gamma" => GlmFamily::Gamma,
      "InverseGaussian" => GlmFamily::InverseGaussian,
      _ => return None,
    })
  }

  fn canonical_link(self) -> GlmLink {
    match self {
      GlmFamily::Gaussian => GlmLink::Identity,
      GlmFamily::Binomial => GlmLink::Logit,
      GlmFamily::Poisson => GlmLink::Log,
      GlmFamily::Gamma => GlmLink::Reciprocal,
      GlmFamily::InverseGaussian => GlmLink::InverseSquare,
    }
  }

  fn variance(self, mu: f64) -> f64 {
    match self {
      GlmFamily::Gaussian => 1.0,
      GlmFamily::Binomial => mu * (1.0 - mu),
      GlmFamily::Poisson => mu,
      GlmFamily::Gamma => mu * mu,
      GlmFamily::InverseGaussian => mu * mu * mu,
    }
  }

  /// `mu` kept strictly inside the family's domain, so that perfectly
  /// separated binomial data or zero Poisson counts (whose estimates run
  /// off to infinity) still give finite weights.
  fn clamp_mean(self, mu: f64) -> f64 {
    const EPS: f64 = 1e-15;
    match self {
      GlmFamily::Binomial => mu.clamp(EPS, 1.0 - EPS),
      GlmFamily::Gaussian => mu,
      _ => mu.max(EPS),
    }
  }

  /// The unit deviance of a response `y` at the mean `mu`.
  fn unit_deviance(self, y: f64, mu: f64) -> f64 {
    let xlogx = |a: f64, b: f64| if a == 0.0 { 0.0 } else { a * (a / b).ln() };
    match self {
      GlmFamily::Gaussian => (y - mu) * (y - mu),
      GlmFamily::Binomial => 2.0 * (xlogx(y, mu) + xlogx(1.0 - y, 1.0 - mu)),
      GlmFamily::Poisson => 2.0 * (xlogx(y, mu) - (y - mu)),
      GlmFamily::Gamma => 2.0 * (-(y / mu).ln() + (y - mu) / mu),
      GlmFamily::InverseGaussian => (y - mu) * (y - mu) / (mu * mu * y),
    }
  }

  /// The log-likelihood of `y` at the mean `mu` with dispersion `phi`.
  fn log_likelihood(self, y: f64, mu: f64, phi: f64) -> f64 {
    use crate::functions::math_ast::ln_gamma;
    use std::f64::consts::PI;
    match self {
      GlmFamily::Gaussian => {
        -0.5 * (2.0 * PI * phi).ln() - (y - mu) * (y - mu) / (2.0 * phi)
      }
      GlmFamily::Binomial => {
        let term = |a: f64, b: f64| if a == 0.0 { 0.0 } else { a * b.ln() };
        term(y, mu) + term(1.0 - y, 1.0 - mu)
      }
      GlmFamily::Poisson => {
        (if y == 0.0 { 0.0 } else { y * mu.ln() }) - mu - ln_gamma(y + 1.0)
      }
      GlmFamily::Gamma => {
        let shape = 1.0 / phi;
        -ln_gamma(shape) - shape * (mu * phi).ln() + (shape - 1.0) * y.ln()
          - y / (mu * phi)
      }
      GlmFamily::InverseGaussian => {
        -0.5 * (2.0 * PI * phi * y * y * y).ln()
          - (y - mu) * (y - mu) / (2.0 * phi * mu * mu * y)
      }
    }
  }

  /// The Anscombe residual of `y` at the mean `mu`: the difference of the
  /// family's variance-stabilizing transform, scaled to unit variance.
  fn anscombe_residual(self, y: f64, mu: f64) -> f64 {
    match self {
      GlmFamily::Gaussian => y - mu,
      GlmFamily::Binomial => {
        use crate::functions::math_ast::ln_gamma;
        use crate::functions::math_ast::numeric_utils::regularized_beta_inc;
        // Beta[z, 2/3, 2/3], the incomplete beta function.
        let a = 2.0 / 3.0;
        let complete = (2.0 * ln_gamma(a) - ln_gamma(2.0 * a)).exp();
        let beta = |z: f64| complete * regularized_beta_inc(z, a, a);
        (beta(y) - beta(mu)) / (mu * (1.0 - mu)).powf(1.0 / 6.0)
      }
      GlmFamily::Poisson => {
        1.5 * (y.powf(2.0 / 3.0) - mu.powf(2.0 / 3.0)) / mu.powf(1.0 / 6.0)
      }
      GlmFamily::Gamma => 3.0 * (y.cbrt() - mu.cbrt()) / mu.cbrt(),
      GlmFamily::InverseGaussian => (y.ln() - mu.ln()) / mu.sqrt(),
    }
  }

  /// Whether the dispersion is estimated rather than fixed at 1.
  fn estimates_dispersion(self) -> bool {
    !matches!(self, GlmFamily::Binomial | GlmFamily::Poisson)
  }

  /// The starting mean for a response `y` with sample mean `ybar`.
  fn initial_mean(self, y: f64, ybar: f64) -> f64 {
    match self {
      GlmFamily::Binomial => f64::midpoint(y, 0.5),
      GlmFamily::Poisson => y + 0.1,
      GlmFamily::Gaussian => y,
      GlmFamily::Gamma | GlmFamily::InverseGaussian => {
        if y > 0.0 {
          y
        } else {
          ybar
        }
      }
    }
  }
}

/// Fit `link(mean) = design . beta` by iteratively reweighted least
/// squares. Where the estimate does not exist (separated data) the
/// coefficients grow until the clamped means stop moving them, and a
/// singular weighted system ends the iteration at the last estimate.
/// `None` when not even the first step can be taken.
fn glm_irls(
  design: &[Vec<f64>],
  y: &[f64],
  family: GlmFamily,
  link: GlmLink,
) -> Option<Vec<f64>> {
  let n = y.len();
  let m = design.first()?.len();
  let ybar = y.iter().sum::<f64>() / n as f64;
  let mut eta: Vec<f64> = y
    .iter()
    .map(|&yi| link.predictor(family.initial_mean(yi, ybar)))
    .collect();
  let mut beta: Option<Vec<f64>> = None;
  for iteration in 0..100 {
    let mut atwa = vec![vec![0.0; m]; m];
    let mut atwz = vec![0.0; m];
    for i in 0..n {
      let mu = family.clamp_mean(link.mean(eta[i]));
      let dmu = link.mean_derivative(mu);
      let variance = family.variance(mu);
      if !(dmu.is_finite()
        && dmu != 0.0
        && variance > 0.0
        && variance.is_finite())
      {
        return beta;
      }
      let w = dmu * dmu / variance;
      let z = eta[i] + (y[i] - mu) / dmu;
      for j in 0..m {
        atwz[j] += design[i][j] * w * z;
        for k in 0..m {
          atwa[j][k] += design[i][j] * w * design[i][k];
        }
      }
    }
    let Some(next) = solve_linear_system(&atwa, &atwz) else {
      return beta;
    };
    if next.iter().any(|b| !b.is_finite()) {
      return beta;
    }
    let previous = beta.unwrap_or_else(|| vec![0.0; m]);
    let change = next
      .iter()
      .zip(&previous)
      .map(|(a, b)| (a - b).abs())
      .fold(0.0, f64::max);
    let scale = next.iter().map(|b| b.abs()).fold(1.0, f64::max);
    eta = design
      .iter()
      .map(|row| row.iter().zip(&next).map(|(a, b)| a * b).sum())
      .collect();
    beta = Some(next);
    if iteration > 0 && change <= 1e-13 * scale {
      break;
    }
  }
  beta
}

/// `GeneralizedLinearModelFit[data, basis, vars, opts]` for the families
/// in `GlmFamily`, with their canonical links or an explicit
/// `"IdentityLink"`, `"LogLink"` or `"LogitLink"`.
///
/// A variable named by `NominalVariables` takes symbolic levels: the basis
/// function that is that variable becomes one `DiscreteIndicator[v, level,
/// levels]` per level but the last (the levels in canonical order), as in
/// wolframscript. Anything this does not cover — another family or link,
/// weights, a nominal variable inside a larger basis function — leaves the
/// call unevaluated.
pub fn generalized_linear_model_fit_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  generalized_linear_model_fit(args, "GeneralizedLinearModelFit", args)
}

/// `LogitModelFit[data, basis, vars, opts]` and `ProbitModelFit[…]`: the
/// binomial generalized linear model with the logit or probit link.
pub fn binomial_model_fit_ast(
  head: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() < 3 {
    return Ok(unevaluated(head, args));
  }
  let rule = |name: &str, value: &str| Expr::Rule {
    pattern: Box::new(Expr::Identifier(name.to_string())),
    replacement: Box::new(Expr::String(value.to_string())),
  };
  let link = if head == "ProbitModelFit" {
    "ProbitLink"
  } else {
    "LogitLink"
  };
  let mut glm_args = args.to_vec();
  glm_args.push(rule("ExponentialFamily", "Binomial"));
  glm_args.push(rule("LinkFunction", link));
  generalized_linear_model_fit(&glm_args, head, args)
}

/// The generalized linear model fit behind `head[original…]`, with the
/// options normalized into `args`; an unevaluated result is written with
/// the caller's head and arguments.
fn generalized_linear_model_fit(
  args: &[Expr],
  head: &str,
  original: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated_call = || Ok(unevaluated(head, original));
  if args.len() < 3 {
    return unevaluated_call();
  }
  let mut family = GlmFamily::Gaussian;
  let mut link: Option<GlmLink> = None;
  let mut nominal_spec: Option<Expr> = None;
  let mut include_constant = true;
  for opt in &args[3..] {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      return unevaluated_call();
    };
    let Expr::Identifier(name) = pattern.as_ref() else {
      return unevaluated_call();
    };
    match (name.as_str(), replacement.as_ref()) {
      ("ExponentialFamily", Expr::String(f)) => match GlmFamily::from_name(f) {
        Some(f) => family = f,
        None => return unevaluated_call(),
      },
      ("LinkFunction", Expr::String(l)) => {
        link = Some(match l.as_str() {
          "IdentityLink" => GlmLink::Identity,
          "LogLink" => GlmLink::Log,
          "LogitLink" => GlmLink::Logit,
          "ProbitLink" => GlmLink::Probit,
          _ => return unevaluated_call(),
        });
      }
      ("LinkFunction", Expr::Identifier(a)) if a == "Automatic" => {}
      ("NominalVariables", spec) => nominal_spec = Some(spec.clone()),
      ("IncludeConstantBasis", value) => {
        include_constant =
          !matches!(value, Expr::Identifier(b) if b == "False");
      }
      _ => return unevaluated_call(),
    }
  }
  let link = link.unwrap_or_else(|| family.canonical_link());

  let vars: Vec<String> = match &args[2] {
    Expr::Identifier(v) => vec![v.clone()],
    Expr::List(items) => {
      let mut names = Vec::with_capacity(items.len());
      for item in items {
        let Expr::Identifier(v) = item else {
          return unevaluated_call();
        };
        names.push(v.clone());
      }
      names
    }
    _ => return unevaluated_call(),
  };
  let nominal: Vec<bool> = vars
    .iter()
    .map(|v| match &nominal_spec {
      None => false,
      Some(Expr::Identifier(a)) if a == "All" => true,
      Some(Expr::Identifier(a)) if a == "None" => false,
      Some(Expr::Identifier(a)) => a == v,
      Some(Expr::List(items)) => items
        .iter()
        .any(|i| matches!(i, Expr::Identifier(a) if a == v)),
      Some(_) => false,
    })
    .collect();

  // Rows `{x1, …, xk, y}`; a vector of responses for one variable is
  // indexed from 1.
  let Expr::List(raw_rows) = &args[0] else {
    return unevaluated_call();
  };
  let k = vars.len();
  let rows: Vec<Vec<Expr>> = if k == 1
    && raw_rows.iter().all(|r| !matches!(r, Expr::List(_)))
  {
    raw_rows
      .iter()
      .enumerate()
      .map(|(i, y)| vec![Expr::Integer(i as i128 + 1), y.clone()])
      .collect()
  } else {
    let mut rows = Vec::with_capacity(raw_rows.len());
    for r in raw_rows {
      match r {
        Expr::List(items) if items.len() == k + 1 => rows.push(items.to_vec()),
        Expr::List(items) if !items.is_empty() => {
          crate::emit_message(&format!(
            "{head}::fitc: The number of coordinates ({}) is not equal to \
             the number of variables ({k}).",
            items.len() - 1
          ));
          return unevaluated_call();
        }
        _ => return unevaluated_call(),
      }
    }
    rows
  };
  if rows.is_empty() {
    return unevaluated_call();
  }
  let mut y = Vec::with_capacity(rows.len());
  for row in &rows {
    match try_eval_to_f64(&row[k]) {
      Some(v) if family != GlmFamily::Binomial || (0.0..=1.0).contains(&v) => {
        y.push(v);
      }
      _ if family == GlmFamily::Binomial => {
        crate::emit_message(&format!(
          "{head}::binrsp: The response values are expected to be between 0 \
           and 1 for binomial models."
        ));
        return unevaluated_call();
      }
      _ => return unevaluated_call(),
    }
  }

  // The levels of each nominal variable, in canonical order.
  let mut levels: Vec<Option<Vec<Expr>>> = Vec::with_capacity(k);
  for (c, &is_nominal) in nominal.iter().enumerate() {
    levels.push(if is_nominal {
      let column = Expr::List(rows.iter().map(|r| r[c].clone()).collect());
      match &evaluate_expr_to_expr(&call1("Union", column))? {
        Expr::List(ls) => Some(ls.to_vec()),
        _ => return unevaluated_call(),
      }
    } else {
      None
    });
  }

  // Expand the basis: a nominal variable becomes its level indicators.
  let user_basis: Vec<Expr> = match &args[1] {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  let mut basis: Vec<Expr> = if include_constant {
    vec![Expr::Integer(1)]
  } else {
    Vec::new()
  };
  for b in user_basis {
    if include_constant && matches!(b, Expr::Integer(1)) {
      continue;
    }
    let nominal_index = vars.iter().position(|v| {
      levels[vars.iter().position(|w| w == v).unwrap()].is_some()
        && matches!(&b, Expr::Identifier(n) if n == v)
    });
    if let Some(c) = nominal_index {
      let ls = levels[c].as_ref().unwrap();
      let all = Expr::List(ls.clone().into());
      for level in &ls[..ls.len().saturating_sub(1)] {
        basis.push(call(
          "DiscreteIndicator",
          vec![
            Expr::Identifier(vars[c].clone()),
            level.clone(),
            all.clone(),
          ],
        ));
      }
    } else if vars.iter().zip(&levels).any(|(v, l)| {
      l.is_some() && !crate::functions::calculus_ast::is_constant_wrt(&b, v)
    }) {
      return unevaluated_call();
    } else {
      basis.push(b);
    }
  }
  if basis.is_empty() {
    return unevaluated_call();
  }

  // Design matrix: each basis function at each row's values.
  let mut design = Vec::with_capacity(rows.len());
  for row in &rows {
    let mut values = Vec::with_capacity(basis.len());
    for b in &basis {
      let at_row = vars.iter().zip(row).fold(b.clone(), |e, (v, value)| {
        crate::syntax::substitute_variable(&e, v, value)
      });
      match try_eval_to_f64(&evaluate_expr_to_expr(&at_row)?) {
        Some(v) => values.push(v),
        None => return unevaluated_call(),
      }
    }
    design.push(values);
  }

  let Some(beta) = glm_irls(&design, &y, family, link) else {
    return unevaluated_call();
  };
  let predicted: Vec<f64> = design
    .iter()
    .map(|row| link.mean(row.iter().zip(&beta).map(|(a, b)| a * b).sum()))
    .collect();

  let predictor = {
    let terms: Vec<Expr> = beta
      .iter()
      .zip(&basis)
      .map(|(c, b)| match b {
        Expr::Integer(1) => Expr::Real(*c),
        other => times(vec![Expr::Real(*c), other.clone()]),
      })
      .collect();
    let raw = if terms.len() == 1 {
      terms.into_iter().next().unwrap()
    } else {
      plus(terms)
    };
    evaluate_expr_to_expr(&raw).unwrap_or(raw)
  };
  let fitted_expr = {
    let raw = link.mean_expr(predictor);
    evaluate_expr_to_expr(&raw).unwrap_or(raw)
  };
  let mut function_body = fitted_expr.clone();
  for (k, name) in vars.iter().enumerate() {
    function_body = crate::syntax::substitute_variable(
      &function_body,
      name,
      &Expr::Slot(k + 1),
    );
  }
  let reals =
    |v: &[f64]| Expr::List(v.iter().map(|x| Expr::Real(*x)).collect());
  let response = Expr::List(rows.iter().map(|r| r[k].clone()).collect());
  let residuals: Vec<f64> =
    y.iter().zip(&predicted).map(|(a, b)| a - b).collect();

  // Goodness of fit. The null model is the constant one (its estimate is
  // the mean response whatever the link), or `eta = 0` without a constant.
  let n = y.len();
  let p = beta.len();
  let deviances: Vec<f64> = y
    .iter()
    .zip(&predicted)
    .map(|(&yi, &mu)| family.unit_deviance(yi, mu))
    .collect();
  let null_mean = if include_constant {
    y.iter().sum::<f64>() / n as f64
  } else {
    link.mean(0.0)
  };
  let null_deviance: f64 = y
    .iter()
    .map(|&yi| family.unit_deviance(yi, null_mean))
    .sum();
  let residual_dof = n.saturating_sub(p);
  // The Pearson estimate of the dispersion, as wolframscript takes it.
  let dispersion = if family.estimates_dispersion() {
    let pearson: f64 = y
      .iter()
      .zip(&predicted)
      .map(|(&yi, &mu)| (yi - mu) * (yi - mu) / family.variance(mu))
      .sum();
    pearson / residual_dof.max(1) as f64
  } else {
    1.0
  };
  let log_likelihood: f64 = y
    .iter()
    .zip(&predicted)
    .map(|(&yi, &mu)| family.log_likelihood(yi, mu, dispersion))
    .sum();
  // The information criteria count the dispersion as a parameter too.
  let parameters = (p + 1) as f64;
  let deviance_residuals: Vec<f64> = residuals
    .iter()
    .zip(&deviances)
    .map(|(r, d)| r.signum() * d.max(0.0).sqrt())
    .collect();
  let pearson_residuals: Vec<f64> = y
    .iter()
    .zip(&predicted)
    .map(|(&yi, &mu)| (yi - mu) / family.variance(mu).sqrt())
    .collect();
  let anscombe_residuals: Vec<f64> = y
    .iter()
    .zip(&predicted)
    .map(|(&yi, &mu)| family.anscombe_residual(yi, mu))
    .collect();
  // Parameter statistics: the covariance is the dispersion times the
  // inverse of the final weighted cross product `Transpose[X] . W . X`, and
  // each estimate is tested with a two-sided z test.
  let mut xtwx = vec![vec![0.0; p]; p];
  for (row, &mu) in design.iter().zip(&predicted) {
    let mu = family.clamp_mean(mu);
    let dmu = link.mean_derivative(mu);
    let w = dmu * dmu / family.variance(mu);
    for a in 0..p {
      for b in 0..p {
        xtwx[a][b] += row[a] * w * row[b];
      }
    }
  }
  let Some(inverse) = invert_square_matrix(&xtwx) else {
    return unevaluated_call();
  };
  let covariance: Vec<Vec<f64>> = inverse
    .iter()
    .map(|row| row.iter().map(|v| v * dispersion).collect())
    .collect();
  let covariance_expr =
    Expr::List(covariance.iter().map(|row| reals(row)).collect());
  let errors: Vec<f64> =
    (0..p).map(|j| covariance[j][j].max(0.0).sqrt()).collect();
  let z_stats: Vec<f64> = (0..p).map(|j| beta[j] / errors[j]).collect();
  let p_values: Vec<f64> = z_stats
    .iter()
    .map(|z| {
      crate::functions::math_ast::numeric_utils::erfc_f64(
        z.abs() / std::f64::consts::SQRT_2,
      )
    })
    .collect();
  // Quantile[NormalDistribution[], 0.975]
  let z_975 = std::f64::consts::SQRT_2
    * crate::functions::math_ast::numeric_utils::inverse_erf_f64(0.95);
  let integer = |v: usize| Expr::Integer(v as i128);
  let mut assoc = vec![
    (str_expr("Deviances"), reals(&deviances)),
    (
      str_expr("ResidualDeviance"),
      Expr::Real(deviances.iter().sum()),
    ),
    (str_expr("ResidualDegreesOfFreedom"), integer(residual_dof)),
    (
      str_expr("NullDegreesOfFreedom"),
      integer(if include_constant { n - 1 } else { n }),
    ),
    (str_expr("DevianceResiduals"), reals(&deviance_residuals)),
    (str_expr("PearsonResiduals"), reals(&pearson_residuals)),
    (str_expr("CovarianceMatrix"), covariance_expr),
    (str_expr("ParameterErrors"), reals(&errors)),
    (str_expr("ParameterZStatistics"), reals(&z_stats)),
    (str_expr("ParameterPValues"), reals(&p_values)),
    (
      str_expr("ParameterTableEntries"),
      Expr::List(
        (0..p)
          .map(|j| reals(&[beta[j], errors[j], z_stats[j], p_values[j]]))
          .collect(),
      ),
    ),
    (
      str_expr("ParameterConfidenceIntervals"),
      Expr::List(
        (0..p)
          .map(|j| {
            reals(&[beta[j] - z_975 * errors[j], beta[j] + z_975 * errors[j]])
          })
          .collect(),
      ),
    ),
    (str_expr("AnscombeResiduals"), reals(&anscombe_residuals)),
    (str_expr("EstimatedDispersion"), Expr::Real(dispersion)),
  ];
  // wolframscript's null deviance for the Gaussian, Gamma and inverse
  // Gaussian families, and its inverse Gaussian likelihood, are neither
  // the constant model's nor the density's; those properties are left out
  // rather than reported differently.
  if !matches!(
    family,
    GlmFamily::Gaussian | GlmFamily::Gamma | GlmFamily::InverseGaussian
  ) {
    assoc.push((str_expr("NullDeviance"), Expr::Real(null_deviance)));
  }
  if !matches!(family, GlmFamily::InverseGaussian) {
    assoc.extend([
      (str_expr("LogLikelihood"), Expr::Real(log_likelihood)),
      (
        str_expr("AIC"),
        Expr::Real(-2.0 * log_likelihood + 2.0 * parameters),
      ),
      (
        str_expr("BIC"),
        Expr::Real(-2.0 * log_likelihood + parameters * (n as f64).ln()),
      ),
    ]);
  }
  assoc.extend([
    (str_expr("Type"), id_expr("GeneralizedLinear")),
    (str_expr("FittedExpression"), fitted_expr),
    (str_expr("BestFitParameters"), reals(&beta)),
    (
      str_expr("IndependentVariables"),
      Expr::List(vars.iter().map(|v| Expr::Identifier(v.clone())).collect()),
    ),
    (str_expr("BasisFunctions"), Expr::List(basis.into())),
    (
      str_expr("DesignMatrix"),
      Expr::List(design.iter().map(|r| reals(r)).collect()),
    ),
    (str_expr("PredictedResponse"), reals(&predicted)),
    (str_expr("Response"), response),
    (str_expr("FitResiduals"), reals(&residuals)),
    (
      str_expr("InputData"),
      Expr::List(rows.into_iter().map(|r| Expr::List(r.into())).collect()),
    ),
    (
      str_expr("Function"),
      Expr::Function {
        body: Box::new(function_body),
      },
    ),
    (
      str_expr("VariableName"),
      Expr::String(vars.first().cloned().unwrap_or_default()),
    ),
  ]);
  Ok(call1("FittedModel", Expr::Association(assoc)))
}
