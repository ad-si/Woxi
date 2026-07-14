#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::functions::math_ast::expr_to_i128;
use crate::syntax::{BinaryOperator, Expr, unevaluated};

/// Resultant[poly1, poly2, var] - Computes the resultant of two polynomials
/// with respect to the given variable.
///
/// The resultant is the determinant of the Sylvester matrix built from
/// the coefficients of the two polynomials.
pub fn resultant_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // A `Modulus -> p` option reduces the (symbolically computed) resultant's
  // coefficients modulo p: the resultant is a polynomial in the input
  // coefficients, so reduction commutes with its computation.
  if args.iter().any(|a| extract_modulus_option(a).is_some()) {
    let mut pos: Vec<Expr> = Vec::new();
    let mut modulus: Option<i128> = None;
    for a in args {
      if let Some(m) = extract_modulus_option(a) {
        modulus = Some(m);
      } else {
        pos.push(a.clone());
      }
    }
    if let Some(p) = modulus {
      let base = resultant_ast(&pos)?;
      return reduce_coeffs_modulus(&base, p);
    }
  }

  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Resultant expects exactly 3 arguments".into(),
    ));
  }

  let poly1 = &args[0];
  let poly2 = &args[1];
  let var = match &args[2] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Third argument of Resultant must be a symbol".into(),
      ));
    }
  };

  // Get degrees
  let deg1 = match max_power_int(poly1, &var) {
    Some(d) => d,
    None => {
      return Ok(unevaluated("Resultant", args));
    }
  };
  let deg2 = match max_power_int(poly2, &var) {
    Some(d) => d,
    None => {
      return Ok(unevaluated("Resultant", args));
    }
  };

  let m = deg1 as usize;
  let n = deg2 as usize;

  if m == 0 && n == 0 {
    // Both constants: resultant is 1 (for nonzero constants)
    return Ok(Expr::Integer(1));
  }

  // Extract coefficients symbolically
  // coeff1[i] = coefficient of x^i in poly1, for i = 0..m
  // coeff2[i] = coefficient of x^i in poly2, for i = 0..n
  let var_expr = Expr::Identifier(var.clone());
  let mut coeff1 = Vec::with_capacity(m + 1);
  let mut coeff2 = Vec::with_capacity(n + 1);

  for i in 0..=m {
    let c = coefficient_ast(&[
      poly1.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    let c = crate::evaluator::evaluate_expr_to_expr(&c)?;
    coeff1.push(c);
  }
  for i in 0..=n {
    let c = coefficient_ast(&[
      poly2.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    let c = crate::evaluator::evaluate_expr_to_expr(&c)?;
    coeff2.push(c);
  }

  // Try integer path first for efficiency
  let int_coeffs1: Option<Vec<i128>> =
    coeff1.iter().map(expr_to_i128).collect();
  let int_coeffs2: Option<Vec<i128>> =
    coeff2.iter().map(expr_to_i128).collect();

  if let (Some(ic1), Some(ic2)) = (int_coeffs1, int_coeffs2) {
    // Integer Sylvester matrix determinant
    let size = m + n;
    if size == 0 {
      return Ok(Expr::Integer(1));
    }
    let det = sylvester_det_integer(&ic1, m, &ic2, n);
    return Ok(Expr::Integer(det));
  }

  // Symbolic path: build Sylvester matrix and compute determinant
  let size = m + n;
  if size == 0 {
    return Ok(Expr::Integer(1));
  }

  let mut matrix: Vec<Vec<Expr>> = Vec::with_capacity(size);

  // First n rows: coefficients of poly1, reversed (leading coeff first)
  for i in 0..n {
    let mut row = vec![Expr::Integer(0); size];
    for j in 0..=m {
      let col = i + m - j;
      row[col] = coeff1[j].clone();
    }
    matrix.push(row);
  }

  // Last m rows: coefficients of poly2, reversed (leading coeff first)
  for i in 0..m {
    let mut row = vec![Expr::Integer(0); size];
    for j in 0..=n {
      let col = i + n - j;
      row[col] = coeff2[j].clone();
    }
    matrix.push(row);
  }

  // Compute symbolic determinant
  let det = symbolic_determinant(&matrix);

  // Simplify the result
  let simplified = crate::evaluator::evaluate_expr_to_expr(&det)?;
  Ok(simplified)
}

/// Reduce all integer coefficients of `expr` modulo `p` (via PolynomialMod),
/// the action of a `Modulus -> p` option on a polynomial result.
pub(super) fn reduce_coeffs_modulus(
  expr: &Expr,
  p: i128,
) -> Result<Expr, InterpreterError> {
  let reduced = Expr::FunctionCall {
    name: "PolynomialMod".to_string(),
    args: vec![expr.clone(), Expr::Integer(p)].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&reduced)
}

/// Subresultants[poly1, poly2, var] - The principal subresultant
/// coefficients s_j for j = 0..Min[deg1, deg2]; s_0 is the resultant.
///
/// s_j is the determinant of the leading (m+n-2j) square block of the
/// matrix whose rows are x^(n-j-1)*p1 ... p1, x^(m-j-1)*p2 ... p2 over
/// the columns for degrees m+n-j-1 down to j. This single formula
/// reproduces wolframscript for every degree combination (m < n needs
/// no swap or sign factor).
pub fn subresultants_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("Subresultants", args);
  // A `Modulus -> p` option computes over GF(p) with the reduced inputs
  // (degrees drop where leading coefficients vanish mod p). Entries are
  // normalized to [0, p), except that for deg1 < deg2 wolframscript
  // presents entry j as the SIGNED swap representative: it normalizes the
  // swapped-argument value and then applies the transposition sign
  // (-1)^((m-j)(n-j)) without renormalizing (e.g. {-6, 1} mod 7).
  if args.iter().any(|a| extract_modulus_option(a).is_some()) {
    let mut pos: Vec<Expr> = Vec::new();
    let mut modulus: Option<i128> = None;
    for a in args {
      if let Some(p) = extract_modulus_option(a) {
        modulus = Some(p);
      } else {
        pos.push(a.clone());
      }
    }
    let (Some(p), 3) = (modulus, pos.len()) else {
      return Ok(unevaluated(args));
    };
    let poly1 = crate::evaluator::evaluate_expr_to_expr(&pos[0])?;
    let poly2 = crate::evaluator::evaluate_expr_to_expr(&pos[1])?;
    let var = match &pos[2] {
      Expr::Identifier(name) => name.clone(),
      _ => return Ok(unevaluated(args)),
    };
    if matches!(poly1, Expr::Integer(0)) || matches!(poly2, Expr::Integer(0)) {
      return Ok(Expr::List(vec![].into()));
    }
    let (Some(m), Some(n)) =
      (max_power_int(&poly1, &var), max_power_int(&poly2, &var))
    else {
      return Ok(unevaluated(args));
    };
    let var_expr = Expr::Identifier(var.clone());
    let mut coeff1 = Vec::with_capacity(m as usize + 1);
    let mut coeff2 = Vec::with_capacity(n as usize + 1);
    for i in 0..=m {
      let c =
        coefficient_ast(&[poly1.clone(), var_expr.clone(), Expr::Integer(i)])?;
      coeff1.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
    }
    for i in 0..=n {
      let c =
        coefficient_ast(&[poly2.clone(), var_expr.clone(), Expr::Integer(i)])?;
      coeff2.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
    }
    let int_coeffs1: Option<Vec<i128>> =
      coeff1.iter().map(expr_to_i128).collect();
    let int_coeffs2: Option<Vec<i128>> =
      coeff2.iter().map(expr_to_i128).collect();
    let (Some(ic1), Some(ic2)) = (int_coeffs1, int_coeffs2) else {
      // Symbolic coefficients: wolframscript ignores the modulus.
      return subresultants_ast(&pos);
    };
    let mut c1: Vec<i128> = ic1.iter().map(|v| v.rem_euclid(p)).collect();
    let mut c2: Vec<i128> = ic2.iter().map(|v| v.rem_euclid(p)).collect();
    trim_int_poly(&mut c1);
    trim_int_poly(&mut c2);
    if c1 == [0] || c2 == [0] {
      return Ok(Expr::List(vec![].into()));
    }
    let (mr, nr) = (int_poly_deg(&c1), int_poly_deg(&c2));
    let entries: Vec<i128> = if mr >= nr {
      psc_int(&c1, &c2).iter().map(|v| v.rem_euclid(p)).collect()
    } else {
      psc_int(&c2, &c1)
        .iter()
        .enumerate()
        .map(|(j, v)| {
          let w = v.rem_euclid(p);
          if (mr - j) % 2 == 1 && (nr - j) % 2 == 1 {
            -w
          } else {
            w
          }
        })
        .collect()
    };
    return Ok(Expr::List(
      entries
        .into_iter()
        .map(Expr::Integer)
        .collect::<Vec<_>>()
        .into(),
    ));
  }
  if args.len() != 3 {
    return Ok(unevaluated(args));
  }
  let poly1 = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let poly2 = crate::evaluator::evaluate_expr_to_expr(&args[1])?;
  let var = match &args[2] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated(args)),
  };

  // A zero polynomial has no subresultant chain
  if matches!(poly1, Expr::Integer(0)) || matches!(poly2, Expr::Integer(0)) {
    return Ok(Expr::List(vec![].into()));
  }

  let m = match max_power_int(&poly1, &var) {
    Some(d) => d as usize,
    None => return Ok(unevaluated(args)),
  };
  let n = match max_power_int(&poly2, &var) {
    Some(d) => d as usize,
    None => return Ok(unevaluated(args)),
  };

  let var_expr = Expr::Identifier(var.clone());
  let mut coeff1 = Vec::with_capacity(m + 1);
  let mut coeff2 = Vec::with_capacity(n + 1);
  for i in 0..=m {
    let c = coefficient_ast(&[
      poly1.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    coeff1.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }
  for i in 0..=n {
    let c = coefficient_ast(&[
      poly2.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    coeff2.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }

  let int_coeffs1: Option<Vec<i128>> =
    coeff1.iter().map(expr_to_i128).collect();
  let int_coeffs2: Option<Vec<i128>> =
    coeff2.iter().map(expr_to_i128).collect();
  let integer_path = int_coeffs1.is_some() && int_coeffs2.is_some();

  let d = m.min(n);
  let mut result = Vec::with_capacity(d + 1);
  for j in 0..=d {
    let size = m + n - 2 * j;
    if size == 0 {
      result.push(Expr::Integer(1));
      continue;
    }
    // Row for x^s * p: entry of degree deg is coeff[deg - s]; the kept
    // columns cover degrees m+n-j-1 down to j (col = m+n-j-1 - deg).
    let top_degree = m + n - j - 1;
    let fill_rows =
      |matrix: &mut Vec<Vec<Expr>>, coeffs: &[Expr], shifts: usize| {
        for i in 0..shifts {
          let s = shifts - 1 - i;
          let mut row = vec![Expr::Integer(0); size];
          for (k, c) in coeffs.iter().enumerate() {
            let deg = k + s;
            if deg >= j && deg <= top_degree {
              row[top_degree - deg] = c.clone();
            }
          }
          matrix.push(row);
        }
      };
    let mut matrix: Vec<Vec<Expr>> = Vec::with_capacity(size);
    fill_rows(&mut matrix, &coeff1, n - j);
    fill_rows(&mut matrix, &coeff2, m - j);

    if integer_path {
      let mut int_matrix: Vec<Vec<i128>> = matrix
        .iter()
        .map(|row| row.iter().map(|e| expr_to_i128(e).unwrap()).collect())
        .collect();
      result.push(Expr::Integer(bareiss_determinant(&mut int_matrix)));
    } else {
      let det = symbolic_determinant(&matrix);
      result.push(crate::evaluator::evaluate_expr_to_expr(&det)?);
    }
  }
  Ok(Expr::List(result.into()))
}

/// Principal subresultant coefficients over the integers for trimmed
/// ascending coefficient vectors — the same matrices as the Expr-based loop
/// in `subresultants_ast`, specialized to i128.
fn psc_int(c1: &[i128], c2: &[i128]) -> Vec<i128> {
  let m = int_poly_deg(c1);
  let n = int_poly_deg(c2);
  let d = m.min(n);
  let mut result = Vec::with_capacity(d + 1);
  for j in 0..=d {
    let size = m + n - 2 * j;
    if size == 0 {
      result.push(1);
      continue;
    }
    let top_degree = m + n - j - 1;
    let mut matrix: Vec<Vec<i128>> = Vec::with_capacity(size);
    for (coeffs, shifts) in [(c1, n - j), (c2, m - j)] {
      for i in 0..shifts {
        let s = shifts - 1 - i;
        let mut row = vec![0i128; size];
        for (idx, &c) in coeffs.iter().enumerate() {
          let deg = idx + s;
          if deg >= j && deg <= top_degree {
            row[top_degree - deg] = c;
          }
        }
        matrix.push(row);
      }
    }
    result.push(bareiss_determinant(&mut matrix));
  }
  result
}

/// SubresultantPolynomials[poly1, poly2, var] - The subresultant polynomial
/// sequence S_0 .. S_n where n = deg(poly2); requires deg(poly1) >= n.
///
/// For j < n, the coefficient of x^k in S_j is the determinant of the
/// matrix whose rows are x^(n-j-1)*p1 ... p1, x^(m-j-1)*p2 ... p2, whose
/// first m+n-2j-1 columns cover degrees m+n-j-1 down to j+1, and whose
/// last column holds the degree-k coefficients (so k = j reproduces the
/// principal subresultant coefficients of Subresultants). The final entry
/// is lc(poly2)^(m-n-1) * poly2: expanded when m > n, and kept as the
/// quotient poly2/lc when m == n, exactly as wolframscript prints it.
pub fn subresultant_polynomials_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Split off a `Modulus -> p` option; remaining args stay in order.
  let mut rebuilt: Vec<Expr> = args.to_vec();
  let mut pos_idx: Vec<usize> = Vec::new();
  let mut modulus: Option<i128> = None;
  for (i, a) in args.iter().enumerate() {
    if let Some(p) = extract_modulus_option(a) {
      modulus = Some(p);
    } else {
      pos_idx.push(i);
    }
  }
  let unevaluated =
    |rebuilt: &[Expr]| unevaluated("SubresultantPolynomials", rebuilt);
  if pos_idx.len() != 3 {
    return Ok(unevaluated(&rebuilt));
  }
  let poly1 = crate::evaluator::evaluate_expr_to_expr(&args[pos_idx[0]])?;
  let poly2 = crate::evaluator::evaluate_expr_to_expr(&args[pos_idx[1]])?;
  rebuilt[pos_idx[0]] = poly1.clone();
  rebuilt[pos_idx[1]] = poly2.clone();
  let var = match &args[pos_idx[2]] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated(&rebuilt)),
  };

  // A zero second polynomial has no subresultant chain.
  if matches!(poly2, Expr::Integer(0)) {
    return Ok(Expr::List(vec![].into()));
  }

  let npolys_message = |poly1: &Expr, poly2: &Expr| {
    let p1 = crate::syntax::expr_to_string(poly1);
    let p2 = crate::syntax::expr_to_string(poly2);
    crate::emit_message(&format!(
      "SubresultantPolynomials::npolys: {p1} and {p2} should be polynomials \
       with exact coefficients and the degree of {p1} in {var} should not \
       be less than the degree of {p2} in {var}."
    ));
  };

  // The zero polynomial has degree -Infinity, so a zero first argument
  // always fails the degree requirement.
  if matches!(poly1, Expr::Integer(0)) {
    npolys_message(&poly1, &poly2);
    return Ok(unevaluated(&rebuilt));
  }

  let m = match max_power_int(&poly1, &var) {
    Some(d) => d as usize,
    None => return Ok(unevaluated(&rebuilt)),
  };
  let n = match max_power_int(&poly2, &var) {
    Some(d) => d as usize,
    None => return Ok(unevaluated(&rebuilt)),
  };
  if m < n || subres_contains_real(&poly1) || subres_contains_real(&poly2) {
    npolys_message(&poly1, &poly2);
    return Ok(unevaluated(&rebuilt));
  }

  let var_expr = Expr::Identifier(var.clone());
  let mut coeff1 = Vec::with_capacity(m + 1);
  let mut coeff2 = Vec::with_capacity(n + 1);
  for i in 0..=m {
    let c = coefficient_ast(&[
      poly1.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    coeff1.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }
  for i in 0..=n {
    let c = coefficient_ast(&[
      poly2.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    coeff2.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }

  let int_coeffs1: Option<Vec<i128>> =
    coeff1.iter().map(expr_to_i128).collect();
  let int_coeffs2: Option<Vec<i128>> =
    coeff2.iter().map(expr_to_i128).collect();
  let integer_path = int_coeffs1.is_some() && int_coeffs2.is_some();
  // Modular arithmetic on symbolic coefficients is not supported.
  if modulus.is_some() && !integer_path {
    return Ok(unevaluated(&rebuilt));
  }

  let var_pow = |k: usize| -> Expr {
    match k {
      0 => Expr::Integer(1),
      1 => var_expr.clone(),
      _ => Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(var_expr.clone()),
        right: Box::new(Expr::Integer(k as i128)),
      },
    }
  };

  let mut entries: Vec<Expr> = Vec::with_capacity(n + 1);
  for j in 0..n {
    let size = m + n - 2 * j;
    let top_degree = m + n - j - 1;
    let mut sum = Expr::Integer(0);
    for k in 0..=j {
      // Rows for x^s * p over the kept columns plus the degree-k column.
      let mut matrix: Vec<Vec<Expr>> = Vec::with_capacity(size);
      for (coeffs, shifts) in [(&coeff1, n - j), (&coeff2, m - j)] {
        for i in 0..shifts {
          let s = shifts - 1 - i;
          let mut row = vec![Expr::Integer(0); size];
          for (idx, c) in coeffs.iter().enumerate() {
            let deg = idx + s;
            if deg > j && deg <= top_degree {
              row[top_degree - deg] = c.clone();
            }
            if deg == k {
              row[size - 1] = c.clone();
            }
          }
          matrix.push(row);
        }
      }
      let det = if integer_path {
        let mut int_matrix: Vec<Vec<i128>> = matrix
          .iter()
          .map(|row| row.iter().map(|e| expr_to_i128(e).unwrap()).collect())
          .collect();
        let mut d = bareiss_determinant(&mut int_matrix);
        if let Some(p) = modulus {
          d = d.rem_euclid(p);
        }
        Expr::Integer(d)
      } else {
        symbolic_determinant(&matrix)
      };
      sum = sym_add(&sum, &sym_mul(&det, &var_pow(k)));
    }
    // Symbolic determinants come out as nested products; wolframscript
    // prints the fully expanded polynomial.
    if !integer_path {
      sum = Expr::FunctionCall {
        name: "Expand".to_string(),
        args: vec![sum].into(),
      };
    }
    entries.push(crate::evaluator::evaluate_expr_to_expr(&sum)?);
  }

  // Final entry S_n = lc^(m-n-1) * poly2. Under Modulus the coefficients
  // of poly2 (and of the expanded product) reduce mod p, but the m == n
  // quotient keeps its unreduced leading-coefficient divisor, matching
  // wolframscript's output shape.
  let lc = coeff2[n].clone();
  let q_coeffs: Vec<Expr> =
    if let (Some(p), Some(ic2)) = (modulus, &int_coeffs2) {
      ic2.iter().map(|c| Expr::Integer(c.rem_euclid(p))).collect()
    } else {
      coeff2.clone()
    };
  let mut q_expanded = Expr::Integer(0);
  for (i, c) in q_coeffs.iter().enumerate() {
    q_expanded = sym_add(&q_expanded, &sym_mul(c, &var_pow(i)));
  }
  let last = if m == n {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(lc),
        right: Box::new(Expr::Integer(-1)),
      }),
      right: Box::new(q_expanded),
    }
  } else if let Some(ic2) = &int_coeffs2 {
    let lc_int = ic2[n];
    let mut scale = 1i128;
    for _ in 0..(m - n - 1) {
      scale *= lc_int;
    }
    let mut sum = Expr::Integer(0);
    for (i, c) in ic2.iter().enumerate() {
      let mut v = scale * c;
      if let Some(p) = modulus {
        v = v.rem_euclid(p);
      }
      sum = sym_add(&sum, &sym_mul(&Expr::Integer(v), &var_pow(i)));
    }
    sum
  } else {
    Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(lc),
          right: Box::new(Expr::Integer((m - n - 1) as i128)),
        }),
        right: Box::new(q_expanded),
      }]
      .into(),
    }
  };
  entries.push(crate::evaluator::evaluate_expr_to_expr(&last)?);
  Ok(Expr::List(entries.into()))
}

/// SubresultantPolynomialRemainders[poly1, poly2, var] - Brown's
/// subresultant polynomial remainder sequence: the two expanded input
/// polynomials followed by each pseudo-remainder divided by the
/// subresultant beta factor, stopping before a zero remainder (though a
/// literal zero second argument is echoed). With `Modulus -> p` the whole
/// chain runs over GF(p) — degrees drop where leading coefficients vanish
/// mod p, so this is not a mere coefficient reduction of the plain chain.
pub fn subresultant_polynomial_remainders_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let mut rebuilt: Vec<Expr> = args.to_vec();
  let mut pos_idx: Vec<usize> = Vec::new();
  let mut modulus: Option<i128> = None;
  for (i, a) in args.iter().enumerate() {
    if let Some(p) = extract_modulus_option(a) {
      modulus = Some(p);
    } else {
      pos_idx.push(i);
    }
  }
  let unevaluated =
    |rebuilt: &[Expr]| unevaluated("SubresultantPolynomialRemainders", rebuilt);
  if pos_idx.len() != 3 {
    return Ok(unevaluated(&rebuilt));
  }
  let poly1 = crate::evaluator::evaluate_expr_to_expr(&args[pos_idx[0]])?;
  let poly2 = crate::evaluator::evaluate_expr_to_expr(&args[pos_idx[1]])?;
  rebuilt[pos_idx[0]] = poly1.clone();
  rebuilt[pos_idx[1]] = poly2.clone();
  let var = match &args[pos_idx[2]] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated(&rebuilt)),
  };

  let npolys_message = |poly1: &Expr, poly2: &Expr| {
    let p1 = crate::syntax::expr_to_string(poly1);
    let p2 = crate::syntax::expr_to_string(poly2);
    crate::emit_message(&format!(
      "SubresultantPolynomialRemainders::npolys: {p1} and {p2} should be \
       polynomials with exact coefficients and the degree of {p1} in {var} \
       should not be less than the degree of {p2} in {var}."
    ));
  };

  // A zero first polynomial fails the degree requirement (its degree is
  // -Infinity) unless the second one is zero too.
  if matches!(poly1, Expr::Integer(0)) && !matches!(poly2, Expr::Integer(0)) {
    npolys_message(&poly1, &poly2);
    return Ok(unevaluated(&rebuilt));
  }

  let m = match max_power_int(&poly1, &var) {
    Some(d) => d as usize,
    None => return Ok(unevaluated(&rebuilt)),
  };
  let n = if matches!(poly2, Expr::Integer(0)) {
    0
  } else {
    match max_power_int(&poly2, &var) {
      Some(d) => d as usize,
      None => return Ok(unevaluated(&rebuilt)),
    }
  };
  if m < n || subres_contains_real(&poly1) || subres_contains_real(&poly2) {
    npolys_message(&poly1, &poly2);
    return Ok(unevaluated(&rebuilt));
  }

  let var_expr = Expr::Identifier(var.clone());
  let var_pow = |k: usize| -> Expr {
    match k {
      0 => Expr::Integer(1),
      1 => var_expr.clone(),
      _ => Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(var_expr.clone()),
        right: Box::new(Expr::Integer(k as i128)),
      },
    }
  };
  let poly_expr = |coeffs: &[Expr]| -> Result<Expr, InterpreterError> {
    let mut sum = Expr::Integer(0);
    for (i, c) in coeffs.iter().enumerate() {
      sum = sym_add(&sum, &sym_mul(c, &var_pow(i)));
    }
    // Symbolic coefficients are multi-term sums; wolframscript prints the
    // fully expanded flat polynomial.
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![sum].into(),
    })
  };

  let mut coeff1 = Vec::with_capacity(m + 1);
  let mut coeff2 = Vec::with_capacity(n + 1);
  for i in 0..=m {
    let c = coefficient_ast(&[
      poly1.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    coeff1.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }
  for i in 0..=n {
    let c = coefficient_ast(&[
      poly2.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    coeff2.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }

  let int_coeffs1: Option<Vec<i128>> =
    coeff1.iter().map(expr_to_i128).collect();
  let int_coeffs2: Option<Vec<i128>> =
    coeff2.iter().map(expr_to_i128).collect();
  // Modular arithmetic on symbolic coefficients is not supported.
  if modulus.is_some() && (int_coeffs1.is_none() || int_coeffs2.is_none()) {
    return Ok(unevaluated(&rebuilt));
  }

  // ---- integer / GF(p) path ----
  if let (Some(ic1), Some(ic2)) = (&int_coeffs1, &int_coeffs2) {
    let mut c1 = ic1.clone();
    let mut c2 = ic2.clone();
    if let Some(p) = modulus {
      for v in c1.iter_mut() {
        *v = v.rem_euclid(p);
      }
      for v in c2.iter_mut() {
        *v = v.rem_euclid(p);
      }
    }
    trim_int_poly(&mut c1);
    trim_int_poly(&mut c2);
    let int_entry = |v: &[i128]| -> Result<Expr, InterpreterError> {
      let coeffs: Vec<Expr> = v.iter().map(|&c| Expr::Integer(c)).collect();
      poly_expr(&coeffs)
    };

    let mut entries = vec![int_entry(&c1)?, int_entry(&c2)?];
    if c2 == [0] {
      return Ok(Expr::List(entries.into()));
    }
    if int_poly_deg(&c1) < int_poly_deg(&c2) || c1 == [0] {
      // Degrees can invert after mod-p reduction.
      npolys_message(&poly1, &poly2);
      return Ok(unevaluated(&rebuilt));
    }

    let mut r0 = c1;
    let mut r1 = c2;
    let mut delta = int_poly_deg(&r0) - int_poly_deg(&r1);
    let mut beta: i128 = if delta.is_multiple_of(2) { -1 } else { 1 };
    let mut psi: i128 = -1;
    while int_poly_deg(&r1) > 0 {
      let rem = int_prem(&r0, &r1, modulus);
      if rem == [0] {
        break;
      }
      let r2: Vec<i128> = if let Some(p) = modulus {
        let inv = mod_inverse(beta.rem_euclid(p), p);
        rem.iter().map(|c| (c * inv).rem_euclid(p)).collect()
      } else {
        rem.iter().map(|c| c / beta).collect()
      };
      entries.push(int_entry(&r2)?);
      let lcb = *r1.last().unwrap();
      if delta > 0 {
        let num = (-lcb).pow(delta as u32);
        psi = if let Some(p) = modulus {
          (num
            * mod_inverse(
              psi.rem_euclid(p).pow((delta - 1) as u32).rem_euclid(p),
              p,
            ))
          .rem_euclid(p)
        } else {
          num / psi.pow((delta - 1) as u32)
        };
      }
      delta = int_poly_deg(&r1) - int_poly_deg(&r2);
      beta = -lcb * psi.pow(delta as u32);
      if let Some(p) = modulus {
        beta = beta.rem_euclid(p);
      }
      r0 = r1;
      r1 = r2;
    }
    return Ok(Expr::List(entries.into()));
  }

  // ---- symbolic path ----
  // Coefficients are kept in expanded canonical form so exact zeros are
  // detected as Integer(0); beta divisions are exact by the subresultant
  // theory, so Cancel removes the divisor completely.
  let simp = |e: &Expr| -> Result<Expr, InterpreterError> {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![e.clone()].into(),
    })
  };
  let exact_div = |num: &Expr, den: &Expr| -> Result<Expr, InterpreterError> {
    if matches!(den, Expr::Integer(1)) {
      return Ok(num.clone());
    }
    if matches!(den, Expr::Integer(-1)) {
      return simp(&sym_mul(&Expr::Integer(-1), num));
    }
    let quotient = Expr::FunctionCall {
      name: "Cancel".to_string(),
      args: vec![Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(num.clone()),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(den.clone()),
          right: Box::new(Expr::Integer(-1)),
        }),
      }]
      .into(),
    };
    simp(&quotient)
  };
  let sym_pow = |base: &Expr, k: usize| -> Result<Expr, InterpreterError> {
    let mut acc = Expr::Integer(1);
    for _ in 0..k {
      acc = sym_mul(&acc, base);
    }
    simp(&acc)
  };

  let mut c1 = coeff1.iter().map(simp).collect::<Result<Vec<_>, _>>()?;
  let mut c2 = coeff2.iter().map(simp).collect::<Result<Vec<_>, _>>()?;
  trim_sym_poly(&mut c1);
  trim_sym_poly(&mut c2);

  let mut entries = vec![poly_expr(&c1)?, poly_expr(&c2)?];
  if sym_poly_is_zero(&c2) {
    return Ok(Expr::List(entries.into()));
  }

  let mut r0 = c1;
  let mut r1 = c2;
  let mut delta = r0.len() - r1.len();
  let mut beta = Expr::Integer(if delta % 2 == 0 { -1 } else { 1 });
  let mut psi = Expr::Integer(-1);
  while r1.len() > 1 {
    let rem = sym_prem(&r0, &r1, &simp)?;
    if sym_poly_is_zero(&rem) {
      break;
    }
    let mut r2 = Vec::with_capacity(rem.len());
    for c in &rem {
      r2.push(exact_div(c, &beta)?);
    }
    trim_sym_poly(&mut r2);
    entries.push(poly_expr(&r2)?);
    let lcb = r1.last().unwrap().clone();
    if delta > 0 {
      let neg_lcb = simp(&sym_mul(&Expr::Integer(-1), &lcb))?;
      psi = exact_div(&sym_pow(&neg_lcb, delta)?, &sym_pow(&psi, delta - 1)?)?;
    }
    delta = r1.len() - r2.len();
    beta = simp(&sym_mul(
      &sym_mul(&Expr::Integer(-1), &lcb),
      &sym_pow(&psi, delta)?,
    ))?;
    r0 = r1;
    r1 = r2;
  }
  Ok(Expr::List(entries.into()))
}

/// Degree of a trimmed ascending coefficient vector (zero polynomial: 0).
fn int_poly_deg(v: &[i128]) -> usize {
  v.len() - 1
}

/// Drop trailing zero coefficients, keeping at least one entry.
fn trim_int_poly(v: &mut Vec<i128>) {
  while v.len() > 1 && *v.last().unwrap() == 0 {
    v.pop();
  }
}

fn trim_sym_poly(v: &mut Vec<Expr>) {
  while v.len() > 1 && matches!(v.last().unwrap(), Expr::Integer(0)) {
    v.pop();
  }
}

fn sym_poly_is_zero(v: &[Expr]) -> bool {
  v.len() == 1 && matches!(v[0], Expr::Integer(0))
}

/// Pseudo-remainder prem(a, b) = lc(b)^(deg a - deg b + 1) * a mod b for
/// trimmed ascending integer coefficient vectors, deg a >= deg b.
fn int_prem(a: &[i128], b: &[i128], modulus: Option<i128>) -> Vec<i128> {
  let db = int_poly_deg(b);
  let lcb = *b.last().unwrap();
  let mut r = a.to_vec();
  let mut e = int_poly_deg(a) + 1 - db;
  while r != [0] && int_poly_deg(&r) >= db {
    let dr = int_poly_deg(&r);
    let lead = r[dr];
    let mut new = vec![0i128; dr + 1];
    for (i, c) in r.iter().enumerate() {
      new[i] = lcb * c;
    }
    for (i, c) in b.iter().enumerate() {
      new[i + dr - db] -= lead * c;
    }
    if let Some(p) = modulus {
      for v in new.iter_mut() {
        *v = v.rem_euclid(p);
      }
    }
    trim_int_poly(&mut new);
    r = new;
    e -= 1;
  }
  for _ in 0..e {
    for v in r.iter_mut() {
      *v *= lcb;
    }
    if let Some(p) = modulus {
      for v in r.iter_mut() {
        *v = v.rem_euclid(p);
      }
    }
  }
  r
}

/// Symbolic pseudo-remainder over expanded Expr coefficient vectors.
fn sym_prem(
  a: &[Expr],
  b: &[Expr],
  simp: &dyn Fn(&Expr) -> Result<Expr, InterpreterError>,
) -> Result<Vec<Expr>, InterpreterError> {
  let db = b.len() - 1;
  let lcb = b.last().unwrap().clone();
  let mut r = a.to_vec();
  let mut e = a.len() - db;
  while !sym_poly_is_zero(&r) && r.len() > db {
    let dr = r.len() - 1;
    let lead = r[dr].clone();
    let mut new = vec![Expr::Integer(0); dr + 1];
    for (i, c) in r.iter().enumerate() {
      new[i] = sym_mul(&lcb, c);
    }
    for (i, c) in b.iter().enumerate() {
      new[i + dr - db] = sym_sub(&new[i + dr - db], &sym_mul(&lead, c));
    }
    // The leading term cancels by construction.
    new[dr] = Expr::Integer(0);
    for v in new.iter_mut() {
      *v = simp(v)?;
    }
    trim_sym_poly(&mut new);
    r = new;
    e -= 1;
  }
  for _ in 0..e {
    for v in r.iter_mut() {
      *v = simp(&sym_mul(v, &lcb))?;
    }
  }
  Ok(r)
}

/// Modular inverse via the extended Euclidean algorithm (gcd(a, p) = 1).
fn mod_inverse(a: i128, p: i128) -> i128 {
  let (mut old_r, mut r) = (a, p);
  let (mut old_s, mut s) = (1i128, 0i128);
  while r != 0 {
    let q = old_r / r;
    (old_r, r) = (r, old_r - q * r);
    (old_s, s) = (s, old_s - q * s);
  }
  old_s.rem_euclid(p)
}

/// True if `expr` contains any inexact (Real/BigFloat) number.
fn subres_contains_real(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(subres_contains_real)
    }
    Expr::BinaryOp { left, right, .. } => {
      subres_contains_real(left) || subres_contains_real(right)
    }
    Expr::UnaryOp { operand, .. } => subres_contains_real(operand),
    _ => false,
  }
}

/// Compute the determinant of the Sylvester matrix for integer coefficients
/// using Gaussian elimination over integers (fraction-free).
fn sylvester_det_integer(
  coeffs1: &[i128],
  deg1: usize,
  coeffs2: &[i128],
  deg2: usize,
) -> i128 {
  let size = deg1 + deg2;
  if size == 0 {
    return 1;
  }

  // Build Sylvester matrix
  let mut matrix = vec![vec![0i128; size]; size];

  // First deg2 rows: coefficients of poly1 (from highest to lowest degree)
  for i in 0..deg2 {
    for j in 0..=deg1 {
      let col = i + deg1 - j;
      matrix[i][col] = coeffs1[j];
    }
  }

  // Last deg1 rows: coefficients of poly2
  for i in 0..deg1 {
    for j in 0..=deg2 {
      let col = i + deg2 - j;
      matrix[deg2 + i][col] = coeffs2[j];
    }
  }

  // Bareiss algorithm (fraction-free Gaussian elimination)
  bareiss_determinant(&mut matrix)
}

/// Bareiss algorithm for exact integer determinant computation
fn bareiss_determinant(matrix: &mut Vec<Vec<i128>>) -> i128 {
  let n = matrix.len();
  if n == 0 {
    return 1;
  }
  if n == 1 {
    return matrix[0][0];
  }

  let mut sign = 1i128;
  let mut prev_pivot = 1i128;

  for k in 0..n {
    // Find pivot
    let mut pivot_row = None;
    for i in k..n {
      if matrix[i][k] != 0 {
        pivot_row = Some(i);
        break;
      }
    }
    let pivot_row = match pivot_row {
      Some(r) => r,
      None => return 0, // singular matrix
    };

    if pivot_row != k {
      matrix.swap(k, pivot_row);
      sign = -sign;
    }

    let pivot = matrix[k][k];

    if k + 1 < n {
      for i in (k + 1)..n {
        for j in (k + 1)..n {
          matrix[i][j] =
            (matrix[i][j] * pivot - matrix[i][k] * matrix[k][j]) / prev_pivot;
        }
      }
    }

    prev_pivot = pivot;
  }

  sign * matrix[n - 1][n - 1]
}

/// Compute symbolic determinant using cofactor expansion
fn symbolic_determinant(matrix: &[Vec<Expr>]) -> Expr {
  let n = matrix.len();
  if n == 0 {
    return Expr::Integer(1);
  }
  if n == 1 {
    return matrix[0][0].clone();
  }
  if n == 2 {
    return sym_sub(
      &sym_mul(&matrix[0][0], &matrix[1][1]),
      &sym_mul(&matrix[0][1], &matrix[1][0]),
    );
  }

  let mut det = Expr::Integer(0);
  for j in 0..n {
    // Skip zero entries to avoid unnecessary computation
    if matches!(&matrix[0][j], Expr::Integer(0)) {
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
    let cofactor = sym_mul(&matrix[0][j], &symbolic_determinant(&minor));
    if j % 2 == 0 {
      det = sym_add(&det, &cofactor);
    } else {
      det = sym_sub(&det, &cofactor);
    }
  }
  det
}

fn sym_add(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(0), _) => b.clone(),
    (_, Expr::Integer(0)) => a.clone(),
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x + y),
    _ => Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}

fn sym_sub(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (_, Expr::Integer(0)) => a.clone(),
    (Expr::Integer(0), _) => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(b.clone()),
    },
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x - y),
    _ => Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}

fn sym_mul(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(0), _) | (_, Expr::Integer(0)) => Expr::Integer(0),
    (Expr::Integer(1), _) => b.clone(),
    (_, Expr::Integer(1)) => a.clone(),
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x * y),
    _ => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}
