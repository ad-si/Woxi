#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, expr_to_string};

// ─── Factor ─────────────────────────────────────────────────────────

/// Factor[expr] - Factor a polynomial expression
/// Threads over List.
pub fn factor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Factor expects exactly 1 argument".into(),
    ));
  }

  // Thread over List
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| factor_ast(&[item.clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }

  // First expand to canonical form
  let expanded = expand_and_combine(&args[0]);

  // Try to find the variable
  let var = match find_single_variable(&expanded) {
    Some(v) => v,
    None => return Ok(expanded), // multivariate or constant — return as is
  };

  // Extract integer coefficients
  let coeffs = match extract_poly_coeffs(&expanded, &var) {
    Some(c) => c,
    None => return Ok(expanded), // non-integer coefficients
  };

  // Factor out GCD of coefficients
  let gcd_coeff = coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if gcd_coeff == 0 {
    return Ok(Expr::Integer(0));
  }

  let reduced_coeffs: Vec<i128> =
    coeffs.iter().map(|c| c / gcd_coeff).collect();

  // Factor out leading negative
  let (sign, reduced_coeffs) =
    if reduced_coeffs.last().map(|&c| c < 0).unwrap_or(false) {
      (
        -1i128,
        reduced_coeffs.iter().map(|c| -c).collect::<Vec<_>>(),
      )
    } else {
      (1, reduced_coeffs)
    };
  let overall = gcd_coeff * sign;

  // Factor the monic-ish polynomial
  let factors = factor_integer_poly(&reduced_coeffs, &var);

  if factors.len() <= 1 && overall == 1 {
    // Couldn't factor further
    return Ok(expanded);
  }

  // Build result: overall * factor1 * factor2 * ...
  let mut result_factors: Vec<Expr> = Vec::new();

  if overall != 1 {
    result_factors.push(Expr::Integer(overall));
  }

  if factors.is_empty() {
    result_factors.push(coeffs_to_expr(&reduced_coeffs, &var));
  } else {
    // Group identical factors: (1+x)*(1+x) → (1+x)^2
    let mut grouped: Vec<(Expr, i128)> = Vec::new();
    for f in &factors {
      let key = expr_to_string(f);
      if let Some(entry) =
        grouped.iter_mut().find(|(e, _)| expr_to_string(e) == key)
      {
        entry.1 += 1;
      } else {
        grouped.push((f.clone(), 1));
      }
    }
    for (factor, count) in grouped {
      if count == 1 {
        result_factors.push(factor);
      } else {
        result_factors.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(factor),
          right: Box::new(Expr::Integer(count)),
        });
      }
    }
  }

  if result_factors.len() == 1 {
    return Ok(result_factors.remove(0));
  }

  Ok(build_product(result_factors))
}

/// GCD of two i128 values.
pub fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Factor an integer polynomial given as coefficients.
/// Returns a list of factor expressions, or empty if can't factor.
pub fn factor_integer_poly(coeffs: &[i128], var: &str) -> Vec<Expr> {
  // Remove trailing zeros (factor out x^k)
  let leading_zeros = coeffs.iter().take_while(|&&c| c == 0).count();
  let trimmed = &coeffs[leading_zeros..];

  let mut factors: Vec<Expr> = Vec::new();

  // Add x^k factor if leading zeros
  if leading_zeros > 0 {
    if leading_zeros == 1 {
      factors.push(Expr::Identifier(var.to_string()));
    } else {
      factors.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier(var.to_string())),
        right: Box::new(Expr::Integer(leading_zeros as i128)),
      });
    }
  }

  if trimmed.len() <= 1 {
    // Constant or empty
    if trimmed.len() == 1 && trimmed[0] != 1 {
      factors.push(Expr::Integer(trimmed[0]));
    }
    return factors;
  }

  // Try to find rational roots and factor
  let mut remaining = trimmed.to_vec();

  loop {
    if remaining.len() <= 1 {
      break;
    }
    if remaining.len() == 2 {
      // Linear: ax + b → (b + a*x) but represented as canonical form
      factors.push(linear_to_expr(remaining[0], remaining[1], var));
      break;
    }

    // Try rational root theorem
    match find_integer_root(&remaining) {
      Some(root) => {
        // Factor out (x - root) which in canonical form is (root_neg + x) → (-root + x)
        factors.push(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(-root)),
          right: Box::new(Expr::Identifier(var.to_string())),
        });
        remaining = divide_by_root(&remaining, root);
      }
      None => {
        // Can't find more integer roots — try polynomial trial division
        let sub_factors = try_factor_no_rational_roots(&remaining, var);
        if sub_factors.is_empty() {
          if remaining != [1] {
            factors.push(coeffs_to_expr(&remaining, var));
          }
        } else {
          factors.extend(sub_factors);
        }
        break;
      }
    }
  }

  // Sort factors by: constant term, then degree, then string representation
  factors.sort_by(|a, b| {
    let ca = factor_constant_term(a);
    let cb = factor_constant_term(b);
    ca.cmp(&cb)
      .then_with(|| factor_degree(a).cmp(&factor_degree(b)))
      .then_with(|| {
        // For same degree, sort by number of terms (fewer terms first)
        factor_term_count(a).cmp(&factor_term_count(b))
      })
      .then_with(|| {
        // For same degree and term count, sort by first non-constant coefficient
        factor_first_nonconst_coeff(a).cmp(&factor_first_nonconst_coeff(b))
      })
      .then_with(|| expr_to_string(a).cmp(&expr_to_string(b)))
  });

  factors
}

/// Get the constant term of a factor expression for sorting.
/// Recursively descends into the leftmost leaf of Plus chains.
pub fn factor_constant_term(expr: &Expr) -> i128 {
  match expr {
    Expr::Integer(n) => *n,
    Expr::Identifier(_) => 0, // x has constant term 0
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      ..
    } => {
      // Recurse into left child to find the constant term
      factor_constant_term(left)
    }
    _ => 0,
  }
}

/// Get the degree of a factor expression for sorting.
pub fn factor_degree(expr: &Expr) -> usize {
  let s = expr_to_string(expr);
  // Count the highest power of the variable
  // Look for x^N patterns
  let mut max_deg = 0usize;
  for cap in s.split('^') {
    // After '^', try to parse the number
    if let Ok(n) = cap
      .chars()
      .take_while(|c| c.is_ascii_digit())
      .collect::<String>()
      .parse::<usize>()
      && n > max_deg
    {
      max_deg = n;
    }
  }
  // If no power found but contains a variable, degree is 1
  if max_deg == 0 && s.chars().any(|c| c.is_ascii_lowercase()) {
    max_deg = 1;
  }
  max_deg
}

/// Count the number of additive terms in a factor expression for sorting.
pub fn factor_term_count(expr: &Expr) -> usize {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => factor_term_count(left) + factor_term_count(right),
    _ => 1,
  }
}

/// Get the first non-constant coefficient of a factor expression for sorting.
/// This finds the coefficient of the lowest-degree non-constant term.
pub fn factor_first_nonconst_coeff(expr: &Expr) -> i128 {
  // Convert the expression back to string and parse the first non-constant term's sign
  let s = expr_to_string(expr);
  // Find the first term with a variable (after the constant)
  // Parse the sign: look for first occurrence of the variable
  // The pattern will be like "1 + x", "1 - x", "1 + x^5", "1 - x^5"
  // After "1", look for " + " or " - " before the variable
  let parts: Vec<&str> =
    s.splitn(2, |c: char| c.is_ascii_lowercase()).collect();
  if parts.len() < 2 {
    return 0;
  }
  let prefix = parts[0].trim_end();
  if prefix.ends_with('+') || prefix.ends_with("+ ") {
    1
  } else if prefix.ends_with('-') || prefix.ends_with("- ") {
    -1
  } else if prefix.ends_with('*') {
    // coefficient * variable — extract the number before *
    let before_star = prefix.trim_end_matches('*').trim();
    // Find the last number in this prefix
    let num_str: String = before_star
      .chars()
      .rev()
      .take_while(|c| c.is_ascii_digit() || *c == '-')
      .collect::<String>()
      .chars()
      .rev()
      .collect();
    num_str.parse().unwrap_or(0)
  } else {
    0
  }
}

/// Build a linear expression from coefficients: c0 + c1*x
pub fn linear_to_expr(c0: i128, c1: i128, var: &str) -> Expr {
  if c1 == 1 {
    if c0 == 0 {
      Expr::Identifier(var.to_string())
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(c0)),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    }
  } else {
    let var_part = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(c1)),
      right: Box::new(Expr::Identifier(var.to_string())),
    };
    if c0 == 0 {
      var_part
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(c0)),
        right: Box::new(var_part),
      }
    }
  }
}

/// Find an integer root of polynomial with given coefficients.
/// Uses rational root theorem: possible roots are ±(factors of c0) / (factors of leading coeff).
pub fn find_integer_root(coeffs: &[i128]) -> Option<i128> {
  if coeffs.is_empty() {
    return None;
  }
  let c0 = coeffs[0]; // constant term
  let lead = coeffs[coeffs.len() - 1]; // leading coefficient

  if c0 == 0 {
    return Some(0);
  }

  // Get divisors of c0 and lead
  let c0_divs = integer_divisors(c0.abs());
  let lead_divs = integer_divisors(lead.abs());

  for &p in &c0_divs {
    for &q in &lead_divs {
      // Try root = p/q (only if it's an integer)
      if p % q == 0 {
        let root = p / q;
        if evaluate_poly(coeffs, root) == 0 {
          return Some(root);
        }
        if evaluate_poly(coeffs, -root) == 0 {
          return Some(-root);
        }
      }
    }
  }
  None
}

/// Evaluate polynomial at integer x.
pub fn evaluate_poly(coeffs: &[i128], x: i128) -> i128 {
  let mut result: i128 = 0;
  let mut power: i128 = 1;
  for &c in coeffs {
    result = result
      .checked_add(c.checked_mul(power).unwrap_or(i128::MAX))
      .unwrap_or(i128::MAX);
    if result == i128::MAX {
      return result; // overflow guard
    }
    power = power.checked_mul(x).unwrap_or(i128::MAX);
  }
  result
}

/// Get all positive divisors of n.
pub fn integer_divisors(n: i128) -> Vec<i128> {
  if n == 0 {
    return vec![1];
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
  divs.sort();
  divs
}

/// Divide polynomial by (x - root) using synthetic division.
pub fn divide_by_root(coeffs: &[i128], root: i128) -> Vec<i128> {
  // coeffs[i] = coefficient of x^i, so coeffs = [c0, c1, c2, ...]
  // We need to divide by (x - root)
  let n = coeffs.len();
  if n <= 1 {
    return vec![];
  }
  let mut result = vec![0i128; n - 1];
  // Synthetic division: work from highest power down
  result[n - 2] = coeffs[n - 1];
  for i in (0..n - 2).rev() {
    result[i] = coeffs[i + 1] + root * result[i + 1];
  }
  result
}

/// Polynomial long division: divide `num` by `den`.
/// Returns (quotient, remainder) as coefficient vectors.
/// Coefficients are stored as [c0, c1, c2, ...] (c_i is coefficient of x^i).
/// Only works when division is exact over integers (remainder should be zero for our use).
pub fn poly_div(num: &[i128], den: &[i128]) -> Option<(Vec<i128>, Vec<i128>)> {
  let n = num.len();
  let m = den.len();
  if m == 0 || m > n {
    return None;
  }
  let lead_den = *den.last().unwrap();
  if lead_den == 0 {
    return None;
  }

  let mut rem = num.to_vec();
  let quot_len = n - m + 1;
  let mut quot = vec![0i128; quot_len];

  for i in (0..quot_len).rev() {
    if rem[i + m - 1] % lead_den != 0 {
      // Not exactly divisible over integers
      return None;
    }
    let coeff = rem[i + m - 1] / lead_den;
    quot[i] = coeff;
    for j in 0..m {
      rem[i + j] -= coeff * den[j];
    }
  }

  // Trim trailing zeros from remainder
  while rem.len() > 1 && *rem.last().unwrap() == 0 {
    rem.pop();
  }

  Some((quot, rem))
}

/// Compute the n-th cyclotomic polynomial as integer coefficients.
/// Φ_1(x) = x - 1
/// Φ_n(x) = (x^n - 1) / ∏_{d|n, d<n} Φ_d(x)
pub fn cyclotomic_poly(n: u64) -> Vec<i128> {
  if n == 1 {
    return vec![-1, 1]; // x - 1
  }

  // Start with x^n - 1
  let mut result = vec![0i128; (n + 1) as usize];
  result[0] = -1;
  result[n as usize] = 1;

  // Divide by Φ_d(x) for all proper divisors d of n
  let divs = divisors_of(n);
  for d in divs {
    if d < n {
      let phi_d = cyclotomic_poly(d);
      if let Some((q, _)) = poly_div(&result, &phi_d) {
        result = q;
      }
    }
  }

  result
}

/// Get all divisors of n in sorted order.
pub fn divisors_of(n: u64) -> Vec<u64> {
  let mut divs = Vec::new();
  let mut i = 1u64;
  while i * i <= n {
    if n.is_multiple_of(i) {
      divs.push(i);
      if i != n / i {
        divs.push(n / i);
      }
    }
    i += 1;
  }
  divs.sort();
  divs
}

/// Try to factor a polynomial (with no rational roots) using polynomial trial division.
/// First tries cyclotomic factors, then Kronecker-style trial division for small degrees.
pub fn try_factor_no_rational_roots(coeffs: &[i128], var: &str) -> Vec<Expr> {
  let deg = coeffs.len() - 1;
  if deg <= 1 {
    return vec![];
  }

  // Try cyclotomic polynomial division:
  // Check divisors up to reasonable size
  let mut remaining = coeffs.to_vec();
  let mut factors: Vec<Vec<i128>> = Vec::new();

  // Try cyclotomic polynomials Φ_n for n up to 2*deg
  // (any cyclotomic factor of a degree-d poly has n ≤ some bound)
  let max_n = (2 * deg) as u64;
  // Collect cyclotomic polys sorted by degree (ascending) for greedy factoring
  let mut cyclo_candidates: Vec<(u64, Vec<i128>)> = Vec::new();
  for n in 2..=max_n {
    let cp = cyclotomic_poly(n);
    if cp.len() - 1 <= deg {
      cyclo_candidates.push((n, cp));
    }
  }
  // Sort by degree so we try smaller factors first
  cyclo_candidates.sort_by_key(|(_, c)| c.len());

  let mut changed = true;
  while changed {
    changed = false;
    for (_n, cp) in &cyclo_candidates {
      if cp.len() > remaining.len() {
        continue;
      }
      if let Some((q, rem)) = poly_div(&remaining, cp)
        && rem.iter().all(|&c| c == 0)
      {
        factors.push(cp.clone());
        remaining = q;
        changed = true;
        // Trim trailing zeros
        while remaining.len() > 1 && *remaining.last().unwrap() == 0 {
          remaining.pop();
        }
        if remaining.len() <= 1 {
          break;
        }
      }
    }
    if remaining.len() <= 1 {
      break;
    }
  }

  if factors.is_empty() {
    // No cyclotomic factors found — try Kronecker's method for small-degree polynomials
    return try_kronecker_factor(coeffs, var);
  }

  // We found cyclotomic factors; handle the remainder
  if remaining.len() > 2 {
    // Try to factor the remainder further with Kronecker
    let sub = try_kronecker_factor(&remaining, var);
    if sub.is_empty() {
      factors.push(remaining);
    } else {
      return factors
        .iter()
        .map(|f| coeffs_to_expr(f, var))
        .chain(sub)
        .collect();
    }
  } else if remaining.len() > 1 || (remaining.len() == 1 && remaining[0] != 1) {
    factors.push(remaining);
  }

  factors.iter().map(|f| coeffs_to_expr(f, var)).collect()
}

/// Kronecker's method for factoring integer polynomials of small degree.
/// Try all monic integer polynomial divisors of degree 2..=deg/2
/// by evaluating at enough points to determine the candidate factor.
pub fn try_kronecker_factor(coeffs: &[i128], var: &str) -> Vec<Expr> {
  let deg = coeffs.len() - 1;
  if deg <= 3 {
    // For degree 2-3, if no rational roots exist, it's irreducible over Z
    return vec![];
  }

  // For degree 4+, try factoring as product of two polynomials
  // Evaluate at points 0, 1, -1, 2, -2, ...
  let max_trial_degree = deg / 2;
  if max_trial_degree > 20 {
    // Too expensive for Kronecker
    return vec![];
  }

  // Evaluate the polynomial at points 0, 1, -1, 2, -2, ... (need max_trial_degree + 1 points)
  let num_points = max_trial_degree + 1;
  let eval_points: Vec<i128> = (0..num_points as i128)
    .flat_map(|i| if i == 0 { vec![0] } else { vec![i, -i] })
    .take(num_points)
    .collect();

  let poly_vals: Vec<i128> = eval_points
    .iter()
    .map(|&x| evaluate_poly(coeffs, x))
    .collect();

  // For each trial degree d (2..=max_trial_degree), try to find a factor
  for trial_deg in 2..=max_trial_degree {
    let n_pts = trial_deg + 1;
    // Get divisors of poly value at each eval point
    let mut divisor_sets: Vec<Vec<i128>> = Vec::new();
    for i in 0..n_pts {
      let val = poly_vals[i];
      if val == 0 {
        // The trial factor also evaluates to 0 at this point
        divisor_sets.push(vec![0]);
      } else {
        let abs_divs = integer_divisors(val.abs());
        let mut divs: Vec<i128> = Vec::new();
        for &d in &abs_divs {
          divs.push(d);
          divs.push(-d);
        }
        divisor_sets.push(divs);
      }
    }

    // Try all combinations of divisor values and interpolate
    let combos = cartesian_product(&divisor_sets);
    for combo in combos {
      // Interpolate: find a polynomial of degree trial_deg with values combo at eval_points
      if let Some(factor_coeffs) =
        lagrange_interpolate_integer(&eval_points[..n_pts], &combo)
      {
        // Check if leading coeff is ±1 (monic or neg-monic) and degree matches
        if factor_coeffs.len() != trial_deg + 1 {
          continue;
        }
        let lead = *factor_coeffs.last().unwrap();
        if lead == 0 {
          continue;
        }

        // Try dividing
        if let Some((q, rem)) = poly_div(coeffs, &factor_coeffs)
          && rem.iter().all(|&c| c == 0)
        {
          // Found a factor! Recursively factor both parts
          let mut result = Vec::new();
          let sub1 = factor_sub_poly(&factor_coeffs, var);
          let sub2 = factor_sub_poly(&q, var);
          result.extend(sub1);
          result.extend(sub2);
          return result;
        }
      }
    }
  }

  vec![]
}

/// Factor a sub-polynomial by trying rational roots first, then trial division.
pub fn factor_sub_poly(coeffs: &[i128], var: &str) -> Vec<Expr> {
  if coeffs.len() <= 1 {
    if coeffs.len() == 1 && coeffs[0] != 1 {
      return vec![Expr::Integer(coeffs[0])];
    }
    return vec![];
  }
  if coeffs.len() == 2 {
    return vec![linear_to_expr(coeffs[0], coeffs[1], var)];
  }

  // Try rational roots first
  let mut remaining = coeffs.to_vec();
  let mut factors: Vec<Expr> = Vec::new();
  loop {
    match find_integer_root(&remaining) {
      Some(root) => {
        factors.push(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(-root)),
          right: Box::new(Expr::Identifier(var.to_string())),
        });
        remaining = divide_by_root(&remaining, root);
        if remaining.len() <= 1 {
          break;
        }
        if remaining.len() == 2 {
          factors.push(linear_to_expr(remaining[0], remaining[1], var));
          return factors;
        }
      }
      None => break,
    }
  }

  if remaining.len() > 1 {
    // Irreducible remainder
    factors.push(coeffs_to_expr(&remaining, var));
  }
  factors
}

/// Compute cartesian product of divisor sets (with size limit to avoid explosion).
pub fn cartesian_product(sets: &[Vec<i128>]) -> Vec<Vec<i128>> {
  if sets.is_empty() {
    return vec![vec![]];
  }

  // Limit total combinations to avoid exponential blowup
  let total: usize = sets.iter().map(|s| s.len()).product();
  if total > 10000 {
    return vec![];
  }

  let mut result = vec![vec![]];
  for set in sets {
    let mut new_result = Vec::new();
    for combo in &result {
      for &val in set {
        let mut new_combo = combo.clone();
        new_combo.push(val);
        new_result.push(new_combo);
      }
    }
    result = new_result;
  }
  result
}

/// Lagrange interpolation over integers.
/// Given points (x_i, y_i), find polynomial with integer coefficients.
/// Returns None if the interpolation doesn't yield integer coefficients.
pub fn lagrange_interpolate_integer(
  xs: &[i128],
  ys: &[i128],
) -> Option<Vec<i128>> {
  let n = xs.len();
  if n == 0 {
    return None;
  }

  // Use the Newton form of interpolation (divided differences)
  let mut divided_diffs = ys.to_vec();

  for j in 1..n {
    for i in (j..n).rev() {
      let num = divided_diffs[i] - divided_diffs[i - 1];
      let den = xs[i] - xs[i - j];
      if den == 0 {
        return None;
      }
      if num % den != 0 {
        return None; // Not integer coefficients
      }
      divided_diffs[i] = num / den;
    }
  }

  // Convert Newton form to standard coefficients
  // P(x) = d[0] + d[1]*(x-x0) + d[2]*(x-x0)*(x-x1) + ...
  let mut coeffs = vec![0i128; n];
  // Build up: start with d[n-1], multiply by (x - x_{n-2}), add d[n-2], etc.
  coeffs[0] = divided_diffs[n - 1];
  for i in (0..n - 1).rev() {
    // Multiply current poly by (x - xs[i])
    // coeffs represents polynomial, multiply by (x - xs[i])
    // New coeffs[j] = coeffs[j-1] - xs[i]*coeffs[j]
    let mut new_coeffs = vec![0i128; n];
    for j in (0..n).rev() {
      new_coeffs[j] = if j > 0 { coeffs[j - 1] } else { 0 } - xs[i] * coeffs[j];
    }
    new_coeffs[0] += divided_diffs[i];
    coeffs = new_coeffs;
  }

  // Trim trailing zeros
  while coeffs.len() > 1 && *coeffs.last().unwrap() == 0 {
    coeffs.pop();
  }

  Some(coeffs)
}

/// FactorList[poly] - list irreducible factors of a polynomial with exponents
pub fn factor_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FactorList expects exactly 1 argument".into(),
    ));
  }

  // First, factor the polynomial
  let factored = factor_ast(args)?;

  // Decompose the factored form into {factor, exponent} pairs
  let mut pairs: Vec<Expr> = Vec::new();
  let mut numeric_coeff = Expr::Integer(1);

  decompose_product(&factored, &mut pairs, &mut numeric_coeff);

  // Build result: {{numeric_coeff, 1}, {factor1, exp1}, ...}
  let mut result = vec![Expr::List(vec![numeric_coeff, Expr::Integer(1)])];
  result.extend(pairs);

  Ok(Expr::List(result))
}

/// Decompose a factored expression into {factor, exponent} pairs.
/// Handles Times[...], Power[base, exp], and literal factors.
pub fn decompose_product(
  expr: &Expr,
  pairs: &mut Vec<Expr>,
  numeric_coeff: &mut Expr,
) {
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      for arg in args {
        decompose_product(arg, pairs, numeric_coeff);
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      decompose_product(left, pairs, numeric_coeff);
      decompose_product(right, pairs, numeric_coeff);
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      // base^exp
      if is_numeric_expr(left) {
        // numeric^exp: handle as coefficient
        *numeric_coeff = crate::functions::math_ast::times_ast(&[
          numeric_coeff.clone(),
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: left.clone(),
            right: right.clone(),
          },
        ])
        .unwrap_or(expr.clone());
      } else {
        pairs.push(Expr::List(vec![*left.clone(), *right.clone()]));
      }
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if is_numeric_expr(&args[0]) {
        *numeric_coeff = crate::functions::math_ast::times_ast(&[
          numeric_coeff.clone(),
          expr.clone(),
        ])
        .unwrap_or(expr.clone());
      } else {
        pairs.push(Expr::List(vec![args[0].clone(), args[1].clone()]));
      }
    }
    Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_) => {
      *numeric_coeff = crate::functions::math_ast::times_ast(&[
        numeric_coeff.clone(),
        expr.clone(),
      ])
      .unwrap_or(expr.clone());
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      *numeric_coeff = crate::functions::math_ast::times_ast(&[
        numeric_coeff.clone(),
        expr.clone(),
      ])
      .unwrap_or(expr.clone());
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      // -expr: multiply coeff by -1 and decompose inner
      *numeric_coeff = crate::functions::math_ast::times_ast(&[
        numeric_coeff.clone(),
        Expr::Integer(-1),
      ])
      .unwrap_or(Expr::Integer(-1));
      decompose_product(operand, pairs, numeric_coeff);
    }
    _ => {
      // Any other expression is a factor with exponent 1
      pairs.push(Expr::List(vec![expr.clone(), Expr::Integer(1)]));
    }
  }
}

/// Check if an expression is purely numeric
pub fn is_numeric_expr(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
    || matches!(expr,
      Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2
    )
}

// ─── FactorTerms ────────────────────────────────────────────────────

/// FactorTerms[poly] - Pull out the greatest common numerical factor
/// from the terms of a polynomial.
/// FactorTerms[poly, x] - Pull out factors that don't depend on x.
pub fn factor_terms_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FactorTerms expects 1 or 2 arguments".into(),
    ));
  }

  // Thread over List
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        let mut new_args = vec![item.clone()];
        if args.len() > 1 {
          new_args.push(args[1].clone());
        }
        factor_terms_ast(&new_args)
      })
      .collect();
    return Ok(Expr::List(results?));
  }

  // Expand to canonical form
  let expanded = expand_and_combine(&args[0]);

  // Collect additive terms
  let terms = collect_additive_terms(&expanded);
  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }
  if terms.len() == 1 {
    return Ok(expanded);
  }

  // First, factor out the numeric GCD from all terms
  let numeric_factored = factor_terms_numeric(&expanded, &terms)?;

  if args.len() == 2 {
    // FactorTerms[poly, x] — also factor out terms not depending on x
    let var = match &args[1] {
      Expr::Identifier(s) => s.clone(),
      _ => return Ok(numeric_factored),
    };
    // Apply variable-dependent factoring to the inner expression
    // (after numeric factor has been pulled out)
    match &numeric_factored {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        // numeric_factor * inner — apply var factoring to inner
        let inner_factored = factor_terms_wrt_var(right, &var)?;
        return Ok(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: left.clone(),
          right: Box::new(inner_factored),
        });
      }
      _ => {
        // No numeric factor was extracted
        return factor_terms_wrt_var(&numeric_factored, &var);
      }
    }
  }

  Ok(numeric_factored)
}

/// Extract the rational coefficient (numerator, denominator) from a term.
/// Returns (numerator, denominator) as i128.
fn extract_rational_coeff(term: &Expr) -> Option<(i128, i128)> {
  match term {
    // Pure integer
    Expr::Integer(n) => Some((*n, 1)),
    // Rational number
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n, *d))
      } else {
        None
      }
    }
    // Negation
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let (n, d) = extract_rational_coeff(operand)?;
      Some((-n, d))
    }
    // Product: extract numeric factors
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    }
    | Expr::FunctionCall { name: _, args: _ }
      if matches!(term, Expr::FunctionCall { name, .. } if name == "Times")
        || matches!(
          term,
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            ..
          }
        ) =>
    {
      let factors = collect_multiplicative_factors(term);
      let mut num: i128 = 1;
      let mut den: i128 = 1;
      for f in &factors {
        match f {
          Expr::Integer(n) => num *= n,
          Expr::FunctionCall { name, args }
            if name == "Rational" && args.len() == 2 =>
          {
            if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
              num *= n;
              den *= d;
            }
          }
          Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand,
          } => match operand.as_ref() {
            Expr::Integer(n) => num *= -n,
            Expr::FunctionCall { name, args }
              if name == "Rational" && args.len() == 2 =>
            {
              if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1])
              {
                num *= -n;
                den *= d;
              }
            }
            _ => {} // non-numeric factor, ignore (it's a variable)
          },
          _ => {} // non-numeric factor (variable part)
        }
      }
      Some((num, den))
    }
    // Identifier or other: coefficient is 1
    Expr::Identifier(_)
    | Expr::BinaryOp {
      op: BinaryOperator::Power,
      ..
    } => Some((1, 1)),
    _ => {
      // Use decompose_term as fallback
      let (coeff, _, _) = decompose_term(term);
      match &coeff {
        Expr::Integer(n) => Some((*n, 1)),
        Expr::FunctionCall { name, args }
          if name == "Rational" && args.len() == 2 =>
        {
          if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
            Some((*n, *d))
          } else {
            None
          }
        }
        _ => None,
      }
    }
  }
}

/// Factor out the GCD of numeric coefficients from all terms.
fn factor_terms_numeric(
  expanded: &Expr,
  terms: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Extract rational coefficients from each term
  let mut coeffs: Vec<(i128, i128)> = Vec::new();
  for term in terms {
    match extract_rational_coeff(term) {
      Some(pair) => coeffs.push(pair),
      None => return Ok(expanded.clone()), // non-numeric coefficient, return as-is
    }
  }

  if coeffs.is_empty() {
    return Ok(expanded.clone());
  }

  // Compute GCD of all numerators and LCM of all denominators
  let num_gcd = coeffs
    .iter()
    .map(|(n, _)| *n)
    .filter(|&n| n != 0)
    .fold(0i128, gcd_i128);
  let den_lcm = coeffs.iter().map(|(_, d)| *d).fold(1i128, lcm_i128);

  // If all non-zero numerators are negative, factor out the negative sign
  let all_negative =
    coeffs.iter().filter(|(n, _)| *n != 0).all(|(n, _)| *n < 0);
  let sign: i128 = if all_negative { -1 } else { 1 };
  let num_gcd = num_gcd * sign;

  if num_gcd.abs() <= 1 && den_lcm == 1 {
    return Ok(expanded.clone());
  }

  // The overall factor is num_gcd / den_lcm
  // Divide each term by this factor
  let mut new_terms: Vec<Expr> = Vec::new();
  for (term, (n, d)) in terms.iter().zip(coeffs.iter()) {
    // New coefficient: (n/d) / (num_gcd/den_lcm) = (n * den_lcm) / (d * num_gcd)
    let new_n = n * den_lcm / (d * num_gcd);
    let var_factors = extract_non_numeric_factors(term);

    let coeff_expr = if new_n == 1 && !var_factors.is_empty() {
      None
    } else if new_n == -1 && !var_factors.is_empty() {
      None // handle sign below
    } else {
      Some(Expr::Integer(new_n.abs()))
    };

    let var_part = if var_factors.is_empty() {
      None
    } else if var_factors.len() == 1 {
      Some(var_factors[0].clone())
    } else {
      let mut product = var_factors[0].clone();
      for f in &var_factors[1..] {
        product = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(product),
          right: Box::new(f.clone()),
        };
      }
      Some(product)
    };

    let term_expr = match (coeff_expr, var_part) {
      (Some(c), Some(v)) => Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(c),
        right: Box::new(v),
      },
      (Some(c), None) => c,
      (None, Some(v)) => v,
      (None, None) => Expr::Integer(1),
    };

    let final_term = if new_n < 0 {
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(term_expr),
      }
    } else {
      term_expr
    };

    new_terms.push(final_term);
  }

  // Build the inner sum
  let inner = build_sum(new_terms);

  // Build the overall factor
  let factor = if den_lcm == 1 {
    Expr::Integer(num_gcd)
  } else {
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(num_gcd), Expr::Integer(den_lcm)],
    }
  };

  // Return factor * inner (or just inner if factor is 1)
  if matches!(&factor, Expr::Integer(1)) {
    Ok(inner)
  } else {
    Ok(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(factor),
      right: Box::new(inner),
    })
  }
}

/// FactorTerms[poly, x] — factor out terms that don't depend on x.
/// Groups terms by power of x, collects the coefficient of each power,
/// then factors out the polynomial GCD of those coefficient expressions.
fn factor_terms_wrt_var(
  expr: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  let terms = collect_additive_terms(expr);

  // Collect coefficients for each power of x
  let mut power_coeffs: std::collections::BTreeMap<i128, Vec<Expr>> =
    std::collections::BTreeMap::new();

  for term in &terms {
    let (power, coeff) = term_var_power_and_coeff(term, var);
    power_coeffs.entry(power).or_default().push(coeff);
  }

  // Sum and simplify coefficients for each power
  let mut summed_coeffs: Vec<(i128, Expr)> = Vec::new();
  for (power, coeffs) in &power_coeffs {
    let sum = if coeffs.len() == 1 {
      coeffs[0].clone()
    } else {
      build_sum(coeffs.clone())
    };
    let simplified = expand_and_combine(&sum);
    summed_coeffs.push((*power, simplified));
  }

  if summed_coeffs.is_empty() {
    return Ok(Expr::Integer(0));
  }

  // Try to find a common symbolic factor among all coefficient expressions.
  // First, factor out the numeric GCD across all coefficient terms.
  let mut all_coeff_terms: Vec<Expr> = Vec::new();
  for (_, coeff) in &summed_coeffs {
    all_coeff_terms.extend(collect_additive_terms(coeff));
  }

  // Try to find a polynomial GCD of the coefficient expressions
  // by looking for a common variable and using polynomial GCD
  let coeff_exprs: Vec<&Expr> = summed_coeffs.iter().map(|(_, c)| c).collect();

  // Check if all coefficients share a common factor
  // Try to find a single variable common to all coefficient expressions
  let mut common_var: Option<String> = None;
  for coeff in &coeff_exprs {
    if let Some(v) = find_single_variable(coeff)
      && v != var
    {
      common_var = Some(v);
      break;
    }
  }

  // If we found a common variable, try polynomial GCD
  if let Some(cv) = &common_var {
    let mut poly_coeffs: Vec<Vec<i128>> = Vec::new();
    let mut all_valid = true;

    for coeff in &coeff_exprs {
      match extract_poly_coeffs(coeff, cv) {
        Some(c) => poly_coeffs.push(c),
        None => {
          all_valid = false;
          break;
        }
      }
    }

    if all_valid && poly_coeffs.len() >= 2 {
      // Compute GCD of all coefficient polynomials
      let mut gcd_poly = poly_coeffs[0].clone();
      for p in &poly_coeffs[1..] {
        if let Some(g) =
          crate::functions::polynomial_ast::poly_gcd(&gcd_poly, p)
        {
          gcd_poly = g;
        } else {
          // GCD computation failed, fall through to numeric GCD
          return factor_terms_numeric(expr, &terms);
        }
      }

      // Check if GCD is non-trivial (not just a constant ±1)
      let is_trivial = gcd_poly.len() <= 1
        && gcd_poly.first().map(|c| c.abs()).unwrap_or(0) <= 1;

      if !is_trivial {
        // Divide each coefficient by the GCD polynomial
        let mut new_terms: Vec<Expr> = Vec::new();
        for (power, coeff) in &summed_coeffs {
          if let Some(c) = extract_poly_coeffs(coeff, cv) {
            let (quotient, _remainder) =
              match crate::functions::polynomial_ast::poly_div(&c, &gcd_poly) {
                Some(qr) => qr,
                None => return factor_terms_numeric(expr, &terms),
              };
            let q_expr = coeffs_to_expr(&quotient, cv);

            let var_power = if *power == 0 {
              q_expr
            } else if *power == 1 {
              Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(q_expr),
                right: Box::new(Expr::Identifier(var.to_string())),
              }
            } else {
              Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(q_expr),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(Expr::Identifier(var.to_string())),
                  right: Box::new(Expr::Integer(*power)),
                }),
              }
            };
            new_terms.push(var_power);
          }
        }

        let inner = expand_and_combine(&build_sum(new_terms));
        let gcd_expr = coeffs_to_expr(&gcd_poly, cv);

        // Factor out numeric GCD from both parts
        let gcd_terms = collect_additive_terms(&gcd_expr);
        let factored_gcd =
          factor_terms_numeric(&gcd_expr, &gcd_terms).unwrap_or(gcd_expr);
        let inner_terms = collect_additive_terms(&inner);
        let factored_inner =
          factor_terms_numeric(&inner, &inner_terms).unwrap_or(inner);

        return Ok(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(factored_gcd),
          right: Box::new(factored_inner),
        });
      }
    }
  }

  // Fall back to numeric GCD factoring
  factor_terms_numeric(expr, &terms)
}

/// Extract non-numeric factors from a term (strip numeric coefficient including Rational).
fn extract_non_numeric_factors(term: &Expr) -> Vec<Expr> {
  match term {
    Expr::Integer(_) | Expr::Real(_) => vec![],
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      vec![] // Rational is purely numeric
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => extract_non_numeric_factors(operand),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    } => {
      let factors = collect_multiplicative_factors(term);
      factors
        .into_iter()
        .filter(|f| {
          !matches!(f, Expr::Integer(_) | Expr::Real(_))
            && !matches!(f, Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2)
            && !matches!(f, Expr::UnaryOp { op: crate::syntax::UnaryOperator::Minus, operand } if matches!(operand.as_ref(), Expr::Integer(_) | Expr::Real(_)))
        })
        .collect()
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      args
        .iter()
        .filter(|f| {
          !matches!(f, Expr::Integer(_) | Expr::Real(_))
            && !matches!(f, Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2)
        })
        .cloned()
        .collect()
    }
    _ => vec![term.clone()],
  }
}

fn lcm_i128(a: i128, b: i128) -> i128 {
  if a == 0 || b == 0 {
    0
  } else {
    (a / gcd_i128(a, b)) * b
  }
}

// ─── FactorSquareFree ──────────────────────────────────────────────

/// FactorSquareFree[poly] - Square-free factorization via Yun's algorithm
/// Groups factors by multiplicity without fully factoring the square-free parts.
pub fn factor_square_free_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FactorSquareFree expects exactly 1 argument".into(),
    ));
  }

  // Thread over List
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| factor_square_free_ast(&[item.clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }

  let expanded = expand_and_combine(&args[0]);

  let var = match find_single_variable(&expanded) {
    Some(v) => v,
    None => return Ok(expanded),
  };

  let coeffs = match extract_poly_coeffs(&expanded, &var) {
    Some(c) => c,
    None => return Ok(expanded),
  };

  // Factor out GCD of coefficients (content)
  let content = coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if content == 0 {
    return Ok(Expr::Integer(0));
  }

  let pp: Vec<i128> = coeffs.iter().map(|c| c / content).collect();

  // Make leading coefficient positive
  let (sign, pp) = if pp.last().map(|&c| c < 0).unwrap_or(false) {
    (-1i128, pp.iter().map(|c| -c).collect::<Vec<_>>())
  } else {
    (1, pp)
  };
  let overall = content * sign;

  // Use full factoring, then group factors with the same multiplicity
  let factors = factor_integer_poly(&pp, &var);

  if factors.is_empty() {
    // No factorization found - return as-is
    if overall == 1 {
      return Ok(expanded);
    }
    let poly_expr = int_coeffs_to_canonical_expr(&pp, &var);
    return Ok(build_product(vec![Expr::Integer(overall), poly_expr]));
  }

  // Group identical factors by string representation
  let mut grouped: Vec<(Expr, i128)> = Vec::new();
  for f in &factors {
    let key = expr_to_string(f);
    if let Some(entry) =
      grouped.iter_mut().find(|(e, _)| expr_to_string(e) == key)
    {
      entry.1 += 1;
    } else {
      grouped.push((f.clone(), 1));
    }
  }

  // Group by multiplicity: multiply factors with the same multiplicity together
  let mut by_mult: std::collections::BTreeMap<i128, Vec<Expr>> =
    std::collections::BTreeMap::new();
  for (factor, mult) in &grouped {
    by_mult.entry(*mult).or_default().push(factor.clone());
  }

  let mut result_factors: Vec<Expr> = Vec::new();

  if overall != 1 {
    result_factors.push(Expr::Integer(overall));
  }

  // Output in descending multiplicity order
  for (&mult, group_factors) in by_mult.iter().rev() {
    // Combine factors with same multiplicity into one product
    let combined = if group_factors.len() == 1 {
      group_factors[0].clone()
    } else {
      // Multiply them together and evaluate to canonical form
      let product = build_product(group_factors.clone());

      expand_and_combine(&product)
    };

    if mult == 1 {
      result_factors.push(combined);
    } else {
      result_factors.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(combined),
        right: Box::new(Expr::Integer(mult)),
      });
    }
  }

  if result_factors.is_empty() {
    return Ok(Expr::Integer(overall));
  }

  if result_factors.len() == 1 {
    return Ok(result_factors.remove(0));
  }

  Ok(build_product(result_factors))
}

/// FactorTermsList[poly] - returns {content, primitive_part}
pub fn factor_terms_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FactorTermsList expects 1 or 2 arguments".into(),
    ));
  }

  let expanded = expand_and_combine(&args[0]);

  // Handle zero
  if matches!(&expanded, Expr::Integer(0)) {
    return Ok(Expr::List(vec![Expr::Integer(1), Expr::Integer(0)]));
  }

  // Handle pure integer
  if let Expr::Integer(n) = &expanded {
    return Ok(Expr::List(vec![Expr::Integer(*n), Expr::Integer(1)]));
  }

  let var = if args.len() == 2 {
    // Explicit variable given
    match &args[1] {
      Expr::Identifier(name) => Some(name.clone()),
      _ => None,
    }
  } else {
    find_single_variable(&expanded)
  };

  let var = match var {
    Some(v) => v,
    None => {
      return Ok(Expr::List(vec![Expr::Integer(1), expanded]));
    }
  };

  let coeffs = match extract_poly_coeffs(&expanded, &var) {
    Some(c) => c,
    None => {
      return Ok(Expr::List(vec![Expr::Integer(1), expanded]));
    }
  };

  // Compute content (GCD of coefficients)
  let content = coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if content == 0 {
    return Ok(Expr::List(vec![Expr::Integer(1), Expr::Integer(0)]));
  }

  // Include sign of leading (highest degree) coefficient in content
  let sign = if coeffs
    .iter()
    .rev()
    .find(|&&c| c != 0)
    .map(|&c| c < 0)
    .unwrap_or(false)
  {
    -1i128
  } else {
    1
  };
  let signed_content = content * sign;

  let pp: Vec<i128> = coeffs.iter().map(|c| c / signed_content).collect();
  let pp_expr = int_coeffs_to_canonical_expr(&pp, &var);

  if args.len() == 2 {
    // Two-arg form: {numerical_content, 1, primitive_part}
    return Ok(Expr::List(vec![
      Expr::Integer(signed_content),
      Expr::Integer(1),
      pp_expr,
    ]));
  }

  Ok(Expr::List(vec![Expr::Integer(signed_content), pp_expr]))
}

/// Convert integer coefficient array to canonical Expr representation
fn int_coeffs_to_canonical_expr(coeffs: &[i128], var: &str) -> Expr {
  // Build a polynomial Expr from coefficients and evaluate to canonical form
  let var_expr = Expr::Identifier(var.to_string());
  let mut terms: Vec<Expr> = Vec::new();

  for (i, &c) in coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let term = if i == 0 {
      Expr::Integer(c)
    } else {
      let x_pow = if i == 1 {
        var_expr.clone()
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(var_expr.clone()),
          right: Box::new(Expr::Integer(i as i128)),
        }
      };
      if c == 1 {
        x_pow
      } else if c == -1 {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(x_pow),
        }
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(c)),
          right: Box::new(x_pow),
        }
      }
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return Expr::Integer(0);
  }
  if terms.len() == 1 {
    return terms.remove(0);
  }

  // Combine into Plus
  let mut result = terms.remove(0);
  for t in terms {
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(t),
    };
  }

  // Evaluate to canonical form
  crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result)
}
