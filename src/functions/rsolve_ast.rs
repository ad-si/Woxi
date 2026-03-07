use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};

/// RSolve[{recurrence, initial_conditions...}, a, n]
/// Solve constant-coefficient linear recurrence relations.
pub fn rsolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated(args));
  }

  // Extract the function name (e.g. "a")
  let func_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated(args)),
  };

  // Extract the variable name (e.g. "n")
  let var_name = match &args[2] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated(args)),
  };

  // Extract equations from the first argument (must be a list)
  let equations = match &args[0] {
    Expr::List(eqs) => eqs.clone(),
    _ => return Ok(unevaluated(args)),
  };

  // Separate the recurrence relation from initial conditions
  let mut recurrence = None;
  let mut initial_conditions: Vec<(i128, Expr)> = Vec::new(); // (index, value)

  for eq in &equations {
    // Each equation should be lhs == rhs
    let (lhs, rhs) = match extract_equation(eq) {
      Some(pair) => pair,
      None => return Ok(unevaluated(args)),
    };

    // Check if this is an initial condition: a[integer] == value
    if let Some((name, idx)) = extract_func_at_integer(&lhs)
      && name == func_name
    {
      initial_conditions.push((idx, rhs));
      continue;
    }
    if let Some((name, idx)) = extract_func_at_integer(&rhs)
      && name == func_name
    {
      initial_conditions.push((idx, lhs));
      continue;
    }

    // Otherwise this is the recurrence relation
    if recurrence.is_some() {
      return Ok(unevaluated(args)); // Multiple recurrences not supported
    }
    recurrence = Some((lhs, rhs));
  }

  let (rec_lhs, rec_rhs) = match recurrence {
    Some(r) => r,
    None => return Ok(unevaluated(args)),
  };

  // Try to solve as constant-coefficient linear recurrence
  // Extract: a[n+k] = c_{k-1}*a[n+k-1] + ... + c_0*a[n]
  if let Some(solution) = solve_const_coeff_linear(
    &rec_lhs,
    &rec_rhs,
    &func_name,
    &var_name,
    &initial_conditions,
  ) {
    // Return {{a -> Function[{n}, solution]}}
    let func_expr = Expr::FunctionCall {
      name: "Function".to_string(),
      args: vec![Expr::List(vec![Expr::Identifier(var_name)]), solution],
    };
    let rule = Expr::Rule {
      pattern: Box::new(Expr::Identifier(func_name)),
      replacement: Box::new(func_expr),
    };
    return Ok(Expr::List(vec![Expr::List(vec![rule])]));
  }

  Ok(unevaluated(args))
}

fn unevaluated(args: &[Expr]) -> Expr {
  Expr::FunctionCall {
    name: "RSolve".to_string(),
    args: args.to_vec(),
  }
}

/// Extract lhs and rhs from an equation (Comparison or FunctionCall Equal)
fn extract_equation(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && operators[0] == crate::syntax::ComparisonOp::Equal =>
    {
      Some((operands[0].clone(), operands[1].clone()))
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      Some((args[0].clone(), args[1].clone()))
    }
    _ => None,
  }
}

/// Check if expr is f[integer] and return (f_name, integer_value)
fn extract_func_at_integer(expr: &Expr) -> Option<(String, i128)> {
  match expr {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      if let Expr::Integer(n) = &args[0] {
        Some((name.clone(), *n))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Try to solve a constant-coefficient linear recurrence.
/// Handles: a[n+k] == c_{k-1}*a[n+k-1] + ... + c_0*a[n] + constant
fn solve_const_coeff_linear(
  lhs: &Expr,
  rhs: &Expr,
  func_name: &str,
  var_name: &str,
  initial_conditions: &[(i128, Expr)],
) -> Option<Expr> {
  // Collect all terms involving func_name[var_name + offset]
  // Move everything to lhs - rhs = 0
  let mut terms: Vec<(i128, i128)> = Vec::new(); // (offset, coefficient)

  // Extract terms from lhs (positive) and rhs (negative when moved)
  collect_recurrence_terms(lhs, func_name, var_name, 1, &mut terms);
  collect_recurrence_terms(rhs, func_name, var_name, -1, &mut terms);

  if terms.is_empty() {
    return None;
  }

  // Combine terms with same offset
  let mut combined: std::collections::HashMap<i128, i128> =
    std::collections::HashMap::new();
  for (offset, coeff) in &terms {
    *combined.entry(*offset).or_insert(0) += coeff;
  }

  // Find the order (max offset - min offset)
  let min_offset = *combined.keys().min()?;
  let max_offset = *combined.keys().max()?;
  let order = (max_offset - min_offset) as usize;

  if order == 0 || order > 10 {
    return None;
  }

  // Build characteristic polynomial coefficients
  // If recurrence is c_k*a[n+k] + c_{k-1}*a[n+k-1] + ... + c_0*a[n] = 0
  // Characteristic equation: c_k*r^k + c_{k-1}*r^{k-1} + ... + c_0 = 0
  let mut char_coeffs = vec![0i128; order + 1];
  for (&offset, &coeff) in &combined {
    let idx = (offset - min_offset) as usize;
    char_coeffs[idx] = coeff;
  }

  // Find roots of the characteristic polynomial
  let roots = find_characteristic_roots(&char_coeffs)?;

  if roots.len() != order {
    return None; // Need all roots for general solution
  }

  // Check initial conditions match order
  if initial_conditions.len() != order {
    return None;
  }

  // Build and solve the system for constants
  // General solution: a[n] = c1*r1^n + c2*r2^n + ...
  // Apply initial conditions to find c1, c2, ...
  let constants = solve_initial_conditions(&roots, initial_conditions)?;

  // Build the solution expression: c1*r1^n + c2*r2^n + ...
  build_solution(&constants, &roots, var_name)
}

/// Collect terms of the form coeff * func[var + offset] from an expression
fn collect_recurrence_terms(
  expr: &Expr,
  func_name: &str,
  var_name: &str,
  sign: i128,
  terms: &mut Vec<(i128, i128)>,
) {
  match expr {
    // Direct: a[n + k] or a[n]
    Expr::FunctionCall { name, args }
      if name == func_name && args.len() == 1 =>
    {
      if let Some(offset) = extract_var_offset(&args[0], var_name) {
        terms.push((offset, sign));
      }
    }
    // c * a[n + k]
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if let Expr::Integer(c) = left.as_ref()
        && let Expr::FunctionCall { name, args } = right.as_ref()
        && name == func_name
        && args.len() == 1
        && let Some(offset) = extract_var_offset(&args[0], var_name)
      {
        terms.push((offset, sign * c));
        return;
      }
      if let Expr::Integer(c) = right.as_ref()
        && let Expr::FunctionCall { name, args } = left.as_ref()
        && name == func_name
        && args.len() == 1
        && let Some(offset) = extract_var_offset(&args[0], var_name)
      {
        terms.push((offset, sign * c));
      }
    }
    // Plus: recurse into sub-terms
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for arg in args {
        collect_recurrence_terms(arg, func_name, var_name, sign, terms);
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      collect_recurrence_terms(left, func_name, var_name, sign, terms);
      collect_recurrence_terms(right, func_name, var_name, sign, terms);
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      collect_recurrence_terms(left, func_name, var_name, sign, terms);
      collect_recurrence_terms(right, func_name, var_name, -sign, terms);
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      collect_recurrence_terms(operand, func_name, var_name, -sign, terms);
    }
    _ => {}
  }
}

/// Extract offset from expressions like `n`, `n + 2`, `2 + n`
fn extract_var_offset(expr: &Expr, var_name: &str) -> Option<i128> {
  match expr {
    Expr::Identifier(name) if name == var_name => Some(0),
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      // n + k or k + n
      if let Expr::Identifier(name) = left.as_ref()
        && name == var_name
        && let Expr::Integer(k) = right.as_ref()
      {
        return Some(*k);
      }
      if let Expr::Identifier(name) = right.as_ref()
        && name == var_name
        && let Expr::Integer(k) = left.as_ref()
      {
        return Some(*k);
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() == 2 => {
      // Plus[n, k] or Plus[k, n]
      for i in 0..2 {
        if let Expr::Identifier(name) = &args[i]
          && name == var_name
          && let Expr::Integer(k) = &args[1 - i]
        {
          return Some(*k);
        }
      }
      None
    }
    _ => None,
  }
}

/// Find integer/rational roots of the characteristic polynomial
fn find_characteristic_roots(coeffs: &[i128]) -> Option<Vec<(i128, i128)>> {
  // Returns roots as (numerator, denominator) pairs
  let mut remaining = coeffs.to_vec();
  let mut roots = Vec::new();

  while remaining.len() > 1 {
    // Try integer roots using rational root theorem
    let last_nonzero = remaining.iter().rposition(|&c| c != 0)?;
    let leading = remaining[last_nonzero];
    let constant = remaining[0];

    if constant == 0 {
      // x = 0 is a root
      roots.push((0i128, 1i128));
      // Divide by x (shift coefficients)
      remaining = remaining[1..].to_vec();
      continue;
    }

    let lead_divs = divisors(leading.abs());
    let const_divs = divisors(constant.abs());

    let mut found = false;
    for &p in &const_divs {
      for &q in &lead_divs {
        for &sign in &[1i128, -1] {
          let num = sign * p;
          let den = q;
          // Evaluate polynomial at num/den: multiply through by den^n
          let mut val = 0i128;
          let _den_power = 1i128;
          for (i, &c) in remaining.iter().enumerate() {
            let num_power = num.checked_pow(i as u32)?;
            // We need: c * num^i * den^(degree-i)
            let degree = remaining.len() - 1;
            let den_p = den.checked_pow((degree - i) as u32)?;
            val =
              val.checked_add(c.checked_mul(num_power)?.checked_mul(den_p)?)?;
          }
          if val == 0 {
            roots.push((num, den));
            // Synthetic division by (den*x - num)
            remaining = synthetic_divide(&remaining, num, den)?;
            found = true;
            break;
          }
        }
        if found {
          break;
        }
      }
      if found {
        break;
      }
    }

    if !found {
      return None; // Can't find all roots
    }
  }

  Some(roots)
}

/// Synthetic division of polynomial by (den*x - num)
fn synthetic_divide(
  coeffs: &[i128],
  num: i128,
  den: i128,
) -> Option<Vec<i128>> {
  if coeffs.len() <= 1 {
    return Some(vec![]);
  }
  let n = coeffs.len() - 1;
  // Division of sum(c_i * x^i) by (den*x - num)
  // Equivalently, divide by (x - num/den) then divide all coefficients by den
  let mut result = vec![0i128; n];
  result[n - 1] = coeffs[n];
  for i in (0..n - 1).rev() {
    // result[i] = coeffs[i+1] + result[i+1] * num / den
    let prod = result[i + 1].checked_mul(num)?;
    if prod % den != 0 {
      return None; // Not exact division
    }
    result[i] = coeffs[i + 1] + prod / den;
  }
  // Verify: coeffs[0] + result[0] * num / den should be 0
  let check = result[0].checked_mul(num)?;
  if coeffs[0] * den + check != 0 {
    return None;
  }
  Some(result)
}

fn divisors(n: i128) -> Vec<i128> {
  let mut result = Vec::new();
  let mut i = 1i128;
  while i * i <= n {
    if n % i == 0 {
      result.push(i);
      if i != n / i {
        result.push(n / i);
      }
    }
    i += 1;
  }
  result
}

/// Solve for constants in the general solution given initial conditions.
/// General solution: a[n] = c1*r1^n + c2*r2^n + ...
/// Returns coefficients as (numerator, denominator) pairs.
fn solve_initial_conditions(
  roots: &[(i128, i128)],
  ics: &[(i128, Expr)],
) -> Option<Vec<(i128, i128)>> {
  let n = roots.len();
  if ics.len() != n {
    return None;
  }

  // Build matrix: A[i][j] = (roots[j].num / roots[j].den)^ics[i].index
  // Working with rationals: (num/den)^k = num^k / den^k
  // We need to solve A * c = b where b[i] = ics[i].value

  // For simplicity, work with integer matrix by multiplying through by lcm of denominators
  // First, build the rational matrix
  let mut matrix: Vec<Vec<(i128, i128)>> = Vec::new(); // (num, den)
  let mut rhs: Vec<(i128, i128)> = Vec::new();

  for (idx, val) in ics {
    let mut row = Vec::new();
    for &(rnum, rden) in roots {
      // (rnum/rden)^idx
      let k = *idx as u32;
      let num_k = rnum.checked_pow(k)?;
      let den_k = rden.checked_pow(k)?;
      row.push((num_k, den_k));
    }
    matrix.push(row);
    // RHS value must be an integer
    match val {
      Expr::Integer(v) => rhs.push((*v, 1)),
      _ => return None,
    }
  }

  // Gaussian elimination with rational arithmetic
  // Augmented matrix: [A | b]
  let mut aug: Vec<Vec<(i128, i128)>> = matrix
    .iter()
    .zip(rhs.iter())
    .map(|(row, b)| {
      let mut r = row.clone();
      r.push(*b);
      r
    })
    .collect();

  for col in 0..n {
    // Find pivot
    let pivot_row = (col..n).find(|&r| aug[r][col].0 != 0)?;
    aug.swap(col, pivot_row);

    let pivot = aug[col][col];
    for row in (col + 1)..n {
      let factor = aug[row][col];
      if factor.0 == 0 {
        continue;
      }
      for j in col..=n {
        // aug[row][j] -= factor/pivot * aug[col][j]
        // = (aug[row][j] * pivot - factor * aug[col][j]) / pivot
        let (an, ad) = aug[row][j];
        let (bn, bd) = aug[col][j];
        let (fn_, fd) = factor;
        let (pn, pd) = pivot;
        // new = an/ad - (fn/fd)/(pn/pd) * bn/bd
        //     = an/ad - (fn*pd)/(fd*pn) * bn/bd
        //     = an/ad - (fn*pd*bn)/(fd*pn*bd)
        let lhs_n = an * fd * pn * bd;
        let lhs_d = ad * fd * pn * bd;
        let rhs_n = fn_ * pd * bn * ad;
        let new_n = lhs_n - rhs_n;
        let new_d = lhs_d;
        let g = gcd_abs(new_n.abs(), new_d.abs());
        aug[row][j] = if g == 0 {
          (0, 1)
        } else {
          let mut nn = new_n / g;
          let mut nd = new_d / g;
          if nd < 0 {
            nn = -nn;
            nd = -nd;
          }
          (nn, nd)
        };
      }
    }
  }

  // Back substitution
  let mut solution = vec![(0i128, 1i128); n];
  for i in (0..n).rev() {
    let (mut sn, mut sd) = aug[i][n]; // RHS
    for j in (i + 1)..n {
      // sn/sd -= aug[i][j] * solution[j]
      let (an, ad) = aug[i][j];
      let (cn, cd) = solution[j];
      let sub_n = an * cn;
      let sub_d = ad * cd;
      // sn/sd - sub_n/sub_d
      sn = sn * sub_d - sub_n * sd;
      sd *= sub_d;
      let g = gcd_abs(sn.abs(), sd.abs());
      if g != 0 {
        sn /= g;
        sd /= g;
      }
      if sd < 0 {
        sn = -sn;
        sd = -sd;
      }
    }
    // solution[i] = (sn/sd) / aug[i][i]
    let (pn, pd) = aug[i][i];
    sn *= pd;
    sd *= pn;
    let g = gcd_abs(sn.abs(), sd.abs());
    if g != 0 {
      sn /= g;
      sd /= g;
    }
    if sd < 0 {
      sn = -sn;
      sd = -sd;
    }
    solution[i] = (sn, sd);
  }

  Some(solution)
}

fn gcd_abs(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a, b);
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Build the solution expression: (c1_num*r1^n + c2_num*r2^n + ...) / common_denom
fn build_solution(
  constants: &[(i128, i128)],
  roots: &[(i128, i128)],
  var_name: &str,
) -> Option<Expr> {
  // Find common denominator for all constants
  let mut common_denom = 1i128;
  for &(_, cd) in constants {
    common_denom = lcm(common_denom, cd);
  }

  // Build terms with integer numerators: (cn * common_denom/cd) * r^n
  let mut terms = Vec::new();

  for (i, &(cn, cd)) in constants.iter().enumerate() {
    let num = cn * (common_denom / cd);
    if num == 0 {
      continue;
    }
    let (rn, rd) = roots[i];

    // Build r^n (with parentheses for negative bases)
    let power_expr = if rn == 1 && rd == 1 {
      None // 1^n = 1
    } else {
      let root_expr = if rd == 1 {
        Expr::Integer(rn)
      } else {
        crate::functions::math_ast::make_rational_pub(rn, rd)
      };
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(root_expr),
        right: Box::new(Expr::Identifier(var_name.to_string())),
      })
    };

    // Build num * r^n (or just num if r=1)
    let term = match power_expr {
      Some(pe) => {
        let abs_num = num.abs();
        let base_term = if abs_num == 1 {
          pe
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(abs_num)),
            right: Box::new(pe),
          }
        };
        if num < 0 {
          Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand: Box::new(base_term),
          }
        } else {
          base_term
        }
      }
      None => Expr::Integer(num), // root is 1, so r^n = 1
    };

    terms.push(term);
  }

  if terms.is_empty() {
    return Some(Expr::Integer(0));
  }

  // Build the numerator sum
  let numerator = if terms.len() == 1 {
    terms.remove(0)
  } else {
    crate::functions::math_ast::plus_ast(&terms).ok()?
  };

  // Divide by common denominator if > 1
  if common_denom == 1 {
    Some(numerator)
  } else {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(Expr::Integer(common_denom)),
    })
  }
}

fn lcm(a: i128, b: i128) -> i128 {
  if a == 0 || b == 0 {
    0
  } else {
    (a / gcd_abs(a.abs(), b.abs())) * b.abs()
  }
}

/// RecurrenceTable[{recurrence, initial_conditions...}, a, {n, nmin, nmax}]
/// Iteratively evaluate a recurrence relation.
pub fn recurrence_table_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(recurrence_table_unevaluated(args));
  }

  // Extract function name (e.g. "a")
  let func_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(recurrence_table_unevaluated(args)),
  };

  // Extract {n, nmin, nmax} range
  let (var_name, nmin, nmax) = match &args[2] {
    Expr::List(items) if items.len() == 3 => {
      let var = match &items[0] {
        Expr::Identifier(s) => s.clone(),
        _ => return Ok(recurrence_table_unevaluated(args)),
      };
      let nmin = match &items[1] {
        Expr::Integer(n) => *n,
        _ => return Ok(recurrence_table_unevaluated(args)),
      };
      let nmax = match &items[2] {
        Expr::Integer(n) => *n,
        _ => return Ok(recurrence_table_unevaluated(args)),
      };
      (var, nmin, nmax)
    }
    _ => return Ok(recurrence_table_unevaluated(args)),
  };

  // Extract equations from first arg
  let equations = match &args[0] {
    Expr::List(eqs) => eqs.clone(),
    _ => return Ok(recurrence_table_unevaluated(args)),
  };

  // Separate initial conditions from the recurrence
  let mut recurrence_eq: Option<(Expr, Expr)> = None;
  let mut initial_conditions: std::collections::HashMap<i128, Expr> =
    std::collections::HashMap::new();

  for eq in &equations {
    let (lhs, rhs) = match extract_equation(eq) {
      Some(pair) => pair,
      None => return Ok(recurrence_table_unevaluated(args)),
    };

    // Check if this is an initial condition: a[integer] == value
    if let Some((name, idx)) = extract_func_at_integer(&lhs)
      && name == func_name
    {
      let val = crate::evaluator::evaluate_expr_to_expr(&rhs)?;
      initial_conditions.insert(idx, val);
      continue;
    }
    if let Some((name, idx)) = extract_func_at_integer(&rhs)
      && name == func_name
    {
      let val = crate::evaluator::evaluate_expr_to_expr(&lhs)?;
      initial_conditions.insert(idx, val);
      continue;
    }

    // This is the recurrence relation
    if recurrence_eq.is_some() {
      return Ok(recurrence_table_unevaluated(args));
    }
    recurrence_eq = Some((lhs, rhs));
  }

  let (rec_lhs, rec_rhs) = match recurrence_eq {
    Some(r) => r,
    None => return Ok(recurrence_table_unevaluated(args)),
  };

  // Determine which side has the "highest" a[n+k] to solve for
  // Normalize: solve for a[n+k] on the LHS, expression on the RHS
  // Find the highest-offset term on LHS that is just a[n+offset]
  let (target_offset, solve_expr) = if let Some(offset) =
    extract_single_func_offset(&rec_lhs, &func_name, &var_name)
  {
    // LHS is a[n+offset], RHS is the expression
    (offset, rec_rhs)
  } else if let Some(offset) =
    extract_single_func_offset(&rec_rhs, &func_name, &var_name)
  {
    // RHS is a[n+offset], LHS is the expression
    (offset, rec_lhs)
  } else {
    return Ok(recurrence_table_unevaluated(args));
  };

  // Now iterate: for each n from nmin to nmax, compute a[n]
  let mut results = Vec::new();
  let mut values = initial_conditions;

  for n in nmin..=nmax {
    if values.contains_key(&n) {
      results.push(values[&n].clone());
      continue;
    }

    // We need to compute a[n]. The recurrence says:
    // a[var + target_offset] = solve_expr
    // So when var = n - target_offset, we get a[n] = solve_expr(var = n - target_offset)
    let var_val = n - target_offset;

    // Substitute var_name = var_val in solve_expr
    let substituted = crate::syntax::substitute_variable(
      &solve_expr,
      &var_name,
      &Expr::Integer(var_val),
    );

    // Now substitute all a[k] references with known values
    let resolved = substitute_func_values(&substituted, &func_name, &values);
    let val = crate::evaluator::evaluate_expr_to_expr(&resolved)?;
    values.insert(n, val.clone());
    results.push(val);
  }

  Ok(Expr::List(results))
}

/// Extract the offset from an expression like a[n+k] or a[n] or a[n-1]
/// Returns Some(k) if the expression is func_name[var_name + k]
fn extract_single_func_offset(
  expr: &Expr,
  func_name: &str,
  var_name: &str,
) -> Option<i128> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == func_name && args.len() == 1 =>
    {
      extract_var_offset(&args[0], var_name)
    }
    _ => None,
  }
}

/// Substitute all occurrences of func_name[integer] with known values
fn substitute_func_values(
  expr: &Expr,
  func_name: &str,
  values: &std::collections::HashMap<i128, Expr>,
) -> Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if name == func_name && args.len() == 1 =>
    {
      // Try to evaluate the argument to an integer
      let evaled_arg = crate::evaluator::evaluate_expr_to_expr(&args[0])
        .unwrap_or(args[0].clone());
      if let Expr::Integer(n) = &evaled_arg
        && let Some(val) = values.get(n)
      {
        return val.clone();
      }
      // Recurse into args
      Expr::FunctionCall {
        name: name.clone(),
        args: args
          .iter()
          .map(|a| substitute_func_values(a, func_name, values))
          .collect(),
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| substitute_func_values(a, func_name, values))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| substitute_func_values(a, func_name, values))
        .collect(),
    ),
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_func_values(left, func_name, values)),
      right: Box::new(substitute_func_values(right, func_name, values)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_func_values(operand, func_name, values)),
    },
    _ => expr.clone(),
  }
}

fn recurrence_table_unevaluated(args: &[Expr]) -> Expr {
  Expr::FunctionCall {
    name: "RecurrenceTable".to_string(),
    args: args.to_vec(),
  }
}
