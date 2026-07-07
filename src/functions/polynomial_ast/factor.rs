#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::functions::calculus_ast::simplify;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

// ─── Factor ─────────────────────────────────────────────────────────

/// Factor[expr] - Factor a polynomial expression
/// Threads over List.
/// For a 2-argument call whose option is not `Modulus`, emit `optx` for
/// genuinely unknown options (valid-but-unimplemented ones like
/// GaussianIntegers/Extension/Trig stay silently unevaluated) and return
/// the unevaluated call.
fn unknown_option_fallback(head: &str, args: &[Expr]) -> Expr {
  let known = |s: &str| {
    matches!(s, "Modulus" | "GaussianIntegers" | "Extension" | "Trig")
  };
  let option_name = match &args[1] {
    Expr::Rule { pattern, .. } => match pattern.as_ref() {
      Expr::Identifier(s) => Some(s.clone()),
      _ => None,
    },
    Expr::FunctionCall { name, args: ra }
      if (name == "Rule" || name == "RuleDelayed") && ra.len() == 2 =>
    {
      match &ra[0] {
        Expr::Identifier(s) => Some(s.clone()),
        _ => None,
      }
    }
    _ => None,
  };
  let call = Expr::FunctionCall {
    name: head.to_string(),
    args: args.to_vec().into(),
  };
  if let Some(opt) = option_name
    && !known(&opt)
  {
    crate::emit_message(&format!(
      "{head}::optx: Unknown option {opt} in {}.",
      expr_to_string(&call)
    ));
  }
  call
}

pub fn factor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  Ok(reorder_factored_product(factor_ast_impl(args)?))
}

/// Put a factored product's monomial factors in canonical position
/// relative to its sum factors (`x*(-1 + x + x^2)`, not
/// `(-1 + x + x^2)*x`), using only the decoded monomial-vs-sum rule so
/// the relative order of sums — which factor construction already gets
/// right (e.g. Factor[x^100 - 1]) — is untouched. A full re-evaluation
/// would apply the evaluator's divergent sum-vs-sum comparison.
fn reorder_factored_product(result: Expr) -> Expr {
  fn collect(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        collect(left, out);
        collect(right, out);
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          collect(a, out);
        }
      }
      other => out.push(other.clone()),
    }
  }
  let is_times = matches!(
    &result,
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    }
  ) || matches!(&result, Expr::FunctionCall { name, .. } if name == "Times");
  if !is_times {
    return result;
  }
  let mut factors: Vec<Expr> = Vec::new();
  collect(&result, &mut factors);

  use crate::functions::math_ast::order_monomial_vs_sum;
  use std::cmp::Ordering;
  let mut swapped_any = false;
  loop {
    let mut swapped = false;
    for i in 0..factors.len().saturating_sub(1) {
      let (a, b) = (&factors[i], &factors[i + 1]);
      let should_swap =
        // sum before monomial, but the monomial sorts first
        order_monomial_vs_sum(b, a) == Some(Ordering::Less)
        // monomial before sum, but the sum sorts first
        || order_monomial_vs_sum(a, b) == Some(Ordering::Greater);
      if should_swap {
        factors.swap(i, i + 1);
        swapped = true;
        swapped_any = true;
      }
    }
    if !swapped {
      break;
    }
  }
  if !swapped_any {
    return result;
  }
  factors
    .into_iter()
    .reduce(|acc, f| Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(acc),
      right: Box::new(f),
    })
    .expect("non-empty factor list")
}

fn factor_ast_impl(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Factor[poly, Modulus -> p] factors over GF(p); inputs the modular
  // engine cannot handle (multivariate, symbolic) stay unevaluated.
  if args.len() == 2 {
    if let Some(p) = extract_modulus_option(&args[1]) {
      return match super::factor_modulus(&args[0], p)? {
        Some(e) => Ok(e),
        None => Ok(Expr::FunctionCall {
          name: "Factor".to_string(),
          args: args.to_vec().into(),
        }),
      };
    }
    return Ok(unknown_option_fallback("Factor", args));
  }
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
    return Ok(Expr::List(results?.into()));
  }

  // Thread over Comparison / Equal / Inequality: factor each operand.
  if let Expr::Comparison {
    operands,
    operators,
  } = &args[0]
  {
    let factored: Result<Vec<Expr>, InterpreterError> =
      operands.iter().map(|o| factor_ast(&[o.clone()])).collect();
    return Ok(Expr::Comparison {
      operands: factored?,
      operators: operators.clone(),
    });
  }

  // Handle rational expressions: factor numerator and denominator separately.
  // This matches Wolfram's behavior where Factor[p/q] = Factor[p]/Factor[q].
  {
    let (num, den) = super::together::extract_num_den(&args[0]);
    if !matches!(&den, Expr::Integer(1)) {
      let factored_num = factor_ast(&[num])?;
      let factored_den = factor_ast(&[den])?;
      return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(factored_num),
        right: Box::new(factored_den),
      });
    }
  }

  // Combine fractions in a Plus before factoring. Wolfram's `Factor` puts
  // sums of rational terms over a common denominator first (so e.g.
  // `Factor[1/x + 1/y] = (x + y)/(x*y)`); without this, we'd return the
  // unmodified Plus. Detect by checking whether any Plus term has a
  // non-trivial denominator, and only run Together if it actually changed
  // the structure (avoids re-factoring loops on plain polynomial sums).
  let plus_terms: Option<Vec<&Expr>> = match &args[0] {
    Expr::FunctionCall { name, args: pa } if name == "Plus" => {
      Some(pa.iter().collect())
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => Some(vec![left.as_ref(), right.as_ref()]),
    _ => None,
  };
  if let Some(terms) = plus_terms
    && terms.iter().any(|t| {
      let (_, d) = super::together::extract_num_den(t);
      !matches!(&d, Expr::Integer(1))
    })
  {
    let combined = super::together::together_expr(&args[0]);
    let (num, den) = super::together::extract_num_den(&combined);
    if !matches!(&den, Expr::Integer(1)) {
      let factored_num = factor_ast(&[num])?;
      let factored_den = factor_ast(&[den])?;
      return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(factored_num),
        right: Box::new(factored_den),
      });
    }
  }

  // First expand to canonical form
  let expanded = expand_and_combine(&args[0]);

  // If expansion produced Laurent terms — negative powers of a variable, e.g.
  // `a/b + a/c` from `a (1/b + 1/c)` — the polynomial factorers below assume a
  // true polynomial and silently drop the `1/(b c)` (returning `a (b + c)`).
  // Combine over a common denominator first and factor numerator/denominator
  // separately, matching Wolfram's `Factor[p/q] = Factor[p]/Factor[q]`.
  if expr_has_negative_var_power(&expanded) {
    let combined = super::together::together_expr(&expanded);
    let (num, den) = super::together::extract_num_den(&combined);
    if !matches!(&den, Expr::Integer(1)) {
      let factored_num = factor_ast(&[num])?;
      let factored_den = factor_ast(&[den])?;
      return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(factored_num),
        right: Box::new(factored_den),
      });
    }
  }

  // Collect all variables
  let mut vars = std::collections::HashSet::new();
  collect_variables(&expanded, &mut vars);
  if vars.is_empty() {
    return Ok(expanded); // constant — return as is
  }
  // A single monomial term (constant × product of variable powers) has no
  // nontrivial polynomial factorization — Factor returns it unchanged. Handle
  // this before the multivariate path, which would otherwise drop the sign of
  // the coefficient when pulling out its content (e.g. Factor[-2 a x] gave
  // `2 a x`). The imaginary unit parses as Identifier("I"), so `-2 I x` is a
  // two-"variable" monomial that hit exactly this bug.
  if collect_additive_terms(&expanded).len() == 1 {
    return crate::evaluator::evaluate_expr_to_expr(&expanded);
  }
  if vars.len() > 1 {
    // Multivariate polynomial — try Kronecker substitution
    return factor_multivariate(&expanded, vars);
  }
  let var = vars.into_iter().next().unwrap();

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

  // Build the polynomial factor expressions, grouping identical factors into
  // powers: (1+x)*(1+x) → (1+x)^2.
  let poly_factor_exprs: Vec<Expr> = if factors.is_empty() {
    vec![coeffs_to_expr(&reduced_coeffs, &var)]
  } else {
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
    grouped
      .into_iter()
      .map(|(factor, count)| {
        if count == 1 {
          factor
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(factor),
            right: Box::new(Expr::Integer(count)),
          }
        }
      })
      .collect()
  };

  // The polynomial part is genuinely factored only when it splits into more
  // than one factor (multiple distinct factors or a repeated power).
  let poly_factored = poly_factor_exprs.len() > 1
    || matches!(
      poly_factor_exprs.first(),
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        ..
      })
    );

  // When only a unit sign was extracted (overall == ±1) and the polynomial
  // didn't factor, wolframscript returns the expanded polynomial unchanged.
  if !poly_factored && (overall == 1 || overall == -1) {
    return Ok(expanded);
  }

  let product = build_product(poly_factor_exprs.clone());

  // overall == -1: wolframscript negates the whole factored product,
  // e.g. -((2 + x)*(3 + x)) and -(1 + x)^2.
  if overall == -1 {
    return Ok(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(product),
    });
  }

  // overall == 1: just the factored product.
  if overall == 1 {
    return Ok(product);
  }

  // Any other integer content (e.g. 2, -2): flat content * factor1 * factor2.
  let mut result_factors: Vec<Expr> = vec![Expr::Integer(overall)];
  result_factors.extend(poly_factor_exprs);
  Ok(build_product(result_factors))
}

/// True when `expr` contains a `Power[base, n]` with a negative exponent whose
/// base mentions a variable (i.e. a genuine `1/var…` Laurent term, not a pure
/// numeric reciprocal like `1/2`). Used to route non-polynomial inputs through
/// Together before the polynomial factorers run.
fn expr_has_negative_var_power(expr: &Expr) -> bool {
  fn is_negative_exp(exp: &Expr) -> bool {
    matches!(exp, Expr::Integer(n) if *n < 0)
      || matches!(
        exp,
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          ..
        }
      )
  }
  fn base_has_var(base: &Expr) -> bool {
    let mut vars = std::collections::HashSet::new();
    collect_variables(base, &mut vars);
    !vars.is_empty()
  }
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      (is_negative_exp(right) && base_has_var(left))
        || expr_has_negative_var_power(left)
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (is_negative_exp(&args[1]) && base_has_var(&args[0]))
        || expr_has_negative_var_power(&args[0])
    }
    Expr::BinaryOp { left, right, .. } => {
      expr_has_negative_var_power(left) || expr_has_negative_var_power(right)
    }
    Expr::UnaryOp { operand, .. } => expr_has_negative_var_power(operand),
    Expr::FunctionCall { args, .. } => {
      args.iter().any(expr_has_negative_var_power)
    }
    Expr::List(items) => items.iter().any(expr_has_negative_var_power),
    _ => false,
  }
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

  // Add x factor for each leading zero (x^k = x * x * ... * x)
  for _ in 0..leading_zeros {
    factors.push(Expr::Identifier(var.to_string()));
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
        // No integer root. Try a non-integer rational root p/q: factor out the
        // primitive linear (q*x - p) and divide it out, e.g.
        // 6x^2+11x+3 -> (2x+3)(3x+1).
        if let Some((p, q)) = find_rational_root(&remaining) {
          let linear = vec![-p, q]; // -p + q*x  ==  q*x - p
          if let Some((quot, rem)) = poly_div(&remaining, &linear)
            && rem == [0]
          {
            factors.push(linear_to_expr(-p, q, var));
            remaining = quot;
            continue;
          }
        }
        // Can't find more rational roots — try polynomial trial division
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

  // Sort factors by: leading coefficient, then constant term, then degree,
  // then string representation. wolframscript orders non-monic factors by
  // their leading coefficient first (e.g. (2x+3)(3x+1)); for monic factors
  // the leading coefficient ties at 1, so the constant-term ordering below
  // is unchanged.
  factors.sort_by(|a, b| {
    let la = factor_leading_coeff(a, var);
    let lb = factor_leading_coeff(b, var);
    let ca = factor_constant_term(a);
    let cb = factor_constant_term(b);
    la.cmp(&lb)
      .then_with(|| ca.cmp(&cb))
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
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().map(factor_term_count).sum()
    }
    _ => 1,
  }
}

/// Leading coefficient (coefficient of the highest power of `var`) of a factor.
/// Used as the primary sort key so non-monic factors order by leading
/// coefficient like wolframscript; returns 0 for non-polynomial factors.
pub fn factor_leading_coeff(expr: &Expr, var: &str) -> i128 {
  crate::functions::polynomial_ast::simplify::extract_poly_coeffs(expr, var)
    .and_then(|c| c.last().copied())
    .unwrap_or(0)
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

/// q^deg * poly(p/q) = Σ coeffs[i] * p^i * q^(deg-i). This is 0 exactly when
/// p/q is a root. Returns None on overflow.
fn poly_value_scaled(coeffs: &[i128], p: i128, q: i128) -> Option<i128> {
  let deg = coeffs.len().checked_sub(1)?;
  let mut total: i128 = 0;
  for (i, &c) in coeffs.iter().enumerate() {
    let pi = p.checked_pow(i as u32)?;
    let qj = q.checked_pow((deg - i) as u32)?;
    total = total.checked_add(c.checked_mul(pi)?.checked_mul(qj)?)?;
  }
  Some(total)
}

/// Rational-root theorem for a non-integer root p/q (q > 1, gcd(p,q)=1).
/// Integer roots are handled separately by `find_integer_root`.
pub fn find_rational_root(coeffs: &[i128]) -> Option<(i128, i128)> {
  if coeffs.len() < 2 {
    return None;
  }
  let c0 = coeffs[0];
  let lead = coeffs[coeffs.len() - 1];
  if c0 == 0 || lead == 0 {
    return None;
  }
  for &q in &integer_divisors(lead.abs()) {
    if q <= 1 {
      continue; // q == 1 would be an integer root
    }
    for &p in &integer_divisors(c0.abs()) {
      if gcd_i128(p, q) != 1 {
        continue;
      }
      for pn in [p, -p] {
        if poly_value_scaled(coeffs, pn, q) == Some(0) {
          return Some((pn, q));
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
      // Bail out (rather than panic) if an intermediate product overflows;
      // the caller treats `None` as "not exactly divisible over integers".
      let prod = coeff.checked_mul(den[j])?;
      rem[i + j] = rem[i + j].checked_sub(prod)?;
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
    // No cyclotomic factors found — try square-free factorization first
    let sqfree = try_square_free_factor(coeffs, var);
    if !sqfree.is_empty() {
      return sqfree;
    }
    // Zassenhaus (mod-p Berlekamp + Hensel lifting) handles the general
    // square-free case; a definitive single-factor answer means the
    // polynomial is irreducible and Kronecker can be skipped.
    if let Some(zf) = super::zassenhaus::zassenhaus_int_factors(coeffs) {
      if zf.len() >= 2 {
        return zf.iter().map(|f| coeffs_to_expr(f, var)).collect();
      }
      return vec![];
    }
    // Fall back to Kronecker's method for small-degree polynomials
    return try_kronecker_factor(coeffs, var);
  }

  // We found cyclotomic factors; handle the remainder
  if remaining.len() > 2 {
    // Try square-free factorization on the remainder first
    let sub = try_square_free_factor(&remaining, var);
    if !sub.is_empty() {
      return factors
        .iter()
        .map(|f| coeffs_to_expr(f, var))
        .chain(sub)
        .collect();
    }
    // Then Zassenhaus, then Kronecker
    if let Some(zf) = super::zassenhaus::zassenhaus_int_factors(&remaining) {
      if zf.len() >= 2 {
        return factors
          .iter()
          .map(|f| coeffs_to_expr(f, var))
          .chain(zf.iter().map(|f| coeffs_to_expr(f, var)))
          .collect();
      }
      factors.push(remaining);
    } else {
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
    }
  } else if remaining.len() > 1 || (remaining.len() == 1 && remaining[0] != 1) {
    factors.push(remaining);
  }

  factors.iter().map(|f| coeffs_to_expr(f, var)).collect()
}

/// Square-free factorization: detect repeated factors using Yun's algorithm,
/// then recursively factor each square-free component.
pub fn try_square_free_factor(coeffs: &[i128], var: &str) -> Vec<Expr> {
  let deg = coeffs.len() - 1;
  if deg <= 2 {
    return vec![];
  }

  let sqfree_parts = yun_square_free(coeffs);

  // Check if we found any multiplicity > 1
  let has_repeated = sqfree_parts.iter().any(|(_, m)| *m > 1);
  if !has_repeated {
    return vec![];
  }

  let mut all_factors: Vec<Expr> = Vec::new();

  for (part_coeffs, multiplicity) in &sqfree_parts {
    // Recursively factor each square-free part
    let sub = factor_sub_poly(part_coeffs, var);
    if sub.is_empty() {
      // part is trivial (constant 1)
      continue;
    }
    // Add each sub-factor with the correct multiplicity
    for _ in 0..*multiplicity {
      all_factors.extend(sub.clone());
    }
  }

  if all_factors.len() > 1 {
    all_factors
  } else {
    vec![]
  }
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
  if args.len() == 2 {
    if let Some(p) = extract_modulus_option(&args[1]) {
      return match super::factor_list_modulus(&args[0], p)? {
        Some(e) => Ok(e),
        None => Ok(Expr::FunctionCall {
          name: "FactorList".to_string(),
          args: args.to_vec().into(),
        }),
      };
    }
    return Ok(unknown_option_fallback("FactorList", args));
  }
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

  // Numeric content comes first, split into numerator/denominator entries,
  // matching wolframscript (e.g. `FactorList[3/4]` → `{{3, 1}, {4, -1}}`).
  let mut result = numeric_content_entries(&numeric_coeff);

  // Then the polynomial factors: numerator factors (positive exponent) first,
  // then denominator factors (negative exponent). `partition` is stable, so
  // the relative order of factors within each group is preserved.
  let (pos, neg): (Vec<Expr>, Vec<Expr>) = pairs
    .into_iter()
    .partition(|p| !factor_exponent_is_negative(p));
  result.extend(pos);
  result.extend(neg);

  Ok(Expr::List(result.into()))
}

/// The leading numeric-content entries of a `FactorList` result. An integer
/// content `n` becomes the single entry `{n, 1}`. A rational `p/q` (in lowest
/// terms, `q > 1`) becomes `{p, 1}` (omitted when `p == 1`) followed by
/// `{q, -1}` — matching wolframscript, which keeps the numerator and
/// denominator whole rather than prime-factoring them.
fn numeric_content_entries(coeff: &Expr) -> Vec<Expr> {
  let entry =
    |value: Expr, exp: i128| Expr::List(vec![value, Expr::Integer(exp)].into());
  if let Expr::FunctionCall { name, args } = coeff
    && name == "Rational"
    && args.len() == 2
  {
    let mut out = Vec::new();
    // The numerator carries the sign; `{1, 1}` is elided.
    if !matches!(&args[0], Expr::Integer(1)) {
      out.push(entry(args[0].clone(), 1));
    }
    out.push(entry(args[1].clone(), -1));
    return out;
  }
  vec![entry(coeff.clone(), 1)]
}

/// Whether a `{factor, exponent}` pair has a negative integer exponent — i.e.
/// it came from a rational-function denominator.
fn factor_exponent_is_negative(pair: &Expr) -> bool {
  matches!(pair, Expr::List(items)
    if items.len() == 2 && matches!(&items[1], Expr::Integer(n) if *n < 0))
}

/// IrreduciblePolynomialQ[poly] - test whether `poly` is irreducible over
/// the rationals (the default coefficient domain). A polynomial is
/// irreducible when, after stripping the constant (variable-free) content,
/// it factors into exactly one non-constant factor that appears to the
/// first power. Pure constants (numbers, Pi, E, ...) are not irreducible
/// polynomials and return False.
pub fn irreducible_polynomial_q_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() == 2 {
    if let Some(p) = extract_modulus_option(&args[1]) {
      return match super::irreducible_polynomial_q_modulus(&args[0], p)? {
        Some(e) => Ok(e),
        None => Ok(Expr::FunctionCall {
          name: "IrreduciblePolynomialQ".to_string(),
          args: args.to_vec().into(),
        }),
      };
    }
    return Ok(unknown_option_fallback("IrreduciblePolynomialQ", args));
  }
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "IrreduciblePolynomialQ expects exactly 1 argument".into(),
    ));
  }

  // A factor is "constant" if it contains no free variables. Numbers and
  // built-in constants such as Pi/E/I all count as constant. Named
  // constants are stored as `Expr::Constant`, which `collect_variables`
  // would otherwise treat as a variable, so check for them explicitly.
  fn has_variable(expr: &Expr) -> bool {
    match expr {
      Expr::Integer(_)
      | Expr::Real(_)
      | Expr::String(_)
      | Expr::BigInteger(_)
      | Expr::BigFloat(_, _)
      | Expr::Constant(_) => false,
      Expr::Identifier(s) => !matches!(
        s.as_str(),
        "True"
          | "False"
          | "Null"
          | "Pi"
          | "E"
          | "I"
          | "Infinity"
          | "ComplexInfinity"
          | "Indeterminate"
      ),
      Expr::List(items) => items.iter().any(has_variable),
      Expr::BinaryOp { left, right, .. } => {
        has_variable(left) || has_variable(right)
      }
      Expr::UnaryOp { operand, .. } => has_variable(operand),
      Expr::FunctionCall { args, .. } => args.iter().any(has_variable),
      _ => true,
    }
  }
  let is_constant_factor = |expr: &Expr| -> bool { !has_variable(expr) };

  // Use FactorList to obtain the {factor, exponent} decomposition, then
  // count the non-constant irreducible factors.
  let list = match factor_list_ast(args) {
    Ok(l) => l,
    Err(_) => {
      return Ok(Expr::Identifier("False".to_string()));
    }
  };

  let Expr::List(ref entries) = list else {
    return Ok(Expr::Identifier("False".to_string()));
  };

  let mut non_constant_count = 0usize;
  let mut has_higher_power = false;
  for entry in entries.iter() {
    if let Expr::List(pair) = entry
      && pair.len() == 2
      && !is_constant_factor(&pair[0])
    {
      non_constant_count += 1;
      // Exponent must be 1 for an irreducible polynomial.
      if !matches!(&pair[1], Expr::Integer(1)) {
        has_higher_power = true;
      }
    }
  }

  let irreducible = non_constant_count == 1 && !has_higher_power;
  Ok(Expr::Identifier(
    if irreducible { "True" } else { "False" }.to_string(),
  ))
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
        pairs.push(Expr::List(vec![*left.clone(), *right.clone()].into()));
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
        pairs.push(Expr::List(vec![args[0].clone(), args[1].clone()].into()));
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
      pairs.push(Expr::List(vec![expr.clone(), Expr::Integer(1)].into()));
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
    return Ok(Expr::List(results?.into()));
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

/// Total degree of a term's non-numeric part: each symbolic factor counts
/// 1, an integer power counts its exponent. Sin[x]^2*y → 3.
fn term_total_degree(term: &Expr) -> i128 {
  let inner = match term {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => operand.as_ref(),
    other => other,
  };
  let mut deg = 0i128;
  for f in collect_multiplicative_factors(inner) {
    let is_numeric = matches!(
      &f,
      Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_)
    ) || matches!(&f, Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2);
    if is_numeric {
      continue;
    }
    deg += match &f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        right,
        ..
      } => match right.as_ref() {
        Expr::Integer(n) => *n,
        _ => 1,
      },
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        match &args[1] {
          Expr::Integer(n) => *n,
          _ => 1,
        }
      }
      _ => 1,
    };
  }
  deg
}

/// Compute the rational content of additive terms: gcd of numerators over
/// lcm of denominators. The content carries the sign of the highest
/// total-degree term's coefficient (ties keep the first in canonical
/// order), matching wolframscript: FactorTerms[2 - 4x - 4x^2] →
/// -2*(-1 + 2x + 2x^2), FactorTerms[3 + 3x^3] → 3*(1 + x^3),
/// FactorTerms[2 Sin[x] - 4 Sin[y]] → 2*(Sin[x] - 2*Sin[y]).
/// Returns (signed num_gcd, den_lcm, per-term coefficients), or None if a
/// term has a non-rational coefficient.
fn rational_content(terms: &[Expr]) -> Option<(i128, i128, Vec<(i128, i128)>)> {
  let mut coeffs: Vec<(i128, i128)> = Vec::new();
  for term in terms {
    coeffs.push(extract_rational_coeff(term)?);
  }
  if coeffs.is_empty() {
    return None;
  }

  let num_gcd = coeffs
    .iter()
    .map(|(n, _)| *n)
    .filter(|&n| n != 0)
    .fold(0i128, gcd_i128);
  let den_lcm = coeffs.iter().map(|(_, d)| *d).fold(1i128, lcm_i128);

  let mut best_deg = i128::MIN;
  let mut sign: i128 = 1;
  for (term, (n, _)) in terms.iter().zip(coeffs.iter()) {
    if *n == 0 {
      continue;
    }
    let d = term_total_degree(term);
    if d > best_deg {
      best_deg = d;
      sign = if *n < 0 { -1 } else { 1 };
    }
  }

  Some((num_gcd * sign, den_lcm, coeffs))
}

/// Divide each term by num_gcd/den_lcm and rebuild the term list.
fn divide_terms_by(
  terms: &[Expr],
  coeffs: &[(i128, i128)],
  num_gcd: i128,
  den_lcm: i128,
) -> Vec<Expr> {
  let mut new_terms: Vec<Expr> = Vec::new();
  for (term, (n, d)) in terms.iter().zip(coeffs.iter()) {
    // New coefficient: (n/d) / (num_gcd/den_lcm) = (n * den_lcm) / (d * num_gcd)
    let new_n = n * den_lcm / (d * num_gcd);
    let var_factors = extract_non_numeric_factors(term);

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

    // Constants stay signed Integers (not UnaryOp Minus around a positive
    // one) and negative coefficients keep their sign inside the Times so a
    // leading term renders `-2*x`, not `-(2*x)`.
    let final_term = match var_part {
      None => Expr::Integer(new_n),
      Some(v) => match new_n {
        1 => v,
        -1 => Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(v),
        },
        _ => Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(new_n)),
          right: Box::new(v),
        },
      },
    };

    new_terms.push(final_term);
  }

  new_terms
}

/// Factor out the GCD of numeric coefficients from all terms.
pub(crate) fn factor_terms_numeric(
  expanded: &Expr,
  terms: &[Expr],
) -> Result<Expr, InterpreterError> {
  let Some((num_gcd, den_lcm, coeffs)) = rational_content(terms) else {
    return Ok(expanded.clone());
  };

  if num_gcd.abs() <= 1 && den_lcm == 1 {
    return Ok(expanded.clone());
  }

  // Display flip: Times[Rational[-1,d], sum] renders as Rational[1,d] times
  // the negated sum in wolframscript — FactorTerms[x/2 - x^2/2] →
  // (x - x^2)/2, not -1/2*(-x + x^2).
  let num_gcd = if num_gcd == -1 && den_lcm != 1 {
    1
  } else {
    num_gcd
  };

  let new_terms = divide_terms_by(terms, &coeffs, num_gcd, den_lcm);

  // Build the inner sum
  let inner = build_sum(new_terms);

  // Build the overall factor
  let factor = if den_lcm == 1 {
    Expr::Integer(num_gcd)
  } else {
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(num_gcd), Expr::Integer(den_lcm)].into(),
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
    return Ok(Expr::List(results?.into()));
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

  // Use Yun's algorithm for square-free factorization, then further factor each component
  let sff = yun_square_free(&pp);

  if sff.is_empty() {
    if overall == 1 {
      return Ok(expanded);
    }
    let poly_expr = int_coeffs_to_canonical_expr(&pp, &var);
    return Ok(build_product(vec![Expr::Integer(overall), poly_expr]));
  }

  // Split out x factors from each square-free component
  let mut all_pairs: Vec<(Vec<i128>, i128)> = Vec::new();
  for (factor_coeffs, mult) in &sff {
    let (x_pow, remaining) = split_x_factor(factor_coeffs);
    if x_pow > 0 {
      // x^x_pow with this multiplicity -> x with mult * x_pow
      all_pairs.push((vec![0, 1], *mult * x_pow as i128));
      if remaining.len() > 1 || (remaining.len() == 1 && remaining[0] != 1) {
        all_pairs.push((remaining, *mult));
      }
    } else {
      all_pairs.push((factor_coeffs.clone(), *mult));
    }
  }

  // Sort by constant term (ascending) for consistent output
  all_pairs.sort_by_key(|(c, _)| c[0]);

  let mut result_factors: Vec<Expr> = Vec::new();

  if overall != 1 {
    result_factors.push(Expr::Integer(overall));
  }

  for (factor_coeffs, mult) in &all_pairs {
    let factor_expr = int_coeffs_to_canonical_expr(factor_coeffs, &var);
    if *mult == 1 {
      result_factors.push(factor_expr);
    } else {
      result_factors.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(factor_expr),
        right: Box::new(Expr::Integer(*mult)),
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

/// FactorSquareFreeList[poly] - returns {{f1, e1}, {f2, e2}, ...}
/// where poly = f1^e1 * f2^e2 * ..., and each fi is square-free.
/// The first entry is always {numeric_coefficient, 1}.
/// Compute the formal derivative of a polynomial given as coefficient vector.
fn poly_derivative(p: &[i128]) -> Vec<i128> {
  if p.len() <= 1 {
    return vec![0];
  }
  p.iter()
    .enumerate()
    .skip(1)
    .map(|(i, &c)| c * i as i128)
    .collect()
}

/// Split a coefficient vector into x^k and the remaining polynomial.
/// E.g., [0, 0, 1, -1] (= x^2 - x^3) becomes (2, [1, -1]) meaning x^2 * (1 - x).
fn split_x_factor(coeffs: &[i128]) -> (usize, Vec<i128>) {
  let leading_zeros = coeffs.iter().take_while(|&&c| c == 0).count();
  if leading_zeros == 0 || leading_zeros >= coeffs.len() {
    return (0, coeffs.to_vec());
  }
  (leading_zeros, coeffs[leading_zeros..].to_vec())
}

/// Square-free factorization using Yun's algorithm.
/// Returns pairs (factor_coeffs, multiplicity) where each factor is square-free.
/// The input polynomial should be primitive (content already extracted).
fn yun_square_free(pp: &[i128]) -> Vec<(Vec<i128>, i128)> {
  use crate::functions::polynomial_ast::cancel::{poly_exact_divide, poly_gcd};

  if pp.len() <= 1 {
    return vec![];
  }

  let fp = poly_derivative(pp);
  let g = match poly_gcd(pp, &fp) {
    Some(g) => g,
    None => return vec![(pp.to_vec(), 1)],
  };

  // If GCD is constant (degree 0), polynomial is already square-free
  let g_is_one =
    g.len() == 1 || (g.iter().rev().skip(1).all(|&c| c == 0) && g[0] != 0);
  if g_is_one {
    return vec![(pp.to_vec(), 1)];
  }

  let mut w = match poly_exact_divide(pp, &g) {
    Some(q) => q,
    None => return vec![(pp.to_vec(), 1)],
  };

  let mut result: Vec<(Vec<i128>, i128)> = Vec::new();
  let mut g = g;
  let mut i: i128 = 1;

  loop {
    if g.len() <= 1 || (g.iter().rev().skip(1).all(|&c| c == 0) && g[0] != 0) {
      break;
    }

    let y = match poly_gcd(&w, &g) {
      Some(y) => y,
      None => break,
    };
    let z = match poly_exact_divide(&w, &y) {
      Some(q) => q,
      None => break,
    };

    // z is the square-free factor with multiplicity i
    let z_is_trivial =
      z.len() == 1 && (z[0] == 1 || z[0] == -1) || z.is_empty();
    if !z_is_trivial {
      // Normalize: make monic-positive
      let mut zn = z;
      if zn.last().map(|&c| c < 0).unwrap_or(false) {
        zn = zn.iter().map(|c| -c).collect();
      }
      let zg = zn.iter().copied().filter(|&c| c != 0).fold(0i128, gcd_i128);
      if zg > 1 {
        zn = zn.iter().map(|c| c / zg).collect();
      }
      result.push((zn, i));
    }

    g = match poly_exact_divide(&g, &y) {
      Some(q) => q,
      None => break,
    };
    w = y;
    i += 1;
  }

  // If w is non-trivial, it's a factor with multiplicity i
  let w_is_trivial = w.len() == 1 && (w[0] == 1 || w[0] == -1) || w.is_empty();
  if !w_is_trivial {
    let mut wn = w;
    if wn.last().map(|&c| c < 0).unwrap_or(false) {
      wn = wn.iter().map(|c| -c).collect();
    }
    let wg = wn.iter().copied().filter(|&c| c != 0).fold(0i128, gcd_i128);
    if wg > 1 {
      wn = wn.iter().map(|c| c / wg).collect();
    }
    result.push((wn, i));
  }

  result
}

pub fn factor_square_free_list_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FactorSquareFreeList expects exactly 1 argument".into(),
    ));
  }

  let expanded = expand_and_combine(&args[0]);

  // Handle zero
  if matches!(&expanded, Expr::Integer(0)) {
    return Ok(Expr::List(
      vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into())].into(),
    ));
  }

  // Handle pure numeric
  if let Expr::Integer(n) = &expanded {
    return Ok(Expr::List(
      vec![Expr::List(vec![Expr::Integer(*n), Expr::Integer(1)].into())].into(),
    ));
  }

  let var = match find_single_variable(&expanded) {
    Some(v) => v,
    None => {
      // Constant expression
      return Ok(Expr::List(
        vec![Expr::List(vec![expanded, Expr::Integer(1)].into())].into(),
      ));
    }
  };

  let coeffs = match extract_poly_coeffs(&expanded, &var) {
    Some(c) => c,
    None => {
      return Ok(Expr::List(
        vec![Expr::List(vec![expanded, Expr::Integer(1)].into())].into(),
      ));
    }
  };

  // Factor out GCD of coefficients (content)
  let content = coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if content == 0 {
    return Ok(Expr::List(
      vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into())].into(),
    ));
  }

  let pp: Vec<i128> = coeffs.iter().map(|c| c / content).collect();

  // Make leading coefficient positive
  let (sign, pp) = if pp.last().map(|&c| c < 0).unwrap_or(false) {
    (-1i128, pp.iter().map(|c| -c).collect::<Vec<_>>())
  } else {
    (1, pp)
  };
  let overall = content * sign;

  // Use Yun's algorithm for square-free factorization, then further factor each component
  let sff = yun_square_free(&pp);

  // Build result list: {overall, 1}, then {factor, mult} pairs
  let mut result = vec![Expr::List(
    vec![Expr::Integer(overall), Expr::Integer(1)].into(),
  )];

  // Split out x factors from each square-free component
  let mut pairs: Vec<(Vec<i128>, i128)> = Vec::new();
  for (factor_coeffs, mult) in &sff {
    let (x_pow, remaining) = split_x_factor(factor_coeffs);
    if x_pow > 0 {
      pairs.push((vec![0, 1], *mult * x_pow as i128));
      if remaining.len() > 1 || (remaining.len() == 1 && remaining[0] != 1) {
        pairs.push((remaining, *mult));
      }
    } else {
      pairs.push((factor_coeffs.clone(), *mult));
    }
  }

  // Sort by constant term (value at x=0) in ascending order (Wolfram convention)
  pairs.sort_by_key(|(c, _)| c[0]);

  for (factor_coeffs, mult) in &pairs {
    let factor_expr = int_coeffs_to_canonical_expr(factor_coeffs, &var);
    result.push(Expr::List(vec![factor_expr, Expr::Integer(*mult)].into()));
  }

  Ok(Expr::List(result.into()))
}

/// Extract the integer GCD content from a multi-variable expanded
/// polynomial. The sign comes from the first (canonical-leading) term —
/// Wolfram's `FactorTermsList[3*(-1+2x)*(-1+y)*(1-a)]` returns
/// `{-3, -1+a+2*x-...}` because the leading term `-6*a*x*y` is negative.
fn extract_multi_var_content(expanded: &Expr) -> Expr {
  // Collect Plus summands.
  let summands: Vec<Expr> = match expanded {
    Expr::FunctionCall { name, args } if name == "Plus" => args.to_vec(),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left,
      right,
    } => {
      let mut v = Vec::new();
      collect_plus(left, &mut v);
      collect_plus(right, &mut v);
      v
    }
    _ => vec![expanded.clone()],
  };

  // Extract the integer coefficient from each term.
  fn term_coeff(term: &Expr) -> Option<i128> {
    match term {
      Expr::Integer(n) => Some(*n),
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand,
      } => term_coeff(operand).map(|c| -c),
      Expr::FunctionCall { name, args } if name == "Times" => {
        // Find an integer coefficient (default to 1, with sign tracked).
        let mut coeff: i128 = 1;
        for a in args {
          if let Expr::Integer(n) = a {
            coeff *= n;
          } else if let Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand,
          } = a
            && let Expr::Integer(n) = operand.as_ref()
          {
            coeff *= -n;
          }
        }
        Some(coeff)
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left,
        right,
      } => match (left.as_ref(), right.as_ref()) {
        (Expr::Integer(n), _) | (_, Expr::Integer(n)) => Some(*n),
        _ => Some(1),
      },
      _ => Some(1),
    }
  }

  let coeffs: Vec<i128> = summands.iter().filter_map(term_coeff).collect();
  if coeffs.is_empty() {
    return Expr::List(vec![Expr::Integer(1), expanded.clone()].into());
  }
  let abs_gcd = coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, |a, b| gcd_i128(a, b.abs()));
  if abs_gcd == 0 {
    return Expr::List(vec![Expr::Integer(1), expanded.clone()].into());
  }
  // Sign: take the sign of the first non-constant non-zero coefficient.
  // In Wolfram canonical Plus order, a bare constant (e.g. `3`) is the
  // first summand; the sign of the FIRST VARIABLE term then determines
  // the content sign. This makes the primitive part's first variable
  // term positive — matching wolframscript:
  //   `FactorTermsList[6*(a+b)*(c-d)]` → `{6, a*c + b*c - a*d - b*d}`
  //     (first term is `6*a*c`; sign +; content +6)
  //   `FactorTermsList[3*(-1+2x)*(-1+y)*(1-a)]` →
  //     `{-3, -1 + a + 2*x - ...}`
  //     (first term is `3` constant, second is `-3*a`; sign −; content −3)
  let sign = {
    let mut found = None;
    for (i, c) in coeffs.iter().enumerate() {
      if *c == 0 {
        continue;
      }
      // Is this term a bare constant? `summands[i]` would be Integer or
      // UnaryOp(Minus, Integer).
      let is_const = matches!(
        &summands[i],
        Expr::Integer(_)
          | Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            ..
          }
      ) && match &summands[i] {
        Expr::Integer(_) => true,
        Expr::UnaryOp { operand, .. } => {
          matches!(operand.as_ref(), Expr::Integer(_))
        }
        _ => false,
      };
      if !is_const {
        found = Some(c.signum());
        break;
      }
    }
    // Fallback: if every term is a constant (single-term polynomial), use
    // its sign.
    found.unwrap_or_else(|| {
      coeffs
        .iter()
        .find(|&&c| c != 0)
        .map(|&c| c.signum())
        .unwrap_or(1)
    })
  };
  let signed_content = abs_gcd * sign;
  // Divide each summand's integer coefficient by signed_content and rebuild
  // the polynomial so the result stays in canonical expanded form (no
  // surrounding `(…)/n` fraction).
  fn divide_term(term: &Expr, div: i128) -> Expr {
    match term {
      Expr::Integer(n) => {
        if n % div == 0 {
          Expr::Integer(n / div)
        } else {
          crate::functions::math_ast::divide_ast(&[
            term.clone(),
            Expr::Integer(div),
          ])
          .unwrap_or_else(|_| term.clone())
        }
      }
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand,
      } => {
        let inner = divide_term(operand, div);
        crate::functions::math_ast::times_ast(&[Expr::Integer(-1), inner])
          .unwrap_or_else(|_| Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand: Box::new(divide_term(operand, div)),
          })
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        let mut new_args = Vec::with_capacity(args.len());
        let mut consumed = false;
        for a in args {
          if !consumed
            && let Expr::Integer(n) = a
            && n % div == 0
          {
            let q = n / div;
            if q != 1 {
              new_args.push(Expr::Integer(q));
            }
            consumed = true;
            continue;
          }
          new_args.push(a.clone());
        }
        if !consumed {
          // No integer factor matched — prepend 1/div as a Rational.
          new_args
            .insert(0, crate::functions::math_ast::make_rational_pub(1, div));
        }
        crate::functions::math_ast::times_ast(&new_args)
          .unwrap_or_else(|_| term.clone())
      }
      _ => crate::functions::math_ast::divide_ast(&[
        term.clone(),
        Expr::Integer(div),
      ])
      .unwrap_or_else(|_| term.clone()),
    }
  }
  let new_summands: Vec<Expr> = summands
    .iter()
    .map(|t| divide_term(t, signed_content))
    .collect();
  let primitive = match crate::functions::math_ast::plus_ast(&new_summands) {
    Ok(p) => p,
    Err(_) => expanded.clone(),
  };
  Expr::List(vec![Expr::Integer(signed_content), primitive].into())
}

fn collect_times_factors(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      let mut fs = collect_times_factors(left);
      fs.extend(collect_times_factors(right));
      fs
    }
    Expr::FunctionCall { name, args } if name == "Times" => args.to_vec(),
    _ => vec![expr.clone()],
  }
}

fn collect_plus(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for a in args {
        collect_plus(a, out);
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left,
      right,
    } => {
      collect_plus(left, out);
      collect_plus(right, out);
    }
    _ => out.push(expr.clone()),
  }
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
    return Ok(Expr::List(vec![Expr::Integer(1), Expr::Integer(0)].into()));
  }

  // Handle pure integer
  if let Expr::Integer(n) = &expanded {
    return Ok(Expr::List(vec![Expr::Integer(*n), Expr::Integer(1)].into()));
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
      // Multi-variable polynomial — extract the integer GCD content of the
      // expanded sum. Wolfram's `FactorTermsList[3*(-1+2*x)*(-1+y)*(1-a)]`
      // returns `{-3, ...}` because the leading (canonical-first) term in
      // the expanded form has a negative coefficient, and the absolute GCD
      // of all coefficients is 3.
      return Ok(extract_multi_var_content(&expanded));
    }
  };

  // 2-arg form with a chosen variable: get the multi-var content, then
  // factor the primitive into (non-var part) × (var part) using Factor.
  // E.g. `FactorTermsList[3*(-1+2x)*(-1+y)*(1-a), x]` →
  // `{-3, 1-a-y+a*y, -1+2*x}`. The numeric content is taken from the
  // multi-var extraction; the polynomial-level split groups the
  // factored result by which factors mention `var`.
  if args.len() == 2 {
    let multi = extract_multi_var_content(&expanded);
    if let Expr::List(parts) = &multi
      && parts.len() == 2
      && let Expr::Integer(num_content) = &parts[0]
    {
      let mut content = Expr::Integer(*num_content);
      let mut primitive = parts[1].clone();
      // Pull rational content the integer-only multi-var pass misses:
      // FactorTermsList[x/2 - x^2/2, x] → {-1/2, 1, -x + x^2}.
      if *num_content == 1 {
        let terms = collect_additive_terms(&primitive);
        if let Some((num_gcd, den_lcm, term_coeffs)) = rational_content(&terms)
          && (num_gcd.abs() != 1 || den_lcm != 1)
        {
          primitive =
            build_sum(divide_terms_by(&terms, &term_coeffs, num_gcd, den_lcm));
          content = if den_lcm == 1 {
            Expr::Integer(num_gcd)
          } else {
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(num_gcd), Expr::Integer(den_lcm)].into(),
            }
          };
        }
      }
      let primitive = &primitive;
      // If the primitive doesn't actually depend on `var`, route to the
      // existing fallback below.
      if contains_var(primitive, &var) {
        let factored = factor_ast(&[primitive.clone()])?;
        let factors = collect_times_factors(&factored);
        let mut var_factors: Vec<Expr> = Vec::new();
        let mut non_var_factors: Vec<Expr> = Vec::new();
        for f in factors {
          if contains_var(&f, &var) {
            var_factors.push(f);
          } else {
            non_var_factors.push(f);
          }
        }
        let combine = |fs: Vec<Expr>| -> Expr {
          if fs.is_empty() {
            Expr::Integer(1)
          } else if fs.len() == 1 {
            fs.into_iter().next().unwrap()
          } else {
            crate::functions::math_ast::times_ast(&fs)
              .unwrap_or(Expr::Integer(1))
          }
        };
        let non_var_part = combine(non_var_factors);
        let var_part = combine(var_factors);
        // Expand both parts so their display matches wolframscript:
        // `1 - a - y + a*y` (not `(-1+a)*(-1+y)`) and `1 - 4*x + 4*x^2`
        // (not `(-1+2*x)^2`).
        let non_var_expanded = expand_and_combine(&non_var_part);
        let var_expanded = expand_and_combine(&var_part);
        return Ok(Expr::List(
          vec![content, non_var_expanded, var_expanded].into(),
        ));
      }
    }
  }

  let coeffs = match extract_poly_coeffs(&expanded, &var) {
    Some(c) => c,
    None => {
      // Rational coefficients: extract the rational content num_gcd/den_lcm
      // (sign of the highest-degree term), matching wolframscript:
      // FactorTermsList[x/2 - x^2/2] → {-1/2, -x + x^2}.
      let terms = collect_additive_terms(&expanded);
      if let Some((num_gcd, den_lcm, term_coeffs)) = rational_content(&terms)
        && (num_gcd.abs() != 1 || den_lcm != 1)
      {
        let inner =
          build_sum(divide_terms_by(&terms, &term_coeffs, num_gcd, den_lcm));
        let content = if den_lcm == 1 {
          Expr::Integer(num_gcd)
        } else {
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(num_gcd), Expr::Integer(den_lcm)].into(),
          }
        };
        if args.len() == 2 {
          // The var-dependent primitive goes in the third slot; a
          // primitive independent of `var` is the non-var second slot
          // (FactorTermsList[3 f, x] → {3, f, 1}).
          if contains_var(&inner, &var) {
            return Ok(Expr::List(
              vec![content, Expr::Integer(1), inner].into(),
            ));
          }
          return Ok(Expr::List(vec![content, inner, Expr::Integer(1)].into()));
        }
        return Ok(Expr::List(vec![content, inner].into()));
      }
      // 2-arg form always returns a 3-element {numeric, non-var, var-part}
      // list. When we can't extract integer coefficients, conservatively
      // split off a numeric factor and route the rest based on whether it
      // depends on the chosen variable.
      if args.len() == 2 {
        if contains_var(&expanded, &var) {
          return Ok(Expr::List(
            vec![Expr::Integer(1), Expr::Integer(1), expanded].into(),
          ));
        }
        let (num_coeff, _key, var_factors) = decompose_term(&expanded);
        let non_var = if var_factors.is_empty() {
          Expr::Integer(1)
        } else if var_factors.len() == 1 {
          var_factors.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: var_factors.into(),
          }
        };
        return Ok(Expr::List(
          vec![num_coeff, non_var, Expr::Integer(1)].into(),
        ));
      }
      return Ok(Expr::List(vec![Expr::Integer(1), expanded].into()));
    }
  };

  // Compute content (GCD of coefficients)
  let content = coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if content == 0 {
    return Ok(Expr::List(vec![Expr::Integer(1), Expr::Integer(0)].into()));
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
    return Ok(Expr::List(
      vec![Expr::Integer(signed_content), Expr::Integer(1), pp_expr].into(),
    ));
  }

  Ok(Expr::List(
    vec![Expr::Integer(signed_content), pp_expr].into(),
  ))
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

// ─── Multivariate Factoring (Kronecker Substitution) ────────────────

/// Match the AST of a^n ± b^n where a, b are distinct identifiers and n ≥ 2.
/// Returns `(a_name, b_name, n, is_minus)` where `is_minus == true` means
/// the second term carries a minus sign (i.e. a^n - b^n). The first variable
/// is whichever side carries the positive sign; for a^n + b^n the order from
/// the source is preserved.
fn match_homogeneous_binomial(
  expr: &Expr,
) -> Option<(String, String, u64, bool)> {
  // Extract ±var^n from a single term. Returns (var, n, positive).
  fn extract_term(t: &Expr) -> Option<(String, u64, bool)> {
    match t {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => {
        let v = match left.as_ref() {
          Expr::Identifier(s) => s.clone(),
          _ => return None,
        };
        let n = match right.as_ref() {
          Expr::Integer(k) if *k >= 2 => *k as u64,
          _ => return None,
        };
        Some((v, n, true))
      }
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        let v = match &args[0] {
          Expr::Identifier(s) => s.clone(),
          _ => return None,
        };
        let n = match &args[1] {
          Expr::Integer(k) if *k >= 2 => *k as u64,
          _ => return None,
        };
        Some((v, n, true))
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => {
        let (v, n, _) = extract_term(operand)?;
        Some((v, n, false))
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } if matches!(left.as_ref(), Expr::Integer(-1)) => {
        let (v, n, _) = extract_term(right)?;
        Some((v, n, false))
      }
      Expr::FunctionCall { name, args }
        if name == "Times"
          && args.len() == 2
          && matches!(&args[0], Expr::Integer(-1)) =>
      {
        let (v, n, _) = extract_term(&args[1])?;
        Some((v, n, false))
      }
      _ => None,
    }
  }

  let (t1, t2) = match expr {
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => (left.as_ref(), right.as_ref()),
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      // a^n - b^n form
      let (v1, n1, sign1) = extract_term(left)?;
      let (v2, n2, _) = extract_term(right)?;
      if !sign1 || v1 == v2 || n1 != n2 {
        return None;
      }
      return Some((v1, v2, n1, true));
    }
    _ => return None,
  };

  let (v1, n1, sign1) = extract_term(t1)?;
  let (v2, n2, sign2) = extract_term(t2)?;
  if v1 == v2 || n1 != n2 {
    return None;
  }
  match (sign1, sign2) {
    (true, false) => Some((v1, v2, n1, true)),
    (false, true) => Some((v2, v1, n1, true)),
    (true, true) => Some((v1, v2, n1, false)),
    (false, false) => None,
  }
}

/// Build the homogeneous d-th cyclotomic polynomial Φ_d(a, b) as an Expr.
/// Φ_d(a, b) = sum_k c_k * a^k * b^(φ_d - k) where c_k is the coefficient
/// of x^k in the univariate Φ_d(x).
fn homogeneous_cyclotomic(d: u64, a: &str, b: &str) -> Expr {
  let coeffs = cyclotomic_poly(d);
  let deg = coeffs.len() - 1;
  let mut terms: Vec<Expr> = Vec::with_capacity(coeffs.len());
  for (k, &c) in coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let a_pow = match k {
      0 => None,
      1 => Some(Expr::Identifier(a.to_string())),
      _ => Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier(a.to_string())),
        right: Box::new(Expr::Integer(k as i128)),
      }),
    };
    let b_exp = deg - k;
    let b_pow = match b_exp {
      0 => None,
      1 => Some(Expr::Identifier(b.to_string())),
      _ => Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier(b.to_string())),
        right: Box::new(Expr::Integer(b_exp as i128)),
      }),
    };
    let mut factors: Vec<Expr> = Vec::new();
    if c.abs() != 1 {
      factors.push(Expr::Integer(c.abs()));
    }
    if let Some(p) = a_pow {
      factors.push(p);
    }
    if let Some(p) = b_pow {
      factors.push(p);
    }
    let body = if factors.is_empty() {
      Expr::Integer(1)
    } else if factors.len() == 1 {
      factors.remove(0)
    } else {
      build_product(factors)
    };
    let signed = if c < 0 {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(body),
      }
    } else {
      body
    };
    terms.push(signed);
  }
  let sum = if terms.len() == 1 {
    terms.remove(0)
  } else {
    build_sum(terms)
  };
  // Canonicalise term order via expand_and_combine (no actual expansion happens
  // since terms are already monomials).
  expand_and_combine(&sum)
}

/// Fast path for `Factor[a^n ± b^n]` via cyclotomic decomposition:
///   a^n - b^n = ∏_{d | n}            Φ_d(a, b)
///   a^n + b^n = ∏_{d | 2n, d ∤ n}    Φ_d(a, b)
/// Returns `None` if the input doesn't match the homogeneous binomial pattern
/// or if the result has only one factor (i.e. already irreducible).
fn try_factor_homogeneous_binomial(expr: &Expr) -> Option<Expr> {
  let (a, b, n, is_minus) = match_homogeneous_binomial(expr)?;

  let ds: Vec<u64> = if is_minus {
    divisors_of(n)
  } else {
    let dvs2n = divisors_of(2 * n);
    let dvsn: std::collections::HashSet<u64> =
      divisors_of(n).into_iter().collect();
    dvs2n.into_iter().filter(|d| !dvsn.contains(d)).collect()
  };

  if ds.len() < 2 {
    return None;
  }

  let mut factors: Vec<Expr> = ds
    .into_iter()
    .map(|d| homogeneous_cyclotomic(d, &a, &b))
    .collect();

  // Sort factors to match wolframscript's display order:
  //   degree ASC, then term count ASC, then min coefficient ASC.
  // This puts smaller-degree factors first, then within a degree the
  // shorter polynomials, then ones whose terms include negatives.
  factors.sort_by(|x, y| {
    let dx = factor_degree(x);
    let dy = factor_degree(y);
    if dx != dy {
      return dx.cmp(&dy);
    }
    let tx = factor_term_count(x);
    let ty = factor_term_count(y);
    if tx != ty {
      return tx.cmp(&ty);
    }
    let mx = min_term_coefficient(x);
    let my = min_term_coefficient(y);
    mx.cmp(&my)
  });

  if factors.len() == 1 {
    return Some(factors.remove(0));
  }
  Some(build_product(factors))
}

/// Minimum integer coefficient across all additive terms of `expr`.
/// Used by the homogeneous-binomial fast path to order Phi_3 / Phi_6
/// style factors (which tie on degree and term count but differ in
/// sign pattern of the middle term).
fn min_term_coefficient(expr: &Expr) -> i128 {
  collect_additive_terms(expr)
    .iter()
    .map(term_integer_coeff)
    .min()
    .unwrap_or(0)
}

/// Factor a multivariate polynomial expression using Kronecker substitution.
fn factor_multivariate(
  expanded: &Expr,
  vars: std::collections::HashSet<String>,
) -> Result<Expr, InterpreterError> {
  // Fast path: a^n ± b^n decomposes directly via cyclotomic polynomials.
  // Skips the expensive Kronecker substitution path, which was effectively
  // hanging on inputs like `Factor[x^10 - y^10]` (degree-110 sparse trial
  // divisions).
  if let Some(result) = try_factor_homogeneous_binomial(expanded) {
    return Ok(result);
  }

  let mut sorted_vars: Vec<String> = vars.into_iter().collect();
  sorted_vars.sort();

  // Extract GCD of all integer coefficients
  let gcd = extract_multivar_gcd(expanded);
  let working = if gcd > 1 {
    divide_expr_by_scalar(expanded, gcd)
  } else {
    expanded.clone()
  };

  // Extract common monomial factor (e.g., x^2*y from all terms)
  let (monomial_factors, working) =
    extract_common_monomial(&working, &sorted_vars);

  // Check if the remaining polynomial still has multiple variables
  let mut remaining_vars = std::collections::HashSet::new();
  collect_variables(&working, &mut remaining_vars);

  let poly_factors = if remaining_vars.len() > 1 {
    // Try Kronecker substitution on the remaining polynomial
    match kronecker_factor_multivar(&working, &sorted_vars) {
      Some((overall_sign, factors)) if !factors.is_empty() => {
        let overall = gcd * overall_sign;
        let mut all_factors = monomial_factors;
        all_factors.extend(factors);
        return Ok(build_multivariate_result(overall, all_factors));
      }
      _ => {
        // Can't factor further; return with monomial factors
        if monomial_factors.is_empty() && gcd == 1 {
          return Ok(expanded.clone());
        }
        let mut all_factors = monomial_factors;
        all_factors.push((working.clone(), 1));
        return Ok(build_multivariate_result(gcd, all_factors));
      }
    }
  } else if remaining_vars.len() == 1 {
    // Remaining is univariate — factor it with existing code
    let var = remaining_vars.into_iter().next().unwrap();
    if let Some(coeffs) = extract_poly_coeffs(&working, &var) {
      let coeff_gcd = coeffs
        .iter()
        .copied()
        .filter(|&c| c != 0)
        .fold(0i128, gcd_i128);
      if coeff_gcd > 0 {
        let reduced: Vec<i128> = coeffs.iter().map(|c| c / coeff_gcd).collect();
        let (sign, reduced) = if reduced.last().map(|&c| c < 0).unwrap_or(false)
        {
          (-1i128, reduced.iter().map(|c| -c).collect::<Vec<_>>())
        } else {
          (1, reduced)
        };
        let uni_factors = factor_integer_poly(&reduced, &var);
        if uni_factors.len() > 1 || !monomial_factors.is_empty() || gcd > 1 {
          let overall = gcd * coeff_gcd * sign;
          let mut all_factors = monomial_factors;
          // Group and add univariate factors
          let mut grouped: Vec<(Expr, usize)> = Vec::new();
          for f in &uni_factors {
            let key = expr_to_string(f);
            if let Some(entry) =
              grouped.iter_mut().find(|(e, _)| expr_to_string(e) == key)
            {
              entry.1 += 1;
            } else {
              grouped.push((f.clone(), 1));
            }
          }
          all_factors.extend(grouped);
          return Ok(build_multivariate_result(overall, all_factors));
        }
      }
    }
    vec![]
  } else {
    vec![]
  };

  if !poly_factors.is_empty() || !monomial_factors.is_empty() {
    let mut all_factors = monomial_factors;
    all_factors.extend(poly_factors);
    return Ok(build_multivariate_result(gcd, all_factors));
  }

  Ok(expanded.clone())
}

/// Core Kronecker substitution algorithm for multivariate factoring.
/// Returns Some((overall_sign, factors)) where factors is Vec<(expr, multiplicity)>,
/// or None if can't factor.
fn kronecker_factor_multivar(
  poly: &Expr,
  sorted_vars: &[String],
) -> Option<(i128, Vec<(Expr, usize)>)> {
  if sorted_vars.len() < 2 {
    return None;
  }

  // Primary variable is the first one alphabetically
  let primary = &sorted_vars[0];
  let secondary_vars = &sorted_vars[1..];

  // Compute D = max degree over all variables + 1
  let mut max_deg: i128 = 0;
  for var in sorted_vars {
    if let Some(d) = max_power_int(poly, var) {
      max_deg = max_deg.max(d);
    }
  }
  let d = (max_deg + 1) as usize;

  // Check that substituted degree won't be too large
  let mut total_deg_bound = d;
  for _ in 1..secondary_vars.len() {
    total_deg_bound = total_deg_bound.saturating_mul(d);
  }
  if total_deg_bound > 200 || max_deg as usize * total_deg_bound > 200 {
    return None; // Degree too large
  }

  // Forward substitution: compute univariate coefficients directly
  // by mapping each multivariate term's exponent vector to a single index
  let coeffs = multivar_to_univar_coeffs(poly, sorted_vars, d)?;

  // Factor out GCD and normalize sign
  let coeff_gcd = coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if coeff_gcd == 0 {
    return None;
  }
  let reduced: Vec<i128> = coeffs.iter().map(|c| c / coeff_gcd).collect();
  let (reduced, uni_sign) = if reduced.last().map(|&c| c < 0).unwrap_or(false) {
    (reduced.iter().map(|c| -c).collect::<Vec<_>>(), -1i128)
  } else {
    (reduced, 1i128)
  };
  let overall_sign = coeff_gcd * uni_sign;

  // Factor the univariate polynomial
  let factor_exprs = factor_integer_poly(&reduced, primary);
  if factor_exprs.len() <= 1 {
    return None; // Couldn't factor
  }

  // Convert factor expressions back to coefficient vectors
  let mut factor_coeffs: Vec<Vec<i128>> = Vec::new();
  for f in &factor_exprs {
    match extract_poly_coeffs(f, primary) {
      Some(c) => factor_coeffs.push(c),
      None => return None,
    }
  }

  // Group identical factors
  let grouped = group_factors(&factor_coeffs);
  if grouped.len() > 15 {
    return None; // Too many distinct factors
  }

  // Try to recombine factors into multivariate factors
  // Pass `reduced` (not `coeffs`) since factors are of the reduced polynomial
  recombine_factors(&reduced, &grouped, sorted_vars, d)
    .map(|factors| (overall_sign, factors))
}

/// Compute univariate coefficient vector from a multivariate polynomial
/// via Kronecker substitution: for each term, map its exponent vector
/// (e_primary, e_var1, e_var2, ...) to index = e_primary + D*e_var1 + D^2*e_var2 + ...
fn multivar_to_univar_coeffs(
  poly: &Expr,
  sorted_vars: &[String],
  d: usize,
) -> Option<Vec<i128>> {
  let terms = collect_additive_terms(poly);
  let mut max_index: usize = 0;
  let mut term_data: Vec<(usize, i128)> = Vec::new();

  for term in &terms {
    // Extract the integer coefficient and exponents for each variable
    let (int_coeff, exponents) = extract_multivar_term_data(term, sorted_vars)?;

    // Compute the univariate index
    let mut index: usize = 0;
    let mut multiplier: usize = 1;
    for &exp in &exponents {
      let Some(term) = (exp as usize).checked_mul(multiplier) else {
        return None;
      };
      let Some(new_index) = index.checked_add(term) else {
        return None;
      };
      index = new_index;
      let Some(new_mult) = multiplier.checked_mul(d) else {
        return None;
      };
      multiplier = new_mult;
    }

    max_index = max_index.max(index);
    term_data.push((index, int_coeff));
  }

  if max_index > 500 {
    return None; // Safety limit
  }

  let mut coeffs = vec![0i128; max_index + 1];
  for (idx, c) in term_data {
    coeffs[idx] += c;
  }

  Some(coeffs)
}

/// Extract integer coefficient and exponent vector from a multivariate term.
/// Returns (coefficient, [exp_var0, exp_var1, ...]) or None if not a valid polynomial term.
fn extract_multivar_term_data(
  term: &Expr,
  sorted_vars: &[String],
) -> Option<(i128, Vec<i128>)> {
  let mut exponents = vec![0i128; sorted_vars.len()];

  // Decompose term using the existing infrastructure
  let (num_coeff, _key, var_factors) = decompose_term(term);

  // Extract numeric coefficient
  let coeff = match &simplify(num_coeff) {
    Expr::Integer(n) => *n,
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      if let Expr::Integer(n) = operand.as_ref() {
        -n
      } else {
        return None;
      }
    }
    _ => return None,
  };

  // Extract exponents from variable factors
  for factor in &var_factors {
    match factor {
      Expr::Identifier(name) => {
        if let Some(pos) = sorted_vars.iter().position(|v| v == name) {
          exponents[pos] += 1;
        } else {
          return None; // Unknown variable
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => {
        if let Expr::Identifier(name) = left.as_ref() {
          if let Expr::Integer(exp) = right.as_ref() {
            if let Some(pos) = sorted_vars.iter().position(|v| v == name) {
              exponents[pos] += exp;
            } else {
              return None;
            }
          } else {
            return None;
          }
        } else {
          return None;
        }
      }
      _ => return None,
    }
  }

  Some((coeff, exponents))
}

/// Reverse Kronecker substitution: convert a univariate coefficient vector
/// back to a multivariate expression.
/// coeffs[k] maps to: coeff * primary^(k mod D) * var1^((k/D) mod D) * var2^((k/D^2) mod D) * ...
fn reverse_kronecker_coeffs(
  coeffs: &[i128],
  sorted_vars: &[String],
  d: usize,
) -> Expr {
  let primary = &sorted_vars[0];
  let secondary_vars = &sorted_vars[1..];
  let mut terms: Vec<Expr> = Vec::new();

  for (k, &c) in coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let mut remaining = k;
    let mut factors: Vec<Expr> = Vec::new();

    // Primary variable exponent
    let primary_exp = remaining % d;
    remaining /= d;
    if primary_exp > 0 {
      if primary_exp == 1 {
        factors.push(Expr::Identifier(primary.clone()));
      } else {
        factors.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Identifier(primary.clone())),
          right: Box::new(Expr::Integer(primary_exp as i128)),
        });
      }
    }

    // Secondary variable exponents
    for sec_var in secondary_vars {
      let exp = remaining % d;
      remaining /= d;
      if exp > 0 {
        if exp == 1 {
          factors.push(Expr::Identifier(sec_var.clone()));
        } else {
          factors.push(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::Identifier(sec_var.clone())),
            right: Box::new(Expr::Integer(exp as i128)),
          });
        }
      }
    }

    // Build the term: c * factors
    let term = if factors.is_empty() {
      Expr::Integer(c)
    } else {
      let var_part = build_product(factors);
      if c == 1 {
        var_part
      } else if c == -1 {
        negate_term(&var_part)
      } else {
        multiply_exprs(&Expr::Integer(c), &var_part)
      }
    };
    terms.push(term);
  }

  if terms.is_empty() {
    Expr::Integer(0)
  } else {
    let raw = build_sum(terms);
    combine_and_build(collect_additive_terms(&raw))
  }
}

/// Group identical factor coefficient vectors and count multiplicities.
fn group_factors(factors: &[Vec<i128>]) -> Vec<(Vec<i128>, usize)> {
  let mut grouped: Vec<(Vec<i128>, usize)> = Vec::new();
  for f in factors {
    if let Some(entry) = grouped.iter_mut().find(|(c, _)| c == f) {
      entry.1 += 1;
    } else {
      grouped.push((f.clone(), 1));
    }
  }
  grouped
}

/// Try to recombine univariate factors into multivariate factors.
/// Returns Some(vec of (factor_expr, multiplicity)) or None.
fn recombine_factors(
  full_coeffs: &[i128],
  grouped: &[(Vec<i128>, usize)],
  sorted_vars: &[String],
  d: usize,
) -> Option<Vec<(Expr, usize)>> {
  let n = grouped.len();

  // Build the original multivariate expression for comparison
  let original = reverse_kronecker_coeffs(full_coeffs, sorted_vars, d);
  let original_expanded = expand_and_combine(&original);

  // For single distinct factor with multiplicity > 1, just reverse-substitute it
  if n == 1 {
    let (ref fc, mult) = grouped[0];
    let candidate = reverse_kronecker_coeffs(fc, sorted_vars, d);
    // Verify: candidate^mult should equal original
    let candidate_expanded = expand_and_combine(&candidate);
    let powered = if mult == 1 {
      candidate_expanded.clone()
    } else {
      expand_and_combine(&Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(candidate_expanded.clone()),
        right: Box::new(Expr::Integer(mult as i128)),
      })
    };
    if exprs_equal(&powered, &original_expanded) {
      return Some(vec![(candidate, mult)]);
    }
    return None;
  }

  // Try all non-trivial subsets of the factor instances.
  // We represent a partition as: for each grouped factor (coeffs, mult),
  // how many copies go to the "left" side (0..=mult).
  // We enumerate assignments and check if the partition works.
  let max_assignments: Vec<usize> = grouped.iter().map(|(_, m)| *m).collect();

  // Total number of factor instances
  let total_instances: usize = max_assignments.iter().sum();
  if total_instances > 20 {
    return None; // Too many to enumerate
  }

  // Enumerate all assignments (for each group, 0..=mult copies to left side)
  let mut assignment = vec![0usize; n];

  loop {
    // Skip trivial partitions (all-left or all-right)
    let left_count: usize = assignment.iter().sum();
    if left_count > 0
      && left_count < total_instances
      && left_count <= total_instances / 2
    {
      // Compute left product (coefficient vector)
      let left_coeffs = compute_subset_product(grouped, &assignment);
      // Compute right = full / left
      if let Some((right_coeffs, rem)) = poly_div(full_coeffs, &left_coeffs)
        && rem.iter().all(|&c| c == 0)
      {
        // Reverse-substitute both sides
        let left_expr = reverse_kronecker_coeffs(&left_coeffs, sorted_vars, d);
        let right_expr =
          reverse_kronecker_coeffs(&right_coeffs, sorted_vars, d);

        // Verify: expand(left * right) == original
        let product = expand_and_combine(&Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(left_expr.clone()),
          right: Box::new(right_expr.clone()),
        });

        if exprs_equal(&product, &original_expanded) {
          // Valid partition! Now recursively try to factor each side.
          let mut result = Vec::new();
          let left_sub = try_refactor_multivar(&left_expr, sorted_vars, d);
          let right_sub = try_refactor_multivar(&right_expr, sorted_vars, d);
          result.extend(left_sub);
          result.extend(right_sub);
          return Some(result);
        }
      }
    }

    // Increment assignment (odometer-style)
    if !increment_assignment(&mut assignment, &max_assignments) {
      break;
    }
  }

  None
}

/// Try to further factor a multivariate expression that came from recombination.
/// Returns a list of (factor, multiplicity) pairs.
fn try_refactor_multivar(
  expr: &Expr,
  sorted_vars: &[String],
  d: usize,
) -> Vec<(Expr, usize)> {
  let expanded = expand_and_combine(expr);

  // Use direct coefficient computation (not symbolic substitution)
  if let Some(coeffs) = multivar_to_univar_coeffs(&expanded, sorted_vars, d) {
    let coeff_gcd = coeffs
      .iter()
      .copied()
      .filter(|&c| c != 0)
      .fold(0i128, gcd_i128);
    if coeff_gcd == 0 {
      return vec![(expanded, 1)];
    }
    let reduced: Vec<i128> = coeffs.iter().map(|c| c / coeff_gcd).collect();
    let (reduced, _) = if reduced.last().map(|&c| c < 0).unwrap_or(false) {
      (reduced.iter().map(|c| -c).collect::<Vec<_>>(), -1i128)
    } else {
      (reduced, 1i128)
    };

    let primary = &sorted_vars[0];
    let factor_exprs = factor_integer_poly(&reduced, primary);
    if factor_exprs.len() > 1 {
      let mut fc: Vec<Vec<i128>> = Vec::new();
      for f in &factor_exprs {
        if let Some(c) = extract_poly_coeffs(f, primary) {
          fc.push(c);
        } else {
          return vec![(expanded, 1)];
        }
      }
      let grouped = group_factors(&fc);
      if let Some(sub_factors) =
        recombine_factors(&reduced, &grouped, sorted_vars, d)
      {
        return sub_factors;
      }
    }
  }

  vec![(expanded, 1)]
}

/// Compute the product of a subset of grouped factors.
/// assignment[i] = how many copies of grouped[i] to include.
fn compute_subset_product(
  grouped: &[(Vec<i128>, usize)],
  assignment: &[usize],
) -> Vec<i128> {
  let mut product = vec![1i128]; // Start with constant 1
  for (i, &count) in assignment.iter().enumerate() {
    let fc = &grouped[i].0;
    for _ in 0..count {
      product = poly_mul_coeffs(&product, fc);
    }
  }
  product
}

/// Multiply two coefficient vectors (polynomial multiplication).
fn poly_mul_coeffs(a: &[i128], b: &[i128]) -> Vec<i128> {
  if a.is_empty() || b.is_empty() {
    return vec![];
  }
  let mut result = vec![0i128; a.len() + b.len() - 1];
  for (i, &ai) in a.iter().enumerate() {
    for (j, &bj) in b.iter().enumerate() {
      result[i + j] += ai * bj;
    }
  }
  result
}

/// Increment an assignment vector (odometer-style).
/// Returns false if we've exhausted all assignments.
fn increment_assignment(assignment: &mut [usize], max_vals: &[usize]) -> bool {
  for i in 0..assignment.len() {
    if assignment[i] < max_vals[i] {
      assignment[i] += 1;
      return true;
    }
    assignment[i] = 0;
  }
  false
}

/// Check if two expanded expressions are equal (by string comparison after normalization).
fn exprs_equal(a: &Expr, b: &Expr) -> bool {
  expr_to_string(a) == expr_to_string(b)
}

/// Extract the GCD of all integer coefficients in a multivariate polynomial.
/// Extract common monomial factor from all terms.
/// Returns (monomial_factors, remaining_polynomial).
/// E.g., for x^2*y + x*y^2, returns ([(x,1), (y,1)], x + y).
fn extract_common_monomial(
  expr: &Expr,
  sorted_vars: &[String],
) -> (Vec<(Expr, usize)>, Expr) {
  let terms = collect_additive_terms(expr);
  if terms.is_empty() {
    return (vec![], expr.clone());
  }

  // Find minimum exponent of each variable across all terms
  let mut min_exps = vec![i128::MAX; sorted_vars.len()];
  let mut valid = true;

  for term in &terms {
    match extract_multivar_term_data(term, sorted_vars) {
      Some((_coeff, exponents)) => {
        for (i, &exp) in exponents.iter().enumerate() {
          min_exps[i] = min_exps[i].min(exp);
        }
      }
      None => {
        valid = false;
        break;
      }
    }
  }

  if !valid {
    return (vec![], expr.clone());
  }

  // Build monomial factors and divide each term
  let mut monomial_factors: Vec<(Expr, usize)> = Vec::new();
  for (i, &min_exp) in min_exps.iter().enumerate() {
    if min_exp > 0 {
      monomial_factors
        .push((Expr::Identifier(sorted_vars[i].clone()), min_exp as usize));
    }
  }

  if monomial_factors.is_empty() {
    return (vec![], expr.clone());
  }

  // Rebuild each term with reduced exponents
  let mut new_terms: Vec<Expr> = Vec::new();
  for term in &terms {
    if let Some((coeff, exponents)) =
      extract_multivar_term_data(term, sorted_vars)
    {
      let mut factors: Vec<Expr> = Vec::new();
      if coeff != 1 {
        factors.push(Expr::Integer(coeff));
      }
      for (i, &exp) in exponents.iter().enumerate() {
        let reduced_exp = exp - min_exps[i];
        if reduced_exp > 0 {
          if reduced_exp == 1 {
            factors.push(Expr::Identifier(sorted_vars[i].clone()));
          } else {
            factors.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(Expr::Identifier(sorted_vars[i].clone())),
              right: Box::new(Expr::Integer(reduced_exp)),
            });
          }
        }
      }
      if factors.is_empty() {
        new_terms.push(Expr::Integer(coeff));
      } else {
        new_terms.push(build_product(factors));
      }
    }
  }

  let remaining = if new_terms.is_empty() {
    Expr::Integer(0)
  } else {
    combine_and_build(new_terms)
  };

  (monomial_factors, remaining)
}

fn extract_multivar_gcd(expr: &Expr) -> i128 {
  let terms = collect_additive_terms(expr);
  let mut gcd = 0i128;
  for term in &terms {
    let c = term_integer_coeff(term);
    gcd = gcd_i128(gcd, c);
  }
  if gcd == 0 { 1 } else { gcd.abs() }
}

/// Extract the integer coefficient from a multivariate term.
fn term_integer_coeff(term: &Expr) -> i128 {
  match term {
    Expr::Integer(n) => *n,
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => -term_integer_coeff(operand),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    } => {
      let factors = collect_multiplicative_factors(term);
      let mut coeff = 1i128;
      for f in &factors {
        match f {
          Expr::Integer(n) => coeff *= n,
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } => {
            coeff = -coeff;
            if let Expr::Integer(n) = operand.as_ref() {
              coeff *= n;
            }
          }
          _ => {}
        }
      }
      coeff
    }
    _ => 1, // variable or power term has implicit coefficient 1
  }
}

/// Divide all terms of a polynomial expression by a scalar.
fn divide_expr_by_scalar(expr: &Expr, scalar: i128) -> Expr {
  if scalar == 1 {
    return expr.clone();
  }
  let terms = collect_additive_terms(expr);
  let mut new_terms: Vec<Expr> = Vec::new();
  for term in &terms {
    new_terms.push(divide_term_by_scalar(term, scalar));
  }
  if new_terms.is_empty() {
    Expr::Integer(0)
  } else {
    combine_and_build(new_terms)
  }
}

/// Divide a single term by a scalar.
fn divide_term_by_scalar(term: &Expr, scalar: i128) -> Expr {
  match term {
    Expr::Integer(n) => Expr::Integer(n / scalar),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let inner = divide_term_by_scalar(operand, scalar);
      negate_term(&inner)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    } => {
      let factors = collect_multiplicative_factors(term);
      let mut new_factors: Vec<Expr> = Vec::new();
      let mut divided = false;
      for f in &factors {
        if !divided {
          if let Expr::Integer(n) = f {
            new_factors.push(Expr::Integer(n / scalar));
            divided = true;
            continue;
          }
          if let Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } = f
            && let Expr::Integer(n) = operand.as_ref()
          {
            let val = -n;
            new_factors.push(Expr::Integer(val / scalar));
            divided = true;
            continue;
          }
        }
        new_factors.push(f.clone());
      }
      if !divided {
        // No integer factor found; the coefficient is implicit 1
        // This shouldn't happen for expanded polynomials with GCD > 1
        return term.clone();
      }
      // Remove factors that are 1
      new_factors.retain(|f| !matches!(f, Expr::Integer(1)));
      if new_factors.is_empty() {
        Expr::Integer(1)
      } else {
        build_product(new_factors)
      }
    }
    _ => {
      // Implicit coefficient 1; 1/scalar = 1 only if scalar=1
      term.clone()
    }
  }
}

/// Build the final factored result from overall coefficient and factor list.
fn build_multivariate_result(
  overall: i128,
  factors: Vec<(Expr, usize)>,
) -> Expr {
  // First, group identical factors by string representation
  let mut grouped: Vec<(Expr, usize)> = Vec::new();
  for (factor, mult) in &factors {
    let f = combine_and_build(collect_additive_terms(factor));
    let key = expr_to_string(&f);
    if let Some(entry) =
      grouped.iter_mut().find(|(e, _)| expr_to_string(e) == key)
    {
      entry.1 += mult;
    } else {
      grouped.push((f, *mult));
    }
  }

  let mut result_factors: Vec<Expr> = Vec::new();

  // If overall is negative, absorb -1 into one of the factors with odd multiplicity
  let mut remaining_overall = overall;
  if remaining_overall < 0 {
    // Find the best factor to negate: prefer one whose first variable term
    // has a negative coefficient (so negation makes the expression look more canonical)
    let mut best_idx: Option<usize> = None;
    for (i, (factor, mult)) in grouped.iter().enumerate() {
      if *mult % 2 == 1 {
        // Check if first variable term is negative
        let s = expr_to_string(factor);
        let starts_negative = s.starts_with('-');
        if starts_negative {
          best_idx = Some(i);
          break;
        }
        if best_idx.is_none() {
          best_idx = Some(i);
        }
      }
    }
    if let Some(idx) = best_idx {
      let terms = collect_additive_terms(&grouped[idx].0);
      let negated: Vec<Expr> = terms.iter().map(negate_term).collect();
      grouped[idx].0 = combine_and_build(negated);
      remaining_overall = -remaining_overall;
    }
  }

  if remaining_overall != 1 {
    result_factors.push(Expr::Integer(remaining_overall));
  }

  for (factor, mult) in &grouped {
    if *mult == 1 {
      result_factors.push(factor.clone());
    } else {
      result_factors.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(factor.clone()),
        right: Box::new(Expr::Integer(*mult as i128)),
      });
    }
  }

  // Sort factors in canonical order:
  // 1. Pure numeric factors come first
  // 2. Then by constant term, degree, term count, coefficients
  result_factors.sort_by(|a, b| {
    let a_is_num = matches!(a, Expr::Integer(_));
    let b_is_num = matches!(b, Expr::Integer(_));
    match (a_is_num, b_is_num) {
      (true, false) => return std::cmp::Ordering::Less,
      (false, true) => return std::cmp::Ordering::Greater,
      _ => {}
    }
    let ca = factor_constant_term(a);
    let cb = factor_constant_term(b);
    ca.cmp(&cb)
      .then_with(|| factor_degree(a).cmp(&factor_degree(b)))
      .then_with(|| factor_term_count(a).cmp(&factor_term_count(b)))
      .then_with(|| {
        factor_first_nonconst_coeff(a).cmp(&factor_first_nonconst_coeff(b))
      })
      .then_with(|| {
        // For multivariate: sort by minimum term coefficient
        // so factors with negative terms sort before positive ones
        let min_a = collect_additive_terms(a)
          .iter()
          .map(term_integer_coeff)
          .min()
          .unwrap_or(0);
        let min_b = collect_additive_terms(b)
          .iter()
          .map(term_integer_coeff)
          .min()
          .unwrap_or(0);
        min_a.cmp(&min_b)
      })
      .then_with(|| expr_to_string(a).cmp(&expr_to_string(b)))
  });

  if result_factors.is_empty() {
    Expr::Integer(1)
  } else if result_factors.len() == 1 {
    result_factors.remove(0)
  } else {
    // Canonicalise the Times ordering so e.g. `Factor[x a == x b + x c]`
    // produces `(b + c)*x` rather than `x*(b + c)`. The factor-specific
    // sort above (by constant term/degree/term-count) is right for
    // grouping and sign placement but doesn't match Wolfram's display
    // order; running through `times_ast` applies the canonical Times
    // sort that Times itself uses.
    crate::functions::math_ast::times_ast(&result_factors)
      .unwrap_or_else(|_| build_product(result_factors))
  }
}
