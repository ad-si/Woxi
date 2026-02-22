#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

/// Plus[args...] - Sum of arguments, with list threading
pub fn plus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(0));
  }

  // Handle Quantity arithmetic before anything else
  if let Some(result) = crate::functions::quantity_ast::try_quantity_plus(args)
  {
    return result;
  }

  // Flatten nested Plus arguments
  let mut flat_args: Vec<Expr> = Vec::new();
  for arg in args {
    match arg {
      Expr::FunctionCall {
        name,
        args: inner_args,
      } if name == "Plus" => {
        flat_args.extend(inner_args.clone());
      }
      _ => flat_args.push(arg.clone()),
    }
  }

  // Check for Infinity + (-Infinity) → Indeterminate
  {
    let mut has_pos_inf = false;
    let mut has_neg_inf = false;
    for arg in &flat_args {
      match arg {
        Expr::Identifier(name) if name == "Infinity" => has_pos_inf = true,
        Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand,
        } if matches!(operand.as_ref(), Expr::Identifier(n) if n == "Infinity") => {
          has_neg_inf = true
        }
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left,
          right,
        } if matches!(left.as_ref(), Expr::Integer(-1))
          && matches!(right.as_ref(), Expr::Identifier(n) if n == "Infinity") =>
        {
          has_neg_inf = true
        }
        _ => {}
      }
    }
    if has_pos_inf && has_neg_inf {
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // Infinity + finite terms → Infinity, -Infinity + finite terms → -Infinity
    if has_pos_inf {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    if has_neg_inf {
      return Ok(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
  }

  // Check for list threading
  let has_list = flat_args.iter().any(|a| matches!(a, Expr::List(_)));
  if has_list {
    return thread_binary_over_lists(&flat_args, |a, b| {
      match (expr_to_num(a), expr_to_num(b)) {
        (Some(x), Some(y)) => Ok(num_to_expr(x + y)),
        _ => Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(a.clone()),
          right: Box::new(b.clone()),
        }),
      }
    });
  }

  // Check if any argument needs BigInt arithmetic (BigInteger or large Integer exceeding f64 precision)
  let has_bigint = flat_args.iter().any(needs_bigint_arithmetic);

  if has_bigint {
    use num_bigint::BigInt;
    let mut big_sum = BigInt::from(0);
    let mut all_int = true;
    let mut symbolic_args: Vec<Expr> = Vec::new();

    for arg in &flat_args {
      match arg {
        Expr::Integer(n) => big_sum += BigInt::from(*n),
        Expr::BigInteger(n) => big_sum += n,
        _ => {
          all_int = false;
          symbolic_args.push(arg.clone());
        }
      }
    }

    if all_int {
      return Ok(bigint_to_expr(big_sum));
    }

    let mut final_args: Vec<Expr> = Vec::new();
    if big_sum != BigInt::from(0) {
      final_args.push(bigint_to_expr(big_sum));
    }
    final_args.extend(symbolic_args);
    if final_args.len() == 1 {
      return Ok(final_args.remove(0));
    }
    return Ok(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: final_args,
    });
  }

  // Classify arguments: exact (Integer/Rational), real (Real), or symbolic
  let mut has_real = false;
  let mut all_numeric = true;
  for arg in &flat_args {
    match arg {
      Expr::Real(_) => has_real = true,
      Expr::Integer(_) => {}
      Expr::FunctionCall { name, args: rargs }
        if name == "Rational"
          && rargs.len() == 2
          && matches!(rargs[0], Expr::Integer(_))
          && matches!(rargs[1], Expr::Integer(_)) => {}
      _ => {
        all_numeric = false;
      }
    }
  }

  // If all numeric and no Reals, use exact rational arithmetic
  if all_numeric && !has_real {
    // Sum as exact rational: (numer, denom)
    let mut sum_n: i128 = 0;
    let mut sum_d: i128 = 1;
    for arg in &flat_args {
      if let Some((n, d)) = expr_to_rational(arg) {
        // sum_n/sum_d + n/d = (sum_n*d + n*sum_d) / (sum_d*d)
        sum_n = sum_n * d + n * sum_d;
        sum_d *= d;
        let g = gcd(sum_n, sum_d);
        sum_n /= g;
        sum_d /= g;
        // Keep denom positive
        if sum_d < 0 {
          sum_n = -sum_n;
          sum_d = -sum_d;
        }
      }
    }
    return Ok(make_rational(sum_n, sum_d));
  }

  // If all numeric but has Reals, use f64
  if all_numeric {
    let mut sum = 0.0;
    for arg in &flat_args {
      if let Some(n) = expr_to_num(arg) {
        sum += n;
      }
    }
    return Ok(Expr::Real(sum));
  }

  {
    // Separate numeric and symbolic terms
    let mut symbolic_args: Vec<Expr> = Vec::new();
    let mut has_exact = false;
    let mut sum_n: i128 = 0;
    let mut sum_d: i128 = 1;
    let mut real_sum: f64 = 0.0;
    let mut has_real_term = false;

    for arg in &flat_args {
      if let Some((n, d)) = expr_to_rational(arg) {
        sum_n = sum_n * d + n * sum_d;
        sum_d *= d;
        let g = gcd(sum_n, sum_d);
        sum_n /= g;
        sum_d /= g;
        if sum_d < 0 {
          sum_n = -sum_n;
          sum_d = -sum_d;
        }
        has_exact = true;
      } else if let Expr::Real(f) = arg {
        real_sum += f;
        has_real_term = true;
      } else {
        symbolic_args.push(arg.clone());
      }
    }

    // Build final args: numeric sum first (if non-zero), then symbolic terms sorted
    let mut final_args: Vec<Expr> = Vec::new();

    // If we have both exact and real, convert exact to f64 and combine
    if has_exact && has_real_term {
      let total = (sum_n as f64) / (sum_d as f64) + real_sum;
      if total != 0.0 {
        final_args.push(Expr::Real(total));
      }
    } else if has_real_term && real_sum != 0.0 {
      final_args.push(Expr::Real(real_sum));
    } else if has_exact && sum_n != 0 {
      final_args.push(make_rational(sum_n, sum_d));
    }

    // Collect like terms: group symbolic terms by their base expression
    // e.g. E + E → 2*E, 3*x + 2*x → 5*x
    let collected = collect_like_terms(&symbolic_args);

    // Sort symbolic terms: polynomial-like terms first, then transcendental functions
    // This gives Mathematica-like ordering where x^2 comes before Sin[x].
    // For alphabetical comparison, strip the leading "-" from negated terms
    // so that -x sorts next to x rather than before everything.
    let mut sorted_symbolic = collected;
    sorted_symbolic.sort_by(|a, b| {
      let pa = term_priority(a);
      let pb = term_priority(b);
      if pa != pb {
        pa.cmp(&pb)
      } else {
        let sa = term_sort_key(a);
        let sb = term_sort_key(b);
        sa.cmp(&sb)
      }
    });
    final_args.extend(sorted_symbolic);

    if final_args.is_empty() {
      Ok(Expr::Integer(0))
    } else if final_args.len() == 1 {
      Ok(final_args.remove(0))
    } else {
      Ok(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: final_args,
      })
    }
  }
}

/// Coefficient: either exact rational or approximate real
#[derive(Clone)]
pub enum Coeff {
  Exact(i128, i128), // (numer, denom)
  Real(f64),
}

impl Coeff {
  fn is_zero(&self) -> bool {
    match self {
      Coeff::Exact(n, _) => *n == 0,
      Coeff::Real(f) => *f == 0.0,
    }
  }
  fn is_one(&self) -> bool {
    match self {
      Coeff::Exact(n, d) => *n == 1 && *d == 1,
      Coeff::Real(f) => *f == 1.0,
    }
  }
  fn to_f64(&self) -> f64 {
    match self {
      Coeff::Exact(n, d) => *n as f64 / *d as f64,
      Coeff::Real(f) => *f,
    }
  }
  fn add(&self, other: &Coeff) -> Coeff {
    match (self, other) {
      (Coeff::Exact(n1, d1), Coeff::Exact(n2, d2)) => {
        let mut sn = n1 * d2 + n2 * d1;
        let mut sd = d1 * d2;
        let g = gcd(sn, sd);
        sn /= g;
        sd /= g;
        if sd < 0 {
          sn = -sn;
          sd = -sd;
        }
        Coeff::Exact(sn, sd)
      }
      _ => Coeff::Real(self.to_f64() + other.to_f64()),
    }
  }
  fn negate(&self) -> Coeff {
    match self {
      Coeff::Exact(n, d) => Coeff::Exact(-n, *d),
      Coeff::Real(f) => Coeff::Real(-f),
    }
  }
  fn to_expr(&self) -> Expr {
    match self {
      Coeff::Exact(n, d) => make_rational(*n, *d),
      Coeff::Real(f) => Expr::Real(*f),
    }
  }
}

/// Decompose a term into (coefficient, base_expression).
/// E.g. `3*x` → (Exact(3,1), x), `x` → (Exact(1,1), x), `-x` → (Exact(-1,1), x),
/// `1.5*x` → (Real(1.5), x), `Rational[3,4]*x` → (Exact(3,4), x).
pub fn decompose_term(e: &Expr) -> (Coeff, Expr) {
  match e {
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      // Check if first arg is a numeric coefficient (integer/rational)
      if let Some((n, d)) = expr_to_rational(&args[0]) {
        let base = if args.len() == 2 {
          args[1].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: args[1..].to_vec(),
          }
        };
        return (Coeff::Exact(n, d), base);
      }
      // Check if first arg is a Real coefficient
      if let Expr::Real(f) = &args[0] {
        let base = if args.len() == 2 {
          args[1].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: args[1..].to_vec(),
          }
        };
        return (Coeff::Real(*f), base);
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if let Some((n, d)) = expr_to_rational(left) {
        return (Coeff::Exact(n, d), *right.clone());
      }
      if let Expr::Real(f) = left.as_ref() {
        return (Coeff::Real(*f), *right.clone());
      }
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let (c, base) = decompose_term(operand);
      return (c.negate(), base);
    }
    _ => {}
  }
  (Coeff::Exact(1, 1), e.clone())
}

/// Collect like terms: group symbolic terms by their base expression
/// and sum their coefficients. E.g. [E, E] → [2*E], [3*x, 2*x] → [5*x].
pub fn collect_like_terms(terms: &[Expr]) -> Vec<Expr> {
  use std::collections::BTreeMap;

  // Group by string representation of base → sum of coefficients
  let mut groups: Vec<(String, Expr, Coeff)> = Vec::new();
  let mut index: BTreeMap<String, usize> = BTreeMap::new();

  for term in terms {
    let (c, base) = decompose_term(term);
    let key = crate::syntax::expr_to_string(&base);
    if let Some(&idx) = index.get(&key) {
      let (_, _, ref mut sum_c) = groups[idx];
      *sum_c = sum_c.add(&c);
    } else {
      index.insert(key.clone(), groups.len());
      groups.push((key, base, c));
    }
  }

  let mut result = Vec::new();
  for (_, base, c) in groups {
    if c.is_zero() {
      continue; // terms cancelled
    }
    if c.is_one() {
      result.push(base);
    } else {
      // Reconstruct as flat Times[coefficient, base_args...] to preserve formatting
      let coeff = c.to_expr();
      let mut times_args = vec![coeff];
      // Flatten base if it's already a Times
      match &base {
        Expr::FunctionCall { name, args: bargs } if name == "Times" => {
          times_args.extend(bargs.clone());
        }
        _ => {
          times_args.push(base);
        }
      }
      result.push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: times_args,
      });
    }
  }
  result
}

/// Extract sort key for a Plus term.
/// For Divide(n, x) where n is an integer, sort by the denominator (x)
/// so that 1/z sorts near z-related terms rather than before all variables.
fn term_sort_key(e: &Expr) -> String {
  // For integer/denominator divisions, sort by denominator
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left,
    right,
  } = e
  {
    if matches!(left.as_ref(), Expr::Integer(_)) {
      return crate::syntax::expr_to_string(right);
    }
  }
  let s = crate::syntax::expr_to_string(e);
  s.strip_prefix('-').unwrap_or(&s).to_string()
}

/// Sort symbolic factors in Times using the same ordering as Wolfram:
/// polynomial-like terms first (variables, powers), then transcendental functions,
/// with alphabetical ordering within each group.
/// Compute term priority for sorting: 0 = polynomial-like, 1 = transcendental.
pub fn term_priority(e: &Expr) -> i32 {
  match e {
    Expr::Identifier(_) => 0,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      ..
    } => 0,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      ..
    } => term_priority(left),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => term_priority(left).max(term_priority(right)),
    Expr::FunctionCall { name, .. } => match name.as_str() {
      "Times" | "Power" | "Plus" | "Rational" => 0,
      "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc" | "Sinh" | "Cosh"
      | "Tanh" | "Coth" | "Sech" | "Csch" | "ArcSin" | "ArcCos" | "ArcTan"
      | "ArcCot" | "ArcSec" | "ArcCsc" | "Exp" | "Log" | "Factorial"
      | "Erf" | "Erfc" => 1,
      _ => 0,
    },
    Expr::UnaryOp { operand, .. } => term_priority(operand),
    _ => 0,
  }
}

/// Sub-priority for Times factor ordering: identifiers before compound expressions.
/// This ensures simple symbols sort before sums/products, matching Wolfram behavior.
pub fn times_factor_subpriority(e: &Expr) -> i32 {
  match e {
    Expr::Identifier(_) | Expr::Constant(_) => 0,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      ..
    } => times_factor_subpriority(left),
    Expr::BinaryOp {
      op:
        crate::syntax::BinaryOperator::Plus | crate::syntax::BinaryOperator::Minus,
      ..
    } => 1,
    Expr::FunctionCall { name, .. } => match name.as_str() {
      "Times" | "Power" | "Rational" => 0,
      "Plus" => 1,
      _ => 2,
    },
    _ => 0,
  }
}

pub fn sort_symbolic_factors(symbolic_args: &mut [Expr]) {
  symbolic_args.sort_by(|a, b| {
    let pa = term_priority(a);
    let pb = term_priority(b);
    if pa != pb {
      return pa.cmp(&pb);
    }
    // Within same priority, identifiers before function calls
    let sa = times_factor_subpriority(a);
    let sb = times_factor_subpriority(b);
    if sa != sb {
      return sa.cmp(&sb);
    }
    crate::syntax::expr_to_string(a).cmp(&crate::syntax::expr_to_string(b))
  });
}

/// Extract (base, exponent) from an expression for power combining in Times.
/// x → (x, 1), x^n → (x, n), Sqrt[x] → (x, 1/2)
pub fn extract_base_exponent(expr: &Expr) -> (Expr, Expr) {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      (args[0].clone(), make_rational(1, 2))
    }
    _ => (expr.clone(), Expr::Integer(1)),
  }
}

/// Combine like bases in a list of symbolic factors: x^a * x^b → x^(a+b)
pub fn combine_like_bases(
  args: Vec<Expr>,
) -> Result<Vec<Expr>, InterpreterError> {
  if args.len() <= 1 {
    return Ok(args);
  }

  // Use string representation of base as grouping key
  let mut groups: Vec<(String, Expr, Vec<Expr>)> = Vec::new(); // (base_key, base, exponents)
  let mut non_combinable: Vec<Expr> = Vec::new();

  for arg in &args {
    // Don't combine Plus, Times, or complex expressions - only identifiers, constants, and powers thereof
    let (base, exp) = extract_base_exponent(arg);
    let combinable = match &base {
      Expr::Identifier(_) | Expr::Constant(_) => true,
      Expr::FunctionCall { name, .. } => {
        matches!(name.as_str(), "Sqrt" | "Log" | "Sin" | "Cos")
      }
      _ => false,
    };
    if !combinable {
      non_combinable.push(arg.clone());
      continue;
    }
    let base_key = crate::syntax::expr_to_string(&base);
    if let Some(group) = groups.iter_mut().find(|(k, _, _)| *k == base_key) {
      group.2.push(exp);
    } else {
      groups.push((base_key, base, vec![exp]));
    }
  }

  let mut result: Vec<Expr> = Vec::new();
  for (_key, base, exponents) in groups {
    if exponents.len() == 1 {
      // Single occurrence — no combining needed, reconstruct original form
      if matches!(&exponents[0], Expr::Integer(1)) {
        result.push(base);
      } else {
        result.push(power_two(&base, &exponents[0])?);
      }
    } else {
      // Multiple occurrences — add exponents
      let combined_exp = plus_ast(&exponents)?;
      if matches!(&combined_exp, Expr::Integer(0)) {
        // x^0 = 1, skip (will be absorbed into coefficient)
        continue;
      }
      result.push(power_two(&base, &combined_exp)?);
    }
  }
  result.extend(non_combinable);

  // Second pass: combine numeric bases with the same fractional exponent
  // e.g. Sqrt[2] * Sqrt[3] = 2^(1/2) * 3^(1/2) → 6^(1/2) = Sqrt[6]
  let mut combined: Vec<Expr> = Vec::new();
  let mut used = vec![false; result.len()];
  for i in 0..result.len() {
    if used[i] {
      continue;
    }
    let (base_i, exp_i) = extract_base_exponent(&result[i]);
    // Only combine integer bases with rational exponents
    let is_numeric_base =
      matches!(&base_i, Expr::Integer(_) | Expr::BigInteger(_));
    let is_rational_exp = matches!(
      &exp_i,
      Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2
    );
    if !is_numeric_base || !is_rational_exp {
      combined.push(result[i].clone());
      continue;
    }
    let exp_key = crate::syntax::expr_to_string(&exp_i);
    let mut bases_to_multiply = vec![base_i];
    for j in (i + 1)..result.len() {
      if used[j] {
        continue;
      }
      let (base_j, exp_j) = extract_base_exponent(&result[j]);
      if crate::syntax::expr_to_string(&exp_j) == exp_key
        && matches!(&base_j, Expr::Integer(_) | Expr::BigInteger(_))
      {
        bases_to_multiply.push(base_j);
        used[j] = true;
      }
    }
    if bases_to_multiply.len() == 1 {
      combined.push(result[i].clone());
    } else {
      let product = times_ast(&bases_to_multiply)?;
      combined.push(power_two(&product, &exp_i)?);
    }
  }

  Ok(combined)
}

/// Times[args...] - Product of arguments, with list threading
pub fn times_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }

  // Handle Quantity arithmetic before anything else
  if let Some(result) = crate::functions::quantity_ast::try_quantity_times(args)
  {
    return result;
  }

  // Flatten nested Times arguments
  let mut flat_args: Vec<Expr> = Vec::new();
  for arg in args {
    match arg {
      Expr::FunctionCall {
        name,
        args: inner_args,
      } if name == "Times" => {
        flat_args.extend(inner_args.clone());
      }
      _ => flat_args.push(arg.clone()),
    }
  }
  let args = &flat_args;

  // Try complex multiplication: if all args can be extracted as exact complex
  // numbers (and at least one has nonzero imaginary part), multiply them.
  if args.len() >= 2 {
    let complex_parts: Vec<_> =
      args.iter().map(try_extract_complex_exact).collect();
    if complex_parts.iter().all(|c| c.is_some()) {
      let has_imaginary = complex_parts.iter().any(|c| {
        if let Some((_, (im, _))) = c {
          *im != 0
        } else {
          false
        }
      });
      if has_imaginary {
        // Multiply all complex parts
        let mut re_n: i128 = complex_parts[0].unwrap().0.0;
        let mut re_d: i128 = complex_parts[0].unwrap().0.1;
        let mut im_n: i128 = complex_parts[0].unwrap().1.0;
        let mut im_d: i128 = complex_parts[0].unwrap().1.1;
        let mut ok = true;
        for cp in &complex_parts[1..] {
          let ((cn, cd), (dn, dd)) = cp.unwrap();
          // (re + im*i) * (cn/cd + dn/dd*i)
          let new_re = (|| {
            let a = re_n.checked_mul(cn)?.checked_mul(im_d)?.checked_mul(dd)?;
            let b = im_n.checked_mul(dn)?.checked_mul(re_d)?.checked_mul(cd)?;
            let num = a.checked_sub(b)?;
            let den =
              re_d.checked_mul(cd)?.checked_mul(im_d)?.checked_mul(dd)?;
            Some((num, den))
          })();
          let new_im = (|| {
            let a = re_n.checked_mul(dn)?.checked_mul(im_d)?.checked_mul(cd)?;
            let b = im_n.checked_mul(cn)?.checked_mul(re_d)?.checked_mul(dd)?;
            let num = a.checked_add(b)?;
            let den =
              re_d.checked_mul(dd)?.checked_mul(im_d)?.checked_mul(cd)?;
            Some((num, den))
          })();
          if let (Some((rn, rd)), Some((in_, id))) = (new_re, new_im) {
            re_n = rn;
            re_d = rd;
            im_n = in_;
            im_d = id;
          } else {
            ok = false;
            break;
          }
        }
        if ok {
          return Ok(complex_rational_to_expr(re_n, re_d, im_n, im_d));
        }
      }
    }
  }

  // Check for list threading
  let has_list = args.iter().any(|a| matches!(a, Expr::List(_)));
  if has_list {
    return thread_binary_over_lists(args, |a, b| {
      match (expr_to_num(a), expr_to_num(b)) {
        (Some(x), Some(y)) => Ok(num_to_expr(x * y)),
        _ => Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(a.clone()),
          right: Box::new(b.clone()),
        }),
      }
    });
  }

  // Check if any argument needs BigInt arithmetic (BigInteger or large Integer exceeding f64 precision)
  let has_bigint = args.iter().any(needs_bigint_arithmetic);

  if has_bigint {
    use num_bigint::BigInt;
    let mut big_product = BigInt::from(1);
    let mut all_int = true;
    let mut symbolic_args: Vec<Expr> = Vec::new();

    for arg in args {
      match arg {
        Expr::Integer(n) => big_product *= BigInt::from(*n),
        Expr::BigInteger(n) => big_product *= n,
        _ => {
          all_int = false;
          symbolic_args.push(arg.clone());
        }
      }
    }

    if all_int {
      return Ok(bigint_to_expr(big_product));
    }

    // 0 * anything = 0
    if big_product == BigInt::from(0) {
      return Ok(Expr::Integer(0));
    }

    symbolic_args = combine_like_bases(symbolic_args)?;
    sort_symbolic_factors(&mut symbolic_args);
    let mut final_args: Vec<Expr> = Vec::new();
    if big_product != BigInt::from(1) {
      final_args.push(bigint_to_expr(big_product));
    }
    final_args.extend(symbolic_args);
    if final_args.len() == 1 {
      return Ok(final_args.remove(0));
    }
    return Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: final_args,
    });
  }

  // Separate into: integers, rationals, reals, and symbolic arguments
  let mut int_product: i128 = 1;
  let mut int_overflow = false;
  let mut has_int = false;
  let mut rat_numer: i128 = 1;
  let mut rat_denom: i128 = 1;
  let mut has_rational = false;
  let mut real_product: f64 = 1.0;
  let mut any_real = false;
  let mut symbolic_args: Vec<Expr> = Vec::new();

  for arg in args {
    match arg {
      Expr::Integer(n) => {
        if let Some(result) = int_product.checked_mul(*n) {
          int_product = result;
        } else {
          int_overflow = true;
        }
        has_int = true;
      }
      Expr::Real(f) => {
        real_product *= f;
        any_real = true;
      }
      Expr::FunctionCall { name, args: rargs }
        if name == "Rational" && rargs.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1]) {
          if let (Some(rn), Some(rd)) =
            (rat_numer.checked_mul(*n), rat_denom.checked_mul(*d))
          {
            rat_numer = rn;
            rat_denom = rd;
          } else {
            int_overflow = true;
          }
          has_rational = true;
        } else {
          symbolic_args.push(arg.clone());
        }
      }
      _ => {
        if let Some(n) = expr_to_num(arg) {
          real_product *= n;
          any_real = true;
        } else {
          symbolic_args.push(arg.clone());
        }
      }
    }
  }

  // If overflow detected, fall back to BigInt arithmetic
  if int_overflow {
    use num_bigint::BigInt;
    let mut big_product = BigInt::from(1);
    let mut sym_args: Vec<Expr> = Vec::new();
    for arg in args {
      match arg {
        Expr::Integer(n) => big_product *= BigInt::from(*n),
        Expr::BigInteger(n) => big_product *= n,
        _ => sym_args.push(arg.clone()),
      }
    }
    if sym_args.is_empty() {
      return Ok(bigint_to_expr(big_product));
    }
    if big_product == BigInt::from(0) {
      return Ok(Expr::Integer(0));
    }
    sym_args = combine_like_bases(sym_args)?;
    sort_symbolic_factors(&mut sym_args);
    let mut final_args: Vec<Expr> = Vec::new();
    if big_product != BigInt::from(1) {
      final_args.push(bigint_to_expr(big_product));
    }
    final_args.extend(sym_args);
    if final_args.len() == 1 {
      return Ok(final_args.remove(0));
    }
    return Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: final_args,
    });
  }

  // If any Real, try to convert symbolic constants (Pi, E, etc.) to floats
  if any_real {
    let mut remaining_symbolic: Vec<Expr> = Vec::new();
    for arg in symbolic_args.drain(..) {
      if let Some(f) = try_eval_to_f64(&arg) {
        real_product *= f;
      } else {
        remaining_symbolic.push(arg);
      }
    }
    symbolic_args = remaining_symbolic;

    let total = (int_product as f64)
      * (rat_numer as f64 / rat_denom as f64)
      * real_product;
    if symbolic_args.is_empty() {
      return Ok(Expr::Real(total));
    }
    if total == 0.0 {
      // Check if any remaining symbolic arg involves I (imaginary unit)
      let has_imag = symbolic_args
        .iter()
        .any(|a| matches!(a, Expr::Identifier(s) if s == "I"));
      if has_imag {
        // 0.0 * I → 0. + 0.*I (Complex form)
        return Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(Expr::Real(0.0)),
          right: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Real(0.0)),
            right: Box::new(Expr::Identifier("I".to_string())),
          }),
        });
      }
      // 0.0 * x → 0. (approximate zero, not exact)
      return Ok(Expr::Real(0.0));
    }
    symbolic_args = combine_like_bases(symbolic_args)?;
    sort_symbolic_factors(&mut symbolic_args);
    let mut final_args: Vec<Expr> = Vec::new();
    if total != 1.0 {
      final_args.push(Expr::Real(total));
    }
    final_args.extend(symbolic_args);
    return if final_args.len() == 1 {
      Ok(final_args.remove(0))
    } else {
      Ok(Expr::FunctionCall {
        name: "Times".to_string(),
        args: final_args,
      })
    };
  }

  // Exact arithmetic: combine integer * rational
  // Result is (int_product * rat_numer) / rat_denom
  let combined_numer = int_product * rat_numer;
  let combined_denom = rat_denom;
  let coeff = if has_rational || (has_int && combined_denom != 1) {
    make_rational(combined_numer, combined_denom)
  } else {
    Expr::Integer(int_product)
  };

  // If all arguments are numeric, return the product
  if symbolic_args.is_empty() {
    return Ok(coeff);
  }

  // 0 * anything = 0
  if combined_numer == 0 {
    return Ok(Expr::Integer(0));
  }

  // Handle Infinity in symbolic args: n * Infinity = ±Infinity
  if symbolic_args.len() == 1 {
    let is_pos_inf =
      matches!(&symbolic_args[0], Expr::Identifier(s) if s == "Infinity");
    let is_neg_inf = match &symbolic_args[0] {
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand,
      } => matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity"),
      _ => false,
    };
    if is_pos_inf || is_neg_inf {
      let coeff_positive = combined_numer > 0;
      // Positive * Infinity or Negative * (-Infinity) → Infinity
      // Negative * Infinity or Positive * (-Infinity) → -Infinity
      let result_positive = coeff_positive == is_pos_inf;
      if result_positive {
        return Ok(Expr::Identifier("Infinity".to_string()));
      } else {
        return Ok(Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(Expr::Identifier("Infinity".to_string())),
        });
      }
    }
  }

  // Times[-1, Plus[args...]] distributes: -1*(a+b+c) → (-a)+(-b)+(-c)
  // This only applies when coefficient is exactly -1 and the sole symbolic arg is Plus
  if matches!(&coeff, Expr::Integer(-1))
    && symbolic_args.len() == 1
    && matches!(&symbolic_args[0], Expr::FunctionCall { name, .. } if name == "Plus")
    && let Expr::FunctionCall {
      name: _,
      args: plus_args,
    } = &symbolic_args[0]
  {
    let negated: Result<Vec<Expr>, InterpreterError> = plus_args
      .iter()
      .map(|a| times_ast(&[Expr::Integer(-1), a.clone()]))
      .collect();
    return plus_ast(&negated?);
  }

  // Combine like bases: x^a * x^b → x^(a+b)
  symbolic_args = combine_like_bases(symbolic_args)?;

  // If all symbolic args canceled (e.g. x^2 * x^(-2)), return coefficient
  if symbolic_args.is_empty() {
    return Ok(coeff);
  }

  // Build final args: coefficient (if not 1) + sorted symbolic terms
  sort_symbolic_factors(&mut symbolic_args);
  let mut final_args: Vec<Expr> = Vec::new();
  let is_unit = matches!(&coeff, Expr::Integer(1));
  if !is_unit {
    final_args.push(coeff);
  }
  final_args.extend(symbolic_args);

  if final_args.len() == 1 {
    Ok(final_args.remove(0))
  } else {
    Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: final_args,
    })
  }
}

/// Minus[a] - Unary negation only
/// Note: Minus with 2 arguments is not valid in Wolfram Language
/// (use Subtract for that)
pub fn minus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Use times_ast for proper distribution: -(a+b) → -a - b
    times_ast(&[Expr::Integer(-1), args[0].clone()])
  } else {
    // Wrong arity - print error to stderr and return unevaluated expression
    eprintln!();
    eprintln!(
      "Minus::argx: Minus called with {} arguments; 1 argument is expected.",
      args.len()
    );
    // Return unevaluated (like Wolfram) — expr_to_string handles display
    Ok(Expr::FunctionCall {
      name: "Minus".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Divide[a, b] - Division with list threading
pub fn divide_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Divide expects exactly 2 arguments".into(),
    ));
  }

  // Check for list threading
  let has_list = args.iter().any(|a| matches!(a, Expr::List(_)));
  if has_list {
    return thread_binary_over_lists(args, divide_two);
  }

  divide_two(&args[0], &args[1])
}

/// Check if an expression represents an infinite quantity
pub fn is_infinity_like(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(name) => name == "Infinity" || name == "ComplexInfinity",
    Expr::FunctionCall { name, .. } => name == "DirectedInfinity",
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => matches!(operand.as_ref(), Expr::Identifier(n) if n == "Infinity"),
    _ => false,
  }
}

/// Helper for division of two arguments
pub fn divide_two(a: &Expr, b: &Expr) -> Result<Expr, InterpreterError> {
  // 0 / expr → 0  (as long as denominator is not also zero)
  let b_is_zero =
    matches!(b, Expr::Integer(0)) || matches!(b, Expr::Real(z) if *z == 0.0);
  if !b_is_zero {
    if matches!(a, Expr::Integer(0)) {
      return Ok(Expr::Integer(0));
    }
    if matches!(a, Expr::Real(f) if *f == 0.0) {
      return Ok(Expr::Real(0.0));
    }
  }

  // finite / Infinity or finite / DirectedInfinity[z] → 0
  if is_infinity_like(b) && !is_infinity_like(a) {
    return Ok(Expr::Integer(0));
  }

  // Handle Quantity division
  if let Some(result) =
    crate::functions::quantity_ast::try_quantity_divide(a, b)
  {
    return result;
  }

  // For two integers, keep as Rational (fraction)
  if let (Expr::Integer(numer), Expr::Integer(denom)) = (a, b) {
    if *denom == 0 {
      return Err(InterpreterError::EvaluationError("Division by zero".into()));
    }
    return Ok(make_rational(*numer, *denom));
  }

  // Simplify (n * expr) / d where n, d are integers → simplify coefficient
  if let Expr::Integer(d) = b
    && *d != 0
  {
    // BinaryOp form
    if let Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } = a
    {
      // (Integer * expr) / Integer → (n/d) * expr
      if let Expr::Integer(n) = left.as_ref() {
        let coeff = make_rational(*n, *d);
        return multiply_scalar_by_expr(&coeff, right);
      }
      // (expr * Integer) / Integer → expr * (n/d)
      if let Expr::Integer(n) = right.as_ref() {
        let coeff = make_rational(*n, *d);
        return multiply_scalar_by_expr(&coeff, left);
      }
    }
    // FunctionCall("Times", ...) form
    if let Expr::FunctionCall { name, args: targs } = a
      && name == "Times"
    {
      for (i, arg) in targs.iter().enumerate() {
        if let Expr::Integer(n) = arg {
          let coeff = make_rational(*n, *d);
          let mut rest: Vec<Expr> = targs
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, e)| e.clone())
            .collect();
          if rest.len() == 1 {
            return multiply_scalar_by_expr(&coeff, &rest.remove(0));
          } else {
            let rest_expr = Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest,
            };
            return multiply_scalar_by_expr(&coeff, &rest_expr);
          }
        }
      }
    }
  }

  // Complex number division: (a + b*I) / integer → simplify
  if let Expr::Integer(d) = b
    && *d != 0
    && let Some(((re_n, re_d), (im_n, im_d))) = try_extract_complex_exact(a)
    && im_n != 0
  {
    // (re + im*I) / d = re/d + (im/d)*I
    let new_re_n = re_n;
    let new_re_d = re_d * *d;
    let new_im_n = im_n;
    let new_im_d = im_d * *d;
    return Ok(complex_rational_to_expr(
      new_re_n, new_re_d, new_im_n, new_im_d,
    ));
  }

  // Exact rational division: Rational/Rational, Rational/Integer, Integer/Rational
  // (a_n/a_d) / (b_n/b_d) = (a_n * b_d) / (a_d * b_n)
  if let (Some((a_n, a_d)), Some((b_n, b_d))) =
    (try_as_rational(a), try_as_rational(b))
  {
    let numer = a_n.checked_mul(b_d);
    let denom = a_d.checked_mul(b_n);
    if let (Some(n), Some(d)) = (numer, denom) {
      if d == 0 {
        return Err(InterpreterError::EvaluationError(
          "Division by zero".into(),
        ));
      }
      return Ok(make_rational(n, d));
    }
  }

  // For reals, perform floating-point division
  // Use try_eval_to_f64 when at least one operand is Real to handle constants like Pi/4.0
  let has_real = matches!(a, Expr::Real(_)) || matches!(b, Expr::Real(_));
  let eval_fn = if has_real {
    |e: &Expr| try_eval_to_f64(e)
  } else {
    |e: &Expr| expr_to_num(e)
  };
  match (eval_fn(a), eval_fn(b)) {
    (Some(x), Some(y)) => {
      if y == 0.0 {
        Err(InterpreterError::EvaluationError("Division by zero".into()))
      } else {
        Ok(Expr::Real(x / y))
      }
    }
    _ => {
      // x / x → 1 for identical symbolic expressions
      if crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b) {
        return Ok(Expr::Integer(1));
      }
      // Flatten nested divisions: (a/b)/c → a/(b*c), a/(b/c) → (a*c)/b
      let (num, den) = flatten_division(a, b);

      Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(num),
        right: Box::new(den),
      })
    }
  }
}

/// Flatten nested divisions into a single numerator and denominator.
/// (a/b)/c → (a, b*c), a/(b/c) → (a*c, b), (a/b)/(c/d) → (a*d, b*c)
pub fn flatten_division(a: &Expr, b: &Expr) -> (Expr, Expr) {
  // Extract (numerator, denominator) from each side
  let (a_num, a_den) = extract_num_den(a);
  let (b_num, b_den) = extract_num_den(b);

  // a/b = (a_num/a_den) / (b_num/b_den) = (a_num * b_den) / (a_den * b_num)
  let num = if a_den.is_none() && b_den.is_none() {
    a_num.clone()
  } else if let Some(bd) = &b_den {
    // a_num * b_den
    build_times_simple(&a_num, bd)
  } else {
    a_num.clone()
  };

  let den = if a_den.is_none() && b_den.is_none() {
    b_num.clone()
  } else if let Some(ad) = &a_den {
    if b_den.is_some() {
      // a_den * b_num
      build_times_simple(ad, &b_num)
    } else {
      // a_den * b_num (b has no denominator)
      build_times_simple(ad, &b_num)
    }
  } else {
    // a has no denominator, b has denominator
    b_num.clone()
  };

  (num, den)
}

/// Extract numerator and optional denominator from an expression.
pub fn extract_num_den(e: &Expr) -> (Expr, Option<Expr>) {
  match e {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => (*left.clone(), Some(*right.clone())),
    _ => (e.clone(), None),
  }
}

/// Build Times[a, b] without full evaluation (just structural)
pub fn build_times_simple(a: &Expr, b: &Expr) -> Expr {
  // For simple integer multiplication, compute directly
  if let (Expr::Integer(x), Expr::Integer(y)) = (a, b) {
    return Expr::Integer(x * y);
  }
  // For 1 * x, just return x
  if matches!(a, Expr::Integer(1)) {
    return b.clone();
  }
  if matches!(b, Expr::Integer(1)) {
    return a.clone();
  }
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(a.clone()),
    right: Box::new(b.clone()),
  }
}

/// Power[a, b] - Exponentiation with list threading
pub fn power_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Power expects exactly 2 arguments".into(),
    ));
  }

  // Check for list threading
  let has_list = args.iter().any(|a| matches!(a, Expr::List(_)));
  if has_list {
    return thread_binary_over_lists(args, power_two);
  }

  power_two(&args[0], &args[1])
}

/// Helper for Power of two arguments
pub fn power_two(base: &Expr, exp: &Expr) -> Result<Expr, InterpreterError> {
  // Handle Quantity^n
  if let Some(result) =
    crate::functions::quantity_ast::try_quantity_power(base, exp)
  {
    return result;
  }

  // x^1 -> x
  if matches!(exp, Expr::Integer(1)) {
    return Ok(base.clone());
  }

  // x^0 -> 1 (for non-zero x; 0^0 is Indeterminate, handled below)
  if matches!(exp, Expr::Integer(0))
    && !matches!(base, Expr::Integer(0))
    && !matches!(base, Expr::Real(f) if *f == 0.0)
  {
    return Ok(Expr::Integer(1));
  }

  // Sqrt[x]^n → x^(n/2)
  if let Expr::FunctionCall { name, args: fargs } = base
    && name == "Sqrt"
    && fargs.len() == 1
    && let Expr::Integer(n) = exp
  {
    if *n == 2 {
      return Ok(fargs[0].clone());
    }
    // Sqrt[x]^n = x^(n/2)
    return power_two(&fargs[0], &make_rational(*n, 2));
  }

  // (base^exp1)^exp2 -> base^(exp1*exp2) when both exponents are integers
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: inner_base,
    right: inner_exp,
  } = base
    && let (Expr::Integer(e1), Expr::Integer(e2)) = (inner_exp.as_ref(), exp)
  {
    let combined = *e1 * *e2;
    return power_two(inner_base, &Expr::Integer(combined));
  }

  // I^n cycles with period 4: I^0=1, I^1=I, I^2=-1, I^3=-I
  if let Expr::Identifier(name) = base
    && name == "I"
    && let Expr::Integer(n) = exp
  {
    let r = ((*n % 4) + 4) % 4; // always non-negative mod
    return Ok(match r {
      0 => Expr::Integer(1),
      1 => Expr::Identifier("I".to_string()),
      2 => Expr::Integer(-1),
      3 => negate_expr(Expr::Identifier("I".to_string())),
      _ => unreachable!(),
    });
  }

  // E^(complex) → Euler's formula: E^(a + b*I) = E^a * (Cos[b] + I*Sin[b])
  if matches!(base, Expr::Constant(c) if c == "E") {
    // Try float complex extraction for the exponent
    if let Some((re, im)) = try_extract_complex_float(exp)
      && im != 0.0
    {
      let mag = re.exp();
      let cos_val = im.cos();
      let sin_val = im.sin();
      let real_part = mag * cos_val;
      let imag_part = mag * sin_val;
      if imag_part == 0.0 {
        return Ok(Expr::Real(real_part));
      }
      if real_part == 0.0 {
        return Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Real(imag_part)),
          right: Box::new(Expr::Identifier("I".to_string())),
        });
      }
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(Expr::Real(real_part)),
        right: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Real(imag_part)),
          right: Box::new(Expr::Identifier("I".to_string())),
        }),
      });
    }
  }

  // Special case: 0^0 is Indeterminate (matches Wolfram)
  let base_is_zero = matches!(base, Expr::Integer(0))
    || matches!(base, Expr::Real(f) if *f == 0.0);
  let exp_is_zero = matches!(exp, Expr::Integer(0))
    || matches!(exp, Expr::Real(f) if *f == 0.0);
  if base_is_zero && exp_is_zero {
    let base_str = crate::syntax::expr_to_string(base);
    let exp_str = crate::syntax::expr_to_string(exp);
    // Align exponent above the base in the warning message
    // "Power::indet: Indeterminate expression " is 39 chars
    // Exponent starts at column 39 + len(base), right-align needs + len(exp)
    let padding = 39 + base_str.len() + exp_str.len();
    eprintln!();
    eprintln!("{:>width$}", exp_str, width = padding);
    eprintln!(
      "Power::indet: Indeterminate expression {}  encountered.",
      base_str
    );
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  // Special case: integer base with negative integer exponent -> Rational
  if let (Expr::Integer(b), Expr::Integer(e)) = (base, exp)
    && *e < 0
  {
    // b^(-n) = 1 / b^n = Rational[1, b^n]
    let pos_exp = (-*e) as u32;
    if let Some(denom) = b.checked_pow(pos_exp) {
      return Ok(Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(denom)],
      });
    }
  }

  // Special case: Rational^Integer -> exact rational result
  if let Expr::FunctionCall {
    name: rname,
    args: rargs,
  } = base
    && rname == "Rational"
    && rargs.len() == 2
    && let (Expr::Integer(num), Expr::Integer(den)) = (&rargs[0], &rargs[1])
    && let Expr::Integer(e) = exp
  {
    if *e > 0 {
      let pe = *e as u32;
      if let (Some(new_num), Some(new_den)) =
        (num.checked_pow(pe), den.checked_pow(pe))
      {
        return Ok(make_rational(new_num, new_den));
      }
    } else if *e < 0 {
      // (a/b)^(-n) = (b/a)^n = b^n / a^n
      let pe = (-*e) as u32;
      if let (Some(new_num), Some(new_den)) =
        (den.checked_pow(pe), num.checked_pow(pe))
      {
        return Ok(make_rational(new_num, new_den));
      }
    }
  }

  // Special case: Integer^Rational — keep symbolic unless result is exact integer
  if let Expr::Integer(b) = base
    && let Expr::FunctionCall { name, args: rargs } = exp
    && name == "Rational"
    && rargs.len() == 2
    && let (Expr::Integer(numer), Expr::Integer(denom)) = (&rargs[0], &rargs[1])
  {
    // Try to compute exact integer root
    let result = (*b as f64).powf(*numer as f64 / *denom as f64);
    if result.fract() == 0.0 && result.is_finite() {
      return Ok(Expr::Integer(result as i128));
    }
    // Simplify n^(p/q) by prime factorization
    // n = p1^k1 * p2^k2 * ...
    // n^(p/q) = product of p_i^(k_i*p/q)
    // Each p_i^(k_i*p/q) = p_i^floor_i * p_i^(rem_i/q)
    if *numer > 0 && *denom > 0 && *b > 0 {
      let d = *denom as u64;
      let n = *numer as u64;
      let mut outside: i128 = 1;
      // Collect (prime, remainder_exponent) pairs for the radical part
      let mut radical_factors: Vec<(i128, u64)> = Vec::new();
      let mut remaining = *b as u64;
      let mut factor = 2u64;
      while factor * factor <= remaining {
        let mut count = 0u64;
        while remaining.is_multiple_of(factor) {
          remaining /= factor;
          count += 1;
        }
        if count > 0 {
          let total = count * n;
          let extracted = total / d;
          let leftover = total % d;
          if extracted > 0 {
            outside *= (factor as i128).pow(extracted as u32);
          }
          if leftover > 0 {
            radical_factors.push((factor as i128, leftover));
          }
        }
        factor += 1;
      }
      if remaining > 1 {
        let total = n; // count=1
        let extracted = total / d;
        let leftover = total % d;
        if extracted > 0 {
          outside *= (remaining as i128).pow(extracted as u32);
        }
        if leftover > 0 {
          radical_factors.push((remaining as i128, leftover));
        }
      }

      let has_radical = !radical_factors.is_empty();
      let has_outside = outside > 1;

      if !has_radical {
        // Fully simplified
        return Ok(Expr::Integer(outside));
      }

      // Only simplify if we actually extracted something outside,
      // or if the radical has fewer prime factors than the original
      // (prevents infinite recursion: 6^(1/3) → 2^(1/3)*3^(1/3) → 6^(1/3) ...)
      if !has_outside && radical_factors.len() > 1 {
        // No simplification possible, keep original form
        if *numer == 1 && *denom == 2 {
          return Ok(Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![base.clone()],
          });
        }
        return Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(base.clone()),
          right: Box::new(exp.clone()),
        });
      }

      // Build radical part: product of p_i^(rem_i/q)
      let mut rad_parts: Vec<Expr> = Vec::new();
      for (prime, rem_exp) in &radical_factors {
        let g = gcd_i128(*rem_exp as i128, d as i128);
        let reduced_num = *rem_exp as i128 / g;
        let reduced_den = d as i128 / g;
        if reduced_den == 1 {
          // Integer power
          rad_parts.push(Expr::Integer(prime.pow(reduced_num as u32)));
        } else if reduced_num == 1 && reduced_den == 2 {
          rad_parts.push(Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![Expr::Integer(*prime)],
          });
        } else {
          rad_parts.push(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(Expr::Integer(*prime)),
            right: Box::new(make_rational(reduced_num, reduced_den)),
          });
        }
      }

      // Combine: outside * rad_parts
      let mut all_factors: Vec<Expr> = Vec::new();
      if has_outside {
        all_factors.push(Expr::Integer(outside));
      }
      all_factors.extend(rad_parts);

      if all_factors.len() == 1 {
        return Ok(all_factors.remove(0));
      }
      return times_ast(&all_factors);
    }

    // If exponent is 1/2, display as Sqrt[base]
    if *numer == 1 && *denom == 2 && *b > 0 {
      return Ok(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![base.clone()],
      });
    }
    // Not exact — keep symbolic
    return Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(base.clone()),
      right: Box::new(exp.clone()),
    });
  }

  // Integer^Integer with non-negative exponent: use exact BigInt arithmetic
  if let (Expr::Integer(b), Expr::Integer(e)) = (base, exp)
    && *e >= 0
  {
    use num_bigint::BigInt;
    let base_big = BigInt::from(*b);
    let result = num_traits::pow::pow(base_big, *e as usize);
    return Ok(bigint_to_expr(result));
  }

  // BigInteger base with integer exponent
  if let (Expr::BigInteger(b), Expr::Integer(e)) = (base, exp)
    && *e >= 0
  {
    let result = num_traits::pow::pow(b.clone(), *e as usize);
    return Ok(bigint_to_expr(result));
  }

  // If either operand is Real, result is Real (even if whole number)
  let has_real = matches!(base, Expr::Real(_)) || matches!(exp, Expr::Real(_));

  match (expr_to_num(base), expr_to_num(exp)) {
    (Some(a), Some(b)) => {
      let result = a.powf(b);
      if has_real {
        Ok(Expr::Real(result))
      } else {
        // Both were integers - use num_to_expr to get Integer when result is whole
        Ok(num_to_expr(result))
      }
    }
    _ => {
      // x^(1/2) → Sqrt[x]
      if let Expr::FunctionCall { name, args: rargs } = exp
        && name == "Rational"
        && rargs.len() == 2
        && matches!(&rargs[0], Expr::Integer(1))
        && matches!(&rargs[1], Expr::Integer(2))
      {
        return Ok(Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![base.clone()],
        });
      }
      Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(base.clone()),
        right: Box::new(exp.clone()),
      })
    }
  }
}

/// Thread a binary operation over lists
pub fn thread_binary_over_lists<F>(
  args: &[Expr],
  op: F,
) -> Result<Expr, InterpreterError>
where
  F: Fn(&Expr, &Expr) -> Result<Expr, InterpreterError>,
{
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Binary operation expects 2 arguments".into(),
    ));
  }

  match (&args[0], &args[1]) {
    (Expr::List(left), Expr::List(right)) => {
      // List + List: element-wise operation
      if left.len() != right.len() {
        return Err(InterpreterError::EvaluationError(
          "Lists must have the same length".into(),
        ));
      }
      let results: Result<Vec<Expr>, _> = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| op(l, r))
        .collect();
      Ok(Expr::List(results?))
    }
    (Expr::List(items), scalar) => {
      // List + scalar: broadcast scalar
      let results: Result<Vec<Expr>, _> =
        items.iter().map(|item| op(item, scalar)).collect();
      Ok(Expr::List(results?))
    }
    (scalar, Expr::List(items)) => {
      // scalar + List: broadcast scalar
      let results: Result<Vec<Expr>, _> =
        items.iter().map(|item| op(scalar, item)).collect();
      Ok(Expr::List(results?))
    }
    _ => op(&args[0], &args[1]),
  }
}

/// Subtract[a, b] - Returns a - b
pub fn subtract_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "Subtract".to_string(),
      args: args.to_vec(),
    });
  }
  // Subtract[a, b] = a + (-1 * b)
  let negated_b = times_ast(&[Expr::Integer(-1), args[1].clone()])?;
  plus_ast(&[args[0].clone(), negated_b])
}

/// Recursively flatten all List arguments for Max/Min
pub fn flatten_lists(args: &[Expr]) -> Vec<&Expr> {
  let mut result = Vec::new();
  for arg in args {
    match arg {
      Expr::List(items) => result.extend(flatten_lists(items)),
      _ => result.push(arg),
    }
  }
  result
}

/// Like try_eval_to_f64 but also handles Infinity/-Infinity (for Max/Min)
pub fn try_eval_to_f64_with_infinity(expr: &Expr) -> Option<f64> {
  // Check by string representation for Infinity forms
  let s = crate::syntax::expr_to_string(expr);
  if s == "Infinity" {
    return Some(f64::INFINITY);
  }
  if s == "-Infinity" {
    return Some(f64::NEG_INFINITY);
  }
  try_eval_to_f64(expr)
}

/// Max[args...] or Max[list] - Maximum value
pub fn max_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Identifier("-Infinity".to_string()));
  }

  // Flatten all nested lists
  let items = flatten_lists(args);
  if items.is_empty() {
    return Ok(Expr::Identifier("-Infinity".to_string()));
  }

  // Separate numeric and symbolic arguments
  let mut best_val: Option<f64> = None;
  let mut best_expr: Option<&Expr> = None;
  let mut symbolic: Vec<Expr> = Vec::new();
  for item in &items {
    if let Some(n) = try_eval_to_f64_with_infinity(item) {
      match best_val {
        Some(m) if n > m => {
          best_val = Some(n);
          best_expr = Some(item);
        }
        None => {
          best_val = Some(n);
          best_expr = Some(item);
        }
        _ => {}
      }
    } else {
      symbolic.push((*item).clone());
    }
  }

  if symbolic.is_empty() {
    // All numeric
    match best_expr {
      Some(expr) => Ok((*expr).clone()),
      None => Ok(num_to_expr(f64::NEG_INFINITY)),
    }
  } else {
    // Mixed: keep max numeric and all symbolic args
    let mut result_args: Vec<Expr> = Vec::new();
    if let Some(expr) = best_expr {
      result_args.push((*expr).clone());
    }
    result_args.extend(symbolic);
    if result_args.len() == 1 {
      Ok(result_args.remove(0))
    } else {
      Ok(Expr::FunctionCall {
        name: "Max".to_string(),
        args: result_args,
      })
    }
  }
}

/// Min[args...] or Min[list] - Minimum value
pub fn min_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }

  // Flatten all nested lists
  let items = flatten_lists(args);
  if items.is_empty() {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }

  // Separate numeric and symbolic arguments
  let mut best_val: Option<f64> = None;
  let mut best_expr: Option<&Expr> = None;
  let mut symbolic: Vec<Expr> = Vec::new();
  for item in &items {
    if let Some(n) = try_eval_to_f64_with_infinity(item) {
      match best_val {
        Some(m) if n < m => {
          best_val = Some(n);
          best_expr = Some(item);
        }
        None => {
          best_val = Some(n);
          best_expr = Some(item);
        }
        _ => {}
      }
    } else {
      symbolic.push((*item).clone());
    }
  }

  if symbolic.is_empty() {
    // All numeric
    match best_expr {
      Some(expr) => Ok((*expr).clone()),
      None => Ok(num_to_expr(f64::INFINITY)),
    }
  } else {
    // Mixed: keep min numeric and all symbolic args
    let mut result_args: Vec<Expr> = Vec::new();
    if let Some(expr) = best_expr {
      result_args.push((*expr).clone());
    }
    result_args.extend(symbolic);
    if result_args.len() == 1 {
      Ok(result_args.remove(0))
    } else {
      Ok(Expr::FunctionCall {
        name: "Min".to_string(),
        args: result_args,
      })
    }
  }
}
