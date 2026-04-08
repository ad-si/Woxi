#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

/// Plus[args...] - Sum of arguments, with list threading
pub fn plus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(0));
  }

  // Handle DateObject subtraction: DateObject[...] - DateObject[...] → Quantity[n, "Days"]
  if let Some(result) = try_date_object_subtraction(args) {
    return result;
  }

  // Handle Quantity arithmetic before anything else
  if let Some(result) = crate::functions::quantity_ast::try_quantity_plus(args)
  {
    return result;
  }

  // Handle Interval arithmetic
  if let Some(result) = crate::functions::interval_ast::try_interval_plus(args)
  {
    return result;
  }

  // Flatten nested Plus arguments (recursive to handle deeply nested Plus)
  let mut flat_args: Vec<Expr> = Vec::new();
  let mut stack: Vec<&Expr> = args.iter().rev().collect();
  while let Some(arg) = stack.pop() {
    match arg {
      Expr::FunctionCall {
        name,
        args: inner_args,
      } if name == "Plus" => {
        for inner in inner_args.iter().rev() {
          stack.push(inner);
        }
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

  // Classify arguments: exact (Integer/Rational), real (Real), bigfloat, or symbolic
  let mut has_real = false;
  let mut has_bigfloat = false;
  let mut all_numeric = true;
  for arg in &flat_args {
    match arg {
      Expr::Real(_) => has_real = true,
      Expr::BigFloat(_, _) => has_bigfloat = true,
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

  // If all numeric and no Reals/BigFloats, use exact rational arithmetic
  if all_numeric && !has_real && !has_bigfloat {
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

  // If all numeric with BigFloat (no machine Real), use precision-tracked arithmetic
  if all_numeric && has_bigfloat && !has_real {
    return bigfloat_plus(&flat_args);
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
    sorted_symbolic.sort_by(compare_plus_terms);
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

fn bigint_gcd(
  a: &num_bigint::BigInt,
  b: &num_bigint::BigInt,
) -> num_bigint::BigInt {
  use num_bigint::BigInt;
  use num_bigint::Sign;
  let mut a = match a.sign() {
    Sign::Minus => -a,
    _ => a.clone(),
  };
  let mut b = match b.sign() {
    Sign::Minus => -b,
    _ => b.clone(),
  };
  while b != BigInt::from(0) {
    let t = &a % &b;
    a = b;
    b = t;
  }
  a
}

/// Coefficient: either exact rational (i128 or BigInt) or approximate real
#[derive(Clone)]
pub enum Coeff {
  Exact(i128, i128), // (numer, denom)
  BigExact(num_bigint::BigInt, num_bigint::BigInt), // (numer, denom)
  Real(f64),
}

impl Coeff {
  fn is_zero(&self) -> bool {
    match self {
      Self::Exact(n, _) => *n == 0,
      Self::BigExact(n, _) => n.sign() == num_bigint::Sign::NoSign,
      Self::Real(f) => *f == 0.0,
    }
  }
  fn is_one(&self) -> bool {
    match self {
      Self::Exact(n, d) => *n == 1 && *d == 1,
      Self::BigExact(n, d) => {
        use num_traits::One;
        n.is_one() && d.is_one()
      }
      Self::Real(f) => *f == 1.0,
    }
  }
  fn is_negative(&self) -> bool {
    match self {
      Self::Exact(n, d) => (*n < 0) != (*d < 0),
      Self::BigExact(n, d) => {
        (n.sign() == num_bigint::Sign::Minus)
          != (d.sign() == num_bigint::Sign::Minus)
      }
      Self::Real(f) => *f < 0.0,
    }
  }
  fn to_f64(&self) -> f64 {
    match self {
      Self::Exact(n, d) => *n as f64 / *d as f64,
      Self::BigExact(n, d) => {
        use num_traits::ToPrimitive;
        n.to_f64().unwrap_or(f64::INFINITY) / d.to_f64().unwrap_or(1.0)
      }
      Self::Real(f) => *f,
    }
  }
  fn to_big(n: i128, d: i128) -> (num_bigint::BigInt, num_bigint::BigInt) {
    (num_bigint::BigInt::from(n), num_bigint::BigInt::from(d))
  }
  fn add(&self, other: &Self) -> Self {
    use num_bigint::BigInt;

    match (self, other) {
      (Self::Exact(n1, d1), Self::Exact(n2, d2)) => {
        // Try i128 first; on overflow, promote to BigInt
        if let (Some(a), Some(b), Some(c)) = (
          n1.checked_mul(*d2),
          n2.checked_mul(*d1),
          d1.checked_mul(*d2),
        ) && let Some(sn) = a.checked_add(b)
        {
          let mut sd = c;
          let mut sn = sn;
          let g = gcd(sn, sd);
          sn /= g;
          sd /= g;
          if sd < 0 {
            sn = -sn;
            sd = -sd;
          }
          return Self::Exact(sn, sd);
        }
        // Overflow: promote to BigInt
        let (n1, d1) = Self::to_big(*n1, *d1);
        let (n2, d2) = Self::to_big(*n2, *d2);
        let mut sn = &n1 * &d2 + &n2 * &d1;
        let mut sd = d1 * d2;
        let g = bigint_gcd(&sn, &sd);
        sn /= &g;
        sd /= g;
        if sd < BigInt::from(0) {
          sn = -sn;
          sd = -sd;
        }
        Self::BigExact(sn, sd)
      }
      (Self::BigExact(n1, d1), Self::BigExact(n2, d2)) => {
        let mut sn = n1 * d2 + n2 * d1;
        let mut sd = d1 * d2;
        let g = bigint_gcd(&sn, &sd);
        sn /= &g;
        sd /= g;
        if sd < BigInt::from(0) {
          sn = -sn;
          sd = -sd;
        }
        Self::BigExact(sn, sd)
      }
      (Self::Exact(n, d), Self::BigExact(..))
      | (Self::BigExact(..), Self::Exact(n, d)) => {
        let big_self;
        let big_other;
        match self {
          Self::BigExact(bn, bd) => {
            big_self = Self::BigExact(bn.clone(), bd.clone());
            big_other = Self::BigExact(BigInt::from(*n), BigInt::from(*d));
          }
          _ => {
            let (bn, bd) = Self::to_big(*n, *d);
            big_self = Self::BigExact(bn, bd);
            big_other = other.clone();
          }
        }
        big_self.add(&big_other)
      }
      _ => Self::Real(self.to_f64() + other.to_f64()),
    }
  }
  fn mul(&self, other: &Self) -> Self {
    use num_bigint::BigInt;

    match (self, other) {
      (Self::Exact(n1, d1), Self::Exact(n2, d2)) => {
        if let (Some(sn), Some(sd)) = (n1.checked_mul(*n2), d1.checked_mul(*d2))
        {
          let mut sn = sn;
          let mut sd = sd;
          let g = gcd(sn, sd);
          sn /= g;
          sd /= g;
          if sd < 0 {
            sn = -sn;
            sd = -sd;
          }
          return Self::Exact(sn, sd);
        }
        let (n1, d1) = Self::to_big(*n1, *d1);
        let (n2, d2) = Self::to_big(*n2, *d2);
        let mut sn = &n1 * &n2;
        let mut sd = d1 * d2;
        let g = bigint_gcd(&sn, &sd);
        sn /= &g;
        sd /= g;
        if sd < BigInt::from(0) {
          sn = -sn;
          sd = -sd;
        }
        Self::BigExact(sn, sd)
      }
      (Self::BigExact(n1, d1), Self::BigExact(n2, d2)) => {
        let mut sn = n1 * n2;
        let mut sd = d1 * d2;
        let g = bigint_gcd(&sn, &sd);
        sn /= &g;
        sd /= g;
        if sd < BigInt::from(0) {
          sn = -sn;
          sd = -sd;
        }
        Self::BigExact(sn, sd)
      }
      (Self::Exact(n, d), Self::BigExact(..))
      | (Self::BigExact(..), Self::Exact(n, d)) => {
        let big_self;
        let big_other;
        match self {
          Self::BigExact(bn, bd) => {
            big_self = Self::BigExact(bn.clone(), bd.clone());
            big_other = Self::BigExact(BigInt::from(*n), BigInt::from(*d));
          }
          _ => {
            let (bn, bd) = Self::to_big(*n, *d);
            big_self = Self::BigExact(bn, bd);
            big_other = other.clone();
          }
        }
        big_self.mul(&big_other)
      }
      _ => Self::Real(self.to_f64() * other.to_f64()),
    }
  }
  fn negate(&self) -> Self {
    match self {
      Self::Exact(n, d) => Self::Exact(-n, *d),
      Self::BigExact(n, d) => Self::BigExact(-n, d.clone()),
      Self::Real(f) => Self::Real(-f),
    }
  }
  fn to_expr(&self) -> Expr {
    match self {
      Self::Exact(n, d) => make_rational(*n, *d),
      Self::BigExact(n, d) => {
        use num_traits::{One, ToPrimitive};
        if d.is_one() {
          bigint_to_expr(n.clone())
        } else if let (Some(ni), Some(di)) = (n.to_i128(), d.to_i128()) {
          make_rational(ni, di)
        } else {
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![bigint_to_expr(n.clone()), bigint_to_expr(d.clone())],
          }
        }
      }
      Self::Real(f) => Expr::Real(*f),
    }
  }
}

/// Decompose a term into (coefficient, base_expression).
/// E.g. `3*x` → (Exact(3,1), x), `x` → (Exact(1,1), x), `-x` → (Exact(-1,1), x),
/// `1.5*x` → (Real(1.5), x), `Rational[3,4]*x` → (Exact(3,4), x).
pub fn decompose_term(e: &Expr) -> (Coeff, Expr) {
  match e {
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      let base_from = |args: &[Expr]| -> Expr {
        if args.len() == 2 {
          args[1].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: args[1..].to_vec(),
          }
        }
      };
      // Check if first arg is a numeric coefficient (integer/rational)
      if let Some((n, d)) = expr_to_rational(&args[0]) {
        let (inner_c, inner_base) = decompose_term(&base_from(args));
        let outer_c = Coeff::Exact(n, d);
        return (outer_c.mul(&inner_c), inner_base);
      }
      // Check if first arg is a BigInteger coefficient
      if let Expr::BigInteger(n) = &args[0] {
        let (inner_c, inner_base) = decompose_term(&base_from(args));
        let outer_c = Coeff::BigExact(n.clone(), num_bigint::BigInt::from(1));
        return (outer_c.mul(&inner_c), inner_base);
      }
      // Check if first arg is a Real coefficient
      if let Expr::Real(f) = &args[0] {
        let (inner_c, inner_base) = decompose_term(&base_from(args));
        let outer_c = Coeff::Real(*f);
        return (outer_c.mul(&inner_c), inner_base);
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if let Some((n, d)) = expr_to_rational(left) {
        let (inner_c, inner_base) = decompose_term(right);
        let outer_c = Coeff::Exact(n, d);
        return (outer_c.mul(&inner_c), inner_base);
      }
      if let Expr::BigInteger(n) = left.as_ref() {
        let (inner_c, inner_base) = decompose_term(right);
        let outer_c = Coeff::BigExact(n.clone(), num_bigint::BigInt::from(1));
        return (outer_c.mul(&inner_c), inner_base);
      }
      if let Expr::Real(f) = left.as_ref() {
        let (inner_c, inner_base) = decompose_term(right);
        let outer_c = Coeff::Real(*f);
        return (outer_c.mul(&inner_c), inner_base);
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => {
      // expr / integer → coefficient 1/d, recursively decompose expr
      if let Some((n, d)) = expr_to_rational(right)
        && n != 0
      {
        let (inner_c, inner_base) = decompose_term(left);
        // Multiply inner coefficient by (d/n) since we're dividing by (n/d)
        let divisor = Coeff::Exact(d, n);
        return (inner_c.mul(&divisor), inner_base);
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
  // Also track the original expression and count for groups with a single term
  let mut groups: Vec<(String, Expr, Coeff, Option<Expr>, usize)> = Vec::new();
  let mut index: BTreeMap<String, usize> = BTreeMap::new();

  for term in terms {
    let (c, base) = decompose_term(term);
    let key = crate::syntax::expr_to_string(&base);
    if let Some(&idx) = index.get(&key) {
      let entry = &mut groups[idx];
      entry.2 = entry.2.add(&c);
      entry.3 = None; // Multiple terms combined, can't preserve original
      entry.4 += 1;
    } else {
      index.insert(key.clone(), groups.len());
      groups.push((key, base, c, Some(term.clone()), 1));
    }
  }

  let mut result = Vec::new();
  for (_, base, c, original, count) in groups {
    if c.is_zero() {
      continue; // terms cancelled
    }
    // If only one term contributed and wasn't combined, AND the coefficient is the same
    // as what decompose_term extracted (meaning no simplification happened), preserve original form.
    // This avoids changing e.g. BinaryOp::Divide(x, 2) to Times[Rational[1,2], x] which
    // would affect sorting and display.
    // But if coefficient is 1 and base differs from original (e.g. Times[1, I] → I),
    // we should still simplify.
    if count == 1
      && !c.is_one()
      && let Some(orig) = original
    {
      result.push(orig);
      continue;
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

/// Extract the earliest (alphabetically first) variable name from an expression.
fn extract_earliest_variable(e: &Expr) -> Option<String> {
  match e {
    Expr::Identifier(s)
      if !matches!(
        s.as_str(),
        "Pi"
          | "E"
          | "I"
          | "Infinity"
          | "True"
          | "False"
          | "Null"
          | "None"
          | "All"
          | "Automatic"
          | "ComplexInfinity"
          | "Indeterminate"
          | "EulerGamma"
          | "GoldenRatio"
          | "Degree"
          | "Catalan"
      ) && s.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) =>
    {
      Some(s.clone())
    }
    Expr::FunctionCall { args, .. } => {
      args.iter().filter_map(extract_earliest_variable).min()
    }
    Expr::BinaryOp { left, right, .. } => {
      let l = extract_earliest_variable(left);
      let r = extract_earliest_variable(right);
      match (l, r) {
        (Some(a), Some(b)) => Some(if a <= b { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
      }
    }
    Expr::UnaryOp { operand, .. } => extract_earliest_variable(operand),
    _ => None,
  }
}

/// Count distinct free (symbolic) variables in an expression.
fn count_free_variables(e: &Expr) -> usize {
  fn collect(e: &Expr, vars: &mut std::collections::BTreeSet<String>) {
    match e {
      Expr::Identifier(s)
        if !matches!(
          s.as_str(),
          "Pi"
            | "E"
            | "I"
            | "Infinity"
            | "True"
            | "False"
            | "Null"
            | "None"
            | "All"
            | "Automatic"
            | "ComplexInfinity"
            | "Indeterminate"
            | "EulerGamma"
            | "GoldenRatio"
            | "Degree"
            | "Catalan"
        ) && s
          .chars()
          .next()
          .map(|c| c.is_alphabetic())
          .unwrap_or(false) =>
      {
        vars.insert(s.clone());
      }
      Expr::FunctionCall { args, .. } => {
        for arg in args {
          collect(arg, vars);
        }
      }
      Expr::BinaryOp { left, right, .. } => {
        collect(left, vars);
        collect(right, vars);
      }
      Expr::UnaryOp { operand, .. } => collect(operand, vars),
      _ => {}
    }
  }
  let mut vars = std::collections::BTreeSet::new();
  collect(e, &mut vars);
  vars.len()
}

/// Extract the latest (alphabetically greatest) free variable in an expression.
fn extract_latest_variable(e: &Expr) -> Option<String> {
  match e {
    Expr::Identifier(s)
      if !matches!(
        s.as_str(),
        "Pi"
          | "E"
          | "I"
          | "Infinity"
          | "True"
          | "False"
          | "Null"
          | "None"
          | "All"
          | "Automatic"
          | "ComplexInfinity"
          | "Indeterminate"
          | "EulerGamma"
          | "GoldenRatio"
          | "Degree"
          | "Catalan"
      ) && s.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) =>
    {
      Some(s.clone())
    }
    Expr::FunctionCall { args, .. } => {
      args.iter().filter_map(extract_latest_variable).max()
    }
    Expr::BinaryOp { left, right, .. } => {
      let l = extract_latest_variable(left);
      let r = extract_latest_variable(right);
      match (l, r) {
        (Some(a), Some(b)) => Some(if a >= b { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
      }
    }
    Expr::UnaryOp { operand, .. } => extract_latest_variable(operand),
    _ => None,
  }
}

/// Extract sort key for a Plus term (fallback for non-polynomial terms).
/// Strip the numeric coefficient so that e.g. `3*Sin[x]` and `5*Cos[x]` sort by
/// their base rather than by the coefficient digits.
fn term_sort_key(e: &Expr) -> String {
  let (_, base) = decompose_term(e);
  let s = crate::syntax::expr_to_string(&base);
  s.strip_prefix('-').unwrap_or(&s).to_string()
}

/// Check if an expression is purely numeric (no symbolic variables).
/// E.g. `5`, `Rational[1,2]`, `Power[5, Rational[-1, 2]]` are all numeric.
fn is_numeric_factor(e: &Expr) -> bool {
  match e {
    Expr::Integer(_) | Expr::Real(_) => true,
    Expr::FunctionCall { name, args } if name == "Rational" => {
      args.iter().all(is_numeric_factor)
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      is_numeric_factor(&args[0]) && is_numeric_factor(&args[1])
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => is_numeric_factor(left) && is_numeric_factor(right),
    _ => false,
  }
}

/// Extract (variable_name, exponent) pairs from a polynomial-like expression base.
/// Returns None if the expression is not a simple polynomial term.
/// E.g. `x` → [(x, 1.0)], `x^2` → [(x, 2.0)], `a*b` → [(a, 1.0), (b, 1.0)]
/// Numeric factors in Times (like `5^(-1/2)` in `x/Sqrt[5]`) are skipped.
fn extract_var_exp_pairs(e: &Expr) -> Option<Vec<(String, f64)>> {
  match e {
    Expr::Identifier(s) => Some(vec![(s.clone(), 1.0)]),
    Expr::Constant(c) => Some(vec![(format!("{c:?}"), 1.0)]),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Identifier(s) = left.as_ref() {
        let exp = expr_to_f64(right)?;
        return Some(vec![(s.clone(), exp)]);
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Expr::Identifier(s) = &args[0] {
        let exp = expr_to_f64(&args[1])?;
        return Some(vec![(s.clone(), exp)]);
      }
      None
    }
    expr if is_sqrt(expr).is_some() => {
      let sqrt_arg = is_sqrt(expr).unwrap();
      if let Expr::Identifier(s) = sqrt_arg {
        return Some(vec![(s.clone(), 0.5)]);
      }
      None
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => {
      // left / right = left * right^(-1)
      let mut pairs = Vec::new();
      if !is_numeric_factor(left) {
        pairs.extend(extract_var_exp_pairs(left)?);
      }
      if !is_numeric_factor(right) {
        let mut right_pairs = extract_var_exp_pairs(right)?;
        for pair in &mut right_pairs {
          pair.1 = -pair.1;
        }
        pairs.extend(right_pairs);
      }
      Some(pairs)
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      let mut pairs = Vec::new();
      if !is_numeric_factor(left) {
        pairs.extend(extract_var_exp_pairs(left)?);
      }
      if !is_numeric_factor(right) {
        pairs.extend(extract_var_exp_pairs(right)?);
      }
      Some(pairs)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut pairs = Vec::new();
      for arg in args {
        if !is_numeric_factor(arg) {
          pairs.extend(extract_var_exp_pairs(arg)?);
        }
      }
      Some(pairs)
    }
    _ => None,
  }
}

/// Convert an expression to an f64 if it represents a number.
fn expr_to_f64(e: &Expr) -> Option<f64> {
  match e {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some(*n as f64 / *d as f64)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Returns true if `e` contains free variables (identifiers that are not
/// well-known constants like I, Infinity).  Used to distinguish transcendental
/// expressions-of-variables (e.g. Sin[2*x]) from pure numeric constants
/// (e.g. Log[15], Sqrt[6]) when sorting Plus terms.
fn has_free_variables(e: &Expr) -> bool {
  match e {
    Expr::Identifier(name) => {
      // I and Infinity are well-known constants, not free variables
      name != "I" && name != "Infinity"
    }
    Expr::Constant(_) => false,
    Expr::Integer(_) | Expr::Real(_) => false,
    Expr::FunctionCall { args, .. } => args.iter().any(has_free_variables),
    Expr::BinaryOp { left, right, .. } => {
      has_free_variables(left) || has_free_variables(right)
    }
    Expr::UnaryOp { operand, .. } => has_free_variables(operand),
    _ => false,
  }
}

/// Returns true if `e` contains a non-standard function call (not in the common
/// algebraic/transcendental set) that has at least one non-numeric argument.
/// This identifies expressions like `Dt[y, x]` or `Derivative[1][f][x]` that
/// represent unevaluated derivatives or opaque functions with free variables.
fn contains_opaque_fn_call(e: &Expr) -> bool {
  match e {
    Expr::FunctionCall { name, args } => {
      let is_known = matches!(
        name.as_str(),
        "Times"
          | "Plus"
          | "Minus"
          | "Power"
          | "Rational"
          | "Sqrt"
          | "Abs"
          | "Sin"
          | "Cos"
          | "Tan"
          | "Cot"
          | "Sec"
          | "Csc"
          | "Sinh"
          | "Cosh"
          | "Tanh"
          | "Coth"
          | "Sech"
          | "Csch"
          | "ArcSin"
          | "ArcCos"
          | "ArcTan"
          | "ArcCot"
          | "ArcSec"
          | "ArcCsc"
          | "ArcSinh"
          | "ArcCosh"
          | "ArcTanh"
          | "ArcCoth"
          | "ArcSech"
          | "ArcCsch"
          | "Exp"
          | "Log"
          | "Erf"
          | "Erfc"
          | "InverseErf"
          | "Factorial"
          | "Binomial"
          | "Re"
          | "Im"
          | "Conjugate"
          | "Floor"
          | "Ceiling"
          | "Round"
          | "Sign"
          | "Gamma"
          | "Beta"
      );
      if !is_known && args.iter().any(|a| !is_numeric_factor(a)) {
        return true;
      }
      args.iter().any(contains_opaque_fn_call)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_opaque_fn_call(left) || contains_opaque_fn_call(right)
    }
    Expr::UnaryOp { operand, .. } => contains_opaque_fn_call(operand),
    _ => false,
  }
}

/// Returns true if `e` is a purely numeric/known-constant expression with
/// no symbolic function calls. Unlike `has_free_variables`, this treats
/// `f[0]` as non-numeric (it is a symbolic call even though its argument
/// is numeric). Used for Plus term sorting to identify true constants like
/// `512 * Rational[-1, 24]` or `(1 + Pi) / 2`.
fn is_numeric_constant(e: &Expr) -> bool {
  match e {
    Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_) => true,
    Expr::Constant(_) => true,
    Expr::Identifier(name) => {
      matches!(name.as_str(), "I" | "Infinity")
    }
    Expr::FunctionCall { name, args } => {
      matches!(
        name.as_str(),
        "Rational"
          | "Complex"
          | "Times"
          | "Plus"
          | "Power"
          | "Sqrt"
          | "Log"
          | "Exp"
          | "DirectedInfinity"
      ) && args.iter().all(is_numeric_constant)
    }
    Expr::BinaryOp { left, right, .. } => {
      is_numeric_constant(left) && is_numeric_constant(right)
    }
    Expr::UnaryOp { operand, .. } => is_numeric_constant(operand),
    _ => false,
  }
}

/// Compare two Plus terms using Wolfram-compatible canonical ordering.
/// For polynomial-like terms, sorts by (variable, exponent) pairs in reverse-lex order:
/// each term's pairs are sorted by variable name descending, then compared
/// lexicographically (variable ascending, exponent ascending, shorter first).
fn compare_plus_terms(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  let pa = term_priority(a);
  let pb = term_priority(b);
  if pa != pb {
    return pa.cmp(&pb);
  }

  let (_, base_a) = decompose_term(a);
  let (_, base_b) = decompose_term(b);

  let pairs_a = extract_var_exp_pairs(&base_a);
  let pairs_b = extract_var_exp_pairs(&base_b);
  let a_has_pairs = pairs_a.is_some();
  let b_has_pairs = pairs_b.is_some();

  match (pairs_a, pairs_b) {
    (Some(mut va), Some(mut vb)) => {
      // Sort each term's pairs by variable name descending
      va.sort_by(|x, y| y.0.cmp(&x.0));
      vb.sort_by(|x, y| y.0.cmp(&x.0));
      // Compare pair-by-pair: variable ascending, then exponent ascending
      for (ap, bp) in va.iter().zip(vb.iter()) {
        let cmp = ap.0.cmp(&bp.0);
        if cmp != std::cmp::Ordering::Equal {
          return cmp;
        }
        let cmp = ap.1.partial_cmp(&bp.1).unwrap_or(std::cmp::Ordering::Equal);
        if cmp != std::cmp::Ordering::Equal {
          return cmp;
        }
      }
      // If all compared pairs equal, shorter comes first
      let cmp = va.len().cmp(&vb.len());
      if cmp != std::cmp::Ordering::Equal {
        return cmp;
      }
      // Tiebreaker: positive coefficients before negative
      let (coeff_a, _) = decompose_term(a);
      let (coeff_b, _) = decompose_term(b);
      coeff_a.is_negative().cmp(&coeff_b.is_negative())
    }
    _ => {
      // If exactly one term has polynomial pairs and the other doesn't,
      // sort the polynomial term first when the non-polynomial term is a
      // transcendental function (priority 1), e.g.
      // Sin[2*x] (sort after x/2), Times[-1, Sin[x]/4], or
      // numeric transcendental constants like Log[15], Sqrt[6].
      // For compound algebraic terms like 4*(3+2*x) vs 8*x, compare by
      // earliest variable: same variable → monomial first.
      if a_has_pairs != b_has_pairs {
        let pair_term = if a_has_pairs { a } else { b };
        let none_term = if b_has_pairs { a } else { b };
        let (_, none_base) = decompose_term(none_term);
        if contains_opaque_fn_call(none_term) || term_priority(&none_base) >= 1
        {
          return if a_has_pairs {
            std::cmp::Ordering::Less
          } else {
            std::cmp::Ordering::Greater
          };
        }
        // Purely numeric constant non-pair term sorts before pair terms
        // (e.g. (1+Pi)/2 comes before x, matching Wolfram).
        // Uses is_numeric_constant (not just !has_free_variables) so that
        // symbolic function calls like f[0] are not treated as constants.
        if is_numeric_constant(&none_base) {
          return if a_has_pairs {
            std::cmp::Ordering::Greater
          } else {
            std::cmp::Ordering::Less
          };
        }
        // For compound algebraic terms with free variables, compare by
        // latest variable relative to the monomial's variable:
        //   latest > monomial → monomial first (e.g. b + Sqrt[a*c])
        //   latest < monomial → compound first (e.g. Sqrt[a] + b)
        //   latest == monomial → monomial first if compound has extra
        //     variables (e.g. b + Sqrt[a+b]), compound first otherwise
        //     (e.g. Sqrt[x] + x)
        if has_free_variables(&none_base) {
          let (_, pair_base) = decompose_term(pair_term);
          let pair_earliest = extract_earliest_variable(&pair_base);
          let none_latest = extract_latest_variable(&none_base);
          if let (Some(pv), Some(nlv)) = (&pair_earliest, &none_latest) {
            if nlv > pv {
              // Non-pair has a later variable → monomial first
              return if a_has_pairs {
                std::cmp::Ordering::Less
              } else {
                std::cmp::Ordering::Greater
              };
            }
            if nlv == pv && count_free_variables(&none_base) > 1 {
              // Same latest variable but compound has extra vars → monomial first
              return if a_has_pairs {
                std::cmp::Ordering::Less
              } else {
                std::cmp::Ordering::Greater
              };
            }
          }
          let none_earliest = extract_earliest_variable(&none_base);
          if let (Some(pv), Some(nv)) = (&pair_earliest, &none_earliest)
            && nv < pv
          {
            // All non-pair variables < monomial's → non-pair first
            return if a_has_pairs {
              std::cmp::Ordering::Greater
            } else {
              std::cmp::Ordering::Less
            };
          }
        }
        // Fall back: pair term sorts first (before the non-pair term).
        // This is consistent with the "no free vars → non-pair first"
        // early return above: constants are the only terms that sort
        // before polynomials; all other non-pair terms sort after.
        // Using a simple consistent rule here prevents transitivity
        // violations that occur when canonical comparison disagrees
        // with the early returns above.
        return if a_has_pairs {
          std::cmp::Ordering::Less
        } else {
          std::cmp::Ordering::Greater
        };
      }
      // When both terms lack polynomial pairs, sort purely numeric
      // constants before non-constant terms. This mirrors the
      // asymmetric block's "numeric constant → non-pair first" rule
      // and prevents transitivity cycles: without it, a constant C,
      // a polynomial P, and a non-pair term T with free vars can form
      // C < P (asymmetric), T > P (asymmetric fallback), C > T
      // (canonical), violating C < P < T ↔ C < T.
      if !a_has_pairs && !b_has_pairs && pa == 0 {
        let a_const = is_numeric_constant(&base_a);
        let b_const = is_numeric_constant(&base_b);
        if a_const && !b_const {
          return std::cmp::Ordering::Less;
        }
        if b_const && !a_const {
          return std::cmp::Ordering::Greater;
        }
      }
      // When both terms lack polynomial pairs, are algebraic (priority 0),
      // and have free variables, compare by earliest variable name
      // (Wolfram sorts by leading variable). Skip for transcendental
      // functions where function-name ordering takes priority.
      // Also skip when both bases are function calls with the same name
      // (e.g. f[x] vs f[h+x]) — structural comparison is more accurate there.
      let same_fn_head = matches!(
        (&base_a, &base_b),
        (Expr::FunctionCall { name: na, .. }, Expr::FunctionCall { name: nb, .. }) if na == nb
      );
      if !a_has_pairs && !b_has_pairs && pa == 0 && pb == 0 && !same_fn_head {
        let a_var = extract_earliest_variable(&base_a);
        let b_var = extract_earliest_variable(&base_b);
        if let (Some(av), Some(bv)) = (&a_var, &b_var) {
          let cmp = av.cmp(bv);
          if cmp != std::cmp::Ordering::Equal {
            return cmp;
          }
        }
      }
      // When both terms are transcendental (priority >= 1) and lack polynomial pairs,
      // compare by primary function name, then by polynomial degree within the product.
      // This sorts x*Cos[x] before Sin[x] (because "Cos" < "Sin"), matching Wolfram.
      if !a_has_pairs && !b_has_pairs && pa >= 1 && pb >= 1 {
        let fn_a = extract_primary_fn_name(&base_a);
        let fn_b = extract_primary_fn_name(&base_b);
        if let (Some(ref na), Some(ref nb)) = (fn_a, fn_b) {
          let cmp = na.cmp(nb);
          if cmp != std::cmp::Ordering::Equal {
            return cmp;
          }
          // Same function: compare by polynomial degree (lower degree first)
          let deg_a = extract_poly_degree_in_product(&base_a);
          let deg_b = extract_poly_degree_in_product(&base_b);
          let cmp = deg_a
            .partial_cmp(&deg_b)
            .unwrap_or(std::cmp::Ordering::Equal);
          if cmp != std::cmp::Ordering::Equal {
            return cmp;
          }
          // For products of transcendental functions (e.g. Cos[b]*Sin[a] vs
          // Cos[a]*Sin[b]), Wolfram sorts by the highest-alphabetical trig factor
          // first, then by argument, then by exponent.
          let cmp = compare_trig_products(&base_a, &base_b);
          if cmp != std::cmp::Ordering::Equal {
            return cmp;
          }
        }
      }
      // For Times terms with complex-number-literal coefficients (e.g. (1+I)*f[x]),
      // compare by the non-coefficient part first, matching Wolfram's behavior.
      if let (Some(stripped_a), Some(stripped_b)) = (
        strip_complex_literal_coeff(&base_a),
        strip_complex_literal_coeff(&base_b),
      ) {
        let cmp = compare_expr_canonical(&stripped_a, &stripped_b);
        if cmp != std::cmp::Ordering::Equal {
          return cmp;
        }
      }
      // Fall back: try structural comparison of function call arguments,
      // then string comparison for non-polynomial terms.
      // This ensures e.g. Log[1 - x] sorts before Log[1 + x] to match Wolfram.
      let cmp = compare_expr_canonical(&base_a, &base_b);
      if cmp != std::cmp::Ordering::Equal {
        return cmp;
      }
      let sa = term_sort_key(a);
      let sb = term_sort_key(b);
      let cmp = sa.cmp(&sb);
      if cmp != std::cmp::Ordering::Equal {
        return cmp;
      }
      // Tiebreaker: positive coefficients before negative
      let (coeff_a, _) = decompose_term(a);
      let (coeff_b, _) = decompose_term(b);
      coeff_a.is_negative().cmp(&coeff_b.is_negative())
    }
  }
}

/// Strip a leading complex-number-literal factor from a Times expression.
/// E.g. Times[(1+I), Sqrt[Pi/2], DiracDelta[w]] → Times[Sqrt[Pi/2], DiracDelta[w]]
/// Returns None if no complex literal is found.
fn strip_complex_literal_coeff(e: &Expr) -> Option<Expr> {
  fn is_complex_literal(e: &Expr) -> bool {
    match e {
      Expr::FunctionCall { name, args } if name == "Plus" => {
        args.iter().all(|a| match a {
          Expr::Integer(_) | Expr::Real(_) => true,
          Expr::Constant(s) if s == "I" => true,
          Expr::FunctionCall { name: tn, args: ta } if tn == "Times" => {
            ta.iter().all(|t| {
              matches!(t, Expr::Integer(_) | Expr::Real(_))
                || matches!(t, Expr::Constant(s) if s == "I")
            })
          }
          _ => false,
        })
      }
      _ => false,
    }
  }

  if let Expr::FunctionCall { name, args } = e
    && name == "Times"
    && args.len() >= 2
    && is_complex_literal(&args[0])
  {
    let rest = &args[1..];
    return Some(if rest.len() == 1 {
      rest[0].clone()
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: rest.to_vec(),
      }
    });
  }
  None
}

/// Extract the primary transcendental function name from a term.
/// For `Sin[x]` → Some("Sin"), for `x*Cos[x]` → Some("Cos") (looking inside Times).
/// Returns None for non-transcendental terms.
fn extract_primary_fn_name(e: &Expr) -> Option<String> {
  match e {
    // Look inside Times products for the earliest (alphabetically) transcendental function
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut best: Option<String> = None;
      for arg in args {
        if let Some(n) = extract_primary_fn_name(arg) {
          best = Some(match best {
            Some(b) if b <= n => b,
            _ => n,
          });
        }
      }
      best
    }
    // Power[f, n] — look inside the base
    Expr::FunctionCall { name, args }
      if name == "Power" && !args.is_empty() =>
    {
      extract_primary_fn_name(&args[0])
    }
    // Bare transcendental function
    Expr::FunctionCall { name, .. } if term_priority(e) >= 1 => {
      Some(name.clone())
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      let l = extract_primary_fn_name(left);
      let r = extract_primary_fn_name(right);
      match (l, r) {
        (Some(a), Some(b)) => Some(if a <= b { a } else { b }),
        (Some(a), None) => Some(a),
        (None, b) => b,
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      ..
    } => extract_primary_fn_name(left),
    _ => None,
  }
}

/// Extract (function_name, argument, exponent) triples from a product of transcendental
/// functions. For `Cos[x]^2*Sin[x]` → [("Cos", x, 2), ("Sin", x, 1)].
fn extract_trig_factors(e: &Expr) -> Vec<(String, Expr, i64)> {
  let mut factors = Vec::new();
  let extract_one = |e: &Expr, factors: &mut Vec<(String, Expr, i64)>| match e {
    Expr::FunctionCall { name, args }
      if term_priority(e) >= 1 && args.len() == 1 =>
    {
      factors.push((name.clone(), args[0].clone(), 1));
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Expr::FunctionCall {
        name: fn_name,
        args: fn_args,
      } = &args[0]
        && term_priority(&args[0]) >= 1
        && fn_args.len() == 1
      {
        let exp = match &args[1] {
          Expr::Integer(n) => *n as i64,
          _ => 0,
        };
        factors.push((fn_name.clone(), fn_args[0].clone(), exp));
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::FunctionCall {
        name: fn_name,
        args: fn_args,
      } = left.as_ref()
        && term_priority(left) >= 1
        && fn_args.len() == 1
      {
        let exp = match right.as_ref() {
          Expr::Integer(n) => *n as i64,
          _ => 0,
        };
        factors.push((fn_name.clone(), fn_args[0].clone(), exp));
      }
    }
    _ => {}
  };
  match e {
    Expr::FunctionCall { name, args } if name == "Times" => {
      for arg in args {
        extract_one(arg, &mut factors);
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      extract_one(left, &mut factors);
      extract_one(right, &mut factors);
    }
    _ => extract_one(e, &mut factors),
  }
  // Sort by function name descending (highest-alphabetical first)
  factors.sort_by(|a, b| b.0.cmp(&a.0));
  factors
}

/// Compare two transcendental product terms by their trig factors.
/// Wolfram sorts by the highest-alphabetical function factor first (e.g., Sin before Cos),
/// then by argument, then by exponent.
fn compare_trig_products(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  let fa = extract_trig_factors(a);
  let fb = extract_trig_factors(b);
  if fa.is_empty() || fb.is_empty() {
    return std::cmp::Ordering::Equal;
  }
  for (ta, tb) in fa.iter().zip(fb.iter()) {
    // Compare function name (descending sort means we compare in ascending for position)
    let cmp = ta.0.cmp(&tb.0);
    if cmp != std::cmp::Ordering::Equal {
      return cmp;
    }
    // Compare argument
    let cmp = compare_expr_canonical(&ta.1, &tb.1);
    if cmp != std::cmp::Ordering::Equal {
      return cmp;
    }
    // Compare exponent (lower first)
    let cmp = ta.2.cmp(&tb.2);
    if cmp != std::cmp::Ordering::Equal {
      return cmp;
    }
  }
  fa.len().cmp(&fb.len())
}

/// Extract the maximum polynomial degree from a Times product's variable factors.
/// For `x*Cos[x]` → 1.0, `x^2*Sin[x]` → 2.0, `Sin[x]` → 0.0.
fn extract_poly_degree_in_product(e: &Expr) -> f64 {
  match e {
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut max_deg = 0.0f64;
      for arg in args {
        if let Some(pairs) = extract_var_exp_pairs(arg) {
          for (_, exp) in pairs {
            max_deg = max_deg.max(exp);
          }
        }
      }
      max_deg
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      let mut deg: f64 = 0.0;
      if let Some(pairs) = extract_var_exp_pairs(left) {
        for (_, exp) in pairs {
          deg = deg.max(exp);
        }
      }
      if let Some(pairs) = extract_var_exp_pairs(right) {
        for (_, exp) in pairs {
          deg = deg.max(exp);
        }
      }
      deg
    }
    _ => 0.0,
  }
}

/// Compare two expressions using Wolfram-compatible canonical ordering.
/// Used as a tie-breaker when Plus term sorting can't distinguish terms
/// via polynomial structure alone.
///
/// Rules:
///   - Integers: by value
///   - Identifiers: alphabetically
///   - FunctionCall with same name: compare arguments element by element
///   - BinaryOp::Plus/Times: compare left then right
///   - UnaryOp::Minus(x) < x (negative sorts before positive)
///   - Fall back to string comparison
fn compare_expr_canonical(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  use std::cmp::Ordering;

  // Assign a type tag for top-level ordering
  fn type_tag(e: &Expr) -> u8 {
    match e {
      Expr::Integer(_) => 0,
      Expr::Real(_) => 1,
      Expr::Constant(_) => 2,
      Expr::Identifier(_) => 3,
      Expr::UnaryOp { .. } => 4,
      Expr::BinaryOp { .. } => 5,
      Expr::FunctionCall { .. } => 6,
      _ => 7,
    }
  }

  let ta = type_tag(a);
  let tb = type_tag(b);

  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) => x.cmp(y),
    (Expr::Real(x), Expr::Real(y)) => {
      x.partial_cmp(y).unwrap_or(Ordering::Equal)
    }
    (Expr::Identifier(x), Expr::Identifier(y)) => {
      let ord = crate::functions::list_helpers_ast::wolfram_string_order(x, y);
      if ord > 0 {
        Ordering::Less
      } else if ord < 0 {
        Ordering::Greater
      } else {
        Ordering::Equal
      }
    }
    (Expr::Constant(x), Expr::Constant(y)) => {
      let ord = crate::functions::list_helpers_ast::wolfram_string_order(x, y);
      if ord > 0 {
        Ordering::Less
      } else if ord < 0 {
        Ordering::Greater
      } else {
        Ordering::Equal
      }
    }
    // Cross-compare Constants and Identifiers using case-insensitive ordering
    // (Wolfram doesn't separate these by type in Times ordering)
    (Expr::Constant(x), Expr::Identifier(y))
    | (Expr::Identifier(x), Expr::Constant(y)) => {
      let ord = crate::functions::list_helpers_ast::wolfram_string_order(x, y);
      if ord > 0 {
        Ordering::Less
      } else if ord < 0 {
        Ordering::Greater
      } else {
        Ordering::Equal
      }
    }
    (
      Expr::FunctionCall { name: na, args: aa },
      Expr::FunctionCall { name: nb, args: ab },
    ) => {
      let cmp = na.cmp(nb);
      if cmp != Ordering::Equal {
        return cmp;
      }
      // Compare arguments element by element
      for (arg_a, arg_b) in aa.iter().zip(ab.iter()) {
        let cmp = compare_expr_canonical(arg_a, arg_b);
        if cmp != Ordering::Equal {
          return cmp;
        }
      }
      aa.len().cmp(&ab.len())
    }
    (
      Expr::BinaryOp {
        op: op_a,
        left: la,
        right: ra,
      },
      Expr::BinaryOp {
        op: op_b,
        left: lb,
        right: rb,
      },
    ) => {
      // Normalize Minus→Plus and Divide→Times for comparison,
      // matching Wolfram's canonical form.
      let (norm_op_a, norm_ra) =
        if *op_a == crate::syntax::BinaryOperator::Minus {
          (
            crate::syntax::BinaryOperator::Plus,
            Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: ra.clone(),
            },
          )
        } else if *op_a == crate::syntax::BinaryOperator::Divide {
          (
            crate::syntax::BinaryOperator::Times,
            Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: ra.clone(),
              right: Box::new(Expr::Integer(-1)),
            },
          )
        } else {
          (*op_a, *ra.clone())
        };
      let (norm_op_b, norm_rb) =
        if *op_b == crate::syntax::BinaryOperator::Minus {
          (
            crate::syntax::BinaryOperator::Plus,
            Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: rb.clone(),
            },
          )
        } else if *op_b == crate::syntax::BinaryOperator::Divide {
          (
            crate::syntax::BinaryOperator::Times,
            Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: rb.clone(),
              right: Box::new(Expr::Integer(-1)),
            },
          )
        } else {
          (*op_b, *rb.clone())
        };
      // Compare by canonical operator name (alphabetical) rather than
      // enum discriminant, so Power sorts before Times, matching Wolfram.
      let op_name = |op: &crate::syntax::BinaryOperator| -> &str {
        match op {
          crate::syntax::BinaryOperator::Plus => "Plus",
          crate::syntax::BinaryOperator::Minus => "Plus",
          crate::syntax::BinaryOperator::Times => "Times",
          crate::syntax::BinaryOperator::Divide => "Times",
          crate::syntax::BinaryOperator::Power => "Power",
          crate::syntax::BinaryOperator::And => "And",
          crate::syntax::BinaryOperator::Or => "Or",
          crate::syntax::BinaryOperator::StringJoin => "StringJoin",
          crate::syntax::BinaryOperator::Alternatives => "Alternatives",
        }
      };
      let cmp = op_name(&norm_op_a).cmp(op_name(&norm_op_b));
      if cmp != Ordering::Equal {
        return cmp;
      }
      let cmp = compare_expr_canonical(la, lb);
      if cmp != Ordering::Equal {
        return cmp;
      }
      compare_expr_canonical(&norm_ra, &norm_rb)
    }
    (
      Expr::UnaryOp { op: _, operand: oa },
      Expr::UnaryOp { op: _, operand: ob },
    ) => compare_expr_canonical(oa, ob),
    // Negated vs non-negated: -x sorts before x
    // Handle both UnaryOp::Minus and Times[-1, x] / Times[negative, x]
    _ if is_negated_form(a) && !is_negated_form(b) => {
      let inner_a = strip_negation(a);
      let inner_cmp = compare_expr_canonical(&inner_a, b);
      if inner_cmp == Ordering::Equal {
        Ordering::Less // -x < x
      } else {
        inner_cmp
      }
    }
    _ if !is_negated_form(a) && is_negated_form(b) => {
      let inner_b = strip_negation(b);
      let inner_cmp = compare_expr_canonical(a, &inner_b);
      if inner_cmp == Ordering::Equal {
        Ordering::Greater // x > -x
      } else {
        inner_cmp
      }
    }
    _ => {
      if ta != tb {
        ta.cmp(&tb)
      } else {
        // Fall back to string comparison
        let sa = crate::syntax::expr_to_string(a);
        let sb = crate::syntax::expr_to_string(b);
        sa.cmp(&sb)
      }
    }
  }
}

/// Check if an expression is a negated form: UnaryOp::Minus(x), Times[-1, x],
/// Times[negative_integer, x], or BinaryOp::Times with -1.
fn is_negated_form(e: &Expr) -> bool {
  match e {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      ..
    } => true,
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      matches!(&args[0], Expr::Integer(n) if *n < 0)
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      ..
    } => matches!(left.as_ref(), Expr::Integer(n) if *n < 0),
    _ => false,
  }
}

/// Strip negation from a negated form, returning the positive inner expression.
fn strip_negation(e: &Expr) -> Expr {
  match e {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => *operand.clone(),
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      if let Expr::Integer(n) = &args[0] {
        let pos_n = -n;
        if pos_n == 1 && args.len() == 2 {
          args[1].clone()
        } else {
          let mut new_args = args.clone();
          new_args[0] = Expr::Integer(pos_n);
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: new_args,
          }
        }
      } else {
        e.clone()
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if let Expr::Integer(n) = left.as_ref() {
        let pos_n = -n;
        if pos_n == 1 {
          *right.clone()
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(pos_n)),
            right: right.clone(),
          }
        }
      } else {
        e.clone()
      }
    }
    _ => e.clone(),
  }
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
      left,
      ..
    } => term_priority(left),
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
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Times" => args.iter().map(term_priority).max().unwrap_or(0),
      "Power" if !args.is_empty() => term_priority(&args[0]),
      "Plus" | "Rational" => 0,
      "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc" | "Sinh" | "Cosh"
      | "Tanh" | "Coth" | "Sech" | "Csch" | "ArcSin" | "ArcCos" | "ArcTan"
      | "ArcCot" | "ArcSec" | "ArcCsc" | "Exp" | "Log" | "Factorial"
      | "Erf" | "Erfc" => 1,
      _ => 0,
    },
    Expr::UnaryOp { operand, .. } => term_priority(operand),
    // CurriedCall like Derivative[1][f][x] — treat as algebraic (priority 0)
    Expr::CurriedCall { .. } => 0,
    _ => 0,
  }
}

/// Sub-priority for Times factor ordering: identifiers before compound expressions.
/// This ensures simple symbols sort before sums/products, matching Wolfram behavior.
pub fn times_factor_subpriority(e: &Expr) -> i32 {
  match e {
    // Imaginary unit I sorts before all other symbolic factors
    Expr::Identifier(name) if name == "I" => -2,
    Expr::Identifier(_) | Expr::Constant(_) => 0,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      ..
    } => {
      let base_sp = times_factor_subpriority(left);
      // Power[Plus[...], n] sorts alongside identifiers (subpriority 0),
      // not after them (subpriority 1). Wolfram puts (1+x)^2 before plain s.
      // Power[func_call, n] (e.g. Sin[x]^2) keeps the function call subpriority
      // so it sorts alongside other function calls.
      if base_sp == 1 { 0 } else { base_sp }
    }
    Expr::BinaryOp {
      op:
        crate::syntax::BinaryOperator::Plus | crate::syntax::BinaryOperator::Minus,
      ..
    } => 1,
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Times" => args.iter().map(times_factor_subpriority).max().unwrap_or(0),
      "Rational" => 0,
      // Complex number literals (1 + I, 1 - I, etc.) sort like numeric constants
      "Plus"
        if args.iter().all(|a| match a {
          Expr::Integer(_) | Expr::Real(_) => true,
          Expr::Constant(s) if s == "I" => true,
          Expr::FunctionCall { name: tn, args: ta } if tn == "Times" => {
            ta.iter().all(|t| {
              matches!(t, Expr::Integer(_) | Expr::Real(_))
                || matches!(t, Expr::Constant(s) if s == "I")
            })
          }
          _ => false,
        }) =>
      {
        -1
      }
      "Plus" => 1,
      // Power[base, exp] and Sqrt[base] sort like their base (same as BinaryOp::Power)
      "Sqrt" if args.len() == 1 => {
        let base_sp = times_factor_subpriority(&args[0]);
        if base_sp == 1 { 0 } else { base_sp }
      }
      "Power" if args.len() == 2 => {
        let base_sp = times_factor_subpriority(&args[0]);
        if base_sp == 1 { 0 } else { base_sp }
      }
      // Numeric-constant function calls (e.g. Log[2], Sqrt[3]) sort before variables
      // but after the imaginary unit I. Only applies to known math functions.
      "Log" | "Sqrt" | "Sin" | "Cos" | "Tan" | "Exp" | "Abs"
        if args.iter().all(|a| {
          matches!(a, Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_))
        }) =>
      {
        -1
      }
      _ => 2,
    },
    // CurriedCall (e.g. Derivative[1][y][x]) sorts after regular function calls
    Expr::CurriedCall { .. } => 3,
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
      // Special case: an additive expression sorts BEFORE an identifier when:
      // 1. The identifier appears with a negative coefficient: (x-y)*y not y*(x-y)
      // 2. The additive is Plus[n, var] where n < 0 and var is the same identifier:
      //    (-3+x)*x not x*(-3+x)   (monic linear factor with negative constant)
      // This matches Wolfram's canonical ordering.
      if sa == 0 && sb == 1 {
        // a is Identifier, b is additive
        if additive_contains_negated(b, a) || additive_is_neg_const_plus_ident(b, a) {
          return std::cmp::Ordering::Greater; // b (additive) before a (identifier)
        }
      } else if sa == 1 && sb == 0 {
        // a is additive, b is Identifier
        if additive_contains_negated(a, b) || additive_is_neg_const_plus_ident(a, b) {
          return std::cmp::Ordering::Less; // a (additive) before b (identifier)
        }
      }
      return sa.cmp(&sb);
    }
    // For FunctionCall with same head, compare arguments structurally
    if let (
      Expr::FunctionCall { name: na, args: aa },
      Expr::FunctionCall { name: nb, args: ab },
    ) = (a, b)
      && na == nb
    {
      // For Power, use sort-key comparison on bases to match Wolfram's
      // Times ordering (e.g. Power[Plus[a,b],-2] before Power[s,-1]
      // because sort key "b" < "s")
      if na == "Power" && aa.len() == 2 && ab.len() == 2 {
        let ak = crate::functions::list_helpers_ast::sorting::expr_sort_key(&aa[0]);
        let bk = crate::functions::list_helpers_ast::sorting::expr_sort_key(&ab[0]);
        let ord = crate::functions::list_helpers_ast::wolfram_string_order(&ak, &bk);
        if ord > 0 {
          return std::cmp::Ordering::Less;
        }
        if ord < 0 {
          return std::cmp::Ordering::Greater;
        }
        // Equal sort keys: tie-break with full base string comparison
        // e.g. Plus[-1, x] vs Plus[1, x] both have sort key "x"
        // but "-1 + x" < "1 + x" lexicographically
        let full_a = crate::syntax::expr_to_string(&aa[0]);
        let full_b = crate::syntax::expr_to_string(&ab[0]);
        let full_ord = crate::functions::list_helpers_ast::wolfram_string_order(&full_a, &full_b);
        if full_ord > 0 {
          return std::cmp::Ordering::Less;
        }
        if full_ord < 0 {
          return std::cmp::Ordering::Greater;
        }
        // Equal bases: compare exponents
        let exp_ord = crate::functions::list_helpers_ast::compare_exprs(&aa[1], &ab[1]);
        if exp_ord > 0 {
          return std::cmp::Ordering::Less;
        }
        if exp_ord < 0 {
          return std::cmp::Ordering::Greater;
        }
        return std::cmp::Ordering::Equal;
      }
      // Same head: compare arguments using Wolfram canonical ordering
      for (arg_a, arg_b) in aa.iter().zip(ab.iter()) {
        let ord =
          crate::functions::list_helpers_ast::compare_exprs(arg_a, arg_b);
        if ord > 0 {
          return std::cmp::Ordering::Less;
        }
        if ord < 0 {
          return std::cmp::Ordering::Greater;
        }
      }
      return aa.len().cmp(&ab.len());
    }
    // For power-like expressions, use sort-key comparison on bases
    // to match Wolfram's Times ordering. This handles both same-variant
    // (e.g. both BinaryOp::Power) and cross-variant (BinaryOp vs FunctionCall) cases.
    let is_power_like = |e: &Expr| -> bool {
      matches!(e, Expr::BinaryOp { op: crate::syntax::BinaryOperator::Power, .. })
        || matches!(e, Expr::FunctionCall { name, args } if (name == "Power" && args.len() == 2))
        || is_sqrt(e).is_some()
    };
    if is_power_like(a) && is_power_like(b) {
      let ak = crate::functions::list_helpers_ast::sorting::expr_sort_key(a);
      let bk = crate::functions::list_helpers_ast::sorting::expr_sort_key(b);
      let ord = crate::functions::list_helpers_ast::wolfram_string_order(&ak, &bk);
      if ord > 0 {
        return std::cmp::Ordering::Less;
      } else if ord < 0 {
        return std::cmp::Ordering::Greater;
      }
      // Equal sort keys: when both bases are function calls with the same head
      // (e.g. f[1] vs f[2]), compare structurally to disambiguate.
      // This matches Wolfram's ordering where Power[f[1], -2] sorts before
      // Power[f[2], 2] (base argument 1 < 2).
      let (base_a, exp_a) = extract_base_exponent(a);
      let (base_b, exp_b) = extract_base_exponent(b);
      let same_fn_base = matches!(
        (&base_a, &base_b),
        (Expr::FunctionCall { name: na, .. }, Expr::FunctionCall { name: nb, .. })
          if na == nb && na != "Plus" && na != "Times"
      );
      // For Plus bases with the same arity, compare structurally.
      // e.g. Plus[-1, x] vs Plus[1, x]: both have sort key "x" and same
      // arg count, but -1 < 1 so (-1+x)^(-1/2) should sort before (1+x)^(-1/2).
      // Don't do this for different arities — let string-length fallback handle
      // e.g. Plus[a, b] vs Plus[1, a, b] where shorter is "simpler".
      let same_arity_plus_base = matches!(
        (&base_a, &base_b),
        (Expr::FunctionCall { name: na, args: aa }, Expr::FunctionCall { name: nb, args: ab })
          if na == "Plus" && nb == "Plus" && aa.len() == ab.len()
      ) || matches!(
        (&base_a, &base_b),
        (Expr::BinaryOp { op: crate::syntax::BinaryOperator::Plus, .. },
         Expr::BinaryOp { op: crate::syntax::BinaryOperator::Plus, .. })
      );
      if same_fn_base || same_arity_plus_base {
        let base_cmp = compare_expr_canonical(&base_a, &base_b);
        if base_cmp != std::cmp::Ordering::Equal {
          return base_cmp;
        }
        // Same base: compare exponents
        let exp_ord = crate::functions::list_helpers_ast::compare_exprs(&exp_a, &exp_b);
        if exp_ord > 0 {
          return std::cmp::Ordering::Less;
        } else if exp_ord < 0 {
          return std::cmp::Ordering::Greater;
        }
      }
      // Fall back to string length then alphabetical
      let as_str = crate::syntax::expr_to_string(a);
      let bs_str = crate::syntax::expr_to_string(b);
      let len_cmp = as_str.len().cmp(&bs_str.len());
      if len_cmp != std::cmp::Ordering::Equal {
        return len_cmp;
      }
      let str_ord = crate::functions::list_helpers_ast::wolfram_string_order(&as_str, &bs_str);
      if str_ord > 0 {
        return std::cmp::Ordering::Less;
      } else if str_ord < 0 {
        return std::cmp::Ordering::Greater;
      }
      // Exactly equal: fall through
    }
    if std::mem::discriminant(a) == std::mem::discriminant(b) {
      compare_expr_canonical(a, b)
    } else {
      // Use sort-key based comparison with wolfram_string_order (case-insensitive)
      // to avoid ASCII case artifacts (e.g. 'S' < 'a' in ASCII would incorrectly
      // put Sqrt[x] before identifier a).
      let ak = crate::functions::list_helpers_ast::sorting::expr_sort_key(a);
      let bk = crate::functions::list_helpers_ast::sorting::expr_sort_key(b);
      let ord = crate::functions::list_helpers_ast::wolfram_string_order(&ak, &bk);
      if ord > 0 {
        std::cmp::Ordering::Less
      } else if ord < 0 {
        std::cmp::Ordering::Greater
      } else {
        // Equal sort keys: fall back to full string comparison
        crate::syntax::expr_to_string(a).cmp(&crate::syntax::expr_to_string(b))
      }
    }
  });
}

/// Check if an additive expression contains a given identifier with a negative coefficient.
/// For example, `(x - y)` contains `y` negated, but `(x + y)` does not.
pub fn additive_contains_negated(additive: &Expr, ident: &Expr) -> bool {
  let ident_name = match ident {
    Expr::Identifier(name) => name.as_str(),
    Expr::Constant(name) => name.as_str(),
    _ => return false,
  };
  match additive {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      right,
      ..
    } => {
      // a - b: check if b == ident
      matches!(right.as_ref(), Expr::Identifier(name) if name == ident_name)
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left,
      right,
    } => {
      // Check if either side is -ident (UnaryMinus(ident) or Times(-1, ident))
      has_negated_ident(left, ident_name)
        || has_negated_ident(right, ident_name)
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().any(|arg| has_negated_ident(arg, ident_name))
    }
    _ => false,
  }
}

/// Check if an expression is a negated form of the given identifier.
fn has_negated_ident(expr: &Expr, ident_name: &str) -> bool {
  match expr {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      matches!(operand.as_ref(), Expr::Identifier(name) if name == ident_name)
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      (matches!(left.as_ref(), Expr::Integer(n) if *n < 0)
        && matches!(right.as_ref(), Expr::Identifier(name) if name == ident_name))
        || (matches!(right.as_ref(), Expr::Integer(n) if *n < 0)
          && matches!(left.as_ref(), Expr::Identifier(name) if name == ident_name))
    }
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && args.iter().any(|a| matches!(a, Expr::Integer(n) if *n < 0))
        && args
          .iter()
          .any(|a| matches!(a, Expr::Identifier(n) if n == ident_name)) =>
    {
      true
    }
    _ => false,
  }
}

/// Check if `additive` is Plus[n, var] where n is a negative number
/// and var is the same identifier as `ident`.
/// This handles the case (-3 + x)*x where the constant is negative
/// but x itself is not negated in the Plus.
pub fn additive_is_neg_const_plus_ident(additive: &Expr, ident: &Expr) -> bool {
  let ident_name = match ident {
    Expr::Identifier(name) | Expr::Constant(name) => name.as_str(),
    _ => return false,
  };

  let is_same_ident = |a: &Expr| -> bool {
    matches!(a, Expr::Identifier(n) | Expr::Constant(n) if n == ident_name)
  };

  // Check for FunctionCall Plus with exactly 2 args: negative number and same variable
  if let Expr::FunctionCall { name, args } = additive
    && name == "Plus"
    && args.len() == 2
  {
    let has_neg_num =
      args.iter().any(|a| matches!(a, Expr::Integer(n) if *n < 0));
    let has_ident = args.iter().any(&is_same_ident);
    return has_neg_num && has_ident;
  }

  // Check for BinaryOp Plus with negative number and same variable
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left,
    right,
  } = additive
  {
    let (l, r) = (left.as_ref(), right.as_ref());
    let has_neg_num = matches!(l, Expr::Integer(n) if *n < 0)
      || matches!(r, Expr::Integer(n) if *n < 0);
    let has_ident = is_same_ident(l) || is_same_ident(r);
    return has_neg_num && has_ident;
  }

  false
}

/// Multiply two exponent expressions, simplifying common cases.
/// E.g. Rational[1,2] * Integer(-1) → Rational[-1,2]
fn multiply_exponents(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x * y),
    (Expr::FunctionCall { name, args }, Expr::Integer(m))
    | (Expr::Integer(m), Expr::FunctionCall { name, args })
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        make_rational(n * m, *d)
      } else {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(a.clone()),
          right: Box::new(b.clone()),
        }
      }
    }
    _ => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}

/// Extract (base, exponent) from an expression for power combining in Times.
/// Check if an expression is "numeric-like" — consists only of integers,
/// constants (E, Pi), rationals, and products/powers thereof.
/// Used to decide whether Sqrt merging is safe (avoids merging Sqrt[x]*Sqrt[y]).
fn is_numeric_like(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => true,
    Expr::Constant(_) => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      true
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().all(is_numeric_like)
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => is_numeric_like(left) && is_numeric_like(right),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => is_numeric_like(left) && is_numeric_like(right),
    _ => false,
  }
}

/// x → (x, 1), x^n → (x, n), Sqrt[x] → (x, 1/2)
/// Power[Sqrt[x], n] → (x, n/2), Power[x^a, b] → (x, a*b)
pub fn extract_base_exponent(expr: &Expr) -> (Expr, Expr) {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      // Only recurse into nested powers when the outer exponent is an integer.
      // For symbolic exponents like (x^(-1))^(1+n/2), keep the inner Power
      // as the base to match Wolfram's canonical form.
      if matches!(right.as_ref(), Expr::Integer(_)) {
        let (inner_base, inner_exp) = extract_base_exponent(left);
        if matches!(inner_exp, Expr::Integer(1)) {
          (*left.clone(), *right.clone())
        } else {
          (inner_base, multiply_exponents(&inner_exp, right))
        }
      } else {
        (*left.clone(), *right.clone())
      }
    }
    expr if is_sqrt(expr).is_some() => {
      (is_sqrt(expr).unwrap().clone(), make_rational(1, 2))
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      // Only recurse when outer exponent is integer
      if matches!(&args[1], Expr::Integer(_)) {
        let (inner_base, inner_exp) = extract_base_exponent(&args[0]);
        if matches!(inner_exp, Expr::Integer(1)) {
          (args[0].clone(), args[1].clone())
        } else {
          (inner_base, multiply_exponents(&inner_exp, &args[1]))
        }
      } else {
        (args[0].clone(), args[1].clone())
      }
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
      Expr::FunctionCall { .. } => true,
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        ..
      }
      | Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        ..
      } => true,
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
        // But Infinity^0 and ComplexInfinity^0 are Indeterminate
        let is_inf = matches!(&base, Expr::Identifier(s) if s == "Infinity" || s == "ComplexInfinity");
        if is_inf {
          return Ok(vec![Expr::Identifier("Indeterminate".to_string())]);
        }
        continue;
      }
      result.push(power_two(&base, &combined_exp)?);
    }
  }
  result.extend(non_combinable);

  // Second pass: combine bases with the same fractional exponent
  // e.g. Sqrt[2] * Sqrt[3] = 2^(1/2) * 3^(1/2) → 6^(1/2) = Sqrt[6]
  // Also handles Sqrt[E] * Sqrt[2] → Sqrt[2*E], etc.
  let mut combined: Vec<Expr> = Vec::new();
  let mut used = vec![false; result.len()];
  for i in 0..result.len() {
    if used[i] {
      continue;
    }
    let (base_i, exp_i) = extract_base_exponent(&result[i]);
    // Only combine bases that are numeric-like (integers, constants, or products thereof)
    // This avoids combining purely symbolic expressions like Sqrt[x]*Sqrt[y]
    let is_combinable_base = is_numeric_like(&base_i);
    let is_rational_exp = matches!(
      &exp_i,
      Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2
    );
    if !is_combinable_base || !is_rational_exp {
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
      if is_numeric_like(&base_j)
        && crate::syntax::expr_to_string(&exp_j) == exp_key
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

  // Handle Interval arithmetic
  if let Some(result) = crate::functions::interval_ast::try_interval_times(args)
  {
    return result;
  }

  // Flatten nested Times arguments (including BinaryOp forms)
  let mut flat_args: Vec<Expr> = Vec::new();
  fn flatten_times(expr: &Expr, out: &mut Vec<Expr>) {
    match expr {
      Expr::FunctionCall {
        name,
        args: inner_args,
      } if name == "Times" => {
        for a in inner_args {
          flatten_times(a, out);
        }
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left,
        right,
      } => {
        flatten_times(left, out);
        flatten_times(right, out);
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left,
        right,
      } => {
        // a/b → a * b^(-1)
        flatten_times(left, out);
        out.push(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: right.clone(),
          right: Box::new(Expr::Integer(-1)),
        });
      }
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand,
      } => {
        // -x → (-1) * x
        out.push(Expr::Integer(-1));
        flatten_times(operand, out);
      }
      _ => out.push(expr.clone()),
    }
  }
  for arg in args {
    flatten_times(arg, &mut flat_args);
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

  // Pre-check: is there an explicit Rational or non-unity Integer coefficient among the args?
  // If so, we can absorb integer denominators from (expr/d) terms into the coefficient.
  // This simplifies e.g. Times[Rational(-1,2), x^3/3] → Rational(-1,6)*x^3
  // but leaves Times[I, Pi/2] intact (no explicit rational coefficient).
  let has_rational_coeff_in_args = args.iter().any(|a| {
    matches!(a, Expr::FunctionCall { name, .. } if name == "Rational")
      || matches!(a, Expr::Integer(n) if *n != 1)
  });

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
      // Power[Integer(n), Integer(neg)] → absorb into rational coefficient
      // e.g. 2^(-1) → rat_denom *= 2, or 3^(-2) → rat_denom *= 9
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: base,
        right: exp,
      } if matches!(base.as_ref(), Expr::Integer(n) if *n != 0)
        && matches!(exp.as_ref(), Expr::Integer(e) if *e < 0) =>
      {
        if let (Expr::Integer(n), Expr::Integer(e)) =
          (base.as_ref(), exp.as_ref())
        {
          let abs_e = (-e) as u32;
          if let Some(pow) = n.checked_pow(abs_e) {
            if let Some(rd) = rat_denom.checked_mul(pow) {
              rat_denom = rd;
              has_rational = true;
            } else {
              int_overflow = true;
            }
          } else {
            symbolic_args.push(arg.clone());
          }
        }
      }
      _ => {
        // If arg is (symbolic_expr / Integer(d)) AND there is already a rational or
        // non-unity integer coefficient among the args, absorb d into the coefficient.
        // E.g. Times[Rational(-1,2), x^3/3] → Rational(-1,6)*x^3
        // But leave Times[I, Pi/2] unchanged (no explicit rational coefficient).
        if has_rational_coeff_in_args
          && let Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: num_expr,
            right: den_expr,
          } = arg
          && let Expr::Integer(d) = den_expr.as_ref()
          && *d != 0
          && expr_to_num(num_expr).is_none()
        {
          if let Some(rd) = rat_denom.checked_mul(*d) {
            rat_denom = rd;
            has_rational = true;
            symbolic_args.push(*num_expr.clone());
          } else {
            symbolic_args.push(arg.clone());
          }
        } else if let Some(n) = expr_to_num(arg) {
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
  let mut coeff = if has_rational || (has_int && combined_denom != 1) {
    make_rational(combined_numer, combined_denom)
  } else {
    Expr::Integer(int_product)
  };

  // If all arguments are numeric, return the product
  if symbolic_args.is_empty() {
    return Ok(coeff);
  }

  // 0 * Infinity = Indeterminate (check before the general 0 * anything = 0 rule)
  if combined_numer == 0 && symbolic_args.iter().any(is_infinity_like) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
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

  // Try to combine integer coefficient with same-base power in symbolic args.
  // Only absorb one factor of the base so the result stays in canonical form:
  // E.g. 2 * 2^(-1/2) → 1 * 2^(1/2) = Sqrt[2]
  // E.g. -2 * 2^(-1/2) → -1 * Sqrt[2]
  // E.g. 4 * 2^(-1/2) → 2 * Sqrt[2]  (absorb one factor of 2)
  if let Expr::Integer(c) = &coeff {
    let cv = *c;
    let abs_cv = cv.unsigned_abs();
    if abs_cv > 1 {
      for i in 0..symbolic_args.len() {
        let (base, exp) = extract_base_exponent(&symbolic_args[i]);
        if let Expr::Integer(b) = &base {
          let bv = *b;
          // Only absorb when exponent is negative — this converts e.g.
          // 2 * 2^(-1/2) → Sqrt[2], 4 * 2^(-1/2) → 2*Sqrt[2]
          // but leaves 2 * Sqrt[2] (exp=1/2) alone to avoid cycles.
          let exp_is_negative = match &exp {
            Expr::Integer(n) => *n < 0,
            Expr::FunctionCall { name, args: rargs }
              if name == "Rational" && rargs.len() == 2 =>
            {
              matches!(&rargs[0], Expr::Integer(n) if *n < 0)
            }
            _ => false,
          };
          if bv > 1 && abs_cv % (bv as u128) == 0 && exp_is_negative {
            // Only absorb one factor of base into the power
            let new_exp = plus_ast(&[Expr::Integer(1), exp])?;
            if matches!(&new_exp, Expr::Integer(0)) {
              symbolic_args.remove(i);
            } else if matches!(&new_exp, Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2 && matches!((&args[0], &args[1]), (Expr::Integer(1), Expr::Integer(2))))
            {
              symbolic_args[i] = crate::functions::math_ast::sqrt_ast(&[base])?;
            } else if matches!(&new_exp, Expr::Integer(1)) {
              symbolic_args[i] = base;
            } else {
              symbolic_args[i] = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(base),
                right: Box::new(new_exp),
              };
            }
            let sign = if cv < 0 { -1i128 } else { 1i128 };
            let remainder = (abs_cv / bv as u128) as i128;
            coeff = Expr::Integer(sign * remainder);
            break;
          }
        }
      }
    }
  }

  // Try to combine Rational coefficient with same-base positive power in symbolic args.
  // This absorbs the base factor from the denominator, converting positive exponent to negative.
  // E.g. Rational[1,6] * 2^(1/2) → Rational[1,3] * 2^(-1/2)  (since 6 = 2*3)
  // E.g. Rational[1,2] * 2^(1/2) → 2^(-1/2)
  if let Expr::FunctionCall {
    name: rname,
    args: rargs,
  } = &coeff
    && rname == "Rational"
    && rargs.len() == 2
    && let (Expr::Integer(cn), Expr::Integer(cd)) = (&rargs[0], &rargs[1])
  {
    let cd_abs = cd.unsigned_abs();
    if cd_abs > 1 {
      for i in 0..symbolic_args.len() {
        let (base, exp) = extract_base_exponent(&symbolic_args[i]);
        if let Expr::Integer(bv) = &base {
          let bv_abs = bv.unsigned_abs();
          // Only for positive fractional exponents and when denominator is divisible by base
          let exp_is_positive_frac = match &exp {
            Expr::FunctionCall { name: en, args: ea }
              if en == "Rational" && ea.len() == 2 =>
            {
              matches!(&ea[0], Expr::Integer(n) if *n > 0)
            }
            _ => false,
          };
          if bv_abs > 1 && exp_is_positive_frac && cd_abs % bv_abs == 0 {
            // Absorb one factor of base from denominator into the power
            let new_exp = plus_ast(&[Expr::Integer(-1), exp])?;
            if matches!(&new_exp, Expr::Integer(0)) {
              symbolic_args.remove(i);
            } else {
              symbolic_args[i] = power_two(&base, &new_exp)?;
            }
            let new_cd = *cd / *bv;
            coeff = make_rational(*cn, new_cd);
            break;
          }
        }
      }
    }
  }

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
    // Return unevaluated (like Wolfram) — error message emitted by centralized arg_count check
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

  // Infinity / Infinity → Indeterminate
  if is_infinity_like(a) && is_infinity_like(b) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
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

  // Handle Interval division
  if let Some(result) =
    crate::functions::interval_ast::try_interval_divide(a, b)
  {
    return result;
  }

  // expr / 1 → expr
  if matches!(b, Expr::Integer(1)) {
    return Ok(a.clone());
  }

  // For two integers, keep as Rational (fraction)
  if let (Expr::Integer(numer), Expr::Integer(denom)) = (a, b) {
    if *denom == 0 {
      // n/0 → ComplexInfinity (with warning), 0/0 → Indeterminate
      crate::emit_message(
        "                                 1\n\
         Power::infy: Infinite expression - encountered.\n\
         \x20                                0",
      );
      if *numer == 0 {
        crate::emit_message(
          "\nInfinity::indet: Indeterminate expression 0 ComplexInfinity encountered.",
        );
        return Ok(Expr::Identifier("Indeterminate".to_string()));
      }
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return Ok(make_rational(*numer, *denom));
  }

  // For BigInteger / BigInteger (or mixed Integer/BigInteger), reduce by GCD
  {
    use crate::functions::math_ast::number_theory::bigint_gcd;
    use num_bigint::BigInt;
    let a_big = expr_to_bigint(a);
    let b_big = expr_to_bigint(b);
    if let (Some(numer), Some(denom)) = (a_big, b_big) {
      use num_traits::Zero;
      if denom.is_zero() {
        crate::emit_message(
          "                                 1\n\
           Power::infy: Infinite expression - encountered.\n\
           \x20                                0",
        );
        if numer.is_zero() {
          crate::emit_message(
            "\nInfinity::indet: Indeterminate expression 0 ComplexInfinity encountered.",
          );
          return Ok(Expr::Identifier("Indeterminate".to_string()));
        }
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      let g = bigint_gcd(numer.clone(), denom.clone());
      let mut rn = &numer / &g;
      let mut rd = &denom / &g;
      // Normalize sign: put sign in numerator
      if rd < BigInt::from(0) {
        rn = -rn;
        rd = -rd;
      }
      if rd == BigInt::from(1) {
        return Ok(bigint_to_expr(rn));
      }
      return Ok(Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![bigint_to_expr(rn), bigint_to_expr(rd)],
      });
    }
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

  // n / Sqrt[m] → delegate to times_ast so coefficient combining works
  // E.g. 2/Sqrt[2] → Sqrt[2], 4/Sqrt[2] → 2*Sqrt[2]
  // Skip for |n|<=1 since 1/Sqrt[m] is already the canonical display form
  if let Expr::Integer(n) = a
    && n.unsigned_abs() > 1
    && is_sqrt(b).is_some()
    && matches!(is_sqrt(b).unwrap(), Expr::Integer(m) if *m > 0)
  {
    let b_inv = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(b.clone()),
      right: Box::new(Expr::Integer(-1)),
    };
    return times_ast(&[Expr::Integer(*n), b_inv]);
  }

  // Sqrt[n] / d → Sqrt[n/d^2] for integer n and d
  // E.g. Sqrt[6]/2 → Sqrt[3/2], matching Wolfram's canonical form
  // Only apply when the reduced denominator has no perfect square factors,
  // to avoid infinite loops (e.g. Sqrt[3]/4 would loop through Sqrt[3/16] → (1/4)*Sqrt[3] → ...)
  if let Expr::Integer(d) = b
    && *d != 0
    && is_sqrt(a).is_some()
    && let Expr::Integer(n) = is_sqrt(a).unwrap()
    && *n > 0
  {
    let denom = d * d;
    let g = gcd(*n, denom);
    let reduced_d = (denom / g) as u64;
    // Check that the reduced denominator has no perfect square factors
    let mut has_sq_factor = false;
    {
      let val = reduced_d;
      let mut f = 2u64;
      while f * f <= val {
        if val.is_multiple_of(f * f) {
          has_sq_factor = true;
          break;
        }
        f += 1;
      }
    }
    if !has_sq_factor {
      return sqrt_ast(&[make_rational(*n, denom)]);
    }
  }

  // c*Sqrt[n] / d → (c/d)*Sqrt[n] → absorbed form via times_ast
  if let Expr::Integer(d) = b
    && *d != 0
    && let Expr::FunctionCall { name, args: targs } = a
    && name == "Times"
    && targs.len() == 2
  {
    let has_sqrt_int = is_sqrt(&targs[1]).is_some()
      && matches!(is_sqrt(&targs[1]).unwrap(), Expr::Integer(n) if *n > 0);
    let has_numeric_coeff = matches!(&targs[0], Expr::Integer(_))
      || matches!(&targs[0], Expr::FunctionCall { name: rn, .. } if rn == "Rational");
    if has_sqrt_int && has_numeric_coeff {
      let coeff = make_rational(1, *d);
      return times_ast(&[targs[0].clone(), coeff, targs[1].clone()]);
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

  // Complex / complex division: (a + b*I) / (c + d*I)
  // Multiply by conjugate: ((a+bi)(c-di)) / (c² + d²)
  if let (Some(((an, ad), (bn, bd))), Some(((cn, cd), (dn, dd)))) =
    (try_extract_complex_exact(a), try_extract_complex_exact(b))
  {
    // Only proceed if denominator has nonzero imaginary part
    if dn != 0 {
      // Compute denominator magnitude squared: c² + d² (as rational)
      // |denom|² = (cn/cd)² + (dn/dd)² = (cn²·dd² + dn²·cd²) / (cd²·dd²)
      if let (Some(cn2), Some(cd2), Some(dn2), Some(dd2)) = (
        cn.checked_mul(cn),
        cd.checked_mul(cd),
        dn.checked_mul(dn),
        dd.checked_mul(dd),
      ) && let (Some(t1), Some(t2), Some(mag_den)) = (
        cn2.checked_mul(dd2),
        dn2.checked_mul(cd2),
        cd2.checked_mul(dd2),
      ) && let Some(mag_num) = t1.checked_add(t2)
        && mag_num != 0
      {
        // Real part of result: (an·cn·bd·dd + bn·dn·ad·cd) / (ad·bd·mag)
        // Actually, let's compute properly:
        // num_real = (an/ad)*(cn/cd) + (bn/bd)*(dn/dd)
        //          = (an*cn*bd*dd + bn*dn*ad*cd) / (ad*cd*bd*dd)
        // num_imag = (bn/bd)*(cn/cd) - (an/ad)*(dn/dd)
        //          = (bn*cn*ad*dd - an*dn*bd*cd) / (ad*cd*bd*dd)
        // result = (num_real + num_imag*I) / (mag_num / mag_den)
        //        = (num_real * mag_den + num_imag * mag_den * I) / (common_den * mag_num)
        if let (Some(nr1), Some(nr2)) = (
          an.checked_mul(cn)
            .and_then(|v| v.checked_mul(bd))
            .and_then(|v| v.checked_mul(dd)),
          bn.checked_mul(dn)
            .and_then(|v| v.checked_mul(ad))
            .and_then(|v| v.checked_mul(cd)),
        ) && let (Some(ni1), Some(ni2)) = (
          bn.checked_mul(cn)
            .and_then(|v| v.checked_mul(ad))
            .and_then(|v| v.checked_mul(dd)),
          an.checked_mul(dn)
            .and_then(|v| v.checked_mul(bd))
            .and_then(|v| v.checked_mul(cd)),
        ) && let (Some(num_re), Some(num_im)) =
          (nr1.checked_add(nr2), ni1.checked_sub(ni2))
        {
          let common_den = ad
            .checked_mul(cd)
            .and_then(|v| v.checked_mul(bd))
            .and_then(|v| v.checked_mul(dd));
          if let Some(cden) = common_den {
            // result = (num_re / cden + num_im / cden * I) / (mag_num / mag_den)
            //        = (num_re * mag_den) / (cden * mag_num) + (num_im * mag_den) / (cden * mag_num) * I
            if let (Some(final_re_n), Some(final_im_n), Some(final_den)) = (
              num_re.checked_mul(mag_den),
              num_im.checked_mul(mag_den),
              cden.checked_mul(mag_num),
            ) {
              return Ok(complex_rational_to_expr(
                final_re_n, final_den, final_im_n, final_den,
              ));
            }
          }
        }
      }
    }
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

      // When both numerator and denominator are Times products containing integer
      // factors, convert to Times[num_factors..., Power[den_factors, -1]...] and let
      // times_ast handle coefficient simplification and canonical ordering.
      // E.g. (2*Sqrt[2]*E^(-6)) / (12*BesselK[...]) → Sqrt[2]/(6*E^6*BesselK[...])
      if let Expr::FunctionCall {
        name: dn,
        args: dargs,
      } = b
        && dn == "Times"
      {
        let num_factors = match a {
          Expr::FunctionCall { name, args } if name == "Times" => args.clone(),
          _ => vec![a.clone()],
        };
        let den_factors = dargs.clone();
        let has_int_num = num_factors.iter().any(|f| matches!(f, Expr::Integer(_))
              || matches!(f, Expr::FunctionCall { name, .. } if name == "Rational"));
        let has_int_den = den_factors.iter().any(|f| matches!(f, Expr::Integer(_))
              || matches!(f, Expr::FunctionCall { name, .. } if name == "Rational"));
        if has_int_num && has_int_den {
          // Only apply when integer factors have GCD > 1 (can actually simplify)
          let num_int = num_factors
            .iter()
            .find_map(|f| match f {
              Expr::Integer(n) => Some(n.unsigned_abs()),
              _ => None,
            })
            .unwrap_or(1);
          let den_int = den_factors
            .iter()
            .find_map(|f| match f {
              Expr::Integer(n) => Some(n.unsigned_abs()),
              _ => None,
            })
            .unwrap_or(1);
          if gcd(num_int as i128, den_int as i128) > 1 {
            let mut all_factors = num_factors;
            for df in den_factors {
              all_factors.push(power_two(&df, &Expr::Integer(-1))?);
            }
            return times_ast(&all_factors);
          }
        }
      }

      // Canonicalize: a/b → Times[a, Power[b, -1]]
      let den_inv = power_two(b, &Expr::Integer(-1))?;
      times_ast(&[a.clone(), den_inv])
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

/// Build the canonical form of `a / b` structurally (without evaluation).
/// Returns `Times[a, Power[b, -1]]` as a FunctionCall.
/// Use this when constructing symbolic results; for evaluated division use `divide_two`.
pub fn make_divide(a: Expr, b: Expr) -> Expr {
  // a / 1 → a
  if matches!(&b, Expr::Integer(1)) {
    return a;
  }
  // 0 / b → 0
  if matches!(&a, Expr::Integer(0)) {
    return Expr::Integer(0);
  }
  let b_inv = Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![b, Expr::Integer(-1)],
  };
  // 1 / b → Power[b, -1]
  if matches!(&a, Expr::Integer(1)) {
    return b_inv;
  }
  // Flatten: if a is already Times, merge b_inv into its args
  if let Expr::FunctionCall { name, args } = &a
    && name == "Times"
  {
    let mut new_args = args.clone();
    new_args.push(b_inv);
    return Expr::FunctionCall {
      name: "Times".to_string(),
      args: new_args,
    };
  }
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![a, b_inv],
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

  // Handle Interval^n
  if let Some(result) =
    crate::functions::interval_ast::try_interval_power(base, exp)
  {
    return result;
  }

  // x^1 -> x
  if matches!(exp, Expr::Integer(1)) {
    return Ok(base.clone());
  }

  // E^Log[x] -> x (inverse function identity)
  if (matches!(base, Expr::Identifier(s) if s == "E")
    || matches!(base, Expr::Constant(s) if s == "E"))
    && let Expr::FunctionCall {
      name,
      args: log_args,
    } = exp
    && name == "Log"
    && log_args.len() == 1
  {
    return Ok(log_args[0].clone());
  }

  // E^(n*Log[x]) -> x^n (generalized inverse)
  if (matches!(base, Expr::Identifier(s) if s == "E")
    || matches!(base, Expr::Constant(s) if s == "E"))
  {
    // Extract coefficient and Log argument from Times[..., Log[x]]
    let factors: Option<Vec<&Expr>> = match exp {
      Expr::FunctionCall { name, args: targs } if name == "Times" => {
        Some(targs.iter().collect())
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left,
        right,
      } => Some(vec![left.as_ref(), right.as_ref()]),
      _ => None,
    };
    if let Some(factors) = factors {
      // Find the Log[x] factor
      let mut log_arg = None;
      let mut coeff_factors: Vec<&Expr> = Vec::new();
      for f in &factors {
        if log_arg.is_none()
          && let Expr::FunctionCall { name, args: la } = f
          && name == "Log"
          && la.len() == 1
        {
          log_arg = Some(&la[0]);
          continue;
        }
        coeff_factors.push(f);
      }
      if let Some(x) = log_arg {
        let coeff = if coeff_factors.len() == 1 {
          coeff_factors[0].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: coeff_factors.iter().map(|e| (*e).clone()).collect(),
          }
        };
        return power_two(x, &coeff);
      }
    }
  }

  // 1^x -> 1 for any finite x (1^Infinity is Indeterminate, handled below)
  if matches!(base, Expr::Integer(1))
    && !matches!(exp, Expr::Identifier(s) if s == "Infinity" || s == "ComplexInfinity")
    && !crate::functions::math_ast::special_functions::is_neg_infinity(exp)
  {
    return Ok(Expr::Integer(1));
  }

  // x^0 -> 1 (for non-zero finite x; 0^0 and Infinity^0 are Indeterminate)
  if matches!(exp, Expr::Integer(0))
    && !matches!(base, Expr::Integer(0))
    && !matches!(base, Expr::Real(f) if *f == 0.0)
    && !matches!(base, Expr::Identifier(s) if s == "Infinity" || s == "ComplexInfinity")
    && !crate::functions::math_ast::special_functions::is_neg_infinity(base)
  {
    return Ok(Expr::Integer(1));
  }

  // Handle Power with Infinity in base or exponent
  let base_is_pos_inf = matches!(base, Expr::Identifier(s) if s == "Infinity");
  let base_is_neg_inf =
    crate::functions::math_ast::special_functions::is_neg_infinity(base);
  let exp_is_pos_inf = matches!(exp, Expr::Identifier(s) if s == "Infinity");
  let exp_is_neg_inf =
    crate::functions::math_ast::special_functions::is_neg_infinity(exp);
  let base_is_complex_inf =
    matches!(base, Expr::Identifier(s) if s == "ComplexInfinity");
  let exp_is_complex_inf =
    matches!(exp, Expr::Identifier(s) if s == "ComplexInfinity");

  // Infinity^0 and ComplexInfinity^0 → Indeterminate
  if matches!(exp, Expr::Integer(0))
    && (base_is_pos_inf || base_is_neg_inf || base_is_complex_inf)
  {
    crate::emit_message(&format!(
      "\n{:>40}\nInfinity::indet: Indeterminate expression {}  encountered.",
      "0",
      crate::syntax::expr_to_string(base)
    ));
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  // Infinity^n: Infinity for positive n, 0 for negative n, ComplexInfinity for complex
  if base_is_pos_inf {
    if let Some(n) = expr_to_num(exp) {
      if n > 0.0 {
        return Ok(Expr::Identifier("Infinity".to_string()));
      } else if n < 0.0 {
        return Ok(Expr::Integer(0));
      }
    }
    if exp_is_pos_inf {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    if exp_is_neg_inf {
      return Ok(Expr::Integer(0));
    }
  }

  // (-Infinity)^n for integer n: Infinity if n>0 even, -Infinity if n>0 odd, 0 if n<0
  if base_is_neg_inf {
    if let Expr::Integer(n) = exp {
      if *n > 0 {
        if n % 2 == 0 {
          return Ok(Expr::Identifier("Infinity".to_string()));
        } else {
          return Ok(negate_expr(Expr::Identifier("Infinity".to_string())));
        }
      } else if *n < 0 {
        return Ok(Expr::Integer(0));
      }
    }
    if exp_is_pos_inf {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
  }

  // E^Infinity → Infinity, E^(-Infinity) → 0
  let base_is_e = matches!(base, Expr::Constant(c) if c == "E")
    || matches!(base, Expr::Identifier(s) if s == "E");
  if base_is_e {
    if exp_is_pos_inf {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    if exp_is_neg_inf {
      return Ok(Expr::Integer(0));
    }
  }

  // base^Infinity where base is a known positive real > 1 → Infinity
  // base^Infinity where 0 < base < 1 → 0
  // base^(-Infinity) where base > 1 → 0
  // base^(-Infinity) where 0 < base < 1 → Infinity
  if (exp_is_pos_inf || exp_is_neg_inf)
    && let Some(b) = expr_to_num(base)
  {
    if b > 1.0 {
      return Ok(if exp_is_pos_inf {
        Expr::Identifier("Infinity".to_string())
      } else {
        Expr::Integer(0)
      });
    } else if b > 0.0 && b < 1.0 {
      return Ok(if exp_is_pos_inf {
        Expr::Integer(0)
      } else {
        Expr::Identifier("Infinity".to_string())
      });
    } else if b == 1.0 {
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    } else if b == 0.0 {
      if exp_is_pos_inf {
        return Ok(Expr::Integer(0));
      }
      if exp_is_neg_inf {
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
    }
    // Negative base with infinite exponent
    if b < 0.0 && (exp_is_pos_inf || exp_is_neg_inf) {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
  }

  // ComplexInfinity^n → ComplexInfinity for positive n, 0 for negative n
  if base_is_complex_inf && let Some(n) = expr_to_num(exp) {
    if n > 0.0 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    } else if n < 0.0 {
      return Ok(Expr::Integer(0));
    }
  }

  // n^ComplexInfinity → Indeterminate for most cases
  if exp_is_complex_inf {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  // (a * b * ...)^n → Times[a^n, b^n, ...] for integer n
  // This matches Wolfram's normalization: Power[Times[...], n] distributes
  if let Expr::Integer(n) = exp {
    let factors: Option<Vec<&Expr>> = match base {
      Expr::FunctionCall { name, args: targs }
        if name == "Times" && targs.len() >= 2 =>
      {
        Some(targs.iter().collect())
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left,
        right,
      } => Some(vec![left.as_ref(), right.as_ref()]),
      _ => None,
    };
    if let Some(factors) = factors {
      let pow_factors: Result<Vec<Expr>, InterpreterError> = factors
        .into_iter()
        .map(|f| power_two(f, &Expr::Integer(*n)))
        .collect();
      return times_ast(&pow_factors?);
    }
  }

  // Sqrt[x]^n → x^(n/2)
  if is_sqrt(base).is_some()
    && let Expr::Integer(n) = exp
  {
    let sqrt_arg = is_sqrt(base).unwrap();
    if *n == 2 {
      return Ok(sqrt_arg.clone());
    }
    // Sqrt[x]^n = x^(n/2)
    return power_two(sqrt_arg, &make_rational(*n, 2));
  }

  // (base^exp1)^exp2 -> base^(exp1*exp2) when outer exponent is integer
  // Handles both BinaryOp::Power and FunctionCall Power forms
  if let Expr::Integer(e2) = exp {
    let inner = match base {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left,
        right,
      } => Some((left.as_ref(), right.as_ref())),
      Expr::FunctionCall { name, args: fargs }
        if name == "Power" && fargs.len() == 2 =>
      {
        Some((&fargs[0], &fargs[1]))
      }
      _ => None,
    };
    if let Some((inner_base, inner_exp)) = inner {
      match inner_exp {
        Expr::Integer(e1) => {
          let combined = *e1 * *e2;
          return power_two(inner_base, &Expr::Integer(combined));
        }
        // (base^(p/q))^n → base^(p*n/q)
        Expr::FunctionCall { name, args: ra }
          if name == "Rational"
            && ra.len() == 2
            && matches!(&ra[0], Expr::Integer(_))
            && matches!(&ra[1], Expr::Integer(_)) =>
        {
          if let (Expr::Integer(p), Expr::Integer(q)) = (&ra[0], &ra[1]) {
            let new_p = p * e2;
            let new_exp = make_rational(new_p, *q);
            return power_two(inner_base, &new_exp);
          }
        }
        // (constant^symbolic_exp)^n → constant^(symbolic_exp * n) for E, Pi, integers
        _ => {
          let is_const_base =
            matches!(inner_base, Expr::Constant(_) | Expr::Integer(_));
          if is_const_base {
            let new_exp = times_ast(&[inner_exp.clone(), Expr::Integer(*e2)])?;
            return power_two(inner_base, &new_exp);
          }
        }
      }
    }
  }

  // (Power[-1, Rational[p,q]])^n → simplify (-1)^(p*n/q)
  // Handles both FunctionCall and BinaryOp representations of Power
  if let Expr::Integer(n) = exp {
    let (inner_base, inner_exp) = match base {
      Expr::FunctionCall { name, args: fargs }
        if name == "Power" && fargs.len() == 2 =>
      {
        (Some(&fargs[0]), Some(&fargs[1]))
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left,
        right,
      } => (Some(left.as_ref()), Some(right.as_ref())),
      _ => (None, None),
    };
    if let (Some(ib), Some(ie)) = (inner_base, inner_exp)
      && matches!(ib, Expr::Integer(-1))
      && let Some((p, q)) = extract_rational_pair(ie)
    {
      let new_p = p * n;
      let new_q = q;
      return simplify_neg1_rational_power(new_p, new_q);
    }
  }

  // (Times[-1, Power[-1, Rational[p,q]]])^n or (UnaryMinus[Power[-1, Rational[p,q]]])^n
  // → (-1)^n * ((-1)^(p/q))^n = (-1)^(n + n*p/q) = (-1)^(n*(q+p)/q)
  // Handles both FunctionCall and BinaryOp representations of the inner Power
  if let Expr::Integer(n) = exp {
    let inner_power = match base {
      Expr::FunctionCall { name, args: targs }
        if name == "Times"
          && targs.len() == 2
          && matches!(&targs[0], Expr::Integer(-1)) =>
      {
        Some(&targs[1])
      }
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand,
      } => Some(operand.as_ref()),
      _ => None,
    };
    if let Some(inner) = inner_power {
      let (inner_base, inner_exp) = match inner {
        Expr::FunctionCall {
          name: pname,
          args: pargs,
        } if pname == "Power" && pargs.len() == 2 => {
          (Some(&pargs[0]), Some(&pargs[1]))
        }
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left,
          right,
        } => (Some(left.as_ref()), Some(right.as_ref())),
        _ => (None, None),
      };
      if let (Some(ib), Some(ie)) = (inner_base, inner_exp)
        && matches!(ib, Expr::Integer(-1))
        && let Some((p, q)) = extract_rational_pair(ie)
      {
        let new_p = n * (q + p);
        let new_q = q;
        return simplify_neg1_rational_power(new_p, new_q);
      }
    }
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

  // (a + b*I)^n for exact complex base with positive integer exponent
  if let Expr::Integer(n) = exp
    && *n >= 2
    && let Some(((re_n, re_d), (im_n, im_d))) = try_extract_complex_exact(base)
    && im_n != 0
  {
    use num_bigint::BigInt;
    use num_traits::{ToPrimitive, Zero};

    // BigInt GCD helper
    fn bigint_gcd(a: &BigInt, b: &BigInt) -> BigInt {
      let mut a = if *a < BigInt::from(0) {
        -a.clone()
      } else {
        a.clone()
      };
      let mut b = if *b < BigInt::from(0) {
        -b.clone()
      } else {
        b.clone()
      };
      while !b.is_zero() {
        let t = &a % &b;
        a = b;
        b = t;
      }
      a
    }

    // Normalize denominators to be positive
    let (re_n, re_d) = if re_d < 0 {
      (-re_n, -re_d)
    } else {
      (re_n, re_d)
    };
    let (im_n, im_d) = if im_d < 0 {
      (-im_n, -im_d)
    } else {
      (im_n, im_d)
    };

    // Common denominator: k = lcm(re_d, im_d)
    let g = gcd_i128(re_d, im_d);
    let k_val = re_d / g * im_d;
    let a = re_n * (k_val / re_d);
    let b_val = im_n * (k_val / im_d);

    // Compute (a + b*I)^n using BigInt exponentiation by squaring
    let a_big = BigInt::from(a);
    let b_big = BigInt::from(b_val);
    let mut result_re = BigInt::from(1);
    let mut result_im = BigInt::from(0);
    let mut sq_re = a_big.clone();
    let mut sq_im = b_big.clone();
    let mut remaining = *n as u64;
    while remaining > 0 {
      if remaining & 1 == 1 {
        let new_re = &result_re * &sq_re - &result_im * &sq_im;
        let new_im = &result_re * &sq_im + &result_im * &sq_re;
        result_re = new_re;
        result_im = new_im;
      }
      remaining >>= 1;
      if remaining > 0 {
        let new_re = &sq_re * &sq_re - &sq_im * &sq_im;
        let new_im = BigInt::from(2) * &sq_re * &sq_im;
        sq_re = new_re;
        sq_im = new_im;
      }
    }

    // Result = result_re / k^n + (result_im / k^n) * I
    let k_big = BigInt::from(k_val);
    let k_n = num_traits::pow::pow(k_big, *n as usize);

    // Reduce fractions
    let g_re = bigint_gcd(&result_re, &k_n);
    let final_re_n = &result_re / &g_re;
    let final_re_d = &k_n / &g_re;
    let g_im = bigint_gcd(&result_im, &k_n);
    let final_im_n = &result_im / &g_im;
    let final_im_d = &k_n / &g_im;

    // Try to convert back to i128
    if let (Some(rn), Some(rd), Some(imn), Some(imd)) = (
      final_re_n.to_i128(),
      final_re_d.to_i128(),
      final_im_n.to_i128(),
      final_im_d.to_i128(),
    ) {
      return Ok(complex_rational_to_expr(rn, rd, imn, imd));
    }

    // BigInt result: build expression for integer denominators
    if final_re_d == BigInt::from(1) && final_im_d == BigInt::from(1) {
      let re_expr = bigint_to_expr(final_re_n);
      let i_expr = Expr::Identifier("I".to_string());
      if result_im.is_zero() {
        return Ok(re_expr);
      }
      let im_expr = bigint_to_expr(final_im_n);
      let imag_term = if matches!(&im_expr, Expr::Integer(1)) {
        i_expr.clone()
      } else if matches!(&im_expr, Expr::Integer(-1)) {
        negate_expr(i_expr.clone())
      } else {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(im_expr),
          right: Box::new(i_expr),
        }
      };
      if result_re.is_zero() {
        return Ok(imag_term);
      }
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(re_expr),
        right: Box::new(imag_term),
      });
    }
  }

  // E^(k*I*Pi) → Cos[k*Pi] + I*Sin[k*Pi] only for multiples of Pi/2
  // Wolfram does not auto-simplify Exp[I*Pi/3] etc.
  if matches!(base, Expr::Constant(c) if c == "E")
    && let Some((numer, denom)) = try_extract_i_pi_rational_multiple(exp)
    && (denom == 1 || denom == 2)
  {
    // Build the symbolic argument k*Pi
    let k_pi =
      crate::functions::math_ast::complex::make_rational_times_pi(numer, denom);
    // Evaluate Cos[k*Pi] and Sin[k*Pi] symbolically
    let cos_val = crate::functions::math_ast::cos_ast(&[k_pi.clone()])?;
    let sin_val = crate::functions::math_ast::sin_ast(&[k_pi])?;
    // Build result: cos_val + I*sin_val
    let sin_is_zero = matches!(&sin_val, Expr::Integer(0));
    let cos_is_zero = matches!(&cos_val, Expr::Integer(0));
    if sin_is_zero {
      return Ok(cos_val);
    }
    let i_expr = Expr::Identifier("I".to_string());
    let imag_term = if matches!(&sin_val, Expr::Integer(1)) {
      i_expr
    } else if matches!(&sin_val, Expr::Integer(-1)) {
      negate_expr(i_expr)
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![sin_val, i_expr],
      }
    };
    if cos_is_zero {
      return Ok(imag_term);
    }
    return Ok(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![cos_val, imag_term],
    });
  }

  // E^(complex) → Euler's formula: E^(a + b*I) = E^a * (Cos[b] + I*Sin[b])
  // Only apply for purely numeric exponents, not symbolic ones like I*Pi/3
  if matches!(base, Expr::Constant(c) if c == "E")
    && try_extract_i_pi_rational_multiple(exp).is_none()
  {
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
    crate::emit_message(&format!(
      "{:>width$}\nPower::indet: Indeterminate expression {}  encountered.",
      exp_str,
      base_str,
      width = padding
    ));
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  // Special case: 0^(negative) = ComplexInfinity (with warning)
  if base_is_zero
    && let Some(e) = expr_to_num(exp)
    && e < 0.0
  {
    let base_str = crate::syntax::expr_to_string(base);
    let padding = 39 + base_str.len() + 1;
    crate::emit_message(&format!(
      "{:>width$}\nPower::infy: Infinite expression - encountered.",
      1,
      width = padding
    ));
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }

  // Special case: integer base with negative integer exponent -> Rational
  if let (Expr::Integer(b), Expr::Integer(e)) = (base, exp)
    && *e < 0
  {
    // b^(-n) = 1 / b^n = Rational[1, b^n]
    let pos_exp = (-*e) as u32;
    if let Some(denom) = b.checked_pow(pos_exp) {
      return Ok(make_rational(1, denom));
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
    // (-1)^(p/q) → simplify via simplify_neg1_rational_power
    if *b == -1 {
      return simplify_neg1_rational_power(*numer, *denom);
    }
    // Negative base (other than -1): (-n)^(p/q) = (-1)^(p/q) * n^(p/q)
    if *b < -1 && *numer > 0 && *denom > 0 {
      let neg1_part = simplify_neg1_rational_power(*numer, *denom)?;
      let pos_part = power_two(&Expr::Integer(-*b), exp)?;
      return times_ast(&[neg1_part, pos_part]);
    }
    // Handle negative rational exponents for integer bases: b^(-p/q) = 1 / b^(p/q)
    // This allows the positive prime factorization code below to simplify the radical.
    // Only applies when base is a positive integer (not symbolic expressions).
    if *numer < 0 && *denom > 0 && *b > 1 {
      let pos_exp = make_rational(-*numer, *denom);
      let pos_result = power_two(base, &pos_exp)?;
      // Only simplify if the positive power actually reduced
      // (avoid infinite recursion when the positive power stays symbolic)
      if !matches!(
        &pos_result,
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          ..
        }
      ) && !matches!(&pos_result, Expr::FunctionCall { name, .. } if name == "Power")
        && is_sqrt(&pos_result).is_none()
      {
        return divide_ast(&[Expr::Integer(1), pos_result]);
      }
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
          rad_parts.push(make_sqrt(Expr::Integer(*prime)));
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

  // x^(1/2) → delegate to sqrt_ast for symbolic simplification
  // This must come before the numeric evaluation to preserve exact results.
  if is_half(exp) {
    return crate::functions::sqrt_ast(&[base.clone()]);
  }

  // If either operand is Real, result is Real (even if whole number)
  let has_real = matches!(base, Expr::Real(_)) || matches!(exp, Expr::Real(_));

  match (expr_to_num(base), expr_to_num(exp)) {
    (Some(a), Some(b)) => {
      let result = a.powf(b);
      if result.is_nan() && a < 0.0 {
        // Negative base with fractional exponent: use complex arithmetic
        // (-x)^r = x^r * e^(i*pi*r) = x^r * (cos(pi*r) + i*sin(pi*r))
        let pos_base = a.abs();
        let magnitude = pos_base.powf(b);
        let angle = std::f64::consts::PI * b;
        let re = magnitude * angle.cos();
        let im = magnitude * angle.sin();
        // Round near-zero components to avoid floating-point noise
        let re = if re.abs() < 1e-15 { 0.0 } else { re };
        let im = if im.abs() < 1e-15 { 0.0 } else { im };
        if im == 0.0 {
          Ok(Expr::Real(re))
        } else {
          // Return re + im*I as a complex expression
          let im_part = if im == 1.0 {
            Expr::Identifier("I".to_string())
          } else if im == -1.0 {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), Expr::Identifier("I".to_string())],
            }
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Real(im), Expr::Identifier("I".to_string())],
            }
          };
          if re == 0.0 {
            Ok(im_part)
          } else {
            Ok(Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![Expr::Real(re), im_part],
            })
          }
        }
      } else if has_real {
        Ok(Expr::Real(result))
      } else {
        // Both were integers - use num_to_expr to get Integer when result is whole
        Ok(num_to_expr(result))
      }
    }
    _ => Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(base.clone()),
      right: Box::new(exp.clone()),
    }),
  }
}

/// Try to extract rational coefficient k = numer/denom from an expression
/// of the form `k * I * Pi`.
///
/// Recognizes these internal forms:
/// - `Times[I, Pi]`              → k = 1/1
/// - `Times[n, I, Pi]`           → k = n/1
/// - `Times[Rational[p,q], I, Pi]` → k = p/q
/// - `Times[k, Times[I, Pi]]`    → recursive
/// - `Times[-1, I, Pi]`          → k = -1/1
fn try_extract_i_pi_rational_multiple(expr: &Expr) -> Option<(i128, i128)> {
  let factors: Vec<&Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().collect()
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => vec![left.as_ref(), right.as_ref()],
    _ => return None,
  };

  let mut has_i = false;
  let mut has_pi = false;
  let mut coeff_numer: i128 = 1;
  let mut coeff_denom: i128 = 1;
  let mut has_numeric_coeff = false;

  for factor in &factors {
    match factor {
      Expr::Identifier(s) if s == "I" => {
        if has_i {
          return None;
        } // Two I factors
        has_i = true;
      }
      Expr::Constant(s) if s == "Pi" => {
        if has_pi {
          return None;
        } // Two Pi factors
        has_pi = true;
      }
      Expr::Integer(n) => {
        if has_numeric_coeff {
          return None;
        }
        coeff_numer = *n;
        coeff_denom = 1;
        has_numeric_coeff = true;
      }
      Expr::FunctionCall { name, args }
        if name == "Rational"
          && args.len() == 2
          && matches!(&args[0], Expr::Integer(_))
          && matches!(&args[1], Expr::Integer(_)) =>
      {
        if has_numeric_coeff {
          return None;
        }
        if let (Expr::Integer(p), Expr::Integer(q)) = (&args[0], &args[1]) {
          coeff_numer = *p;
          coeff_denom = *q;
          has_numeric_coeff = true;
        }
      }
      // Nested Times (e.g., Times[Rational[2,3], Times[I, Pi]])
      nested => {
        if let Some((n, d)) = try_extract_i_pi_rational_multiple(nested) {
          if has_i || has_pi {
            return None;
          }
          has_i = true;
          has_pi = true;
          coeff_numer *= n;
          coeff_denom *= d;
        } else {
          return None; // Unknown factor
        }
      }
    }
  }

  if has_i && has_pi {
    Some((coeff_numer, coeff_denom))
  } else {
    None
  }
}

/// Extract (p, q) from a Rational[p, q] FunctionCall expression.
fn extract_rational_pair(expr: &Expr) -> Option<(i128, i128)> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Rational"
    && args.len() == 2
    && let Expr::Integer(p) = &args[0]
    && let Expr::Integer(q) = &args[1]
  {
    Some((*p, *q))
  } else {
    None
  }
}

/// Simplify (-1)^(p/q) where p and q are integers.
/// Reduces p mod 2q to the canonical range, then builds the expression.
fn simplify_neg1_rational_power(
  p: i128,
  q: i128,
) -> Result<Expr, InterpreterError> {
  let g = gcd_i128(p.abs(), q.abs());
  let (mut p, mut q) = (p / g, q / g);
  if q < 0 {
    p = -p;
    q = -q;
  }
  // Reduce p mod 2q to canonical range [0, 2q)
  let period = 2 * q;
  p = ((p % period) + period) % period;

  // (-1)^0 = 1
  if p == 0 {
    return Ok(Expr::Integer(1));
  }
  // (-1)^1 = -1
  if p == q {
    return Ok(Expr::Integer(-1));
  }
  // If p > q, factor out (-1)^1 = -1: (-1)^(p/q) = -(-1)^((p-q)/q)
  if p > q {
    let remainder = p - q;
    let g2 = gcd_i128(remainder, q);
    let rp = remainder / g2;
    let rq = q / g2;
    let inner = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![Expr::Integer(-1), make_rational_pub(rp, rq)],
    };
    return Ok(negate_expr(inner));
  }
  // (-1)^(1/2) = I
  if p == 1 && q == 2 {
    return Ok(Expr::Identifier("I".to_string()));
  }
  // 0 < p < q: return (-1)^(p/q)
  Ok(Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![Expr::Integer(-1), make_rational_pub(p, q)],
  })
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
      Expr::Association(pairs) => {
        for (_, v) in pairs {
          result.push(v);
        }
      }
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

  // Handle Interval in Max
  if let Some(result) = crate::functions::interval_ast::try_interval_max(args) {
    return result;
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

  // Handle Interval in Min
  if let Some(result) = crate::functions::interval_ast::try_interval_min(args) {
    return result;
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

/// Sum BigFloat (precision-tagged) numbers with precision tracking.
/// Handles BigFloat + BigFloat and BigFloat + Integer/Rational.
fn bigfloat_plus(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Extract (f64_value, precision) for each argument
  // BigFloat: use stored precision; Integer/Rational: infinite precision
  let mut sum_val: f64 = 0.0;
  let mut sum_error: f64 = 0.0;

  for arg in args {
    match arg {
      Expr::BigFloat(digits, prec) => {
        let v: f64 = digits.parse().unwrap_or(0.0);
        let p = *prec;
        sum_val += v;
        sum_error += v.abs() * 10f64.powf(-p);
      }
      Expr::Integer(n) => {
        sum_val += *n as f64;
        // Integer has infinite precision, contributes 0 error
      }
      Expr::FunctionCall { name, args: rargs }
        if name == "Rational" && rargs.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1]) {
          sum_val += *n as f64 / *d as f64;
        }
      }
      _ => {}
    }
  }

  // Compute result precision
  let result_prec: f64 = if sum_val.abs() < 1e-300 || sum_error <= 0.0 {
    // Fallback: use min finite precision among BigFloat args
    args
      .iter()
      .filter_map(|a| {
        if let Expr::BigFloat(_, p) = a {
          Some(*p)
        } else {
          None
        }
      })
      .fold(None, |acc: Option<f64>, p| {
        Some(acc.map_or(p, |a: f64| a.min(p)))
      })
      .unwrap_or(1.0)
  } else {
    let p = sum_val.abs().log10() - sum_error.log10();
    p.max(1.0)
  };

  // Format result value with the right number of significant digits
  let display_prec = (result_prec.round() as usize).max(1);
  let result_str = format_bigfloat_value(sum_val, display_prec);
  Ok(Expr::BigFloat(result_str, result_prec))
}

/// Format an f64 value as a BigFloat digit string with the given significant digits.
fn format_bigfloat_value(value: f64, sig_digits: usize) -> String {
  if value == 0.0 {
    return "0.".to_string();
  }
  let sign = if value < 0.0 { "-" } else { "" };
  let abs_val = value.abs();
  let magnitude = abs_val.log10().floor() as i32;
  let decimal_places = ((sig_digits as i32) - magnitude - 1).max(0) as usize;
  let formatted = format!("{}{:.prec$}", sign, abs_val, prec = decimal_places);
  if !formatted.contains('.') {
    format!("{}.", formatted)
  } else {
    formatted
  }
}

/// Check if Plus args represent DateObject subtraction (d1 - d2) and handle it.
/// Returns Some(Ok(Quantity[n, "Days"])) if applicable.
fn try_date_object_subtraction(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if args.len() != 2 {
    return None;
  }

  fn is_date_object(e: &Expr) -> bool {
    matches!(e, Expr::FunctionCall { name, .. } if name == "DateObject")
  }

  fn extract_negated_date(e: &Expr) -> Option<&Expr> {
    // Matches Times[-1, DateObject[...]] or UnaryOp(Minus, DateObject[...])
    match e {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left,
        right,
      } => {
        if matches!(left.as_ref(), Expr::Integer(-1)) && is_date_object(right) {
          Some(right)
        } else if matches!(right.as_ref(), Expr::Integer(-1))
          && is_date_object(left)
        {
          Some(left)
        } else {
          None
        }
      }
      Expr::FunctionCall { name, args: fargs }
        if name == "Times" && fargs.len() == 2 =>
      {
        if matches!(&fargs[0], Expr::Integer(-1)) && is_date_object(&fargs[1]) {
          Some(&fargs[1])
        } else if matches!(&fargs[1], Expr::Integer(-1))
          && is_date_object(&fargs[0])
        {
          Some(&fargs[0])
        } else {
          None
        }
      }
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand,
      } if is_date_object(operand) => Some(operand),
      _ => None,
    }
  }

  // Pattern: DateObject[...] + Times[-1, DateObject[...]]
  // i.e. d1 - d2
  if is_date_object(&args[0])
    && let Some(d2) = extract_negated_date(&args[1])
  {
    return Some(crate::functions::datetime_ast::date_difference_ast(&[
      d2.clone(),
      args[0].clone(),
    ]));
  }
  if is_date_object(&args[1])
    && let Some(d1) = extract_negated_date(&args[0])
  {
    return Some(crate::functions::datetime_ast::date_difference_ast(&[
      args[1].clone(),
      d1.clone(),
    ]));
  }

  None
}
