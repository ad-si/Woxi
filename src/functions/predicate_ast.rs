//! AST-native predicate functions.
//!
//! These functions work directly with `Expr` AST nodes.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Helper to create boolean result
fn bool_expr(b: bool) -> Expr {
  Expr::Identifier(if b { "True" } else { "False" }.to_string())
}

/// NumberQ[expr] - Tests if the expression is a number
pub fn number_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NumberQ expects exactly 1 argument".into(),
    ));
  }
  let is_number = is_numeric_expr(&args[0]);
  Ok(bool_expr(is_number))
}

/// Check if an expression is a numeric quantity (Integer, Real, Rational, Complex)
fn is_numeric_expr(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _) => true,
    // Rational[n, d]
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      is_numeric_expr(&args[0]) && is_numeric_expr(&args[1])
    }
    // Complex[re, im]
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      is_numeric_expr(&args[0]) && is_numeric_expr(&args[1])
    }
    // I alone
    Expr::Identifier(name) if name == "I" => true,
    // n * I or I * n
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      let has_i = matches!(left.as_ref(), Expr::Identifier(n) if n == "I")
        || matches!(right.as_ref(), Expr::Identifier(n) if n == "I");
      if has_i {
        let other = if matches!(left.as_ref(), Expr::Identifier(n) if n == "I")
        {
          right
        } else {
          left
        };
        is_numeric_expr(other)
      } else {
        is_numeric_expr(left) && is_numeric_expr(right)
      }
    }
    // a + b*I or a - b*I
    Expr::BinaryOp {
      op:
        crate::syntax::BinaryOperator::Plus | crate::syntax::BinaryOperator::Minus,
      left,
      right,
    } => is_numeric_expr(left) && is_numeric_expr(right),
    // Times[...] or Plus[...] with all numeric parts
    Expr::FunctionCall { name, args }
      if (name == "Times" || name == "Plus") && !args.is_empty() =>
    {
      args.iter().all(is_numeric_expr)
    }
    // Unary minus
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => is_numeric_expr(operand),
    // a / b where both are numeric
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => is_numeric_expr(left) && is_numeric_expr(right),
    _ => false,
  }
}

/// RealValuedNumberQ[expr] - Tests if the expression is a real-valued number
/// Returns True for Integer, BigInteger, Rational, Real, BigFloat
/// Returns False for Complex, symbolic expressions, etc.
pub fn real_valued_number_q_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RealValuedNumberQ expects exactly 1 argument".into(),
    ));
  }
  let is_real = match &args[0] {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _) => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      true
    }
    Expr::FunctionCall { name, args }
      if (name == "Underflow" || name == "Overflow") && args.is_empty() =>
    {
      true
    }
    _ => false,
  };
  Ok(bool_expr(is_real))
}

/// IntegerQ[expr] - Tests if the expression is an integer (not a real like 3.0)
pub fn integer_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "IntegerQ expects exactly 1 argument".into(),
    ));
  }
  // Both Expr::Integer and Expr::BigInteger are integers
  let is_integer = matches!(&args[0], Expr::Integer(_) | Expr::BigInteger(_));
  Ok(bool_expr(is_integer))
}

/// EvenQ[n] - Tests if a number is even
pub fn even_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "EvenQ expects exactly 1 argument".into(),
    ));
  }
  let is_even = match &args[0] {
    Expr::Integer(n) => n % 2 == 0,
    Expr::BigInteger(n) => {
      use num_traits::Zero;
      (n % num_bigint::BigInt::from(2)).is_zero()
    }
    Expr::Real(f) => {
      if f.fract() == 0.0 {
        (*f as i64) % 2 == 0
      } else {
        return Ok(bool_expr(false));
      }
    }
    _ => return Ok(bool_expr(false)),
  };
  Ok(bool_expr(is_even))
}

/// OddQ[n] - Tests if a number is odd
pub fn odd_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "OddQ expects exactly 1 argument".into(),
    ));
  }
  let is_odd = match &args[0] {
    Expr::Integer(n) => n % 2 != 0,
    Expr::BigInteger(n) => {
      use num_traits::Zero;
      !(n % num_bigint::BigInt::from(2)).is_zero()
    }
    Expr::Real(f) => {
      if f.fract() == 0.0 {
        (*f as i64) % 2 != 0
      } else {
        return Ok(bool_expr(false));
      }
    }
    _ => return Ok(bool_expr(false)),
  };
  Ok(bool_expr(is_odd))
}

/// ListQ[expr] - Tests if the expression is a list
pub fn list_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ListQ expects exactly 1 argument".into(),
    ));
  }
  let is_list = matches!(&args[0], Expr::List(_));
  Ok(bool_expr(is_list))
}

/// StringQ[expr] - Tests if the expression is a string
pub fn string_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "StringQ expects exactly 1 argument".into(),
    ));
  }
  let is_string = matches!(&args[0], Expr::String(_));
  Ok(bool_expr(is_string))
}

/// AtomQ[expr] - Tests if the expression is atomic (not compound)
pub fn atom_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AtomQ expects exactly 1 argument".into(),
    ));
  }
  let is_atom = match &args[0] {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::Identifier(_)
    | Expr::Constant(_) => true,
    // Rational[n, d] and Complex[re, im] are atoms in Mathematica
    Expr::FunctionCall { name, .. }
      if name == "Rational" || name == "Complex" =>
    {
      true
    }
    _ => false,
  };
  Ok(bool_expr(is_atom))
}

/// NumericQ[expr] - Tests if the expression is numeric (evaluates to a number)
pub fn numeric_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NumericQ expects exactly 1 argument".into(),
    ));
  }
  Ok(bool_expr(is_numeric_q(&args[0])))
}

/// Check if an expression is numeric (can be numerically evaluated)
fn is_numeric_q(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _) => true,
    Expr::Constant(_) => true,
    Expr::Identifier(name) => {
      if name == "I" || name == "Infinity" {
        return true;
      }
      // Check for user-defined NumericQ downvalue (e.g., NumericQ[a] = True)
      // FUNC_DEFS stores (params, conditions, defaults, heads, body)
      crate::FUNC_DEFS.with(|m| {
        let defs = m.borrow();
        if let Some(overloads) = defs.get("NumericQ") {
          for (params, conditions, _defaults, _heads, body) in overloads {
            if params.len() == 1 {
              // Try to match this symbol against the DownValue conditions
              if let Some(Some(cond)) = conditions.first()
                && let Expr::Comparison {
                  operands,
                  operators,
                } = cond
                && operators.len() == 1
                && operators[0] == crate::syntax::ComparisonOp::SameQ
                && operands.len() == 2
                && let Expr::Identifier(cond_val) = &operands[1]
                && cond_val == name
              {
                let body_str = crate::syntax::expr_to_string(body);
                return body_str == "True";
              }
            }
          }
        }
        false
      })
    }
    Expr::FunctionCall { name, args, .. } => {
      is_numeric_function(name) && args.iter().all(is_numeric_q)
    }
    Expr::BinaryOp { left, right, .. } => {
      is_numeric_q(left) && is_numeric_q(right)
    }
    Expr::UnaryOp { operand, .. } => is_numeric_q(operand),
    _ => false,
  }
}

/// Known numeric functions (have the NumericFunction attribute in Wolfram Language)
fn is_numeric_function(name: &str) -> bool {
  matches!(
    name,
    "Plus"
      | "Times"
      | "Power"
      | "Sqrt"
      | "Rational"
      | "Sin"
      | "Cos"
      | "Tan"
      | "Cot"
      | "Sec"
      | "Csc"
      | "ArcSin"
      | "ArcCos"
      | "ArcTan"
      | "ArcCot"
      | "ArcSec"
      | "ArcCsc"
      | "Sinh"
      | "Cosh"
      | "Tanh"
      | "Coth"
      | "Sech"
      | "Csch"
      | "Exp"
      | "Log"
      | "Log2"
      | "Log10"
      | "Abs"
      | "Arg"
      | "Re"
      | "Im"
      | "Conjugate"
      | "Sign"
      | "Round"
      | "Floor"
      | "Ceiling"
      | "Max"
      | "Min"
      | "Mod"
      | "Quotient"
      | "GCD"
      | "LCM"
      | "Factorial"
      | "Factorial2"
      | "Gamma"
      | "Beta"
      | "Binomial"
      | "Fibonacci"
      | "EulerPhi"
      | "BernoulliB"
      | "Zeta"
      | "N"
  ) || crate::FUNC_ATTRS.with(|m| {
    m.borrow()
      .get(name)
      .is_some_and(|attrs| attrs.contains(&"NumericFunction".to_string()))
  })
}

/// PositiveQ[x] - Tests if x is a positive number
/// Check if an expression is known to be strictly positive.
fn is_known_positive(expr: &Expr) -> Option<bool> {
  match expr {
    Expr::Integer(n) => Some(*n > 0),
    Expr::BigInteger(n) => Some(*n > num_bigint::BigInt::from(0)),
    Expr::Real(f) => Some(*f > 0.0),
    Expr::Constant(c) => match c.as_str() {
      "Pi" | "E" | "Degree" => Some(true),
      _ => None,
    },
    Expr::Identifier(name) => match name.as_str() {
      "Infinity" => Some(true),
      _ => None,
    },
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => is_known_positive(operand).map(|p| !p),
    // Times[-1, x] is negative of x (e.g. -Pi parses as Times[-1, Pi])
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(args[0], Expr::Integer(-1)) =>
    {
      is_known_positive(&args[1]).map(|p| !p)
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(-1)) => {
      is_known_positive(right).map(|p| !p)
    }
    _ => None,
  }
}

/// Check if an expression is known to be strictly negative.
fn is_known_negative(expr: &Expr) -> Option<bool> {
  match expr {
    Expr::Integer(n) => Some(*n < 0),
    Expr::BigInteger(n) => Some(*n < num_bigint::BigInt::from(0)),
    Expr::Real(f) => Some(*f < 0.0),
    Expr::Constant(_) => Some(false), // Pi, E, Degree are all positive
    Expr::Identifier(name) => match name.as_str() {
      "Infinity" => Some(false),
      _ => None,
    },
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => is_known_positive(operand),
    // Times[-1, x] is negative of x (e.g. -Pi parses as Times[-1, Pi])
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(args[0], Expr::Integer(-1)) =>
    {
      is_known_positive(&args[1])
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(-1)) => is_known_positive(right),
    _ => None,
  }
}

pub fn positive_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PositiveQ expects exactly 1 argument".into(),
    ));
  }
  let is_positive = is_known_positive(&args[0]).unwrap_or(false);
  Ok(bool_expr(is_positive))
}

/// NegativeQ[x] - Tests if x is a negative number
pub fn negative_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NegativeQ expects exactly 1 argument".into(),
    ));
  }
  match is_known_negative(&args[0]) {
    Some(val) => Ok(bool_expr(val)),
    None => Ok(Expr::FunctionCall {
      name: "Negative".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// NonPositiveQ[x] - Tests if x is <= 0
pub fn non_positive_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NonPositiveQ expects exactly 1 argument".into(),
    ));
  }
  // NonPositive: x <= 0, i.e. negative or zero
  let is_non_positive =
    is_known_negative(&args[0]).unwrap_or(false) || is_zero(&args[0]);
  Ok(bool_expr(is_non_positive))
}

/// NonNegativeQ[x] - Tests if x is >= 0
pub fn non_negative_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NonNegativeQ expects exactly 1 argument".into(),
    ));
  }
  // NonNegative: x >= 0, i.e. positive or zero
  let is_non_negative =
    is_known_positive(&args[0]).unwrap_or(false) || is_zero(&args[0]);
  Ok(bool_expr(is_non_negative))
}

fn is_zero(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(0)) || matches!(expr, Expr::Real(f) if *f == 0.0)
}

/// PrimeQ[n] - Tests if n is a prime number
pub fn prime_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PrimeQ expects exactly 1 argument".into(),
    ));
  }
  if let Expr::BigInteger(n) = &args[0] {
    use num_traits::Signed;
    let abs_n = n.abs();
    return Ok(bool_expr(crate::functions::math_ast::is_prime_bigint(
      &abs_n,
    )));
  }
  let n = match &args[0] {
    Expr::Integer(n) => n.abs(),
    Expr::Real(f) => {
      if f.fract() == 0.0 {
        f.abs() as i128
      } else {
        return Ok(bool_expr(false));
      }
    }
    _ => return Ok(bool_expr(false)),
  };

  let is_prime = if n <= 1 {
    false
  } else if n <= 3 {
    true
  } else if n % 2 == 0 || n % 3 == 0 {
    false
  } else if n.unsigned_abs() > (1u128 << 53) {
    // For large integers, use Miller-Rabin instead of trial division
    crate::functions::math_ast::is_prime_bigint(&num_bigint::BigInt::from(n))
  } else {
    let mut i = 5i128;
    let mut result = true;
    while i * i <= n {
      if n % i == 0 || n % (i + 2) == 0 {
        result = false;
        break;
      }
      i += 6;
    }
    result
  };
  Ok(bool_expr(is_prime))
}

/// CompositeQ[n] - Tests if n is a composite (non-prime > 1) number
pub fn composite_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CompositeQ expects exactly 1 argument".into(),
    ));
  }
  // Delegate to PrimeQ and negate
  let prime_result = prime_q_ast(args)?;
  let is_prime = matches!(&prime_result, Expr::Identifier(s) if s == "True");

  // CompositeQ is True only for n > 1 that are not prime
  let n_gt_1 = match &args[0] {
    Expr::Integer(n) => *n > 1,
    Expr::BigInteger(n) => *n > num_bigint::BigInt::from(1),
    Expr::Real(f) if f.fract() == 0.0 => *f > 1.0,
    _ => return Ok(bool_expr(false)),
  };

  Ok(bool_expr(n_gt_1 && !is_prime))
}

/// PrimePowerQ[n] - Tests if n is a power of a prime
pub fn prime_power_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PrimePowerQ expects exactly 1 argument".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) => n.unsigned_abs(),
    Expr::BigInteger(n) => {
      use num_traits::Signed;
      let abs_n = n.abs();
      // Check if it's a prime power via factorization
      // For BigInteger, convert to u128 if possible
      match abs_n.clone().try_into() {
        Ok(v) => v,
        Err(_) => {
          // Too large, check if it's prime itself
          return Ok(bool_expr(crate::functions::math_ast::is_prime_bigint(
            &abs_n,
          )));
        }
      }
    }
    _ => return Ok(bool_expr(false)),
  };
  if n <= 1 {
    return Ok(bool_expr(false));
  }
  // Find a prime factor and check if n is a power of it
  let mut p: u128 = 0;
  let mut m = n;
  if m % 2 == 0 {
    p = 2;
    while m % 2 == 0 {
      m /= 2;
    }
  } else {
    let mut i: u128 = 3;
    while i * i <= m {
      if m % i == 0 {
        p = i;
        while m % i == 0 {
          m /= i;
        }
        break;
      }
      i += 2;
    }
    if p == 0 {
      // n itself is prime (no factor found)
      return Ok(bool_expr(true));
    }
  }
  // n is a prime power iff m == 1 after dividing out the single prime factor
  Ok(bool_expr(m == 1 && p > 0))
}

/// AssociationQ[expr] - Tests if the expression is an association
pub fn association_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AssociationQ expects exactly 1 argument".into(),
    ));
  }
  let is_assoc = matches!(&args[0], Expr::Association(_));
  Ok(bool_expr(is_assoc))
}

/// MemberQ[list, elem] - Tests if elem is a member of list
pub fn member_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MemberQ expects exactly 2 arguments".into(),
    ));
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => return Ok(bool_expr(false)),
  };

  let target_str = crate::syntax::expr_to_string(&args[1]);
  for item in list {
    if crate::syntax::expr_to_string(item) == target_str {
      return Ok(bool_expr(true));
    }
  }
  Ok(bool_expr(false))
}

/// FreeQ[expr, form] - Tests if expr is free of form
pub fn free_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "FreeQ expects exactly 2 arguments".into(),
    ));
  }

  let form = &args[1];
  let form_str = crate::syntax::expr_to_string(form);

  fn expr_str_eq(a: &Expr, b: &Expr) -> bool {
    crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
  }

  /// Check if `needles` args are a subset of `haystack` args (for Flat+Orderless ops)
  fn is_args_subset(haystack: &[Expr], needles: &[Expr]) -> bool {
    let mut used = vec![false; haystack.len()];
    'outer: for needle in needles {
      for (i, h) in haystack.iter().enumerate() {
        if !used[i] && expr_str_eq(h, needle) {
          used[i] = true;
          continue 'outer;
        }
      }
      return false;
    }
    true
  }

  fn is_form_symbol(form: &Expr, name: &str) -> bool {
    matches!(form, Expr::Identifier(s) if s == name)
  }

  fn contains_form(expr: &Expr, form: &Expr, form_str: &str) -> bool {
    // Check exact match via string comparison
    if crate::syntax::expr_to_string(expr) == form_str {
      return true;
    }
    match expr {
      Expr::List(items) => {
        // Check if form matches the head "List"
        if is_form_symbol(form, "List") {
          return true;
        }
        items.iter().any(|e| contains_form(e, form, form_str))
      }
      Expr::FunctionCall {
        name,
        args: fn_args,
        ..
      } => {
        // Check if form is a symbol matching this function's head
        if is_form_symbol(form, name) {
          return true;
        }
        // For Flat+Orderless functions (Plus, Times), check if form's args
        // are a subset of this function's args
        if let Expr::FunctionCall {
          name: form_name,
          args: form_args,
          ..
        } = form
          && name == form_name
          && !form_args.is_empty()
          && form_args.len() < fn_args.len()
          && (name == "Plus" || name == "Times")
          && is_args_subset(fn_args, form_args)
        {
          return true;
        }
        fn_args.iter().any(|e| contains_form(e, form, form_str))
      }
      Expr::BinaryOp {
        op, left, right, ..
      } => {
        // Check if operator's head matches the form symbol
        if let Expr::Identifier(s) = form {
          let matches = match op {
            crate::syntax::BinaryOperator::Plus => s == "Plus",
            crate::syntax::BinaryOperator::Minus => s == "Plus",
            crate::syntax::BinaryOperator::Times => s == "Times",
            crate::syntax::BinaryOperator::Power => s == "Power",
            crate::syntax::BinaryOperator::Divide => s == "Times",
            _ => false,
          };
          if matches {
            return true;
          }
        }
        contains_form(left, form, form_str)
          || contains_form(right, form, form_str)
      }
      Expr::UnaryOp { operand, .. } => contains_form(operand, form, form_str),
      _ => false,
    }
  }

  Ok(bool_expr(!contains_form(&args[0], form, &form_str)))
}

/// SquareFreeQ[n] - Tests if an integer has no repeated prime factors
pub fn square_free_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "SquareFreeQ expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => {
      let n = *n;
      if n == 0 {
        return Ok(bool_expr(false));
      }
      let mut num = n.unsigned_abs();
      // Check factor of 2
      let mut count = 0;
      while num % 2 == 0 {
        count += 1;
        num /= 2;
        if count > 1 {
          return Ok(bool_expr(false));
        }
      }
      // Check odd factors
      let mut i: u128 = 3;
      while i * i <= num {
        count = 0;
        while num % i == 0 {
          count += 1;
          num /= i;
          if count > 1 {
            return Ok(bool_expr(false));
          }
        }
        i += 2;
      }
      Ok(bool_expr(true))
    }
    Expr::BigInteger(n) => {
      use num_traits::Zero;
      let mut num = if n < &num_bigint::BigInt::from(0) {
        -n.clone()
      } else {
        n.clone()
      };
      if num.is_zero() {
        return Ok(bool_expr(false));
      }
      let two = num_bigint::BigInt::from(2);
      let mut count = 0;
      while &num % &two == num_bigint::BigInt::from(0) {
        count += 1;
        num /= &two;
        if count > 1 {
          return Ok(bool_expr(false));
        }
      }
      let mut i = num_bigint::BigInt::from(3);
      while &i * &i <= num {
        count = 0;
        while &num % &i == num_bigint::BigInt::from(0) {
          count += 1;
          num /= &i;
          if count > 1 {
            return Ok(bool_expr(false));
          }
        }
        i += 2;
      }
      Ok(bool_expr(true))
    }
    _ => Ok(Expr::FunctionCall {
      name: "SquareFreeQ".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// PalindromeQ[expr] - Tests if expr is a palindrome
/// Works with strings, lists, and integers
pub fn palindrome_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PalindromeQ expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::String(s) => {
      let is_palindrome = s.chars().eq(s.chars().rev());
      Ok(bool_expr(is_palindrome))
    }
    Expr::List(items) => {
      let is_palindrome = items.iter().zip(items.iter().rev()).all(|(a, b)| {
        crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
      });
      Ok(bool_expr(is_palindrome))
    }
    Expr::Integer(n) => {
      let s = n.to_string();
      let is_palindrome = s.chars().eq(s.chars().rev());
      Ok(bool_expr(is_palindrome))
    }
    Expr::BigInteger(n) => {
      let s = n.to_string();
      let is_palindrome = s.chars().eq(s.chars().rev());
      Ok(bool_expr(is_palindrome))
    }
    _ => Ok(Expr::FunctionCall {
      name: "PalindromeQ".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Divisible[n, m] - Tests if n is divisible by m
/// Returns unevaluated if arguments are not exact numbers (non-integer Reals)
pub fn divisible_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Divisible expects exactly 2 arguments".into(),
    ));
  }

  // Check if first argument is a non-exact number (Real with fractional part)
  let n = match &args[0] {
    Expr::Integer(n) => num_bigint::BigInt::from(*n),
    Expr::BigInteger(n) => n.clone(),
    Expr::Real(f) if f.fract() == 0.0 => num_bigint::BigInt::from(*f as i128),
    Expr::Real(_) => {
      // Non-exact number - return unevaluated
      return Ok(Expr::FunctionCall {
        name: "Divisible".to_string(),
        args: args.to_vec(),
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Divisible".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Check if second argument is a non-exact number
  let m = match &args[1] {
    Expr::Integer(m) => num_bigint::BigInt::from(*m),
    Expr::BigInteger(m) => m.clone(),
    Expr::Real(f) if f.fract() == 0.0 => num_bigint::BigInt::from(*f as i128),
    Expr::Real(_) => {
      // Non-exact number - return unevaluated
      return Ok(Expr::FunctionCall {
        name: "Divisible".to_string(),
        args: args.to_vec(),
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Divisible".to_string(),
        args: args.to_vec(),
      });
    }
  };

  {
    use num_traits::Zero;
    if m.is_zero() {
      return Err(InterpreterError::EvaluationError(
        "Divisible: divisor cannot be zero".into(),
      ));
    }

    Ok(bool_expr((n % m).is_zero()))
  }
}

/// Check if an expression represents a complex number (has nonzero imaginary part).
/// In Wolfram Language, I, 3*I, 2+3*I, Complex[a,b] are all Complex atoms.
fn is_complex_number(expr: &Expr) -> bool {
  // I itself is Complex[0, 1]
  if let Expr::Identifier(name) = expr {
    return name == "I";
  }
  // Check exact complex extraction (integer/rational components)
  if let Some((_, (im_num, _))) =
    super::math_ast::try_extract_complex_exact(expr)
    && im_num != 0
  {
    return true;
  }
  // Check float complex extraction
  if let Some((_, im)) = super::math_ast::try_extract_complex_float(expr)
    && im != 0.0
  {
    return true;
  }
  // Check structural presence of I (e.g. 0. + 0.*I)
  expr_contains_i(expr)
}

/// Check if an expression structurally contains the imaginary unit I
/// in a multiplication context (e.g. Times[0., I]).
fn expr_contains_i(expr: &Expr) -> bool {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left,
      right,
    } => expr_contains_i(left) || expr_contains_i(right),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      matches!(left.as_ref(), Expr::Identifier(s) if s == "I")
        || matches!(right.as_ref(), Expr::Identifier(s) if s == "I")
    }
    Expr::FunctionCall { name, args } if name == "Times" => args
      .iter()
      .any(|a| matches!(a, Expr::Identifier(s) if s == "I")),
    _ => false,
  }
}

/// Head[expr] - Returns the head of an expression
pub fn head_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Head expects exactly 1 argument".into(),
    ));
  }
  // Check for complex number patterns before the general match
  if is_complex_number(&args[0]) {
    return Ok(Expr::Identifier("Complex".to_string()));
  }
  let head = match &args[0] {
    Expr::Integer(_) | Expr::BigInteger(_) => "Integer",
    Expr::Real(_) | Expr::BigFloat(_, _) => "Real",
    Expr::String(_) => "String",
    Expr::Identifier(_) => "Symbol",
    Expr::List(_) => "List",
    Expr::Association(_) => "Association",
    Expr::FunctionCall { name, .. } => {
      return Ok(Expr::Identifier(name.clone()));
    }
    Expr::Rule { .. } => "Rule",
    Expr::RuleDelayed { .. } => "RuleDelayed",
    Expr::BinaryOp { op, .. } => {
      use crate::syntax::BinaryOperator;
      match op {
        BinaryOperator::Plus => "Plus",
        BinaryOperator::Minus => "Plus", // Minus is represented as Plus internally
        BinaryOperator::Times => "Times",
        BinaryOperator::Divide => "Times", // Divide is represented as Times internally
        BinaryOperator::Power => "Power",
        BinaryOperator::And => "And",
        BinaryOperator::Or => "Or",
        BinaryOperator::StringJoin => "StringJoin",
        BinaryOperator::Alternatives => "Alternatives",
      }
    }
    Expr::UnaryOp { op, .. } => {
      use crate::syntax::UnaryOperator;
      match op {
        UnaryOperator::Minus => "Times",
        UnaryOperator::Not => "Not",
      }
    }
    Expr::Comparison { .. } => "Comparison",
    Expr::Map { .. } => "Map",
    Expr::Apply { .. } => "Apply",
    Expr::Part { .. } => "Part",
    Expr::Function { .. } | Expr::NamedFunction { .. } => "Function",
    Expr::Pattern { name, .. } if name.is_empty() => "Blank",
    Expr::Pattern { .. } => "Pattern",
    Expr::PatternTest { .. } => "PatternTest",
    Expr::Image { .. } => "Image",
    Expr::Slot(_) => "Slot",
    _ => "Symbol",
  };
  Ok(Expr::Identifier(head.to_string()))
}

/// Length[expr] - Returns the number of elements at the top level of an expression
pub fn length_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Length expects exactly 1 argument".into(),
    ));
  }
  let len = match &args[0] {
    Expr::List(items) => items.len() as i128,
    Expr::Association(items) => items.len() as i128,
    // Rational[n, d] and Complex[re, im] are atoms with length 0
    Expr::FunctionCall { name, .. }
      if name == "Rational" || name == "Complex" =>
    {
      0
    }
    Expr::FunctionCall { args, .. } => args.len() as i128,
    Expr::BinaryOp { .. } => 2,
    Expr::UnaryOp { .. } => 1,
    Expr::Comparison { operands, .. } => operands.len() as i128,
    // Atoms: Integer, Real, String, Identifier, BigFloat, BigInteger, etc.
    _ => 0,
  };
  Ok(Expr::Integer(len))
}

/// Depth[expr] - Returns the depth of an expression
pub fn depth_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Depth expects exactly 1 argument".into(),
    ));
  }

  fn calc_depth(expr: &Expr) -> i128 {
    match expr {
      Expr::List(items) => 1 + items.iter().map(calc_depth).max().unwrap_or(0),
      Expr::FunctionCall { args, .. } => {
        1 + args.iter().map(calc_depth).max().unwrap_or(0)
      }
      Expr::Association(items) => {
        1 + items
          .iter()
          .flat_map(|(k, v)| [calc_depth(k), calc_depth(v)])
          .max()
          .unwrap_or(0)
      }
      _ => 1,
    }
  }

  Ok(Expr::Integer(calc_depth(&args[0])))
}

/// LeafCount[expr] - counts the number of atoms in the expression tree (including heads)
pub fn leaf_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LeafCount expects exactly 1 argument".into(),
    ));
  }

  fn count_leaves(expr: &Expr) -> i128 {
    match expr {
      // Atoms: count as 1
      Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(_, _)
      | Expr::String(_)
      | Expr::Identifier(_)
      | Expr::Constant(_) => 1,
      // List: 1 for the List head + sum of all elements
      Expr::List(items) => 1 + items.iter().map(count_leaves).sum::<i128>(),
      // FunctionCall: 1 for the head + sum of all args
      Expr::FunctionCall { args, .. } => {
        1 + args.iter().map(count_leaves).sum::<i128>()
      }
      // BinaryOp: 1 for the operator head + left + right
      Expr::BinaryOp { left, right, .. } => {
        1 + count_leaves(left) + count_leaves(right)
      }
      // UnaryOp: 1 for the operator head + operand
      Expr::UnaryOp { operand, .. } => 1 + count_leaves(operand),
      // Comparison: 1 for each operator + each operand
      Expr::Comparison { operands, .. } => {
        operands.iter().map(count_leaves).sum::<i128>()
      }
      // Association: 1 for head + entries
      Expr::Association(items) => {
        1 + items
          .iter()
          .map(|(k, v)| 1 + count_leaves(k) + count_leaves(v))
          .sum::<i128>()
      }
      // Anything else: treat as atom
      _ => 1,
    }
  }

  Ok(Expr::Integer(count_leaves(&args[0])))
}

/// Helper to format a real number
fn format_real_helper(f: f64) -> String {
  if f.fract() == 0.0 && f.abs() < 1e15 {
    format!("{:.1}", f)
  } else {
    format!("{}", f)
  }
}

/// Helper to convert Expr to its FullForm string representation
pub fn expr_to_full_form(expr: &Expr) -> String {
  match expr {
    Expr::Integer(n) => n.to_string(),
    Expr::BigInteger(n) => n.to_string(),
    Expr::Real(f) => format_real_helper(*f),
    Expr::BigFloat(digits, prec) => format!("{}`{}.", digits, prec),
    Expr::String(s) => format!("\"{}\"", s),
    Expr::Identifier(s) => s.clone(),
    Expr::Slot(n) => {
      if *n == 1 {
        "Slot[1]".to_string()
      } else {
        format!("Slot[{}]", n)
      }
    }
    Expr::Constant(c) => c.clone(),
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_full_form).collect();
      format!("List[{}]", parts.join(", "))
    }
    Expr::FunctionCall { name, args } => {
      let parts: Vec<String> = args.iter().map(expr_to_full_form).collect();
      format!("{}[{}]", name, parts.join(", "))
    }
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator;

      // Helper to collect all operands for associative operators (Plus, Times)
      fn collect_operands(
        expr: &Expr,
        target_op: &BinaryOperator,
      ) -> Vec<String> {
        match expr {
          Expr::BinaryOp { op, left, right } if op == target_op => {
            let mut parts = collect_operands(left, target_op);
            parts.extend(collect_operands(right, target_op));
            parts
          }
          _ => vec![expr_to_full_form(expr)],
        }
      }

      match op {
        BinaryOperator::Plus => {
          let parts = collect_operands(expr, &BinaryOperator::Plus);
          format!("Plus[{}]", parts.join(", "))
        }
        BinaryOperator::Times => {
          let parts = collect_operands(expr, &BinaryOperator::Times);
          format!("Times[{}]", parts.join(", "))
        }
        BinaryOperator::Minus => {
          format!(
            "Plus[{}, Times[-1, {}]]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
        BinaryOperator::Divide => {
          format!(
            "Times[{}, Power[{}, -1]]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
        BinaryOperator::Power => {
          format!(
            "Power[{}, {}]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
        BinaryOperator::And => {
          format!(
            "And[{}, {}]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
        BinaryOperator::Or => {
          format!(
            "Or[{}, {}]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
        BinaryOperator::StringJoin => {
          format!(
            "StringJoin[{}, {}]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
        BinaryOperator::Alternatives => {
          format!(
            "Alternatives[{}, {}]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
      }
    }
    Expr::UnaryOp { op, operand } => {
      use crate::syntax::UnaryOperator;
      match op {
        UnaryOperator::Minus => {
          format!("Times[-1, {}]", expr_to_full_form(operand))
        }
        UnaryOperator::Not => {
          format!("Not[{}]", expr_to_full_form(operand))
        }
      }
    }
    Expr::Comparison {
      operands,
      operators,
    } => {
      use crate::syntax::ComparisonOp;
      // For single comparison like a < b, return Less[a, b]
      if operators.len() == 1 {
        let func_name = match &operators[0] {
          ComparisonOp::Equal => "Equal",
          ComparisonOp::NotEqual => "Unequal",
          ComparisonOp::Less => "Less",
          ComparisonOp::LessEqual => "LessEqual",
          ComparisonOp::Greater => "Greater",
          ComparisonOp::GreaterEqual => "GreaterEqual",
          ComparisonOp::SameQ => "SameQ",
          ComparisonOp::UnsameQ => "UnsameQ",
        };
        let parts: Vec<String> =
          operands.iter().map(expr_to_full_form).collect();
        format!("{}[{}]", func_name, parts.join(", "))
      } else {
        // Mixed comparison chain
        let parts: Vec<String> =
          operands.iter().map(expr_to_full_form).collect();
        format!("Inequality[{}]", parts.join(", "))
      }
    }
    Expr::CompoundExpr(exprs) => {
      let parts: Vec<String> = exprs.iter().map(expr_to_full_form).collect();
      format!("CompoundExpression[{}]", parts.join(", "))
    }
    Expr::Association(items) => {
      let parts: Vec<String> = items
        .iter()
        .map(|(k, v)| {
          format!("Rule[{}, {}]", expr_to_full_form(k), expr_to_full_form(v))
        })
        .collect();
      format!("Association[{}]", parts.join(", "))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => {
      format!(
        "Rule[{}, {}]",
        expr_to_full_form(pattern),
        expr_to_full_form(replacement)
      )
    }
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      format!(
        "RuleDelayed[{}, {}]",
        expr_to_full_form(pattern),
        expr_to_full_form(replacement)
      )
    }
    Expr::ReplaceAll { expr, rules } => {
      format!(
        "ReplaceAll[{}, {}]",
        expr_to_full_form(expr),
        expr_to_full_form(rules)
      )
    }
    Expr::ReplaceRepeated { expr, rules } => {
      format!(
        "ReplaceRepeated[{}, {}]",
        expr_to_full_form(expr),
        expr_to_full_form(rules)
      )
    }
    Expr::Map { func, list } => {
      format!(
        "Map[{}, {}]",
        expr_to_full_form(func),
        expr_to_full_form(list)
      )
    }
    Expr::Apply { func, list } => {
      format!(
        "Apply[{}, {}]",
        expr_to_full_form(func),
        expr_to_full_form(list)
      )
    }
    Expr::MapApply { func, list } => {
      format!(
        "MapApply[{}, {}]",
        expr_to_full_form(func),
        expr_to_full_form(list)
      )
    }
    Expr::PrefixApply { func, arg } => {
      format!("{}[{}]", expr_to_full_form(func), expr_to_full_form(arg))
    }
    Expr::Postfix { expr, func } => {
      format!("{}[{}]", expr_to_full_form(func), expr_to_full_form(expr))
    }
    Expr::Part { expr, index } => {
      // Flatten nested Part into Part[expr, i1, i2, ...]
      let mut indices = vec![expr_to_full_form(index)];
      let mut base = expr.as_ref();
      while let Expr::Part {
        expr: inner_expr,
        index: inner_index,
      } = base
      {
        indices.push(expr_to_full_form(inner_index));
        base = inner_expr.as_ref();
      }
      indices.reverse();
      format!("Part[{}, {}]", expr_to_full_form(base), indices.join(", "))
    }
    Expr::Function { body } => {
      format!("Function[{}]", expr_to_full_form(body))
    }
    Expr::NamedFunction { params, body } => {
      if params.len() == 1 {
        format!("Function[{}, {}]", params[0], expr_to_full_form(body))
      } else {
        format!(
          "Function[List[{}], {}]",
          params.join(", "),
          expr_to_full_form(body)
        )
      }
    }
    Expr::Pattern { name, head } => {
      if let Some(h) = head {
        format!("Pattern[{}, Blank[{}]]", name, h)
      } else {
        format!("Pattern[{}, Blank[]]", name)
      }
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => {
      let pattern_part = if let Some(h) = head {
        format!("Pattern[{}, Blank[{}]]", name, h)
      } else {
        format!("Pattern[{}, Blank[]]", name)
      };
      format!("Optional[{}, {}]", pattern_part, expr_to_full_form(default))
    }
    Expr::PatternTest { name, test } => {
      let blank_part = if name.is_empty() {
        "Blank[]".to_string()
      } else {
        format!("Pattern[{}, Blank[]]", name)
      };
      format!("PatternTest[{}, {}]", blank_part, expr_to_full_form(test))
    }
    Expr::Raw(s) => s.clone(),
    Expr::Image { .. } => "-Image-".to_string(),
    Expr::CurriedCall { func, args } => {
      // CurriedCall[f[a]][b] displays as f[a][b]
      let args_str: Vec<String> = args.iter().map(expr_to_full_form).collect();
      format!("{}[{}]", expr_to_full_form(func), args_str.join(", "))
    }
  }
}

/// FullForm[expr] - Returns a symbolic FullForm wrapper (like wolframscript).
/// The display layer (expr_to_output) renders the inner expr in FullForm notation.
pub fn full_form_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  Ok(Expr::FunctionCall {
    name: "FullForm".to_string(),
    args: vec![arg.clone()],
  })
}

/// Construct[f, a, b, c, ...] - Creates function call f[a, b, c, ...]
/// In Wolfram, Construct[f, a, b, c] returns f[a, b, c] (all args together)
/// To get curried calls like f[a][b][c], use Fold[Construct, f, {a, b, c}]
pub fn construct_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Construct expects at least 1 argument".into(),
    ));
  }

  if args.len() == 1 {
    // Construct[f] returns f
    return Ok(args[0].clone());
  }

  // Get the head (function name)
  let head = &args[0];
  let func_args = &args[1..];

  // Build the function name from the head expression
  let head_name = match head {
    Expr::Identifier(name) => name.clone(),
    Expr::FunctionCall {
      name,
      args: inner_args,
    } if inner_args.is_empty() => name.clone(),
    // If head is already a function call like f[a], we need to use it as string for nested application
    _ => crate::syntax::expr_to_string(head),
  };

  // Construct[f, a, b, c] => f[a, b, c]
  Ok(Expr::FunctionCall {
    name: head_name,
    args: func_args.to_vec(),
  })
}

/// LeapYearQ[{year}] - Tests if a year is a leap year
pub fn leap_year_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let year = match &args[0] {
    Expr::List(items) if !items.is_empty() => match &items[0] {
      Expr::Integer(n) => *n,
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        match n.to_i128() {
          Some(v) => v,
          None => return Ok(bool_expr(false)),
        }
      }
      _ => return Ok(bool_expr(false)),
    },
    Expr::Integer(_) | Expr::BigInteger(_) => return Ok(bool_expr(false)),
    _ => return Ok(bool_expr(false)),
  };
  let is_leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
  Ok(bool_expr(is_leap))
}

/// MatchQ[expr, pattern] - Tests if an expression matches a pattern
pub fn match_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MatchQ expects exactly 2 arguments".into(),
    ));
  }
  let matches =
    crate::functions::list_helpers_ast::matches_pattern_ast(&args[0], &args[1]);
  Ok(bool_expr(matches))
}

/// SubsetQ[list1, list2] - Tests if list2 is a subset of list1
pub fn subset_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "SubsetQ expects exactly 2 arguments".into(),
    ));
  }
  match (&args[0], &args[1]) {
    (Expr::List(superset), Expr::List(subset)) => {
      // Check that every element in subset appears in superset
      let superset_strs: Vec<String> =
        superset.iter().map(crate::syntax::expr_to_string).collect();
      for elem in subset {
        let s = crate::syntax::expr_to_string(elem);
        if !superset_strs.contains(&s) {
          return Ok(Expr::Identifier("False".to_string()));
        }
      }
      Ok(Expr::Identifier("True".to_string()))
    }
    _ => Ok(Expr::FunctionCall {
      name: "SubsetQ".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// OptionQ[expr] - Tests if expr is a Rule or RuleDelayed or a list thereof
pub fn option_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "OptionQ expects exactly 1 argument".into(),
    ));
  }
  fn is_option(expr: &Expr) -> bool {
    match expr {
      Expr::Rule { .. } | Expr::RuleDelayed { .. } => true,
      Expr::List(items) => items.iter().all(is_option),
      _ => false,
    }
  }
  Ok(bool_expr(is_option(&args[0])))
}
