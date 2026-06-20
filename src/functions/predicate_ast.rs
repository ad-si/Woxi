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

/// RealValuedNumericQ[expr] - True if `expr` is a numeric quantity whose value
/// is a real number. Unlike RealValuedNumberQ (which requires an explicit real
/// number literal), this accepts any numeric expression — constants, exact
/// irrationals, and numeric functions — provided its value is real:
/// `RealValuedNumericQ[Pi]`, `RealValuedNumericQ[Sqrt[2]]`,
/// `RealValuedNumericQ[Sin[1]]` are all True, while `I`, complex numbers,
/// `Sqrt[-1]`, symbols, and strings are False.
pub fn real_valued_numeric_q_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RealValuedNumericQ expects exactly 1 argument".into(),
    ));
  }
  // Must be numeric to begin with.
  if !is_numeric_q(&args[0]) {
    return Ok(bool_expr(false));
  }
  // Numerically evaluate and check the result is a real-number atom (not a
  // complex value, which carries the head Complex or stays as `I`).
  let n_call = Expr::FunctionCall {
    name: "N".to_string(),
    args: vec![args[0].clone()].into(),
  };
  let evaluated = crate::evaluator::evaluate_expr_to_expr(&n_call)?;
  let is_real = matches!(
    evaluated,
    Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(_, _)
  ) || matches!(
    &evaluated,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2
  );
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

/// An exact element of a real quadratic field: `a + b*sqrt(d)` with `a`, `b`
/// rational and `d` a square-free integer > 1 (so a non-zero `b` means the
/// value is a quadratic irrational). A pure rational has `b == 0`.
#[derive(Clone, Copy)]
struct QuadNum {
  a: (i128, i128), // rational part (num, den), den > 0
  b: (i128, i128), // sqrt coefficient (num, den), den > 0
  d: i128,         // square-free radicand > 1 (only meaningful when b != 0)
}

fn qi_gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = a % b;
    a = b;
    b = t;
  }
  a.max(1)
}

fn qi_reduce(n: i128, d: i128) -> (i128, i128) {
  if d == 0 {
    return (n, 0);
  }
  let g = qi_gcd(n, d);
  let (mut n, mut d) = (n / g, d / g);
  if d < 0 {
    n = -n;
    d = -d;
  }
  (n, d)
}

fn qi_add(x: (i128, i128), y: (i128, i128)) -> (i128, i128) {
  qi_reduce(x.0 * y.1 + y.0 * x.1, x.1 * y.1)
}

fn qi_mul(x: (i128, i128), y: (i128, i128)) -> (i128, i128) {
  qi_reduce(x.0 * y.0, x.1 * y.1)
}

impl QuadNum {
  fn rational(n: i128, den: i128) -> QuadNum {
    QuadNum {
      a: qi_reduce(n, den),
      b: (0, 1),
      d: 1,
    }
  }
  fn is_rational(&self) -> bool {
    self.b.0 == 0
  }
  /// Add two quadratic numbers, requiring a common field (or one rational).
  fn add(self, o: QuadNum) -> Option<QuadNum> {
    if self.is_rational() {
      return Some(QuadNum {
        a: qi_add(self.a, o.a),
        b: o.b,
        d: o.d,
      });
    }
    if o.is_rational() {
      return Some(QuadNum {
        a: qi_add(self.a, o.a),
        b: self.b,
        d: self.d,
      });
    }
    if self.d != o.d {
      return None; // different quadratic fields
    }
    Some(QuadNum {
      a: qi_add(self.a, o.a),
      b: qi_add(self.b, o.b),
      d: self.d,
    })
  }
  fn neg(self) -> QuadNum {
    QuadNum {
      a: (-self.a.0, self.a.1),
      b: (-self.b.0, self.b.1),
      d: self.d,
    }
  }
  /// Multiply: (a1 + b1√d)(a2 + b2√d) = (a1a2 + b1b2 d) + (a1b2 + a2b1)√d.
  fn mul(self, o: QuadNum) -> Option<QuadNum> {
    if self.is_rational() {
      return Some(QuadNum {
        a: qi_mul(self.a, o.a),
        b: qi_mul(self.a, o.b),
        d: o.d,
      });
    }
    if o.is_rational() {
      return Some(QuadNum {
        a: qi_mul(self.a, o.a),
        b: qi_mul(self.b, o.a),
        d: self.d,
      });
    }
    if self.d != o.d {
      return None;
    }
    let cross = qi_add(qi_mul(self.a, o.b), qi_mul(self.b, o.a));
    let rat = qi_add(
      qi_mul(self.a, o.a),
      qi_mul(qi_mul(self.b, o.b), (self.d, 1)),
    );
    Some(QuadNum {
      a: rat,
      b: cross,
      d: self.d,
    })
  }
  /// Inverse: 1/(a + b√d) = (a - b√d) / (a² - b² d).
  fn inv(self) -> Option<QuadNum> {
    let denom = qi_add(
      qi_mul(self.a, self.a),
      (
        -qi_mul(qi_mul(self.b, self.b), (self.d, 1)).0,
        qi_mul(qi_mul(self.b, self.b), (self.d, 1)).1,
      ),
    );
    if denom.0 == 0 {
      return None;
    }
    Some(QuadNum {
      a: qi_mul(self.a, (denom.1, denom.0)),
      b: qi_mul((-self.b.0, self.b.1), (denom.1, denom.0)),
      d: self.d,
    })
  }
}

/// Extract `k` and square-free `m` from `n = k^2 * m` for a positive integer.
fn qi_square_free(mut n: i128) -> (i128, i128) {
  let mut k = 1i128;
  let mut f = 2i128;
  while f * f <= n {
    while n % (f * f) == 0 {
      n /= f * f;
      k *= f;
    }
    f += 1;
  }
  (k, n)
}

/// Check if an expression is Rational[-1, 2].
fn is_neg_half(expr: &Expr) -> bool {
  matches!(
    expr,
    Expr::FunctionCall { name, args }
      if name == "Rational"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(-1))
        && matches!(&args[1], Expr::Integer(2))
  )
}

/// `Sqrt[r]` for an exact rational radicand `r`, as a quadratic number.
fn sqrt_of_rational(rad: &Expr) -> Option<QuadNum> {
  let (p, q) = match rad {
    Expr::Integer(n) => (*n, 1),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(q)) => (*p, *q),
        _ => return None,
      }
    }
    _ => return None,
  };
  if p <= 0 || q <= 0 {
    return None; // not a positive real radicand
  }
  // sqrt(p/q) = sqrt(p*q)/q.
  let (k, m) = qi_square_free(p * q);
  if m == 1 {
    Some(QuadNum::rational(k, q))
  } else {
    Some(QuadNum {
      a: (0, 1),
      b: qi_reduce(k, q),
      d: m,
    })
  }
}

/// Try to interpret an exact expression as an element of a single real
/// quadratic field. Returns `None` for anything outside (machine reals,
/// symbols, non-real radicals, higher-degree algebraics, …).
fn as_quad_num(expr: &Expr) -> Option<QuadNum> {
  use crate::syntax::{BinaryOperator, UnaryOperator};
  // Sqrt[r] (matches both the Power[r, 1/2] and Sqrt[r] representations).
  if let Some(rad) = crate::functions::is_sqrt(expr) {
    return sqrt_of_rational(rad);
  }
  match expr {
    Expr::Integer(n) => Some(QuadNum::rational(*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&args[0], &args[1]) {
        Some(QuadNum::rational(*p, *q))
      } else {
        None
      }
    }
    // 1/Sqrt[r], represented as Power[r, Rational[-1, 2]].
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } if is_neg_half(right) => sqrt_of_rational(left)?.inv(),
    Expr::FunctionCall { name, args }
      if name == "Power" && args.len() == 2 && is_neg_half(&args[1]) =>
    {
      sqrt_of_rational(&args[0])?.inv()
    }
    Expr::Constant(c) | Expr::Identifier(c) if c == "GoldenRatio" => {
      // (1 + Sqrt[5]) / 2.
      Some(QuadNum {
        a: (1, 2),
        b: (1, 2),
        d: 5,
      })
    }
    Expr::BinaryOp { op, left, right } => {
      let l = as_quad_num(left)?;
      let r = as_quad_num(right)?;
      match op {
        BinaryOperator::Plus => l.add(r),
        BinaryOperator::Times => l.mul(r),
        BinaryOperator::Divide => l.mul(r.inv()?),
        _ => None,
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(as_quad_num(operand)?.neg()),
    Expr::FunctionCall { name, args } if name == "Plus" && !args.is_empty() => {
      let mut acc = as_quad_num(&args[0])?;
      for a in &args[1..] {
        acc = acc.add(as_quad_num(a)?)?;
      }
      Some(acc)
    }
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      let mut acc = as_quad_num(&args[0])?;
      for a in &args[1..] {
        acc = acc.mul(as_quad_num(a)?)?;
      }
      Some(acc)
    }
    _ => None,
  }
}

/// QuadraticIrrationalQ[x] - True if x is a real quadratic irrational.
pub fn quadratic_irrational_q_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "QuadraticIrrationalQ expects exactly 1 argument".into(),
    ));
  }
  let is_qi = match as_quad_num(&args[0]) {
    Some(q) => !q.is_rational() && q.d > 1,
    None => false,
  };
  Ok(bool_expr(is_qi))
}

/// EvenQ[n] - Tests if a number is even
pub fn even_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "EvenQ expects exactly 1 argument".into(),
    ));
  }
  // EvenQ applies only to actual integers. A machine Real (even a
  // whole-valued one like 2.0) is not an integer, so wolframscript returns
  // False, e.g. EvenQ[2.0] = False.
  let is_even = match &args[0] {
    Expr::Integer(n) => n % 2 == 0,
    Expr::BigInteger(n) => {
      use num_traits::Zero;
      (n % num_bigint::BigInt::from(2)).is_zero()
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
  // OddQ applies only to actual integers; a machine Real (e.g. 3.0) is not
  // an integer, so wolframscript returns False.
  let is_odd = match &args[0] {
    Expr::Integer(n) => n % 2 != 0,
    Expr::BigInteger(n) => {
      use num_traits::Zero;
      !(n % num_bigint::BigInt::from(2)).is_zero()
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
  let is_atom = if is_complex_number(&args[0]) {
    true
  } else {
    match &args[0] {
      Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::String(_)
      | Expr::Identifier(_)
      | Expr::Constant(_) => true,
      // Rational[n, d] and Complex[re, im] are atoms in Mathematica
      Expr::FunctionCall { name, .. }
        if name == "Rational"
          || name == "Complex"
          || name == "ByteArray"
          || name == "NumericArray" =>
      {
        true
      }
      _ => false,
    }
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
pub fn is_numeric_q_pub(expr: &Expr) -> bool {
  is_numeric_q(expr)
}

fn is_numeric_q(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _) => true,
    Expr::Constant(_) => true,
    Expr::Identifier(name) => {
      // I and named mathematical constants are numeric
      // Infinity is NOT numeric in Wolfram Language
      // (NumericQ[Infinity] returns False)
      if matches!(
        name.as_str(),
        "I"
          | "GoldenRatio"
          | "EulerGamma"
          | "Catalan"
          | "Khinchin"
          | "Glaisher"
      ) {
        return true;
      }
      // Check for user-defined NumericQ downvalue (e.g., NumericQ[a] = True)
      // FUNC_DEFS stores (params, conditions, defaults, heads, body)
      crate::FUNC_DEFS.with(|m| {
        let defs = m.borrow();
        if let Some(overloads) = defs.get("NumericQ") {
          for (params, conditions, _defaults, _heads, _blank_types, body) in
            overloads
          {
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
/// Numerically evaluate a NumericQ expression (e.g. Sin[11]) and apply
/// `cmp` to the resulting f64. Returns None when the expression isn't
/// purely real-numeric (e.g. it has free user symbols, or evaluates to
/// a complex / non-finite value).
fn sign_via_numeric(expr: &Expr, cmp: impl Fn(f64) -> bool) -> Option<bool> {
  // Only fire when NumericQ would succeed — that's how Wolfram decides
  // it can numerically evaluate. Avoids forcing Negative[a] to True or
  // anything weird for unevaluated user symbols.
  if !is_numeric_q_pub(expr) {
    return None;
  }
  let val = super::math_ast::try_eval_to_f64(expr)?;
  if !val.is_finite() {
    return None;
  }
  Some(cmp(val))
}

/// Detect a non-real complex value: Complex[_, b] with non-zero b, or its
/// expanded forms `b*I`, `a + b*I`, etc.
fn has_nonzero_imag_part(expr: &Expr) -> bool {
  use super::math_ast::{try_extract_complex_exact, try_extract_complex_float};
  if let Some((_, (im_n, _))) = try_extract_complex_exact(expr)
    && im_n != 0
  {
    return true;
  }
  if let Some((_, im)) = try_extract_complex_float(expr)
    && im != 0.0
    && expr_has_real_or_i(expr)
  {
    return true;
  }
  false
}

/// True if `expr` contains a Real/BigFloat or the literal symbol `I` (so
/// the float-extraction result is meaningful and not just a coerced symbolic
/// constant like Pi).
fn expr_has_real_or_i(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::Identifier(s) if s == "I" => true,
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(expr_has_real_or_i)
    }
    Expr::BinaryOp { left, right, .. } => {
      expr_has_real_or_i(left) || expr_has_real_or_i(right)
    }
    Expr::UnaryOp { operand, .. } => expr_has_real_or_i(operand),
    _ => false,
  }
}

/// Check if an expression is known to be strictly positive.
fn is_known_positive(expr: &Expr) -> Option<bool> {
  match expr {
    Expr::Integer(n) => Some(*n > 0),
    Expr::BigInteger(n) => Some(*n > num_bigint::BigInt::from(0)),
    Expr::Real(f) => Some(*f > 0.0),
    // Rational[n, d]: positive when n and d have the same sign
    Expr::FunctionCall { name, args }
      if name == "Rational"
        && args.len() == 2
        && matches!(
          (&args[0], &args[1]),
          (Expr::Integer(_), Expr::Integer(_))
        ) =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n > 0 && *d > 0) || (*n < 0 && *d < 0))
      } else {
        None
      }
    }
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
    // Log[c] is positive iff c > 1 (for real positive c).
    Expr::FunctionCall { name, args } if name == "Log" && args.len() == 1 => {
      match &args[0] {
        Expr::Integer(n) => Some(*n > 1),
        Expr::Real(f) => Some(*f > 1.0),
        Expr::Constant(c) if c == "E" || c == "Pi" => Some(true),
        _ => None,
      }
    }
    // Sqrt[c] is positive when c is positive.
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      is_known_positive(&args[0])
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
    // Rational[n, d]: negative when n and d have opposite signs
    Expr::FunctionCall { name, args }
      if name == "Rational"
        && args.len() == 2
        && matches!(
          (&args[0], &args[1]),
          (Expr::Integer(_), Expr::Integer(_))
        ) =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n > 0 && *d < 0) || (*n < 0 && *d > 0))
      } else {
        None
      }
    }
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

/// Sign predicate on a single-segment `Interval[{a, b}]`. Returns `Some` with
/// the result (a Boolean, or the unevaluated call when the interval straddles
/// zero and the sign is indeterminate), or `None` when `expr` is not such an
/// interval. `name` selects Positive / Negative / NonNegative / NonPositive.
fn interval_sign(expr: &Expr, name: &str) -> Option<Expr> {
  let (a, b) = match expr {
    Expr::FunctionCall {
      name: head,
      args: iargs,
    } if head == "Interval" && iargs.len() == 1 => match &iargs[0] {
      Expr::List(seg) if seg.len() == 2 => {
        let a = super::math_ast::try_eval_to_f64(&seg[0])?;
        let b = super::math_ast::try_eval_to_f64(&seg[1])?;
        (a, b)
      }
      _ => return None,
    },
    _ => return None,
  };
  // Strict straddle: the sign cannot be determined.
  if a < 0.0 && b > 0.0 {
    return Some(Expr::FunctionCall {
      name: name.to_string(),
      args: vec![expr.clone()].into(),
    });
  }
  let val = match name {
    "Positive" => a > 0.0,
    "Negative" => b < 0.0,
    "NonNegative" => a >= 0.0,
    "NonPositive" => b <= 0.0,
    _ => return None,
  };
  Some(bool_expr(val))
}

pub fn positive_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PositiveQ expects exactly 1 argument".into(),
    ));
  }
  if let Some(r) = interval_sign(&args[0], "Positive") {
    return Ok(r);
  }
  // Positive on a non-real complex value is False.
  if has_nonzero_imag_part(&args[0]) {
    return Ok(bool_expr(false));
  }
  if let Some(val) = is_known_positive(&args[0]) {
    return Ok(bool_expr(val));
  }
  // NumericQ inputs (e.g. Sin[11]) get evaluated to a real f64 to decide.
  if let Some(b) = sign_via_numeric(&args[0], |f| f > 0.0) {
    return Ok(bool_expr(b));
  }
  Ok(Expr::FunctionCall {
    name: "Positive".to_string(),
    args: args.to_vec().into(),
  })
}

/// NegativeQ[x] - Tests if x is a negative number
pub fn negative_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NegativeQ expects exactly 1 argument".into(),
    ));
  }
  if let Some(r) = interval_sign(&args[0], "Negative") {
    return Ok(r);
  }
  // Negative on a non-real complex value is False.
  if has_nonzero_imag_part(&args[0]) {
    return Ok(bool_expr(false));
  }
  if let Some(val) = is_known_negative(&args[0]) {
    return Ok(bool_expr(val));
  }
  if let Some(b) = sign_via_numeric(&args[0], |f| f < 0.0) {
    return Ok(bool_expr(b));
  }
  Ok(Expr::FunctionCall {
    name: "Negative".to_string(),
    args: args.to_vec().into(),
  })
}

/// NonPositiveQ[x] - Tests if x is <= 0
pub fn non_positive_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NonPositiveQ expects exactly 1 argument".into(),
    ));
  }
  if let Some(r) = interval_sign(&args[0], "NonPositive") {
    return Ok(r);
  }
  // A non-real complex value is not NonPositive.
  if has_nonzero_imag_part(&args[0]) {
    return Ok(bool_expr(false));
  }
  // NonPositive: x <= 0, i.e. negative or zero
  if is_zero(&args[0]) {
    return Ok(bool_expr(true));
  }
  if let Some(val) = is_known_negative(&args[0]) {
    return Ok(bool_expr(val));
  }
  // Real-valued numeric expressions (Pi - 3, Sqrt[2] - 2, …) are decided
  // by their numeric value, matching Positive/Negative.
  if let Some(b) = sign_via_numeric(&args[0], |f| f <= 0.0) {
    return Ok(bool_expr(b));
  }
  Ok(Expr::FunctionCall {
    name: "NonPositive".to_string(),
    args: args.to_vec().into(),
  })
}

/// NonNegativeQ[x] - Tests if x is >= 0
pub fn non_negative_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NonNegativeQ expects exactly 1 argument".into(),
    ));
  }
  if let Some(r) = interval_sign(&args[0], "NonNegative") {
    return Ok(r);
  }
  // A non-real complex value is not NonNegative.
  if has_nonzero_imag_part(&args[0]) {
    return Ok(bool_expr(false));
  }
  // NonNegative: x >= 0, i.e. positive or zero
  if is_zero(&args[0]) {
    return Ok(bool_expr(true));
  }
  if let Some(val) = is_known_positive(&args[0]) {
    return Ok(bool_expr(val));
  }
  // Real-valued numeric expressions (Pi - 3, Sqrt[2] - 2, …) are decided
  // by their numeric value, matching Positive/Negative.
  if let Some(b) = sign_via_numeric(&args[0], |f| f >= 0.0) {
    return Ok(bool_expr(b));
  }
  Ok(Expr::FunctionCall {
    name: "NonNegative".to_string(),
    args: args.to_vec().into(),
  })
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
    let abs_n = if n.is_negative() {
      -n.clone()
    } else {
      n.clone()
    };
    return Ok(bool_expr(crate::functions::math_ast::is_prime_bigint(
      &abs_n,
    )));
  }
  let n = match &args[0] {
    Expr::Integer(n) => n.abs(),
    Expr::Real(f) if f.fract() == 0.0 && *f > 0.0 => *f as i128,
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
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "MemberQ expects 2 or 3 arguments".into(),
    ));
  }

  let pattern = &args[1];

  // Parse optional level spec (3rd argument)
  let level_spec = if args.len() == 3 {
    parse_level_spec(&args[2])
  } else {
    (1, 1) // default: level {1}
  };

  fn search_at_levels(
    expr: &Expr,
    pattern: &Expr,
    level_spec: (i64, i64),
    current_level: i64,
  ) -> bool {
    if current_level >= level_spec.0
      && current_level <= level_spec.1
      && crate::functions::list_helpers_ast::matches_pattern_ast(expr, pattern)
    {
      return true;
    }
    if current_level < level_spec.1 {
      match expr {
        Expr::List(items) => {
          for item in items {
            if search_at_levels(item, pattern, level_spec, current_level + 1) {
              return true;
            }
          }
        }
        Expr::Association(pairs) => {
          for (_, v) in pairs {
            if search_at_levels(v, pattern, level_spec, current_level + 1) {
              return true;
            }
          }
        }
        Expr::FunctionCall { args: fn_args, .. } => {
          for arg in fn_args {
            if search_at_levels(arg, pattern, level_spec, current_level + 1) {
              return true;
            }
          }
        }
        _ => {}
      }
    }
    false
  }

  Ok(bool_expr(search_at_levels(
    &args[0], pattern, level_spec, 0,
  )))
}

/// Parse a level spec like {2}, {1, 3}, Infinity, or a plain integer.
fn parse_level_spec(spec: &Expr) -> (i64, i64) {
  use num_traits::ToPrimitive;
  match spec {
    Expr::Integer(n) => (1, n.to_i64().unwrap_or(1)),
    Expr::Identifier(s) if s == "Infinity" => (1, i64::MAX),
    Expr::List(items) if items.len() == 1 => {
      if let Expr::Integer(n) = &items[0] {
        let lvl = n.to_i64().unwrap_or(1);
        (lvl, lvl)
      } else {
        (1, 1)
      }
    }
    Expr::List(items) if items.len() == 2 => {
      let lo = if let Expr::Integer(n) = &items[0] {
        n.to_i64().unwrap_or(1)
      } else {
        1
      };
      let hi = if let Expr::Integer(n) = &items[1] {
        n.to_i64().unwrap_or(1)
      } else if let Expr::Identifier(s) = &items[1] {
        if s == "Infinity" { i64::MAX } else { 1 }
      } else {
        1
      };
      (lo, hi)
    }
    _ => (1, 1),
  }
}

/// FreeQ[expr, form] - Tests if expr is free of form
pub fn free_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // FreeQ[expr, form, levelspec]: True when no part of expr at the
  // given levels matches form. Invalid specs emit ::level.
  if args.len() == 3 {
    let is_level_int = |e: &Expr| {
      matches!(e, Expr::Integer(_))
        || matches!(e, Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity")
        || matches!(e, Expr::FunctionCall { name, args }
          if name == "DirectedInfinity" && args.len() == 1)
    };
    let spec_ok = match &args[2] {
      e if is_level_int(e) => true,
      Expr::List(parts) => match parts.as_slice() {
        [n] => matches!(n, Expr::Integer(_)),
        [m, n] => is_level_int(m) && is_level_int(n),
        _ => false,
      },
      _ => false,
    };
    if !spec_ok {
      crate::emit_message(&format!(
        "FreeQ::level: Level specification {} is not of the form n, {{n}} or {{m, n}}.",
        crate::syntax::format_expr(&args[2], crate::syntax::ExprForm::Output)
      ));
      return Ok(Expr::FunctionCall {
        name: "FreeQ".to_string(),
        args: args.to_vec().into(),
      });
    }
    let parts = crate::functions::list_helpers_ast::level_unified_ast(&[
      args[0].clone(),
      args[2].clone(),
    ])?;
    if let Expr::List(ref parts) = parts {
      let any = parts.iter().any(|p| {
        crate::functions::list_helpers_ast::matches_pattern_ast(p, &args[1])
      });
      return Ok(Expr::Identifier(
        if any { "False" } else { "True" }.to_string(),
      ));
    }
    return Ok(Expr::FunctionCall {
      name: "FreeQ".to_string(),
      args: args.to_vec().into(),
    });
  }
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

  fn contains_form(
    expr: &Expr,
    form: &Expr,
    form_str: &str,
    use_pattern: bool,
  ) -> bool {
    // Check pattern match or exact string match
    if use_pattern {
      if crate::functions::list_helpers_ast::matches_pattern_ast(expr, form) {
        return true;
      }
    } else if crate::syntax::expr_to_string(expr) == form_str {
      return true;
    }
    match expr {
      Expr::List(items) => {
        // Check if form matches the head "List"
        if is_form_symbol(form, "List") {
          return true;
        }
        // When using pattern matching, also check if the head identifier
        // itself matches the pattern (e.g. _Symbol matches "List" since
        // Head[List] is Symbol).
        if use_pattern {
          let head_expr = Expr::Identifier("List".to_string());
          if crate::functions::list_helpers_ast::matches_pattern_ast(
            &head_expr, form,
          ) {
            return true;
          }
        }
        items
          .iter()
          .any(|e| contains_form(e, form, form_str, use_pattern))
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
        // When using pattern matching, also check if the head identifier
        // itself matches the pattern.
        if use_pattern {
          let head_expr = Expr::Identifier(name.clone());
          if crate::functions::list_helpers_ast::matches_pattern_ast(
            &head_expr, form,
          ) {
            return true;
          }
        }
        // For Flat+Orderless functions (Plus, Times), check if form's args
        // are a subset of this function's args
        if !use_pattern
          && let Expr::FunctionCall {
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
        fn_args
          .iter()
          .any(|e| contains_form(e, form, form_str, use_pattern))
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
        contains_form(left, form, form_str, use_pattern)
          || contains_form(right, form, form_str, use_pattern)
      }
      Expr::UnaryOp { operand, .. } => {
        contains_form(operand, form, form_str, use_pattern)
      }
      _ => false,
    }
  }

  // Detect if the form is or contains a pattern (Blank, Pattern, etc.)
  fn contains_pattern(form: &Expr) -> bool {
    if matches!(
      form,
      Expr::Pattern { .. }
        | Expr::PatternTest { .. }
        | Expr::PatternOptional { .. }
    ) {
      return true;
    }
    if matches!(form, Expr::FunctionCall { name, .. }
      if name == "Blank" || name == "BlankSequence" || name == "BlankNullSequence"
        || name == "Alternatives" || name == "Repeated" || name == "RepeatedNull"
        || name == "Pattern" || name == "Except" || name == "Condition"
        || name == "PatternTest"
    ) {
      return true;
    }
    if matches!(
      form,
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Alternatives,
        ..
      }
    ) {
      return true;
    }
    match form {
      Expr::FunctionCall { args, .. } => args.iter().any(contains_pattern),
      Expr::BinaryOp { left, right, .. } => {
        contains_pattern(left) || contains_pattern(right)
      }
      Expr::UnaryOp { operand, .. } => contains_pattern(operand),
      Expr::List(items) => items.iter().any(contains_pattern),
      _ => false,
    }
  }
  let use_pattern = contains_pattern(form);

  Ok(bool_expr(!contains_form(
    &args[0],
    form,
    &form_str,
    use_pattern,
  )))
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
      args: args.to_vec().into(),
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
      args: args.to_vec().into(),
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
        args: args.to_vec().into(),
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Divisible".to_string(),
        args: args.to_vec().into(),
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
        args: args.to_vec().into(),
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Divisible".to_string(),
        args: args.to_vec().into(),
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
/// Symbolic combinations like `Pi*I` or `a*I` stay as Times (not Complex).
pub fn is_complex_number(expr: &Expr) -> bool {
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
  // Float complex extraction: only treat as Complex when the expression
  // actually contains a Real / BigFloat literal — symbolic constants like
  // Pi or E should keep Times[Pi, I] as a Times expression, not coerce to
  // a float complex.
  if contains_real_literal(expr)
    && let Some((_, im)) = super::math_ast::try_extract_complex_float(expr)
    && im != 0.0
  {
    return true;
  }
  false
}

/// Check whether the expression contains any actual Real/BigFloat node.
fn contains_real_literal(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::BinaryOp { left, right, .. } => {
      contains_real_literal(left) || contains_real_literal(right)
    }
    Expr::UnaryOp { operand, .. } => contains_real_literal(operand),
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(contains_real_literal)
    }
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
    // f[a][b] — the head is the inner call expression itself, not a symbol.
    Expr::CurriedCall { func, .. } => {
      return Ok((**func).clone());
    }
    Expr::Rule { .. } => "Rule",
    Expr::RuleDelayed { .. } => "RuleDelayed",
    Expr::BinaryOp { op, left, .. } => {
      use crate::syntax::BinaryOperator;
      match op {
        BinaryOperator::Plus => "Plus",
        BinaryOperator::Minus => "Plus", // Minus is represented as Plus internally
        BinaryOperator::Times => "Times",
        // In Wolfram, a/b is Times[a, Power[b, -1]], but 1/b is Power[b, -1]
        BinaryOperator::Divide => {
          if matches!(left.as_ref(), Expr::Integer(1)) {
            "Power"
          } else {
            "Times"
          }
        }
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
    Expr::Comparison { operators, .. } => {
      use crate::syntax::ComparisonOp;
      // A uniform chain (all operators the same) has head equal to that operator.
      // A mixed chain has head "Inequality". A single-op comparison also uses
      // the operator name directly.
      let first = operators.first().copied();
      let all_same = operators.iter().all(|op| Some(*op) == first);
      if all_same {
        match first {
          Some(ComparisonOp::Equal) => "Equal",
          Some(ComparisonOp::NotEqual) => "Unequal",
          Some(ComparisonOp::Less) => "Less",
          Some(ComparisonOp::LessEqual) => "LessEqual",
          Some(ComparisonOp::Greater) => "Greater",
          Some(ComparisonOp::GreaterEqual) => "GreaterEqual",
          Some(ComparisonOp::SameQ) => "SameQ",
          Some(ComparisonOp::UnsameQ) => "UnsameQ",
          None => "Equal",
        }
      } else {
        "Inequality"
      }
    }
    Expr::Map { .. } => "Map",
    Expr::Apply { .. } => "Apply",
    Expr::Part { .. } => "Part",
    Expr::Function { .. } | Expr::NamedFunction { .. } => "Function",
    Expr::Pattern {
      name, blank_type, ..
    } if name.is_empty() => match blank_type {
      2 => "BlankSequence",
      3 => "BlankNullSequence",
      _ => "Blank",
    },
    Expr::Pattern { .. } => "Pattern",
    Expr::PatternTest { .. } => "PatternTest",
    Expr::Image { .. } => "Image",
    Expr::Graphics { is_3d, .. } => {
      if *is_3d {
        "Graphics3D"
      } else {
        "Graphics"
      }
    }
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
  // Unevaluated[expr] is consumed by Length: count elements of expr directly.
  let stripped = crate::evaluator::strip_unevaluated(&args[0]);
  // SparseArray[Automatic, dims, default, rules]: Length is its first
  // dimension, like a dense array (not the count of canonical-form parts).
  if let Expr::FunctionCall { name, args: sa } = &stripped
    && name == "SparseArray"
    && sa.len() == 4
    && let Expr::List(dims) = &sa[1]
    && let Some(Expr::Integer(d)) = dims.first()
  {
    return Ok(Expr::Integer(*d));
  }
  let len = match &stripped {
    Expr::List(items) => items.len() as i128,
    Expr::Association(items) => items.len() as i128,
    // Rational[n, d] and Complex[re, im] are atoms with length 0
    Expr::FunctionCall { name, .. }
      if name == "Rational" || name == "Complex" =>
    {
      0
    }
    // ByteArray["base64"] — length is number of bytes
    Expr::FunctionCall { name, args }
      if name == "ByteArray" && args.len() == 1 =>
    {
      if let Expr::String(b64) = &args[0] {
        use base64::Engine;
        let engine = base64::engine::general_purpose::STANDARD;
        if let Ok(decoded) = engine.decode(b64) {
          decoded.len() as i128
        } else {
          1
        }
      } else if let Expr::List(items) = &args[0] {
        items.len() as i128
      } else {
        1
      }
    }
    Expr::FunctionCall { args, .. } => args.len() as i128,
    Expr::BinaryOp { .. } | Expr::UnaryOp { .. } => {
      if let Some((_head, ha_args)) =
        crate::functions::list_helpers_ast::expr_to_head_args(&stripped)
      {
        ha_args.len() as i128
      } else {
        0
      }
    }
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
    // Complex numbers are atoms with depth 1
    if is_complex_number(expr) {
      return 1;
    }
    match expr {
      Expr::List(items) => 1 + items.iter().map(calc_depth).max().unwrap_or(0),
      Expr::FunctionCall { args, .. } => {
        1 + args.iter().map(calc_depth).max().unwrap_or(0)
      }
      // Depth counts positive-index parts only; the head (part 0) is
      // ignored, so only the arguments contribute.
      Expr::CurriedCall { args, .. } => {
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
      // CurriedCall f[a,b][x,y]: head expression contributes its own leaves; no extra 1
      Expr::CurriedCall { func, args } => {
        count_leaves(func) + args.iter().map(count_leaves).sum::<i128>()
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

/// ByteCount[expr] - gives the number of bytes used internally to store expr
pub fn byte_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ByteCount expects exactly 1 argument".into(),
    ));
  }

  fn count_bytes(expr: &Expr) -> i128 {
    match expr {
      // Atoms with data
      Expr::Integer(_) => 16, // machine integer: 16 bytes (Wolfram's representation)
      Expr::BigInteger(n) => {
        let (_, bytes) = n.to_bytes_le();
        16 + bytes.len() as i128
      }
      Expr::Real(_) => 16, // machine real: 16 bytes (Wolfram's representation)
      Expr::BigFloat(s, _) => 16 + s.len() as i128,
      // Strings: 32-byte header + 8 bytes per 8 characters of content
      Expr::String(s) => 32 + ((s.len() / 8) * 8) as i128,
      // Symbols and constants are shared, so 0 bytes
      Expr::Identifier(_) => 0,
      Expr::Constant(_) => 0,
      // Slots
      Expr::Slot(_) | Expr::SlotSequence(_) => 8,
      // Compound expressions: 40-byte base + 8 bytes per slot + recursive sizes
      Expr::List(items) => {
        40 + 8 * items.len() as i128
          + items.iter().map(count_bytes).sum::<i128>()
      }
      Expr::FunctionCall { name, args } => {
        // Rational and Complex are packed types with fixed 56-byte size
        if name == "Rational" || name == "Complex" {
          return 56;
        }
        40 + 8 * args.len() as i128 + args.iter().map(count_bytes).sum::<i128>()
      }
      Expr::BinaryOp { left, right, .. } => {
        16 + count_bytes(left) + count_bytes(right)
      }
      Expr::UnaryOp { operand, .. } => 8 + count_bytes(operand),
      Expr::Comparison { operands, .. } => {
        8 * operands.len() as i128
          + operands.iter().map(count_bytes).sum::<i128>()
      }
      Expr::CompoundExpr(items) => {
        8 * items.len() as i128 + items.iter().map(count_bytes).sum::<i128>()
      }
      Expr::Association(items) => items
        .iter()
        .map(|(k, v)| 16 + count_bytes(k) + count_bytes(v))
        .sum::<i128>(),
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => 16 + count_bytes(pattern) + count_bytes(replacement),
      Expr::ReplaceAll { expr, rules }
      | Expr::ReplaceRepeated { expr, rules } => {
        16 + count_bytes(expr) + count_bytes(rules)
      }
      Expr::Map { func, list }
      | Expr::Apply { func, list }
      | Expr::MapApply { func, list } => {
        16 + count_bytes(func) + count_bytes(list)
      }
      _ => 8, // fallback for other node types
    }
  }

  Ok(Expr::Integer(count_bytes(&args[0])))
}

/// Helper to format a real number
/// Helper to convert Expr to its FullForm string representation
pub fn expr_to_full_form(expr: &Expr) -> String {
  crate::functions::expr_form::render_full_form(expr)
}

/// FullForm[expr] - Returns a symbolic FullForm wrapper (like wolframscript).
/// The display layer (expr_to_output) renders the inner expr in FullForm notation.
pub fn full_form_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  Ok(Expr::FunctionCall {
    name: "FullForm".to_string(),
    args: vec![arg.clone()].into(),
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
    args: func_args.to_vec().into(),
  })
}

/// LeapYearQ[year] or LeapYearQ[{year}] or LeapYearQ[DateObject[...]] - Tests if a year is a leap year
pub fn leap_year_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let extract_year = |e: &Expr| -> Option<i128> {
    match e {
      Expr::Integer(n) => Some(*n),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        n.to_i128()
      }
      _ => None,
    }
  };
  let year = match &args[0] {
    // Bare integers are not accepted by Wolfram's LeapYearQ
    Expr::Integer(_) | Expr::BigInteger(_) => {
      return Ok(bool_expr(false));
    }
    // Accept list format: LeapYearQ[{year}] or LeapYearQ[{year, month, day}]
    Expr::List(items) if !items.is_empty() => match extract_year(&items[0]) {
      Some(y) => y,
      None => return Ok(bool_expr(false)),
    },
    // Accept DateObject: LeapYearQ[DateObject[{year, ...}]]
    Expr::FunctionCall { name, args: dargs }
      if name == "DateObject" && !dargs.is_empty() =>
    {
      match &dargs[0] {
        Expr::List(items) if !items.is_empty() => match extract_year(&items[0])
        {
          Some(y) => y,
          None => return Ok(bool_expr(false)),
        },
        _ => return Ok(bool_expr(false)),
      }
    }
    // Accept string date specs: "2024" (year only) or ISO "2024-03-01".
    // Wolfram parses the string as a date and tests its year, so take the
    // leading run of digits as the year.
    Expr::String(s) => {
      let digits: String = s
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
      match digits.parse::<i128>() {
        Ok(y) if !digits.is_empty() => y,
        _ => return Ok(bool_expr(false)),
      }
    }
    // Other forms return False
    _ => {
      return Ok(bool_expr(false));
    }
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
  // Use the bindings-tracking matcher so that repeated pattern variables
  // like {a_, b_, a_} correctly require both `a_` positions to bind to
  // the same value.
  let matches =
    crate::evaluator::pattern_matching::match_pattern(&args[0], &args[1])
      .is_some();
  Ok(bool_expr(matches))
}

/// SubsetQ[list1, list2] - Tests if list2 is a subset of list1
/// The display name of an expression's head, for ::heads messages.
fn head_name(e: &Expr) -> String {
  crate::evaluator::evaluate_function_call_ast("Head", std::slice::from_ref(e))
    .map(|h| crate::syntax::format_expr(&h, crate::syntax::ExprForm::Output))
    .unwrap_or_else(|_| "Symbol".to_string())
}

/// Element slices for the two subjects of SubsetQ/DisjointQ/
/// IntersectingQ. Differing heads emit `<F>::heads: Heads <H1> and
/// <H2> at positions 1 and 2 are expected to be the same.` and return
/// the unevaluated call; matching nonatomic heads (List or any common
/// head) yield their elements.
#[allow(clippy::type_complexity)]
fn same_head_elements<'a>(
  fname: &str,
  args: &'a [Expr],
) -> Result<Option<(&'a [Expr], &'a [Expr])>, Expr> {
  let elements = |e: &'a Expr| -> Option<(&'a str, &'a [Expr])> {
    match e {
      Expr::List(items) => Some(("List", items.as_slice())),
      Expr::FunctionCall { name, args } => {
        Some((name.as_str(), args.as_slice()))
      }
      _ => None,
    }
  };
  match (elements(&args[0]), elements(&args[1])) {
    (Some((h1, a)), Some((h2, b))) if h1 == h2 => Ok(Some((a, b))),
    _ => {
      let h1 = head_name(&args[0]);
      let h2 = head_name(&args[1]);
      if h1 != h2 {
        crate::emit_message(&format!(
          "{}::heads: Heads {} and {} at positions 1 and 2 are expected to be the same.",
          fname, h1, h2
        ));
        return Err(Expr::FunctionCall {
          name: fname.to_string(),
          args: args.to_vec().into(),
        });
      }
      Ok(None)
    }
  }
}

pub fn subset_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "SubsetQ expects 2 or 3 arguments".into(),
    ));
  }
  // Optional `SameTest -> f` (or `{SameTest -> f}`) as the third argument.
  let same_test: Option<&Expr> = if args.len() == 3 {
    match extract_same_test(&args[2]) {
      Some(f) => Some(f),
      None => {
        return Ok(Expr::FunctionCall {
          name: "SubsetQ".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    None
  };
  let (superset, subset) = match same_head_elements("SubsetQ", &args[..2]) {
    Ok(Some(pair)) => pair,
    Ok(None) => {
      return Ok(Expr::FunctionCall {
        name: "SubsetQ".to_string(),
        args: args.to_vec().into(),
      });
    }
    Err(unevaluated) => return Ok(unevaluated),
  };
  if let Some(test) = same_test {
    // Every subset element y must be equivalent to some superset element x,
    // i.e. test[x, y] evaluates to True.
    for y in subset {
      let found = superset.iter().any(|x| {
        matches!(
          crate::functions::list_helpers_ast::apply_func_to_two_args(test, x, y),
          Ok(Expr::Identifier(ref s)) if s == "True"
        )
      });
      if !found {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
    return Ok(Expr::Identifier("True".to_string()));
  }
  // Default: structural membership.
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

/// Extract the function `f` from a `SameTest -> f` rule, or from a singleton
/// list `{SameTest -> f}`. Returns None for anything else.
fn extract_same_test(opt: &Expr) -> Option<&Expr> {
  let rule = match opt {
    Expr::List(items) if items.len() == 1 => &items[0],
    other => other,
  };
  match rule {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "SameTest") => {
      Some(replacement)
    }
    _ => None,
  }
}

/// DisjointQ[a, b] - True if the subjects share no common elements.
pub fn disjoint_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  intersecting_or_disjoint(args, "DisjointQ", false)
}

/// IntersectingQ[a, b] - True if the subjects share a common element.
pub fn intersecting_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  intersecting_or_disjoint(args, "IntersectingQ", true)
}

fn intersecting_or_disjoint(
  args: &[Expr],
  fname: &str,
  want_common: bool,
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects 2 or 3 arguments",
      fname
    )));
  }
  // Optional `SameTest -> f` (or `{SameTest -> f}`) as the third argument.
  let same_test: Option<&Expr> = if args.len() == 3 {
    match extract_same_test(&args[2]) {
      Some(f) => Some(f),
      None => {
        return Ok(Expr::FunctionCall {
          name: fname.to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    None
  };
  let (a, b) = match same_head_elements(fname, &args[..2]) {
    Ok(Some(pair)) => pair,
    Ok(None) => {
      return Ok(Expr::FunctionCall {
        name: fname.to_string(),
        args: args.to_vec().into(),
      });
    }
    Err(unevaluated) => return Ok(unevaluated),
  };
  let has_common = if let Some(test) = same_test {
    // A common element exists iff some pair satisfies test[x, y] (x from the
    // first subject, y from the second).
    a.iter().any(|x| {
      b.iter().any(|y| {
        matches!(
          crate::functions::list_helpers_ast::apply_func_to_two_args(test, x, y),
          Ok(Expr::Identifier(ref s)) if s == "True"
        )
      })
    })
  } else {
    let a_strs: Vec<String> =
      a.iter().map(crate::syntax::expr_to_string).collect();
    b.iter()
      .any(|e| a_strs.contains(&crate::syntax::expr_to_string(e)))
  };
  Ok(Expr::Identifier(
    if has_common == want_common {
      "True"
    } else {
      "False"
    }
    .to_string(),
  ))
}

/// PossibleZeroQ[expr] - Tests if expr is possibly zero
/// Uses symbolic simplification and numeric evaluation to determine
/// whether an expression could be zero.
pub fn possible_zero_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "PossibleZeroQ".to_string(),
      args: args.to_vec().into(),
    });
  }
  let expr = &args[0];

  // 1. Structural zero check
  if is_structural_zero(expr) {
    return Ok(bool_expr(true));
  }

  // 2. Non-numeric atoms that can never be zero
  match expr {
    Expr::String(_) => return Ok(bool_expr(false)),
    Expr::Identifier(name) => match name.as_str() {
      "True" | "False" | "Infinity" | "ComplexInfinity" | "Indeterminate" => {
        return Ok(bool_expr(false));
      }
      _ => {}
    },
    _ => {}
  }

  // 3. Known nonzero constants
  if let Expr::Constant(c) = expr {
    match c.as_str() {
      "Pi" | "E" | "Degree" | "EulerGamma" | "Catalan" | "GoldenRatio"
      | "Glaisher" | "Khinchin" => {
        return Ok(bool_expr(false));
      }
      _ => {}
    }
  }

  // 4. Known positive/negative numbers are not zero
  if let Some(true) = is_known_positive(expr) {
    return Ok(bool_expr(false));
  }
  if let Some(true) = is_known_negative(expr) {
    return Ok(bool_expr(false));
  }

  // 5. Simplify the expression and check
  let simplified = crate::functions::polynomial_ast::simplify_expr(expr);
  if is_structural_zero(&simplified) {
    return Ok(bool_expr(true));
  }

  // 6. Try numeric evaluation on the simplified expression
  if let Some(val) = crate::functions::math_ast::try_eval_to_f64(&simplified) {
    return Ok(bool_expr(val.abs() < 1e-10));
  }

  // 7. Try numeric evaluation on the original expression
  if let Some(val) = crate::functions::math_ast::try_eval_to_f64(expr) {
    return Ok(bool_expr(val.abs() < 1e-10));
  }

  // 8. For complex-valued expressions, evaluate via N[] and check magnitude.
  let n_call = Expr::FunctionCall {
    name: "N".to_string(),
    args: vec![expr.clone()].into(),
  };
  if let Ok(num) = crate::evaluator::evaluate_expr_to_expr(&n_call)
    && let Some((re, im)) =
      crate::functions::math_ast::try_extract_complex_float(&num)
    && (re.abs() + im.abs()) < 1e-10
  {
    return Ok(bool_expr(true));
  }

  // 9. For expressions we can't evaluate numerically, return False
  // (matches Wolfram behavior for symbolic unknowns like x)
  Ok(bool_expr(false))
}

/// Check if an expression is structurally zero (literal 0, 0.0, 0/1, Complex[0,0], etc.)
fn is_structural_zero(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(0) => true,
    Expr::Real(f) => *f == 0.0,
    Expr::BigInteger(n) => {
      use num_traits::Zero;
      n.is_zero()
    }
    Expr::BigFloat(digits, _) => digits.parse::<f64>().is_ok_and(|f| f == 0.0),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      is_structural_zero(&args[0])
    }
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      is_structural_zero(&args[0]) && is_structural_zero(&args[1])
    }
    _ => false,
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
