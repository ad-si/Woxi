//! Machine complex arithmetic for compiled numeric expressions.
//!
//! [`super::solve::compile_numeric`] compiles an expression of real
//! variables to float arithmetic. Where a real-valued wrapper (`Re`, `Im`,
//! `Abs`, `Arg`) holds a genuinely complex argument — the characteristic
//! function `Re[(-I t)^d E^(-I t x - Abs[t]^α (1 - I β …))]` integrated by
//! `NIntegrate`, say — the argument is compiled here instead, so each point
//! costs a few complex operations rather than a symbolic evaluation.

use super::solve::{NumNode, compile_numeric};
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

/// A complex machine number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct C {
  pub re: f64,
  pub im: f64,
}

impl C {
  const fn new(re: f64, im: f64) -> Self {
    C { re, im }
  }
  fn add(self, o: C) -> C {
    C::new(self.re + o.re, self.im + o.im)
  }
  fn mul(self, o: C) -> C {
    if self.im == 0.0 && o.im == 0.0 {
      return C::new(self.re * o.re, 0.0);
    }
    C::new(
      self.re * o.re - self.im * o.im,
      self.re * o.im + self.im * o.re,
    )
  }
  fn neg(self) -> C {
    C::new(-self.re, -self.im)
  }
  fn div(self, o: C) -> C {
    if o.im == 0.0 {
      return C::new(self.re / o.re, self.im / o.re);
    }
    // Smith's algorithm keeps the intermediate products in range.
    if o.re.abs() >= o.im.abs() {
      let r = o.im / o.re;
      let d = o.re + o.im * r;
      C::new((self.re + self.im * r) / d, (self.im - self.re * r) / d)
    } else {
      let r = o.re / o.im;
      let d = o.re * r + o.im;
      C::new((self.re * r + self.im) / d, (self.im * r - self.re) / d)
    }
  }
  fn abs(self) -> f64 {
    self.re.hypot(self.im)
  }
  fn arg(self) -> f64 {
    self.im.atan2(self.re)
  }
  fn exp(self) -> C {
    let m = self.re.exp();
    if self.im == 0.0 {
      return C::new(m, 0.0);
    }
    C::new(m * self.im.cos(), m * self.im.sin())
  }
  fn ln(self) -> C {
    C::new(self.abs().ln(), self.arg())
  }
  fn sqrt(self) -> C {
    if self.im == 0.0 && self.re >= 0.0 {
      return C::new(self.re.sqrt(), 0.0);
    }
    let m = self.abs();
    let re = f64::midpoint(m, self.re).sqrt();
    let im = ((m - self.re) / 2.0).sqrt();
    C::new(re, if self.im < 0.0 { -im } else { im })
  }
  fn powi(self, n: i64) -> C {
    let mut base = if n < 0 {
      C::new(1.0, 0.0).div(self)
    } else {
      self
    };
    let mut k = n.unsigned_abs();
    let mut acc = C::new(1.0, 0.0);
    while k > 0 {
      if k & 1 == 1 {
        acc = acc.mul(base);
      }
      base = base.mul(base);
      k >>= 1;
    }
    acc
  }
  fn pow(self, e: C) -> C {
    if e.im == 0.0 && e.re.fract() == 0.0 && e.re.abs() < 1e9 {
      return self.powi(e.re as i64);
    }
    if self.re == 0.0 && self.im == 0.0 {
      return if e.re > 0.0 {
        C::new(0.0, 0.0)
      } else {
        C::new(f64::NAN, f64::NAN)
      };
    }
    if self.im == 0.0 && self.re > 0.0 && e.im == 0.0 {
      return C::new(self.re.powf(e.re), 0.0);
    }
    e.mul(self.ln()).exp()
  }
  fn sin(self) -> C {
    C::new(
      self.re.sin() * self.im.cosh(),
      self.re.cos() * self.im.sinh(),
    )
  }
  fn cos(self) -> C {
    C::new(
      self.re.cos() * self.im.cosh(),
      -self.re.sin() * self.im.sinh(),
    )
  }
  fn sinh(self) -> C {
    C::new(
      self.re.sinh() * self.im.cos(),
      self.re.cosh() * self.im.sin(),
    )
  }
  fn cosh(self) -> C {
    C::new(
      self.re.cosh() * self.im.cos(),
      self.re.sinh() * self.im.sin(),
    )
  }
  fn is_finite(self) -> bool {
    self.re.is_finite() && self.im.is_finite()
  }
}

/// An expression compiled to complex arithmetic over real variables.
pub(crate) enum CNode {
  Const(C),
  /// A real subexpression, compiled by `compile_numeric`.
  Real(NumNode),
  Add(Vec<CNode>),
  Mul(Vec<CNode>),
  Neg(Box<CNode>),
  Div(Box<CNode>, Box<CNode>),
  Pow(Box<CNode>, Box<CNode>),
  Func(fn(C) -> C, Box<CNode>),
}

impl CNode {
  pub(crate) fn eval(&self, point: &[f64]) -> C {
    match self {
      CNode::Const(c) => *c,
      CNode::Real(n) => C::new(n.eval(point), 0.0),
      CNode::Add(xs) => xs
        .iter()
        .fold(C::new(0.0, 0.0), |acc, x| acc.add(x.eval(point))),
      CNode::Mul(xs) => xs
        .iter()
        .fold(C::new(1.0, 0.0), |acc, x| acc.mul(x.eval(point))),
      CNode::Neg(a) => a.eval(point).neg(),
      CNode::Div(a, b) => a.eval(point).div(b.eval(point)),
      CNode::Pow(a, b) => a.eval(point).pow(b.eval(point)),
      CNode::Func(f, a) => f(a.eval(point)),
    }
  }

  pub(crate) fn has_fallback(&self) -> bool {
    match self {
      CNode::Const(_) => false,
      CNode::Real(n) => n.has_fallback(),
      CNode::Add(xs) | CNode::Mul(xs) => xs.iter().any(CNode::has_fallback),
      CNode::Neg(a) | CNode::Func(_, a) => a.has_fallback(),
      CNode::Div(a, b) | CNode::Pow(a, b) => {
        a.has_fallback() || b.has_fallback()
      }
    }
  }
}

/// The real-valued part of a complex value a wrapper takes.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Part {
  Re,
  Im,
  Abs,
  Arg,
}

impl Part {
  pub(crate) fn of(self, z: C) -> f64 {
    if !z.is_finite() {
      return f64::NAN;
    }
    match self {
      Part::Re => z.re,
      Part::Im => z.im,
      Part::Abs => z.abs(),
      Part::Arg => z.arg(),
    }
  }
}

/// Whether `expr` mentions the imaginary unit.
fn mentions_i(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(n) => n == "I",
    Expr::Constant(n) => n == "I",
    Expr::FunctionCall { name, args } => {
      name == "Complex" || args.iter().any(mentions_i)
    }
    Expr::BinaryOp { left, right, .. } => mentions_i(left) || mentions_i(right),
    Expr::UnaryOp { operand, .. } => mentions_i(operand),
    Expr::List(items) => items.iter().any(mentions_i),
    _ => false,
  }
}

/// `expr` compiled to complex arithmetic in `vars` (bound to real values),
/// or `None` if some piece is neither a known function nor real.
pub(crate) fn compile_complex(expr: &Expr, vars: &[String]) -> Option<CNode> {
  if !mentions_i(expr) {
    // Elementary functions of a real argument can still leave the reals
    // (`Sqrt[x]`, `Log[x]` for x < 0), so they are compiled complex below;
    // anything else real goes to the real compiler.
    let elementary = matches!(expr, Expr::FunctionCall { name, .. }
      if matches!(name.as_str(), "Sqrt" | "Log" | "Power"))
      || matches!(
        expr,
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          ..
        }
      );
    if !elementary {
      return compile_numeric(expr, vars).map(CNode::Real);
    }
  }
  let c = |e: &Expr| compile_complex(e, vars);
  let func = |f: fn(C) -> C, args: &[Expr]| -> Option<CNode> {
    match args {
      [a] => Some(CNode::Func(f, Box::new(c(a)?))),
      _ => None,
    }
  };
  match expr {
    Expr::Integer(n) => Some(CNode::Const(C::new(*n as f64, 0.0))),
    Expr::Real(r) => Some(CNode::Const(C::new(*r, 0.0))),
    Expr::Identifier(n) | Expr::Constant(n) if n == "I" => {
      Some(CNode::Const(C::new(0.0, 1.0)))
    }
    Expr::Identifier(_) | Expr::Constant(_) | Expr::BigFloat(..) => {
      compile_numeric(expr, vars).map(CNode::Real)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(CNode::Neg(Box::new(c(operand)?))),
    Expr::BinaryOp { op, left, right } => {
      let (l, r) = (c(left)?, c(right)?);
      Some(match op {
        BinaryOperator::Plus => CNode::Add(vec![l, r]),
        BinaryOperator::Minus => CNode::Add(vec![l, CNode::Neg(Box::new(r))]),
        BinaryOperator::Times => CNode::Mul(vec![l, r]),
        BinaryOperator::Divide => CNode::Div(Box::new(l), Box::new(r)),
        BinaryOperator::Power => CNode::Pow(Box::new(l), Box::new(r)),
        _ => return None,
      })
    }
    Expr::FunctionCall { name, args } => match (name.as_str(), &args[..]) {
      ("Complex", [re, im]) => Some(CNode::Const(C::new(
        crate::functions::math_ast::try_eval_to_f64(re)?,
        crate::functions::math_ast::try_eval_to_f64(im)?,
      ))),
      ("Plus", _) => {
        Some(CNode::Add(args.iter().map(c).collect::<Option<_>>()?))
      }
      ("Times", _) => {
        Some(CNode::Mul(args.iter().map(c).collect::<Option<_>>()?))
      }
      ("Power", [a, b]) => Some(CNode::Pow(Box::new(c(a)?), Box::new(c(b)?))),
      ("Rational" | "Divide", [a, b]) => {
        Some(CNode::Div(Box::new(c(a)?), Box::new(c(b)?)))
      }
      ("Subtract", [a, b]) => {
        Some(CNode::Add(vec![c(a)?, CNode::Neg(Box::new(c(b)?))]))
      }
      ("Minus", [a]) => Some(CNode::Neg(Box::new(c(a)?))),
      ("Exp", _) => func(C::exp, args),
      ("Log", [_]) => func(C::ln, args),
      ("Sqrt", _) => func(C::sqrt, args),
      ("Sin", _) => func(C::sin, args),
      ("Cos", _) => func(C::cos, args),
      ("Tan", _) => func(|z| z.sin().div(z.cos()), args),
      ("Sinh", _) => func(C::sinh, args),
      ("Cosh", _) => func(C::cosh, args),
      ("Tanh", _) => func(|z| z.sinh().div(z.cosh()), args),
      ("Conjugate", _) => func(|z| C::new(z.re, -z.im), args),
      ("Re", _) => func(|z| C::new(z.re, 0.0), args),
      ("Im", _) => func(|z| C::new(z.im, 0.0), args),
      ("Abs", _) => func(|z| C::new(z.abs(), 0.0), args),
      ("Arg", _) => func(|z| C::new(z.arg(), 0.0), args),
      ("Sign", _) => func(
        |z| {
          let m = z.abs();
          if m == 0.0 {
            C::new(0.0, 0.0)
          } else {
            C::new(z.re / m, z.im / m)
          }
        },
        args,
      ),
      _ => None,
    },
    _ => None,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn eval_at(src: &str, x: f64) -> C {
    let e = crate::syntax::string_to_expr(src).unwrap();
    compile_complex(&e, &["x".to_string()]).unwrap().eval(&[x])
  }

  #[test]
  fn complex_arithmetic_matches_closed_forms() {
    let z = eval_at("E^(I x)", std::f64::consts::FRAC_PI_2);
    assert!(z.re.abs() < 1e-15 && (z.im - 1.0).abs() < 1e-15);
    let z = eval_at("(-I x)^2", 3.0);
    assert_eq!(z, C::new(-9.0, 0.0));
    let z = eval_at("(-I x)^-1", 2.0);
    assert_eq!(z, C::new(0.0, 0.5));
    let z = eval_at("Sqrt[x]", -4.0);
    assert_eq!(z, C::new(0.0, 2.0));
    let z = eval_at("Abs[x]^(3/2) (1 - I Sign[x])", -1.0);
    assert_eq!(z, C::new(1.0, 1.0));
  }
}
