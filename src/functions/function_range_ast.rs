//! FunctionRange[f, x, y] for a recognized family of real functions,
//! returning wolframscript's condition forms verbatim:
//! polynomials (linear -> True, quadratics -> vertex bound, pure even
//! powers -> y >= 0, odd -> True), Sin/Cos -> -1 <= y <= 1, Tan/Log ->
//! True, E^x -> y > 0, Sqrt/Abs -> y >= 0, Cosh -> 1 <= y,
//! Tanh -> -1 < y < 1, and 1/(1 + x^2) -> Inequality[0, Less, y,
//! LessEqual, 1].

use crate::InterpreterError;
use crate::syntax::{BinaryOperator, ComparisonOp, Expr, unevaluated};

pub fn function_range_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("FunctionRange", args);
  if args.len() != 3 {
    return Ok(unevaluated(args));
  }
  let x_var = match &args[1] {
    Expr::Identifier(v) => v.clone(),
    _ => return Ok(unevaluated(args)),
  };
  let y = match &args[2] {
    y @ Expr::Identifier(_) => y.clone(),
    _ => return Ok(unevaluated(args)),
  };
  let f = &args[0];

  let cmp =
    |operands: Vec<Expr>, operators: Vec<ComparisonOp>| Expr::Comparison {
      operands,
      operators,
    };
  let true_expr = || Expr::Identifier("True".to_string());
  let y_ge =
    |v: Expr| cmp(vec![y.clone(), v], vec![ComparisonOp::GreaterEqual]);
  let y_le = |v: Expr| cmp(vec![y.clone(), v], vec![ComparisonOp::LessEqual]);

  // Unary special functions of the bare variable
  if let Expr::FunctionCall { name, args: fargs } = f
    && fargs.len() == 1
    && matches!(&fargs[0], Expr::Identifier(v) if *v == x_var)
  {
    return Ok(match name.as_str() {
      // -1 <= y <= 1
      "Sin" | "Cos" => cmp(
        vec![Expr::Integer(-1), y.clone(), Expr::Integer(1)],
        vec![ComparisonOp::LessEqual, ComparisonOp::LessEqual],
      ),
      "Tan" | "Log" | "Sinh" | "ArcSinh" => true_expr(),
      "Sqrt" | "Abs" => y_ge(Expr::Integer(0)),
      "Exp" => cmp(
        vec![y.clone(), Expr::Integer(0)],
        vec![ComparisonOp::Greater],
      ),
      // 1 <= y (wolframscript puts the constant first here)
      "Cosh" => cmp(
        vec![Expr::Integer(1), y.clone()],
        vec![ComparisonOp::LessEqual],
      ),
      // -1 < y < 1
      "Tanh" => cmp(
        vec![Expr::Integer(-1), y.clone(), Expr::Integer(1)],
        vec![ComparisonOp::Less, ComparisonOp::Less],
      ),
      _ => unevaluated(args),
    });
  }

  // E^x (Power form)
  let is_x = |e: &Expr| matches!(e, Expr::Identifier(v) if *v == x_var);
  if let Some((base, exp)) = as_power(f) {
    if matches!(&base, Expr::Identifier(n) | Expr::Constant(n) if n == "E")
      && is_x(&exp)
    {
      return Ok(cmp(
        vec![y.clone(), Expr::Integer(0)],
        vec![ComparisonOp::Greater],
      ));
    }
    // Sqrt[x] evaluates to Power[x, 1/2] before dispatch
    if is_x(&base)
      && matches!(&exp, Expr::FunctionCall { name, args: ra }
        if name == "Rational"
          && ra.len() == 2
          && matches!(&ra[0], Expr::Integer(1))
          && matches!(&ra[1], Expr::Integer(2)))
    {
      return Ok(y_ge(Expr::Integer(0)));
    }
    // 1/(1 + x^2): Inequality[0, Less, y, LessEqual, 1]
    if matches!(&exp, Expr::Integer(-1))
      && let Expr::FunctionCall { name, args: pargs } = &base
      && name == "Plus"
      && pargs.len() == 2
      && matches!(&pargs[0], Expr::Integer(1))
      && matches!(as_power(&pargs[1]), Some((b, Expr::Integer(2))) if is_x(&b))
    {
      return Ok(Expr::FunctionCall {
        name: "Inequality".to_string(),
        args: vec![
          Expr::Integer(0),
          Expr::Identifier("Less".to_string()),
          y.clone(),
          Expr::Identifier("LessEqual".to_string()),
          Expr::Integer(1),
        ]
        .into(),
      });
    }
  }

  // Polynomials with rational coefficients
  let coeffs = match poly_coeffs(f, &x_var) {
    Some(c) => c,
    None => return Ok(unevaluated(args)),
  };
  let degree = coeffs.len().saturating_sub(1);
  match degree {
    // Non-constant linear functions cover all reals
    1 => Ok(true_expr()),
    // Quadratic: bounded by the vertex value c - b^2/(4a)
    2 => {
      let (c, b, a) = (coeffs[0], coeffs[1], coeffs[2]);
      // v = c - b^2/(4a) computed exactly
      let num = c.0 * 4 * a.0 * b.1 * b.1 - b.0 * b.0 * c.1 * a.1;
      let den = c.1 * 4 * a.0 * b.1 * b.1;
      let v = simplify_frac(num, den);
      let v_expr = if v.1 == 1 {
        Expr::Integer(v.0)
      } else {
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(v.0), Expr::Integer(v.1)].into(),
        }
      };
      if a.0 > 0 {
        Ok(y_ge(v_expr))
      } else {
        Ok(y_le(v_expr))
      }
    }
    // Pure powers x^k: even k is bounded below by 0, odd covers the reals
    k if k >= 3
      && coeffs[..k].iter().all(|&(n, _)| n == 0)
      && coeffs[k] == (1, 1) =>
    {
      if k % 2 == 0 {
        Ok(y_ge(Expr::Integer(0)))
      } else {
        Ok(true_expr())
      }
    }
    _ => Ok(unevaluated(args)),
  }
}

fn as_power(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      Some((args[0].clone(), args[1].clone()))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => Some(((**left).clone(), (**right).clone())),
    _ => None,
  }
}

fn simplify_frac(n: i128, d: i128) -> (i128, i128) {
  fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
      let t = b;
      b = a % b;
      a = t;
    }
    a.max(1)
  }
  let g = gcd(n, d);
  let (mut n, mut d) = (n / g, d / g);
  if d < 0 {
    n = -n;
    d = -d;
  }
  (n, d)
}

/// Ascending rational coefficients of a polynomial in `var`; None for
/// non-polynomial input.
fn poly_coeffs(expr: &Expr, var: &str) -> Option<Vec<(i128, i128)>> {
  let coeff_list =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "CoefficientList".to_string(),
      args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
    })
    .ok()?;
  let items = match &coeff_list {
    Expr::List(items) if items.len() >= 2 => items,
    _ => return None,
  };
  let mut out = Vec::with_capacity(items.len());
  for item in items {
    match item {
      Expr::Integer(n) => out.push((*n, 1)),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          out.push((*n, *d));
        } else {
          return None;
        }
      }
      _ => return None,
    }
  }
  // Leading coefficient must be nonzero for the degree logic
  while out.len() > 1 && out.last() == Some(&(0, 1)) {
    out.pop();
  }
  Some(out)
}
