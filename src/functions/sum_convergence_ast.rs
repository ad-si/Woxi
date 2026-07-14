//! SumConvergence[f, n] for a recognized family of terms, matching
//! wolframscript's verdicts:
//! - rational functions P(n)/Q(n): True iff deg Q - deg P >= 2
//! - alternating (-1)^n * P/Q: True iff deg Q - deg P >= 1
//! - geometric r^n (numeric r): True iff |r| < 1, polynomial factors
//!   never change the verdict for |r| != 1
//! - bare x^n (symbolic): Abs[x] < 1; bare 1/n^p (symbolic): Re[p] > 1
//! - 1/n! decay: True

use crate::InterpreterError;
use crate::syntax::{BinaryOperator, ComparisonOp, Expr, UnaryOperator};

pub fn sum_convergence_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "SumConvergence".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let n_var = match &args[1] {
    Expr::Identifier(v) => v.clone(),
    _ => return Ok(unevaluated(args)),
  };
  let f = &args[0];

  // Bare symbolic-base power x^n -> Abs[x] < 1
  if let Some((base, exp)) = as_power(f)
    && matches!(&exp, Expr::Identifier(v) if *v == n_var)
    && matches!(&base, Expr::Identifier(b) if *b != n_var)
  {
    return Ok(Expr::Comparison {
      operands: vec![
        Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![base.clone()].into(),
        },
        Expr::Integer(1),
      ],
      operators: vec![ComparisonOp::Less],
    });
  }
  // Bare symbolic p-series 1/n^p (Power[n, -p]) -> Re[p] > 1
  if let Some((base, exp)) = as_power(f)
    && matches!(&base, Expr::Identifier(v) if *v == n_var)
    && let Some(p) = negated_symbol(&exp, &n_var)
  {
    return Ok(Expr::Comparison {
      operands: vec![
        Expr::FunctionCall {
          name: "Re".to_string(),
          args: vec![p].into(),
        },
        Expr::Integer(1),
      ],
      operators: vec![ComparisonOp::Greater],
    });
  }

  // Factor classification
  let mut net_degree: i128 = 0; // polynomial degree of the term
  let mut geo_ratio: Option<(f64, bool)> = None; // (|r|, alternating-sign)
  let mut inv_factorial = false;
  let mut ok = true;
  classify_factors(
    f,
    &n_var,
    1,
    &mut net_degree,
    &mut geo_ratio,
    &mut inv_factorial,
    &mut ok,
  );
  if !ok {
    return Ok(unevaluated(args));
  }

  // 1/n! decay dominates any polynomial factor
  if inv_factorial {
    return Ok(Expr::Identifier("True".to_string()));
  }

  let verdict = |converges: bool| {
    Expr::Identifier(if converges { "True" } else { "False" }.to_string())
  };
  match geo_ratio {
    Some((r_abs, alternating)) => {
      if r_abs < 1.0 {
        Ok(verdict(true))
      } else if r_abs > 1.0 {
        Ok(verdict(false))
      } else if alternating {
        // |r| == 1 with sign alternation: alternating series test
        Ok(verdict(net_degree <= -1))
      } else {
        Ok(verdict(net_degree <= -2))
      }
    }
    None => Ok(verdict(net_degree <= -2)),
  }
}

/// Decompose into polynomial-degree, geometric-ratio, and factorial
/// parts. `power_sign` carries +1/-1 through Divide denominators.
#[allow(clippy::too_many_arguments)]
fn classify_factors(
  expr: &Expr,
  n_var: &str,
  power_sign: i128,
  net_degree: &mut i128,
  geo_ratio: &mut Option<(f64, bool)>,
  inv_factorial: &mut bool,
  ok: &mut bool,
) {
  if !*ok {
    return;
  }
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args {
        classify_factors(
          a,
          n_var,
          power_sign,
          net_degree,
          geo_ratio,
          inv_factorial,
          ok,
        );
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      classify_factors(
        left,
        n_var,
        power_sign,
        net_degree,
        geo_ratio,
        inv_factorial,
        ok,
      );
      classify_factors(
        right,
        n_var,
        power_sign,
        net_degree,
        geo_ratio,
        inv_factorial,
        ok,
      );
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      classify_factors(
        left,
        n_var,
        power_sign,
        net_degree,
        geo_ratio,
        inv_factorial,
        ok,
      );
      classify_factors(
        right,
        n_var,
        -power_sign,
        net_degree,
        geo_ratio,
        inv_factorial,
        ok,
      );
    }
    // Numeric constants are irrelevant (zero would be trivially
    // convergent, but wolframscript treats that separately; bail)
    Expr::Integer(0) => *ok = false,
    Expr::Integer(_) | Expr::Real(_) => {}
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 => {}
    _ => {
      // n-polynomial factor (n, n^2+1, ...)
      if let Some(deg) = poly_degree(expr, n_var) {
        *net_degree += power_sign * deg;
        return;
      }
      if let Some((base, exp)) = as_power(expr) {
        // poly^k
        if let Expr::Integer(k) = exp
          && let Some(deg) = poly_degree(&base, n_var)
        {
          *net_degree += power_sign * k * deg;
          return;
        }
        // r^n / r^(-n) with numeric r
        if let Some(direction) = n_linear_sign(&exp, n_var) {
          if let Some(r) = crate::functions::math_ast::expr_to_num(&base) {
            let r_abs = r.abs();
            let effective = if power_sign * direction > 0 {
              r_abs
            } else if r_abs != 0.0 {
              1.0 / r_abs
            } else {
              *ok = false;
              return;
            };
            let alternating = r < 0.0;
            let entry = geo_ratio.get_or_insert((1.0, false));
            entry.0 *= effective;
            entry.1 ^= alternating;
            return;
          }
          *ok = false;
          return;
        }
        // (n!)^(-1)
        if matches!(&base, Expr::FunctionCall { name, args }
          if name == "Factorial"
            && args.len() == 1
            && matches!(&args[0], Expr::Identifier(v) if v == n_var))
          && matches!(&exp, Expr::Integer(k) if power_sign * k < 0)
        {
          *inv_factorial = true;
          return;
        }
        *ok = false;
        return;
      }
      // n! in a denominator
      if matches!(expr, Expr::FunctionCall { name, args }
        if name == "Factorial"
          && args.len() == 1
          && matches!(&args[0], Expr::Identifier(v) if v == n_var))
        && power_sign < 0
      {
        *inv_factorial = true;
        return;
      }
      *ok = false;
    }
  }
}

/// Degree of a polynomial in `var` (None for non-polynomials).
fn poly_degree(expr: &Expr, var: &str) -> Option<i128> {
  if matches!(expr, Expr::Identifier(v) if v == var) {
    return Some(1);
  }
  if crate::functions::calculus_ast::is_constant_wrt(expr, var) {
    return None; // constants handled by the caller
  }
  let coeff_list =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "CoefficientList".to_string(),
      args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
    })
    .ok()?;
  match &coeff_list {
    Expr::List(items) if items.len() >= 2 => {
      let numeric = |e: &Expr| {
        matches!(e, Expr::Integer(_) | Expr::Real(_))
          || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
      };
      if items.iter().all(numeric) {
        Some((items.len() - 1) as i128)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Match exponent n / -n, returning +1 / -1.
fn n_linear_sign(exp: &Expr, n_var: &str) -> Option<i128> {
  match exp {
    Expr::Identifier(v) if v == n_var => Some(1),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } if matches!(operand.as_ref(), Expr::Identifier(v) if v == n_var) => {
      Some(-1)
    }
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(-1))
        && matches!(&args[1], Expr::Identifier(v) if v == n_var) =>
    {
      Some(-1)
    }
    _ => None,
  }
}

/// Match exponent == -p for a symbol p (Times[-1, p] / UnaryMinus(p)).
fn negated_symbol(exp: &Expr, n_var: &str) -> Option<Expr> {
  match exp {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => match operand.as_ref() {
      p @ Expr::Identifier(v) if v != n_var => Some(p.clone()),
      _ => None,
    },
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(-1)) =>
    {
      match &args[1] {
        p @ Expr::Identifier(v) if *v != n_var => Some(p.clone()),
        _ => None,
      }
    }
    _ => None,
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
