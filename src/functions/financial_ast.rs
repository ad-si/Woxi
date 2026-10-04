//! `FinancialDerivative` for European options (Black-Scholes-Merton).
//!
//! The result is built as a symbolic expression in the contract parameters and
//! handed back to the evaluator, so exact, symbolic and machine-precision
//! inputs all behave like any other closed-form expression.

use crate::InterpreterError;
use crate::syntax::{Expr, string_to_expr, substitute_variables};

/// Look up a (string-keyed) option in a list of rules.
fn lookup(rules: &Expr, key: &str) -> Option<Expr> {
  let Expr::List(items) = rules else {
    return None;
  };
  items.iter().find_map(|item| {
    let (lhs, rhs) = match item {
      Expr::Rule {
        pattern,
        replacement,
      } => ((**pattern).clone(), (**replacement).clone()),
      Expr::FunctionCall { name, args }
        if name == "Rule" && args.len() == 2 =>
      {
        (args[0].clone(), args[1].clone())
      }
      _ => return None,
    };
    matches!(&lhs, Expr::String(s) if s == key).then_some(rhs)
  })
}

/// Closed-form Black-Scholes-Merton expression for `greek` in terms of the
/// placeholder symbols S (spot), K (strike), T (expiry), r, v, q.
fn greek_template(call: bool, greek: &str) -> Option<&'static str> {
  // Standard normal CDF / PDF
  macro_rules! cdf {
    ($x:expr) => {
      concat!("((1 + Erf[", $x, "/Sqrt[2]])/2)")
    };
  }
  macro_rules! pdf {
    ($x:expr) => {
      concat!("(Exp[-(", $x, ")^2/2]/Sqrt[2 Pi])")
    };
  }
  macro_rules! d1 {
    () => {
      "((Log[S/K] + (r - q + v^2/2) T)/(v Sqrt[T]))"
    };
  }
  macro_rules! d2 {
    () => {
      "((Log[S/K] + (r - q - v^2/2) T)/(v Sqrt[T]))"
    };
  }
  Some(match (call, greek) {
    (true, "Value") => {
      concat!("S Exp[-q T] ", cdf!(d1!()), " - K Exp[-r T] ", cdf!(d2!()))
    }
    (false, "Value") => concat!(
      "K Exp[-r T] ",
      cdf!(concat!("-", d2!())),
      " - S Exp[-q T] ",
      cdf!(concat!("-", d1!()))
    ),
    (true, "Delta") => concat!("Exp[-q T] ", cdf!(d1!())),
    (false, "Delta") => concat!("-Exp[-q T] ", cdf!(concat!("-", d1!()))),
    (_, "Gamma") => concat!("Exp[-q T] ", pdf!(d1!()), "/(S v Sqrt[T])"),
    (_, "Vega") => concat!("S Exp[-q T] ", pdf!(d1!()), " Sqrt[T]"),
    (true, "Theta") => concat!(
      "-S Exp[-q T] ",
      pdf!(d1!()),
      " v/(2 Sqrt[T]) - r K Exp[-r T] ",
      cdf!(d2!()),
      " + q S Exp[-q T] ",
      cdf!(d1!())
    ),
    (false, "Theta") => concat!(
      "-S Exp[-q T] ",
      pdf!(d1!()),
      " v/(2 Sqrt[T]) + r K Exp[-r T] ",
      cdf!(concat!("-", d2!())),
      " - q S Exp[-q T] ",
      cdf!(concat!("-", d1!()))
    ),
    (true, "Rho") => concat!("K T Exp[-r T] ", cdf!(d2!())),
    (false, "Rho") => concat!("-K T Exp[-r T] ", cdf!(concat!("-", d2!()))),
    _ => return None,
  })
}

/// `FinancialDerivative[{"European", "Call"|"Put"}, {"StrikePrice" -> k,
/// "Expiration" -> t}, {"InterestRate" -> r, "Volatility" -> v,
/// "CurrentPrice" -> s, "Dividend" -> q}, greeks]`.
///
/// `greeks` is a single name or a list of names; it defaults to `"Value"`.
/// Returns `None` (stay unevaluated) for anything that isn't a recognised
/// European call/put with all required parameters.
pub fn financial_derivative_ast(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if args.len() < 3 || args.len() > 4 {
    return None;
  }
  let call = match &args[0] {
    Expr::List(items) if items.len() == 2 => match (&items[0], &items[1]) {
      (Expr::String(a), Expr::String(b)) if a == "European" => match b.as_str()
      {
        "Call" => true,
        "Put" => false,
        _ => return None,
      },
      _ => return None,
    },
    _ => return None,
  };

  let strike = lookup(&args[1], "StrikePrice")?;
  let expiry = lookup(&args[1], "Expiration")?;
  let rate = lookup(&args[2], "InterestRate")?;
  let vol = lookup(&args[2], "Volatility")?;
  let spot = lookup(&args[2], "CurrentPrice")?;
  let dividend = lookup(&args[2], "Dividend").unwrap_or(Expr::Integer(0));

  let build = |greek: &str| -> Option<Result<Expr, InterpreterError>> {
    let tmpl = greek_template(call, greek)?;
    let parsed = string_to_expr(tmpl).ok()?;
    let bound = substitute_variables(
      &parsed,
      &[
        ("S", &spot),
        ("K", &strike),
        ("T", &expiry),
        ("r", &rate),
        ("v", &vol),
        ("q", &dividend),
      ],
    );
    Some(crate::evaluator::evaluate_expr_to_expr(&bound))
  };

  match args.get(3) {
    None => build("Value"),
    Some(Expr::String(g)) => build(g),
    Some(Expr::List(gs)) => {
      let mut out = Vec::with_capacity(gs.len());
      for g in gs {
        let Expr::String(g) = g else {
          return None;
        };
        match build(g)? {
          Ok(v) => out.push(v),
          Err(e) => return Some(Err(e)),
        }
      }
      Some(Ok(Expr::List(out.into())))
    }
    _ => None,
  }
}
