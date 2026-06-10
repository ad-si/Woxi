//! Resolve[Exists[...]/ForAll[...], Reals] — quantifier elimination for
//! univariate polynomial conditions, matching wolframscript:
//! parameter-free formulas decide via Reduce (with complex solutions
//! discarded over the reals), and the parametrized doc-example families
//! Exists[x, x^even == c] -> c >= 0 and ForAll[x, x^even + c > 0] ->
//! c > 0 return their conditions.

use crate::InterpreterError;
use crate::syntax::{ComparisonOp, Expr};

pub fn resolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "Resolve".to_string(),
    args: args.to_vec().into(),
  };
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated(args));
  }
  // Optional domain: only Reals (or omitted, which defaults to complexes
  // in Wolfram, but the rules below are domain-agnostic for the
  // supported families)
  if args.len() == 2 && !matches!(&args[1], Expr::Identifier(d) if d == "Reals")
  {
    return Ok(unevaluated(args));
  }
  let over_reals = args.len() == 2;

  let (head, var, cond) = match &args[0] {
    Expr::FunctionCall { name, args: qargs }
      if (name == "Exists" || name == "ForAll") && qargs.len() == 2 =>
    {
      match &qargs[0] {
        Expr::Identifier(v) => (name.as_str(), v.clone(), qargs[1].clone()),
        _ => return Ok(unevaluated(args)),
      }
    }
    _ => return Ok(unevaluated(args)),
  };

  let truth = |b: bool| {
    Ok(Expr::Identifier(
      if b { "True" } else { "False" }.to_string(),
    ))
  };

  // Parametrized templates
  if let Some(result) = parametric_template(head, &var, &cond) {
    return Ok(result);
  }

  // Parameter-free path: decide via Reduce
  if has_free_symbol(&cond, &var) {
    return Ok(unevaluated(args));
  }
  match head {
    "Exists" => {
      let reduced =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Reduce".to_string(),
          args: vec![cond.clone(), Expr::Identifier(var.clone())].into(),
        })?;
      match &reduced {
        Expr::Identifier(s) if s == "False" => truth(false),
        Expr::Identifier(s) if s == "True" => truth(true),
        Expr::FunctionCall { name, .. } if name == "Reduce" => {
          Ok(unevaluated(args))
        }
        solution => {
          if over_reals {
            // Discard complex solution branches
            truth(any_real_branch(solution))
          } else {
            truth(true)
          }
        }
      }
    }
    "ForAll" => {
      // ForAll[x, cond] == !Exists[x, !cond] for invertible comparisons
      let negated = match negate_comparison(&cond) {
        Some(n) => n,
        None => return Ok(unevaluated(args)),
      };
      let reduced =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Reduce".to_string(),
          args: vec![negated, Expr::Identifier(var.clone())].into(),
        })?;
      match &reduced {
        Expr::Identifier(s) if s == "False" => truth(true),
        Expr::Identifier(s) if s == "True" => truth(false),
        Expr::FunctionCall { name, .. } if name == "Reduce" => {
          Ok(unevaluated(args))
        }
        solution => {
          if over_reals {
            truth(!any_real_branch(solution))
          } else {
            truth(false)
          }
        }
      }
    }
    _ => Ok(unevaluated(args)),
  }
}

/// The documented parametrized families:
/// Exists[x, x^even == c] -> c >= 0
/// ForAll[x, x^even + c > 0] -> c > 0 (and >= 0 for GreaterEqual)
fn parametric_template(head: &str, var: &str, cond: &Expr) -> Option<Expr> {
  let (operands, op) = match cond {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => {
      (operands, operators[0])
    }
    _ => return None,
  };
  let is_even_power_of_var = |e: &Expr| -> bool {
    let (base, exp) = match e {
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left,
        right,
      } => (&**left as &Expr, &**right as &Expr),
      _ => return false,
    };
    matches!(base, Expr::Identifier(v) if v == var)
      && matches!(exp, Expr::Integer(k) if *k >= 2 && k % 2 == 0)
  };
  let free_symbol = |e: &Expr| -> Option<Expr> {
    match e {
      Expr::Identifier(s) if s != var => Some(e.clone()),
      _ => None,
    }
  };

  match head {
    // Exists[x, x^even == c]
    "Exists" if op == ComparisonOp::Equal => {
      let c = if is_even_power_of_var(&operands[0]) {
        free_symbol(&operands[1])?
      } else if is_even_power_of_var(&operands[1]) {
        free_symbol(&operands[0])?
      } else {
        return None;
      };
      Some(Expr::Comparison {
        operands: vec![c, Expr::Integer(0)],
        operators: vec![ComparisonOp::GreaterEqual],
      })
    }
    // ForAll[x, x^even + c > 0] (Plus arrives canonically as c + x^even)
    "ForAll"
      if (op == ComparisonOp::Greater || op == ComparisonOp::GreaterEqual)
        && matches!(&operands[1], Expr::Integer(0)) =>
    {
      let terms: Vec<&Expr> = match &operands[0] {
        Expr::FunctionCall { name, args }
          if name == "Plus" && args.len() == 2 =>
        {
          args.iter().collect()
        }
        _ => return None,
      };
      let c = if is_even_power_of_var(terms[1]) {
        free_symbol(terms[0])?
      } else if is_even_power_of_var(terms[0]) {
        free_symbol(terms[1])?
      } else {
        return None;
      };
      Some(Expr::Comparison {
        operands: vec![c, Expr::Integer(0)],
        operators: vec![op],
      })
    }
    _ => None,
  }
}

/// Any identifier other than `var` (and protected constants)?
fn has_free_symbol(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(s) => {
      s != var && !matches!(s.as_str(), "True" | "False" | "Pi" | "E" | "I")
    }
    Expr::Constant(_) => false,
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(|a| has_free_symbol(a, var))
    }
    Expr::Comparison { operands, .. } => {
      operands.iter().any(|a| has_free_symbol(a, var))
    }
    Expr::BinaryOp { left, right, .. } => {
      has_free_symbol(left, var) || has_free_symbol(right, var)
    }
    Expr::UnaryOp { operand, .. } => has_free_symbol(operand, var),
    _ => false,
  }
}

/// Negate a simple comparison (chains and equations stay unsupported).
fn negate_comparison(cond: &Expr) -> Option<Expr> {
  match cond {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => {
      let flipped = match operators[0] {
        ComparisonOp::Less => ComparisonOp::GreaterEqual,
        ComparisonOp::LessEqual => ComparisonOp::Greater,
        ComparisonOp::Greater => ComparisonOp::LessEqual,
        ComparisonOp::GreaterEqual => ComparisonOp::Less,
        _ => return None,
      };
      Some(Expr::Comparison {
        operands: operands.clone(),
        operators: vec![flipped],
      })
    }
    _ => None,
  }
}

/// Does any Or-branch of a Reduce solution avoid the imaginary unit?
fn any_real_branch(solution: &Expr) -> bool {
  let branches: Vec<&Expr> = match solution {
    Expr::FunctionCall { name, args } if name == "Or" => args.iter().collect(),
    other => vec![other],
  };
  branches.iter().any(|b| !contains_imaginary(b))
}

fn contains_imaginary(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(s) | Expr::Constant(s) => s == "I",
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(contains_imaginary)
    }
    Expr::Comparison { operands, .. } => {
      operands.iter().any(contains_imaginary)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_imaginary(left) || contains_imaginary(right)
    }
    Expr::UnaryOp { operand, .. } => contains_imaginary(operand),
    _ => false,
  }
}
