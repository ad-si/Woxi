#[allow(unused_imports)]
use super::*;
use crate::syntax::{BinaryOperator, ComparisonOp, Expr, expr_to_string};

// ─── helpers ────────────────────────────────────────────────────────

use std::collections::HashMap;

/// Extract a map of variable_name → exponent from a list of variable factors.
/// E.g. [x^2, y] → {"x": 2, "y": 1}
pub fn extract_exponent_map(var_factors: &[Expr]) -> HashMap<String, i128> {
  let mut map = HashMap::new();
  for f in var_factors {
    match f {
      Expr::Identifier(name) => {
        *map.entry(name.clone()).or_insert(0) += 1;
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => {
        let name = expr_to_string(left);
        let exp = match right.as_ref() {
          Expr::Integer(n) => *n,
          _ => 1,
        };
        *map.entry(name).or_insert(0) += exp;
      }
      _ => {
        let name = expr_to_string(f);
        *map.entry(name).or_insert(0) += 1;
      }
    }
  }
  map
}

pub use crate::syntax::bool_expr;

/// Build a FunctionCall expression.
pub fn mk_call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

/// Build an Integer expression.
pub fn mk_int(n: i128) -> Expr {
  Expr::Integer(n)
}

/// Build a `Rational[n, d]` expression.
pub fn mk_ratio(n: i128, d: i128) -> Expr {
  Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
  }
}

/// Build `Times[a, b]`.
pub fn mk_times(a: Expr, b: Expr) -> Expr {
  mk_call("Times", vec![a, b])
}

/// Build `Power[base, exp]`.
pub fn mk_power(base: Expr, exp: Expr) -> Expr {
  mk_call("Power", vec![base, exp])
}

// ─── Ascending trimmed polynomial coefficient vectors ────────────────

/// Drop trailing zero coefficients (keeping at least one entry).
pub fn trim(v: &mut Vec<i128>) {
  while v.len() > 1 && *v.last().unwrap() == 0 {
    v.pop();
  }
}

/// True for the zero polynomial `[0]`.
pub fn is_zero(v: &[i128]) -> bool {
  v == [0]
}

/// Degree of a trimmed coefficient vector.
pub fn deg(v: &[i128]) -> usize {
  v.len() - 1
}

// ─── Helper functions for building expressions ─────────────────────

/// Build an equality comparison: lhs == rhs
pub fn make_equality(lhs: &Expr, rhs: &Expr) -> Expr {
  Expr::Comparison {
    operands: vec![lhs.clone(), rhs.clone()],
    operators: vec![ComparisonOp::Equal],
  }
}

/// Build a comparison expression.
pub fn make_comparison(lhs: &Expr, rhs: &Expr, op: CompOp) -> Expr {
  let comp_op = match op {
    CompOp::Equal => ComparisonOp::Equal,
    CompOp::NotEqual => ComparisonOp::NotEqual,
    CompOp::Less => ComparisonOp::Less,
    CompOp::LessEqual => ComparisonOp::LessEqual,
    CompOp::Greater => ComparisonOp::Greater,
    CompOp::GreaterEqual => ComparisonOp::GreaterEqual,
  };
  Expr::Comparison {
    operands: vec![lhs.clone(), rhs.clone()],
    operators: vec![comp_op],
  }
}

/// Build a compound inequality: low op1 var op2 high
/// Always produces Inequality[low, Op1, var, Op2, high] as an Expr::FunctionCall,
/// matching Wolfram's Reduce which always returns Inequality head.
pub fn make_compound_inequality(
  low: &Expr,
  op1: CompOp,
  var: &str,
  op2: CompOp,
  high: &Expr,
) -> Expr {
  let op1_name = match op1 {
    CompOp::Less => "Less",
    CompOp::LessEqual => "LessEqual",
    CompOp::Greater => "Greater",
    CompOp::GreaterEqual => "GreaterEqual",
    _ => "Less",
  };
  let op2_name = match op2 {
    CompOp::Less => "Less",
    CompOp::LessEqual => "LessEqual",
    CompOp::Greater => "Greater",
    CompOp::GreaterEqual => "GreaterEqual",
    _ => "Less",
  };
  Expr::FunctionCall {
    name: "Inequality".to_string(),
    args: vec![
      low.clone(),
      Expr::Identifier(op1_name.to_string()),
      Expr::Identifier(var.to_string()),
      Expr::Identifier(op2_name.to_string()),
      high.clone(),
    ]
    .into(),
  }
}

/// Build an Or expression from a list of terms.
pub fn build_or(mut terms: Vec<Expr>) -> Expr {
  // Filter out False
  terms.retain(|t| !matches!(t, Expr::Identifier(s) if s == "False"));

  if terms.is_empty() {
    return Expr::Identifier("False".to_string());
  }
  if terms.len() == 1 {
    return terms.remove(0);
  }
  terms
    .iter()
    .skip(1)
    .fold(terms[0].clone(), |acc, t| Expr::BinaryOp {
      op: BinaryOperator::Or,
      left: Box::new(acc),
      right: Box::new(t.clone()),
    })
}

/// Build a sum from parts.
pub fn build_sum_from_parts(parts: &[Expr]) -> Expr {
  if parts.is_empty() {
    Expr::Integer(0)
  } else {
    parts
      .iter()
      .skip(1)
      .fold(parts[0].clone(), |acc, p| add_exprs(&acc, p))
  }
}

/// Flip an inequality operator.
pub fn flip_op(op: CompOp) -> CompOp {
  match op {
    CompOp::Less => CompOp::Greater,
    CompOp::LessEqual => CompOp::GreaterEqual,
    CompOp::Greater => CompOp::Less,
    CompOp::GreaterEqual => CompOp::LessEqual,
    other => other,
  }
}

/// Combine two results with Or.
pub fn or_results(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Identifier(s), _) if s == "False" => b.clone(),
    (_, Expr::Identifier(s)) if s == "False" => a.clone(),
    (Expr::Identifier(s), _) if s == "True" => {
      Expr::Identifier("True".to_string())
    }
    (_, Expr::Identifier(s)) if s == "True" => {
      Expr::Identifier("True".to_string())
    }
    _ => Expr::BinaryOp {
      op: BinaryOperator::Or,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}

/// Combine two results with And.
pub fn and_results(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Identifier(s), _) if s == "True" => b.clone(),
    (_, Expr::Identifier(s)) if s == "True" => a.clone(),
    (Expr::Identifier(s), _) if s == "False" => {
      Expr::Identifier("False".to_string())
    }
    (_, Expr::Identifier(s)) if s == "False" => {
      Expr::Identifier("False".to_string())
    }
    _ => Expr::BinaryOp {
      op: BinaryOperator::And,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}

/// Compare two expressions for ordering (used to sort solutions).
pub(super) fn compare_exprs(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  // Try numeric comparison first
  if let (Some(va), Some(vb)) = (expr_to_number(a), expr_to_number(b)) {
    return va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal);
  }
  // Compare solution values within equalities
  if let (Some((_, rhs_a, _)), Some((_, rhs_b, _))) =
    (extract_comparison(a), extract_comparison(b))
  {
    if let (Some(va), Some(vb)) =
      (expr_to_number(&rhs_a), expr_to_number(&rhs_b))
    {
      return va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal);
    }
    // For symbolic expressions, compare string representations
    let sa = expr_to_string(&rhs_a);
    let sb = expr_to_string(&rhs_b);
    // Negative values come first
    let a_neg = sa.starts_with('-');
    let b_neg = sb.starts_with('-');
    if a_neg && !b_neg {
      return std::cmp::Ordering::Less;
    }
    if !a_neg && b_neg {
      return std::cmp::Ordering::Greater;
    }
    return sa.cmp(&sb);
  }
  let sa = expr_to_string(a);
  let sb = expr_to_string(b);
  sa.cmp(&sb)
}
