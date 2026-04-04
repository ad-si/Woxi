#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

use crate::functions::calculus_ast::simplify;

// ─── Refine ─────────────────────────────────────────────────────────

/// Refine[expr, assumption] - Simplify an expression under assumptions.
/// Refine[expr] - Simplify using default assumptions.
/// E.g. Refine[Sqrt[x^2], x > 0] → x
pub fn refine_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Refine expects 1 or 2 arguments".into(),
    ));
  }

  // Single argument: just evaluate/return the expression
  if args.len() == 1 {
    return Ok(args[0].clone());
  }

  let expr = &args[0];
  let assumption = &args[1];

  // Extract assumption info
  let info = extract_assumption_info(assumption);

  // Recursively simplify the expression under the assumption
  let result = refine_expr(expr, &info, assumption);

  Ok(result)
}

/// Information extracted from assumptions
struct AssumptionInfo {
  positive_vars: Vec<String>,
  negative_vars: Vec<String>,
  real_vars: Vec<String>,
  integer_vars: Vec<String>,
}

/// Extract all assumption information from the assumption expression.
fn extract_assumption_info(assumption: &Expr) -> AssumptionInfo {
  let mut info = AssumptionInfo {
    positive_vars: Vec::new(),
    negative_vars: Vec::new(),
    real_vars: Vec::new(),
    integer_vars: Vec::new(),
  };
  extract_assumptions_inner(assumption, &mut info);
  // Positive/negative vars are also real
  for v in &info.positive_vars {
    if !info.real_vars.contains(v) {
      info.real_vars.push(v.clone());
    }
  }
  for v in &info.negative_vars {
    if !info.real_vars.contains(v) {
      info.real_vars.push(v.clone());
    }
  }
  info
}

/// Check if an expression is a non-negative numeric constant (integer, real, or rational).
fn is_nonnegative_constant(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(n) => *n >= 0,
    Expr::BigInteger(n) => *n >= num_bigint::BigInt::from(0),
    Expr::Real(f) => *f >= 0.0,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(a), Expr::Integer(b)) => {
          (*a >= 0 && *b > 0) || (*a <= 0 && *b < 0)
        }
        _ => false,
      }
    }
    _ => false,
  }
}

/// Check if an expression is a non-positive numeric constant.
fn is_nonpositive_constant(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(n) => *n <= 0,
    Expr::BigInteger(n) => *n <= num_bigint::BigInt::from(0),
    Expr::Real(f) => *f <= 0.0,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(a), Expr::Integer(b)) => {
          (*a <= 0 && *b > 0) || (*a >= 0 && *b < 0)
        }
        _ => false,
      }
    }
    _ => false,
  }
}

fn extract_assumptions_inner(assumption: &Expr, info: &mut AssumptionInfo) {
  match assumption {
    Expr::Comparison {
      operands,
      operators,
    } => {
      if operands.len() == 2 && operators.len() == 1 {
        let op = &operators[0];
        let left = &operands[0];
        let right = &operands[1];

        // x > c, x >= c where c >= 0 → positive
        if matches!(
          op,
          crate::syntax::ComparisonOp::Greater
            | crate::syntax::ComparisonOp::GreaterEqual
        ) && let Expr::Identifier(name) = left
          && is_nonnegative_constant(right)
        {
          info.positive_vars.push(name.clone());
        }

        // c < x, c <= x where c >= 0 → positive
        if matches!(
          op,
          crate::syntax::ComparisonOp::Less
            | crate::syntax::ComparisonOp::LessEqual
        ) && let Expr::Identifier(name) = right
          && is_nonnegative_constant(left)
        {
          info.positive_vars.push(name.clone());
        }

        // x < c, x <= c where c <= 0 → negative
        if matches!(
          op,
          crate::syntax::ComparisonOp::Less
            | crate::syntax::ComparisonOp::LessEqual
        ) && let Expr::Identifier(name) = left
          && is_nonpositive_constant(right)
        {
          info.negative_vars.push(name.clone());
        }

        // c > x, c >= x where c <= 0 → negative
        if matches!(
          op,
          crate::syntax::ComparisonOp::Greater
            | crate::syntax::ComparisonOp::GreaterEqual
        ) && let Expr::Identifier(name) = right
          && is_nonpositive_constant(left)
        {
          info.negative_vars.push(name.clone());
        }
      }
    }
    // Element[x, domain]
    Expr::FunctionCall { name, args }
      if name == "Element" && args.len() == 2 =>
    {
      if let Expr::Identifier(var_name) = &args[0]
        && let Expr::Identifier(domain) = &args[1]
      {
        match domain.as_str() {
          "Reals" => info.real_vars.push(var_name.clone()),
          "Integers" | "Primes" => {
            info.integer_vars.push(var_name.clone());
            info.real_vars.push(var_name.clone());
          }
          "Rationals" | "Algebraics" => {
            info.real_vars.push(var_name.clone());
          }
          _ => {}
        }
      }
    }
    // And[cond1, cond2, ...]
    Expr::FunctionCall { name, args } if name == "And" => {
      for arg in args {
        extract_assumptions_inner(arg, info);
      }
    }
    _ => {}
  }
}

/// Check if the assumption implies a comparison is True or False.
/// Returns Some(true) for True, Some(false) for False, None if undetermined.
fn check_comparison_under_assumption(
  expr: &Expr,
  info: &AssumptionInfo,
  assumption: &Expr,
) -> Option<bool> {
  // Check if the comparison is identical to the assumption
  if expr_to_string(expr) == expr_to_string(assumption) {
    return Some(true);
  }
  // Also check individual conjuncts in And assumptions
  if let Expr::FunctionCall { name, args } = assumption
    && name == "And"
  {
    for arg in args {
      if expr_to_string(expr) == expr_to_string(arg) {
        return Some(true);
      }
    }
  }

  if let Expr::Comparison {
    operands,
    operators,
  } = expr
    && operands.len() == 2
    && operators.len() == 1
  {
    let left = &operands[0];
    let right = &operands[1];
    let op = &operators[0];

    // Patterns: x > 0, x >= 0, x < 0, etc. where x is a known positive/negative var
    if let Expr::Identifier(var_name) = left
      && (is_nonnegative_constant(right) || is_nonpositive_constant(right))
    {
      let rhs_is_zero = matches!(right, Expr::Integer(0));

      if info.positive_vars.contains(var_name) {
        // x is positive
        match op {
          crate::syntax::ComparisonOp::Greater if rhs_is_zero => {
            return Some(true);
          }
          crate::syntax::ComparisonOp::GreaterEqual
            if is_nonnegative_constant(right) && rhs_is_zero =>
          {
            return Some(true);
          }
          crate::syntax::ComparisonOp::Less
            if is_nonnegative_constant(right) && rhs_is_zero =>
          {
            return Some(false);
          }
          crate::syntax::ComparisonOp::LessEqual
            if is_nonpositive_constant(right) && rhs_is_zero =>
          {
            return Some(false);
          }
          _ => {}
        }
      }
      if info.negative_vars.contains(var_name) {
        // x is negative
        match op {
          crate::syntax::ComparisonOp::Less if rhs_is_zero => {
            return Some(true);
          }
          crate::syntax::ComparisonOp::LessEqual
            if is_nonpositive_constant(right) && rhs_is_zero =>
          {
            return Some(true);
          }
          crate::syntax::ComparisonOp::Greater
            if is_nonpositive_constant(right) && rhs_is_zero =>
          {
            return Some(false);
          }
          crate::syntax::ComparisonOp::GreaterEqual
            if is_nonnegative_constant(right) && rhs_is_zero =>
          {
            return Some(false);
          }
          _ => {}
        }
      }
    }

    // Check implication from specific numeric bounds in assumption
    // e.g., x > 1 implies x > 0 (True) and x < 0 (False)
    if let Expr::Identifier(var_name) = left
      && let Some(bound) = get_lower_bound(var_name, assumption)
    {
      // We know var_name > bound (or >= bound)
      if let Expr::Integer(rhs_val) = right
        && let Expr::Integer(bound_val) = &bound
      {
        match op {
          crate::syntax::ComparisonOp::Greater if *bound_val > *rhs_val => {
            return Some(true);
          }
          crate::syntax::ComparisonOp::GreaterEqual
            if *bound_val >= *rhs_val =>
          {
            return Some(true);
          }
          crate::syntax::ComparisonOp::Less if *bound_val >= *rhs_val => {
            return Some(false);
          }
          crate::syntax::ComparisonOp::LessEqual if *bound_val > *rhs_val => {
            return Some(false);
          }
          _ => {}
        }
      }
    }
  }

  None
}

/// Get the lower bound of a variable from the assumption.
/// Returns the bound value if the assumption is of the form var > bound or var >= bound.
fn get_lower_bound(var_name: &str, assumption: &Expr) -> Option<Expr> {
  match assumption {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => {
      // var > c or var >= c
      if matches!(
        operators[0],
        crate::syntax::ComparisonOp::Greater
          | crate::syntax::ComparisonOp::GreaterEqual
      ) && let Expr::Identifier(name) = &operands[0]
        && name == var_name
      {
        return Some(operands[1].clone());
      }
      // c < var or c <= var
      if matches!(
        operators[0],
        crate::syntax::ComparisonOp::Less
          | crate::syntax::ComparisonOp::LessEqual
      ) && let Expr::Identifier(name) = &operands[1]
        && name == var_name
      {
        return Some(operands[0].clone());
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      for arg in args {
        if let Some(bound) = get_lower_bound(var_name, arg) {
          return Some(bound);
        }
      }
      None
    }
    _ => None,
  }
}

/// Recursively apply Refine simplification rules.
fn refine_expr(expr: &Expr, info: &AssumptionInfo, assumption: &Expr) -> Expr {
  match expr {
    // Comparisons: check if they can be resolved under assumptions
    Expr::Comparison { .. } => {
      if let Some(result) =
        check_comparison_under_assumption(expr, info, assumption)
      {
        return Expr::Identifier(
          if result { "True" } else { "False" }.to_string(),
        );
      }
      expr.clone()
    }

    // (var^n)^(1/m) → var^(n/m) when var >= 0 and n divisible by m
    // Also handles Sqrt[var^2] as special case (m=2, n=2)
    // For var < 0: only simplifies when n is even
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: outer_base,
      right: outer_exp,
    } if extract_rational_1_over(outer_exp).is_some() => {
      let m = extract_rational_1_over(outer_exp).unwrap();
      if let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: base,
        right: exp,
      } = outer_base.as_ref()
        && let Expr::Integer(n) = exp.as_ref()
        && *n > 0
        && n % m == 0
        && let Expr::Identifier(var_name) = base.as_ref()
      {
        let reduced = n / m;
        if info.positive_vars.contains(var_name) {
          return make_power_or_identity(base, reduced);
        }
        if info.negative_vars.contains(var_name) && n % 2 == 0 {
          // x^n is positive when n is even, so (x^n)^(1/m) = |x|^(n/m)
          let abs_power = make_power_or_identity(base, reduced);
          // If reduced is even, (-x)^reduced = x^reduced
          if reduced % 2 == 0 {
            return abs_power;
          }
          return Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand: Box::new(abs_power),
          };
        }
      }
      // Recurse
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(refine_expr(outer_base, info, assumption)),
        right: Box::new(refine_expr(outer_exp, info, assumption)),
      }
    }

    // Abs[var] → var when var > 0, -var when var < 0
    Expr::FunctionCall { name, args } if name == "Abs" && args.len() == 1 => {
      let refined_arg = refine_expr(&args[0], info, assumption);
      // Check if the refined argument is a product of known-sign variables
      if let Some(result) = simplify_abs_with_signs(&refined_arg, info) {
        return result;
      }
      Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![refined_arg],
      }
    }

    // Sign[var] → 1 when var > 0, -1 when var < 0
    Expr::FunctionCall { name, args } if name == "Sign" && args.len() == 1 => {
      if let Expr::Identifier(var_name) = &args[0] {
        if info.positive_vars.contains(var_name) {
          return Expr::Integer(1);
        }
        if info.negative_vars.contains(var_name) {
          return Expr::Integer(-1);
        }
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      Expr::FunctionCall {
        name: "Sign".to_string(),
        args: vec![refined_arg],
      }
    }

    // Arg[var] → 0 when var > 0, Pi when var < 0
    Expr::FunctionCall { name, args } if name == "Arg" && args.len() == 1 => {
      if let Expr::Identifier(var_name) = &args[0] {
        if info.positive_vars.contains(var_name) {
          return Expr::Integer(0);
        }
        if info.negative_vars.contains(var_name) {
          return Expr::Constant("Pi".to_string());
        }
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      Expr::FunctionCall {
        name: "Arg".to_string(),
        args: vec![refined_arg],
      }
    }

    // Re[var] → var when var ∈ Reals
    Expr::FunctionCall { name, args } if name == "Re" && args.len() == 1 => {
      if let Expr::Identifier(var_name) = &args[0]
        && info.real_vars.contains(var_name)
      {
        return args[0].clone();
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      Expr::FunctionCall {
        name: "Re".to_string(),
        args: vec![refined_arg],
      }
    }

    // Im[var] → 0 when var ∈ Reals
    Expr::FunctionCall { name, args } if name == "Im" && args.len() == 1 => {
      if let Expr::Identifier(var_name) = &args[0]
        && info.real_vars.contains(var_name)
      {
        return Expr::Integer(0);
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      Expr::FunctionCall {
        name: "Im".to_string(),
        args: vec![refined_arg],
      }
    }

    // Floor[var] → var when var ∈ Integers
    Expr::FunctionCall { name, args } if name == "Floor" && args.len() == 1 => {
      if let Expr::Identifier(var_name) = &args[0]
        && info.integer_vars.contains(var_name)
      {
        return args[0].clone();
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      Expr::FunctionCall {
        name: "Floor".to_string(),
        args: vec![refined_arg],
      }
    }

    // Ceiling[var] → var when var ∈ Integers
    Expr::FunctionCall { name, args }
      if name == "Ceiling" && args.len() == 1 =>
    {
      if let Expr::Identifier(var_name) = &args[0]
        && info.integer_vars.contains(var_name)
      {
        return args[0].clone();
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      Expr::FunctionCall {
        name: "Ceiling".to_string(),
        args: vec![refined_arg],
      }
    }

    // ConditionalExpression[val, cond] → val if cond matches assumption
    Expr::FunctionCall { name, args }
      if name == "ConditionalExpression" && args.len() == 2 =>
    {
      let cond_str = expr_to_string(&args[1]);
      let assumption_str = expr_to_string(assumption);
      if cond_str == assumption_str {
        return refine_expr(&args[0], info, assumption);
      }
      Expr::FunctionCall {
        name: name.clone(),
        args: args
          .iter()
          .map(|a| refine_expr(a, info, assumption))
          .collect(),
      }
    }

    // Recurse into function calls
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| refine_expr(a, info, assumption))
        .collect(),
    },

    // Recurse into binary ops
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(refine_expr(left, info, assumption)),
      right: Box::new(refine_expr(right, info, assumption)),
    },

    // Recurse into unary ops
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(refine_expr(operand, info, assumption)),
    },

    // Recurse into lists
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|i| refine_expr(i, info, assumption))
        .collect(),
    ),

    // Everything else: return as-is
    _ => expr.clone(),
  }
}

/// Extract m from Rational[1, m] (i.e. check if expr is 1/m for positive integer m).
fn extract_rational_1_over(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(1), Expr::Integer(d)) = (&args[0], &args[1])
        && *d > 0
      {
        return Some(*d);
      }
      None
    }
    _ => None,
  }
}

/// Build Power[base, exp] or just base if exp == 1.
fn make_power_or_identity(base: &Expr, exp: i128) -> Expr {
  if exp == 1 {
    base.clone()
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(base.clone()),
      right: Box::new(Expr::Integer(exp)),
    }
  }
}

/// Try to simplify Abs[expr] when we know the signs of variables.
/// Returns Some(simplified) if possible, None otherwise.
fn simplify_abs_with_signs(expr: &Expr, info: &AssumptionInfo) -> Option<Expr> {
  match expr {
    Expr::Identifier(name) => {
      if info.positive_vars.contains(name) {
        Some(expr.clone())
      } else if info.negative_vars.contains(name) {
        Some(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(expr.clone()),
        })
      } else {
        None
      }
    }
    // For products: Abs[a * b] = Abs[a] * Abs[b], then simplify each
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let left_sign = get_sign(left, info);
      let right_sign = get_sign(right, info);
      match (left_sign, right_sign) {
        (Some(_), Some(_)) => {
          // We know both signs, compute sign of product
          let left_abs = abs_of_known_sign(left, info)?;
          let right_abs = abs_of_known_sign(right, info)?;
          let product_sign = get_sign(expr, info)?;
          let abs_product = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(left_abs),
            right: Box::new(right_abs),
          };
          if product_sign > 0 {
            Some(abs_product)
          } else {
            Some(abs_product)
          }
        }
        _ => None,
      }
    }
    // Times as FunctionCall
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      // Check if all factors have known sign
      let mut all_known = true;
      for arg in args {
        if get_sign(arg, info).is_none() {
          all_known = false;
          break;
        }
      }
      if all_known {
        let abs_factors: Vec<Expr> = args
          .iter()
          .filter_map(|a| abs_of_known_sign(a, info))
          .collect();
        if abs_factors.len() == args.len() {
          if abs_factors.len() == 1 {
            return Some(abs_factors.into_iter().next().unwrap());
          }
          return Some(Expr::FunctionCall {
            name: "Times".to_string(),
            args: abs_factors,
          });
        }
      }
      None
    }
    _ => None,
  }
}

/// Get the sign of an expression: Some(1) for positive, Some(-1) for negative, None if unknown.
fn get_sign(expr: &Expr, info: &AssumptionInfo) -> Option<i8> {
  match expr {
    Expr::Identifier(name) => {
      if info.positive_vars.contains(name) {
        Some(1)
      } else if info.negative_vars.contains(name) {
        Some(-1)
      } else {
        None
      }
    }
    Expr::Integer(n) => {
      if *n > 0 {
        Some(1)
      } else if *n < 0 {
        Some(-1)
      } else {
        Some(0)
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let ls = get_sign(left, info)?;
      let rs = get_sign(right, info)?;
      Some(ls * rs)
    }
    _ => None,
  }
}

/// Get the absolute value of an expression with known sign.
fn abs_of_known_sign(expr: &Expr, info: &AssumptionInfo) -> Option<Expr> {
  let sign = get_sign(expr, info)?;
  if sign >= 0 {
    Some(expr.clone())
  } else {
    Some(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(expr.clone()),
    })
  }
}

// ─── Simplify ───────────────────────────────────────────────────────

/// Simplify[expr] or Simplify[expr, Assumptions -> cond] - User-facing simplification
pub fn simplify_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Simplify expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 2 {
    return simplify_with_assumptions(&args[0], &args[1], false);
  }
  Ok(simplify_expr(&args[0]))
}

/// FullSimplify[expr] or FullSimplify[expr, Assumptions -> cond]
pub fn full_simplify_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FullSimplify expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 2 {
    return simplify_with_assumptions(&args[0], &args[1], true);
  }
  // Thread over Lists
  if let Expr::List(items) = &args[0] {
    let results: Vec<Expr> = items.iter().map(full_simplify_expr).collect();
    return Ok(Expr::List(results));
  }
  Ok(full_simplify_expr(&args[0]))
}

/// Apply Simplify or FullSimplify with an Assumptions option.
fn simplify_with_assumptions(
  expr: &Expr,
  opts: &Expr,
  full: bool,
) -> Result<Expr, InterpreterError> {
  // Extract Assumptions -> value from the options argument
  let assumption = match opts {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      if let Expr::Identifier(name) = pattern.as_ref() {
        if name == "Assumptions" {
          Some(replacement.as_ref().clone())
        } else {
          None
        }
      } else {
        None
      }
    }
    _ => None,
  };

  if let Some(assumption_val) = assumption {
    // Save previous $Assumptions
    let prev = crate::ENV.with(|e| e.borrow().get("$Assumptions").cloned());

    // Set $Assumptions
    let val = expr_to_string(&assumption_val);
    crate::ENV.with(|e| {
      e.borrow_mut()
        .insert("$Assumptions".to_string(), crate::StoredValue::Raw(val))
    });

    let result = if full {
      full_simplify_expr(expr)
    } else {
      simplify_expr(expr)
    };

    // Restore previous $Assumptions
    crate::ENV.with(|e| {
      let mut env = e.borrow_mut();
      if let Some(v) = prev {
        env.insert("$Assumptions".to_string(), v);
      } else {
        env.remove("$Assumptions");
      }
    });

    Ok(result)
  } else {
    // Unknown option, just simplify normally
    if full {
      Ok(full_simplify_expr(expr))
    } else {
      Ok(simplify_expr(expr))
    }
  }
}

/// FullSimplify: more aggressive than Simplify.
/// Expands, applies trig identities, factors out common terms, and tries factoring.
pub fn full_simplify_expr(expr: &Expr) -> Expr {
  // Thread over Lists
  if let Expr::List(items) = expr {
    let results: Vec<Expr> = items.iter().map(full_simplify_expr).collect();
    return Expr::List(results);
  }

  // Combine Abs quotients/products before other simplification
  let abs_combined = simplify_abs_products(expr);

  // First apply regular simplification
  let simplified = simplify_expr(&abs_combined);

  // Then expand fully and combine
  let expanded = expand_and_combine(&simplified);

  // Apply trig identities
  let trig_simplified = apply_trig_identities(&expanded);

  // Keep track of the best (simplest) form using leaf count as complexity.
  // Include the pre-expansion simplified form as a candidate — expand_and_combine
  // can undo fraction-combining done by Simplify.
  let mut best = trig_simplified.clone();
  let mut best_complexity = leaf_count(&best);
  {
    let c = leaf_count(&simplified);
    if c <= best_complexity {
      best = simplified.clone();
      best_complexity = c;
    }
  }

  // Try factoring (Factor[expr]) — prefer factored forms (use <=)
  if let Ok(factored) =
    crate::functions::polynomial_ast::factor_ast(&[trig_simplified.clone()])
  {
    let c = leaf_count(&factored);
    if c <= best_complexity {
      best = factored;
      best_complexity = c;
    }
  }

  // Try FactorTerms to factor out common numeric/symbolic terms
  let terms = collect_additive_terms(&trig_simplified);
  if terms.len() >= 2 {
    if let Ok(factored) = crate::functions::polynomial_ast::factor_terms_ast(&[
      trig_simplified.clone(),
    ]) {
      let c = leaf_count(&factored);
      if c <= best_complexity {
        best = factored;
        best_complexity = c;
      }
    }

    // Try extracting common symbolic factors from all terms
    if let Some(factored) = factor_common_symbolic(&trig_simplified, &terms) {
      let c = leaf_count(&factored);
      if c <= best_complexity {
        best = factored;
        best_complexity = c;
      }
    }

    // Try factoring out minimum power of common base
    if let Some(factored) = factor_common_power_base(&terms) {
      let c = leaf_count(&factored);
      if c <= best_complexity {
        best = factored;
        best_complexity = c;
      }
    }
  }

  // Try combining like-denominator terms
  let with_fracs = combine_like_denominator_terms(&best);
  {
    let c = leaf_count(&with_fracs);
    if c < best_complexity {
      best = with_fracs;
      best_complexity = c;
    }
  }

  // Try Together + factor + cancel
  let together = try_together_simplify(&best);
  {
    let c = leaf_count(&together);
    if c < best_complexity {
      best = together;
      best_complexity = c;
    }
  }

  let _ = best_complexity; // suppress unused warning
  best
}

/// Full simplification: expand, combine like terms, simplify.
pub fn simplify_expr(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::Constant(_)
    | Expr::Identifier(_) => expr.clone(),

    // Thread over Lists
    Expr::List(items) => Expr::List(items.iter().map(simplify_expr).collect()),

    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let num = simplify_expr(left);
      let den = simplify_expr(right);
      // Try to cancel: expand both and see if we can simplify
      simplify_division(&num, &den)
    }

    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let base = simplify_expr(left);
      let exp = simplify_expr(right);
      simplify(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(exp),
      })
    }

    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let l = simplify_expr(left);
      let r = simplify_expr(right);
      // Combine powers: x * x → x^2, x^a * x^b → x^(a+b)
      simplify_product(&l, &r)
    }

    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus,
      ..
    } => {
      let combined = expand_and_combine(expr);
      let trig = apply_trig_identities(&combined);
      let mut best = trig;
      let mut best_c = leaf_count(&best);
      // Try combining like-denominator terms
      let with_fracs = combine_like_denominator_terms(&best);
      let c = leaf_count(&with_fracs);
      if c < best_c {
        best = with_fracs;
        best_c = c;
      }
      // Try Together + factor + cancel
      let together = try_together_simplify(&best);
      let c = leaf_count(&together);
      if c < best_c {
        best = together;
      }
      best
    }

    Expr::UnaryOp { op, operand } => {
      let inner = simplify_expr(operand);
      simplify(Expr::UnaryOp {
        op: *op,
        operand: Box::new(inner),
      })
    }

    // Handle FunctionCall forms of Plus, Times, Power
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let combined = expand_and_combine(expr);
        let trig = apply_trig_identities(&combined);
        let mut best = trig;
        let mut best_c = leaf_count(&best);
        let with_fracs = combine_like_denominator_terms(&best);
        let c = leaf_count(&with_fracs);
        if c < best_c {
          best = with_fracs;
          best_c = c;
        }
        let together = try_together_simplify(&best);
        let c = leaf_count(&together);
        if c < best_c {
          best = together;
        }
        best
      }
      "Times" => {
        // Check for fraction form: Times[..., Power[den, -1]]
        let (num, den) = super::together::extract_num_den(expr);
        if !matches!(&den, Expr::Integer(1)) {
          let s_num = simplify_expr(&num);
          let s_den = simplify_expr(&den);
          return simplify_division(&s_num, &s_den);
        }
        if args.len() == 2 {
          let l = simplify_expr(&args[0]);
          let r = simplify_expr(&args[1]);
          simplify_product(&l, &r)
        } else {
          expr.clone()
        }
      }
      "Power" if args.len() == 2 => {
        let base = simplify_expr(&args[0]);
        let exp = simplify_expr(&args[1]);
        simplify(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(base),
          right: Box::new(exp),
        })
      }
      "Rational" if args.len() == 2 => expr.clone(),
      "ConditionalExpression" if args.len() == 2 => {
        simplify_conditional_expression(&args[0], &args[1])
      }
      _ => expr.clone(),
    },

    _ => simplify(expr.clone()),
  }
}

/// Simplify ConditionalExpression[value, cond] under current $Assumptions.
/// If cond matches $Assumptions → return Simplify[value]
/// If $Assumptions negates cond → return Undefined
/// Otherwise → ConditionalExpression[Simplify[value], cond]
pub fn simplify_conditional_expression(value: &Expr, cond: &Expr) -> Expr {
  let cond_str = expr_to_string(cond);

  // Get $Assumptions from environment (default: "True")
  let assumptions_str = crate::ENV
    .with(|e| {
      e.borrow().get("$Assumptions").map(|sv| match sv {
        crate::StoredValue::Raw(s) => s.clone(),
        crate::StoredValue::ExprVal(e) => expr_to_string(e),
        _ => "True".to_string(),
      })
    })
    .unwrap_or_else(|| "True".to_string());

  if cond_str == assumptions_str {
    // Condition matches assumptions → strip ConditionalExpression
    simplify_expr(value)
  } else if assumptions_str == format!("!{}", cond_str)
    || assumptions_str == format!(" !{}", cond_str)
    || assumptions_str == format!("Not[{}]", cond_str)
  {
    // Assumptions negate the condition → Undefined
    Expr::Identifier("Undefined".to_string())
  } else {
    // Keep ConditionalExpression with simplified value
    Expr::FunctionCall {
      name: "ConditionalExpression".to_string(),
      args: vec![simplify_expr(value), cond.clone()],
    }
  }
}

/// Apply trigonometric identities to a sum expression.
/// Detects a*Sin[x]^2 + a*Cos[x]^2 → a and similar patterns.
pub fn apply_trig_identities(expr: &Expr) -> Expr {
  let terms = collect_additive_terms(expr);
  if terms.len() < 2 {
    return expr.clone();
  }

  // Look for pairs: coeff*Sin[arg]^2 + coeff*Cos[arg]^2 → coeff
  let mut used = vec![false; terms.len()];
  let mut result_terms: Vec<Expr> = Vec::new();

  for i in 0..terms.len() {
    if used[i] {
      continue;
    }
    if let Some((coeff_i, arg_i, is_sin_i)) = extract_trig_squared(&terms[i]) {
      // Look for matching pair
      for j in (i + 1)..terms.len() {
        if used[j] {
          continue;
        }
        if let Some((coeff_j, arg_j, is_sin_j)) =
          extract_trig_squared(&terms[j])
          && is_sin_i != is_sin_j
          && expr_to_string(&arg_i) == expr_to_string(&arg_j)
          && expr_to_string(&coeff_i) == expr_to_string(&coeff_j)
        {
          // Found matching pair: coeff*Sin[x]^2 + coeff*Cos[x]^2 = coeff
          result_terms.push(coeff_i.clone());
          used[i] = true;
          used[j] = true;
          break;
        }
      }
    }
    if !used[i] {
      result_terms.push(terms[i].clone());
    }
  }

  if result_terms.len() == terms.len() {
    // No simplification happened
    return expr.clone();
  }

  // Re-combine to simplify (e.g. 1 + 1 → 2)
  if let Ok(result) = crate::functions::math_ast::plus_ast(&result_terms) {
    result
  } else {
    build_sum(result_terms)
  }
}

/// Try to extract (coefficient, argument, is_sin) from a term like coeff*Sin[arg]^2 or coeff*Cos[arg]^2.
pub fn extract_trig_squared(term: &Expr) -> Option<(Expr, Expr, bool)> {
  // Pattern: Sin[arg]^2 or Cos[arg]^2 (coefficient = 1)
  if let Some((func, arg)) = match_trig_squared(term) {
    return Some((Expr::Integer(1), arg, func == "Sin"));
  }

  // Pattern: coeff * Sin[arg]^2 or coeff * Cos[arg]^2
  let factors = collect_multiplicative_factors(term);
  if factors.len() < 2 {
    return None;
  }

  // Find the trig^2 factor
  for (idx, f) in factors.iter().enumerate() {
    if let Some((func, arg)) = match_trig_squared(f) {
      let mut coeff_factors: Vec<Expr> = Vec::new();
      for (j, g) in factors.iter().enumerate() {
        if j != idx {
          coeff_factors.push(g.clone());
        }
      }
      let coeff = if coeff_factors.len() == 1 {
        coeff_factors.remove(0)
      } else {
        build_product(coeff_factors)
      };
      return Some((coeff, arg, func == "Sin"));
    }
  }
  None
}

/// Match Sin[arg]^2 or Cos[arg]^2, returning ("Sin"/"Cos", arg).
pub fn match_trig_squared(expr: &Expr) -> Option<(&str, Expr)> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } if matches!(right.as_ref(), Expr::Integer(2)) => {
      if let Expr::FunctionCall { name, args } = left.as_ref()
        && (name == "Sin" || name == "Cos")
        && args.len() == 1
      {
        return Some((name.as_str(), args[0].clone()));
      }
      None
    }
    _ => None,
  }
}

/// Simplify a product, combining powers.
pub fn simplify_product(a: &Expr, b: &Expr) -> Expr {
  // x * x → x^2
  if expr_to_string(a) == expr_to_string(b) {
    return Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(a.clone()),
      right: Box::new(Expr::Integer(2)),
    };
  }

  // x^a * x^b → x^(a+b)
  let (base_a, exp_a) = extract_base_exp(a);
  let (base_b, exp_b) = extract_base_exp(b);
  if expr_to_string(&base_a) == expr_to_string(&base_b) {
    let new_exp = simplify(Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(exp_a),
      right: Box::new(exp_b),
    });
    return simplify(Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(base_a),
      right: Box::new(new_exp),
    });
  }

  simplify(Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(a.clone()),
    right: Box::new(b.clone()),
  })
}

/// Extract base and exponent from a power expression.
pub fn extract_base_exp(expr: &Expr) -> (Expr, Expr) {
  extract_base_and_exp(expr)
}

/// Simplify a division by trying polynomial cancellation.
pub fn simplify_division(num: &Expr, den: &Expr) -> Expr {
  // If same expression, return 1
  if expr_to_string(num) == expr_to_string(den) {
    return Expr::Integer(1);
  }

  // Try: if denominator is a single factor, try polynomial division
  // E.g. (x^2 - 1) / (x - 1) → x + 1
  // We try to do this by expanding the numerator and attempting polynomial long division
  let num_expanded = expand_and_combine(num);
  let den_expanded = expand_and_combine(den);

  // Try to find the variable
  if let Some(var) = find_single_variable(&num_expanded)
    && let Some(quotient) =
      poly_divide_single_var(&num_expanded, &den_expanded, &var)
  {
    return quotient;
  }

  // Use divide_two for proper evaluation (distributes powers, creates Rationals, etc.)
  if let Ok(result) =
    crate::functions::math_ast::divide_two(&num_expanded, &den_expanded)
  {
    return result;
  }
  crate::functions::math_ast::make_divide(num_expanded, den_expanded)
}

/// Find a single variable in an expression (for univariate polynomial division).
pub fn find_single_variable(expr: &Expr) -> Option<String> {
  let mut vars = std::collections::HashSet::new();
  collect_variables(expr, &mut vars);
  if vars.len() == 1 {
    vars.into_iter().next()
  } else {
    None
  }
}

/// Collect all variable names from an expression.
pub(super) fn collect_variables(
  expr: &Expr,
  vars: &mut std::collections::HashSet<String>,
) {
  match expr {
    Expr::Identifier(name)
      if name != "True" && name != "False" && name != "Null" =>
    {
      vars.insert(name.clone());
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_variables(left, vars);
      collect_variables(right, vars);
    }
    Expr::UnaryOp { operand, .. } => collect_variables(operand, vars),
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_variables(a, vars);
      }
    }
    Expr::List(items) => {
      for i in items {
        collect_variables(i, vars);
      }
    }
    _ => {}
  }
}

/// Try polynomial long division of num/den in a single variable.
/// Returns Some(quotient) if den divides num exactly.
pub fn poly_divide_single_var(
  num: &Expr,
  den: &Expr,
  var: &str,
) -> Option<Expr> {
  let num_coeffs = extract_poly_coeffs(num, var)?;
  let den_coeffs = extract_poly_coeffs(den, var)?;

  if den_coeffs.is_empty() {
    return None;
  }

  let num_deg = num_coeffs.len() as i128 - 1;
  let den_deg = den_coeffs.len() as i128 - 1;

  if num_deg < den_deg {
    return None;
  }

  // Polynomial long division with integer/rational coefficients
  let mut remainder = num_coeffs.clone();
  let mut quotient = vec![0i128; (num_deg - den_deg + 1) as usize];
  let lead_den = *den_coeffs.last()?;

  if lead_den == 0 {
    return None;
  }

  for i in (0..quotient.len()).rev() {
    let rem_idx = i + den_coeffs.len() - 1;
    if rem_idx >= remainder.len() {
      continue;
    }
    if remainder[rem_idx] % lead_den != 0 {
      return None; // Not exactly divisible with integers
    }
    let q = remainder[rem_idx] / lead_den;
    quotient[i] = q;
    for j in 0..den_coeffs.len() {
      remainder[i + j] -= q * den_coeffs[j];
    }
  }

  // Check remainder is zero
  if remainder.iter().any(|&c| c != 0) {
    return None;
  }

  // Build quotient polynomial
  Some(coeffs_to_expr(&quotient, var))
}

/// Extract integer polynomial coefficients from expr, indexed by power.
/// coeffs[i] = coefficient of var^i
pub fn extract_poly_coeffs(expr: &Expr, var: &str) -> Option<Vec<i128>> {
  let terms = collect_additive_terms(expr);
  let mut max_pow: i128 = 0;
  let mut term_data: Vec<(i128, i128)> = Vec::new(); // (power, integer_coeff)

  for term in &terms {
    let (power, coeff) = term_var_power_and_coeff(term, var);
    if power < 0 {
      return None; // non-polynomial term
    }
    let int_coeff = match &simplify(coeff) {
      Expr::Integer(n) => *n,
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => {
        if let Expr::Integer(n) = operand.as_ref() {
          -n
        } else {
          return None;
        }
      }
      _ => return None, // non-integer coefficient
    };
    max_pow = max_pow.max(power);
    term_data.push((power, int_coeff));
  }

  let mut coeffs = vec![0i128; (max_pow + 1) as usize];
  for (power, c) in term_data {
    coeffs[power as usize] += c;
  }

  Some(coeffs)
}

/// Build a polynomial expression from integer coefficients.
/// coeffs[i] = coefficient of var^i
pub fn coeffs_to_expr(coeffs: &[i128], var: &str) -> Expr {
  let mut terms: Vec<Expr> = Vec::new();

  for (i, &c) in coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let var_part = if i == 0 {
      None
    } else if i == 1 {
      Some(Expr::Identifier(var.to_string()))
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier(var.to_string())),
        right: Box::new(Expr::Integer(i as i128)),
      })
    };

    let term = match (c, var_part) {
      (c, None) => Expr::Integer(c),
      (1, Some(v)) => v,
      (-1, Some(v)) => negate_term(&v),
      (c, Some(v)) => Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(c)),
        right: Box::new(v),
      },
    };
    terms.push(term);
  }

  if terms.is_empty() {
    Expr::Integer(0)
  } else {
    build_sum(terms)
  }
}

/// Group additive terms by their denominator and combine like-denominator groups.
/// E.g. a/x + b/x + c/y → (a + b)/x + c/y
fn combine_like_denominator_terms(expr: &Expr) -> Expr {
  let terms = collect_additive_terms(expr);
  if terms.len() < 2 {
    return expr.clone();
  }

  // Extract (numerator, denominator) for each term
  let fractions: Vec<(Expr, Expr)> =
    terms.iter().map(super::together::extract_num_den).collect();

  // Group terms by denominator string
  let mut groups: Vec<(String, Expr, Vec<Expr>)> = Vec::new(); // (den_str, den_expr, numerators)
  for (num, den) in &fractions {
    let den_str = expr_to_string(den);
    if let Some(group) = groups.iter_mut().find(|(ds, _, _)| *ds == den_str) {
      group.2.push(num.clone());
    } else {
      groups.push((den_str, den.clone(), vec![num.clone()]));
    }
  }

  // If no group has >1 term, no combining happened
  if groups.iter().all(|(_, _, nums)| nums.len() <= 1) {
    return expr.clone();
  }

  // Build combined terms
  let mut result_terms: Vec<Expr> = Vec::new();
  for (_, den, nums) in groups {
    let combined_num = if nums.len() == 1 {
      nums.into_iter().next().unwrap()
    } else {
      expand_and_combine(&build_sum(nums))
    };
    if matches!(&den, Expr::Integer(1)) {
      result_terms.push(combined_num);
    } else {
      result_terms.push(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(combined_num),
        right: Box::new(den),
      });
    }
  }

  if result_terms.len() == 1 {
    result_terms.remove(0)
  } else {
    build_sum(result_terms)
  }
}

/// Try Together + factor + cancel to simplify a sum of fractions.
fn try_together_simplify(expr: &Expr) -> Expr {
  let combined = together_expr(expr);

  // If result is a fraction, try to factor and cancel
  match &combined {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let num = *left.clone();
      let den = *right.clone();

      // Factor the numerator aggressively: extract numeric GCD, then symbolic common factors
      let factored_num = factor_numerator_fully(&num);

      // Cancel common factors between numerator and denominator
      let result = cancel_symbolic_factors(&factored_num, &den);
      // Evaluate to canonicalize (flatten nested Times, distribute powers, sort factors)
      if let Ok(canonical) = crate::evaluator::evaluate_expr_to_expr(&result) {
        canonical
      } else {
        result
      }
    }
    _ => combined,
  }
}

/// Factor a numerator fully: extract numeric GCD, then factor common symbolic terms.
fn factor_numerator_fully(num: &Expr) -> Expr {
  // First try FactorTerms (numeric GCD)
  let after_numeric = if let Ok(f) =
    crate::functions::polynomial_ast::factor_terms_ast(&[num.clone()])
  {
    f
  } else {
    num.clone()
  };

  // If FactorTerms produced coeff * inner_sum, try factor_common_symbolic on the inner sum
  let factored = extract_and_factor_inner_sum(&after_numeric);

  // Also try factor_common_symbolic directly on the original (in case FactorTerms didn't help)
  let terms = collect_additive_terms(num);
  if terms.len() >= 2
    && let Some(f) = factor_common_symbolic(num, &terms)
  {
    // Pick the more factored form (fewer leaves)
    if leaf_count(&f) <= leaf_count(&factored) {
      return f;
    }
  }

  factored
}

/// If expr is a product containing a sum factor, try factor_common_symbolic on that sum.
/// E.g. 2*(a^4*k*q + a^4*k*q*(1+s)^(15/4)) → 2*a^4*k*q*(1 + (1+s)^(15/4))
fn extract_and_factor_inner_sum(expr: &Expr) -> Expr {
  let factors = collect_multiplicative_factors(expr);
  if factors.len() < 2 {
    return expr.clone();
  }

  // Find a factor that is a sum (has multiple additive terms)
  for (idx, f) in factors.iter().enumerate() {
    let terms = collect_additive_terms(f);
    if terms.len() >= 2 {
      // Try to factor out common symbolic factors from this sum
      if let Some(factored_sum) = factor_common_symbolic(f, &terms) {
        // Rebuild the product with the factored sum replacing the original
        let mut new_factors: Vec<Expr> = Vec::new();
        for (j, g) in factors.iter().enumerate() {
          if j == idx {
            new_factors.push(factored_sum.clone());
          } else {
            new_factors.push(g.clone());
          }
        }
        return build_product(new_factors);
      }
    }
  }

  expr.clone()
}

/// Combine Abs quotients: Abs[a]/Abs[b] → Abs[a/b], Abs[a]*Abs[b] → Abs[a*b]
/// Then expand the inner expression so e.g. Abs[1+x^3]/Abs[x] → Abs[x^(-1)+x^2]
fn simplify_abs_products(expr: &Expr) -> Expr {
  let factors = collect_multiplicative_factors(expr);
  if factors.len() < 2 {
    return expr.clone();
  }

  let mut abs_numerators: Vec<Expr> = Vec::new();
  let mut abs_denominators: Vec<Expr> = Vec::new();
  let mut other_factors: Vec<Expr> = Vec::new();

  for factor in &factors {
    match factor {
      // Abs[x] in the numerator
      Expr::FunctionCall { name, args } if name == "Abs" && args.len() == 1 => {
        abs_numerators.push(args[0].clone());
      }
      // Power[Abs[x], -1] i.e. 1/Abs[x]
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if matches!(right.as_ref(), Expr::Integer(-1))
        || matches!(
          right.as_ref(),
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand
          } if matches!(operand.as_ref(), Expr::Integer(1))
        ) =>
      {
        if let Expr::FunctionCall { name, args } = left.as_ref() {
          if name == "Abs" && args.len() == 1 {
            abs_denominators.push(args[0].clone());
          } else {
            other_factors.push(factor.clone());
          }
        } else {
          other_factors.push(factor.clone());
        }
      }
      // FunctionCall Power[Abs[x], -1]
      Expr::FunctionCall { name, args }
        if name == "Power"
          && args.len() == 2
          && matches!(&args[1], Expr::Integer(-1)) =>
      {
        if let Expr::FunctionCall {
          name: inner_name,
          args: inner_args,
        } = &args[0]
        {
          if inner_name == "Abs" && inner_args.len() == 1 {
            abs_denominators.push(inner_args[0].clone());
          } else {
            other_factors.push(factor.clone());
          }
        } else {
          other_factors.push(factor.clone());
        }
      }
      _ => {
        other_factors.push(factor.clone());
      }
    }
  }

  // Only combine if we have at least 2 Abs factors (numerator + denominator)
  if abs_numerators.len() + abs_denominators.len() < 2 {
    return expr.clone();
  }

  // Build inner numerator
  let inner_num = if abs_numerators.is_empty() {
    Expr::Integer(1)
  } else {
    build_product(abs_numerators)
  };

  // Build inner expression: multiply numerators with Power[denominator, -1]
  let mut inner_parts = vec![inner_num];
  for d in abs_denominators {
    inner_parts.push(Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(d),
      right: Box::new(Expr::Integer(-1)),
    });
  }
  let inner = build_product(inner_parts);

  // Expand the inner expression so (1+x^3)/x becomes x^(-1)+x^2
  let inner_expanded = expand_and_combine(&inner);

  let combined_abs = Expr::FunctionCall {
    name: "Abs".to_string(),
    args: vec![inner_expanded],
  };

  if other_factors.is_empty() {
    combined_abs
  } else {
    other_factors.push(combined_abs);
    build_product(other_factors)
  }
}

/// Count the complexity of an expression (leaf nodes + internal nodes).
/// Used as a metric for choosing the simplest form.
fn leaf_count(expr: &Expr) -> usize {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::Constant(_)
    | Expr::Identifier(_) => 1,
    Expr::BinaryOp { left, right, .. } => {
      1 + leaf_count(left) + leaf_count(right)
    }
    Expr::UnaryOp { operand, .. } => 1 + leaf_count(operand),
    Expr::FunctionCall { args, .. } => {
      1 + args.iter().map(leaf_count).sum::<usize>()
    }
    Expr::List(items) => items.iter().map(leaf_count).sum::<usize>().max(1),
    _ => 1,
  }
}

/// Factor out common symbolic factors from additive terms.
/// e.g., `2*a^2 - 2*a^2*Sin[theta]` → `2*a^2*(1 - Sin[theta])`
fn factor_common_symbolic(_expr: &Expr, terms: &[Expr]) -> Option<Expr> {
  if terms.len() < 2 {
    return None;
  }

  // For each term, get the set of multiplicative factor strings
  let term_factor_sets: Vec<Vec<(String, Expr)>> = terms
    .iter()
    .map(|t| {
      let factors = collect_multiplicative_factors(t);
      // Flatten negation but track it
      let mut result: Vec<(String, Expr)> = Vec::new();
      for f in &factors {
        match f {
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } => {
            let inner = collect_multiplicative_factors(operand);
            result.push(("-1".to_string(), Expr::Integer(-1)));
            for i in &inner {
              result.push((expr_to_string(i), i.clone()));
            }
          }
          _ => result.push((expr_to_string(f), f.clone())),
        }
      }
      result
    })
    .collect();

  // Find factors common to ALL terms (by string comparison)
  // Exclude integer factors (handled by factor_terms_numeric already)
  let first_factors: Vec<(String, Expr)> = term_factor_sets[0]
    .iter()
    .filter(|(s, _)| {
      // Skip pure integers and -1
      s != "-1" && s.parse::<i128>().is_err()
    })
    .cloned()
    .collect();

  let mut common_factors: Vec<(String, Expr)> = Vec::new();
  for (s, e) in &first_factors {
    if term_factor_sets[1..]
      .iter()
      .all(|tfs| tfs.iter().any(|(ts, _)| ts == s))
    {
      // Check for duplicates in common_factors
      if !common_factors.iter().any(|(cs, _)| cs == s) {
        common_factors.push((s.clone(), e.clone()));
      }
    }
  }

  if common_factors.is_empty() {
    return None;
  }

  // Remove common factors from each term
  let mut new_terms: Vec<Expr> = Vec::new();
  for tfs in &term_factor_sets {
    let mut remaining: Vec<Expr> = Vec::new();
    let mut used: Vec<bool> = vec![false; common_factors.len()];

    for (s, e) in tfs {
      let mut is_common = false;
      for (ci, (cs, _)) in common_factors.iter().enumerate() {
        if !used[ci] && s == cs {
          used[ci] = true;
          is_common = true;
          break;
        }
      }
      if !is_common {
        remaining.push(e.clone());
      }
    }

    if remaining.is_empty() {
      new_terms.push(Expr::Integer(1));
    } else if remaining.len() == 1 {
      new_terms.push(remaining.remove(0));
    } else {
      new_terms.push(build_product(remaining));
    }
  }

  // Build result: common_factor * (sum of new_terms)
  let common_expr = if common_factors.len() == 1 {
    common_factors[0].1.clone()
  } else {
    build_product(common_factors.into_iter().map(|(_, e)| e).collect())
  };

  let sum = expand_and_combine(&build_sum(new_terms));

  // Also try to factor numeric GCD from the inner sum
  let inner = if let Ok(factored) =
    crate::functions::polynomial_ast::factor_terms_ast(&[sum.clone()])
  {
    factored
  } else {
    sum
  };

  // Build result with proper ordering: numeric_coeff * symbolic_factors * (inner_sum)
  // Extract numeric factor from inner if present
  let (num_factor, remainder) = match &inner {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(_)) => {
      (Some(*left.clone()), *right.clone())
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      // -expr → factor of -1
      match operand.as_ref() {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left,
          right,
        } if matches!(left.as_ref(), Expr::Integer(n) if *n > 0) => {
          if let Expr::Integer(n) = left.as_ref() {
            (Some(Expr::Integer(-n)), *right.clone())
          } else {
            (Some(Expr::Integer(-1)), *operand.clone())
          }
        }
        _ => (Some(Expr::Integer(-1)), *operand.clone()),
      }
    }
    _ => (None, inner),
  };

  // Check if we should negate the coefficient and inner sum to match canonical form.
  // Wolfram convention: the first symbolic (non-constant) term in the inner sum
  // should have a positive coefficient. If not, negate both.
  let (final_num, final_remainder) = {
    let inner_terms = collect_additive_terms(&remainder);
    // Find first non-constant term
    let first_symbolic =
      inner_terms.iter().find(|t| !matches!(t, Expr::Integer(_)));
    let should_negate = if let Some(sym_term) = first_symbolic {
      // Check if it has a negative leading coefficient
      matches!(
        sym_term,
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          ..
        }
      ) || matches!(sym_term, Expr::BinaryOp { op: BinaryOperator::Times, left, .. }
            if matches!(left.as_ref(), Expr::Integer(n) if *n < 0)
              || matches!(left.as_ref(), Expr::UnaryOp { op: UnaryOperator::Minus, .. }))
    } else {
      false
    };
    if should_negate {
      let negated_remainder = expand_and_combine(&negate_term(&remainder));
      let negated_num = num_factor
        .map(|nf| match nf {
          Expr::Integer(n) => Expr::Integer(-n),
          _ => negate_term(&nf),
        })
        .unwrap_or(Expr::Integer(-1));
      (Some(negated_num), negated_remainder)
    } else {
      (num_factor, remainder)
    }
  };

  let mut factors: Vec<Expr> = Vec::new();
  if let Some(nf) = final_num {
    factors.push(nf);
  }
  factors.push(common_expr);
  factors.push(final_remainder);

  Some(build_product(factors))
}

/// Factor out the minimum power of a common base from additive terms.
/// E.g. (1+s)^(-3/2) + (1+s)^(9/4) → (1+s)^(-3/2) * (1 + (1+s)^(15/4))
///
/// Works by:
/// 1. Decomposing each additive term into (coefficient, base, rational_exponent) triples
/// 2. Finding a base that appears in all terms with rational exponents
/// 3. Factoring out the minimum exponent
fn factor_common_power_base(terms: &[Expr]) -> Option<Expr> {
  if terms.len() < 2 {
    return None;
  }

  // For each term, extract the multiplicative factors and find power-like bases
  // A term like k*q*(1+s)^(-3/2)/(2*a^4) has factors: [k, q, (1+s)^(-3/2), Power[a^4,-1], Rational[1,2]]
  // We look for bases that appear as powers across all terms

  // Extract (coefficient_factors, base_string, rational_exponent) for each term
  struct PowerInfo {
    base_str: String,
    base: Expr,
    numer: i128,
    denom: i128,
  }

  fn extract_rational_exp(exp: &Expr) -> Option<(i128, i128)> {
    match exp {
      Expr::Integer(n) => Some((*n, 1)),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some((*n, *d))
        } else {
          None
        }
      }
      // Handle Times[-1, Rational[p, q]] → (-p, q)
      Expr::FunctionCall { name, args }
        if name == "Times"
          && args.len() == 2
          && matches!(&args[0], Expr::Integer(-1))
          && matches!(&args[1], Expr::FunctionCall { name: rn, args: ra }
            if rn == "Rational" && ra.len() == 2) =>
      {
        if let Expr::FunctionCall { args: ra, .. } = &args[1] {
          if let (Expr::Integer(n), Expr::Integer(d)) = (&ra[0], &ra[1]) {
            Some((-n, *d))
          } else {
            None
          }
        } else {
          None
        }
      }
      // Handle BinaryOp representations
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => {
        if let (Expr::Integer(n), Expr::Integer(d)) =
          (left.as_ref(), right.as_ref())
        {
          Some((*n, *d))
        } else {
          None
        }
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => extract_rational_exp(operand).map(|(n, d)| (-n, d)),
      _ => None,
    }
  }

  // For each term, collect multiplicative factors and extract power bases
  fn get_power_bases(factors: &[Expr]) -> Vec<PowerInfo> {
    let mut result = Vec::new();
    for f in factors {
      let (base, exp) = extract_base_and_exp(f);
      // Skip atoms — we only want compound bases like (1+s)
      if matches!(
        &base,
        Expr::Integer(_) | Expr::Constant(_) | Expr::Identifier(_)
      ) {
        continue;
      }
      if let Some((n, d)) = extract_rational_exp(&exp) {
        let bs = expr_to_string(&base);
        result.push(PowerInfo {
          base_str: bs,
          base,
          numer: n,
          denom: d,
        });
      }
    }
    result
  }

  let term_factors: Vec<Vec<Expr>> =
    terms.iter().map(collect_multiplicative_factors).collect();
  let term_powers: Vec<Vec<PowerInfo>> =
    term_factors.iter().map(|f| get_power_bases(f)).collect();

  // Find bases that appear in ALL terms
  if term_powers.is_empty() || term_powers[0].is_empty() {
    return None;
  }

  for candidate in &term_powers[0] {
    let bs = &candidate.base_str;
    // Check if this base appears in all other terms
    let in_all = term_powers[1..]
      .iter()
      .all(|tp| tp.iter().any(|p| &p.base_str == bs));
    if !in_all {
      continue;
    }

    // Find the minimum exponent across all terms
    let mut min_n = candidate.numer;
    let mut min_d = candidate.denom;
    for tp in &term_powers[1..] {
      for p in tp {
        if &p.base_str == bs {
          // Compare p.numer/p.denom with min_n/min_d
          if p.numer * min_d < min_n * p.denom {
            min_n = p.numer;
            min_d = p.denom;
          }
          break;
        }
      }
    }

    // Factor out base^(min_n/min_d) from each term
    let mut new_terms = Vec::new();
    for (i, _term) in terms.iter().enumerate() {
      // Find the exponent of this base in this term
      let pi = term_powers[i].iter().find(|p| &p.base_str == bs).unwrap();
      // Subtract min exponent: new_exp = pi.exp - min_exp
      let diff_n = pi.numer * min_d - min_n * pi.denom;
      let diff_d = pi.denom * min_d;
      // Simplify the fraction
      let g = crate::functions::math_ast::gcd(diff_n, diff_d);
      let sn = diff_n / g;
      let sd = diff_d / g;

      // Remove the old power factor and replace with the new exponent
      let factors = &term_factors[i];
      let mut new_factors: Vec<Expr> = Vec::new();
      let mut replaced = false;
      for f in factors {
        let (fb, _fe) = extract_base_and_exp(f);
        if !replaced && expr_to_string(&fb) == *bs {
          replaced = true;
          if sn == 0 {
            // base^0 = 1, skip it
          } else if sn == 1 && sd == 1 {
            new_factors.push(candidate.base.clone());
          } else if sd == 1 {
            new_factors.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(candidate.base.clone()),
              right: Box::new(Expr::Integer(sn)),
            });
          } else {
            new_factors.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(candidate.base.clone()),
              right: Box::new(crate::functions::math_ast::make_rational(
                sn, sd,
              )),
            });
          }
        } else {
          new_factors.push(f.clone());
        }
      }
      if new_factors.is_empty() {
        new_terms.push(Expr::Integer(1));
      } else if new_factors.len() == 1 {
        new_terms.push(new_factors.remove(0));
      } else {
        new_terms.push(build_product(new_factors));
      }
    }

    // Build: base^(min_exp) * (sum of new_terms)
    let min_power = if min_n == 1 && min_d == 1 {
      candidate.base.clone()
    } else if min_d == 1 {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(candidate.base.clone()),
        right: Box::new(Expr::Integer(min_n)),
      }
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(candidate.base.clone()),
        right: Box::new(crate::functions::math_ast::make_rational(
          min_n, min_d,
        )),
      }
    };
    let inner_sum = build_sum(new_terms);
    // Evaluate the inner sum to simplify
    let simplified_sum =
      if let Ok(s) = crate::evaluator::evaluate_expr_to_expr(&inner_sum) {
        s
      } else {
        inner_sum
      };
    let result = build_product(vec![min_power, simplified_sum]);
    // Evaluate to get canonical form
    if let Ok(r) = crate::evaluator::evaluate_expr_to_expr(&result) {
      return Some(r);
    }
    return Some(result);
  }

  None
}
