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

  // Single argument: use $Assumptions if available
  if args.len() == 1 {
    let assumptions_str = crate::ENV
      .with(|e| {
        e.borrow().get("$Assumptions").map(|sv| match sv {
          crate::StoredValue::Raw(s) => s.clone(),
          crate::StoredValue::ExprVal(e) => expr_to_string(e),
          _ => "True".to_string(),
        })
      })
      .unwrap_or_else(|| "True".to_string());
    if assumptions_str != "True" {
      // Parse the assumption string back to an Expr
      if let Ok(parsed) = crate::syntax::string_to_expr(&assumptions_str)
        && let Ok(assumption_expr) =
          crate::evaluator::evaluate_expr_to_expr(&parsed)
      {
        let info = extract_assumption_info(&assumption_expr);
        let result = refine_expr(&args[0], &info, &assumption_expr);
        return Ok(result);
      }
    }
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
  positive_vars: Vec<String>, // x > 0 (strictly positive)
  nonnegative_vars: Vec<String>, // x >= 0
  negative_vars: Vec<String>, // x < 0 (strictly negative)
  real_vars: Vec<String>,
  integer_vars: Vec<String>,
  /// Raw assumptions preserved for advanced reasoning
  raw_assumptions: Vec<Expr>,
}

/// Extract all assumption information from the assumption expression.
fn extract_assumption_info(assumption: &Expr) -> AssumptionInfo {
  let mut info = AssumptionInfo {
    positive_vars: Vec::new(),
    nonnegative_vars: Vec::new(),
    negative_vars: Vec::new(),
    real_vars: Vec::new(),
    integer_vars: Vec::new(),
    raw_assumptions: Vec::new(),
  };
  extract_assumptions_inner(assumption, &mut info);
  // Positive vars are also non-negative
  for v in &info.positive_vars {
    if !info.nonnegative_vars.contains(v) {
      info.nonnegative_vars.push(v.clone());
    }
  }
  // Positive/negative/nonnegative vars are also real
  for v in info
    .positive_vars
    .iter()
    .chain(info.negative_vars.iter())
    .chain(info.nonnegative_vars.iter())
    .cloned()
    .collect::<Vec<_>>()
  {
    if !info.real_vars.contains(&v) {
      info.real_vars.push(v);
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
  // Store raw assumption for advanced reasoning
  info.raw_assumptions.push(assumption.clone());

  match assumption {
    Expr::Comparison {
      operands,
      operators,
    } => {
      if operands.len() == 2 && operators.len() == 1 {
        let op = &operators[0];
        let left = &operands[0];
        let right = &operands[1];

        // x > c where c >= 0 → positive (strictly)
        if matches!(op, crate::syntax::ComparisonOp::Greater)
          && let Expr::Identifier(name) = left
          && is_nonnegative_constant(right)
        {
          info.positive_vars.push(name.clone());
        }
        // x >= c where c > 0 → positive; where c == 0 → nonnegative
        if matches!(op, crate::syntax::ComparisonOp::GreaterEqual)
          && let Expr::Identifier(name) = left
          && is_nonnegative_constant(right)
        {
          if is_positive_constant(right) {
            info.positive_vars.push(name.clone());
          } else {
            info.nonnegative_vars.push(name.clone());
          }
        }

        // c < x where c >= 0 → positive
        if matches!(op, crate::syntax::ComparisonOp::Less)
          && let Expr::Identifier(name) = right
          && is_nonnegative_constant(left)
        {
          info.positive_vars.push(name.clone());
        }
        // c <= x where c > 0 → positive; where c == 0 → nonnegative
        if matches!(op, crate::syntax::ComparisonOp::LessEqual)
          && let Expr::Identifier(name) = right
          && is_nonnegative_constant(left)
        {
          if is_positive_constant(left) {
            info.positive_vars.push(name.clone());
          } else {
            info.nonnegative_vars.push(name.clone());
          }
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
    // Element[x, domain] or Element[Alternatives[a, b, ...], domain]
    Expr::FunctionCall { name, args }
      if name == "Element" && args.len() == 2 =>
    {
      let vars = extract_element_vars(&args[0]);
      if let Expr::Identifier(domain) = &args[1] {
        for var_name in vars {
          match domain.as_str() {
            "Reals" => {
              if !info.real_vars.contains(&var_name) {
                info.real_vars.push(var_name);
              }
            }
            "Integers" | "Primes" => {
              if !info.integer_vars.contains(&var_name) {
                info.integer_vars.push(var_name.clone());
              }
              if !info.real_vars.contains(&var_name) {
                info.real_vars.push(var_name);
              }
            }
            "Rationals" | "Algebraics" => {
              if !info.real_vars.contains(&var_name) {
                info.real_vars.push(var_name);
              }
            }
            "PositiveReals" | "PositiveIntegers" | "PositiveRationals" => {
              info.positive_vars.push(var_name.clone());
              if !info.real_vars.contains(&var_name) {
                info.real_vars.push(var_name);
              }
            }
            "NonNegativeReals"
            | "NonNegativeIntegers"
            | "NonNegativeRationals" => {
              info.nonnegative_vars.push(var_name.clone());
              if !info.real_vars.contains(&var_name) {
                info.real_vars.push(var_name);
              }
            }
            _ => {}
          }
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

/// Extract variable names from an Element first argument.
/// Handles: single Identifier, Alternatives[a, b, ...] (BinaryOp or FunctionCall form)
fn extract_element_vars(expr: &Expr) -> Vec<String> {
  match expr {
    Expr::Identifier(name) => vec![name.clone()],
    // Alternatives as BinaryOp: a | b
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Alternatives,
      left,
      right,
    } => {
      let mut vars = extract_element_vars(left);
      vars.extend(extract_element_vars(right));
      vars
    }
    // Alternatives as FunctionCall: Alternatives[a, b, ...]
    Expr::FunctionCall { name, args } if name == "Alternatives" => {
      args.iter().flat_map(extract_element_vars).collect()
    }
    _ => vec![],
  }
}

/// Check if an expression is a strictly positive constant.
fn is_positive_constant(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(n) => *n > 0,
    Expr::BigInteger(n) => *n > num_bigint::BigInt::from(0),
    Expr::Real(f) => *f > 0.0,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(a), Expr::Integer(b)) => {
          (*a > 0 && *b > 0) || (*a < 0 && *b < 0)
        }
        _ => false,
      }
    }
    _ => false,
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

    // For compound expressions like x - y > 0:
    // Check if we can determine the sign of the LHS expression
    if matches!(right, Expr::Integer(0)) {
      match op {
        crate::syntax::ComparisonOp::Greater => {
          if is_provably_positive_under_assumptions(left, info) {
            return Some(true);
          }
        }
        crate::syntax::ComparisonOp::GreaterEqual => {
          if is_provably_nonneg_under_assumptions(left, info) {
            return Some(true);
          }
        }
        _ => {}
      }
    }
  }

  None
}

/// Check if an expression is provably positive given assumption info.
/// Handles sums where we know signs: nonneg + positive = positive, etc.
fn is_provably_positive_under_assumptions(
  expr: &Expr,
  info: &AssumptionInfo,
) -> bool {
  let terms = collect_additive_terms(expr);
  let mut has_strictly_positive = false;
  for term in &terms {
    match get_sign_under_assumptions(term, info) {
      Some(1) => has_strictly_positive = true,
      Some(0) => {} // nonnegative, fine
      Some(-1) => return false,
      None => return false,
      _ => return false,
    }
  }
  has_strictly_positive
}

/// Check if an expression is provably non-negative given assumption info.
fn is_provably_nonneg_under_assumptions(
  expr: &Expr,
  info: &AssumptionInfo,
) -> bool {
  let terms = collect_additive_terms(expr);
  for term in &terms {
    match get_sign_under_assumptions(term, info) {
      Some(s) if s >= 0 => {}
      _ => return false,
    }
  }
  true
}

/// Get sign info: 1 = strictly positive, 0 = nonnegative, -1 = negative, None = unknown.
fn get_sign_under_assumptions(
  expr: &Expr,
  info: &AssumptionInfo,
) -> Option<i8> {
  match expr {
    Expr::Integer(n) => {
      if *n > 0 {
        Some(1)
      } else if *n == 0 {
        Some(0)
      } else {
        Some(-1)
      }
    }
    Expr::Identifier(name) => {
      if info.positive_vars.contains(name) {
        Some(1)
      } else if info.nonnegative_vars.contains(name) {
        Some(0)
      } else if info.negative_vars.contains(name) {
        Some(-1)
      } else {
        None
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => get_sign_under_assumptions(operand, info).map(|s| -s),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let ls = get_sign_under_assumptions(left, info)?;
      let rs = get_sign_under_assumptions(right, info)?;
      Some(ls * rs)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut result: i8 = 1;
      for a in args {
        result *= get_sign_under_assumptions(a, info)?;
      }
      Some(result)
    }
    _ => None,
  }
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
      // Try algebraic reasoning for equations/inequalities
      if let Some(result) = check_algebraic_comparison(expr, info, assumption) {
        return Expr::Identifier(
          if result { "True" } else { "False" }.to_string(),
        );
      }
      expr.clone()
    }

    // (var^n)^(1/m) → var^(n/m) when var >= 0 and n divisible by m
    // Also handles Sqrt[var^2] as special case (m=2, n=2)
    // For var < 0: only simplifies when n is even
    // For var ∈ Reals: (x^2)^(1/2) → Abs[x]
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
        if info.positive_vars.contains(var_name)
          || info.nonnegative_vars.contains(var_name)
        {
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
        // For real vars with unknown sign and even n: (x^n)^(1/m) → Abs[x]^reduced
        if info.real_vars.contains(var_name) && n % 2 == 0 {
          if reduced == 1 {
            return Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![base.as_ref().clone()],
            };
          }
          return Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![base.as_ref().clone()],
            }),
            right: Box::new(Expr::Integer(reduced)),
          };
        }
      }
      // Handle product base: (x^2 * y^2 * ...)^(1/m) → refine each factor
      if let Some(result) = refine_product_root(outer_base, m, info, assumption)
      {
        return result;
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

    // Sign[expr] → 1 when expr > 0, -1 when expr < 0
    Expr::FunctionCall { name, args } if name == "Sign" && args.len() == 1 => {
      if let Expr::Identifier(var_name) = &args[0] {
        if info.positive_vars.contains(var_name) {
          return Expr::Integer(1);
        }
        if info.negative_vars.contains(var_name) {
          return Expr::Integer(-1);
        }
      }
      // Check if expression is provably positive (e.g., x^2 - xy + y^2 + 1 with x,y real)
      if is_provably_positive(&args[0], info) {
        return Expr::Integer(1);
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

    // Re[expr] → simplify when all vars are real
    Expr::FunctionCall { name, args } if name == "Re" && args.len() == 1 => {
      if let Expr::Identifier(var_name) = &args[0]
        && info.real_vars.contains(var_name)
      {
        return args[0].clone();
      }
      // Re[a + b*I] with a, b ∈ Reals → a
      if let Some(real_part) = extract_real_part(&args[0], info) {
        return real_part;
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      // Try again after refining
      if let Some(real_part) = extract_real_part(&refined_arg, info) {
        return real_part;
      }
      Expr::FunctionCall {
        name: "Re".to_string(),
        args: vec![refined_arg],
      }
    }

    // Im[expr] → simplify when all vars are real
    Expr::FunctionCall { name, args } if name == "Im" && args.len() == 1 => {
      if let Expr::Identifier(var_name) = &args[0]
        && info.real_vars.contains(var_name)
      {
        return Expr::Integer(0);
      }
      // Im[a + b*I] with a, b ∈ Reals → b
      if let Some(imag_part) = extract_imag_part(&args[0], info) {
        return imag_part;
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      if let Some(imag_part) = extract_imag_part(&refined_arg, info) {
        return imag_part;
      }
      Expr::FunctionCall {
        name: "Im".to_string(),
        args: vec![refined_arg],
      }
    }

    // Floor[expr] → expr when expr is known integer
    Expr::FunctionCall { name, args } if name == "Floor" && args.len() == 1 => {
      if is_known_integer(&args[0], info) {
        let refined = refine_expr(&args[0], info, assumption);
        return refined;
      }
      // Check if we can determine bounds: Floor[x] with a < x <= b where a,b integers
      if let Some(val) = refine_floor_ceiling(&args[0], info, assumption, true)
      {
        return val;
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      if is_known_integer(&refined_arg, info) {
        return refined_arg;
      }
      Expr::FunctionCall {
        name: "Floor".to_string(),
        args: vec![refined_arg],
      }
    }

    // Ceiling[expr] → expr when expr is known integer; or evaluate with bounds
    Expr::FunctionCall { name, args }
      if name == "Ceiling" && args.len() == 1 =>
    {
      if is_known_integer(&args[0], info) {
        let refined = refine_expr(&args[0], info, assumption);
        return refined;
      }
      if let Some(val) = refine_floor_ceiling(&args[0], info, assumption, false)
      {
        return val;
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      if is_known_integer(&refined_arg, info) {
        return refined_arg;
      }
      Expr::FunctionCall {
        name: "Ceiling".to_string(),
        args: vec![refined_arg],
      }
    }

    // Sin[k*Pi] → 0 when k ∈ Integers
    Expr::FunctionCall { name, args } if name == "Sin" && args.len() == 1 => {
      if is_integer_multiple_of_pi(&args[0], info) {
        return Expr::Integer(0);
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      Expr::FunctionCall {
        name: "Sin".to_string(),
        args: vec![refined_arg],
      }
    }

    // Cos[x + k*Pi] → (-1)^k * Cos[x] when k ∈ Integers
    Expr::FunctionCall { name, args } if name == "Cos" && args.len() == 1 => {
      if let Some((non_pi_part, k_expr)) = split_integer_pi_part(&args[0], info)
      {
        // Cos[x + k*Pi] = (-1)^k * Cos[x]
        let cos_x = if matches!(&non_pi_part, Expr::Integer(0)) {
          Expr::Integer(1)
        } else {
          Expr::FunctionCall {
            name: "Cos".to_string(),
            args: vec![non_pi_part],
          }
        };
        return Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(k_expr),
          }),
          right: Box::new(cos_x),
        };
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      Expr::FunctionCall {
        name: "Cos".to_string(),
        args: vec![refined_arg],
      }
    }

    // ArcTan[Tan[x]] → x when -Pi/2 < Re[x] < Pi/2
    Expr::FunctionCall { name, args }
      if name == "ArcTan" && args.len() == 1 =>
    {
      if let Expr::FunctionCall {
        name: inner_name,
        args: inner_args,
      } = &args[0]
        && inner_name == "Tan"
        && inner_args.len() == 1
      {
        // Check if -Pi/2 < Re[x] < Pi/2 is in assumptions
        if is_in_arctan_range(&inner_args[0], info, assumption) {
          return inner_args[0].clone();
        }
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      Expr::FunctionCall {
        name: "ArcTan".to_string(),
        args: vec![refined_arg],
      }
    }

    // Log[x] with x < 0 → I*Pi + Log[-x]
    // Log[x^p] with -1 < p < 1 → p*Log[x]
    Expr::FunctionCall { name, args } if name == "Log" && args.len() == 1 => {
      if let Some(result) = refine_log(&args[0], info, assumption) {
        return result;
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![refined_arg],
      }
    }

    // Element[expr, domain] → True/False under assumptions
    Expr::FunctionCall { name, args }
      if name == "Element" && args.len() == 2 =>
    {
      if let Some(result) = refine_element(&args[0], &args[1], info) {
        return result;
      }
      let refined_args: Vec<Expr> = args
        .iter()
        .map(|a| refine_expr(a, info, assumption))
        .collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: refined_args,
      }
    }

    // FractionalPart[a] under assumptions
    Expr::FunctionCall { name, args }
      if name == "FractionalPart" && args.len() == 1 =>
    {
      if let Some(result) = refine_fractional_part(&args[0], info, assumption) {
        return result;
      }
      let refined_arg = refine_expr(&args[0], info, assumption);
      Expr::FunctionCall {
        name: "FractionalPart".to_string(),
        args: vec![refined_arg],
      }
    }

    // Mod[a, m] under assumptions
    Expr::FunctionCall { name, args } if name == "Mod" && args.len() == 2 => {
      if let Some(result) = refine_mod(&args[0], &args[1], info, assumption) {
        return result;
      }
      let refined_args: Vec<Expr> = args
        .iter()
        .map(|a| refine_expr(a, info, assumption))
        .collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: refined_args,
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

    // Times[...] as FunctionCall: check for a^p * b^p → (a*b)^p
    Expr::FunctionCall { name, args } if name == "Times" => {
      let refined: Vec<Expr> = args
        .iter()
        .map(|a| refine_expr(a, info, assumption))
        .collect();
      // Try pairwise combining of same-exponent powers with positive bases
      if refined.len() == 2
        && let Some(result) =
          try_combine_power_product(&refined[0], &refined[1], info)
      {
        return result;
      }
      Expr::FunctionCall {
        name: name.clone(),
        args: refined,
      }
    }

    // Power[...] as FunctionCall: handle nested power simplification
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Some(result) =
        try_simplify_nested_power(&args[0], &args[1], info, assumption)
      {
        return result;
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

    // Recurse into binary ops, with special handling for products and powers
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let l = refine_expr(left, info, assumption);
      let r = refine_expr(right, info, assumption);
      // a^p * b^p with a > 0 && b > 0 → (a*b)^p
      if let Some(result) = try_combine_power_product(&l, &r, info) {
        return result;
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(l),
        right: Box::new(r),
      }
    }

    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: outer_base,
      right: outer_exp,
    } => {
      // (a^b)^c with certain conditions on b
      if let Some(result) =
        try_simplify_nested_power(outer_base, outer_exp, info, assumption)
      {
        return result;
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(refine_expr(outer_base, info, assumption)),
        right: Box::new(refine_expr(outer_exp, info, assumption)),
      }
    }

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
      if info.positive_vars.contains(name)
        || info.nonnegative_vars.contains(name)
      {
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
      } else if info.nonnegative_vars.contains(name) {
        Some(1) // treat nonneg as positive for Abs simplification
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

/// Refine (product)^(1/m) by splitting into individual factors.
/// E.g., (x^2 * y^2)^(1/2) with x >= 0, y < 0 → x * (-y) = -(x*y)
fn refine_product_root(
  base: &Expr,
  m: i128,
  info: &AssumptionInfo,
  assumption: &Expr,
) -> Option<Expr> {
  let factors = collect_multiplicative_factors(base);
  if factors.len() < 2 {
    return None;
  }

  // Check if each factor is of the form var^n where n is divisible by m
  // and the variable has known sign
  let mut refined_factors = Vec::new();
  let mut all_simplified = true;

  for factor in &factors {
    // Try to refine (factor)^(1/m)
    let root_expr = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(factor.clone()),
      right: Box::new(Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(m)],
      }),
    };
    let refined = refine_expr(&root_expr, info, assumption);
    // Check if it actually simplified (different from input)
    if expr_to_string(&refined) != expr_to_string(&root_expr) {
      refined_factors.push(refined);
    } else {
      all_simplified = false;
      break;
    }
  }

  if all_simplified && !refined_factors.is_empty() {
    let product = build_product(refined_factors);
    // Evaluate to canonical form
    if let Ok(evaled) = crate::evaluator::evaluate_expr_to_expr(&product) {
      return Some(evaled);
    }
    return Some(product);
  }
  None
}

// ─── Refine helper functions ────────────────────────────────────────

/// Check if an expression is known to be an integer under assumptions.
/// Handles: integer vars, integer constants, sums/products of integers, powers.
fn is_known_integer(expr: &Expr, info: &AssumptionInfo) -> bool {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) => true,
    Expr::Identifier(name) => info.integer_vars.contains(name),
    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus | BinaryOperator::Times,
      left,
      right,
    } => is_known_integer(left, info) && is_known_integer(right, info),
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      // integer^positive_integer = integer
      is_known_integer(left, info)
        && (is_known_positive(right, info) && is_known_integer(right, info))
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_known_integer(operand, info),
    Expr::FunctionCall { name, args }
      if (name == "Plus" || name == "Times") && !args.is_empty() =>
    {
      args.iter().all(|a| is_known_integer(a, info))
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      is_known_integer(&args[0], info)
        && is_known_positive(&args[1], info)
        && is_known_integer(&args[1], info)
    }
    Expr::FunctionCall { name, args }
      if (name == "Floor" || name == "Ceiling") && args.len() == 1 =>
    {
      is_known_real(&args[0], info)
    }
    _ => false,
  }
}

/// Check if an expression is known to be real under assumptions.
fn is_known_real(expr: &Expr, info: &AssumptionInfo) -> bool {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => true,
    Expr::Constant(c) => matches!(c.as_str(), "Pi" | "E" | "EulerGamma"),
    Expr::Identifier(name) => info.real_vars.contains(name),
    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus | BinaryOperator::Times,
      left,
      right,
    } => is_known_real(left, info) && is_known_real(right, info),
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      // a^n with a > 0 and n real, or a real and n integer
      if is_known_real(left, info) && is_known_integer(right, info) {
        return true;
      }
      if is_known_positive(left, info) && is_known_real(right, info) {
        return true;
      }
      false
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => is_known_real(left, info) && is_known_real(right, info),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_known_real(operand, info),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Rational" if args.len() == 2 => true,
      "Plus" | "Times" => args.iter().all(|a| is_known_real(a, info)),
      "Power" if args.len() == 2 => {
        if is_known_real(&args[0], info) && is_known_integer(&args[1], info) {
          return true;
        }
        if is_known_positive(&args[0], info) && is_known_real(&args[1], info) {
          return true;
        }
        false
      }
      "Floor" | "Ceiling" if args.len() == 1 => is_known_real(&args[0], info),
      "Abs" | "Sign" if args.len() == 1 => is_known_real(&args[0], info),
      "Sqrt" if args.len() == 1 => {
        is_known_real(&args[0], info) && is_known_nonnegative(&args[0], info)
      }
      "Log" if args.len() == 1 => is_known_positive(&args[0], info),
      "Gamma" if args.len() == 1 => is_known_positive(&args[0], info),
      "Sin" | "Cos" | "Tan" | "Exp" if args.len() == 1 => {
        is_known_real(&args[0], info)
      }
      _ => false,
    },
    _ => false,
  }
}

/// Check if expression is known to be positive.
fn is_known_positive(expr: &Expr, info: &AssumptionInfo) -> bool {
  match expr {
    Expr::Integer(n) => *n > 0,
    Expr::Identifier(name) => info.positive_vars.contains(name),
    Expr::Constant(c) => matches!(c.as_str(), "Pi" | "E" | "EulerGamma"),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      is_positive_constant(expr)
    }
    Expr::FunctionCall { name, args } if name == "Gamma" && args.len() == 1 => {
      is_known_positive(&args[0], info)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      // pos + nonneg = pos, nonneg + pos = pos
      (is_known_positive(left, info) && is_known_nonnegative(right, info))
        || (is_known_nonnegative(left, info) && is_known_positive(right, info))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      (is_known_positive(left, info) && is_known_positive(right, info))
        || (is_known_negative_expr(left, info)
          && is_known_negative_expr(right, info))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      // positive^anything_real = positive
      is_known_positive(left, info) && is_known_real(right, info)
    }
    Expr::FunctionCall { name, args } if name == "Plus" && !args.is_empty() => {
      // At least one positive, rest nonneg
      let mut has_positive = false;
      for a in args {
        if is_known_positive(a, info) {
          has_positive = true;
        } else if !is_known_nonnegative(a, info) {
          return false;
        }
      }
      has_positive
    }
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      // All positive, or even number of negatives
      args.iter().all(|a| is_known_positive(a, info))
    }
    _ => false,
  }
}

fn is_known_negative_expr(expr: &Expr, info: &AssumptionInfo) -> bool {
  match expr {
    Expr::Integer(n) => *n < 0,
    Expr::Identifier(name) => info.negative_vars.contains(name),
    _ => false,
  }
}

/// Check if expression is known to be non-negative.
fn is_known_nonnegative(expr: &Expr, info: &AssumptionInfo) -> bool {
  if is_known_positive(expr, info) {
    return true;
  }
  match expr {
    Expr::Integer(n) => *n >= 0,
    Expr::Identifier(name) => {
      info.nonnegative_vars.contains(name) || info.positive_vars.contains(name)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      // x^2 is non-negative for real x, x^(2n) in general
      if let Expr::Integer(n) = right.as_ref()
        && n % 2 == 0
        && is_known_real(left, info)
      {
        return true;
      }
      false
    }
    _ => false,
  }
}

/// Extract real part from a + b*I when a, b are real.
fn extract_real_part(expr: &Expr, info: &AssumptionInfo) -> Option<Expr> {
  // Collect additive terms and separate real from imaginary
  let terms = collect_additive_terms(expr);
  let mut real_terms = Vec::new();
  let mut has_imag = false;

  for term in &terms {
    if contains_imaginary_unit(term) {
      has_imag = true;
    } else if is_known_real(term, info) {
      real_terms.push(term.clone());
    } else {
      return None; // Unknown term
    }
  }

  if !has_imag {
    return None; // No imaginary component, Re doesn't simplify this way
  }

  if real_terms.is_empty() {
    Some(Expr::Integer(0))
  } else if real_terms.len() == 1 {
    Some(real_terms.remove(0))
  } else {
    Some(build_sum(real_terms))
  }
}

/// Extract imaginary part from a + b*I when a, b are real.
fn extract_imag_part(expr: &Expr, info: &AssumptionInfo) -> Option<Expr> {
  let terms = collect_additive_terms(expr);
  let mut imag_terms = Vec::new();
  let mut has_real = false;

  for term in &terms {
    if let Some(coeff) = extract_imag_coefficient(term) {
      if is_known_real(&coeff, info) {
        imag_terms.push(coeff);
      } else {
        return None;
      }
    } else if is_known_real(term, info) {
      has_real = true;
    } else {
      return None;
    }
  }

  if imag_terms.is_empty() && has_real {
    return Some(Expr::Integer(0));
  }
  if imag_terms.is_empty() {
    return None;
  }

  if imag_terms.len() == 1 {
    Some(imag_terms.remove(0))
  } else {
    Some(build_sum(imag_terms))
  }
}

/// Check if an expression contains the imaginary unit I.
fn contains_imaginary_unit(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(name) if name == "I" => true,
    Expr::FunctionCall { name, args }
      if name == "Complex"
        && args.len() == 2
        && matches!(&args[1], Expr::Integer(n) if *n != 0) =>
    {
      true
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => contains_imaginary_unit(left) || contains_imaginary_unit(right),
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().any(contains_imaginary_unit)
    }
    _ => false,
  }
}

/// Extract the coefficient of I from a term like b*I or Complex[0, b].
fn extract_imag_coefficient(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      if matches!(&args[0], Expr::Integer(0)) {
        return Some(args[1].clone());
      }
      None
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if is_imaginary_unit(left) {
        Some(*right.clone())
      } else if is_imaginary_unit(right) {
        Some(*left.clone())
      } else {
        None
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      for (i, arg) in args.iter().enumerate() {
        if is_imaginary_unit(arg) {
          let remaining: Vec<Expr> = args
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, a)| a.clone())
            .collect();
          if remaining.len() == 1 {
            return Some(remaining.into_iter().next().unwrap());
          }
          return Some(build_product(remaining));
        }
      }
      None
    }
    _ => None,
  }
}

/// Check if expr is the imaginary unit I.
fn is_imaginary_unit(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(name) if name == "I" => true,
    Expr::FunctionCall { name, args }
      if name == "Complex"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(0))
        && matches!(&args[1], Expr::Integer(1)) =>
    {
      true
    }
    _ => false,
  }
}

/// Check if an expression is an integer multiple of Pi.
/// E.g., k*Pi where k ∈ Integers.
fn is_integer_multiple_of_pi(expr: &Expr, info: &AssumptionInfo) -> bool {
  match expr {
    Expr::Constant(c) if c == "Pi" => true,
    Expr::Integer(0) => true,
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      (is_known_integer(left, info) && is_pi(right))
        || (is_pi(left) && is_known_integer(right, info))
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      (is_known_integer(&args[0], info) && is_pi(&args[1]))
        || (is_pi(&args[0]) && is_known_integer(&args[1], info))
    }
    _ => false,
  }
}

fn is_pi(expr: &Expr) -> bool {
  matches!(expr, Expr::Constant(c) if c == "Pi")
}

/// Split an expression into (non-Pi part, integer k) where expr = non_pi + k*Pi.
/// Returns None if no integer multiple of Pi can be factored out.
fn split_integer_pi_part(
  expr: &Expr,
  info: &AssumptionInfo,
) -> Option<(Expr, Expr)> {
  let terms = collect_additive_terms(expr);
  let mut pi_k: Option<Expr> = None;
  let mut non_pi_terms = Vec::new();

  for term in &terms {
    if let Some(k) = extract_pi_integer_coefficient(term, info) {
      if pi_k.is_some() {
        return None; // Multiple Pi terms, too complex
      }
      pi_k = Some(k);
    } else {
      non_pi_terms.push(term.clone());
    }
  }

  let k = pi_k?;
  let non_pi = if non_pi_terms.is_empty() {
    Expr::Integer(0)
  } else if non_pi_terms.len() == 1 {
    non_pi_terms.remove(0)
  } else {
    build_sum(non_pi_terms)
  };
  Some((non_pi, k))
}

/// Extract integer coefficient k from k*Pi.
fn extract_pi_integer_coefficient(
  expr: &Expr,
  info: &AssumptionInfo,
) -> Option<Expr> {
  if is_pi(expr) {
    return Some(Expr::Integer(1)); // Just Pi = 1*Pi
  }
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if is_pi(right) && is_known_integer(left, info) {
        Some(*left.clone())
      } else if is_pi(left) && is_known_integer(right, info) {
        Some(*right.clone())
      } else {
        None
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if is_pi(&args[1]) && is_known_integer(&args[0], info) {
        Some(args[0].clone())
      } else if is_pi(&args[0]) && is_known_integer(&args[1], info) {
        Some(args[1].clone())
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Check if -Pi/2 < Re[x] < Pi/2 is stated in assumptions.
fn is_in_arctan_range(
  _x: &Expr,
  _info: &AssumptionInfo,
  assumption: &Expr,
) -> bool {
  // Check if the assumption directly states this range.
  // Look for chained comparison: -Pi/2 < Re[x] < Pi/2
  // or in And assumptions
  check_arctan_range_in_assumption(assumption)
}

fn check_arctan_range_in_assumption(assumption: &Expr) -> bool {
  match assumption {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 3 && operators.len() == 2 => {
      // Check pattern: -Pi/2 < Re[x] < Pi/2
      matches!(
        (&operators[0], &operators[1]),
        (
          crate::syntax::ComparisonOp::Less,
          crate::syntax::ComparisonOp::Less
        )
      ) && is_negative_pi_over_2(&operands[0])
        && is_pi_over_2(&operands[2])
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      args.iter().any(check_arctan_range_in_assumption)
    }
    _ => false,
  }
}

fn is_pi_over_2(expr: &Expr) -> bool {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      (is_pi(left) && is_rational_half(right))
        || (is_rational_half(left) && is_pi(right))
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      (is_pi(&args[0]) && is_rational_half(&args[1]))
        || (is_rational_half(&args[0]) && is_pi(&args[1]))
    }
    _ => false,
  }
}

fn is_negative_pi_over_2(expr: &Expr) -> bool {
  match expr {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_pi_over_2(operand),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      // Check for patterns like Rational[-1,2]*Pi or -1*Pi/2
      if is_pi(right) && is_neg_rational_half(left) {
        return true;
      }
      if is_pi(left) && is_neg_rational_half(right) {
        return true;
      }
      false
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if is_pi(&args[1]) && is_neg_rational_half(&args[0]) {
        return true;
      }
      if is_pi(&args[0]) && is_neg_rational_half(&args[1]) {
        return true;
      }
      false
    }
    _ => false,
  }
}

fn is_rational_half(expr: &Expr) -> bool {
  matches!(
    expr,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2
        && matches!(&args[0], Expr::Integer(1))
        && matches!(&args[1], Expr::Integer(2))
  )
}

fn is_neg_rational_half(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!((&args[0], &args[1]), (Expr::Integer(-1), Expr::Integer(2)))
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_rational_half(operand),
    _ => false,
  }
}

/// Refine Log[x] under assumptions.
fn refine_log(
  arg: &Expr,
  info: &AssumptionInfo,
  _assumption: &Expr,
) -> Option<Expr> {
  // Log[x] with x < 0 → I*Pi + Log[-x]
  if let Expr::Identifier(var_name) = arg
    && info.negative_vars.contains(var_name)
  {
    // Return I*Pi + Log[-x] as a FunctionCall to avoid evaluation issues
    return Some(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Identifier("I".to_string()),
            Expr::Constant("Pi".to_string()),
          ],
        },
        Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand: Box::new(arg.clone()),
          }],
        },
      ],
    });
  }

  // Log[x^p] with -1 < p < 1 → p*Log[x]
  // This is the identity log(x^p) = p*log(x) when p is in (-1, 1)
  if let Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: base,
    right: exp,
  } = arg
    && is_var_in_open_range(exp, -1, 1, info)
  {
    return Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(*exp.clone()),
      right: Box::new(Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![*base.clone()],
      }),
    });
  }

  None
}

/// Check if a variable is in an open range (lo, hi) based on chained comparison assumptions.
fn is_var_in_open_range(
  expr: &Expr,
  lo: i128,
  hi: i128,
  info: &AssumptionInfo,
) -> bool {
  // Check raw assumptions for chained comparisons like lo < var < hi
  for raw in &info.raw_assumptions {
    if let Expr::Comparison {
      operands,
      operators,
    } = raw
      && operands.len() == 3
      && operators.len() == 2
      && matches!(
        (&operators[0], &operators[1]),
        (
          crate::syntax::ComparisonOp::Less,
          crate::syntax::ComparisonOp::Less
        )
      )
    {
      // Pattern: lo_expr < var < hi_expr
      if expr_to_string(&operands[1]) == expr_to_string(expr)
        && matches!(&operands[0], Expr::Integer(n) if *n == lo)
        && matches!(&operands[2], Expr::Integer(n) if *n == hi)
      {
        return true;
      }
      // Also check with UnaryOp negation for negative lo
      if expr_to_string(&operands[1]) == expr_to_string(expr)
        && matches!(&operands[2], Expr::Integer(n) if *n == hi)
        && let Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } = &operands[0]
        && matches!(operand.as_ref(), Expr::Integer(n) if *n == -lo)
      {
        return true;
      }
    }
  }
  false
}

/// Refine Element[expr, domain] under assumptions.
fn refine_element(
  expr: &Expr,
  domain: &Expr,
  info: &AssumptionInfo,
) -> Option<Expr> {
  if let Expr::Identifier(dom) = domain {
    match dom.as_str() {
      "Reals" => {
        if is_known_real(expr, info) {
          return Some(Expr::Identifier("True".to_string()));
        }
      }
      "Integers" => {
        if is_known_integer(expr, info) {
          return Some(Expr::Identifier("True".to_string()));
        }
      }
      _ => {}
    }
  }
  None
}

/// Refine Floor/Ceiling with numeric bounds.
/// For Floor: if we know a < x <= b with a, b integers and b = a + 1, then Floor[x] = a.
/// For Ceiling: if we know a < x <= b with integer b, then Ceiling[x] = b.
fn refine_floor_ceiling(
  arg: &Expr,
  info: &AssumptionInfo,
  _assumption: &Expr,
  is_floor: bool,
) -> Option<Expr> {
  // Look for bounds on the variable from raw assumptions
  if let Expr::Identifier(var_name) = arg {
    // Look for chained comparison: a < var <= b or a <= var < b etc.
    for raw in &info.raw_assumptions {
      if let Some((lo, lo_strict, hi, hi_strict)) =
        extract_bounds_for_var(var_name, raw)
        && !is_floor
      {
        // Ceiling[x] with a < x <= b (integer b) → b
        if !hi_strict {
          // hi is inclusive
          if let Expr::Integer(hi_val) = &hi
            && lo_strict
          {
            // a < x <= b
            if let Expr::Integer(lo_val) = &lo
              && *hi_val > *lo_val
            {
              return Some(Expr::Integer(*hi_val));
            }
          }
        }
        // Ceiling[x] with a < x < b (and no integers between a and b except possibly b-like)
        // General: if a < x and x is NOT an integer, ceiling = floor(a) + 1
      }
    }
  }
  None
}

/// Extract bounds (lo, lo_is_strict, hi, hi_is_strict) for a variable from a comparison.
/// Returns (lo_expr, lo_strict, hi_expr, hi_strict) from patterns like:
///   lo < var <= hi  →  (lo, true, hi, false)
///   lo <= var < hi  →  (lo, false, hi, true)
fn extract_bounds_for_var(
  var_name: &str,
  assumption: &Expr,
) -> Option<(Expr, bool, Expr, bool)> {
  match assumption {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 3 && operators.len() == 2 => {
      // Check if middle operand is our variable
      if let Expr::Identifier(name) = &operands[1]
        && name == var_name
      {
        let lo_strict =
          matches!(operators[0], crate::syntax::ComparisonOp::Less);
        let hi_strict =
          matches!(operators[1], crate::syntax::ComparisonOp::Less);
        return Some((
          operands[0].clone(),
          lo_strict,
          operands[2].clone(),
          hi_strict,
        ));
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      for arg in args {
        if let Some(result) = extract_bounds_for_var(var_name, arg) {
          return Some(result);
        }
      }
      None
    }
    _ => None,
  }
}

/// Refine FractionalPart[a] under assumptions.
fn refine_fractional_part(
  arg: &Expr,
  info: &AssumptionInfo,
  assumption: &Expr,
) -> Option<Expr> {
  // If we know Mod[a, 1] == value from assumptions, and a < 0,
  // then FractionalPart[a] = value - 1
  // In Wolfram, FractionalPart[x] = x - Floor[x]
  // For negative x: if Mod[x, 1] = 1/3, then FractionalPart = 1/3 - 1 = -2/3

  // Look for Mod[a, 1] == value in assumptions
  if let Some(mod_val) = find_mod_value_in_assumptions(arg, 1, assumption) {
    // Check if a < 0
    if let Expr::Identifier(var_name) = arg
      && info.negative_vars.contains(var_name)
    {
      // FractionalPart[a] = Mod[a, 1] - 1 for negative a
      let diff = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(mod_val),
        right: Box::new(Expr::Integer(1)),
      };
      if let Ok(result) = crate::evaluator::evaluate_expr_to_expr(&diff) {
        return Some(result);
      }
      return Some(crate::functions::calculus_ast::simplify(diff));
    }
    // For positive a: FractionalPart[a] = Mod[a, 1]
    return Some(mod_val);
  }
  None
}

/// Find a value for Mod[expr, m] from assumptions.
fn find_mod_value_in_assumptions(
  expr: &Expr,
  m: i128,
  assumption: &Expr,
) -> Option<Expr> {
  match assumption {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && matches!(operators[0], crate::syntax::ComparisonOp::Equal) =>
    {
      // Check if one side is Mod[expr, m]
      if is_mod_of(expr, m, &operands[0]) {
        return Some(operands[1].clone());
      }
      if is_mod_of(expr, m, &operands[1]) {
        return Some(operands[0].clone());
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      for arg in args {
        if let Some(val) = find_mod_value_in_assumptions(expr, m, arg) {
          return Some(val);
        }
      }
      None
    }
    _ => None,
  }
}

/// Check if check_expr is Mod[target_expr, m].
fn is_mod_of(target_expr: &Expr, m: i128, check_expr: &Expr) -> bool {
  if let Expr::FunctionCall { name, args } = check_expr
    && name == "Mod"
    && args.len() == 2
    && expr_to_string(&args[0]) == expr_to_string(target_expr)
    && matches!(&args[1], Expr::Integer(n) if *n == m)
  {
    return true;
  }
  false
}

/// Refine Mod[a, m] under assumptions.
fn refine_mod(
  a: &Expr,
  m: &Expr,
  info: &AssumptionInfo,
  assumption: &Expr,
) -> Option<Expr> {
  // If Element[(a + k)/m, Integers] for some k, then Mod[a, m] = m - k
  if let Expr::Integer(m_val) = m
    && let Some(result) =
      find_mod_from_integer_assumption(a, *m_val, info, assumption)
  {
    return Some(result);
  }
  None
}

/// Check if assumptions imply Element[(a + k)/m, Integers] for some k.
/// If so, a ≡ -k (mod m), meaning Mod[a, m] = m - k (when k > 0) or -k (when k <= 0).
fn find_mod_from_integer_assumption(
  a: &Expr,
  m_val: i128,
  info: &AssumptionInfo,
  assumption: &Expr,
) -> Option<Expr> {
  match assumption {
    Expr::FunctionCall { name, args }
      if name == "Element" && args.len() == 2 =>
    {
      if let Expr::Identifier(domain) = &args[1]
        && domain == "Integers"
      {
        // Check if args[0] is (a + k) / m
        if let Some(k) = extract_linear_div_offset(&args[0], a, m_val) {
          // (a + k) / m ∈ Integers means a ≡ -k (mod m)
          // Mod[a, m] = (m - k) % m
          let result = ((m_val - k) % m_val + m_val) % m_val;
          return Some(Expr::Integer(result));
        }
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      for arg in args {
        if let Some(result) =
          find_mod_from_integer_assumption(a, m_val, info, arg)
        {
          return Some(result);
        }
      }
      None
    }
    _ => None,
  }
}

/// Check if expr is (a + k) / m and return k.
/// Handles forms like: Times[Rational[1, m], Plus[a, k]] or
/// Times[Power[m, -1], Plus[a, k]]
fn extract_linear_div_offset(expr: &Expr, a: &Expr, m: i128) -> Option<i128> {
  // Try to match: Rational[1, m] * (a + k) or (a + k) / m
  // After evaluation, this may appear as Times[Rational[1, m], Plus[a, k]]
  // or BinaryOp Divide

  // Collect multiplicative factors
  let factors = collect_multiplicative_factors(expr);

  // Find the 1/m factor and the sum factor
  let mut found_inv_m = false;
  let mut sum_factor: Option<&Expr> = None;

  for f in &factors {
    if is_rational_1_over_m(f, m) || is_power_m_neg1(f, m) {
      found_inv_m = true;
    } else {
      sum_factor = Some(f);
    }
  }

  if !found_inv_m {
    return None;
  }

  let sum = sum_factor?;

  // Check if sum is a + k
  let terms = collect_additive_terms(sum);
  let a_str = expr_to_string(a);
  let mut found_a = false;
  let mut k: i128 = 0;

  for term in &terms {
    if expr_to_string(term) == a_str {
      found_a = true;
    } else if let Expr::Integer(n) = term {
      k += n;
    } else {
      return None; // Unknown term
    }
  }

  if found_a { Some(k) } else { None }
}

fn is_rational_1_over_m(expr: &Expr, m: i128) -> bool {
  matches!(
    expr,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2
        && matches!(&args[0], Expr::Integer(1))
        && matches!(&args[1], Expr::Integer(d) if *d == m)
  )
}

fn is_power_m_neg1(expr: &Expr, m: i128) -> bool {
  matches!(
    expr,
    Expr::BinaryOp { op: BinaryOperator::Power, left, right }
      if matches!(left.as_ref(), Expr::Integer(n) if *n == m)
        && matches!(right.as_ref(), Expr::Integer(-1))
  )
}

/// Try to combine a^p * b^p → (a*b)^p when a > 0 and b > 0.
fn try_combine_power_product(
  left: &Expr,
  right: &Expr,
  info: &AssumptionInfo,
) -> Option<Expr> {
  let (base_l, exp_l) = extract_base_and_exp(left);
  let (base_r, exp_r) = extract_base_and_exp(right);

  // Same exponent, both bases positive
  if expr_to_string(&exp_l) == expr_to_string(&exp_r)
    && !matches!(&exp_l, Expr::Integer(1))
    && is_known_positive(&base_l, info)
    && is_known_positive(&base_r, info)
  {
    return Some(Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(base_l),
        right: Box::new(base_r),
      }),
      right: Box::new(exp_l),
    });
  }
  None
}

/// Try to simplify nested powers (a^b)^c under assumptions.
fn try_simplify_nested_power(
  outer_base: &Expr,
  outer_exp: &Expr,
  info: &AssumptionInfo,
  assumption: &Expr,
) -> Option<Expr> {
  // (a^b)^c with -1 < b < 1 → a^(b*c)
  if let Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: base,
    right: inner_exp,
  } = outer_base
    && is_var_in_open_range(inner_exp, -1, 1, info)
  {
    return Some(Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base.clone(),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: inner_exp.clone(),
        right: Box::new(refine_expr(outer_exp, info, assumption)),
      }),
    });
  }
  None
}

/// Check algebraic comparisons under assumptions.
/// Handles cases like:
/// - a^2 - b^2 + 1 == 0 with a + b == 0 → substitute and check
/// - a^2 - a*b + b^2 >= 0 with a, b real → True (positive-definite)
/// - (x-1)^2 + (y-2)^2 < 3/2 with x^2 + y^2 <= 1 → False (geometric)
fn check_algebraic_comparison(
  expr: &Expr,
  info: &AssumptionInfo,
  assumption: &Expr,
) -> Option<bool> {
  if let Expr::Comparison {
    operands,
    operators,
  } = expr
    && operands.len() == 2
    && operators.len() == 1
  {
    let op = &operators[0];
    let left = &operands[0];
    let right = &operands[1];

    // For >= 0 comparisons: check if left - right is provably non-negative
    match op {
      crate::syntax::ComparisonOp::GreaterEqual => {
        if matches!(right, Expr::Integer(0))
          && is_provably_nonnegative(left, info)
        {
          return Some(true);
        }
        // General: check if left - right >= 0
        let diff = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(left.clone()),
          right: Box::new(right.clone()),
        };
        let simplified = crate::functions::calculus_ast::simplify(diff);
        if is_provably_nonnegative(&simplified, info) {
          return Some(true);
        }
      }
      crate::syntax::ComparisonOp::Equal => {
        // Try substitution from equation assumptions
        if let Some(result) =
          check_equation_by_substitution(left, right, op, info, assumption)
        {
          return Some(result);
        }
      }
      crate::syntax::ComparisonOp::Less => {
        // Try substitution-based reasoning
        if let Some(result) =
          check_inequality_by_substitution(left, right, op, info, assumption)
        {
          return Some(result);
        }
      }
      _ => {}
    }
  }
  None
}

/// Check if an expression is provably non-negative for all real values of its variables.
fn is_provably_nonnegative(expr: &Expr, info: &AssumptionInfo) -> bool {
  if is_provably_positive(expr, info) {
    return true;
  }

  match expr {
    Expr::Integer(n) => return *n >= 0,
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Integer(n) = right.as_ref()
        && n % 2 == 0
        && is_known_real(left, info)
      {
        return true;
      }
    }
    _ => {}
  }

  // Check if it's a nonnegative-definite quadratic form
  let expanded = expand_and_combine(expr);
  let terms = collect_additive_terms(&expanded);
  if let Some(true) = check_quadratic_form_nonnegative(&terms, info) {
    return true;
  }

  false
}

/// Check if a quadratic form is nonnegative-definite (>= 0 for all real values).
fn check_quadratic_form_nonnegative(
  terms: &[Expr],
  info: &AssumptionInfo,
) -> Option<bool> {
  let mut vars = std::collections::HashSet::new();
  for term in terms {
    collect_variables(term, &mut vars);
  }

  if vars.len() > 2 {
    return None;
  }
  for v in &vars {
    if !info.real_vars.contains(v) {
      return None;
    }
  }

  let vars_vec: Vec<String> = vars.into_iter().collect();

  if vars_vec.len() == 2 {
    let x = &vars_vec[0];
    let y = &vars_vec[1];
    if let Some((a, b, c, d, e, f)) =
      extract_bivariate_quadratic_coefficients(terms, x, y)
      && a >= 0
      && 4 * a * c - b * b >= 0
    {
      if d == 0 && e == 0 && f >= 0 {
        return Some(true);
      }
      let det = 4 * a * c - b * b;
      if det > 0 {
        let num = a * e * e - b * d * e + c * d * d;
        if f * det >= num {
          return Some(true);
        }
      }
    }
  } else if vars_vec.len() == 1 {
    let var = &vars_vec[0];
    if let Some((a, b, c)) = extract_quadratic_coefficients(terms, var)
      && a > 0
      && 4 * a * c >= b * b
    {
      return Some(true);
    }
  }
  None
}

/// Check if an expression is provably positive for all real values.
fn is_provably_positive(expr: &Expr, info: &AssumptionInfo) -> bool {
  match expr {
    Expr::Integer(n) if *n > 0 => return true,
    Expr::Constant(c) if matches!(c.as_str(), "Pi" | "E") => return true,
    _ => {}
  }

  // Try to recognize positive-definite quadratic forms
  // Expand and check if it's a sum of terms that are individually >= 0
  // with at least one strictly > 0
  let expanded = expand_and_combine(expr);
  let terms = collect_additive_terms(&expanded);

  // Check for pattern: sum of even-power terms + positive constant
  // e.g., x^2 - x*y + y^2 + 1
  // The quadratic form x^2 - xy + y^2 has discriminant b^2 - 4ac = 1 - 4 = -3 < 0
  // and leading coefficient > 0, so it's positive definite
  if let Some(result) = check_quadratic_form_positive(&terms, info) {
    return result;
  }

  false
}

/// Check if a sum of terms forms a positive-definite quadratic expression.
fn check_quadratic_form_positive(
  terms: &[Expr],
  info: &AssumptionInfo,
) -> Option<bool> {
  // Collect variables
  let mut vars = std::collections::HashSet::new();
  for term in terms {
    collect_variables(term, &mut vars);
  }

  // Only handle 1-2 variable cases for now
  if vars.len() > 2 {
    return None;
  }

  // Check that all vars are real
  for v in &vars {
    if !info.real_vars.contains(v) {
      return None;
    }
  }

  let vars_vec: Vec<String> = vars.into_iter().collect();

  if vars_vec.len() == 1 {
    // Single variable: check if ax^2 + bx + c with a > 0 and b^2 - 4ac < 0
    let var = &vars_vec[0];
    let (a, b, c) = extract_quadratic_coefficients(terms, var)?;
    if a > 0 && b * b - 4 * a * c < 0 {
      return Some(true);
    }
    if a > 0 && c > 0 && b * b < 4 * a * c {
      return Some(true);
    }
  } else if vars_vec.len() == 2 {
    // Two variables: check positive definiteness
    // ax^2 + bxy + cy^2 + dx + ey + f
    let x = &vars_vec[0];
    let y = &vars_vec[1];
    if let Some(coeffs) = extract_bivariate_quadratic_coefficients(terms, x, y)
    {
      let (a, b, c, d, e, f) = coeffs;
      // The quadratic form ax^2 + bxy + cy^2 is positive definite when a > 0 and 4ac - b^2 > 0
      // With linear terms: we complete the square
      // The minimum is at (x0, y0) and the minimum value must be > 0
      if a > 0 && 4 * a * c - b * b > 0 {
        // Minimum value of the quadratic form:
        // f - (4*a*e^2 - 4*b*d*e + 4*c*d^2) / (4*(4*a*c - b^2))
        // Simplified: f - (a*e^2 - b*d*e + c*d^2) / (4*a*c - b^2)
        let det = 4 * a * c - b * b;
        let num = a * e * e - b * d * e + c * d * d;
        // min_val = f - num/det, so min_val > 0 iff f*det > num
        if f * det > num {
          return Some(true);
        }
        // If no linear terms and f >= 0
        if d == 0 && e == 0 && f > 0 {
          return Some(true);
        }
        if d == 0 && e == 0 && f == 0 {
          return Some(false); // Can be zero
        }
      }
    }
  }
  None
}

/// Extract coefficients (a, b, c) from ax^2 + bx + c.
fn extract_quadratic_coefficients(
  terms: &[Expr],
  var: &str,
) -> Option<(i128, i128, i128)> {
  let mut a: i128 = 0;
  let mut b: i128 = 0;
  let mut c: i128 = 0;

  for term in terms {
    let (power, coeff) = term_var_power_and_coeff(term, var);
    let coeff_val = match &crate::functions::calculus_ast::simplify(coeff) {
      Expr::Integer(n) => *n,
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } if matches!(operand.as_ref(), Expr::Integer(n) if *n > 0) => {
        if let Expr::Integer(n) = operand.as_ref() {
          -n
        } else {
          return None;
        }
      }
      _ => return None,
    };
    match power {
      0 => c += coeff_val,
      1 => b += coeff_val,
      2 => a += coeff_val,
      _ => return None, // Higher degree
    }
  }

  Some((a, b, c))
}

/// Extract bivariate quadratic coefficients (a, b, c, d, e, f) from
/// ax^2 + bxy + cy^2 + dx + ey + f.
fn extract_bivariate_quadratic_coefficients(
  terms: &[Expr],
  x: &str,
  y: &str,
) -> Option<(i128, i128, i128, i128, i128, i128)> {
  let mut a: i128 = 0; // x^2
  let mut b: i128 = 0; // x*y
  let mut c: i128 = 0; // y^2
  let mut d: i128 = 0; // x
  let mut e: i128 = 0; // y
  let mut f: i128 = 0; // constant

  for term in terms {
    let (x_pow, y_pow, coeff) = term_bivariate_powers_and_coeff(term, x, y)?;
    if x_pow + y_pow > 2 {
      return None;
    }
    match (x_pow, y_pow) {
      (0, 0) => f += coeff,
      (1, 0) => d += coeff,
      (0, 1) => e += coeff,
      (2, 0) => a += coeff,
      (1, 1) => b += coeff,
      (0, 2) => c += coeff,
      _ => return None,
    }
  }

  Some((a, b, c, d, e, f))
}

/// Extract (x_power, y_power, integer_coefficient) from a term in two variables.
fn term_bivariate_powers_and_coeff(
  term: &Expr,
  x: &str,
  y: &str,
) -> Option<(i128, i128, i128)> {
  // Handle negated terms: -(expr) → negate and recurse
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = term
  {
    let (xp, yp, c) = term_bivariate_powers_and_coeff(operand, x, y)?;
    return Some((xp, yp, -c));
  }

  let factors = collect_multiplicative_factors(term);
  let mut x_pow: i128 = 0;
  let mut y_pow: i128 = 0;
  let mut coeff: i128 = 1;

  for f in &factors {
    match f {
      Expr::Integer(n) => coeff *= n,
      Expr::Identifier(name) if name == x => x_pow += 1,
      Expr::Identifier(name) if name == y => y_pow += 1,
      // BinaryOp Power
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => {
        if let Expr::Identifier(name) = left.as_ref()
          && let Expr::Integer(p) = right.as_ref()
        {
          if name == x {
            x_pow += p;
          } else if name == y {
            y_pow += p;
          } else {
            return None;
          }
        } else {
          return None;
        }
      }
      // FunctionCall Power[var, n]
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        if let Expr::Identifier(var_name) = &args[0]
          && let Expr::Integer(p) = &args[1]
        {
          if var_name == x {
            x_pow += p;
          } else if var_name == y {
            y_pow += p;
          } else {
            return None;
          }
        } else {
          return None;
        }
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => {
        coeff = -coeff;
        match operand.as_ref() {
          Expr::Integer(n) => coeff *= n,
          Expr::Identifier(name) if name == x => x_pow += 1,
          Expr::Identifier(name) if name == y => y_pow += 1,
          _ => return None,
        }
      }
      _ => return None,
    }
  }

  Some((x_pow, y_pow, coeff))
}

/// Check equation by substitution from assumptions.
fn check_equation_by_substitution(
  left: &Expr,
  right: &Expr,
  _op: &crate::syntax::ComparisonOp,
  _info: &AssumptionInfo,
  assumption: &Expr,
) -> Option<bool> {
  // If assumption is an equation like a + b == 0 (meaning b = -a),
  // substitute and check if left == right
  let substitutions = extract_equation_substitutions(assumption);
  if substitutions.is_empty() {
    return None;
  }

  for (var, replacement) in &substitutions {
    let new_left = substitute_var(left, var, replacement);
    let new_right = substitute_var(right, var, replacement);

    // Use full evaluation to simplify (handles (-a)^2 → a^2 etc.)
    let new_left_eval = crate::evaluator::evaluate_expr_to_expr(&new_left)
      .unwrap_or_else(|_| expand_and_combine(&new_left));
    let new_right_eval = crate::evaluator::evaluate_expr_to_expr(&new_right)
      .unwrap_or_else(|_| expand_and_combine(&new_right));

    // Simplify left - right and check if it's a nonzero constant
    let diff = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(new_left_eval),
      right: Box::new(new_right_eval),
    };
    let diff_eval = crate::evaluator::evaluate_expr_to_expr(&diff)
      .unwrap_or_else(|_| expand_and_combine(&diff));
    match &diff_eval {
      Expr::Integer(n) if *n != 0 => return Some(false),
      Expr::Integer(0) => return Some(true),
      _ => {}
    }
  }
  None
}

/// Check inequality by substitution from assumptions.
fn check_inequality_by_substitution(
  left: &Expr,
  right: &Expr,
  _op: &crate::syntax::ComparisonOp,
  info: &AssumptionInfo,
  assumption: &Expr,
) -> Option<bool> {
  // For patterns like (x-1)^2 + (y-2)^2 < 3/2 with x^2 + y^2 <= 1
  // Try to evaluate the maximum of left under the constraint

  // Try numeric bound checking: substitute extremes from constraints
  let ineq_constraints = extract_inequality_constraints(assumption);
  if ineq_constraints.is_empty() {
    return None;
  }

  // For the simple case: check if the LHS has a known minimum > RHS or known maximum < RHS
  // under the constraints.
  // We'll use the approach: expand left - right and check bounds.

  let diff = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(left.clone()),
    right: Box::new(right.clone()),
  };
  let expanded = expand_and_combine(&diff);

  // If we can prove diff >= 0 under the constraints, then left < right is False
  if is_provably_nonnegative_under_constraints(&expanded, info, assumption) {
    return Some(false);
  }

  None
}

fn is_provably_nonnegative_under_constraints(
  expr: &Expr,
  info: &AssumptionInfo,
  assumption: &Expr,
) -> bool {
  // Expand and try to show it's non-negative
  // Strategy: substitute constraint bounds and check
  let expanded = expand_and_combine(expr);

  // Check if it's directly non-negative
  if is_provably_nonnegative(&expanded, info) {
    return true;
  }

  // For x^2 + y^2 <= 1: substitute x^2 + y^2 with max value (1)
  // and check remaining terms
  // Try: given a <= constraint (like x^2 + y^2 <= 1),
  // check if expr >= 0 when substituting the constraint bound.

  // Extract comparison constraints
  let constraints = extract_inequality_constraints(assumption);
  for (lhs, rhs, _is_leq) in &constraints {
    // Try substituting lhs = rhs (upper bound) in the expression
    let lhs_str = expr_to_string(lhs);
    let rhs_str = expr_to_string(rhs);

    // Find lhs terms in the expanded expression and substitute with rhs
    let terms = collect_additive_terms(&expanded);
    let mut new_terms = Vec::new();
    let mut substituted = false;

    for term in &terms {
      let term_str = expr_to_string(term);
      if term_str == lhs_str {
        new_terms.push(rhs.clone());
        substituted = true;
      } else {
        // Check for coefficient * lhs pattern
        let factors = collect_multiplicative_factors(term);
        let mut found = false;
        let mut coeff_factors = Vec::new();
        for f in &factors {
          if !found && expr_to_string(f) == lhs_str {
            found = true;
            coeff_factors.push(rhs.clone());
          } else {
            coeff_factors.push(f.clone());
          }
        }
        if found {
          new_terms.push(build_product(coeff_factors));
          substituted = true;
        } else {
          new_terms.push(term.clone());
        }
      }
    }

    if substituted && !new_terms.is_empty() {
      let substituted_expr = build_sum(new_terms);
      let simplified = expand_and_combine(&substituted_expr);
      // For the substituted expression (upper bound), if it's >= 0,
      // then the original is also >= 0 when lhs <= rhs
      // (only works if the expression is monotonically non-decreasing in lhs)
      // Be conservative: only if it simplifies to a non-negative constant
      match &simplified {
        Expr::Integer(n) if *n >= 0 => return true,
        Expr::FunctionCall { name, args }
          if name == "Rational" && args.len() == 2 =>
        {
          if is_nonnegative_constant(&simplified) {
            return true;
          }
        }
        _ => {}
      }
      // Also try evaluating
      if let Ok(evaled) = crate::evaluator::evaluate_expr_to_expr(&simplified) {
        match &evaled {
          Expr::Integer(n) if *n >= 0 => return true,
          _ => {
            if is_nonnegative_constant(&evaled) {
              return true;
            }
          }
        }
      }
    }

    // Also try the lhs_str in the full expression string
    let expr_str = expr_to_string(&expanded);
    if expr_str.contains(&lhs_str) {
      // Direct string substitution approach — evaluate numerically
      let sub_expr_str = expr_str.replace(&lhs_str, &rhs_str);
      if let Ok(parsed) =
        crate::evaluator::evaluate_expr_to_expr(&Expr::Identifier(sub_expr_str))
        && is_nonnegative_constant(&parsed)
      {
        return true;
      }
    }
  }

  // Strategy: Cauchy-Schwarz bound for sum-of-squares constraints.
  // For a constraint like x^2 + y^2 <= C and expression like
  // x^2 + y^2 - 2x - 4y + 7/2, decompose into quadratic + linear + constant
  // and use Cauchy-Schwarz to bound the linear part.
  if check_nonneg_via_cauchy_schwarz(&expanded, &constraints) {
    return true;
  }

  false
}

/// Use Cauchy-Schwarz inequality to prove an expression is non-negative
/// under sum-of-squares constraints.
///
/// For expression `a*x^2 + c*y^2 + d*x + e*y + f` with constraint
/// `a_c*x^2 + c_c*y^2 <= C`:
/// - The excess quadratic `(a-a_c)*x^2 + (c-c_c)*y^2 >= 0`
/// - By Cauchy-Schwarz: `d*x + e*y >= -S * sqrt(Q)` where
///   `S = sqrt(d^2/a_c + e^2/c_c)` and `Q = a_c*x^2 + c_c*y^2`
/// - So expr >= `Q - S*sqrt(Q) + f`, minimize over `0 <= sqrt(Q) <= sqrt(C)`
fn check_nonneg_via_cauchy_schwarz(
  expanded: &Expr,
  constraints: &[(Expr, Expr, bool)],
) -> bool {
  let expr_terms = collect_additive_terms(expanded);
  let mut expr_vars = std::collections::HashSet::new();
  collect_variables(expanded, &mut expr_vars);

  if expr_vars.is_empty() || expr_vars.len() > 2 {
    return false;
  }

  let vars_vec: Vec<String> = expr_vars.into_iter().collect();

  for (lhs, rhs, is_leq) in constraints {
    if !*is_leq {
      continue;
    }

    let c_bound = match const_expr_to_f64(rhs) {
      Some(v) if v > 0.0 => v,
      _ => continue,
    };

    let lhs_expanded = expand_and_combine(lhs);
    let lhs_terms = collect_additive_terms(&lhs_expanded);

    if vars_vec.len() == 2 {
      let x = &vars_vec[0];
      let y = &vars_vec[1];

      let (a_c, b_c, c_c, d_c, e_c, f_c) =
        match extract_bivariate_quadratic_f64(&lhs_terms, x, y) {
          Some(v) => v,
          None => continue,
        };

      // Constraint LHS must be a pure sum of squares
      if a_c <= 0.0
        || c_c <= 0.0
        || b_c.abs() > 1e-10
        || d_c.abs() > 1e-10
        || e_c.abs() > 1e-10
        || f_c.abs() > 1e-10
      {
        continue;
      }

      let (a_e, b_e, c_e, d_e, e_e, f_e) =
        match extract_bivariate_quadratic_f64(&expr_terms, x, y) {
          Some(v) => v,
          None => continue,
        };

      // Expression quadratic part must dominate the constraint's
      if a_e < a_c - 1e-10 || c_e < c_c - 1e-10 || b_e.abs() > 1e-10 {
        continue;
      }

      // S^2 for the weighted Cauchy-Schwarz bound
      let s_sq = d_e * d_e / a_c + e_e * e_e / c_c;
      let s = s_sq.sqrt();
      let t_crit = s / 2.0;
      let t_max = c_bound.sqrt();

      let min_val = if t_crit <= t_max {
        f_e - s_sq / 4.0
      } else {
        c_bound - s * t_max + f_e
      };

      if min_val >= -1e-9 {
        return true;
      }
    } else if vars_vec.len() == 1 {
      let x = &vars_vec[0];
      let y_dummy = "";

      let (a_c, _, _, d_c, _, f_c) =
        match extract_bivariate_quadratic_f64(&lhs_terms, x, y_dummy) {
          Some(v) => v,
          None => continue,
        };

      if a_c <= 0.0 || d_c.abs() > 1e-10 || f_c.abs() > 1e-10 {
        continue;
      }

      let (a_e, _, _, d_e, _, f_e) =
        match extract_bivariate_quadratic_f64(&expr_terms, x, y_dummy) {
          Some(v) => v,
          None => continue,
        };

      if a_e < a_c - 1e-10 {
        continue;
      }

      let s_sq = d_e * d_e / a_c;
      let s = s_sq.sqrt();
      let t_crit = s / 2.0;
      let t_max = c_bound.sqrt();

      let min_val = if t_crit <= t_max {
        f_e - s_sq / 4.0
      } else {
        c_bound - s * t_max + f_e
      };

      if min_val >= -1e-9 {
        return true;
      }
    }
  }

  false
}

/// Convert a constant expression to f64.
fn const_expr_to_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(num), Expr::Integer(den)) = (&args[0], &args[1]) {
        Some(*num as f64 / *den as f64)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Extract bivariate quadratic coefficients as f64.
/// Returns (a, b, c, d, e, f) for ax^2 + bxy + cy^2 + dx + ey + f.
/// Handles rational coefficients.
fn extract_bivariate_quadratic_f64(
  terms: &[Expr],
  x: &str,
  y: &str,
) -> Option<(f64, f64, f64, f64, f64, f64)> {
  let mut a = 0.0f64;
  let mut b = 0.0f64;
  let mut c = 0.0f64;
  let mut d = 0.0f64;
  let mut e = 0.0f64;
  let mut f = 0.0f64;

  for term in terms {
    let (x_pow, y_pow, coeff) =
      term_bivariate_powers_and_coeff_f64(term, x, y)?;
    if x_pow + y_pow > 2 {
      return None;
    }
    match (x_pow, y_pow) {
      (0, 0) => f += coeff,
      (1, 0) => d += coeff,
      (0, 1) => e += coeff,
      (2, 0) => a += coeff,
      (1, 1) => b += coeff,
      (0, 2) => c += coeff,
      _ => return None,
    }
  }

  Some((a, b, c, d, e, f))
}

/// Extract (x_power, y_power, f64_coefficient) from a term.
fn term_bivariate_powers_and_coeff_f64(
  term: &Expr,
  x: &str,
  y: &str,
) -> Option<(i128, i128, f64)> {
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = term
  {
    let (xp, yp, c) = term_bivariate_powers_and_coeff_f64(operand, x, y)?;
    return Some((xp, yp, -c));
  }

  let factors = collect_multiplicative_factors(term);
  let mut x_pow: i128 = 0;
  let mut y_pow: i128 = 0;
  let mut coeff: f64 = 1.0;

  for f in &factors {
    match f {
      Expr::Integer(n) => coeff *= *n as f64,
      Expr::Real(r) => coeff *= r,
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(num), Expr::Integer(den)) = (&args[0], &args[1]) {
          coeff *= *num as f64 / *den as f64;
        } else {
          return None;
        }
      }
      Expr::Identifier(name) if name == x => x_pow += 1,
      Expr::Identifier(name) if !y.is_empty() && name == y => y_pow += 1,
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => {
        if let Expr::Identifier(name) = left.as_ref()
          && let Expr::Integer(p) = right.as_ref()
        {
          if name == x {
            x_pow += p;
          } else if !y.is_empty() && name == y {
            y_pow += p;
          } else {
            return None;
          }
        } else {
          return None;
        }
      }
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        if let Expr::Identifier(var_name) = &args[0]
          && let Expr::Integer(p) = &args[1]
        {
          if var_name == x {
            x_pow += p;
          } else if !y.is_empty() && var_name == y {
            y_pow += p;
          } else {
            return None;
          }
        } else {
          return None;
        }
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => {
        coeff = -coeff;
        match operand.as_ref() {
          Expr::Integer(n) => coeff *= *n as f64,
          Expr::Identifier(name) if name == x => x_pow += 1,
          Expr::Identifier(name) if !y.is_empty() && name == y => y_pow += 1,
          _ => return None,
        }
      }
      _ => return None,
    }
  }

  Some((x_pow, y_pow, coeff))
}

/// Extract inequality constraints from assumptions.
/// Returns (lhs, rhs, is_leq) for lhs <= rhs or lhs < rhs.
fn extract_inequality_constraints(
  assumption: &Expr,
) -> Vec<(Expr, Expr, bool)> {
  let mut result = Vec::new();
  extract_inequality_constraints_inner(assumption, &mut result);
  result
}

fn extract_inequality_constraints_inner(
  assumption: &Expr,
  result: &mut Vec<(Expr, Expr, bool)>,
) {
  match assumption {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => match &operators[0] {
      crate::syntax::ComparisonOp::LessEqual => {
        result.push((operands[0].clone(), operands[1].clone(), true));
      }
      crate::syntax::ComparisonOp::Less => {
        result.push((operands[0].clone(), operands[1].clone(), false));
      }
      crate::syntax::ComparisonOp::GreaterEqual => {
        result.push((operands[1].clone(), operands[0].clone(), true));
      }
      crate::syntax::ComparisonOp::Greater => {
        result.push((operands[1].clone(), operands[0].clone(), false));
      }
      _ => {}
    },
    Expr::FunctionCall { name, args } if name == "And" => {
      for arg in args {
        extract_inequality_constraints_inner(arg, result);
      }
    }
    _ => {}
  }
}

/// Extract equation substitutions from assumption.
/// E.g., a + b == 0 → b = -a.
fn extract_equation_substitutions(assumption: &Expr) -> Vec<(String, Expr)> {
  let mut result = Vec::new();
  extract_equation_substitutions_inner(assumption, &mut result);
  result
}

fn extract_equation_substitutions_inner(
  assumption: &Expr,
  result: &mut Vec<(String, Expr)>,
) {
  match assumption {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && matches!(operators[0], crate::syntax::ComparisonOp::Equal) =>
    {
      // Try to solve for a variable
      // Simple case: a + b == 0 → b = -a
      let lhs = &operands[0];
      let rhs = &operands[1];

      if matches!(rhs, Expr::Integer(0)) {
        // lhs == 0, try to isolate a variable
        let terms = collect_additive_terms(lhs);
        for (i, term) in terms.iter().enumerate() {
          if let Expr::Identifier(var_name) = term {
            // var = -(sum of other terms)
            let other_terms: Vec<Expr> = terms
              .iter()
              .enumerate()
              .filter(|(j, _)| *j != i)
              .map(|(_, t)| negate_term(t))
              .collect();
            if other_terms.len() == 1 {
              result.push((var_name.clone(), other_terms[0].clone()));
            } else if !other_terms.is_empty() {
              result.push((var_name.clone(), build_sum(other_terms)));
            }
          }
        }
      }
    }
    Expr::FunctionCall { name, args } if name == "And" => {
      for arg in args {
        extract_equation_substitutions_inner(arg, result);
      }
    }
    _ => {}
  }
}

/// Substitute a variable with a replacement expression.
fn substitute_var(expr: &Expr, var: &str, replacement: &Expr) -> Expr {
  match expr {
    Expr::Identifier(name) if name == var => replacement.clone(),
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_var(left, var, replacement)),
      right: Box::new(substitute_var(right, var, replacement)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_var(operand, var, replacement)),
    },
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| substitute_var(a, var, replacement))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|i| substitute_var(i, var, replacement))
        .collect(),
    ),
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands
        .iter()
        .map(|o| substitute_var(o, var, replacement))
        .collect(),
      operators: operators.clone(),
    },
    _ => expr.clone(),
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
  // Single argument: consult $Assumptions from the environment (e.g. set by
  // Assuming[...]) and apply refinement if any are active.
  let simplified = simplify_expr_with_together(&args[0]);
  Ok(apply_active_assumptions(&simplified))
}

/// Run `simplify_expr` and also try `together_expr`, both at the top level and
/// recursively on sub-expressions, picking the leaf-smallest result. This lets
/// nested fraction forms (e.g. continued fractions like `1 + 1/(1 + 1/(1 + 1/x))`
/// or `1/(1 + 1/x)`) collapse into a single fraction without having to sprinkle
/// Together calls through every branch of `simplify_expr`. Sub-expression
/// combining handles the case where combining the whole expression makes the
/// leaf count larger but combining an inner fraction still helps.
fn simplify_expr_with_together(expr: &Expr) -> Expr {
  let simplified = simplify_expr(expr);
  let mut best = simplified.clone();
  let mut best_c = leaf_count(&best);

  // Candidate 1: Together the whole simplified expression.
  let togethered = super::together::together_expr(&simplified);
  let tc = leaf_count(&togethered);
  if tc < best_c {
    // Re-run simplify_expr to absorb any cancellations Together exposed.
    let resimplified = simplify_expr(&togethered);
    let rc = leaf_count(&resimplified);
    if rc <= tc {
      best = resimplified;
      best_c = rc;
    } else {
      best = togethered;
      best_c = tc;
    }
  }

  // Candidate 2: Together sub-expressions (leaves outer structure alone).
  // This helps cases like `1 + 1/(1 + 1/x)` where the whole-expression Together
  // is tied with the original, but combining only the inner fraction is
  // strictly better.
  let sub_togethered = together_subexpressions(&simplified);
  let sc = leaf_count(&sub_togethered);
  if sc < best_c {
    best = sub_togethered;
    let _ = best_c;
  }

  best
}

/// Apply `together_expr` only to proper sub-expressions of `expr`, leaving the
/// outermost operator untouched. Used to combine inner fractions inside an
/// expression whose top-level combining doesn't help.
fn together_subexpressions(expr: &Expr) -> Expr {
  match expr {
    Expr::BinaryOp { op, left, right } => {
      let l = super::together::together_expr(left);
      let r = super::together::together_expr(right);
      Expr::BinaryOp {
        op: *op,
        left: Box::new(l),
        right: Box::new(r),
      }
    }
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(super::together::together_expr(operand)),
    },
    Expr::FunctionCall { name, args } if name == "Plus" || name == "Times" => {
      let new_args: Vec<Expr> =
        args.iter().map(super::together::together_expr).collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args,
      }
    }
    _ => expr.clone(),
  }
}

/// FullSimplify[expr] or FullSimplify[expr, assum]
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
    let results: Vec<Expr> = items
      .iter()
      .map(|e| apply_active_assumptions(&full_simplify_expr_with_together(e)))
      .collect();
    return Ok(Expr::List(results));
  }
  let simplified = full_simplify_expr_with_together(&args[0]);
  Ok(apply_active_assumptions(&simplified))
}

/// Full-simplify variant that also tries Together (whole and sub-expression)
/// and picks the simpler form.
fn full_simplify_expr_with_together(expr: &Expr) -> Expr {
  let simplified = full_simplify_expr(expr);
  let mut best = simplified.clone();
  let mut best_c = leaf_count(&best);

  let togethered = super::together::together_expr(&simplified);
  let tc = leaf_count(&togethered);
  if tc < best_c {
    let resimplified = full_simplify_expr(&togethered);
    let rc = leaf_count(&resimplified);
    if rc <= tc {
      best = resimplified;
      best_c = rc;
    } else {
      best = togethered;
      best_c = tc;
    }
  }

  let sub_togethered = together_subexpressions(&simplified);
  let sc = leaf_count(&sub_togethered);
  if sc < best_c {
    best = sub_togethered;
    let _ = best_c;
  }

  best
}

/// Retrieve the current `$Assumptions` from the environment, if any, as an Expr.
/// Returns None when `$Assumptions` is unset or equals `True`.
fn current_assumptions() -> Option<Expr> {
  let assumptions_str = crate::ENV.with(|e| {
    e.borrow().get("$Assumptions").map(|sv| match sv {
      crate::StoredValue::Raw(s) => s.clone(),
      crate::StoredValue::ExprVal(e) => expr_to_string(e),
      _ => "True".to_string(),
    })
  })?;
  if assumptions_str == "True" || assumptions_str.is_empty() {
    return None;
  }
  let parsed = crate::syntax::string_to_expr(&assumptions_str).ok()?;
  crate::evaluator::evaluate_expr_to_expr(&parsed).ok()
}

/// Apply any currently active `$Assumptions` (set by `Assuming[...]` or
/// `Simplify[expr, assum]`) to the given already-simplified expression by
/// running it through `refine_expr`. Returns the original expression unchanged
/// when no assumptions are active.
fn apply_active_assumptions(expr: &Expr) -> Expr {
  if let Some(assumption_expr) = current_assumptions() {
    let info = extract_assumption_info(&assumption_expr);
    refine_expr(expr, &info, &assumption_expr)
  } else {
    expr.clone()
  }
}

/// Apply Simplify or FullSimplify with an explicit assumption argument.
///
/// Accepts either the direct form `Simplify[expr, assum]` (where `assum` is a
/// predicate like `x > 0`) or the option form `Simplify[expr, Assumptions -> assum]`.
/// The assumption is combined with any existing `$Assumptions` (e.g. set by a
/// surrounding `Assuming[...]`) using `And`, so nested assumptions accumulate.
fn simplify_with_assumptions(
  expr: &Expr,
  opts: &Expr,
  full: bool,
) -> Result<Expr, InterpreterError> {
  // Extract the assumption: either from `Assumptions -> assum` or taken directly.
  let assumption_val = match opts {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Assumptions") => {
      replacement.as_ref().clone()
    }
    _ => opts.clone(),
  };

  // Combine with any already-active $Assumptions (e.g. from an outer Assuming)
  // so `Assuming[x > 0, Simplify[expr, y > 0]]` uses both.
  let combined = if let Some(prev_assum) = current_assumptions() {
    Expr::FunctionCall {
      name: "And".to_string(),
      args: vec![prev_assum, assumption_val.clone()],
    }
  } else {
    assumption_val.clone()
  };

  // Save previous $Assumptions
  let prev = crate::ENV.with(|e| e.borrow().get("$Assumptions").cloned());

  // Set $Assumptions to the combined expression so any nested Simplify/Refine
  // calls inside the expression also see it.
  let val = expr_to_string(&combined);
  crate::ENV.with(|e| {
    e.borrow_mut()
      .insert("$Assumptions".to_string(), crate::StoredValue::Raw(val))
  });

  let simplified = if full {
    full_simplify_expr_with_together(expr)
  } else {
    simplify_expr_with_together(expr)
  };

  // Apply refinement using the combined assumption.
  let info = extract_assumption_info(&combined);
  let result = refine_expr(&simplified, &info, &combined);

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

  // Try partial factoring by variable connectivity. Splits a sum into
  // variable-disjoint groups and factors each group separately. This is what
  // turns `1 + c^2 + 2*c*d + d^2` into `1 + (c + d)^2`.
  if let Some(pf) = try_partial_factor_components(&trig_simplified) {
    let c = leaf_count(&pf);
    if c < best_complexity {
      best = pf;
      best_complexity = c;
    }
  }

  // Try Collect[expr, v] for each free variable, recursively full-simplifying
  // each collected coefficient. This produces compact nested forms like
  // `(a + b)^2 + 2*(a + b)*x + (1 + (c + d)^2)*x^2` for polynomials in x.
  // Skipped at greater depth to keep the combinatorial blow-up bounded — each
  // level multiplies work by the number of free variables.
  let cur_depth = FULL_SIMPLIFY_DEPTH.with(|d| d.get());
  if cur_depth < MAX_COLLECT_SIMPLIFY_DEPTH
    && let Some(cs) = try_collect_recursive_simplify(&trig_simplified)
  {
    let c = leaf_count(&cs);
    if c < best_complexity {
      best = cs;
      best_complexity = c;
    }
  }

  // If `best` is a Times that contains a Plus factor (e.g. `y * big_sum`),
  // recursively full-simplify the inner sum so that nested factoring kicks in.
  // This is cheap (at most one recursive call per factor) so we always run it,
  // bounded by the overall `MAX_FULL_SIMPLIFY_DEPTH` guard.
  let inner_simplified = simplify_inside_times(&best);
  {
    let c = leaf_count(&inner_simplified);
    if c < best_complexity {
      best = inner_simplified;
      best_complexity = c;
    }
  }

  let _ = best_complexity; // suppress unused warning
  best
}

// ─── Recursion guard for nested full_simplify ──────────────────────────────

thread_local! {
  static FULL_SIMPLIFY_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

const MAX_FULL_SIMPLIFY_DEPTH: usize = 3;

/// Beyond this depth we disable the expensive `Collect[...]`-based candidate
/// in `full_simplify_expr`, only relying on the cheaper Factor/partial-factor
/// paths. Empirically one level is enough to reach the nested factored form
/// for most practical inputs while keeping the combinatorial blow-up bounded.
const MAX_COLLECT_SIMPLIFY_DEPTH: usize = 1;

/// Run `full_simplify_expr` while incrementing the recursion-depth counter.
/// Returns `None` if the depth limit has been reached, so callers can fall
/// back to leaving the sub-expression alone.
fn full_simplify_recursive(expr: &Expr) -> Option<Expr> {
  let depth = FULL_SIMPLIFY_DEPTH.with(|d| d.get());
  if depth >= MAX_FULL_SIMPLIFY_DEPTH {
    return None;
  }
  FULL_SIMPLIFY_DEPTH.with(|d| d.set(depth + 1));
  let result = full_simplify_expr(expr);
  FULL_SIMPLIFY_DEPTH.with(|d| d.set(depth));
  Some(result)
}

/// Group additive terms by variable connectivity and factor each group
/// separately. Returns `Some(_)` only when at least two variable-disjoint
/// components exist *and* the result is strictly simpler than the input.
fn try_partial_factor_components(expr: &Expr) -> Option<Expr> {
  let terms = collect_additive_terms(expr);
  if terms.len() < 3 {
    return None;
  }

  // Variables for each term.
  let term_vars: Vec<std::collections::BTreeSet<String>> = terms
    .iter()
    .map(|t| {
      let mut set = std::collections::BTreeSet::new();
      collect_free_vars_simple(t, &mut set);
      set
    })
    .collect();

  // Union–find over term indices: connected if they share any variable.
  let n = terms.len();
  let mut parent: Vec<usize> = (0..n).collect();
  fn find(p: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while p[r] != r {
      r = p[r];
    }
    let mut cur = i;
    while p[cur] != r {
      let next = p[cur];
      p[cur] = r;
      cur = next;
    }
    r
  }
  for i in 0..n {
    for j in (i + 1)..n {
      if term_vars[i].is_empty() || term_vars[j].is_empty() {
        continue;
      }
      if !term_vars[i].is_disjoint(&term_vars[j]) {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri != rj {
          parent[ri] = rj;
        }
      }
    }
  }

  // Bucket terms by component root.
  let mut groups: std::collections::BTreeMap<usize, Vec<Expr>> =
    std::collections::BTreeMap::new();
  for (i, term) in terms.iter().enumerate() {
    let root = if term_vars[i].is_empty() {
      // Constants form their own singleton component.
      n + i
    } else {
      find(&mut parent, i)
    };
    groups.entry(root).or_default().push(term.clone());
  }

  if groups.len() < 2 {
    return None;
  }

  let original_complexity = leaf_count(expr);
  let mut result_parts: Vec<Expr> = Vec::new();
  for (_, group_terms) in groups {
    let group_sum = build_sum(group_terms);
    if let Ok(factored) =
      crate::functions::polynomial_ast::factor_ast(&[group_sum.clone()])
    {
      // Pick whichever is simpler for this component.
      if leaf_count(&factored) <= leaf_count(&group_sum) {
        result_parts.push(factored);
      } else {
        result_parts.push(group_sum);
      }
    } else {
      result_parts.push(group_sum);
    }
  }

  let result = crate::functions::math_ast::plus_ast(&result_parts)
    .unwrap_or_else(|_| build_sum(result_parts));
  if leaf_count(&result) < original_complexity {
    Some(result)
  } else {
    None
  }
}

/// Lightweight free-variable collector that ignores built-in constants.
fn collect_free_vars_simple(
  expr: &Expr,
  out: &mut std::collections::BTreeSet<String>,
) {
  match expr {
    Expr::Identifier(name)
      if !crate::functions::polynomial_ast::is_builtin_constant_sa(name) =>
    {
      out.insert(name.clone());
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_free_vars_simple(left, out);
      collect_free_vars_simple(right, out);
    }
    Expr::UnaryOp { operand, .. } => collect_free_vars_simple(operand, out),
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_free_vars_simple(a, out);
      }
    }
    Expr::List(items) | Expr::CompoundExpr(items) => {
      for it in items {
        collect_free_vars_simple(it, out);
      }
    }
    _ => {}
  }
}

/// Try `Collect[expr, v]` for each free variable `v`, recursively
/// full-simplifying each resulting coefficient. Returns the simplest variant
/// that strictly improves on `expr`'s leaf count.
fn try_collect_recursive_simplify(expr: &Expr) -> Option<Expr> {
  // Only meaningful for sums with multiple terms.
  let terms = collect_additive_terms(expr);
  if terms.len() < 2 {
    return None;
  }

  let mut vars_set = std::collections::BTreeSet::new();
  collect_free_vars_simple(expr, &mut vars_set);
  if vars_set.len() < 2 {
    return None;
  }

  let original = leaf_count(expr);
  let mut best: Option<(Expr, usize)> = None;

  for var in &vars_set {
    let collected = match crate::functions::polynomial_ast::collect_ast(&[
      expr.clone(),
      Expr::Identifier(var.clone()),
    ]) {
      Ok(c) => c,
      Err(_) => continue,
    };

    let simplified_raw = match simplify_collected_coefficients(&collected, var)
    {
      Some(s) => s,
      None => continue,
    };
    // Pull out any common symbolic factor that's shared across all terms of
    // the collected sum without going back through `expand_and_combine`, which
    // would undo the nested factoring we just performed.
    let simplified = pull_common_factor(&simplified_raw);
    let c = leaf_count(&simplified);
    if c < original && best.as_ref().map(|(_, bc)| c < *bc).unwrap_or(true) {
      best = Some((simplified, c));
    }
  }

  best.map(|(e, _)| e)
}

/// Pull out a common multiplicative factor shared across all top-level
/// additive terms of `expr`, without re-expanding the sub-factors. This is
/// used to post-process the result of a Collect-based candidate so that e.g.
/// `y*(a+b)^2 + 2*y*(a+b)*x + y*(1+(c+d)^2)*x^2` becomes
/// `((a+b)^2 + 2*(a+b)*x + (1+(c+d)^2)*x^2)*y`.
fn pull_common_factor(expr: &Expr) -> Expr {
  let terms = collect_additive_terms(expr);
  if terms.len() < 2 {
    return expr.clone();
  }

  let term_factor_strs: Vec<Vec<(String, Expr)>> = terms
    .iter()
    .map(|t| {
      collect_multiplicative_factors(t)
        .into_iter()
        .filter_map(|f| {
          // Exclude pure numeric/−1 factors — those are handled elsewhere.
          if matches!(&f, Expr::Integer(_) | Expr::Real(_)) {
            return None;
          }
          if matches!(&f, Expr::UnaryOp { op: UnaryOperator::Minus, operand }
            if matches!(operand.as_ref(), Expr::Integer(_)))
          {
            return None;
          }
          Some((expr_to_string(&f), f))
        })
        .collect()
    })
    .collect();

  // Find non-numeric factor strings common to every term.
  let mut common: Vec<(String, Expr)> = Vec::new();
  for (s, e) in &term_factor_strs[0] {
    if term_factor_strs[1..]
      .iter()
      .all(|ts| ts.iter().any(|(k, _)| k == s))
      && !common.iter().any(|(k, _)| k == s)
    {
      common.push((s.clone(), e.clone()));
    }
  }

  if common.is_empty() {
    return expr.clone();
  }

  // Strip one occurrence of each common factor from each term.
  let mut stripped: Vec<Expr> = Vec::with_capacity(terms.len());
  for term in &terms {
    let mut factors = collect_multiplicative_factors(term);
    for (s, _) in &common {
      if let Some(pos) = factors.iter().position(|f| &expr_to_string(f) == s) {
        factors.remove(pos);
      }
    }
    let new_term = if factors.is_empty() {
      Expr::Integer(1)
    } else if factors.len() == 1 {
      factors.into_iter().next().unwrap()
    } else {
      build_product(factors)
    };
    stripped.push(new_term);
  }

  let stripped_sum = crate::functions::math_ast::plus_ast(&stripped)
    .unwrap_or_else(|_| build_sum(stripped));
  let common_expr = if common.len() == 1 {
    common.into_iter().next().unwrap().1
  } else {
    build_product(common.into_iter().map(|(_, e)| e).collect())
  };

  // Build final product: (common) * (remaining sum).
  Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(stripped_sum),
    right: Box::new(common_expr),
  }
}

/// Walk the result of `Collect[expr, var]` and full-simplify each coefficient
/// (the part of each additive term that doesn't depend on `var`).
fn simplify_collected_coefficients(
  collected: &Expr,
  var: &str,
) -> Option<Expr> {
  let terms = collect_additive_terms(collected);

  // Group terms by power-of-var first, summing the per-term coefficients,
  // so that all `x^0` parts of the collected expression get full-simplified
  // together rather than each leaf in isolation.
  let mut power_groups: Vec<(i128, Vec<Expr>)> = Vec::new();
  for term in &terms {
    let (power, coeff) = term_var_power_and_coeff(term, var);
    if power < 0 {
      // Sentinel: term has a complex var-dependence. Bail out.
      return None;
    }
    if let Some(entry) = power_groups.iter_mut().find(|(p, _)| *p == power) {
      entry.1.push(coeff);
    } else {
      power_groups.push((power, vec![coeff]));
    }
  }
  power_groups.sort_by_key(|(p, _)| *p);

  let mut new_terms: Vec<Expr> = Vec::with_capacity(power_groups.len());
  for (power, coeffs) in power_groups {
    let summed_coeff = if coeffs.len() == 1 {
      coeffs.into_iter().next().unwrap()
    } else {
      build_sum(coeffs)
    };
    let simplified_coeff =
      full_simplify_recursive(&summed_coeff).unwrap_or(summed_coeff);
    let var_part: Option<Expr> = match power {
      0 => None,
      1 => Some(Expr::Identifier(var.to_string())),
      _ => Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier(var.to_string())),
        right: Box::new(Expr::Integer(power)),
      }),
    };
    let new_term = match (simplified_coeff, var_part) {
      (c, None) => c,
      (Expr::Integer(1), Some(v)) => v,
      (Expr::Integer(-1), Some(v)) => Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(v),
      },
      (c, Some(v)) => multiply_exprs(&c, &v),
    };
    new_terms.push(new_term);
  }

  Some(
    crate::functions::math_ast::plus_ast(&new_terms)
      .unwrap_or_else(|_| build_sum(new_terms)),
  )
}

/// If `expr` is `factor * Plus[...]` (or `Times[..., Plus[...]]`),
/// recursively full-simplify the inner Plus and rebuild the product.
fn simplify_inside_times(expr: &Expr) -> Expr {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let new_left = simplify_plus_factor(left);
      let new_right = simplify_plus_factor(right);
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(new_left),
        right: Box::new(new_right),
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let new_args: Vec<Expr> = args.iter().map(simplify_plus_factor).collect();
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: new_args,
      }
    }
    _ => expr.clone(),
  }
}

fn simplify_plus_factor(expr: &Expr) -> Expr {
  if is_plus_expr(expr)
    && let Some(simpler) = full_simplify_recursive(expr)
    && leaf_count(&simpler) < leaf_count(expr)
  {
    return simpler;
  }
  expr.clone()
}

fn is_plus_expr(expr: &Expr) -> bool {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus,
      ..
    } => true,
    Expr::FunctionCall { name, .. } if name == "Plus" => true,
    _ => false,
  }
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
        best_c = c;
      }
      // Try trig polynomial simplification (Pythagorean sub + power reduction)
      if let Some(trig_reduced) = try_trig_polynomial_simplify(&best) {
        let c = leaf_count(&trig_reduced);
        if c < best_c {
          best = trig_reduced;
        }
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
          best_c = c;
        }
        // Try trig polynomial simplification
        if let Some(trig_reduced) = try_trig_polynomial_simplify(&best) {
          let c = leaf_count(&trig_reduced);
          if c < best_c {
            best = trig_reduced;
          }
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
/// Try to simplify trig ratios:
/// Sin[x]/Cos[x] → Tan[x], Cos[x]/Sin[x] → Cot[x],
/// 1/Sin[x] → Csc[x], 1/Cos[x] → Sec[x], 1/Tan[x] → Cot[x]
fn try_simplify_trig_ratio(num: &Expr, den: &Expr) -> Option<Expr> {
  // Both numerator and denominator are trig functions with same arg
  if let Expr::FunctionCall {
    name: n_name,
    args: n_args,
  } = num
    && let Expr::FunctionCall {
      name: d_name,
      args: d_args,
    } = den
    && n_args.len() == 1
    && d_args.len() == 1
    && expr_to_string(&n_args[0]) == expr_to_string(&d_args[0])
  {
    let result_name = match (n_name.as_str(), d_name.as_str()) {
      ("Sin", "Cos") => Some("Tan"),
      ("Cos", "Sin") => Some("Cot"),
      _ => None,
    };
    if let Some(name) = result_name {
      return Some(Expr::FunctionCall {
        name: name.to_string(),
        args: vec![n_args[0].clone()],
      });
    }
  }

  // 1/Sin[x] → Csc[x], 1/Cos[x] → Sec[x], 1/Tan[x] → Cot[x]
  if matches!(num, Expr::Integer(1))
    && let Expr::FunctionCall { name, args } = den
    && args.len() == 1
  {
    let result_name = match name.as_str() {
      "Sin" => Some("Csc"),
      "Cos" => Some("Sec"),
      "Tan" => Some("Cot"),
      _ => None,
    };
    if let Some(rname) = result_name {
      return Some(Expr::FunctionCall {
        name: rname.to_string(),
        args: vec![args[0].clone()],
      });
    }
  }

  None
}

pub fn simplify_division(num: &Expr, den: &Expr) -> Expr {
  // If same expression, return 1
  if expr_to_string(num) == expr_to_string(den) {
    return Expr::Integer(1);
  }

  // Trig ratio simplification: Sin[x]/Cos[x] → Tan[x], Cos[x]/Sin[x] → Cot[x], etc.
  if let Some(result) = try_simplify_trig_ratio(num, den) {
    return result;
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
  let basic = if let Ok(result) =
    crate::functions::math_ast::divide_two(&num_expanded, &den_expanded)
  {
    result
  } else {
    crate::functions::math_ast::make_divide(num_expanded, den_expanded)
  };

  // Try factoring the result: Factor[p/q] = Factor[p]/Factor[q] often produces
  // a simpler form (e.g. x^2/(1-3x+3x^2-x^3) → x^2/(-1+x)^3).
  if let Ok(factored) = super::factor::factor_ast(&[basic.clone()]) {
    let fc = leaf_count(&factored);
    let bc = leaf_count(&basic);
    if fc <= bc {
      return factored;
    }
  }

  basic
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

// ─── Trig Polynomial Simplification ──────────────────────────────

/// Parse a term as coeff * Sin[arg]^a * Cos[arg]^b.
/// Returns (coeff, sin_power, cos_power, arg) or None.
fn parse_trig_monomial(term: &Expr) -> Option<(i128, i128, i128, Expr)> {
  let mut coeff: i128 = 1;
  let mut sin_pow: i128 = 0;
  let mut cos_pow: i128 = 0;
  let mut trig_arg: Option<Expr> = None;

  let (is_neg, inner) = match term {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => (true, operand.as_ref()),
    _ => (false, term),
  };
  if is_neg {
    coeff = -1;
  }

  let factors = super::expand::collect_multiplicative_factors(inner);

  for factor in &factors {
    match factor {
      Expr::Integer(n) => {
        coeff *= n;
      }
      _ => {
        let (base, exp) = super::expand::extract_base_and_exp(factor);
        let exp_val = match &exp {
          Expr::Integer(n) => *n,
          _ => return None,
        };
        match &base {
          Expr::FunctionCall { name, args }
            if args.len() == 1 && (name == "Sin" || name == "Cos") =>
          {
            if let Some(ref existing) = trig_arg {
              if expr_to_string(&args[0]) != expr_to_string(existing) {
                return None;
              }
            } else {
              trig_arg = Some(args[0].clone());
            }
            if name == "Sin" {
              sin_pow += exp_val;
            } else {
              cos_pow += exp_val;
            }
          }
          _ => return None,
        }
      }
    }
  }

  let arg = trig_arg?;
  Some((coeff, sin_pow, cos_pow, arg))
}

/// Try to simplify a sum of trig monomials by:
/// 1. Factoring out common Sin/Cos powers
/// 2. Substituting Sin²=1-Cos² to get a polynomial in Cos
/// 3. Applying TrigReduce for power reduction to multiple-angle form
/// 4. Factoring the result
pub fn try_trig_polynomial_simplify(expr: &Expr) -> Option<Expr> {
  let terms = collect_additive_terms(expr);
  if terms.len() < 2 {
    return None;
  }

  let mut trig_arg: Option<Expr> = None;
  let mut parsed: Vec<(i128, i128, i128)> = Vec::new();

  for term in &terms {
    let (c, sp, cp, a) = parse_trig_monomial(term)?;
    if let Some(ref existing) = trig_arg {
      if expr_to_string(&a) != expr_to_string(existing) {
        return None;
      }
    } else {
      trig_arg = Some(a);
    }
    parsed.push((c, sp, cp));
  }

  let trig_arg = trig_arg?;
  if parsed.len() < 2 {
    return None;
  }

  let min_sin = parsed.iter().map(|t| t.1).min()?;
  let min_cos = parsed.iter().map(|t| t.2).min()?;

  // Reduce powers by factoring out common base
  let inner: Vec<(i128, i128, i128)> = parsed
    .iter()
    .map(|&(c, s, cp)| (c, s - min_sin, cp - min_cos))
    .collect();

  // Need some remaining sin powers to substitute (otherwise nothing to do)
  let has_sin = inner.iter().any(|t| t.1 > 0);
  let has_cos = inner.iter().any(|t| t.2 > 0);
  if !has_sin && !has_cos {
    return None;
  }

  let cos_expr = Expr::FunctionCall {
    name: "Cos".to_string(),
    args: vec![trig_arg.clone()],
  };
  let sin_expr = Expr::FunctionCall {
    name: "Sin".to_string(),
    args: vec![trig_arg.clone()],
  };

  let mut best: Option<Expr> = None;
  let mut best_lc = leaf_count(expr);

  // Try substituting Sin²=1-Cos² if all remaining sin powers are even
  let all_sin_even = inner.iter().all(|t| t.1 % 2 == 0);
  if all_sin_even
    && has_sin
    && let Some(result) = try_trig_sub_and_reduce(
      &inner, &cos_expr, &sin_expr, true, min_sin, min_cos,
    )
  {
    let lc = leaf_count(&result);
    if lc < best_lc {
      best = Some(result);
      best_lc = lc;
    }
  }

  // Try substituting Cos²=1-Sin² if all remaining cos powers are even
  let all_cos_even = inner.iter().all(|t| t.2 % 2 == 0);
  if all_cos_even
    && has_cos
    && let Some(result) = try_trig_sub_and_reduce(
      &inner, &cos_expr, &sin_expr, false, min_sin, min_cos,
    )
  {
    let lc = leaf_count(&result);
    if lc < best_lc {
      best = Some(result);
      #[allow(unused_assignments)]
      {
        best_lc = lc;
      }
    }
  }

  best
}

/// Compute binomial coefficient C(n, k).
fn binom(n: i128, k: i128) -> i128 {
  if k < 0 || k > n {
    return 0;
  }
  let k = k.min(n - k);
  let mut result = 1i128;
  for i in 0..k {
    result = result * (n - i) / (i + 1);
  }
  result
}

fn integer_gcd(mut a: i128, mut b: i128) -> i128 {
  a = a.abs();
  b = b.abs();
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Perform Pythagorean substitution and power reduction using integer arithmetic.
/// If `sub_sin`: substitute Sin²=1-Cos², producing polynomial in Cos.
/// If `!sub_sin`: substitute Cos²=1-Sin², producing polynomial in Sin.
fn try_trig_sub_and_reduce(
  inner: &[(i128, i128, i128)],
  cos_expr: &Expr,
  sin_expr: &Expr,
  sub_sin: bool,
  min_sin: i128,
  min_cos: i128,
) -> Option<Expr> {
  use std::collections::HashMap;

  // Step 1: Substitute sin²=1-cos² (or cos²=1-sin²) and build polynomial in kept trig function.
  // For each term: coeff * (1-kept²)^(sub_pow/2) * kept^keep_pow
  // = coeff * sum_{j=0}^{sub_pow/2} C(sub_pow/2, j) * (-1)^j * kept^(2j + keep_pow)
  let mut cos_poly: HashMap<i128, i128> = HashMap::new();

  for &(coeff, sin_pow, cos_pow) in inner {
    let (sub_half, keep_pow) = if sub_sin {
      (sin_pow / 2, cos_pow)
    } else {
      (cos_pow / 2, sin_pow)
    };

    for j in 0..=sub_half {
      let sign = if j % 2 == 0 { 1i128 } else { -1 };
      let binom_val = binom(sub_half, j);
      let power = 2 * j + keep_pow;
      let contrib = coeff * sign * binom_val;
      *cos_poly.entry(power).or_insert(0) += contrib;
    }
  }

  // Remove zero coefficients
  cos_poly.retain(|_, v| *v != 0);

  if cos_poly.is_empty() {
    return None;
  }

  // Check all remaining powers are even (required for clean power reduction)
  let all_even = cos_poly.keys().all(|&p| p % 2 == 0);
  if !all_even {
    return None;
  }

  // Step 2: Apply power reduction formulas with integer arithmetic.
  // For even power n: trig^n = (1/2^n) * [C(n,n/2) + 2*sum_{k=0}^{n/2-1} C(n,k)*cos((n-2k)*arg)]
  // We use a common denominator: 2^max_power

  let max_power = *cos_poly.keys().max()?;
  if max_power == 0 {
    // Just a constant — no trig reduction needed
    let c = *cos_poly.get(&0)?;
    // Build result with outer factors
    return build_outer_result(
      c,
      1,
      &HashMap::new(),
      cos_expr,
      sin_expr,
      min_sin,
      min_cos,
      sub_sin,
    );
  }

  let common_denom = 1i128.checked_shl(max_power as u32)?;

  // Accumulate: multi_angle → integer_numerator (with common_denom)
  let mut angle_coeffs: HashMap<i128, i128> = HashMap::new(); // angle_multiplier → numerator

  for (&power, &coeff) in &cos_poly {
    if power == 0 {
      // Constant term: coeff * common_denom
      *angle_coeffs.entry(0).or_insert(0) += coeff * common_denom;
    } else {
      // power is even, apply reduction formula
      let n = power;
      let half_n = n / 2;
      let this_denom = 1i128.checked_shl(n as u32)?;
      let scale = common_denom / this_denom;

      // Constant contribution: C(n, n/2) * scale
      let const_binom = binom(n, half_n);
      *angle_coeffs.entry(0).or_insert(0) += coeff * const_binom * scale;

      // Cos[(n-2k)*arg] contributions
      for k in 0..half_n {
        let angle_mult = n - 2 * k;
        let binom_val = binom(n, k);
        *angle_coeffs.entry(angle_mult).or_insert(0) +=
          coeff * 2 * binom_val * scale;
      }
    }
  }

  // Remove zero coefficients
  angle_coeffs.retain(|_, v| *v != 0);
  if angle_coeffs.is_empty() {
    return None;
  }

  // Step 3: Simplify - find GCD of all numerators and denominator
  let mut g = common_denom;
  for &v in angle_coeffs.values() {
    g = integer_gcd(g, v);
  }
  let final_denom = common_denom / g;

  // Divide all numerators by g
  let simplified_coeffs: HashMap<i128, i128> =
    angle_coeffs.iter().map(|(&k, &v)| (k, v / g)).collect();

  // Factor out GCD of all simplified numerators
  let mut num_gcd = 0i128;
  for &v in simplified_coeffs.values() {
    num_gcd = integer_gcd(num_gcd, v);
  }
  if num_gcd == 0 {
    return None;
  }

  let factored_coeffs: HashMap<i128, i128> = simplified_coeffs
    .iter()
    .map(|(&k, &v)| (k, v / num_gcd))
    .collect();

  build_outer_result(
    num_gcd,
    final_denom,
    &factored_coeffs,
    cos_expr,
    sin_expr,
    min_sin,
    min_cos,
    sub_sin,
  )
}

/// Build the final result expression from the factored trig polynomial.
fn build_outer_result(
  num_factor: i128,
  denom: i128,
  angle_coeffs: &std::collections::HashMap<i128, i128>,
  cos_expr: &Expr,
  sin_expr: &Expr,
  min_sin: i128,
  min_cos: i128,
  sub_sin: bool,
) -> Option<Expr> {
  // Extract the trig argument from cos_expr
  let trig_arg = match cos_expr {
    Expr::FunctionCall { args, .. } if !args.is_empty() => &args[0],
    _ => return None,
  };

  // The trig function used for multiple-angle terms
  let multi_angle_fn = if sub_sin { "Cos" } else { "Sin" };

  // Build the inner sum: sum of angle_coeff * Cos/Sin[mult*arg]
  let mut sum_terms: Vec<Expr> = Vec::new();

  // Sort angles for deterministic output (constant first, then ascending)
  let mut angles: Vec<i128> = angle_coeffs.keys().cloned().collect();
  angles.sort();

  for &angle_mult in &angles {
    let coeff = angle_coeffs[&angle_mult];
    if coeff == 0 {
      continue;
    }

    if angle_mult == 0 {
      // Constant term
      sum_terms.push(Expr::Integer(coeff));
    } else {
      // Build Cos[mult*arg] or Sin[mult*arg]
      let angle_arg = if angle_mult == 1 {
        trig_arg.clone()
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(angle_mult)),
          right: Box::new(trig_arg.clone()),
        }
      };
      let trig_call = Expr::FunctionCall {
        name: multi_angle_fn.to_string(),
        args: vec![angle_arg],
      };
      let term = if coeff == 1 {
        trig_call
      } else if coeff == -1 {
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(trig_call),
        }
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(coeff)),
          right: Box::new(trig_call),
        }
      };
      sum_terms.push(term);
    }
  }

  // Build the inner sum
  let inner = if sum_terms.is_empty() {
    Expr::Integer(1)
  } else if sum_terms.len() == 1 && angle_coeffs.len() == 1 {
    sum_terms.into_iter().next().unwrap()
  } else {
    super::expand::build_sum(sum_terms)
  };

  // Build the numeric factor: num_factor / denom
  let numeric = if denom == 1 {
    if num_factor == 1 {
      None
    } else {
      Some(Expr::Integer(num_factor))
    }
  } else {
    Some(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(num_factor), Expr::Integer(denom)],
    })
  };

  // Assemble: numeric * inner * Sin[x]^min_sin * Cos[x]^min_cos
  let mut factors: Vec<Expr> = Vec::new();

  if let Some(n) = numeric {
    factors.push(n);
  }

  // Wrap inner in parens by keeping it as a sum
  factors.push(inner);

  if min_sin > 0 {
    let outer_sin = if min_sin == 1 {
      sin_expr.clone()
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(sin_expr.clone()),
        right: Box::new(Expr::Integer(min_sin)),
      }
    };
    factors.push(outer_sin);
  }

  if min_cos > 0 {
    let outer_cos = if min_cos == 1 {
      cos_expr.clone()
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(cos_expr.clone()),
        right: Box::new(Expr::Integer(min_cos)),
      }
    };
    factors.push(outer_cos);
  }

  let result = super::expand::build_product(factors);

  // Evaluate to canonical form
  crate::evaluator::evaluate_expr_to_expr(&result).ok()
}
