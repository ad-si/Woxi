//! AST-native calculus functions (D, Integrate).
//!
//! These functions work directly with `Expr` AST nodes for symbolic differentiation
//! and integration.

use crate::InterpreterError;
use crate::functions::math_ast::{gcd_i128, is_sqrt, make_sqrt};
use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, unevaluated,
};

thread_local! {
  /// Active `NonConstants` context for `D`. Holds the symbol names that must be
  /// treated as depending on the differentiation variable, together with the
  /// original `NonConstants -> {…}` option rule used to render the carried
  /// `D[a, x, NonConstants -> {…}]` derivative terms.
  static NON_CONSTANTS: std::cell::RefCell<Option<(Vec<String>, Expr)>> =
    const { std::cell::RefCell::new(None) };
}

/// Run `f` with the given `NonConstants` context installed, restoring the
/// previous context afterwards.
fn with_non_constants<F, R>(names: Vec<String>, rule: Expr, f: F) -> R
where
  F: FnOnce() -> R,
{
  let prev = NON_CONSTANTS.with(|nc| nc.borrow_mut().replace((names, rule)));
  let result = f();
  NON_CONSTANTS.with(|nc| *nc.borrow_mut() = prev);
  result
}

/// D[expr, var] or D[expr, {var, n}] or D[expr, x, y, ...] - Symbolic differentiation
pub fn d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "D expects at least 2 arguments".into(),
    ));
  }

  // Trailing option rules (e.g. `NonConstants -> {a}`) must not be treated as
  // extra differentiation variables. wolframscript: `D[x^2, x,
  // NonConstants -> {a}]` is `2 x`, not `0`.
  fn option_name(e: &Expr) -> Option<String> {
    if let Expr::Rule { pattern, .. } = e
      && let Expr::Identifier(s) = pattern.as_ref()
    {
      Some(s.clone())
    } else {
      None
    }
  }
  if args[1..].iter().any(|a| option_name(a).is_some()) {
    let expr = &args[0];
    let unevaluated = || unevaluated("D", args);
    let mut diff_vars: Vec<Expr> = Vec::new();
    let mut non_constants: Vec<Expr> = Vec::new();
    for a in &args[1..] {
      match option_name(a).as_deref() {
        Some("NonConstants") => {
          if let Expr::Rule { replacement, .. } = a {
            match replacement.as_ref() {
              Expr::List(vs) => non_constants.extend(vs.iter().cloned()),
              other => non_constants.push(other.clone()),
            }
          }
        }
        // NonConstants is D's only option. Any other option is unknown, so
        // wolframscript emits `D::optx` and leaves the call unevaluated.
        Some(opt) => {
          crate::emit_message(&format!(
            "D::optx: Unknown option {} in {}.",
            opt,
            crate::syntax::expr_to_output(&unevaluated())
          ));
          return Ok(unevaluated());
        }
        None => diff_vars.push(a.clone()),
      }
    }
    if diff_vars.is_empty() {
      return Ok(unevaluated());
    }
    // Differentiate against the real variables, treating every NonConstants
    // symbol as dependent on the differentiation variable. Its derivative is
    // carried symbolically as `D[a, x, NonConstants -> {…}]`, e.g.
    // `D[a x^2, x, NonConstants -> {a}]` = `2 a x + x^2 D[a, x, NonConstants -> {a}]`.
    let names: Vec<String> = non_constants
      .iter()
      .filter_map(|nc| match nc {
        Expr::Identifier(n) => Some(n.clone()),
        _ => None,
      })
      .collect();
    let rule = Expr::Rule {
      pattern: Box::new(Expr::Identifier("NonConstants".to_string())),
      replacement: Box::new(Expr::List(non_constants.into())),
    };
    let mut new_args = vec![expr.clone()];
    new_args.extend(diff_vars);
    return with_non_constants(names, rule, || d_ast(&new_args));
  }

  // D[expr, x, y, ...] — mixed partial derivatives: differentiate sequentially
  if args.len() > 2 {
    // First differentiate with respect to the last variable
    let inner = d_ast(&[args[0].clone(), args[1].clone()])?;
    // Then differentiate the result with respect to remaining variables
    let mut remaining = vec![inner];
    remaining.extend_from_slice(&args[2..]);
    return d_ast(&remaining);
  }

  // Thread over lists in the first argument: D[{f1, f2, ...}, var] -> {D[f1, var], D[f2, var], ...}
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|item| d_ast(&[item.clone(), args[1].clone()]))
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  // Handle D[expr, {{x, y, ...}}] — gradient/Jacobian
  if let Expr::List(outer_items) = &args[1]
    && outer_items.len() == 1
    && let Expr::List(vars) = &outer_items[0]
  {
    if let Expr::List(_) = &args[0] {
      // D[{f1, f2, ...}, {{x, y, ...}}] → Jacobian matrix
      let expr_list = match &args[0] {
        Expr::List(items) => items.clone(),
        _ => unreachable!(),
      };
      let mut rows = Vec::new();
      for f in &expr_list {
        let mut row = Vec::new();
        for v in vars {
          row.push(d_ast(&[f.clone(), v.clone()])?);
        }
        rows.push(Expr::List(row.into()));
      }
      return Ok(Expr::List(rows.into()));
    } else {
      // D[f, {{x, y, ...}}] → gradient vector
      let mut result = Vec::new();
      for v in vars {
        result.push(d_ast(&[args[0].clone(), v.clone()])?);
      }
      return Ok(Expr::List(result.into()));
    }
  }

  // Handle D[expr, {{x, y, ...}, n}] — n-th order derivative tensor.
  // Recursively: result[i_1][i_2]…[i_n] = D[expr, v_{i_1}, …, v_{i_n}].
  // Base case n == 0 returns `expr` itself; n == 1 yields the gradient.
  if let Expr::List(outer_items) = &args[1]
    && outer_items.len() == 2
    && let Expr::List(vars) = &outer_items[0]
    && let Expr::Integer(n_val) = &outer_items[1]
    && *n_val >= 0
  {
    let n = *n_val as usize;
    if n == 0 {
      return Ok(args[0].clone());
    }
    let mut rows = Vec::with_capacity(vars.len());
    let inner_spec = Expr::List(
      vec![Expr::List(vars.clone()), Expr::Integer((n - 1) as i128)].into(),
    );
    for v in vars {
      let first = d_ast(&[args[0].clone(), v.clone()])?;
      let nested = d_ast(&[first, inner_spec.clone()])?;
      rows.push(nested);
    }
    return Ok(Expr::List(rows.into()));
  }

  // If args[1] is a 2-element List that isn't a valid {var, n} (n must
  // be a non-negative integer), return unevaluated — matching
  // wolframscript for forms like D[expr, {x, y}] with symbolic y.
  if let Expr::List(items) = &args[1]
    && items.len() == 2
    && !matches!(&items[1], Expr::Integer(n) if *n >= 0)
  {
    return Ok(unevaluated("D", args));
  }

  // Handle D[expr, {var, n}] for higher-order derivatives.
  if let Expr::List(items) = &args[1]
    && items.len() == 2
    && let Expr::Integer(n_val) = &items[1]
    && *n_val >= 0
  {
    let n = *n_val as usize;
    // Identifier variable — symbolic differentiation.
    if let Expr::Identifier(var_name) = &items[0] {
      let mut result = args[0].clone();
      for _ in 0..n {
        result = differentiate(&result, var_name)?;
        result = simplify(result);
      }
      return Ok(result);
    }
    // Slot variable: replace `Slot[k]` with a fresh symbol, run the
    // standard chain-rule differentiation, then put the slot back.
    if let Expr::Slot(_) = &items[0] {
      let fresh = "$__DSlotVar__";
      let body = replace_subexpr_simple(
        &args[0],
        &items[0],
        &Expr::Identifier(fresh.to_string()),
      );
      let mut result = body;
      for _ in 0..n {
        result = differentiate(&result, fresh)?;
        result = simplify(result);
      }
      result = replace_subexpr_simple(
        &result,
        &Expr::Identifier(fresh.to_string()),
        &items[0],
      );
      return Ok(result);
    }
    // Non-symbol variable specifier (e.g. x[k]) — apply
    // differentiate_wrt_expr n times.
    let mut result = args[0].clone();
    for _ in 0..n {
      result = differentiate_wrt_expr(&result, &items[0])?;
      result = simplify(result);
    }
    return Ok(result);
  }

  // Slot variable: same fresh-symbol trick as the {Slot, n} branch.
  if let Expr::Slot(_) = &args[1] {
    let fresh = "$__DSlotVar__";
    let body = replace_subexpr_simple(
      &args[0],
      &args[1],
      &Expr::Identifier(fresh.to_string()),
    );
    let result = differentiate(&body, fresh)?;
    let result = simplify(result);
    return Ok(replace_subexpr_simple(
      &result,
      &Expr::Identifier(fresh.to_string()),
      &args[1],
    ));
  }

  // Get the variable name
  let var_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    // For indexed variables like x[k], differentiate with respect to the full expression
    // This treats x[k] as an atomic unit — D[x[i], x[k]] = 0 for symbolic i, k
    other => {
      return differentiate_wrt_expr(&args[0], other);
    }
  };

  // Differentiate the expression and cancel common factors in fractions
  let result = differentiate(&args[0], &var_name)?;
  // Only apply Cancel when the result is a fraction to avoid expanding
  // products. Keep the raw quotient sign: wolframscript's D does not
  // canonicalize (D[ArcCoth[x^2], x] stays (2*x)/(1 - x^4)).
  let (_, den) =
    crate::functions::polynomial_ast::together::extract_num_den(&result);
  if !matches!(&den, Expr::Integer(1)) {
    Ok(
      crate::functions::polynomial_ast::cancel_expr_keep_quotient_sign(&result),
    )
  } else {
    Ok(result)
  }
}

/// Recursively replace every occurrence of `target` in `expr` with `replacement`.
/// Used by D[expr, {Slot[k], n}] to swap a slot for a fresh symbolic variable.
fn replace_subexpr_simple(
  expr: &Expr,
  target: &Expr,
  replacement: &Expr,
) -> Expr {
  use crate::evaluator::pattern_matching::expr_equal;
  if expr_equal(expr, target) {
    return replacement.clone();
  }
  match expr {
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| replace_subexpr_simple(a, target, replacement))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| replace_subexpr_simple(a, target, replacement))
        .collect(),
    ),
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(replace_subexpr_simple(left, target, replacement)),
      right: Box::new(replace_subexpr_simple(right, target, replacement)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(replace_subexpr_simple(operand, target, replacement)),
    },
    Expr::CurriedCall { func, args } => Expr::CurriedCall {
      func: Box::new(replace_subexpr_simple(func, target, replacement)),
      args: args
        .iter()
        .map(|a| replace_subexpr_simple(a, target, replacement))
        .collect(),
    },
    _ => expr.clone(),
  }
}

/// Differentiate an expression with respect to a non-symbol expression (e.g., x[k]).
/// For symbolic indexed variables, D[f[x[i]], x[k]] = 0 when we can't determine equality.
fn differentiate_wrt_expr(
  expr: &Expr,
  var_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  // A bare number (3) or an arithmetic-compound expression (2 x, x + 1, x^2,
  // x/2, a b) is not a valid symbol to differentiate against — Wolfram emits
  // `D::ivar` and returns the call unevaluated. Symbol-headed forms such as
  // x[k] or Sin[x] stay valid and fall through.
  fn is_invalid_dvar(e: &Expr) -> bool {
    match e {
      Expr::Integer(_)
      | Expr::Real(_)
      | Expr::BigInteger(_)
      | Expr::BigFloat(_, _) => true,
      Expr::BinaryOp { op, .. } => matches!(
        op,
        BinaryOperator::Plus
          | BinaryOperator::Minus
          | BinaryOperator::Times
          | BinaryOperator::Divide
          | BinaryOperator::Power
      ),
      Expr::FunctionCall { name, .. } => matches!(
        name.as_str(),
        "Plus" | "Minus" | "Times" | "Divide" | "Power" | "Rational"
      ),
      _ => false,
    }
  }
  if is_invalid_dvar(var_expr) {
    crate::emit_message(&format!(
      "D::ivar: {} is not a valid variable.",
      crate::syntax::expr_to_message_form(var_expr)
    ));
    return Ok(Expr::FunctionCall {
      name: "D".to_string(),
      args: vec![expr.clone(), var_expr.clone()].into(),
    });
  }
  // If the expression is structurally equal to the variable, derivative is 1
  if crate::syntax::expr_to_string(expr)
    == crate::syntax::expr_to_string(var_expr)
  {
    return Ok(Expr::Integer(1));
  }
  // For products: use product rule
  if let Expr::BinaryOp {
    op: BinaryOperator::Times,
    left,
    right,
  } = expr
  {
    let dl = differentiate_wrt_expr(left, var_expr)?;
    let dr = differentiate_wrt_expr(right, var_expr)?;
    let term1 = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(dl),
      right: right.clone(),
    });
    let term2 = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: left.clone(),
      right: Box::new(dr),
    });
    return Ok(simplify(Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(term1),
      right: Box::new(term2),
    }));
  }
  // For sums
  if let Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left,
    right,
  } = expr
  {
    let dl = differentiate_wrt_expr(left, var_expr)?;
    let dr = differentiate_wrt_expr(right, var_expr)?;
    return Ok(simplify(Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(dl),
      right: Box::new(dr),
    }));
  }
  // For FunctionCall with Plus/Times
  if let Expr::FunctionCall { name, args } = expr {
    if name == "Times" && args.len() >= 2 {
      // Product of multiple terms
      let mut result_terms = Vec::new();
      for (i, arg) in args.iter().enumerate() {
        let darg = differentiate_wrt_expr(arg, var_expr)?;
        if !matches!(darg, Expr::Integer(0)) {
          let mut factors = Vec::new();
          for (j, a) in args.iter().enumerate() {
            if i == j {
              factors.push(darg.clone());
            } else {
              factors.push(a.clone());
            }
          }
          result_terms.push(
            crate::functions::math_ast::times_ast(&factors).unwrap_or(
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: factors.into(),
              },
            ),
          );
        }
      }
      if result_terms.is_empty() {
        return Ok(Expr::Integer(0));
      }
      return crate::functions::math_ast::plus_ast(&result_terms).or(Ok(
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: result_terms.into(),
        },
      ));
    }
    if name == "Plus" {
      let derivs: Result<Vec<_>, _> = args
        .iter()
        .map(|a| differentiate_wrt_expr(a, var_expr))
        .collect();
      let d = derivs?;
      return crate::functions::math_ast::plus_ast(&d).or(Ok(
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: d.into(),
        },
      ));
    }
  }
  // Otherwise, treat as constant w.r.t. var_expr → 0
  Ok(Expr::Integer(0))
}

/// Integrate[expr, var] or Integrate[expr, {var, lo, hi}] - Symbolic integration
pub fn integrate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Integrate expects at least 2 arguments".into(),
    ));
  }

  // Separate trailing option rules (`Assumptions -> …`, `GenerateConditions ->
  // …`, etc.) from the integration specs so they are not mistaken for extra
  // integration variables by the multivariate path below. An `Assumptions`
  // option is honoured by evaluating the integral under the Assuming mechanism
  // (which sets `$Assumptions`, consulted by the definite-integral and limit
  // paths); any other option is accepted and ignored.
  {
    let mut assumption: Option<Expr> = None;
    let mut had_option = false;
    let mut effective: Vec<Expr> = vec![args[0].clone()];
    for arg in &args[1..] {
      if let Expr::Rule {
        pattern,
        replacement,
      } = arg
      {
        had_option = true;
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Assumptions")
        {
          assumption = Some(replacement.as_ref().clone());
        }
        continue;
      }
      effective.push(arg.clone());
    }
    if had_option {
      let inner = Expr::FunctionCall {
        name: "Integrate".to_string(),
        args: effective.into(),
      };
      return match assumption {
        Some(cond) => crate::evaluator::assuming_ast(&[cond, inner]),
        None => crate::evaluator::evaluate_expr_to_expr(&inner),
      };
    }
  }

  // Multi-variable integration: Integrate[f, spec1, spec2, ..., specN]
  // = Integrate[Integrate[f, spec2, ..., specN], spec1]
  // i.e., innermost integration variable is listed last
  if args.len() > 2 {
    let mut inner_args = vec![args[0].clone()];
    inner_args.extend_from_slice(&args[2..]);
    let inner = Expr::FunctionCall {
      name: "Integrate".to_string(),
      args: inner_args.into(),
    };
    let inner_result = crate::evaluator::evaluate_expr_to_expr(&inner)?;
    return integrate_ast(&[inner_result, args[1].clone()]);
  }

  // Check if the second argument is {var, lo, hi} (definite integral)
  if let Expr::List(items) = &args[1]
    && items.len() == 3
  {
    let var_name = match &items[0] {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Ok(unevaluated("Integrate", args));
      }
    };
    let lo = &items[1];
    let hi = &items[2];

    // Constant integrand with an infinite value: the general result is
    // sign(c)·Infinity whenever the integration interval is non-empty. The
    // antiderivative-then-substitute path fails here because `c·0` on an
    // infinite `c` evaluates to Indeterminate. Handle the signed-Infinity
    // case directly. A non-empty range is any range where lo and hi aren't
    // structurally identical (compared via their printed form).
    if is_constant_wrt(&args[0], &var_name) {
      use crate::syntax::expr_to_string;
      let lo_ne_hi = expr_to_string(lo) != expr_to_string(hi);
      // -Infinity must be returned through the evaluator so it gets
      // canonicalised to the same form `DirectedInfinity[-1]` / unary-minus
      // that the REPL displays as `-Infinity`.
      let return_neg_infinity = || -> Result<Expr, InterpreterError> {
        crate::evaluator::evaluate_expr_to_expr(&Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(Expr::Identifier("Infinity".to_string())),
        })
      };
      match &args[0] {
        Expr::Identifier(s) if s == "Infinity" && lo_ne_hi => {
          return Ok(Expr::Identifier("Infinity".to_string()));
        }
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity")
          && lo_ne_hi =>
        {
          return return_neg_infinity();
        }
        Expr::FunctionCall { name, args: da }
          if name == "DirectedInfinity"
            && da.len() == 1
            && matches!(&da[0], Expr::Integer(-1))
            && lo_ne_hi =>
        {
          return return_neg_infinity();
        }
        _ => {}
      }
      // For a finite constant integrand with finite bounds, return
      // (hi - lo) * integrand in factored form rather than expanding through
      // the antiderivative path (which yields `hi*c - lo*c`). Wolfram keeps
      // the (-lo + hi) factor. Skip when either bound is infinite — those
      // are divergent improper integrals and need the antiderivative path's
      // divergence detection.
      let bounds_finite = !is_infinity(lo)
        && !is_negative_infinity(lo)
        && !is_infinity(hi)
        && !is_negative_infinity(hi);
      if bounds_finite {
        let width = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(hi.clone()),
          right: Box::new(lo.clone()),
        };
        let product = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(width),
          right: Box::new(args[0].clone()),
        };
        return crate::evaluator::evaluate_expr_to_expr(&product);
      }
    }

    // Try known definite integrals first
    if let Some(result) = try_definite_integral(&args[0], &var_name, lo, hi) {
      return Ok(result);
    }

    // ∫_lo^hi Sqrt[a + b*x^2] dx for a monic quadratic radicand (b = ±1,
    // positive constant a): use the continuous ArcSin/ArcSinh antiderivative
    // so the closed form is exact (e.g. the semicircle area Pi/2).
    if let Some(result) =
      try_definite_sqrt_quadratic(&args[0], &var_name, lo, hi)
    {
      return Ok(result);
    }

    // Fall back: compute indefinite integral and evaluate at bounds
    if let Some(antideriv) = integrate(&args[0], &var_name) {
      let antideriv = simplify(antideriv);
      let antideriv = crate::evaluator::evaluate_expr_to_expr(&antideriv)
        .unwrap_or(antideriv);
      // F(hi) - F(lo)
      // When a boundary is ±Infinity, use Limit instead of direct substitution
      // to correctly handle indeterminate forms like 0 * Infinity.
      let at_hi = if is_infinity(hi) || is_negative_infinity(hi) {
        let limit_expr = Expr::FunctionCall {
          name: "Limit".to_string(),
          args: vec![
            antideriv.clone(),
            Expr::FunctionCall {
              name: "Rule".to_string(),
              args: vec![Expr::Identifier(var_name.clone()), hi.clone()].into(),
            },
          ]
          .into(),
        };
        crate::evaluator::evaluate_expr_to_expr(&limit_expr)?
      } else {
        let sub = crate::syntax::substitute_variable(&antideriv, &var_name, hi);
        crate::evaluator::evaluate_expr_to_expr(&sub)?
      };
      let at_lo = if is_infinity(lo) || is_negative_infinity(lo) {
        let limit_expr = Expr::FunctionCall {
          name: "Limit".to_string(),
          args: vec![
            antideriv.clone(),
            Expr::FunctionCall {
              name: "Rule".to_string(),
              args: vec![Expr::Identifier(var_name.clone()), lo.clone()].into(),
            },
          ]
          .into(),
        };
        crate::evaluator::evaluate_expr_to_expr(&limit_expr)?
      } else {
        let sub = crate::syntax::substitute_variable(&antideriv, &var_name, lo);
        crate::evaluator::evaluate_expr_to_expr(&sub)?
      };
      // Apply Horner form to polynomial factors in products,
      // matching Wolfram's canonical output for definite integrals.
      let at_hi = hornerize_product_polys(at_hi);
      let at_lo = hornerize_product_polys(at_lo);
      let result = simplify(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(at_hi),
        right: Box::new(at_lo),
      });
      // Resolve boundary terms like `0^(1 + n)` using any active assumptions
      // (e.g. from an enclosing `Assuming[n > 0, …]` / the `Assumptions` option):
      // `0^k -> 0` when `k` is provably positive. A no-op when no assumptions
      // are active. Matches wolframscript's Integrate[x^n, {x, 0, 1}] under
      // `n > 0` giving `1/(1 + n)` rather than leaving the `0^(1+n)` term.
      let result =
        crate::functions::polynomial_ast::apply_active_assumptions(&result);
      let result =
        crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result);
      // Divergent improper integral: if at least one bound is infinite and
      // the evaluated difference is non-finite (±Infinity, ComplexInfinity,
      // Indeterminate), treat the integral as divergent and return it
      // unevaluated — matching wolframscript's Integrate::idiv behaviour.
      let has_infinite_bound = is_infinity(lo)
        || is_negative_infinity(lo)
        || is_infinity(hi)
        || is_negative_infinity(hi);
      if has_infinite_bound && is_nonfinite_result(&result) {
        return Ok(unevaluated("Integrate", args));
      }
      return Ok(result);
    }

    // Return unevaluated
    return Ok(unevaluated("Integrate", args));
  }

  // Indefinite integral: Integrate[expr, var]
  let var_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(unevaluated("Integrate", args));
    }
  };

  // SeriesData integrand: integrate the truncated power series term-by-term
  // when the integration variable is the series variable. For a term
  //   c_k (x - x0)^((nmin+k)/den),
  // the antiderivative is
  //   c_k den/(nmin+k+den) (x - x0)^((nmin+k+den)/den),
  // so nmin and nmax each rise by `den` and each coefficient is scaled by
  // `den/(nmin+k+den)`. (The constant of integration is dropped, matching
  // wolframscript.) A term with exponent -1 would integrate to a logarithm,
  // which has no power-series form, so that case is left unevaluated.
  if let Expr::FunctionCall { name, args: sd } = &args[0]
    && name == "SeriesData"
    && sd.len() == 6
    && matches!(&sd[0], Expr::Identifier(v) if *v == var_name)
    && let (
      Expr::List(coeffs),
      Expr::Integer(nmin),
      Expr::Integer(nmax),
      Expr::Integer(den),
    ) = (&sd[2], &sd[3], &sd[4], &sd[5])
  {
    let (nmin, nmax, den) = (*nmin, *nmax, *den);
    // A term with exponent -1 integrates to a logarithm — but only if its
    // coefficient is actually non-zero (a zero coefficient there is harmless).
    let has_log = coeffs.iter().enumerate().any(|(k, c)| {
      nmin + k as i128 + den == 0 && !matches!(c, Expr::Integer(0))
    });
    // A malformed series (e.g. from `Series[Sqrt[x], …]`, which Woxi cannot
    // yet expand) carries non-finite coefficients; don't integrate those.
    let finite_coeffs = coeffs.iter().all(|c| {
      !matches!(c,
        Expr::Identifier(s)
          if s == "ComplexInfinity" || s == "Infinity" || s == "Indeterminate")
        && !matches!(c, Expr::FunctionCall { name, .. } if name == "DirectedInfinity")
    });
    if den > 0 && !has_log && finite_coeffs {
      let mut new_coeffs: Vec<Expr> = Vec::with_capacity(coeffs.len());
      for (k, c) in coeffs.iter().enumerate() {
        let exp_denom = nmin + k as i128 + den;
        // exp_denom == 0 is the -1-power slot; `has_log` already guaranteed
        // its coefficient is zero, so the integrated coefficient is just 0
        // (and `Rational[den, 0]` must be avoided).
        if exp_denom == 0 {
          new_coeffs.push(Expr::Integer(0));
          continue;
        }
        let factor = Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(den), Expr::Integer(exp_denom)].into(),
        };
        let term = crate::functions::math_ast::times_ast(&[factor, c.clone()])
          .unwrap_or(Expr::Integer(0));
        new_coeffs.push(simplify(term));
      }
      let result = Expr::FunctionCall {
        name: "SeriesData".to_string(),
        args: vec![
          sd[0].clone(),
          sd[1].clone(),
          Expr::List(new_coeffs.into()),
          Expr::Integer(nmin + den),
          Expr::Integer(nmax + den),
          Expr::Integer(den),
        ]
        .into(),
      };
      return crate::evaluator::evaluate_expr_to_expr(&result);
    }
  }

  // Integrate the expression
  match integrate(&args[0], &var_name) {
    Some(result) => {
      let simplified = simplify(result);
      let evaluated = crate::evaluator::evaluate_expr_to_expr(&simplified)
        .unwrap_or(simplified);
      Ok(evaluated)
    }
    None => Ok(unevaluated("Integrate", args)),
  }
}

/// Apply Horner form to polynomial factors within product expressions.
/// E.g., Times[E^a, 2 - 2a + a^2] → Times[E^a, 2 + (-2+a)*a]
/// This matches Wolfram's canonical output for definite integral results.
fn hornerize_product_polys(expr: Expr) -> Expr {
  match &expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      let new_args: Vec<Expr> = args.iter().map(try_horner_if_poly).collect();
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: new_args.into(),
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(try_horner_if_poly(left)),
      right: Box::new(try_horner_if_poly(right)),
    },
    _ => expr,
  }
}

/// If the expression is a polynomial of degree >= 2 in a single variable,
/// convert it to Horner form.
fn try_horner_if_poly(expr: &Expr) -> Expr {
  // Check if the expression is an additive (Plus) expression — only
  // those can be polynomials of degree >= 2 that benefit from Horner form.
  let is_plus = matches!(expr,
    Expr::FunctionCall { name, .. } if name == "Plus")
    || matches!(
      expr,
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        ..
      }
    );
  if !is_plus {
    return expr.clone();
  }
  // Try to apply HornerForm; it will auto-detect variables and
  // return the expression unchanged if it's not a polynomial of degree >= 2.
  match crate::functions::polynomial_ast::horner::horner_form_ast(&[
    expr.clone()
  ]) {
    Ok(ref h)
      if crate::syntax::expr_to_string(h)
        != crate::syntax::expr_to_string(expr) =>
    {
      h.clone()
    }
    _ => expr.clone(),
  }
}

/// Check if an expression represents Infinity
fn is_infinity(expr: &Expr) -> bool {
  matches!(expr, Expr::Identifier(name) if name == "Infinity")
}

/// Check if an expression represents a non-finite value — ±Infinity,
/// ComplexInfinity, DirectedInfinity[...], or Indeterminate.
fn is_nonfinite_result(expr: &Expr) -> bool {
  if is_infinity(expr) || is_negative_infinity(expr) {
    return true;
  }
  matches!(expr, Expr::Identifier(name)
    if name == "ComplexInfinity" || name == "Indeterminate")
    || matches!(expr, Expr::FunctionCall { name, .. }
      if name == "DirectedInfinity")
}

/// Check if an expression represents -Infinity (via UnaryOp::Minus or Times[-1, Infinity])
fn is_negative_infinity(expr: &Expr) -> bool {
  match expr {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_infinity(operand),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      (matches!(left.as_ref(), Expr::Integer(-1)) && is_infinity(right))
        || (matches!(right.as_ref(), Expr::Integer(-1)) && is_infinity(left))
    }
    Expr::Integer(n) if *n < 0 => false, // negative number, not -Infinity
    _ => false,
  }
}

/// Numeric value of an integration bound, mapping ±Infinity to ±f64::INFINITY.
fn bound_to_f64(e: &Expr) -> Option<f64> {
  if is_infinity(e) {
    return Some(f64::INFINITY);
  }
  if is_negative_infinity(e) {
    return Some(f64::NEG_INFINITY);
  }
  crate::functions::math_ast::try_eval_to_f64(e)
}

/// Evaluate `∫ g(x) DiracDelta[c x + d] dx` over `(lo, hi)` via the sifting
/// property. Requires the delta argument to be linear in `var` with a constant
/// nonzero coefficient. Handles a root strictly inside (→ `g(x0)/|c|`), outside
/// (→ `0`), on a boundary (→ `g(x0)/|c| * HeavisideTheta[0]`), and a symbolic
/// root over the whole real line (→ `ConditionalExpression[g(x0)/|c|, x0 ∈
/// Reals]`). Symbolic roots with finite bounds are left unevaluated.
fn try_dirac_delta_integral(
  integrand: &Expr,
  var: &str,
  lo: &Expr,
  hi: &Expr,
) -> Option<Expr> {
  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e).ok();
  let at = |e: &Expr, v: Expr| crate::syntax::substitute_variable(e, var, &v);

  // Split the integrand into the (single) DiracDelta factor and the rest.
  let factors = flatten_times_factors(integrand);
  let mut delta_arg: Option<Expr> = None;
  let mut others: Vec<Expr> = Vec::new();
  for f in &factors {
    if let Expr::FunctionCall { name, args } = f
      && name == "DiracDelta"
      && args.len() == 1
    {
      if delta_arg.is_some() {
        return None; // more than one DiracDelta — not handled
      }
      delta_arg = Some(args[0].clone());
      continue;
    }
    others.push(f.clone());
  }
  let arg = delta_arg?;

  // The argument must be linear in var: d = arg(0), c = arg(1) - arg(0).
  let d = eval(at(&arg, Expr::Integer(0)))?;
  let c_plus_d = eval(at(&arg, Expr::Integer(1)))?;
  let c = eval(Expr::FunctionCall {
    name: "Subtract".to_string(),
    args: vec![c_plus_d, d.clone()].into(),
  })?;
  // c must be a constant, nonzero number.
  let c_f = crate::functions::math_ast::try_eval_to_f64(&c)?;
  if c_f == 0.0 {
    return None;
  }
  // Confirm linearity: arg(2) - (2 c + d) must vanish.
  let arg_at_2 = eval(at(&arg, Expr::Integer(2)))?;
  let linearity_residual = eval(Expr::FunctionCall {
    name: "Subtract".to_string(),
    args: vec![
      arg_at_2,
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(2), c.clone()].into(),
          },
          d.clone(),
        ]
        .into(),
      },
    ]
    .into(),
  })?;
  if !matches!(linearity_residual, Expr::Integer(0)) {
    return None; // nonlinear delta argument
  }

  // Root x0 = -d / c.
  let root = eval(Expr::FunctionCall {
    name: "Divide".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), d].into(),
      },
      c.clone(),
    ]
    .into(),
  })?;
  // g(x): the product of the non-delta factors.
  let g = if others.is_empty() {
    Expr::Integer(1)
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: others.into(),
    }
  };
  // Sifted value g(x0)/|c|. Defined symbolically so it also works when the
  // root is symbolic.
  let g_at_root = eval(at(&g, root.clone()))?;
  let sifted = eval(Expr::FunctionCall {
    name: "Divide".to_string(),
    args: vec![
      g_at_root,
      Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![c].into(),
      },
    ]
    .into(),
  })?;

  match crate::functions::math_ast::try_eval_to_f64(&root) {
    // Numeric root: position it relative to the (numeric) bounds.
    Some(root_f) => {
      let lo_f = bound_to_f64(lo)?;
      let hi_f = bound_to_f64(hi)?;
      if root_f > lo_f && root_f < hi_f {
        // Strictly inside the interval.
        Some(sifted)
      } else if root_f < lo_f || root_f > hi_f {
        // Strictly outside: the delta never fires.
        Some(Expr::Integer(0))
      } else {
        // Root exactly on a boundary: g(x0)/|c| * HeavisideTheta[0].
        eval(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            sifted,
            Expr::FunctionCall {
              name: "HeavisideTheta".to_string(),
              args: vec![Expr::Integer(0)].into(),
            },
          ]
          .into(),
        })
      }
    }
    // Symbolic root over the whole real line: the delta always fires for a
    // real root → ConditionalExpression[g(x0)/|c|, x0 ∈ Reals].
    None if is_negative_infinity(lo) && is_infinity(hi) => {
      Some(Expr::FunctionCall {
        name: "ConditionalExpression".to_string(),
        args: vec![
          sifted,
          Expr::FunctionCall {
            name: "Element".to_string(),
            args: vec![root, Expr::Identifier("Reals".to_string())].into(),
          },
        ]
        .into(),
      })
    }
    // Symbolic root with finite bounds: position is undetermined — leave it
    // unevaluated (Wolfram returns a Piecewise/HeavisideTheta form).
    None => None,
  }
}

/// Try to evaluate a definite integral using known closed-form results
fn try_definite_integral(
  integrand: &Expr,
  var: &str,
  lo: &Expr,
  hi: &Expr,
) -> Option<Expr> {
  // Definite integral of a Piecewise: split [lo, hi] at the piece boundaries
  // and integrate each piece over the sub-interval where its condition holds.
  // The antiderivative path uses a discontinuous default-0 antiderivative, so
  // a bounded-support condition like `0 < x < 1` otherwise integrates to 0.
  if let Some(result) = try_piecewise_definite_integral(integrand, var, lo, hi)
  {
    return Some(result);
  }

  // Definite integral of Abs of a linear argument. The continuous antiderivative
  // of |u| for u = a*var + b (a a non-zero constant) is u*Abs[u]/(2 a), so
  //   ∫_lo^hi |u| dvar = F(hi) - F(lo).
  // Restricted to finite numeric bounds; symbolic bounds return a conditional in
  // wolframscript, which is out of scope here.
  if let Expr::FunctionCall { name, args } = integrand
    && name == "Abs"
    && args.len() == 1
    && is_finite_real_number(lo)
    && is_finite_real_number(hi)
    && let Ok(slope) = differentiate(&args[0], var)
  {
    let slope = crate::evaluator::evaluate_expr_to_expr(&simplify(slope))
      .unwrap_or(Expr::Integer(0));
    let slope_is_zero = matches!(&slope, Expr::Integer(0))
      || matches!(&slope, Expr::Real(f) if *f == 0.0);
    if !contains_var(&slope, var) && !slope_is_zero {
      let u = args[0].clone();
      let abs_u = Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![u.clone()].into(),
      };
      // u * Abs[u] / (2 a)
      let antideriv = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(u),
          right: Box::new(abs_u),
        }),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(2)),
          right: Box::new(slope),
        }),
      };
      let at_hi = crate::syntax::substitute_variable(&antideriv, var, hi);
      let at_lo = crate::syntax::substitute_variable(&antideriv, var, lo);
      let at_hi = crate::evaluator::evaluate_expr_to_expr(&at_hi).ok()?;
      let at_lo = crate::evaluator::evaluate_expr_to_expr(&at_lo).ok()?;
      let result = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(at_hi),
        right: Box::new(at_lo),
      };
      return Some(
        crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result),
      );
    }
  }

  // DiracDelta sifting property:
  //   ∫_lo^hi g(x) DiracDelta[c x + d] dx = g(x0)/|c|  when lo < x0 < hi,
  // 0 when x0 is strictly outside [lo, hi], g(x0)/|c| * HeavisideTheta[0] when
  // x0 lands on a boundary, and ConditionalExpression[g(x0)/|c|, x0 ∈ Reals]
  // for a symbolic root over (-∞, ∞), where x0 = -d/c is the root of the
  // (linear) delta argument.
  if let Some(result) = try_dirac_delta_integral(integrand, var, lo, hi) {
    return Some(result);
  }

  // Gaussian integral: ∫_{-∞}^{∞} E^(-a*x^2) dx = Sqrt[Pi/a]
  if is_negative_infinity(lo)
    && is_infinity(hi)
    && let Some(coeff) = match_gaussian(integrand, var)
  {
    let result = match coeff {
      Expr::Integer(1) => make_sqrt(Expr::Constant("Pi".to_string())),
      _ => make_sqrt(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(coeff),
      }),
    };
    // Re-evaluate so e.g. `Sqrt[Pi/(1/4)]` collapses to `2*Sqrt[Pi]`.
    return Some(
      crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result),
    );
  }

  // Gaussian moment: ∫ c * x^n * E^(-a*x^2) dx over (-∞,∞) or (0,∞).
  {
    let full_range = is_negative_infinity(lo) && is_infinity(hi);
    let half_range = matches!(lo, Expr::Integer(0)) && is_infinity(hi);
    if (full_range || half_range)
      && let Some((n, coeff, consts)) = match_gaussian_moment(integrand, var)
    {
      let result = gaussian_moment_result(n, &coeff, &consts, full_range);
      return Some(
        crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result),
      );
    }
  }

  // Half-Gaussian: ∫_0^{∞} E^(-a*x^2) dx = Sqrt[Pi/a]/2
  if matches!(lo, Expr::Integer(0))
    && is_infinity(hi)
    && let Some(coeff) = match_gaussian(integrand, var)
  {
    let sqrt_part = match coeff {
      Expr::Integer(1) => make_sqrt(Expr::Constant("Pi".to_string())),
      _ => make_sqrt(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(coeff),
      }),
    };
    let result = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(sqrt_part),
      right: Box::new(Expr::Integer(2)),
    };
    return Some(
      crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result),
    );
  }

  // Fresnel C/S: ∫_0^z Cos[Pi var^2 / 2] d var = FresnelC[z]
  //              ∫_0^z Sin[Pi var^2 / 2] d var = FresnelS[z]
  if matches!(lo, Expr::Integer(0))
    && let Expr::FunctionCall { name, args } = integrand
    && (name == "Cos" || name == "Sin")
    && args.len() == 1
    && is_pi_x_squared_over_two(&args[0], var)
  {
    let head = if name == "Cos" {
      "FresnelC"
    } else {
      "FresnelS"
    };
    return Some(Expr::FunctionCall {
      name: head.to_string(),
      args: vec![hi.clone()].into(),
    });
  }

  // Bessel integral representation:
  //   ∫_0^Pi Cos[n · Sin[var]] d var = Pi · BesselJ[0, n]
  // for any `n` independent of `var`.
  if matches!(lo, Expr::Integer(0))
    && is_pi(hi)
    && let Expr::FunctionCall {
      name: cos_name,
      args: cos_args,
    } = integrand
    && cos_name == "Cos"
    && cos_args.len() == 1
    && let Some(n) = extract_coefficient_of_sin_var(&cos_args[0], var)
    && !expr_depends_on_var(&n, var)
  {
    let bessel = Expr::FunctionCall {
      name: "BesselJ".to_string(),
      args: vec![Expr::Integer(0), n].into(),
    };
    return Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Constant("Pi".to_string())),
      right: Box::new(bessel),
    });
  }

  // Euler's log-trig integrals:
  //   ∫_0^{Pi/2} Log[Sin[x]] dx = ∫_0^{Pi/2} Log[Cos[x]] dx = -(Pi·Log[2])/2
  //   ∫_0^{Pi/2} Log[Tan[x]] dx = ∫_0^{Pi/2} Log[Cot[x]] dx = 0
  //   ∫_0^{Pi}   Log[Sin[x]] dx = -Pi·Log[2]
  if matches!(lo, Expr::Integer(0))
    && let Expr::FunctionCall {
      name: log_name,
      args: log_args,
    } = integrand
    && log_name == "Log"
    && log_args.len() == 1
    && let Expr::FunctionCall {
      name: trig_name,
      args: trig_args,
    } = &log_args[0]
    && trig_args.len() == 1
    && matches!(&trig_args[0], Expr::Identifier(n) if n == var)
  {
    let pi_log2 = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Constant("Pi".to_string()),
        Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![Expr::Integer(2)].into(),
        },
      ]
      .into(),
    };
    if is_pi_over_two(hi) {
      match trig_name.as_str() {
        "Sin" | "Cos" => {
          let result = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(-1), Expr::Integer(2)].into(),
              },
              pi_log2,
            ]
            .into(),
          };
          return Some(
            crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result),
          );
        }
        // The Tan/Cot integrands are antisymmetric about Pi/4 (Tan and Cot
        // swap under x -> Pi/2 - x), so the two halves cancel exactly.
        "Tan" | "Cot" => return Some(Expr::Integer(0)),
        _ => {}
      }
    } else if is_pi(hi) && trig_name == "Sin" {
      let result = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), pi_log2].into(),
      };
      return Some(
        crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result),
      );
    }
  }

  // Linearity in constant factors: ∫ c·f(x) dx = c·∫ f(x) dx for any factor
  // c independent of the integration variable. Recursing on the var-dependent
  // part lets every known-integral rule above compose with constant
  // multiples. Runs last so rules with their own coefficient handling (e.g.
  // DiracDelta sifting, Gaussian moments) keep their specialised results.
  {
    let (negated, base) = match integrand {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => (true, operand.as_ref()),
      _ => (false, integrand),
    };
    let factors = flatten_times_factors(base);
    if negated || factors.len() > 1 {
      let (mut consts, dependent): (Vec<Expr>, Vec<Expr>) = factors
        .into_iter()
        .partition(|f| !expr_depends_on_var(f, var));
      if negated {
        consts.push(Expr::Integer(-1));
      }
      if !consts.is_empty() && !dependent.is_empty() {
        let rest = if dependent.len() == 1 {
          dependent.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: dependent.into(),
          }
        };
        if let Some(inner) = try_definite_integral(&rest, var, lo, hi) {
          consts.push(inner);
          let product = Expr::FunctionCall {
            name: "Times".to_string(),
            args: consts.into(),
          };
          return Some(
            crate::evaluator::evaluate_expr_to_expr(&product)
              .unwrap_or(product),
          );
        }
      }
    }
  }

  None
}

/// Match the constant `Pi/2` in either its parsed (`Pi/2` division) or
/// evaluated (`Times[Rational[1, 2], Pi]`) form.
fn is_pi_over_two(e: &Expr) -> bool {
  let is_half = |e: &Expr| {
    matches!(
      e,
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2
           && matches!(&args[0], Expr::Integer(1))
           && matches!(&args[1], Expr::Integer(2))
    )
  };
  match e {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => is_pi(left) && matches!(right.as_ref(), Expr::Integer(2)),
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      (is_half(&args[0]) && is_pi(&args[1]))
        || (is_pi(&args[0]) && is_half(&args[1]))
    }
    _ => false,
  }
}

fn is_pi(e: &Expr) -> bool {
  matches!(e, Expr::Constant(c) if c == "Pi")
}

/// Match `Sin[var]` (coefficient 1) or `n * Sin[var]` for some `n`. Returns
/// the `n` coefficient on success.
fn extract_coefficient_of_sin_var(expr: &Expr, var: &str) -> Option<Expr> {
  let is_sin_var = |e: &Expr| match e {
    Expr::FunctionCall { name, args } => {
      name == "Sin"
        && args.len() == 1
        && matches!(&args[0], Expr::Identifier(n) if n == var)
    }
    _ => false,
  };
  if is_sin_var(expr) {
    return Some(Expr::Integer(1));
  }
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut found = false;
      let mut other: Vec<Expr> = Vec::new();
      for a in args {
        if is_sin_var(a) {
          found = true;
        } else {
          other.push(a.clone());
        }
      }
      if !found {
        return None;
      }
      match other.len() {
        0 => Some(Expr::Integer(1)),
        1 => Some(other.pop().unwrap()),
        _ => Some(Expr::FunctionCall {
          name: "Times".to_string(),
          args: other.into(),
        }),
      }
    }
    _ => None,
  }
}

fn expr_depends_on_var(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(n) => n == var,
    Expr::BinaryOp { left, right, .. } => {
      expr_depends_on_var(left, var) || expr_depends_on_var(right, var)
    }
    Expr::UnaryOp { operand, .. } => expr_depends_on_var(operand, var),
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|a| expr_depends_on_var(a, var))
    }
    Expr::List(items) => items.iter().any(|a| expr_depends_on_var(a, var)),
    _ => false,
  }
}

/// Match `Pi * var^2 / 2` in any canonical-times ordering.
fn is_pi_x_squared_over_two(expr: &Expr, var: &str) -> bool {
  // Normalize: we're looking for Times[Pi, Power[var, 2], Rational[1, 2]]
  // or equivalent Divide forms.
  fn is_half(e: &Expr) -> bool {
    matches!(
      e,
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2
           && matches!(&args[0], Expr::Integer(1))
           && matches!(&args[1], Expr::Integer(2))
    )
  }
  fn is_var_squared(e: &Expr, var: &str) -> bool {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => {
        matches!(left.as_ref(), Expr::Identifier(n) if n == var)
          && matches!(right.as_ref(), Expr::Integer(2))
      }
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        matches!(&args[0], Expr::Identifier(n) if n == var)
          && matches!(&args[1], Expr::Integer(2))
      }
      _ => false,
    }
  }
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      // Expect exactly three factors: Pi, var^2, 1/2 in any order.
      if args.len() != 3 {
        return false;
      }
      let mut have_pi = false;
      let mut have_sq = false;
      let mut have_half = false;
      for a in args {
        if is_pi(a) {
          have_pi = true;
        } else if is_var_squared(a, var) {
          have_sq = true;
        } else if is_half(a) {
          have_half = true;
        } else {
          return false;
        }
      }
      have_pi && have_sq && have_half
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      // (Pi * var^2) / 2
      matches!(right.as_ref(), Expr::Integer(2))
        && matches!(
          left.as_ref(),
          Expr::FunctionCall { name, args }
            if name == "Times" && args.len() == 2
               && ((is_pi(&args[0]) && is_var_squared(&args[1], var))
                   || (is_pi(&args[1]) && is_var_squared(&args[0], var)))
        )
    }
    _ => false,
  }
}

/// Try to match an expression as E^(-a*x^2) where a is a positive constant.
/// Returns Some(a) if it matches, None otherwise.
fn match_gaussian(expr: &Expr, var: &str) -> Option<Expr> {
  // Match E^(exponent) where E is the constant. Both `BinaryOp::Power` (from
  // direct `^` parsing) and `FunctionCall["Power", [E, exponent]]` (from
  // canonical-form simplification of e.g. `Exp[-(x/2)^2]`) are accepted.
  let exponent = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Constant(c) if c == "E") {
        Some(right.as_ref())
      } else {
        None
      }
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if matches!(&args[0], Expr::Constant(c) if c == "E") {
        Some(&args[1])
      } else {
        None
      }
    }
    _ => None,
  }?;

  // Match -a*x^2 or -(x^2) forms in the exponent
  match_neg_a_x_squared(exponent, var)
}

/// Flatten a (possibly nested) product into its individual factors. A
/// non-product expression yields a single-element list.
fn flatten_times_factors(expr: &Expr) -> Vec<Expr> {
  fn rec(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        rec(left, out);
        rec(right, out);
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          rec(a, out);
        }
      }
      _ => out.push(e.clone()),
    }
  }
  let mut out = Vec::new();
  rec(expr, &mut out);
  out
}

/// Match `var` or `var^k` (k a non-negative integer) and return the exponent.
fn match_var_power(expr: &Expr, var: &str) -> Option<i128> {
  if matches!(expr, Expr::Identifier(n) if n == var) {
    return Some(1);
  }
  let (base, exp) = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => (left.as_ref(), right.as_ref()),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    _ => return None,
  };
  if matches!(base, Expr::Identifier(n) if n == var)
    && let Expr::Integer(k) = exp
    && *k >= 0
  {
    return Some(*k);
  }
  None
}

/// Match an integrand of the form `c * var^n * E^(-a*var^2)` (factors in any
/// order). Returns `(n, a, consts)` where `n >= 0` is the total power of
/// `var`, `a` the positive Gaussian coefficient, and `consts` the var-free
/// factors. Requires at least two factors so a bare Gaussian is left to the
/// dedicated handler.
fn match_gaussian_moment(
  expr: &Expr,
  var: &str,
) -> Option<(i128, Expr, Vec<Expr>)> {
  let factors = flatten_times_factors(expr);
  if factors.len() < 2 {
    return None;
  }
  let mut coeff_a: Option<Expr> = None;
  let mut power: i128 = 0;
  let mut consts: Vec<Expr> = Vec::new();
  for f in &factors {
    if let Some(a) = match_gaussian(f, var) {
      if coeff_a.is_some() {
        return None; // more than one Gaussian factor
      }
      coeff_a = Some(a);
    } else if let Some(n) = match_var_power(f, var) {
      power = power.checked_add(n)?;
    } else if !contains_var(f, var) {
      consts.push(f.clone());
    } else {
      return None; // unrecognized var-dependent factor
    }
  }
  coeff_a.map(|a| (power, a, consts))
}

/// Build the closed form of `∫ c * x^n * E^(-a*x^2) dx` over `(-∞,∞)` (when
/// `full_range`) or `(0,∞)`.
///
/// * Even `n = 2m`: `(2m-1)!! / (2a)^m * Sqrt[Pi/a]`, halved over `(0,∞)`.
/// * Odd `n = 2k+1`: `0` over `(-∞,∞)`, and `k! / (2 a^(k+1))` over `(0,∞)`.
///
/// `consts` are the var-free factors multiplied back in.
fn gaussian_moment_result(
  n: i128,
  coeff: &Expr,
  consts: &[Expr],
  full_range: bool,
) -> Expr {
  let pow = |base: Expr, exp: i128| Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![base, Expr::Integer(exp)].into(),
  };
  let half = || Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
  };
  let sqrt_pi_over_a = || match coeff {
    Expr::Integer(1) => make_sqrt(Expr::Constant("Pi".to_string())),
    _ => make_sqrt(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Constant("Pi".to_string())),
      right: Box::new(coeff.clone()),
    }),
  };

  let mut factors: Vec<Expr> = Vec::new();
  if n % 2 == 0 {
    let m = n / 2;
    factors.push(Expr::FunctionCall {
      name: "Factorial2".to_string(),
      args: vec![Expr::Integer(2 * m - 1)].into(),
    });
    // (2 a)^(-m)
    factors.push(pow(
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(2), coeff.clone()].into(),
      },
      -m,
    ));
    factors.push(sqrt_pi_over_a());
    if !full_range {
      factors.push(half());
    }
  } else if full_range {
    return Expr::Integer(0);
  } else {
    let k = (n - 1) / 2;
    factors.push(Expr::FunctionCall {
      name: "Factorial".to_string(),
      args: vec![Expr::Integer(k)].into(),
    });
    factors.push(pow(coeff.clone(), -(k + 1)));
    factors.push(half());
  }
  factors.extend(consts.iter().cloned());
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: factors.into(),
  }
}

/// Match an exponent expression as -a*x^2 and return 'a'.
/// Handles forms: -x^2, -(x^2), -a*x^2, Times[-1, x, x], etc.
fn match_neg_a_x_squared(expr: &Expr, var: &str) -> Option<Expr> {
  match expr {
    // UnaryOp::Minus wrapping something
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      // -x^2 => a=1
      if is_var_squared(operand, var) {
        return Some(Expr::Integer(1));
      }
      // -(a*x^2) => a
      if let Some(coeff) = match_a_x_squared(operand, var) {
        return Some(coeff);
      }
      None
    }
    // Times[-1, x^2] or Times[x^2, -1]
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      // -1 * x^2
      if matches!(left.as_ref(), Expr::Integer(-1))
        && is_var_squared(right, var)
      {
        return Some(Expr::Integer(1));
      }
      if matches!(right.as_ref(), Expr::Integer(-1))
        && is_var_squared(left, var)
      {
        return Some(Expr::Integer(1));
      }
      // -1 * (a * x^2)
      if matches!(left.as_ref(), Expr::Integer(-1))
        && let Some(coeff) = match_a_x_squared(right, var)
      {
        return Some(coeff);
      }
      if matches!(right.as_ref(), Expr::Integer(-1))
        && let Some(coeff) = match_a_x_squared(left, var)
      {
        return Some(coeff);
      }
      // (-a) * x^2 where a is a negative integer
      if let Expr::Integer(n) = left.as_ref()
        && *n < 0
        && is_var_squared(right, var)
      {
        return Some(Expr::Integer(-*n));
      }
      if let Expr::Integer(n) = right.as_ref()
        && *n < 0
        && is_var_squared(left, var)
      {
        return Some(Expr::Integer(-*n));
      }
      None
    }
    // FunctionCall("Times", [...]) form
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Find a negative numeric factor (Integer or Rational) and check that
      // the remainder is x^2 or a*x^2, returning the positive coefficient.
      // Handles e.g. `Times[Rational[-1, 4], Power[x, 2]]` from the
      // simplified form of `Exp[-(x/2)^2]`.
      fn neg_numeric_to_positive(e: &Expr) -> Option<Expr> {
        match e {
          Expr::Integer(n) if *n < 0 => Some(Expr::Integer(-n)),
          Expr::Real(f) if *f < 0.0 => Some(Expr::Real(-f)),
          Expr::FunctionCall { name, args }
            if name == "Rational" && args.len() == 2 =>
          {
            if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
              let (n, d) = if *d < 0 { (-*n, -*d) } else { (*n, *d) };
              if n < 0 {
                return Some(Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(-n), Expr::Integer(d)].into(),
                });
              }
            }
            None
          }
          _ => None,
        }
      }
      for (i, arg) in args.iter().enumerate() {
        if let Some(positive_coeff) = neg_numeric_to_positive(arg) {
          let mut rest: Vec<Expr> = args
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, e)| e.clone())
            .collect();
          let rest_expr = if rest.len() == 1 {
            rest.remove(0)
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest.into(),
            }
          };
          if matches!(arg, Expr::Integer(-1)) {
            // Times[-1, x^2] => a=1; Times[-1, b*x^2] => b
            if is_var_squared(&rest_expr, var) {
              return Some(Expr::Integer(1));
            }
            if let Some(coeff) = match_a_x_squared(&rest_expr, var) {
              return Some(coeff);
            }
          } else {
            // Times[-coeff, x^2] => coeff (positive form)
            if is_var_squared(&rest_expr, var) {
              return Some(positive_coeff);
            }
          }
        }
      }
      None
    }
    // BinaryOp::Minus: 0 - x^2 or similar (unlikely but handle)
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Integer(0)) {
        if is_var_squared(right, var) {
          return Some(Expr::Integer(1));
        }
        if let Some(coeff) = match_a_x_squared(right, var) {
          return Some(coeff);
        }
      }
      None
    }
    _ => None,
  }
}

/// Check if expr is x^(-2) (where x is the variable). Accepts both the
/// parsed `BinaryOp::Power` form and the `Power[x, -2]` FunctionCall form.
/// Used to recognise `Exp[-1/x^2]` for the special antiderivative
/// `x*Exp[-1/x^2] + Sqrt[Pi]*Erf[1/x]`.
fn is_var_to_minus_two(expr: &Expr, var: &str) -> bool {
  if let Expr::BinaryOp {
    op: BinaryOperator::Power,
    left,
    right,
  } = expr
  {
    return matches!(left.as_ref(), Expr::Identifier(name) if name == var)
      && matches!(right.as_ref(), Expr::Integer(-2));
  }
  if let Expr::FunctionCall { name, args } = expr
    && name == "Power"
    && args.len() == 2
  {
    return matches!(&args[0], Expr::Identifier(n) if n == var)
      && matches!(&args[1], Expr::Integer(-2));
  }
  false
}

/// Match `-1/x^2` (i.e. `-x^(-2)`) and return `Some(())`. Currently only
/// the specific `a == 1` case is recognised since that's the only Erf-
/// closed-form antiderivative we surface.
fn match_neg_inverse_x_squared(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_var_to_minus_two(operand, var),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      (matches!(left.as_ref(), Expr::Integer(-1))
        && is_var_to_minus_two(right, var))
        || (matches!(right.as_ref(), Expr::Integer(-1))
          && is_var_to_minus_two(left, var))
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      (matches!(&args[0], Expr::Integer(-1))
        && is_var_to_minus_two(&args[1], var))
        || (matches!(&args[1], Expr::Integer(-1))
          && is_var_to_minus_two(&args[0], var))
    }
    _ => false,
  }
}

/// Check if expr is x^2 (where x is the variable). Accepts both the parsed
/// `BinaryOp::Power` form and the canonicalised `Power[x, 2]` FunctionCall
/// form (which is what falls out of simplifying `(x/2)^2` etc.).
fn is_var_squared(expr: &Expr, var: &str) -> bool {
  if let Expr::BinaryOp {
    op: BinaryOperator::Power,
    left,
    right,
  } = expr
  {
    return matches!(left.as_ref(), Expr::Identifier(name) if name == var)
      && matches!(right.as_ref(), Expr::Integer(2));
  }
  if let Expr::FunctionCall { name, args } = expr
    && name == "Power"
    && args.len() == 2
  {
    return matches!(&args[0], Expr::Identifier(n) if n == var)
      && matches!(&args[1], Expr::Integer(2));
  }
  false
}

/// Match a*x^2 and return 'a'
fn match_a_x_squared(expr: &Expr, var: &str) -> Option<Expr> {
  if let Expr::BinaryOp {
    op: BinaryOperator::Times,
    left,
    right,
  } = expr
  {
    // a * x^2
    if is_constant_wrt(left, var) && is_var_squared(right, var) {
      return Some(*left.clone());
    }
    // x^2 * a
    if is_var_squared(left, var) && is_constant_wrt(right, var) {
      return Some(*right.clone());
    }
  }
  // FunctionCall("Times", [...]) form
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
  {
    // Find x^2 factor, rest is 'a'
    for (i, arg) in args.iter().enumerate() {
      if is_var_squared(arg, var) {
        let rest: Vec<Expr> = args
          .iter()
          .enumerate()
          .filter(|(j, _)| *j != i)
          .map(|(_, e)| e.clone())
          .collect();
        if rest.iter().all(|a| is_constant_wrt(a, var)) {
          return if rest.len() == 1 {
            Some(rest[0].clone())
          } else {
            Some(Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest.into(),
            })
          };
        }
      }
    }
  }
  None
}

/// Check if expression is constant with respect to a variable
pub fn is_constant_wrt(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _)
    | Expr::String(_)
    | Expr::Constant(_) => true,
    Expr::Identifier(name) => name != var,
    Expr::List(items) => items.iter().all(|e| is_constant_wrt(e, var)),
    Expr::BinaryOp { left, right, .. } => {
      is_constant_wrt(left, var) && is_constant_wrt(right, var)
    }
    Expr::UnaryOp { operand, .. } => is_constant_wrt(operand, var),
    Expr::FunctionCall { args, .. } => {
      args.iter().all(|e| is_constant_wrt(e, var))
    }
    Expr::CurriedCall { func, args } => {
      is_constant_wrt(func, var) && args.iter().all(|e| is_constant_wrt(e, var))
    }
    // A comparison/inequality is constant w.r.t. `var` iff all of its operands
    // are (e.g. `x >= 5` is constant w.r.t. `y`). Without this it fell to the
    // catch-all below and was reported as always depending on every variable.
    Expr::Comparison { operands, .. } => {
      operands.iter().all(|e| is_constant_wrt(e, var))
    }
    _ => false,
  }
}

/// Differentiate an expression with respect to a variable
/// Public wrapper for differentiate - used by Derivative[n][f][x] evaluation
pub fn differentiate_expr(
  expr: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  differentiate(expr, var)
}

/// Parse a stringified curried-derivative head of the form
/// `"Derivative[i1, i2, ...][fname]"` (the result of stringifying a
/// `CurriedCall { func: FunctionCall("Derivative", [...]), args: [Identifier(f)] }`
/// in `apply_curried_call`'s fallback path) into `(indices, fname)`. Returns
/// `None` if the name doesn't parse as that exact shape.
fn parse_stringified_derivative_head(
  name: &str,
) -> Option<(Vec<Expr>, String)> {
  // Match prefix "Derivative["
  let rest = name.strip_prefix("Derivative[")?;
  // Find the closing "]" of the indices list (no nested brackets in indices)
  let close = rest.find(']')?;
  let indices_str = &rest[..close];
  // Then "[fname]" follows immediately
  let after = &rest[close + 1..];
  let fname = after.strip_prefix('[').and_then(|s| s.strip_suffix(']'))?;
  if fname.is_empty() {
    return None;
  }
  let indices: Result<Vec<Expr>, _> = indices_str
    .split(',')
    .map(|s| s.trim().parse::<i128>().map(Expr::Integer))
    .collect();
  Some((indices.ok()?, fname.to_string()))
}

/// In the derivative of a `Piecewise`, each piece boundary becomes a point of
/// non-differentiability that falls to the `Indeterminate` default, so an
/// inclusive comparison (`x <= 0`, `x >= 0`) is tightened to its strict form
/// (`x < 0`, `x > 0`) — matching wolframscript. Only bare two-operand
/// comparisons are rewritten; compound/other conditions are left untouched.
fn strip_piecewise_condition_boundary(cond: &Expr) -> Expr {
  match cond {
    Expr::Comparison {
      operands,
      operators,
    } if operators.len() == 1 => {
      let new_op = match operators[0] {
        ComparisonOp::LessEqual => ComparisonOp::Less,
        ComparisonOp::GreaterEqual => ComparisonOp::Greater,
        other => other,
      };
      Expr::Comparison {
        operands: operands.clone(),
        operators: vec![new_op],
      }
    }
    Expr::FunctionCall { name, args } if args.len() == 2 => {
      let new_name = match name.as_str() {
        "LessEqual" => "Less",
        "GreaterEqual" => "Greater",
        _ => return cond.clone(),
      };
      Expr::FunctionCall {
        name: new_name.to_string(),
        args: args.clone(),
      }
    }
    _ => cond.clone(),
  }
}

fn differentiate(expr: &Expr, var: &str) -> Result<Expr, InterpreterError> {
  match expr {
    // Constants
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => {
      Ok(Expr::Integer(0))
    }

    // Slots are independent variables of a pure function. When the
    // differentiation variable is itself one slot (substituted through
    // a fresh symbol), other Slot[k] nodes contribute zero rather than
    // an unresolved D[#k, x] term.
    Expr::Slot(_) | Expr::SlotSequence(_) => Ok(Expr::Integer(0)),

    // Variable
    Expr::Identifier(name) => {
      if name == var {
        Ok(Expr::Integer(1))
      } else {
        // A symbol listed in the active `NonConstants` context depends on the
        // differentiation variable, so its derivative is the unevaluated
        // `D[name, var, NonConstants -> {…}]` rather than zero.
        let carried = NON_CONSTANTS.with(|nc| {
          nc.borrow().as_ref().and_then(|(names, rule)| {
            names.iter().any(|n| n == name).then(|| Expr::FunctionCall {
              name: "D".to_string(),
              args: vec![
                Expr::Identifier(name.clone()),
                Expr::Identifier(var.to_string()),
                rule.clone(),
              ]
              .into(),
            })
          })
        });
        Ok(carried.unwrap_or(Expr::Integer(0)))
      }
    }

    // Binary operations
    Expr::BinaryOp { op, left, right } => {
      use BinaryOperator::*;
      match op {
        Plus => {
          // d/dx[a + b] = d/dx[a] + d/dx[b]
          let da = differentiate(left, var)?;
          let db = differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Plus,
            left: Box::new(da),
            right: Box::new(db),
          }))
        }
        Minus => {
          // d/dx[a - b] = d/dx[a] - d/dx[b]
          let da = differentiate(left, var)?;
          let db = differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Minus,
            left: Box::new(da),
            right: Box::new(db),
          }))
        }
        Times => {
          // Product rule: d/dx[a * b] = a' * b + a * b'
          let da = differentiate(left, var)?;
          let db = differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Plus,
            left: Box::new(Expr::BinaryOp {
              op: Times,
              left: Box::new(da),
              right: right.clone(),
            }),
            right: Box::new(Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(db),
            }),
          }))
        }
        Divide => {
          // Rewrite a/b as a * b^(-1) to use power+product rule
          // instead of quotient rule (avoids exponential expression growth)
          if is_constant_wrt(right, var) {
            // d/dx[a / c] = (d/dx a) / c
            // Use times_ast with b^(-1) so integer coefficients cancel
            let da = differentiate(left, var)?;
            let result = crate::functions::math_ast::times_ast(&[
              da,
              Expr::BinaryOp {
                op: Power,
                left: right.clone(),
                right: Box::new(Expr::Integer(-1)),
              },
            ])
            .unwrap_or_else(|_| Expr::BinaryOp {
              op: Divide,
              left: Box::new(
                differentiate(left, var).unwrap_or(Expr::Integer(0)),
              ),
              right: right.clone(),
            });
            Ok(simplify(result))
          } else if is_constant_wrt(left, var) {
            // d/dx[c / b] = c * d/dx[b^(-1)] = -c * b' / b^2
            let rewritten = Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(Expr::BinaryOp {
                op: Power,
                left: right.clone(),
                right: Box::new(Expr::Integer(-1)),
              }),
            };
            differentiate(&rewritten, var)
          } else {
            // d/dx[a / b] = d/dx[a * b^(-1)] (product rule + power rule)
            let rewritten = Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(Expr::BinaryOp {
                op: Power,
                left: right.clone(),
                right: Box::new(Expr::Integer(-1)),
              }),
            };
            differentiate(&rewritten, var)
          }
        }
        Power => {
          // Power rule for x^n: n * x^(n-1) * x'
          // Use Plus[-1, n] to match Wolfram's canonical form (-1 + n)
          if is_constant_wrt(right, var) {
            let df = differentiate(left, var)?;
            Ok(simplify(Expr::BinaryOp {
              op: Times,
              left: Box::new(Expr::BinaryOp {
                op: Times,
                left: right.clone(),
                right: Box::new(Expr::BinaryOp {
                  op: Power,
                  left: left.clone(),
                  right: Box::new(Expr::BinaryOp {
                    op: Plus,
                    left: Box::new(Expr::Integer(-1)),
                    right: right.clone(),
                  }),
                }),
              }),
              right: Box::new(df),
            }))
          } else if matches!(left.as_ref(), Expr::Constant(c) if c == "E") {
            // d/dx[E^g(x)] = E^g(x) * g'(x)  (since Log[E] = 1)
            let dg = differentiate(right, var)?;
            Ok(simplify(Expr::BinaryOp {
              op: Times,
              left: Box::new(expr.clone()),
              right: Box::new(dg),
            }))
          } else if is_constant_wrt(left, var) {
            // d/dx[a^g(x)] = a^g(x) * ln(a) * g'(x)
            let dg = differentiate(right, var)?;
            Ok(simplify(Expr::BinaryOp {
              op: Times,
              left: Box::new(Expr::BinaryOp {
                op: Times,
                left: Box::new(expr.clone()),
                right: Box::new(Expr::FunctionCall {
                  name: "Log".to_string(),
                  args: vec![*left.clone()].into(),
                }),
              }),
              right: Box::new(dg),
            }))
          } else {
            // General case: d/dx[f(x)^g(x)] = f(x)^g(x) * (g'(x)*Log[f(x)] + g(x)*f'(x)/f(x))
            // This is logarithmic differentiation
            let df = differentiate(left, var)?;
            let dg = differentiate(right, var)?;
            Ok(simplify(Expr::BinaryOp {
              op: Times,
              left: Box::new(expr.clone()), // f^g
              right: Box::new(Expr::BinaryOp {
                op: Plus,
                left: Box::new(Expr::BinaryOp {
                  op: Times,
                  left: Box::new(dg), // g'
                  right: Box::new(Expr::FunctionCall {
                    name: "Log".to_string(),
                    args: vec![*left.clone()].into(), // Log[f]
                  }),
                }),
                right: Box::new(Expr::BinaryOp {
                  op: Times,
                  left: right.clone(), // g
                  right: Box::new(Expr::BinaryOp {
                    op: Times,
                    left: Box::new(df), // f'
                    right: Box::new(Expr::BinaryOp {
                      op: Power,
                      left: left.clone(),                 // f
                      right: Box::new(Expr::Integer(-1)), // f^(-1)
                    }),
                  }),
                }),
              }),
            }))
          }
        }
        _ => Ok(Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
        }),
      }
    }

    // Unary minus
    Expr::UnaryOp { op, operand } => {
      if matches!(op, UnaryOperator::Minus) {
        let d = differentiate(operand, var)?;
        Ok(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(d),
        })
      } else {
        Ok(Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
        })
      }
    }

    // Function calls
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        // `List[a, b, ...]` differentiates element-wise. Without this
        // special case, the generic chain rule below produces the
        // unresolved `Derivative[1, ...][List][...]` form, which
        // `Derivative[n][List]` then refuses to fold into a pure
        // function. Apply the same rule the `Expr::List` arm does
        // further down so `Derivative[1][List]` returns `{1}&` and
        // `Derivative[0, 0, 1][List]` returns `{0, 0, 1}&`.
        "List" => {
          let mut diffed: Vec<Expr> = Vec::with_capacity(args.len());
          for a in args {
            diffed.push(differentiate(a, var)?);
          }
          Ok(Expr::List(diffed.into()))
        }
        // `Piecewise[{{v_i, c_i}, …}, default]` differentiates value-by-value:
        // each piece keeps its condition with the value differentiated, and the
        // default becomes `Indeterminate` (the derivative is undefined at the
        // piece boundaries). Without this the generic chain rule emits a
        // `Derivative[1, 0][Piecewise][…]` mess. This matches wolframscript for
        // the common multi-piece form whose conditions cover the line except at
        // isolated boundary points; wolframscript's finer default handling
        // (keeping D[default] for a single-piece continuous default, or
        // splitting a discontinuous default region) is not reproduced.
        "Piecewise" if !args.is_empty() => {
          if let Expr::List(pieces) = &args[0] {
            let mut diffed_pieces: Vec<Expr> = Vec::with_capacity(pieces.len());
            for piece in pieces.iter() {
              if let Expr::List(pair) = piece
                && pair.len() == 2
              {
                let dval = differentiate(&pair[0], var)?;
                diffed_pieces.push(Expr::List(
                  vec![dval, strip_piecewise_condition_boundary(&pair[1])]
                    .into(),
                ));
              } else {
                // Unexpected piece shape — fall back to unevaluated D[…].
                return Ok(Expr::FunctionCall {
                  name: "D".to_string(),
                  args: vec![expr.clone(), Expr::Identifier(var.to_string())]
                    .into(),
                });
              }
            }
            Ok(Expr::FunctionCall {
              name: "Piecewise".to_string(),
              args: vec![
                Expr::List(diffed_pieces.into()),
                Expr::Identifier("Indeterminate".to_string()),
              ]
              .into(),
            })
          } else {
            Ok(Expr::FunctionCall {
              name: "D".to_string(),
              args: vec![expr.clone(), Expr::Identifier(var.to_string())]
                .into(),
            })
          }
        }
        // SeriesData[var, x0, {coeffs}, nmin, nmax, denom]: differentiate
        // element-wise through the coefficient list and, when the center
        // x0 itself depends on `var`, apply the chain rule on the
        // `(var - x0)^p` factors. The chain-rule contribution at slot
        // `i` is `-((nmin + i + den)/den) * coeffs[i+den] * D[x0, var]`,
        // which both shrinks the coefficient list by `den` slots and
        // drops `nmax` by `den`. Strips leading zero coefficients (and
        // bumps nmin) so e.g. d/da of `1 - a x + a²/2 x²` collapses to
        // `SeriesData[x, 0, {-1, a}, 1, 3, 1]`, and so the F-series
        // identity `D[Series[F[x],{x, g[y], n}], y]` collapses to the
        // empty series `SeriesData[x, g[y], {}, n, n, 1]`.
        "SeriesData" if args.len() == 6 => {
          let coeffs = match &args[2] {
            Expr::List(items) => items.clone(),
            _ => {
              return Ok(Expr::FunctionCall {
                name: "D".to_string(),
                args: vec![expr.clone(), Expr::Identifier(var.to_string())]
                  .into(),
              });
            }
          };
          let nmin_val = match &args[3] {
            Expr::Integer(n) => *n,
            _ => 0,
          };
          let nmax_val = match &args[4] {
            Expr::Integer(n) => *n,
            _ => 0,
          };
          let den_val = match &args[5] {
            Expr::Integer(n) => *n,
            _ => 1,
          };
          let center = &args[1];

          // If `var` is the series variable itself, apply the term-by-term
          // power rule for the truncated series:
          //   d/dv [ c_i (v - x0)^((nmin+i)/den) ]
          //     = c_i (nmin+i)/den (v - x0)^((nmin+i-den)/den).
          // The exponents shift down by one integer power, so nmin and nmax
          // each drop by `den` and every coefficient is scaled by its
          // exponent. (Coefficients of a valid SeriesData never depend on the
          // expansion variable, so there is no coefficient-wise term here.)
          if matches!(&args[0], Expr::Identifier(v) if v == var) {
            let mut dcoeffs: Vec<Expr> = Vec::with_capacity(coeffs.len());
            for (i, c) in coeffs.iter().enumerate() {
              let exp_num = nmin_val + i as i128;
              let factor = if den_val == 1 {
                Expr::Integer(exp_num)
              } else {
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(exp_num), Expr::Integer(den_val)]
                    .into(),
                }
              };
              let term =
                crate::functions::math_ast::times_ast(&[factor, c.clone()])
                  .unwrap_or(Expr::Integer(0));
              dcoeffs.push(simplify(term));
            }
            let mut new_nmin = nmin_val - den_val;
            let new_nmax = nmax_val - den_val;
            // Trim leading zero coefficients (bumping nmin) — e.g. the
            // constant term differentiates to a zero at the lowest power.
            while !dcoeffs.is_empty() && matches!(&dcoeffs[0], Expr::Integer(0))
            {
              dcoeffs.remove(0);
              new_nmin += 1;
            }
            // Trim trailing zero coefficients while keeping nmax.
            while dcoeffs.len() > 1
              && matches!(dcoeffs.last(), Some(Expr::Integer(0)))
            {
              dcoeffs.pop();
            }
            let (out_nmin, out_coeffs) = if dcoeffs.is_empty() {
              (new_nmax, Vec::new())
            } else {
              (new_nmin, dcoeffs)
            };
            return Ok(Expr::FunctionCall {
              name: "SeriesData".to_string(),
              args: vec![
                args[0].clone(),
                args[1].clone(),
                Expr::List(out_coeffs.into()),
                Expr::Integer(out_nmin),
                Expr::Integer(new_nmax),
                Expr::Integer(den_val),
              ]
              .into(),
            });
          }

          // First, the element-wise contribution from differentiating
          // each coefficient.
          let mut diffed: Vec<Expr> = Vec::with_capacity(coeffs.len());
          for c in &coeffs {
            diffed.push(simplify(differentiate(c, var)?));
          }

          let center_depends = !is_constant_wrt(center, var);
          let den_step = den_val.max(1) as usize;

          let (mut new_coeffs, new_nmax) =
            if center_depends && coeffs.len() > den_step {
              // Chain-rule shift on (var - x0)^p factors. Shrinks both the
              // coefficient list and `nmax` by `den` slots.
              let dx0 = simplify(differentiate(center, var)?);
              let new_len = coeffs.len() - den_step;
              let mut shifted: Vec<Expr> = Vec::with_capacity(new_len);
              for i in 0..new_len {
                let pow_num = nmin_val + (i as i128) + den_val;
                let power_factor = if den_val == 1 {
                  Expr::Integer(pow_num)
                } else {
                  Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(pow_num), Expr::Integer(den_val)]
                      .into(),
                  }
                };
                let next_coeff = coeffs[i + den_step].clone();
                // shift_term = power_factor * next_coeff * dx0
                let shift_term = crate::functions::math_ast::times_ast(&[
                  power_factor,
                  next_coeff,
                  dx0.clone(),
                ])
                .unwrap_or(Expr::Integer(0));
                // new_coeff_i = diffed[i] - shift_term
                let neg_shift = crate::functions::math_ast::times_ast(&[
                  Expr::Integer(-1),
                  shift_term,
                ])
                .unwrap_or(Expr::Integer(0));
                let combined = crate::functions::math_ast::plus_ast(&[
                  diffed[i].clone(),
                  neg_shift,
                ])
                .unwrap_or_else(|_| diffed[i].clone());
                // Run a full evaluation so subterms like `g'[y]*F''[g[y]]
                // - F''[g[y]]*g'[y]` collapse to `0` via Times's canonical
                // ordering (the local `simplify` doesn't always sort
                // commutative factors the same way).
                let canon = crate::evaluator::evaluate_expr_to_expr(&combined)
                  .unwrap_or_else(|_| simplify(combined));
                shifted.push(canon);
              }
              (shifted, nmax_val - den_val)
            } else if center_depends {
              // Center depends on var but coefficient list is too short to
              // shift; the result is just the all-zero series with one
              // less integer power of accuracy.
              (Vec::new(), nmax_val - den_val)
            } else {
              (diffed, nmax_val)
            };

          let mut new_nmin = nmin_val;

          // Treat empty inner SeriesData as structurally zero so leading
          // zero trimming collapses series-of-series correctly.
          let is_zero_expr = |e: &Expr| -> bool {
            matches!(e, Expr::Integer(0))
              || matches!(e, Expr::FunctionCall { name, args }
                          if name == "SeriesData"
                          && args.len() == 6
                          && matches!(&args[2], Expr::List(items)
                                      if items.is_empty()))
          };

          while !new_coeffs.is_empty() && is_zero_expr(&new_coeffs[0]) {
            new_coeffs.remove(0);
            new_nmin += 1;
          }
          while new_coeffs.len() > 1
            && new_coeffs.last().map(is_zero_expr).unwrap_or(false)
          {
            new_coeffs.pop();
          }
          if new_coeffs.is_empty() {
            if center_depends {
              return Ok(Expr::FunctionCall {
                name: "SeriesData".to_string(),
                args: vec![
                  args[0].clone(),
                  args[1].clone(),
                  Expr::List(vec![].into()),
                  Expr::Integer(new_nmax),
                  Expr::Integer(new_nmax),
                  Expr::Integer(den_val),
                ]
                .into(),
              });
            }
            return Ok(Expr::Integer(0));
          }
          Ok(Expr::FunctionCall {
            name: "SeriesData".to_string(),
            args: vec![
              args[0].clone(),
              args[1].clone(),
              Expr::List(new_coeffs.into()),
              Expr::Integer(new_nmin),
              Expr::Integer(new_nmax),
              Expr::Integer(den_val),
            ]
            .into(),
          })
        }
        "Sin" if args.len() == 1 => {
          // d/dx[sin(f(x))] = cos(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Cos".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Cos" if args.len() == 1 => {
          // d/dx[cos(f(x))] = -sin(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(Expr::FunctionCall {
                name: "Sin".to_string(),
                args: args.clone(),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Gudermannian" if args.len() == 1 => {
          // d/dx[gd(f(x))] = Sech[f(x)] * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Sech".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "InverseGudermannian" if args.len() == 1 => {
          // d/dx[gd^-1(f(x))] = Sec[f(x)] * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Sec".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Haversine" if args.len() == 1 => {
          // d/dx[haversine(f(x))] = Sin[f(x)]/2 * f'(x)
          let df = differentiate(&args[0], var)?;
          let half = Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
          };
          Ok(simplify(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              half,
              Expr::FunctionCall {
                name: "Sin".to_string(),
                args: args.clone(),
              },
              df,
            ]
            .into(),
          }))
        }
        "Tan" if args.len() == 1 => {
          // d/dx[tan(f(x))] = sec^2(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(Expr::FunctionCall {
                name: "Sec".to_string(),
                args: args.clone(),
              }),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(df),
          }))
        }
        "Sec" if args.len() == 1 => {
          // d/dx[sec(f(x))] = sec(f(x)) * tan(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::FunctionCall {
                name: "Sec".to_string(),
                args: args.clone(),
              }),
              right: Box::new(Expr::FunctionCall {
                name: "Tan".to_string(),
                args: args.clone(),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Csc" if args.len() == 1 => {
          // d/dx[csc(f(x))] = -csc(f(x)) * cot(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(Expr::FunctionCall {
                  name: "Csc".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::FunctionCall {
                  name: "Cot".to_string(),
                  args: args.clone(),
                }),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Cot" if args.len() == 1 => {
          // d/dx[cot(f(x))] = -csc^2(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Power,
                left: Box::new(Expr::FunctionCall {
                  name: "Csc".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Sinh" if args.len() == 1 => {
          // d/dx[sinh(f(x))] = cosh(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Cosh".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Cosh" if args.len() == 1 => {
          // d/dx[cosh(f(x))] = sinh(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Sinh".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Tanh" if args.len() == 1 => {
          // d/dx[tanh(f(x))] = sech^2(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(Expr::FunctionCall {
                name: "Sech".to_string(),
                args: args.clone(),
              }),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(df),
          }))
        }
        "Sech" if args.len() == 1 => {
          // d/dx[sech(f(x))] = -sech(f(x)) * tanh(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(Expr::FunctionCall {
                  name: "Sech".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::FunctionCall {
                  name: "Tanh".to_string(),
                  args: args.clone(),
                }),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Csch" if args.len() == 1 => {
          // d/dx[csch(f(x))] = -coth(f(x)) * csch(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(Expr::FunctionCall {
                  name: "Coth".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::FunctionCall {
                  name: "Csch".to_string(),
                  args: args.clone(),
                }),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Coth" if args.len() == 1 => {
          // d/dx[coth(f(x))] = -csch^2(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Power,
                left: Box::new(Expr::FunctionCall {
                  name: "Csch".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "ArcSin" if args.len() == 1 => {
          // d/dx[arcsin(f(x))] = f'(x) / sqrt(1 - f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_minus_f_sq = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Power,
                left: Box::new(args[0].clone()),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
          };
          let sqrt_expr = make_sqrt(one_minus_f_sq);
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![sqrt_expr, Expr::Integer(-1)].into(),
            }),
          }))
        }
        "ArcCos" if args.len() == 1 => {
          // d/dx[arccos(f(x))] = -f'(x) / sqrt(1 - f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_minus_f_sq = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Power,
                left: Box::new(args[0].clone()),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
          };
          let sqrt_expr = make_sqrt(one_minus_f_sq);
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(df),
            }),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![sqrt_expr, Expr::Integer(-1)].into(),
            }),
          }))
        }
        "ArcTan" if args.len() == 2 => {
          // Two-argument arctangent (angle of the point (u, v)). Apply the
          // chain rule term by term to match Wolfram's form, which keeps the
          // sum of partials rather than a single combined fraction:
          //   d/dx ArcTan[u, v] = (-v/(u^2+v^2)) u' + (u/(u^2+v^2)) v'
          let u = args[0].clone();
          let v = args[1].clone();
          let du = differentiate(&u, var)?;
          let dv = differentiate(&v, var)?;
          let denom = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(u.clone()),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(v.clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          };
          let inv_denom = Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![denom, Expr::Integer(-1)].into(),
          };
          // partial wrt first arg: -v / (u^2 + v^2)
          let d_first = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(v),
            }),
            right: Box::new(inv_denom.clone()),
          };
          // partial wrt second arg: u / (u^2 + v^2)
          let d_second = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(u),
            right: Box::new(inv_denom),
          };
          let term1 = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(d_first),
            right: Box::new(du),
          };
          let term2 = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(d_second),
            right: Box::new(dv),
          };
          // Order the second-argument term first to match Wolfram's output,
          // e.g. ArcTan[u, v]' = u v'/(u^2+v^2) - v u'/(u^2+v^2).
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(term2),
            right: Box::new(term1),
          }))
        }
        "ArcTan" if args.len() == 1 => {
          // d/dx[arctan(f(x))] = f'(x) / (1 + f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_plus_f_sq = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          };
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![one_plus_f_sq, Expr::Integer(-1)].into(),
            }),
          }))
        }
        "ArcCot" if args.len() == 1 => {
          // d/dx[arccot(f(x))] = -f'(x) / (1 + f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_plus_f_sq = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          };
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(df),
            }),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![one_plus_f_sq, Expr::Integer(-1)].into(),
            }),
          }))
        }
        "ArcSinh" if args.len() == 1 => {
          // d/dx[arcsinh(f(x))] = f'(x) / sqrt(1 + f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_plus_f_sq = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          };
          let sqrt_expr = make_sqrt(one_plus_f_sq);
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![sqrt_expr, Expr::Integer(-1)].into(),
            }),
          }))
        }
        "ArcCosh" if args.len() == 1 => {
          // d/dx[arccosh(f(x))] = f'(x) / (sqrt(f(x) - 1) * sqrt(f(x) + 1))
          // Using factored form to match Wolfram's branch-cut-aware convention
          let df = differentiate(&args[0], var)?;
          let f_minus_one = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(args[0].clone()),
          };
          let f_plus_one = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(args[0].clone()),
          };
          let sqrt_minus = make_sqrt(f_minus_one);
          let sqrt_plus = make_sqrt(f_plus_one);
          // f'(x) / (Sqrt[f-1] * Sqrt[f+1])
          let denom = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(sqrt_minus),
            right: Box::new(sqrt_plus),
          };
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![denom, Expr::Integer(-1)].into(),
            }),
          }))
        }
        "ArcTanh" if args.len() == 1 => {
          // d/dx[arctanh(f(x))] = f'(x) / (1 - f(x)^2)
          let df = differentiate(&args[0], var)?;
          let one_minus_f_sq = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Power,
                left: Box::new(args[0].clone()),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
          };
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![one_minus_f_sq, Expr::Integer(-1)].into(),
            }),
          }))
        }
        "ArcCoth" if args.len() == 1 => {
          // d/dx[arccoth(f(x))] = f'(x) / (1 - f(x)^2) (same as ArcTanh)
          let df = differentiate(&args[0], var)?;
          let one_minus_f_sq = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Power,
                left: Box::new(args[0].clone()),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
          };
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![one_minus_f_sq, Expr::Integer(-1)].into(),
            }),
          }))
        }
        "Exp" if args.len() == 1 => {
          // d/dx[e^f(x)] = e^f(x) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Exp".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Log" if args.len() == 1 => {
          // d/dx[ln(f(x))] = f'(x) * f(x)^(-1)
          let df = differentiate(&args[0], var)?;
          let power_neg_one = Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![args[0].clone(), Expr::Integer(-1)].into(),
          };
          if matches!(df, Expr::Integer(1)) {
            Ok(power_neg_one)
          } else {
            crate::functions::math_ast::times_ast(&[df, power_neg_one])
          }
        }
        // Handle evaluated Plus[a, b, ...] (FunctionCall form of +)
        "Plus" if args.len() >= 2 => {
          let mut result = differentiate(&args[0], var)?;
          for arg in &args[1..] {
            let d = differentiate(arg, var)?;
            result = simplify(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(result),
              right: Box::new(d),
            });
          }
          Ok(result)
        }
        // Handle evaluated Times[a, b, ...] (FunctionCall form of *)
        "Times" if args.len() >= 2 => {
          // Generalized product rule:
          // D[f1*f2*...*fn, x] = sum_i(f1*...*D[fi]*...*fn)
          // Each term replaces one factor with its derivative.
          let mut sum_terms: Vec<Expr> = Vec::new();
          for i in 0..args.len() {
            let di = differentiate(&args[i], var)?;
            // Skip if derivative is zero (constant factor)
            if matches!(&di, Expr::Integer(0)) {
              continue;
            }
            if let Expr::Real(f) = &di
              && *f == 0.0
            {
              continue;
            }
            // Build the product: all factors with the i-th replaced by its derivative
            let mut product_args: Vec<Expr> = Vec::new();
            for (j, arg) in args.iter().enumerate() {
              if j == i {
                product_args.push(di.clone());
              } else {
                product_args.push(arg.clone());
              }
            }
            // Simplify this individual product term using times_ast
            let term = if product_args.len() == 1 {
              simplify(product_args.remove(0))
            } else {
              crate::functions::math_ast::times_ast(&product_args)?
            };
            if !matches!(&term, Expr::Integer(0)) {
              sum_terms.push(term);
            }
          }

          if sum_terms.is_empty() {
            Ok(Expr::Integer(0))
          } else if sum_terms.len() == 1 {
            Ok(sum_terms.remove(0))
          } else {
            // Combine terms using plus_ast for proper like-term collection
            // without expanding product sub-expressions
            crate::functions::math_ast::plus_ast(&sum_terms).or_else(|_| {
              Ok(Expr::FunctionCall {
                name: "Plus".to_string(),
                args: sum_terms.into(),
              })
            })
          }
        }
        // Handle evaluated Power[base, exp] (FunctionCall form of ^)
        "Power" if args.len() == 2 => differentiate(
          &Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(args[0].clone()),
            right: Box::new(args[1].clone()),
          },
          var,
        ),
        "Sqrt" if args.len() == 1 => {
          // d/dx[sqrt(f(x))] = f'(x) / (2 * sqrt(f(x)))
          let df = differentiate(&args[0], var)?;
          if matches!(df, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(df),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(2)),
              right: Box::new(make_sqrt(args[0].clone())),
            }),
          }))
        }
        // Abs[f(x)]: Abs is non-analytic, so wolframscript keeps the derivative
        // as the unevaluated Abs' (Derivative[1][Abs][f]) times the chain-rule
        // factor f', rather than rewriting it to Sign[f]. (Limits over the
        // reals rewrite Abs' back to Sign internally — see abs_deriv_to_sign.)
        "Abs" if args.len() == 1 => {
          let df = differentiate(&args[0], var)?;
          if matches!(df, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let deriv_expr = Expr::CurriedCall {
            func: Box::new(Expr::CurriedCall {
              func: Box::new(Expr::FunctionCall {
                name: "Derivative".to_string(),
                args: vec![Expr::Integer(1)].into(),
              }),
              args: vec![Expr::Identifier("Abs".to_string())],
            }),
            args: args.to_vec(),
          };
          if matches!(df, Expr::Integer(1)) {
            Ok(deriv_expr)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(df),
              right: Box::new(deriv_expr),
            }))
          }
        }
        // RealAbs[f(x)]: d/dx[RealAbs[f]] = f'(x) * f / RealAbs[f]
        // (matching wolframscript's preferred form, which avoids Sign).
        "RealAbs" if args.len() == 1 => {
          let df = differentiate(&args[0], var)?;
          if matches!(df, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          // Build: df * f / RealAbs[f]
          let f = args[0].clone();
          let real_abs = Expr::FunctionCall {
            name: "RealAbs".to_string(),
            args: vec![f.clone()].into(),
          };
          let f_over_abs = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(f),
            right: Box::new(real_abs),
          };
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(df),
            right: Box::new(f_over_abs),
          }))
        }
        // Sign derivative: D[Sign[f(x)], x] = Derivative[1][Sign][f(x)] * f'(x)
        // (Wolfram keeps the unevaluated Sign' instead of 2*DiracDelta[x])
        "Sign" if args.len() == 1 => {
          let df = differentiate(&args[0], var)?;
          if matches!(df, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          // wolframscript keeps the unevaluated Sign' rather than rewriting
          // Sign = Abs' and reporting Abs'' (Derivative[2][Abs]).
          let deriv_expr = Expr::CurriedCall {
            func: Box::new(Expr::CurriedCall {
              func: Box::new(Expr::FunctionCall {
                name: "Derivative".to_string(),
                args: vec![Expr::Integer(1)].into(),
              }),
              args: vec![Expr::Identifier("Sign".to_string())],
            }),
            args: args.to_vec(),
          };
          if matches!(df, Expr::Integer(1)) {
            Ok(deriv_expr)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(df),
              right: Box::new(deriv_expr),
            }))
          }
        }
        // Cylinder-function derivatives w.r.t. the argument z:
        //   D[BesselJ/Y/HankelH1/HankelH2[n,z], z] = (F[n-1,z] - F[n+1,z]) / 2
        //   D[BesselI[n,z], z]                    = (I[n-1,z] + I[n+1,z]) / 2
        //   D[BesselK[n,z], z]                    = -(K[n-1,z] + K[n+1,z]) / 2
        "BesselJ" | "BesselY" | "BesselI" | "BesselK" | "HankelH1"
        | "HankelH2"
          if args.len() == 2 =>
        {
          let dn = differentiate(&args[0], var)?;
          let dz = differentiate(&args[1], var)?;
          if matches!(dn, Expr::Integer(0)) && matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          if !matches!(dn, Expr::Integer(0)) {
            // Derivative w.r.t. order: leave unevaluated
            return Ok(Expr::FunctionCall {
              name: "D".to_string(),
              args: vec![expr.clone(), Expr::Identifier(var.to_string())]
                .into(),
            });
          }
          // Use plus_ast for canonical ordering: n-1 → Plus[-1, n], n+1 → Plus[1, n]
          let n_minus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(-1),
            args[0].clone(),
          ])
          .unwrap_or_else(|_| {
            simplify(Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(1)),
            })
          });
          let n_plus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(1),
            args[0].clone(),
          ])
          .unwrap_or_else(|_| {
            simplify(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(1)),
            })
          });
          // Evaluate the neighbour orders so a concrete n resolves its
          // negative-order symmetry (e.g. BesselK[-1, z] -> BesselK[1, z]).
          let f_nm1 =
            crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
              name: name.clone(),
              args: vec![n_minus_1, args[1].clone()].into(),
            })?;
          let f_np1 =
            crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
              name: name.clone(),
              args: vec![n_plus_1, args[1].clone()].into(),
            })?;
          // BesselI adds the neighbours; BesselK adds and negates; the rest
          // (J, Y, HankelH1, HankelH2) subtract.
          let numer = match name.as_str() {
            "BesselI" => simplify(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(f_nm1),
              right: Box::new(f_np1),
            }),
            "BesselK" => simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(-1)),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(f_nm1),
                right: Box::new(f_np1),
              }),
            }),
            _ => simplify(Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(f_nm1),
              right: Box::new(f_np1),
            }),
          };
          let half_diff = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(numer),
            right: Box::new(Expr::Integer(2)),
          };
          // Chain rule: multiply by dz
          if matches!(dz, Expr::Integer(1)) {
            Ok(half_diff)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(half_diff),
            }))
          }
        }
        // ExpIntegralEi[z]: D[ExpIntegralEi[z], z] = E^z / z
        "ExpIntegralEi" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let exp_z = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::Constant("E".to_string())),
            right: Box::new(args[0].clone()),
          };
          let result = simplify(Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(exp_z),
            right: Box::new(args[0].clone()),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // SinIntegral[z]: D[SinIntegral[z], z] = Sinc[z] * z'. Wolfram prints
        // the derivative with the named Sinc rather than Sin[z]/z.
        "SinIntegral" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let sinc = Expr::FunctionCall {
            name: "Sinc".to_string(),
            args: vec![args[0].clone()].into(),
          };
          if matches!(dz, Expr::Integer(1)) {
            Ok(sinc)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(sinc),
            }))
          }
        }
        // PolyLog[n, z]: D[_, z] = PolyLog[n-1, z] / z.
        "PolyLog" if args.len() == 2 => {
          let dn = differentiate(&args[0], var)?;
          let dz = differentiate(&args[1], var)?;
          if matches!(dn, Expr::Integer(0)) && matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          if !matches!(dn, Expr::Integer(0)) {
            return Ok(Expr::FunctionCall {
              name: "D".to_string(),
              args: vec![expr.clone(), Expr::Identifier(var.to_string())]
                .into(),
            });
          }
          let n_minus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(-1),
            args[0].clone(),
          ])?;
          let pl =
            crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
              name: "PolyLog".to_string(),
              args: vec![n_minus_1, args[1].clone()].into(),
            })?;
          let result = simplify(Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(pl),
            right: Box::new(args[1].clone()),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // ExpIntegralE[n, z]: D[_, z] = -ExpIntegralE[n-1, z].
        "ExpIntegralE" if args.len() == 2 => {
          let dn = differentiate(&args[0], var)?;
          let dz = differentiate(&args[1], var)?;
          if matches!(dn, Expr::Integer(0)) && matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          if !matches!(dn, Expr::Integer(0)) {
            return Ok(Expr::FunctionCall {
              name: "D".to_string(),
              args: vec![expr.clone(), Expr::Identifier(var.to_string())]
                .into(),
            });
          }
          let n_minus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(-1),
            args[0].clone(),
          ])?;
          let neg_e =
            crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::Integer(-1),
                Expr::FunctionCall {
                  name: "ExpIntegralE".to_string(),
                  args: vec![n_minus_1, args[1].clone()].into(),
                },
              ]
              .into(),
            })?;
          if matches!(dz, Expr::Integer(1)) {
            Ok(neg_e)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(neg_e),
            }))
          }
        }
        // AiryAiPrime[z] -> z AiryAi[z], AiryBiPrime[z] -> z AiryBi[z] (the
        // Airy ODE y'' = z y), each times z'.
        "AiryAiPrime" | "AiryBiPrime" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let base = if name == "AiryAiPrime" {
            "AiryAi"
          } else {
            "AiryBi"
          };
          let result = simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(args[0].clone()),
            right: Box::new(Expr::FunctionCall {
              name: base.to_string(),
              args: vec![args[0].clone()].into(),
            }),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // CosIntegral[z] -> Cos[z]/z, SinhIntegral[z] -> Sinh[z]/z,
        // CoshIntegral[z] -> Cosh[z]/z, each times z'.
        "CosIntegral" | "SinhIntegral" | "CoshIntegral" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let head = match name.as_str() {
            "CosIntegral" => "Cos",
            "SinhIntegral" => "Sinh",
            _ => "Cosh",
          };
          let f = Expr::FunctionCall {
            name: head.to_string(),
            args: vec![args[0].clone()].into(),
          };
          let result = simplify(Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(f),
            right: Box::new(args[0].clone()),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Incomplete elliptic integrals w.r.t. the amplitude phi:
        //   D[EllipticF[phi, m], phi] = 1 / Sqrt[1 - m Sin[phi]^2]
        //   D[EllipticE[phi, m], phi] = Sqrt[1 - m Sin[phi]^2]
        // A derivative w.r.t. the parameter m is left unevaluated.
        "EllipticF" | "EllipticE" if args.len() == 2 => {
          let dphi = differentiate(&args[0], var)?;
          let dm = differentiate(&args[1], var)?;
          if matches!(dphi, Expr::Integer(0)) && matches!(dm, Expr::Integer(0))
          {
            return Ok(Expr::Integer(0));
          }
          if !matches!(dm, Expr::Integer(0)) {
            return Ok(Expr::FunctionCall {
              name: "D".to_string(),
              args: vec![expr.clone(), Expr::Identifier(var.to_string())]
                .into(),
            });
          }
          // 1 - m Sin[phi]^2
          let inner =
            crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(args[1].clone()),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(Expr::FunctionCall {
                    name: "Sin".to_string(),
                    args: vec![args[0].clone()].into(),
                  }),
                  right: Box::new(Expr::Integer(2)),
                }),
              }),
            })
            .unwrap_or_else(|_| args[0].clone());
          let sqrt = Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![inner].into(),
          };
          let deriv = if name == "EllipticF" {
            Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(sqrt),
            }
          } else {
            sqrt
          };
          if matches!(dphi, Expr::Integer(1)) {
            Ok(deriv)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dphi),
              right: Box::new(deriv),
            }))
          }
        }
        // FresnelS[z]: D = Sin[(Pi z^2)/2] * z'
        // FresnelC[z]: D = Cos[(Pi z^2)/2] * z'
        "FresnelS" | "FresnelC" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          // (Pi * z^2) / 2, evaluated so a compound argument's square expands.
          let inner =
            crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(Expr::Constant("Pi".to_string())),
                right: Box::new(Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(args[0].clone()),
                  right: Box::new(Expr::Integer(2)),
                }),
              }),
              right: Box::new(Expr::Integer(2)),
            })
            .unwrap_or_else(|_| args[0].clone());
          let outer_head = if name == "FresnelS" { "Sin" } else { "Cos" };
          let g = Expr::FunctionCall {
            name: outer_head.to_string(),
            args: vec![inner].into(),
          };
          Ok(if matches!(dz, Expr::Integer(1)) {
            simplify(g)
          } else {
            simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(g),
              right: Box::new(dz),
            })
          })
        }
        // LogGamma[z]: D = PolyGamma[0, z] * z'
        // LogIntegral[z]: D = z' / Log[z]
        // AiryAi[z]: D = AiryAiPrime[z] * z'; AiryBi[z]: D = AiryBiPrime[z] * z'
        "LogGamma" | "LogIntegral" | "AiryAi" | "AiryBi" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let g = match name.as_str() {
            "LogGamma" => Expr::FunctionCall {
              name: "PolyGamma".to_string(),
              args: vec![Expr::Integer(0), args[0].clone()].into(),
            },
            "LogIntegral" => Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Log".to_string(),
                  args: vec![args[0].clone()].into(),
                },
                Expr::Integer(-1),
              ]
              .into(),
            },
            "AiryAi" => Expr::FunctionCall {
              name: "AiryAiPrime".to_string(),
              args: args.clone(),
            },
            _ => Expr::FunctionCall {
              name: "AiryBiPrime".to_string(),
              args: args.clone(),
            },
          };
          Ok(if matches!(dz, Expr::Integer(1)) {
            simplify(g)
          } else {
            simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(g),
            })
          })
        }
        // PolyGamma[z] = PolyGamma[0, z] (digamma): D = PolyGamma[1, z] z'
        "PolyGamma" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let g = Expr::FunctionCall {
            name: "PolyGamma".to_string(),
            args: vec![Expr::Integer(1), args[0].clone()].into(),
          };
          Ok(if matches!(dz, Expr::Integer(1)) {
            simplify(g)
          } else {
            simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(g),
            })
          })
        }
        // PolyGamma[n, z], n free of var: D[PolyGamma[n, z], z] =
        //   PolyGamma[n + 1, z] z'. When n depends on var there is no
        //   elementary form, so this arm only matches for constant n.
        "PolyGamma" if args.len() == 2 && is_constant_wrt(&args[0], var) => {
          let dz = differentiate(&args[1], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          // n + 1, canonically ordered (1 + n for symbolic n, k+1 for integer k)
          let n_plus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(1),
            args[0].clone(),
          ])
          .unwrap_or_else(|_| {
            simplify(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(1)),
            })
          });
          let g = Expr::FunctionCall {
            name: "PolyGamma".to_string(),
            args: vec![n_plus_1, args[1].clone()].into(),
          };
          Ok(if matches!(dz, Expr::Integer(1)) {
            simplify(g)
          } else {
            simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(g),
            })
          })
        }
        // Beta[z, a, b] (incomplete Beta), a and b free of var:
        //   D[Beta[z, a, b], z] = z^(a-1) (1-z)^(b-1) z'
        "Beta"
          if args.len() == 3
            && is_constant_wrt(&args[1], var)
            && is_constant_wrt(&args[2], var) =>
        {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let z = &args[0];
          // z^(a-1)
          let a_minus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(-1),
            args[1].clone(),
          ])
          .unwrap_or_else(|_| args[1].clone());
          let z_pow = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(z.clone()),
            right: Box::new(a_minus_1),
          };
          // (1-z)^(b-1)
          let b_minus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(-1),
            args[2].clone(),
          ])
          .unwrap_or_else(|_| args[2].clone());
          let one_minus_z = simplify(Expr::BinaryOp {
            op: BinaryOperator::Minus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(z.clone()),
          });
          let omz_pow = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(one_minus_z),
            right: Box::new(b_minus_1),
          };
          // Wolfram orders the (1-z) factor before the z factor; build the
          // product directly in that order rather than through the canonical
          // Times sorter (which would put the bare-symbol power first).
          let g = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(omz_pow),
            right: Box::new(z_pow),
          };
          Ok(if matches!(dz, Expr::Integer(1)) {
            g
          } else {
            simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(g),
            })
          })
        }
        // Hypergeometric1F1[a, b, z], a and b free of var:
        //   D = (a/b) Hypergeometric1F1[a+1, b+1, z] z'
        "Hypergeometric1F1"
          if args.len() == 3
            && is_constant_wrt(&args[0], var)
            && is_constant_wrt(&args[1], var) =>
        {
          let dz = differentiate(&args[2], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let a_plus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(1),
            args[0].clone(),
          ])
          .unwrap_or_else(|_| args[0].clone());
          let b_plus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(1),
            args[1].clone(),
          ])
          .unwrap_or_else(|_| args[1].clone());
          let f = Expr::FunctionCall {
            name: "Hypergeometric1F1".to_string(),
            args: vec![a_plus_1, b_plus_1, args[2].clone()].into(),
          };
          // (a * F) / b
          let g = simplify(Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![args[0].clone(), f].into(),
            }),
            right: Box::new(args[1].clone()),
          });
          Ok(if matches!(dz, Expr::Integer(1)) {
            g
          } else {
            simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(g),
            })
          })
        }
        // Hypergeometric2F1[a, b, c, z], a, b, c free of var:
        //   D = (a b / c) Hypergeometric2F1[a+1, b+1, c+1, z] z'
        "Hypergeometric2F1"
          if args.len() == 4
            && is_constant_wrt(&args[0], var)
            && is_constant_wrt(&args[1], var)
            && is_constant_wrt(&args[2], var) =>
        {
          let dz = differentiate(&args[3], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let a_plus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(1),
            args[0].clone(),
          ])
          .unwrap_or_else(|_| args[0].clone());
          let b_plus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(1),
            args[1].clone(),
          ])
          .unwrap_or_else(|_| args[1].clone());
          let c_plus_1 = crate::functions::math_ast::plus_ast(&[
            Expr::Integer(1),
            args[2].clone(),
          ])
          .unwrap_or_else(|_| args[2].clone());
          let f = Expr::FunctionCall {
            name: "Hypergeometric2F1".to_string(),
            args: vec![a_plus_1, b_plus_1, c_plus_1, args[3].clone()].into(),
          };
          // (a * b * F) / c
          let g = simplify(Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![args[0].clone(), args[1].clone(), f].into(),
            }),
            right: Box::new(args[2].clone()),
          });
          Ok(if matches!(dz, Expr::Integer(1)) {
            g
          } else {
            simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(g),
            })
          })
        }
        // InverseErf[z]:  D = (Sqrt[Pi]/2) E^(InverseErf[z]^2) z'
        // InverseErfc[z]: D = -(Sqrt[Pi]/2) E^(InverseErfc[z]^2) z'
        "InverseErf" | "InverseErfc" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          // E^(F[z]^2) * Sqrt[Pi] / 2.
          let f_sq = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::FunctionCall {
              name: name.clone(),
              args: args.clone(),
            }),
            right: Box::new(Expr::Integer(2)),
          };
          let core = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(Expr::Constant("E".to_string())),
                  right: Box::new(f_sq),
                },
                Expr::FunctionCall {
                  name: "Sqrt".to_string(),
                  args: vec![Expr::Constant("Pi".to_string())].into(),
                },
              ]
              .into(),
            }),
            right: Box::new(Expr::Integer(2)),
          };
          // InverseErfc carries an overall minus sign.
          let g = if name == "InverseErf" {
            core
          } else {
            Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(core),
            }
          };
          Ok(if matches!(dz, Expr::Integer(1)) {
            simplify(g)
          } else {
            simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(g),
            })
          })
        }
        // Gamma[z]: D[Gamma[z], z] = Gamma[z] * PolyGamma[0, z]
        "Gamma" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let result = simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Gamma".to_string(),
              args: args.clone(),
            }),
            right: Box::new(Expr::FunctionCall {
              name: "PolyGamma".to_string(),
              args: vec![Expr::Integer(0), args[0].clone()].into(),
            }),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Factorial[z] = Gamma[1 + z], so
        //   D[z!, z] = Gamma[1 + z] PolyGamma[0, 1 + z] z'.
        // (wolframscript keeps the Gamma[1 + z] form rather than z! here.)
        "Factorial" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let one_plus_z = simplify(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(args[0].clone()),
          });
          let result = simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Gamma".to_string(),
              args: vec![one_plus_z.clone()].into(),
            }),
            right: Box::new(Expr::FunctionCall {
              name: "PolyGamma".to_string(),
              args: vec![Expr::Integer(0), one_plus_z].into(),
            }),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Gamma[a, z] (upper incomplete), a free of var:
        //   d/dvar Gamma[a, z] = -z^(a-1) E^(-z) z'
        // When a depends on var the derivative needs the ∂/∂a term (a
        // MeijerG expression), so this arm only matches for constant a and
        // otherwise falls through to the generic handler.
        "Gamma" if args.len() == 2 && is_constant_wrt(&args[0], var) => {
          let dz = differentiate(&args[1], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let a = &args[0];
          let z = &args[1];
          // z^(a-1)
          let z_pow = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(z.clone()),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(a.clone()),
              right: Box::new(Expr::Integer(1)),
            }),
          };
          // E^(-z)
          let exp_neg_z = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::Constant("E".to_string())),
            right: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(z.clone()),
            }),
          };
          // -z^(a-1) E^(-z), times z' (chain rule).
          let core = Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(z_pow),
              right: Box::new(exp_neg_z),
            }),
          };
          let full = if matches!(dz, Expr::Integer(1)) {
            core
          } else {
            Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(core),
            }
          };
          crate::evaluator::evaluate_expr_to_expr(&full)
        }
        // Gamma[a, z0, z1] = Gamma[a, z0] - Gamma[a, z1], a free of var:
        // differentiate the difference so the two-argument rule applies.
        "Gamma" if args.len() == 3 && is_constant_wrt(&args[0], var) => {
          let diff = Expr::BinaryOp {
            op: BinaryOperator::Minus,
            left: Box::new(Expr::FunctionCall {
              name: "Gamma".to_string(),
              args: vec![args[0].clone(), args[1].clone()].into(),
            }),
            right: Box::new(Expr::FunctionCall {
              name: "Gamma".to_string(),
              args: vec![args[0].clone(), args[2].clone()].into(),
            }),
          };
          let d = differentiate(&diff, var)?;
          crate::evaluator::evaluate_expr_to_expr(&d)
        }
        // RealSign[u]: D[RealSign[u], u] = Piecewise[{{0, u != 0}}, Indeterminate]
        "RealSign" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let result = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![
              Expr::List(
                vec![Expr::List(
                  vec![
                    Expr::Integer(0),
                    Expr::Comparison {
                      operands: vec![args[0].clone(), Expr::Integer(0)],
                      operators: vec![ComparisonOp::NotEqual],
                    },
                  ]
                  .into(),
                )]
                .into(),
              ),
              Expr::Identifier("Indeterminate".to_string()),
            ]
            .into(),
          };
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // KroneckerDelta is 0 away from the discrete coincidence locus and has
        // zero derivative everywhere it is defined, so D[KroneckerDelta[…], x] = 0.
        "KroneckerDelta" => Ok(Expr::Integer(0)),
        // Floor/Ceiling are locally constant with a jump at the integers:
        //   D[Floor[u], x]   = D[u, x] Piecewise[{{0, u > Floor[u]}},   Indeterminate]
        //   D[Ceiling[u], x] = D[u, x] Piecewise[{{0, u < Ceiling[u]}}, Indeterminate]
        // (0 off the jumps, Indeterminate on them).
        "Floor" | "Ceiling" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let step_call = Expr::FunctionCall {
            name: name.clone(),
            args: vec![args[0].clone()].into(),
          };
          let cmp_op = if name == "Floor" {
            ComparisonOp::Greater
          } else {
            ComparisonOp::Less
          };
          let cond = Expr::Comparison {
            operands: vec![args[0].clone(), step_call],
            operators: vec![cmp_op],
          };
          let pw = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![
              Expr::List(
                vec![Expr::List(vec![Expr::Integer(0), cond].into())].into(),
              ),
              Expr::Identifier("Indeterminate".to_string()),
            ]
            .into(),
          };
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(dz),
            right: Box::new(pw),
          }))
        }
        // UnitStep[x]: D[UnitStep[x], x] = Piecewise[{{Indeterminate, x == 0}}, 0]
        // HeavisideTheta[z]: D[HeavisideTheta[z], z] = DiracDelta[z], with the
        // chain rule for a composite argument: D[HeavisideTheta[u], x] =
        // DiracDelta[u] D[u, x]. (Distinct from UnitStep, whose derivative is
        // the Piecewise below.)
        "HeavisideTheta" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          // Evaluate DiracDelta[u] so the scaling law folds a constant factor
          // (e.g. DiracDelta[2 x] -> DiracDelta[x]/2, giving the WL-canonical
          // D[HeavisideTheta[2 x], x] = DiracDelta[x]).
          let raw_dirac = Expr::FunctionCall {
            name: "DiracDelta".to_string(),
            args: vec![args[0].clone()].into(),
          };
          let dirac = crate::evaluator::evaluate_expr_to_expr(&raw_dirac)
            .unwrap_or(raw_dirac);
          Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(dz),
            right: Box::new(dirac),
          }))
        }
        "UnitStep" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let result = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![
              Expr::List(
                vec![Expr::List(
                  vec![
                    Expr::Identifier("Indeterminate".to_string()),
                    Expr::Comparison {
                      operands: vec![args[0].clone(), Expr::Integer(0)],
                      operators: vec![ComparisonOp::Equal],
                    },
                  ]
                  .into(),
                )]
                .into(),
              ),
              Expr::Integer(0),
            ]
            .into(),
          };
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Ramp[z]: D[Ramp[z], z] = Piecewise[{{0, z<0}, {1, z>0}}, Indeterminate]
        // (Ramp is flat below 0, the identity above, with a corner at 0).
        "Ramp" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let clause = |val: Expr, op: ComparisonOp| {
            Expr::List(
              vec![
                val,
                Expr::Comparison {
                  operands: vec![args[0].clone(), Expr::Integer(0)],
                  operators: vec![op],
                },
              ]
              .into(),
            )
          };
          let result = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![
              Expr::List(
                vec![
                  clause(Expr::Integer(0), ComparisonOp::Less),
                  clause(Expr::Integer(1), ComparisonOp::Greater),
                ]
                .into(),
              ),
              Expr::Identifier("Indeterminate".to_string()),
            ]
            .into(),
          };
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Erf[z0, z1] = Erf[z1] - Erf[z0]: differentiate the difference so
        // the one-argument rule supplies each term's derivative.
        "Erf" if args.len() == 2 => {
          let diff = Expr::BinaryOp {
            op: BinaryOperator::Minus,
            left: Box::new(Expr::FunctionCall {
              name: "Erf".to_string(),
              args: vec![args[1].clone()].into(),
            }),
            right: Box::new(Expr::FunctionCall {
              name: "Erf".to_string(),
              args: vec![args[0].clone()].into(),
            }),
          };
          differentiate(&diff, var)
        }
        // Erf[z]: D[Erf[z], z] = 2*E^(-z^2)/Sqrt[Pi]
        "Erf" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let z_sq = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(args[0].clone()),
            right: Box::new(Expr::Integer(2)),
          };
          let exp_neg_z2 = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::Constant("E".to_string())),
            right: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(z_sq),
            }),
          };
          let two_over_sqrt_pi = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expr::Integer(2)),
            right: Box::new(make_sqrt(Expr::Constant("Pi".to_string()))),
          };
          let result = simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(two_over_sqrt_pi),
            right: Box::new(exp_neg_z2),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Erfc[z]: D[Erfc[z], z] = -2*E^(-z^2)/Sqrt[Pi]
        "Erfc" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let z_sq = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(args[0].clone()),
            right: Box::new(Expr::Integer(2)),
          };
          let neg_exp_neg_z2 = Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(Expr::Constant("E".to_string())),
              right: Box::new(Expr::UnaryOp {
                op: UnaryOperator::Minus,
                operand: Box::new(z_sq),
              }),
            }),
          };
          let two_over_sqrt_pi = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expr::Integer(2)),
            right: Box::new(make_sqrt(Expr::Constant("Pi".to_string()))),
          };
          let result = simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(two_over_sqrt_pi),
            right: Box::new(neg_exp_neg_z2),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Erfi[z]: D[Erfi[z], z] = 2*E^(z^2)/Sqrt[Pi]
        "Erfi" if args.len() == 1 => {
          let dz = differentiate(&args[0], var)?;
          if matches!(dz, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          let z_sq = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(args[0].clone()),
            right: Box::new(Expr::Integer(2)),
          };
          let exp_z2 = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::Constant("E".to_string())),
            right: Box::new(z_sq),
          };
          let two_over_sqrt_pi = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expr::Integer(2)),
            right: Box::new(make_sqrt(Expr::Constant("Pi".to_string()))),
          };
          let result = simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(two_over_sqrt_pi),
            right: Box::new(exp_z2),
          });
          if matches!(dz, Expr::Integer(1)) {
            Ok(result)
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dz),
              right: Box::new(result),
            }))
          }
        }
        // Handle Rational[n, d] as constant
        "Rational" if args.len() == 2 => Ok(Expr::Integer(0)),
        // Handle Integrate[f, {t, a, b}] via Leibniz integral rule
        "Integrate" if args.len() == 2 => {
          if let Expr::List(spec) = &args[1]
            && spec.len() == 3
          {
            // D[Integrate[f[t, x], {t, a(x), b(x)}], x]
            //   = Integrate[∂f/∂x, {t, a(x), b(x)}]
            //     + f[b(x), x] * b'(x)
            //     - f[a(x), x] * a'(x)
            let integrand = &args[0];
            let int_var = &spec[0];
            let lo = &spec[1];
            let hi = &spec[2];

            let int_var_name = match int_var {
              Expr::Identifier(n) => n.as_str(),
              _ => "",
            };

            let da = differentiate(lo, var)?;
            let db = differentiate(hi, var)?;

            // Substitute integration variable with upper/lower bound in integrand
            let f_at_hi = if !int_var_name.is_empty() {
              crate::syntax::substitute_variable(integrand, int_var_name, hi)
            } else {
              integrand.clone()
            };
            let f_at_lo = if !int_var_name.is_empty() {
              crate::syntax::substitute_variable(integrand, int_var_name, lo)
            } else {
              integrand.clone()
            };

            let mut terms: Vec<Expr> = Vec::new();

            // Inside-integral term: Integrate[∂f/∂x, {t, a(x), b(x)}].
            // Only include when the integrand actually depends on `var`,
            // since otherwise the partial is identically 0.
            if !is_constant_wrt(integrand, var) {
              let inner_deriv = differentiate(integrand, var)?;
              if !matches!(inner_deriv, Expr::Integer(0)) {
                terms.push(Expr::FunctionCall {
                  name: "Integrate".to_string(),
                  args: vec![
                    inner_deriv,
                    Expr::List(
                      vec![int_var.clone(), lo.clone(), hi.clone()].into(),
                    ),
                  ]
                  .into(),
                });
              }
            }

            // f[b(x)] * b'(x) term
            if !matches!(db, Expr::Integer(0)) {
              let term = if matches!(db, Expr::Integer(1)) {
                simplify(f_at_hi)
              } else {
                simplify(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(simplify(f_at_hi)),
                  right: Box::new(db),
                })
              };
              terms.push(term);
            }
            // -f[a(x)] * a'(x) term
            if !matches!(da, Expr::Integer(0)) {
              let term = if matches!(da, Expr::Integer(1)) {
                simplify(f_at_lo)
              } else {
                simplify(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(simplify(f_at_lo)),
                  right: Box::new(da),
                })
              };
              terms.push(Expr::UnaryOp {
                op: UnaryOperator::Minus,
                operand: Box::new(term),
              });
            }

            if terms.is_empty() {
              Ok(Expr::Integer(0))
            } else if terms.len() == 1 {
              Ok(simplify(terms.remove(0)))
            } else {
              let mut result = terms[0].clone();
              for t in terms.iter().skip(1) {
                result = Expr::BinaryOp {
                  op: BinaryOperator::Plus,
                  left: Box::new(result),
                  right: Box::new(t.clone()),
                };
              }
              Ok(simplify(result))
            }
          } else {
            // Indefinite integral Integrate[g, t]. With the integration
            // variable t:
            //   D[Integrate[g, t], t] = g          (fundamental theorem), and
            //   D[Integrate[g, t], x] = Integrate[D[g, x], t]
            //                                       (differentiation under the
            //                                        integral sign).
            let int_var = match &args[1] {
              Expr::Identifier(n) => Some(n.clone()),
              Expr::List(s) if s.len() == 1 => match &s[0] {
                Expr::Identifier(n) => Some(n.clone()),
                _ => None,
              },
              _ => None,
            };
            match int_var {
              Some(iv) if iv == var => Ok(args[0].clone()),
              Some(iv) => {
                let dg = differentiate(&args[0], var)?;
                crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                  name: "Integrate".to_string(),
                  args: vec![dg, Expr::Identifier(iv)].into(),
                })
              }
              None => Ok(Expr::FunctionCall {
                name: "D".to_string(),
                args: vec![expr.clone(), Expr::Identifier(var.to_string())]
                  .into(),
              }),
            }
          }
        }
        // Flattened Derivative[n, f, x]: this is the evaluated form of
        // Derivative[n][f][x] (n-th derivative of f at x).
        // D[Derivative[n, f, x], x] = Derivative[n+1][f][x] * D[x, x]
        // via chain rule, where D[x,x] = 1 for simple variable.
        "Derivative" if args.len() == 3 => {
          if is_constant_wrt(expr, var) {
            return Ok(Expr::Integer(0));
          }
          let order = &args[0];
          let func = &args[1];
          let inner = &args[2];
          let d_inner = differentiate(inner, var)?;
          if matches!(d_inner, Expr::Integer(0)) {
            return Ok(Expr::Integer(0));
          }
          // Build Derivative[n+1][f][x]
          let new_order = if let Expr::Integer(k) = order {
            Expr::Integer(k + 1)
          } else {
            Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(order.clone()),
            }
          };
          let deriv_expr = Expr::CurriedCall {
            func: Box::new(Expr::CurriedCall {
              func: Box::new(Expr::FunctionCall {
                name: "Derivative".to_string(),
                args: vec![new_order].into(),
              }),
              args: vec![func.clone()],
            }),
            args: vec![inner.clone()],
          };
          if matches!(&d_inner, Expr::Integer(1)) {
            Ok(simplify(deriv_expr))
          } else {
            Ok(simplify(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(d_inner),
              right: Box::new(deriv_expr),
            }))
          }
        }
        // General chain rule for unknown functions: D[f[g1(x),...,gn(x)], x]
        // = Sum_i Derivative[0,...,1,...,0][f][g1,...,gn] * D[gi, x]
        _ => {
          // If the function is entirely constant w.r.t. var, return 0
          if is_constant_wrt(expr, var) {
            return Ok(Expr::Integer(0));
          }

          // Compute derivatives of each argument
          let n = args.len();
          let mut dargs: Vec<Expr> = Vec::with_capacity(n);
          for arg in args {
            dargs.push(differentiate(arg, var)?);
          }

          // The derivative index for an argument that is itself a list is a
          // (structurally matching) list of zeros, mirroring Wolfram:
          // D[f[x, {1, 2, 3}], x] = Derivative[1, {0, 0, 0}][f][x, {1, 2, 3}].
          fn structured_zero(arg: &Expr) -> Expr {
            match arg {
              Expr::List(items) => Expr::List(
                items.iter().map(structured_zero).collect::<Vec<_>>().into(),
              ),
              _ => Expr::Integer(0),
            }
          }
          // An argument that does not depend on `var` has an all-zero
          // derivative (a scalar 0 or a list of zeros) and contributes nothing.
          fn is_all_zero(e: &Expr) -> bool {
            match e {
              Expr::Integer(0) => true,
              Expr::Real(f) => *f == 0.0,
              Expr::List(items) => items.iter().all(is_all_zero),
              _ => false,
            }
          }

          // Build chain rule sum
          let mut terms: Vec<Expr> = Vec::new();
          for i in 0..n {
            if is_all_zero(&dargs[i]) {
              continue;
            }

            // Build Derivative[0,...,1,...,0][f][g1,...,gn]
            let deriv_indices: Vec<Expr> = (0..n)
              .map(|j| {
                if j == i {
                  Expr::Integer(1)
                } else {
                  structured_zero(&args[j])
                }
              })
              .collect();

            // If `name` is itself the stringified head of an outer Derivative
            // (e.g. `"Derivative[1, 0][F]"` — produced when curried-derivative
            // application falls through to `expr_to_string` in
            // `apply_curried_call`), parse those indices and combine them with
            // the new `deriv_indices` so we end up with a flat
            // `Derivative[a+1, b][F][...]` rather than a nested
            // `Derivative[0, 1][Derivative[a, b][F]][...]`.
            let deriv_expr = if let Some((outer_indices, inner_fn_name)) =
              parse_stringified_derivative_head(name)
              && outer_indices.len() == n
            {
              let combined: Vec<Expr> = outer_indices
                .iter()
                .zip(deriv_indices.iter())
                .map(|(a, b)| match (a, b) {
                  (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x + y),
                  // A structured-zero (list) index adds nothing: keep `a`.
                  _ if is_all_zero(b) => a.clone(),
                  _ => Expr::BinaryOp {
                    op: BinaryOperator::Plus,
                    left: Box::new(a.clone()),
                    right: Box::new(b.clone()),
                  },
                })
                .collect();
              Expr::CurriedCall {
                func: Box::new(Expr::CurriedCall {
                  func: Box::new(Expr::FunctionCall {
                    name: "Derivative".to_string(),
                    args: combined.into(),
                  }),
                  args: vec![Expr::Identifier(inner_fn_name)],
                }),
                args: args.to_vec(),
              }
            } else {
              Expr::CurriedCall {
                func: Box::new(Expr::CurriedCall {
                  func: Box::new(Expr::FunctionCall {
                    name: "Derivative".to_string(),
                    args: deriv_indices.into(),
                  }),
                  args: vec![Expr::Identifier(name.clone())],
                }),
                args: args.to_vec(),
              }
            };

            if matches!(&dargs[i], Expr::Integer(1)) {
              terms.push(deriv_expr);
            } else {
              terms.push(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(dargs[i].clone()),
                right: Box::new(deriv_expr),
              });
            }
          }

          if terms.is_empty() {
            Ok(Expr::Integer(0))
          } else if terms.len() == 1 {
            Ok(simplify(terms.remove(0)))
          } else {
            let mut result = terms[0].clone();
            for term in &terms[1..] {
              result = Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(result),
                right: Box::new(term.clone()),
              };
            }
            Ok(simplify(result))
          }
        }
      }
    }

    // CurriedCall: handles Derivative[n1,...,nk][f][g1,...,gk] expressions
    // and InverseFunction[f][x]
    Expr::CurriedCall { func, args } => {
      // Handle InverseFunction[f][x]:
      // D[InverseFunction[f][x], x] = 1 / Derivative[1][f][InverseFunction[f][x]]
      if let Expr::FunctionCall {
        name: inv_name,
        args: inv_args,
      } = func.as_ref()
        && inv_name == "InverseFunction"
        && inv_args.len() == 1
        && args.len() == 1
      {
        let dx = differentiate(&args[0], var)?;
        if matches!(dx, Expr::Integer(0)) {
          return Ok(Expr::Integer(0));
        }
        // 1 / f'(InverseFunction[f][x])
        let deriv_f_at_inv = Expr::CurriedCall {
          func: Box::new(Expr::CurriedCall {
            func: Box::new(Expr::FunctionCall {
              name: "Derivative".to_string(),
              args: vec![Expr::Integer(1)].into(),
            }),
            args: inv_args.to_vec(),
          }),
          args: vec![expr.clone()],
        };
        let result = Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(deriv_f_at_inv),
          right: Box::new(Expr::Integer(-1)),
        };
        if matches!(dx, Expr::Integer(1)) {
          return Ok(result);
        } else {
          return Ok(simplify(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(dx),
            right: Box::new(result),
          }));
        }
      }

      // Check if this is Derivative[...][f][args]
      if let Expr::CurriedCall {
        func: inner_func,
        args: func_names,
      } = func.as_ref()
        && let Expr::FunctionCall {
          name: deriv_name,
          args: indices,
        } = inner_func.as_ref()
        && deriv_name == "Derivative"
        && func_names.len() == 1
        && indices.len() == args.len()
      {
        // D[Derivative[n1,...,nk][f][g1,...,gk], x]
        // = Sum_i Derivative[n1,...,ni+1,...,nk][f][g1,...,gk] * D[gi, x]
        let n = args.len();
        let mut dargs: Vec<Expr> = Vec::with_capacity(n);
        for arg in args {
          dargs.push(differentiate(arg, var)?);
        }

        let mut terms: Vec<Expr> = Vec::new();
        for i in 0..n {
          if matches!(&dargs[i], Expr::Integer(0)) {
            continue;
          }

          // Increment the i-th derivative index
          let new_indices: Vec<Expr> = indices
            .iter()
            .enumerate()
            .map(|(j, idx)| {
              if j == i {
                if let Expr::Integer(k) = idx {
                  Expr::Integer(k + 1)
                } else {
                  Expr::BinaryOp {
                    op: BinaryOperator::Plus,
                    left: Box::new(Expr::Integer(1)),
                    right: Box::new(idx.clone()),
                  }
                }
              } else {
                idx.clone()
              }
            })
            .collect();

          let deriv_expr = Expr::CurriedCall {
            func: Box::new(Expr::CurriedCall {
              func: Box::new(Expr::FunctionCall {
                name: "Derivative".to_string(),
                args: new_indices.into(),
              }),
              args: func_names.clone(),
            }),
            args: args.clone(),
          };

          if matches!(&dargs[i], Expr::Integer(1)) {
            terms.push(deriv_expr);
          } else {
            terms.push(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(dargs[i].clone()),
              right: Box::new(deriv_expr),
            });
          }
        }

        return if terms.is_empty() {
          Ok(Expr::Integer(0))
        } else if terms.len() == 1 {
          Ok(simplify(terms.remove(0)))
        } else {
          let mut result = terms[0].clone();
          for term in &terms[1..] {
            result = Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(result),
              right: Box::new(term.clone()),
            };
          }
          Ok(simplify(result))
        };
      }
      // Fallback for other CurriedCall forms
      if is_constant_wrt(expr, var) {
        Ok(Expr::Integer(0))
      } else {
        Ok(Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
        })
      }
    }

    // Lists differentiate element-wise. The fallback below would fold a
    // constant-w.r.t.-`var` list to a single `0` via `is_constant_wrt`,
    // which trips the `Derivative[n][List]` chain (after the first
    // derivative the list contains only constants like `{1}`, and the
    // second pass needs to yield `{0}` rather than `0`).
    Expr::List(items) => {
      let mut diffed: Vec<Expr> = Vec::with_capacity(items.len());
      for item in items {
        diffed.push(differentiate(item, var)?);
      }
      Ok(Expr::List(diffed.into()))
    }

    _ => {
      if is_constant_wrt(expr, var) {
        Ok(Expr::Integer(0))
      } else {
        Ok(Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
        })
      }
    }
  }
}

/// Build `expr / divisor`, simplifying to just `expr` when `divisor == 1`.
/// For integer divisors, produces `Rational[1, n] * expr` to match Wolfram output.
fn make_divided(expr: Expr, divisor: Expr) -> Expr {
  match &divisor {
    Expr::Integer(1) => expr,
    // expr / (a/b) → expr * b/a = (b * expr) / a
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: num,
      right: den,
    } => {
      let result = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: den.clone(),
          right: Box::new(expr),
        }),
        right: num.clone(),
      };
      simplify(result)
    }
    // expr / Rational[a, b] → (b * expr) / a
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      let num = &args[0]; // a
      let den = &args[1]; // b
      let result = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(den.clone()),
          right: Box::new(expr),
        }),
        right: Box::new(num.clone()),
      };
      simplify(result)
    }
    _ => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(expr),
      right: Box::new(divisor),
    },
  }
}

/// Build `-expr / divisor`, simplifying to `-expr` when `divisor == 1`.
/// For integer divisors, produces `Rational[-1, n] * expr` to match Wolfram output.
fn make_neg_divided(expr: Expr, divisor: Expr) -> Expr {
  match &divisor {
    Expr::Integer(1) => Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(expr),
    },
    Expr::Integer(n) => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(crate::functions::math_ast::make_rational(-1, *n)),
      right: Box::new(expr),
    },
    _ => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(expr),
      }),
      right: Box::new(divisor),
    },
  }
}

/// Extract a trig function with its power from an expression.
/// Returns (function_name, argument, power) for Sin[f]^n or Cos[f]^n patterns.
/// Power defaults to 1 if not explicitly raised.
fn extract_trig_factor(expr: &Expr) -> Option<(&str, &Expr, i64)> {
  // Sin[f] or Cos[f] (power = 1)
  if let Expr::FunctionCall { name, args } = expr
    && args.len() == 1
    && (name == "Sin" || name == "Cos")
  {
    return Some((name.as_str(), &args[0], 1));
  }
  // Sin[f]^n or Cos[f]^n as BinaryOp
  if let Expr::BinaryOp {
    op: BinaryOperator::Power,
    left,
    right,
  } = expr
    && let Expr::Integer(n) = right.as_ref()
    && *n >= 1
    && let Expr::FunctionCall { name, args } = left.as_ref()
    && args.len() == 1
    && (name == "Sin" || name == "Cos")
  {
    return Some((name.as_str(), &args[0], *n as i64));
  }
  // Power[Sin[f], n] or Power[Cos[f], n] as FunctionCall
  if let Expr::FunctionCall { name, args } = expr
    && name == "Power"
    && args.len() == 2
    && let Expr::Integer(n) = &args[1]
    && *n >= 1
    && let Expr::FunctionCall {
      name: trig_name,
      args: trig_args,
    } = &args[0]
    && trig_args.len() == 1
    && (trig_name == "Sin" || trig_name == "Cos")
  {
    return Some((trig_name.as_str(), &trig_args[0], *n as i64));
  }
  None
}

/// The reciprocal of a (hyperbolic) trig function, used to rewrite
/// `Cos[x]^-n` as `Sec[x]^n` etc. so the named-reciprocal integrators apply.
fn reciprocal_trig_name(name: &str) -> Option<&'static str> {
  Some(match name {
    "Cos" => "Sec",
    "Sin" => "Csc",
    "Tan" => "Cot",
    "Sec" => "Cos",
    "Csc" => "Sin",
    "Cot" => "Tan",
    "Cosh" => "Sech",
    "Sinh" => "Csch",
    "Tanh" => "Coth",
    "Sech" => "Cosh",
    "Csch" => "Sinh",
    "Coth" => "Tanh",
    _ => return None,
  })
}

/// Try to integrate `E^(a x) Sin[b x]` or `E^(a x) Cos[b x]` (with `a`, `b`
/// constant w.r.t. `var` and the exponent / trig argument linear in `var`):
///   ∫ E^(a x) Sin[b x] dx = E^(a x) (a Sin[b x] - b Cos[b x]) / (a^2 + b^2)
///   ∫ E^(a x) Cos[b x] dx = E^(a x) (a Cos[b x] + b Sin[b x]) / (a^2 + b^2)
/// A constant phase/offset in either argument is preserved automatically by
/// carrying the original exponent and trig argument through unchanged.
fn try_integrate_exp_trig_product(
  factors: &[&Expr],
  var: &str,
) -> Option<Expr> {
  if factors.len() != 2 {
    return None;
  }

  // The exponent of an `E^expo` / `Exp[expo]` factor, else None.
  let exp_exponent = |f: &Expr| -> Option<Expr> {
    match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if matches!(left.as_ref(), Expr::Constant(c) | Expr::Identifier(c) if c == "E") => {
        Some((**right).clone())
      }
      Expr::FunctionCall { name, args }
        if name == "Power"
          && args.len() == 2
          && matches!(&args[0], Expr::Constant(c) | Expr::Identifier(c) if c == "E") =>
      {
        Some(args[1].clone())
      }
      Expr::FunctionCall { name, args } if name == "Exp" && args.len() == 1 => {
        Some(args[0].clone())
      }
      _ => None,
    }
  };
  // `(head, arg)` if `f` is `Sin[arg]` or `Cos[arg]`, else None.
  let trig_parts = |f: &Expr| -> Option<(String, Expr)> {
    if let Expr::FunctionCall { name, args } = f
      && (name == "Sin" || name == "Cos")
      && args.len() == 1
    {
      Some((name.clone(), args[0].clone()))
    } else {
      None
    }
  };

  // Try both orderings of the two factors.
  for (ef, tf) in [(factors[0], factors[1]), (factors[1], factors[0])] {
    let Some(exponent) = exp_exponent(ef) else {
      continue;
    };
    let Some((trig, arg)) = trig_parts(tf) else {
      continue;
    };
    // a = d/dx(exponent), b = d/dx(arg); both must be constant (linear args).
    let a = differentiate(&exponent, var).ok()?;
    let b = differentiate(&arg, var).ok()?;
    if !is_constant_wrt(&a, var) || !is_constant_wrt(&b, var) {
      continue;
    }

    let times = |x: Expr, y: Expr| Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![x, y].into(),
    };
    let cos = Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![arg.clone()].into(),
    };
    let sin = Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![arg.clone()].into(),
    };
    // Numerator combination (Cos term first, matching wolframscript):
    //   Sin → -b Cos + a Sin ; Cos → a Cos + b Sin
    let combo = if trig == "Sin" {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          times(times(Expr::Integer(-1), b.clone()), cos),
          times(a.clone(), sin),
        ]
        .into(),
      }
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![times(a.clone(), cos), times(b.clone(), sin)].into(),
      }
    };
    // Denominator a^2 + b^2.
    let denom = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(a.clone()),
          right: Box::new(Expr::Integer(2)),
        },
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(b.clone()),
          right: Box::new(Expr::Integer(2)),
        },
      ]
      .into(),
    };
    let result = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(times(ef.clone(), combo)),
      right: Box::new(denom),
    };
    return crate::evaluator::evaluate_expr_to_expr(&result).ok();
  }
  None
}

/// Try to integrate a product of Sin[f]^m * Cos[f]^n where f is linear in var.
/// Handles:
///   - Sin[f] * Cos[f]^n → -Cos[f]^(n+1) / ((n+1)*a)
///   - Sin[f]^m * Cos[f] → Sin[f]^(m+1) / ((m+1)*a)
///   - General odd power cases via reduction
fn try_integrate_sin_cos_product(factors: &[&Expr], var: &str) -> Option<Expr> {
  let mut sin_arg: Option<&Expr> = None;
  let mut sin_power: i64 = 0;
  let mut cos_arg: Option<&Expr> = None;
  let mut cos_power: i64 = 0;

  for factor in factors {
    if let Some((name, arg, power)) = extract_trig_factor(factor) {
      match name {
        "Sin" => {
          if sin_power > 0 && !expr_str_eq(sin_arg.unwrap(), arg) {
            return None;
          }
          sin_arg = Some(arg);
          sin_power += power;
        }
        "Cos" => {
          if cos_power > 0 && !expr_str_eq(cos_arg.unwrap(), arg) {
            return None;
          }
          cos_arg = Some(arg);
          cos_power += power;
        }
        _ => return None,
      }
    } else {
      // Non-trig factor that depends on var: can't handle
      if !is_constant_wrt(factor, var) {
        return None;
      }
    }
  }

  // Need both Sin and Cos present
  if sin_power == 0 || cos_power == 0 {
    return None;
  }

  let sin_a = sin_arg?;
  let cos_a = cos_arg?;
  // Arguments must be the same
  if !expr_str_eq(sin_a, cos_a) {
    return None;
  }

  let arg = sin_a;
  let coeff = try_match_linear_arg(arg, var)?;

  // Special case: Sin[a*x] * Cos[a*x] with |a| > 1 — use the double-angle
  // identity so the result matches wolframscript's preferred form:
  //   Sin[f] * Cos[f] = Sin[2f] / 2
  //   ∫ Sin[f] * Cos[f] dx = -Cos[2f] / (4*a)
  // (wolframscript keeps -Cos[x]^2/2 for the plain a=1 case, but switches
  // to -Cos[2a*x]/(4a) for larger coefficients such as a=2, 3, …)
  if sin_power == 1 && cos_power == 1 {
    let coeff_is_nontrivial = match &coeff {
      Expr::Integer(n) => n.abs() != 1,
      Expr::Real(r) => (r.abs() - 1.0).abs() > f64::EPSILON,
      _ => false,
    };
    if coeff_is_nontrivial {
      let double_arg = simplify(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(arg.clone()),
      });
      let cos_double = Expr::FunctionCall {
        name: "Cos".to_string(),
        args: vec![double_arg].into(),
      };
      let divisor = simplify(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(4)),
        right: Box::new(coeff.clone()),
      });
      return Some(make_neg_divided(cos_double, divisor));
    }
  }

  // When sin_power is odd (priority) or == 1: use u = Cos[f] substitution
  // ∫ Sin[f]^m * Cos[f]^n dx where m is odd:
  //   Factor out Sin[f], convert Sin[f]^(m-1) = (1-Cos[f]^2)^((m-1)/2)
  //   u = Cos[f], du = -a*Sin[f]dx
  //   = -1/a * ∫ (1-u^2)^((m-1)/2) * u^n du
  //
  // When sin_power == 1: ∫ Sin[f] * Cos[f]^n dx = -Cos[f]^(n+1) / ((n+1)*a)
  if sin_power == 1 {
    let new_power = cos_power + 1;
    let cos_expr = Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![arg.clone()].into(),
    };
    let power_expr = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(cos_expr),
      right: Box::new(Expr::Integer(new_power as i128)),
    };
    // divisor = (n+1) * a
    let total_divisor = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(new_power as i128)),
      right: Box::new(coeff),
    });
    return Some(make_neg_divided(power_expr, total_divisor));
  }

  // When cos_power == 1: ∫ Sin[f]^m * Cos[f] dx = Sin[f]^(m+1) / ((m+1)*a)
  if cos_power == 1 {
    let new_power = sin_power + 1;
    let sin_expr = Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![arg.clone()].into(),
    };
    let power_expr = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(sin_expr),
      right: Box::new(Expr::Integer(new_power as i128)),
    };
    // divisor = (m+1) * a
    let total_divisor = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(new_power as i128)),
      right: Box::new(coeff),
    });
    return Some(make_divided(power_expr, total_divisor));
  }

  // General case: both powers > 1
  // If sin_power is odd: reduce using Sin^2 = 1 - Cos^2
  if sin_power % 2 == 1 {
    // Sin[f]^m * Cos[f]^n = Sin[f] * (1-Cos[f]^2)^((m-1)/2) * Cos[f]^n
    // Expand (1-Cos[f]^2)^k and integrate each term with the sin_power=1 rule
    let k = (sin_power - 1) / 2;
    let cos_f = Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![arg.clone()].into(),
    };
    // Expand (1-u^2)^k using binomial theorem
    // = sum_{j=0}^{k} C(k,j) * (-1)^j * u^(2j)
    // So integral = sum_{j=0}^{k} C(k,j) * (-1)^j * ∫ Sin[f] * Cos[f]^(n+2j) dx
    //            = sum_{j=0}^{k} C(k,j) * (-1)^j * (-Cos[f]^(n+2j+1) / ((n+2j+1)*a))
    let mut terms: Vec<Expr> = Vec::new();
    for j in 0..=k {
      let binom = crate::functions::binomial_coeff(k as i128, j as i128);
      let sign = if j % 2 == 0 { 1i128 } else { -1 };
      let new_cos_power = cos_power + 2 * j;
      let new_power = new_cos_power + 1;
      let cos_power_expr = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(cos_f.clone()),
        right: Box::new(Expr::Integer(new_power as i128)),
      };
      // coefficient = binom * sign * (-1) / ((new_power) * a)
      // = -binom * sign / (new_power * a)
      let numer = -sign * binom;
      let total_divisor = simplify(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(new_power as i128)),
        right: Box::new(coeff.clone()),
      });
      let term = if numer == 1 {
        make_divided(cos_power_expr, total_divisor)
      } else if numer == -1 {
        make_neg_divided(cos_power_expr, total_divisor)
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(make_divided(Expr::Integer(numer), total_divisor)),
          right: Box::new(cos_power_expr),
        }
      };
      terms.push(term);
    }
    return Some(if terms.len() == 1 {
      terms.remove(0)
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      }
    });
  }

  // If cos_power is odd: reduce using Cos^2 = 1 - Sin^2
  if cos_power % 2 == 1 {
    let k = (cos_power - 1) / 2;
    let sin_f = Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![arg.clone()].into(),
    };
    let mut terms: Vec<Expr> = Vec::new();
    for j in 0..=k {
      let binom = crate::functions::binomial_coeff(k as i128, j as i128);
      let sign = if j % 2 == 0 { 1i128 } else { -1 };
      let new_sin_power = sin_power + 2 * j;
      let new_power = new_sin_power + 1;
      let sin_power_expr = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(sin_f.clone()),
        right: Box::new(Expr::Integer(new_power as i128)),
      };
      let numer = sign * binom;
      let total_divisor = simplify(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(new_power as i128)),
        right: Box::new(coeff.clone()),
      });
      let term = if numer == 1 {
        make_divided(sin_power_expr, total_divisor)
      } else if numer == -1 {
        make_neg_divided(sin_power_expr, total_divisor)
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(make_divided(Expr::Integer(numer), total_divisor)),
          right: Box::new(sin_power_expr),
        }
      };
      terms.push(term);
    }
    return Some(if terms.len() == 1 {
      terms.remove(0)
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      }
    });
  }

  // Both even: use double-angle reduction (not yet implemented)
  None
}

/// Extract any trig function with an integer power (possibly negative) from a
/// factor: `Tan[f]`, `Sec[f]^3`, `Cos[f]^-2`, … Returns (name, argument, power).
fn extract_any_trig_factor(expr: &Expr) -> Option<(&str, &Expr, i64)> {
  const TRIGS: [&str; 6] = ["Sin", "Cos", "Tan", "Cot", "Sec", "Csc"];
  if let Expr::FunctionCall { name, args } = expr
    && args.len() == 1
    && TRIGS.contains(&name.as_str())
  {
    return Some((name.as_str(), &args[0], 1));
  }
  let (base, n) = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => match right.as_ref() {
      Expr::Integer(n) => (left.as_ref(), *n),
      _ => return None,
    },
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      match &args[1] {
        Expr::Integer(n) => (&args[0], *n),
        _ => return None,
      }
    }
    _ => return None,
  };
  if let Expr::FunctionCall { name, args } = base
    && args.len() == 1
    && TRIGS.contains(&name.as_str())
  {
    return Some((name.as_str(), &args[0], n as i64));
  }
  None
}

/// Integrate products of trig functions of a common linear argument that
/// reduce to `Sin[u]^p / Cos[u]` or `Cos[u]^p / Sin[u]` (e.g. `Sin[x]*Tan[x]`,
/// `Sin[x]^2/Cos[x]`, `Cos[x]*Cot[x]`), matching wolframscript's closed forms:
///
///   ∫ Sin[u]^p/Cos[u] dx, p even:
///     ArcTanh[Sin[u]]/a - Σ_{j odd < p} Sin[u]^j/(j a)
///   ∫ Sin[u]^p/Cos[u] dx, p odd (k=(p-1)/2):
///     -Log[Cos[u]]/a + Σ_{j=1..k} (-1)^(j+1) C(k,j) Cos[u]^(2j)/(2j a)
///   ∫ Cos[u]^p/Sin[u] dx, p odd (k=(p-1)/2):
///     Log[Sin[u]]/a + Σ_{j=1..k} (-1)^j C(k,j) Sin[u]^(2j)/(2j a)
///   ∫ Cos[u]^2/Sin[u] dx:
///     Cos[u]/a + Log[Tan[u/2]]/a
///
/// (`u = a x`.) Even p ≥ 4 in the Cos family needs wolframscript's
/// multiple-angle presentation and is left unevaluated.
fn try_integrate_trig_quotient(
  num_factors: &[&Expr],
  den_factors: &[&Expr],
  var: &str,
) -> Option<Expr> {
  let mut factors: Vec<(&str, &Expr, i64)> = Vec::new();
  for factor in num_factors {
    factors.push(extract_any_trig_factor(factor)?);
  }
  for factor in den_factors {
    let (name, a, power) = extract_any_trig_factor(factor)?;
    factors.push((name, a, -power));
  }
  let mut sin_pow: i64 = 0;
  let mut cos_pow: i64 = 0;
  let mut arg: Option<&Expr> = None;
  for &(name, a, power) in &factors {
    match arg {
      None => arg = Some(a),
      Some(existing) => {
        if !expr_str_eq(existing, a) {
          return None;
        }
      }
    }
    let (ds, dc) = match name {
      "Sin" => (1, 0),
      "Cos" => (0, 1),
      "Tan" => (1, -1),
      "Cot" => (-1, 1),
      "Sec" => (0, -1),
      "Csc" => (-1, 0),
      _ => return None,
    };
    sin_pow += ds * power;
    cos_pow += dc * power;
  }
  let arg = arg?;
  let coeff = try_match_linear_arg(arg, var)?;

  // Build `Func[u]^n` (n == 1 gives the bare call).
  let trig_pow = |name: &str, n: i64| -> Expr {
    let call = Expr::FunctionCall {
      name: name.to_string(),
      args: vec![arg.clone()].into(),
    };
    if n == 1 {
      call
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(call),
        right: Box::new(Expr::Integer(n as i128)),
      }
    }
  };
  // Build `n/d * expr / a` with the rational coefficient reduced.
  let coeff_term = |n: i128, d: i128, expr: Expr, a: &Expr| -> Expr {
    let divisor = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(d)),
      right: Box::new(a.clone()),
    });
    if n == 1 {
      make_divided(expr, divisor)
    } else if n == -1 {
      make_neg_divided(expr, divisor)
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(make_divided(Expr::Integer(n), divisor)),
        right: Box::new(expr),
      }
    }
  };
  let wrap_call = |name: &str, inner: Expr| -> Expr {
    Expr::FunctionCall {
      name: name.to_string(),
      args: vec![inner].into(),
    }
  };

  if cos_pow == -1 && sin_pow == 1 {
    // ∫ Sin[u]/Cos[u] dx = -Log[Cos[u]]/a
    return Some(coeff_term(
      -1,
      1,
      wrap_call("Log", trig_pow("Cos", 1)),
      &coeff,
    ));
  }
  if sin_pow == -1 && cos_pow == 1 {
    // ∫ Cos[u]/Sin[u] dx = Log[Sin[u]]/a
    return Some(coeff_term(
      1,
      1,
      wrap_call("Log", trig_pow("Sin", 1)),
      &coeff,
    ));
  }

  if cos_pow == -1 && sin_pow >= 2 {
    let p = sin_pow;
    let mut terms: Vec<Expr> = Vec::new();
    if p % 2 == 0 {
      // ArcTanh[Sin[u]]/a - Sin[u]/a - Sin[u]^3/(3a) - … - Sin[u]^(p-1)/((p-1)a)
      terms.push(coeff_term(
        1,
        1,
        wrap_call("ArcTanh", trig_pow("Sin", 1)),
        &coeff,
      ));
      let mut j = 1;
      while j < p {
        terms.push(coeff_term(-1, j as i128, trig_pow("Sin", j), &coeff));
        j += 2;
      }
    } else {
      // -Log[Cos[u]]/a + Σ (-1)^(j+1) C(k,j) Cos[u]^(2j)/(2j a)
      let k = (p - 1) / 2;
      terms.push(coeff_term(
        -1,
        1,
        wrap_call("Log", trig_pow("Cos", 1)),
        &coeff,
      ));
      for j in 1..=k {
        let binom = crate::functions::binomial_coeff(k as i128, j as i128);
        let sign = if j % 2 == 1 { 1 } else { -1 };
        let g = gcd_i128(binom, 2 * j as i128).max(1);
        terms.push(coeff_term(
          sign * binom / g,
          2 * j as i128 / g,
          trig_pow("Cos", 2 * j),
          &coeff,
        ));
      }
    }
    return Some(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    });
  }

  if sin_pow == -1 && cos_pow >= 2 {
    let p = cos_pow;
    let mut terms: Vec<Expr> = Vec::new();
    if p == 2 {
      // Cos[u]/a + Log[Tan[u/2]]/a
      terms.push(coeff_term(1, 1, trig_pow("Cos", 1), &coeff));
      let half_arg = simplify(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(arg.clone()),
        right: Box::new(Expr::Integer(2)),
      });
      let tan_half = Expr::FunctionCall {
        name: "Tan".to_string(),
        args: vec![half_arg].into(),
      };
      terms.push(coeff_term(1, 1, wrap_call("Log", tan_half), &coeff));
    } else if p % 2 == 1 {
      // Log[Sin[u]]/a + Σ (-1)^j C(k,j) Sin[u]^(2j)/(2j a)
      let k = (p - 1) / 2;
      terms.push(coeff_term(
        1,
        1,
        wrap_call("Log", trig_pow("Sin", 1)),
        &coeff,
      ));
      for j in 1..=k {
        let binom = crate::functions::binomial_coeff(k as i128, j as i128);
        let sign = if j % 2 == 1 { -1 } else { 1 };
        let g = gcd_i128(binom, 2 * j as i128).max(1);
        terms.push(coeff_term(
          sign * binom / g,
          2 * j as i128 / g,
          trig_pow("Sin", 2 * j),
          &coeff,
        ));
      }
    } else {
      // Even p ≥ 4 needs wolframscript's multiple-angle presentation.
      return None;
    }
    return Some(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    });
  }

  None
}

/// Split an expression into multiplicative numerator/denominator factor
/// references, flattening nested Times and Divide.
fn collect_times_factor_refs<'a>(
  expr: &'a Expr,
  num: &mut Vec<&'a Expr>,
  den: &mut Vec<&'a Expr>,
) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      collect_times_factor_refs(left, num, den);
      collect_times_factor_refs(right, num, den);
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      collect_times_factor_refs(left, num, den);
      // Denominator factors flip roles below it: a/(b/c) = a*c/b
      collect_times_factor_refs(right, den, num);
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      for a in args {
        collect_times_factor_refs(a, num, den);
      }
    }
    _ => num.push(expr),
  }
}

/// Build the antiderivative of Exp[-a*x^2] (`erf_name` = "Erf") or of
/// Exp[a*x^2] (`erf_name` = "Erfi", for the positive-coefficient branch):
///   Sqrt[Pi/a]/2 * Erf[Sqrt[a]*x]  (general a)
///   (Sqrt[Pi]*Erf[x])/2            (when a=1)
/// The Erfi form is identical with Erf replaced by Erfi.
fn make_gaussian_antiderivative(
  var: &str,
  coeff: &Expr,
  erf_name: &str,
) -> Expr {
  let var_expr = Expr::Identifier(var.to_string());
  let (erf_arg, prefix) = match coeff {
    Expr::Integer(1) => {
      // a=1: Erf[x], prefix = Sqrt[Pi]
      (var_expr, make_sqrt(Expr::Constant("Pi".to_string())))
    }
    Expr::Integer(n) if *n != 1 => {
      // concrete integer a: (Sqrt[Pi/a]*Erf[Sqrt[a]*x])/2 — matches Wolfram output
      let sqrt_a = make_sqrt(coeff.clone());
      let erf_arg = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(sqrt_a),
        right: Box::new(var_expr),
      };
      let prefix = make_sqrt(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(coeff.clone()),
      });
      (erf_arg, prefix)
    }
    _ => {
      // symbolic a: (Sqrt[Pi]*Erf[Sqrt[a]*x])/(2*Sqrt[a]) — matches Wolfram output
      let sqrt_a = make_sqrt(coeff.clone());
      let erf_arg = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(sqrt_a.clone()),
        right: Box::new(var_expr),
      };
      let prefix = make_sqrt(Expr::Constant("Pi".to_string()));
      let erf_expr = Expr::FunctionCall {
        name: erf_name.to_string(),
        args: vec![erf_arg].into(),
      };
      // (Sqrt[Pi] * Erf[Sqrt[a]*x]) / (2 * Sqrt[a])
      return Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(prefix),
          right: Box::new(erf_expr),
        }),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(2)),
          right: Box::new(sqrt_a),
        }),
      };
    }
  };
  let erf_expr = Expr::FunctionCall {
    name: erf_name.to_string(),
    args: vec![erf_arg].into(),
  };
  // a=1 case: (Sqrt[Pi] * Erf[x]) / 2
  Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(prefix),
      right: Box::new(erf_expr),
    }),
    right: Box::new(Expr::Integer(2)),
  }
}

/// Match `E^(a*x^2)` with a positive (or symbolic) `a`, returning `a`. Explicit
/// negative-coefficient forms are left to `match_neg_a_x_squared` (the Erf
/// branch), so this recognizes the Erfi branch `Exp[a*x^2]` with a > 0.
fn match_pos_a_x_squared(expr: &Expr, var: &str) -> Option<Expr> {
  if is_var_squared(expr, var) {
    return Some(Expr::Integer(1));
  }
  let coeff = match_a_x_squared(expr, var)?;
  if matches!(&coeff, Expr::Integer(n) if *n < 0) {
    return None;
  }
  Some(coeff)
}

/// Build the Fresnel antiderivative of Sin[a*x^2] (`fresnel_name` = "FresnelS")
/// or Cos[a*x^2] (`fresnel_name` = "FresnelC"):
///   ∫ Sin[a x^2] dx = Sqrt[Pi/2]/Sqrt[a] * FresnelS[Sqrt[a]*Sqrt[2/Pi]*x].
/// The factored radicals are left for the evaluator to canonicalize, matching
/// wolframscript's printed form.
fn make_fresnel_antiderivative(
  var: &str,
  coeff: &Expr,
  fresnel_name: &str,
) -> Expr {
  let x = Expr::Identifier(var.to_string());
  let pi = Expr::Constant("Pi".to_string());
  let sqrt_pi_2 = make_sqrt(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(pi.clone()),
    right: Box::new(Expr::Integer(2)),
  });
  let sqrt_2_pi = make_sqrt(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::Integer(2)),
    right: Box::new(pi),
  });
  let sqrt_a = make_sqrt(coeff.clone());
  let simplify = |e: Expr| -> Expr {
    let cloned = e.clone();
    crate::evaluator::evaluate_function_call_ast("Simplify", &[e])
      .unwrap_or_else(|_| {
        crate::evaluator::evaluate_expr_to_expr(&cloned).unwrap_or(cloned)
      })
  };
  // Simplify the Fresnel argument first — Sqrt[a] Sqrt[2/Pi] x collapses to x
  // for a = Pi/2 and to (2 x)/Sqrt[Pi] for a = 2, while a symbolic coefficient
  // stays split, matching wolframscript.
  let fresnel_arg = simplify(Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![sqrt_a.clone(), sqrt_2_pi, x].into(),
  });
  let fresnel = Expr::FunctionCall {
    name: fresnel_name.to_string(),
    args: vec![fresnel_arg].into(),
  };
  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      sqrt_pi_2,
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(sqrt_a),
        right: Box::new(Expr::Integer(-1)),
      },
      fresnel,
    ]
    .into(),
  };
  // Simplify the prefix Sqrt[Pi/2]/Sqrt[a] (e.g. to Sqrt[Pi/6] or Sqrt[Pi]/2).
  simplify(result)
}

/// Try to match an expression as `a*var` where `a` is constant w.r.t. `var`,
/// or just `var` (returning `Integer(1)`).
/// Returns Some(a) if it matches, None otherwise.
fn try_match_linear_arg(expr: &Expr, var: &str) -> Option<Expr> {
  match expr {
    Expr::Identifier(name) if name == var => Some(Expr::Integer(1)),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if is_constant_wrt(left, var)
        && matches!(right.as_ref(), Expr::Identifier(name) if name == var)
      {
        Some(*left.clone())
      } else if is_constant_wrt(right, var)
        && matches!(left.as_ref(), Expr::Identifier(name) if name == var)
      {
        Some(*right.clone())
      } else {
        None
      }
    }
    // x/c form: var/const → coefficient is 1/const (i.e., Rational[1,c] for integer c)
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Identifier(name) if name == var)
        && is_constant_wrt(right, var)
      {
        // coefficient = 1/right — use division_ast for proper Rational creation
        if let Ok(result) = crate::functions::math_ast::divide_ast(&[
          Expr::Integer(1),
          *right.clone(),
        ]) {
          Some(result)
        } else {
          Some(Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expr::Integer(1)),
            right: right.clone(),
          })
        }
      } else if is_constant_wrt(right, var) {
        // (expr)/const where expr might be a*x → coefficient is a/const
        if let Some(inner_coeff) = try_match_linear_arg(left, var) {
          if let Ok(result) = crate::functions::math_ast::divide_ast(&[
            inner_coeff.clone(),
            *right.clone(),
          ]) {
            Some(result)
          } else {
            Some(Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(inner_coeff),
              right: right.clone(),
            })
          }
        } else {
          None
        }
      } else {
        None
      }
    }
    // FunctionCall("Times", [coeff, var]) form
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Find the variable factor and collect the rest as coefficient
      let mut var_idx = None;
      for (i, arg) in args.iter().enumerate() {
        if matches!(arg, Expr::Identifier(n) if n == var) {
          var_idx = Some(i);
          break;
        }
      }
      if let Some(vi) = var_idx {
        let rest: Vec<Expr> = args
          .iter()
          .enumerate()
          .filter(|(i, _)| *i != vi)
          .map(|(_, e)| e.clone())
          .collect();
        if rest.iter().all(|a| is_constant_wrt(a, var)) {
          if rest.len() == 1 {
            Some(rest[0].clone())
          } else {
            Some(Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest.into(),
            })
          }
        } else {
          None
        }
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Decompose a rational-valued constant expression into `(p, q)` with `q > 0`
/// in lowest terms. Returns `None` for anything that isn't `Integer(n)` or
/// `Rational[p, q]` (already reduced by construction).
fn decompose_rational_coeff(coeff: &Expr) -> Option<(i128, i128)> {
  match coeff {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&args[0], &args[1])
        && *q > 0
      {
        return Some((*p, *q));
      }
      None
    }
    _ => None,
  }
}

/// Build the antiderivative of `ArcSin[a*var]` (or `ArcCos[a*var]`) where
/// `a = p/q` is rational, expressed in the form used by Mathematica's
/// `Integrate`:
///
/// ```text
/// ∫ ArcSin[a x] dx  =  x*ArcSin[a x] + Sqrt[q^2 - p^2 x^2] / p
/// ∫ ArcCos[a x] dx  =  x*ArcCos[a x] - Sqrt[q^2 - p^2 x^2] / p
/// ```
///
/// Requires `p != 0`. For `a = 1/q`, this collapses nicely to
/// `Sqrt[q^2 - x^2]`. For `a = p` (integer), it yields
/// `Sqrt[1 - p^2 x^2] / p`.
fn arcsin_arccos_linear_antideriv(
  fname: &str,
  args: &[Expr],
  var: &str,
  p: i128,
  q: i128,
) -> Option<Expr> {
  if p == 0 {
    return None;
  }
  let x = Expr::Identifier(var.to_string());
  let x_sq = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(x.clone()),
    right: Box::new(Expr::Integer(2)),
  };
  // Sqrt argument: q^2 - p^2 * x^2 (with p^2 elided when p^2 == 1).
  let p_sq = p.checked_mul(p)?;
  let q_sq = q.checked_mul(q)?;
  let scaled_x_sq = if p_sq == 1 {
    x_sq
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(p_sq)),
      right: Box::new(x_sq),
    }
  };
  let sqrt_arg = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(Expr::Integer(q_sq)),
    right: Box::new(scaled_x_sq),
  };
  let sqrt_expr = crate::functions::math_ast::make_sqrt(sqrt_arg);
  let sqrt_over_p = if p == 1 {
    sqrt_expr
  } else if p == -1 {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(sqrt_expr),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(sqrt_expr),
      right: Box::new(Expr::Integer(p)),
    }
  };
  let inverse_trig = unevaluated(fname, args);
  let x_times_atrig = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(Expr::Identifier(var.to_string())),
    right: Box::new(inverse_trig),
  };
  let join_op = if fname == "ArcCos" {
    BinaryOperator::Minus
  } else {
    BinaryOperator::Plus
  };
  Some(Expr::BinaryOp {
    op: join_op,
    left: Box::new(x_times_atrig),
    right: Box::new(sqrt_over_p),
  })
}

/// Extract the coefficient of `var` from a linear expression `a*var + b`.
/// Returns `Some(a)` if the expression is linear in `var`, `None` otherwise.
/// Works by differentiating the expression: if d/dx(expr) is constant w.r.t. var,
/// then expr is linear and the derivative is the coefficient.
fn extract_linear_coefficient(expr: &Expr, var: &str) -> Option<Expr> {
  // Check that expr actually depends on var
  if is_constant_wrt(expr, var) {
    return None;
  }
  // Differentiate: if expr = a*x + b, then d/dx(expr) = a (constant)
  let deriv = differentiate(expr, var).ok()?;
  let deriv = simplify(deriv);
  if is_constant_wrt(&deriv, var) {
    Some(deriv)
  } else {
    None
  }
}

/// Try to integrate Sin[f(x)]^2 or Cos[f(x)]^2 using power-reduction identities.
/// sin²(a*x) = x/2 - sin(2*a*x)/(4*a)
/// cos²(a*x) = x/2 + sin(2*a*x)/(4*a)
/// ∫ of a two-factor product that is exactly the derivative of an elementary
/// reciprocal-trig / hyperbolic function:
///   Sec[u] Tan[u]   ->  Sec[u]
///   Csc[u] Cot[u]   -> -Csc[u]
///   Sech[u] Tanh[u] -> -Sech[u]
///   Csch[u] Coth[u] -> -Csch[u]
/// where u = a*var is linear in var; the result carries the 1/a factor.
fn try_integrate_derivative_product(
  factors: &[&Expr],
  var: &str,
) -> Option<Expr> {
  if factors.len() != 2 {
    return None;
  }
  let head_arg = |e: &Expr| -> Option<(String, Expr)> {
    if let Expr::FunctionCall { name, args } = e
      && args.len() == 1
    {
      Some((name.clone(), args[0].clone()))
    } else {
      None
    }
  };
  let (n0, a0) = head_arg(factors[0])?;
  let (n1, a1) = head_arg(factors[1])?;
  // Both factors must share the same (linear) argument.
  if crate::syntax::expr_to_string(&a0) != crate::syntax::expr_to_string(&a1) {
    return None;
  }
  let coeff = try_match_linear_arg(&a0, var)?;
  let mut names = [n0.as_str(), n1.as_str()];
  names.sort();
  let (head, negate) = match (names[0], names[1]) {
    ("Sec", "Tan") => ("Sec", false),
    ("Cot", "Csc") => ("Csc", true),
    ("Sech", "Tanh") => ("Sech", true),
    ("Coth", "Csch") => ("Csch", true),
    _ => return None,
  };
  let f = Expr::FunctionCall {
    name: head.to_string(),
    args: vec![a0].into(),
  };
  Some(if negate {
    make_neg_divided(f, coeff)
  } else {
    make_divided(f, coeff)
  })
}

fn try_integrate_trig_squared(base: &Expr, var: &str) -> Option<Expr> {
  if let Expr::FunctionCall { name, args } = base
    && args.len() == 1
  {
    let is_sin = name == "Sin";
    let is_cos = name == "Cos";
    let is_sec = name == "Sec";
    let is_csc = name == "Csc";
    // ∫ Sec[a*x]^2 dx = Tan[a*x]/a
    if is_sec {
      let coeff = try_match_linear_arg(&args[0], var)?;
      let tan_expr = Expr::FunctionCall {
        name: "Tan".to_string(),
        args: args.clone(),
      };
      return Some(make_divided(tan_expr, coeff));
    }
    // ∫ Csc[a*x]^2 dx = -Cot[a*x]/a
    if is_csc {
      let coeff = try_match_linear_arg(&args[0], var)?;
      let cot_expr = Expr::FunctionCall {
        name: "Cot".to_string(),
        args: args.clone(),
      };
      return Some(make_neg_divided(cot_expr, coeff));
    }
    // ∫ Sech[a*x]^2 dx = Tanh[a*x]/a
    if name == "Sech" {
      let coeff = try_match_linear_arg(&args[0], var)?;
      let tanh_expr = Expr::FunctionCall {
        name: "Tanh".to_string(),
        args: args.clone(),
      };
      return Some(make_divided(tanh_expr, coeff));
    }
    // ∫ Csch[a*x]^2 dx = -Coth[a*x]/a
    if name == "Csch" {
      let coeff = try_match_linear_arg(&args[0], var)?;
      let coth_expr = Expr::FunctionCall {
        name: "Coth".to_string(),
        args: args.clone(),
      };
      return Some(make_neg_divided(coth_expr, coeff));
    }
    // ∫ Tan[a*x]^2 dx  = -ArcTan[Tan[a*x]]/a  + Tan[a*x]/a
    // ∫ Cot[a*x]^2 dx  = -ArcTan[Tan[a*x]]/a  - Cot[a*x]/a
    // ∫ Tanh[a*x]^2 dx =  ArcTanh[Tanh[a*x]]/a - Tanh[a*x]/a
    // ∫ Coth[a*x]^2 dx =  ArcTanh[Tanh[a*x]]/a - Coth[a*x]/a
    // wolframscript keeps the linear ("x") term as ArcTan[Tan[u]] /
    // ArcTanh[Tanh[u]] for these (it does not simplify them back to `x`).
    if matches!(name.as_str(), "Tan" | "Cot" | "Tanh" | "Coth") {
      let coeff = try_match_linear_arg(&args[0], var)?;
      let u = args[0].clone();
      let call = |f: &str, a: Expr| Expr::FunctionCall {
        name: f.to_string(),
        args: vec![a].into(),
      };
      let hyperbolic = matches!(name.as_str(), "Tanh" | "Coth");
      let (inv, inner) = if hyperbolic {
        ("ArcTanh", "Tanh")
      } else {
        ("ArcTan", "Tan")
      };
      let x_inner = call(inv, call(inner, u.clone()));
      // x-term: hyperbolic +, circular -.
      let x_term = if hyperbolic {
        make_divided(x_inner, coeff.clone())
      } else {
        make_neg_divided(x_inner, coeff.clone())
      };
      let direct = call(name.as_str(), u);
      // direct term: Tan +, Cot/Tanh/Coth -.
      let direct_term = if name == "Tan" {
        make_divided(direct, coeff)
      } else {
        make_neg_divided(direct, coeff)
      };
      return Some(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(x_term),
        right: Box::new(direct_term),
      });
    }

    // ∫ Sinh[a*x]^2 dx = Sinh[2*a*x]/(4*a) - x/2
    // ∫ Cosh[a*x]^2 dx = Sinh[2*a*x]/(4*a) + x/2
    if name == "Sinh" || name == "Cosh" {
      let coeff = try_match_linear_arg(&args[0], var)?;
      let double_arg = simplify(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(args[0].clone()),
      });
      let sinh_double = Expr::FunctionCall {
        name: "Sinh".to_string(),
        args: vec![double_arg].into(),
      };
      let four_a = simplify(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(4)),
        right: Box::new(coeff),
      });
      let sinh_term = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(sinh_double),
        right: Box::new(four_a),
      };
      let x_half = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Identifier(var.to_string())),
        right: Box::new(Expr::Integer(2)),
      };
      return Some(Expr::BinaryOp {
        op: if name == "Cosh" {
          BinaryOperator::Plus
        } else {
          BinaryOperator::Minus
        },
        left: Box::new(sinh_term),
        right: Box::new(x_half),
      });
    }

    if !is_sin && !is_cos {
      return None;
    }
    let coeff = try_match_linear_arg(&args[0], var)?;
    // Build: 2*a*x
    let double_arg = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(args[0].clone()),
    });
    // sin(2*a*x)
    let sin_double = Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![double_arg].into(),
    };
    // 4*a
    let four_a = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(4)),
      right: Box::new(coeff),
    });
    // x/2
    let x_half = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Identifier(var.to_string())),
      right: Box::new(Expr::Integer(2)),
    };
    // sin(2*a*x)/(4*a)
    let sin_term = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(sin_double),
      right: Box::new(four_a),
    };
    if is_sin {
      // x/2 - sin(2*a*x)/(4*a)
      Some(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(x_half),
        right: Box::new(sin_term),
      })
    } else {
      // x/2 + sin(2*a*x)/(4*a)
      Some(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(x_half),
        right: Box::new(sin_term),
      })
    }
  } else {
    None
  }
}

/// Try to integrate Sin[a*x]^n or Cos[a*x]^n for positive integer n ≥ 3
/// using Chebyshev expansion (multiple angle formula).
///
/// For odd n = 2m+1, m = (n-1)/2:
///   sin^n(x) = (1/2^n) * 2 * Sum_{k=0}^{m} (-1)^(m-k) * C(n,k) * sin((n-2k)*x)
///   cos^n(x) = (1/2^n) * 2 * Sum_{k=0}^{m} C(n,k) * cos((n-2k)*x)
/// For even n = 2m, m = n/2:
///   sin^n(x) = (1/2^n) * [C(n,m) + 2 * Sum_{k=0}^{m-1} (-1)^(m-k) * C(n,k) * cos((n-2k)*x)]
///   cos^n(x) = (1/2^n) * [C(n,m) + 2 * Sum_{k=0}^{m-1} C(n,k) * cos((n-2k)*x)]
fn try_integrate_trig_power(base: &Expr, n: i128, var: &str) -> Option<Expr> {
  if n < 3 {
    return None;
  }
  // Bail out for huge powers (e.g. Sin[x]^1000): the Chebyshev expansion
  // would generate `n/2` binomial-coefficient products that overflow
  // i128 long before we can build them. Without this guard `binomial`
  // panics and any caller wrapped in TimeConstrained tears the kernel
  // down. ~85 is the largest n where C(n, n/2) still fits in i128.
  if n > 85 {
    return None;
  }
  let (name, arg) = match base {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      let is_sin = name == "Sin";
      let is_cos = name == "Cos";
      if !is_sin && !is_cos {
        return None;
      }
      (name.as_str(), &args[0])
    }
    _ => return None,
  };

  // Match linear argument a*x
  let coeff = try_match_linear_arg(arg, var)?;
  let is_sin = name == "Sin";

  // Build the Chebyshev expansion terms and integrate each
  let mut terms: Vec<Expr> = Vec::new();
  let m = n / 2;
  let is_odd = n % 2 != 0;

  if !is_odd {
    // Constant term: C(n,m) / 2^n * x
    let binom_mid = crate::functions::binomial_coeff(n, m);
    let denom = 1i128 << n; // 2^n
    let g = gcd_i128(binom_mid, denom);
    let const_num = binom_mid / g;
    let const_den = denom / g;
    let numer_term = if const_num == 1 {
      Expr::Identifier(var.to_string())
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(const_num)),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    };
    let const_term = if const_den == 1 {
      numer_term
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(numer_term),
        right: Box::new(Expr::Integer(const_den)),
      }
    };
    terms.push(const_term);
  }

  // Odd power: n = 2m+1
  // For sin: integral of (-1)^(m-k)*C(n,k)*sin((n-2k)*x) → absorb -1/k into coefficient
  //   coeff = (-1)^(m-k+1) * C(n,k), trig = Cos[(n-2k)*x]
  // For cos: integral of C(n,k)*cos((n-2k)*x) → coeff = C(n,k)/k, trig = Sin[(n-2k)*x]
  for k in 0..n - m {
    // n-m=m+1 for n odd, m for n even
    let freq = n - 2 * k; // always positive since k ≤ m and n=2m+1
    let binom = crate::functions::binomial_coeff(n, k);

    // Build the trig argument: freq * a * x
    let freq_arg = if matches!(&coeff, Expr::Integer(1)) {
      if freq == 1 {
        Expr::Identifier(var.to_string())
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(freq)),
          right: Box::new(Expr::Identifier(var.to_string())),
        }
      }
    } else {
      simplify(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(freq)),
        right: Box::new(arg.clone()),
      })
    };

    // For sin^n: integral coefficient includes the -1 from integrating sin
    // sign = (-1)^(m-k) for the Chebyshev expansion, then *(-1) for integral of sin
    // = (-1)^(m-k+1)
    // For cos^n: sign = +1 (Chebyshev) and integral of cos gives sin (no sign change)
    let coeff_num = if is_sin && (m - k + is_odd as i128) % 2 != 0 {
      -2 * binom
    } else {
      2 * binom
    };

    // Integrated: sin(freq*x)/(freq) for both sin^n and cos^n even powers
    // cos(freq*x)/(freq) for sin^n odd powers.
    let integrated_trig = Expr::FunctionCall {
      name: if is_sin && is_odd { "Cos" } else { "Sin" }.to_string(),
      args: vec![freq_arg].into(),
    };

    // Total coefficient: coeff_num / (freq * 4^m)
    let power_2n = 1i128 << n; // 2^n
    let denom = freq * power_2n; // freq * 2^n
    let g = gcd_i128(coeff_num, denom);
    let num = coeff_num / g;
    let den = denom / g;

    let term = if matches!(&coeff, Expr::Integer(1)) {
      make_fraction_term(num, den, integrated_trig)
    } else {
      let den_expr = simplify(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(den)),
        right: Box::new(coeff.clone()),
      });
      make_fraction_term_expr(num, den_expr, integrated_trig)
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return None;
  }

  // Combine terms using plus_ast for canonical ordering
  let result = crate::functions::math_ast::plus_ast(&terms).ok()?;
  Some(result)
}

/// Build a term: (num/den) * expr, simplified
fn make_fraction_term(num: i128, den: i128, expr: Expr) -> Expr {
  if den == 1 {
    if num == 1 {
      expr
    } else if num == -1 {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(expr),
      }
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(num)),
        right: Box::new(expr),
      }
    }
  } else if num == 1 {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(expr),
      right: Box::new(Expr::Integer(den)),
    }
  } else if num == -1 {
    // -(expr/den) so plus_ast displays as "- expr/den"
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(expr),
        right: Box::new(Expr::Integer(den)),
      }),
    }
  } else if num < 0 {
    // -(|num|*expr/den) so plus_ast displays as "- num*expr/den"
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-num)),
          right: Box::new(expr),
        }),
        right: Box::new(Expr::Integer(den)),
      }),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(num)),
        right: Box::new(expr),
      }),
      right: Box::new(Expr::Integer(den)),
    }
  }
}

/// Build a term: (num/den_expr) * expr, simplified
fn make_fraction_term_expr(num: i128, den_expr: Expr, expr: Expr) -> Expr {
  let num_expr = if num == 1 {
    expr
  } else if num == -1 {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(expr),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(num)),
      right: Box::new(expr),
    }
  };
  Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(num_expr),
    right: Box::new(den_expr),
  }
}

/// Try to match ∫ E^(a*x) / (c*x) dx = ExpIntegralEi[a*x] / c
/// Handles: E^(a*x) / x, E^(a*x) / (c*x), E^x / x, E^x / (c*x)
/// Also handles Exp[a*x] function form.
fn try_match_exp_over_linear(
  numerator: &Expr,
  denominator: &Expr,
  var: &str,
) -> Option<Expr> {
  // Check if numerator is E^(a*x) (Power form or Exp function form)
  let exp_linear_arg = match numerator {
    // E^(a*x) as BinaryOp::Power
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } if matches!(base.as_ref(), Expr::Constant(c) if c == "E") => {
      try_match_linear_arg(exp, var)
    }
    // Exp[a*x] as FunctionCall
    Expr::FunctionCall { name, args } if name == "Exp" && args.len() == 1 => {
      try_match_linear_arg(&args[0], var)
    }
    _ => None,
  };

  let linear_coeff = exp_linear_arg?; // a in E^(a*x)

  // Check if denominator is c*x or just x
  let denom_const = match denominator {
    // Just x
    Expr::Identifier(name) if name == var => Some(Expr::Integer(1)),
    // c*x (BinaryOp form)
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if is_constant_wrt(left, var)
        && matches!(right.as_ref(), Expr::Identifier(name) if name == var)
      {
        Some(*left.clone())
      } else if is_constant_wrt(right, var)
        && matches!(left.as_ref(), Expr::Identifier(name) if name == var)
      {
        Some(*right.clone())
      } else {
        None
      }
    }
    // Times[c, x] (FunctionCall form)
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if is_constant_wrt(&args[0], var)
        && matches!(&args[1], Expr::Identifier(name) if name == var)
      {
        Some(args[0].clone())
      } else if is_constant_wrt(&args[1], var)
        && matches!(&args[0], Expr::Identifier(name) if name == var)
      {
        Some(args[1].clone())
      } else {
        None
      }
    }
    _ => None,
  };

  let denom_const = denom_const?; // c in c*x

  // Build ExpIntegralEi[a*x]
  let ei_arg = if matches!(&linear_coeff, Expr::Integer(1)) {
    Expr::Identifier(var.to_string())
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(linear_coeff),
      right: Box::new(Expr::Identifier(var.to_string())),
    }
  };
  let ei_expr = Expr::FunctionCall {
    name: "ExpIntegralEi".to_string(),
    args: vec![ei_arg].into(),
  };

  // Return ExpIntegralEi[a*x] / c
  if matches!(&denom_const, Expr::Integer(1)) {
    Some(ei_expr)
  } else {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(ei_expr),
      right: Box::new(denom_const),
    })
  }
}

/// Try to match ∫ f(a*x) / (c*x) dx for a trigonometric or hyperbolic `f`,
/// giving the corresponding exponential-integral special function:
///   Sin  → SinIntegral,   Cos  → CosIntegral,
///   Sinh → SinhIntegral,  Cosh → CoshIntegral.
/// e.g. ∫ Sin[x]/x dx = SinIntegral[x] and ∫ Cos[a x]/x dx = CosIntegral[a x].
fn try_match_si_ci_over_linear(
  numerator: &Expr,
  denominator: &Expr,
  var: &str,
) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = numerator else {
    return None;
  };
  if args.len() != 1 {
    return None;
  }
  let integral_name = match name.as_str() {
    "Sin" => "SinIntegral",
    "Cos" => "CosIntegral",
    "Sinh" => "SinhIntegral",
    "Cosh" => "CoshIntegral",
    _ => return None,
  };
  let linear_coeff = try_match_linear_arg(&args[0], var)?; // a in f(a*x)
  let denom_const = try_match_linear_arg(denominator, var)?; // c in c*x

  let si_arg = if matches!(&linear_coeff, Expr::Integer(1)) {
    Expr::Identifier(var.to_string())
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(linear_coeff),
      right: Box::new(Expr::Identifier(var.to_string())),
    }
  };
  let si_expr = Expr::FunctionCall {
    name: integral_name.to_string(),
    args: vec![si_arg].into(),
  };

  if matches!(&denom_const, Expr::Integer(1)) {
    Some(si_expr)
  } else {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(si_expr),
      right: Box::new(denom_const),
    })
  }
}

/// Check if an expression is Rational[-1, 2] (i.e., exponent -1/2)
fn is_rational_neg_half(expr: &Expr) -> bool {
  matches!(
    expr,
    Expr::FunctionCall { name, args }
      if name == "Rational"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(-1))
        && matches!(&args[1], Expr::Integer(2))
  )
}

/// Try to integrate (base)^(-1/2) for special forms:
/// ∫ (1 - x^2)^(-1/2) dx = ArcSin[x]
/// ∫ (1 + x^2)^(-1/2) dx = ArcSinh[x]
fn try_integrate_inverse_sqrt(base: &Expr, var: &str) -> Option<Expr> {
  // Try to match the base as a + b*x^2 by extracting polynomial coefficients
  let base_eval =
    crate::evaluator::evaluate_expr_to_expr(base).unwrap_or(base.clone());
  let var_expr = Expr::Identifier(var.to_string());

  // Use CoefficientList to extract coefficients
  let coeff_result = crate::functions::polynomial_ast::coefficient_list_ast(&[
    base_eval, var_expr,
  ])
  .ok()?;
  let coeffs = match &coeff_result {
    Expr::List(items) => items,
    _ => return None,
  };

  // We need exactly a polynomial of degree 2 with no linear term: a + 0*x + b*x^2
  if coeffs.len() != 3 {
    return None;
  }
  // Linear coefficient must be zero
  let c1_val = crate::functions::math_ast::try_eval_to_f64(&coeffs[1])?;
  if c1_val.abs() > 1e-15 {
    return None;
  }

  let a = &coeffs[0]; // constant term
  let b = &coeffs[2]; // x^2 coefficient

  let a_val = crate::functions::math_ast::try_eval_to_f64(a)?;
  let b_val = crate::functions::math_ast::try_eval_to_f64(b)?;

  if a_val <= 0.0 {
    return None;
  }

  if b_val < 0.0 {
    // ∫ (a - |b|*x^2)^(-1/2) dx = (1/sqrt(|b|)) * ArcSin[x * sqrt(|b|/a)]
    let abs_b = simplify(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(b.clone()),
    });
    let ratio = simplify(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(abs_b.clone()),
      right: Box::new(a.clone()),
    });
    let sqrt_ratio = simplify(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![ratio].into(),
    });
    let sqrt_abs_b = simplify(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![abs_b].into(),
    });
    let arg = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Identifier(var.to_string())),
      right: Box::new(sqrt_ratio),
    });
    let arcsin = Expr::FunctionCall {
      name: "ArcSin".to_string(),
      args: vec![arg].into(),
    };
    Some(make_divided(arcsin, sqrt_abs_b))
  } else {
    // ∫ (a + b*x^2)^(-1/2) dx = (1/sqrt(b)) * ArcSinh[x * sqrt(b/a)]
    let ratio = simplify(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(b.clone()),
      right: Box::new(a.clone()),
    });
    let sqrt_ratio = simplify(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![ratio].into(),
    });
    let sqrt_b = simplify(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![b.clone()].into(),
    });
    let arg = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Identifier(var.to_string())),
      right: Box::new(sqrt_ratio),
    });
    let arcsinh = Expr::FunctionCall {
      name: "ArcSinh".to_string(),
      args: vec![arg].into(),
    };
    Some(make_divided(arcsinh, sqrt_b))
  }
}

/// Build a *continuous* antiderivative of `Sqrt[a + b*x^2]` for a positive
/// numeric constant term `a` and any non-zero numeric quadratic coefficient
/// `b` (with no linear term):
///   b < 0:  (x*Sqrt[a + b*x^2])/2 + (a*ArcSin[x*Sqrt[-b/a]])/(2*Sqrt[-b])
///   b > 0:  (x*Sqrt[a + b*x^2])/2 + (a*ArcSinh[x*Sqrt[b/a]])/(2*Sqrt[b])
/// Unlike wolframscript's displayed indefinite forms (which use ArcTan/ArcTanh
/// and have a removable singularity at the radical's zeros), the ArcSin/ArcSinh
/// forms are continuous, so substituting the integration bounds yields exact
/// closed forms. Returns the antiderivative expression `F(var)`.
fn sqrt_quadratic_antiderivative(base: &Expr, var: &str) -> Option<Expr> {
  use BinaryOperator::*;
  let base_eval = crate::evaluator::evaluate_expr_to_expr(base)
    .unwrap_or_else(|_| base.clone());
  let var_expr = Expr::Identifier(var.to_string());
  let coeff_result = crate::functions::polynomial_ast::coefficient_list_ast(&[
    base_eval, var_expr,
  ])
  .ok()?;
  let coeffs = match &coeff_result {
    Expr::List(items) => items,
    _ => return None,
  };
  // Need exactly a degree-2 polynomial: a + (linear)*x + b*x^2.
  if coeffs.len() != 3 {
    return None;
  }
  // Linear coefficient must be zero.
  let c1_val = crate::functions::math_ast::try_eval_to_f64(&coeffs[1])?;
  if c1_val.abs() > 1e-15 {
    return None;
  }
  // The x^2 coefficient `b` must be a non-zero numeric constant.
  let b = &coeffs[2];
  let b_val = crate::functions::math_ast::try_eval_to_f64(b)?;
  if b_val.abs() < 1e-15 {
    return None;
  }
  let b_neg = b_val < 0.0;
  let a = &coeffs[0]; // constant term
  // The constant term must be a positive numeric constant so the ArcSin/ArcSinh
  // form is real-valued and the square roots are well defined.
  let a_val = crate::functions::math_ast::try_eval_to_f64(a)?;
  if a_val <= 0.0 {
    return None;
  }

  let sqrt_base = make_sqrt(base.clone());
  let x = Expr::Identifier(var.to_string());
  // First term: (x*Sqrt[base])/2.
  let first = Expr::BinaryOp {
    op: Divide,
    left: Box::new(Expr::BinaryOp {
      op: Times,
      left: Box::new(x.clone()),
      right: Box::new(sqrt_base),
    }),
    right: Box::new(Expr::Integer(2)),
  };
  // |b| for the b < 0 branch (ArcSin needs the positive coefficient).
  let abs_b = if b_neg {
    simplify(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(b.clone()),
    })
  } else {
    b.clone()
  };
  // arc_arg = x * Sqrt[|b|/a]
  let ratio = simplify(Expr::BinaryOp {
    op: Divide,
    left: Box::new(abs_b.clone()),
    right: Box::new(a.clone()),
  });
  let arc_arg = simplify(Expr::BinaryOp {
    op: Times,
    left: Box::new(x),
    right: Box::new(make_sqrt(ratio)),
  });
  let arc = Expr::FunctionCall {
    name: (if b_neg { "ArcSin" } else { "ArcSinh" }).to_string(),
    args: vec![arc_arg].into(),
  };
  // Second term: (a*arc) / (2*Sqrt[|b|]).
  let denom = simplify(Expr::BinaryOp {
    op: Times,
    left: Box::new(Expr::Integer(2)),
    right: Box::new(make_sqrt(abs_b)),
  });
  let second = make_divided(
    Expr::BinaryOp {
      op: Times,
      left: Box::new(a.clone()),
      right: Box::new(arc),
    },
    denom,
  );
  Some(Expr::BinaryOp {
    op: Plus,
    left: Box::new(first),
    right: Box::new(second),
  })
}

/// Definite integral of `Sqrt[a + b*x^2]` over `[lo, hi]` for a monic quadratic
/// radicand with positive numeric constant `a`. Uses the continuous ArcSin /
/// ArcSinh antiderivative so the result is an exact closed form (e.g.
/// `Integrate[Sqrt[1 - x^2], {x, -1, 1}]` → `Pi/2`,
/// `Integrate[Sqrt[4 - x^2], {x, -2, 2}]` → `2*Pi`).
fn try_definite_sqrt_quadratic(
  integrand: &Expr,
  var: &str,
  lo: &Expr,
  hi: &Expr,
) -> Option<Expr> {
  let base = crate::functions::math_ast::is_sqrt(integrand)?.clone();
  let antideriv = sqrt_quadratic_antiderivative(&base, var)?;
  let antideriv =
    crate::evaluator::evaluate_expr_to_expr(&antideriv).unwrap_or(antideriv);
  let sub_hi = crate::syntax::substitute_variable(&antideriv, var, hi);
  let at_hi = crate::evaluator::evaluate_expr_to_expr(&sub_hi).ok()?;
  let sub_lo = crate::syntax::substitute_variable(&antideriv, var, lo);
  let at_lo = crate::evaluator::evaluate_expr_to_expr(&sub_lo).ok()?;
  if is_nonfinite_result(&at_hi) || is_nonfinite_result(&at_lo) {
    return None;
  }
  let diff = simplify(Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(at_hi),
    right: Box::new(at_lo),
  });
  let result = crate::evaluator::evaluate_expr_to_expr(&diff).unwrap_or(diff);
  // A complex result means the radicand goes negative somewhere on [lo, hi]
  // (the integrand isn't real over the whole interval). Bail so the general
  // machinery (or an unevaluated result) handles that case rather than
  // emitting an analytic-continuation form that diverges from wolframscript.
  if expr_contains_imaginary(&result) {
    return None;
  }
  Some(result)
}

/// True if `expr` structurally contains the imaginary unit `I` or a `Complex`
/// head (used to reject out-of-domain Sqrt-quadratic definite integrals).
fn expr_contains_imaginary(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(s) | Expr::Constant(s) => s == "I",
    Expr::FunctionCall { name, args } => {
      name == "Complex" || args.iter().any(expr_contains_imaginary)
    }
    Expr::BinaryOp { left, right, .. } => {
      expr_contains_imaginary(left) || expr_contains_imaginary(right)
    }
    Expr::UnaryOp { operand, .. } => expr_contains_imaginary(operand),
    Expr::List(items) => items.iter().any(expr_contains_imaginary),
    _ => false,
  }
}

/// Try to integrate a rational function (numerator/denominator where both are polynomials).
/// Uses polynomial long division + partial fraction decomposition.
fn try_integrate_rational(
  num_expr: &Expr,
  den_expr: &Expr,
  var: &str,
) -> Option<Expr> {
  use crate::functions::polynomial_ast::{
    build_sum, coeffs_to_expr, divide_by_root, evaluate_poly,
    expand_and_combine, extract_poly_coeffs, find_integer_root, gcd_i128,
    poly_long_divide,
  };

  // Step 1: Extract polynomial coefficients
  let num_expanded = expand_and_combine(num_expr);
  let den_expanded = expand_and_combine(den_expr);
  let num_coeffs = extract_poly_coeffs(&num_expanded, var)?;
  let den_coeffs = extract_poly_coeffs(&den_expanded, var)?;

  // Need at least degree 1 denominator
  if den_coeffs.len() <= 1 {
    return None;
  }

  // Step 2: Polynomial long division (if needed)
  let (quotient_integral, proper_num) = if num_coeffs.len() >= den_coeffs.len()
  {
    let (quotient, remainder) = poly_long_divide(&num_coeffs, &den_coeffs);
    // Check that division actually worked (poly_long_divide returns (vec![0], original) on failure)
    if quotient == vec![0] && remainder == num_coeffs {
      return None;
    }
    let quot_expr = coeffs_to_expr(&quotient, var);
    let quot_integral = integrate(&quot_expr, var)?;
    // If remainder is all zeros, just return the quotient integral
    if remainder.iter().all(|&c| c == 0) {
      return Some(quot_integral);
    }
    (Some(quot_integral), remainder)
  } else {
    (None, num_coeffs)
  };

  // If proper numerator is all zeros, return just quotient integral
  if proper_num.iter().all(|&c| c == 0) {
    return quotient_integral;
  }

  // Step 3: Factor denominator
  let gcd_coeff = den_coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if gcd_coeff == 0 {
    return None;
  }
  let reduced: Vec<i128> = den_coeffs.iter().map(|c| c / gcd_coeff).collect();
  let (sign, reduced) = if reduced.last().map(|&c| c < 0).unwrap_or(false) {
    (-1i128, reduced.iter().map(|c| -c).collect::<Vec<_>>())
  } else {
    (1, reduced)
  };
  let overall_factor = gcd_coeff * sign;

  // Find integer roots
  let mut remaining = reduced.clone();
  let mut roots: Vec<i128> = Vec::new();

  loop {
    if remaining.len() <= 1 {
      break;
    }
    match find_integer_root(&remaining) {
      Some(root) => {
        roots.push(root);
        remaining = divide_by_root(&remaining, root);
      }
      None => break,
    }
  }

  // Remaining factor: if degree > 2, bail out
  // Trim trailing zeros from remaining
  while remaining.len() > 1 && remaining.last() == Some(&0) {
    remaining.pop();
  }
  let remaining_deg =
    if remaining.len() <= 1 && remaining.first().copied().unwrap_or(0) != 0 {
      0 // constant
    } else if remaining.len() <= 1 {
      0
    } else {
      remaining.len() - 1
    };
  if remaining_deg > 2 {
    return None;
  }

  // Must have at least one root or a quadratic remaining to do something useful
  if roots.is_empty() && remaining_deg < 2 {
    return None;
  }

  // Sort roots for consistent output (descending, so linear factors appear in ascending order)
  roots.sort_by(|a, b| b.cmp(a));

  // Step 4: Compute residues for linear roots
  let mut log_terms: Vec<Expr> = Vec::new();

  for (i, &root) in roots.iter().enumerate() {
    let num_at_root = evaluate_poly(&proper_num, root);
    let mut den_product = 1i128;
    for (j, &other_root) in roots.iter().enumerate() {
      if i != j {
        den_product = den_product.checked_mul(root - other_root)?;
      }
    }
    // Include remaining factor evaluation
    if remaining_deg > 0 {
      let rem_at_root = evaluate_poly(&remaining, root);
      den_product = den_product.checked_mul(rem_at_root)?;
    } else if remaining.len() == 1 && remaining[0] != 0 && remaining[0] != 1 {
      den_product = den_product.checked_mul(remaining[0])?;
    }
    den_product = den_product.checked_mul(overall_factor)?;

    if den_product == 0 {
      return None;
    }

    // A_i = num_at_root / den_product as reduced fraction
    let g = gcd_i128(num_at_root.abs(), den_product.abs());
    let (mut an, mut ad) = (num_at_root / g, den_product / g);
    if ad < 0 {
      an = -an;
      ad = -ad;
    }

    if an == 0 {
      continue;
    }

    // Step 5: Build Log terms
    // Convention: argument is positive at x=0
    // root > 0: Log[root - x], root < 0: Log[-root + x], root = 0: Log[x]
    let log_arg = if root == 0 {
      Expr::Identifier(var.to_string())
    } else if root > 0 {
      // Log[root - x]
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(Expr::Integer(root)),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    } else {
      // root < 0: Log[-root + x]
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-root)),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    };

    let log_expr = Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![log_arg].into(),
    };

    // Build coefficient * Log[...]
    let term = if ad == 1 {
      if an == 1 {
        log_expr
      } else if an == -1 {
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(log_expr),
        }
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(an)),
          right: Box::new(log_expr),
        }
      }
    } else {
      // Use Rational form: Rational[an, ad] * Log[...]
      // But for positive fractions, Wolfram outputs Log[...]/ad
      // and for negative, -Log[...]/ad, etc.
      let abs_an = an.abs();
      let frac_expr = if abs_an == 1 {
        Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(log_expr.clone()),
          right: Box::new(Expr::Integer(ad)),
        }
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(crate::functions::math_ast::make_rational(abs_an, ad)),
          right: Box::new(log_expr.clone()),
        }
      };
      if an < 0 {
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(frac_expr),
        }
      } else {
        frac_expr
      }
    };

    log_terms.push(term);
  }

  // Step 6 & 7: Handle quadratic part
  let mut quad_terms: Vec<Expr> = Vec::new();

  if remaining_deg == 2 {
    // Remaining is a*x^2 + b*x + c (monic after normalization from factoring)
    // remaining = [c, b, a] where a should be 1 (monic from divide_by_root)
    let rem_a = remaining[2];
    let rem_b = remaining[1];
    let rem_c = remaining[0];

    // Normalize to monic: x^2 + (b/a)*x + (c/a)
    // Since we factored out integer roots, remaining should already be monic (a=1)
    if rem_a != 1 {
      return None; // Can't handle non-monic quadratics easily
    }

    let b = rem_b; // coefficient of x
    let c = rem_c; // constant term

    // discriminant: b^2 - 4c (for x^2 + bx + c)
    let disc = b * b - 4 * c;
    if disc > 0 {
      // Real but irrational roots: integration produces logs with √disc.
      // Only handle the case where there are no extracted integer roots and
      // the numerator is a constant — sufficient for `Integrate[1/(x^2+bx+c), x]`
      // style queries.  Wolframscript prints
      // `(Log[-b + s - 2x] - Log[b + s + 2x])/s` (simplified by gcd of the
      // integer pieces), so reproduce that shape here.
      if !roots.is_empty() || proper_num.len() != 1 {
        return None;
      }
      let n = proper_num[0];
      if n == 0 {
        return Some(Expr::Integer(0));
      }
      // Factor disc = k^2 * m with m square-free.
      let (k, m) = {
        let mut outside = 1i128;
        let mut inside = disc;
        let mut factor = 2i128;
        while factor * factor <= inside {
          while inside % (factor * factor) == 0 {
            outside *= factor;
            inside /= factor * factor;
          }
          factor += 1;
        }
        (outside, inside)
      };
      // Common divisor between the (-b, s, 2x) integer pieces. If g > 1 we
      // can divide every coefficient inside the Log args by g — wolframscript
      // does this so `1/(x^2 - 4x + 1)` prints with `2 + Sqrt[3] - x` rather
      // than `4 + 2*Sqrt[3] - 2x`. The outer denominator stays at the
      // unsimplified `k * Sqrt[m]` (the Log[g] terms cancel between the two
      // arguments).
      let g_bk = gcd_i128(b.abs(), k);
      let g = gcd_i128(g_bk, 2);
      let neg_b_g = -b / g;
      let b_g = b / g;
      let k_g = k / g;
      let two_g = 2 / g;

      // Construct the `k_g * Sqrt[m]` factor that appears inside each Log
      // argument. Collapses to `Sqrt[m]` when `k_g == 1` and to a bare
      // integer when `m == 1`.
      let inner_sqrt_factor = if m == 1 {
        Expr::Integer(k_g)
      } else if k_g == 1 {
        make_sqrt(Expr::Integer(m))
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(k_g)),
          right: Box::new(make_sqrt(Expr::Integer(m))),
        }
      };

      // Helper: build `coeff * x` (or just `x` when coeff == 1).
      let var_term = |coeff: i128| -> Expr {
        if coeff == 1 {
          Expr::Identifier(var.to_string())
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(coeff)),
            right: Box::new(Expr::Identifier(var.to_string())),
          }
        }
      };

      // arg_a = neg_b_g + inner_sqrt_factor - two_g*x
      // (wolframscript orders these as `-b + sqrt - x`, so do the same here).
      let arg_a = {
        let mut acc = if neg_b_g != 0 {
          Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(neg_b_g)),
            right: Box::new(inner_sqrt_factor.clone()),
          }
        } else {
          inner_sqrt_factor.clone()
        };
        acc = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(acc),
          right: Box::new(var_term(two_g)),
        };
        acc
      };
      // arg_b = b_g + inner_sqrt_factor + two_g*x
      let arg_b = {
        let mut acc = if b_g != 0 {
          Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(b_g)),
            right: Box::new(inner_sqrt_factor.clone()),
          }
        } else {
          inner_sqrt_factor.clone()
        };
        acc = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(acc),
          right: Box::new(var_term(two_g)),
        };
        acc
      };

      let log_a = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![arg_a].into(),
      };
      let log_b = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![arg_b].into(),
      };
      // `Log[arg_a] - Log[arg_b]`, then multiplied by `n` and divided by
      // `k * Sqrt[m]`.
      let log_diff = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(log_a),
        right: Box::new(log_b),
      };
      let outer_sqrt = if m == 1 {
        Expr::Integer(k)
      } else if k == 1 {
        make_sqrt(Expr::Integer(m))
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(k)),
          right: Box::new(make_sqrt(Expr::Integer(m))),
        }
      };
      let result_inner = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(log_diff),
        right: Box::new(outer_sqrt),
      };
      let combined = if n == 1 {
        result_inner
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(n / overall_factor)),
          right: Box::new(result_inner),
        }
      };
      // Combine with quotient integral if any.
      let final_expr = match quotient_integral {
        Some(qi) => Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(qi),
          right: Box::new(combined),
        },
        None => combined,
      };
      return Some(crate::functions::polynomial_ast::expand_and_combine(
        &final_expr,
      ));
    }
    if disc == 0 {
      // Repeated real root: 1/(x - r)^2 or similar. Bail out for now.
      return None;
    }

    // Compute (Bx + C) numerator of partial fraction for quadratic factor.
    // Use polynomial identity:
    // proper_num(x) = sum(A_j * overall_factor * cofactor_j(x) * quad(x))
    //                 + (Bx + C) * overall_factor * linear_product(x)
    // where cofactor_j(x) = prod_{k!=j}(x - r_k), linear_product(x) = prod(x - r_j)
    //
    // Strategy: compute the sum as a rational polynomial, subtract from proper_num,
    // then divide by overall_factor * linear_product to get (Bx + C).

    // Build linear_product polynomial = prod(x - r_j)
    let mut lin_prod_poly = vec![1i128]; // start with constant 1
    for &root in &roots {
      // Multiply by (x - root): new[i] = old[i-1] - root * old[i]
      let mut new_poly = vec![0i128; lin_prod_poly.len() + 1];
      for (i, &coeff) in lin_prod_poly.iter().enumerate() {
        new_poly[i] -= root * coeff;
        new_poly[i + 1] += coeff;
      }
      lin_prod_poly = new_poly;
    }

    // Compute the sum of residue contributions as a rational polynomial.
    // Use a common denominator: multiply proper_num by LCM of all ad_j,
    // subtract the integer contributions, then divide.
    // First, collect residues as (an_j, ad_j) pairs.
    let mut residues: Vec<(i128, i128)> = Vec::new();
    for (i, &root) in roots.iter().enumerate() {
      let num_at_root = evaluate_poly(&proper_num, root);
      let mut den_prod = 1i128;
      for (j, &other_root) in roots.iter().enumerate() {
        if i != j {
          den_prod *= root - other_root;
        }
      }
      let rem_at_root = evaluate_poly(&remaining, root);
      den_prod *= rem_at_root;
      den_prod *= overall_factor;
      if den_prod == 0 {
        return None;
      }
      let g = gcd_i128(num_at_root.abs(), den_prod.abs());
      let (mut an, mut ad) = (num_at_root / g, den_prod / g);
      if ad < 0 {
        an = -an;
        ad = -ad;
      }
      residues.push((an, ad));
    }

    // Compute LCM of all denominators
    let mut lcm = 1i128;
    for &(_, ad) in &residues {
      let g = gcd_i128(lcm, ad);
      lcm = (lcm / g).checked_mul(ad)?;
    }

    // Build: lcm * proper_num - sum(lcm/ad_j * an_j * overall_factor * cofactor_j * quad)
    // = lcm * (Bx + C) * overall_factor * linear_product
    let mut residual: Vec<i128> = proper_num.iter().map(|&c| c * lcm).collect();
    // Pad to degree of den
    let target_deg = den_coeffs.len() - 1;
    while residual.len() <= target_deg {
      residual.push(0);
    }

    let quad_poly = remaining.clone(); // [c, b, 1]

    for (i, &_root) in roots.iter().enumerate() {
      let (an, ad) = residues[i];
      let scale = (lcm / ad) * an * overall_factor;

      // Build cofactor_j(x) = prod_{k!=j}(x - r_k)
      let mut cofactor = vec![1i128];
      for (j, &other_root) in roots.iter().enumerate() {
        if i != j {
          let mut new_poly = vec![0i128; cofactor.len() + 1];
          for (k, &coeff) in cofactor.iter().enumerate() {
            new_poly[k] -= other_root * coeff;
            new_poly[k + 1] += coeff;
          }
          cofactor = new_poly;
        }
      }

      // Multiply cofactor by quad_poly
      let mut product = vec![0i128; cofactor.len() + quad_poly.len() - 1];
      for (ci, &cv) in cofactor.iter().enumerate() {
        for (qi, &qv) in quad_poly.iter().enumerate() {
          product[ci + qi] += cv * qv;
        }
      }

      // Subtract scale * product from residual
      for (k, &pv) in product.iter().enumerate() {
        if k < residual.len() {
          residual[k] -= scale * pv;
        }
      }
    }

    // Now residual = lcm * (Bx + C) * overall_factor * linear_product
    // Divide residual by (overall_factor * linear_product)
    let mut divisor_poly = lin_prod_poly.clone();
    // Multiply divisor by overall_factor
    for c in &mut divisor_poly {
      *c *= overall_factor;
    }
    // Polynomial division: residual / divisor_poly should give (lcm * B)x + (lcm * C)
    // Use poly_long_divide
    let (bc_scaled, rem_check) = poly_long_divide(&residual, &divisor_poly);
    if !rem_check.iter().all(|&c| c == 0) {
      return None;
    }

    // Trim trailing zeros from bc_scaled
    let mut bc_scaled = bc_scaled;
    while bc_scaled.len() > 1 && bc_scaled.last() == Some(&0) {
      bc_scaled.pop();
    }

    // bc_scaled should be degree 1: [lcm*C, lcm*B]
    if bc_scaled.len() > 2 {
      return None;
    }
    let bc_c_scaled = if bc_scaled.is_empty() {
      0
    } else {
      bc_scaled[0]
    };
    let bc_b_scaled = if bc_scaled.len() < 2 { 0 } else { bc_scaled[1] };

    // Divide by lcm to get B and C as rationals
    let g_b = gcd_i128(bc_b_scaled.abs(), lcm);
    let (big_b_num, big_b_den) = (bc_b_scaled / g_b, lcm / g_b);
    let g_c = gcd_i128(bc_c_scaled.abs(), lcm);
    let (big_c_num, big_c_den) = (bc_c_scaled / g_c, lcm / g_c);

    // Convert to common B and C as integers (if possible) or rationals
    // For the integration formula, we need B and C as rationals
    // big_b = big_b_num / big_b_den, big_c = big_c_num / big_c_den
    // But our formula uses integer B and C. If they're not integers, we need to adjust.
    // Use common denominator for B and C as rationals
    let common_den = {
      let g = gcd_i128(big_b_den, big_c_den);
      (big_b_den / g).checked_mul(big_c_den)?
    };
    let big_b_int = big_b_num * (common_den / big_b_den);
    let big_c_int = big_c_num * (common_den / big_c_den);
    // The quadratic partial fraction is (big_b_int * x + big_c_int) / (common_den * (x^2+bx+c))

    // Step 7: Integrate (big_b_int * x + big_c_int) / (common_den * (x^2 + bx + c))
    // = (1/common_den) * Integrate[(big_b_int * x + big_c_int) / (x^2 + bx + c)]
    // Split: (Bx + C) = (B/2)(2x + b) + (C - Bb/2)
    // where B = big_b_int, C = big_c_int
    // Integral of (2x+b)/(x^2+bx+c) = Log[x^2+bx+c]
    // Integral of 1/(x^2+bx+c) = (2/sqrt(4c-b^2)) * ArcTan[(2x+b)/sqrt(4c-b^2)]

    let neg_disc = 4 * c - b * b; // = -(b^2-4c) > 0 since disc < 0
    if neg_disc <= 0 {
      return None;
    }

    // Build quadratic expr for Log: c + b*x + x^2 (Wolfram orders by ascending power)
    let quad_log_arg = if b == 0 && c == 1 {
      // 1 + x^2
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Identifier(var.to_string())),
          right: Box::new(Expr::Integer(2)),
        }),
      }
    } else {
      coeffs_to_expr(&[c, b, 1], var)
    };

    // Log part coefficient: B/(2*common_den)
    // = big_b_int / (2 * common_den)
    let log_total_num = big_b_int;
    let log_total_den = 2 * common_den;
    let g_log = gcd_i128(log_total_num.abs(), log_total_den);
    let (log_num, log_den) = (log_total_num / g_log, log_total_den / g_log);

    if log_num != 0 {
      let log_expr = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![quad_log_arg.clone()].into(),
      };
      let log_term = build_coeff_times_expr(log_num, log_den, log_expr);
      quad_terms.push(log_term);
    }

    // ArcTan part coefficient: (2*big_c_int - big_b_int*b) / (common_den * sqrt(4c - b^2))
    let arctan_coeff_num = 2 * big_c_int - big_b_int * b;
    if arctan_coeff_num != 0 {
      // Extract perfect square factor from neg_disc: neg_disc = k^2 * m (m square-free)
      let (k, m) = {
        let mut outside = 1i128;
        let mut inside = neg_disc;
        let mut factor = 2i128;
        while factor * factor <= inside {
          while inside % (factor * factor) == 0 {
            outside *= factor;
            inside /= factor * factor;
          }
          factor += 1;
        }
        (outside, inside)
      };
      // sqrt(neg_disc) = k * sqrt(m), where sqrt(1) = 1

      // Simplify ArcTan argument: (b + 2*x) / (k * sqrt(m))
      // The numerator coefficients are b and 2; divide both and k by their common factor.
      let inner_gcd = if b == 0 { 2 } else { gcd_i128(b.abs(), 2) };
      let g = gcd_i128(inner_gcd, k);
      let k_reduced = k / g;
      let b_simplified = b / g;
      let two_simplified = 2 / g;
      let arctan_coeff_num = arctan_coeff_num / g;

      // Build ArcTan argument denominator expression
      let sqrt_denom = if m == 1 {
        if k_reduced <= 1 {
          None // denominator is 1
        } else {
          Some(Expr::Integer(k_reduced))
        }
      } else if k_reduced <= 1 {
        Some(make_sqrt(Expr::Integer(m)))
      } else {
        Some(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(k_reduced)),
          right: Box::new(make_sqrt(Expr::Integer(m))),
        })
      };

      // Build ArcTan argument numerator: b_simplified + two_simplified * x
      let arctan_numerator = if b_simplified == 0 {
        if two_simplified == 1 {
          Expr::Identifier(var.to_string())
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(two_simplified)),
            right: Box::new(Expr::Identifier(var.to_string())),
          }
        }
      } else {
        let x_term = if two_simplified == 1 {
          Expr::Identifier(var.to_string())
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(two_simplified)),
            right: Box::new(Expr::Identifier(var.to_string())),
          }
        };
        Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(b_simplified)),
          right: Box::new(x_term),
        }
      };

      // ArcTan argument: numerator / denom (or just numerator if denom is 1)
      let arctan_inner = if let Some(denom) = &sqrt_denom {
        Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(arctan_numerator),
          right: Box::new(denom.clone()),
        }
      } else {
        arctan_numerator
      };

      let arctan_expr = Expr::FunctionCall {
        name: "ArcTan".to_string(),
        args: vec![arctan_inner].into(),
      };

      // Full coefficient: arctan_coeff_num / (common_den * k_reduced * sqrt(m))
      // Reduce: arctan_coeff_num / common_den first
      let g_at = gcd_i128(arctan_coeff_num.abs(), common_den);
      let at_num = arctan_coeff_num / g_at;
      let at_den_int = common_den / g_at;

      // Build effective sqrt expression combining at_den_int, k_reduced, and sqrt(m)
      let int_factor = at_den_int * k_reduced;
      let effective_sqrt = if m == 1 {
        Expr::Integer(int_factor)
      } else if int_factor == 1 {
        make_sqrt(Expr::Integer(m))
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(int_factor)),
          right: Box::new(make_sqrt(Expr::Integer(m))),
        }
      };

      let arctan_term = build_arctan_term(at_num, &effective_sqrt, arctan_expr);
      quad_terms.push(arctan_term);
    }
  } else if remaining_deg == 0 && roots.is_empty() {
    // No roots and no quadratic - can't decompose
    return None;
  }

  // Step 8: Combine all terms
  let mut all_terms: Vec<Expr> = Vec::new();
  if let Some(qi) = quotient_integral {
    all_terms.push(qi);
  }
  // ArcTan terms come before Log terms in Wolfram output ordering
  // Actually, looking at the expected outputs:
  // x/(1-x^3) → -(ArcTan[...]/Sqrt[3]) - Log[1 - x]/3 + Log[1 + x + x^2]/6
  // (2x+3)/(x^2+x+1) → (4*ArcTan[...])/Sqrt[3] + Log[...]
  // So ArcTan terms come first, then Log terms
  all_terms.extend(quad_terms);
  all_terms.extend(log_terms);

  if all_terms.is_empty() {
    return None;
  }

  Some(build_sum(all_terms))
}

/// Build an ArcTan term: coeff_num * arctan_expr / sqrt_expr
/// Handles simplification for common cases.
fn build_arctan_term(
  coeff_num: i128,
  sqrt_expr: &Expr,
  arctan_expr: Expr,
) -> Expr {
  let abs_coeff = coeff_num.abs();

  let term = if let Expr::Integer(s) = sqrt_expr {
    // sqrt is an integer - can simplify
    let g = gcd_i128(abs_coeff, *s);
    let reduced_num = abs_coeff / g;
    let reduced_den = *s / g;

    if reduced_den == 1 {
      if reduced_num == 1 {
        arctan_expr.clone()
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(reduced_num)),
          right: Box::new(arctan_expr.clone()),
        }
      }
    } else if reduced_num == 1 {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(arctan_expr.clone()),
        right: Box::new(Expr::Integer(reduced_den)),
      }
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(reduced_num)),
          right: Box::new(arctan_expr.clone()),
        }),
        right: Box::new(Expr::Integer(reduced_den)),
      }
    }
  } else {
    // sqrt is Sqrt[n] - put it in denominator
    if abs_coeff == 1 {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(arctan_expr.clone()),
        right: Box::new(sqrt_expr.clone()),
      }
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(abs_coeff)),
          right: Box::new(arctan_expr.clone()),
        }),
        right: Box::new(sqrt_expr.clone()),
      }
    }
  };

  if coeff_num < 0 {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(term),
    }
  } else {
    term
  }
}

/// Build (num/den) * expr, handling sign and simplifications.
fn build_coeff_times_expr(num: i128, den: i128, expr: Expr) -> Expr {
  let abs_num = num.abs();

  let unsigned_term = if den == 1 {
    if abs_num == 1 {
      expr
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(abs_num)),
        right: Box::new(expr),
      }
    }
  } else if abs_num == 1 {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(expr),
      right: Box::new(Expr::Integer(den)),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(crate::functions::math_ast::make_rational(abs_num, den)),
      right: Box::new(expr),
    }
  };

  if num < 0 {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(unsigned_term),
    }
  } else {
    unsigned_term
  }
}

/// LIATE priority for integration by parts: higher = choose as u first
/// Log > Inverse trig > Algebraic > Trig > Exponential
fn liate_priority(expr: &Expr, var: &str) -> i32 {
  match expr {
    Expr::FunctionCall { name, .. } => match name.as_str() {
      "Log" => 5,
      "ArcSin" | "ArcCos" | "ArcTan" => 4,
      "Sin" | "Cos" | "Tan" | "Sec" | "Csc" | "Cot" | "Sinh" | "Cosh" => 2,
      "Exp" => 1,
      _ => 3,
    },
    // x, x^n => algebraic
    Expr::Identifier(_) => 3,
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    }
      // Any constant^(var-dependent) is exponential (E^x, e^x, 2^x, etc.)
      if is_constant_wrt(left, var) && !is_constant_wrt(right, var) => {
        1
      }
    _ => 3,
  }
}

/// Convert FunctionCall("Power", [base, exp]) → BinaryOp(Power, base, exp)
/// so that times_ast can combine powers with matching bases.
fn normalize_power(expr: Expr) -> Expr {
  if let Expr::FunctionCall { ref name, ref args } = expr
    && name == "Power"
    && args.len() == 2
  {
    return Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(args[0].clone()),
      right: Box::new(args[1].clone()),
    };
  }
  expr
}

/// Check if an expression is a pure exponential E^f(x).
fn is_exponential(expr: &Expr) -> bool {
  matches!(expr, Expr::BinaryOp {
    op: BinaryOperator::Power,
    left,
    ..
  } if matches!(left.as_ref(), Expr::Constant(c) if c == "E"))
}

/// Compare two expressions by their string representation.
pub(crate) fn expr_str_eq(a: &Expr, b: &Expr) -> bool {
  crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
}

/// Try to remove a specific factor from a product expression.
/// E.g., try_remove_factor(Times[2, E^x, (-1+x)], E^x) → Some(Times[2, (-1+x)])
/// E.g., try_remove_factor(E^x, E^x) → Some(Integer(1))
fn try_remove_factor(expr: &Expr, factor: &Expr) -> Option<Expr> {
  if expr_str_eq(expr, factor) {
    return Some(Expr::Integer(1));
  }
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      for (i, arg) in args.iter().enumerate() {
        if expr_str_eq(arg, factor) {
          let remaining: Vec<_> = args
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, a)| a.clone())
            .collect();
          return Some(if remaining.len() == 1 {
            remaining.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: remaining.into(),
            }
          });
        }
      }
      None
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if expr_str_eq(left, factor) {
        Some(*right.clone())
      } else if expr_str_eq(right, factor) {
        Some(*left.clone())
      } else {
        None
      }
    }
    _ => None,
  }
}

std::thread_local! {
  // Recursion depth counter for integration by parts
  static IBP_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Check if expr is a polynomial in var (derivatives eventually reach 0).
fn is_polynomial_in(expr: &Expr, var: &str) -> bool {
  let mut current = expr.clone();
  for _ in 0..20 {
    if matches!(&current, Expr::Integer(0)) {
      return true;
    }
    match differentiate(&current, var) {
      Ok(d) => current = simplify(d),
      Err(_) => return false,
    }
  }
  false
}

/// Match `1/p(x)` where `p` is an irreducible-looking univariate polynomial
/// of degree ≥ 5 with no rational roots. Returns the wolframscript-style
/// `RootSum[Function[p in #1], Function[Log[x - #1] / p'(#1)]]` antiderivative.
/// Returns `None` for everything else, leaving the standard integration
/// rules in charge for low-degree denominators that have closed forms.
fn try_integrate_one_over_poly_rootsum(expr: &Expr, var: &str) -> Option<Expr> {
  use crate::functions::polynomial_ast::{
    expand_and_combine, extract_poly_coeffs, find_integer_root,
  };
  use BinaryOperator::*;
  // Match `Power[poly, -1]` in either BinaryOp or FunctionCall form.
  let poly_raw: Expr = match expr {
    Expr::BinaryOp {
      op: Power,
      left,
      right,
    } if matches!(right.as_ref(), Expr::Integer(-1)) => (**left).clone(),
    Expr::FunctionCall { name, args }
      if name == "Power"
        && args.len() == 2
        && matches!(&args[1], Expr::Integer(-1)) =>
    {
      args[0].clone()
    }
    _ => return None,
  };
  // Need an integer-coefficient univariate polynomial in `var` of degree ≥ 5.
  let poly = expand_and_combine(&poly_raw);
  let coeffs = extract_poly_coeffs(&poly, var)?;
  if coeffs.len() < 6 {
    return None;
  }
  // Skip when wolframscript's preferred form is plain Log/ArcTan: any
  // polynomial with a rational root factors out a linear term and the
  // standard partial-fractions path takes over instead. (`find_integer_root`
  // is sufficient for monic polynomials, which is the common case here.)
  if find_integer_root(&coeffs).is_some() {
    return None;
  }
  // p'(x) = sum k * c_k * x^(k-1). Use the symbolic differentiator so the
  // result is in the same canonical form as the integrand.
  let pprime = differentiate(&poly, var).ok()?;
  let pprime = simplify(pprime);
  // Substitute `x → #1` in both poly and p' to build the Function bodies.
  let poly_in_slot =
    crate::syntax::substitute_variable(&poly, var, &Expr::Slot(1));
  let pprime_in_slot =
    crate::syntax::substitute_variable(&pprime, var, &Expr::Slot(1));
  // Inner function body: Log[x - #1] / p'(#1).
  let var_expr = Expr::Identifier(var.to_string());
  let log_arg = Expr::BinaryOp {
    op: Minus,
    left: Box::new(var_expr),
    right: Box::new(Expr::Slot(1)),
  };
  let log_term = Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![log_arg].into(),
  };
  let body = Expr::BinaryOp {
    op: Divide,
    left: Box::new(log_term),
    right: Box::new(pprime_in_slot),
  };
  let log_fn = Expr::Function {
    body: Box::new(body),
  };
  let poly_fn = Expr::Function {
    body: Box::new(poly_in_slot),
  };
  Some(Expr::FunctionCall {
    name: "RootSum".to_string(),
    args: vec![poly_fn, log_fn].into(),
  })
}

/// Closed-form integration of polynomial × constant-base exponential:
/// ∫ P(x) * a^(c*x) dx = a^(c*x) * Σ_{k=0}^{deg} (-1)^k * P^(k)(x) / r^(k+1)
/// where r = c * Log[a] is the effective rate.
fn try_integrate_poly_times_const_exp(
  poly: &Expr,
  exponential: &Expr,
  base: &Expr,
  coeff: &Expr,
  var: &str,
) -> Option<Expr> {
  use BinaryOperator::*;

  // For base E, Log[E] = 1, so rate = coeff directly.
  // For other bases, rate = coeff * Log[base].
  let rate = if matches!(base, Expr::Constant(c) if c == "E") {
    simplify(coeff.clone())
  } else {
    let log_base = Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![base.clone()].into(),
    };
    simplify(Expr::BinaryOp {
      op: Times,
      left: Box::new(coeff.clone()),
      right: Box::new(log_base),
    })
  };

  // Collect derivatives of poly until we reach 0
  let mut derivs = vec![poly.clone()];
  let mut current = poly.clone();
  for _ in 0..20 {
    match differentiate(&current, var) {
      Ok(d) => {
        let d = simplify(d);
        if matches!(&d, Expr::Integer(0)) {
          break;
        }
        derivs.push(d.clone());
        current = d;
      }
      Err(_) => return None,
    }
  }

  // For numeric rates (e.g., base E with fractional coeff), compute each term
  // directly with 1/rate^(k+1) to get clean integer coefficients.
  // For symbolic rates (non-E bases involving Log), use the common-denominator
  // form: (exponential * Σ P^(k)(x)*rate^(n-1-k)) / rate^n.
  let is_numeric_rate = matches!(&rate, Expr::Integer(_))
    || matches!(&rate, Expr::FunctionCall { name, .. } if name == "Rational");

  let n = derivs.len();

  if is_numeric_rate {
    // Direct approach: Σ (-1)^k * P^(k)(x) / rate^(k+1)
    let inv_rate =
      crate::functions::math_ast::divide_ast(&[Expr::Integer(1), rate.clone()])
        .unwrap_or_else(|_| Expr::BinaryOp {
          op: Divide,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(rate.clone()),
        });

    let mut num_terms = Vec::new();
    for (k, deriv) in derivs.iter().enumerate() {
      let k1 = k as i128 + 1;
      let inv_rate_factor = if k1 == 1 {
        inv_rate.clone()
      } else {
        crate::functions::math_ast::power_two(&inv_rate, &Expr::Integer(k1))
          .unwrap_or_else(|_| Expr::BinaryOp {
            op: Power,
            left: Box::new(inv_rate.clone()),
            right: Box::new(Expr::Integer(k1)),
          })
      };

      let mut term = simplify(Expr::BinaryOp {
        op: Times,
        left: Box::new(deriv.clone()),
        right: Box::new(inv_rate_factor),
      });

      if k % 2 == 1 {
        term = simplify(Expr::BinaryOp {
          op: Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(term),
        });
      }

      num_terms.push(term);
    }

    let numerator = if num_terms.len() == 1 {
      num_terms.into_iter().next().unwrap()
    } else {
      let combined = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: num_terms.into(),
      };
      crate::functions::polynomial_ast::expand_and_combine(&combined)
    };

    let result = simplify(Expr::BinaryOp {
      op: Times,
      left: Box::new(exponential.clone()),
      right: Box::new(numerator),
    });

    Some(result)
  } else {
    // Common-denominator approach: (exponential * Σ P^(k)(x)*rate^(n-1-k)) / rate^n
    let mut num_terms = Vec::new();
    for (k, deriv) in derivs.iter().enumerate() {
      let rate_power = n as i128 - 1 - k as i128;
      let rate_factor = if rate_power == 0 {
        Expr::Integer(1)
      } else if rate_power == 1 {
        rate.clone()
      } else {
        Expr::BinaryOp {
          op: Power,
          left: Box::new(rate.clone()),
          right: Box::new(Expr::Integer(rate_power)),
        }
      };

      let mut term = simplify(Expr::BinaryOp {
        op: Times,
        left: Box::new(deriv.clone()),
        right: Box::new(rate_factor),
      });

      if k % 2 == 1 {
        term = simplify(Expr::BinaryOp {
          op: Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(term),
        });
      }

      num_terms.push(term);
    }

    let numerator = if num_terms.len() == 1 {
      num_terms.into_iter().next().unwrap()
    } else {
      let combined = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: num_terms.into(),
      };
      crate::functions::polynomial_ast::expand_and_combine(&combined)
    };

    let denom = if n == 1 {
      rate
    } else {
      Expr::BinaryOp {
        op: Power,
        left: Box::new(rate),
        right: Box::new(Expr::Integer(n as i128)),
      }
    };

    let result_num = simplify(Expr::BinaryOp {
      op: Times,
      left: Box::new(exponential.clone()),
      right: Box::new(numerator),
    });

    Some(Expr::BinaryOp {
      op: Divide,
      left: Box::new(result_num),
      right: Box::new(denom),
    })
  }
}

/// Try u-substitution for a product of two factors: ∫ f(x) * g(h(x)) dx.
/// If f(x) is proportional to h'(x), the integral is (1/c) * G(h(x))
/// where G is the antiderivative of g and f(x) = c * h'(x).
fn try_u_substitution_binary(
  left: &Expr,
  right: &Expr,
  var: &str,
) -> Option<Expr> {
  // Try both orderings: left * g(right_inner) and right * g(left_inner)
  for (factor, composite) in [(left, right), (right, left)] {
    // Look for composite functions: Exp[h(x)], Sin[h(x)], Cos[h(x)], etc.
    let (outer_fn, inner) = match composite {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: base,
        right: exp,
      } => {
        // E^h(x) or base^h(x)
        if matches!(base.as_ref(), Expr::Constant(c) if c == "E")
          || matches!(base.as_ref(), Expr::Identifier(c) if c == "E")
        {
          ("Exp", exp.as_ref())
        } else {
          continue;
        }
      }
      Expr::FunctionCall { name, args }
        if args.len() == 1
          && matches!(name.as_str(), "Sin" | "Cos" | "Tan" | "Log" | "Exp") =>
      {
        (name.as_str(), &args[0])
      }
      _ => continue,
    };

    // Compute h'(x) and check if factor is proportional to h'(x)
    let h_deriv = differentiate(inner, var).ok()?;
    if is_constant_wrt(&h_deriv, var) && !is_constant_wrt(factor, var) {
      continue; // h'(x) is constant but factor depends on var — no match
    }
    // Try to compute factor / h'(x) and check if it's constant
    let ratio = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(factor.clone()),
      right: Box::new(h_deriv.clone()),
    })
    .ok()?;
    if !is_constant_wrt(&ratio, var) {
      continue;
    }
    // factor = ratio * h'(x), so ∫ factor * g(h(x)) dx = ratio * G(h(x))
    // where G is the antiderivative of g.
    // Compute G(h(x)) directly based on the outer function.
    let antideriv_of_h = match outer_fn {
      "Exp" => composite.clone(), // G(Exp) = Exp, so G(h(x)) = Exp[h(x)]
      "Sin" => Expr::FunctionCall {
        name: "Cos".to_string(),
        args: vec![inner.clone()].into(),
      }, // ∫ Sin = -Cos → handle sign below
      "Cos" => Expr::FunctionCall {
        name: "Sin".to_string(),
        args: vec![inner.clone()].into(),
      },
      _ => {
        continue; // Other functions: skip for now
      }
    };
    // For Sin, the antiderivative is -Cos, so flip the ratio sign
    let ratio = if outer_fn == "Sin" {
      crate::evaluator::evaluate_expr_to_expr(&Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(ratio),
      })
      .ok()?
    } else {
      ratio
    };
    // Multiply by the constant ratio
    let final_result =
      crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(ratio),
        right: Box::new(antideriv_of_h),
      })
      .ok()?;
    return Some(final_result);
  }
  // Strategy 2: ∫ c * f'(x) * f(x) dx = c * f(x)^2 / 2
  // More generally: ∫ c * h'(x) * g(h(x)) dx where one factor is
  // the derivative of the other (not wrapped in a composite function).
  for (candidate_h, other) in [(left, right), (right, left)] {
    // Skip trivial cases (variable itself)
    if matches!(candidate_h, Expr::Identifier(name) if name == var) {
      continue;
    }
    let h_deriv = differentiate(candidate_h, var).ok()?;
    if is_constant_wrt(&h_deriv, var) {
      continue; // h'(x) is constant — not useful
    }
    // Check if other = c * h'(x)
    let ratio = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(other.clone()),
      right: Box::new(h_deriv),
    })
    .ok()?;
    if !is_constant_wrt(&ratio, var) {
      continue;
    }
    // ∫ c * h'(x) * h(x) dx = c * h(x)^2 / 2
    let result = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(ratio),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(candidate_h.clone()),
          right: Box::new(Expr::Integer(2)),
        }),
        right: Box::new(Expr::Integer(2)),
      }),
    })
    .ok()?;
    return Some(result);
  }
  // Strategy 3: ∫ c * h'(x) * h(x)^p dx = c * h(x)^(p+1)/(p+1) for a constant
  // power p ≠ -1, the u = h(x) substitution.  Catches e.g.
  //   ∫ x (a + b x^2)^p dx = (a + b x^2)^(p+1) / (2 b (p+1)).
  // Restricted to NON-integer p: integer powers are handled elsewhere and
  // Wolfram expands positive-integer results into polynomials (so the closed
  // form here would not match its output).
  for (composite, other) in [(left, right), (right, left)] {
    let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = composite
    else {
      continue;
    };
    // base must depend on var; exp must be a constant, non-integer power.
    if is_constant_wrt(base, var)
      || !is_constant_wrt(exp, var)
      || matches!(exp.as_ref(), Expr::Integer(_))
    {
      continue;
    }
    let Ok(h_deriv) = differentiate(base, var) else {
      continue;
    };
    if is_constant_wrt(&h_deriv, var) {
      continue;
    }
    // other = c * h'(x)?
    let Ok(ratio) = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(other.clone()),
      right: Box::new(h_deriv),
    }) else {
      continue;
    };
    if !is_constant_wrt(&ratio, var) {
      continue;
    }
    // p + 1 (guaranteed ≠ 0 since p is not the integer -1)
    let Ok(p_plus_1) =
      crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: exp.clone(),
        right: Box::new(Expr::Integer(1)),
      })
    else {
      continue;
    };
    if matches!(p_plus_1, Expr::Integer(0)) {
      continue;
    }
    // ratio * base^(p+1) / (p+1)
    let Ok(result) = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(ratio),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: base.clone(),
          right: Box::new(p_plus_1.clone()),
        }),
        right: Box::new(p_plus_1),
      }),
    }) else {
      continue;
    };
    return Some(result);
  }
  None
}

/// Try integration by parts: ∫ u dv = u*v - ∫ v du
/// `factors` are the factors that depend on `var`.
/// We pick `u` using the LIATE heuristic and `dv` is the product of the remaining factors.
fn try_integration_by_parts(factors: &[&Expr], var: &str) -> Option<Expr> {
  if factors.len() < 2 {
    return None;
  }

  // Limit recursion depth to prevent infinite loops
  let depth = IBP_DEPTH.with(|d| d.get());
  if depth >= 5 {
    return None;
  }

  // Find the factor with the highest LIATE priority → that becomes u
  let mut best_u_idx = 0;
  let mut best_priority = liate_priority(factors[0], var);
  for (i, f) in factors.iter().enumerate().skip(1) {
    let p = liate_priority(f, var);
    if p > best_priority {
      best_priority = p;
      best_u_idx = i;
    }
  }

  let u = factors[best_u_idx];

  // dv is the product of the remaining factors
  let dv_factors: Vec<&Expr> = factors
    .iter()
    .enumerate()
    .filter(|(i, _)| *i != best_u_idx)
    .map(|(_, f)| *f)
    .collect();

  // Special case: polynomial × constant-base exponential (including E base)
  // Use closed-form formula: ∫ P(x)*a^(cx) dx = a^(cx) * Σ (-1)^k P^(k)(x) / (c*Log[a])^(k+1)
  // For a=E, rate = c*Log[E] = c, giving direct polynomial-times-E^(cx) integration.
  if dv_factors.len() == 1
    && let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = dv_factors[0]
    && is_constant_wrt(base, var)
    && !is_constant_wrt(exp, var)
    && let Some(coeff) = try_match_linear_arg(exp, var)
    && is_polynomial_in(u, var)
  {
    return try_integrate_poly_times_const_exp(
      u,
      dv_factors[0],
      base,
      &coeff,
      var,
    );
  }

  let dv_expr = if dv_factors.len() == 1 {
    dv_factors[0].clone()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: dv_factors.iter().map(|f| (*f).clone()).collect(),
    }
  };

  // v = ∫ dv
  let v = integrate(&dv_expr, var)?;

  // du = D[u, var]
  let du = differentiate(u, var).ok()?;
  let du = simplify(du);

  // Decompose v into (v_core, v_denom) where v = v_core / v_denom
  // This allows proper fraction handling: u*v = (u*v_core)/v_denom
  let (v_core, v_denom) = if let Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: v_num,
    right: v_den,
  } = &v
  {
    if is_constant_wrt(v_den, var) {
      (*v_num.clone(), Some(*v_den.clone()))
    } else {
      (v.clone(), None)
    }
  } else {
    (v.clone(), None)
  };

  // Result: u*v - ∫ v*du dx
  // When v is a fraction a/b, compute uv as (u*a)/b for proper display
  // e.g. Log[x] * (x^2/2) → (x^2*Log[x])/2 instead of x^2/2*Log[x]
  let uv = {
    let u_times_core = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(u.clone()),
      right: Box::new(v_core.clone()),
    });
    if let Some(ref denom) = v_denom {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(u_times_core),
        right: Box::new(denom.clone()),
      }
    } else {
      u_times_core
    }
  };

  // If du is 0, the integral is just u*v
  if matches!(&du, Expr::Integer(0)) {
    return Some(uv);
  }

  // Normalize du: convert FunctionCall("Power", [base, exp]) → BinaryOp(Power)
  // so that times_ast can combine powers like x^2 * x^(-1) → x
  let du = normalize_power(du);

  // Compute v*du, decomposing v = numerator/constant for better simplification
  // (e.g., v = x^2/2, du = x^(-1) → numerator*du = x^2 * x^(-1) = x → v*du = x/2)
  let v_du = if let Some(ref denom) = v_denom {
    let num_du =
      crate::functions::math_ast::times_ast(&[v_core.clone(), du]).ok()?;
    simplify(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(num_du),
      right: Box::new(denom.clone()),
    })
  } else {
    crate::functions::math_ast::times_ast(&[v, du]).ok()?
  };

  // Integrate v*du with increased depth
  IBP_DEPTH.with(|d| d.set(depth + 1));
  let int_v_du = integrate(&v_du, var);
  IBP_DEPTH.with(|d| d.set(depth));

  let int_v_du = int_v_du?;
  // Simplify to flatten nested products (e.g., BinaryOp(Times, 2, E^x*...) → Times[2, E^x, ...])
  let int_v_du = simplify(int_v_du);

  // Try to factor out common exponential factor (e.g., E^x from E^x*x - E^x → E^x*(-1+x))
  // This matches Wolfram's output form for exponential integrals.
  if is_exponential(&v_core)
    && let Some(quotient) = try_remove_factor(&int_v_du, &v_core)
  {
    // result = v_core * (u - quotient)
    // Expand the inner expression so e.g. x^2 - 2*(-1+x) becomes 2 - 2*x + x^2
    let inner =
      crate::functions::math_ast::subtract_ast(&[u.clone(), quotient]).ok()?;
    let inner = crate::functions::polynomial_ast::expand_and_combine(&inner);
    let result = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(v_core),
      right: Box::new(inner),
    });
    return Some(result);
  }

  Some(simplify(Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(uv),
    right: Box::new(int_v_du),
  }))
}

/// Integrate an expression with respect to a variable
/// ∫ c·g'(x)/g(x) dx = c·Log[g(x)]. Detects an integrand that is a constant
/// multiple of the logarithmic derivative of some sub-expression `g` (a
/// denominator factor). This catches transcendental `g` — Log[x], Sin[x],
/// 1 + E^x, … — that the polynomial-only rational path does not.
///
/// The test is exact: if `integrand · g / g'` evaluates to a value free of
/// `var`, then `integrand == c · g'/g` identically, so the antiderivative is
/// `c · Log[g]`.
/// ∫ c·g'(x)/g(x)^n dx = c·g(x)^(1-n)/(1-n) for an integer n >= 2 (the u = g(x)
/// substitution). Returns None unless the denominator is a power g^n (n >= 2)
/// and the numerator is a nonzero constant multiple of g'(x).
fn try_integrate_power_derivative(expr: &Expr, var: &str) -> Option<Expr> {
  // Extract the denominator: literal `a / b` or the Times[..., Power[d, -k]] form.
  let den = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      right,
      ..
    } => (**right).clone(),
    _ => extract_quotient_from_times(expr).map(|(_, d)| d)?,
  };
  // Denominator must be g^n with an integer n >= 2.
  let (g, n): (Expr, i128) = match &den {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => match right.as_ref() {
      Expr::Integer(k) => ((**left).clone(), *k),
      _ => return None,
    },
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      match &args[1] {
        Expr::Integer(k) => (args[0].clone(), *k),
        _ => return None,
      }
    }
    _ => return None,
  };
  if n < 2 || is_constant_wrt(&g, var) {
    return None;
  }
  let dg = differentiate(&g, var).ok()?;
  // ratio = expr · g^n / g' = numerator / g'  must be a nonzero constant.
  let ratio = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      expr.clone(),
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![g.clone(), Expr::Integer(n)].into(),
      },
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![dg, Expr::Integer(-1)].into(),
      },
    ]
    .into(),
  };
  let c = crate::evaluator::evaluate_expr_to_expr(&ratio).ok()?;
  if !is_constant_wrt(&c, var) || matches!(&c, Expr::Integer(0)) {
    return None;
  }
  // result = c/(1 - n) * g^(1 - n)
  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(c),
        right: Box::new(Expr::Integer(1 - n)),
      },
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![g, Expr::Integer(1 - n)].into(),
      },
    ]
    .into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&result).ok()
}

fn try_integrate_log_derivative(expr: &Expr, var: &str) -> Option<Expr> {
  // Decompose into a denominator. Either a literal `a / b` or the canonical
  // Times[..., Power[d, -k]] form.
  let den = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      right,
      ..
    } => (**right).clone(),
    _ => extract_quotient_from_times(expr).map(|(_, d)| d)?,
  };

  // Candidate `g`s: the whole denominator and each of its factors.
  let mut candidates = vec![den.clone()];
  candidates.extend(
    crate::functions::polynomial_ast::collect_multiplicative_factors(&den),
  );

  for g in candidates {
    if is_constant_wrt(&g, var) {
      continue;
    }
    let dg = match differentiate(&g, var) {
      Ok(d) => d,
      Err(_) => continue,
    };
    // ratio = integrand · g / g'
    let ratio = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        expr.clone(),
        g.clone(),
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![dg, Expr::Integer(-1)].into(),
        },
      ]
      .into(),
    };
    let ratio_val = match crate::evaluator::evaluate_expr_to_expr(&ratio) {
      Ok(v) => v,
      Err(_) => continue,
    };
    if matches!(&ratio_val, Expr::Integer(0)) {
      continue;
    }
    let log_g = Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![g].into(),
    };
    // n = -1: integrand = c·g'/g → c·Log[g].
    if is_constant_wrt(&ratio_val, var) {
      return Some(if matches!(&ratio_val, Expr::Integer(1)) {
        log_g
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![ratio_val, log_g].into(),
        }
      });
    }
    // General power: integrand = c·Log[g]^n·g'/g (n ≠ -1) →
    // c·Log[g]^(n+1)/(n+1). Here ratio_val = c·Log[g]^n.
    if let Some((coeff, n)) =
      match_const_times_log_power(&ratio_val, &log_g, var)
      && n != -1
    {
      let new_exp = n + 1;
      let log_pow = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![log_g, Expr::Integer(new_exp)].into(),
      };
      let result = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          coeff,
          log_pow,
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(new_exp)].into(),
          },
        ]
        .into(),
      };
      if let Ok(v) = crate::evaluator::evaluate_expr_to_expr(&result) {
        return Some(v);
      }
    }
  }
  None
}

/// Match `c · Log[g]^n` (with `Log[g]` given as `log_g`): a constant `c` times a
/// single integer power of `log_g`. Returns `(c, n)`. Used to integrate
/// `Log[g]^n · g'/g` via the power rule `Log[g]^(n+1)/(n+1)`.
fn match_const_times_log_power(
  ratio: &Expr,
  log_g: &Expr,
  var: &str,
) -> Option<(Expr, i128)> {
  use crate::syntax::expr_to_string;
  let log_g_str = expr_to_string(log_g);
  let as_log_power = |e: &Expr| -> Option<i128> {
    if expr_to_string(e) == log_g_str {
      return Some(1);
    }
    if let Expr::FunctionCall { name, args } = e
      && name == "Power"
      && args.len() == 2
      && expr_to_string(&args[0]) == log_g_str
      && let Expr::Integer(k) = &args[1]
    {
      return Some(*k);
    }
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } = e
      && expr_to_string(left) == log_g_str
      && let Expr::Integer(k) = right.as_ref()
    {
      return Some(*k);
    }
    None
  };
  let factors =
    crate::functions::polynomial_ast::collect_multiplicative_factors(ratio);
  let mut n: Option<i128> = None;
  let mut consts: Vec<Expr> = Vec::new();
  for f in factors {
    if let Some(k) = as_log_power(&f) {
      if n.is_some() {
        return None; // more than one Log[g] power factor
      }
      n = Some(k);
    } else if is_constant_wrt(&f, var) {
      consts.push(f);
    } else {
      return None; // a var-dependent factor that isn't a power of Log[g]
    }
  }
  let n = n?;
  let coeff = match consts.len() {
    0 => Expr::Integer(1),
    1 => consts.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: consts.into(),
    },
  };
  Some((coeff, n))
}

/// Exact integer square root, or None when `n` is not a perfect square.
fn int_sqrt_exact(n: i128) -> Option<i128> {
  if n < 0 {
    return None;
  }
  let mut x = (n as f64).sqrt() as i128;
  while x > 0 && x * x > n {
    x -= 1;
  }
  while (x + 1) * (x + 1) <= n {
    x += 1;
  }
  if x * x == n { Some(x) } else { None }
}

/// `Sqrt[e]` taken symbolically: perfect-square integers and even powers
/// reduce (`a^2 -> a`, `9 -> 3`), everything else becomes `Power[e, 1/2]`.
fn symbolic_sqrt(e: &Expr) -> Expr {
  let half = || Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
  };
  match e {
    Expr::Integer(n) if *n >= 0 => {
      if let Some(r) = int_sqrt_exact(*n) {
        return Expr::Integer(r);
      }
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Expr::Integer(k) = &args[1]
        && *k > 0
        && k % 2 == 0
      {
        return if *k == 2 {
          args[0].clone()
        } else {
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![args[0].clone(), Expr::Integer(k / 2)].into(),
          }
        };
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Integer(k) = right.as_ref()
        && *k > 0
        && k % 2 == 0
      {
        return if *k == 2 {
          (**left).clone()
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: left.clone(),
            right: Box::new(Expr::Integer(k / 2)),
          }
        };
      }
    }
    _ => {}
  }
  Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![e.clone(), half()].into(),
  }
}

/// True for a constant Wolfram treats as positive: a bare symbol or an even
/// power (a square). Used to decide the sign of a quadratic's constant term.
fn is_manifestly_positive_symbolic(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(_))
    || matches!(e, Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2
          && matches!(&args[1], Expr::Integer(k) if *k > 0 && k % 2 == 0))
    || matches!(e, Expr::BinaryOp { op: BinaryOperator::Power, right, .. }
        if matches!(right.as_ref(), Expr::Integer(k) if *k > 0 && k % 2 == 0))
}

/// Classify the constant term `q` of a quadratic denominator as
/// `(is_negative, Sqrt[|q|])`. Positive constants (positive numbers, squares,
/// bare symbols) give the ArcTan branch; a negated symbolic square (`-a^2`)
/// gives the ArcTanh branch. Returns None for ambiguous signs (e.g. a bare
/// negative number), so those defer to the partial-fraction/Log path.
fn classify_quadratic_const(q: &Expr) -> Option<(bool, Expr)> {
  match q {
    Expr::Integer(n) if *n > 0 => Some((false, symbolic_sqrt(q))),
    Expr::FunctionCall { name, args }
      if name == "Rational"
        && args.len() == 2
        && matches!((&args[0], &args[1]),
          (Expr::Integer(a), Expr::Integer(b)) if *a > 0 && *b > 0) =>
    {
      Some((false, symbolic_sqrt(q)))
    }
    _ if is_manifestly_positive_symbolic(q) => Some((false, symbolic_sqrt(q))),
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(-1))
        && is_manifestly_positive_symbolic(&args[1]) =>
    {
      Some((true, symbolic_sqrt(&args[1])))
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } if is_manifestly_positive_symbolic(operand) => {
      Some((true, symbolic_sqrt(operand)))
    }
    _ => None,
  }
}

/// Coefficient `p` of `var^2` in `term` when `term` is `p * var^2` with `p`
/// constant w.r.t. `var`; None otherwise.
fn coeff_of_var_squared(term: &Expr, var: &str) -> Option<Expr> {
  let is_var_sq = |e: &Expr| {
    matches!(e, Expr::BinaryOp { op: BinaryOperator::Power, left, right }
        if matches!(left.as_ref(), Expr::Identifier(n) if n == var)
          && matches!(right.as_ref(), Expr::Integer(2)))
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2
          && matches!(&args[0], Expr::Identifier(n) if n == var)
          && matches!(&args[1], Expr::Integer(2)))
  };
  if is_var_sq(term) {
    return Some(Expr::Integer(1));
  }
  let factors: Vec<Expr> = match term {
    Expr::FunctionCall { name, args } if name == "Times" => args.to_vec(),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => vec![(**left).clone(), (**right).clone()],
    _ => return None,
  };
  let mut coeff: Vec<Expr> = Vec::new();
  let mut found = false;
  for f in factors {
    if is_var_sq(&f) {
      if found {
        return None;
      }
      found = true;
    } else if is_constant_wrt(&f, var) {
      coeff.push(f);
    } else {
      return None;
    }
  }
  if !found {
    return None;
  }
  Some(match coeff.len() {
    0 => Expr::Integer(1),
    1 => coeff.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: coeff.into(),
    },
  })
}

/// ∫ 1/(p x^2 + q) dx for constants p, q (numeric or symbolic):
///   q > 0:  ArcTan[Sqrt[p/q] x] / Sqrt[p q]
///   q = -a^2 (symbolic): -ArcTanh[Sqrt[p/a^2] x] / Sqrt[p a^2]
/// Numeric `x^2 - c` (c a positive number) is deferred to the Log path.
fn try_integrate_reciprocal_quadratic(expr: &Expr, var: &str) -> Option<Expr> {
  // expr must be 1/base.
  let base = match expr {
    Expr::FunctionCall { name, args }
      if name == "Power"
        && args.len() == 2
        && matches!(&args[1], Expr::Integer(-1)) =>
    {
      args[0].clone()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } if matches!(right.as_ref(), Expr::Integer(-1)) => (**left).clone(),
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(1)) => (**right).clone(),
    _ => return None,
  };

  let terms = crate::functions::polynomial_ast::collect_additive_terms(&base);
  if terms.len() < 2 {
    return None;
  }
  let mut p: Option<Expr> = None;
  let mut q_terms: Vec<Expr> = Vec::new();
  for t in &terms {
    if let Some(coeff) = coeff_of_var_squared(t, var) {
      if p.is_some() {
        return None; // two x^2 terms: not a simple quadratic
      }
      p = Some(coeff);
    } else if is_constant_wrt(t, var) {
      q_terms.push(t.clone());
    } else {
      return None; // a linear term or other var dependence: defer
    }
  }
  let p = p?;
  if q_terms.is_empty() {
    return None;
  }
  let q = match q_terms.len() {
    1 => q_terms.into_iter().next().unwrap(),
    _ => crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".to_string(),
      args: q_terms.into(),
    })
    .ok()?,
  };

  // Leading coefficient must read as positive; the constant term decides the
  // ArcTan vs ArcTanh branch.
  let (p_neg, sqrt_p) = classify_quadratic_const(&p)?;
  if p_neg {
    return None;
  }
  let (is_neg, sqrt_q) = classify_quadratic_const(&q)?;

  let pow_neg1 = |e: Expr| Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![e, Expr::Integer(-1)].into(),
  };
  // arg = sqrt_p * x / sqrt_q
  let arg = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      sqrt_p.clone(),
      Expr::Identifier(var.to_string()),
      pow_neg1(sqrt_q.clone()),
    ]
    .into(),
  };
  // norm = sqrt_p * sqrt_q
  let norm = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![sqrt_p, sqrt_q].into(),
  };
  let func = if is_neg { "ArcTanh" } else { "ArcTan" };
  let inner = Expr::FunctionCall {
    name: func.to_string(),
    args: vec![arg].into(),
  };
  let mut result_factors = vec![inner, pow_neg1(norm)];
  if is_neg {
    result_factors.insert(0, Expr::Integer(-1));
  }
  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: result_factors.into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&result).ok()
}

fn integrate(expr: &Expr, var: &str) -> Option<Expr> {
  // ∫ 1/(p x^2 + q) dx = ArcTan / Sqrt or -ArcTanh / Sqrt for constants p, q
  // (handles symbolic a^2 and leading coefficients that the integer-coefficient
  // rational path misses, e.g. 1/(x^2 + a^2), 1/(9 x^2 + 1)).
  if let Some(result) = try_integrate_reciprocal_quadratic(expr, var) {
    return Some(result);
  }

  // General constant check: ∫ c dy = c*y for any expression c independent of y
  // (handles compound expressions like x^2, Sin[x], etc. when integrating w.r.t. a different variable)
  if is_constant_wrt(expr, var) {
    return Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(expr.clone()),
      right: Box::new(Expr::Identifier(var.to_string())),
    });
  }

  // Closed form for `1/p(x)` with `p` an irreducible-looking degree-≥5
  // univariate polynomial: wolframscript emits a `RootSum[…]` shape.
  if let Some(rs) = try_integrate_one_over_poly_rootsum(expr, var) {
    return Some(rs);
  }

  // Trig quotients (Sin[x]*Sec[x], Cos[x]*Csc[x], Sin[x]^2/Cos[x], …) must
  // run before the generic log-derivative and by-parts rules, which produce
  // non-Wolfram forms for them (e.g. Log[1 - Cos[x]^2]/2, -1 - Log[Cos[x]]).
  {
    let mut num_factors: Vec<&Expr> = Vec::new();
    let mut den_factors: Vec<&Expr> = Vec::new();
    collect_times_factor_refs(expr, &mut num_factors, &mut den_factors);
    if num_factors.len() + den_factors.len() >= 2
      && let Some(result) =
        try_integrate_trig_quotient(&num_factors, &den_factors, var)
    {
      return Some(result);
    }
  }

  // ∫ c·g'(x)/g(x) dx = c·Log[g(x)] for a sub-expression g (covers
  // transcendental g such as Log[x], Sin[x], 1 + E^x).
  if let Some(result) = try_integrate_log_derivative(expr, var) {
    return Some(result);
  }

  // ∫ c·g'(x)/g(x)^n dx = c·g(x)^(1-n)/(1-n) (n >= 2), the u = g(x) substitution.
  if let Some(result) = try_integrate_power_derivative(expr, var) {
    return Some(result);
  }

  match expr {
    // Constant: ∫ c dx = c*x
    Expr::Integer(n) => Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(*n)),
      right: Box::new(Expr::Identifier(var.to_string())),
    }),
    Expr::Real(f) => Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Real(*f)),
      right: Box::new(Expr::Identifier(var.to_string())),
    }),

    // Variable: ∫ x dx = x^2/2, ∫ c dx = c*x
    Expr::Identifier(name) => {
      if name == var {
        Some(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::Identifier(var.to_string())),
            right: Box::new(Expr::Integer(2)),
          }),
          right: Box::new(Expr::Integer(2)),
        })
      } else {
        // Constant * x
        Some(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Identifier(name.clone())),
          right: Box::new(Expr::Identifier(var.to_string())),
        })
      }
    }

    // Binary operations
    Expr::BinaryOp { op, left, right } => {
      use BinaryOperator::*;
      match op {
        Plus => {
          // ∫ (a + b) dx = ∫ a dx + ∫ b dx
          let int_a = integrate(left, var)?;
          let int_b = integrate(right, var)?;
          Some(Expr::BinaryOp {
            op: Plus,
            left: Box::new(int_a),
            right: Box::new(int_b),
          })
        }
        Minus => {
          // ∫ (a - b) dx = ∫ a dx - ∫ b dx
          let int_a = integrate(left, var)?;
          let int_b = integrate(right, var)?;
          Some(Expr::BinaryOp {
            op: Minus,
            left: Box::new(int_a),
            right: Box::new(int_b),
          })
        }
        Times => {
          // c * f(x) where c is constant
          if is_constant_wrt(left, var) {
            let int_b = integrate(right, var)?;
            Some(Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(int_b),
            })
          } else if is_constant_wrt(right, var) {
            let int_a = integrate(left, var)?;
            Some(Expr::BinaryOp {
              op: Times,
              left: right.clone(),
              right: Box::new(int_a),
            })
          } else {
            // Both factors depend on var: try u-substitution first
            if let Some(result) = try_u_substitution_binary(left, right, var) {
              return Some(result);
            }
            // Try trig product
            if let Some(result) =
              try_integrate_sin_cos_product(&[left, right], var)
            {
              return Some(result);
            }
            // Trig quotients like Sin[x]*Tan[x] (= Sin^2/Cos)
            if let Some(result) =
              try_integrate_trig_quotient(&[left, right], &[], var)
            {
              return Some(result);
            }
            // Fall back to integration by parts
            try_integration_by_parts(&[left, right], var)
          }
        }
        Divide => {
          // f(x) / c where c is constant
          if is_constant_wrt(right, var) {
            let int_a = integrate(left, var)?;
            Some(Expr::BinaryOp {
              op: Divide,
              left: Box::new(int_a),
              right: right.clone(),
            })
          } else {
            // If denominator is x^n, rewrite as numerator * x^(-n)
            if let Expr::BinaryOp {
              op: Power,
              left: base,
              right: exp,
            } = right.as_ref()
              && let Expr::Identifier(name) = base.as_ref()
              && name == var
              && is_constant_wrt(exp, var)
            {
              let neg_exp = Expr::UnaryOp {
                op: UnaryOperator::Minus,
                operand: exp.clone(),
              };
              let x_neg_n = Expr::BinaryOp {
                op: Power,
                left: base.clone(),
                right: Box::new(neg_exp),
              };
              let rewritten = Expr::BinaryOp {
                op: Times,
                left: left.clone(),
                right: Box::new(x_neg_n),
              };
              if let Some(result) = integrate(&rewritten, var) {
                return Some(result);
              }
            }
            // ∫ E^(a*x) / (c*x) dx = ExpIntegralEi[a*x] / c
            // ∫ E^(a*x) / x dx = ExpIntegralEi[a*x]
            if let Some(result) = try_match_exp_over_linear(left, right, var) {
              return Some(result);
            }
            // ∫ Sin[a*x]/(c*x) dx = SinIntegral[a*x]/c (and Cos/Sinh/Cosh)
            if let Some(result) = try_match_si_ci_over_linear(left, right, var)
            {
              return Some(result);
            }
            // Trig quotients like Sin[x]^2/Cos[x]
            if let Some(result) =
              try_integrate_trig_quotient(&[left], &[right], var)
            {
              return Some(result);
            }
            // Try rational function integration (partial fractions)
            try_integrate_rational(left, right, var)
          }
        }
        Power => {
          // ∫ 1/Log[x] dx = LogIntegral[x]
          if let Expr::FunctionCall {
            name: lname,
            args: largs,
          } = left.as_ref()
            && lname == "Log"
            && largs.len() == 1
            && matches!(&largs[0], Expr::Identifier(nm) if nm == var)
            && matches!(right.as_ref(), Expr::Integer(-1))
          {
            return Some(Expr::FunctionCall {
              name: "LogIntegral".to_string(),
              args: vec![Expr::Identifier(var.to_string())].into(),
            });
          }
          // ∫ Log[x]^n dx = x · Σ_{k=0}^{n} (-1)^(n-k) (n!/k!) Log[x]^k
          // for integer n ≥ 1 (repeated integration by parts). Building the
          // expanded form reproduces wolframscript's ordering, e.g.
          // ∫ Log[x]^2 dx → 2 x - 2 x Log[x] + x Log[x]^2.
          if let Expr::FunctionCall {
            name: lname,
            args: largs,
          } = left.as_ref()
            && lname == "Log"
            && largs.len() == 1
            && matches!(&largs[0], Expr::Identifier(nm) if nm == var)
            && let Expr::Integer(n) = right.as_ref()
            && *n >= 1
          {
            let n = *n;
            let fact = |m: i128| -> i128 { (1..=m).product() };
            let nf = fact(n);
            let logx = Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![Expr::Identifier(var.to_string())].into(),
            };
            let mut sum: Option<Expr> = None;
            for k in 0..=n {
              let mag = nf / fact(k);
              let coeff = if (n - k) % 2 == 0 { mag } else { -mag };
              let logpow = match k {
                0 => Expr::Integer(1),
                1 => logx.clone(),
                _ => Expr::BinaryOp {
                  op: Power,
                  left: Box::new(logx.clone()),
                  right: Box::new(Expr::Integer(k)),
                },
              };
              let term = Expr::BinaryOp {
                op: Times,
                left: Box::new(Expr::Integer(coeff)),
                right: Box::new(logpow),
              };
              sum = Some(match sum {
                None => term,
                Some(s) => Expr::BinaryOp {
                  op: Plus,
                  left: Box::new(s),
                  right: Box::new(term),
                },
              });
            }
            let prod = Expr::BinaryOp {
              op: Times,
              left: Box::new(Expr::Identifier(var.to_string())),
              right: Box::new(sum.unwrap()),
            };
            if let Ok(expanded) =
              crate::evaluator::evaluate_function_call_ast("Expand", &[prod])
            {
              return Some(expanded);
            }
          }
          // ∫ 1/(a + b*x)^n dx for n ≥ 2 integer → -(a+b*x)^(1-n)/(b*(n-1))
          // Wolfram keeps this factored, matching wolframscript's
          // `Integrate[1/(1+x)^4, x]` → `-1/(3*(1+x)^3)`. Only applies when
          // the base is linear in the variable and n is a negative integer.
          if !matches!(left.as_ref(), Expr::Identifier(name) if name == var)
            && let Expr::Integer(n) = right.as_ref()
            && *n <= -2
            && let Some(slope) = extract_linear_coefficient(left, var)
          {
            let new_exp = Expr::Integer(*n + 1);
            let new_pow = Expr::BinaryOp {
              op: Power,
              left: left.clone(),
              right: Box::new(new_exp.clone()),
            };
            let divisor = simplify(Expr::BinaryOp {
              op: Times,
              left: Box::new(slope),
              right: Box::new(new_exp),
            });
            return Some(Expr::BinaryOp {
              op: Divide,
              left: Box::new(new_pow),
              right: Box::new(divisor),
            });
          }
          // ∫ x^n dx = x^(n+1)/(n+1) where n is constant
          if let Expr::Identifier(name) = left.as_ref()
            && name == var
            && is_constant_wrt(right, var)
          {
            // When the exponent is a Real half-integer (e.g. 3.5), Wolfram
            // promotes the *exponent* of the antiderivative to a Rational
            // (`x^(9/2)`) while keeping the divisor as a Real (`/ 4.5`,
            // displayed as `0.222... * x^(9/2)`). We build a separate
            // `divisor_exp` so the exponent stays Rational while the
            // division uses the Real, matching wolframscript's split.
            let (new_exp, divisor_exp) = if let Expr::Real(f) = right.as_ref()
              && (f * 2.0).fract() == 0.0
            {
              let two_n_plus_2 = ((f + 1.0) * 2.0).round() as i128;
              (
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(two_n_plus_2), Expr::Integer(2)]
                    .into(),
                },
                Expr::Real(f + 1.0),
              )
            } else {
              let s = simplify(Expr::BinaryOp {
                op: Plus,
                left: right.clone(),
                right: Box::new(Expr::Integer(1)),
              });
              (s.clone(), s)
            };
            // Special case: ∫ x^(-1) dx = Log[x]
            if matches!(&new_exp, Expr::Integer(0)) {
              return Some(Expr::FunctionCall {
                name: "Log".to_string(),
                args: vec![Expr::Identifier(var.to_string())].into(),
              });
            }
            let power_expr = Expr::BinaryOp {
              op: Power,
              left: left.clone(),
              right: Box::new(new_exp.clone()),
            };
            // When new_exp is a negative integer, use Wolfram canonical form:
            // x^n / n where n < 0 → Times[Rational[1, n], Power[x, n]]
            // e.g. x^(-2)/(-2) → Times[Rational[-1, 2], Power[x, -2]] → -1/2*1/x^2
            if let Expr::Integer(n) = &new_exp {
              let n = *n;
              if n < 0 {
                let abs_n = -n;
                // Build Rational[-1, abs_n] (= 1/n since n < 0)
                let coeff = if abs_n == 1 {
                  Expr::Integer(-1)
                } else {
                  Expr::FunctionCall {
                    name: "Rational".to_string(),
                    args: vec![Expr::Integer(-1), Expr::Integer(abs_n)].into(),
                  }
                };
                return Some(Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![coeff, power_expr].into(),
                });
              }
            }
            return Some(Expr::BinaryOp {
              op: Divide,
              left: Box::new(power_expr),
              right: Box::new(divisor_exp),
            });
          }
          // ∫ E^x dx = E^x, ∫ E^(a*x) dx = E^(a*x)/a,
          // ∫ E^(-a*x^2) dx = Gaussian integral
          if matches!(left.as_ref(), Expr::Constant(c) if c == "E") {
            let exp_arg = right.as_ref();
            // ∫ E^x dx = E^x
            if let Expr::Identifier(n) = exp_arg
              && n == var
            {
              return Some(expr.clone());
            }
            // ∫ E^(a*x) dx = E^(a*x)/a
            if let Some(coeff) = try_match_linear_arg(exp_arg, var) {
              return Some(make_divided(expr.clone(), coeff));
            }
            // ∫ E^(-a*x^2) dx = Sqrt[Pi/a]/2 * Erf[Sqrt[a]*x]
            if let Some(coeff) = match_neg_a_x_squared(exp_arg, var) {
              return Some(make_gaussian_antiderivative(var, &coeff, "Erf"));
            }
            // ∫ E^(a*x^2) dx = Sqrt[Pi/a]/2 * Erfi[Sqrt[a]*x]  (a > 0)
            if let Some(coeff) = match_pos_a_x_squared(exp_arg, var) {
              return Some(make_gaussian_antiderivative(var, &coeff, "Erfi"));
            }
            // ∫ E^(-1/x^2) dx = x*E^(-1/x^2) + Sqrt[Pi]*Erf[1/x]
            // Standard integration-by-parts result; matches wolframscript
            // (which surfaces the `Erf[1/x]` term).
            if match_neg_inverse_x_squared(exp_arg, var) {
              let var_expr = Expr::Identifier(var.to_string());
              let inv_x = Expr::BinaryOp {
                op: BinaryOperator::Power,
                left: Box::new(var_expr.clone()),
                right: Box::new(Expr::Integer(-1)),
              };
              let term1 = Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(var_expr),
                right: Box::new(expr.clone()),
              };
              let term2 = Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(make_sqrt(Expr::Constant("Pi".to_string()))),
                right: Box::new(Expr::FunctionCall {
                  name: "Erf".to_string(),
                  args: vec![inv_x].into(),
                }),
              };
              return Some(Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(term1),
                right: Box::new(term2),
              });
            }
          }
          // ∫ a^x dx = a^x / Log[a], ∫ a^(c*x) dx = a^(c*x) / (c*Log[a])
          // where a is any constant base (not E, which is handled above)
          if is_constant_wrt(left, var) && !is_constant_wrt(right, var) {
            let exp_arg = right.as_ref();
            let log_a = Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![*left.clone()].into(),
            };
            // ∫ a^x dx = a^x / Log[a]
            if let Expr::Identifier(n) = exp_arg
              && n == var
            {
              return Some(Expr::BinaryOp {
                op: Divide,
                left: Box::new(expr.clone()),
                right: Box::new(log_a),
              });
            }
            // ∫ a^(c*x) dx = a^(c*x) / (c * Log[a])
            if let Some(coeff) = try_match_linear_arg(exp_arg, var) {
              let divisor = simplify(Expr::BinaryOp {
                op: Times,
                left: Box::new(coeff),
                right: Box::new(log_a),
              });
              return Some(Expr::BinaryOp {
                op: Divide,
                left: Box::new(expr.clone()),
                right: Box::new(divisor),
              });
            }
          }
          // ∫ Sin[x]^n dx, ∫ Cos[x]^n dx using Chebyshev expansion
          if let Expr::Integer(n) = right.as_ref()
            && *n >= 2
            && let Some(result) = if *n == 2 {
              try_integrate_trig_squared(left, var)
            } else {
              try_integrate_trig_power(left, *n, var)
            }
          {
            return Some(result);
          }
          // ∫ 1/Trig[x]^|n| dx: rewrite to the reciprocal-trig form
          // (Cos[x]^-2 → Sec[x]^2, Sin[x]^-2 → Csc[x]^2, …) and retry, so the
          // named-reciprocal integrators apply. The rewrite is an exact
          // identity; if the reciprocal form isn't integrable it falls through
          // unchanged.
          if let Expr::Integer(n) = right.as_ref()
            && *n < 0
            && let Expr::FunctionCall {
              name: tname,
              args: targs,
            } = left.as_ref()
            && targs.len() == 1
            && let Some(recip) = reciprocal_trig_name(tname)
          {
            let recip_call = Expr::FunctionCall {
              name: recip.to_string(),
              args: vec![targs[0].clone()].into(),
            };
            // |n| == 1 yields the bare reciprocal (Sec[x]); higher powers keep
            // the Power wrapper (Sec[x]^2).
            let rewritten = if *n == -1 {
              recip_call
            } else {
              Expr::BinaryOp {
                op: Power,
                left: Box::new(recip_call),
                right: Box::new(Expr::Integer(-*n)),
              }
            };
            if let Some(result) = integrate(&rewritten, var) {
              return Some(result);
            }
          }
          // ∫ (a*x + b)^n dx where n >= 3 and the base is linear in var:
          // Use substitution: result = (a*x + b)^(n+1) / ((n+1) * a)
          // For n == 2, Wolfram expands instead, so we skip to the expand path.
          if let Expr::Integer(n) = right.as_ref()
            && *n >= 3
            && !is_constant_wrt(left, var)
            && let Some(a) = extract_linear_coefficient(left, var)
          {
            let n1 = Expr::Integer(*n + 1);
            let base_pow = Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(left.as_ref().clone()),
              right: Box::new(n1.clone()),
            };
            let denom = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![n1, a].into(),
            };
            return Some(Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(base_pow),
              right: Box::new(denom),
            });
          }
          // ∫ f(x)^n dx where n is a positive integer: try expanding (used for n == 2)
          if let Expr::Integer(n) = right.as_ref()
            && *n >= 2
            && !is_constant_wrt(left, var)
          {
            let expanded =
              crate::functions::polynomial_ast::expand_and_combine(expr);
            if !expr_str_eq(&expanded, expr) {
              return integrate(&expanded, var);
            }
          }
          // ∫ (1 - x^2)^(-1/2) dx = ArcSin[x]
          // ∫ (1 + x^2)^(-1/2) dx = ArcSinh[x]
          if is_rational_neg_half(right)
            && !is_constant_wrt(left, var)
            && let Some(result) = try_integrate_inverse_sqrt(left, var)
          {
            return Some(result);
          }
          // base^(-n) where base depends on var: treat as 1/base^n (rational)
          if let Expr::Integer(n) = right.as_ref()
            && *n < 0
            && !is_constant_wrt(left, var)
          {
            let denom = if *n == -1 {
              left.as_ref().clone()
            } else {
              Expr::BinaryOp {
                op: Power,
                left: left.clone(),
                right: Box::new(Expr::Integer(-*n)),
              }
            };
            // Try exp over linear
            if let Some(result) =
              try_match_exp_over_linear(&Expr::Integer(1), &denom, var)
            {
              return Some(result);
            }
            // Try rational function integration
            if let Some(result) =
              try_integrate_rational(&Expr::Integer(1), &denom, var)
            {
              return Some(result);
            }
          }
          None
        }
        _ => None,
      }
    }

    // Function calls
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        // Piecewise[{{val_i, cond_i}, …}, default?] is integrated piece by
        // piece. When the result has exactly two complementary conditions
        // (the second is the negation of the first), wolframscript drops
        // the second and uses its antiderivative as the default —
        // `∫ Piecewise[{{1, x ≤ 0}, {-1, x > 0}}] dx` →
        // `Piecewise[{{x, x ≤ 0}}, -x]`.
        "Piecewise" if !args.is_empty() => {
          let pieces = match &args[0] {
            Expr::List(items) => items,
            _ => return None,
          };
          let mut new_pieces: Vec<Expr> = Vec::with_capacity(pieces.len());
          for item in pieces {
            if let Expr::List(pair) = item
              && pair.len() == 2
            {
              let antideriv = simplify(integrate(&pair[0], var)?);
              new_pieces
                .push(Expr::List(vec![antideriv, pair[1].clone()].into()));
            } else {
              return None;
            }
          }
          let default_expr = if args.len() == 2 {
            simplify(integrate(&args[1], var)?)
          } else {
            Expr::Integer(0)
          };
          // Detect 2-piece complementary conditions: drop the second piece
          // and promote its antiderivative to the default. Recognise
          // `x <= 0` ↔ `x > 0`, `x < 0` ↔ `x >= 0`, and (for symmetry) the
          // mirrored pair, anchored at any constant `c`.
          let is_complement =
            |a: &Expr, b: &Expr| -> bool { conditions_are_complementary(a, b) };
          // The default-0 case behaves like the no-default case: the two
          // complementary conditions partition the axis, so the default
          // is unreachable. (Piecewise evaluation normalizes a missing
          // default to an explicit 0, so both spellings arrive here.)
          let no_reachable_default = args.len() == 1
            || (args.len() == 2 && matches!(&args[1], Expr::Integer(0)));
          if no_reachable_default
            && new_pieces.len() == 2
            && let Expr::List(pair_a) = &new_pieces[0]
            && let Expr::List(pair_b) = &new_pieces[1]
            && pair_a.len() == 2
            && pair_b.len() == 2
            && is_complement(&pair_a[1], &pair_b[1])
          {
            return Some(Expr::FunctionCall {
              name: "Piecewise".to_string(),
              args: vec![
                Expr::List(vec![new_pieces[0].clone()].into()),
                pair_b[0].clone(),
              ]
              .into(),
            });
          }
          Some(Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: if args.len() == 2 {
              vec![Expr::List(new_pieces.into()), default_expr].into()
            } else {
              vec![Expr::List(new_pieces.into())].into()
            },
          })
        }
        "Plus" if args.len() >= 2 => {
          // ∫ (a + b + ...) dx = ∫ a dx + ∫ b dx + ...
          let integrals: Option<Vec<Expr>> =
            args.iter().map(|arg| integrate(arg, var)).collect();
          integrals.map(|ints| Expr::FunctionCall {
            name: "Plus".to_string(),
            args: ints.into(),
          })
        }
        // ∫ RealSign[x] dx = Abs[x] (RealSign is the derivative of Abs
        // for real arguments, away from 0).
        "RealSign" if args.len() == 1 => {
          if matches!(&args[0], Expr::Identifier(n) if n == var) {
            Some(Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![Expr::Identifier(var.to_string())].into(),
            })
          } else {
            None
          }
        }
        // ∫ RealAbs[x] dx = (x * RealAbs[x]) / 2. Differentiating
        // (x*RealAbs[x])/2 with the rule d/dx[RealAbs[x]] = x/RealAbs[x]
        // collapses x^2/RealAbs[x] back to RealAbs[x], so the chosen
        // antiderivative recovers the original integrand (away from 0).
        "RealAbs" if args.len() == 1 => {
          if matches!(&args[0], Expr::Identifier(n) if n == var) {
            let x = Expr::Identifier(var.to_string());
            let real_abs_x = Expr::FunctionCall {
              name: "RealAbs".to_string(),
              args: vec![x.clone()].into(),
            };
            Some(Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(x),
                right: Box::new(real_abs_x),
              }),
              right: Box::new(Expr::Integer(2)),
            })
          } else {
            None
          }
        }
        "Sin" if args.len() == 1 => {
          // ∫ sin(a*x) dx = -cos(a*x)/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let cos_expr = Expr::FunctionCall {
              name: "Cos".to_string(),
              args: args.clone(),
            };
            return Some(make_neg_divided(cos_expr, coeff));
          }
          // ∫ Sin[a*x^2] dx = Sqrt[Pi/2]/Sqrt[a] * FresnelS[Sqrt[a] Sqrt[2/Pi] x]
          if let Some(coeff) = match_pos_a_x_squared(&args[0], var) {
            return Some(make_fresnel_antiderivative(var, &coeff, "FresnelS"));
          }
          None
        }
        // Inverse trig antiderivatives for rational-linear arguments
        // `a*var` (no additive constant). Matches Mathematica's chosen
        // normalization where `q^2` and `p^2 x^2` live inside the Sqrt.
        "ArcSin" | "ArcCos" if args.len() == 1 => {
          let coeff = try_match_linear_arg(&args[0], var)?;
          let (p, q) = decompose_rational_coeff(&coeff)?;
          arcsin_arccos_linear_antideriv(name, args, var, p, q)
        }
        "ArcTan" if args.len() == 1 => {
          if let Expr::Identifier(name) = &args[0]
            && name == var
          {
            let x = Expr::Identifier(var.to_string());
            let x_sq = Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(x.clone()),
              right: Box::new(Expr::Integer(2)),
            };
            let one_plus = Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(x_sq),
            };
            let log_term = Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![one_plus].into(),
            };
            // x ArcTan[x] - Log[1 + x^2] / 2
            return Some(Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(x),
                right: Box::new(Expr::FunctionCall {
                  name: "ArcTan".to_string(),
                  args: args.clone(),
                }),
              }),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Divide,
                left: Box::new(log_term),
                right: Box::new(Expr::Integer(2)),
              }),
            });
          }
          None
        }
        // Integration by parts of the exponential/trig integral functions
        // (argument = x); each is x F[x] plus the elementary ∫ x F'[x]:
        //   ∫ SinIntegral[x]    = x SinIntegral[x]  + Cos[x]
        //   ∫ CosIntegral[x]    = x CosIntegral[x]  - Sin[x]
        //   ∫ SinhIntegral[x]   = x SinhIntegral[x] - Cosh[x]
        //   ∫ CoshIntegral[x]   = x CoshIntegral[x] - Sinh[x]
        //   ∫ ExpIntegralEi[x]  = x ExpIntegralEi[x] - E^x
        "SinIntegral" | "CosIntegral" | "SinhIntegral" | "CoshIntegral"
        | "ExpIntegralEi"
          if args.len() == 1 =>
        {
          if let Expr::Identifier(n) = &args[0]
            && n == var
          {
            let x = Expr::Identifier(var.to_string());
            let call = |head: &str, a: Expr| Expr::FunctionCall {
              name: head.to_string(),
              args: vec![a].into(),
            };
            let neg = |e: Expr| Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(-1)),
              right: Box::new(e),
            };
            let correction = match name.as_str() {
              "SinIntegral" => call("Cos", x.clone()),
              "CosIntegral" => neg(call("Sin", x.clone())),
              "SinhIntegral" => neg(call("Cosh", x.clone())),
              "CoshIntegral" => neg(call("Sinh", x.clone())),
              _ => neg(Expr::BinaryOp {
                op: BinaryOperator::Power,
                left: Box::new(Expr::Constant("E".to_string())),
                right: Box::new(x.clone()),
              }),
            };
            let x_f = Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(x),
              right: Box::new(Expr::FunctionCall {
                name: name.clone(),
                args: args.clone(),
              }),
            };
            return Some(simplify(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(x_f),
              right: Box::new(correction),
            }));
          }
          None
        }
        // Integration by parts of the inverse hyperbolic functions (arg = x):
        //   ∫ ArcSinh[x] = x ArcSinh[x] - Sqrt[1 + x^2]
        //   ∫ ArcCosh[x] = x ArcCosh[x] - Sqrt[-1 + x] Sqrt[1 + x]
        //   ∫ ArcTanh[x] = x ArcTanh[x] + Log[1 - x^2]/2
        //   ∫ ArcCoth[x] = x ArcCoth[x] + Log[1 - x^2]/2
        "ArcSinh" | "ArcCosh" | "ArcTanh" | "ArcCoth" if args.len() == 1 => {
          if let Expr::Identifier(n) = &args[0]
            && n == var
          {
            let x = Expr::Identifier(var.to_string());
            let power = |b: Expr, e: Expr| Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(b),
              right: Box::new(e),
            };
            let times = |a: Expr, b: Expr| Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(a),
              right: Box::new(b),
            };
            let plus = |a: Expr, b: Expr| Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(a),
              right: Box::new(b),
            };
            let minus = |a: Expr, b: Expr| Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(a),
              right: Box::new(b),
            };
            let sqrt = |e: Expr| Expr::FunctionCall {
              name: "Sqrt".to_string(),
              args: vec![e].into(),
            };
            let x_sq = power(x.clone(), Expr::Integer(2));
            let correction = match name.as_str() {
              // -Sqrt[1 + x^2]
              "ArcSinh" => {
                times(Expr::Integer(-1), sqrt(plus(Expr::Integer(1), x_sq)))
              }
              // -Sqrt[-1 + x] Sqrt[1 + x] (wolframscript keeps the split radical)
              "ArcCosh" => times(
                Expr::Integer(-1),
                times(
                  sqrt(plus(Expr::Integer(-1), x.clone())),
                  sqrt(plus(Expr::Integer(1), x.clone())),
                ),
              ),
              // Log[1 - x^2]/2  (ArcTanh and ArcCoth share this)
              _ => Expr::BinaryOp {
                op: BinaryOperator::Divide,
                left: Box::new(Expr::FunctionCall {
                  name: "Log".to_string(),
                  args: vec![minus(Expr::Integer(1), x_sq)].into(),
                }),
                right: Box::new(Expr::Integer(2)),
              },
            };
            let x_f = times(
              x,
              Expr::FunctionCall {
                name: name.clone(),
                args: args.clone(),
              },
            );
            return Some(simplify(plus(x_f, correction)));
          }
          None
        }
        // Integration by parts of the error / Fresnel functions (argument = x):
        //   ∫ Erf[x]      = x Erf[x]      + E^(-x^2)/Sqrt[Pi]
        //   ∫ Erfc[x]     = x Erfc[x]     - E^(-x^2)/Sqrt[Pi]
        //   ∫ Erfi[x]     = x Erfi[x]     - E^(x^2)/Sqrt[Pi]
        //   ∫ FresnelS[x] = x FresnelS[x] + Cos[(Pi x^2)/2]/Pi
        //   ∫ FresnelC[x] = x FresnelC[x] - Sin[(Pi x^2)/2]/Pi
        "Erf" | "Erfc" | "Erfi" | "FresnelS" | "FresnelC"
          if args.len() == 1 =>
        {
          if let Expr::Identifier(n) = &args[0]
            && n == var
          {
            let x = Expr::Identifier(var.to_string());
            let power = |b: Expr, e: Expr| Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(b),
              right: Box::new(e),
            };
            let times = |a: Expr, b: Expr| Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(a),
              right: Box::new(b),
            };
            let divide = |a: Expr, b: Expr| Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(a),
              right: Box::new(b),
            };
            let x_sq = power(x.clone(), Expr::Integer(2));
            let sqrt_pi = Expr::FunctionCall {
              name: "Sqrt".to_string(),
              args: vec![Expr::Constant("Pi".to_string())].into(),
            };
            let neg = |e: Expr| times(Expr::Integer(-1), e);
            // exp(-x^2)/Sqrt[Pi] used by Erf/Erfc.
            let gauss = || {
              divide(
                Expr::Integer(1),
                times(
                  power(Expr::Constant("E".to_string()), x_sq.clone()),
                  sqrt_pi.clone(),
                ),
              )
            };
            // (Pi x^2)/2 argument for the Fresnel corrections.
            let fresnel_arg = || {
              divide(
                times(Expr::Constant("Pi".to_string()), x_sq.clone()),
                Expr::Integer(2),
              )
            };
            let correction = match name.as_str() {
              "Erf" => gauss(),
              "Erfc" => neg(gauss()),
              "Erfi" => neg(divide(
                power(Expr::Constant("E".to_string()), x_sq.clone()),
                sqrt_pi.clone(),
              )),
              "FresnelS" => divide(
                Expr::FunctionCall {
                  name: "Cos".to_string(),
                  args: vec![fresnel_arg()].into(),
                },
                Expr::Constant("Pi".to_string()),
              ),
              _ => neg(divide(
                Expr::FunctionCall {
                  name: "Sin".to_string(),
                  args: vec![fresnel_arg()].into(),
                },
                Expr::Constant("Pi".to_string()),
              )),
            };
            let x_f = times(
              x,
              Expr::FunctionCall {
                name: name.clone(),
                args: args.clone(),
              },
            );
            return Some(simplify(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(x_f),
              right: Box::new(correction),
            }));
          }
          None
        }
        "Cos" if args.len() == 1 => {
          // ∫ cos(a*x) dx = sin(a*x)/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let sin_expr = Expr::FunctionCall {
              name: "Sin".to_string(),
              args: args.clone(),
            };
            return Some(make_divided(sin_expr, coeff));
          }
          // ∫ Cos[a*x^2] dx = Sqrt[Pi/2]/Sqrt[a] * FresnelC[Sqrt[a] Sqrt[2/Pi] x]
          if let Some(coeff) = match_pos_a_x_squared(&args[0], var) {
            return Some(make_fresnel_antiderivative(var, &coeff, "FresnelC"));
          }
          None
        }
        "Exp" if args.len() == 1 => {
          // ∫ e^x dx = e^x
          if let Expr::Identifier(n) = &args[0]
            && n == var
          {
            return Some(Expr::FunctionCall {
              name: "Exp".to_string(),
              args: args.clone(),
            });
          }
          // ∫ e^(a*x) dx = e^(a*x)/a  (linear argument)
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let exp_expr = Expr::FunctionCall {
              name: "Exp".to_string(),
              args: args.clone(),
            };
            return Some(make_divided(exp_expr, coeff));
          }
          // ∫ Exp[-a*x^2] dx = Sqrt[Pi/a]/2 * Erf[Sqrt[a]*x]
          // (when a=1: Sqrt[Pi]/2 * Erf[x])
          if let Some(coeff) = match_neg_a_x_squared(&args[0], var) {
            return Some(make_gaussian_antiderivative(var, &coeff, "Erf"));
          }
          // ∫ Exp[a*x^2] dx = Sqrt[Pi/a]/2 * Erfi[Sqrt[a]*x]  (a > 0)
          if let Some(coeff) = match_pos_a_x_squared(&args[0], var) {
            return Some(make_gaussian_antiderivative(var, &coeff, "Erfi"));
          }
          None
        }
        "Sinh" if args.len() == 1 => {
          // ∫ sinh(a*x) dx = cosh(a*x)/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let cosh_expr = Expr::FunctionCall {
              name: "Cosh".to_string(),
              args: args.clone(),
            };
            return Some(make_divided(cosh_expr, coeff));
          }
          None
        }
        "Cosh" if args.len() == 1 => {
          // ∫ cosh(a*x) dx = sinh(a*x)/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let sinh_expr = Expr::FunctionCall {
              name: "Sinh".to_string(),
              args: args.clone(),
            };
            return Some(make_divided(sinh_expr, coeff));
          }
          None
        }
        "Tan" if args.len() == 1 => {
          // ∫ tan(a*x) dx = -Log[Cos[a*x]]/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let cos_expr = Expr::FunctionCall {
              name: "Cos".to_string(),
              args: args.clone(),
            };
            let log_cos = Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![cos_expr].into(),
            };
            return Some(make_neg_divided(log_cos, coeff));
          }
          None
        }
        "Cot" if args.len() == 1 => {
          // ∫ cot(a*x) dx = Log[Sin[a*x]]/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let sin_expr = Expr::FunctionCall {
              name: "Sin".to_string(),
              args: args.clone(),
            };
            let log_sin = Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![sin_expr].into(),
            };
            return Some(make_divided(log_sin, coeff));
          }
          None
        }
        "Tanh" if args.len() == 1 => {
          // ∫ tanh(a*x) dx = Log[Cosh[a*x]]/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let cosh_expr = Expr::FunctionCall {
              name: "Cosh".to_string(),
              args: args.clone(),
            };
            let log_cosh = Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![cosh_expr].into(),
            };
            return Some(make_divided(log_cosh, coeff));
          }
          None
        }
        "Coth" if args.len() == 1 => {
          // ∫ coth(a*x) dx = Log[Sinh[a*x]]/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let sinh_expr = Expr::FunctionCall {
              name: "Sinh".to_string(),
              args: args.clone(),
            };
            let log_sinh = Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![sinh_expr].into(),
            };
            return Some(make_divided(log_sinh, coeff));
          }
          None
        }
        "Sec" if args.len() == 1 => {
          // ∫ sec(a*x) dx = ArcCoth[Sin[a*x]]/a (wolframscript's form)
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let sin_expr = Expr::FunctionCall {
              name: "Sin".to_string(),
              args: args.clone(),
            };
            let arccoth = Expr::FunctionCall {
              name: "ArcCoth".to_string(),
              args: vec![sin_expr].into(),
            };
            return Some(make_divided(arccoth, coeff));
          }
          None
        }
        "Csc" if args.len() == 1 => {
          // ∫ csc(a*x) dx = -ArcTanh[Cos[a*x]]/a (wolframscript's form)
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let cos_expr = Expr::FunctionCall {
              name: "Cos".to_string(),
              args: args.clone(),
            };
            let arctanh = Expr::FunctionCall {
              name: "ArcTanh".to_string(),
              args: vec![cos_expr].into(),
            };
            return Some(make_neg_divided(arctanh, coeff));
          }
          None
        }
        "Sech" if args.len() == 1 => {
          // ∫ sech(a*x) dx = -ArcCot[Sinh[a*x]]/a (wolframscript's form)
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let sinh_expr = Expr::FunctionCall {
              name: "Sinh".to_string(),
              args: args.clone(),
            };
            let arccot = Expr::FunctionCall {
              name: "ArcCot".to_string(),
              args: vec![sinh_expr].into(),
            };
            return Some(make_neg_divided(arccot, coeff));
          }
          None
        }
        "Csch" if args.len() == 1 => {
          // ∫ csch(a*x) dx = -ArcTanh[Cosh[a*x]]/a (wolframscript's form)
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let cosh_expr = Expr::FunctionCall {
              name: "Cosh".to_string(),
              args: args.clone(),
            };
            let arctanh = Expr::FunctionCall {
              name: "ArcTanh".to_string(),
              args: vec![cosh_expr].into(),
            };
            return Some(make_neg_divided(arctanh, coeff));
          }
          None
        }
        "Log" if args.len() == 1 => {
          // ∫ Log[x] dx = -x + x*Log[x]
          if let Expr::Identifier(name) = &args[0]
            && name == var
          {
            let x = Expr::Identifier(var.to_string());
            return Some(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::UnaryOp {
                op: UnaryOperator::Minus,
                operand: Box::new(x.clone()),
              }),
              right: Box::new(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(x),
                right: Box::new(Expr::FunctionCall {
                  name: "Log".to_string(),
                  args: args.clone(),
                }),
              }),
            });
          }
          // ∫ Log[Log[u]] dx with u = a*x + b linear in x:
          //   x*Log[Log[u]] - LogIntegral[u]/a        (b == 0)
          //   (u*Log[Log[u]])/a - LogIntegral[u]/a    (b != 0)
          if let Expr::FunctionCall {
            name: inner_name,
            args: inner_args,
          } = &args[0]
            && inner_name == "Log"
            && inner_args.len() == 1
          {
            let u = inner_args[0].clone();
            if let Some(coeff) = extract_linear_coefficient(&u, var) {
              let log_log = Expr::FunctionCall {
                name: "Log".to_string(),
                args: args.clone(),
              };
              let first = if try_match_linear_arg(&u, var).is_some() {
                // u = a*x, so u/a = x
                Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(Expr::Identifier(var.to_string())),
                  right: Box::new(log_log),
                }
              } else {
                make_divided(
                  Expr::BinaryOp {
                    op: BinaryOperator::Times,
                    left: Box::new(u.clone()),
                    right: Box::new(log_log),
                  },
                  coeff.clone(),
                )
              };
              let li = Expr::FunctionCall {
                name: "LogIntegral".to_string(),
                args: vec![u].into(),
              };
              return Some(Expr::BinaryOp {
                op: BinaryOperator::Plus,
                left: Box::new(first),
                right: Box::new(make_neg_divided(li, coeff)),
              });
            }
          }
          None
        }
        "Times" => {
          // ∫ (c1 * c2 * ... * f(x)) dx = c1 * c2 * ... * ∫ f(x) dx
          let (const_factors, var_factors): (Vec<_>, Vec<_>) =
            args.iter().partition(|a| is_constant_wrt(a, var));
          if var_factors.len() == 1 {
            let int_var = integrate(var_factors[0], var)?;
            let const_expr = if const_factors.is_empty() {
              return Some(int_var);
            } else if const_factors.len() == 1 {
              const_factors[0].clone()
            } else {
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: const_factors.into_iter().cloned().collect(),
              }
            };
            Some(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(const_expr),
              right: Box::new(int_var),
            })
          } else if var_factors.is_empty() {
            // All constant: ∫ c dx = c*x
            let const_expr = Expr::FunctionCall {
              name: "Times".to_string(),
              args: args.clone(),
            };
            Some(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(const_expr),
              right: Box::new(Expr::Identifier(var.to_string())),
            })
          } else {
            // Direct-derivative products: Sec*Tan, Csc*Cot, Sech*Tanh,
            // Csch*Coth (carrying through any constant factors).
            if let Some(result) =
              try_integrate_derivative_product(&var_factors, var)
            {
              if const_factors.is_empty() {
                return Some(result);
              }
              let const_expr = if const_factors.len() == 1 {
                const_factors[0].clone()
              } else {
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: const_factors.iter().map(|e| (*e).clone()).collect(),
                }
              };
              return Some(Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(const_expr),
                right: Box::new(result),
              });
            }
            // Multiple variable-dependent factors: check for fraction form
            // Times[..., Power[den, -1]] → treat as numerator / denominator
            let mut num_var_factors: Vec<Expr> = Vec::new();
            let mut den_factors: Vec<Expr> = Vec::new();
            for vf in &var_factors {
              // Extract base and negative exponent from Power[base, -n]
              let neg_power = match vf {
                Expr::FunctionCall {
                  name: pname,
                  args: pargs,
                } if pname == "Power" && pargs.len() == 2 => {
                  if let Expr::Integer(n) = &pargs[1] {
                    if *n < 0 {
                      Some((pargs[0].clone(), *n))
                    } else {
                      None
                    }
                  } else {
                    None
                  }
                }
                Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left,
                  right,
                } => {
                  if let Expr::Integer(n) = right.as_ref() {
                    if *n < 0 {
                      Some((*left.clone(), *n))
                    } else {
                      None
                    }
                  } else {
                    None
                  }
                }
                _ => None,
              };
              if let Some((base, neg_exp)) = neg_power {
                if neg_exp == -1 {
                  den_factors.push(base);
                } else {
                  den_factors.push(Expr::BinaryOp {
                    op: BinaryOperator::Power,
                    left: Box::new(base),
                    right: Box::new(Expr::Integer(-neg_exp)),
                  });
                }
              } else {
                num_var_factors.push((*vf).clone());
              }
            }
            if !den_factors.is_empty() {
              let numerator = if num_var_factors.is_empty() {
                Expr::Integer(1)
              } else if num_var_factors.len() == 1 {
                num_var_factors.remove(0)
              } else {
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: num_var_factors.into(),
                }
              };
              let denominator = if den_factors.len() == 1 {
                den_factors.remove(0)
              } else {
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: den_factors.into(),
                }
              };
              // Helper to multiply constant factors back to a result
              let apply_const = |result: Expr| -> Expr {
                if const_factors.is_empty() {
                  result
                } else {
                  let const_expr = if const_factors.len() == 1 {
                    const_factors[0].clone()
                  } else {
                    Expr::FunctionCall {
                      name: "Times".to_string(),
                      args: const_factors
                        .iter()
                        .map(|e| (*e).clone())
                        .collect(),
                    }
                  };
                  Expr::BinaryOp {
                    op: BinaryOperator::Times,
                    left: Box::new(const_expr),
                    right: Box::new(result),
                  }
                }
              };
              // Try the same logic as the Divide arm
              if is_constant_wrt(&denominator, var)
                && let Some(int_num) = integrate(&numerator, var)
              {
                return Some(apply_const(
                  crate::functions::math_ast::make_divide(int_num, denominator),
                ));
              }
              // Try exp over linear: ∫ E^(a*x) / (c*x) dx
              if let Some(result) =
                try_match_exp_over_linear(&numerator, &denominator, var)
              {
                return Some(apply_const(result));
              }
              // ∫ Sin[a*x]/(c*x) dx = SinIntegral[a*x]/c (and Cos/Sinh/Cosh)
              if let Some(result) =
                try_match_si_ci_over_linear(&numerator, &denominator, var)
              {
                return Some(apply_const(result));
              }
              // Try rational function integration
              if let Some(result) =
                try_integrate_rational(&numerator, &denominator, var)
              {
                return Some(apply_const(result));
              }
            }
            // Try trig product: Sin[f]^m * Cos[f]^n
            let var_refs: Vec<&Expr> = var_factors.to_vec();
            // ∫ E^(a x) Sin[b x] dx / ∫ E^(a x) Cos[b x] dx
            if let Some(et_result) =
              try_integrate_exp_trig_product(&var_refs, var)
            {
              if const_factors.is_empty() {
                return Some(et_result);
              } else {
                let const_expr = if const_factors.len() == 1 {
                  const_factors[0].clone()
                } else {
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: const_factors.into_iter().cloned().collect(),
                  }
                };
                return Some(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(const_expr),
                  right: Box::new(et_result),
                });
              }
            }
            if let Some(trig_result) =
              try_integrate_sin_cos_product(&var_refs, var)
                .or_else(|| try_integrate_trig_quotient(&var_refs, &[], var))
            {
              if const_factors.is_empty() {
                return Some(trig_result);
              } else {
                let const_expr = if const_factors.len() == 1 {
                  const_factors[0].clone()
                } else {
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: const_factors.into_iter().cloned().collect(),
                  }
                };
                return Some(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(const_expr),
                  right: Box::new(trig_result),
                });
              }
            }
            // Try u-substitution for pairs of variable-dependent factors
            if var_factors.len() == 2
              && let Some(usub_result) =
                try_u_substitution_binary(var_factors[0], var_factors[1], var)
            {
              let result = if const_factors.is_empty() {
                usub_result
              } else {
                let const_expr = if const_factors.len() == 1 {
                  const_factors[0].clone()
                } else {
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: const_factors.into_iter().cloned().collect(),
                  }
                };
                Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(const_expr),
                  right: Box::new(usub_result),
                }
              };
              return Some(result);
            }
            // Fall through to integration by parts
            if let Some(ibp_result) = try_integration_by_parts(&var_refs, var) {
              // Multiply back the constant factors
              if const_factors.is_empty() {
                Some(ibp_result)
              } else {
                let const_expr = if const_factors.len() == 1 {
                  const_factors[0].clone()
                } else {
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: const_factors.into_iter().cloned().collect(),
                  }
                };
                Some(Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(const_expr),
                  right: Box::new(ibp_result),
                })
              }
            } else {
              None
            }
          }
        }
        // Power[base, exp] as FunctionCall → normalize to BinaryOp and recurse
        "Power" if args.len() == 2 => {
          let as_binop = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(args[0].clone()),
            right: Box::new(args[1].clone()),
          };
          integrate(&as_binop, var)
        }
        _ => None,
      }
    }

    // Unary minus
    Expr::UnaryOp { op, operand } => {
      if matches!(op, UnaryOperator::Minus) {
        let int = integrate(operand, var)?;
        Some(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(int),
        })
      } else {
        None
      }
    }

    _ => None,
  }
}

/// Simplify an expression
pub fn simplify(mut expr: Expr) -> Expr {
  match &mut expr {
    Expr::BinaryOp { op, left, right } => {
      let op = *op;
      let left = simplify(*std::mem::replace(left, Box::new(Expr::Integer(0))));
      let right =
        simplify(*std::mem::replace(right, Box::new(Expr::Integer(0))));

      use BinaryOperator::*;
      match (&op, &left, &right) {
        // 0 + x = x
        (Plus, Expr::Integer(0), _) => return right,
        // x + 0 = x
        (Plus, _, Expr::Integer(0)) => return left,
        // 0 * x = 0
        (Times, Expr::Integer(0), _) | (Times, _, Expr::Integer(0)) => {
          return Expr::Integer(0);
        }
        // 1 * x = x
        (Times, Expr::Integer(1), _) => return right,
        // x * 1 = x
        (Times, _, Expr::Integer(1)) => return left,
        // x - 0 = x
        (Minus, _, Expr::Integer(0)) => return left,
        // 0 - n = -n  (for integers)
        (Minus, Expr::Integer(0), Expr::Integer(n)) => {
          return Expr::Integer(-n);
        }
        // 0 - (-x) = x
        (
          Minus,
          Expr::Integer(0),
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          },
        ) => {
          return *operand.clone();
        }
        // 0 - x = -x  (general)
        (Minus, Expr::Integer(0), _) => {
          return Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand: Box::new(right),
          };
        }
        // x / 1 = x
        (Divide, _, Expr::Integer(1)) => return left,
        // x^0 = 1
        (Power, _, Expr::Integer(0)) => return Expr::Integer(1),
        // x^1 = x
        (Power, _, Expr::Integer(1)) => return left,
        // 0^n = 0 (for n > 0)
        (Power, Expr::Integer(0), Expr::Integer(n)) if *n > 0 => {
          return Expr::Integer(0);
        }
        // 1^n = 1
        (Power, Expr::Integer(1), _) => return Expr::Integer(1),
        // Numeric simplification
        (Plus, Expr::Integer(a), Expr::Integer(b)) => {
          return Expr::Integer(a + b);
        }
        (Minus, Expr::Integer(a), Expr::Integer(b)) => {
          return Expr::Integer(a - b);
        }
        (Times, Expr::Integer(a), Expr::Integer(b)) => {
          return Expr::Integer(a * b);
        }
        _ => {}
      }

      // For Power, delegate to power_two for proper expansion (e.g. (3*x)^2 → 9*x^2)
      // Only for non-negative exponents to preserve canonical form
      if matches!(op, Power)
        && matches!(&right, Expr::Integer(n) if *n >= 0)
        && let Ok(result) = crate::functions::math_ast::power_two(&left, &right)
      {
        return result;
      }
      // For Times, delegate to times_ast for proper flattening and sorting
      if matches!(op, Times)
        && let Ok(result) =
          crate::functions::math_ast::times_ast(&[left.clone(), right.clone()])
      {
        return result;
      }
      // For Plus, delegate to plus_ast for proper sorting
      if matches!(op, Plus)
        && let Ok(result) =
          crate::functions::math_ast::plus_ast(&[left.clone(), right.clone()])
      {
        return result;
      }

      Expr::BinaryOp {
        op,
        left: Box::new(left),
        right: Box::new(right),
      }
    }
    Expr::UnaryOp { op, operand } => {
      let op = *op;
      let operand =
        simplify(*std::mem::replace(operand, Box::new(Expr::Integer(0))));
      if matches!(&op, UnaryOperator::Minus) {
        if let Expr::Integer(0) = operand {
          return Expr::Integer(0);
        }
        if let Expr::Integer(n) = operand {
          return Expr::Integer(-n);
        }
        // --x → x (double negation)
        if let Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: ref inner,
        } = operand
        {
          return *inner.clone();
        }
      }
      Expr::UnaryOp {
        op,
        operand: Box::new(operand),
      }
    }
    Expr::FunctionCall { name, args } => {
      let name = std::mem::take(name);
      let args: Vec<Expr> =
        std::mem::take(args).into_iter().map(simplify).collect();

      // Delegate Power[base, exp] to power_two for proper expansion
      // Only for non-negative exponents to avoid distributing (a*b)^(-n)
      // which changes canonical form (e.g. Sqrt[-1+x]*Sqrt[1+x] factor ordering)
      if name == "Power" && args.len() == 2 {
        let is_non_neg = matches!(&args[1], Expr::Integer(n) if *n >= 0);
        if is_non_neg
          && let Ok(result) =
            crate::functions::math_ast::power_two(&args[0], &args[1])
        {
          return result;
        }
      }
      // Convert Exp[x] → E^x
      if name == "Exp"
        && args.len() == 1
        && let Ok(result) = crate::functions::math_ast::power_two(
          &Expr::Constant("E".to_string()),
          &args[0],
        )
      {
        return result;
      }
      // Delegate Sqrt to handle Sqrt[0] → 0, Sqrt[1] → 1, etc.
      if name == "Sqrt" && args.len() == 1 {
        let canonical = make_sqrt(args[0].clone());
        if let Some(inner) = is_sqrt(&canonical) {
          if matches!(inner, Expr::Integer(0)) {
            return Expr::Integer(0);
          }
          if matches!(inner, Expr::Integer(1)) {
            return Expr::Integer(1);
          }
        }
      }

      Expr::FunctionCall {
        name,
        args: args.into(),
      }
    }
    _ => expr,
  }
}

/// Extract base and exponent from Power expressions (both BinaryOp and FunctionCall forms)
fn extract_power(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => Some((*left.clone(), *right.clone())),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      Some((args[0].clone(), args[1].clone()))
    }
    _ => None,
  }
}

/// True if `expr` contains a super-polynomially growing function (Factorial,
/// Gamma, …) applied to a var-dependent argument. Substituting a large probe
/// value (1e6) into such an expression would materialize an astronomically
/// large number, so the numeric limit heuristics must skip it and rely on the
/// structural divergence analysis instead.
fn contains_explosive_of_var(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      (matches!(
        name.as_str(),
        "Factorial" | "Factorial2" | "Gamma" | "Hyperfactorial" | "BarnesG"
      ) && args.iter().any(|a| !is_constant_wrt(a, var)))
        || args.iter().any(|a| contains_explosive_of_var(a, var))
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_explosive_of_var(left, var)
        || contains_explosive_of_var(right, var)
    }
    Expr::UnaryOp { operand, .. } => contains_explosive_of_var(operand, var),
    Expr::List(items) => {
      items.iter().any(|a| contains_explosive_of_var(a, var))
    }
    _ => false,
  }
}

/// True if `expr` contains a `HarmonicNumber[…]` whose argument depends on
/// `var`. Used both to skip such expressions during cheap probes and to drive
/// the asymptotic rewrite below.
fn contains_harmonic_of_var(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      (name.as_str() == "HarmonicNumber"
        && args.iter().any(|a| !is_constant_wrt(a, var)))
        || args.iter().any(|a| contains_harmonic_of_var(a, var))
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_harmonic_of_var(left, var)
        || contains_harmonic_of_var(right, var)
    }
    Expr::UnaryOp { operand, .. } => contains_harmonic_of_var(operand, var),
    Expr::List(items) => items.iter().any(|a| contains_harmonic_of_var(a, var)),
    _ => false,
  }
}

/// True when the argument `g` of a `HarmonicNumber[g]` tends to +Infinity as
/// `var -> +Infinity` (so its asymptotic expansion is valid). Confirmed with a
/// cheap two-point probe — `g` is a plain expression here (no HarmonicNumber /
/// Factorial of `var`, which are excluded), so probing it is safe and fast.
fn harmonic_arg_tends_to_pos_inf(g: &Expr, var: &str) -> bool {
  if is_constant_wrt(g, var)
    || contains_explosive_of_var(g, var)
    || contains_harmonic_of_var(g, var)
  {
    return false;
  }
  let probe = |v: i128| -> Option<f64> {
    let s = crate::syntax::substitute_variable(g, var, &Expr::Integer(v));
    crate::evaluator::evaluate_expr_to_expr(&s)
      .ok()
      .and_then(|e| crate::functions::math_ast::try_eval_to_f64(&e))
  };
  matches!((probe(1000), probe(1_000_000)), (Some(a), Some(b)) if b > a && b > 1000.0)
}

/// The asymptotic expansion of `HarmonicNumber[g]` for large `g`:
///   Log[g] + EulerGamma + 1/(2 g) - 1/(12 g^2) + 1/(120 g^4).
/// Enough Euler–Maclaurin terms to resolve the common `n`-, `n^2`-scaled
/// limits exactly (matching wolframscript).
fn harmonic_asymptotic(g: &Expr) -> Expr {
  let pow = |e: i128| Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![g.clone(), Expr::Integer(e)].into(),
  };
  let rat = |n: i128, d: i128| Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
  };
  let term = |n: i128, d: i128, p: i128| Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![rat(n, d), pow(p)].into(),
  };
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![g.clone()].into(),
      },
      Expr::Identifier("EulerGamma".to_string()),
      term(1, 2, -1),
      term(-1, 12, -2),
      term(1, 120, -4),
    ]
    .into(),
  }
}

/// Replace every `HarmonicNumber[g]` (1-arg, `g -> +Infinity`) in `expr` by its
/// asymptotic expansion so a limit at +Infinity resolves symbolically — and,
/// crucially, so the numeric fallback never substitutes a huge probe into
/// `HarmonicNumber`, whose exact rational value is astronomically expensive to
/// compute (a hang). Returns Some(rewritten) iff a replacement was made.
fn rewrite_harmonic_asymptotic(expr: &Expr, var: &str) -> Option<Expr> {
  match expr {
    Expr::FunctionCall { name, args }
      if name.as_str() == "HarmonicNumber"
        && args.len() == 1
        && harmonic_arg_tends_to_pos_inf(&args[0], var) =>
    {
      Some(harmonic_asymptotic(&args[0]))
    }
    Expr::FunctionCall { name, args } => {
      let mut changed = false;
      let new_args: Vec<Expr> = args
        .iter()
        .map(|a| match rewrite_harmonic_asymptotic(a, var) {
          Some(n) => {
            changed = true;
            n
          }
          None => a.clone(),
        })
        .collect();
      changed.then(|| Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      })
    }
    Expr::BinaryOp { op, left, right } => {
      let nl = rewrite_harmonic_asymptotic(left, var);
      let nr = rewrite_harmonic_asymptotic(right, var);
      (nl.is_some() || nr.is_some()).then(|| Expr::BinaryOp {
        op: *op,
        left: Box::new(nl.unwrap_or_else(|| (**left).clone())),
        right: Box::new(nr.unwrap_or_else(|| (**right).clone())),
      })
    }
    Expr::UnaryOp { op, operand } => rewrite_harmonic_asymptotic(operand, var)
      .map(|n| Expr::UnaryOp {
        op: *op,
        operand: Box::new(n),
      }),
    Expr::List(items) => {
      let mut changed = false;
      let new: Vec<Expr> = items
        .iter()
        .map(|a| match rewrite_harmonic_asymptotic(a, var) {
          Some(n) => {
            changed = true;
            n
          }
          None => a.clone(),
        })
        .collect();
      changed.then(|| Expr::List(new.into()))
    }
    _ => None,
  }
}

/// Check if an expression approaches 1 when var -> Infinity
fn eval_at_infinity_is_one(expr: &Expr, var: &str) -> bool {
  if contains_explosive_of_var(expr, var) {
    return false;
  }
  // Substitute a large value and check if close to 1
  let subst =
    crate::syntax::substitute_variable(expr, var, &Expr::Integer(1_000_000));
  if let Ok(val) = crate::evaluator::evaluate_expr_to_expr(&subst)
    && let Some(f) = crate::functions::math_ast::try_eval_to_f64(&val)
  {
    return (f - 1.0).abs() < 0.01;
  }
  // Symbolic fallback for a base with free parameters (e.g. 1 + a/n): with
  // var -> Infinity the var-dependent terms vanish and the base is exactly 1.
  let subst_inf = crate::syntax::substitute_variable(
    expr,
    var,
    &Expr::Identifier("Infinity".to_string()),
  );
  matches!(
    crate::evaluator::evaluate_expr_to_expr(&subst_inf),
    Ok(Expr::Integer(1))
  )
}

/// Check if an expression diverges to infinity when var -> Infinity
fn eval_at_infinity_diverges(expr: &Expr, var: &str) -> Option<bool> {
  if contains_explosive_of_var(expr, var) {
    return None;
  }
  let subst =
    crate::syntax::substitute_variable(expr, var, &Expr::Integer(1_000_000));
  if let Ok(val) = crate::evaluator::evaluate_expr_to_expr(&subst)
    && let Some(f) = crate::functions::math_ast::try_eval_to_f64(&val)
  {
    if f > 1e5 {
      return Some(true); // positive infinity
    }
    if f < -1e5 {
      return Some(false); // negative infinity (returns Some(false) for sign)
    }
  }
  None
}

/// Classify a sum of `c_i * trig[poly_i(var)]` summands as
/// `Interval[{-bound, bound}]` (if any two arguments have different
/// polynomial degrees in `var`) or `Indeterminate` (same polynomial
/// degree everywhere). Returns `None` if any summand isn't of that
/// shape — the caller should fall back to its other heuristics.
fn limit_bounded_oscillating_sum(expr: &Expr, var_name: &str) -> Option<Expr> {
  // Collect additive terms.
  let terms: Vec<Expr> = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let mut v =
        crate::functions::polynomial_ast::collect_additive_terms(left);
      v.extend(crate::functions::polynomial_ast::collect_additive_terms(
        right,
      ));
      v
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut v = Vec::new();
      for a in args.iter() {
        v.extend(crate::functions::polynomial_ast::collect_additive_terms(a));
      }
      v
    }
    _ => return None,
  };
  // At least two terms needed.
  if terms.len() < 2 {
    return None;
  }

  // For each term, extract (|coefficient|, argument-degree).
  // Sin/Cos accepts arbitrary polynomial in `var_name`. A trig argument
  // that doesn't depend on `var_name` is treated as constant (term
  // doesn't oscillate) and disqualifies the heuristic.
  let mut coeffs_abs: Vec<i128> = Vec::with_capacity(terms.len());
  let mut degrees: Vec<i128> = Vec::with_capacity(terms.len());
  for t in &terms {
    let (coeff_expr, body) = split_constant_factor(t, var_name)?;
    let trig_arg = match &body {
      Expr::FunctionCall { name, args }
        if (name == "Sin" || name == "Cos") && args.len() == 1 =>
      {
        &args[0]
      }
      _ => return None,
    };
    if !contains_var(trig_arg, var_name) {
      return None;
    }
    let deg =
      crate::functions::polynomial_ast::max_power_int(trig_arg, var_name)?;
    let abs_coeff = expr_to_abs_int(&coeff_expr)?;
    coeffs_abs.push(abs_coeff);
    degrees.push(deg);
  }

  // All same degree → Indeterminate. Otherwise → Interval[{-bound, bound}].
  let all_same_degree = degrees.windows(2).all(|w| w[0] == w[1]);
  if all_same_degree {
    return Some(Expr::Identifier("Indeterminate".to_string()));
  }
  let bound: i128 = coeffs_abs.iter().sum();
  Some(Expr::FunctionCall {
    name: "Interval".to_string(),
    args: vec![Expr::List(
      vec![Expr::Integer(-bound), Expr::Integer(bound)].into(),
    )]
    .into(),
  })
}

/// Split an additive term `t` into `(coefficient, body)` where the
/// coefficient is the product of factors free of `var_name` and `body`
/// is the unique remaining factor. Returns None if there isn't exactly
/// one var-dependent factor.
fn split_constant_factor(t: &Expr, var_name: &str) -> Option<(Expr, Expr)> {
  // Pull out a leading unary minus.
  let (sign, inner) = match t {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => (-1i128, &**operand),
    _ => (1i128, t),
  };
  let factors =
    crate::functions::polynomial_ast::collect_multiplicative_factors(inner);
  let (const_factors, var_factors): (Vec<Expr>, Vec<Expr>) = factors
    .into_iter()
    .partition(|f| !contains_var(f, var_name));
  if var_factors.len() != 1 {
    return None;
  }
  let coeff = if const_factors.is_empty() {
    Expr::Integer(sign)
  } else {
    let product = if const_factors.len() == 1 {
      const_factors.into_iter().next().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: const_factors.into(),
      }
    };
    if sign == -1 {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(product),
      }
    } else {
      product
    }
  };
  Some((coeff, var_factors.into_iter().next().unwrap()))
}

/// Try to evaluate an Expr to its absolute integer value. Used by the
/// bounded-oscillating-sum heuristic where coefficients must be
/// integer-typed to compose a clean Interval bound.
fn expr_to_abs_int(expr: &Expr) -> Option<i128> {
  let evaluated = crate::evaluator::evaluate_expr_to_expr(expr)
    .unwrap_or_else(|_| expr.clone());
  if let Expr::Integer(n) = evaluated {
    return Some(n.unsigned_abs() as i128);
  }
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = &evaluated
    && let Expr::Integer(n) = **operand
  {
    return Some(n.unsigned_abs() as i128);
  }
  None
}

/// Cheap test: does `expr` mention `var_name` anywhere?
fn contains_var(expr: &Expr, var_name: &str) -> bool {
  !is_constant_wrt(expr, var_name)
}

/// Whether any `FunctionCall` inside `expr` has the given head name.
fn expr_contains_head(expr: &Expr, head: &str) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      name == head || args.iter().any(|a| expr_contains_head(a, head))
    }
    Expr::BinaryOp { left, right, .. } => {
      expr_contains_head(left, head) || expr_contains_head(right, head)
    }
    Expr::UnaryOp { operand, .. } => expr_contains_head(operand, head),
    Expr::List(items) => items.iter().any(|i| expr_contains_head(i, head)),
    _ => false,
  }
}

/// Whether `expr` is a finite real number literal (integer, rational, or real).
fn is_finite_real_number(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_))
    || matches!(expr, Expr::FunctionCall { name, .. } if name == "Rational")
}

/// Structurally determine whether `expr` diverges to +Infinity (`Some(1)`) or
/// -Infinity (`Some(-1)`) as `var -> point`. Only the `point == +Infinity`
/// direction is handled; whenever the growth cannot be classified this returns
/// `None` so the caller falls back to the numeric heuristic. This catches the
/// slowly-growing monotonic forms (Log[x], Sqrt[x], x^(1/3), Log[Log[x]], …)
/// that never reach the numeric `|f| > 1e5` divergence threshold.
fn diverges_to_infinity(expr: &Expr, var: &str, point: &Expr) -> Option<i32> {
  if !is_infinity(point) {
    return None;
  }
  diverges_pos_infinity(expr, var)
}

fn diverges_pos_infinity(expr: &Expr, var: &str) -> Option<i32> {
  // The variable itself diverges to +Infinity.
  if let Expr::Identifier(name) = expr {
    return if name == var { Some(1) } else { None };
  }
  // Anything free of `var` stays finite.
  if is_constant_wrt(expr, var) {
    return None;
  }

  // -f
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = expr
  {
    return diverges_pos_infinity(operand, var).map(|s| -s);
  }

  // a - b  ==  a + (-b)
  if let Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left,
    right,
  } = expr
  {
    let plus = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        (**left).clone(),
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: right.clone(),
        },
      ]
      .into(),
    };
    return diverges_pos_infinity(&plus, var);
  }

  // Sqrt[g] and Log[g] (single argument): diverge iff their argument does.
  if let Expr::FunctionCall { name, args } = expr
    && args.len() == 1
    && (name == "Sqrt" || name == "Log")
  {
    return (diverges_pos_infinity(&args[0], var) == Some(1)).then_some(1);
  }

  // n!, n!!, Gamma[n] diverge to +Infinity when their argument does. (This
  // also lets the Log[g] case above resolve Log[n!] -> Infinity.)
  if let Expr::FunctionCall { name, args } = expr
    && args.len() == 1
    && (name == "Factorial" || name == "Factorial2" || name == "Gamma")
  {
    return (diverges_pos_infinity(&args[0], var) == Some(1)).then_some(1);
  }

  // base ^ exp, with a positive constant exponent.
  if let Some((base, exp)) = extract_power(expr)
    && is_constant_wrt(&exp, var)
    && let Some(ev) = crate::functions::math_ast::try_eval_to_f64(&exp)
    && ev > 0.0
  {
    return match diverges_pos_infinity(&base, var)? {
      1 => Some(1), // (+Infinity)^positive -> +Infinity
      -1 => match &exp {
        // (-Infinity)^k: sign by parity, only safe for integer powers.
        Expr::Integer(k) => Some(if k % 2 == 0 { 1 } else { -1 }),
        _ => None,
      },
      _ => None,
    };
  }

  // A product of powers of `var` and logarithms with positive net polynomial
  // order diverges (e.g. x/Log[x]); the sign is that of the expression at a
  // large argument. (net_poly_order returns None for Sin/E^x/… so those aren't
  // misclassified here.)
  if let Some(order) = net_poly_order(expr, var)
    && order > 0.0
    && let Some(v) = eval_at_large_n(expr, var, 1_000_000)
    && v != 0.0
  {
    return Some(if v > 0.0 { 1 } else { -1 });
  }

  // Product / sum of factors.
  times_divergence(expr, var).or_else(|| plus_divergence(expr, var))
}

/// Sign of divergence of a product, or `None` if it doesn't diverge / can't be
/// classified. A zero or non-divergent variable factor disqualifies it.
fn times_divergence(expr: &Expr, var: &str) -> Option<i32> {
  let factors: Vec<Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Times" => args.to_vec(),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => vec![(**left).clone(), (**right).clone()],
    _ => return None,
  };
  let mut total = 1i32;
  let mut saw_divergent = false;
  for f in &factors {
    if is_constant_wrt(f, var) {
      let v = crate::functions::math_ast::try_eval_to_f64(f)?;
      if v == 0.0 {
        return None;
      }
      if v < 0.0 {
        total = -total;
      }
    } else {
      total *= diverges_pos_infinity(f, var)?;
      saw_divergent = true;
    }
  }
  saw_divergent.then_some(total)
}

/// Sign of divergence of a sum. Finite (constant) terms are ignored; a
/// non-constant term that doesn't itself diverge makes the sum unclassifiable,
/// and a mix of +Infinity and -Infinity terms is left indeterminate.
fn plus_divergence(expr: &Expr, var: &str) -> Option<i32> {
  let terms: Vec<Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => args.to_vec(),
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => vec![(**left).clone(), (**right).clone()],
    _ => return None,
  };
  let mut has_pos = false;
  let mut has_neg = false;
  for t in &terms {
    if is_constant_wrt(t, var) {
      continue; // finite contribution
    }
    match diverges_pos_infinity(t, var) {
      Some(1) => has_pos = true,
      Some(-1) => has_neg = true,
      _ => return None, // bounded/oscillating/unknown term: bail
    }
  }
  match (has_pos, has_neg) {
    (true, false) => Some(1),
    (false, true) => Some(-1),
    _ => None, // no divergent term, or indeterminate Infinity - Infinity
  }
}

/// Structurally determine whether `expr` decays to 0 as `var -> +Infinity`.
/// Recognizes a negative constant power of a diverging base (1/Log[x],
/// 1/Sqrt[x], Log[x]^(-2), x^(-1/3)) and products of such with finite
/// factors (2/Log[x]). Returns false whenever it can't prove decay, so the
/// numeric path still gets a chance.
/// Net polynomial order of `var` in a product of powers of `var`, `Sqrt`s and
/// logarithms (logs count as order 0, since they grow slower than any positive
/// power). Returns `None` for any other var-dependent factor (Sin, E^x, …) so
/// such expressions aren't misclassified. Used to prove decay for mixed
/// log/power products like `Log[x]/Sqrt[x]` (order -1/2 -> 0).
fn net_poly_order(expr: &Expr, var: &str) -> Option<f64> {
  if is_constant_wrt(expr, var) {
    return Some(0.0);
  }
  match expr {
    Expr::Identifier(v) if v == var => Some(1.0),
    // Sqrt[g] has half the order of g.
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      Some(net_poly_order(&args[0], var)? / 2.0)
    }
    // Any (real) power of a diverging logarithm is sub-polynomial: order 0.
    Expr::FunctionCall { name, args }
      if name == "Log"
        && args.len() == 1
        && diverges_pos_infinity(&args[0], var) == Some(1) =>
    {
      Some(0.0)
    }
    // Times: orders add.
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut total = 0.0;
      for a in args.iter() {
        total += net_poly_order(a, var)?;
      }
      Some(total)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => Some(net_poly_order(left, var)? + net_poly_order(right, var)?),
    _ => {
      // base^exp with a constant exponent scales the base's order; this also
      // covers Log[g]^q (base order 0 -> 0) and x^p.
      if let Some((base, exp)) = extract_power(expr)
        && is_constant_wrt(&exp, var)
        && let Some(ev) = crate::functions::math_ast::try_eval_to_f64(&exp)
      {
        return Some(net_poly_order(&base, var)? * ev);
      }
      None
    }
  }
}

fn tends_to_zero(expr: &Expr, var: &str) -> bool {
  // A negative constant power of a base that diverges to +/-Infinity.
  if let Some((base, exp)) = extract_power(expr)
    && is_constant_wrt(&exp, var)
    && let Some(ev) = crate::functions::math_ast::try_eval_to_f64(&exp)
    && ev < 0.0
    && diverges_pos_infinity(&base, var).is_some()
  {
    return true;
  }
  // A product of powers of `var` and logarithms whose net polynomial order is
  // negative decays to 0 (e.g. Log[x]/Sqrt[x], Log[x]^5/x). The power of `var`
  // dominates any logarithmic factor.
  if let Some(order) = net_poly_order(expr, var)
    && order < 0.0
  {
    return true;
  }
  // A product whose factors are all finite constants or decaying, with at
  // least one decaying factor (e.g. 2/Log[x]). A diverging factor (x/Log[x])
  // disqualifies it.
  let factors: Option<Vec<Expr>> = match expr {
    Expr::FunctionCall { name, args } if name == "Times" => Some(args.to_vec()),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => Some(vec![(**left).clone(), (**right).clone()]),
    _ => None,
  };
  if let Some(factors) = factors {
    let mut saw_zero = false;
    for f in &factors {
      if is_constant_wrt(f, var) {
        continue; // finite factor
      } else if tends_to_zero(f, var) {
        saw_zero = true;
      } else {
        return false; // diverging or unclassifiable factor
      }
    }
    return saw_zero;
  }
  false
}

/// Detect decay-to-a-finite-value limits at +Infinity that the numeric path
/// misses for slowly-decaying terms: a pure 1/g(x) -> 0, or a sum whose only
/// var-dependent terms decay to 0 (1 + 1/Log[x] -> 1, 5 - 3/Sqrt[x] -> 5).
fn limit_decay_at_infinity(
  expr: &Expr,
  var: &str,
  point: &Expr,
) -> Option<Expr> {
  if !is_infinity(point) {
    return None;
  }
  if tends_to_zero(expr, var) {
    return Some(Expr::Integer(0));
  }
  // Sum: every var-dependent term must decay to 0; the limit is then the sum
  // of the constant terms.
  let terms: Vec<Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => args.to_vec(),
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => vec![(**left).clone(), (**right).clone()],
    _ => return None,
  };
  let mut constant_terms: Vec<Expr> = Vec::new();
  let mut saw_decaying = false;
  for t in &terms {
    if is_constant_wrt(t, var) {
      constant_terms.push(t.clone());
    } else if tends_to_zero(t, var) {
      saw_decaying = true;
    } else {
      return None; // a term that neither stays constant nor decays
    }
  }
  if !saw_decaying {
    return None;
  }
  let sum = match constant_terms.len() {
    0 => Expr::Integer(0),
    1 => constant_terms.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Plus".to_string(),
      args: constant_terms.into(),
    },
  };
  crate::evaluator::evaluate_expr_to_expr(&sum).ok()
}

/// The radicand of `Sqrt[P]` / `Power[P, 1/2]`, or `mag^2` for a polynomial
/// `mag` (so a polynomial term `p` counts as `Sqrt[p^2]` in a difference).
fn sqrt_radicand(mag: &Expr, _var: &str) -> Option<Expr> {
  if let Expr::FunctionCall { name, args } = mag
    && name == "Sqrt"
    && args.len() == 1
  {
    return Some(args[0].clone());
  }
  if let Some((base, exp)) = extract_power(mag)
    && matches!(
      &exp,
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2
          && matches!((&args[0], &args[1]), (Expr::Integer(1), Expr::Integer(2)))
    )
  {
    return Some(base);
  }
  // Treat the term as Sqrt[mag^2]; coefflist later rejects non-polynomials.
  let sq = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(mag.clone()),
    right: Box::new(Expr::Integer(2)),
  };
  crate::evaluator::evaluate_function_call_ast("Expand", &[sq]).ok()
}

/// CoefficientList[P, var] as a vector of exact coefficient expressions
/// (ascending degree), or None if `P` isn't a polynomial in `var`.
fn poly_coefflist(p: &Expr, var: &str) -> Option<Vec<Expr>> {
  let cl = Expr::FunctionCall {
    name: "CoefficientList".to_string(),
    args: vec![p.clone(), Expr::Identifier(var.to_string())].into(),
  };
  let evaled = crate::evaluator::evaluate_expr_to_expr(&cl).ok()?;
  match &evaled {
    Expr::List(items) => Some(items.iter().cloned().collect()),
    _ => None,
  }
}

/// Limit at +Infinity of a two-term difference `Sqrt[A] - Sqrt[B]` (a
/// polynomial term `p` counting as `Sqrt[p^2]`), via the leading-coefficient
/// asymptotics. For radicands of equal degree `d` and equal leading
/// coefficient `c`, `Sqrt[A] - Sqrt[B] ~ (a1 - b1)/(2 Sqrt[c]) * x^(d/2 - 1)`,
/// where `a1`, `b1` are the next coefficients: 0 for `d < 2`, the finite
/// constant for `d == 2`. Other shapes return None.
fn limit_sqrt_difference(expr: &Expr, var: &str, point: &Expr) -> Option<Expr> {
  if !is_infinity(point) {
    return None;
  }
  let terms: Vec<Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().cloned().collect()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => vec![(**left).clone(), (**right).clone()],
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => vec![
      (**left).clone(),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), (**right).clone()].into(),
      },
    ],
    _ => return None,
  };
  if terms.len() != 2 {
    return None;
  }
  // Classify each term: sign from a numeric probe, radicand of its magnitude.
  let mut pos_rad: Option<Expr> = None;
  let mut neg_rad: Option<Expr> = None;
  for t in &terms {
    let probe = eval_at_large_n(t, var, 1_000_000)?;
    if probe == 0.0 {
      return None;
    }
    let mag = if probe > 0.0 {
      t.clone()
    } else {
      simplify(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), t.clone()].into(),
      })
    };
    let radicand = sqrt_radicand(&mag, var)?;
    if probe > 0.0 {
      if pos_rad.is_some() {
        return None;
      }
      pos_rad = Some(radicand);
    } else {
      if neg_rad.is_some() {
        return None;
      }
      neg_rad = Some(radicand);
    }
  }
  let (a, b) = (pos_rad?, neg_rad?);
  let ca = poly_coefflist(&a, var)?;
  let cb = poly_coefflist(&b, var)?;
  let d = ca.len().checked_sub(1)?;
  // Both radicands must be genuine polynomials of equal degree (>= 1) with the
  // same positive leading coefficient — otherwise the difference diverges.
  if d == 0 || cb.len().checked_sub(1)? != d || !expr_str_eq(&ca[d], &cb[d]) {
    return None;
  }
  if !matches!(
    crate::functions::math_ast::try_eval_to_f64(&ca[d]),
    Some(c) if c > 0.0
  ) {
    return None;
  }
  if d < 2 {
    // x^(d/2 - 1) -> 0 (d == 1).
    return Some(Expr::Integer(0));
  }
  if d == 2 {
    // (a1 - b1) / (2 Sqrt[c]).
    let result = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(ca[d - 1].clone()),
        right: Box::new(cb[d - 1].clone()),
      }),
      right: Box::new(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(2),
          Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![ca[d].clone()].into(),
          },
        ]
        .into(),
      }),
    };
    return crate::evaluator::evaluate_expr_to_expr(&result).ok();
  }
  None
}

/// Handle limits at infinity.
/// Strategies:
/// 1. If expr is constant wrt var, return it directly
/// 2. Direct substitution heuristic (evaluate at large n) to classify the limit
/// 3. For f^g where f->1, g->inf: Limit = E^(Limit[g*(f-1)])
/// 4. For expressions going to 0 or a constant: detect via structure
fn limit_at_infinity(
  expr: &Expr,
  var_name: &str,
  point: &Expr,
) -> Result<Expr, InterpreterError> {
  // If the expression is constant wrt the variable, return it
  if is_constant_wrt(expr, var_name) {
    return crate::evaluator::evaluate_expr_to_expr(expr);
  }

  // Pull out multiplicative factors that are constant w.r.t. `var_name`.
  // Lets `Limit[a*f(n), n -> Infinity]` reduce to `a * Limit[f(n), n -> Infinity]`
  // — without this, our numeric-fallback path bails when the expression
  // contains free variables besides the limit target. Only apply when at
  // least one factor would actually be peeled off.
  let factors =
    crate::functions::polynomial_ast::collect_multiplicative_factors(expr);
  if factors.len() >= 2 {
    let (constant_factors, var_factors): (Vec<Expr>, Vec<Expr>) = factors
      .into_iter()
      .partition(|f| is_constant_wrt(f, var_name));
    if !constant_factors.is_empty() && !var_factors.is_empty() {
      let var_part = if var_factors.len() == 1 {
        var_factors.into_iter().next().unwrap()
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: var_factors.into(),
        }
      };
      let inner_limit = limit_at_infinity(&var_part, var_name, point)?;
      // Only commit to the factored form if the recursive limit actually
      // resolved (didn't come back wrapped in a Limit[...] head).
      if !matches!(&inner_limit, Expr::FunctionCall { name, .. } if name == "Limit")
      {
        let mut all = constant_factors;
        all.push(inner_limit);
        let product = if all.len() == 1 {
          all.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: all.into(),
          }
        };
        return crate::evaluator::evaluate_expr_to_expr(&product);
      }
    }
  }

  // Handle var itself: Limit[n, n -> Infinity] = Infinity
  if let Expr::Identifier(name) = expr
    && name == var_name
  {
    return Ok(point.clone());
  }

  // Oscillating trig functions at +/- Infinity have no limit:
  // Limit[Sin[x], x -> Infinity] etc. -> Indeterminate.
  if let Expr::FunctionCall { name, args: targs } = expr
    && targs.len() == 1
    && matches!(name.as_str(), "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc")
    && let Expr::Identifier(arg_name) = &targs[0]
    && arg_name == var_name
  {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  // Bounded oscillating sum at infinity: a sum of Sin/Cos terms with
  // polynomial arguments accumulates densely in [-bound, bound] when at
  // least one pair of trig arguments has different polynomial degrees
  // in `var_name` (wolframscript's heuristic for incommensurate
  // oscillation). With all arguments at the same polynomial degree the
  // sum is quasi-periodic and the limit stays Indeterminate.
  if let Some(result) = limit_bounded_oscillating_sum(expr, var_name) {
    return Ok(result);
  }

  // Handle f^g form (e.g., (1 + 1/n)^n -> E, or E^(-m/(m+1)) -> E^(-1))
  if let Some((base, exp)) = extract_power(expr) {
    // Constant base, variable exponent: lift the limit into the exponent.
    // Limit[c^g(n), n -> point] = c^Limit[g, n -> point] when c is free of n.
    if is_constant_wrt(&base, var_name) && !is_constant_wrt(&exp, var_name) {
      let rule = Expr::Rule {
        pattern: Box::new(Expr::Identifier(var_name.to_string())),
        replacement: Box::new(point.clone()),
      };
      let exp_limit = limit_ast(&[exp.clone(), rule])?;
      // Only commit if the inner Limit fully resolved.
      if !matches!(&exp_limit, Expr::FunctionCall { name, .. } if name == "Limit")
      {
        let result = Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(base.clone()),
          right: Box::new(exp_limit),
        };
        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
    }
    // Check if base -> 1 and exponent -> Infinity (1^Infinity indeterminate form)
    if eval_at_infinity_is_one(&base, var_name)
      && eval_at_infinity_diverges(&exp, var_name).is_some()
    {
      // Use identity: Limit[f^g] = E^(Limit[g * (f - 1)]) when f -> 1, g -> inf
      let f_minus_1 = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(base),
        right: Box::new(Expr::Integer(1)),
      };
      let product = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(exp),
        right: Box::new(f_minus_1),
      };
      // Simplify g*(f - 1) first so a symbolic coefficient cancels (e.g.
      // n*((1 + a/n) - 1) → a); otherwise limit_ast's numeric fallback can't
      // resolve the parameterized form.
      let product =
        crate::evaluator::evaluate_expr_to_expr(&product).unwrap_or(product);
      // Take the limit of g * (f - 1) as var -> Infinity
      let rule = Expr::Rule {
        pattern: Box::new(Expr::Identifier(var_name.to_string())),
        replacement: Box::new(point.clone()),
      };
      let exponent_limit = limit_ast(&[product, rule])?;

      // Return E^limit once the exponent limit has resolved — either a clean
      // number or a parameter-only expression like `a` (giving E^a). The only
      // rejected case is an unresolved Limit[...] still containing the var.
      let resolved = !matches!(&exponent_limit,
        Expr::FunctionCall { name, .. } if name == "Limit")
        && is_constant_wrt(&exponent_limit, var_name);
      if is_clean_value(&exponent_limit) || resolved {
        let result = simplify(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::Constant("E".to_string())),
          right: Box::new(exponent_limit),
        });
        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
    }
  }

  // Structural detection of monotonic divergence to +/-Infinity for forms the
  // numeric fast-path (which needs |f| > 1e5 at n = 1e6) misses because they
  // grow slowly: Log[x], Sqrt[x], x^(1/3), Log[2 x], Log[Log[x]], Log[x]^2, …
  if let Some(s) = diverges_to_infinity(expr, var_name, point) {
    return Ok(if s >= 0 {
      Expr::Identifier("Infinity".to_string())
    } else {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      }
    });
  }

  // Structural detection of decay to a finite value for slowly-decaying
  // reciprocals the numeric path misses: 1/Log[x] -> 0, 2/Log[x] -> 0,
  // 1 + 1/Log[x] -> 1, 5 - 3/Sqrt[x] -> 5.
  if let Some(result) = limit_decay_at_infinity(expr, var_name, point) {
    return Ok(result);
  }

  // Conjugate-difference forms (Sqrt[x^2+x] - x -> 1/2, Sqrt[n+1] - Sqrt[n]
  // -> 0) are ∞ - ∞ indeterminates the numeric path can't recognize exactly.
  if let Some(result) = limit_sqrt_difference(expr, var_name, point) {
    return Ok(result);
  }

  // Direct substitution of the limit point: if substituting `point` yields a
  // finite, variable-free value (e.g. HarmonicNumber[Infinity, 2] -> Pi^2/6,
  // ArcTan[Infinity] -> Pi/2, 1/n -> 0), that is the limit. Indeterminate
  // forms (0*Inf, Inf-Inf, Inf/Inf) evaluate to Indeterminate / ComplexInfinity
  // / Infinity and are rejected here, deferring to the heuristics below. This
  // also avoids the numeric fallback evaluating functions like HarmonicNumber
  // at n = 10^7, which would sum ten million terms.
  {
    let subst = crate::syntax::substitute_variable(expr, var_name, point);
    // The substitution is speculative — indeterminate forms like Inf/Inf emit
    // messages (e.g. Infinity::indet) we then discard, so suppress them.
    crate::push_quiet();
    let evaluated = crate::evaluator::evaluate_expr_to_expr(&subst);
    crate::pop_quiet();
    if let Ok(val) = evaluated
      && is_constant_wrt(&val, var_name)
      && (is_finite_value(&val)
        || is_infinity(&val)
        || is_negative_infinity(&val))
    {
      return Ok(val);
    }
  }

  // For simple expressions, try evaluating at two large values to detect convergence
  let sign = if is_negative_infinity(point) { -1 } else { 1 };
  let val1 = eval_at_large_n(expr, var_name, sign * 1_000_000);
  let val2 = eval_at_large_n(expr, var_name, sign * 10_000_000);
  if let (Some(f1), Some(f2)) = (val1, val2) {
    // Both diverging to +infinity
    if f1 > 1e5 && f2 > f1 {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    // Both diverging to -infinity
    if f1 < -1e5 && f2 < f1 {
      return Ok(Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
    // Approaching zero: both values small and getting smaller
    if f2.abs() < 1e-4 && f2.abs() < f1.abs() {
      return Ok(Expr::Integer(0));
    }
    // Check convergence: values should be close (relative difference)
    let diff = (f1 - f2).abs();
    let scale = f1.abs().max(f2.abs()).max(1e-15);
    if diff / scale < 0.01 {
      // Converging to a nonzero limit — determine the value
      // Check if the limit is a known integer
      let rounded = f2.round();
      if (f2 - rounded).abs() < 1e-4 {
        return Ok(Expr::Integer(rounded as i128));
      }
      // Check for a simple rational p/q — the ratio-of-leading-coefficients
      // limit of a rational function (e.g. (x^2+1)/(2x^2-3) -> 1/2). The probe
      // points are a decade apart (n, 10n), so for the typical ~1/n
      // convergence a Richardson step `f2 + (f2-f1)/9` removes the leading
      // error term, sharpening the estimate. The tight tolerance keeps
      // irrational limits (Pi/2, Log[2], …) from matching.
      let l_ext = f2 + (f2 - f1) / 9.0;
      for q in 2..=36i128 {
        let pf = l_ext * q as f64;
        let p = pf.round();
        if (pf - p).abs() < 1e-6 {
          return Ok(crate::functions::math_ast::make_rational(p as i128, q));
        }
      }
      // Check for known constants
      if (f2 - std::f64::consts::E).abs() < 1e-3 {
        return Ok(Expr::Constant("E".to_string()));
      }
      if (f2 - std::f64::consts::PI).abs() < 1e-3 {
        return Ok(Expr::Constant("Pi".to_string()));
      }
      // Check for common multiples/fractions of Pi
      let pi = std::f64::consts::PI;
      let pi_fractions: &[(f64, i128, i128)] = &[
        (pi / 2.0, 1, 2),
        (-pi / 2.0, -1, 2),
        (pi / 3.0, 1, 3),
        (-pi / 3.0, -1, 3),
        (pi / 4.0, 1, 4),
        (-pi / 4.0, -1, 4),
        (pi / 6.0, 1, 6),
        (-pi / 6.0, -1, 6),
        (2.0 * pi / 3.0, 2, 3),
        (-2.0 * pi / 3.0, -2, 3),
        (3.0 * pi / 4.0, 3, 4),
        (-3.0 * pi / 4.0, -3, 4),
        (5.0 * pi / 6.0, 5, 6),
        (-5.0 * pi / 6.0, -5, 6),
        (-pi, -1, 1),
        (2.0 * pi, 2, 1),
        (-2.0 * pi, -2, 1),
      ];
      for &(val, numer, denom) in pi_fractions {
        if (f2 - val).abs() < 1e-3 {
          if denom == 1 {
            if numer == -1 {
              return Ok(Expr::UnaryOp {
                op: UnaryOperator::Minus,
                operand: Box::new(Expr::Constant("Pi".to_string())),
              });
            }
            return Ok(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(numer)),
              right: Box::new(Expr::Constant("Pi".to_string())),
            });
          }
          return Ok(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(numer), Expr::Integer(denom)].into(),
              },
              Expr::Constant("Pi".to_string()),
            ]
            .into(),
          });
        }
      }
    }
  }

  // Expressions with exponential growth in `var` (Exp/Sinh/Cosh/Tanh, E^x):
  // the 1e6 probe above overflows E^x, so probe at moderate points where the
  // exponentials stay in f64 range and the decaying part is negligible.
  if let Some(v) = exp_growth_limit_at_infinity(expr, var_name, point) {
    return Ok(v);
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "Limit".to_string(),
    args: vec![
      expr.clone(),
      Expr::Rule {
        pattern: Box::new(Expr::Identifier(var_name.to_string())),
        replacement: Box::new(point.clone()),
      },
    ]
    .into(),
  })
}

/// True when `expr` contains an exponential-growth subterm in `var`: Exp / Sinh
/// / Cosh / Tanh / Coth / Sech / Csch with a var-dependent argument, or E raised
/// to a var-dependent power.
fn contains_exponential_of_var(expr: &Expr, var: &str) -> bool {
  // A growing exponential: E (or any numeric base > 1) raised to a
  // var-dependent power. Bases <= 1 decay and are handled by the ordinary
  // numeric probe.
  let is_e_pow = |base: &Expr, exp: &Expr| {
    if is_constant_wrt(exp, var) {
      return false;
    }
    matches!(base, Expr::Constant(c) if c == "E")
      || matches!(
        crate::functions::math_ast::try_eval_to_f64(base),
        Some(b) if b > 1.0
      )
  };
  match expr {
    Expr::FunctionCall { name, args } => {
      (matches!(
        name.as_str(),
        "Exp" | "Sinh" | "Cosh" | "Tanh" | "Coth" | "Sech" | "Csch"
      ) && args.iter().any(|a| !is_constant_wrt(a, var)))
        || (name == "Power" && args.len() == 2 && is_e_pow(&args[0], &args[1]))
        || args.iter().any(|a| contains_exponential_of_var(a, var))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } if is_e_pow(left, right) => true,
    Expr::BinaryOp { left, right, .. } => {
      contains_exponential_of_var(left, var)
        || contains_exponential_of_var(right, var)
    }
    Expr::UnaryOp { operand, .. } => contains_exponential_of_var(operand, var),
    _ => false,
  }
}

/// True when `expr` contains a Power whose exponent depends on `var` (c^x,
/// x^x, …) — the case whose exact integer value explodes at large probes.
fn contains_var_power(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      (name == "Power" && args.len() == 2 && !is_constant_wrt(&args[1], var))
        || (name == "Exp" && !is_constant_wrt(&args[0], var))
        || args.iter().any(|a| contains_var_power(a, var))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      right,
      ..
    } if !is_constant_wrt(right, var) => true,
    Expr::BinaryOp { left, right, .. } => {
      contains_var_power(left, var) || contains_var_power(right, var)
    }
    Expr::UnaryOp { operand, .. } => contains_var_power(operand, var),
    _ => false,
  }
}

/// For x -> ±Infinity limits of expressions with exponential growth in `var`,
/// probe two moderate points (E^180 ≈ 1e78 stays within f64 range while
/// E^-90 ≈ 1e-39 makes decaying terms negligible) and recognise a clean
/// integer / simple-rational / zero / divergent limit.
fn exp_growth_limit_at_infinity(
  expr: &Expr,
  var: &str,
  point: &Expr,
) -> Option<Expr> {
  if !contains_exponential_of_var(expr, var) {
    return None;
  }
  let sign: i128 = if is_negative_infinity(point) { -1 } else { 1 };
  let f1 = eval_at_large_n(expr, var, sign * 90)?;
  let f2 = eval_at_large_n(expr, var, sign * 180)?;
  if !f1.is_finite() || !f2.is_finite() {
    return None;
  }
  // Divergence to ±Infinity.
  if f1.abs() > 1e5 && f2.abs() > f1.abs() {
    return Some(if f2 > 0.0 {
      Expr::Identifier("Infinity".to_string())
    } else {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      }
    });
  }
  // Approaching zero — both probes tiny and shrinking (the relative-agreement
  // gate below would otherwise reject these, since the values differ by orders
  // of magnitude even though both -> 0).
  if f1.abs() < 1e-6 && f2.abs() < f1.abs() {
    return Some(Expr::Integer(0));
  }
  // Require tight agreement between the two probes before committing.
  let diff = (f1 - f2).abs();
  let scale = f1.abs().max(f2.abs()).max(1e-12);
  if diff / scale > 1e-6 {
    return None;
  }
  let rounded = f2.round();
  if (f2 - rounded).abs() < 1e-9 {
    return Some(Expr::Integer(rounded as i128));
  }
  // Recognise a simple rational p/q (smallest denominator wins).
  for q in 2..=36i128 {
    let pf = f2 * q as f64;
    let p = pf.round();
    if (pf - p).abs() < 1e-7 {
      return Some(crate::functions::math_ast::make_rational(p as i128, q));
    }
  }
  None
}

/// True when `e` is a finite value: a number, or a constant expression (e.g.
/// Pi^2/6, Zeta[3]) that reduces to a finite real under N. The Infinity /
/// Indeterminate / ComplexInfinity / Undefined sentinels are explicitly
/// rejected so indeterminate forms don't get mistaken for a finite limit.
fn is_finite_value(e: &Expr) -> bool {
  if matches!(e, Expr::Identifier(n) | Expr::Constant(n)
    if n == "Infinity"
      || n == "Indeterminate"
      || n == "ComplexInfinity"
      || n == "Undefined")
  {
    return false;
  }
  // Reject any expression that still carries an infinity/indeterminate head.
  if matches!(e, Expr::FunctionCall { name, .. }
    if name == "DirectedInfinity" || name == "Indeterminate")
  {
    return false;
  }
  if let Some(f) = crate::functions::math_ast::try_eval_to_f64(e) {
    return f.is_finite();
  }
  // Symbolic constant such as Pi^2/6 or Zeta[3]: confirm it is finite via N.
  let napprox = Expr::FunctionCall {
    name: "N".to_string(),
    args: vec![e.clone()].into(),
  };
  if let Ok(nv) = crate::evaluator::evaluate_expr_to_expr(&napprox)
    && let Some(f) = crate::functions::math_ast::try_eval_to_f64(&nv)
  {
    return f.is_finite();
  }
  false
}

/// Evaluate an expression numerically at var = n
fn eval_at_large_n(expr: &Expr, var: &str, n: i128) -> Option<f64> {
  if contains_explosive_of_var(expr, var) {
    return None;
  }
  // At very large probe points, an expression with a variable exponent (c^x,
  // x^x) would otherwise require building astronomically large *exact*
  // integers (e.g. 3^1000000 has ~477k digits), which hangs. Substitute a
  // float there so the arithmetic stays in f64 — overflowing to Infinity
  // instantly — rather than computing the giant BigInteger. Moderate probes
  // (used by the exponential-limit fallback) keep exact integer substitution.
  let point = if n.abs() > 10_000 && contains_var_power(expr, var) {
    Expr::Real(n as f64)
  } else {
    Expr::Integer(n)
  };
  let subst = crate::syntax::substitute_variable(expr, var, &point);
  let val = crate::evaluator::evaluate_expr_to_expr(&subst).ok()?;
  if let Some(f) = crate::functions::math_ast::try_eval_to_f64(&val) {
    return Some(f);
  }
  // Some functions stay symbolic at an exact integer argument but evaluate
  // numerically under N[...] (e.g. ArcCot[1000000]); fall back to that so the
  // numeric limit heuristic can still classify them.
  let napprox = Expr::FunctionCall {
    name: "N".to_string(),
    args: vec![val].into(),
  };
  let nval = crate::evaluator::evaluate_expr_to_expr(&napprox).ok()?;
  crate::functions::math_ast::try_eval_to_f64(&nval)
}

/// Check if an expression is a "clean" value (integer, real, constant, or rational)
fn is_clean_value(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      true
    }
    _ => crate::functions::math_ast::try_eval_to_f64(expr).is_some(),
  }
}

/// MaxLimit[f, x -> a] - largest limiting value (from above/right)
/// MinLimit[f, x -> a] - smallest limiting value (from below/left)
pub fn max_limit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  one_sided_limit_ast(args, "MaxLimit", LimitDirection::FromAbove)
}

pub fn min_limit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  one_sided_limit_ast(args, "MinLimit", LimitDirection::FromBelow)
}

fn one_sided_limit_ast(
  args: &[Expr],
  fn_name: &str,
  direction: LimitDirection,
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated(fn_name, args));
  }

  // Build Limit[expr, rule, Direction -> dir]
  let dir_str = match direction {
    LimitDirection::FromAbove => "FromAbove",
    LimitDirection::FromBelow => "FromBelow",
    LimitDirection::TwoSided => "FromAbove",
  };
  let direction_opt = Expr::Rule {
    pattern: Box::new(Expr::Identifier("Direction".to_string())),
    replacement: Box::new(Expr::String(dir_str.to_string())),
  };

  let result = limit_ast(&[args[0].clone(), args[1].clone(), direction_opt])?;

  // Check if result is unevaluated Limit
  if let Expr::FunctionCall { name, .. } = &result
    && name == "Limit"
  {
    return Ok(unevaluated(fn_name, args));
  }

  Ok(result)
}

/// Direction for one-sided limits
#[derive(Debug, Clone, Copy, PartialEq)]
enum LimitDirection {
  /// Two-sided limit (default)
  TwoSided,
  /// From above (x -> x0+), i.e. from the right
  FromAbove,
  /// From below (x -> x0-), i.e. from the left
  FromBelow,
}

/// Parse the Direction option from a Rule like `Direction -> "FromAbove"`
fn parse_direction(option: &Expr) -> Option<LimitDirection> {
  if let Expr::Rule {
    pattern,
    replacement,
  } = option
    && let Expr::Identifier(name) = pattern.as_ref()
    && name == "Direction"
  {
    match replacement.as_ref() {
      Expr::String(s) if s == "FromAbove" => {
        return Some(LimitDirection::FromAbove);
      }
      Expr::String(s) if s == "FromBelow" => {
        return Some(LimitDirection::FromBelow);
      }
      // Direction -> -1 means from above (from the right, x -> x0+)
      Expr::Integer(n) if *n == -1 => {
        return Some(LimitDirection::FromAbove);
      }
      // Direction -> 1 means from below (from the left, x -> x0-)
      Expr::Integer(n) if *n == 1 => {
        return Some(LimitDirection::FromBelow);
      }
      _ => {}
    }
  }
  None
}

/// Check if two scalar comparisons cover the whole real line, i.e. one is
/// the negation of the other. Recognises the four pair shapes that a
/// Piecewise-on-the-real-line typically takes: `x ≤ c` ↔ `x > c` and
/// `x < c` ↔ `x ≥ c`, in either order. Used by Integrate[Piecewise[…]] to
/// collapse two complementary pieces into one piece + default.
fn conditions_are_complementary(a: &Expr, b: &Expr) -> bool {
  let extract = |c: &Expr| -> Option<(Expr, ComparisonOp, Expr)> {
    if let Expr::Comparison {
      operands,
      operators,
    } = c
      && operators.len() == 1
      && operands.len() == 2
    {
      Some((operands[0].clone(), operators[0], operands[1].clone()))
    } else {
      None
    }
  };
  let (la, opa, ra) = match extract(a) {
    Some(t) => t,
    None => return false,
  };
  let (lb, opb, rb) = match extract(b) {
    Some(t) => t,
    None => return false,
  };
  // Same variable on the left, same constant on the right.
  if crate::syntax::expr_to_string(&la) != crate::syntax::expr_to_string(&lb)
    || crate::syntax::expr_to_string(&ra) != crate::syntax::expr_to_string(&rb)
  {
    return false;
  }
  matches!(
    (opa, opb),
    (ComparisonOp::LessEqual, ComparisonOp::Greater)
      | (ComparisonOp::Greater, ComparisonOp::LessEqual)
      | (ComparisonOp::Less, ComparisonOp::GreaterEqual)
      | (ComparisonOp::GreaterEqual, ComparisonOp::Less)
  )
}

/// Definite integral of a `Piecewise` with a zero default: integrate each
/// piece's value over the sub-interval of `[lo, hi]` where its condition
/// holds, and sum. Returns `None` (leaving the generic path to try) when the
/// integrand is not such a Piecewise, when any condition is not a simple
/// interval in `var`, when a bound is not numerically comparable, or when a
/// piece's sub-integral cannot be evaluated.
fn try_piecewise_definite_integral(
  integrand: &Expr,
  var: &str,
  lo: &Expr,
  hi: &Expr,
) -> Option<Expr> {
  let (pieces, default) = match integrand {
    Expr::FunctionCall { name, args }
      if name == "Piecewise" && !args.is_empty() =>
    {
      let pieces = match &args[0] {
        Expr::List(p) => p,
        _ => return None,
      };
      (pieces, args.get(1).cloned().unwrap_or(Expr::Integer(0)))
    }
    _ => return None,
  };
  // Only the zero-default case is handled; a non-zero default would need the
  // complement of the pieces (and diverges over an unbounded axis).
  let default_is_zero = matches!(&default, Expr::Integer(0))
    || matches!(&default, Expr::Real(f) if *f == 0.0);
  if !default_is_zero {
    return None;
  }

  let bound_f64 = |e: &Expr| -> Option<f64> {
    crate::functions::math_ast::try_eval_to_f64_with_infinity(e)
  };
  let mut terms: Vec<Expr> = Vec::new();
  for piece in pieces.iter() {
    let (val, cond) = match piece {
      Expr::List(p) if p.len() == 2 => (&p[0], &p[1]),
      _ => return None,
    };
    let (cond_lo, cond_hi) = piecewise_condition_bounds(cond, var)?;
    // Clip the condition's interval to the integration bounds.
    let eff_lo =
      tighter_definite_bound(lo, cond_lo.as_ref(), true, &bound_f64)?;
    let eff_hi =
      tighter_definite_bound(hi, cond_hi.as_ref(), false, &bound_f64)?;
    let (elf, ehf) = (bound_f64(&eff_lo)?, bound_f64(&eff_hi)?);
    if elf >= ehf {
      continue; // empty sub-interval contributes nothing
    }
    let sub = integrate_ast(&[
      val.clone(),
      Expr::List(
        vec![Expr::Identifier(var.to_string()), eff_lo, eff_hi].into(),
      ),
    ])
    .ok()?;
    // If the piece could not be integrated, bail to the generic path.
    if expr_mentions_head(&sub, "Integrate") {
      return None;
    }
    terms.push(sub);
  }
  if terms.is_empty() {
    return Some(Expr::Integer(0));
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  Some(crate::evaluator::evaluate_expr_to_expr(&sum).unwrap_or(sum))
}

/// Resolve a Piecewise condition into an interval `(lower, upper)` for `var`,
/// where `None` on a side means unbounded. Understands single comparisons,
/// uniform chained comparisons (`a < x < b`), the `Inequality[…]` form,
/// `Abs[x] < c`, and `And` intersections.
fn piecewise_condition_bounds(
  cond: &Expr,
  var: &str,
) -> Option<(Option<Expr>, Option<Expr>)> {
  let is_var = |e: &Expr| matches!(e, Expr::Identifier(n) if n == var);
  let is_abs_var = |e: &Expr| {
    matches!(e,
    Expr::FunctionCall { name, args }
      if name == "Abs" && args.len() == 1 && is_var(&args[0]))
  };
  let less =
    |o: ComparisonOp| matches!(o, ComparisonOp::Less | ComparisonOp::LessEqual);
  let greater = |o: ComparisonOp| {
    matches!(o, ComparisonOp::Greater | ComparisonOp::GreaterEqual)
  };
  match cond {
    Expr::Identifier(t) if t == "True" => Some((None, None)),
    Expr::Comparison {
      operands,
      operators,
    } if operators.len() == 1 && operands.len() == 2 => {
      let op = operators[0];
      // Abs[x] < c / <= c -> (-c, c).
      if is_abs_var(&operands[0]) && less(op) {
        let c = operands[1].clone();
        return Some((Some(negate_bound(&c)), Some(c)));
      }
      if is_abs_var(&operands[1]) && greater(op) {
        let c = operands[0].clone();
        return Some((Some(negate_bound(&c)), Some(c)));
      }
      // var OP c, or c OP var (flip the operator).
      let (op, other) = if is_var(&operands[0]) {
        (op, operands[1].clone())
      } else if is_var(&operands[1]) {
        (flip_comparison_op(op), operands[0].clone())
      } else {
        return None;
      };
      if greater(op) {
        Some((Some(other), None))
      } else if less(op) {
        Some((None, Some(other)))
      } else {
        None
      }
    }
    // a < x < b (uniform chained comparison).
    Expr::Comparison {
      operands,
      operators,
    } if operators.len() == 2
      && operands.len() == 3
      && is_var(&operands[1]) =>
    {
      let (o1, o2) = (operators[0], operators[1]);
      if less(o1) && less(o2) {
        Some((Some(operands[0].clone()), Some(operands[2].clone())))
      } else if greater(o1) && greater(o2) {
        Some((Some(operands[2].clone()), Some(operands[0].clone())))
      } else {
        None
      }
    }
    // Inequality[a, op1, x, op2, b] (mixed operators).
    Expr::FunctionCall { name, args }
      if name == "Inequality" && args.len() == 5 && is_var(&args[2]) =>
    {
      let o1 = comparison_op_from_symbol(&args[1])?;
      let o2 = comparison_op_from_symbol(&args[3])?;
      if less(o1) && less(o2) {
        Some((Some(args[0].clone()), Some(args[4].clone())))
      } else {
        None
      }
    }
    // And[…] / a && b: intersect the bounds.
    Expr::FunctionCall { name, args } if name == "And" => {
      let mut lo: Option<Expr> = None;
      let mut hi: Option<Expr> = None;
      for c in args.iter() {
        let (l, h) = piecewise_condition_bounds(c, var)?;
        lo = intersect_bound(lo, l, true)?;
        hi = intersect_bound(hi, h, false)?;
      }
      Some((lo, hi))
    }
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => {
      let (l1, h1) = piecewise_condition_bounds(left, var)?;
      let (l2, h2) = piecewise_condition_bounds(right, var)?;
      Some((
        intersect_bound(l1, l2, true)?,
        intersect_bound(h1, h2, false)?,
      ))
    }
    _ => None,
  }
}

/// Negate a numeric bound expression (`c` -> `-c`), keeping integers exact.
fn negate_bound(c: &Expr) -> Expr {
  match c {
    Expr::Integer(n) => Expr::Integer(-n),
    _ => Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(c.clone()),
    },
  }
}

fn flip_comparison_op(op: ComparisonOp) -> ComparisonOp {
  use ComparisonOp::*;
  match op {
    Less => Greater,
    LessEqual => GreaterEqual,
    Greater => Less,
    GreaterEqual => LessEqual,
    other => other,
  }
}

fn comparison_op_from_symbol(e: &Expr) -> Option<ComparisonOp> {
  use ComparisonOp::*;
  match e {
    Expr::Identifier(s) => match s.as_str() {
      "Less" => Some(Less),
      "LessEqual" => Some(LessEqual),
      "Greater" => Some(Greater),
      "GreaterEqual" => Some(GreaterEqual),
      _ => None,
    },
    _ => None,
  }
}

/// Combine two interval bounds on the same side, keeping the tighter one
/// (larger for the lower bound, smaller for the upper). `None` means unbounded.
fn intersect_bound(
  a: Option<Expr>,
  b: Option<Expr>,
  is_lower: bool,
) -> Option<Option<Expr>> {
  match (a, b) {
    (None, x) | (x, None) => Some(x),
    (Some(a), Some(b)) => {
      let af = crate::functions::math_ast::try_eval_to_f64_with_infinity(&a)?;
      let bf = crate::functions::math_ast::try_eval_to_f64_with_infinity(&b)?;
      let pick_a = if is_lower { af >= bf } else { af <= bf };
      Some(Some(if pick_a { a } else { b }))
    }
  }
}

/// Clip an integration bound against a condition bound, returning the tighter.
/// `is_lower` picks the larger of the two for a lower bound (smaller for upper).
fn tighter_definite_bound(
  int_bound: &Expr,
  cond_bound: Option<&Expr>,
  is_lower: bool,
  bound_f64: &dyn Fn(&Expr) -> Option<f64>,
) -> Option<Expr> {
  match cond_bound {
    None => Some(int_bound.clone()),
    Some(c) => {
      let ib = bound_f64(int_bound)?;
      let cb = bound_f64(c)?;
      let pick_int = if is_lower { ib >= cb } else { ib <= cb };
      Some(if pick_int {
        int_bound.clone()
      } else {
        c.clone()
      })
    }
  }
}

/// True if `expr` contains a `FunctionCall` with the given head anywhere.
fn expr_mentions_head(expr: &Expr, head: &str) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      name == head || args.iter().any(|a| expr_mentions_head(a, head))
    }
    Expr::BinaryOp { left, right, .. } => {
      expr_mentions_head(left, head) || expr_mentions_head(right, head)
    }
    Expr::UnaryOp { operand, .. } => expr_mentions_head(operand, head),
    Expr::List(items) => items.iter().any(|a| expr_mentions_head(a, head)),
    _ => false,
  }
}

/// Check if an expression contains a Piecewise function call anywhere.
fn contains_piecewise(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      if name == "Piecewise" {
        return true;
      }
      args.iter().any(contains_piecewise)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_piecewise(left) || contains_piecewise(right)
    }
    Expr::UnaryOp { operand, .. } => contains_piecewise(operand),
    _ => false,
  }
}

/// Extract (numerator, denominator) from a canonicalized Times expression.
/// Recognizes patterns like Times[Power[den, -1], num] or
/// Times[num, Power[den, -1]] including multi-factor products.
fn extract_quotient_from_times(expr: &Expr) -> Option<(Expr, Expr)> {
  let factors: Vec<&Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().collect()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => vec![left.as_ref(), right.as_ref()],
    _ => return None,
  };

  // Find Power[something, -1] factor(s) — those form the denominator
  let mut den_factors: Vec<Expr> = Vec::new();
  let mut num_factors: Vec<Expr> = Vec::new();

  for factor in &factors {
    // Check for Power[base, negative_integer] — these represent 1/base^|n|
    let inverse_base = match factor {
      Expr::FunctionCall { name, args }
        if name == "Power"
          && args.len() == 2
          && matches!(&args[1], Expr::Integer(n) if *n < 0) =>
      {
        if let Expr::Integer(n) = &args[1] {
          if *n == -1 {
            Some(args[0].clone())
          } else {
            // Power[base, -k] → denominator is Power[base, k]
            Some(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![args[0].clone(), Expr::Integer(-*n)].into(),
            })
          }
        } else {
          None
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if matches!(right.as_ref(), Expr::Integer(n) if *n < 0) => {
        if let Expr::Integer(n) = right.as_ref() {
          if *n == -1 {
            Some(left.as_ref().clone())
          } else {
            Some(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: left.clone(),
              right: Box::new(Expr::Integer(-*n)),
            })
          }
        } else {
          None
        }
      }
      _ => None,
    };

    if let Some(den) = inverse_base {
      den_factors.push(den);
    } else {
      num_factors.push((*factor).clone());
    }
  }

  if den_factors.is_empty() {
    return None; // No denominator found
  }

  let numerator = if num_factors.len() == 1 {
    num_factors.pop().unwrap()
  } else if num_factors.is_empty() {
    Expr::Integer(1)
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: num_factors.into(),
    }
  };

  let denominator = if den_factors.len() == 1 {
    den_factors.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: den_factors.into(),
    }
  };

  Some((numerator, denominator))
}

/// Evaluate an expression numerically at var = point + delta
fn eval_near_point(
  expr: &Expr,
  var: &str,
  point: &Expr,
  delta: f64,
) -> Option<f64> {
  let point_val = crate::functions::math_ast::try_eval_to_f64(point)?;
  let val_at = point_val + delta;
  let subst =
    crate::syntax::substitute_variable(expr, var, &Expr::Real(val_at));
  let result = crate::evaluator::evaluate_expr_to_expr(&subst).ok()?;
  crate::functions::math_ast::try_eval_to_f64(&result)
}

/// Compute a one-sided limit numerically by evaluating at points approaching x0
/// For a one-sided limit, cross-check the direct-substitution value against the
/// numerical one-sided limit. They agree when the function is continuous at the
/// point (return the exact `direct`); they disagree at a jump discontinuity
/// (e.g. Floor, Ceiling, Sign, UnitStep, FractionalPart), where the one-sided
/// limit — not the value at the point — is the answer, so return the numerical
/// result. Continuous cases keep their exact (possibly symbolic) form.
/// Heads that introduce a jump discontinuity (the function is piecewise
/// constant or otherwise steps). A two-sided limit at such a step does not
/// exist even though direct substitution returns a finite value.
fn contains_jump_discontinuous_head(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      matches!(
        name.as_str(),
        "Floor"
          | "Ceiling"
          | "Round"
          | "IntegerPart"
          | "FractionalPart"
          | "Sign"
          | "RealSign"
          | "UnitStep"
          | "HeavisideTheta"
          | "Mod"
          | "Quotient"
          | "KroneckerDelta"
          | "DiscreteDelta"
          | "Boole"
      ) || args.iter().any(contains_jump_discontinuous_head)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_jump_discontinuous_head(left)
        || contains_jump_discontinuous_head(right)
    }
    Expr::UnaryOp { operand, .. } => contains_jump_discontinuous_head(operand),
    _ => false,
  }
}

fn reconcile_one_sided_direct(
  direct: Expr,
  expr: &Expr,
  var_name: &str,
  point: &Expr,
  direction: LimitDirection,
) -> Expr {
  if matches!(direction, LimitDirection::TwoSided) {
    // A finite value from direct substitution is the wrong answer when the
    // expression steps at the point: Limit[Floor[x], x -> 2] is Indeterminate
    // (left -> 1, right -> 2), not Floor[2] = 2. Detect a genuine jump by
    // comparing the two one-sided numerical limits; only override when both
    // sides resolve numerically and clearly disagree.
    if contains_jump_discontinuous_head(expr)
      && let Some(below) = numerical_one_sided_limit(
        expr,
        var_name,
        point,
        LimitDirection::FromBelow,
      )
      && let Some(above) = numerical_one_sided_limit(
        expr,
        var_name,
        point,
        LimitDirection::FromAbove,
      )
      && let (Some(lo), Some(hi)) = (
        crate::functions::math_ast::try_eval_to_f64(&below),
        crate::functions::math_ast::try_eval_to_f64(&above),
      )
      && lo.is_finite()
      && hi.is_finite()
      && (lo - hi).abs() > 1e-6 * (1.0 + lo.abs().max(hi.abs()))
    {
      return Expr::Identifier("Indeterminate".to_string());
    }
    return direct;
  }
  let d_f = match crate::functions::math_ast::try_eval_to_f64(&direct) {
    Some(v) if v.is_finite() => v,
    _ => return direct,
  };
  let num = match numerical_one_sided_limit(expr, var_name, point, direction) {
    Some(n) => n,
    None => return direct,
  };
  let n_f = match crate::functions::math_ast::try_eval_to_f64(&num) {
    Some(v) if v.is_finite() => v,
    _ => return direct,
  };
  let tol = 1e-4 * (1.0 + d_f.abs());
  if (d_f - n_f).abs() <= tol {
    direct
  } else {
    num
  }
}

fn numerical_one_sided_limit(
  expr: &Expr,
  var_name: &str,
  point: &Expr,
  direction: LimitDirection,
) -> Option<Expr> {
  let sign = match direction {
    LimitDirection::FromAbove => 1.0,
    LimitDirection::FromBelow => -1.0,
    LimitDirection::TwoSided => return None,
  };

  // Evaluate at decreasing distances from the point
  let deltas = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12];
  let mut vals = Vec::new();
  for &d in &deltas {
    if let Some(v) = eval_near_point(expr, var_name, point, sign * d) {
      if v.is_nan() {
        return None;
      }
      vals.push(v);
    } else {
      return None;
    }
  }

  // Check for immediate infinity (even at the first sample point)
  if vals.iter().any(|v| v.is_infinite()) {
    // Determine sign from the first non-infinite value, or from the infinite ones
    let sign_positive = vals
      .iter()
      .find(|v| !v.is_infinite())
      .map(|v| *v > 0.0)
      .unwrap_or_else(|| vals[0].is_sign_positive());
    if sign_positive {
      return Some(Expr::Identifier("Infinity".to_string()));
    } else {
      return Some(Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
  }

  // Check if the values are monotonically diverging (sign consistent, magnitude increasing)
  let all_positive = vals.iter().all(|&v| v > 0.0);
  let all_negative = vals.iter().all(|&v| v < 0.0);
  let magnitudes_increasing = vals.windows(2).all(|w| w[1].abs() > w[0].abs());

  if magnitudes_increasing && (all_positive || all_negative) {
    // Check that the growth is unbounded (magnitude at least doubles over the range)
    if vals.last().unwrap().abs() > 2.0 * vals.first().unwrap().abs() {
      if all_positive {
        return Some(Expr::Identifier("Infinity".to_string()));
      } else {
        return Some(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(Expr::Identifier("Infinity".to_string())),
        });
      }
    }
  }

  // Check convergence to a finite value
  let last = *vals.last().unwrap();
  let second_last = vals[vals.len() - 2];
  let diff = (last - second_last).abs();
  let scale = last.abs().max(second_last.abs()).max(1e-15);
  if diff / scale < 0.01 || diff < 1e-10 {
    // Converging — determine the value
    let rounded = last.round();
    if (last - rounded).abs() < 1e-6 {
      return Some(Expr::Integer(rounded as i128));
    }
    // Check for known constants
    if (last - std::f64::consts::E).abs() < 1e-4 {
      return Some(Expr::Constant("E".to_string()));
    }
    if (last - std::f64::consts::PI).abs() < 1e-4 {
      return Some(Expr::Constant("Pi".to_string()));
    }
    return Some(Expr::Real(last));
  }

  None
}

/// Compute a two-sided limit numerically by checking both sides agree
fn numerical_two_sided_limit(
  expr: &Expr,
  var_name: &str,
  point: &Expr,
) -> Option<Expr> {
  let from_above =
    numerical_one_sided_limit(expr, var_name, point, LimitDirection::FromAbove);
  let from_below =
    numerical_one_sided_limit(expr, var_name, point, LimitDirection::FromBelow);

  match (from_above, from_below) {
    (Some(a), Some(b)) => {
      // Check if both sides agree
      let a_val = crate::functions::math_ast::try_eval_to_f64(&a);
      let b_val = crate::functions::math_ast::try_eval_to_f64(&b);
      match (a_val, b_val) {
        (Some(av), Some(bv)) => {
          let diff = (av - bv).abs();
          let scale = av.abs().max(bv.abs()).max(1e-15);
          if diff / scale < 0.01 || diff < 1e-10 {
            // Both sides converge to the same value
            return Some(a);
          }
          // Sides disagree — indeterminate
          Some(Expr::Identifier("Indeterminate".to_string()))
        }
        _ => {
          // At least one side is infinite — check if they match symbolically
          let a_str = crate::syntax::expr_to_string(&a);
          let b_str = crate::syntax::expr_to_string(&b);
          if a_str == b_str {
            return Some(a);
          }
          // Different infinities — indeterminate
          Some(Expr::Identifier("Indeterminate".to_string()))
        }
      }
    }
    _ => None,
  }
}

thread_local! {
  static LIMIT_PRODUCT_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// How a single factor of a product behaves at the limit point.
enum FactorAtPoint {
  Zero,
  Infinite,
  Finite,
  Unknown,
}

/// Substitute `var -> point` into `factor` and classify the resulting value.
fn classify_factor_at(factor: &Expr, var: &str, point: &Expr) -> FactorAtPoint {
  let subst = crate::syntax::substitute_variable(factor, var, point);
  let saved = crate::snapshot_warnings();
  crate::push_quiet();
  let res = crate::evaluator::evaluate_expr_to_expr(&subst);
  crate::pop_quiet();
  crate::restore_warnings(saved);
  let val = match res {
    Ok(v) => v,
    Err(_) => return FactorAtPoint::Unknown,
  };
  if matches!(&val, Expr::Integer(0))
    || matches!(&val, Expr::Real(f) if *f == 0.0)
  {
    FactorAtPoint::Zero
  } else if is_infinity(&val)
    || is_negative_infinity(&val)
    || matches!(&val, Expr::Identifier(n) if n == "ComplexInfinity")
    || matches!(&val, Expr::FunctionCall { name, .. } if name == "DirectedInfinity")
  {
    FactorAtPoint::Infinite
  } else if matches!(&val, Expr::Identifier(n) if n == "Indeterminate") {
    FactorAtPoint::Unknown
  } else if crate::functions::math_ast::try_eval_to_f64(&val)
    .is_some_and(|x| x != 0.0)
    || matches!(&val, Expr::Constant(_))
    || is_numeric_complex_constant(&val)
  {
    FactorAtPoint::Finite
  } else {
    FactorAtPoint::Unknown
  }
}

/// Build a Times of the given factors (or the single factor / 1).
fn product_of(mut factors: Vec<Expr>) -> Expr {
  match factors.len() {
    0 => Expr::Integer(1),
    1 => factors.pop().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: factors.into(),
    },
  }
}

/// Resolve a `0 * Infinity` indeterminate product at a finite point.
///
/// Splits the product into the factor(s) diverging to infinity (`N`) and the
/// remaining factors that go to zero/finite values (`D`), then evaluates the
/// equivalent `Limit[N / (1/D)]` (an Infinity/Infinity form) via one L'Hopital
/// step. Returns `None` when the expression is not a clean `0 * Infinity`
/// product, so the caller can fall back to the numerical path.
fn limit_zero_times_infinity(
  expr: &Expr,
  var: &str,
  point: &Expr,
  rule_arg: &Expr,
) -> Option<Expr> {
  let factors: Vec<Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().cloned().collect()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => vec![left.as_ref().clone(), right.as_ref().clone()],
    _ => return None,
  };
  if factors.len() < 2 {
    return None;
  }

  let mut inf_factors: Vec<Expr> = Vec::new();
  let mut rest_factors: Vec<Expr> = Vec::new();
  let mut saw_zero = false;
  for f in &factors {
    match classify_factor_at(f, var, point) {
      FactorAtPoint::Infinite => inf_factors.push(f.clone()),
      FactorAtPoint::Zero => {
        saw_zero = true;
        rest_factors.push(f.clone());
      }
      FactorAtPoint::Finite => rest_factors.push(f.clone()),
      FactorAtPoint::Unknown => return None,
    }
  }
  // Only a genuine 0 * Infinity form qualifies.
  if !saw_zero || inf_factors.is_empty() || rest_factors.is_empty() {
    return None;
  }

  // Guard against pathological unbounded recursion.
  let depth = LIMIT_PRODUCT_DEPTH.with(|d| d.get());
  if depth >= 12 {
    return None;
  }

  // N = (product of diverging factors), D = (product of the rest, which
  // includes the vanishing factor so D -> 0). A product N * D can be rewritten
  // as a quotient two ways, and the choice decides whether one L'Hopital step
  // resolves it or the derivatives blow up:
  //   * 0/0 form:        D / (1/N)   — differentiate D (the zero) over 1/N
  //   * Infinity/Inf:    N / (1/D)   — differentiate N (the divergence) over 1/D
  // The 0/0 orientation is tried first: for e.g. Tan[Pi x/2] Log[2-x] it gives
  // Log[2-x] / Cot[Pi x/2], which resolves in one step to 2/Pi, whereas the
  // Infinity/Infinity orientation differentiates Tan into ever-larger Sec/Log
  // powers that never resolve. The original orientation is kept as a fallback
  // for forms the 0/0 one cannot close.
  let inf_part = product_of(inf_factors);
  let rest_part = product_of(rest_factors);

  // Build `d/dx[num] / d/dx[1/den]` (one L'Hopital step on `num / (1/den)`),
  // then recurse. Returns None if the step doesn't resolve or blows up.
  let try_orientation = |num: &Expr, den: &Expr| -> Option<Expr> {
    let recip = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(den.clone()),
    };
    let dn = differentiate(num, var).ok()?;
    let d_recip = differentiate(&recip, var).ok()?;
    // Evaluate (not just `simplify`) the quotient so fractional/negative powers
    // recombine — e.g. (1/x)/(-1/2 x^(-3/2)) collapses to -2 Sqrt[x], which the
    // recursive Limit can then resolve by direct substitution.
    let quotient = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(dn),
      right: Box::new(d_recip),
    };
    let new_expr =
      crate::evaluator::evaluate_expr_to_expr(&quotient).unwrap_or(quotient);
    // Skip an orientation whose derivative ratio has already blown up.
    if expr_node_count_capped(&new_expr, 160) >= 160 {
      return None;
    }
    LIMIT_PRODUCT_DEPTH.with(|d| d.set(depth + 1));
    let result = limit_ast(&[new_expr, rule_arg.clone()]);
    LIMIT_PRODUCT_DEPTH.with(|d| d.set(depth));
    match result {
      Ok(val) if !matches!(&val, Expr::FunctionCall { name, .. } if name == "Limit") => {
        Some(val)
      }
      _ => None,
    }
  };

  try_orientation(&rest_part, &inf_part)
    .or_else(|| try_orientation(&inf_part, &rest_part))
}

thread_local! {
  static LIMIT_POWER_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
  static LIMIT_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// RAII guard that decrements the global `limit_ast` recursion counter on every
/// exit path. Bounds repeated L'Hôpital / product-rewrite recursion so a
/// pathological limit (e.g. `Limit[Tan[Pi x/2] Log[2 - x], x -> 1]`, whose
/// derivatives grow without ever resolving) returns unevaluated instead of
/// hanging.
struct LimitDepthGuard;
impl Drop for LimitDepthGuard {
  fn drop(&mut self) {
    LIMIT_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
  }
}

/// Total node count of an expression, capped at `limit` (returns early once the
/// cap is reached). Used to detect L'Hôpital derivative blow-up cheaply.
fn expr_node_count_capped(expr: &Expr, limit: usize) -> usize {
  fn go(expr: &Expr, count: &mut usize, limit: usize) {
    if *count >= limit {
      return;
    }
    *count += 1;
    match expr {
      Expr::FunctionCall { args, .. } | Expr::List(args) => {
        for a in args.iter() {
          go(a, count, limit);
        }
      }
      Expr::BinaryOp { left, right, .. } => {
        go(left, count, limit);
        go(right, count, limit);
      }
      Expr::UnaryOp { operand, .. } => go(operand, count, limit),
      Expr::Rule {
        pattern,
        replacement,
      } => {
        go(pattern, count, limit);
        go(replacement, count, limit);
      }
      _ => {}
    }
  }
  let mut count = 0;
  go(expr, &mut count, limit);
  count
}

/// True for a (resolved) limit value that is `±Infinity`, `ComplexInfinity`, or
/// a `DirectedInfinity[...]` — but NOT `Indeterminate`. Used to classify the
/// exponent/base of a power-form limit.
fn is_infinite_value(expr: &Expr) -> bool {
  is_infinity(expr)
    || is_negative_infinity(expr)
    || matches!(expr, Expr::Identifier(name) if name == "ComplexInfinity")
    || matches!(expr, Expr::FunctionCall { name, .. } if name == "DirectedInfinity")
}

/// Numerically check that `result` is the limit of `fg` (a power `f^g`) as
/// `var -> point`, by sampling `fg` just off the point and comparing. Used to
/// guard the symbolic power-form transform against a mis-evaluated inner limit:
/// only a result that agrees with the actual values near the point is accepted.
fn power_limit_matches_numerically(
  fg: &Expr,
  result: &Expr,
  var: &str,
  point: &Expr,
) -> bool {
  let x0 = match crate::functions::math_ast::try_eval_to_f64(point) {
    Some(v) if v.is_finite() => v,
    _ => return false,
  };
  let target = match crate::functions::math_ast::try_eval_to_f64(result) {
    Some(v) if v.is_finite() => v,
    _ => return false,
  };
  let tol = 0.03 * (1.0 + target.abs()) + 1e-9;
  for &delta in &[1e-5_f64, -1e-5, 1e-4, -1e-4] {
    let sample = Expr::Real(x0 + delta);
    let sub = crate::syntax::substitute_variable(fg, var, &sample);
    if let Some(v) = crate::functions::math_ast::try_eval_to_f64(&sub)
      && v.is_finite()
      && (v - target).abs() <= tol
    {
      return true;
    }
  }
  false
}

/// Resolve an indeterminate power-form limit `Limit[f^g, x -> x0]`. The forms
/// `0^0`, `1^Infinity`, and `Infinity^0` are indeterminate; for each, the limit
/// equals `Exp[Limit[g * Log[f], x -> x0]]`. The base/exponent sub-limits are
/// classified (with `Indeterminate` treated as a possibly-infinite candidate,
/// since Woxi returns Indeterminate for e.g. `Limit[1/x, x -> 0]`), and the
/// transformed result is accepted only after a numerical cross-check so a
/// mis-evaluated inner limit can never produce a wrong answer. Returns None
/// when `expr` is not a power, the form is not a candidate, the transformed
/// limit is not finite, or the numerical check fails.
fn limit_power_form(
  args: &[Expr],
  var_name: &str,
  point: &Expr,
) -> Option<Expr> {
  let (base, exp) = match &args[0] {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => ((**left).clone(), (**right).clone()),
    Expr::FunctionCall { name, args: pargs }
      if name == "Power" && pargs.len() == 2 =>
    {
      (pargs[0].clone(), pargs[1].clone())
    }
    _ => return None,
  };

  // Guard against unbounded re-entry (the transformed limit is itself a Limit).
  let depth = LIMIT_POWER_DEPTH.with(|d| d.get());
  if depth > 4 {
    return None;
  }

  // Build inner-limit argument lists that preserve the point (and direction).
  let inner_args = |e: Expr| {
    let mut v = vec![e, args[1].clone()];
    if args.len() == 3 {
      v.push(args[2].clone());
    }
    v
  };

  LIMIT_POWER_DEPTH.with(|d| d.set(depth + 1));
  let base_lim = limit_ast(&inner_args(base.clone()));
  let exp_lim = limit_ast(&inner_args(exp.clone()));
  LIMIT_POWER_DEPTH.with(|d| d.set(depth));

  let (base_lim, exp_lim) = match (base_lim, exp_lim) {
    (Ok(b), Ok(e)) => (b, e),
    _ => return None,
  };

  let is_indet = |e: &Expr| matches!(e, Expr::Identifier(n) | Expr::Constant(n) if n == "Indeterminate");
  let base_zero = matches!(&base_lim, Expr::Integer(0))
    || matches!(&base_lim, Expr::Real(f) if *f == 0.0);
  let base_one = matches!(&base_lim, Expr::Integer(1))
    || matches!(&base_lim, Expr::Real(f) if *f == 1.0);
  let base_inf = is_infinite_value(&base_lim) || is_indet(&base_lim);
  let exp_zero = matches!(&exp_lim, Expr::Integer(0))
    || matches!(&exp_lim, Expr::Real(f) if *f == 0.0);
  let exp_inf = is_infinite_value(&exp_lim) || is_indet(&exp_lim);

  // Candidate indeterminate forms: 0^0, 1^Infinity, Infinity^0.
  let candidate =
    (base_zero && exp_zero) || (base_one && exp_inf) || (base_inf && exp_zero);
  if !candidate {
    return None;
  }

  // L = Limit[g * Log[f], x -> x0]; the power limit is Exp[L].
  let g_log_f = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      exp,
      Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![base].into(),
      },
    ]
    .into(),
  };
  LIMIT_POWER_DEPTH.with(|d| d.set(depth + 1));
  let l = limit_ast(&inner_args(g_log_f));
  LIMIT_POWER_DEPTH.with(|d| d.set(depth));
  let l = l.ok()?;
  if !is_finite_limit_value(&l) {
    return None;
  }

  let exp_l = Expr::FunctionCall {
    name: "Exp".to_string(),
    args: vec![l].into(),
  };
  let result = crate::evaluator::evaluate_expr_to_expr(&exp_l).ok()?;
  if !power_limit_matches_numerically(&args[0], &result, var_name, point) {
    return None;
  }
  Some(result)
}

/// Limit[expr, x -> x0] - Compute the limit of expr as x approaches x0
/// Limit[expr, x -> x0, Direction -> "FromAbove"] - One-sided limit
/// Rewrite `Derivative[1][Abs][g]` to `Sign[g]` throughout an expression. Used
/// only inside the limit machinery: `D[Abs[x], x]` keeps the unevaluated `Abs'`
/// to match wolframscript, but over the reals `Abs'[g] = Sign[g]`, which the
/// L'Hôpital step needs to resolve quotients like `Abs[x]/x`.
fn abs_deriv_to_sign(expr: &Expr) -> Expr {
  // Detect Derivative[1][Abs][g] == CurriedCall{ CurriedCall{ Derivative[1],
  // [Abs] }, [g] }.
  if let Expr::CurriedCall { func, args } = expr
    && args.len() == 1
    && let Expr::CurriedCall {
      func: inner_func,
      args: inner_args,
    } = func.as_ref()
    && inner_args.len() == 1
    && matches!(&inner_args[0], Expr::Identifier(s) if s == "Abs")
    && let Expr::FunctionCall { name, args: order } = inner_func.as_ref()
    && name == "Derivative"
    && order.len() == 1
    && matches!(&order[0], Expr::Integer(1))
  {
    return Expr::FunctionCall {
      name: "Sign".to_string(),
      args: vec![abs_deriv_to_sign(&args[0])].into(),
    };
  }
  match expr {
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(abs_deriv_to_sign(left)),
      right: Box::new(abs_deriv_to_sign(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(abs_deriv_to_sign(operand)),
    },
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(abs_deriv_to_sign)
        .collect::<Vec<_>>()
        .into(),
    },
    other => other.clone(),
  }
}

/// A function head that Limit may safely substitute through: a structural or
/// arithmetic head, or any built-in function. An undefined user symbol (e.g.
/// `f`) is not — its continuity at the limit point is unknown.
fn limit_head_is_continuous_safe(name: &str) -> bool {
  matches!(
    name,
    "Plus"
      | "Times"
      | "Power"
      | "Subtract"
      | "Divide"
      | "Minus"
      | "Rational"
      | "Complex"
      | "List"
  ) || crate::evaluator::get_builtin_function_info(name).is_some()
}

/// True when `expr` applies an unknown (non-built-in) function to a
/// subexpression involving `var`, so its limit at a finite point can't be found
/// by substitution (`f[x]`, `Sin[f[x]]`, `g[x] h[x]`) — but not `f[0]`, whose
/// argument is free of `var`.
fn contains_unknown_function_of_var(expr: &Expr, var: &str) -> bool {
  use crate::functions::polynomial_ast::contains_var;
  match expr {
    Expr::FunctionCall { name, args } => {
      if !limit_head_is_continuous_safe(name)
        && args.iter().any(|a| contains_var(a, var))
      {
        return true;
      }
      args
        .iter()
        .any(|a| contains_unknown_function_of_var(a, var))
    }
    Expr::CurriedCall { func, args } => {
      contains_unknown_function_of_var(func, var)
        || args
          .iter()
          .any(|a| contains_unknown_function_of_var(a, var))
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_unknown_function_of_var(left, var)
        || contains_unknown_function_of_var(right, var)
    }
    Expr::UnaryOp { operand, .. } => {
      contains_unknown_function_of_var(operand, var)
    }
    Expr::List(items) => items
      .iter()
      .any(|i| contains_unknown_function_of_var(i, var)),
    _ => false,
  }
}

pub fn limit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Limit expects 2 or 3 arguments".into(),
    ));
  }

  // Bound recursion: repeated L'Hôpital / product rewrites can differentiate an
  // expression that never resolves, exploding its size. Past the bound, return
  // the call unevaluated rather than hanging.
  let cur_depth = LIMIT_DEPTH.with(|d| d.get());
  if cur_depth > 16 {
    return Ok(unevaluated("Limit", args));
  }
  LIMIT_DEPTH.with(|d| d.set(cur_depth + 1));
  let _depth_guard = LimitDepthGuard;

  // Parse optional Direction from 3rd argument
  let direction = if args.len() == 3 {
    parse_direction(&args[2]).unwrap_or(LimitDirection::TwoSided)
  } else {
    LimitDirection::TwoSided
  };

  // Second argument must be a Rule: x -> x0
  let (var_name, point) = match &args[1] {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let name = match pattern.as_ref() {
        Expr::Identifier(n) => n.clone(),
        _ => {
          return Ok(unevaluated("Limit", args));
        }
      };
      (name, replacement.as_ref().clone())
    }
    _ => {
      return Ok(unevaluated("Limit", args));
    }
  };

  // The limit of an expression that doesn't involve the limit variable is the
  // expression itself (Limit[Log[a], x -> 0] = Log[a], at any point).
  if !crate::functions::polynomial_ast::contains_var(&args[0], &var_name) {
    return Ok(args[0].clone());
  }

  // A limit whose expression applies an unknown (undefined) function to
  // something involving the limit variable stays unevaluated: the function
  // could be discontinuous at the point, so wolframscript does not substitute.
  // Limit[f[x], x -> 0] is kept, but Limit[f[0] + x, x -> 0] = f[0] (the f[0]
  // argument is a var-free constant, so only the `+ x` term is substituted).
  if contains_unknown_function_of_var(&args[0], &var_name) {
    return Ok(unevaluated("Limit", args));
  }

  // Handle limits at Infinity
  if is_infinity(&point) || is_negative_infinity(&point) {
    // Replace HarmonicNumber[g] (g -> +Infinity) by its asymptotic expansion so
    // the limit resolves symbolically instead of the numeric fallback summing
    // HarmonicNumber at a huge probe value (astronomically slow — a hang).
    if is_infinity(&point)
      && let Some(rewritten) = rewrite_harmonic_asymptotic(&args[0], &var_name)
    {
      // Evaluate first so cancelling Log terms (e.g. Log[n] - Log[n]) collapse
      // before the limit is taken.
      let collapsed = crate::evaluator::evaluate_expr_to_expr(&rewritten)
        .unwrap_or(rewritten);
      return limit_ast(&[collapsed, args[1].clone()]);
    }
    let result = limit_at_infinity(&args[0], &var_name, &point)?;
    // Fallback: FunctionExpand collapses Gamma ratios that the direct
    // asymptotic path does not recognize (e.g. Gamma[x+1]/Gamma[x] -> x, whose
    // limit at Infinity is then Infinity). Retry only when the expansion
    // actually changed the expression, so this cannot loop.
    if matches!(&result, Expr::FunctionCall { name, .. } if name == "Limit")
      && expr_contains_head(&args[0], "Gamma")
    {
      let expanded =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "FunctionExpand".to_string(),
          args: vec![args[0].clone()].into(),
        })?;
      if crate::syntax::expr_to_string(&expanded)
        != crate::syntax::expr_to_string(&args[0])
      {
        return limit_ast(&[expanded, args[1].clone()]);
      }
    }
    return Ok(result);
  }

  // Limit[-f] = -Limit[f]: peel a Minus wrapper (Simplify produces forms
  // like -((Pi - z)/Sin[z])) so the exact L'Hôpital paths apply instead
  // of the low-accuracy numeric fallback.
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = &args[0]
  {
    let mut inner_args = vec![(**operand).clone(), args[1].clone()];
    if args.len() == 3 {
      inner_args.push(args[2].clone());
    }
    let inner = limit_ast(&inner_args)?;
    if !matches!(&inner, Expr::FunctionCall { name, .. } if name == "Limit") {
      return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(inner),
      });
    }
    return Ok(unevaluated("Limit", args));
  }

  // Functions with known poles (Zeta at 1, Gamma at nonpositive integers)
  // get truncated Laurent models so pole-crossing products resolve exactly
  // (Limit[(z-1)*Zeta[z], z -> 1] is 1); without this, the product
  // strategies return a wrong 0 or a low-accuracy numeric probe.
  {
    let mut applied = false;
    let mut gamma_applied = false;
    let rewritten = rewrite_pole_models(
      &args[0],
      &var_name,
      &point,
      &mut applied,
      &mut gamma_applied,
    );
    if applied
      && !(gamma_applied
        && gamma_model_pathological(&args[0], &var_name, &point))
    {
      // Evaluate first (unevaluated PolyGamma constants hang Simplify's
      // rewrite loop), then Simplify so the model's pole factor cancels
      // against the rest of the product ((z+3)*(1/(z+3) + ...) must
      // expand before the limit, or the 0*Infinity strategies fall back
      // to a numeric probe).
      let evaluated = crate::evaluator::evaluate_expr_to_expr(&rewritten)
        .unwrap_or(rewritten);
      let simplified =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Simplify".to_string(),
          args: vec![evaluated.clone()].into(),
        })
        .unwrap_or(evaluated);
      let mut inner_args = vec![simplified, args[1].clone()];
      if args.len() == 3 {
        inner_args.push(args[2].clone());
      }
      return limit_ast(&inner_args);
    }
  }

  // Reciprocal trig forms miss the exact quotient paths (Simplify turns
  // (Pi+z)/Sin[z] into (Pi+z)*Csc[z], which only resolved numerically);
  // rewrite them as Sin/Cos quotients before picking a strategy.
  if let Some(rewritten) = rewrite_reciprocal_trig(&args[0]) {
    let mut inner_args = vec![rewritten, args[1].clone()];
    if args.len() == 3 {
      inner_args.push(args[2].clone());
    }
    return limit_ast(&inner_args);
  }

  // For one-sided limits, skip direct substitution when the expression
  // contains Piecewise, because substituting the exact boundary point may
  // select the wrong branch (e.g. the >= branch instead of the < branch).
  // In that case we fall through to numerical evaluation which correctly
  // approaches from the specified direction.
  let skip_direct_sub = contains_piecewise(&args[0]);

  // Strategy: try direct substitution first.
  // Suppress messages during trial substitution (e.g. Power::infy for Sin[x]/x at x=0)
  if !skip_direct_sub {
    let substituted =
      crate::syntax::substitute_variable(&args[0], &var_name, &point);
    let saved_warnings = crate::snapshot_warnings();
    crate::push_quiet();
    let result = crate::evaluator::evaluate_expr_to_expr(&substituted);
    crate::pop_quiet();
    crate::restore_warnings(saved_warnings);

    match result {
      Ok(ref val) => {
        // Check if the result is a valid numeric value (not Indeterminate, ComplexInfinity, etc.)
        match val {
          Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => {
            return Ok(reconcile_one_sided_direct(
              val.clone(),
              &args[0],
              &var_name,
              &point,
              direction,
            ));
          }
          Expr::FunctionCall { name, args: fargs }
            if name == "Rational" && fargs.len() == 2 =>
          {
            // Check for 0/0 indeterminate form
            if matches!(&fargs[1], Expr::Integer(0)) {
              // Fall through to L'Hôpital
            } else {
              return Ok(reconcile_one_sided_direct(
                val.clone(),
                &args[0],
                &var_name,
                &point,
                direction,
              ));
            }
          }
          Expr::FunctionCall { name, .. }
            if name == "DirectedInfinity" || name == "Indeterminate" =>
          {
            // Fall through to try other methods
          }
          v if is_infinity(v) || is_negative_infinity(v) => {
            // Direct substitution produced a real, directed +/-Infinity
            // (e.g. Log[0] -> -Infinity); that signed value is the limit.
            // ComplexInfinity / DirectedInfinity (an unsigned or genuinely
            // two-sided divergence such as 1/x at 0) are handled by the arm
            // above and stay Indeterminate, so they are not affected.
            return result;
          }
          _ => {
            // Check if it evaluates to a number via N[]
            if crate::functions::math_ast::try_eval_to_f64(val).is_some() {
              return Ok(reconcile_one_sided_direct(
                val.clone(),
                &args[0],
                &var_name,
                &point,
                direction,
              ));
            }
            // Accept a finite complex-number constant (e.g. -I/2). Direct
            // substitution at a point where the expression is continuous is
            // exactly the limit; for complex points the result is a pure
            // numeric-complex form that try_eval_to_f64 can't see.
            if is_numeric_complex_constant(val) {
              return result;
            }
            // A substituted result free of the limit variable is the value at
            // a point of continuity (e.g. Limit[a x, x -> 2] = 2 a). Skip the
            // indeterminate/divergent markers so 0/0, 1/0, … still fall through
            // to L'Hôpital and the other strategies below.
            let is_special_marker = matches!(
              val,
              Expr::Identifier(s) | Expr::Constant(s)
                if s == "Indeterminate"
                  || s == "ComplexInfinity"
                  || s == "Infinity"
                  || s == "Undefined"
            ) || is_infinity(val)
              || is_negative_infinity(val)
              || matches!(val, Expr::FunctionCall { name, .. } if name == "DirectedInfinity" || name == "Indeterminate");
            if !is_special_marker
              && !crate::functions::polynomial_ast::contains_var(val, &var_name)
            {
              return Ok(reconcile_one_sided_direct(
                val.clone(),
                &args[0],
                &var_name,
                &point,
                direction,
              ));
            }
            // Return unevaluated if substitution doesn't yield a clean result
          }
        }
      }
      Err(_) => {
        // Substitution failed (e.g., division by zero)
      }
    }
  }

  // Try L'Hôpital's rule for 0/0 forms
  // Extract numerator and denominator from either BinaryOp::Divide or
  // the canonical Times[Power[den, -1], num] form.
  let num_den: Option<(Expr, Expr)> = if let Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: num,
    right: den,
  } = &args[0]
  {
    Some((*num.clone(), *den.clone()))
  } else {
    extract_quotient_from_times(&args[0])
  };

  if let Some((numerator, denominator)) = num_den {
    let num_at_point =
      crate::syntax::substitute_variable(&numerator, &var_name, &point);
    let den_at_point =
      crate::syntax::substitute_variable(&denominator, &var_name, &point);
    let saved2 = crate::snapshot_warnings();
    crate::push_quiet();
    let num_val = crate::evaluator::evaluate_expr_to_expr(&num_at_point);
    let den_val = crate::evaluator::evaluate_expr_to_expr(&den_at_point);
    crate::pop_quiet();
    crate::restore_warnings(saved2);

    let num_is_zero = matches!(&num_val, Ok(Expr::Integer(0)))
      || matches!(&num_val, Ok(Expr::Real(f)) if *f == 0.0);
    let den_is_zero = matches!(&den_val, Ok(Expr::Integer(0)))
      || matches!(&den_val, Ok(Expr::Real(f)) if *f == 0.0);

    if num_is_zero && den_is_zero {
      // Apply L'Hôpital: Limit[f'/g', x -> x0]
      if let (Ok(df), Ok(dg)) = (
        differentiate(&numerator, &var_name),
        differentiate(&denominator, &var_name),
      ) {
        // Limits run over the reals, where Abs'[g] = Sign[g]. D keeps the
        // unevaluated Derivative[1][Abs][g] (matching wolframscript), so rewrite
        // it here before recursing, otherwise Abs[x]/x-style quotients stall.
        let new_expr = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(abs_deriv_to_sign(&simplify(df))),
          right: Box::new(abs_deriv_to_sign(&simplify(dg))),
        };
        // Bail out of L'Hôpital if the derivative ratio has blown up: some
        // forms (e.g. Tan-based products) differentiate into ever-larger
        // expressions that never resolve, so recursing would only get slower.
        // Fall through to the numerical methods instead.
        if expr_node_count_capped(&new_expr, 160) < 160 {
          // Preserve the Direction option so one-sided limits of quotients
          // that reduce to a discontinuous function (e.g. Abs[x]/x -> Sign[x])
          // resolve to the correct side.
          let mut next_args = vec![new_expr, args[1].clone()];
          if args.len() == 3 {
            next_args.push(args[2].clone());
          }
          return limit_ast(&next_args);
        }
      }
    }
  }

  // Handle the 0 * Infinity indeterminate product at a finite point by
  // rewriting it as an Infinity/Infinity quotient and applying L'Hopital,
  // e.g. Limit[x Log[x], x -> 0] -> Limit[Log[x] / (1/x), x -> 0] = 0.
  // (Genuine quotients are already handled by the 0/0 block above, so this
  // only fires for products that direct substitution left Indeterminate.)
  if let Some(result) =
    limit_zero_times_infinity(&args[0], &var_name, &point, &args[1])
  {
    return Ok(result);
  }

  // Indeterminate power forms f^g (0^0, 1^Infinity, Infinity^0): resolve via
  // Exp[Limit[g Log[f]]] for an exact symbolic result, before the numerical
  // fallback (which would otherwise yield an approximate or wrong value, e.g.
  // (Cos[x])^(1/x^2) -> 1 instead of 1/Sqrt[E]).
  if let Some(result) = limit_power_form(args, &var_name, &point) {
    return Ok(result);
  }

  // Numerical approach for one-sided or two-sided limits
  match direction {
    LimitDirection::FromAbove | LimitDirection::FromBelow => {
      if let Some(result) =
        numerical_one_sided_limit(&args[0], &var_name, &point, direction)
      {
        return Ok(result);
      }
    }
    LimitDirection::TwoSided => {
      if let Some(result) =
        numerical_two_sided_limit(&args[0], &var_name, &point)
      {
        return Ok(result);
      }
    }
  }

  // Return unevaluated
  Ok(unevaluated("Limit", args))
}

/// The limit of `(z - z0)^m * f` used by `residue_ast`. Residue assumes the
/// non-pole part is analytic at `z0`, so when the limit of an unknown-function
/// factor stays symbolic (`Limit[f[z], z -> 0]`, since Limit itself no longer
/// substitutes into undefined functions), the analytic value equals the direct
/// substitution — giving `Residue[f[z]/z, {z, 0}] = f[0]`, `.../z^2 = f'[0]`.
fn analytic_residue_limit(
  expr: &Expr,
  var_name: &str,
  z0: &Expr,
) -> Result<Expr, InterpreterError> {
  let rule = Expr::Rule {
    pattern: Box::new(Expr::Identifier(var_name.to_string())),
    replacement: Box::new(z0.clone()),
  };
  let lim = limit_ast(&[expr.clone(), rule])?;
  if !matches!(&lim, Expr::FunctionCall { name, .. } if name == "Limit") {
    return Ok(lim);
  }
  let sub = crate::syntax::substitute_variable(expr, var_name, z0);
  crate::evaluator::evaluate_expr_to_expr(&sub)
}

/// Returns true if a (resolved) limit result is a finite value — i.e. it is
/// not Infinity / ComplexInfinity / Indeterminate / DirectedInfinity[...] and
/// is not an unresolved `Limit[...]` wrapper. Used by `residue_ast` to detect
/// the order of a pole.
fn is_finite_limit_value(expr: &Expr) -> bool {
  fn scan(expr: &Expr) -> bool {
    match expr {
      Expr::Identifier(s) => {
        !matches!(s.as_str(), "Infinity" | "ComplexInfinity" | "Indeterminate")
      }
      Expr::FunctionCall { name, args } => {
        if matches!(
          name.as_str(),
          "DirectedInfinity" | "Limit" | "Indeterminate"
        ) {
          return false;
        }
        args.iter().all(scan)
      }
      Expr::BinaryOp { left, right, .. } => scan(left) && scan(right),
      Expr::UnaryOp { operand, .. } => scan(operand),
      Expr::List(items) => items.iter().all(scan),
      _ => true,
    }
  }
  scan(expr)
}

/// Returns true if `expr` is a finite complex (or real) number constant — i.e.
/// built only from numeric atoms and the imaginary unit `I` combined with
/// arithmetic operators, containing no free variables and no infinities.
fn is_numeric_complex_constant(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) | Expr::Real(_) => true,
    Expr::Identifier(s) => s == "I",
    Expr::Constant(_) => true,
    Expr::FunctionCall { name, args } => {
      matches!(
        name.as_str(),
        "Rational" | "Complex" | "Times" | "Plus" | "Power"
      ) && args.iter().all(is_numeric_complex_constant)
    }
    Expr::BinaryOp { op, left, right } => {
      matches!(
        op,
        BinaryOperator::Plus
          | BinaryOperator::Minus
          | BinaryOperator::Times
          | BinaryOperator::Divide
          | BinaryOperator::Power
      ) && is_numeric_complex_constant(left)
        && is_numeric_complex_constant(right)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_numeric_complex_constant(operand),
    _ => false,
  }
}

/// Residue[expr, {z, z0}] — the residue of `expr` at the point `z = z0`, i.e.
/// the coefficient of `(z - z0)^(-1)` in the Laurent expansion.
///
/// Strategy: find the order `m` of the pole as the smallest `m >= 1` for which
/// `Limit[(z - z0)^m expr, z -> z0]` is finite, then apply the standard formula
///   Res = 1/(m-1)! * Limit[ d^(m-1)/dz^(m-1) ((z - z0)^m expr), z -> z0 ].
/// If no finite `m` is found within the search bound (e.g. an essential
/// singularity), the call is returned unevaluated.
/// Rewrite reciprocal trig heads (Csc, Sec, Cot and hyperbolic versions)
/// as Sin/Cos quotients. Returns None when nothing changed.
fn rewrite_reciprocal_trig(e: &Expr) -> Option<Expr> {
  fn walk(e: &Expr, changed: &mut bool) -> Expr {
    match e {
      Expr::FunctionCall { name, args } if args.len() == 1 => {
        let arg = walk(&args[0], changed);
        let recip = |head: &str| Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::FunctionCall {
            name: head.to_string(),
            args: vec![arg.clone()].into(),
          }),
          right: Box::new(Expr::Integer(-1)),
        };
        let quot = |num_head: &str, den_head: &str| Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::FunctionCall {
            name: num_head.to_string(),
            args: vec![arg.clone()].into(),
          }),
          right: Box::new(Expr::FunctionCall {
            name: den_head.to_string(),
            args: vec![arg.clone()].into(),
          }),
        };
        match name.as_str() {
          "Csc" => {
            *changed = true;
            recip("Sin")
          }
          "Sec" => {
            *changed = true;
            recip("Cos")
          }
          "Cot" => {
            *changed = true;
            quot("Cos", "Sin")
          }
          "Csch" => {
            *changed = true;
            recip("Sinh")
          }
          "Sech" => {
            *changed = true;
            recip("Cosh")
          }
          "Coth" => {
            *changed = true;
            quot("Cosh", "Sinh")
          }
          _ => Expr::FunctionCall {
            name: name.clone(),
            args: vec![arg].into(),
          },
        }
      }
      Expr::FunctionCall { name, args } => Expr::FunctionCall {
        name: name.clone(),
        args: args
          .iter()
          .map(|a| walk(a, changed))
          .collect::<Vec<_>>()
          .into(),
      },
      Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
        op: *op,
        left: Box::new(walk(left, changed)),
        right: Box::new(walk(right, changed)),
      },
      Expr::UnaryOp { op, operand } => Expr::UnaryOp {
        op: *op,
        operand: Box::new(walk(operand, changed)),
      },
      Expr::List(items) => {
        Expr::List(items.iter().map(|a| walk(a, changed)).collect())
      }
      other => other.clone(),
    }
  }
  let mut changed = false;
  let out = walk(e, &mut changed);
  changed.then_some(out)
}

/// Power of `var` in a monomial `c*var^k` (c free of `var`); None when the
/// expression is not a monomial in `var`.
fn monomial_power_in(e: &Expr, var: &str) -> Option<i128> {
  use BinaryOperator as Op;
  if !crate::functions::polynomial_ast::contains_var(e, var) {
    return Some(0);
  }
  match e {
    Expr::Identifier(v) if v == var => Some(1),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => monomial_power_in(operand, var),
    Expr::BinaryOp {
      op: Op::Power,
      left,
      right,
    } => match right.as_ref() {
      Expr::Integer(k) => Some(monomial_power_in(left, var)? * k),
      _ => None,
    },
    Expr::BinaryOp {
      op: Op::Times,
      left,
      right,
    } => Some(monomial_power_in(left, var)? + monomial_power_in(right, var)?),
    Expr::BinaryOp {
      op: Op::Divide,
      left,
      right,
    } => Some(monomial_power_in(left, var)? - monomial_power_in(right, var)?),
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut total = 0i128;
      for a in args.iter() {
        total += monomial_power_in(a, var)?;
      }
      Some(total)
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      match &args[1] {
        Expr::Integer(k) => Some(monomial_power_in(&args[0], var)? * k),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Entire functions whose composition with a reciprocal monomial is safe for
/// the essential-singularity substitution below.
const ENTIRE_HEADS: [&str; 5] = ["Exp", "Sin", "Cos", "Sinh", "Cosh"];

/// Gate for the essential-singularity substitution `u -> 1/w`: the shifted
/// expression (singularity at `var` = 0) must be built from polynomials in
/// `var`, monomial denominators, and whitelisted entire functions of
/// reciprocal monomials. Any other singular structure (e.g. a 1-z
/// denominator) makes the formal w-series the expansion at infinity, NOT the
/// local Laurent series, so the gate must reject it even though the series
/// machinery would happily expand it. Returns (safe, has_essential).
fn essential_series_safe(e: &Expr, var: &str) -> (bool, bool) {
  use BinaryOperator as Op;
  if !crate::functions::polynomial_ast::contains_var(e, var) {
    return (true, false);
  }
  let both = |a: &Expr, b: &Expr| {
    let (sa, ea) = essential_series_safe(a, var);
    let (sb, eb) = essential_series_safe(b, var);
    (sa && sb, ea || eb)
  };
  match e {
    Expr::Identifier(v) if v == var => (true, false),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => essential_series_safe(operand, var),
    Expr::BinaryOp {
      op: Op::Plus | Op::Minus | Op::Times,
      left,
      right,
    } => both(left, right),
    Expr::BinaryOp {
      op: Op::Divide,
      left,
      right,
    } => {
      if monomial_power_in(right, var).is_some() {
        essential_series_safe(left, var)
      } else {
        (false, false)
      }
    }
    Expr::BinaryOp {
      op: Op::Power,
      left,
      right,
    } => match right.as_ref() {
      Expr::Integer(k) if *k >= 0 => essential_series_safe(left, var),
      Expr::Integer(_) => (monomial_power_in(left, var).is_some(), false),
      exp
        if crate::functions::polynomial_ast::contains_var(exp, var)
        // c^u with a variable exponent: essential iff u is a reciprocal
        // monomial and the base is constant.
        && !crate::functions::polynomial_ast::contains_var(left, var)
          && matches!(monomial_power_in(exp, var), Some(k) if k < 0) =>
      {
        (true, true)
      }
      _ => (false, false),
    },
    Expr::FunctionCall { name, args }
      if (name == "Plus" || name == "Times") && !args.is_empty() =>
    {
      let mut safe = true;
      let mut ess = false;
      for a in args.iter() {
        let (s, e2) = essential_series_safe(a, var);
        safe &= s;
        ess |= e2;
      }
      (safe, ess)
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      essential_series_safe(
        &Expr::BinaryOp {
          op: Op::Power,
          left: Box::new(args[0].clone()),
          right: Box::new(args[1].clone()),
        },
        var,
      )
    }
    Expr::FunctionCall { name, args }
      if ENTIRE_HEADS.contains(&name.as_str()) && args.len() == 1 =>
    {
      if matches!(monomial_power_in(&args[0], var), Some(k) if k < 0) {
        (true, true)
      } else {
        (false, false)
      }
    }
    _ => (false, false),
  }
}

/// A Gamma pole model multiplied by ANOTHER pole at a nonzero point sends
/// the downstream Simplify calls into a pathologically slow polynomial
/// blowup with PolyGamma constants (minutes of compute where wolframscript
/// answers instantly), so those inputs stay unevaluated. Monomial
/// denominators at 0 (Gamma[z]/z^2) and zero factors ((z+3)*Gamma[z])
/// simplify fast and keep the model.
fn gamma_model_pathological(expr: &Expr, var: &str, z0: &Expr) -> bool {
  if matches!(z0, Expr::Integer(0)) {
    return false;
  }
  fn strip_modeled(e: &Expr) -> Expr {
    match e {
      Expr::FunctionCall { name, .. } if name == "Gamma" || name == "Zeta" => {
        Expr::Integer(1)
      }
      Expr::FunctionCall { name, args } => Expr::FunctionCall {
        name: name.clone(),
        args: args.iter().map(strip_modeled).collect::<Vec<_>>().into(),
      },
      Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
        op: *op,
        left: Box::new(strip_modeled(left)),
        right: Box::new(strip_modeled(right)),
      },
      Expr::UnaryOp { op, operand } => Expr::UnaryOp {
        op: *op,
        operand: Box::new(strip_modeled(operand)),
      },
      other => other.clone(),
    }
  }
  let stripped = strip_modeled(expr);
  if !crate::functions::polynomial_ast::contains_var(&stripped, var) {
    return false;
  }
  let saved = crate::snapshot_warnings();
  crate::push_quiet();
  let at_point = crate::evaluator::evaluate_expr_to_expr(
    &crate::syntax::substitute_variable(&stripped, var, z0),
  );
  crate::pop_quiet();
  crate::restore_warnings(saved);
  match at_point {
    Ok(v) => !is_finite_limit_value(&v),
    Err(_) => true,
  }
}

/// Truncated Laurent models for functions with known poles: Zeta at 1 and
/// Gamma at nonpositive integers. Three terms each (u^-1, constant, u), so
/// results are valid for total pole order <= 3; the caller caps the pole
/// order accordingly. Without the models, Limit falls back to a numeric
/// probe near the pole (Residue[Gamma[z], {z, -3}] stayed unevaluated) or
/// even returns a wrong 0 (Residue[Zeta[z], {z, 1}]).
fn rewrite_pole_models(
  e: &Expr,
  var: &str,
  z0: &Expr,
  applied: &mut bool,
  gamma_applied: &mut bool,
) -> Expr {
  use BinaryOperator as Op;
  let value_at = |u: &Expr| -> Option<Expr> {
    crate::evaluator::evaluate_expr_to_expr(
      &crate::syntax::substitute_variable(u, var, z0),
    )
    .ok()
  };
  let plus = |items: Vec<Expr>| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: items.into(),
  };
  let times = |items: Vec<Expr>| Expr::FunctionCall {
    name: "Times".to_string(),
    args: items.into(),
  };
  let pow = |b: Expr, k: i128| Expr::BinaryOp {
    op: Op::Power,
    left: Box::new(b),
    right: Box::new(Expr::Integer(k)),
  };
  if let Expr::FunctionCall { name, args } = e
    && args.len() == 1
    && crate::functions::polynomial_ast::contains_var(&args[0], var)
  {
    let u = &args[0];
    if name == "Zeta" && matches!(value_at(u), Some(Expr::Integer(1))) {
      // Zeta[u] = 1/(u-1) + EulerGamma - StieltjesGamma[1]*(u-1) + O((u-1)^2)
      *applied = true;
      let um1 = plus(vec![u.clone(), Expr::Integer(-1)]);
      return plus(vec![
        pow(um1.clone(), -1),
        Expr::Identifier("EulerGamma".to_string()),
        times(vec![
          Expr::Integer(-1),
          Expr::FunctionCall {
            name: "StieltjesGamma".to_string(),
            args: vec![Expr::Integer(1)].into(),
          },
          um1,
        ]),
      ]);
    }
    if name == "Gamma"
      && let Some(Expr::Integer(n)) = value_at(u)
      && (-20..=0).contains(&n)
    {
      *gamma_applied = true;
      // Gamma[u] near -k: (-1)^k/k! * (1/(u+k) + PolyGamma[0,k+1]
      //   + (u+k)*(PolyGamma[0,k+1]^2 + PolyGamma[1,k+1])/2) + O((u+k)^2)
      *applied = true;
      let k = -n;
      let mut fact: i128 = 1;
      for j in 2..=k {
        fact *= j;
      }
      let sign = if k % 2 == 0 { 1 } else { -1 };
      let upk = plus(vec![u.clone(), Expr::Integer(k)]);
      let psi0 = Expr::FunctionCall {
        name: "PolyGamma".to_string(),
        args: vec![Expr::Integer(0), Expr::Integer(k + 1)].into(),
      };
      let psi1 = Expr::FunctionCall {
        name: "PolyGamma".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(k + 1)].into(),
      };
      // Emit a canonical atom (an uncanonical Rational[-1, 1] wrapper
      // sends downstream Simplify calls into a rewrite loop), and
      // pre-evaluate the constant coefficients: half-evaluated shapes
      // like Times[-1, 1 - EulerGamma] inside the sum also make
      // Simplify's rewrite loop cycle without terminating.
      let coeff = if fact == 1 {
        Expr::Integer(sign)
      } else {
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(sign), Expr::Integer(fact)].into(),
        }
      };
      return times(vec![
        coeff,
        plus(vec![
          pow(upk.clone(), -1),
          psi0.clone(),
          times(vec![
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
            },
            upk,
            plus(vec![pow(psi0, 2), psi1]),
          ]),
        ]),
      ]);
    }
  }
  match e {
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| rewrite_pole_models(a, var, z0, applied, gamma_applied))
        .collect::<Vec<_>>()
        .into(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(rewrite_pole_models(
        left,
        var,
        z0,
        applied,
        gamma_applied,
      )),
      right: Box::new(rewrite_pole_models(
        right,
        var,
        z0,
        applied,
        gamma_applied,
      )),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(rewrite_pole_models(
        operand,
        var,
        z0,
        applied,
        gamma_applied,
      )),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| rewrite_pole_models(a, var, z0, applied, gamma_applied))
        .collect(),
    ),
    other => other.clone(),
  }
}

pub fn residue_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("Residue", args));

  if args.len() != 2 {
    return unevaluated();
  }

  let expr = &args[0];
  let (var_name, z0) = match &args[1] {
    Expr::List(items) if items.len() == 2 => match &items[0] {
      Expr::Identifier(n) => (n.clone(), items[1].clone()),
      _ => return unevaluated(),
    },
    _ => return unevaluated(),
  };

  // --- essential singularities -------------------------------------
  // (z-z0)^m * expr has a finite-order pole for NO m at an essential
  // singularity, and the limit loop below can even return a wrong
  // bounded-oscillation value (z*Sin[1/z] -> 0 while the residue of
  // Sin[1/z] is 1). When the gate holds, substitute z -> z0 + 1/w: the
  // Laurent coefficient c_{-1} at z0 becomes the w^1 series coefficient
  // of the composed expression, which the ordinary series machinery can
  // extract (Exp[1/z] -> 1, z^2*Sin[1/z] -> -1/6).
  {
    let shifted = if matches!(&z0, Expr::Integer(0)) {
      expr.clone()
    } else {
      crate::syntax::substitute_variable(
        expr,
        &var_name,
        &Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Identifier(var_name.clone())),
          right: Box::new(z0.clone()),
        },
      )
    };
    let (series_safe, has_essential) =
      essential_series_safe(&shifted, &var_name);
    if has_essential {
      if series_safe {
        let composed = crate::syntax::substitute_variable(
          &shifted,
          &var_name,
          &Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::Identifier(var_name.clone())),
            right: Box::new(Expr::Integer(-1)),
          },
        );
        let coeff =
          crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "SeriesCoefficient".to_string(),
            args: vec![
              composed,
              Expr::List(
                vec![
                  Expr::Identifier(var_name.clone()),
                  Expr::Integer(0),
                  Expr::Integer(1),
                ]
                .into(),
              ),
            ]
            .into(),
          });
        if let Ok(c) = coeff
          && !matches!(&c, Expr::FunctionCall { name, .. }
            if name == "SeriesCoefficient" || name == "SeriesData")
          && !contains_real(&c)
        {
          return Ok(c);
        }
      }
      // Unsafe or unresolved essential singularity: the pole-order loop
      // would be wrong, not just incomplete.
      return unevaluated();
    }
  }

  // --- known-pole Laurent models -------------------------------------
  let mut model_applied = false;
  let mut gamma_applied = false;
  let modeled = rewrite_pole_models(
    expr,
    &var_name,
    &z0,
    &mut model_applied,
    &mut gamma_applied,
  );
  if model_applied
    && gamma_applied
    && gamma_model_pathological(expr, &var_name, &z0)
  {
    return unevaluated();
  }
  // Evaluate the model before Simplify sees it: unevaluated PolyGamma
  // constants inside the quotient send Simplify's rewrite loop into a
  // non-terminating cycle (pre-existing; evaluation folds them to
  // EulerGamma forms that simplify cleanly).
  let expr = &if model_applied {
    crate::evaluator::evaluate_expr_to_expr(&modeled).unwrap_or(modeled)
  } else {
    expr.clone()
  };

  // Build g = (z - z0)^m * expr.
  let build_g = |m: i128| -> Expr {
    let shift = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::Identifier(var_name.clone())),
      right: Box::new(z0.clone()),
    };
    let powered = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(shift),
      right: Box::new(Expr::Integer(m)),
    };
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(powered),
      right: Box::new(expr.clone()),
    }
  };

  // Fully simplify via the `Simplify` builtin (the internal `simplify` helper
  // does not cancel `(z - z0)^m * p / (z - z0)^m` the way the builtin does;
  // differentiating an uncancelled form would create spurious ∞ − ∞ limits).
  // The input is evaluated first: Simplify's rewrite loop can cycle without
  // terminating on half-canonical constructed trees (e.g. a raw
  // Times[Power[Minus[z, -1], 1], ...] with Times[-1, 1 - EulerGamma]
  // constants inside).
  let simplify_full = |e: Expr| -> Expr {
    let e = crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e);
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Simplify".to_string(),
      args: vec![e].into(),
    })
    .unwrap_or_else(|_| Expr::Identifier("Indeterminate".to_string()))
  };

  // The internal pole-order probes evaluate at the singularity and would
  // print Power::infy etc.; wolframscript's Residue emits no messages.
  struct QuietGuard((Vec<String>, Vec<String>));
  impl Drop for QuietGuard {
    fn drop(&mut self) {
      crate::pop_quiet();
      crate::restore_warnings(std::mem::take(&mut self.0));
    }
  }
  let _quiet = QuietGuard({
    let saved = crate::snapshot_warnings();
    crate::push_quiet();
    saved
  });

  // The truncated Laurent models above carry three terms, so residues are
  // only trustworthy up to total pole order 3 when one was substituted.
  const MAX_ORDER: i128 = 12;
  let max_order = if model_applied { 3 } else { MAX_ORDER };
  for m in 1..=max_order {
    let g = simplify_full(build_g(m));
    let base_limit = analytic_residue_limit(&g, &var_name, &z0)?;
    if !is_finite_limit_value(&base_limit) {
      continue;
    }

    // Pole of order m. For a simple pole (m == 1) the residue is just the
    // limit itself; otherwise differentiate m-1 times before taking the limit.
    let to_limit = if m == 1 {
      g
    } else {
      let mut deriv = g;
      for _ in 0..(m - 1) {
        // Recombine and re-simplify after each derivative: the product
        // rule leaves cancelling pole terms like 1/(z-1) - (z-1)/(z-1)^2
        // spread across the sum, which the limit only resolves
        // numerically otherwise; Together folds them into one quotient.
        let raw = differentiate_expr(&deriv, &var_name)?;
        let together =
          crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "Together".to_string(),
            args: vec![raw.clone()].into(),
          })
          .unwrap_or(raw);
        deriv = simplify_full(together);
      }
      deriv
    };

    let limit_val = analytic_residue_limit(&to_limit, &var_name, &z0)?;
    if !is_finite_limit_value(&limit_val) {
      // Differentiation/limit didn't resolve cleanly — give up.
      return unevaluated();
    }

    // Divide by (m-1)! (a no-op for a simple pole, where (m-1)! == 1).
    let mut fact: i128 = 1;
    for k in 2..m {
      fact *= k;
    }
    let result = if fact == 1 {
      simplify_full(limit_val)
    } else {
      simplify_full(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(limit_val),
        right: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(fact)].into(),
        }),
      })
    };

    // Only emit exact results. A floating-point value means the underlying
    // limit was resolved numerically (e.g. a 0·∞ trig form) and would not
    // match the exact symbolic residue — return unevaluated instead.
    if contains_real(&result) {
      return unevaluated();
    }
    return Ok(result);
  }

  unevaluated()
}

/// Returns true if `expr` contains any floating-point (`Real`) atom.
fn contains_real(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) => true,
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(contains_real)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_real(left) || contains_real(right)
    }
    Expr::UnaryOp { operand, .. } => contains_real(operand),
    _ => false,
  }
}

/// Parse a series coefficient expression into an exact rational `(num, den)`.
/// Returns None for symbolic (non-rational) coefficients.
fn series_coeff_to_rat(e: &Expr) -> Option<(i128, i128)> {
  match e {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => series_coeff_to_rat(operand).map(|(n, d)| (-n, d)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(n), Expr::Integer(d)) if *d != 0 => {
          Some(rat_reduce(*n, *d))
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// Solve a square rational linear system `A x = b` by Gaussian elimination.
/// Returns None if the matrix is singular.
fn rat_linear_solve(
  mut a: Vec<Vec<(i128, i128)>>,
  mut b: Vec<(i128, i128)>,
) -> Option<Vec<(i128, i128)>> {
  let n = b.len();
  for col in 0..n {
    // Find a pivot row with a nonzero entry in this column.
    let pivot = (col..n).find(|&r| a[r][col] != (0, 1))?;
    a.swap(col, pivot);
    b.swap(col, pivot);
    let inv_pivot = rat_div((1, 1), a[col][col]);
    // Normalise the pivot row.
    for j in col..n {
      a[col][j] = rat_mul(a[col][j], inv_pivot);
    }
    b[col] = rat_mul(b[col], inv_pivot);
    // Eliminate this column from every other row.
    for r in 0..n {
      if r == col || a[r][col] == (0, 1) {
        continue;
      }
      let factor = a[r][col];
      for j in col..n {
        a[r][j] =
          rat_add(a[r][j], (-factor.0 * a[col][j].0, factor.1 * a[col][j].1));
        a[r][j] = rat_reduce(a[r][j].0, a[r][j].1);
      }
      b[r] = rat_reduce(
        b[r].0 * (factor.1 * b[col].1) - factor.0 * b[col].0 * b[r].1,
        b[r].1 * (factor.1 * b[col].1),
      );
    }
  }
  Some(b)
}

/// PadeApproximant[f, {x, x0, {m, n}}] — the [m/n] Padé approximant of `f`
/// about `x = x0`: the rational function P(x)/Q(x) with deg P <= m, deg Q <= n,
/// Q(x0) = 1, whose Taylor series agrees with `f` through order m+n.
///
/// Only Padé approximants with exact rational Taylor coefficients and a
/// non-singular denominator system are handled; degenerate (singular) systems
/// — e.g. the Padé of an already-rational function — return unevaluated.
pub fn pade_approximant_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("PadeApproximant", args));
  if args.len() != 2 {
    return unevaluated();
  }
  // Parse {x, x0, {m, n}}.
  let spec = match &args[1] {
    Expr::List(items) if items.len() == 3 => items,
    _ => return unevaluated(),
  };
  let var = match &spec[0] {
    Expr::Identifier(v) => v.clone(),
    _ => return unevaluated(),
  };
  let x0 = spec[1].clone();
  let (m, n) = match &spec[2] {
    Expr::List(mn) if mn.len() == 2 => {
      match (
        crate::functions::math_ast::expr_to_i128(&mn[0]),
        crate::functions::math_ast::expr_to_i128(&mn[1]),
      ) {
        (Some(m), Some(n)) if m >= 0 && n >= 0 => (m, n),
        _ => return unevaluated(),
      }
    }
    // A single non-negative integer n is the diagonal [n/n] approximant,
    // i.e. PadeApproximant[f, {x, x0, n}] == PadeApproximant[f, {x, x0, {n, n}}].
    other => match crate::functions::math_ast::expr_to_i128(other) {
      Some(n) if n >= 0 => (n, n),
      _ => return unevaluated(),
    },
  };
  let order = (m + n) as usize;

  // Taylor coefficients c_0 .. c_{m+n} (of (x - x0)) via Series.
  let series = series_ast(&[
    args[0].clone(),
    Expr::List(
      vec![
        Expr::Identifier(var.clone()),
        x0.clone(),
        Expr::Integer(m + n),
      ]
      .into(),
    ),
  ])?;
  let mut c = vec![(0i128, 1i128); order + 1];
  match &series {
    Expr::FunctionCall { name, args: sd }
      if name == "SeriesData" && sd.len() == 6 =>
    {
      let coeffs = match &sd[2] {
        Expr::List(c) => c,
        _ => return unevaluated(),
      };
      let nmin = match crate::functions::math_ast::expr_to_i128(&sd[3]) {
        Some(v) => v,
        None => return unevaluated(),
      };
      let den = match crate::functions::math_ast::expr_to_i128(&sd[5]) {
        Some(v) => v,
        None => return unevaluated(),
      };
      if den != 1 {
        return unevaluated();
      }
      for (i, cf) in coeffs.iter().enumerate() {
        let k = nmin + i as i128;
        if k < 0 {
          return unevaluated(); // Laurent terms — out of scope.
        }
        if (k as usize) <= order {
          match series_coeff_to_rat(cf) {
            Some(r) => c[k as usize] = r,
            None => return unevaluated(),
          }
        }
      }
    }
    // Series returned a bare value (f free of the variable): constant function.
    other => match series_coeff_to_rat(other) {
      Some(r) => c[0] = r,
      None => return unevaluated(),
    },
  }

  let cval = |i: i128| -> (i128, i128) {
    if i < 0 || i as usize > order {
      (0, 1)
    } else {
      c[i as usize]
    }
  };

  // Denominator coefficients q_1..q_n solve  sum_j c_{m+k-j} q_j = -c_{m+k}.
  let nn = n as usize;
  let mut q_full = vec![(0i128, 1i128); nn + 1];
  q_full[0] = (1, 1);
  if nn > 0 {
    let mut amat = vec![vec![(0i128, 1i128); nn]; nn];
    let mut bvec = vec![(0i128, 1i128); nn];
    for r in 0..nn {
      for s in 0..nn {
        amat[r][s] = cval(m + r as i128 - s as i128);
      }
      let rhs = cval(m + r as i128 + 1);
      bvec[r] = (-rhs.0, rhs.1);
    }
    match rat_linear_solve(amat, bvec) {
      Some(sol) => {
        for (s, &qv) in sol.iter().enumerate() {
          q_full[s + 1] = qv;
        }
      }
      None => return unevaluated(),
    }
  }

  // Numerator p_k = sum_{j=0}^{min(k,n)} q_j c_{k-j}, k = 0..m.
  let mut p = vec![(0i128, 1i128); (m + 1) as usize];
  for (k, pk) in p.iter_mut().enumerate() {
    let mut acc = (0i128, 1i128);
    for j in 0..=nn.min(k) {
      acc = rat_add(acc, rat_mul(q_full[j], cval(k as i128 - j as i128)));
    }
    *pk = acc;
  }

  // Build polynomial expressions in the base (x - x0).
  let base = if matches!(&x0, Expr::Integer(0)) {
    Expr::Identifier(var.clone())
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::Identifier(var.clone())),
      right: Box::new(x0.clone()),
    }
  };
  let build_poly = |coeffs: &[(i128, i128)]| -> Expr {
    let mut terms: Vec<Expr> = Vec::new();
    for (k, &coef) in coeffs.iter().enumerate() {
      if coef == (0, 1) {
        continue;
      }
      let power = match k {
        0 => Expr::Integer(1),
        1 => base.clone(),
        _ => Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(base.clone()),
          right: Box::new(Expr::Integer(k as i128)),
        },
      };
      let term =
        crate::functions::math_ast::times_ast(&[rat_to_expr(coef), power])
          .unwrap_or(Expr::Integer(0));
      terms.push(term);
    }
    if terms.is_empty() {
      return Expr::Integer(0);
    }
    let sum =
      crate::functions::math_ast::plus_ast(&terms).unwrap_or(Expr::Integer(0));
    crate::evaluator::evaluate_expr_to_expr(&sum).unwrap_or(sum)
  };

  let p_expr = build_poly(&p);
  let q_expr = build_poly(&q_full);

  // Result P/Q. Building Times[P, Power[Q, -1]] keeps the displayed form as a
  // ratio of the two polynomials (Woxi does not auto-combine it).
  let result = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(p_expr),
    right: Box::new(q_expr),
  };
  crate::evaluator::evaluate_expr_to_expr(&result).or_else(|_| unevaluated())
}

/// Truncated product of two rational coefficient vectors (indexed by power),
/// keeping terms up to and including order `n`.
fn poly_mul_trunc(
  a: &[(i128, i128)],
  b: &[(i128, i128)],
  n: usize,
) -> Vec<(i128, i128)> {
  let mut out = vec![(0i128, 1i128); n + 1];
  for (i, &ai) in a.iter().enumerate() {
    if i > n || ai == (0, 1) {
      continue;
    }
    for (j, &bj) in b.iter().enumerate() {
      if i + j > n {
        break;
      }
      out[i + j] = rat_add(out[i + j], rat_mul(ai, bj));
    }
  }
  out
}

/// InverseSeries[s] — power-series reversion. Given a `SeriesData` for
/// `y = a1 x + a2 x^2 + ...` (expanded about 0, with `a1 != 0` and no constant
/// term), returns the `SeriesData` for the inverse `x = b1 y + b2 y^2 + ...`
/// such that the composition is the identity.
///
/// The coefficients are found order by order: requiring `[y^n] f(g(y)) = 0`
/// for `n >= 2` isolates `b_n` linearly, since only the `m = 1` term `a1 b_n`
/// involves the unknown — every `g^m` with `m >= 2` reaches `b_n` only at order
/// `> n`. Only series with exact rational coefficients and integer step
/// (`den == 1`) are handled; anything else is returned unevaluated.
pub fn inverse_series_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("InverseSeries", args));
  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }

  let (var, x0, coeffs, nmax, den) = match &args[0] {
    Expr::FunctionCall { name, args: sd }
      if name == "SeriesData" && sd.len() == 6 =>
    {
      let coeffs = match &sd[2] {
        Expr::List(c) => c.clone(),
        _ => return unevaluated(),
      };
      let nmin = match crate::functions::math_ast::expr_to_i128(&sd[3]) {
        Some(v) => v,
        None => return unevaluated(),
      };
      let nmax = match crate::functions::math_ast::expr_to_i128(&sd[4]) {
        Some(v) => v,
        None => return unevaluated(),
      };
      let den = match crate::functions::math_ast::expr_to_i128(&sd[5]) {
        Some(v) => v,
        None => return unevaluated(),
      };
      // Reversion about 0 requires f(0)=0 (no constant term) and a present
      // linear term, i.e. the series starts exactly at x^1.
      if nmin != 1 || den != 1 || !matches!(&sd[1], Expr::Integer(0)) {
        return unevaluated();
      }
      (sd[0].clone(), sd[1].clone(), coeffs, nmax, den)
    }
    _ => return unevaluated(),
  };
  let _ = den;

  let hi = (nmax - 1) as usize; // highest tracked order (coeff of x^hi)
  if hi == 0 {
    return unevaluated();
  }

  // a[k] = coefficient of x^k (a[0] unused). coeffs[i] is the order-(1+i) term.
  let mut a = vec![(0i128, 1i128); hi + 1];
  for (i, c) in coeffs.iter().enumerate() {
    let order = 1 + i;
    if order > hi {
      break;
    }
    match series_coeff_to_rat(c) {
      Some(r) => a[order] = r,
      None => return unevaluated(),
    }
  }
  if a[1] == (0, 1) {
    return unevaluated();
  }

  // b[k] = coefficient of y^k in the inverse series.
  let mut b = vec![(0i128, 1i128); hi + 1];
  b[1] = rat_div((1, 1), a[1]);
  for n in 2..=hi {
    // S = sum_{m=2}^{n} a[m] * [y^n] g^m, using only b[1..n-1].
    let mut gv = vec![(0i128, 1i128); n + 1];
    gv[1..n].copy_from_slice(&b[1..n]);
    let mut gpow = gv.clone(); // g^1
    let mut s = (0i128, 1i128);
    for m in 2..=n {
      gpow = poly_mul_trunc(&gpow, &gv, n); // g^m
      if a[m] != (0, 1) {
        s = rat_add(s, rat_mul(a[m], gpow[n]));
      }
    }
    // 0 = a[1] * b[n] + S  =>  b[n] = -S / a[1].
    b[n] = rat_div((-s.0, s.1), a[1]);
  }

  // Drop trailing zero coefficients (as wolframscript does) while keeping the
  // truncation order nmax unchanged.
  let mut last = hi;
  while last > 1 && b[last] == (0, 1) {
    last -= 1;
  }
  let out_coeffs: Vec<Expr> = (1..=last).map(|k| rat_to_expr(b[k])).collect();
  // The optional second argument renames the series variable in the result.
  let out_var = if args.len() == 2 {
    args[1].clone()
  } else {
    var
  };
  Ok(Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      out_var,
      x0,
      Expr::List(out_coeffs.into()),
      Expr::Integer(1),
      Expr::Integer(nmax),
      Expr::Integer(1),
    ]
    .into(),
  })
}

/// Parsed integer-power SeriesData: variable, center, coefficients (lowest to
/// highest), nmin, nmax. Only `den == 1` series are accepted.
struct ParsedSeries {
  var: Expr,
  center: Expr,
  coeffs: Vec<Expr>,
  nmin: i128,
  nmax: i128,
}

fn parse_integer_series(e: &Expr) -> Option<ParsedSeries> {
  if let Expr::FunctionCall { name, args } = e
    && name == "SeriesData"
    && args.len() == 6
    && let Expr::List(coeffs) = &args[2]
  {
    let nmin = crate::functions::math_ast::expr_to_i128(&args[3])?;
    let nmax = crate::functions::math_ast::expr_to_i128(&args[4])?;
    let den = crate::functions::math_ast::expr_to_i128(&args[5])?;
    if den != 1 {
      return None;
    }
    return Some(ParsedSeries {
      var: args[0].clone(),
      center: args[1].clone(),
      coeffs: coeffs.to_vec(),
      nmin,
      nmax,
    });
  }
  None
}

/// True when a value obtained by substituting the expansion point is a
/// non-finite placeholder (`Indeterminate`, `ComplexInfinity`, a
/// `DirectedInfinity`, or `Infinity`). Such a value means the direct Taylor
/// evaluation has hit a removable singularity or a pole, so the quotient
/// long-division fallback should be tried instead.
fn is_singular_series_value(e: &Expr) -> bool {
  match e {
    Expr::Constant(name) | Expr::Identifier(name) => {
      name == "Indeterminate" || name == "ComplexInfinity" || name == "Infinity"
    }
    Expr::FunctionCall { name, .. } => name == "DirectedInfinity",
    _ => false,
  }
}

/// Split an expression into `(numerator, denominator)` for series long
/// division. Unlike `Numerator`/`Denominator` (which canonicalize, e.g.
/// `1/Sin[x] → Csc[x]` with denominator 1), this keeps the literal quotient so
/// the denominator's vanishing factor is exposed. Handles `a/b`,
/// `Power[base, negative]`, and `Times[...]` with negative-power factors.
fn split_for_series(expr: &Expr) -> Option<(Expr, Expr)> {
  let times_of = |factors: Vec<Expr>| match factors.len() {
    0 => Expr::Integer(1),
    1 => factors.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: factors.into(),
    },
  };
  // Power[base, n] with n < 0 (in any of its AST spellings) → base^(-n) as a
  // denominator factor.
  let neg_power_den = |factor: &Expr| -> Option<Expr> {
    let (base, pos_exp) =
      crate::functions::math_ast::complex::get_negative_power_exponent(factor)?;
    Some(if matches!(pos_exp, Expr::Integer(1)) {
      base
    } else {
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![base, pos_exp].into(),
      }
    })
  };
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => Some(((**left).clone(), (**right).clone())),
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut num_f = Vec::new();
      let mut den_f = Vec::new();
      for a in args.iter() {
        if let Some(d) = neg_power_den(a) {
          den_f.push(d);
        } else {
          num_f.push(a.clone());
        }
      }
      if den_f.is_empty() {
        None
      } else {
        Some((times_of(num_f), times_of(den_f)))
      }
    }
    // Bare reciprocal power, e.g. Power[x, -2] or Sin[x]^-1.
    _ => neg_power_den(expr).map(|den| (Expr::Integer(1), den)),
  }
}

/// Power→coefficient map of an integer-power series, keyed by exponent.
/// Coefficients that are literally zero are dropped so the lowest key is the
/// valuation. A constant (var-independent) expression maps to a single
/// coefficient at power 0. Returns None when the expression is neither an
/// integer-power series nor a constant.
fn series_coeff_map(
  e: &Expr,
  var_name: &str,
) -> Option<std::collections::BTreeMap<i128, Expr>> {
  let mut map = std::collections::BTreeMap::new();
  if let Some(ps) = parse_integer_series(e) {
    for (i, c) in ps.coeffs.iter().enumerate() {
      let power = ps.nmin + i as i128;
      if !matches!(c, Expr::Integer(0)) {
        map.insert(power, c.clone());
      }
    }
    return Some(map);
  }
  // Bare constant (e.g. numerator `1`): series_ast returns the value itself.
  if is_constant_wrt(e, var_name) {
    if !matches!(e, Expr::Integer(0)) {
      map.insert(0, e.clone());
    }
    return Some(map);
  }
  None
}

/// Series of a quotient `num / den` about `x0` to the requested `order`, used
/// when the direct Taylor evaluation fails because numerator and denominator
/// both vanish at `x0` (removable singularity, e.g. `Sin[x]/x`) or only the
/// denominator vanishes (pole, e.g. `Cot[x]`). Expands numerator and
/// denominator into power series and performs power-series long division:
/// `num = den * quotient` solved coefficient by coefficient. Returns None when
/// either piece is not an integer-power series or the numerator is zero.
fn try_series_quotient(
  num: &Expr,
  den: &Expr,
  var_name: &str,
  x0: &Expr,
  order: i128,
) -> Option<Expr> {
  // Expand numerator and denominator to a generous order so the long-division
  // recurrence has enough terms even after the leading powers cancel.
  let guard: i128 = 8;
  let make_args = |f: &Expr| {
    vec![
      f.clone(),
      Expr::List(
        vec![
          Expr::Identifier(var_name.to_string()),
          x0.clone(),
          Expr::Integer(order + guard),
        ]
        .into(),
      ),
    ]
  };
  let num_s = series_ast(&make_args(num)).ok()?;
  let den_s = series_ast(&make_args(den)).ok()?;

  let num_map = series_coeff_map(&num_s, var_name)?;
  let den_map = series_coeff_map(&den_s, var_name)?;

  // Valuations (lowest nonzero power) of numerator and denominator.
  let m = *den_map.keys().next()?; // denominator must be nonzero
  let bm = den_map.get(&m)?.clone();
  let p = match num_map.keys().next() {
    Some(p) => *p,
    None => return Some(Expr::Integer(0)), // numerator identically zero
  };
  let q_min = p - m;

  // Solve num = den * quotient for the quotient coefficients c_l, ascending.
  // c_l = (a_{l+m} - sum_{j>m} b_j c_{l+m-j}) / b_m
  let mut c_map: std::collections::BTreeMap<i128, Expr> =
    std::collections::BTreeMap::new();
  for l in q_min..=order {
    let target = l + m;
    let mut terms: Vec<Expr> =
      vec![num_map.get(&target).cloned().unwrap_or(Expr::Integer(0))];
    for (&j, bj) in den_map.iter() {
      if j == m {
        continue;
      }
      if let Some(cl) = c_map.get(&(target - j)) {
        let prod = crate::functions::math_ast::times_ast(&[
          Expr::Integer(-1),
          bj.clone(),
          cl.clone(),
        ])
        .ok()?;
        terms.push(prod);
      }
    }
    let numer = crate::functions::math_ast::plus_ast(&terms).ok()?;
    let quot = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numer),
      right: Box::new(bm.clone()),
    };
    let cl = crate::evaluator::evaluate_expr_to_expr(&quot).ok()?;
    c_map.insert(l, cl);
  }

  // Assemble the coefficient vector from q_min..=order, then trim leading and
  // trailing zeros the way the direct path does.
  let mut coefficients: Vec<Expr> = (q_min..=order)
    .map(|l| c_map.get(&l).cloned().unwrap_or(Expr::Integer(0)))
    .collect();
  let mut nmin = q_min;
  while !coefficients.is_empty() && matches!(coefficients[0], Expr::Integer(0))
  {
    coefficients.remove(0);
    nmin += 1;
  }
  if coefficients.is_empty() {
    return Some(Expr::Integer(0));
  }
  while coefficients.len() > 1
    && matches!(coefficients.last(), Some(Expr::Integer(0)))
  {
    coefficients.pop();
  }

  Some(Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name.to_string()),
      x0.clone(),
      Expr::List(coefficients.into()),
      Expr::Integer(nmin),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ]
    .into(),
  })
}

/// Truncated product of two power→coefficient maps, dropping terms at power
/// `>= max_power`. Coefficients are arbitrary expressions.
fn series_poly_mul(
  a: &std::collections::BTreeMap<i128, Expr>,
  b: &std::collections::BTreeMap<i128, Expr>,
  max_power: i128,
) -> std::collections::BTreeMap<i128, Expr> {
  let mut out: std::collections::BTreeMap<i128, Vec<Expr>> =
    std::collections::BTreeMap::new();
  for (pa, ca) in a {
    for (pb, cb) in b {
      let p = pa + pb;
      if p >= max_power {
        continue;
      }
      let term =
        crate::functions::math_ast::times_ast(&[ca.clone(), cb.clone()])
          .unwrap_or(Expr::Integer(0));
      out.entry(p).or_default().push(term);
    }
  }
  out
    .into_iter()
    .map(|(p, terms)| {
      let summed = crate::functions::math_ast::plus_ast(&terms)
        .unwrap_or(Expr::Integer(0));
      let canon =
        crate::evaluator::evaluate_expr_to_expr(&summed).unwrap_or(summed);
      (p, canon)
    })
    .collect()
}

/// Compose two integer-power series: `outer(inner(x))`. The inner series must
/// have a zero constant term (nmin >= 1) and the outer must be expanded about
/// 0. Returns None for forms outside this scope.
fn compose_series_pair(outer: &Expr, inner: &Expr) -> Option<Expr> {
  let s1 = parse_integer_series(outer)?;
  let s2 = parse_integer_series(inner)?;

  // Inner must have zero constant term; outer must be expanded about 0.
  if s2.nmin < 1 || !matches!(&s1.center, Expr::Integer(0)) {
    return None;
  }

  // h(x) = inner, as a power→coefficient map (all powers >= 1).
  let mut h_map: std::collections::BTreeMap<i128, Expr> =
    std::collections::BTreeMap::new();
  for (idx, c) in s2.coeffs.iter().enumerate() {
    h_map.insert(s2.nmin + idx as i128, c.clone());
  }
  let nmin_h = s2.nmin;

  // Result truncation order: M = min(nmin_h + nmax1 - 1, nmax2).
  let big_m = (nmin_h + s1.nmax - 1).min(s2.nmax);

  // Accumulate sum_i A_i * h^i, with A_i the coefficient of u^i in the outer
  // series. hpow tracks h^i (starting at h^0 = 1).
  let mut result: std::collections::BTreeMap<i128, Vec<Expr>> =
    std::collections::BTreeMap::new();
  let mut hpow: std::collections::BTreeMap<i128, Expr> =
    std::collections::BTreeMap::new();
  hpow.insert(0, Expr::Integer(1));

  for i in 0..s1.nmax {
    if i >= s1.nmin {
      let a_idx = (i - s1.nmin) as usize;
      if a_idx < s1.coeffs.len() {
        let a_i = &s1.coeffs[a_idx];
        if !matches!(a_i, Expr::Integer(0)) {
          for (p, c) in &hpow {
            if *p < big_m {
              let term = crate::functions::math_ast::times_ast(&[
                a_i.clone(),
                c.clone(),
              ])
              .unwrap_or(Expr::Integer(0));
              result.entry(*p).or_default().push(term);
            }
          }
        }
      }
    }
    // Prepare h^(i+1); stop once its lowest power leaves the truncation.
    if (i + 1) * nmin_h >= big_m {
      break;
    }
    hpow = series_poly_mul(&hpow, &h_map, big_m);
  }

  // Collapse each power's term list to a single canonical coefficient.
  let mut coeff_map: std::collections::BTreeMap<i128, Expr> =
    std::collections::BTreeMap::new();
  for (p, terms) in result {
    let summed =
      crate::functions::math_ast::plus_ast(&terms).unwrap_or(Expr::Integer(0));
    let canon =
      crate::evaluator::evaluate_expr_to_expr(&summed).unwrap_or(summed);
    coeff_map.insert(p, canon);
  }

  // Determine the coefficient range and build a dense list, trimming leading
  // and trailing zeros (matching wolframscript) while keeping nmax = M.
  let is_zero = |e: &Expr| matches!(e, Expr::Integer(0));
  let mut nmin_r = match coeff_map.iter().find(|(_, c)| !is_zero(c)) {
    Some((p, _)) => *p,
    None => big_m, // all zero
  };
  let mut dense: Vec<Expr> = Vec::new();
  if nmin_r < big_m {
    let mut hi = big_m - 1;
    while hi > nmin_r && coeff_map.get(&hi).map(&is_zero).unwrap_or(true) {
      hi -= 1;
    }
    for p in nmin_r..=hi {
      dense.push(coeff_map.get(&p).cloned().unwrap_or(Expr::Integer(0)));
    }
  } else {
    nmin_r = big_m;
  }

  Some(Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      s2.var.clone(),
      s2.center.clone(),
      Expr::List(dense.into()),
      Expr::Integer(nmin_r),
      Expr::Integer(big_m),
      Expr::Integer(1),
    ]
    .into(),
  })
}

/// ComposeSeries[s1, s2, ...] — substitute each series into the previous one,
/// i.e. s1(s2(s3(...))). Inner series must have a zero constant term and the
/// outer ones must be expanded about 0; otherwise the call is unevaluated.
pub fn compose_series_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("ComposeSeries", args));
  if args.len() < 2 {
    return unevaluated();
  }
  // Fold from the right: a(b(c)) = compose(a, compose(b, c)).
  let mut acc = args[args.len() - 1].clone();
  for outer in args[..args.len() - 1].iter().rev() {
    match compose_series_pair(outer, &acc) {
      Some(r) => acc = r,
      None => return unevaluated(),
    }
  }
  Ok(acc)
}

/// Rational arithmetic helpers for coefficient-based series computation
/// Represents a rational number as (numerator, denominator) with denominator > 0
fn rat_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  let num = a.0 * b.1 + b.0 * a.1;
  let den = a.1 * b.1;
  rat_reduce(num, den)
}

fn rat_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  rat_reduce(a.0 * b.0, a.1 * b.1)
}

fn rat_div(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  if b.0 < 0 {
    rat_reduce(-a.0 * b.1, a.1 * -b.0)
  } else {
    rat_reduce(a.0 * b.1, a.1 * b.0)
  }
}

fn rat_reduce(num: i128, den: i128) -> (i128, i128) {
  if num == 0 {
    return (0, 1);
  }
  let g = gcd_i128(num, den);
  let (n, d) = (num / g, den / g);
  if d < 0 { (-n, -d) } else { (n, d) }
}

fn rat_to_expr(r: (i128, i128)) -> Expr {
  if r.1 == 1 {
    Expr::Integer(r.0)
  } else {
    crate::functions::math_ast::make_rational(r.0, r.1)
  }
}

/// Cauchy product of two coefficient vectors (convolution)
fn cauchy_product(
  a: &[(i128, i128)],
  b: &[(i128, i128)],
  n: usize,
) -> (i128, i128) {
  let mut sum = (0i128, 1i128);
  for k in 0..=n {
    if k < a.len() && (n - k) < b.len() {
      sum = rat_add(sum, rat_mul(a[k], b[n - k]));
    }
  }
  sum
}

/// Compute Tan[x] series coefficients around x=0 up to order n
/// Uses recurrence: tan'(x) = 1 + tan²(x), t(0) = 0
fn tan_series_coeffs(order: usize) -> Vec<(i128, i128)> {
  let mut t: Vec<(i128, i128)> = vec![(0, 1)]; // t_0 = 0
  for n in 0..order {
    // (n+1)*t_{n+1} = delta_{n,0} + sum_{k=0..n} t_k * t_{n-k}
    let mut rhs = cauchy_product(&t, &t, n);
    if n == 0 {
      rhs = rat_add(rhs, (1, 1));
    }
    // t_{n+1} = rhs / (n+1)
    t.push(rat_div(rhs, ((n + 1) as i128, 1)));
  }
  t
}

/// Compute Sec[x] series coefficients around x=0 up to order n
/// Uses: sec'(x) = sec(x)*tan(x), sec(0) = 1
/// If s = sec, t = tan: s'= s*t, t' = 1 + t^2 = s^2
fn sec_series_coeffs(order: usize) -> Vec<(i128, i128)> {
  let mut s: Vec<(i128, i128)> = vec![(1, 1)]; // s_0 = 1
  let mut t: Vec<(i128, i128)> = vec![(0, 1)]; // t_0 = 0
  for n in 0..order {
    // (n+1)*t_{n+1} = sum_{k=0..n} s_k * s_{n-k}
    let t_rhs = cauchy_product(&s, &s, n);
    t.push(rat_div(t_rhs, ((n + 1) as i128, 1)));
    // (n+1)*s_{n+1} = sum_{k=0..n} s_k * t_{n-k}
    let s_rhs = cauchy_product(&s, &t, n);
    s.push(rat_div(s_rhs, ((n + 1) as i128, 1)));
  }
  s
}

/// Compute Csc[x] series coefficients around x=0 up to order n
/// csc(x) = 1/sin(x), has a pole at x=0 so series starts at x^{-1}
/// Returns (coefficients, nmin) where nmin is the starting power
fn csc_series_coeffs(order: usize) -> (Vec<(i128, i128)>, i128) {
  // csc(x) = 1/x + x/6 + 7*x^3/360 + ...
  // Use: sin(x)*csc(x) = 1, solve for csc coefficients
  // sin coefficients: s_1 = 1, s_3 = -1/6, s_5 = 1/120, ...
  let total = order + 2; // need extra terms since csc starts at x^{-1}
  let mut sin_c: Vec<(i128, i128)> = Vec::new();
  let mut factorial = 1i128;
  for k in 0..=total {
    if k % 2 == 0 {
      sin_c.push((0, 1));
    } else {
      let sign = if (k / 2) % 2 == 0 { 1 } else { -1 };
      sin_c.push((sign, factorial));
    }
    if k < total {
      factorial *= (k + 1) as i128;
    }
  }
  // Recompute: sin_c[k] = coefficient of x^k in sin(x)
  // We need sin_c as rationals properly
  let mut sin_coeffs: Vec<(i128, i128)> = Vec::new();
  let mut fact = 1i128;
  for k in 0..=total {
    if k > 1 {
      fact *= k as i128;
    }
    if k % 2 == 0 {
      sin_coeffs.push((0, 1));
    } else {
      let sign = if (k / 2) % 2 == 0 { 1i128 } else { -1 };
      sin_coeffs.push(rat_reduce(sign, fact));
    }
  }

  // csc(x) = c_{-1}/x + c_1*x + c_3*x^3 + ...
  // sin(x)*csc(x) = 1
  // Shift: let csc(x) = (1/x) * C(x) where C(x) = c_0 + c_1*x + c_2*x^2 + ...
  // Then sin(x) * C(x) / x = 1, so sin(x)/x * C(x) = 1
  // sinc(x) = sin(x)/x = 1 - x^2/6 + x^4/120 - ...
  // sinc coefficients: sinc_k = sin_coeffs[k+1] (shift by one)
  let mut sinc: Vec<(i128, i128)> = Vec::new();
  for k in 0..=total {
    if k + 1 < sin_coeffs.len() {
      sinc.push(sin_coeffs[k + 1]);
    } else {
      sinc.push((0, 1));
    }
  }

  // C(x) = 1/sinc(x), so sinc*C = 1
  // c_0 = 1/sinc_0 = 1
  // c_n = -(1/sinc_0) * sum_{k=1..n} sinc_k * c_{n-k}
  let mut c: Vec<(i128, i128)> = vec![(1, 1)];
  for n in 1..=total {
    let mut s = (0i128, 1i128);
    for k in 1..=n {
      if k < sinc.len() && (n - k) < c.len() {
        s = rat_add(s, rat_mul(sinc[k], c[n - k]));
      }
    }
    c.push(rat_reduce(-s.0, s.1));
  }

  // csc(x) = c_0/x + c_1 + c_2*x + ... = c_0*x^{-1} + c_1*x^0 + ...
  // In SeriesData format, nmin = -1, coefficients are c_0, c_1, c_2, ...
  // But only keep up to order
  let keep = (order as i128 + 2) as usize; // from x^{-1} to x^{order}
  let coeffs: Vec<(i128, i128)> = c.into_iter().take(keep).collect();
  (coeffs, -1)
}

/// Compute Cot[x] series coefficients around x=0
/// Uses: cot'(x) = -1 - cot²(x) (but cot has pole at 0)
/// cot(x) = cos(x)/sin(x) = 1/x - x/3 - x^3/45 - ...
/// Returns (coefficients, nmin)
fn cot_series_coeffs(order: usize) -> (Vec<(i128, i128)>, i128) {
  // Use cos(x)/sin(x) = cot(x)
  // cos(x) = cot(x)*sin(x)
  // Similar to csc: let cot(x) = (1/x)*C(x)
  // cos(x) = C(x)*sin(x)/x = C(x)*sinc(x)
  let total = order + 2;

  let mut sin_coeffs: Vec<(i128, i128)> = Vec::new();
  let mut cos_coeffs: Vec<(i128, i128)> = Vec::new();
  let mut fact = 1i128;
  for k in 0..=total {
    if k > 1 {
      fact *= k as i128;
    }
    if k % 2 == 0 {
      let sign = if (k / 2) % 2 == 0 { 1i128 } else { -1 };
      cos_coeffs.push(rat_reduce(sign, fact));
      sin_coeffs.push((0, 1));
    } else {
      cos_coeffs.push((0, 1));
      let sign = if (k / 2) % 2 == 0 { 1i128 } else { -1 };
      sin_coeffs.push(rat_reduce(sign, fact));
    }
  }

  // sinc = sin(x)/x
  let mut sinc: Vec<(i128, i128)> = Vec::new();
  for k in 0..=total {
    if k + 1 < sin_coeffs.len() {
      sinc.push(sin_coeffs[k + 1]);
    } else {
      sinc.push((0, 1));
    }
  }

  // C(x)*sinc(x) = cos(x), where cot(x) = C(x)/x
  // c_0 = cos_0/sinc_0 = 1
  // c_n = (cos_n - sum_{k=1..n} sinc_k * c_{n-k}) / sinc_0
  let mut c: Vec<(i128, i128)> = vec![(1, 1)];
  for n in 1..=total {
    let mut s = if n < cos_coeffs.len() {
      cos_coeffs[n]
    } else {
      (0, 1)
    };
    for k in 1..=n {
      if k < sinc.len() && (n - k) < c.len() {
        s = rat_add(s, rat_mul((-sinc[k].0, sinc[k].1), c[n - k]));
      }
    }
    c.push(s);
  }

  let keep = (order as i128 + 2) as usize;
  let coeffs: Vec<(i128, i128)> = c.into_iter().take(keep).collect();
  (coeffs, -1)
}

/// Try coefficient-based series for known functions around x=0.
/// Returns Some((coefficients, nmin)) if handled, None otherwise.
fn try_fast_series(
  expr: &Expr,
  var_name: &str,
  x0: &Expr,
  order: i128,
) -> Option<(Vec<(i128, i128)>, i128)> {
  // Only for expansion around 0
  if !matches!(x0, Expr::Integer(0)) {
    return None;
  }

  // Only for f[var] form
  match expr {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      // Check inner arg is the variable
      if !matches!(&args[0], Expr::Identifier(v) if v == var_name) {
        return None;
      }
      let n = order as usize;
      match name.as_str() {
        "Tan" => Some((tan_series_coeffs(n), 0)),
        "Sec" => Some((sec_series_coeffs(n), 0)),
        "Csc" => Some(csc_series_coeffs(n)),
        "Cot" => Some(cot_series_coeffs(n)),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Series[expr, {x, x0, n}] - Taylor series expansion
/// Normalize a `SeriesData[var, x0, coeffs, nmin, nmax, den]` by trimming
/// leading zero coefficients (advancing `nmin`) and trailing zero coefficients
/// (keeping `nmax`), matching wolframscript. Returns `Some(normalized)` only
/// when this actually changes the expression, so an already-normal SeriesData
/// is left unevaluated (no re-evaluation churn).
pub fn normalize_series_data(args: &[Expr]) -> Option<Expr> {
  if args.len() != 6 {
    return None;
  }
  let Expr::List(items) = &args[2] else {
    return None;
  };
  let nmin = match &args[3] {
    Expr::Integer(n) => *n,
    _ => return None,
  };
  let nmax = match &args[4] {
    Expr::Integer(n) => *n,
    _ => return None,
  };
  let mut coeffs: Vec<Expr> = items.to_vec();
  let mut new_nmin = nmin;
  while !coeffs.is_empty() && matches!(&coeffs[0], Expr::Integer(0)) {
    coeffs.remove(0);
    new_nmin += 1;
  }
  while !coeffs.is_empty() && matches!(coeffs.last(), Some(Expr::Integer(0))) {
    coeffs.pop();
  }
  // All-zero series collapses to the empty O[(var-x0)^nmax] form (nmin = nmax).
  let out_nmin = if coeffs.is_empty() { nmax } else { new_nmin };
  // No change → leave as-is.
  if out_nmin == nmin && coeffs.len() == items.len() {
    return None;
  }
  Some(Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      args[0].clone(),
      args[1].clone(),
      Expr::List(coeffs.into()),
      Expr::Integer(out_nmin),
      args[4].clone(),
      args[5].clone(),
    ]
    .into(),
  })
}

/// Flatten an expression into its additive summands, descending through
/// nested `Plus` (FunctionCall or BinaryOp) and treating `a - b` as
/// `a + (-1)*b`. A non-additive expression yields a single-element vector.
fn additive_terms(expr: &Expr) -> Vec<Expr> {
  use BinaryOperator as B;
  match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().flat_map(additive_terms).collect()
    }
    Expr::BinaryOp {
      op: B::Plus,
      left,
      right,
    } => {
      let mut t = additive_terms(left);
      t.extend(additive_terms(right));
      t
    }
    Expr::BinaryOp {
      op: B::Minus,
      left,
      right,
    } => {
      let mut t = additive_terms(left);
      let neg = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), (**right).clone()].into(),
      };
      t.extend(additive_terms(&neg));
      t
    }
    _ => vec![expr.clone()],
  }
}

/// Flatten an expression into its multiplicative factors, descending through
/// nested `Times` (FunctionCall or BinaryOp). A non-product yields a
/// single-element vector.
fn multiplicative_factors(expr: &Expr) -> Vec<Expr> {
  use BinaryOperator as B;
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().flat_map(multiplicative_factors).collect()
    }
    Expr::BinaryOp {
      op: B::Times,
      left,
      right,
    } => {
      let mut f = multiplicative_factors(left);
      f.extend(multiplicative_factors(right));
      f
    }
    _ => vec![expr.clone()],
  }
}

/// Reduce a fraction `(num, den)` (den > 0) to lowest terms with positive den.
fn reduce_frac(num: i128, den: i128) -> (i128, i128) {
  let g = gcd_i128(num.abs(), den.abs()).max(1);
  let (mut n, mut d) = (num / g, den / g);
  if d < 0 {
    n = -n;
    d = -d;
  }
  (n, d)
}

/// If `f` has a leading non-integer rational power of the shift `(var - x0)`
/// (so that its expansion about `x0` is a genuine Puiseux series), return the
/// reduced exponent `(p, q)` with `q > 1` together with the analytic cofactor
/// `g = f / (var - x0)^(p/q)`. Handles `(var - x0)^(p/q)` directly and products
/// such as `Sqrt[x] Exp[x]`. The power exponents must be explicit integers or
/// `Rational`s; anything symbolic (or a purely integer total exponent) returns
/// `None`. `Sqrt[x]` expanded about a point where it is analytic (e.g. x0 = 1)
/// has no `(var - x0)` power factor and so returns `None`.
fn leading_fractional_power(
  f: &Expr,
  var: &str,
  x0: &Expr,
) -> Option<((i128, i128), Expr)> {
  let pow_exp = |e: &Expr| -> Option<(i128, i128)> {
    match e {
      Expr::Integer(k) => Some((*k, 1)),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        match (&args[0], &args[1]) {
          (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => Some((*p, *q)),
          _ => None,
        }
      }
      _ => None,
    }
  };
  // Whether `e` is the shift base `(var - x0)`: the bare variable when x0 == 0,
  // otherwise any expression that simplifies to `var - x0`.
  let is_base = |e: &Expr| -> bool {
    if matches!(x0, Expr::Integer(0)) {
      return matches!(e, Expr::Identifier(s) if s == var);
    }
    let diff = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        e.clone(),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), Expr::Identifier(var.to_string())]
            .into(),
        },
        x0.clone(),
      ]
      .into(),
    };
    matches!(
      crate::evaluator::evaluate_expr_to_expr(&diff),
      Ok(Expr::Integer(0))
    )
  };
  // Exponent contributed by a single factor that is a power of `(var - x0)`.
  let factor_exp = |fac: &Expr| -> Option<Option<(i128, i128)>> {
    // Outer Option: Some(None) = factor is part of g; Some(Some(r)) = power of
    // the base with exponent r; None = unrepresentable (bail).
    if is_base(fac) {
      return Some(Some((1, 1)));
    }
    match fac {
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 && is_base(&args[0]) =>
      {
        pow_exp(&args[1]).map(Some)
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if is_base(left) => pow_exp(right).map(Some),
      // A factor that is not a power of the base is part of the cofactor g
      // (e.g. Exp[x], 1 + x, Sqrt[1 + x] — all analytic at x0).
      _ => Some(None),
    }
  };

  let factors = multiplicative_factors(f);
  let mut num = 0i128;
  let mut den = 1i128;
  let mut g_factors: Vec<Expr> = Vec::new();
  for fac in &factors {
    match factor_exp(fac)? {
      Some((p, q)) => {
        // num/den += p/q
        num = num * q + p * den;
        den *= q;
        let (rn, rd) = reduce_frac(num, den);
        num = rn;
        den = rd;
      }
      None => g_factors.push(fac.clone()),
    }
  }
  let (p, q) = reduce_frac(num, den);
  if q <= 1 {
    return None; // integer total exponent — not a Puiseux case.
  }
  let g = if g_factors.is_empty() {
    Expr::Integer(1)
  } else if g_factors.len() == 1 {
    g_factors.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: g_factors.into(),
    }
  };
  Some(((p, q), g))
}

pub fn series_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Series expects at least 2 arguments".into(),
    ));
  }

  // Extract options (e.g., Assumptions -> x > 0) from remaining args
  let mut option_args = Vec::new();
  let mut spec_args = Vec::new();
  for arg in &args[2..] {
    if let Expr::Rule { .. } = arg {
      option_args.push(arg.clone());
    } else {
      spec_args.push(arg.clone());
    }
  }

  // Handle multivariate: Series[expr, {x, x0, nx}, {y, y0, ny}, ...]
  if !spec_args.is_empty() {
    // First expand in the first variable (pass options too)
    let mut first_args = vec![args[0].clone(), args[1].clone()];
    first_args.extend(option_args.clone());
    let first_result = series_ast(&first_args)?;
    // Then expand coefficients in each subsequent variable
    let mut result = first_result;
    for spec in &spec_args {
      result = expand_series_data_coefficients(&result, spec)?;
    }
    return Ok(result);
  }

  // Second argument: {x, x0, n}
  let (var_name, x0, order) = match &args[1] {
    Expr::List(items) if items.len() == 3 => {
      let name = match &items[0] {
        Expr::Identifier(n) => n.clone(),
        _ => {
          return Ok(unevaluated("Series", args));
        }
      };
      let order = match &items[2] {
        Expr::Integer(n) => *n,
        _ => {
          return Ok(unevaluated("Series", args));
        }
      };
      (name, items[1].clone(), order)
    }
    _ => {
      return Ok(unevaluated("Series", args));
    }
  };

  // Zeta[var] around var == 1 has a simple pole (residue 1). Its Laurent
  // series is 1/(var-1) + Sum_{n>=0} (-1)^n StieltjesGamma[n]/n! (var-1)^n,
  // with StieltjesGamma[0] = EulerGamma. The generic Taylor path would hit the
  // pole (Zeta[1] = ComplexInfinity) and emit Derivative[Zeta][1] terms.
  if order >= 0
    && matches!(&x0, Expr::Integer(1))
    && let Expr::FunctionCall { name, args: fa } = &args[0]
    && name == "Zeta"
    && fa.len() == 1
    && matches!(&fa[0], Expr::Identifier(v) if *v == var_name)
  {
    let mut coeffs = vec![Expr::Integer(1)]; // order -1: 1/(var-1)
    for n in 0..=order {
      let sign = if n % 2 == 0 { 1 } else { -1 };
      let mut fact: i128 = 1;
      for j in 2..=n {
        fact *= j;
      }
      let sg = Expr::FunctionCall {
        name: "StieltjesGamma".to_string(),
        args: vec![Expr::Integer(n)].into(),
      };
      // (-1)^n StieltjesGamma[n] / n! (StieltjesGamma[0] folds to EulerGamma)
      let coeff = if sign == 1 && fact == 1 {
        sg
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(sign), Expr::Integer(fact)].into(),
            },
            sg,
          ]
          .into(),
        }
      };
      coeffs.push(crate::evaluator::evaluate_expr_to_expr(&coeff)?);
    }
    return Ok(Expr::FunctionCall {
      name: "SeriesData".to_string(),
      args: vec![
        Expr::Identifier(var_name.clone()),
        Expr::Integer(1),
        Expr::List(coeffs.into()),
        Expr::Integer(-1),
        Expr::Integer(order + 1),
        Expr::Integer(1),
      ]
      .into(),
    });
  }

  // If the expression does not depend on the expansion variable, its series
  // is the expression itself (wolframscript returns it directly, with no
  // SeriesData wrapper or O-term).
  if is_constant_wrt(&args[0], &var_name) {
    return crate::evaluator::evaluate_expr_to_expr(&args[0]);
  }

  // Gamma[x] has a simple pole at x = 0; the direct coefficient path samples
  // Gamma[0] = ComplexInfinity and fails. Use the identity Gamma[x] = x!/x:
  // expand the analytic factorial one order higher, then divide by x by
  // shifting every exponent down one integer power (nmin/nmax -= den, the
  // coefficient list stays verbatim). Routing the numerator through the
  // Factorial series keeps the coefficients in wolframscript's canonical form
  // (e.g. (6*EulerGamma^2 + Pi^2)/12), which the generic `.../x` Laurent
  // division would otherwise re-fold into (EulerGamma^2 + Pi^2/6)/2.
  if matches!(&x0, Expr::Integer(0))
    && let Expr::FunctionCall { name, args: ga } = &args[0]
    && name == "Gamma"
    && ga.len() == 1
    && matches!(&ga[0], Expr::Identifier(n) if *n == var_name)
  {
    let fact_series = series_ast(&[
      Expr::FunctionCall {
        name: "Factorial".to_string(),
        args: vec![ga[0].clone()].into(),
      },
      Expr::List(
        vec![ga[0].clone(), Expr::Integer(0), Expr::Integer(order + 1)].into(),
      ),
    ])?;
    if let Expr::FunctionCall { name, args: sd } = &fact_series
      && name == "SeriesData"
      && sd.len() == 6
      && let Expr::Integer(nmin) = &sd[3]
      && let Expr::Integer(nmax) = &sd[4]
      && let Expr::Integer(den) = &sd[5]
    {
      return Ok(Expr::FunctionCall {
        name: "SeriesData".to_string(),
        args: vec![
          sd[0].clone(),
          sd[1].clone(),
          sd[2].clone(),
          Expr::Integer(nmin - den),
          Expr::Integer(nmax - den),
          sd[5].clone(),
        ]
        .into(),
      });
    }
  }

  // Laurent sums: the direct coefficient path samples the integrand near x0
  // and chokes on a summand with a pole there (e.g. `1/x + 1 + x` yields
  // ComplexInfinity coefficients). Series is linear, so when some summand is
  // singular at x0, expand each summand on its own and add the results — each
  // single term (including a bare `1/x`) expands correctly, and SeriesData
  // addition recombines them. Analytic sums (all summands finite at x0) fall
  // through to the normal path unchanged.
  {
    let terms = additive_terms(&args[0]);
    if terms.len() >= 2 {
      let needs_split = terms.iter().any(|t| {
        let at_x0 = crate::syntax::substitute_variable(t, &var_name, &x0);
        // A pole at x0: the direct coefficient sampling chokes on it.
        let has_pole = matches!(
          crate::evaluator::evaluate_expr_to_expr(&at_x0),
          Ok(Expr::Identifier(ref s))
            if s == "ComplexInfinity" || s == "Indeterminate" || s == "Infinity"
        ) || matches!(
          crate::evaluator::evaluate_expr_to_expr(&at_x0),
          Ok(Expr::FunctionCall { ref name, .. }) if name == "DirectedInfinity"
        );
        // A fractional leading power of (x - x0) needs the Puiseux path, which
        // applies to a single term — so split the sum and expand each.
        // Mixed-denominator SeriesData add correctly.
        let is_fractional =
          leading_fractional_power(t, &var_name, &x0).is_some();
        has_pole || is_fractional
      });
      if needs_split {
        let mut expanded: Vec<Expr> = Vec::with_capacity(terms.len());
        for t in &terms {
          let mut term_args = vec![t.clone(), args[1].clone()];
          term_args.extend(option_args.clone());
          expanded.push(series_ast(&term_args)?);
        }
        return crate::evaluator::evaluate_function_call_ast("Plus", &expanded);
      }
    }
  }

  // Puiseux (fractional-power) expansion about x0. The direct path assumes
  // integer powers and evaluates 0^(p/q); instead, factor f = (x-x0)^(p/q) g(x)
  // with g analytic at x0, expand g normally about x0, and shift exponents by
  // p/q. The result is a SeriesData with den = q whose coefficients are g's
  // coefficients interleaved with q-1 zeros (consecutive numerators step by q).
  if let Some(((p, q), g)) = leading_fractional_power(&args[0], &var_name, &x0)
  {
    // Expand g to enough integer orders M so that p/q + M reaches `order`.
    // M = max(0, floor(order - p/q)) = max(0, floor((order*q - p)/q)).
    let m = (((order * q - p) as f64) / q as f64).floor() as i128;
    let m = m.max(0);
    let g_series = series_ast(&[
      g,
      Expr::List(
        vec![
          Expr::Identifier(var_name.clone()),
          x0.clone(),
          Expr::Integer(m),
        ]
        .into(),
      ),
    ])?;
    // Map integer power k -> coefficient g_k (default 0).
    let g_coeff = |k: i128| -> Expr {
      match &g_series {
        Expr::FunctionCall { name, args: sd }
          if name == "SeriesData" && sd.len() == 6 =>
        {
          if let (Expr::List(cs), Some(gnmin)) =
            (&sd[2], crate::functions::math_ast::expr_to_i128(&sd[3]))
          {
            let idx = k - gnmin;
            if idx >= 0 && (idx as usize) < cs.len() {
              return cs[idx as usize].clone();
            }
          }
          Expr::Integer(0)
        }
        // g analytic and free of var (e.g. g == 1): only the k=0 coefficient.
        other => {
          if k == 0 {
            other.clone()
          } else {
            Expr::Integer(0)
          }
        }
      }
    };
    // Build coefficients at consecutive numerators p, p+1, ..., p + m*q.
    let mut coeffs: Vec<Expr> = Vec::with_capacity((m * q + 1) as usize);
    for j in 0..=(m * q) {
      coeffs.push(if j % q == 0 {
        g_coeff(j / q)
      } else {
        Expr::Integer(0)
      });
    }
    // Trim trailing zero coefficients (nmax still records the O-term).
    while coeffs.len() > 1 && matches!(coeffs.last(), Some(Expr::Integer(0))) {
      coeffs.pop();
    }
    let nmax = (order * q).max(p) + 1;
    return Ok(Expr::FunctionCall {
      name: "SeriesData".to_string(),
      args: vec![
        Expr::Identifier(var_name.clone()),
        x0.clone(),
        Expr::List(coeffs.into()),
        Expr::Integer(p),
        Expr::Integer(nmax),
        Expr::Integer(q),
      ]
      .into(),
    });
  }

  // Series expansion at Infinity: substitute x -> 1/t, simplify, expand at
  // t = 0, then relabel the resulting SeriesData to base Infinity (a t^k term
  // at 0 is an x^-k term at Infinity, so the coefficients/exponents carry over
  // unchanged). Restricted to rational functions of the expansion variable so
  // that special functions (ExpIntegralEi, Log, …) keep their dedicated
  // asymptotic handlers further below.
  fn is_rational_function(expr: &Expr) -> bool {
    use {BinaryOperator as B, UnaryOperator as U};
    match expr {
      Expr::Integer(_)
      | Expr::Real(_)
      | Expr::BigInteger(_)
      | Expr::Identifier(_) => true,
      Expr::BinaryOp { op, left, right } => match op {
        B::Plus | B::Minus | B::Times | B::Divide => {
          is_rational_function(left) && is_rational_function(right)
        }
        B::Power => {
          is_rational_function(left)
            && matches!(right.as_ref(), Expr::Integer(_))
        }
        _ => false,
      },
      Expr::UnaryOp {
        op: U::Minus,
        operand,
      } => is_rational_function(operand),
      Expr::FunctionCall { name, args } => match name.as_str() {
        "Plus" | "Times" => args.iter().all(is_rational_function),
        "Power" => {
          args.len() == 2
            && is_rational_function(&args[0])
            && matches!(&args[1], Expr::Integer(_))
        }
        "Rational" => true,
        _ => false,
      },
      _ => false,
    }
  }
  if matches!(&x0, Expr::Identifier(s) if s == "Infinity")
    && is_rational_function(&args[0])
  {
    let temp = format!("{}$si", var_name);
    let recip = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![Expr::Identifier(temp.clone()), Expr::Integer(-1)].into(),
    };
    let substituted =
      crate::syntax::substitute_variable(&args[0], &var_name, &recip);
    // Compute the t = 0 expansion of the substituted form quietly: a function
    // that grows at infinity yields a pole at t = 0, whose intermediate
    // evaluation can emit spurious `Power::infy` messages even though the final
    // SeriesData is correct. wolframscript emits no such message here.
    crate::push_quiet();
    let inner_result: Result<Expr, InterpreterError> = (|| {
      // Cancel[Together[…]] clears the nested 1/t fractions and reduces the
      // result to a single cancelled rational so the t = 0 expansion is clean.
      let simplified =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Cancel".to_string(),
          args: vec![Expr::FunctionCall {
            name: "Together".to_string(),
            args: vec![substituted].into(),
          }]
          .into(),
        })?;
      let spec = Expr::List(
        vec![
          Expr::Identifier(temp.clone()),
          Expr::Integer(0),
          Expr::Integer(order),
        ]
        .into(),
      );
      let mut inner_args = vec![simplified, spec];
      inner_args.extend(option_args.clone());
      series_ast(&inner_args)
    })();
    crate::pop_quiet();
    let inner = inner_result?;
    if let Expr::FunctionCall {
      name: sd_name,
      args: sd,
    } = &inner
      && sd_name == "SeriesData"
      && sd.len() == 6
      && let Expr::List(coeffs) = &sd[2]
      && !coeffs.is_empty()
      && coeffs.iter().all(|c| is_constant_wrt(c, &temp))
    {
      let mut new_sd = sd.to_vec();
      new_sd[0] = Expr::Identifier(var_name.clone());
      new_sd[1] = Expr::Identifier("Infinity".to_string());
      return Ok(Expr::FunctionCall {
        name: "SeriesData".to_string(),
        args: new_sd.into(),
      });
    }
    // Could not produce a clean expansion — leave the call symbolic instead of
    // emitting a bogus SeriesData.
    return Ok(unevaluated("Series", args));
  }

  // Series[QFactorial[n, q], {q, 0, k}] — expand the q-factorial directly
  // as the polynomial product (1+q)·(1+q+q²)·…·(1+q+…+q^{n-1}), truncating
  // at each step. The default Series path tries to expand the rational
  // form (1-q^n)/(1-q) which blows up combinatorially.
  if let Expr::FunctionCall {
    name: qf_name,
    args: qf_args,
  } = &args[0]
    && qf_name == "QFactorial"
    && qf_args.len() == 2
    && matches!(&qf_args[1], Expr::Identifier(v) if v == &var_name)
    && matches!(&x0, Expr::Integer(0))
    && order >= 0
    && let Expr::Integer(n) = &qf_args[0]
    && *n >= 0
  {
    return Ok(qfactorial_series_at_zero(&var_name, *n as usize, order));
  }

  // Series[BarnesG[x], {x, 0, n}] — the direct-differentiation path stalls
  // because BarnesG'[0] is formally indeterminate. Inject the closed-form
  // coefficients from the asymptotic expansion at z = 0 instead.
  if let Expr::FunctionCall {
    name: bname,
    args: bargs,
  } = &args[0]
    && bname == "BarnesG"
    && bargs.len() == 1
    && matches!(&bargs[0], Expr::Identifier(v) if v == &var_name)
    && matches!(&x0, Expr::Integer(0))
    && order >= 1
  {
    return Ok(barnes_g_series_at_zero(&var_name, order));
  }

  // Series[x!, {x, 0, n}] — coefficients involve PolyGamma derivatives at 1
  // (ψ′(1) = π²/6, etc.) that Woxi does not reduce to closed form. Inject the
  // known low-order coefficients so the result matches wolframscript.
  if let Expr::FunctionCall {
    name: fname,
    args: fargs,
  } = &args[0]
    && fname == "Factorial"
    && fargs.len() == 1
    && matches!(&fargs[0], Expr::Identifier(v) if v == &var_name)
    && matches!(&x0, Expr::Integer(0))
    && (0..=2).contains(&order)
  {
    return Ok(factorial_series_at_zero(&var_name, order));
  }

  // Series[x!!, {x, 0, n}] — same issue as Factorial: low-order coefficients
  // involve closed-form combinations of EulerGamma, Log[2], Log[Pi] and Pi^2.
  if let Expr::FunctionCall {
    name: f2name,
    args: f2args,
  } = &args[0]
    && f2name == "Factorial2"
    && f2args.len() == 1
    && matches!(&f2args[0], Expr::Identifier(v) if v == &var_name)
    && matches!(&x0, Expr::Integer(0))
    && (0..=2).contains(&order)
  {
    return Ok(factorial2_series_at_zero(&var_name, order));
  }

  // Series[Hyperfactorial[x], {x, 0, n}] — closed-form low-order Taylor
  // coefficients at 0 involve Log[2*Pi], EulerGamma, and Log[2*Pi]^2.
  if let Expr::FunctionCall {
    name: hfname,
    args: hfargs,
  } = &args[0]
    && hfname == "Hyperfactorial"
    && hfargs.len() == 1
    && matches!(&hfargs[0], Expr::Identifier(v) if v == &var_name)
    && matches!(&x0, Expr::Integer(0))
    && (0..=2).contains(&order)
  {
    return Ok(hyperfactorial_series_at_zero(&var_name, order));
  }

  // Series[Pochhammer[x, 1/2], {x, 0, n}] — closed-form low-order expansion.
  // Pochhammer[x, a] = Gamma[x + a] / Gamma[x] is singular at x = 0, so the
  // direct-Derivative path produces `Pochhammer[0, 1/2]` and partial
  // derivatives that Woxi can't reduce. Inject the known coefficients.
  if let Expr::FunctionCall {
    name: pname,
    args: pargs,
  } = &args[0]
    && pname == "Pochhammer"
    && pargs.len() == 2
    && matches!(&pargs[0], Expr::Identifier(v) if v == &var_name)
    && matches!(&pargs[1], Expr::FunctionCall { name: rn, args: ra }
      if rn == "Rational"
        && ra.len() == 2
        && matches!(&ra[0], Expr::Integer(1))
        && matches!(&ra[1], Expr::Integer(2)))
    && matches!(&x0, Expr::Integer(0))
    && (0..=2).contains(&order)
  {
    return Ok(pochhammer_half_series_at_zero(&var_name, order));
  }

  // Series[FactorialPower[x, n], {x, 0, ord}] — FactorialPower[x, n] with
  // positive integer n is the polynomial x*(x-1)*(x-2)*...*(x-n+1) whose
  // coefficients are signed Stirling numbers of the first kind. The
  // direct-differentiation path leaves `Derivative[k, 0][FactorialPower]`
  // placeholders; inject the polynomial coefficients instead.
  if let Expr::FunctionCall {
    name: fpname,
    args: fpargs,
  } = &args[0]
    && fpname == "FactorialPower"
    && fpargs.len() == 2
    && matches!(&fpargs[0], Expr::Identifier(v) if v == &var_name)
    && let Expr::Integer(n) = &fpargs[1]
    && *n >= 0
    && matches!(&x0, Expr::Integer(0))
    && order >= 0
  {
    return Ok(factorial_power_series_at_zero(&var_name, *n, order));
  }

  // Series[WeberE[v, z], {z, 0, n}] and Series[AngerJ[v, z], {z, 0, n}] —
  // the direct-Derivative path stalls because Woxi can't reduce
  // `Derivative[0, k][WeberE][v, 0]` symbolically. Inject the closed-form
  // coefficients (Cos/Sin and (v^2 - j^2) factors) instead. Skip for integer
  // ν, where the formula has 0/0 indeterminate forms (and AngerJ[n, x] is
  // BesselJ[n, x] anyway).
  if let Expr::FunctionCall {
    name: webname,
    args: webargs,
  } = &args[0]
    && (webname == "WeberE" || webname == "AngerJ")
    && webargs.len() == 2
    && matches!(&webargs[1], Expr::Identifier(v) if v == &var_name)
    && !matches!(&webargs[0], Expr::Integer(_))
    && matches!(&x0, Expr::Integer(0))
    && order >= 0
  {
    let is_weber = webname == "WeberE";
    return Ok(weber_anger_series_at_zero(
      &var_name,
      &webargs[0],
      order,
      is_weber,
    ));
  }

  // LucasL[n, var] with non-integer n: rewrite using the closed-form
  // first dominant root `((x + Sqrt[x^2 + 4])/2)^n`. Series of this
  // form yields the same coefficients as Wolfram's `Series[LucasL[n, x], …]`
  // (the second-root contribution drops out of the real series).
  if let Expr::FunctionCall {
    name: lname,
    args: largs,
  } = &args[0]
    && lname == "LucasL"
    && largs.len() == 2
    && matches!(&largs[1], Expr::Identifier(v) if v == &var_name)
    && !matches!(&largs[0], Expr::Integer(_))
  {
    let n_expr = largs[0].clone();
    let x_expr = largs[1].clone();
    // ((x + Sqrt[x^2 + 4])/2)^n
    let rewritten = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
            },
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                x_expr.clone(),
                Expr::FunctionCall {
                  name: "Sqrt".to_string(),
                  args: vec![Expr::FunctionCall {
                    name: "Plus".to_string(),
                    args: vec![
                      Expr::Integer(4),
                      Expr::FunctionCall {
                        name: "Power".to_string(),
                        args: vec![x_expr, Expr::Integer(2)].into(),
                      },
                    ]
                    .into(),
                  }]
                  .into(),
                },
              ]
              .into(),
            },
          ]
          .into(),
        },
        n_expr,
      ]
      .into(),
    };
    let mut new_args = vec![rewritten, args[1].clone()];
    new_args.extend(option_args.clone());
    return series_ast(&new_args);
  }

  // Try fast coefficient-based computation for known functions
  if let Some((rat_coeffs, nmin)) =
    try_fast_series(&args[0], &var_name, &x0, order)
  {
    let mut coefficients: Vec<Expr> =
      rat_coeffs.iter().map(|r| rat_to_expr(*r)).collect();
    let mut actual_nmin = nmin;

    // Strip leading zeros
    while !coefficients.is_empty()
      && matches!(coefficients[0], Expr::Integer(0))
    {
      coefficients.remove(0);
      actual_nmin += 1;
    }

    // Strip trailing zeros
    while coefficients.len() > 1
      && matches!(coefficients.last(), Some(Expr::Integer(0)))
    {
      coefficients.pop();
    }

    if coefficients.is_empty() {
      return Ok(Expr::Integer(0));
    }

    return Ok(Expr::FunctionCall {
      name: "SeriesData".to_string(),
      args: vec![
        Expr::Identifier(var_name),
        x0,
        Expr::List(coefficients.into()),
        Expr::Integer(actual_nmin),
        Expr::Integer(order + 1),
        Expr::Integer(1),
      ]
      .into(),
    });
  }

  // Fast path for ExpIntegralEi series
  if let Expr::FunctionCall {
    name: fname,
    args: fargs,
  } = &args[0]
    && fname == "ExpIntegralEi"
    && fargs.len() == 1
    && matches!(&fargs[0], Expr::Identifier(v) if v == &var_name)
  {
    if matches!(&x0, Expr::Integer(0)) {
      // Series[ExpIntegralEi[x], {x, 0, n}]
      // Ei(x) = EulerGamma + Log[|x|] + Σ_{k=1}^n x^k / (k * k!)
      // For x > 0: Ei(x) = EulerGamma + Log[x] + ...
      // For x < 0: Ei(x) = EulerGamma + Log[-x] + ...
      // Check Assumptions option for sign of x
      let mut assume_negative = false;
      for opt in &option_args {
        if let Expr::Rule {
          pattern,
          replacement,
        } = opt
          && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Assumptions")
        {
          // Check if the assumption is x < 0
          if let Expr::Comparison {
            operands,
            operators,
          } = replacement.as_ref()
            && operands.len() == 2
            && matches!(&operands[0], Expr::Identifier(v) if v == &var_name)
            && matches!(&operands[1], Expr::Integer(0))
            && operators.len() == 1
            && matches!(&operators[0], ComparisonOp::Less)
          {
            assume_negative = true;
          }
        }
      }

      let log_arg = if assume_negative {
        // Log[-x]
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), Expr::Identifier(var_name.clone())]
            .into(),
        }
      } else {
        // Log[x]
        Expr::Identifier(var_name.clone())
      };
      let c0 = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Identifier("EulerGamma".to_string()),
          Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![log_arg].into(),
          },
        ]
        .into(),
      };
      let mut coefficients = vec![c0];
      let mut factorial: i128 = 1;
      for k in 1..=order {
        factorial *= k;
        // c_k = 1 / (k * k!)
        coefficients.push(rat_to_expr((1, k * factorial)));
      }

      return Ok(Expr::FunctionCall {
        name: "SeriesData".to_string(),
        args: vec![
          Expr::Identifier(var_name),
          x0,
          Expr::List(coefficients.into()),
          Expr::Integer(0),
          Expr::Integer(order + 1),
          Expr::Integer(1),
        ]
        .into(),
      });
    }

    if matches!(&x0, Expr::Identifier(s) if s == "Infinity" || s == "DirectedInfinity")
      || matches!(&x0, Expr::FunctionCall { name, args: a } if name == "DirectedInfinity" && a.len() == 1 && matches!(&a[0], Expr::Integer(1)))
    {
      // Series[ExpIntegralEi[x], {x, Infinity, n}]
      // Asymptotic expansion: Ei(x) ~ E^x/x * Σ_{k=0}^{n-1} k!/x^k + regularization
      // The result in terms of SeriesData at Infinity:
      // SeriesData[x, Infinity, {coeffs...}, -1, -(n+1), -1]
      // where coefficients are k! (factorials)
      // But the output format from Wolfram is specific. Let me construct it differently.
      // Actually, Wolfram returns it as a proper SeriesData which // Normal gives the expression.
      //
      // Normal of the asymptotic expansion gives:
      // E^x * Sum[k!/x^(k+1), {k, 0, n-1}] + (Log[-1/x] - Log[-x] + 2*Log[x])/2
      //
      // Let me construct the SeriesData directly.
      // SeriesData[x, Infinity, {1, 1, 2, 6, 24, 120, ...}, 1, n+1, 1]
      // where the coefficients are k! and the powers are x^(-k-1)
      // The convention for series at infinity is different.

      // Build the Normal form directly since the SeriesData at infinity is complex
      // E^x*(n!/x^(n+1) + ... + 1/x^2 + 1/x) + (Log[-1/x] - Log[-x] + 2*Log[x])/2
      let mut exp_terms = Vec::new();
      let mut fact: i128 = 1;
      for k in 0..order {
        if k > 0 {
          fact *= k;
        }
        // k!/x^(k+1) = fact * Power[x, -(k+1)]
        let power = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::Identifier(var_name.clone()),
            Expr::Integer(-(k + 1)),
          ]
          .into(),
        };
        if fact == 1 {
          exp_terms.push(power);
        } else {
          exp_terms.push(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(fact), power].into(),
          });
        }
      }
      // Reverse to show highest power first (matching Wolfram output order)
      exp_terms.reverse();

      // E^x * (sum of terms)
      let exp_x = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          Expr::Constant("E".to_string()),
          Expr::Identifier(var_name.clone()),
        ]
        .into(),
      };
      let exp_part = Expr::FunctionCall {
        name: "Times".to_string(),
        args: {
          let mut a = vec![exp_x];
          if exp_terms.len() == 1 {
            a.push(exp_terms.into_iter().next().unwrap());
          } else {
            a.push(Expr::FunctionCall {
              name: "Plus".to_string(),
              args: exp_terms.into(),
            });
          }
          a.into()
        },
      };

      // Regularization term: (Log[-1/x] - Log[-x] + 2*Log[x])/2
      let log_neg_inv_x = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(-1),
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![Expr::Identifier(var_name.clone()), Expr::Integer(-1)]
                .into(),
            },
          ]
          .into(),
        }]
        .into(),
      };
      let log_neg_x = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), Expr::Identifier(var_name.clone())]
            .into(),
        }]
        .into(),
      };
      let two_log_x = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(2),
          Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![Expr::Identifier(var_name.clone())].into(),
          },
        ]
        .into(),
      };
      let log_sum = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          log_neg_inv_x,
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), log_neg_x].into(),
          },
          two_log_x,
        ]
        .into(),
      };
      let reg_term = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(log_sum),
        right: Box::new(Expr::Integer(2)),
      };

      let result = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(exp_part),
        right: Box::new(reg_term),
      };
      return Ok(result);
    }
  }

  // Quotient with a removable singularity or pole at x0: the direct Taylor
  // loop below evaluates f^(k)(x0) by substitution, which yields Indeterminate
  // or ComplexInfinity when numerator and denominator both vanish (e.g.
  // Sin[x]/x → 1 - x^2/6 + ...). Detect that case and fall back to
  // power-series long division of numerator by denominator.
  {
    let probe = crate::syntax::substitute_variable(&args[0], &var_name, &x0);
    // Evaluate the probe quietly: substituting x0 into a removable singularity
    // (e.g. Sin[x]/x → 0/0) intentionally produces ComplexInfinity/Indeterminate
    // here, which would otherwise emit spurious `Power::infy`/`Infinity::indet`
    // messages even though we recover the correct series below. wolframscript
    // emits no such message for these expansions. Snapshot/restore the message
    // buffers around the probe (push_quiet only silences printing, not capture).
    let snapshot = crate::snapshot_warnings();
    crate::push_quiet();
    let probe_val = crate::evaluator::evaluate_expr_to_expr(&probe);
    crate::pop_quiet();
    crate::restore_warnings(snapshot);
    if let Ok(val) = probe_val
      && is_singular_series_value(&val)
      && let Some((num, den)) = split_for_series(&args[0])
      && let Some(series) =
        try_series_quotient(&num, &den, &var_name, &x0, order)
    {
      return Ok(series);
    }
  }

  // Compute Taylor coefficients: f^(k)(x0) / k!
  let mut coefficients = Vec::new();
  let mut current_expr = args[0].clone();

  for k in 0..=order {
    // Evaluate current derivative at x0
    let substituted =
      crate::syntax::substitute_variable(&current_expr, &var_name, &x0);
    let value = crate::evaluator::evaluate_expr_to_expr(&substituted)?;

    // Compute k!
    let mut factorial = 1i128;
    for i in 2..=k {
      factorial *= i;
    }

    // Coefficient = value / k!
    let coeff = if matches!(&value, Expr::Integer(0)) {
      Expr::Integer(0)
    } else if factorial == 1 {
      value
    } else {
      // value / factorial
      match &value {
        Expr::Integer(n) => {
          let g = gcd_i128(*n, factorial);
          let (num, den) = (n / g, factorial / g);
          if den == 1 {
            Expr::Integer(num)
          } else {
            crate::functions::math_ast::make_rational(num, den)
          }
        }
        // Handle Rational[n, d] / factorial → Rational[n, d*factorial] simplified
        Expr::FunctionCall { name, args: rargs }
          if name == "Rational"
            && rargs.len() == 2
            && matches!(&rargs[0], Expr::Integer(_))
            && matches!(&rargs[1], Expr::Integer(_)) =>
        {
          if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1]) {
            crate::functions::math_ast::make_rational(*n, d * factorial)
          } else {
            Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(value),
              right: Box::new(Expr::Integer(factorial)),
            }
          }
        }
        _ => {
          // value / factorial — fold the factorial into any leading
          // numeric coefficient via times_ast so e.g.
          // `Times[Rational[1, 2], Derivative[…]] / 2` collapses to
          // `Times[Rational[1, 4], Derivative[…]]` rather than leaving a
          // `BinaryOp::Divide` outside.
          let inv = Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(factorial)].into(),
          };
          let val_clone = value.clone();
          crate::functions::math_ast::times_ast(&[value, inv]).unwrap_or(
            Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(val_clone),
              right: Box::new(Expr::Integer(factorial)),
            },
          )
        }
      }
    };

    // Re-evaluate the coefficient so nested rational sub-expressions
    // collapse to a single canonical fraction. Without this the Taylor
    // expansion of `((x + Sqrt[x^2 + 4])/2)^(3/2)` keeps coefficients
    // like `((3*1)/8 + (3*1/2)/8)/2` instead of `9/32`, and so e.g.
    // `Series[LucasL[3/2, x], {x, 0, 5}]` doesn't match wolframscript.
    let coeff_simplified =
      crate::evaluator::evaluate_expr_to_expr(&coeff).unwrap_or(coeff);
    coefficients.push(coeff_simplified);

    // Differentiate for the next iteration (unless this is the last)
    if k < order {
      current_expr = match differentiate(&current_expr, &var_name) {
        Ok(d) => simplify(d),
        Err(_) => {
          return Ok(unevaluated("Series", args));
        }
      };
    }
  }

  // Strip leading zero coefficients and adjust nmin
  let mut nmin: i128 = 0;
  while !coefficients.is_empty() && matches!(coefficients[0], Expr::Integer(0))
  {
    coefficients.remove(0);
    nmin += 1;
  }

  // If all coefficients are zero, return 0
  if coefficients.is_empty() {
    return Ok(Expr::Integer(0));
  }

  // Strip trailing zero coefficients (wolframscript drops them from the
  // coefficient list while keeping the truncation order nmax). The leading
  // strip above guarantees the first element is nonzero, so this never empties
  // the list.
  while coefficients.len() > 1
    && matches!(coefficients.last(), Some(Expr::Integer(0)))
  {
    coefficients.pop();
  }

  // Build SeriesData[x, x0, {c0, c1, ...}, nmin, nmax, 1]
  Ok(Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name),
      x0,
      Expr::List(coefficients.into()),
      Expr::Integer(nmin),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ]
    .into(),
  })
}

/// Expand each coefficient of a SeriesData in a new variable.
/// Used for multivariate Series: each coefficient becomes a SeriesData in the new variable.
fn expand_series_data_coefficients(
  series: &Expr,
  spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // series should be SeriesData[var, x0, {coeffs}, nmin, nmax, den]
  if let Expr::FunctionCall { name, args } = series
    && name == "SeriesData"
    && args.len() == 6
    && let Expr::List(coeffs) = &args[2]
  {
    // Expand each coefficient in the new variable
    let mut new_coeffs = Vec::new();
    for c in coeffs {
      let expanded = series_ast(&[c.clone(), spec.clone()])?;
      new_coeffs.push(expanded);
    }
    return Ok(Expr::FunctionCall {
      name: "SeriesData".to_string(),
      args: vec![
        args[0].clone(), // var
        args[1].clone(), // x0
        Expr::List(new_coeffs.into()),
        args[3].clone(), // nmin
        args[4].clone(), // nmax
        args[5].clone(), // den
      ]
      .into(),
    });
  }
  // If not a SeriesData, just expand the expression
  series_ast(&[series.clone(), spec.clone()])
}

/// NIntegrate[expr, {var, lo, hi}] - Numerical integration
/// Detect `Exp[α·var²]` integrand where `α` is constant w.r.t. `var` —
/// matches `Exp[arg]` (FunctionCall), `E^arg` (BinaryOp::Power), and
/// `Power[E, arg]` (FunctionCall) shapes. The inner `arg` must
/// factor as `α · var^2` (any order, optional unary `-` baked into α).
/// Returns the α coefficient when matched.
fn detect_gaussian_coefficient(
  integrand: &Expr,
  var_name: &str,
) -> Option<Expr> {
  // Peel `Exp[arg]` or `E^arg`.
  let arg: &Expr = match integrand {
    Expr::FunctionCall { name, args } if name == "Exp" && args.len() == 1 => {
      &args[0]
    }
    Expr::FunctionCall { name, args }
      if name == "Power"
        && args.len() == 2
        && (matches!(&args[0], Expr::Identifier(s) if s == "E")
          || matches!(&args[0], Expr::Constant(s) if s == "E")) =>
    {
      &args[1]
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Identifier(s) if s == "E")
      || matches!(left.as_ref(), Expr::Constant(s) if s == "E") =>
    {
      right.as_ref()
    }
    _ => return None,
  };

  // `arg` should be `α · var^2` (possibly with α split into multiple
  // factors). Collect all multiplicative factors, separate out the
  // `var^2`, and take the product of everything else as `α`.
  let factors =
    crate::functions::polynomial_ast::collect_multiplicative_factors(arg);
  let mut var_sq_count = 0;
  let mut alpha_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    let is_var_sq = match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => {
        matches!(left.as_ref(), Expr::Identifier(s) if s == var_name)
          && matches!(right.as_ref(), Expr::Integer(2))
      }
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        matches!(&args[0], Expr::Identifier(s) if s == var_name)
          && matches!(&args[1], Expr::Integer(2))
      }
      _ => false,
    };
    if is_var_sq {
      var_sq_count += 1;
      continue;
    }
    // Anything that mentions `var` disqualifies the match.
    if !is_constant_wrt(f, var_name) {
      return None;
    }
    alpha_factors.push(f.clone());
  }
  if var_sq_count != 1 {
    return None;
  }
  let alpha = if alpha_factors.is_empty() {
    Expr::Integer(1)
  } else if alpha_factors.len() == 1 {
    alpha_factors.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: alpha_factors.into(),
    }
  };
  crate::evaluator::evaluate_expr_to_expr(&alpha).ok()
}

/// Build `Sqrt[π/(-α)] · (Erf[hi·Sqrt[-α]] − Erf[lo·Sqrt[-α]]) / 2` as
/// a fully evaluated `Expr`. When `precision` is `Some(p)`, wraps the
/// result in `N[..., p]` so the output is a `p`-digit BigFloat.
fn gaussian_closed_form_integral(
  alpha: &Expr,
  lo: f64,
  hi: f64,
  precision: Option<i128>,
) -> Result<Expr, InterpreterError> {
  // Build `-α` and `Sqrt[-α]` as exact expressions so high-precision
  // evaluation later doesn't suffer f64-to-Real downcasting.
  let neg_alpha = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(-1), alpha.clone()].into(),
  };
  let neg_alpha_eval = crate::evaluator::evaluate_expr_to_expr(&neg_alpha)?;
  let sqrt_neg_alpha = Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![neg_alpha_eval.clone()].into(),
  };
  let pi_over_neg_alpha = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::Constant("Pi".to_string())),
    right: Box::new(neg_alpha_eval.clone()),
  };
  let sqrt_pi_over = Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![pi_over_neg_alpha].into(),
  };

  let bound_expr = |v: f64| -> Expr {
    // Bounds come in as f64; turn integer-valued ones back into exact
    // Integers (so e.g. lo=-1, hi=1 → exact -1, 1).
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e18 {
      Expr::Integer(v as i128)
    } else {
      Expr::Real(v)
    }
  };
  let hi_arg = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![bound_expr(hi), sqrt_neg_alpha.clone()].into(),
  };
  let lo_arg = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![bound_expr(lo), sqrt_neg_alpha].into(),
  };
  let erf_hi = Expr::FunctionCall {
    name: "Erf".to_string(),
    args: vec![hi_arg].into(),
  };
  let erf_lo = Expr::FunctionCall {
    name: "Erf".to_string(),
    args: vec![lo_arg].into(),
  };
  let diff = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(erf_hi),
    right: Box::new(erf_lo),
  };
  let half = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(diff),
    right: Box::new(Expr::Integer(2)),
  };
  let result_symbolic = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![sqrt_pi_over, half].into(),
  };

  if let Some(p) = precision {
    let n_call = Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![result_symbolic, Expr::Integer(p)].into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&n_call);
  }
  // No WorkingPrecision: f64.
  let n_call = Expr::FunctionCall {
    name: "N".to_string(),
    args: vec![result_symbolic].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&n_call)
}

pub fn nintegrate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "NIntegrate expects at least 2 arguments".into(),
    ));
  }

  // Parse options from additional arguments (Tolerance, Method, MaxRecursion, etc.)
  let mut tolerance = 1e-10_f64;
  let mut max_recursion = 50_u32;
  let mut working_precision: Option<i128> = None;
  // `EvaluationMonitor :> expr` — evaluated (with the integration variable
  // bound) at every sampled point, e.g. to `Sow` the abscissae.
  let mut monitor: Option<Expr> = None;
  // Recognised method names — anything else triggers NIntegrate::bdmtd.
  // These are the standard Wolfram NIntegrate strategies; Woxi treats
  // them all as the same adaptive Simpson backend internally for now.
  const KNOWN_METHODS: &[&str] = &[
    "Automatic",
    "AdaptiveMonteCarlo",
    "AdaptiveQuasiMonteCarlo",
    "DoubleExponential",
    "GaussKronrod",
    "GaussLegendre",
    "GlobalAdaptive",
    "InterpolationPointsSubdivision",
    "LocalAdaptive",
    "MonteCarlo",
    "MultidimensionalRule",
    "QuasiMonteCarlo",
    "Romberg",
    "Simpson",
    "Trapezoidal",
  ];
  let mut bad_method_call = false;
  for opt in &args[2..] {
    let (opt_name, opt_value) = match opt {
      Expr::FunctionCall { name, args: rargs }
        if (name == "Rule" || name == "RuleDelayed") && rargs.len() == 2 =>
      {
        match &rargs[0] {
          Expr::Identifier(s) => (s.as_str(), &rargs[1]),
          _ => continue,
        }
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => match pattern.as_ref() {
        Expr::Identifier(s) => (s.as_str(), replacement.as_ref()),
        _ => continue,
      },
      _ => continue,
    };
    match opt_name {
      "Tolerance" => {
        if let Some(t) = crate::functions::math_ast::try_eval_to_f64(opt_value)
        {
          tolerance = t;
        }
      }
      "MaxRecursion" => {
        if let Some(n) = crate::functions::math_ast::expr_to_i128(opt_value) {
          max_recursion = n.max(1) as u32;
        }
      }
      "WorkingPrecision" => {
        if let Some(p) = crate::functions::math_ast::expr_to_i128(opt_value)
          && p > 0
        {
          working_precision = Some(p);
        }
      }
      "Method" => {
        // Validate the method name. Strings like "GaussLegendre" must
        // match exactly; identifiers like Automatic are accepted too.
        // Lists like {"GaussLegendre", "Points" -> 5} are accepted by
        // checking only the leading method name.
        let method_name: Option<String> = match opt_value {
          Expr::String(s) => Some(s.clone()),
          Expr::Identifier(s) => Some(s.clone()),
          Expr::List(items) if !items.is_empty() => match &items[0] {
            Expr::String(s) => Some(s.clone()),
            Expr::Identifier(s) => Some(s.clone()),
            _ => None,
          },
          _ => None,
        };
        if let Some(name) = method_name
          && !KNOWN_METHODS.contains(&name.as_str())
        {
          // Wolfram emits one bdmtd per attempted recursion, but we
          // surface a single message and bail to the unevaluated form.
          crate::emit_message(
            "NIntegrate::bdmtd: The Method option should be a built-in method name or a list with a name followed by method options.",
          );
          bad_method_call = true;
        }
      }
      "EvaluationMonitor" => {
        monitor = Some(opt_value.clone());
      }
      _ => {} // Ignore unrecognised options (Wolfram does too).
    }
  }
  if bad_method_call {
    return Ok(unevaluated("NIntegrate", args));
  }

  // Iterated / multi-dimensional integration: any argument after the first
  // range that is itself a range `{var, lo, hi, …}` (rather than an option
  // Rule) is an inner integration variable. Integrate over the outer variable
  // and, for each sampled abscissa, substitute it into the integrand and the
  // remaining ranges — whose bounds may depend on the outer variable, e.g.
  // `{y, 0, x}` — before integrating the rest recursively.
  let is_range = |e: &Expr| {
    matches!(e, Expr::List(items)
      if items.len() >= 3 && matches!(&items[0], Expr::Identifier(_)))
  };
  if args[2..].iter().any(is_range) {
    let mut inner_ranges: Vec<Expr> = Vec::new();
    let mut options: Vec<Expr> = Vec::new();
    for a in &args[2..] {
      if is_range(a) && options.is_empty() {
        inner_ranges.push(a.clone());
      } else {
        options.push(a.clone());
      }
    }
    let (Expr::List(outer), true) = (&args[1], is_range(&args[1])) else {
      return Ok(unevaluated("NIntegrate", args));
    };
    let Expr::Identifier(ovar) = &outer[0] else {
      return Ok(unevaluated("NIntegrate", args));
    };
    let ovar = ovar.clone();
    let olo = crate::evaluator::evaluate_expr_to_expr(&outer[1])
      .ok()
      .and_then(|e| expr_to_bound(&e));
    let ohi = crate::evaluator::evaluate_expr_to_expr(&outer[outer.len() - 1])
      .ok()
      .and_then(|e| expr_to_bound(&e));
    let (Some(olo), Some(ohi)) = (olo, ohi) else {
      return Ok(unevaluated("NIntegrate", args));
    };
    // Only finite outer bounds are handled here; an infinite outer range in an
    // iterated integral falls through to the unevaluated form.
    if !olo.is_finite() || !ohi.is_finite() {
      return Ok(unevaluated("NIntegrate", args));
    }
    let integrand0 = args[0].clone();
    let inner_eval = |x: f64| -> Option<f64> {
      let x_expr = Expr::Real(x);
      let mut inner_args: Vec<Expr> =
        Vec::with_capacity(1 + inner_ranges.len() + options.len());
      inner_args.push(crate::syntax::substitute_variable(
        &integrand0,
        &ovar,
        &x_expr,
      ));
      for r in &inner_ranges {
        inner_args.push(crate::syntax::substitute_variable(r, &ovar, &x_expr));
      }
      inner_args.extend(options.iter().cloned());
      let result = nintegrate_ast(&inner_args).ok()?;
      crate::functions::math_ast::try_eval_to_f64(&result)
        .filter(|v| v.is_finite())
    };
    return match adaptive_simpson(
      &inner_eval,
      olo,
      ohi,
      tolerance,
      max_recursion,
    ) {
      Some(v) => Ok(Expr::Real(v)),
      None => Err(InterpreterError::EvaluationError(
        "NIntegrate: failed to converge or integrand is not numeric".into(),
      )),
    };
  }

  // Second argument is `{var, a, b}` or, with interior waypoints (e.g. a
  // singularity to break the interval at), `{var, a, b, c, …}`.
  let (var_name, bounds) = match &args[1] {
    Expr::List(items) if items.len() >= 3 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "NIntegrate: first element of integration range must be a symbol"
              .into(),
          ));
        }
      };
      // Evaluate the boundary points — support Infinity/-Infinity at the ends.
      let mut bounds = Vec::with_capacity(items.len() - 1);
      for item in items.iter().skip(1) {
        let e = crate::evaluator::evaluate_expr_to_expr(item)?;
        bounds.push(expr_to_bound(&e).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "NIntegrate: integration bound must be numeric or Infinity".into(),
          )
        })?);
      }
      (var_name, bounds)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "NIntegrate expects {var, lo, hi} as second argument".into(),
      ));
    }
  };
  let lo = bounds[0];
  let hi = *bounds.last().unwrap();

  let integrand = &args[0];

  // Gaussian fast path: detect `Exp[α x²]` (or `E^(α x²)`) where α is a
  // numeric constant w.r.t. the integration variable. Use the closed
  // form `Sqrt[π/(-α)] * (Erf[hi·Sqrt[-α]] − Erf[lo·Sqrt[-α]])/2` so
  // narrow-peak inputs (e.g. α = -10^8) don't time out in adaptive
  // Simpson. When WorkingPrecision is set, evaluate via N[...] at that
  // precision so the output matches Wolfram's BigFloat form.
  if !lo.is_infinite()
    && !hi.is_infinite()
    && let Some(alpha) = detect_gaussian_coefficient(integrand, &var_name)
  {
    return gaussian_closed_form_integral(&alpha, lo, hi, working_precision);
  }

  // Evaluate the integrand at a point. A point where the integrand is not
  // finite (e.g. the removable singularity of Sin[x]/x at x = 0) is replaced
  // by its limit via a tiny perturbation, so adaptive quadrature does not
  // abort at such a point.
  let eval_at = |x: f64| -> Option<f64> {
    // Fire the EvaluationMonitor at each sampled abscissa (with the variable
    // bound to x), e.g. so `Sow[x]` records the sample locations.
    if let Some(m) = &monitor {
      let sub =
        crate::syntax::substitute_variable(m, &var_name, &Expr::Real(x));
      let _ = crate::evaluator::evaluate_expr_to_expr(&sub);
    }
    let raw = |xx: f64| -> Option<f64> {
      let substituted = crate::syntax::substitute_variable(
        integrand,
        &var_name,
        &Expr::Real(xx),
      );
      let evaluated =
        crate::evaluator::evaluate_expr_to_expr(&substituted).ok()?;
      crate::functions::math_ast::try_eval_to_f64(&evaluated)
        .filter(|v| v.is_finite())
    };
    if let Some(v) = raw(x) {
      return Some(v);
    }
    let d = x.abs().max(1.0) * 1e-10;
    raw(x + d).or_else(|| raw(x - d))
  };

  // Integrate one sub-interval [a, b], handling infinite end bounds via a
  // tangent substitution. Interior waypoints are always finite.
  let integrate_segment = |a: f64, b: f64| -> Option<f64> {
    let a_inf = a.is_infinite() && a < 0.0;
    let b_inf = b.is_infinite() && b > 0.0;
    let half_pi = std::f64::consts::FRAC_PI_2;
    let eps = 1e-10;
    if a_inf && b_inf {
      // (-∞, ∞): x = tan(t), dx = sec²(t) dt over (-π/2, π/2).
      let g = |t: f64| eval_at(t.tan()).map(|v| v / (t.cos() * t.cos()));
      adaptive_simpson(
        &g,
        -half_pi + eps,
        half_pi - eps,
        tolerance,
        max_recursion,
      )
    } else if a_inf {
      // (-∞, b): x = b - tan(t), t in (0, π/2).
      let g = |t: f64| eval_at(b - t.tan()).map(|v| v / (t.cos() * t.cos()));
      adaptive_simpson(&g, eps, half_pi - eps, tolerance, max_recursion)
    } else if b_inf {
      // (a, ∞): x = a + tan(t), t in (0, π/2).
      let g = |t: f64| eval_at(a + t.tan()).map(|v| v / (t.cos() * t.cos()));
      adaptive_simpson(&g, eps, half_pi - eps, tolerance, max_recursion)
    } else {
      adaptive_simpson(&eval_at, a, b, tolerance, max_recursion)
    }
  };

  // Sum the integral over each consecutive pair of boundary points. A single
  // `{var, a, b}` range is just one segment.
  let mut total = 0.0;
  for pair in bounds.windows(2) {
    match integrate_segment(pair[0], pair[1]) {
      Some(v) => total += v,
      None => {
        return Err(InterpreterError::EvaluationError(
          "NIntegrate: failed to converge or integrand is not numeric".into(),
        ));
      }
    }
  }
  Ok(Expr::Real(total))
}

/// Convert an expression to an f64 bound, supporting Infinity/-Infinity
fn expr_to_bound(expr: &Expr) -> Option<f64> {
  if matches!(expr, Expr::Identifier(s) if s == "Infinity") {
    return Some(f64::INFINITY);
  }
  if crate::functions::math_ast::is_neg_infinity(expr) {
    return Some(f64::NEG_INFINITY);
  }
  // DirectedInfinity[1] = Infinity, DirectedInfinity[-1] = -Infinity
  if let Expr::FunctionCall { name, args } = expr
    && name == "DirectedInfinity"
    && args.len() == 1
  {
    if matches!(&args[0], Expr::Integer(1)) {
      return Some(f64::INFINITY);
    }
    if matches!(&args[0], Expr::Integer(-1)) {
      return Some(f64::NEG_INFINITY);
    }
  }
  crate::functions::math_ast::try_eval_to_f64(expr)
}

/// Adaptive Simpson's quadrature
fn adaptive_simpson(
  f: &dyn Fn(f64) -> Option<f64>,
  a: f64,
  b: f64,
  tol: f64,
  max_depth: u32,
) -> Option<f64> {
  let fa = f(a)?;
  let fb = f(b)?;
  let m = (a + b) / 2.0;
  let fm = f(m)?;
  let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
  // A binary refinement to depth `max_depth` can visit up to 2^max_depth nodes
  // — for an integrand that never meets the tolerance (e.g. the oscillatory
  // Sin[x]/x transformed onto a finite interval) this never terminates in
  // practice. Cap the total number of subdivision nodes so NIntegrate always
  // returns in bounded time, falling back to the current estimate. Convergent
  // integrands stop at `error.abs() < tol` long before this ceiling, so it only
  // bounds the divergent fallback; keep it small enough that even in a debug
  // build the fallback returns in ~1s (each node re-interprets the integrand
  // expression, which is slow unoptimized) rather than brushing the test
  // harness's per-test timeout under parallel CI load.
  let budget = std::cell::Cell::new(10_000u64);
  adaptive_simpson_rec(f, a, b, tol, whole, fa, fm, fb, max_depth, &budget)
}

#[allow(clippy::too_many_arguments)]
fn adaptive_simpson_rec(
  f: &dyn Fn(f64) -> Option<f64>,
  a: f64,
  b: f64,
  tol: f64,
  whole: f64,
  fa: f64,
  fm: f64,
  fb: f64,
  depth: u32,
  budget: &std::cell::Cell<u64>,
) -> Option<f64> {
  let m = (a + b) / 2.0;
  let m1 = (a + m) / 2.0;
  let m2 = (m + b) / 2.0;
  let fm1 = f(m1)?;
  let fm2 = f(m2)?;
  let h = b - a;
  let left = h / 12.0 * (fa + 4.0 * fm1 + fm);
  let right = h / 12.0 * (fm + 4.0 * fm2 + fb);
  let refined = left + right;
  let error = (refined - whole) / 15.0;

  // Stop refining at the depth limit, when converged, or when the global node
  // budget is exhausted (the latter bounds wall-clock for non-converging
  // integrands).
  if depth == 0 || error.abs() < tol || budget.get() == 0 {
    Some(refined + error)
  } else {
    budget.set(budget.get() - 1);
    let left_result = adaptive_simpson_rec(
      f,
      a,
      m,
      tol / 2.0,
      left,
      fa,
      fm1,
      fm,
      depth - 1,
      budget,
    )?;
    let right_result = adaptive_simpson_rec(
      f,
      m,
      b,
      tol / 2.0,
      right,
      fm,
      fm2,
      fb,
      depth - 1,
      budget,
    )?;
    Some(left_result + right_result)
  }
}

/// Orthogonal-curvilinear scale factors h_i for a named coordinate system, so
/// the gradient component along variable i is `(1/h_i) ∂f/∂x_i`. Returns None
/// for an unknown system or a mismatched variable count.
fn coord_scale_factors(cs: &str, vars: &[Expr]) -> Option<Vec<Expr>> {
  let one = || Expr::Integer(1);
  let sin = |e: &Expr| Expr::FunctionCall {
    name: "Sin".to_string(),
    args: vec![e.clone()].into(),
  };
  let times = |a: Expr, b: Expr| Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![a, b].into(),
  };
  match cs {
    // h_i = 1 for every Cartesian axis.
    "Cartesian" => Some(vars.iter().map(|_| one()).collect()),
    // {r, θ}: h = {1, r}.
    "Polar" if vars.len() == 2 => Some(vec![one(), vars[0].clone()]),
    // {r, θ, z}: h = {1, r, 1}.
    "Cylindrical" if vars.len() == 3 => {
      Some(vec![one(), vars[0].clone(), one()])
    }
    // {r, θ, φ} (θ the polar angle): h = {1, r, r Sin[θ]}.
    "Spherical" if vars.len() == 3 => Some(vec![
      one(),
      vars[0].clone(),
      times(vars[0].clone(), sin(&vars[1])),
    ]),
    _ => None,
  }
}

/// Build a product expression, dropping trivial factors of 1.
fn cc_product(factors: Vec<Expr>) -> Expr {
  let kept: Vec<Expr> = factors
    .into_iter()
    .filter(|f| !matches!(f, Expr::Integer(1)))
    .collect();
  match kept.len() {
    0 => Expr::Integer(1),
    1 => kept.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: kept.into(),
    },
  }
}

/// Build `1/e` as `Power[e, -1]` (left as 1 when e is 1).
fn cc_reciprocal(e: Expr) -> Expr {
  if matches!(e, Expr::Integer(1)) {
    return Expr::Integer(1);
  }
  Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![e, Expr::Integer(-1)].into(),
  }
}

/// Divergence in orthogonal curvilinear coordinates with scale factors h:
/// Div F = (1/J) Σ_i ∂/∂x_i( (J/h_i) F_i ),  J = Π h_i.
fn divergence_curvilinear(
  funcs: &[Expr],
  var_names: &[String],
  scales: &[Expr],
) -> Result<Expr, InterpreterError> {
  let jac = cc_product(scales.to_vec());
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let mut terms = Vec::with_capacity(funcs.len());
  for i in 0..funcs.len() {
    let coef = eval(&cc_product(vec![
      jac.clone(),
      cc_reciprocal(scales[i].clone()),
    ]))?;
    let inner = eval(&cc_product(vec![coef, funcs[i].clone()]))?;
    terms.push(differentiate_expr(&inner, &var_names[i])?);
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  eval(&cc_product(vec![sum, cc_reciprocal(jac)]))
}

/// Laplacian in orthogonal curvilinear coordinates with scale factors h:
/// Lap f = (1/J) Σ_i ∂/∂x_i( (J/h_i²) ∂f/∂x_i ),  J = Π h_i.
fn laplacian_curvilinear(
  f: &Expr,
  var_names: &[String],
  scales: &[Expr],
) -> Result<Expr, InterpreterError> {
  let jac = cc_product(scales.to_vec());
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let mut terms = Vec::with_capacity(var_names.len());
  for (i, var) in var_names.iter().enumerate() {
    let coef = eval(&cc_product(vec![
      jac.clone(),
      cc_reciprocal(scales[i].clone()),
      cc_reciprocal(scales[i].clone()),
    ]))?;
    let dfi = differentiate_expr(f, var)?;
    let inner = eval(&cc_product(vec![coef, dfi]))?;
    terms.push(differentiate_expr(&inner, var)?);
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  eval(&cc_product(vec![sum, cc_reciprocal(jac)]))
}

/// Curl in orthogonal curvilinear coordinates with scale factors h.
/// 2D ({F1, F2}) gives the scalar (1/(h1 h2))[∂(h2 F2)/∂x1 − ∂(h1 F1)/∂x2].
/// 3D gives the vector whose i-th component (with (i,j,k) cyclic) is
/// (1/(hj hk))[∂(hk Fk)/∂xj − ∂(hj Fj)/∂xk].
fn curl_curvilinear(
  field: &[Expr],
  var_names: &[String],
  scales: &[Expr],
) -> Result<Expr, InterpreterError> {
  let eval = crate::evaluator::evaluate_expr_to_expr;
  // (1/(h_a h_b)) [ ∂(h_b F_b)/∂x_a − ∂(h_a F_a)/∂x_b ]
  let component = |a: usize, b: usize| -> Result<Expr, InterpreterError> {
    let prefactor =
      cc_reciprocal(cc_product(vec![scales[a].clone(), scales[b].clone()]));
    let hb_fb = eval(&cc_product(vec![scales[b].clone(), field[b].clone()]))?;
    let d1 = differentiate_expr(&hb_fb, &var_names[a])?;
    let ha_fa = eval(&cc_product(vec![scales[a].clone(), field[a].clone()]))?;
    let d2 = differentiate_expr(&ha_fa, &var_names[b])?;
    let inner = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![d1, cc_product(vec![Expr::Integer(-1), d2])].into(),
    };
    eval(&cc_product(vec![prefactor, inner]))
  };
  if field.len() == 2 {
    return component(0, 1);
  }
  // 3D: component i uses axes (j, k) = ((i+1)%3, (i+2)%3).
  let mut comps = Vec::with_capacity(3);
  for i in 0..3 {
    comps.push(component((i + 1) % 3, (i + 2) % 3)?);
  }
  Ok(Expr::List(comps.into()))
}

/// Extract the coordinate-system scale factors from a 3rd argument, or None if
/// it is not a recognized system string / matched variable list.
fn coord_scales_from_arg(cs_arg: &Expr, vars: &[Expr]) -> Option<Vec<Expr>> {
  let cs = match cs_arg {
    Expr::String(s) | Expr::Identifier(s) => s.as_str(),
    _ => return None,
  };
  coord_scale_factors(cs, vars)
}

/// Grad[f, {x1, x2, ...}] - Gradient of a scalar function.
/// Grad[f, {x1, ...}, "Coordinates"] uses orthogonal-curvilinear scale factors.
pub fn grad_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Grad expects 2 or 3 arguments".into(),
    ));
  }
  let unevaluated = || Ok(unevaluated("Grad", args));
  let vars = match &args[1] {
    Expr::List(items) => items,
    _ => return unevaluated(),
  };

  // Scale factors: all 1 for the plain (Cartesian) form, or system-specific.
  let scales: Vec<Expr> = if args.len() == 3 {
    let cs = match &args[2] {
      Expr::String(s) | Expr::Identifier(s) => s.as_str(),
      _ => return unevaluated(),
    };
    match coord_scale_factors(cs, vars) {
      Some(h) => h,
      None => return unevaluated(),
    }
  } else {
    vars.iter().map(|_| Expr::Integer(1)).collect()
  };

  // All variables must be plain symbols.
  let mut var_names = Vec::with_capacity(vars.len());
  for var in vars.iter() {
    match var {
      Expr::Identifier(s) => var_names.push(s.clone()),
      _ => return unevaluated(),
    }
  }

  // Grad appends the derivative index as the LAST axis of the result, so for a
  // scalar field f the result is the gradient vector {∂f/∂x_j}, and for an
  // array field f the gradient vector is appended at each leaf — e.g. for a
  // vector field result[[i, j]] = ∂f_i/∂x_j. Recurse over the field structure
  // to handle arbitrary tensor rank.
  grad_field(&args[0], &var_names, &scales)
}

/// Build the gradient of `field` with respect to `var_names`, applying the
/// per-variable scale factors. The derivative axis becomes the innermost axis.
fn grad_field(
  field: &Expr,
  var_names: &[String],
  scales: &[Expr],
) -> Result<Expr, InterpreterError> {
  if let Expr::List(items) = field {
    let mut rows = Vec::with_capacity(items.len());
    for item in items.iter() {
      rows.push(grad_field(item, var_names, scales)?);
    }
    return Ok(Expr::List(rows.into()));
  }

  let mut components = Vec::with_capacity(var_names.len());
  for (var_name, h) in var_names.iter().zip(scales) {
    let deriv = differentiate_expr(field, var_name)?;
    // component = (1/h) ∂f/∂x; the scale factor is 1 for Cartesian axes.
    let comp = if matches!(h, Expr::Integer(1)) {
      deriv
    } else {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          deriv,
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![h.clone(), Expr::Integer(-1)].into(),
          },
        ]
        .into(),
      }
    };
    let evald = crate::evaluator::evaluate_expr_to_expr(&comp)?;
    components.push(evald);
  }
  Ok(Expr::List(components.into()))
}

/// Wronskian[{f1, ..., fn}, x] = determinant of matrix of derivatives
pub fn wronskian_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Wronskian expects exactly 2 arguments".into(),
    ));
  }
  let funcs = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(unevaluated("Wronskian", args));
    }
  };
  let var_name = match &args[1] {
    Expr::Identifier(s) => s,
    _ => {
      return Ok(unevaluated("Wronskian", args));
    }
  };

  let n = funcs.len();
  if n == 0 {
    return Ok(Expr::Integer(1));
  }

  // Build matrix: M[i][j] = D^j[funcs[i], x]
  let mut matrix_rows = Vec::with_capacity(n);
  for f in funcs {
    let mut row = Vec::with_capacity(n);
    let mut current = crate::evaluator::evaluate_expr_to_expr(f)?;
    row.push(current.clone());
    for _j in 1..n {
      current = differentiate_expr(&current, var_name)?;
      current = crate::evaluator::evaluate_expr_to_expr(&current)?;
      row.push(current.clone());
    }
    matrix_rows.push(Expr::List(row.into()));
  }

  let matrix = Expr::List(matrix_rows.into());
  let det = crate::functions::linear_algebra_ast::det_ast(&[matrix])?;
  let result = crate::evaluator::evaluate_expr_to_expr(&det)?;
  // Apply trig identities (e.g. Sin[x]^2 + Cos[x]^2 → 1) to simplify the determinant
  let simplified =
    crate::functions::polynomial_ast::apply_trig_identities(&result);
  let simplified_str = crate::syntax::expr_to_string(&simplified);
  let result_str = crate::syntax::expr_to_string(&result);
  if simplified_str != result_str {
    crate::evaluator::evaluate_expr_to_expr(&simplified)
  } else {
    Ok(result)
  }
}

/// Div[{f1, f2, ...}, {x1, x2, ...}] = divergence = Sum[D[fi, xi]]
pub fn div_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Div expects 2 or 3 arguments".into(),
    ));
  }
  let unevaluated = || Ok(unevaluated("Div", args));
  let funcs = match &args[0] {
    Expr::List(items) => items,
    _ => return unevaluated(),
  };
  let vars = match &args[1] {
    Expr::List(items) => items,
    _ => return unevaluated(),
  };

  // The 3-argument form uses orthogonal-curvilinear scale factors. It is
  // defined for a rank-1 vector field, so the outer length must match.
  if args.len() == 3 {
    if funcs.len() != vars.len() {
      return unevaluated();
    }
    let scales = match coord_scales_from_arg(&args[2], vars) {
      Some(h) => h,
      None => return unevaluated(),
    };
    let mut var_names = Vec::with_capacity(vars.len());
    for var in vars {
      match var {
        Expr::Identifier(s) => var_names.push(s.clone()),
        _ => return unevaluated(),
      }
    }
    return divergence_curvilinear(funcs, &var_names, &scales);
  }

  // Div contracts the LAST index of the field with the derivative variables.
  // For a rank-1 vector field f this is the scalar Σ_j ∂f_j/∂x_j; for an
  // array field it recurses over the outer structure, so a rank-2 tensor T
  // gives the vector result[[i]] = Σ_j ∂T[[i, j]]/∂x_j.
  let mut var_names = Vec::with_capacity(vars.len());
  for var in vars {
    match var {
      Expr::Identifier(s) => var_names.push(s.clone()),
      _ => return unevaluated(),
    }
  }
  match divergence_field(&args[0], &var_names)? {
    Some(result) => Ok(result),
    None => unevaluated(),
  }
}

/// Divergence of an array `field` contracting its last index with `var_names`.
/// Returns `None` when the field shape does not match the variable count.
fn divergence_field(
  field: &Expr,
  var_names: &[String],
) -> Result<Option<Expr>, InterpreterError> {
  let Expr::List(items) = field else {
    return Ok(None);
  };
  // Recurse into higher-rank fields: if elements are themselves lists, the last
  // index is deeper, so map the divergence over the outer structure.
  if items.iter().any(|e| matches!(e, Expr::List(_))) {
    if !items.iter().all(|e| matches!(e, Expr::List(_))) {
      return Ok(None);
    }
    let mut rows = Vec::with_capacity(items.len());
    for item in items.iter() {
      match divergence_field(item, var_names)? {
        Some(r) => rows.push(r),
        None => return Ok(None),
      }
    }
    return Ok(Some(Expr::List(rows.into())));
  }

  // Base case: a rank-1 vector field. Contract Σ_j ∂f_j/∂x_j.
  if items.len() != var_names.len() {
    return Ok(None);
  }
  let mut terms = Vec::with_capacity(items.len());
  for (f, var_name) in items.iter().zip(var_names.iter()) {
    let deriv = differentiate_expr(f, var_name)?;
    terms.push(crate::evaluator::evaluate_expr_to_expr(&deriv)?);
  }
  if terms.len() == 1 {
    return Ok(Some(terms.into_iter().next().unwrap()));
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  Ok(Some(crate::evaluator::evaluate_expr_to_expr(&sum)?))
}

/// Laplacian[f, {x1, x2, ...}] = Sum of second partial derivatives
pub fn laplacian_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Laplacian expects 2 or 3 arguments".into(),
    ));
  }
  let unevaluated = || Ok(unevaluated("Laplacian", args));
  let vars = match &args[1] {
    Expr::List(items) => items,
    _ => return unevaluated(),
  };

  // The 3-argument form uses orthogonal-curvilinear scale factors.
  if args.len() == 3 {
    let scales = match coord_scales_from_arg(&args[2], vars) {
      Some(h) => h,
      None => return unevaluated(),
    };
    let mut var_names = Vec::with_capacity(vars.len());
    for var in vars {
      match var {
        Expr::Identifier(s) => var_names.push(s.clone()),
        _ => return unevaluated(),
      }
    }
    return laplacian_curvilinear(&args[0], &var_names, &scales);
  }

  let mut terms = Vec::with_capacity(vars.len());
  for var in vars {
    let var_name = match var {
      Expr::Identifier(s) => s,
      _ => return unevaluated(),
    };
    // Second derivative: D[D[f, x], x]
    let first = differentiate_expr(&args[0], var_name)?;
    let second = differentiate_expr(&first, var_name)?;
    let evald = crate::evaluator::evaluate_expr_to_expr(&second)?;
    terms.push(evald);
  }

  // Sum all terms
  if terms.len() == 1 {
    return Ok(terms.into_iter().next().unwrap());
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&sum)
}

/// Curl[{f1, f2}, {x1, x2}] - 2D curl (scalar)
/// Curl[{f1, f2, f3}, {x1, x2, x3}] - 3D curl (vector)
/// Rank (nesting depth) and dimensions of a rectangular array expression.
/// A non-list is rank 0 with no dimensions; dimensions are taken from the
/// first element at each level.
fn tensor_shape(expr: &Expr) -> (usize, Vec<usize>) {
  match expr {
    Expr::List(items) if !items.is_empty() => {
      let (sub_rank, sub_dims) = tensor_shape(&items[0]);
      let mut dims = Vec::with_capacity(sub_dims.len() + 1);
      dims.push(items.len());
      dims.extend(sub_dims);
      (sub_rank + 1, dims)
    }
    Expr::List(_) => (1, vec![0]),
    _ => (0, Vec::new()),
  }
}

pub fn curl_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Curl expects 2 or 3 arguments".into(),
    ));
  }
  let vars = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(unevaluated("Curl", args));
    }
  };

  // The 3-argument form uses orthogonal-curvilinear scale factors and requires
  // a vector field of 2 (scalar result) or 3 (vector result) components.
  if args.len() == 3 {
    let unevaluated = || Ok(unevaluated("Curl", args));
    let field = match &args[0] {
      Expr::List(items) => items,
      _ => return unevaluated(),
    };
    if field.len() != vars.len() || !(field.len() == 2 || field.len() == 3) {
      return unevaluated();
    }
    let scales = match coord_scales_from_arg(&args[2], vars) {
      Some(h) => h,
      None => return unevaluated(),
    };
    let mut var_names = Vec::with_capacity(vars.len());
    for v in vars {
      match v {
        Expr::Identifier(s) => var_names.push(s.clone()),
        _ => return unevaluated(),
      }
    }
    return curl_curvilinear(field, &var_names, &scales);
  }
  // Curl[s, {x, y}] for a scalar s: the 2D scalar curl equals
  // {-D[s, y], D[s, x]} (the perpendicular gradient). Matches Wolfram.
  if !matches!(&args[0], Expr::List(_)) && vars.len() == 2 {
    let var1 = match &vars[0] {
      Expr::Identifier(s) => s,
      _ => {
        return Ok(unevaluated("Curl", args));
      }
    };
    let var2 = match &vars[1] {
      Expr::Identifier(s) => s,
      _ => {
        return Ok(unevaluated("Curl", args));
      }
    };
    let dsdy = differentiate_expr(&args[0], var2)?;
    let dsdx = differentiate_expr(&args[0], var1)?;
    let neg_dsdy = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(dsdy),
    })?;
    return Ok(Expr::List(vec![neg_dsdy, dsdx].into()));
  }
  let field = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(unevaluated("Curl", args));
    }
  };

  // Validate the field's rank and dimensions against the space dimension n,
  // matching wolframscript. The computed branches below assume a rank-1 vector
  // field of matching length; other shapes either have no curl (error message)
  // or use the antisymmetric-tensor form that Woxi does not yet compute (left
  // unevaluated).
  let n = vars.len();
  let (rank, dims) = tensor_shape(&args[0]);
  let unevaluated = || unevaluated("Curl", args);
  let field_str = crate::syntax::expr_to_string(&args[0]);
  if rank == 1 {
    if dims[0] != n {
      crate::emit_message(&format!(
        "Curl::ndimv: There is no {}-dimensional curl for the {}-dimensional vector {}.",
        n, dims[0], field_str
      ));
      return Ok(unevaluated());
    }
    // dims[0] == n: a 2- or 3-vector is computed below; higher-dimensional
    // vectors give an antisymmetric tensor that is left unevaluated.
  } else {
    // rank >= 2
    if dims.iter().any(|&d| d != n) {
      let dims_list =
        Expr::List(dims.iter().map(|&d| Expr::Integer(d as i128)).collect());
      crate::emit_message(&format!(
        "Curl::ndimt: {} has dimensions {} and is therefore not a tensor in {}-dimensional space.",
        field_str,
        crate::syntax::expr_to_string(&dims_list),
        n
      ));
      return Ok(unevaluated());
    }
    if rank >= n {
      crate::emit_message(&format!(
        "Curl::hrank: Tensor expression {} does not have a curl because its rank, {}, is greater than or equal to the dimension {}.",
        field_str, rank, n
      ));
      return Ok(unevaluated());
    }
    // rank < n with matching dimensions: a valid higher-rank curl that Woxi
    // does not yet compute; leave unevaluated rather than return a wrong value.
    return Ok(unevaluated());
  }

  if field.len() == 2 && vars.len() == 2 {
    // 2D curl: dF2/dx1 - dF1/dx2
    let var1 = match &vars[0] {
      Expr::Identifier(s) => s,
      _ => {
        return Ok(unevaluated());
      }
    };
    let var2 = match &vars[1] {
      Expr::Identifier(s) => s,
      _ => {
        return Ok(unevaluated());
      }
    };
    let df2_dx1 = differentiate_expr(&field[1], var1)?;
    let df1_dx2 = differentiate_expr(&field[0], var2)?;
    let result = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(df2_dx1),
      right: Box::new(df1_dx2),
    };
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else if field.len() == 3 && vars.len() == 3 {
    // 3D curl: {dF3/dx2 - dF2/dx3, dF1/dx3 - dF3/dx1, dF2/dx1 - dF1/dx2}
    let var_names: Vec<&str> = vars
      .iter()
      .map(|v| match v {
        Expr::Identifier(s) => Ok(s.as_str()),
        _ => Err(InterpreterError::EvaluationError(
          "Curl: variables must be symbols".into(),
        )),
      })
      .collect::<Result<Vec<_>, _>>()?;

    let mut components = Vec::new();
    // Curl component i = dF_{(i+2)%3}/dx_{(i+1)%3} - dF_{(i+1)%3}/dx_{(i+2)%3}
    for i in 0..3 {
      let j = (i + 1) % 3;
      let k = (i + 2) % 3;
      let d1 = differentiate_expr(&field[k], var_names[j])?;
      let d2 = differentiate_expr(&field[j], var_names[k])?;
      let comp = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(d1),
        right: Box::new(d2),
      };
      components.push(crate::evaluator::evaluate_expr_to_expr(&comp)?);
    }
    Ok(Expr::List(components.into()))
  } else {
    Ok(unevaluated())
  }
}

/// Dt[expr, var] - Total derivative
pub fn dt_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Dt expects exactly 2 arguments".into(),
    ));
  }
  let var = match &args[1] {
    Expr::Identifier(s) => s.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Dt: second argument must be a variable".into(),
      ));
    }
  };
  let result = total_differentiate(&args[0], &var)?;
  Ok(simplify(result))
}

/// Check if an expression is a true constant (number or named constant).
/// Unlike is_constant_wrt, this does NOT consider other variables as constant.
fn is_true_constant(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_))
}

/// Named symbols whose total differential is 0 (they carry the Constant
/// attribute), so they are not treated as Dt variables.
const DT_CONSTANT_SYMBOLS: &[&str] = &[
  "Pi",
  "E",
  "Degree",
  "EulerGamma",
  "GoldenRatio",
  "GoldenAngle",
  "Catalan",
  "Khinchin",
  "Glaisher",
  "I",
  "Infinity",
  "ComplexInfinity",
  "Indeterminate",
  "True",
  "False",
];

/// Collect the free atomic symbols of `expr` (the Dt variables), skipping
/// numeric/named constants and function heads. `Sin[x]` contributes `x`, not
/// `Sin`; `a x^2 + Sin[y]` contributes `a`, `x`, `y`.
fn collect_dt_symbols(expr: &Expr, out: &mut Vec<String>) {
  match expr {
    Expr::Identifier(n)
      if !DT_CONSTANT_SYMBOLS.contains(&n.as_str()) && !out.contains(n) =>
    {
      out.push(n.clone());
    }
    Expr::FunctionCall { args, .. } => {
      for a in args.iter() {
        collect_dt_symbols(a, out);
      }
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_dt_symbols(left, out);
      collect_dt_symbols(right, out);
    }
    Expr::UnaryOp { operand, .. } => collect_dt_symbols(operand, out),
    Expr::List(items) => {
      for i in items.iter() {
        collect_dt_symbols(i, out);
      }
    }
    _ => {}
  }
}

/// Dt[expr] — total differential: the sum over each free variable v of
/// D[expr, v] * Dt[v] (matching wolframscript). A constant gives 0 and a bare
/// variable stays as the unevaluated Dt[v].
pub fn dt_total_differential_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let expr = &args[0];
  // Numeric/named constants: Dt = 0.
  if is_true_constant(expr) {
    return Ok(Expr::Integer(0));
  }
  let dt_of = |sym: &str| Expr::FunctionCall {
    name: "Dt".to_string(),
    args: vec![Expr::Identifier(sym.to_string())].into(),
  };
  // A bare variable is the base case: Dt[x] stays unevaluated (Dt[Pi] = 0).
  if let Expr::Identifier(n) = expr {
    if DT_CONSTANT_SYMBOLS.contains(&n.as_str()) {
      return Ok(Expr::Integer(0));
    }
    return Ok(dt_of(n));
  }
  let mut syms: Vec<String> = Vec::new();
  collect_dt_symbols(expr, &mut syms);
  if syms.is_empty() {
    return Ok(Expr::Integer(0));
  }
  let mut sum: Option<Expr> = None;
  for v in &syms {
    let partial = differentiate(expr, v)?;
    if matches!(partial, Expr::Integer(0)) {
      continue;
    }
    let term = simplify(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(partial),
      right: Box::new(dt_of(v)),
    });
    sum = Some(match sum {
      None => term,
      Some(acc) => Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(acc),
        right: Box::new(term),
      },
    });
  }
  Ok(match sum {
    None => Expr::Integer(0),
    Some(s) => simplify(s),
  })
}

/// Total differentiation: like differentiate but treats all symbols as potentially
/// dependent on the differentiation variable (returns Dt[y, x] instead of 0).
fn total_differentiate(
  expr: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  use BinaryOperator::*;

  match expr {
    // Constants
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => {
      Ok(Expr::Integer(0))
    }

    // Variable
    Expr::Identifier(name) => {
      if name == var {
        Ok(Expr::Integer(1))
      } else {
        // Other variables may depend on var: return Dt[y, x]
        Ok(Expr::FunctionCall {
          name: "Dt".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
        })
      }
    }

    // Binary operations
    Expr::BinaryOp { op, left, right } => match op {
      Plus => {
        let da = total_differentiate(left, var)?;
        let db = total_differentiate(right, var)?;
        Ok(simplify(Expr::BinaryOp {
          op: Plus,
          left: Box::new(da),
          right: Box::new(db),
        }))
      }
      Minus => {
        let da = total_differentiate(left, var)?;
        let db = total_differentiate(right, var)?;
        Ok(simplify(Expr::BinaryOp {
          op: Minus,
          left: Box::new(da),
          right: Box::new(db),
        }))
      }
      Times => {
        let da = total_differentiate(left, var)?;
        let db = total_differentiate(right, var)?;
        Ok(simplify(Expr::BinaryOp {
          op: Plus,
          left: Box::new(Expr::BinaryOp {
            op: Times,
            left: Box::new(da),
            right: right.clone(),
          }),
          right: Box::new(Expr::BinaryOp {
            op: Times,
            left: left.clone(),
            right: Box::new(db),
          }),
        }))
      }
      Divide => {
        let da = total_differentiate(left, var)?;
        let db = total_differentiate(right, var)?;
        Ok(simplify(Expr::BinaryOp {
          op: Divide,
          left: Box::new(Expr::BinaryOp {
            op: Minus,
            left: Box::new(Expr::BinaryOp {
              op: Times,
              left: Box::new(da),
              right: right.clone(),
            }),
            right: Box::new(Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(db),
            }),
          }),
          right: Box::new(Expr::BinaryOp {
            op: Power,
            left: right.clone(),
            right: Box::new(Expr::Integer(2)),
          }),
        }))
      }
      Power => {
        if is_true_constant(right) || is_constant_wrt(right, var) {
          // f(x)^n: n * f(x)^(n-1) * Dt[f, x]
          let df = total_differentiate(left, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(Expr::BinaryOp {
              op: Times,
              left: right.clone(),
              right: Box::new(Expr::BinaryOp {
                op: Power,
                left: left.clone(),
                right: Box::new(Expr::BinaryOp {
                  op: Plus,
                  left: Box::new(Expr::Integer(-1)),
                  right: right.clone(),
                }),
              }),
            }),
            right: Box::new(df),
          }))
        } else if matches!(left.as_ref(), Expr::Constant(c) if c == "E") {
          // E^g: E^g * Dt[g, x]
          let dg = total_differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(expr.clone()),
            right: Box::new(dg),
          }))
        } else {
          // General f^g: return unevaluated
          Ok(Expr::FunctionCall {
            name: "Dt".to_string(),
            args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
          })
        }
      }
      _ => Ok(Expr::FunctionCall {
        name: "Dt".to_string(),
        args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
      }),
    },

    // Unary minus
    Expr::UnaryOp { op, operand } => {
      if matches!(op, UnaryOperator::Minus) {
        let d = total_differentiate(operand, var)?;
        Ok(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(d),
        })
      } else {
        Ok(Expr::FunctionCall {
          name: "Dt".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
        })
      }
    }

    // Known function calls - chain rule
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Sin" if args.len() == 1 => {
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(Expr::FunctionCall {
              name: "Cos".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Cos" if args.len() == 1 => {
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand: Box::new(Expr::FunctionCall {
                name: "Sin".to_string(),
                args: args.clone(),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Tan" if args.len() == 1 => {
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(Expr::BinaryOp {
              op: Power,
              left: Box::new(Expr::FunctionCall {
                name: "Sec".to_string(),
                args: args.clone(),
              }),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(df),
          }))
        }
        "Log" if args.len() == 1 => {
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(Expr::BinaryOp {
              op: Power,
              left: Box::new(args[0].clone()),
              right: Box::new(Expr::Integer(-1)),
            }),
            right: Box::new(df),
          }))
        }
        "Exp" if args.len() == 1 => {
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Times,
            left: Box::new(expr.clone()),
            right: Box::new(df),
          }))
        }
        "Sqrt" if args.len() == 1 => {
          // d/dx[sqrt(f)] = f'/(2*sqrt(f))
          let df = total_differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Divide,
            left: Box::new(df),
            right: Box::new(Expr::BinaryOp {
              op: Times,
              left: Box::new(Expr::Integer(2)),
              right: Box::new(expr.clone()),
            }),
          }))
        }
        "Plus" => {
          // Sum rule
          let mut terms = Vec::new();
          for arg in args {
            terms.push(total_differentiate(arg, var)?);
          }
          if terms.is_empty() {
            return Ok(Expr::Integer(0));
          }
          let mut result = terms.remove(0);
          for t in terms {
            result = Expr::BinaryOp {
              op: Plus,
              left: Box::new(result),
              right: Box::new(t),
            };
          }
          Ok(simplify(result))
        }
        "Times" if args.len() >= 2 => {
          // Product rule for n factors
          let mut sum_terms = Vec::new();
          for i in 0..args.len() {
            let di = total_differentiate(&args[i], var)?;
            let mut product = di;
            for (j, arg) in args.iter().enumerate() {
              if j != i {
                product = Expr::BinaryOp {
                  op: Times,
                  left: Box::new(product),
                  right: Box::new(arg.clone()),
                };
              }
            }
            sum_terms.push(product);
          }
          let mut result = sum_terms.remove(0);
          for t in sum_terms {
            result = Expr::BinaryOp {
              op: Plus,
              left: Box::new(result),
              right: Box::new(t),
            };
          }
          Ok(simplify(result))
        }
        _ => {
          // Unknown function: return unevaluated
          Ok(Expr::FunctionCall {
            name: "Dt".to_string(),
            args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
          })
        }
      }
    }

    _ => Ok(Expr::FunctionCall {
      name: "Dt".to_string(),
      args: vec![expr.clone(), Expr::Identifier(var.to_string())].into(),
    }),
  }
}

/// Asymptotic[f, x -> x0] returns the leading (lowest-order non-zero) term of
/// the series expansion of f at the finite point x0. Asymptotic[f, {x, x0, n}]
/// returns the series truncated at order n (Normal[Series[f, {x, x0, n}]]).
/// Infinite expansion points and other forms are left unevaluated.
pub fn asymptotic_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("Asymptotic", args));
  if args.len() != 2 {
    return unevaluated();
  }
  let f = &args[0];

  match &args[1] {
    // {x, x0, n} → Normal[Series[f, {x, x0, n}]].
    Expr::List(items)
      if items.len() == 3 && matches!(&items[0], Expr::Identifier(_)) =>
    {
      let series = crate::evaluator::evaluate_function_call_ast(
        "Series",
        &[f.clone(), args[1].clone()],
      )?;
      crate::evaluator::evaluate_function_call_ast("Normal", &[series])
    }
    // x -> x0 → leading term.
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let var = match pattern.as_ref() {
        Expr::Identifier(name) => name.clone(),
        _ => return unevaluated(),
      };
      let x0 = replacement.as_ref().clone();
      // Only finite expansion points are handled here.
      let is_infinite = matches!(&x0, Expr::Identifier(n) if n == "Infinity" || n == "ComplexInfinity")
        || matches!(&x0, Expr::UnaryOp { op: UnaryOperator::Minus, operand }
            if matches!(operand.as_ref(), Expr::Identifier(n) if n == "Infinity"));
      if is_infinite {
        return unevaluated();
      }
      // The leading term is a finite quantity even when the series of f at x0
      // transiently divides by zero (e.g. Sin[x]/x); discard the messages those
      // intermediate steps emit, the way wolframscript's Asymptotic does.
      let snapshot = crate::snapshot_warnings();
      let mut found: Option<Expr> = None;
      for &order in &[6i128, 16, 40] {
        let spec = Expr::List(
          vec![
            Expr::Identifier(var.clone()),
            x0.clone(),
            Expr::Integer(order),
          ]
          .into(),
        );
        let series = crate::evaluator::evaluate_function_call_ast(
          "Series",
          &[f.clone(), spec],
        )?;
        if let Some(term) = leading_series_term(&series, &var, &x0) {
          found = Some(term);
          break;
        }
        // A non-SeriesData, non-zero result (e.g. a constant) is the answer.
        if !matches!(&series, Expr::FunctionCall { name, .. } if name == "SeriesData")
          && !matches!(&series, Expr::Integer(0))
          && !matches!(&series, Expr::Real(z) if *z == 0.0)
        {
          found = Some(series);
          break;
        }
      }
      crate::restore_warnings(snapshot);
      match found {
        Some(term) => crate::evaluator::evaluate_expr_to_expr(&term),
        None => unevaluated(),
      }
    }
    _ => unevaluated(),
  }
}

/// Extract the leading (first non-zero) term `c*(x-x0)^order` from a SeriesData
/// expression. Returns None when every stored coefficient is zero.
fn leading_series_term(series: &Expr, var: &str, x0: &Expr) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = series else {
    return None;
  };
  if name != "SeriesData" || args.len() < 6 {
    return None;
  }
  // SeriesData[var, x0, {coeffs}, nmin, nmax, denom]
  let coeffs = match &args[2] {
    Expr::List(c) => c,
    _ => return None,
  };
  let nmin = crate::functions::math_ast::expr_to_i128(&args[3])?;
  let denom = crate::functions::math_ast::expr_to_i128(&args[5])?;
  let is_zero = |e: &Expr| {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(z) if *z == 0.0)
  };
  for (i, c) in coeffs.iter().enumerate() {
    if is_zero(c) {
      continue;
    }
    let order = nmin + i as i128;
    if order == 0 {
      return Some(c.clone());
    }
    let exp = if denom == 1 {
      Expr::Integer(order)
    } else {
      crate::functions::math_ast::make_rational(order, denom)
    };
    let base = if matches!(x0, Expr::Integer(0)) {
      Expr::Identifier(var.to_string())
    } else {
      Expr::FunctionCall {
        name: "Subtract".to_string(),
        args: vec![Expr::Identifier(var.to_string()), x0.clone()].into(),
      }
    };
    let pow = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![base, exp].into(),
    };
    return Some(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![c.clone(), pow].into(),
    });
  }
  None
}

/// AsymptoticSolve[eqn, x -> x0, n] — find asymptotic solutions of eqn near x = x0 to order n.
///
/// Uses Series expansion and iterative coefficient solving.
pub fn asymptotic_solve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated("AsymptoticSolve", args));
  }

  // Parse the equation: eqn can be f == 0 or just f (treated as f == 0)
  let f_expr = match &args[0] {
    Expr::Comparison {
      operands,
      operators,
    } if operators.len() == 1
      && operators[0] == ComparisonOp::Equal
      && operands.len() == 2 =>
    {
      // f == g becomes f - g
      if matches!(&operands[1], Expr::Integer(0)) {
        operands[0].clone()
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(operands[0].clone()),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(operands[1].clone()),
          }),
        }
      }
    }
    // Also handle FunctionCall "Equal"
    Expr::FunctionCall {
      name,
      args: eq_args,
    } if name == "Equal" && eq_args.len() == 2 => {
      if matches!(&eq_args[1], Expr::Integer(0)) {
        eq_args[0].clone()
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(eq_args[0].clone()),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(eq_args[1].clone()),
          }),
        }
      }
    }
    other => other.clone(),
  };

  // Parse x -> x0
  let (var_name, x0) = match &args[1] {
    Expr::Rule {
      pattern,
      replacement,
    } => match pattern.as_ref() {
      Expr::Identifier(name) => (name.clone(), *replacement.clone()),
      _ => {
        return Ok(unevaluated("AsymptoticSolve", args));
      }
    },
    _ => {
      return Ok(unevaluated("AsymptoticSolve", args));
    }
  };

  // Parse order: must be a list {param, val, n} or a rule param -> val.
  // A plain integer is not valid (Wolfram returns unevaluated).
  let order = match &args[2] {
    Expr::List(items) if items.len() == 3 => match &items[2] {
      Expr::Integer(n) => *n,
      _ => {
        return Ok(unevaluated("AsymptoticSolve", args));
      }
    },
    Expr::Integer(n) => *n,
    _ => {
      return Ok(unevaluated("AsymptoticSolve", args));
    }
  };

  // When 3rd arg is a plain integer (not a list/rule with perturbation param),
  // Wolfram returns unevaluated.
  if matches!(&args[2], Expr::Integer(_)) {
    return Ok(unevaluated("AsymptoticSolve", args));
  }

  if order < 1 {
    return Ok(Expr::List(vec![].into()));
  }

  // Compute the series expansion of f around x0
  let series_result = series_ast(&[
    f_expr.clone(),
    Expr::List(
      vec![
        Expr::Identifier(var_name.clone()),
        x0.clone(),
        Expr::Integer(order),
      ]
      .into(),
    ),
  ])?;

  // Extract SeriesData coefficients
  let (coeffs, _min_power) = match extract_series_coefficients(&series_result) {
    Some(c) => c,
    None => {
      return Ok(unevaluated("AsymptoticSolve", args));
    }
  };

  if coeffs.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Use the series to find solutions via InverseSeries approach:
  // f(x) = c0 + c1*(x-x0) + c2*(x-x0)^2 + ... = 0
  // If c0 == 0, x = x0 is already a solution. Look at the structure.
  // If c0 != 0, we need to solve for x-x0.

  // Use Newton-like iteration on the truncated polynomial
  // Build the polynomial: sum of c_k * t^(k + min_power) where t = x - x0
  // and solve this polynomial for t using Solve

  // Build the polynomial expression in a temporary variable
  let t_var = Expr::Identifier("AsymptoticSolve$t".to_string());

  let mut poly_terms: Vec<Expr> = Vec::new();
  for (i, coeff) in coeffs.iter().enumerate() {
    if matches!(coeff, Expr::Integer(0)) {
      continue;
    }
    let power = _min_power + i as i128;
    let term = if power == 0 {
      coeff.clone()
    } else if power == 1 {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(coeff.clone()),
        right: Box::new(t_var.clone()),
      }
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(coeff.clone()),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(t_var.clone()),
          right: Box::new(Expr::Integer(power)),
        }),
      }
    };
    poly_terms.push(term);
  }

  if poly_terms.is_empty() {
    // f is identically zero to this order — any x works
    return Ok(Expr::List(
      vec![Expr::List(
        vec![Expr::Rule {
          pattern: Box::new(Expr::Identifier(var_name)),
          replacement: Box::new(x0),
        }]
        .into(),
      )]
      .into(),
    ));
  }

  let poly_expr = if poly_terms.len() == 1 {
    poly_terms.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: poly_terms.into(),
    }
  };

  // Solve poly_expr == 0 for t
  use crate::evaluator::evaluate_expr_to_expr;

  let solve_expr = Expr::FunctionCall {
    name: "Solve".to_string(),
    args: vec![
      Expr::Comparison {
        operands: vec![poly_expr, Expr::Integer(0)],
        operators: vec![ComparisonOp::Equal],
      },
      Expr::Identifier("AsymptoticSolve$t".to_string()),
    ]
    .into(),
  };

  let solutions = evaluate_expr_to_expr(&solve_expr)?;

  // Convert solutions from t -> val to x -> x0 + val
  match &solutions {
    Expr::List(sol_list) => {
      let mut result = Vec::new();
      for sol in sol_list {
        if let Expr::List(rules) = sol {
          let mut new_rules = Vec::new();
          for rule in rules {
            if let Expr::Rule { replacement, .. } = rule {
              // x = x0 + t
              let x_val = if matches!(x0, Expr::Integer(0)) {
                *replacement.clone()
              } else {
                Expr::BinaryOp {
                  op: BinaryOperator::Plus,
                  left: Box::new(x0.clone()),
                  right: replacement.clone(),
                }
              };
              let simplified = evaluate_expr_to_expr(&x_val)?;
              new_rules.push(Expr::Rule {
                pattern: Box::new(Expr::Identifier(var_name.clone())),
                replacement: Box::new(simplified),
              });
            }
          }
          if !new_rules.is_empty() {
            result.push(Expr::List(new_rules.into()));
          }
        }
      }
      Ok(Expr::List(result.into()))
    }
    _ => Ok(unevaluated("AsymptoticSolve", args)),
  }
}

/// Extract coefficients from a SeriesData expression.
/// Returns (coefficients, min_power).
fn extract_series_coefficients(expr: &Expr) -> Option<(Vec<Expr>, i128)> {
  match expr {
    Expr::FunctionCall { name, args } if name == "SeriesData" => {
      // SeriesData[var, x0, coeffs_list, nmin, nmax, den]
      if args.len() >= 4 {
        let coeffs = match &args[2] {
          Expr::List(items) => items.clone(),
          _ => return None,
        };
        let nmin = match &args[3] {
          Expr::Integer(n) => *n,
          _ => return None,
        };
        let den = if args.len() >= 6 {
          match &args[5] {
            Expr::Integer(d) => *d,
            _ => 1,
          }
        } else {
          1
        };
        // The actual power of the i-th coefficient is (nmin + i) / den
        // For simplicity, handle den == 1
        if den == 1 {
          Some((coeffs.to_vec(), nmin))
        } else {
          None
        }
      } else {
        None
      }
    }
    _ => None,
  }
}

/// DiscreteConvolve[f, g, n, m] — discrete convolution.
///
/// Computes Sum[f /. n -> k, g /. m -> (m - k), {k, -Infinity, Infinity}]
/// by building a Sum expression and evaluating it.
/// If `kernel` is `KroneckerDelta[n + d]` with the coefficient of `n` equal to
/// 1, return the constant offset `d` (which may itself be symbolic but must be
/// free of `n`). Returns None for any other kernel.
fn kronecker_delta_offset(kernel: &Expr, n_var: &str) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = kernel else {
    return None;
  };
  if name != "KroneckerDelta" || args.len() != 1 {
    return None;
  }
  let arg = &args[0];
  let at = |v: i128| -> Option<Expr> {
    crate::evaluator::evaluate_expr_to_expr(
      &crate::syntax::substitute_variable(arg, n_var, &Expr::Integer(v)),
    )
    .ok()
  };
  let d = at(0)?; // arg(0) = d
  let a1 = at(1)?; // arg(1) = 1 + d, so the coefficient of n is a1 - d
  let slope = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(a1),
    right: Box::new(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), d.clone()].into(),
    }),
  })
  .ok()?;
  if matches!(slope, Expr::Integer(1)) && is_constant_wrt(&d, n_var) {
    Some(d)
  } else {
    None
  }
}

/// True if `expr` applies an unrecognized (non-built-in) function head to an
/// argument that depends on `var` — e.g. an opaque `g[n]`. Used to keep
/// DiscreteConvolve's KroneckerDelta sifting from evaluating expressions that
/// wolframscript itself leaves symbolic.
fn discrete_convolve_has_opaque_fn(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      let head_opaque =
        crate::evaluator::get_builtin_function_info(name).is_none();
      if head_opaque && args.iter().any(|a| !is_constant_wrt(a, var)) {
        return true;
      }
      args.iter().any(|a| discrete_convolve_has_opaque_fn(a, var))
    }
    Expr::BinaryOp { left, right, .. } => {
      discrete_convolve_has_opaque_fn(left, var)
        || discrete_convolve_has_opaque_fn(right, var)
    }
    Expr::UnaryOp { operand, .. } => {
      discrete_convolve_has_opaque_fn(operand, var)
    }
    Expr::List(items) => items
      .iter()
      .any(|e| discrete_convolve_has_opaque_fn(e, var)),
    _ => false,
  }
}

pub fn discrete_convolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 4 {
    return Ok(unevaluated("DiscreteConvolve", args));
  }

  let f_expr = &args[0];
  let g_expr = &args[1];
  let n_var = match &args[2] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(unevaluated("DiscreteConvolve", args));
    }
  };
  let m_var = match &args[3] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(unevaluated("DiscreteConvolve", args));
    }
  };

  // KroneckerDelta sifting: convolving with KroneckerDelta[n + d] shifts the
  // other sequence — DiscreteConvolve[h(n), KroneckerDelta[n + d], n, m] = h(m + d)
  // (and symmetrically, since convolution is commutative). wolframscript fires
  // this only for recognized functions, leaving an opaque undefined function of
  // n unevaluated; replicate that so we do not out-evaluate wolframscript.
  let sift = |other: &Expr, d: &Expr| -> Option<Expr> {
    if discrete_convolve_has_opaque_fn(other, &n_var) {
      return None;
    }
    let point = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Identifier(m_var.clone())),
      right: Box::new(d.clone()),
    };
    let point = crate::evaluator::evaluate_expr_to_expr(&point).ok()?;
    let substituted = crate::syntax::substitute_variable(other, &n_var, &point);
    crate::evaluator::evaluate_expr_to_expr(&substituted).ok()
  };
  if let Some(d) = kronecker_delta_offset(g_expr, &n_var)
    && let Some(res) = sift(f_expr, &d)
  {
    return Ok(res);
  }
  if let Some(d) = kronecker_delta_offset(f_expr, &n_var)
    && let Some(res) = sift(g_expr, &d)
  {
    return Ok(res);
  }

  // Build: Sum[(f /. n -> k$dc) * (g /. m -> (m - k$dc)), {k$dc, -Infinity, Infinity}]
  let k_var = "DiscreteConvolve$k";
  let k_expr = Expr::Identifier(k_var.to_string());

  // f /. n -> k
  let f_sub = crate::syntax::substitute_variable(f_expr, &n_var, &k_expr);

  // g /. n -> (m - k). Both f and g are functions of the convolution variable
  // n; the discrete convolution is Sum_k f[k] g[m-k], so g's n becomes m - k
  // (NOT m — g is not a function of the output variable).
  let m_minus_k = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(Expr::Identifier(m_var.clone())),
    right: Box::new(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(k_expr.clone()),
    }),
  };
  let g_sub = crate::syntax::substitute_variable(g_expr, &n_var, &m_minus_k);

  // Build the product
  let product = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(f_sub),
    right: Box::new(g_sub),
  };

  // Build Sum[product, {k, -Infinity, Infinity}]
  let sum_expr = Expr::FunctionCall {
    name: "Sum".to_string(),
    args: vec![
      product,
      Expr::List(
        vec![
          k_expr,
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(-1),
              Expr::Identifier("Infinity".to_string()),
            ]
            .into(),
          },
          Expr::Identifier("Infinity".to_string()),
        ]
        .into(),
      ),
    ]
    .into(),
  };

  let result = crate::evaluator::evaluate_expr_to_expr(&sum_expr)?;
  // If the infinite sum did not reduce to a closed form it still references
  // the internal summation variable; in that case keep DiscreteConvolve
  // symbolic rather than leaking the raw `Sum[…, {k$dc, -Infinity, Infinity}]`.
  if !is_constant_wrt(&result, k_var) {
    return Ok(unevaluated("DiscreteConvolve", args));
  }
  Ok(result)
}

/// FrenetSerretSystem[{f1, f2, ...}, t] - Frenet-Serret system for a parametric curve
/// Returns {{curvatures...}, {tangent, normal, ...}} where:
/// - 2D: {{κ}, {T, N}}
/// - 3D: {{κ, τ}, {T, N, B}}
pub fn frenet_serret_system_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "FrenetSerretSystem expects exactly 2 arguments".into(),
    ));
  }

  // If the first argument is a scalar function f[t], treat it as the 2D curve {t, f[t]}
  let owned_components: Vec<Expr>;
  let components: &[Expr] = match &args[0] {
    Expr::List(items) => items,
    _ => {
      owned_components = vec![args[1].clone(), args[0].clone()];
      &owned_components
    }
  };

  let var_name = match &args[1] {
    Expr::Identifier(s) => s.as_str(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "FrenetSerretSystem: second argument must be a variable".into(),
      ));
    }
  };

  let n = components.len();
  if !(2..=3).contains(&n) {
    return Ok(unevaluated("FrenetSerretSystem", args));
  }

  let eval = |e: &Expr| -> Result<Expr, InterpreterError> {
    crate::evaluator::evaluate_expr_to_expr(e)
  };

  // Compute first and second derivatives of each component
  let mut r1 = Vec::with_capacity(n); // r'
  let mut r2 = Vec::with_capacity(n); // r''
  for c in components {
    let d1 = differentiate_expr(c, var_name)?;
    let d1 = eval(&d1)?;
    let d2 = differentiate_expr(&d1, var_name)?;
    let d2 = eval(&d2)?;
    r1.push(d1);
    r2.push(d2);
  }

  // speed_sq = sum of r'[i]^2
  let speed_sq = sum_of_squares(&r1);
  let speed_sq = eval(&speed_sq)?;

  // speed = Sqrt[speed_sq]
  let speed = make_sqrt(speed_sq.clone());

  // T = r' / speed (unit tangent)
  let tangent: Vec<Expr> = r1
    .iter()
    .map(|c| {
      eval(&Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(c.clone()),
        right: Box::new(speed.clone()),
      })
    })
    .collect::<Result<Vec<_>, _>>()?;

  if n == 2 {
    // 2D case
    // κ = (x'*y'' - y'*x'') / (speed_sq)^(3/2)
    let numerator = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(r1[0].clone()),
        right: Box::new(r2[1].clone()),
      }),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(r1[1].clone()),
        right: Box::new(r2[0].clone()),
      }),
    };
    let denom = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(speed_sq),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(3)),
        right: Box::new(Expr::Integer(2)),
      }),
    };
    let kappa = eval(&Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(denom),
    })?;

    // N = {-T2, T1} (rotate tangent 90° counterclockwise)
    let normal = vec![
      eval(&Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(tangent[1].clone()),
      })?,
      tangent[0].clone(),
    ];

    Ok(Expr::List(
      vec![
        Expr::List(vec![kappa].into()),
        Expr::List(
          vec![Expr::List(tangent.into()), Expr::List(normal.into())].into(),
        ),
      ]
      .into(),
    ))
  } else {
    // 3D case
    // r''' for torsion
    let mut r3 = Vec::with_capacity(3);
    for c in &r2 {
      let d3 = differentiate_expr(c, var_name)?;
      r3.push(eval(&d3)?);
    }

    // cross = r' × r''
    let cross = cross_product_3d(&r1, &r2);
    let cross: Vec<Expr> = cross
      .into_iter()
      .map(|c| eval(&c))
      .collect::<Result<Vec<_>, _>>()?;

    // norm_cross_sq = ||cross||^2
    let norm_cross_sq = sum_of_squares(&cross);
    let norm_cross_sq = eval(&norm_cross_sq)?;

    // Check if curvature is zero (straight line case)
    let is_zero_curvature = matches!(&norm_cross_sq, Expr::Integer(0));
    if is_zero_curvature {
      let zero_vec = Expr::List(
        vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)].into(),
      );
      return Ok(Expr::List(
        vec![
          Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
          Expr::List(
            vec![Expr::List(tangent.into()), zero_vec.clone(), zero_vec].into(),
          ),
        ]
        .into(),
      ));
    }

    // norm_cross = ||cross||
    let norm_cross = make_sqrt(norm_cross_sq.clone());

    // κ = ||cross|| / ||r'||^3 = norm_cross / speed_sq^(3/2)
    let speed_cubed = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(speed_sq),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(3)),
        right: Box::new(Expr::Integer(2)),
      }),
    };
    let kappa = eval(&Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(norm_cross.clone()),
      right: Box::new(speed_cubed),
    })?;

    // τ = (r' × r'') · r''' / ||r' × r''||^2
    let dot_cross_r3 = dot_product(&cross, &r3);
    let dot_cross_r3 = eval(&dot_cross_r3)?;
    let tau = eval(&Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(dot_cross_r3),
      right: Box::new(norm_cross_sq),
    })?;

    // B = (r' × r'') / ||r' × r''|| (unit binormal)
    let binormal: Vec<Expr> = cross
      .iter()
      .map(|c| {
        eval(&Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(c.clone()),
          right: Box::new(norm_cross.clone()),
        })
      })
      .collect::<Result<Vec<_>, _>>()?;

    // N = B × T (unit normal)
    let normal_cross = cross_product_3d(&binormal, &tangent);
    let normal: Vec<Expr> = normal_cross
      .into_iter()
      .map(|c| eval(&c))
      .collect::<Result<Vec<_>, _>>()?;

    Ok(Expr::List(
      vec![
        Expr::List(vec![kappa, tau].into()),
        Expr::List(
          vec![
            Expr::List(tangent.into()),
            Expr::List(normal.into()),
            Expr::List(binormal.into()),
          ]
          .into(),
        ),
      ]
      .into(),
    ))
  }
}

/// Helper: compute sum of squares of expressions
fn sum_of_squares(items: &[Expr]) -> Expr {
  let squared: Vec<Expr> = items
    .iter()
    .map(|e| Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(e.clone()),
      right: Box::new(Expr::Integer(2)),
    })
    .collect();
  if squared.len() == 1 {
    squared.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: squared.into(),
    }
  }
}

/// Helper: compute cross product of two 3D vectors
fn cross_product_3d(a: &[Expr], b: &[Expr]) -> Vec<Expr> {
  vec![
    // a[1]*b[2] - a[2]*b[1]
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(a[1].clone()),
        right: Box::new(b[2].clone()),
      }),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(a[2].clone()),
        right: Box::new(b[1].clone()),
      }),
    },
    // a[2]*b[0] - a[0]*b[2]
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(a[2].clone()),
        right: Box::new(b[0].clone()),
      }),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(a[0].clone()),
        right: Box::new(b[2].clone()),
      }),
    },
    // a[0]*b[1] - a[1]*b[0]
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(a[0].clone()),
        right: Box::new(b[1].clone()),
      }),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(a[1].clone()),
        right: Box::new(b[0].clone()),
      }),
    },
  ]
}

/// ArcCurvature[curve, t] - curvature of a parametric curve
/// For 2D: κ = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)
/// For 3D: κ = ||r' × r''|| / ||r'||^3
/// For scalar f[t]: treated as the 2D curve {t, f[t]}
/// AsymptoticIntegrate[f, x, {x, x0, n}] - series expansion of the antiderivative
/// Computes the antiderivative of f, then expands as a series to order n.
pub fn asymptotic_integrate_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated("AsymptoticIntegrate", args));
  }

  let f = &args[0];
  let var = &args[1];
  let spec = &args[2];

  // Definite-integral form: AsymptoticIntegrate[f, {t, a, b}, x -> x0]
  //                        or AsymptoticIntegrate[f, {t, a, b}, {x, x0, n}]
  // Expand f as a series in x at x0, then integrate over t from a to b.
  if let Expr::List(int_items) = var
    && int_items.len() == 3
    && matches!(&int_items[0], Expr::Identifier(_))
  {
    // leading_only = true for `x -> x0`, false for `{x, x0, n}`
    // For the rule form, expand the series order until a non-zero coefficient
    // is found (capped to avoid runaway expansion).
    let (x_var_expr, x0_expr, leading_only, mut order) = match spec {
      Expr::Rule {
        pattern,
        replacement,
      } => (
        pattern.as_ref().clone(),
        replacement.as_ref().clone(),
        true,
        1,
      ),
      Expr::FunctionCall { name, args: rargs }
        if name == "Rule" && rargs.len() == 2 =>
      {
        (rargs[0].clone(), rargs[1].clone(), true, 1)
      }
      Expr::List(items) if items.len() == 3 => {
        let n =
          crate::functions::math_ast::expr_to_i128(&items[2]).unwrap_or(1);
        (items[0].clone(), items[1].clone(), false, n)
      }
      _ => {
        return Ok(unevaluated("AsymptoticIntegrate", args));
      }
    };

    let max_order = if leading_only { 20 } else { order };
    let mut series_result;
    loop {
      let series_spec = Expr::List(
        vec![x_var_expr.clone(), x0_expr.clone(), Expr::Integer(order)].into(),
      );
      series_result = series_ast(&[f.clone(), series_spec])?;
      if !leading_only {
        break;
      }
      // Check if the series is identically zero up to this order; if so,
      // expand further.
      let all_zero = match &series_result {
        Expr::Integer(0) => true,
        Expr::FunctionCall { name, args: sargs }
          if name == "SeriesData" && sargs.len() >= 6 =>
        {
          if let Expr::List(coeffs) = &sargs[2] {
            coeffs.iter().all(|c| matches!(c, Expr::Integer(0)))
          } else {
            false
          }
        }
        _ => false,
      };
      if !all_zero || order >= max_order {
        break;
      }
      order += 1;
    }

    if let Expr::FunctionCall { name, args: sargs } = &series_result
      && name == "SeriesData"
      && sargs.len() >= 6
      && let Expr::List(coeffs) = &sargs[2]
    {
      let nmin =
        crate::functions::math_ast::expr_to_i128(&sargs[3]).unwrap_or(0);
      let den =
        crate::functions::math_ast::expr_to_i128(&sargs[5]).unwrap_or(1);

      let mut terms: Vec<Expr> = Vec::new();
      for (i, coeff) in coeffs.iter().enumerate() {
        let integrated = integrate_ast(&[coeff.clone(), var.clone()])?;
        let integrated_val =
          crate::evaluator::evaluate_expr_to_expr(&integrated)?;
        if matches!(integrated_val, Expr::Integer(0)) {
          continue;
        }
        let power_num = nmin + i as i128;
        let base = if matches!(&x0_expr, Expr::Integer(0)) {
          x_var_expr.clone()
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Minus,
            left: Box::new(x_var_expr.clone()),
            right: Box::new(x0_expr.clone()),
          }
        };
        let power_expr = if power_num == 0 {
          Expr::Integer(1)
        } else if power_num == den {
          base
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(base),
            right: Box::new(crate::functions::math_ast::make_rational(
              power_num, den,
            )),
          }
        };
        let term = if matches!(power_expr, Expr::Integer(1)) {
          integrated_val
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(integrated_val),
            right: Box::new(power_expr),
          }
        };
        terms.push(crate::evaluator::evaluate_expr_to_expr(&term)?);
        if leading_only {
          break;
        }
      }

      if terms.is_empty() {
        return Ok(Expr::Integer(0));
      }
      let result = if terms.len() == 1 {
        terms.into_iter().next().unwrap()
      } else {
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: terms.into(),
        }
      };
      return crate::evaluator::evaluate_expr_to_expr(&result);
    }

    // Fallback: series wasn't a SeriesData (e.g. f is constant).
    // Use Normal -> Integrate path.
    let polynomial =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Normal".to_string(),
        args: vec![series_result].into(),
      })?;
    let integrated = integrate_ast(&[polynomial, var.clone()])?;
    return crate::evaluator::evaluate_expr_to_expr(&integrated);
  }

  // spec must be {x, x0, n}
  if !matches!(spec, Expr::List(items) if items.len() == 3) {
    return Ok(unevaluated("AsymptoticIntegrate", args));
  }

  // Try to compute the exact antiderivative first
  let antideriv_result = integrate_ast(&[f.clone(), var.clone()]);

  if let Ok(ref antideriv) = antideriv_result {
    // Check if integrate succeeded (not unevaluated)
    let is_unevaluated = matches!(antideriv, Expr::FunctionCall { name, .. } if name == "Integrate");
    if !is_unevaluated {
      // Expand the antiderivative as a series
      let series_result = series_ast(&[antideriv.clone(), spec.clone()])?;
      // Convert SeriesData to Normal polynomial
      let normal = Expr::FunctionCall {
        name: "Normal".to_string(),
        args: vec![series_result].into(),
      };
      return crate::evaluator::evaluate_expr_to_expr(&normal);
    }
  }

  // Fallback: integrate the series term by term
  let series_result = series_ast(&[f.clone(), spec.clone()])?;

  match &series_result {
    Expr::FunctionCall { name, args: sargs }
      if name == "SeriesData" && sargs.len() >= 6 =>
    {
      if let Expr::List(coeffs) = &sargs[2] {
        let x0 = &sargs[1];
        let nmin =
          crate::functions::math_ast::expr_to_i128(&sargs[3]).unwrap_or(0);
        let nmax =
          crate::functions::math_ast::expr_to_i128(&sargs[4]).unwrap_or(0);
        let den =
          crate::functions::math_ast::expr_to_i128(&sargs[5]).unwrap_or(1);

        let var_name = match var {
          Expr::Identifier(s) => s.clone(),
          _ => {
            return Ok(unevaluated("AsymptoticIntegrate", args));
          }
        };

        // Integrate each series term: coeff * (x-x0)^(k/den) -> coeff/(k/den+1) * (x-x0)^(k/den+1)
        let mut terms = Vec::new();
        for (i, coeff) in coeffs.iter().enumerate() {
          if matches!(coeff, Expr::Integer(0)) {
            continue;
          }
          let power_num = nmin + i as i128;
          // After integration: new power = (power_num + den) / den
          let new_power_num = power_num + den;
          // Skip if the integrated power exceeds the requested order n
          // nmax is the truncation order of the original series
          if new_power_num > nmax {
            continue;
          }

          let base = if matches!(x0, Expr::Integer(0)) {
            Expr::Identifier(var_name.clone())
          } else {
            Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(Expr::Identifier(var_name.clone())),
              right: Box::new(x0.clone()),
            }
          };

          // coeff / (new_power_num / den) * (x - x0)^(new_power_num/den)
          // = coeff * den / new_power_num * (x - x0)^(new_power_num/den)
          let factor = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(coeff.clone()),
            right: Box::new(crate::functions::math_ast::make_rational(
              den,
              new_power_num,
            )),
          };
          let power_expr = if new_power_num == den {
            base.clone()
          } else {
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(base),
              right: Box::new(crate::functions::math_ast::make_rational(
                new_power_num,
                den,
              )),
            }
          };
          let term = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(factor),
            right: Box::new(power_expr),
          };
          terms.push(crate::evaluator::evaluate_expr_to_expr(&term)?);
        }

        if terms.is_empty() {
          return Ok(Expr::Integer(0));
        }

        let result = if terms.len() == 1 {
          terms.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: terms.into(),
          }
        };

        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
    }
    _ => {}
  }

  Ok(unevaluated("AsymptoticIntegrate", args))
}

pub fn arc_curvature_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Reuse FrenetSerretSystem and extract the curvature
  let fss = frenet_serret_system_ast(args)?;
  // FrenetSerretSystem returns {{κ, ...}, {T, N, ...}}
  // Extract first element of first list
  if let Expr::List(outer) = &fss
    && let Some(Expr::List(curvatures)) = outer.first()
    && let Some(kappa) = curvatures.first()
  {
    // wolframscript simplifies the curvature (e.g. 1/Sqrt[Cos^2+Sin^2] -> 1).
    let simplified =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Simplify".to_string(),
        args: vec![kappa.clone()].into(),
      })
      .unwrap_or_else(|_| kappa.clone());
    return Ok(simplified);
  }
  // Fallback: return unevaluated
  Ok(unevaluated("ArcCurvature", args))
}

/// Check if an expression contains a variable by name.
fn expr_has_var(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(name) => name == var,
    Expr::Integer(_) | Expr::Real(_) | Expr::String(_) | Expr::Constant(_) => {
      false
    }
    Expr::BinaryOp { left, right, .. } => {
      expr_has_var(left, var) || expr_has_var(right, var)
    }
    Expr::UnaryOp { operand, .. } => expr_has_var(operand, var),
    Expr::FunctionCall { name: n, args } => {
      if n == "Rational" {
        return false;
      }
      args.iter().any(|a| expr_has_var(a, var))
    }
    Expr::List(items) => items.iter().any(|a| expr_has_var(a, var)),
    _ => false,
  }
}

/// Try to compute DifferenceDelta for exponential expressions a^f(x).
/// Returns Some(simplified_expr) if the expression is Power[base, exponent]
/// where base doesn't depend on the variable.
/// Δ[a^f(x), {x, h}] = a^f(x) * (a^(f(x+h)-f(x)) - 1)
fn try_exponential_delta(expr: &Expr, var: &str, step: &Expr) -> Option<Expr> {
  let (base, exponent) = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => (left.as_ref(), right.as_ref()),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    _ => return None,
  };

  // Base must not contain the variable
  if expr_has_var(base, var) {
    return None;
  }

  // Exponent must contain the variable
  if !expr_has_var(exponent, var) {
    return None;
  }

  // Compute f(x+h) - f(x) for the exponent
  let x_plus_h = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(Expr::Identifier(var.to_string())),
    right: Box::new(step.clone()),
  };
  let shifted_exp =
    crate::syntax::substitute_variable(exponent, var, &x_plus_h);

  // delta_exp = Expand[shifted_exp - exponent]
  let delta_exp = Expr::FunctionCall {
    name: "Expand".to_string(),
    args: vec![Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(shifted_exp),
      right: Box::new(exponent.clone()),
    }]
    .into(),
  };

  // Result: base^exponent * (base^delta_exp - 1)
  let result = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(expr.clone()),
    right: Box::new(Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base.clone()),
        right: Box::new(delta_exp),
      }),
      right: Box::new(Expr::Integer(1)),
    }),
  };

  Some(result)
}

/// Try to compute DifferenceDelta for Sin/Cos using sum-to-product identities.
/// Δ[Sin[f(x)]] = 2*Sin[Δf/2]*Sin[Pi/2 + (f(x+h)+f(x))/2]
/// Δ[Cos[f(x)]] = -2*Sin[(f(x+h)+f(x))/2]*Sin[Δf/2]
/// where Δf = f(x+h) - f(x).
fn try_trig_delta(expr: &Expr, var: &str, step: &Expr) -> Option<Expr> {
  let (fn_name, arg) = match expr {
    Expr::FunctionCall { name, args }
      if (name == "Sin" || name == "Cos") && args.len() == 1 =>
    {
      (name.as_str(), &args[0])
    }
    _ => return None,
  };

  // Argument must contain the variable
  if !expr_has_var(arg, var) {
    return None;
  }

  let x_plus_h = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(Expr::Identifier(var.to_string())),
    right: Box::new(step.clone()),
  };
  let shifted_arg = crate::syntax::substitute_variable(arg, var, &x_plus_h);

  // Compute half_delta = (shifted_arg - arg) / 2 via evaluation
  let half_delta_raw = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(shifted_arg.clone()),
        right: Box::new(arg.clone()),
      }]
      .into(),
    }),
    right: Box::new(Expr::Integer(2)),
  };
  let half_delta =
    crate::evaluator::evaluate_expr_to_expr(&half_delta_raw).ok()?;

  // Build second Sin argument:
  // For Sin: arg + (step + Pi)/2  (= arg + half_delta + Pi/2, combined fraction)
  // For Cos: arg + (step - Pi)/2
  // We construct (step ± Pi)/2 as a single fraction, then add arg.
  let pi_term = if fn_name == "Sin" {
    Expr::Constant("Pi".to_string())
  } else {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Constant("Pi".to_string())),
    }
  };
  // Build (2*half_delta + Pi) / 2  for the constant part, but use direct
  // (step + Pi) / 2 to get a cleaner fraction when possible.
  // General form: half_delta + Pi/2 + arg, combined as arg + (2*half_delta ± Pi)/2
  let const_part = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(2)),
        right: Box::new(half_delta.clone()),
      }),
      right: Box::new(pi_term),
    }),
    right: Box::new(Expr::Integer(2)),
  };
  let second_arg_expr = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(const_part),
    right: Box::new(arg.clone()),
  };

  let coeff = if fn_name == "Sin" {
    Expr::Integer(2)
  } else {
    Expr::Integer(-2)
  };

  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      coeff,
      Expr::FunctionCall {
        name: "Sin".to_string(),
        args: vec![half_delta].into(),
      },
      Expr::FunctionCall {
        name: "Sin".to_string(),
        args: vec![second_arg_expr].into(),
      },
    ]
    .into(),
  };

  Some(result)
}

/// DiscreteShift[f, x] — the discrete shift operator: substitutes x → x+1 in f.
/// DiscreteShift[f, {x, k}] shifts by k. Additional variable arguments are each
/// shifted by 1 (`DiscreteShift[f, x, y]`). The substituted result is expanded
/// only when its top-level head is Plus, matching wolframscript. A specifier
/// that is not a valid variable emits `General::ivar` and stays unevaluated.
pub fn discrete_shift_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DiscreteShift", args));
  if args.is_empty() {
    return unevaluated();
  }
  // With no variable specifier the shift is the identity.
  if args.len() == 1 {
    return crate::evaluator::evaluate_expr_to_expr(&args[0]);
  }

  // Parse and validate every variable specifier first so a single bad one
  // leaves the whole expression unevaluated (no partial shifts).
  let mut specs: Vec<(String, Expr)> = Vec::with_capacity(args.len() - 1);
  for spec in &args[1..] {
    match spec {
      Expr::Identifier(name) => {
        specs.push((name.clone(), Expr::Integer(1)));
      }
      Expr::List(items)
        if items.len() == 2 && matches!(&items[0], Expr::Identifier(_)) =>
      {
        if let Expr::Identifier(name) = &items[0] {
          specs.push((name.clone(), items[1].clone()));
        }
      }
      _ => {
        crate::emit_message(&format!(
          "General::ivar: {} is not a valid variable.",
          crate::syntax::expr_to_string(spec)
        ));
        return unevaluated();
      }
    }
  }

  // Apply each shift x → x + k.
  let mut result = args[0].clone();
  for (var, k) in &specs {
    let shifted = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![k.clone(), Expr::Identifier(var.clone())].into(),
    };
    result = crate::syntax::substitute_variable(&result, var, &shifted);
  }
  let mut result = crate::evaluator::evaluate_expr_to_expr(&result)?;

  // With an integer shift wolframscript combines the result over a common
  // denominator, so `DiscreteShift[1/(2 n + 1), n]` becomes `(3 + 2 n)^-1`
  // rather than the unsimplified `(1 + 2 (1 + n))^-1`. A symbolic shift is left
  // as-is (e.g. `(1 + 2 (k + n))^-1`), so only fold for a purely integer step.
  if specs.iter().all(|(_, k)| matches!(k, Expr::Integer(_))) {
    result = crate::functions::polynomial_ast::together_ast(&[result])?;
  }

  // wolframscript expands the result only when its top-level head is Plus.
  let head =
    crate::evaluator::evaluate_function_call_ast("Head", &[result.clone()])?;
  if matches!(&head, Expr::Identifier(h) if h == "Plus") {
    return crate::functions::polynomial_ast::expand_ast(&[result]);
  }
  Ok(result)
}

/// DiscreteRatio[f, x] — the multiplicative analog of DifferenceDelta: the
/// ratio operator R[f] = DiscreteShift[f, x] / f (i.e. f(x+1)/f(x)).
/// DiscreteRatio[f, {x, k}] applies the operator k times, and the optional step
/// in `{x, k, h}` makes each shift x → x+h. Extra variable arguments compose
/// the operator in each variable. The result is simplified to match
/// wolframscript.
pub fn discrete_ratio_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DiscreteRatio", args));
  if args.is_empty() {
    return unevaluated();
  }
  if args.len() == 1 {
    return crate::evaluator::evaluate_expr_to_expr(&args[0]);
  }

  enum SpecResult {
    Ok(String, usize, Expr),
    Symbolic,
    Ivar(Expr),
    Dvar(Expr),
  }
  fn parse_spec(spec: &Expr) -> SpecResult {
    use crate::functions::math_ast::expr_to_i128;
    match spec {
      Expr::Identifier(v) => SpecResult::Ok(v.clone(), 1, Expr::Integer(1)),
      Expr::List(items)
        if !items.is_empty() && matches!(&items[0], Expr::Identifier(_)) =>
      {
        let Expr::Identifier(v) = &items[0] else {
          unreachable!()
        };
        if items.len() == 1 {
          return SpecResult::Ok(v.clone(), 1, Expr::Integer(1));
        }
        let order = match expr_to_i128(&items[1]) {
          Some(k) if k >= 0 => k as usize,
          Some(_) => return SpecResult::Dvar(spec.clone()),
          None if matches!(&items[1], Expr::Identifier(_)) => {
            return SpecResult::Symbolic;
          }
          None => return SpecResult::Dvar(spec.clone()),
        };
        let step = if items.len() >= 3 {
          items[2].clone()
        } else {
          Expr::Integer(1)
        };
        SpecResult::Ok(v.clone(), order, step)
      }
      _ => SpecResult::Ivar(spec.clone()),
    }
  }

  let mut specs: Vec<(String, usize, Expr)> = Vec::new();
  for spec in &args[1..] {
    match parse_spec(spec) {
      SpecResult::Ok(v, o, s) => specs.push((v, o, s)),
      SpecResult::Symbolic => return unevaluated(),
      SpecResult::Ivar(sp) => {
        crate::emit_message(&format!(
          "General::ivar: {} is not a valid variable.",
          crate::syntax::expr_to_string(&sp)
        ));
        return unevaluated();
      }
      SpecResult::Dvar(sp) => {
        crate::emit_message(&format!(
          "DiscreteRatio::dvar: Ratio specifier {} does not have the form \
           {{variable, n}} or {{variable, n, h}}, where n is a non-negative \
           machine integer.",
          crate::syntax::expr_to_string(&sp)
        ));
        return unevaluated();
      }
    }
  }

  let mut result = args[0].clone();
  for (var, order, step) in &specs {
    for _ in 0..*order {
      // R[g] = DiscreteShift[g, {var, step}] / g.
      let shift_spec =
        Expr::List(vec![Expr::Identifier(var.clone()), step.clone()].into());
      let shifted = discrete_shift_ast(&[result.clone(), shift_spec])?;
      let ratio = Expr::FunctionCall {
        name: "Divide".to_string(),
        args: vec![shifted, result.clone()].into(),
      };
      result = crate::evaluator::evaluate_expr_to_expr(&ratio)?;
    }
  }
  // Cancel common factors to reach wolframscript's canonical ratio form
  // (Simplify would over-split e.g. (1+n)/n into 1 + 1/n).
  let canceled =
    crate::evaluator::evaluate_function_call_ast("Cancel", &[result])?;
  // FunctionExpand collapses factorial/Gamma/Pochhammer ratios that Cancel
  // leaves alone — e.g. DiscreteRatio[n!, n] = (1 + n)!/n! → 1 + n — while
  // preserving plain rational forms like (2 + n)/n, matching wolframscript.
  // Only adopt it when it fully reduces the ratio (no Gamma/Factorial left):
  // for higher-order or scaled arguments (e.g. (2 n)!) woxi can't reduce the
  // residual Gamma ratio, so keep the cancelled form rather than swap one
  // unreduced display for another.
  let expanded = crate::evaluator::evaluate_function_call_ast(
    "FunctionExpand",
    &[canceled.clone()],
  )?;
  let fully_reduced = !expr_mentions_head(&expanded, "Gamma")
    && !expr_mentions_head(&expanded, "Factorial");
  Ok(if fully_reduced { expanded } else { canceled })
}

/// DifferenceDelta[f, x] = f(x+1) - f(x)
/// DifferenceDelta[f, {x, n}] = n-th order forward difference with step 1
/// DifferenceDelta[f, {x, n, h}] = n-th order forward difference with step h
pub fn difference_delta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated("DifferenceDelta", args));
  }

  let expr = &args[0];

  // Parse second argument: x, {x, n}, or {x, n, h}
  let (var_name, order, step) = if args.len() == 1 {
    return Ok(unevaluated("DifferenceDelta", args));
  } else {
    match &args[1] {
      Expr::Identifier(name) => (name.clone(), 1usize, Expr::Integer(1)),
      Expr::List(items) if !items.is_empty() => {
        let var = match &items[0] {
          Expr::Identifier(name) => name.clone(),
          _ => {
            return Ok(unevaluated("DifferenceDelta", args));
          }
        };
        let n = if items.len() >= 2 {
          match crate::functions::math_ast::expr_to_i128(&items[1]) {
            Some(n) if n >= 0 => n as usize,
            _ => {
              return Ok(unevaluated("DifferenceDelta", args));
            }
          }
        } else {
          1
        };
        let h = if items.len() >= 3 {
          items[2].clone()
        } else {
          Expr::Integer(1)
        };
        (var, n, h)
      }
      _ => {
        return Ok(unevaluated("DifferenceDelta", args));
      }
    }
  };

  if order == 0 {
    return crate::evaluator::evaluate_expr_to_expr(expr);
  }

  // Apply forward difference operator n times
  let mut current = expr.clone();
  for _ in 0..order {
    // Special case: Power[base, exponent] where base is independent of the variable.
    // Δ[base^f(x), {x, h}] = base^f(x) * (base^Δ[f(x)] - 1), which for f(x)=x
    // gives base^x * (base^h - 1). This avoids unsimplified a^(x+h) - a^x forms.
    if let Some(result) = try_exponential_delta(&current, &var_name, &step) {
      current = crate::evaluator::evaluate_expr_to_expr(&result)?;
      continue;
    }

    // Special case: Sin[f(x)] or Cos[f(x)] — use sum-to-product identities.
    // Δ[Sin[f(x)]] = 2*Cos[(f(x+h)+f(x))/2]*Sin[(f(x+h)-f(x))/2]
    // Δ[Cos[f(x)]] = -2*Sin[(f(x+h)+f(x))/2]*Sin[(f(x+h)-f(x))/2]
    // Then Cos[θ] → Sin[Pi/2 + θ] and -Sin[θ] → Sin[-Pi/2 + θ] to match Wolfram.
    if let Some(result) = try_trig_delta(&current, &var_name, &step) {
      current = crate::evaluator::evaluate_expr_to_expr(&result)?;
      continue;
    }

    // f(x + h)
    let x_plus_h = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Identifier(var_name.clone())),
      right: Box::new(step.clone()),
    };
    let shifted =
      crate::syntax::substitute_variable(&current, &var_name, &x_plus_h);
    // f(x + h) - f(x)
    let diff = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(shifted),
      right: Box::new(current.clone()),
    };
    // wolframscript canonicalizes with a numeric step by factoring (which also
    // combines rational terms over a common denominator, e.g.
    // Δ[1/(2 n + 1)] = -2/((1 + 2 n) (3 + 2 n))); with a symbolic step it leaves
    // the result expanded (e.g. Δ[x^2, {x, 1, h}] = h^2 + 2 h x).
    let head = if matches!(step, Expr::Integer(_)) {
      "Factor"
    } else {
      "Expand"
    };
    let canonicalized = Expr::FunctionCall {
      name: head.to_string(),
      args: vec![diff].into(),
    };
    current = crate::evaluator::evaluate_expr_to_expr(&canonicalized)?;
  }

  Ok(current)
}

/// DifferenceQuotient[f, {x, h}] = (f(x+h) - f(x)) / h
/// DifferenceQuotient[f, x] = f(x+1) - f(x) (i.e. DifferenceDelta with step 1)
/// DifferenceQuotient[f, {x, h}, n] = n-th order difference quotient
pub fn difference_quotient_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Ok(unevaluated("DifferenceQuotient", args));
  }

  let expr = &args[0];

  // Parse second argument: only {x, h} form is supported (bare x returns unevaluated)
  let (var_name, step) = if args.len() >= 2 {
    match &args[1] {
      Expr::List(items) if items.len() == 2 => {
        let var = match &items[0] {
          Expr::Identifier(name) => name.clone(),
          _ => {
            return Ok(unevaluated("DifferenceQuotient", args));
          }
        };
        (var, items[1].clone())
      }
      _ => {
        return Ok(unevaluated("DifferenceQuotient", args));
      }
    }
  } else {
    return Ok(unevaluated("DifferenceQuotient", args));
  };

  let order = if args.len() == 3 {
    match crate::functions::math_ast::expr_to_i128(&args[2]) {
      Some(n) if n >= 0 => n as usize,
      _ => {
        return Ok(unevaluated("DifferenceQuotient", args));
      }
    }
  } else {
    1
  };

  if order == 0 {
    return crate::evaluator::evaluate_expr_to_expr(expr);
  }

  // Apply difference quotient n times: (DifferenceDelta[f, {x, 1, h}]) / h^n
  // Build DifferenceDelta args
  let delta_args = vec![
    expr.clone(),
    Expr::List(
      vec![
        Expr::Identifier(var_name.clone()),
        Expr::Integer(order as i128),
        step.clone(),
      ]
      .into(),
    ),
  ];
  let delta_result = difference_delta_ast(&delta_args)?;

  // Divide by h^n
  let divisor = if order == 1 {
    step.clone()
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(step.clone()),
      right: Box::new(Expr::Integer(order as i128)),
    }
  };

  let quotient = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(delta_result),
    right: Box::new(divisor),
  };
  let result = Expr::FunctionCall {
    name: "Cancel".to_string(),
    args: vec![quotient].into(),
  };

  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Helper: compute dot product of two vectors
fn dot_product(a: &[Expr], b: &[Expr]) -> Expr {
  let terms: Vec<Expr> = a
    .iter()
    .zip(b.iter())
    .map(|(ai, bi)| Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(ai.clone()),
      right: Box::new(bi.clone()),
    })
    .collect();
  if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    }
  }
}

/// Compute Series[QFactorial[n, q], {q, 0, order}] as a polynomial.
/// QFactorial[n, q] = prod_{i=1..n} (1 + q + q^2 + ... + q^{i-1}).
fn qfactorial_series_at_zero(var_name: &str, n: usize, order: i128) -> Expr {
  let limit = order.max(0) as usize;
  // Start with constant polynomial 1.
  let mut coeffs: Vec<i128> = vec![1];
  for i in 1..=n {
    // Multiply by (1 + q + q^2 + ... + q^{i-1}).
    let factor_len = i;
    let new_len = (coeffs.len() + factor_len - 1).min(limit + 1);
    let mut new_coeffs = vec![0i128; new_len];
    for (a, ca) in coeffs.iter().enumerate() {
      if *ca == 0 {
        continue;
      }
      for b in 0..factor_len {
        let idx = a + b;
        if idx > limit {
          break;
        }
        new_coeffs[idx] += ca;
      }
    }
    coeffs = new_coeffs;
  }
  // Ensure the coefficient list has exactly limit+1 entries.
  coeffs.resize(limit + 1, 0);
  let coeff_exprs: Vec<Expr> = coeffs.into_iter().map(Expr::Integer).collect();
  Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name.to_string()),
      Expr::Integer(0),
      Expr::List(coeff_exprs.into()),
      Expr::Integer(0),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ]
    .into(),
  }
}

/// Build SeriesData[var, 0, {...}, 1, order+1, 1] for `Series[BarnesG[var], {var, 0, order}]`.
fn barnes_g_series_at_zero(var_name: &str, order: i128) -> Expr {
  let mut coeffs: Vec<Expr> = Vec::with_capacity(order.max(0) as usize);
  for k in 1..=order {
    coeffs.push(barnes_g_series_coefficient(k));
  }
  Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name.to_string()),
      Expr::Integer(0),
      Expr::List(coeffs.into()),
      Expr::Integer(1),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ]
    .into(),
  }
}

/// Coefficient `a_k` in the expansion BarnesG[z] = sum_{k≥1} a_k z^k around 0.
fn barnes_g_series_coefficient(k: i128) -> Expr {
  let evaluated = |e: Expr| -> Expr {
    crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e)
  };
  let int = |n: i128| Expr::Integer(n);
  let sym = |s: &str| Expr::Identifier(s.to_string());
  let constant = |s: &str| Expr::Constant(s.to_string());
  let plus = |args: Vec<Expr>| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: args.into(),
  };
  let times = |args: Vec<Expr>| Expr::FunctionCall {
    name: "Times".to_string(),
    args: args.into(),
  };
  let log = |arg: Expr| Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![arg].into(),
  };
  let rational = |p: i128, q: i128| Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![int(p), int(q)].into(),
  };
  let two_pi = times(vec![int(2), constant("Pi")]);
  let log_2pi = log(two_pi);
  // (-1 + Log[2*Pi]) / 2
  let half_term =
    times(vec![rational(1, 2), plus(vec![int(-1), log_2pi.clone()])]);

  match k {
    1 => int(1),
    2 => evaluated(plus(vec![sym("EulerGamma"), half_term])),
    _ => Expr::FunctionCall {
      name: "Derivative".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Derivative".to_string(),
          args: vec![int(k)].into(),
        },
        sym("BarnesG"),
      ]
      .into(),
    },
  }
}

/// `Series[x!, {x, 0, order}]` — Taylor expansion of Factorial[x] at 0,
/// expressed with EulerGamma and Pi to match wolframscript's display.
fn factorial_series_at_zero(var_name: &str, order: i128) -> Expr {
  let int = |n: i128| Expr::Integer(n);
  let sym = |s: &str| Expr::Identifier(s.to_string());
  let constant = |s: &str| Expr::Constant(s.to_string());
  let plus = |args: Vec<Expr>| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: args.into(),
  };
  let times = |args: Vec<Expr>| Expr::FunctionCall {
    name: "Times".to_string(),
    args: args.into(),
  };
  let power = |base: Expr, exp: Expr| Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![base, exp].into(),
  };
  let rational = |p: i128, q: i128| Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![int(p), int(q)].into(),
  };

  let mut coeffs: Vec<Expr> = Vec::with_capacity(order.max(0) as usize + 1);
  for k in 0..=order {
    let raw = match k {
      0 => int(1),
      1 => times(vec![int(-1), sym("EulerGamma")]),
      2 => {
        // (6 EulerGamma^2 + Pi^2) / 12.
        let num = plus(vec![
          times(vec![int(6), power(sym("EulerGamma"), int(2))]),
          power(constant("Pi"), int(2)),
        ]);
        times(vec![rational(1, 12), num])
      }
      _ => unreachable!(),
    };
    coeffs.push(crate::evaluator::evaluate_expr_to_expr(&raw).unwrap_or(raw));
  }
  Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name.to_string()),
      Expr::Integer(0),
      Expr::List(coeffs.into()),
      Expr::Integer(0),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ]
    .into(),
  }
}

/// `Series[WeberE[v, z], {z, 0, order}]` (`is_weber == true`) or
/// `Series[AngerJ[v, z], {z, 0, order}]` (`is_weber == false`) using their
/// closed-form Taylor coefficients at z = 0:
///
/// WeberE: c_k(v) = (1 - (-1)^k Cos[π v]) / (π * denom_k(v))
/// AngerJ: c_k(v) = (-1)^k Sin[π v] / (π * denom_k(v))
///
/// where denom_k(v) is the product of (v^2 - j^2) over j ∈ {k, k-2, …}
/// down to 1 (if k odd) or 0 (if k even); j = 0 contributes a bare `v`.
fn weber_anger_series_at_zero(
  var_name: &str,
  nu: &Expr,
  order: i128,
  is_weber: bool,
) -> Expr {
  let int = |n: i128| Expr::Integer(n);
  let constant = |s: &str| Expr::Constant(s.to_string());
  let times = |args: Vec<Expr>| Expr::FunctionCall {
    name: "Times".to_string(),
    args: args.into(),
  };
  let power = |base: Expr, exp: Expr| Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![base, exp].into(),
  };
  let plus = |args: Vec<Expr>| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: args.into(),
  };
  let divide = |left: Expr, right: Expr| Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(left),
    right: Box::new(right),
  };

  let pi = constant("Pi");
  let nu_pi = times(vec![nu.clone(), pi.clone()]);

  // Build the numerator for coefficient k:
  //   WeberE: 1 - (-1)^k * Cos[π v]    (i.e. 1 - Cos[πν] if k even, 1 + Cos[πν] if k odd)
  //   AngerJ:        (-1)^k * Sin[π v]
  let make_numerator = |k: i128| -> Expr {
    let trig = Expr::FunctionCall {
      name: if is_weber { "Cos" } else { "Sin" }.to_string(),
      args: vec![nu_pi.clone()].into(),
    };
    let k_even = k % 2 == 0;
    if is_weber {
      // 1 - sign*Cos where sign = (-1)^k
      if k_even {
        plus(vec![int(1), times(vec![int(-1), trig])])
      } else {
        plus(vec![int(1), trig])
      }
    } else if k_even {
      trig
    } else {
      times(vec![int(-1), trig])
    }
  };

  // Build the denominator factors (without the leading π):
  //   For j = 0: factor is v (just `nu`)
  //   For j > 0: factor is v^2 - j^2
  // Returns a single Expr (Times if multiple factors).
  let make_denom_no_pi = |k: i128| -> Expr {
    let mut factors: Vec<Expr> = Vec::new();
    // Collect j values: k, k-2, k-4, ..., down to 0 or 1.
    let mut j = k;
    while j >= 0 {
      if j == 0 {
        factors.push(nu.clone());
      } else {
        // v^2 - j^2
        factors.push(plus(vec![
          times(vec![int(-j * j), int(1)]),
          power(nu.clone(), int(2)),
        ]));
      }
      j -= 2;
    }
    factors.reverse();
    if factors.len() == 1 {
      factors.remove(0)
    } else {
      times(factors)
    }
  };

  let mut coeffs: Vec<Expr> = Vec::with_capacity((order.max(0) + 1) as usize);
  for k in 0..=order {
    let num = make_numerator(k);
    let denom_no_pi = make_denom_no_pi(k);
    let denom = times(vec![pi.clone(), denom_no_pi]);
    let raw = divide(num, denom);
    coeffs.push(crate::evaluator::evaluate_expr_to_expr(&raw).unwrap_or(raw));
  }
  Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name.to_string()),
      Expr::Integer(0),
      Expr::List(coeffs.into()),
      Expr::Integer(0),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ]
    .into(),
  }
}

/// `Series[x!!, {x, 0, order}]` — Taylor expansion of `Factorial2[x]` at 0.
/// Coefficients involve EulerGamma, Log[2], Log[Pi], Log[64], and Pi^2.
fn factorial2_series_at_zero(var_name: &str, order: i128) -> Expr {
  let int = |n: i128| Expr::Integer(n);
  let sym = |s: &str| Expr::Identifier(s.to_string());
  let constant = |s: &str| Expr::Constant(s.to_string());
  let plus = |args: Vec<Expr>| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: args.into(),
  };
  let times = |args: Vec<Expr>| Expr::FunctionCall {
    name: "Times".to_string(),
    args: args.into(),
  };
  let power = |base: Expr, exp: Expr| Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![base, exp].into(),
  };
  let rational = |p: i128, q: i128| Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![int(p), int(q)].into(),
  };
  let log = |arg: Expr| Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![arg].into(),
  };

  let mut coeffs: Vec<Expr> = Vec::with_capacity(order.max(0) as usize + 1);
  for k in 0..=order {
    let raw = match k {
      0 => int(1),
      1 => {
        // (-EulerGamma + Log[2]) / 2
        times(vec![
          rational(1, 2),
          plus(vec![times(vec![int(-1), sym("EulerGamma")]), log(int(2))]),
        ])
      }
      2 => {
        // (6 (EulerGamma - Log[2])^2 + Pi^2 (1 + Log[64] - 6 Log[Pi])) / 48
        let eg_minus_log2 =
          plus(vec![sym("EulerGamma"), times(vec![int(-1), log(int(2))])]);
        let part_a = times(vec![int(6), power(eg_minus_log2, int(2))]);
        let part_b = times(vec![
          power(constant("Pi"), int(2)),
          plus(vec![
            int(1),
            log(int(64)),
            times(vec![int(-6), log(constant("Pi"))]),
          ]),
        ]);
        times(vec![rational(1, 48), plus(vec![part_a, part_b])])
      }
      _ => unreachable!(),
    };
    coeffs.push(crate::evaluator::evaluate_expr_to_expr(&raw).unwrap_or(raw));
  }
  Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name.to_string()),
      Expr::Integer(0),
      Expr::List(coeffs.into()),
      Expr::Integer(0),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ]
    .into(),
  }
}

/// `Series[FactorialPower[x, n], {x, 0, order}]` for non-negative integer n.
///
/// `FactorialPower[x, n]` is the falling factorial x*(x-1)*...*(x-n+1), a
/// polynomial whose Taylor coefficients at 0 are the signed Stirling numbers
/// of the first kind s(n, k). The leading non-zero coefficient is s(n, 1)
/// at x^1 (or s(0, 0) = 1 when n = 0).
fn factorial_power_series_at_zero(
  var_name: &str,
  n: i128,
  order: i128,
) -> Expr {
  let n_usize = n as usize;
  let stirling = signed_stirling_first_row(n_usize);
  let (min_idx, max_idx) = if n == 0 {
    (0usize, 0usize)
  } else {
    (1usize, n_usize)
  };
  let upper = max_idx.min(order.max(0) as usize);
  let coeffs: Vec<Expr> = (min_idx..=upper)
    .map(|k| Expr::Integer(stirling.get(k).copied().unwrap_or(0)))
    .collect();
  Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name.to_string()),
      Expr::Integer(0),
      Expr::List(coeffs.into()),
      Expr::Integer(min_idx as i128),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ]
    .into(),
  }
}

/// `Series[Hyperfactorial[x], {x, 0, order}]` — closed-form low-order
/// Taylor coefficients of `Hyperfactorial[x]` at 0.
/// `Series[Pochhammer[x, 1/2], {x, 0, order}]` for orders 0..=2.
///   Pochhammer[x, 1/2] = Sqrt[π] x − Sqrt[π] Log[4] x² + O(x³).
fn pochhammer_half_series_at_zero(var_name: &str, order: i128) -> Expr {
  let int = |n: i128| Expr::Integer(n);
  let constant = |s: &str| Expr::Constant(s.to_string());
  let times = |args: Vec<Expr>| Expr::FunctionCall {
    name: "Times".to_string(),
    args: args.into(),
  };
  let sqrt_pi = Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![constant("Pi")].into(),
  };
  let log = |arg: Expr| Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![arg].into(),
  };

  // Coefficients starting at x^1 (so SeriesData min = 1).
  let mut coeffs: Vec<Expr> = Vec::with_capacity(order.max(0) as usize);
  for k in 1..=order {
    let raw = match k {
      1 => sqrt_pi.clone(),
      2 => times(vec![int(-1), sqrt_pi.clone(), log(int(4))]),
      _ => unreachable!(),
    };
    coeffs.push(crate::evaluator::evaluate_expr_to_expr(&raw).unwrap_or(raw));
  }
  Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name.to_string()),
      Expr::Integer(0),
      Expr::List(coeffs.into()),
      Expr::Integer(1),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ]
    .into(),
  }
}

fn hyperfactorial_series_at_zero(var_name: &str, order: i128) -> Expr {
  let int = |n: i128| Expr::Integer(n);
  let sym = |s: &str| Expr::Identifier(s.to_string());
  let constant = |s: &str| Expr::Constant(s.to_string());
  let plus = |args: Vec<Expr>| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: args.into(),
  };
  let times = |args: Vec<Expr>| Expr::FunctionCall {
    name: "Times".to_string(),
    args: args.into(),
  };
  let power = |base: Expr, exp: Expr| Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![base, exp].into(),
  };
  let rational = |p: i128, q: i128| Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![int(p), int(q)].into(),
  };
  let log = |arg: Expr| Expr::FunctionCall {
    name: "Log".to_string(),
    args: vec![arg].into(),
  };
  let two_pi = times(vec![int(2), constant("Pi")]);
  let log_2pi = log(two_pi);

  let mut coeffs: Vec<Expr> = Vec::with_capacity(order.max(0) as usize + 1);
  for k in 0..=order {
    let raw = match k {
      0 => int(1),
      1 => {
        // (1 - Log[2*Pi]) / 2
        times(vec![
          rational(1, 2),
          plus(vec![int(1), times(vec![int(-1), log_2pi.clone()])]),
        ])
      }
      2 => {
        // (5 - 4 EulerGamma - 2 Log[2 Pi] + Log[2 Pi]^2) / 8
        times(vec![
          rational(1, 8),
          plus(vec![
            int(5),
            times(vec![int(-4), sym("EulerGamma")]),
            times(vec![int(-2), log_2pi.clone()]),
            power(log_2pi.clone(), int(2)),
          ]),
        ])
      }
      _ => unreachable!(),
    };
    coeffs.push(crate::evaluator::evaluate_expr_to_expr(&raw).unwrap_or(raw));
  }
  Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name.to_string()),
      Expr::Integer(0),
      Expr::List(coeffs.into()),
      Expr::Integer(0),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ]
    .into(),
  }
}

/// Row `n` of the signed Stirling numbers of the first kind.
/// `out[k] = s(n, k)`, satisfying x(x-1)...(x-n+1) = sum_k s(n, k) x^k.
fn signed_stirling_first_row(n: usize) -> Vec<i128> {
  // Start at row 0: s(0, 0) = 1, all other s(0, k) = 0.
  let mut row = vec![0i128; n + 1];
  row[0] = 1;
  let mut next = vec![0i128; n + 1];
  // Build up row by row using s(m+1, k) = s(m, k-1) - m * s(m, k).
  for m in 0..n {
    for k in 0..=(m + 1) {
      let from_left = if k > 0 { row[k - 1] } else { 0 };
      let from_above = row[k];
      next[k] = from_left - (m as i128) * from_above;
    }
    (row, next) = (next, row);
  }
  row
}
