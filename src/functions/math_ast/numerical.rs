use super::*;
use crate::evaluator::Attributes;
use crate::syntax::substitute_variable;

pub fn n_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "N expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 2 {
    // N[expr, MachinePrecision] is identical to N[expr] — return a
    // machine-precision Real, not a BigFloat.
    if matches!(&args[1], Expr::Identifier(s) if s == "MachinePrecision") {
      return n_eval(&args[0]);
    }
    // N[expr, {precision, accuracy}] — the effective precision of the result is
    // min(p, a + Log10[|value|]); the digits are those of N[expr, p]. For a
    // value of magnitude ≈ 1 this equals N[expr, p], but e.g. N[1/3, {6, 6}]
    // carries the accuracy-limited precision tag 5.5228… rather than 6.
    if let Expr::List(spec) = &args[1]
      && spec.len() == 2
      && let (Some(p), Some(a)) =
        (try_eval_to_f64(&spec[0]), try_eval_to_f64(&spec[1]))
      && p > 0.0
      && a > 0.0
    {
      let magnitude = try_eval_to_f64(&args[0]).map(f64::abs).or_else(|| {
        try_extract_complex_float(&args[0]).map(|(re, im)| re.hypot(im))
      });
      let effective = match magnitude {
        Some(m) if m > 0.0 => p.min(a + m.log10()),
        _ => p,
      };
      return n_eval_arbitrary(&args[0], effective);
    }
    // N[expr, precision] — arbitrary-precision evaluation. Precision
    // can be a numeric expression (e.g. N[Pi, Pi] uses ≈3.14159… as
    // the precision marker); we preserve the full f64 value so the
    // emitted BigFloat carries the exact precision tag wolframscript
    // shows.
    let precision_f64 = match &args[1] {
      Expr::Integer(n) if *n > 0 => *n as f64,
      other => {
        if let Some(v) = try_eval_to_f64(other)
          && v.floor() as i128 > 0
        {
          v
        } else {
          crate::emit_message(&format!(
            "N::precbd: Requested precision {} is not a machine-sized real number between $MinPrecision and $MaxPrecision.",
            expr_to_string(other)
          ));
          return Ok(unevaluated("N", args));
        }
      }
    };
    return n_eval_arbitrary_precision_form(&args[0], precision_f64);
  }
  n_eval(&args[0])
}

/// Arbitrary-precision `N[expr, p]` for the plain-precision spec (not the
/// `{precision, accuracy}` form). Exact zeros collapse to the exact integer 0
/// — `N[0, p]` stays `0` (Head Integer, Precision Infinity) in wolframscript
/// rather than becoming a precision-tagged BigFloat zero. Lists are handled
/// element-wise so e.g. `N[{0, 1/3}, 20]` keeps the zero exact while the
/// other elements pick up the precision tag.
fn n_eval_arbitrary_precision_form(
  expr: &Expr,
  precision: f64,
) -> Result<Expr, InterpreterError> {
  let is_exact_zero = match expr {
    Expr::Integer(n) => *n == 0,
    Expr::BigInteger(n) => n.sign() == Sign::NoSign,
    _ => false,
  };
  if is_exact_zero {
    return Ok(Expr::Integer(0));
  }
  if let Expr::List(items) = expr {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|e| n_eval_arbitrary_precision_form(e, precision))
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  if let Expr::FunctionCall { name, args } = expr
    && let Some(sparse) = n_sparse_array(name, args, |e| {
      n_eval_arbitrary_precision_form(e, precision)
    })
  {
    return sparse;
  }
  n_eval_arbitrary(expr, precision)
}

/// Does `N` leave the right operand of `op` exact?
///
/// Wolfram's `N` never numericizes an exact rational exponent: `N[x^2]` stays
/// `x^2` (not `x^2.`) and `N[Sqrt[x]]` stays `Sqrt[x]`. Keeping the exponent
/// exact is what lets `Solve`/`NSolve` still recognise `N[poly]` as a
/// polynomial — `N[BernoulliB[20, z]] == 0` is solvable only because the
/// `z^20` term keeps its integer exponent. A non-rational exponent is still
/// numericized (`N[x^Pi]` → `x^3.14159…`).
fn keeps_exact_exponent(op: BinaryOperator, exponent: &Expr) -> bool {
  op == BinaryOperator::Power && is_exact_rational(exponent)
}

/// Is this an exact rational literal (integer, big integer, `p/q`, or a
/// negation of one)?
fn is_exact_rational(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) => true,
    Expr::FunctionCall { name, args } if name == "Rational" => {
      args.len() == 2 && args.iter().all(is_exact_rational)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_exact_rational(operand),
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => is_exact_rational(left) && is_exact_rational(right),
    _ => false,
  }
}

/// `$MachinePrecision`: the decimal digits of a machine real.
pub(crate) const MACHINE_PRECISION_DIGITS: f64 = 15.954589770191003;

/// Recursively convert an expression to numeric (Real) form
pub fn n_eval(expr: &Expr) -> Result<Expr, InterpreterError> {
  let value = n_eval_machine(expr)?;
  // An exact quantity whose parts overflow a machine number (a solution
  // formula with `Sqrt[10^400]` inside) is evaluated at higher precision.
  if matches!(value, Expr::Real(v) if !v.is_finite())
    && !matches!(expr, Expr::Real(_))
    && let Some(beyond) = n_eval_beyond_machine_range(expr)
  {
    return Ok(beyond);
  }
  Ok(value)
}

/// `N` of a `SparseArray[Automatic, dims, background, {1, {rowptr, cols},
/// values}]` reaches only its background and values: the dimensions and
/// the index structure stay integers.
fn n_sparse_array(
  name: &str,
  args: &[Expr],
  n: impl Fn(&Expr) -> Result<Expr, InterpreterError>,
) -> Option<Result<Expr, InterpreterError>> {
  if name != "SparseArray" || args.len() != 4 {
    return None;
  }
  let Expr::List(parts) = &args[3] else {
    return None;
  };
  if parts.len() != 3 {
    return None;
  }
  let build = || -> Result<Expr, InterpreterError> {
    let mut new_parts = parts.to_vec();
    new_parts[2] = n(&parts[2])?;
    Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: vec![
        args[0].clone(),
        args[1].clone(),
        n(&args[2])?,
        Expr::List(new_parts.into()),
      ]
      .into(),
    })
  };
  Some(build())
}

/// `expr` with every numeric subexpression free of slots and symbols
/// numericized and everything else left unevaluated.
fn n_held_numbers(expr: &Expr) -> Expr {
  let numeric = matches!(expr, Expr::Integer(_) | Expr::BigInteger(_))
    || matches!(expr, Expr::FunctionCall { name, .. } if name == "Rational");
  if numeric && let Ok(v) = n_eval(expr) {
    return v;
  }
  crate::syntax::map_children(expr, &n_held_numbers)
}

fn n_eval_machine(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(Expr::Real(*n as f64)),
    Expr::Real(_) => Ok(expr.clone()),
    // N[BigFloat] without an explicit precision collapses to a
    // machine-precision Real, matching wolframscript:
    // `N[1.01234567890123456789]` → `1.0123456789012346`.
    Expr::BigFloat(digits, _) => {
      if let Ok(f) = digits.parse::<f64>() {
        Ok(Expr::Real(f))
      } else {
        Ok(expr.clone())
      }
    }
    Expr::List(items) => {
      let results: Result<Vec<Expr>, _> = items.iter().map(n_eval).collect();
      Ok(Expr::List(results?.into()))
    }
    // A pure function's body is held, so N reaches only its numbers:
    // `N[FoldList[Plus, 0, #] &]` is `FoldList[Plus, 0., #] &`.
    Expr::Function { body } => Ok(Expr::Function {
      body: Box::new(n_held_numbers(body)),
    }),
    Expr::BigInteger(n) => Ok(Expr::Real(
      n.to_string().parse::<f64>().unwrap_or(f64::INFINITY),
    )),
    Expr::FunctionCall { name, args } => {
      if let Some(sparse) = n_sparse_array(name, args, n_eval) {
        return sparse;
      }
      if let Some(value) = user_n_value(expr, None)? {
        return n_eval(&value);
      }
      // First try to evaluate the whole expression to a number
      if let Some(v) = try_eval_to_f64(expr).filter(|v| !v.is_nan()) {
        return Ok(Expr::Real(v));
      }
      // N[Integrate[expr, {var, a, b}]] — when symbolic Integrate
      // didn't simplify (e.g. Abs[Sin[phi]]), fall back to NIntegrate.
      // Matches wolframscript's behaviour of computing the value
      // numerically rather than returning the unevaluated form.
      if name == "Integrate"
        && args.len() == 2
        && let Expr::List(spec) = &args[1]
        && spec.len() == 3
        && matches!(&spec[0], Expr::Identifier(_))
        && let Ok(r) = crate::functions::calculus_ast::nintegrate_ast(args)
        && matches!(&r, Expr::Real(_) | Expr::Integer(_))
      {
        return Ok(r);
      }
      // Special case for functions that stay symbolic when called
      // directly with a Real but have a numeric value triggered by N[].
      if args.len() == 1
        && let Some(n) = expr_to_i128(&args[0])
      {
        if name == "AiryAiZero"
          && let Some(r) = airy_ai_zero_n_eval(n)
        {
          return Ok(r);
        }
        if name == "AiryBiZero"
          && let Some(r) = airy_bi_zero_n_eval(n)
        {
          return Ok(r);
        }
        if name == "ZetaZero"
          && let Some(r) = zeta_zero_n_eval(n)
        {
          return Ok(r);
        }
      }
      if name == "ZetaZero"
        && args.len() == 2
        && let Some(k) = expr_to_i128(&args[0])
        && let Some(t) = super::try_eval_to_f64(&args[1])
        && let Some(r) = zeta_zero_after_n_eval(k, t)
      {
        return Ok(r);
      }
      // RootSum[poly &, fn &] — apply fn to each (complex) root of poly
      // and sum, returning a machine-precision Real (or Complex when the
      // imaginary part doesn't cancel).
      if name == "RootSum"
        && args.len() == 2
        && let Some(r) = root_sum_n_eval(&args[0], &args[1])
      {
        return Ok(r);
      }
      // `Derivative[n][f][x]` (stored flat as `Derivative[n, f, x]`): the
      // order and the function are held, only the point is numericized.
      if name == "Derivative"
        && args.len() == 3
        && matches!(&args[1], Expr::Identifier(_))
      {
        let point = n_eval(&args[2])?;
        return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: name.clone(),
          args: vec![args[0].clone(), args[1].clone(), point].into(),
        });
      }
      // AlgebraicNumber[θ, {c0, c1, …}] — c0 + c1 θ + ….
      if let Some(v) =
        crate::functions::polynomial_ast::algebraic_number_value(expr)
      {
        return n_eval(&v);
      }
      // Root[{f, x0}] — the transcendental root near x0.
      if name == "Root"
        && let [Expr::List(spec)] = args.as_slice()
        && let [f, x0] = spec.as_slice()
        && let Some(r) =
          crate::functions::polynomial_ast::transcendental_root_value(f, x0)
      {
        return Ok(r);
      }
      // Root[poly &, k] / Root[poly &, k, 0] — the k-th root numerically.
      if name == "Root"
        && (args.len() == 2 || args.len() == 3)
        && let Some(r) = root_n_eval(&args[0], &args[1])
      {
        return Ok(r);
      }
      // Honour NHoldAll / NHoldFirst / NHoldRest attributes (built-in or
      // user-set). When a slot is held, leave the argument literal and
      // skip the recursive N application.
      let attrs = {
        let builtin = crate::evaluator::get_builtin_attributes(name);
        let user = crate::FUNC_ATTRS
          .with(|m| m.borrow().get(name).copied().unwrap_or_default());
        let mut combined = builtin;
        combined.add(user.to_u32());
        combined
      };
      let hold_all = attrs.contains(Attributes::NHoldAll);
      let hold_first = attrs.contains(Attributes::NHoldFirst);
      let hold_rest = attrs.contains(Attributes::NHoldRest);
      // Every argument held: nothing changes, and evaluating the call again
      // would only repeat the messages it already issued (`N[ZetaZero[0]]`).
      if hold_all
        || args.is_empty()
        || (hold_first && (args.len() == 1 || hold_rest))
      {
        return Ok(expr.clone());
      }
      let new_args: Vec<Expr> = if hold_all {
        args.to_vec()
      } else {
        let mut out = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
          let held = (i == 0 && hold_first) || (i > 0 && hold_rest);
          if held {
            out.push(a.clone());
          } else {
            out.push(n_eval(a)?);
          }
        }
        out
      };
      // Re-evaluate the function with numeric arguments
      let new_expr = Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      };
      match crate::evaluator::evaluate_expr_to_expr(&new_expr) {
        Ok(result) => Ok(result),
        Err(_) => Ok(new_expr),
      }
    }
    Expr::BinaryOp { op, left, right } => {
      if let Some(v) = try_eval_to_f64(expr).filter(|v| !v.is_nan()) {
        return Ok(Expr::Real(v));
      }
      let new_right = if keeps_exact_exponent(*op, right) {
        (**right).clone()
      } else {
        n_eval(right)?
      };
      let new_expr = binop(*op, n_eval(left)?, new_right);
      // Re-evaluate to allow numeric simplification (e.g. complex powers)
      let result = match crate::evaluator::evaluate_expr_to_expr(&new_expr) {
        Ok(result) => result,
        Err(_) => new_expr,
      };
      // If the result is still a Power with complex operands, force
      // numeric evaluation via z^w = exp(w * log(z))
      if let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: ref base,
        right: ref exp,
      } = result
        && let (Some((a, b)), Some((c, d))) = (
          try_extract_complex_float(base),
          try_extract_complex_float(exp),
        )
        && (b != 0.0 || d != 0.0)
      {
        let abs_z = (a * a + b * b).sqrt();
        if abs_z > 0.0 {
          let ln_abs = abs_z.ln();
          let arg_z = b.atan2(a);
          let re_exp = c * ln_abs - d * arg_z;
          let im_exp = d * ln_abs + c * arg_z;
          let mag = re_exp.exp();
          let re = mag * im_exp.cos();
          let im = mag * im_exp.sin();
          let re = if re.abs() < 1e-15 { 0.0 } else { re };
          let im = if im.abs() < 1e-15 { 0.0 } else { im };
          if im == 0.0 {
            // Preserve complex form: re + 0.*I (matching Wolfram's convention
            // for complex power results where imaginary part is numerically zero)
            return Ok(plus2(
              Expr::Real(re),
              times2(Expr::Real(0.0), id_expr("I")),
            ));
          }
          return Ok(build_complex_float_expr(re, im));
        }
      }
      Ok(result)
    }
    // Rule: keep the pattern, apply N to the replacement
    Expr::Rule {
      pattern,
      replacement,
    } => Ok(Expr::Rule {
      pattern: pattern.clone(),
      replacement: Box::new(n_eval(replacement)?),
    }),
    // Comparison (`==`, `<`, …): N maps over the operands, so
    // `N[x == 1/3]` → `x == 0.333333`. Solve/NSolve rely on this to see
    // the inexact coefficients of `N[eqn]`.
    Expr::Comparison {
      operands,
      operators,
    } => {
      let new_operands: Result<Vec<Expr>, _> =
        operands.iter().map(n_eval).collect();
      let new_expr = Expr::Comparison {
        operands: new_operands?,
        operators: operators.clone(),
      };
      Ok(match crate::evaluator::evaluate_expr_to_expr(&new_expr) {
        Ok(result) => result,
        Err(_) => new_expr,
      })
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      if let Some(v) = try_eval_to_f64(expr).filter(|v| !v.is_nan()) {
        return Ok(Expr::Real(v));
      }
      crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Real(-1.0), n_eval(operand)?],
      )
    }
    Expr::Identifier(name) | Expr::Constant(name) if name == "I" => {
      crate::evaluator::evaluate_function_call_ast(
        "Complex",
        &[Expr::Real(0.0), Expr::Real(1.0)],
      )
    }
    Expr::Identifier(_) | Expr::Constant(_) => {
      if let Some(v) = try_eval_to_f64(expr) {
        return Ok(Expr::Real(v));
      }
      if let Some(value) = user_n_value(expr, None)? {
        return n_eval(&value);
      }
      // No built-in numeric value for this symbol — try to invoke
      // `N[expr]` so any user-installed `N[a] = 10.9` style DownValue
      // gets a chance to fire (matching Wolfram's NValues lookup).
      // Skip when no user `N` DownValues exist: the built-in dispatch for
      // `N` is `n_ast`, which would call back into `n_eval` here — an
      // infinite loop that exhausts WASM's smaller stack before the
      // RECURSION_LIMIT guard trips.
      let has_user_n_rules = crate::FUNC_DEFS
        .with(|m| m.borrow().get("N").is_some_and(|v| !v.is_empty()));
      if !has_user_n_rules {
        return Ok(expr.clone());
      }
      let original_str = expr_to_string(expr);
      let n_call_str = format!("N[{original_str}]");
      match crate::evaluator::evaluate_function_call_ast(
        "N",
        std::slice::from_ref(expr),
      ) {
        Ok(result) => {
          let result_str = expr_to_string(&result);
          if result_str == n_call_str {
            Ok(expr.clone())
          } else {
            Ok(result)
          }
        }
        Err(_) => Ok(expr.clone()),
      }
    }
    _ => {
      if let Some(v) = try_eval_to_f64(expr) {
        Ok(Expr::Real(v))
      } else {
        Ok(expr.clone())
      }
    }
  }
}

/// Convert decimal digit precision to the nominal bit-precision.
/// `astro-float` internally rounds up to 64-bit word boundaries.
/// Minimum 128 bits (2 words) to avoid precision issues with small values.
pub fn nominal_bits(precision: usize) -> usize {
  // Compute decimal precision plus ~36-bit guard, padded to the next
  // 64-bit word boundary. Mirrors Wolfram's bit budget so output digit
  // counts line up:
  //   p ≤ 8        →  64 bits  → 19 digits
  //   9 ≤ p ≤ 27   → 128 bits  → 38 digits
  //   28 ≤ p ≤ 47  → 192 bits  → 58 digits
  //   48 ≤ p ≤ 66  → 256 bits  → 77 digits
  // We keep a 128-bit minimum because astro-float's 64-bit operations
  // are unstable for some inputs (e.g. division produces convert_to_radix
  // errors).
  let base_bits =
    (precision as f64 * std::f64::consts::LOG2_10).ceil() as usize + 36;
  let bits = (base_bits + 63) & !63;
  bits.max(128)
}

/// Number of decimal digits after the point that wolframscript shows for an
/// arbitrary-precision number `bf` of precision `precision`.
///
/// WL keeps `ceil(p·log2 10) + 33` bits below the leading bit, in 64-bit
/// limbs aligned to the binary point, and displays every decimal digit the
/// fraction limbs resolve: `floor((64k + 1)·log10 2)` digits for `k` limbs
/// (19, 38, 58, 77, …). The digit count therefore depends on the magnitude
/// as well as on the precision — `N[4/3, 9]` shows 19 fraction digits,
/// `N[1/7, 9]` 38 — and turns negative when even the units limb lies below
/// the precision: `N[10^40/7, 3]` rounds away 19 integer digits.
pub fn wl_fraction_digits(bf: &astro_float::BigFloat, precision: f64) -> i64 {
  let fraction_limbs = wl_fraction_limbs(bf, precision);
  ((64 * fraction_limbs + 1) as f64 * std::f64::consts::LOG10_2).floor() as i64
}

/// The number of 64-bit limbs below the binary point WL keeps for `bf` at
/// `precision` (see [`wl_fraction_digits`]).
fn wl_fraction_limbs(bf: &astro_float::BigFloat, precision: f64) -> i64 {
  // `bf` is `0.1… × 2^e` (leading mantissa bit set), so it has `e` bits
  // above the binary point.
  let int_bits = bf.exponent().map_or(0, |e| e as i64);
  let kept_bits =
    (precision.max(0.0) * std::f64::consts::LOG2_10).ceil() as i64 + 33;
  (kept_bits - int_bits).div_euclid(64)
    + i64::from((kept_bits - int_bits).rem_euclid(64) != 0)
}

/// Extract an integer value from an Expr, if it is one.
fn try_as_integer(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      if let Expr::Integer(n) = operand.as_ref() {
        Some(-n)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Compute base^n for integer n, handling negative exponents.
/// Uses BigFloat::powi for the absolute value, then inverts if needed.
pub(crate) fn bigfloat_powi(
  base: &astro_float::BigFloat,
  n: i128,
  bits: usize,
  rm: astro_float::RoundingMode,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if n == 0 {
    return BigFloat::from_i32(1, bits);
  }

  let abs_n = n.unsigned_abs() as usize;
  let result = base.powi(abs_n, bits, rm);

  if n < 0 {
    BigFloat::from_i32(1, bits).div(&result, bits, rm)
  } else {
    result
  }
}

/// Largest decimal exponent an overflowed machine real is promoted to.
pub const MAX_PROMOTED_DECIMAL_EXPONENT: f64 = 10_000.0;

/// `a^b` for a positive machine real `a` whose machine result overflowed:
/// WL promotes it to an arbitrary-precision number whose precision is the
/// machine precision less the digits `b·ln a` amplifies the input error by.
pub fn machine_power_out_of_range(
  a: f64,
  b: f64,
) -> Option<Result<Expr, InterpreterError>> {
  use astro_float::{BigFloat, RoundingMode};
  if !(a > 0.0 && a.is_finite() && b.is_finite()) {
    return None;
  }
  // The promoted number stores every decimal digit, so a result with an
  // absurd magnitude stays a machine overflow (`Infinity`).
  if (b * a.log10()).abs() > MAX_PROMOTED_DECIMAL_EXPONENT {
    return None;
  }
  let rm = RoundingMode::ToEven;
  let result_prec =
    MACHINE_PRECISION_DIGITS - (b * a.ln()).abs().max(1.0).log10();
  let bits = nominal_bits(MACHINE_PRECISION_DIGITS.ceil() as usize);
  let computed = with_astro_consts(|cc| {
    let base = BigFloat::from_f64(a, bits);
    let exponent = BigFloat::from_f64(b, bits);
    let result = base.pow(&exponent, bits, rm, cc);
    let digits = wl_fraction_digits(&result, result_prec);
    bigfloat_to_string(&result, Some(digits), rm, cc)
      .map(|s| Expr::BigFloat(s, result_prec))
  });
  Some(computed.and_then(|r| r))
}

/// Run `f` with this thread's astro-float constants cache, built once:
/// building it per call cost far more than the arithmetic it serves (every
/// promoted machine power of a plot's samples built one).
pub(crate) fn with_astro_consts<R>(
  f: impl FnOnce(&mut astro_float::Consts) -> R,
) -> Result<R, InterpreterError> {
  thread_local! {
    static CONSTS: std::cell::RefCell<Option<astro_float::Consts>> =
      const { std::cell::RefCell::new(None) };
  }
  CONSTS.with(|slot| {
    let mut slot = slot.borrow_mut();
    if slot.is_none() {
      *slot = Some(
        astro_float::Consts::new()
          .map_err(|e| InterpreterError::EvaluationError(format!("{e}")))?,
      );
    }
    Ok(f(slot.as_mut().unwrap()))
  })
}

/// `Exp[x]` for a precision-tagged BigFloat `x`. Computes e^x in astro-float
/// and tracks precision as `input_prec - log10(|x|)` (matching wolframscript:
/// Exp[N[1,30]] keeps precision 30, Exp[N[10,30]] drops to 29). Returns None
/// when `x` is zero (handled elsewhere) so the caller can defer.
pub fn bigfloat_exp(
  digits: &str,
  prec: f64,
) -> Option<Result<Expr, InterpreterError>> {
  use astro_float::{BigFloat, Consts, Radix, RoundingMode};
  let x_val: f64 = digits.parse().ok()?;
  if x_val == 0.0 {
    return None;
  }
  // Past the representable decimal exponent the result is a machine
  // overflow or underflow, as for a machine real.
  if x_val.abs() * std::f64::consts::LOG10_E > MAX_PROMOTED_DECIMAL_EXPONENT {
    return Some(Ok(if x_val > 0.0 {
      Expr::Identifier("Infinity".to_string())
    } else {
      Expr::Real(0.0)
    }));
  }
  let rm = RoundingMode::ToEven;
  let bits = nominal_bits(prec.ceil() as usize);
  let mut cc = match Consts::new() {
    Ok(c) => c,
    Err(e) => {
      return Some(Err(InterpreterError::EvaluationError(format!("{e}"))));
    }
  };
  let x = BigFloat::parse(digits, Radix::Dec, bits, rm, &mut cc);
  let result = x.exp(bits, rm, &mut cc);
  let result_prec = prec - x_val.abs().log10();
  let digits = wl_fraction_digits(&result, result_prec);
  match bigfloat_to_string(&result, Some(digits), rm, &mut cc) {
    Ok(s) => Some(Ok(Expr::BigFloat(s, result_prec))),
    Err(e) => Some(Err(e)),
  }
}

/// `Log[x]` (natural log) for a precision-tagged BigFloat `x > 0`. Computes
/// ln(x) in astro-float and tracks precision as `input_prec + log10(|ln x|)`
/// (matching wolframscript: Log[N[2,30]] has precision 30 + log10(ln 2)).
/// Returns None for non-positive `x` (complex/zero result — deferred) or when
/// `x == 1` (ln = 0, special-cased by the caller).
pub fn bigfloat_log(
  digits: &str,
  prec: f64,
) -> Option<Result<Expr, InterpreterError>> {
  use astro_float::{BigFloat, Consts, Radix, RoundingMode};
  let x_val: f64 = digits.parse().ok()?;
  // Log[-x] = Log[x] + I Pi, both parts known to the same accuracy.
  if let Some(abs_digits) = digits.strip_prefix('-')
    && x_val < 0.0
  {
    let im_prec = prec + std::f64::consts::PI.log10();
    let rm = RoundingMode::ToEven;
    let bits = nominal_bits(im_prec.ceil() as usize);
    let mut cc = Consts::new().ok()?;
    let pi = cc.pi(bits, rm);
    let pi_digits = wl_fraction_digits(&pi, im_prec);
    let im_term = match bigfloat_to_string(&pi, Some(pi_digits), rm, &mut cc) {
      Ok(s) => times(vec![Expr::BigFloat(s, im_prec), id_expr("I")]),
      Err(e) => return Some(Err(e)),
    };
    if x_val == -1.0 {
      return Some(Ok(im_term));
    }
    return Some(
      bigfloat_log(abs_digits, prec)?.map(|re| plus(vec![re, im_term])),
    );
  }
  if x_val <= 0.0 || x_val == 1.0 {
    return None;
  }
  let rm = RoundingMode::ToEven;
  let bits = nominal_bits(prec.ceil() as usize);
  let mut cc = match Consts::new() {
    Ok(c) => c,
    Err(e) => {
      return Some(Err(InterpreterError::EvaluationError(format!("{e}"))));
    }
  };
  let x = BigFloat::parse(digits, Radix::Dec, bits, rm, &mut cc);
  let result = x.ln(bits, rm, &mut cc);
  // ln x itself, not ln of x read as a machine number: x may lie past the
  // machine range (1.*^308 * 10. is 1.*^309), where that reads as Infinity.
  let ln_val: f64 = bigfloat_to_string(&result, Some(17), rm, &mut cc)
    .ok()
    .and_then(|s| s.replace("*^", "e").parse().ok())
    .unwrap_or_else(|| x_val.ln());
  if !ln_val.is_finite() {
    return None;
  }
  let result_prec = prec + ln_val.abs().log10();
  let digits = wl_fraction_digits(&result, result_prec);
  match bigfloat_to_string(&result, Some(digits), rm, &mut cc) {
    Ok(s) => Some(Ok(Expr::BigFloat(s, result_prec))),
    Err(e) => Some(Err(e)),
  }
}

/// Arbitrary-precision `PolyGamma[n, z]` for an integer order `n ≥ 0` and a
/// real `z` that is not a pole, accurate to about `bits` bits.
///
/// Shifts `z` up to `x = z + m ≥ N` with `ψ⁽ⁿ⁾(z) = ψ⁽ⁿ⁾(z + m) +
/// (-1)ⁿ⁺¹ n! Σ 1/(z + k)ⁿ⁺¹` and sums the asymptotic series there:
/// `ψ(x) ~ ln x − 1/(2x) − Σ B₂ₖ/(2k x²ᵏ)` and, for `n ≥ 1`,
/// `ψ⁽ⁿ⁾(x) ~ (-1)ⁿ⁺¹ [(n−1)!/xⁿ + n!/(2xⁿ⁺¹) + Σ B₂ₖ (2k+n−1)!/((2k)! x²ᵏ⁺ⁿ)]`.
/// Returns `None` at a pole and for `z` so negative that the shift would
/// take too long.
pub fn polygamma_bigfloat(
  n: usize,
  z: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Option<astro_float::BigFloat> {
  use astro_float::BigFloat;
  let z_f64 = bigfloat_to_f64_approx(z)?;
  if !z_f64.is_finite() {
    return None;
  }
  // Far left of the origin the upward shift would take too many steps;
  // reflect ψ(z) = ψ(1 − z) − π cot(πz) instead (higher orders would need
  // the n-th derivative of the cotangent).
  if z_f64 < -1.0e3 {
    if n != 0 || z.fract().is_zero() {
      return None;
    }
    let wp = bits + 64;
    let one = BigFloat::from_i32(1, wp);
    let pi_z = cc.pi(wp, rm).mul(z, wp, rm);
    let cot = pi_z.cos(wp, rm, cc).div(&pi_z.sin(wp, rm, cc), wp, rm);
    let reflected = polygamma_bigfloat(0, &one.sub(z, wp, rm), wp, rm, cc)?;
    let mut r = reflected.sub(&cc.pi(wp, rm).mul(&cot, wp, rm), wp, rm);
    r.set_precision(bits, rm).ok()?;
    return Some(r);
  }
  // Guard bits absorb the cancellation between the shift sum and the
  // asymptotic value (ψ's roots, the large terms next to a pole).
  let wp = bits + 64;
  let one = BigFloat::from_i32(1, wp);
  let shift_target = asymptotic_shift(wp) + n;

  // A non-positive integer is a pole.
  if z_f64 <= 0.0 {
    let frac = z.fract();
    if frac.is_zero() {
      return None;
    }
  }

  let int = |k: usize| BigFloat::from_u64(k as u64, wp);
  let factorial =
    |k: usize| (2..=k).fold(one.clone(), |f, j| f.mul(&int(j), wp, rm));
  let half = BigFloat::from_f64(0.5, wp);

  // Σ 1/(z+k)ⁿ⁺¹ kept as one fraction, so the loop only multiplies.
  let mut x = z.clone();
  let mut shift_num = BigFloat::from_i32(0, wp);
  let mut shift_den = one.clone();
  let target = int(shift_target);
  while x.cmp(&target).is_some_and(|o| o < 0) {
    let p = bigfloat_powi(&x, (n + 1) as i128, wp, rm);
    shift_num = shift_num.mul(&p, wp, rm).add(&shift_den, wp, rm);
    shift_den = shift_den.mul(&p, wp, rm);
    x = x.add(&one, wp, rm);
  }
  let shift_sum = shift_num.div(&shift_den, wp, rm);

  let inv_x = one.div(&x, wp, rm);
  let inv_x2 = inv_x.mul(&inv_x, wp, rm);
  // Leading terms and the power x^-(2k+n) of the first Bernoulli term.
  let (mut asym, mut x_pow) = if n == 0 {
    let lead =
      ln_above_integer(&x, wp, rm, cc).sub(&inv_x.mul(&half, wp, rm), wp, rm);
    (lead, inv_x2.clone())
  } else {
    let x_n = bigfloat_powi(&inv_x, n as i128, wp, rm);
    let lead = factorial(n - 1).mul(&x_n, wp, rm).add(
      &factorial(n)
        .mul(&x_n, wp, rm)
        .mul(&inv_x, wp, rm)
        .mul(&half, wp, rm),
      wp,
      rm,
    );
    (lead, x_n.mul(&inv_x2, wp, rm))
  };
  let eps_exp = asym.exponent().unwrap_or(0) as i64 - wp as i64;
  let mut prev_mag = i64::MAX;
  for k in 1..=(4 * wp) {
    // Coefficient B₂ₖ/(2k) for ψ, B₂ₖ (2k+n−1)!/(2k)! = B₂ₖ/(2k) ·
    // (2k)(2k+1)…(2k+n−1) otherwise.
    let coeff = (0..n).fold(bernoulli_over_index(k, wp, rm, cc), |c, j| {
      c.mul(&int(2 * k + j), wp, rm)
    });
    let term = coeff.mul(&x_pow, wp, rm);
    let mag = term.exponent().map_or(i64::MIN, |e| e as i64);
    // The series is asymptotic: stop once the terms are negligible, or
    // once they start growing again.
    if mag < eps_exp || mag > prev_mag {
      break;
    }
    prev_mag = mag;
    asym = if n == 0 {
      asym.sub(&term, wp, rm)
    } else {
      asym.add(&term, wp, rm)
    };
    x_pow = x_pow.mul(&inv_x2, wp, rm);
  }

  // ψ⁽ⁿ⁾(z) = (-1)ⁿ⁺¹ [inner(x) + n! Σ 1/(z+k)ⁿ⁺¹], with inner(x) = -ψ(x)
  // for n = 0.
  let result = if n == 0 {
    asym.sub(&shift_sum, wp, rm)
  } else {
    let total = asym.add(&factorial(n).mul(&shift_sum, wp, rm), wp, rm);
    if n.is_multiple_of(2) {
      total.neg()
    } else {
      total
    }
  };
  let mut result = result;
  result.set_precision(bits, rm).ok()?;
  Some(result)
}

/// How far up to shift the argument of an asymptotic (Bernoulli) series so
/// that it reaches `wp` bits: its smallest term is about `e^(-2πx)`, and the
/// shifted argument must stay ≥ 16 for [`ln_above_integer`].
fn asymptotic_shift(wp: usize) -> usize {
  ((wp as f64 * std::f64::consts::LN_2 / std::f64::consts::TAU).ceil() as usize
    + 4)
    .max(16)
}

/// Arbitrary-precision `Zeta[s, a]` (`signed = false`) or
/// `HurwitzZeta[s, a]` (`signed = true`) for real `s ≠ 1`, accurate to about
/// `bits` bits. The two differ only in the terms with `k + a < 0`:
/// `Zeta` sums `((k + a)²)^(-s/2)`, `HurwitzZeta` `(k + a)^(-s)` (complex
/// unless `s` is an integer, which gives `None`). A term with `k + a = 0` is
/// left out. `Zeta[s] = Zeta[s, 1]`.
///
/// Euler–Maclaurin at `x = a + N`: `Σₖ₌₀ᴺ⁻¹ (a + k)^(-s) + x^(1−s)/(s − 1) +
/// x^(-s)/2 + Σⱼ B₂ⱼ/(2j)! · s(s+1)…(s+2j−2) · x^(1−s−2j)`.
pub fn hurwitz_zeta_bigfloat(
  s: &astro_float::BigFloat,
  a: &astro_float::BigFloat,
  signed: bool,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Option<astro_float::BigFloat> {
  use astro_float::BigFloat;
  let s_f64 = bigfloat_to_f64_approx(s)?;
  let a_f64 = bigfloat_to_f64_approx(a)?;
  if !s_f64.is_finite() || !a_f64.is_finite() || a_f64 < -1.0e5 {
    return None;
  }
  let wp = bits + 64;
  let one = BigFloat::from_i32(1, wp);
  if s.sub(&one, wp, rm).is_zero() {
    return None;
  }
  let s_int = s.fract().is_zero().then_some(s_f64 as i128);
  let neg_s = s.neg();
  // x^(-s), exact repeated multiplication for an integer `s`.
  let pow_neg_s = |x: &BigFloat, cc: &mut astro_float::Consts| match s_int {
    Some(n) => bigfloat_powi(x, -n, wp, rm),
    None if x.is_positive() => neg_s
      .mul(&ln_above_integer(x, wp, rm, cc), wp, rm)
      .exp(wp, rm, cc),
    None => x.pow(&neg_s, wp, rm, cc),
  };

  let target = BigFloat::from_u64(
    (asymptotic_shift(wp) as f64 + s_f64.abs()).ceil() as u64,
    wp,
  );
  let mut sum = BigFloat::from_i32(0, wp);
  let mut x = a.clone();
  while x.cmp(&target).is_some_and(|o| o < 0) {
    if !x.is_zero() {
      let term = match (x.is_negative(), signed, s_int) {
        (false, _, _) => pow_neg_s(&x, cc),
        (true, false, _) => pow_neg_s(&x.abs(), cc),
        (true, true, Some(_)) => pow_neg_s(&x, cc),
        (true, true, None) => return None,
      };
      sum = sum.add(&term, wp, rm);
    }
    x = x.add(&one, wp, rm);
  }

  let x_neg_s = pow_neg_s(&x, cc);
  let s_minus_1 = s.sub(&one, wp, rm);
  sum = sum
    .add(&x_neg_s.mul(&x, wp, rm).div(&s_minus_1, wp, rm), wp, rm)
    .add(&x_neg_s.mul(&BigFloat::from_f64(0.5, wp), wp, rm), wp, rm);

  let inv_x = one.div(&x, wp, rm);
  let inv_x2 = inv_x.mul(&inv_x, wp, rm);
  // T_j = s(s+1)…(s+2j−2)/(2j−1)! · x^(1−s−2j); T_1 = s x^(-s-1).
  let mut t = s.mul(&x_neg_s, wp, rm).mul(&inv_x, wp, rm);
  let eps_exp = sum.exponent().unwrap_or(0) as i64 - wp as i64;
  let mut prev_mag = i64::MAX;
  for j in 1..=(4 * wp) {
    if t.is_zero() {
      break;
    }
    // B₂ⱼ/(2j)! · (s)₂ⱼ₋₁ x^(1−s−2j) = B₂ⱼ/(2j) · T_j.
    let term = bernoulli_over_index(j, wp, rm, cc).mul(&t, wp, rm);
    let mag = term.exponent().map_or(i64::MIN, |e| e as i64);
    if mag < eps_exp || mag > prev_mag {
      break;
    }
    prev_mag = mag;
    sum = sum.add(&term, wp, rm);
    let twoj = (2 * j) as i64;
    let rise = s.add(&BigFloat::from_i64(twoj - 1, wp), wp, rm).mul(
      &s.add(&BigFloat::from_i64(twoj, wp), wp, rm),
      wp,
      rm,
    );
    t = t.mul(&rise, wp, rm).mul(&inv_x2, wp, rm).div(
      &BigFloat::from_i64(twoj * (twoj + 1), wp),
      wp,
      rm,
    );
  }
  sum.set_precision(bits, rm).ok()?;
  Some(sum)
}

/// `HarmonicNumber[x, r]` for real `x > -1` and real `r`, to `bits` bits:
/// `ψ(x + 1) + γ` at `r = 1`, else `ζ(r) − ζ(r, x + 1)`, with guard bits
/// grown past the cancellation of the difference.
pub(crate) fn harmonic_number_bigfloat(
  x: &astro_float::BigFloat,
  r: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Option<astro_float::BigFloat> {
  use astro_float::BigFloat;
  let exp = |b: &BigFloat| {
    if b.is_zero() {
      i64::MIN / 4
    } else {
      i64::from(b.exponent().unwrap_or(0))
    }
  };
  if x.cmp(&BigFloat::from_i32(-1, bits))? <= 0 {
    return None;
  }
  let mut guard = 64;
  loop {
    let wp = bits + guard;
    let one = BigFloat::from_i32(1, wp);
    let x1 = x.add(&one, wp, rm);
    let (a, b) = if r.sub(&one, wp, rm).is_zero() {
      (
        polygamma_bigfloat(0, &x1, wp, rm, cc)?,
        polygamma_bigfloat(0, &one, wp, rm, cc)?,
      )
    } else {
      (
        hurwitz_zeta_bigfloat(r, &one, false, wp, rm, cc)?,
        hurwitz_zeta_bigfloat(r, &x1, false, wp, rm, cc)?,
      )
    };
    let diff = a.sub(&b, wp, rm);
    let lost = (exp(&a).max(exp(&b)) - exp(&diff)).max(0) as usize;
    if lost + 16 <= guard || guard > 16 * bits + 4096 {
      return Some(diff.round(bits, rm));
    }
    guard = lost + 64;
  }
}

/// `expr` (a function call with real arguments) as a machine number,
/// computed at 128 bits through the arbitrary-precision path and rounded
/// once; `None` when that path has no value for it.
pub(crate) fn machine_function_value(expr: &Expr) -> Option<f64> {
  let rm = astro_float::RoundingMode::ToEven;
  let v = with_thread_consts(|cc| expr_to_bigfloat(expr, 128, rm, cc).ok())?;
  bigfloat_to_f64_approx(&v).filter(|v| v.is_finite())
}

/// The order of `PolyGamma[n, z]` as a non-negative integer — also from an
/// integral `Real`/`BigFloat`, which is what `N` turns the order into.
pub fn polygamma_order(n: &Expr) -> Option<usize> {
  match n {
    Expr::Integer(k) if *k >= 0 => usize::try_from(*k).ok(),
    Expr::Real(f) if *f >= 0.0 && f.fract() == 0.0 && *f < 1.0e6 => {
      Some(*f as usize)
    }
    Expr::BigFloat(digits, _) => {
      let f: f64 = digits.parse().ok()?;
      (f >= 0.0 && f.fract() == 0.0 && f < 1.0e6).then_some(f as usize)
    }
    _ => None,
  }
}

/// Runs `f` with this thread's cached astro-float constants, which the
/// machine-precision wrappers around the BigFloat special functions share
/// instead of rebuilding them on every call.
pub(crate) fn with_thread_consts<T>(
  f: impl FnOnce(&mut astro_float::Consts) -> Option<T>,
) -> Option<T> {
  thread_local! {
    static CONSTS: std::cell::RefCell<Option<astro_float::Consts>> =
      const { std::cell::RefCell::new(None) };
  }
  CONSTS.with(|slot| {
    let mut slot = slot.borrow_mut();
    if slot.is_none() {
      *slot = Some(astro_float::Consts::new().ok()?);
    }
    f(slot.as_mut()?)
  })
}

/// Machine `PolyGamma[n, x]`, rounded from a 64-bit result (worked out with
/// 64 guard bits) so it is accurate to the last bit (the f64 recurrence-plus-series loses up to ~40
/// ulps to cancellation). `None` at a pole.
pub fn polygamma_f64(n: usize, x: f64) -> Option<f64> {
  use astro_float::{BigFloat, RoundingMode};
  with_thread_consts(|cc| {
    let z = BigFloat::from_f64(x, 64);
    let r = polygamma_bigfloat(n, &z, 64, RoundingMode::ToEven, cc)?;
    bigfloat_to_f64_approx(&r)
  })
}

/// `ln x` for `x > 0`, as `ln N + 2 atanh(δ/(2N + δ))` with `N = ⌊x⌋ ≥ 16`
/// and `δ = x − N` (a smaller `x` is first scaled by an exact power of two):
/// `ln N` is cached and the atanh series needs only multiplications, which
/// makes it several times faster than a general `BigFloat::ln` for the
/// series arguments of PolyGamma and Zeta.
pub(crate) fn ln_above_integer(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;
  thread_local! {
    static LN_CACHE: std::cell::RefCell<std::collections::HashMap<(usize, u64), BigFloat>> =
      std::cell::RefCell::new(std::collections::HashMap::new());
    static ODD_RECIPROCALS: std::cell::RefCell<std::collections::HashMap<usize, Vec<BigFloat>>> =
      std::cell::RefCell::new(std::collections::HashMap::new());
  }
  let n = match bigfloat_to_f64_approx(x) {
    Some(v) if (16.0..1.0e6).contains(&v) => v.floor() as u64,
    // Scale by an exact power of two into [16, 32): ln x = ln(2ᵏ x) − k ln 2.
    Some(v) if v > 0.0 && v.is_finite() => {
      let k = 4 - v.log2().floor() as i64;
      let mut scaled = x.clone();
      scaled
        .set_exponent(x.exponent().unwrap_or(0) + k as astro_float::Exponent);
      if bigfloat_to_f64_approx(&scaled)
        .is_none_or(|w| !(16.0..32.0).contains(&w))
      {
        return x.ln(bits, rm, cc);
      }
      let ln2 = cc.ln_2(bits, rm);
      return ln_above_integer(&scaled, bits, rm, cc).sub(
        &ln2.mul(&BigFloat::from_i64(k, bits), bits, rm),
        bits,
        rm,
      );
    }
    _ => return x.ln(bits, rm, cc),
  };
  let n_bf = BigFloat::from_u64(n, bits);
  let ln_n = LN_CACHE.with(|cache| {
    cache
      .borrow_mut()
      .entry((bits, n))
      .or_insert_with(|| n_bf.ln(bits, rm, cc))
      .clone()
  });
  let delta = x.sub(&n_bf, bits, rm);
  if delta.is_zero() {
    return ln_n;
  }
  // u = δ/(2N + δ) < 1/33, so each term gains more than 10 bits.
  let u = delta.div(&n_bf.add(&n_bf, bits, rm).add(&delta, bits, rm), bits, rm);
  let u2 = u.mul(&u, bits, rm);
  let terms = bits / 10 + 2;
  let atanh = ODD_RECIPROCALS.with(|cache| {
    let mut cache = cache.borrow_mut();
    let recips = cache.entry(bits).or_default();
    while recips.len() < terms {
      let k = recips.len() as i32;
      let one = BigFloat::from_i32(1, bits);
      recips.push(one.div(&BigFloat::from_i32(2 * k + 1, bits), bits, rm));
    }
    // Horner: u (1 + u²/3 + u⁴/5 + …)
    let mut sum = BigFloat::from_i32(0, bits);
    for r in recips[..terms].iter().rev() {
      sum = sum.mul(&u2, bits, rm).add(r, bits, rm);
    }
    sum.mul(&u, bits, rm)
  });
  ln_n.add(&atanh.add(&atanh, bits, rm), bits, rm)
}

/// Machine `Zeta[s, a]`/`HurwitzZeta[s, a]` (see [`hurwitz_zeta_bigfloat`])
/// rounded from a 64-bit result, accurate to the last bit where the f64
/// Euler–Maclaurin sum loses digits — badly so for negative `s`, whose
/// terms grow and cancel. `None` at a pole or for a complex value.
pub fn hurwitz_zeta_f64(s: f64, a: f64, signed: bool) -> Option<f64> {
  use astro_float::{BigFloat, RoundingMode};
  with_thread_consts(|cc| {
    let (sb, ab) = (BigFloat::from_f64(s, 64), BigFloat::from_f64(a, 64));
    let r =
      hurwitz_zeta_bigfloat(&sb, &ab, signed, 64, RoundingMode::ToEven, cc)?;
    bigfloat_to_f64_approx(&r)
  })
}

/// B₂ₖ/(2k) as a BigFloat of `bits` bits, cached per precision. A value
/// cached at a higher precision is rounded down rather than rederived from
/// the exact rational, so callers stepping through many precisions (each
/// term of a series to its own accuracy) pay for each B₂ₖ only once.
fn bernoulli_over_index(
  k: usize,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;
  thread_local! {
    static CACHE: std::cell::RefCell<std::collections::HashMap<usize, Vec<BigFloat>>> =
      std::cell::RefCell::new(std::collections::HashMap::new());
  }
  CACHE.with(|cache| {
    let mut cache = cache.borrow_mut();
    if cache.get(&bits).is_none_or(|values| values.len() <= k)
      && let Some(mut v) = cache
        .iter()
        .filter(|(b, values)| **b > bits && values.len() > k)
        .min_by_key(|(b, _)| **b)
        .map(|(_, values)| values[k].clone())
      && v.set_precision(bits, rm).is_ok()
    {
      return v;
    }
    let values = cache.entry(bits).or_default();
    while values.len() <= k {
      // Index 0 is an unused placeholder, so that index k holds B₂ₖ/(2k).
      let k = values.len();
      if k == 0 {
        values.push(BigFloat::from_i32(0, bits));
        continue;
      }
      let (num, den) = crate::functions::math_ast::bernoulli_big(2 * k);
      let den = den * num_bigint::BigInt::from(2 * k);
      let parse = |v: &num_bigint::BigInt, cc: &mut astro_float::Consts| {
        BigFloat::parse(&v.to_string(), astro_float::Radix::Dec, bits, rm, cc)
      };
      let b = parse(&num, cc).div(&parse(&den, cc), bits, rm);
      values.push(b);
    }
    values[k].clone()
  })
}

/// A BigFloat's value rounded to the nearest `f64` (normal range only).
pub(crate) fn bigfloat_to_f64_approx(
  bf: &astro_float::BigFloat,
) -> Option<f64> {
  if bf.is_zero() {
    return Some(0.0);
  }
  let (words, _, sign, exp, _) = bf.as_raw_parts()?;
  // The mantissa is 0.1… in little-endian words (32 or 64 bits wide,
  // depending on the target). Collect its leading 64 bits into `top`; any set
  // bit below them only matters as a sticky bit for round-to-nearest of the
  // u64 → f64 conversion.
  let per_top = (64 / usize::BITS) as usize;
  let used = per_top.min(words.len());
  if used == 0 {
    return None;
  }
  let split = words.len() - used;
  let mut top: u64 = 0;
  for &w in words[split..].iter().rev() {
    // `w` is already a u64 on 64-bit targets but a u32 on wasm32.
    #[allow(clippy::unnecessary_cast)]
    let word = w as u64;
    top = (top << (usize::BITS % 64)) | word;
  }
  top <<= 64 - used as u32 * usize::BITS;
  let sticky = u64::from(words[..split].iter().any(|&w| w != 0));
  // Scale into [1/2, 1) first and apply 2^exp in two halves, so neither
  // factor over- or underflows for a result near the ends of f64's range.
  let half_exp = exp / 2;
  let v = (top | sticky) as f64
    * 2f64.powi(-64)
    * 2f64.powi(half_exp)
    * 2f64.powi(exp - half_exp);
  Some(if sign == astro_float::Sign::Neg {
    -v
  } else {
    v
  })
}

/// Arbitrary-precision Gamma via Spouge's approximation.
///
/// `Γ(z+1) = (z+a)^(z+1/2) · e^{−(z+a)} · (c₀ + Σ_{k=1}^{a−1} c_k/(z+k))`
///
/// with `c_0 = √(2π)` and `c_k = (-1)^(k-1)/(k-1)! · (a-k)^(k-1/2) · e^(a-k)`.
/// The relative error is below `(2π)^{−a}`, which fixes `a` from the
/// requested bits. The `c_k` alternate with magnitudes up to about `e^a`, so
/// the sum is formed with `a·log₂e` guard bits; the coefficients depend only
/// on the precision and are cached. We shift the input by 1 (so the formula
/// receives `z = input - 1`) and reflect to positive arguments via the
/// reflection identity when `Re(z) < 1/2`.
pub(crate) fn gamma_bigfloat(
  z_in: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // (2π)^(−a) = 2^(−2.65 a) must clear the requested bits.
  let a_int: i64 = (((bits + 8) as f64 / 2.65).ceil() as i64 + 2).max(8);
  let w = bits + (a_int as f64 * std::f64::consts::LOG2_E).ceil() as usize + 8;

  // Reflection: for z < 1/2 use Γ(z) = π / (sin(π·z) · Γ(1−z)).
  let half = BigFloat::from_f64(0.5, w);
  if z_in.cmp(&half).is_some_and(|o| o < 0) {
    let pi = cc.pi(w, rm);
    let one_minus_z = BigFloat::from_i32(1, w).sub(z_in, w, rm);
    let sin_piz = pi.mul(z_in, w, rm).sin(w, rm, cc);
    let gamma_1mz = gamma_bigfloat(&one_minus_z, w, rm, cc);
    return pi.div(&sin_piz.mul(&gamma_1mz, w, rm), bits, rm);
  }

  thread_local! {
    static COEFFS: std::cell::RefCell<std::collections::HashMap<usize, std::rc::Rc<Vec<BigFloat>>>> =
      std::cell::RefCell::new(std::collections::HashMap::new());
  }
  let coeffs = if let Some(c) = COEFFS.with(|c| c.borrow().get(&w).cloned()) {
    c
  } else {
    // c_0 = √(2π), c_k = (-1)^(k-1)/(k-1)! · (a-k)^(k-1/2) · e^(a-k).
    let two_pi = cc.pi(w, rm).mul(&BigFloat::from_i32(2, w), w, rm);
    let mut c = vec![two_pi.sqrt(w, rm)];
    let mut factorial = BigFloat::from_i32(1, w); // (k-1)!
    for k in 1..a_int {
      if k > 1 {
        factorial = factorial.mul(&BigFloat::from_i64(k - 1, w), w, rm);
      }
      let a_minus_k = BigFloat::from_i64(a_int - k, w);
      let half_offset = BigFloat::from_i64(k - 1, w).add(&half, w, rm);
      let c_k = a_minus_k
        .pow(&half_offset, w, rm, cc)
        .mul(&a_minus_k.exp(w, rm, cc), w, rm)
        .div(&factorial, w, rm);
      c.push(if k % 2 == 0 { c_k.neg() } else { c_k });
    }
    let c = std::rc::Rc::new(c);
    COEFFS.with(|m| m.borrow_mut().insert(w, c.clone()));
    c
  };

  // Work with z = z_in - 1 so the formula computes Γ(z+1) = Γ(z_in).
  let z = z_in.sub(&BigFloat::from_i32(1, w), w, rm);
  let z_plus_a = z.add(&BigFloat::from_i64(a_int, w), w, rm);
  // (z+a)^(z+½) e^(−(z+a)) as one exponential.
  let outer = ln_above_integer(&z_plus_a, w, rm, cc)
    .mul(&z.add(&half, w, rm), w, rm)
    .sub(&z_plus_a, w, rm)
    .exp(w, rm, cc);
  // Σ c_k/(z+k) as a single fraction num/den: software division is far
  // slower than multiplication, so divide once at the end.
  let mut num = BigFloat::from_i32(0, w);
  let mut den = BigFloat::from_i32(1, w);
  for (k, c_k) in coeffs.iter().enumerate().skip(1) {
    let z_plus_k = z.add(&BigFloat::from_i64(k as i64, w), w, rm);
    num = num.mul(&z_plus_k, w, rm).add(&c_k.mul(&den, w, rm), w, rm);
    den = den.mul(&z_plus_k, w, rm);
  }
  let sum = coeffs[0].add(&num.div(&den, w, rm), w, rm);
  outer.mul(&sum, bits, rm)
}

/// Convert a BigFloat to a decimal string.
/// If `max_fraction_digits` is Some(n), round to `n` digits AFTER the
/// decimal point (a negative `n` rounds off integer digits), so e.g. with
/// `n = 58`, Sqrt[2] yields 1 + 58 = 59 significant digits, Sin[1] 58 and
/// 1/30 (0.0333…) 57 — matching how Wolfram caps display digits relative to
/// the decimal point (see [`wl_fraction_digits`]).
pub fn bigfloat_to_string(
  bf: &astro_float::BigFloat,
  max_fraction_digits: Option<i64>,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<String, InterpreterError> {
  if bf.is_zero() {
    return Ok("0.".to_string());
  }

  let (sign, digits, exponent) = bf
    .convert_to_radix(astro_float::Radix::Dec, rm, cc)
    .map_err(|e| {
      InterpreterError::EvaluationError(format!("N: format error: {e}"))
    })?;

  if digits.is_empty() || digits.iter().all(|&d| d == 0) {
    return Ok("0.".to_string());
  }

  let is_negative = sign == astro_float::Sign::Neg;
  let prefix = if is_negative { "-" } else { "" };

  // Convert digit values to ASCII string
  let digit_str: String = digits.iter().map(|&d| (b'0' + d) as char).collect();

  // Round to `max_frac` digits after the dot if requested. The value is
  // 0.d₁d₂… × 10^exp, so that is `max_frac + exp` significant digits.
  // Rounding (not truncation) the last shown digit matches Wolfram. Carries
  // that ripple past the leading digit increment the decimal exponent.
  let (digit_str, decimal_exp) = if let Some(max_frac) = max_fraction_digits {
    let exp_i64 = exponent as i64;
    let total_keep = (max_frac + exp_i64).max(1) as usize;
    if digit_str.len() > total_keep {
      let kept = &digit_str[..total_keep];
      let next_digit = digit_str.as_bytes()[total_keep] - b'0';
      let mut bytes: Vec<u8> = kept.bytes().collect();
      let mut carry = u8::from(next_digit >= 5);
      // Round-half-up. (Wolfram uses banker's rounding internally, but the
      // BigFloat is already at higher precision than `total_keep` so the
      // next digit is rarely exactly 5 with no further nonzero digits.)
      for b in bytes.iter_mut().rev() {
        if carry == 0 {
          break;
        }
        let v = *b - b'0' + carry;
        if v >= 10 {
          *b = b'0';
          carry = 1;
        } else {
          *b = b'0' + v;
          carry = 0;
        }
      }
      let new_exp = if carry == 1 {
        // All 9s rolled over: prepend a 1, drop the trailing 0, bump exponent.
        bytes.pop();
        bytes.insert(0, b'1');
        exp_i64 + 1
      } else {
        exp_i64
      };
      let s: String = String::from_utf8(bytes).expect("ascii digits");
      (s, new_exp)
    } else {
      (digit_str, exp_i64)
    }
  } else {
    (digit_str, exponent as i64)
  };

  if decimal_exp <= 0 {
    // Number like 0.000xxxx
    let zeros = (-decimal_exp) as usize;
    let trimmed = digit_str.trim_end_matches('0');
    if trimmed.is_empty() {
      Ok(format!("{prefix}0."))
    } else {
      let frac = format!("{}{}", "0".repeat(zeros), trimmed);
      let frac = frac.trim_end_matches('0');
      if frac.is_empty() {
        Ok(format!("{prefix}0."))
      } else {
        Ok(format!("{prefix}0.{frac}"))
      }
    }
  } else {
    let dp = decimal_exp as usize;
    if dp >= digit_str.len() {
      // All digits are in the integer part
      let padded = format!("{}{}", digit_str, "0".repeat(dp - digit_str.len()));
      Ok(format!("{prefix}{padded}."))
    } else {
      // Some digits before decimal, some after
      let int_part = &digit_str[..dp];
      let frac_part = digit_str[dp..].trim_end_matches('0');
      if frac_part.is_empty() {
        Ok(format!("{prefix}{int_part}."))
      } else {
        Ok(format!("{prefix}{int_part}.{frac_part}"))
      }
    }
  }
}

/// SetPrecision[expr, precision] — set every numeric leaf of `expr` to a
/// fixed precision.
///
/// Unlike `N`, SetPrecision walks the expression tree symbolically: numeric
/// leaves are converted to BigFloat (or, with MachinePrecision, demoted to
/// a machine-precision Real), while symbolic heads and identifiers are
/// preserved verbatim. The rewritten expression is then evaluated again, as
/// wolframscript does with any result a function hands back: the numeric
/// leaves are what unlock the numeric evaluation, so `SetPrecision[Tanh[1],
/// 3]` is `Tanh[1.`3.]` and therefore `0.762`, not a literal `Tanh[1.00]`.
pub fn set_precision_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("SetPrecision", args));
  }
  let expr = &args[0];
  let target = &args[1];

  if matches!(target, Expr::Identifier(s) if s == "MachinePrecision") {
    return set_precision_machine(expr);
  }
  if matches!(target, Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity")
  {
    let exact = set_precision_exact(expr);
    if crate::evaluator::pattern_matching::expr_equal(&exact, expr) {
      return Ok(exact);
    }
    return crate::evaluator::evaluate_expr_to_expr(&exact);
  }

  let precision_f64 = match target {
    Expr::Integer(n) if *n > 0 => *n as f64,
    other => {
      if let Some(v) = try_eval_to_f64(other)
        && v > 0.0
      {
        v
      } else {
        return Ok(unevaluated("SetPrecision", args));
      }
    }
  };

  use astro_float::{Consts, RoundingMode};
  let mut cc = Consts::new().map_err(|e| {
    InterpreterError::EvaluationError(format!("BigFloat init error: {e}"))
  })?;
  let rm = RoundingMode::ToEven;
  let prec_usize = precision_f64.floor().max(1.0) as usize;
  let bits = nominal_bits(prec_usize) + 64;

  let rewritten = set_precision_walk(expr, precision_f64, bits, rm, &mut cc)?;
  // Re-evaluate only when the walk actually changed something, so an
  // expression with no numeric leaf to convert cannot loop.
  if crate::evaluator::pattern_matching::expr_equal(&rewritten, expr) {
    return Ok(rewritten);
  }
  crate::evaluator::evaluate_expr_to_expr(&rewritten)
}

/// Walk `expr` and convert every numeric leaf to a BigFloat at `precision`.
fn set_precision_walk(
  expr: &Expr,
  precision: f64,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _) => leaf_to_bigfloat(expr, precision, bits, rm, cc),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      leaf_to_bigfloat(expr, precision, bits, rm, cc)
    }
    Expr::List(items) => {
      let mut out = Vec::with_capacity(items.len());
      for item in items {
        out.push(set_precision_walk(item, precision, bits, rm, cc)?);
      }
      Ok(Expr::List(out.into()))
    }
    Expr::FunctionCall { name, args } => {
      let mut out = Vec::with_capacity(args.len());
      for a in args {
        out.push(set_precision_walk(a, precision, bits, rm, cc)?);
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: out.into(),
      })
    }
    Expr::BinaryOp { op, left, right } => {
      let l = set_precision_walk(left, precision, bits, rm, cc)?;
      let r = set_precision_walk(right, precision, bits, rm, cc)?;
      Ok(binop(*op, l, r))
    }
    Expr::UnaryOp { op, operand } => {
      let o = set_precision_walk(operand, precision, bits, rm, cc)?;
      Ok(Expr::UnaryOp {
        op: *op,
        operand: Box::new(o),
      })
    }
    // Symbolic real constants (Pi, E, Degree, GoldenRatio, EulerGamma, …)
    // numericize to the requested precision, matching N[c, p]:
    // SetPrecision[Pi, 5] -> 3.1415926535897932385`5.. expr_to_bigfloat
    // errors for a bare symbol (e.g. x), so those pass through unchanged.
    Expr::Constant(_) | Expr::Identifier(_) => {
      leaf_to_bigfloat(expr, precision, bits, rm, cc)
        .or_else(|_| Ok(expr.clone()))
    }
    // Every other compound node (comparisons, rules, …) has its children
    // set the same way.
    _ => {
      let cc = std::cell::RefCell::new(cc);
      let failure = std::cell::RefCell::new(None);
      let mapped =
        crate::syntax::map_children(expr, &|child| match set_precision_walk(
          child,
          precision,
          bits,
          rm,
          &mut cc.borrow_mut(),
        ) {
          Ok(e) => e,
          Err(e) => {
            failure.borrow_mut().get_or_insert(e);
            child.clone()
          }
        });
      match failure.into_inner() {
        Some(e) => Err(e),
        None => Ok(mapped),
      }
    }
  }
}

/// Convert a numeric leaf to a BigFloat at the requested precision.
fn leaf_to_bigfloat(
  expr: &Expr,
  precision: f64,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  wl_bigfloat_expr(expr, precision, bits, rm, cc)
}

/// `expr` as the arbitrary-precision number WL makes of it at `precision`,
/// evaluated with (at least) `bits` of mantissa.
fn wl_bigfloat_expr(
  expr: &Expr,
  precision: f64,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  let mut bf = expr_to_bigfloat(expr, bits, rm, cc)?;
  // A real function leaving its real domain (`Log` of a negative sum) is
  // NaN here; the caller then evaluates in the complex plane.
  if bf.is_nan() {
    return Err(InterpreterError::EvaluationError("N: complex value".into()));
  }
  // WL's value is the number rounded to its limb grid below the binary
  // point, and the shown digits are that binary value's. For a small
  // magnitude the grid is finer than `bits` resolve (`N[10^-8/7, 20]` keeps
  // 192 fraction bits of a number whose leading bit is 2^-30), so the value
  // is computed afresh at the grid's resolution before being rounded to it.
  let limbs = wl_fraction_limbs(&bf, precision);
  if limbs > 0 {
    let fraction_bits = 64 * limbs as usize;
    let int_bits = bf.exponent().map_or(0, |e| e as i64);
    let needed = (fraction_bits as i64 + int_bits + 64).max(0) as usize;
    if needed > bits {
      bf = expr_to_bigfloat(expr, (needed + 63) & !63, rm, cc)?;
    }
    bf = bf.round(fraction_bits, rm);
  }
  let digits = wl_fraction_digits(&bf, precision);
  let decimal = bigfloat_to_string(&bf, Some(digits), rm, cc)?;
  Ok(Expr::BigFloat(decimal, precision))
}

/// `expr`, a function call with real numeric arguments, as an
/// arbitrary-precision number at `precision` digits — or `None` when the
/// arbitrary-precision path has no value for it.
pub(crate) fn bigfloat_function_value(
  expr: &Expr,
  precision: f64,
) -> Option<Expr> {
  let bits = nominal_bits(precision.floor().max(1.0) as usize) + 64;
  let rm = astro_float::RoundingMode::ToEven;
  with_thread_consts(|cc| wl_bigfloat_expr(expr, precision, bits, rm, cc).ok())
}

/// SetAccuracy[expr, accuracy] — set every numeric leaf of `expr` to a fixed
/// accuracy (absolute uncertainty 10^-accuracy). For a leaf of magnitude |v|
/// this is a precision of `accuracy + Log10[|v|]`; a zero leaf becomes the
/// accuracy-form `0``accuracy`. Mirrors SetPrecision but the precision is
/// computed per leaf from its magnitude.
pub fn set_accuracy_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("SetAccuracy", args);
  if args.len() != 2 {
    return Ok(unevaluated());
  }
  let accuracy = match &args[1] {
    Expr::Integer(n) => *n as f64,
    other => match try_eval_to_f64(other) {
      Some(v) => v,
      None => return Ok(unevaluated()),
    },
  };

  use astro_float::{Consts, RoundingMode};
  let mut cc = Consts::new().map_err(|e| {
    InterpreterError::EvaluationError(format!("BigFloat init error: {e}"))
  })?;
  set_accuracy_walk(&args[0], accuracy, RoundingMode::ToEven, &mut cc)
}

/// Walk `expr`, setting every numeric leaf to `accuracy` (see `set_accuracy_ast`).
fn set_accuracy_walk(
  expr: &Expr,
  accuracy: f64,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _) => {
      leaf_to_bigfloat_at_accuracy(expr, accuracy, rm, cc)
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      leaf_to_bigfloat_at_accuracy(expr, accuracy, rm, cc)
    }
    Expr::List(items) => {
      let mut out = Vec::with_capacity(items.len());
      for item in items {
        out.push(set_accuracy_walk(item, accuracy, rm, cc)?);
      }
      Ok(Expr::List(out.into()))
    }
    Expr::FunctionCall { name, args } => {
      let mut out = Vec::with_capacity(args.len());
      for a in args {
        out.push(set_accuracy_walk(a, accuracy, rm, cc)?);
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: out.into(),
      })
    }
    Expr::BinaryOp { op, left, right } => Ok(binop(
      *op,
      set_accuracy_walk(left, accuracy, rm, cc)?,
      set_accuracy_walk(right, accuracy, rm, cc)?,
    )),
    Expr::UnaryOp { op, operand } => Ok(Expr::UnaryOp {
      op: *op,
      operand: Box::new(set_accuracy_walk(operand, accuracy, rm, cc)?),
    }),
    // Symbolic real constants (Pi, E, …) numericize like SetPrecision; bare
    // symbols pass through unchanged.
    Expr::Constant(_) | Expr::Identifier(_) => {
      leaf_to_bigfloat_at_accuracy(expr, accuracy, rm, cc)
        .or_else(|_| Ok(expr.clone()))
    }
    _ => Ok(expr.clone()),
  }
}

/// Convert a numeric leaf to a BigFloat at the given absolute `accuracy`.
fn leaf_to_bigfloat_at_accuracy(
  expr: &Expr,
  accuracy: f64,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  let v = try_eval_to_f64(expr).ok_or_else(|| {
    InterpreterError::EvaluationError("non-numeric leaf".to_string())
  })?;
  if v == 0.0 {
    // Zero has no relative precision; it carries the accuracy directly in the
    // `0``accuracy` form (stored as BigFloat with digits "0").
    return Ok(Expr::BigFloat("0".to_string(), accuracy));
  }
  let precision = accuracy + v.abs().log10();
  let prec_usize = precision.floor().max(1.0) as usize;
  let bits = nominal_bits(prec_usize) + 64;
  leaf_to_bigfloat(expr, precision, bits, rm, cc)
}

/// SetPrecision[..., MachinePrecision] — walk the tree and demote numeric
/// leaves to machine-precision Reals; symbolic parts stay as-is.
fn set_precision_machine(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(Expr::Real(*n as f64)),
    Expr::BigInteger(n) => Ok(Expr::Real(
      n.to_string().parse::<f64>().unwrap_or(f64::INFINITY),
    )),
    Expr::Real(_) => Ok(expr.clone()),
    Expr::BigFloat(digits, _) => {
      Ok(Expr::Real(digits.parse::<f64>().unwrap_or(f64::NAN)))
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Some(p), Some(q)) =
        (try_eval_to_f64(&args[0]), try_eval_to_f64(&args[1]))
      {
        Ok(Expr::Real(p / q))
      } else {
        Ok(expr.clone())
      }
    }
    Expr::List(items) => {
      let mut out = Vec::with_capacity(items.len());
      for item in items {
        out.push(set_precision_machine(item)?);
      }
      Ok(Expr::List(out.into()))
    }
    Expr::FunctionCall { name, args } => {
      let mut out = Vec::with_capacity(args.len());
      for a in args {
        out.push(set_precision_machine(a)?);
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: out.into(),
      })
    }
    Expr::BinaryOp { op, left, right } => {
      let l = set_precision_machine(left)?;
      let r = set_precision_machine(right)?;
      Ok(binop(*op, l, r))
    }
    Expr::UnaryOp { op, operand } => {
      let o = set_precision_machine(operand)?;
      Ok(Expr::UnaryOp {
        op: *op,
        operand: Box::new(o),
      })
    }
    // Comparisons, rules and the other compound nodes
    _ => Ok(crate::syntax::map_children(expr, &|child| {
      set_precision_machine(child).unwrap_or_else(|_| child.clone())
    })),
  }
}

/// `SetPrecision[expr, Infinity]`: every inexact number replaced by the exact
/// rational it stands for — a machine real by its binary value
/// (`0.1` → `3602879701896397/36028797018963968`), an arbitrary-precision
/// one by its digits (`1.5`30` → `3/2`). Complex numbers convert part by
/// part when the result is evaluated.
fn set_precision_exact(expr: &Expr) -> Expr {
  use num_bigint::BigInt;
  match expr {
    Expr::Real(f) if f.is_finite() => {
      if *f == 0.0 {
        return Expr::Integer(0);
      }
      let bits = f.to_bits();
      let negative = bits >> 63 == 1;
      let exponent = ((bits >> 52) & 0x7ff) as i64;
      let fraction = bits & ((1u64 << 52) - 1);
      // value = mantissa * 2^power
      let (mantissa, power) = if exponent == 0 {
        (fraction, -1074)
      } else {
        (fraction | (1u64 << 52), exponent - 1075)
      };
      let mut num = BigInt::from(mantissa);
      if negative {
        num = -num;
      }
      let one = BigInt::from(1);
      if power >= 0 {
        crate::functions::math_ast::number_theory::make_rational_expr(
          &(num << power as usize),
          &one,
        )
      } else {
        crate::functions::math_ast::number_theory::make_rational_expr(
          &num,
          &(one << (-power) as usize),
        )
      }
    }
    Expr::BigFloat(digits, _) => {
      let mantissa = digits.split("*^").next().unwrap_or(digits);
      let exponent: i64 = digits
        .split("*^")
        .nth(1)
        .and_then(|e| e.parse().ok())
        .unwrap_or(0);
      let (int_part, frac_part) =
        mantissa.split_once('.').unwrap_or((mantissa, ""));
      let Ok(num) = format!("{int_part}{frac_part}").parse::<BigInt>() else {
        return expr.clone();
      };
      let scale = exponent - frac_part.len() as i64;
      let ten = BigInt::from(10);
      let one = BigInt::from(1);
      if scale >= 0 {
        crate::functions::math_ast::number_theory::make_rational_expr(
          &(num * ten.pow(scale as u32)),
          &one,
        )
      } else {
        crate::functions::math_ast::number_theory::make_rational_expr(
          &num,
          &ten.pow((-scale) as u32),
        )
      }
    }
    _ => crate::syntax::map_children(expr, &set_precision_exact),
  }
}

/// N of an exact number whose machine evaluation overflows: evaluated with
/// 30 digits, a value inside the machine range comes back as a machine
/// real; a value outside it stays a big number at machine precision, as
/// in wolframscript (`N[10^400]` is `1.`15.95*^400`).
fn n_eval_beyond_machine_range(expr: &Expr) -> Option<Expr> {
  let value = n_eval_arbitrary(expr, 30.0).ok()?;
  match &value {
    Expr::BigFloat(digits, _) => match digits.parse::<f64>() {
      Ok(f) if f.is_finite() => Some(Expr::Real(f)),
      _ => n_eval_arbitrary(expr, MACHINE_PRECISION_DIGITS).ok(),
    },
    Expr::Real(f) if f.is_finite() => Some(value),
    _ => None,
  }
}

/// N[expr, precision] — arbitrary-precision numeric evaluation using BigFloat
pub fn n_eval_arbitrary(
  expr: &Expr,
  precision: f64,
) -> Result<Expr, InterpreterError> {
  // Handle List recursively at the Expr level
  if let Expr::List(items) = expr {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|e| n_eval_arbitrary(e, precision))
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  // Floor of the precision tag is what bit/digit budgets are sized
  // from — astro-float can't operate on a fractional precision, but
  // we still want the resulting BigFloat to carry the original f64
  // precision marker.
  let prec_usize = precision.floor().max(1.0) as usize;

  // Machine-precision Reals already live at MachinePrecision and cannot be
  // promoted by N — wolframscript returns the Real unchanged.
  if let Expr::Real(_) = expr {
    return Ok(expr.clone());
  }

  // BigFloat input with already-tracked precision: wolframscript clamps the
  // result precision to the input's. `N[1.012345...123, 50]` keeps the
  // 24-digit-equivalent precision marker the literal carried in, rather
  // than fabricating extra digits we don't have. Re-emit the same digits
  // with the original precision marker.
  if let Expr::BigFloat(digits, prec_f64) = expr {
    let prec_floor = if *prec_f64 > 0.0 { *prec_f64 } else { 0.0 };
    if precision > prec_floor {
      return Ok(Expr::BigFloat(digits.clone(), prec_floor));
    }
  }

  use astro_float::{Consts, RoundingMode};

  let mut cc = Consts::new().map_err(|e| {
    InterpreterError::EvaluationError(format!("BigFloat init error: {e}"))
  })?;
  let rm = RoundingMode::ToEven;

  // Compute at one extra 64-bit word above the nominal display precision
  // so astro-float's `convert_to_radix` always emits at least
  // `max_display_digits` decimal digits — without the extra word, e.g. a
  // 192-bit Sqrt[2] only yields 58 digits and N[Sqrt[2], 40] truncates one
  // digit short of Wolfram's 59-digit display.
  let bits = nominal_bits(prec_usize) + 64;

  // Try full conversion to BigFloat first (fast path for purely numeric expressions)
  if let Ok(result) = wl_bigfloat_expr(expr, precision, bits, rm, &mut cc) {
    Ok(result)
  } else {
    // Try complex BigFloat evaluation (handles expressions with I)
    if let Ok((mut re, im)) = expr_to_complex_bigfloat(expr, bits, rm, &mut cc)
    {
      // A real part below the working precision of the imaginary one is
      // rounding noise of an exactly imaginary value (`N[ArcCosh[1/2], 20]`
      // is ln|1/2 + Sqrt[3]/2 I| = 0 plus Pi/3 I).
      if let (Some(re_exp), Some(im_exp)) = (re.exponent(), im.exponent())
        && !re.is_zero()
        && !im.is_zero()
        && i64::from(re_exp) + bits as i64 - 16 < i64::from(im_exp)
      {
        re = astro_float::BigFloat::from_i32(0, bits);
      }
      // For complex function results, compute per-component precision markers
      if let Expr::FunctionCall { name: _, args } = expr
        && !im.is_zero()
        && args.len() == 1
        && let Ok(input_complex) =
          expr_to_complex_bigfloat(&args[0], bits, rm, &mut cc)
        && let Ok((prec_re_str, prec_im_str)) =
          compute_complex_precision_markers(
            &input_complex.0,
            &input_complex.1,
            &re,
            &im,
            prec_usize,
            rm,
            &mut cc,
          )
      {
        return build_complex_result_with_string_precision(
          &re,
          &im,
          &prec_re_str,
          &prec_im_str,
          prec_usize,
          rm,
          &mut cc,
        );
      }
      return build_complex_bigfloat_result(&re, &im, prec_usize, rm, &mut cc);
    }
    // Fall back to partial evaluation: convert numeric sub-expressions
    // to arbitrary precision while leaving symbolic parts as-is
    n_eval_arbitrary_partial(expr, precision, bits, rm, &mut cc)
  }
}

/// Recursively apply arbitrary-precision evaluation to sub-expressions.
/// Numeric parts are converted to BigFloat; symbolic parts are left as-is.
fn n_eval_arbitrary_partial(
  expr: &Expr,
  precision: f64,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  // An exact zero stays exact under `N[expr, p]` (Head Integer, Precision
  // Infinity), just like at the top level — `N[x == 0, 10]` is `x == 0`.
  if matches!(expr, Expr::Integer(0))
    || matches!(expr, Expr::BigInteger(n) if n.sign() == Sign::NoSign)
  {
    return Ok(Expr::Integer(0));
  }
  // Lists are handled element-wise so nested lists (e.g. inside an
  // equation) pick up the precision tag too.
  if let Expr::List(items) = expr {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|e| n_eval_arbitrary_partial(e, precision, bits, rm, cc))
      .collect();
    return Ok(Expr::List(results?.into()));
  }
  // A machine-precision Real already carries the maximum information
  // an f64 can hold — N cannot recover more digits from it, so leave
  // it alone (matches wolframscript: `N[F[1.2, 2/9], $MachinePrecision]`
  // keeps the `1.2` literal and only expands the rational `2/9`).
  if let Expr::Real(_) = expr {
    return Ok(expr.clone());
  }
  // BigFloat input with a lower precision tag: keep the original tag
  // — N can't manufacture digits the input didn't carry. Matches
  // wolframscript: `N[F[1.2\`3, 2/9], 5]` keeps the `1.2\`3` literal at
  // precision 3 while promoting only the rational to precision 5.
  if let Expr::BigFloat(_, prec_f64) = expr {
    let prec_in = if *prec_f64 > 0.0 { *prec_f64 } else { 0.0 };
    if precision > prec_in {
      return Ok(expr.clone());
    }
  }
  // If the whole expression can be converted to BigFloat, do it
  if let Ok(result) = wl_bigfloat_expr(expr, precision, bits, rm, cc) {
    return Ok(result);
  }

  match expr {
    Expr::FunctionCall { .. }
      if let Some(value) = user_n_value(expr, Some(precision))? =>
    {
      n_eval_arbitrary(&value, precision)
    }
    Expr::FunctionCall { name, args } => {
      // Honour NHoldAll / NHoldFirst / NHoldRest so e.g. `N[Out[0], 50]`
      // doesn't recurse into Out's slot and rewrite the index as a
      // BigFloat. (Out has NHoldFirst; the `0` slot must stay literal.)
      let attrs = {
        let builtin = crate::evaluator::get_builtin_attributes(name);
        let user = crate::FUNC_ATTRS
          .with(|m| m.borrow().get(name).copied().unwrap_or_default());
        let mut combined = builtin;
        combined.add(user.to_u32());
        combined
      };
      let hold_all = attrs.contains(Attributes::NHoldAll);
      let hold_first = attrs.contains(Attributes::NHoldFirst);
      let hold_rest = attrs.contains(Attributes::NHoldRest);
      let new_args: Result<Vec<Expr>, _> = args
        .iter()
        .enumerate()
        .map(|(i, a)| {
          let held = hold_all || (hold_first && i == 0) || (hold_rest && i > 0);
          if held {
            Ok(a.clone())
          } else {
            n_eval_arbitrary_partial(a, precision, bits, rm, cc)
          }
        })
        .collect();
      let new_expr = Expr::FunctionCall {
        name: name.clone(),
        args: new_args?.into(),
      };
      // Try to re-evaluate after converting numeric args
      match crate::evaluator::evaluate_expr_to_expr(&new_expr) {
        Ok(result) => Ok(result),
        Err(_) => Ok(new_expr),
      }
    }
    Expr::BinaryOp { op, left, right } => {
      let l = n_eval_arbitrary_partial(left, precision, bits, rm, cc)?;
      let r = if keeps_exact_exponent(*op, right) {
        (**right).clone()
      } else {
        n_eval_arbitrary_partial(right, precision, bits, rm, cc)?
      };
      Ok(binop(*op, l, r))
    }
    // Comparison chains: N maps over the operands (`N[x == 1/3, 5]` →
    // `x == 0.33333`).
    Expr::Comparison {
      operands,
      operators,
    } => {
      let new_operands: Result<Vec<Expr>, _> = operands
        .iter()
        .map(|o| n_eval_arbitrary_partial(o, precision, bits, rm, cc))
        .collect();
      Ok(Expr::Comparison {
        operands: new_operands?,
        operators: operators.clone(),
      })
    }
    Expr::UnaryOp { op, operand } => {
      let inner = n_eval_arbitrary_partial(operand, precision, bits, rm, cc)?;
      Ok(Expr::UnaryOp {
        op: *op,
        operand: Box::new(inner),
      })
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Ok(Expr::Rule {
      pattern: pattern.clone(),
      replacement: Box::new(n_eval_arbitrary_partial(
        replacement,
        precision,
        bits,
        rm,
        cc,
      )?),
    }),
    // Identifier: check user-installed `N[sym, p] = value` rules. The
    // canonical LHS Wolfram stores under NValues is
    // `N[sym, {p., Infinity}]`; match on numeric `p` equal to (or below)
    // the requested precision. Also fire for "any-precision" LHS shapes
    // like `N[sym, _]` (Blank[]) or `N[sym, x_]` (Pattern[x, Blank[]]),
    // which match every requested precision.
    Expr::Identifier(_) => match user_n_value(expr, Some(precision))? {
      Some(value) => n_eval_arbitrary(&value, precision),
      None => Ok(expr.clone()),
    },
    // Other non-numeric expressions: leave as-is
    _ => Ok(expr.clone()),
  }
}

/// The value of a user NValue rule for `expr` — `N[a] = v`, `N[f[1]] = v`,
/// `N[f[x_]] := body`, stored under the NValues of `a` or `f` — at machine
/// precision (`precision` None) or at `precision` digits. The rule's
/// target is matched against `expr` as written, before its arguments are
/// numericized, as Wolfram does (`N[f[1]] = v` does not fire for `f[1.]`).
pub(crate) fn user_n_value(
  expr: &Expr,
  precision: Option<f64>,
) -> Result<Option<Expr>, InterpreterError> {
  let tag = match expr {
    Expr::Identifier(name) => name,
    Expr::FunctionCall { name, .. } => name,
    _ => return Ok(None),
  };
  let Some(entries) = crate::evaluator::assignment::N_VALUES
    .with(|m| m.borrow().get(tag).cloned())
  else {
    return Ok(None);
  };
  // Wolfram asks for the value at machine precision as
  // `N[expr, {$MachinePrecision, $MachinePrecision}]` and then
  // `N[expr, $MachinePrecision]`, and at `p` digits as `N[expr, p.]` and
  // then `N[expr, {p., Infinity}]`; each rule's left side is matched
  // against those calls in turn (so `N[c, p_?(# > 10 &)] := p` gives
  // `11.` for `N[c, 11]`). A literal `MachinePrecision` on a stored left
  // side stands for its value.
  let machine = Expr::Real(MACHINE_PRECISION_DIGITS);
  let candidates: Vec<Expr> = match precision {
    None => vec![
      Expr::List(vec![machine.clone(), machine.clone()].into()),
      machine,
    ],
    Some(p) => {
      let p = Expr::Real(p);
      vec![
        p.clone(),
        Expr::List(vec![p, Expr::Identifier("Infinity".to_string())].into()),
      ]
    }
  };
  let machine_symbol_to_value = |e: &Expr| -> Expr {
    match e {
      Expr::Identifier(s) if s == "MachinePrecision" => {
        Expr::Real(MACHINE_PRECISION_DIGITS)
      }
      Expr::List(items) => Expr::List(
        items
          .iter()
          .map(|i| match i {
            Expr::Identifier(s) if s == "MachinePrecision" => {
              Expr::Real(MACHINE_PRECISION_DIGITS)
            }
            other => other.clone(),
          })
          .collect(),
      ),
      other => other.clone(),
    }
  };
  for candidate in &candidates {
    let call = Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![expr.clone(), candidate.clone()].into(),
    };
    for (lhs, rhs) in &entries {
      let Expr::FunctionCall { name, args } = lhs else {
        continue;
      };
      if name != "N" || args.len() != 2 {
        continue;
      }
      let pattern = Expr::FunctionCall {
        name: "N".to_string(),
        args: vec![args[0].clone(), machine_symbol_to_value(&args[1])].into(),
      };
      if let Some(bindings) =
        crate::evaluator::pattern_matching::match_pattern(&call, &pattern)
        && let Some(value) =
          crate::evaluator::pattern_matching::instantiate_replacement(
            rhs, &bindings,
          )?
      {
        return Ok(Some(value));
      }
    }
  }
  Ok(None)
}

/// Recursively convert an Expr to a BigFloat with the given precision.
pub fn expr_to_bigfloat(
  expr: &Expr,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<astro_float::BigFloat, InterpreterError> {
  use astro_float::BigFloat;

  match expr {
    Expr::Integer(n) => Ok(BigFloat::from_i128(*n, bits)),
    Expr::BigInteger(n) => {
      // Convert BigInt to BigFloat by parsing its decimal string
      let s = n.to_string();
      Ok(BigFloat::parse(&s, astro_float::Radix::Dec, bits, rm, cc))
    }
    Expr::Real(f) => Ok(BigFloat::from_f64(*f, bits)),
    Expr::BigFloat(digits, _) => Ok(BigFloat::parse(
      digits,
      astro_float::Radix::Dec,
      bits,
      rm,
      cc,
    )),
    // A variable holding `Pi`/`E`/`Degree` substitutes in as `Identifier`
    // rather than `Constant` (unlike a literal `Pi` token), so both variants
    // are accepted here — matching how GoldenRatio, EulerGamma, etc. below
    // already handle their `Identifier` form.
    Expr::Constant(name) | Expr::Identifier(name)
      if matches!(name.as_str(), "Pi" | "-Pi" | "E" | "Degree") =>
    {
      match name.as_str() {
        "Pi" | "-Pi" => {
          let pi = cc.pi(bits, rm);
          if name == "-Pi" { Ok(pi.neg()) } else { Ok(pi) }
        }
        "E" => Ok(cc.e(bits, rm)),
        "Degree" => {
          let pi = cc.pi(bits, rm);
          let d180 = BigFloat::from_i32(180, bits);
          Ok(pi.div(&d180, bits, rm))
        }
        _ => unreachable!(),
      }
    }
    Expr::Constant(name) => Err(InterpreterError::EvaluationError(format!(
      "N: cannot evaluate constant {name} to arbitrary precision"
    ))),
    Expr::Identifier(name) if name == "GoldenRatio" => {
      // GoldenRatio = (1 + Sqrt[5]) / 2
      let five = BigFloat::from_i32(5, bits);
      let sqrt5 = five.sqrt(bits, rm);
      let one = BigFloat::from_i32(1, bits);
      let numer = one.add(&sqrt5, bits, rm);
      let two = BigFloat::from_i32(2, bits);
      Ok(numer.div(&two, bits, rm))
    }
    Expr::Identifier(name) if name == "GoldenAngle" => {
      // GoldenAngle = Pi * (3 - Sqrt[5]); the subtraction cancels leading
      // bits, so work with guard bits and round at the end
      let wbits = bits + 64;
      let five = BigFloat::from_i32(5, wbits);
      let sqrt5 = five.sqrt(wbits, rm);
      let three = BigFloat::from_i32(3, wbits);
      let factor = three.sub(&sqrt5, wbits, rm);
      let pi = astro_float::Consts::new()
        .map_or_else(|_| BigFloat::from_i32(0, wbits), |mut c| c.pi(wbits, rm));
      Ok(pi.mul(&factor, bits, rm))
    }
    Expr::Identifier(name) if name == "EulerGamma" => {
      Ok(compute_euler_gamma(bits, rm, cc))
    }
    Expr::Identifier(name) if name == "Catalan" => {
      Ok(compute_catalan(bits, rm, cc))
    }
    Expr::Identifier(name) if name == "Glaisher" => {
      Ok(compute_glaisher(bits, rm, cc))
    }
    Expr::Identifier(name) if name == "Khinchin" => {
      Ok(compute_khinchin(bits, rm, cc))
    }
    // MachinePrecision = Log10[2^53] = 53 * Log10[2].
    // `N[MachinePrecision, 30]` should yield the arbitrary-precision
    // decimal expansion `15.9545897701910033463281614204`.
    Expr::Identifier(name) if name == "MachinePrecision" => {
      let two = BigFloat::from_i32(2, bits);
      let ten = BigFloat::from_i32(10, bits);
      let log2 = two.ln(bits, rm, cc);
      let log10 = ten.ln(bits, rm, cc);
      let log10_2 = log2.div(&log10, bits, rm);
      let fifty_three = BigFloat::from_i32(53, bits);
      Ok(fifty_three.mul(&log10_2, bits, rm))
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let val = expr_to_bigfloat(operand, bits, rm, cc)?;
      Ok(val.neg())
    }
    Expr::BinaryOp { op, left, right } => {
      // For integer exponents, use powi (repeated squaring) for efficiency
      if matches!(op, BinaryOperator::Power)
        && let Some(n) = try_as_integer(right)
      {
        let base = expr_to_bigfloat(left, bits, rm, cc)?;
        return Ok(bigfloat_powi(&base, n, bits, rm));
      }
      let l = expr_to_bigfloat(left, bits, rm, cc)?;
      let r = expr_to_bigfloat(right, bits, rm, cc)?;
      match op {
        BinaryOperator::Plus => Ok(l.add(&r, bits, rm)),
        BinaryOperator::Minus => Ok(l.sub(&r, bits, rm)),
        BinaryOperator::Times => Ok(l.mul(&r, bits, rm)),
        BinaryOperator::Divide => Ok(l.div(&r, bits, rm)),
        // A negative base under a non-integer exponent is complex: left to
        // `expr_to_complex_bigfloat` rather than computed as NaN
        BinaryOperator::Power if l.is_negative() => {
          Err(InterpreterError::EvaluationError(
            "N: complex power of a negative base".into(),
          ))
        }
        BinaryOperator::Power => Ok(l.pow(&r, bits, rm, cc)),
        _ => Err(InterpreterError::EvaluationError(
          "N: unsupported binary operator for arbitrary precision".into(),
        )),
      }
    }
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Rational" if args.len() == 2 => {
          let n = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let d = expr_to_bigfloat(&args[1], bits, rm, cc)?;
          Ok(n.div(&d, bits, rm))
        }
        // Root[f, k] / Root[f, k, 0] — Newton-refine the (real) k-th root
        // to `bits` of precision, e.g. `N[Root[#^3 - # - 1 &, 1], 30]` (the
        // plastic constant).
        "Root" if args.len() == 2 || args.len() == 3 => {
          root_n_eval_arbitrary(&args[0], &args[1], bits, rm).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "N: cannot evaluate Root[…] to arbitrary precision".into(),
            )
          })
        }
        "Sqrt" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.sqrt(bits, rm))
        }
        "Sin" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.sin(bits, rm, cc))
        }
        "Cos" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.cos(bits, rm, cc))
        }
        "Tan" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.tan(bits, rm, cc))
        }
        "Exp" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.exp(bits, rm, cc))
        }
        "Log" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.ln(bits, rm, cc))
        }
        "Log" if args.len() == 2 => {
          let base = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let val = expr_to_bigfloat(&args[1], bits, rm, cc)?;
          Ok(val.log(&base, bits, rm, cc))
        }
        "Abs" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.abs())
        }
        "Sinh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.sinh(bits, rm, cc))
        }
        "Cosh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.cosh(bits, rm, cc))
        }
        "Tanh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.tanh(bits, rm, cc))
        }
        "ArcSin" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.asin(bits, rm, cc))
        }
        "ArcCos" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.acos(bits, rm, cc))
        }
        "ArcTan" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.atan(bits, rm, cc))
        }
        "ArcSinh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.asinh(bits, rm, cc))
        }
        "ArcCosh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.acosh(bits, rm, cc))
        }
        "ArcTanh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.atanh(bits, rm, cc))
        }
        // The reciprocal circular/hyperbolic functions and the inverses of
        // the reciprocals. astro-float has no primitive for any of them, but
        // each is one `1/x` away from a primitive that it does have — and
        // leaving them out is what made `N[Coth[1], 20]` come back as
        // `Coth[1.`20.]`.
        "Cot" | "Sec" | "Csc" | "Coth" | "Sech" | "Csch" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          // `Cot`/`Coth` go through `Cos/Sin` and `Cosh/Sinh` rather than
          // `1/Tan`: reciprocating an already-rounded tangent costs a unit
          // in the last displayed digit against wolframscript.
          let (num, den) = match name.as_str() {
            "Cot" => (v.cos(bits, rm, cc), v.sin(bits, rm, cc)),
            "Sec" => (BigFloat::from_i32(1, bits), v.cos(bits, rm, cc)),
            "Csc" => (BigFloat::from_i32(1, bits), v.sin(bits, rm, cc)),
            "Coth" => (v.cosh(bits, rm, cc), v.sinh(bits, rm, cc)),
            "Sech" => (BigFloat::from_i32(1, bits), v.cosh(bits, rm, cc)),
            _ => (BigFloat::from_i32(1, bits), v.sinh(bits, rm, cc)),
          };
          Ok(num.div(&den, bits, rm))
        }
        "ArcCot" | "ArcSec" | "ArcCsc" | "ArcCoth" | "ArcSech" | "ArcCsch"
          if args.len() == 1 =>
        {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let r = BigFloat::from_i32(1, bits).div(&v, bits, rm);
          Ok(match name.as_str() {
            "ArcCot" => r.atan(bits, rm, cc),
            "ArcSec" => r.acos(bits, rm, cc),
            "ArcCsc" => r.asin(bits, rm, cc),
            "ArcCoth" => r.atanh(bits, rm, cc),
            "ArcSech" => r.acosh(bits, rm, cc),
            _ => r.asinh(bits, rm, cc),
          })
        }
        "Zeta" | "HurwitzZeta" if !args.is_empty() && args.len() <= 2 => {
          let sv = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let av = match args.get(1) {
            Some(a) => expr_to_bigfloat(a, bits, rm, cc)?,
            None => BigFloat::from_i32(1, bits),
          };
          // HurwitzZeta has a pole at every non-positive integer a when
          // s > 0 (the k = -a term); Zeta leaves that term out.
          let hurwitz_pole = name == "HurwitzZeta"
            && av.fract().is_zero()
            && !av.is_positive()
            && sv.is_positive();
          (!hurwitz_pole)
            .then(|| {
              hurwitz_zeta_bigfloat(
                &sv,
                &av,
                name == "HurwitzZeta",
                bits,
                rm,
                cc,
              )
            })
            .flatten()
            .ok_or_else(|| {
              InterpreterError::EvaluationError(format!(
                "N: cannot evaluate {name}[...] to arbitrary precision"
              ))
            })
        }
        "PolyGamma" if !args.is_empty() && args.len() <= 2 => {
          let (n, z) = match &args[..] {
            [z] => (Some(0), z),
            [n, z] => (polygamma_order(n), z),
            _ => unreachable!(),
          };
          let z = expr_to_bigfloat(z, bits, rm, cc)?;
          n.and_then(|n| polygamma_bigfloat(n, &z, bits, rm, cc))
            .ok_or_else(|| {
              InterpreterError::EvaluationError(
                "N: cannot evaluate PolyGamma[...] to arbitrary precision"
                  .into(),
              )
            })
        }
        // HarmonicNumber[x, r] = Zeta[r] - HurwitzZeta[r, x + 1], and
        // PolyGamma[x + 1] + EulerGamma at r = 1; real x > -1.
        "HarmonicNumber" if (1..=2).contains(&args.len()) => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let r = match args.get(1) {
            Some(r) => expr_to_bigfloat(r, bits, rm, cc)?,
            None => BigFloat::from_i32(1, bits),
          };
          harmonic_number_bigfloat(&x, &r, bits, rm, cc).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "N: cannot evaluate HarmonicNumber[...] to arbitrary precision"
                .into(),
            )
          })
        }
        "LogGamma" if args.len() == 1 => {
          let z = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(gamma_bigfloat(&z, bits, rm, cc).ln(bits, rm, cc))
        }
        "Plus" => {
          // Evaluated Plus[a, b, c, ...] as a function call
          let mut result = BigFloat::from_i32(0, bits);
          for arg in args {
            let v = expr_to_bigfloat(arg, bits, rm, cc)?;
            result = result.add(&v, bits, rm);
          }
          Ok(result)
        }
        "Times" => {
          let mut result = BigFloat::from_i32(1, bits);
          for arg in args {
            let v = expr_to_bigfloat(arg, bits, rm, cc)?;
            result = result.mul(&v, bits, rm);
          }
          Ok(result)
        }
        "Power" if args.len() == 2 => {
          if let Some(n) = try_as_integer(&args[1]) {
            let base = expr_to_bigfloat(&args[0], bits, rm, cc)?;
            return Ok(bigfloat_powi(&base, n, bits, rm));
          }
          let base = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          if base.is_negative() {
            return Err(InterpreterError::EvaluationError(
              "N: complex power of a negative base".into(),
            ));
          }
          let exp = expr_to_bigfloat(&args[1], bits, rm, cc)?;
          Ok(base.pow(&exp, bits, rm, cc))
        }
        "AiryAi" | "AiryBi" | "AiryAiPrime" | "AiryBiPrime"
          if args.len() == 1 =>
        {
          use crate::functions::math_ast::bessel_bigfloat::{
            AiryKind, airy_bigfloat,
          };
          let kind = match name.as_str() {
            "AiryAi" => AiryKind::Ai,
            "AiryBi" => AiryKind::Bi,
            "AiryAiPrime" => AiryKind::AiPrime,
            _ => AiryKind::BiPrime,
          };
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          airy_bigfloat(kind, &x, bits, rm, cc).ok_or_else(|| {
            InterpreterError::EvaluationError(format!("N: {name}[0] is exact"))
          })
        }
        "BesselJ" | "BesselY" | "BesselI" | "BesselK" if args.len() == 2 => {
          use crate::functions::math_ast::bessel_bigfloat::{
            BesselKind, bessel_bigfloat,
          };
          let kind = match name.as_str() {
            "BesselJ" => BesselKind::J,
            "BesselY" => BesselKind::Y,
            "BesselI" => BesselKind::I,
            _ => BesselKind::K,
          };
          let nu = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let x = expr_to_bigfloat(&args[1], bits, rm, cc)?;
          bessel_bigfloat(kind, &nu, &x, bits, rm, cc).ok_or_else(|| {
            InterpreterError::EvaluationError(format!(
              "N: {name}[...] needs a positive argument here"
            ))
          })
        }
        "Gamma" if args.len() == 1 => {
          let z = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(gamma_bigfloat(&z, bits, rm, cc))
        }
        // Non-integer degrees of the orthogonal-polynomial family (integer
        // degrees expand to exact polynomials before N sees them).
        "LegendreP"
        | "LaguerreL"
        | "HermiteH"
        | "Fibonacci"
        | "QPochhammer"
        | "Hypergeometric1F1"
        | "Hypergeometric0F1"
        | "Hypergeometric0F1Regularized"
        | "FresnelS"
        | "FresnelC"
        | "LogisticSigmoid"
          if (1..=3).contains(&args.len()) =>
        {
          let v: Vec<BigFloat> = args
            .iter()
            .map(|a| expr_to_bigfloat(a, bits, rm, cc))
            .collect::<Result<_, _>>()?;
          crate::functions::math_ast::orthogonal_bigfloat::orthogonal_bigfloat(
            name, &v, bits, rm, cc,
          )
          .ok_or_else(|| {
            InterpreterError::EvaluationError(format!(
              "N: cannot evaluate {name}[...] to arbitrary precision"
            ))
          })
        }
        "Erf" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(bigfloat_erf(&x, bits, rm, cc))
        }
        "Erfc" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(bigfloat_erfc(&x, bits, rm, cc))
        }
        "Erfi" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(bigfloat_erfi(&x, bits, rm, cc))
        }
        "ExpIntegralEi" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          crate::functions::math_ast::exp_integral_ei_bigfloat(&x, bits, rm, cc)
            .ok_or_else(|| {
              InterpreterError::EvaluationError(
                "N: ExpIntegralEi has a pole at 0".into(),
              )
            })
        }
        // The real-valued cases of the exponential-integral family; a
        // complex value (negative argument) goes through the complex path.
        "ExpIntegralE" | "LogIntegral" | "Gamma" => {
          let (re, im) =
            exp_integral_family_bigfloat(name, args, bits, rm, cc)?;
          if im.is_zero() {
            Ok(re)
          } else {
            Err(InterpreterError::EvaluationError(format!(
              "N: {name}[...] is complex here"
            )))
          }
        }
        "ChampernowneNumber" if args.len() <= 1 => {
          let base = match args.first() {
            None => 10,
            Some(Expr::Integer(b)) if *b >= 2 => *b,
            _ => {
              return Err(InterpreterError::EvaluationError(
                "N: invalid ChampernowneNumber base".into(),
              ));
            }
          };
          Ok(compute_champernowne(base, bits, rm, cc))
        }
        _ => Err(InterpreterError::EvaluationError(format!(
          "N: cannot evaluate {name}[...] to arbitrary precision"
        ))),
      }
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "N: cannot evaluate expression to arbitrary precision: {}",
      expr_to_string(expr)
    ))),
  }
}

/// `z^w` for complex arbitrary-precision `z` and `w`. An integer exponent
/// multiplies out exactly (a negative one through the reciprocal); any other
/// is `exp(w log z)` on the principal branch, `log z = ln|z| + i arg z`.
fn complex_bigfloat_pow(
  (zr, zi): (&astro_float::BigFloat, &astro_float::BigFloat),
  (wr, wi): (&astro_float::BigFloat, &astro_float::BigFloat),
  integer_exponent: Option<i128>,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(astro_float::BigFloat, astro_float::BigFloat), InterpreterError> {
  use astro_float::BigFloat;
  let zero = || BigFloat::from_i32(0, bits);
  let mul = |(ar, ai): (&BigFloat, &BigFloat),
             (br, bi): (&BigFloat, &BigFloat)| {
    (
      ar.mul(br, bits, rm).sub(&ai.mul(bi, bits, rm), bits, rm),
      ar.mul(bi, bits, rm).add(&ai.mul(br, bits, rm), bits, rm),
    )
  };
  if let Some(n) = integer_exponent {
    if zi.is_zero() {
      return Ok((bigfloat_powi(zr, n, bits, rm), zero()));
    }
    // Repeated squaring on the complex base
    let (mut br, mut bi) = if n < 0 {
      let denom = zr.mul(zr, bits, rm).add(&zi.mul(zi, bits, rm), bits, rm);
      (zr.div(&denom, bits, rm), zi.neg().div(&denom, bits, rm))
    } else {
      (zr.clone(), zi.clone())
    };
    let mut k = n.unsigned_abs();
    let (mut rr, mut ri) = (BigFloat::from_i32(1, bits), zero());
    while k > 0 {
      if k & 1 == 1 {
        (rr, ri) = mul((&rr, &ri), (&br, &bi));
      }
      (br, bi) = mul((&br, &bi), (&br, &bi));
      k >>= 1;
    }
    return Ok((rr, ri));
  }
  if zr.is_zero() && zi.is_zero() {
    if wr.is_positive() {
      return Ok((zero(), zero()));
    }
    return Err(InterpreterError::EvaluationError(
      "N: 0 to a non-positive power".into(),
    ));
  }
  // arg z, quadrant by quadrant
  let pi = cc.pi(bits, rm);
  let arg = if zr.is_zero() {
    let half_pi = pi.div(&BigFloat::from_i32(2, bits), bits, rm);
    if zi.is_negative() {
      half_pi.neg()
    } else {
      half_pi
    }
  } else {
    let base = zi.div(zr, bits, rm).atan(bits, rm, cc);
    if zr.is_positive() {
      base
    } else if zi.is_negative() {
      base.sub(&pi, bits, rm)
    } else {
      base.add(&pi, bits, rm)
    }
  };
  let modulus_sq = zr.mul(zr, bits, rm).add(&zi.mul(zi, bits, rm), bits, rm);
  let ln_modulus =
    modulus_sq
      .ln(bits, rm, cc)
      .div(&BigFloat::from_i32(2, bits), bits, rm);
  // w log z
  let (er, ei) = mul((wr, wi), (&ln_modulus, &arg));
  let scale = er.exp(bits, rm, cc);
  Ok((
    scale.mul(&ei.cos(bits, rm, cc), bits, rm),
    scale.mul(&ei.sin(bits, rm, cc), bits, rm),
  ))
}

/// Convert an expression to a complex (BigFloat, BigFloat) pair with given precision.
/// Returns (real_part, imaginary_part) as BigFloats.
/// Handles expressions involving the imaginary unit I.
fn expr_to_complex_bigfloat(
  expr: &Expr,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(astro_float::BigFloat, astro_float::BigFloat), InterpreterError> {
  use astro_float::BigFloat;

  // Fast path: if purely real, delegate (a NaN means a real function left
  // its domain, e.g. Log of a negative sum, and the value is complex)
  if let Ok(val) = expr_to_bigfloat(expr, bits, rm, cc)
    && !val.is_nan()
  {
    return Ok((val, BigFloat::from_i32(0, bits)));
  }

  match expr {
    // I → (0, 1)
    Expr::Identifier(name) if name == "I" => {
      Ok((BigFloat::from_i32(0, bits), BigFloat::from_i32(1, bits)))
    }
    // Unary minus
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (re, im) = expr_to_complex_bigfloat(operand, bits, rm, cc)?;
      Ok((re.neg(), im.neg()))
    }
    // Binary operations
    Expr::BinaryOp { op, left, right } => {
      let (lr, li) = expr_to_complex_bigfloat(left, bits, rm, cc)?;
      let (rr, ri) = expr_to_complex_bigfloat(right, bits, rm, cc)?;
      match op {
        BinaryOperator::Plus => {
          Ok((lr.add(&rr, bits, rm), li.add(&ri, bits, rm)))
        }
        BinaryOperator::Minus => {
          Ok((lr.sub(&rr, bits, rm), li.sub(&ri, bits, rm)))
        }
        BinaryOperator::Times => {
          // (lr + li*i) * (rr + ri*i) = (lr*rr - li*ri) + (lr*ri + li*rr)*i
          let re = lr.mul(&rr, bits, rm).sub(&li.mul(&ri, bits, rm), bits, rm);
          let im = lr.mul(&ri, bits, rm).add(&li.mul(&rr, bits, rm), bits, rm);
          Ok((re, im))
        }
        BinaryOperator::Divide => {
          // (lr + li*i) / (rr + ri*i)
          let denom =
            rr.mul(&rr, bits, rm).add(&ri.mul(&ri, bits, rm), bits, rm);
          let re = lr.mul(&rr, bits, rm).add(&li.mul(&ri, bits, rm), bits, rm);
          let im = li.mul(&rr, bits, rm).sub(&lr.mul(&ri, bits, rm), bits, rm);
          Ok((re.div(&denom, bits, rm), im.div(&denom, bits, rm)))
        }
        BinaryOperator::Power => complex_bigfloat_pow(
          (&lr, &li),
          (&rr, &ri),
          try_as_integer(right),
          bits,
          rm,
          cc,
        ),
        _ => Err(InterpreterError::EvaluationError(
          "N: unsupported binary operator for complex arbitrary precision"
            .into(),
        )),
      }
    }
    // Function calls
    Expr::FunctionCall { name, args } => match name.as_str() {
      // A complex `Root[f, k]`: the real case is `expr_to_bigfloat`'s.
      "Root" if args.len() == 2 || args.len() == 3 => {
        root_n_eval_arbitrary_complex(&args[0], &args[1], bits, rm).ok_or_else(
          || {
            InterpreterError::EvaluationError(
              "N: cannot evaluate Root[…] to arbitrary precision".into(),
            )
          },
        )
      }
      "Times" => {
        let mut result =
          (BigFloat::from_i32(1, bits), BigFloat::from_i32(0, bits));
        for arg in args {
          let (rr, ri) = expr_to_complex_bigfloat(arg, bits, rm, cc)?;
          let (lr, li) = result;
          let re = lr.mul(&rr, bits, rm).sub(&li.mul(&ri, bits, rm), bits, rm);
          let im = lr.mul(&ri, bits, rm).add(&li.mul(&rr, bits, rm), bits, rm);
          result = (re, im);
        }
        Ok(result)
      }
      "Plus" => {
        let mut result =
          (BigFloat::from_i32(0, bits), BigFloat::from_i32(0, bits));
        for arg in args {
          let (rr, ri) = expr_to_complex_bigfloat(arg, bits, rm, cc)?;
          result = (result.0.add(&rr, bits, rm), result.1.add(&ri, bits, rm));
        }
        Ok(result)
      }
      "Complex" if args.len() == 2 => {
        let re = expr_to_bigfloat(&args[0], bits, rm, cc)?;
        let im = expr_to_bigfloat(&args[1], bits, rm, cc)?;
        Ok((re, im))
      }
      "Sin" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        let sin_a = a.sin(bits, rm, cc);
        let cos_a = a.cos(bits, rm, cc);
        let cosh_b = b.cosh(bits, rm, cc);
        let sinh_b = b.sinh(bits, rm, cc);
        Ok((sin_a.mul(&cosh_b, bits, rm), cos_a.mul(&sinh_b, bits, rm)))
      }
      "Cos" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        let sin_a = a.sin(bits, rm, cc);
        let cos_a = a.cos(bits, rm, cc);
        let cosh_b = b.cosh(bits, rm, cc);
        let sinh_b = b.sinh(bits, rm, cc);
        Ok((
          cos_a.mul(&cosh_b, bits, rm),
          sin_a.mul(&sinh_b, bits, rm).neg(),
        ))
      }
      "Tan" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // tan(z) = sin(z)/cos(z)
        let sin_a = a.sin(bits, rm, cc);
        let cos_a = a.cos(bits, rm, cc);
        let cosh_b = b.cosh(bits, rm, cc);
        let sinh_b = b.sinh(bits, rm, cc);
        // sin(z) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        let sr = sin_a.mul(&cosh_b, bits, rm);
        let si = cos_a.mul(&sinh_b, bits, rm);
        // cos(z) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        let cr = cos_a.mul(&cosh_b, bits, rm);
        let ci = sin_a.mul(&sinh_b, bits, rm).neg();
        // (sr + si*i) / (cr + ci*i)
        let denom = cr.mul(&cr, bits, rm).add(&ci.mul(&ci, bits, rm), bits, rm);
        let re = sr.mul(&cr, bits, rm).add(&si.mul(&ci, bits, rm), bits, rm);
        let im = si.mul(&cr, bits, rm).sub(&sr.mul(&ci, bits, rm), bits, rm);
        Ok((re.div(&denom, bits, rm), im.div(&denom, bits, rm)))
      }
      "Exp" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // exp(a+bi) = exp(a)*(cos(b) + i*sin(b))
        let exp_a = a.exp(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        Ok((exp_a.mul(&cos_b, bits, rm), exp_a.mul(&sin_b, bits, rm)))
      }
      "Sinh" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
        let sinh_a = a.sinh(bits, rm, cc);
        let cosh_a = a.cosh(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        Ok((sinh_a.mul(&cos_b, bits, rm), cosh_a.mul(&sin_b, bits, rm)))
      }
      "Cosh" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
        let sinh_a = a.sinh(bits, rm, cc);
        let cosh_a = a.cosh(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        Ok((cosh_a.mul(&cos_b, bits, rm), sinh_a.mul(&sin_b, bits, rm)))
      }
      "Tanh" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // tanh(z) = sinh(z)/cosh(z)
        let sinh_a = a.sinh(bits, rm, cc);
        let cosh_a = a.cosh(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        let sr = sinh_a.mul(&cos_b, bits, rm);
        let si = cosh_a.mul(&sin_b, bits, rm);
        let cr = cosh_a.mul(&cos_b, bits, rm);
        let ci = sinh_a.mul(&sin_b, bits, rm);
        let denom = cr.mul(&cr, bits, rm).add(&ci.mul(&ci, bits, rm), bits, rm);
        let re = sr.mul(&cr, bits, rm).add(&si.mul(&ci, bits, rm), bits, rm);
        let im = si.mul(&cr, bits, rm).sub(&sr.mul(&ci, bits, rm), bits, rm);
        Ok((re.div(&denom, bits, rm), im.div(&denom, bits, rm)))
      }
      "Log" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // log(a+bi) = ln(|z|) + i*arg(z)
        // |z| = sqrt(a^2 + b^2), arg(z) = atan2(b, a)
        let abs_sq = a.mul(&a, bits, rm).add(&b.mul(&b, bits, rm), bits, rm);
        let abs_val = abs_sq.sqrt(bits, rm);
        let ln_abs = abs_val.ln(bits, rm, cc);
        // atan2(b, a) implemented using atan
        let arg = if a.is_zero() {
          let half_pi =
            cc.pi(bits, rm).div(&BigFloat::from_i32(2, bits), bits, rm);
          if b.is_negative() {
            half_pi.neg()
          } else {
            half_pi
          }
        } else if a.is_negative() {
          let atan_val = b.div(&a, bits, rm).atan(bits, rm, cc);
          if b.is_negative() {
            atan_val.sub(&cc.pi(bits, rm), bits, rm)
          } else {
            atan_val.add(&cc.pi(bits, rm), bits, rm)
          }
        } else {
          b.div(&a, bits, rm).atan(bits, rm, cc)
        };
        Ok((ln_abs, arg))
      }
      "Sqrt" if args.len() == 1 => {
        // sqrt(a+bi) = sqrt((|z|+a)/2) + i*sign(b)*sqrt((|z|-a)/2)
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        let abs_sq = a.mul(&a, bits, rm).add(&b.mul(&b, bits, rm), bits, rm);
        let abs_val = abs_sq.sqrt(bits, rm);
        let two = BigFloat::from_i32(2, bits);
        let re = abs_val.add(&a, bits, rm).div(&two, bits, rm).sqrt(bits, rm);
        let mut im =
          abs_val.sub(&a, bits, rm).div(&two, bits, rm).sqrt(bits, rm);
        if b.is_negative() {
          im = im.neg();
        }
        Ok((re, im))
      }
      "Rational" if args.len() == 2 => {
        let n = expr_to_bigfloat(&args[0], bits, rm, cc)?;
        let d = expr_to_bigfloat(&args[1], bits, rm, cc)?;
        Ok((n.div(&d, bits, rm), BigFloat::from_i32(0, bits)))
      }
      "Power" if args.len() == 2 => {
        let (zr, zi) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        let (wr, wi) = expr_to_complex_bigfloat(&args[1], bits, rm, cc)?;
        complex_bigfloat_pow(
          (&zr, &zi),
          (&wr, &wi),
          try_as_integer(&args[1]),
          bits,
          rm,
          cc,
        )
      }
      "Abs" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        let abs_sq = a.mul(&a, bits, rm).add(&b.mul(&b, bits, rm), bits, rm);
        Ok((abs_sq.sqrt(bits, rm), BigFloat::from_i32(0, bits)))
      }
      // Principal values through Log and Sqrt, whose branch cuts give
      // Wolfram's: ArcSin[3] = Pi/2 - 1.76… I, ArcCos[3] = 1.76… I,
      // ArcCosh[1/2] = 1.04… I.
      "ArcSin" | "ArcCos" | "ArcCosh" if args.len() == 1 => {
        let z = args[0].clone();
        let i = || Expr::Identifier("I".to_string());
        let sqrt = |e: Expr| Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![e].into(),
        };
        let log = |e: Expr| Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![e].into(),
        };
        let arcsin = |z: Expr| {
          times(vec![
            Expr::Integer(-1),
            i(),
            log(plus(vec![
              times(vec![i(), z.clone()]),
              sqrt(plus(vec![
                Expr::Integer(1),
                times(vec![
                  Expr::Integer(-1),
                  crate::helpers::pow(z, Expr::Integer(2)),
                ]),
              ])),
            ])),
          ])
        };
        let rewritten = match name.as_str() {
          "ArcSin" => arcsin(z),
          "ArcCos" => plus(vec![
            times(vec![
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
              },
              Expr::Constant("Pi".to_string()),
            ]),
            times(vec![Expr::Integer(-1), arcsin(z)]),
          ]),
          _ => log(plus(vec![
            z.clone(),
            times(vec![
              sqrt(plus(vec![z.clone(), Expr::Integer(1)])),
              sqrt(plus(vec![z, Expr::Integer(-1)])),
            ]),
          ])),
        };
        expr_to_complex_bigfloat(&rewritten, bits, rm, cc)
      }
      "ExpIntegralEi" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        Ok(complex_exp_integral_ei(&a, &b, bits, rm, cc))
      }
      "ExpIntegralE" | "LogIntegral" | "Gamma" => {
        exp_integral_family_bigfloat(name, args, bits, rm, cc)
      }
      _ => Err(InterpreterError::EvaluationError(format!(
        "N: cannot evaluate {name}[...] to complex arbitrary precision"
      ))),
    },
    _ => Err(InterpreterError::EvaluationError(format!(
      "N: cannot evaluate expression to complex arbitrary precision: {}",
      expr_to_string(expr)
    ))),
  }
}

/// Build a properly formatted complex result from BigFloat real and imaginary parts.
fn build_complex_bigfloat_result(
  re: &astro_float::BigFloat,
  im: &astro_float::BigFloat,
  precision: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  if im.is_zero() {
    let re_str = bigfloat_to_string(re, None, rm, cc)?;
    return Ok(Expr::BigFloat(re_str, precision as f64));
  }
  let p = precision as f64;
  complex_bigfloat_expr(re, im, p, p, rm, cc)
}

/// Build a complex result with per-component precision markers.
fn build_complex_result_with_string_precision(
  re: &astro_float::BigFloat,
  im: &astro_float::BigFloat,
  prec_re_str: &str,
  prec_im_str: &str,
  precision: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  let parse = |s: &str| s.parse::<f64>().unwrap_or(precision as f64);
  complex_bigfloat_expr(re, im, parse(prec_re_str), parse(prec_im_str), rm, cc)
}

/// The canonical evaluated form of an arbitrary-precision complex number,
/// `Plus[re, Times[im, I]]` (just `Times[im, I]` for a zero real part), the
/// same tree evaluating `1.5`30 + 2.`30 I` builds — so arithmetic on an `N`
/// result sees numbers, not an unevaluated operator form.
fn complex_bigfloat_expr(
  re: &astro_float::BigFloat,
  im: &astro_float::BigFloat,
  prec_re: f64,
  prec_im: f64,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  let im_term = times(vec![
    Expr::BigFloat(bigfloat_to_string(im, None, rm, cc)?, prec_im),
    id_expr("I"),
  ]);
  if re.is_zero() {
    return Ok(im_term);
  }
  let re_bf = Expr::BigFloat(bigfloat_to_string(re, None, rm, cc)?, prec_re);
  Ok(plus(vec![re_bf, im_term]))
}

/// Compute per-component precision markers for a complex function evaluation.
///
/// Uses the formula:
///   accuracy = p + log10(|input|) - log10(|output|)
///   precision_component = accuracy + log10(|component|)
///
/// Computes using BigFloat arithmetic for accuracy, then converts to f64.
fn compute_complex_precision_markers(
  in_re: &astro_float::BigFloat,
  in_im: &astro_float::BigFloat,
  out_re: &astro_float::BigFloat,
  out_im: &astro_float::BigFloat,
  precision: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(String, String), InterpreterError> {
  use astro_float::BigFloat;
  let pbits = 256; // high precision for the log computations

  // |input| = sqrt(in_re^2 + in_im^2)
  let in_abs_sq =
    in_re
      .mul(in_re, pbits, rm)
      .add(&in_im.mul(in_im, pbits, rm), pbits, rm);
  let in_abs = in_abs_sq.sqrt(pbits, rm);

  // |output| = sqrt(out_re^2 + out_im^2)
  let out_abs_sq = out_re.mul(out_re, pbits, rm).add(
    &out_im.mul(out_im, pbits, rm),
    pbits,
    rm,
  );
  let out_abs = out_abs_sq.sqrt(pbits, rm);

  // log10(x) = ln(x) / ln(10)
  let ln10 = BigFloat::from_i32(10, pbits).ln(pbits, rm, cc);
  let log10_in = in_abs.ln(pbits, rm, cc).div(&ln10, pbits, rm);
  let log10_out = out_abs.ln(pbits, rm, cc).div(&ln10, pbits, rm);
  let log10_re = out_re.abs().ln(pbits, rm, cc).div(&ln10, pbits, rm);
  let log10_im = out_im.abs().ln(pbits, rm, cc).div(&ln10, pbits, rm);

  let p_bf = BigFloat::from_i32(precision as i32, pbits);
  // accuracy = p + log10(|input|) - log10(|output|)
  let accuracy = p_bf.add(&log10_in, pbits, rm).sub(&log10_out, pbits, rm);

  // precision_re = accuracy + log10(|re|)
  let prec_re_bf = accuracy.add(&log10_re, pbits, rm);
  // precision_im = accuracy + log10(|im|)
  let prec_im_bf = accuracy.add(&log10_im, pbits, rm);

  // The rule models a function whose output scales like its input (Sin,
  // Exp near 1 + I). Where it does not (`Log` of a number near 0 has a
  // large output from a tiny input) it would claim a precision far from the
  // requested one, even a negative one; the caller then marks both parts
  // with the requested precision instead.
  let leading = bigfloat_to_f64_approx(&prec_re_bf)
    .zip(bigfloat_to_f64_approx(&prec_im_bf))
    .map(|(a, b)| a.max(b));
  if !leading.is_some_and(|l| (l - precision as f64).abs() <= 1.0) {
    return Err(InterpreterError::EvaluationError(
      "N: precision markers do not apply".into(),
    ));
  }

  // Format precision markers directly from BigFloat with f64-like precision
  // This avoids f64 rounding issues by going from BigFloat → string directly
  let prec_re_s = format_bigfloat_as_precision_marker(&prec_re_bf, rm, cc)?;
  let prec_im_s = format_bigfloat_as_precision_marker(&prec_im_bf, rm, cc)?;

  Ok((prec_re_s, prec_im_s))
}

/// Format a BigFloat as a precision marker string, matching Wolfram's formatting.
/// Wolfram displays precision markers with ~15-17 significant digits (f64-like precision),
/// stripping trailing zeros.
fn format_bigfloat_as_precision_marker(
  bf: &astro_float::BigFloat,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<String, InterpreterError> {
  // Format with ~17 significant digits (matching f64 precision)
  let full_str = bigfloat_to_string(bf, Some(17), rm, cc)?;

  // Parse to f64 and back to get Wolfram-style formatting
  // Wolfram uses a formatting that shows all distinguishing digits
  let val: f64 = full_str.trim_end_matches('.').parse().unwrap_or(0.0);

  if val.fract() == 0.0 {
    Ok(format!("{}.", val as i64))
  } else {
    // Use Rust's Debug format which shows the exact f64 representation
    // with minimum digits to uniquely represent the value
    Ok(format!("{val:?}"))
  }
}

/// Rescale[x, {xmin, xmax}] - rescales x to [0,1]
/// Rescale[x, {xmin, xmax}, {ymin, ymax}] - rescales x to [ymin,ymax]
/// Rescale[list] - rescales list elements to [0,1] based on min/max
/// Recursively collect numeric leaves of a (possibly nested) list for the
/// 1-argument `Rescale`. Sets `ok = false` on any non-numeric, non-list leaf
/// and clears `all_int` when a `Real` leaf is seen.
fn rescale_collect_leaves(
  expr: &Expr,
  vals: &mut Vec<f64>,
  int_vals: &mut Vec<i128>,
  all_int: &mut bool,
  ok: &mut bool,
) {
  match expr {
    Expr::List(items) => {
      for it in items {
        rescale_collect_leaves(it, vals, int_vals, all_int, ok);
      }
    }
    Expr::Integer(n) => {
      vals.push(*n as f64);
      int_vals.push(*n);
    }
    Expr::Real(f) => {
      vals.push(*f);
      *all_int = false;
    }
    _ => *ok = false,
  }
}

/// Rebuild a (possibly nested) list, rescaling each numeric leaf to [0, 1]
/// using the global `min`/range. Integer data stays exact via `make_rational`;
/// degenerate (zero-range) data maps every leaf to 0, matching wolframscript.
fn rescale_rebuild(
  expr: &Expr,
  all_int: bool,
  min_i: i128,
  range_i: i128,
  min_f: f64,
  range_f: f64,
  degenerate: bool,
) -> Expr {
  match expr {
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|it| {
          rescale_rebuild(
            it, all_int, min_i, range_i, min_f, range_f, degenerate,
          )
        })
        .collect(),
    ),
    // Degenerate (zero-range) data maps every leaf to 0. An inexact list
    // gives the machine real 0., an exact list the Integer 0.
    _ if degenerate => {
      if all_int {
        Expr::Integer(0)
      } else {
        Expr::Real(0.0)
      }
    }
    Expr::Integer(n) if all_int => make_rational(n - min_i, range_i),
    _ => {
      // Reached only for an inexact list (all_int is false), so the result
      // stays a Real even at a whole-number value (Rescale[{1., 2., 3.}] =
      // {0., 0.5, 1.}, not {0, 0.5, 1}). num_to_expr would collapse a whole
      // number to an exact Integer, discarding the inexactness.
      let x = match expr {
        Expr::Integer(n) => *n as f64,
        Expr::Real(f) => *f,
        _ => 0.0,
      };
      Expr::Real((x - min_f) / range_f)
    }
  }
}

/// Collect the numeric leaves of a nested list together with their values;
/// `None` when a leaf has no numeric value.
fn collect_numeric_leaves<'a>(
  expr: &'a Expr,
  out: &mut Vec<(&'a Expr, f64)>,
) -> Option<()> {
  match expr {
    Expr::List(items) => {
      for it in items {
        collect_numeric_leaves(it, out)?;
      }
      Some(())
    }
    leaf => {
      out.push((leaf, try_eval_to_f64(leaf)?));
      Some(())
    }
  }
}

/// Rescale[list] for exact numeric constants such as `Log[3]` or `Pi`:
/// each leaf `x` becomes `(x - min)/(max - min)`, kept symbolic.
fn rescale_symbolic_numeric_list(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let mut leaves = Vec::new();
  if collect_numeric_leaves(&args[0], &mut leaves).is_none()
    || leaves.is_empty()
  {
    return Ok(unevaluated("Rescale", args));
  }
  let (mut min_e, mut min_v) = leaves[0];
  let (mut max_e, mut max_v) = leaves[0];
  for &(e, v) in &leaves[1..] {
    if v < min_v {
      (min_e, min_v) = (e, v);
    }
    if v > max_v {
      (max_e, max_v) = (e, v);
    }
  }
  if (max_v - min_v).abs() < f64::EPSILON {
    return Ok(unevaluated("Rescale", args));
  }
  let span = plus(vec![max_e.clone(), neg(min_e.clone())]);
  fn rebuild(
    expr: &Expr,
    min_e: &Expr,
    span: &Expr,
  ) -> Result<Expr, InterpreterError> {
    match expr {
      Expr::List(items) => Ok(Expr::List(
        items
          .iter()
          .map(|it| rebuild(it, min_e, span))
          .collect::<Result<Vec<_>, _>>()?
          .into(),
      )),
      leaf => crate::evaluator::evaluate_expr_to_expr(&times(vec![
        plus(vec![leaf.clone(), neg(min_e.clone())]),
        pow(span.clone(), Expr::Integer(-1)),
      ])),
    }
  }
  rebuild(&args[0], min_e, &span)
}

/// Rescale with symbolic (or complex) bounds, in the forms wolframscript
/// gives: `Rescale[x, {a, b}]` is `(x - a)/(b - a)` (so `Rescale[0, {a, b}]`
/// is `-(a/(-a + b))` and `Rescale[a, {b, a}]` is 1), and
/// `Rescale[x, {a, b}, {c, d}]` is the line
/// `-(a d - b c)/(b - a) + (d - c) x/(b - a)`.
fn rescale_symbolic(
  x: &Expr,
  a: &Expr,
  b: &Expr,
  target: Option<(&Expr, &Expr)>,
) -> Result<Expr, InterpreterError> {
  let inv_span = pow(plus(vec![b.clone(), neg(a.clone())]), Expr::Integer(-1));
  let Some((c, d)) = target else {
    return crate::evaluator::evaluate_expr_to_expr(&times(vec![
      plus(vec![x.clone(), neg(a.clone())]),
      inv_span,
    ]));
  };
  let constant = times(vec![
    Expr::Integer(-1),
    plus(vec![
      times(vec![a.clone(), d.clone()]),
      neg(times(vec![b.clone(), c.clone()])),
    ]),
    inv_span.clone(),
  ]);
  let slope_term = times(vec![
    plus(vec![d.clone(), neg(c.clone())]),
    x.clone(),
    inv_span,
  ]);
  crate::evaluator::evaluate_expr_to_expr(&plus(vec![constant, slope_term]))
}

pub fn rescale_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Rescale expects 1 to 3 arguments".into(),
    ));
  }

  // Rescale[list] - auto-detect the global min/max across all (possibly
  // nested) elements and rescale each, preserving the list structure.
  if args.len() == 1 {
    if let Expr::List(items) = &args[0] {
      if items.is_empty() {
        return Ok(Expr::List(vec![].into()));
      }
      let mut vals = Vec::new();
      let mut int_vals: Vec<i128> = Vec::new();
      let mut all_int = true;
      let mut ok = true;
      rescale_collect_leaves(
        &args[0],
        &mut vals,
        &mut int_vals,
        &mut all_int,
        &mut ok,
      );
      if !ok {
        return rescale_symbolic_numeric_list(args);
      }
      if vals.is_empty() {
        return Ok(unevaluated("Rescale", args));
      }
      let min_f = vals.iter().copied().fold(f64::INFINITY, f64::min);
      let max_f = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
      let degenerate = (max_f - min_f).abs() < f64::EPSILON;
      let (min_i, range_i) = if all_int {
        let mn = *int_vals.iter().min().unwrap();
        let mx = *int_vals.iter().max().unwrap();
        (mn, mx - mn)
      } else {
        (0, 0)
      };
      return Ok(rescale_rebuild(
        &args[0],
        all_int,
        min_i,
        range_i,
        min_f,
        max_f - min_f,
        degenerate,
      ));
    }
    // Single non-list value needs {xmin, xmax}
    return Ok(unevaluated("Rescale", args));
  }

  // Handle list first argument: Rescale[{x1, x2, ...}, range] maps over elements
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|item| {
        let mut new_args = args.to_vec();
        new_args[0] = item.clone();
        rescale_ast(&new_args)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  // Rescale[x, {xmin, xmax}] or Rescale[x, {xmin, xmax}, {ymin, ymax}]
  let range = match &args[1] {
    Expr::List(r) if r.len() == 2 => r,
    _ => {
      return Ok(unevaluated("Rescale", args));
    }
  };

  let (ymin, ymax) = if args.len() == 3 {
    match &args[2] {
      Expr::List(r) if r.len() == 2 => (&r[0], &r[1]),
      _ => {
        return Ok(unevaluated("Rescale", args));
      }
    }
  } else {
    (&Expr::Integer(0) as &Expr, &Expr::Integer(1) as &Expr)
  };

  let unevaluated = || Ok(unevaluated("Rescale", args));

  // A literal exact number: Integer, BigInteger, Real or Rational[p, q].
  let is_numeric_literal = |e: &Expr| -> bool {
    matches!(e, Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_))
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2)
  };

  // Symbolic (or complex) bounds/targets ({a, b}) give the line form of
  // `rescale_symbolic`. Otherwise the bounds are real numbers. Exact
  // numeric constants (`Log[3]`, as in a weighting of `Log[n/df]` values)
  // are numbers too, as long as the value being rescaled is numeric as well.
  let bounds = [&range[0], &range[1], ymin, ymax];
  if bounds.iter().any(|b| try_eval_to_f64(b).is_none()) {
    return rescale_symbolic(
      &args[0],
      &range[0],
      &range[1],
      (args.len() == 3).then_some((ymin, ymax)),
    );
  }
  let literal_bounds = bounds.iter().all(|b| is_numeric_literal(b));
  if !literal_bounds && try_eval_to_f64(&args[0]).is_none() {
    return unevaluated();
  }
  let xmin_n = try_eval_to_f64(&range[0]).unwrap();
  let xmax_n = try_eval_to_f64(&range[1]).unwrap();
  if (xmax_n - xmin_n).abs() < f64::EPSILON {
    return unevaluated();
  }
  // For a non-literal x (a symbol or constant like Pi), only xmin == 0 is
  // safe: with a nonzero xmin wolframscript distributes the offset
  // (10 + 5 (x - 2) -> 5 x) in a way Woxi's evaluator leaves factored.
  if literal_bounds
    && !is_numeric_literal(&args[0])
    && xmin_n.abs() >= f64::EPSILON
  {
    return unevaluated();
  }

  // Build and evaluate  ymin + (x - xmin) * (ymax - ymin) / (xmax - xmin).
  // Evaluating the symbolic expression keeps exact and symbolic values
  // (Pi/10, 1/3, x/10, …) instead of floatifying, and a Real input still
  // yields a Real.
  let neg = |e: &Expr| neg(e.clone());
  let sub = |a: &Expr, b: &Expr| plus(vec![a.clone(), neg(b)]);
  let x_minus_min = sub(&args[0], &range[0]);
  let y_span = sub(ymax, ymin);
  let x_span = sub(&range[1], &range[0]);
  let fraction =
    times(vec![x_minus_min, y_span, pow(x_span, Expr::Integer(-1))]);
  let result = plus(vec![ymin.clone(), fraction]);
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Normalize[v] - normalizes a vector to unit length
/// Normalize[{3, 4}] => {3/5, 4/5}
/// Normalize[{0, 0, 0}] => {0, 0, 0}
/// Norm[v] - Euclidean norm (L2) of a vector
/// Norm[v, p] - Lp norm
pub fn norm_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Norm expects 1 or 2 arguments".into(),
    ));
  }
  // A SparseArray argument is handled via its dense form.
  if let Some(dense) =
    crate::functions::list_helpers_ast::densify_sparse_array(&args[0])
  {
    let mut new_args = args.to_vec();
    new_args[0] = dense;
    return norm_ast(&new_args);
  }
  // An empty vector has no norm: emit Norm::nvm and stay unevaluated rather
  // than collapsing to 0. (wolframscript parity)
  if matches!(&args[0], Expr::List(items) if items.is_empty()) {
    crate::emit_message(
      "Norm::nvm: The first Norm argument should be a scalar, vector or matrix.",
    );
    return Ok(unevaluated("Norm", args));
  }
  // Determine the norm parameter p
  let p_expr = if args.len() == 2 {
    Some(args[1].clone())
  } else {
    None
  };
  let p_val = match &p_expr {
    Some(e) => try_eval_to_f64(e),
    None => Some(2.0),
  };
  // Check for Infinity norm
  let is_infinity = match &p_expr {
    Some(Expr::Identifier(s)) if s == "Infinity" => true,
    Some(Expr::FunctionCall { name, args })
      if name == "DirectedInfinity" && args.len() == 1 =>
    {
      true
    }
    _ => p_val == Some(f64::INFINITY),
  };

  // Emit Norm::ptype and leave the call unevaluated for an out-of-range norm
  // parameter (a numeric p < 1 for a vector, or a non-{1,2,Infinity,
  // "Frobenius"} value for a matrix).
  let ptype = || {
    if let Some(p) = &p_expr {
      crate::emit_message(&format!(
        "Norm::ptype: The second argument of Norm, {}, should be a symbol, Infinity or an integer or real number not less than 1 for vector p-norms; or 1, 2, Infinity or \"Frobenius\" for matrix norms.",
        expr_to_string(p)
      ));
    }
    Ok(unevaluated("Norm", args))
  };

  // Norm[matrix, "Frobenius"] — sqrt of sum of squared absolute values
  // across every entry of the (rectangular) matrix.
  if matches!(&p_expr, Some(Expr::String(s)) if s == "Frobenius")
    && let Expr::List(rows) = &args[0]
    && rows.iter().all(|r| matches!(r, Expr::List(_)))
  {
    let mut flat: Vec<Expr> = Vec::new();
    for row in rows {
      if let Expr::List(cells) = row {
        flat.extend(cells.iter().cloned());
      }
    }
    // Unlike the machine-precision vector 2-norm (which Wolfram computes
    // with the scaled formula, see below), Frobenius sums the plain
    // squares: Norm[{{1.5,2.5},{3.5,4.5}}, "Frobenius"] → Sqrt[41.] =
    // 6.4031242374328485, not the scaled 6.403124237432849.
    if flat.iter().any(contains_inexact_real) {
      let vals: Option<Vec<f64>> = flat.iter().map(try_eval_to_f64).collect();
      if let Some(vals) = vals {
        let sum: f64 = vals.iter().map(|x| x * x).sum();
        return Ok(Expr::Real(sum.sqrt()));
      }
    }
    return norm_ast(&[Expr::List(flat.into())]);
  }

  // For a vector or scalar the Frobenius norm is just the ordinary 2-norm
  // (Abs for a scalar); only the matrix case above needs special handling.
  if matches!(&p_expr, Some(Expr::String(s)) if s == "Frobenius") {
    return norm_ast(std::slice::from_ref(&args[0]));
  }

  // Matrix norms: args[0] is a rectangular list of lists. These differ from
  // the element-wise vector norms below:
  //   p = 1        → maximum absolute column sum
  //   p = Infinity → maximum absolute row sum
  //   p = 2        → spectral norm (largest singular value)
  //                  = Sqrt[Max[Eigenvalues[Transpose[A].A]]]
  if let Expr::List(rows) = &args[0]
    && !rows.is_empty()
    && rows.iter().all(|r| matches!(r, Expr::List(_)))
  {
    let mat: Vec<Vec<Expr>> = rows
      .iter()
      .map(|r| match r {
        Expr::List(c) => c.to_vec(),
        _ => unreachable!(),
      })
      .collect();
    // A matrix has scalar entries. If any entry is itself a list the argument
    // is a rank >= 3 tensor, which Norm does not accept: emit Norm::nvm and
    // stay unevaluated (matching wolframscript) rather than threading the
    // matrix-norm formula over the deeper structure.
    if mat
      .iter()
      .any(|row| row.iter().any(|c| matches!(c, Expr::List(_))))
    {
      crate::emit_message(
        "Norm::nvm: The first Norm argument should be a scalar, vector or matrix.",
      );
      return Ok(unevaluated("Norm", args));
    }
    let ncols = mat[0].len();
    if ncols > 0 && mat.iter().all(|r| r.len() == ncols) {
      use crate::evaluator::evaluate_expr_to_expr;
      let abs = |e: &Expr| call1("Abs", e.clone());
      let sum = |terms: Vec<Expr>| plus(terms);
      let max_of = |terms: Vec<Expr>| call("Max", terms);

      if is_infinity {
        // Maximum absolute row sum.
        let row_sums: Vec<Expr> = mat
          .iter()
          .map(|row| sum(row.iter().map(&abs).collect()))
          .collect();
        return evaluate_expr_to_expr(&max_of(row_sums));
      }
      if p_val == Some(1.0) {
        // Maximum absolute column sum.
        let col_sums: Vec<Expr> = (0..ncols)
          .map(|j| sum(mat.iter().map(|row| abs(&row[j])).collect()))
          .collect();
        return evaluate_expr_to_expr(&max_of(col_sums));
      }
      if p_val == Some(2.0) {
        // Spectral norm: Sqrt[Max[Eigenvalues[Transpose[A].A]]].
        let a = args[0].clone();
        let at = call1("Transpose", a.clone());
        let ata = call("Dot", vec![at, a]);
        let eig = call1("Eigenvalues", ata);
        let sqrt = call1("Sqrt", max_of(vec![eig]));
        return evaluate_expr_to_expr(&sqrt);
      }
      // Only 1, 2, Infinity and "Frobenius" are valid matrix norms; anything
      // else emits ptype and stays unevaluated.
      return ptype();
    }
  }

  // A vector (or scalar) p-norm requires a numeric p not less than 1; a
  // numeric p < 1 is rejected with ptype (symbolic p stays symbolic).
  if let Some(p) = p_val
    && p.is_finite()
    && p < 1.0
  {
    return ptype();
  }

  match &args[0] {
    Expr::List(items) => {
      let p = p_val.unwrap_or(2.0);

      // "Inexact" mode: any item contains a Real/BigFloat leaf — collapse
      // to a machine-precision numeric result, mirroring Wolfram's
      // behavior (Norm[{1.0, 2, 3}] → 3.741…).
      // Otherwise stay in "exact" mode and build a symbolic expression.
      // A vector that also holds symbolic entries (Norm[{-2., y}]) cannot
      // collapse to a number and takes the symbolic path below, keeping its
      // inexact entries inexact.
      let inexact = items.iter().any(contains_inexact_real)
        && items.iter().all(|item| try_eval_to_f64(item).is_some());

      if inexact {
        let mut vals = Vec::with_capacity(items.len());
        for item in items {
          match try_eval_to_f64(item) {
            Some(v) => vals.push(v),
            None => {
              return Ok(unevaluated("Norm", args));
            }
          }
        }
        // An inexact vector gives an inexact norm even at a whole-number value
        // (Norm[{3., 4.}] = 5., not 5), so return Real rather than num_to_expr
        // (which would collapse a whole number to an exact Integer).
        if p == 1.0 {
          let result: f64 = vals.iter().map(|x| x.abs()).sum();
          return Ok(Expr::Real(result));
        }
        if is_infinity {
          let result = vals.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
          return Ok(Expr::Real(result));
        }
        if p == 2.0 {
          // Wolfram computes the machine-precision 2-norm with the
          // overflow-safe scaled formula m*Sqrt[Sum[(x/m)^2]], m = Max[Abs],
          // whose rounding differs from Sqrt[Sum[x^2]] by an ulp
          // (Norm[{10.9, -10.3}] → 14.996666296213968).
          let m = vals.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
          if m == 0.0 {
            return Ok(Expr::Real(0.0));
          }
          let sum: f64 = vals
            .iter()
            .map(|x| {
              let r = x / m;
              r * r
            })
            .sum();
          return Ok(Expr::Real(m * sum.sqrt()));
        }
        let sum: f64 = vals.iter().map(|x| x.abs().powf(p)).sum();
        return Ok(Expr::Real(sum.powf(1.0 / p)));
      }

      // Exact/symbolic mode.
      use crate::evaluator::evaluate_expr_to_expr;

      // For each item decide whether to wrap in Abs: if the item is a
      // numerically-evaluable (hence known-real) expression — integers,
      // rationals, Pi, Sin[1], 2 Sin[2], … — drop the Abs and build
      // item^p directly so that known scalars combine. For unknown
      // symbols (x, f[x]) keep Abs to preserve correctness over ℂ.
      let is_real_valued =
        |item: &Expr| -> bool { try_eval_to_f64(item).is_some() };

      if is_infinity {
        // Max[Abs[x], Abs[y], ...]
        let abs_items: Vec<Expr> = items
          .iter()
          .map(|item| call1("Abs", item.clone()))
          .collect();
        return evaluate_expr_to_expr(&call("Max", abs_items));
      }

      // Match against `p_val`/`p_expr` (not the defaulted `p`) so a symbolic p
      // falls through to the general p-norm instead of the 2-norm.
      if p_val == Some(1.0) {
        // Sum of Abs[item]
        let terms: Vec<Expr> = items
          .iter()
          .map(|item| call1("Abs", item.clone()))
          .collect();
        let sum = if terms.len() == 1 {
          terms.into_iter().next().unwrap()
        } else {
          plus(terms)
        };
        return evaluate_expr_to_expr(&sum);
      }

      if p_expr.is_none() || p_val == Some(2.0) {
        // A one-element vector IS its Abs: Norm[{1 + Pi}] → 1 + Pi and
        // Norm[{x}] → Abs[x], never Sqrt[(1 + Pi)^2]. (wolframscript
        // parity; differential fuzzer, seed 1783520505113402110)
        if items.len() == 1 {
          return crate::evaluator::evaluate_function_call_ast(
            "Abs",
            &[items[0].clone()],
          );
        }
        // Sqrt[Plus[item^2, ...]] (or Abs[item]^2 for unknown items).
        // Wolfram squares Abs[item], so a numerically negative exact item
        // flips sign first: Norm[{-46, -63, 1 - 81 + Pi}] carries
        // (80 - Pi)^2, not (-80 + Pi)^2.
        let sq_items: Vec<Expr> = items
          .iter()
          .map(|item| {
            let base = if is_real_valued(item) {
              if try_eval_to_f64(item).is_some_and(|v| v < 0.0) {
                evaluate_expr_to_expr(&neg(item.clone()))
                  .unwrap_or_else(|_| item.clone())
              } else {
                item.clone()
              }
            } else {
              call1("Abs", item.clone())
            };
            pow(base, Expr::Integer(2))
          })
          .collect();
        let sum = if sq_items.len() == 1 {
          sq_items.into_iter().next().unwrap()
        } else {
          plus(sq_items)
        };
        let sum_eval = evaluate_expr_to_expr(&sum)?;
        return evaluate_expr_to_expr(&make_sqrt(sum_eval));
      }

      // General p-norm (p is a number other than 1/2/Infinity, or symbolic):
      // (Sum Abs[item]^p)^(1/p). Abs of a known real evaluates away, so
      // numeric inputs collapse (Norm[{3, 4}, 4] -> 337^(1/4)).
      let p_term = p_expr.clone().unwrap_or(Expr::Integer(2));
      let power = |base: Expr, exp: Expr| pow(base, exp);
      let terms: Vec<Expr> = items
        .iter()
        .map(|item| power(call1("Abs", item.clone()), p_term.clone()))
        .collect();
      let sum = if terms.len() == 1 {
        terms.into_iter().next().unwrap()
      } else {
        plus(terms)
      };
      let result = power(sum, power(p_term, Expr::Integer(-1)));
      evaluate_expr_to_expr(&result)
    }
    // Norm of a scalar. For a number (real or complex) Norm[x] = Abs[x],
    // evaluated symbolically so exact values are preserved (Norm[Pi] -> Pi,
    // Norm[2/3] -> 2/3, Norm[3 + 4 I] -> 5). For a non-numeric scalar (x,
    // a + b I, …) wolframscript leaves Norm unevaluated.
    _ => {
      if crate::functions::predicate_ast::is_numeric_q(&args[0]) {
        crate::evaluator::evaluate_function_call_ast("Abs", &[args[0].clone()])
      } else {
        Ok(unevaluated("Norm", args))
      }
    }
  }
}

pub fn normalize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 2 {
    // Normalize[v, f] — divides v by f[v]
    let norm_val =
      crate::functions::list_helpers_ast::apply_func_ast(&args[1], &args[0])?;
    // v / norm_val
    let result = crate::evaluator::evaluate_expr_to_expr(&div2(
      args[0].clone(),
      norm_val,
    ))?;
    return Ok(result);
  }
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Normalize expects 1 or 2 arguments".into(),
    ));
  }
  if let Expr::List(items) = &args[0] {
    if items.is_empty() {
      return Ok(Expr::List(vec![].into()));
    }
    // Compute the Euclidean norm
    let mut vals = Vec::new();
    let mut all_int = true;
    let mut int_vals: Vec<i128> = Vec::new();
    for item in items {
      match item {
        Expr::Integer(n) => {
          vals.push(*n as f64);
          int_vals.push(*n);
        }
        Expr::Real(f) => {
          vals.push(*f);
          all_int = false;
        }
        _ => {
          // Symbolic case: return {elem/Sqrt[sum_of_squares], ...}
          // like Mathematica does for Normalize[{a, b}] → {a/Sqrt[a^2+b^2], b/Sqrt[a^2+b^2]}
          let squared_terms: Vec<Expr> = items
            .iter()
            .map(|e| pow2(call1("Abs", e.clone()), Expr::Integer(2)))
            .collect();
          let sum_of_squares = if squared_terms.len() == 1 {
            squared_terms.into_iter().next().unwrap()
          } else {
            plus(squared_terms)
          };
          // Evaluate the norm so numeric entries collapse the Abs-of-square
          // sum (e.g. Normalize[{1, I}] -> {1/Sqrt[2], I/Sqrt[2]} rather than
          // leaving Sqrt[Abs[1]^2 + Abs[I]^2]); a fully symbolic vector keeps
          // the Abs form unchanged.
          let norm_expr = crate::evaluator::evaluate_expr_to_expr(&make_sqrt(
            sum_of_squares,
          ))?;
          let result: Vec<Expr> = items
            .iter()
            .map(|e| {
              crate::evaluator::evaluate_expr_to_expr(&div2(
                e.clone(),
                norm_expr.clone(),
              ))
            })
            .collect::<Result<Vec<_>, _>>()?;
          return Ok(Expr::List(result.into()));
        }
      }
    }
    let norm_sq: f64 = vals.iter().map(|x| x * x).sum();
    if norm_sq == 0.0 {
      return Ok(args[0].clone());
    }
    let norm = norm_sq.sqrt();

    if all_int {
      // Exact: each element / Sqrt[sum_sq], handed to the evaluator rather
      // than built literally. It owns both reductions this needs — a
      // perfect-square norm (`{3, 4}` → `{3/5, 4/5}`) and the radical
      // canonicalisation wolframscript applies component-wise
      // (`3/Sqrt[18]` → `1/Sqrt[2]`, `5/Sqrt[70]` → `Sqrt[5/14]`). Built
      // literally the components came out as `3/Sqrt[18]`.
      let sum_sq: i128 = int_vals.iter().map(|x| x * x).sum();
      let norm_expr = make_sqrt(Expr::Integer(sum_sq));
      let result: Vec<Expr> = int_vals
        .iter()
        .map(|x| {
          crate::evaluator::evaluate_expr_to_expr(&div2(
            Expr::Integer(*x),
            norm_expr.clone(),
          ))
        })
        .collect::<Result<Vec<_>, _>>()?;
      return Ok(Expr::List(result.into()));
    }

    // Float path. Wolfram scales the vector as `v/norm`, which is a
    // multiply by the reciprocal — the last bit of each component follows
    // from that order, so dividing instead moves it (`Normalize[{3., 4.}]`
    // is `{0.6000000000000001, 0.8}`, not `{0.6, 0.8}`). A component that
    // lands on a whole number stays a machine real.
    let inv = 1.0 / norm;
    let result: Vec<Expr> = vals.iter().map(|x| Expr::Real(x * inv)).collect();
    Ok(Expr::List(result.into()))
  } else {
    // Scalar: Normalize[x] = x / Norm[x]
    let norm_val = norm_ast(args)?;
    // If norm is 0, return the original
    let is_zero = match &norm_val {
      Expr::Integer(0) => true,
      Expr::Real(f) if *f == 0.0 => true,
      _ => false,
    };
    if is_zero {
      return Ok(args[0].clone());
    }
    crate::evaluator::evaluate_function_call_ast(
      "Divide",
      &[args[0].clone(), norm_val],
    )
  }
}

/// Unitize[x] - returns 0 for 0, 1 for anything else
/// Unitize[x, dx] - returns 0 when |x| < dx, 1 otherwise
/// Unitize[list] - maps over lists
pub fn unitize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Unitize expects 1 or 2 arguments".into(),
    ));
  }

  // Two-argument form: Unitize[x, dx]
  if args.len() == 2 {
    // Thread over lists in the first argument.
    if let Expr::List(items) = &args[0] {
      let results: Result<Vec<Expr>, InterpreterError> = items
        .iter()
        .map(|x| unitize_ast(&[x.clone(), args[1].clone()]))
        .collect();
      return Ok(Expr::List(results?.into()));
    }

    let x_val = try_eval_to_f64(&args[0]);
    let tol_val = try_eval_to_f64(&args[1]);
    if let (Some(x), Some(tol)) = (x_val, tol_val) {
      if x.abs() < tol {
        return Ok(Expr::Integer(0));
      }
      return Ok(Expr::Integer(1));
    }

    // Non-numeric arguments remain unevaluated.
    return Ok(unevaluated("Unitize", args));
  }

  match &args[0] {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(_) => Ok(Expr::Integer(1)),
    Expr::Real(f) if *f == 0.0 => Ok(Expr::Integer(0)),
    Expr::Real(_) => Ok(Expr::Integer(1)),
    Expr::List(items) => {
      let results: Result<Vec<Expr>, InterpreterError> = items
        .iter()
        .map(|x| unitize_ast(std::slice::from_ref(x)))
        .collect();
      Ok(Expr::List(results?.into()))
    }
    _ => {
      // Numeric-eval fallback: if the argument collapses to a finite
      // non-zero real, Unitize is 1 (covers Pi, E, EulerGamma,
      // Sqrt[2], 2*Pi, 3/4, Log[2], ...). Zero numeric results are
      // already handled because Pi - Pi etc. simplify symbolically
      // before reaching Unitize.
      if let Some(v) = try_eval_to_f64(&args[0])
        && v.is_finite()
        && v != 0.0
      {
        return Ok(Expr::Integer(1));
      }
      Ok(unevaluated("Unitize", args))
    }
  }
}

/// Precision[x] - the number of significant decimal digits
/// Returns MachinePrecision for machine reals, Infinity for exact numbers
pub fn precision_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Precision expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Constant(_) => {
      Ok(id_expr("Infinity"))
    }
    Expr::Real(_) => Ok(id_expr("MachinePrecision")),
    Expr::BigFloat(digits, prec) => {
      // A literal zero BigFloat (e.g. `0.`20`, `0.``3`) reports
      // precision 0 in Wolfram — there are no significant digits when
      // the value itself is zero, so precision = -Log10(|0|) is treated
      // as 0 rather than the spec'd precision.
      if crate::functions::math_ast::numeric_utils::bigfloat_digits_are_zero(
        digits,
      ) {
        Ok(Expr::Real(0.0))
      } else {
        Ok(Expr::Real(*prec))
      }
    }
    Expr::Identifier(name)
      if name == "Infinity" || name == "ComplexInfinity" =>
    {
      Ok(id_expr("Infinity"))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      ..
    } => {
      // Exact rationals like 1/2 have infinite precision
      Ok(id_expr("Infinity"))
    }
    Expr::List(items) => {
      // Precision of a list is the minimum precision of its elements,
      // with one Wolfram-specific quirk: when both MachinePrecision and
      // arbitrary-precision Reals appear in the same list, the whole
      // result is MachinePrecision (machine arithmetic infects the mix).
      let mp: f64 = 15.954589770191003;
      let mut min_prec: Option<f64> = None;
      let mut min_is_machine = false;
      let mut saw_machine = false;
      let mut saw_arb = false;
      for item in items {
        let p = precision_ast(std::slice::from_ref(item))?;
        match p {
          Expr::Identifier(ref name) if name == "Infinity" => {}
          Expr::Identifier(ref name) if name == "MachinePrecision" => {
            saw_machine = true;
            match min_prec {
              None => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              Some(v) if mp < v => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              _ => {}
            }
          }
          Expr::Real(f) => {
            saw_arb = true;
            match min_prec {
              None => {
                min_prec = Some(f);
                min_is_machine = false;
              }
              Some(v) if f < v => {
                min_prec = Some(f);
                min_is_machine = false;
              }
              _ => {}
            }
          }
          _ => {}
        }
      }
      if saw_machine && saw_arb {
        return Ok(id_expr("MachinePrecision"));
      }
      match min_prec {
        Some(_) if min_is_machine => Ok(id_expr("MachinePrecision")),
        Some(p) => Ok(Expr::Real(p)),
        None => Ok(id_expr("Infinity")),
      }
    }
    // For symbolic expressions, check if any subexpression has finite precision.
    // If the minimum comes from a machine-real element, return the symbol
    // MachinePrecision (matches wolframscript).
    Expr::FunctionCall { args: fargs, .. } => {
      let mp: f64 = 15.954589770191003;
      let mut min_prec: Option<f64> = None;
      let mut min_is_machine = false;
      for arg in fargs {
        let p = precision_ast(std::slice::from_ref(arg))?;
        match p {
          Expr::Identifier(ref name) if name == "Infinity" => {}
          Expr::Identifier(ref name) if name == "MachinePrecision" => {
            match min_prec {
              None => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              Some(v) if mp < v => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              _ => {}
            }
          }
          Expr::Real(f) => match min_prec {
            None => {
              min_prec = Some(f);
              min_is_machine = false;
            }
            Some(v) if f < v => {
              min_prec = Some(f);
              min_is_machine = false;
            }
            _ => {}
          },
          _ => {}
        }
      }
      match min_prec {
        Some(_) if min_is_machine => Ok(id_expr("MachinePrecision")),
        Some(p) => Ok(Expr::Real(p)),
        None => Ok(id_expr("Infinity")),
      }
    }
    _ => Ok(id_expr("Infinity")),
  }
}

/// Accuracy[x] - the number of significant decimal digits to the right of the decimal point
/// Returns Infinity for exact numbers, computes from precision for approximate numbers
pub fn accuracy_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Accuracy expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Constant(_) => {
      Ok(id_expr("Infinity"))
    }
    Expr::Real(f) => {
      // Accuracy = MachinePrecision - Log10[Abs[x]]
      // MachinePrecision ≈ 15.9546
      let machine_precision = 15.954589770191003_f64;
      if *f == 0.0 {
        // Accuracy[0.] is the negative log of the smallest distinguishable
        // machine number, i.e. MachinePrecision - Log10[$MinMachineNumber].
        // Wolfram uses 2^-1022 (smallest normalised double) here.
        let min_machine_number = f64::MIN_POSITIVE;
        return Ok(Expr::Real(machine_precision - min_machine_number.log10()));
      }
      let accuracy = machine_precision - f.abs().log10();
      Ok(Expr::Real(accuracy))
    }
    // Arbitrary-precision: Accuracy = Precision - Log10[Abs[x]]. The
    // BigFloat carries the precision in its own field; for x == 0 the
    // precision *is* the accuracy (Wolfram stores `0``a` as a 0 with
    // accuracy a, which we represent as a BigFloat whose precision field
    // already holds a — see the parser).
    Expr::BigFloat(digits, prec) => {
      let val: f64 = digits.parse().unwrap_or(0.0);
      let prec_f = *prec;
      if val == 0.0 {
        return Ok(Expr::Real(prec_f));
      }
      Ok(Expr::Real(prec_f - val.abs().log10()))
    }
    Expr::Identifier(name)
      if name == "Infinity"
        || name == "ComplexInfinity"
        || name == "Indeterminate" =>
    {
      Ok(id_expr("Infinity"))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      ..
    } => Ok(id_expr("Infinity")),
    // Complex number with finite-accuracy parts: apply Wolfram's formula
    // Accuracy[Complex[re, im]] = -Log10[Sqrt[10^(-2*Acc[re]) + 10^(-2*Acc[im])]].
    // Without this, `Accuracy[Complex[3.00``2, 4.00``2]]` would just take
    // min(2, 2) = 2 instead of the correct 1.8494…
    _ if try_complex_accuracy(&args[0]).is_some() => {
      Ok(try_complex_accuracy(&args[0]).unwrap())
    }
    // Compound arithmetic in BinaryOp form: take the minimum accuracy
    // across both operands (mirrors the FunctionCall Plus/Times path
    // below). Otherwise `Plus[BigFloat, Times[BigFloat, I]]` — the
    // canonical form of `Complex[3.00``2, 4.00``2]` — would fall through
    // to Infinity even though both operands have finite accuracy.
    Expr::BinaryOp {
      op:
        BinaryOperator::Plus
        | BinaryOperator::Minus
        | BinaryOperator::Times
        | BinaryOperator::Power,
      left,
      right,
    } => {
      let mut min_finite: Option<f64> = None;
      for arg in [left.as_ref(), right.as_ref()] {
        let a = accuracy_ast(std::slice::from_ref(arg))?;
        match a {
          Expr::Identifier(ref n) if n == "Infinity" => {}
          Expr::Real(v) => {
            min_finite = Some(min_finite.map_or(v, |m| m.min(v)));
          }
          _ => {
            return Ok(unevaluated("Accuracy", args));
          }
        }
      }
      Ok(match min_finite {
        Some(v) => Expr::Real(v),
        None => id_expr("Infinity"),
      })
    }
    // Symbolic identifiers (variables) have infinite accuracy
    Expr::Identifier(_) => Ok(id_expr("Infinity")),
    // For symbolic expressions and lists, take the minimum of the
    // children's accuracies. Wolfram: Accuracy[F[1.3, Pi, A]] → 15.840…,
    // picking the less-accurate Real over the exact Pi/A. Lists are
    // handled the same way: Accuracy[{1, 1.``5}] → 5.
    Expr::FunctionCall { args: fargs, .. } | Expr::List(fargs) => {
      let mut min_finite: Option<f64> = None;
      for arg in fargs {
        let a = accuracy_ast(std::slice::from_ref(arg))?;
        match a {
          Expr::Identifier(ref n) if n == "Infinity" => {}
          Expr::Real(v) => {
            min_finite = Some(min_finite.map_or(v, |m| m.min(v)));
          }
          _ => {
            // Some child still couldn't be reduced to a number — keep the
            // call symbolic rather than guess.
            return Ok(unevaluated("Accuracy", args));
          }
        }
      }
      Ok(match min_finite {
        Some(v) => Expr::Real(v),
        None => id_expr("Infinity"),
      })
    }
    _ => Ok(id_expr("Infinity")),
  }
}

/// Detect a complex number `Plus[real_part, Times[imag_part, I]]` (in
/// either FunctionCall or BinaryOp form) where both parts have finite
/// accuracy, and return Wolfram's combined accuracy:
///   Acc = -Log10[Sqrt[10^(-2*Acc[re]) + 10^(-2*Acc[im])]].
/// Returns None if the structure isn't a recognised real+imag pair, if
/// either part has infinite accuracy, or if there's no I factor — in
/// those cases the caller falls back to the generic min-of-children path.
fn try_complex_accuracy(expr: &Expr) -> Option<Expr> {
  let plus_args: Vec<Expr> = match expr {
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() == 2 => {
      args.to_vec()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => vec![*left.clone(), *right.clone()],
    _ => return None,
  };
  let split = |term: &Expr| -> Option<(bool, Expr)> {
    // Returns (is_imaginary, raw_part). For pure imaginary `I`, raw_part
    // is the implicit coefficient `1`.
    if matches!(term, Expr::Identifier(s) if s == "I") {
      return Some((true, Expr::Integer(1)));
    }
    let factors: Vec<&Expr> = match term {
      Expr::FunctionCall { name, args } if name == "Times" => {
        args.iter().collect()
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => vec![left.as_ref(), right.as_ref()],
      _ => return Some((false, term.clone())),
    };
    let i_count = factors
      .iter()
      .filter(|f| matches!(f, Expr::Identifier(s) if s == "I"))
      .count();
    if i_count == 0 {
      return Some((false, term.clone()));
    }
    if i_count != 1 {
      return None;
    }
    let others: Vec<Expr> = factors
      .iter()
      .filter(|f| !matches!(f, Expr::Identifier(s) if s == "I"))
      .map(|f| (*f).clone())
      .collect();
    let coeff = if others.is_empty() {
      Expr::Integer(1)
    } else if others.len() == 1 {
      others.into_iter().next().unwrap()
    } else {
      times(others)
    };
    Some((true, coeff))
  };
  let (a_imag, a_part) = split(&plus_args[0])?;
  let (b_imag, b_part) = split(&plus_args[1])?;
  let (re_part, im_part) = match (a_imag, b_imag) {
    (false, true) => (a_part, b_part),
    (true, false) => (b_part, a_part),
    _ => return None,
  };
  let Expr::Real(re_acc) = accuracy_ast(&[re_part]).ok()? else {
    return None;
  };
  let Expr::Real(im_acc) = accuracy_ast(&[im_part]).ok()? else {
    return None;
  };
  let combined = -((10f64.powf(-2.0 * re_acc) + 10f64.powf(-2.0 * im_acc))
    .sqrt()
    .log10());
  Some(Expr::Real(combined))
}

/// PowerExpand[expr] - expand powers of products and powers
/// Rules: (a^b)^c -> a^(b*c), (a*b)^c -> a^c * b^c
pub fn power_expand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "PowerExpand expects 1 or 2 arguments".into(),
    ));
  }
  // Second argument (Assumptions) is accepted but not used
  Ok(power_expand_recursive(&args[0]))
}

fn power_expand_recursive(expr: &Expr) -> Expr {
  // Helper to extract (base, exponent) from any Power representation
  let extract_power = |e: &Expr| -> Option<(Expr, Expr)> {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => Some((*left.clone(), *right.clone())),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        Some((args[0].clone(), args[1].clone()))
      }
      // Sqrt[x] = Power[x, 1/2]
      expr if is_sqrt(expr).is_some() => {
        let sqrt_arg = is_sqrt(expr).unwrap();
        Some((
          sqrt_arg.clone(),
          call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
        ))
      }
      _ => None,
    }
  };

  // Helper to recursively collect all Times factors (flattening nested Times
  // and converting Divide to Times with Power[..., -1])
  fn collect_times_factors(e: &Expr) -> Vec<Expr> {
    match e {
      Expr::FunctionCall { name, args }
        if name == "Times" && args.len() >= 2 =>
      {
        args.iter().flat_map(collect_times_factors).collect()
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        let mut factors = collect_times_factors(left);
        factors.extend(collect_times_factors(right));
        factors
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => {
        let mut factors = collect_times_factors(left);
        factors.push(pow2(*right.clone(), Expr::Integer(-1)));
        factors
      }
      _ => vec![e.clone()],
    }
  }

  let extract_times = |e: &Expr| -> Option<Vec<Expr>> {
    let factors = collect_times_factors(e);
    if factors.len() >= 2 {
      Some(factors)
    } else {
      None
    }
  };

  match expr {
    _ if extract_power(expr).is_some() => {
      let (raw_base, raw_exp) = extract_power(expr).unwrap();
      let base = power_expand_recursive(&raw_base);
      let exp = power_expand_recursive(&raw_exp);

      // (a^b)^c -> a^(b*c)
      if let Some((inner_base, inner_exp)) = extract_power(&base) {
        let new_exp = match times_ast(&[inner_exp.clone(), exp.clone()]) {
          Ok(r) => r,
          Err(_) => times2(inner_exp, exp),
        };
        return match crate::evaluator::evaluate_expr_to_expr(&pow2(
          inner_base, new_exp,
        )) {
          Ok(r) => r,
          Err(_) => expr.clone(),
        };
      }

      // (a*b*...)^c -> a^c * b^c * ...
      // Also applies (a^r)^c -> a^(r*c) for each factor
      if let Some(factors) = extract_times(&base) {
        let expanded: Vec<Expr> = factors
          .iter()
          .map(|factor| {
            // If factor is itself a power, apply (a^r)^c -> a^(r*c)
            if let Some((inner_base, inner_exp)) = extract_power(factor) {
              let new_exp = match times_ast(&[inner_exp.clone(), exp.clone()]) {
                Ok(r) => r,
                Err(_) => times2(inner_exp, exp.clone()),
              };
              match power_two(&inner_base, &new_exp) {
                Ok(r) => r,
                Err(_) => pow2(inner_base, new_exp),
              }
            } else {
              match power_two(factor, &exp) {
                Ok(r) => r,
                Err(_) => pow2(factor.clone(), exp.clone()),
              }
            }
          })
          .collect();
        return match times_ast(&expanded) {
          Ok(r) => r,
          Err(_) => times(expanded),
        };
      }

      // E^(a*Log[b]) -> b^a  and  E^Log[b] -> b
      if matches!(&base, Expr::Constant(c) if c == "E") {
        // Check if exponent is Log[b]
        if let Expr::FunctionCall { name: ln, args: la } = &exp
          && ln == "Log"
          && la.len() == 1
        {
          return la[0].clone();
        }
        // Check if exponent is a product containing Log[b]
        let exp_factors = collect_times_factors(&exp);
        if exp_factors.len() >= 2 {
          // Find the Log factor
          let log_idx = exp_factors.iter().position(|f| {
            matches!(f, Expr::FunctionCall { name, args } if name == "Log" && args.len() == 1)
          });
          if let Some(idx) = log_idx
            && let Expr::FunctionCall { args: la, .. } = &exp_factors[idx]
          {
            let log_arg = la[0].clone();
            // Remaining factors form the new exponent
            let mut remaining: Vec<Expr> = exp_factors.clone();
            remaining.remove(idx);
            let new_exp = if remaining.len() == 1 {
              remaining.into_iter().next().unwrap()
            } else {
              match times_ast(&remaining) {
                Ok(r) => r,
                Err(_) => {
                  return pow2(base, exp);
                }
              }
            };
            return match power_two(&log_arg, &new_exp) {
              Ok(r) => r,
              Err(_) => pow2(log_arg, new_exp),
            };
          }
        }
      }

      pow2(base, exp)
    }
    // Log expansion rules (assuming positive reals):
    // Log[a*b*...] -> Log[a] + Log[b] + ...
    // Log[a^b] -> b*Log[a]
    Expr::FunctionCall { name, args } if name == "Log" && args.len() == 1 => {
      // First, recursively expand the argument
      let expanded_arg = power_expand_recursive(&args[0]);

      // Log of a quotient: Log[a/b] -> Log[a] - Log[b]
      // Convert a/b to Times[a, Power[b, -1]] and fall through to product rule
      let expanded_arg = match &expanded_arg {
        Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left,
          right,
        } => {
          times(vec![*left.clone(), pow2(*right.clone(), Expr::Integer(-1))])
        }
        _ => expanded_arg,
      };

      // Log of a product: Log[a*b*...] -> Log[a] + Log[b] + ...
      if let Some(factors) = extract_times(&expanded_arg) {
        let log_terms: Vec<Expr> = factors
          .iter()
          .map(|f| power_expand_recursive(&call1("Log", f.clone())))
          .collect();
        return match plus_ast(&log_terms) {
          Ok(r) => r,
          Err(_) => plus(log_terms),
        };
      }

      // Log of a power: Log[a^b] -> b*Log[a]
      if let Some((base, exp)) = extract_power(&expanded_arg) {
        let log_expr = call1("Log", base);
        // Evaluate Log[base] to simplify cases like Log[E] -> 1
        let log_base = crate::evaluator::evaluate_expr_to_expr(&log_expr)
          .unwrap_or(log_expr);
        let log_base = power_expand_recursive(&log_base);
        return match times_ast(&[exp, log_base]) {
          Ok(r) => r,
          Err(_) => call1("Log", expanded_arg),
        };
      }

      // Evaluate Log to simplify cases like Log[E] -> 1, Log[1] -> 0
      let log_expr = call1("Log", expanded_arg.clone());
      crate::evaluator::evaluate_expr_to_expr(&log_expr).unwrap_or(log_expr)
    }
    Expr::FunctionCall { name, args } => {
      let new_args: Vec<Expr> =
        args.iter().map(power_expand_recursive).collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }
    Expr::BinaryOp { op, left, right } => binop(
      *op,
      power_expand_recursive(left),
      power_expand_recursive(right),
    ),
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(power_expand_recursive(operand)),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(power_expand_recursive).collect())
    }
    _ => expr.clone(),
  }
}

// ─── Variables ──────────────────────────────────────────────────────

/// Variables[expr] - list of variables in a polynomial expression
pub fn variables_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Variables expects exactly 1 argument".into(),
    ));
  }
  // Variables only applies to polynomials. A relational or logical expression
  // (equation, inequality, And/Or/...) is not a polynomial, so wolframscript
  // returns {} rather than treating the whole expression as one variable.
  let head = crate::evaluator::evaluate_function_call_ast(
    "Head",
    std::slice::from_ref(&args[0]),
  )?;
  if let Expr::Identifier(h) = &head
    && matches!(
      h.as_str(),
      "Equal"
        | "Unequal"
        | "Less"
        | "Greater"
        | "LessEqual"
        | "GreaterEqual"
        | "Inequality"
        | "And"
        | "Or"
        | "Not"
        | "Nand"
        | "Nor"
        | "Xor"
        | "Implies"
        | "Equivalent"
        | "SameQ"
        | "UnsameQ"
    )
  {
    return Ok(Expr::List(vec![].into()));
  }
  let mut vars = Vec::new();
  collect_variables(&args[0], &mut vars);
  // Deduplicate while preserving order
  let mut seen = std::collections::HashSet::new();
  vars.retain(|v| seen.insert(expr_to_string(v)));
  // For List input, sort in canonical order (alphabetical);
  // for non-List input, preserve first-appearance order (matching Wolfram).
  if matches!(&args[0], Expr::List(_)) {
    vars.sort_by_key(expr_to_string);
  }
  Ok(Expr::List(vars.into()))
}

fn collect_variables(expr: &Expr, vars: &mut Vec<Expr>) {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::BigInteger(_)
    | Expr::BigFloat(_, _) => {}
    Expr::Identifier(s) => {
      // Skip built-in constants
      if !matches!(
        s.as_str(),
        "True"
          | "False"
          | "Null"
          | "Pi"
          | "E"
          | "I"
          | "Infinity"
          | "ComplexInfinity"
          | "Indeterminate"
      ) {
        vars.push(expr.clone());
      }
    }
    Expr::List(items) => {
      for item in items {
        collect_variables(item, vars);
      }
    }
    Expr::BinaryOp { op, left, right } => {
      match op {
        BinaryOperator::Plus
        | BinaryOperator::Minus
        | BinaryOperator::Times
        | BinaryOperator::Power
        | BinaryOperator::Divide => {
          collect_variables(left, vars);
          collect_variables(right, vars);
        }
        _ => {
          // Treat as atomic variable-like term
          vars.push(expr.clone());
        }
      }
    }
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Plus" | "Times" | "Power" | "Rational" => {
          for arg in args {
            collect_variables(arg, vars);
          }
        }
        _ => {
          // Non-polynomial function (like Sin, Cos) — treat as variable
          vars.push(expr.clone());
        }
      }
    }
    Expr::UnaryOp { op: _, operand } => {
      collect_variables(operand, vars);
    }
    _ => {
      vars.push(expr.clone());
    }
  }
}

/// LinearRecurrence[ker, init, n] - generates a linear recurrence sequence of length n.
/// LinearRecurrence[ker, init, {nmin, nmax}] - returns elements nmin through nmax.
/// LinearRecurrence[ker, init, {n}] - returns the list containing only the nth element.
pub fn linear_recurrence_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated("LinearRecurrence", args));
  }

  let kernel = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(unevaluated("LinearRecurrence", args));
    }
  };

  let init = match &args[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(unevaluated("LinearRecurrence", args));
    }
  };

  // Parse the third argument: n, {n}, or {nmin, nmax}
  let (total_n, range) = match &args[2] {
    Expr::Integer(n) if *n >= 0 => (*n as usize, None),
    Expr::List(items) if items.len() == 1 => {
      if let Some(n) = expr_to_i128(&items[0]) {
        if n < 1 {
          return linear_recurrence_backward(&kernel[..], &init[..], n, n).map(
            |r| r.unwrap_or_else(|| unevaluated("LinearRecurrence", args)),
          );
        }
        (n as usize, Some((n as usize, n as usize)))
      } else {
        return Ok(unevaluated("LinearRecurrence", args));
      }
    }
    Expr::List(items) if items.len() == 2 => {
      if let (Some(nmin), Some(nmax)) =
        (expr_to_i128(&items[0]), expr_to_i128(&items[1]))
        && nmin <= nmax
      {
        if nmin < 1 {
          return linear_recurrence_backward(
            &kernel[..],
            &init[..],
            nmin,
            nmax,
          )
          .map(|r| r.unwrap_or_else(|| unevaluated("LinearRecurrence", args)));
        }
        (nmax as usize, Some((nmin as usize, nmax as usize)))
      } else {
        return Ok(unevaluated("LinearRecurrence", args));
      }
    }
    _ => {
      return Ok(unevaluated("LinearRecurrence", args));
    }
  };

  let mut seq = init.clone();

  // Extend sequence to total_n elements
  while seq.len() < total_n {
    let mut next = Expr::Integer(0);
    for (i, coeff) in kernel.iter().enumerate() {
      let idx = seq.len() - 1 - i;
      let term = times2(coeff.clone(), seq[idx].clone());
      next = plus2(next, term);
    }
    // Evaluate the expression to simplify
    let evaluated = crate::evaluator::evaluate_expr_to_expr(&next)?;
    seq.push(evaluated);
  }

  match range {
    None => Ok(Expr::List(seq[..total_n].to_vec().into())),
    Some((nmin, nmax)) => Ok(Expr::List(seq[nmin - 1..nmax].to_vec().into())),
  }
}

/// LinearRecurrence[ker, init, {nmin, nmax}] with nmin < 1: the terms before
/// the initial values solve the recurrence for its oldest term,
/// a[n - k] = (a[n] - Σ_{i<k} ker[i] a[n - i]) / ker[k], which needs a
/// nonzero last kernel entry.
fn linear_recurrence_backward(
  kernel: &[Expr],
  init: &[Expr],
  nmin: i128,
  nmax: i128,
) -> Result<Option<Expr>, InterpreterError> {
  let k = kernel.len();
  if k == 0 || init.len() < k || is_expr_zero(&kernel[k - 1]) {
    return Ok(None);
  }
  // seq[j] holds a[first + j].
  let mut seq: std::collections::VecDeque<Expr> =
    init.iter().cloned().collect();
  let mut first: i128 = 1;
  while first > nmin {
    let mut rest = Expr::Integer(0);
    for i in 1..k {
      rest = plus2(rest, times2(kernel[i - 1].clone(), seq[k - 1 - i].clone()));
    }
    let older = crate::evaluator::evaluate_expr_to_expr(&times2(
      plus2(seq[k - 1].clone(), times2(Expr::Integer(-1), rest)),
      pow2(kernel[k - 1].clone(), Expr::Integer(-1)),
    ))?;
    seq.push_front(older);
    first -= 1;
  }
  while (seq.len() as i128) < nmax - first + 1 {
    let mut next = Expr::Integer(0);
    for (i, coeff) in kernel.iter().enumerate() {
      let idx = seq.len() - 1 - i;
      next = plus2(next, times2(coeff.clone(), seq[idx].clone()));
    }
    seq.push_back(crate::evaluator::evaluate_expr_to_expr(&next)?);
  }
  let from = (nmin - first) as usize;
  let to = (nmax - first) as usize;
  Ok(Some(Expr::List(
    seq.range(from..=to).cloned().collect::<Vec<_>>().into(),
  )))
}

/// EuclideanDistance[u, v] - Euclidean distance between two points
/// WarpingDistance[s1, s2] - dynamic time warping distance between two numeric
/// sequences, with the default Manhattan (|a - b|) local cost. The result is
/// the accumulated cost along the optimal warping path (a Real, matching
/// wolframscript). Returns None when an argument isn't a numeric sequence.
fn warping_distance_value(a: &Expr, b: &Expr) -> Option<f64> {
  let to_vec = |e: &Expr| -> Option<Vec<f64>> {
    match e {
      Expr::List(items) => items.iter().map(try_eval_to_f64).collect(),
      _ => None,
    }
  };
  let a = to_vec(a)?;
  let b = to_vec(b)?;
  let (n, m) = (a.len(), b.len());
  if n == 0 || m == 0 {
    return None;
  }
  let inf = f64::INFINITY;
  let mut dp = vec![vec![inf; m + 1]; n + 1];
  dp[0][0] = 0.0;
  for i in 1..=n {
    for j in 1..=m {
      let cost = (a[i - 1] - b[j - 1]).abs();
      dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
    }
  }
  Some(dp[n][m])
}

/// Backtrace the DTW cost matrix, returning the correspondence path as a
/// pair of 1-indexed index lists into `a` and `b`. On ties the diagonal step
/// is preferred, then the step advancing the longer remaining sequence,
/// matching wolframscript's WarpingCorrespondence path.
fn warping_correspondence_path(
  a: &Expr,
  b: &Expr,
) -> Option<(Vec<i128>, Vec<i128>)> {
  let to_vec = |e: &Expr| -> Option<Vec<f64>> {
    match e {
      Expr::List(items) => items.iter().map(try_eval_to_f64).collect(),
      _ => None,
    }
  };
  let a = to_vec(a)?;
  let b = to_vec(b)?;
  let (n, m) = (a.len(), b.len());
  if n == 0 || m == 0 {
    return None;
  }
  let inf = f64::INFINITY;
  let mut dp = vec![vec![inf; m + 1]; n + 1];
  dp[0][0] = 0.0;
  for i in 1..=n {
    for j in 1..=m {
      let cost = (a[i - 1] - b[j - 1]).abs();
      dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
    }
  }

  let (mut i, mut j) = (n, m);
  let mut pi = vec![i as i128];
  let mut pj = vec![j as i128];
  while i > 1 || j > 1 {
    let diag = if i > 1 && j > 1 {
      dp[i - 1][j - 1]
    } else {
      inf
    };
    let up = if i > 1 { dp[i - 1][j] } else { inf };
    let left = if j > 1 { dp[i][j - 1] } else { inf };
    let best = diag.min(up).min(left);
    // Prefer the diagonal on ties; otherwise advance the coordinate with more
    // remaining steps so both sequences are consumed evenly.
    if diag <= best {
      i -= 1;
      j -= 1;
    } else if up == best && (j == 1 || up <= left) {
      i -= 1;
    } else {
      j -= 1;
    }
    pi.push(i as i128);
    pj.push(j as i128);
  }
  pi.reverse();
  pj.reverse();
  Some((pi, pj))
}

/// WarpingCorrespondence[s1, s2] — the dynamic-time-warping alignment path
/// between two numeric sequences, given as a pair of 1-indexed index lists.
pub fn warping_correspondence_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("WarpingCorrespondence", args));
  if args.len() != 2 {
    return unevaluated();
  }
  if let Some((pi, pj)) = warping_correspondence_path(&args[0], &args[1]) {
    Ok(Expr::List(
      vec![
        Expr::List(
          pi.into_iter().map(Expr::Integer).collect::<Vec<_>>().into(),
        ),
        Expr::List(
          pj.into_iter().map(Expr::Integer).collect::<Vec<_>>().into(),
        ),
      ]
      .into(),
    ))
  } else {
    let is_numeric_vec = |e: &Expr| {
      matches!(e, Expr::List(items) if !items.is_empty()
        && items.iter().all(|x| try_eval_to_f64(x).is_some()))
    };
    if let Some(bad) = args.iter().find(|a| !is_numeric_vec(a)) {
      crate::emit_message(&format!(
        "WarpingCorrespondence::invarg: Expecting a real-valued numeric or \
         Boolean vector or matrix instead of {}.",
        crate::syntax::format_expr(bad, crate::syntax::ExprForm::Output)
      ));
    }
    unevaluated()
  }
}

pub fn warping_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 2 {
    if let Some(d) = warping_distance_value(&args[0], &args[1]) {
      return Ok(Expr::Real(d));
    }
    // Identify the offending (non-numeric-vector) argument, matching the
    // wolframscript WarpingDistance::invarg message.
    let is_numeric_vec = |e: &Expr| {
      matches!(e, Expr::List(items) if !items.is_empty()
        && items.iter().all(|x| try_eval_to_f64(x).is_some()))
    };
    if let Some(bad) = args.iter().find(|a| !is_numeric_vec(a)) {
      crate::emit_message(&format!(
        "WarpingDistance::invarg: Expecting a real-valued numeric or Boolean \
         vector or matrix instead of {}.",
        crate::syntax::format_expr(bad, crate::syntax::ExprForm::Output)
      ));
    }
  }
  Ok(unevaluated("WarpingDistance", args))
}

pub fn euclidean_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "EuclideanDistance expects exactly 2 arguments".into(),
    ));
  }

  if let (Expr::List(u), Expr::List(v)) = (&args[0], &args[1]) {
    if u.len() != v.len() {
      return Err(InterpreterError::EvaluationError(
        "EuclideanDistance: vectors must have the same length".into(),
      ));
    }
    // Build Sqrt[Sum[Abs[u_i - v_i]^2]]
    let mut sum_args = Vec::new();
    for (ui, vi) in u.iter().zip(v.iter()) {
      let diff = crate::evaluator::evaluate_function_call_ast(
        "Subtract",
        &[ui.clone(), vi.clone()],
      )?;
      let abs_diff =
        crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
      let sq = crate::evaluator::evaluate_function_call_ast(
        "Power",
        &[abs_diff, Expr::Integer(2)],
      )?;
      sum_args.push(sq);
    }
    let sum = crate::evaluator::evaluate_function_call_ast("Plus", &sum_args)?;
    crate::evaluator::evaluate_function_call_ast("Sqrt", &[sum])
  } else {
    // Scalar distance: Abs[u - v]
    let diff = crate::evaluator::evaluate_function_call_ast(
      "Subtract",
      &[args[0].clone(), args[1].clone()],
    )?;
    crate::evaluator::evaluate_function_call_ast("Abs", &[diff])
  }
}

/// SquaredEuclideanDistance[u, v] - squared Euclidean distance
pub fn squared_euclidean_distance_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "SquaredEuclideanDistance expects exactly 2 arguments".into(),
    ));
  }

  if let (Expr::List(u), Expr::List(v)) = (&args[0], &args[1]) {
    if u.len() != v.len() {
      return Err(InterpreterError::EvaluationError(
        "SquaredEuclideanDistance: vectors must have the same length".into(),
      ));
    }
    let mut sum_args = Vec::new();
    for (ui, vi) in u.iter().zip(v.iter()) {
      let diff = crate::evaluator::evaluate_function_call_ast(
        "Subtract",
        &[ui.clone(), vi.clone()],
      )?;
      let abs_diff =
        crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
      let sq = crate::evaluator::evaluate_function_call_ast(
        "Power",
        &[abs_diff, Expr::Integer(2)],
      )?;
      sum_args.push(sq);
    }
    crate::evaluator::evaluate_function_call_ast("Plus", &sum_args)
  } else {
    let diff = crate::evaluator::evaluate_function_call_ast(
      "Subtract",
      &[args[0].clone(), args[1].clone()],
    )?;
    let abs_diff =
      crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
    crate::evaluator::evaluate_function_call_ast(
      "Power",
      &[abs_diff, Expr::Integer(2)],
    )
  }
}

/// Core DFT computation shared by Fourier and InverseFourier.
/// `sign` is +1 for Fourier, -1 for InverseFourier (before applying `b`).
/// FourierParameters {a, b}: F_s = n^((a-1)/2) * sum_{r=0}^{n-1} u_r * exp(2*pi*i*b*(r*s)/n)
pub(crate) fn dft_core(
  data: &[(f64, f64)],
  param_a: f64,
  param_b: f64,
  inverse: bool,
) -> Vec<(f64, f64)> {
  let n = data.len();
  if n == 0 {
    return vec![];
  }
  let nf = n as f64;
  // For Fourier: scaling = n^((a-1)/2), exponent sign from b
  // For InverseFourier: scaling = n^((-1-a)/2), exponent sign from -b
  let (scaling, exp_sign) = if inverse {
    (nf.powf((-1.0 - param_a) / 2.0), -param_b)
  } else {
    (nf.powf((param_a - 1.0) / 2.0), param_b)
  };

  // For power-of-2 sizes, use Cooley-Tukey FFT (O(n log n)) instead of
  // the O(n^2) DFT.
  if n >= 2 && n.is_power_of_two() {
    let mut x: Vec<(f64, f64)> = data.to_vec();
    fft_pow2_in_place(&mut x, exp_sign);
    for (re, im) in &mut x {
      *re *= scaling;
      *im *= scaling;
    }
    return x;
  }

  let two_pi_over_n = 2.0 * std::f64::consts::PI / nf;
  let mut result = Vec::with_capacity(n);

  for s in 0..n {
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for r in 0..n {
      let angle = two_pi_over_n * exp_sign * (r as f64) * (s as f64);
      let (sin_a, cos_a) = angle.sin_cos();
      let (ur, ui) = data[r];
      // (ur + ui*i) * (cos_a + sin_a*i) = (ur*cos_a - ui*sin_a) + (ur*sin_a + ui*cos_a)*i
      sum_re += ur * cos_a - ui * sin_a;
      sum_im += ur * sin_a + ui * cos_a;
    }
    result.push((scaling * sum_re, scaling * sum_im));
  }
  result
}

/// In-place Cooley-Tukey radix-2 FFT. `data.len()` must be a power of 2.
/// `exp_sign` is +1 or -1, selecting forward vs inverse twiddle direction.
/// Scaling is left to the caller.
pub(crate) fn fft_pow2_in_place(data: &mut [(f64, f64)], exp_sign: f64) {
  let n = data.len();
  // Bit-reverse permutation.
  let mut j = 0usize;
  for i in 1..n {
    let mut bit = n >> 1;
    while j & bit != 0 {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if i < j {
      data.swap(i, j);
    }
  }
  // Iterative butterflies.
  let mut size = 2;
  while size <= n {
    let half = size / 2;
    let theta = exp_sign * 2.0 * std::f64::consts::PI / size as f64;
    let (sin_t, cos_t) = theta.sin_cos();
    let mut i = 0;
    while i < n {
      // Twiddle for this butterfly group.
      let mut w_re = 1.0_f64;
      let mut w_im = 0.0_f64;
      for k in 0..half {
        let a = data[i + k];
        let b = data[i + k + half];
        let t_re = w_re * b.0 - w_im * b.1;
        let t_im = w_re * b.1 + w_im * b.0;
        data[i + k] = (a.0 + t_re, a.1 + t_im);
        data[i + k + half] = (a.0 - t_re, a.1 - t_im);
        // Advance twiddle.
        let _ = k;
        let new_w_re = w_re * cos_t - w_im * sin_t;
        let new_w_im = w_re * sin_t + w_im * cos_t;
        w_re = new_w_re;
        w_im = new_w_im;
      }
      i += size;
    }
    size <<= 1;
  }
}

/// Round a floating-point number to clean up near-integer/near-half values.
/// This accounts for floating-point errors in DFT trig computations.
fn fourier_round(x: f64) -> f64 {
  if x.abs() < 1e-14 {
    return 0.0;
  }
  // Check if x is very close to an integer or half-integer
  let rounded = x.round();
  if (x - rounded).abs() < 1e-14 {
    return rounded;
  }
  // Check half-integer
  let doubled = (x * 2.0).round();
  if (x * 2.0 - doubled).abs() < 1e-13 {
    return doubled / 2.0;
  }
  x
}

/// Build an Expr for a Fourier/InverseFourier result element.
/// If `force_complex` is true, always output as Complex (even if im == 0).
pub fn fourier_result_to_expr(re: f64, im: f64, force_complex: bool) -> Expr {
  let re = fourier_round(re);
  let im = fourier_round(im);

  if re != 0.0 && im != 0.0 {
    // The evaluated form of `Complex[re, im]`, built directly: a large
    // transform has one per element.
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Real(re),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Real(im), Expr::Identifier("I".to_string())].into(),
        },
      ]
      .into(),
    }
  } else if force_complex || im != 0.0 {
    // Build via Complex[re, im] which the evaluator will format as a + b*I
    crate::evaluator::evaluate_function_call_ast(
      "Complex",
      &[Expr::Real(re), Expr::Real(im)],
    )
    .unwrap_or_else(|_| build_complex_float_expr(re, im))
  } else {
    Expr::Real(re)
  }
}

/// Parse FourierParameters option from args, returning (a, b).
/// Default is {0, 1}.
fn parse_fourier_parameters(
  args: &[Expr],
) -> Result<(f64, f64), InterpreterError> {
  for arg in args {
    if let Expr::Rule {
      pattern,
      replacement,
    } = arg
      && matches!(pattern.as_ref(), Expr::Identifier(name) if name == "FourierParameters")
    {
      if let Expr::List(params) = replacement.as_ref()
        && params.len() == 2
      {
        let a = try_eval_to_f64(&params[0]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "FourierParameters: first parameter must be numeric".into(),
          )
        })?;
        let b = try_eval_to_f64(&params[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "FourierParameters: second parameter must be numeric".into(),
          )
        })?;
        return Ok((a, b));
      }
      return Err(InterpreterError::EvaluationError(
        "FourierParameters must be a list of two numbers".into(),
      ));
    }
  }
  Ok((0.0, 1.0)) // default
}

/// Shared implementation for Fourier and InverseFourier
fn fourier_impl(
  func_name: &str,
  args: &[Expr],
  inverse: bool,
) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{func_name} expects 1 or 2 arguments"
    )));
  }

  // A SparseArray stands for the dense array it represents.
  if let Some(dense) =
    crate::functions::list_helpers_ast::densify_sparse_array(&args[0])
  {
    let mut dense_args = args.to_vec();
    dense_args[0] = dense;
    return fourier_impl(func_name, &dense_args, inverse);
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      crate::emit_message(&format!(
        "{}::fftl: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
        func_name,
        expr_to_string(&args[0])
      ));
      return Ok(unevaluated(func_name, args));
    }
  };

  // Rectangular arrays of rank >= 2 are transformed along every axis.
  if matches!(items[0], Expr::List(_)) {
    return fourier_array_impl(func_name, args, inverse);
  }

  // Extract numeric values as complex pairs
  let mut data: Vec<(f64, f64)> = Vec::with_capacity(items.len());
  for item in items {
    if let Some((re, im)) = try_extract_complex_float(item) {
      data.push((re, im));
    } else {
      crate::emit_message(&format!(
        "{}::fftl: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
        func_name,
        expr_to_string(&args[0])
      ));
      return Ok(unevaluated(func_name, args));
    }
  }

  let (param_a, param_b) = parse_fourier_parameters(&args[1..])?;
  let result = dft_core(&data, param_a, param_b, inverse);

  // Determine if any element has nonzero imaginary part
  let any_complex = result.iter().any(|(_, im)| {
    let im_r = fourier_round(*im);
    im_r != 0.0
  });

  let exprs: Vec<Expr> = result
    .iter()
    .map(|&(re, im)| fourier_result_to_expr(re, im, any_complex))
    .collect();

  Ok(Expr::List(exprs.into()))
}

/// Multi-dimensional Fourier/InverseFourier of a rectangular array: the
/// separable transform applies the 1D transform along each axis in turn.
fn fourier_array_impl(
  func_name: &str,
  args: &[Expr],
  inverse: bool,
) -> Result<Expr, InterpreterError> {
  let bad = || {
    crate::emit_message(&format!(
      "{}::fftl: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
      func_name,
      expr_to_string(&args[0])
    ));
    Ok(unevaluated(func_name, args))
  };

  // Determine the shape by descending along first elements.
  let mut shape: Vec<usize> = Vec::new();
  let mut cur = &args[0];
  while let Expr::List(l) = cur {
    if l.is_empty() {
      return bad();
    }
    shape.push(l.len());
    cur = &l[0];
  }

  // Flatten in row-major order, validating rectangularity.
  fn flatten(e: &Expr, shape: &[usize], out: &mut Vec<(f64, f64)>) -> bool {
    match shape {
      [] => match try_extract_complex_float(e) {
        Some(c) => {
          out.push(c);
          true
        }
        None => false,
      },
      [n, rest @ ..] => match e {
        Expr::List(l) if l.len() == *n => {
          l.iter().all(|x| flatten(x, rest, out))
        }
        _ => false,
      },
    }
  }
  let mut data: Vec<(f64, f64)> = Vec::with_capacity(shape.iter().product());
  if !flatten(&args[0], &shape, &mut data) {
    return bad();
  }

  let (param_a, param_b) = parse_fourier_parameters(&args[1..])?;

  // Transform along each axis.
  let total = data.len();
  let mut stride = total;
  for &n in &shape {
    stride /= n;
    let block = stride * n;
    let mut line = vec![(0.0, 0.0); n];
    for base in (0..total).step_by(block) {
      for off in 0..stride {
        for (k, slot) in line.iter_mut().enumerate() {
          *slot = data[base + off + k * stride];
        }
        let t = dft_core(&line, param_a, param_b, inverse);
        for (k, v) in t.into_iter().enumerate() {
          data[base + off + k * stride] = v;
        }
      }
    }
  }

  // Imaginary parts that are pure round-off relative to the magnitude of the
  // whole array (e.g. `InverseFourier` of a thresholded, Hermitian-symmetric
  // spectrum of a large-valued image) are dropped, so the real round trip
  // stays real instead of turning every entry into `a + 10^-12 I`.
  let scale = data
    .iter()
    .fold(0.0_f64, |m, (re, im)| m.max(re.abs()).max(im.abs()));
  if scale > 0.0 && data.iter().all(|(_, im)| im.abs() <= scale * 1e-13) {
    for (_, im) in &mut data {
      *im = 0.0;
    }
  }

  let any_complex = data.iter().any(|(_, im)| fourier_round(*im) != 0.0);

  fn build(data: &[(f64, f64)], shape: &[usize], any_complex: bool) -> Expr {
    match shape {
      [] => fourier_result_to_expr(data[0].0, data[0].1, any_complex),
      [n, rest @ ..] => {
        let sub: usize = rest.iter().product();
        Expr::List(
          (0..*n)
            .map(|i| build(&data[i * sub..(i + 1) * sub], rest, any_complex))
            .collect(),
        )
      }
    }
  }
  Ok(build(&data, &shape, any_complex))
}

/// Fourier[list] or Fourier[list, opts] - Discrete Fourier transform
pub fn fourier_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  fourier_impl("Fourier", args, false)
}

/// InverseFourier[list] or InverseFourier[list, opts] - Inverse discrete Fourier transform
pub fn inverse_fourier_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  fourier_impl("InverseFourier", args, true)
}

/// Core discrete cosine transform of a real list `u`.
/// `m` selects the DCT method: 1, 2 (default), 3, or 4.
/// Mirrors the normalisation used by Wolfram's `FourierDCT`.
fn dct_core(u: &[f64], m: i64) -> Vec<f64> {
  let n = u.len();
  let nf = n as f64;
  let pi = std::f64::consts::PI;
  let mut out = Vec::with_capacity(n);
  for s in 0..n {
    let sf = s as f64;
    let mut acc = 0.0;
    match m {
      1 => {
        if n == 1 {
          acc = u[0];
        } else {
          for (r, &ur) in u.iter().enumerate() {
            let w = if r == 0 || r == n - 1 { 0.5 } else { 1.0 };
            acc += w * ur * (pi * (r as f64) * sf / (nf - 1.0)).cos();
          }
          acc *= (2.0 / (nf - 1.0)).sqrt();
        }
      }
      3 => {
        for (r, &ur) in u.iter().enumerate() {
          let w = if r == 0 { 0.5 } else { 1.0 };
          acc +=
            w * ur * (pi * (2.0 * sf + 1.0) * (r as f64) / (2.0 * nf)).cos();
        }
        acc *= 2.0 / nf.sqrt();
      }
      4 => {
        for (r, &ur) in u.iter().enumerate() {
          acc += ur
            * (pi * (2.0 * (r as f64) + 1.0) * (2.0 * sf + 1.0) / (4.0 * nf))
              .cos();
        }
        acc *= (2.0 / nf).sqrt();
      }
      // default: type 2
      _ => {
        for (r, &ur) in u.iter().enumerate() {
          acc += ur * (pi * (2.0 * (r as f64) + 1.0) * sf / (2.0 * nf)).cos();
        }
        acc /= nf.sqrt();
      }
    }
    out.push(acc);
  }
  out
}

/// FourierDCT[list] or FourierDCT[list, m] — discrete cosine transform.
/// `m` (1..4) selects the DCT method; default is 2.
pub fn fourier_dct_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FourierDCT expects 1 or 2 arguments".into(),
    ));
  }

  let unevaluated = || {
    crate::emit_message(&format!(
      "FourierDCT::fftl: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
      expr_to_string(&args[0])
    ));
    Ok(unevaluated("FourierDCT", args))
  };

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return unevaluated(),
  };

  // Extract real numeric values. Any non-real entry leaves it unevaluated.
  let mut data: Vec<f64> = Vec::with_capacity(items.len());
  for item in items {
    match try_extract_complex_float(item) {
      Some((re, 0.0)) => data.push(re),
      _ => return unevaluated(),
    }
  }

  // Optional method argument (1..4); default 2.
  let m = if args.len() == 2 {
    match try_eval_to_f64(&args[1]) {
      Some(v) if (v - v.round()).abs() < 1e-12 && (1.0..=4.0).contains(&v) => {
        v.round() as i64
      }
      _ => return unevaluated(),
    }
  } else {
    2
  };

  let result = dct_core(&data, m);
  let exprs: Vec<Expr> = result.iter().map(|&x| Expr::Real(x)).collect();
  Ok(Expr::List(exprs.into()))
}

/// DiscreteHilbertTransform[list] — the periodic discrete Hilbert transform
/// of a list of real numbers. Computed via the DFT: multiply the spectrum by
/// -i on the positive-frequency half and +i on the negative-frequency half
/// (zeroing the DC and, for even length, the Nyquist bin), then invert.
/// Matches Wolfram's Chop behaviour: entries that vanish become the exact
/// integer 0 while the rest stay machine-precision reals.
pub fn discrete_hilbert_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DiscreteHilbertTransform", args));
  // Emit the ::data message and leave the expression unevaluated, matching
  // wolframscript's handling of empty or non-real arguments.
  let data_err = || {
    crate::emit_message(&format!(
      "DiscreteHilbertTransform::data: {} is empty or not a real valued numerical array.",
      expr_to_string(&args[0])
    ));
    unevaluated()
  };

  if args.len() != 1 {
    return unevaluated();
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return data_err(),
  };

  // Real numeric entries only.
  let mut data: Vec<(f64, f64)> = Vec::with_capacity(items.len());
  for item in items {
    match try_extract_complex_float(item) {
      Some((re, 0.0)) => data.push((re, 0.0)),
      _ => return data_err(),
    }
  }

  let n = data.len();

  // Forward DFT with no scaling: X[k] = sum_r u_r exp(-2 pi i r k / n).
  let x = dft_core(&data, 1.0, -1.0, false);

  // Apply the Hilbert multiplier in the frequency domain.
  let mut y = vec![(0.0f64, 0.0f64); n];
  for k in 1..n {
    let (re, im) = x[k];
    if 2 * k < n {
      // -i * (re + i im) = im - i re
      y[k] = (im, -re);
    } else if 2 * k == n {
      // Nyquist bin (even length): zeroed.
      y[k] = (0.0, 0.0);
    } else {
      // +i * (re + i im) = -im + i re
      y[k] = (-im, re);
    }
  }

  // Inverse DFT (scaling 1/n) recovers the real-valued transform.
  let out = dft_core(&y, 1.0, -1.0, true);

  let exprs: Vec<Expr> = out
    .iter()
    .map(|&(re, _im)| {
      if re.abs() < 1e-10 {
        Expr::Integer(0)
      } else {
        Expr::Real(fourier_round(re))
      }
    })
    .collect();
  Ok(Expr::List(exprs.into()))
}

fn dst_core(u: &[f64], m: i64) -> Vec<f64> {
  let n = u.len();
  let nf = n as f64;
  let pi = std::f64::consts::PI;
  let mut out = Vec::with_capacity(n);
  for s in 0..n {
    let sf = s as f64;
    let mut acc = 0.0;
    match m {
      1 => {
        for (r, &ur) in u.iter().enumerate() {
          acc += ur * (pi * ((r + 1) as f64) * (sf + 1.0) / (nf + 1.0)).sin();
        }
        acc *= (2.0 / (nf + 1.0)).sqrt();
      }
      3 => {
        // Transpose of type 2; the Nyquist term (r = n-1) is halved
        for (r, &ur) in u.iter().enumerate() {
          let w = if r == n - 1 { 0.5 } else { 1.0 };
          acc += w
            * ur
            * (pi * (2.0 * sf + 1.0) * ((r + 1) as f64) / (2.0 * nf)).sin();
        }
        acc *= 2.0 / nf.sqrt();
      }
      4 => {
        for (r, &ur) in u.iter().enumerate() {
          acc += ur
            * (pi * (2.0 * (r as f64) + 1.0) * (2.0 * sf + 1.0) / (4.0 * nf))
              .sin();
        }
        acc *= (2.0 / nf).sqrt();
      }
      // default: type 2
      _ => {
        for (r, &ur) in u.iter().enumerate() {
          acc += ur
            * (pi * (2.0 * (r as f64) + 1.0) * (sf + 1.0) / (2.0 * nf)).sin();
        }
        acc /= nf.sqrt();
      }
    }
    out.push(acc);
  }
  out
}

/// FourierDST[list] or FourierDST[list, m] — discrete sine transform.
/// `m` (1..4) selects the DST method; default is 2, mirroring
/// FourierDCT's conventions (including the numericization of exact
/// input and the ::fftl message for non-numeric arguments).
pub fn fourier_dst_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FourierDST expects 1 or 2 arguments".into(),
    ));
  }

  let unevaluated = || {
    crate::emit_message(&format!(
      "FourierDST::fftl: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
      expr_to_string(&args[0])
    ));
    Ok(unevaluated("FourierDST", args))
  };

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return unevaluated(),
  };

  let mut data: Vec<f64> = Vec::with_capacity(items.len());
  for item in items {
    match try_extract_complex_float(item) {
      Some((re, 0.0)) => data.push(re),
      _ => return unevaluated(),
    }
  }

  let m = if args.len() == 2 {
    match try_eval_to_f64(&args[1]) {
      Some(v) if (v - v.round()).abs() < 1e-12 && (1.0..=4.0).contains(&v) => {
        v.round() as i64
      }
      _ => return unevaluated(),
    }
  } else {
    2
  };

  let result = dst_core(&data, m);
  let exprs: Vec<Expr> = result.iter().map(|&x| Expr::Real(x)).collect();
  Ok(Expr::List(exprs.into()))
}

/// Wynn epsilon algorithm for series acceleration.
/// Takes a sequence of partial sums and returns an accelerated estimate.
/// Polynomial extrapolation using Neville's algorithm.
/// Given points (x_i, y_i) representing (1/n, S_n), extrapolate to x=0.
fn neville_extrapolation(xs: &[f64], ys: &[f64]) -> f64 {
  let n = xs.len();
  if n == 0 {
    return 0.0;
  }
  if n == 1 {
    return ys[0];
  }

  let mut c = ys.to_vec();
  for j in 1..n {
    for i in (j..n).rev() {
      // Neville's algorithm for interpolation at x=0:
      // c[i] = (x_i * c[i-1] - x_{i-j} * c[i]) / (x_i - x_{i-j})
      // Since we evaluate at x=0, this simplifies to:
      let denom = xs[i] - xs[i - j];
      if denom.abs() < 1e-300 {
        continue;
      }
      c[i] = (xs[i] * c[i - 1] - xs[i - j] * c[i]) / denom;
    }
  }
  c[n - 1]
}

fn is_infinite_upper(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(s) if s == "Infinity")
}

/// `f` at `var = at`, as a machine complex number.
fn complex_value_at(f: &Expr, var: &str, at: &Expr) -> Option<(f64, f64)> {
  let sub = substitute_variable(f, var, at);
  let v = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "N".to_string(),
    args: vec![sub].into(),
  })
  .ok()?;
  let (re, im) = crate::functions::list_helpers_ast::expr_to_complex_parts(&v)?;
  (re.is_finite() && im.is_finite()).then_some((re, im))
}

/// `Sum[f, {var, min, Infinity}]` for a smooth, decaying, possibly complex
/// summand by Euler–Maclaurin: the first terms summed directly, the tail as
/// `∫_N^∞ f + f(N)/2 - Σ B_2k/(2k)! f^(2k-1)(N)`. Unlike extrapolation in
/// `1/n`, this copes with tails such as `Log[x] x^(-2 - 2 I)` whose terms
/// oscillate in `Log[x]`.
fn nsum_euler_maclaurin(f: &Expr, var: &str, min: i64) -> Option<(f64, f64)> {
  const DIRECT: i64 = 40;
  let n = min + DIRECT;
  let mut sum = (0.0, 0.0);
  for k in min..n {
    let (re, im) = complex_value_at(f, var, &Expr::Integer(k as i128))?;
    sum.0 += re;
    sum.1 += im;
  }
  let at_n = Expr::Integer(n as i128);
  // ∫_N^∞ f(x) dx with x = N e^u: the integrand f(N e^u) N e^u decays
  // exponentially in u and oscillates at most at a fixed rate, so
  // unit-width Gauss–Legendre panels integrate it to machine precision.
  const NODES: [(f64, f64); 5] = [
    (0.0, 0.568_888_888_888_888_9),
    (0.538_469_310_105_683, 0.478_628_670_499_366_5),
    (-0.538_469_310_105_683, 0.478_628_670_499_366_5),
    (0.906_179_845_938_664, 0.236_926_885_056_189_1),
    (-0.906_179_845_938_664, 0.236_926_885_056_189_1),
  ];
  let nf = n as f64;
  let mut integral = (0.0, 0.0);
  let mut quiet_panels = 0;
  for panel in 0..4000 {
    let mid = f64::from(panel) + 0.5;
    let mut piece = (0.0, 0.0);
    for (t, w) in NODES {
      let x = nf * (mid + t / 2.0).exp();
      let (re, im) = complex_value_at(f, var, &Expr::Real(x))?;
      piece.0 += w / 2.0 * re * x;
      piece.1 += w / 2.0 * im * x;
    }
    integral.0 += piece.0;
    integral.1 += piece.1;
    let size = integral.0.hypot(integral.1);
    if piece.0.hypot(piece.1) <= 1e-17 * size.max(1e-300) {
      quiet_panels += 1;
      if quiet_panels >= 3 {
        break;
      }
    } else {
      quiet_panels = 0;
    }
    if panel == 3999 {
      return None;
    }
  }
  let (ire, iim) = integral;
  let (fre, fim) = complex_value_at(f, var, &at_n)?;
  sum.0 += ire + fre / 2.0;
  sum.1 += iim + fim / 2.0;
  // B_2/2!, B_4/4!, B_6/6!: the corrections subtract these times the odd
  // derivatives at N.
  const COEFFS: [f64; 3] = [1.0 / 12.0, -1.0 / 720.0, 1.0 / 30240.0];
  let mut derivative = f.clone();
  for (k, c) in COEFFS.iter().enumerate() {
    let times = if k == 0 { 1 } else { 2 };
    for _ in 0..times {
      derivative =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![derivative, Expr::Identifier(var.to_string())].into(),
        })
        .ok()?;
    }
    let (dre, dim) = complex_value_at(&derivative, var, &at_n)?;
    sum.0 -= c * dre;
    sum.1 -= c * dim;
  }
  Some(sum)
}

/// NSum[expr, {i, min, max}] - Numerical summation
/// `{f, {i, a, b, s}, …}` with a step `s` other than 1 rewritten to iterate
/// a fresh variable `j` from 0 in unit steps, `i` being `a + s j` (an upper
/// bound of Infinity stays infinite); `None` without such a step.
fn unit_step_iteration(args: &[Expr]) -> Option<Vec<Expr>> {
  let Expr::List(items) = &args[1] else {
    return None;
  };
  let [Expr::Identifier(var), lo, hi, step] = items.as_slice() else {
    return None;
  };
  if matches!(step, Expr::Integer(1)) {
    return None;
  }
  let j = "NumericIterationStep$".to_string();
  let value = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      lo.clone(),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![step.clone(), Expr::Identifier(j.clone())].into(),
      },
    ]
    .into(),
  })
  .ok()?;
  let upper = if is_infinite_upper(hi) {
    hi.clone()
  } else {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Floor".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Divide".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Subtract".to_string(),
            args: vec![hi.clone(), lo.clone()].into(),
          },
          step.clone(),
        ]
        .into(),
      }]
      .into(),
    })
    .ok()?
  };
  let mut out = vec![
    substitute_variable(&args[0], var, &value),
    Expr::List(
      vec![Expr::Identifier(j), Expr::Integer(0), upper].into(),
    ),
  ];
  // Later ranges may have bounds that depend on the stepped variable.
  out.extend(args[2..].iter().map(|a| substitute_variable(a, var, &value)));
  Some(out)
}

pub fn nsum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Ok(unevaluated("NSum", args));
  }
  if let Some(unit_step) = unit_step_iteration(args) {
    return nsum_ast(&unit_step);
  }

  let body = &args[0];
  let iter_spec = &args[1];

  let items = match iter_spec {
    Expr::List(items) if items.len() >= 2 => items,
    _ => {
      return Ok(unevaluated("NSum", args));
    }
  };

  let var_name = match &items[0] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(unevaluated("NSum", args));
    }
  };

  if items.len() < 3 {
    return Ok(unevaluated("NSum", args));
  }

  let min_val = match try_eval_to_f64(&items[1]) {
    Some(v) => v as i64,
    None => {
      return Ok(unevaluated("NSum", args));
    }
  };

  // Check for infinite sum
  let is_infinite = matches!(&items[2], Expr::Identifier(s) if s == "Infinity");

  // Iterated multi-range NSum: any argument after args[1] that is itself a
  // range `{var, lo, hi}` (rather than an option Rule) is an inner summation
  // variable. Each outer term is then the inner NSum over the remaining ranges
  // (with the outer variable substituted into the body and those ranges, whose
  // bounds may depend on it, e.g. {j, 1, i}).
  let is_range = |e: &Expr| {
    matches!(e, Expr::List(it)
      if it.len() >= 3 && matches!(&it[0], Expr::Identifier(_)))
  };
  let mut inner_ranges: Vec<Expr> = Vec::new();
  let mut inner_options: Vec<Expr> = Vec::new();
  for a in &args[2..] {
    if is_range(a) && inner_options.is_empty() {
      inner_ranges.push(a.clone());
    } else {
      inner_options.push(a.clone());
    }
  }
  // A complex summand (`Log[x]/x^(2 + 2 I)`) is summed as its real and
  // imaginary parts; the summation below works with real terms.
  if inner_ranges.is_empty()
    && !crate::functions::math_ast::numeric_utils::expr_contains_head(
      body, "Re",
    )
    && !crate::functions::math_ast::numeric_utils::expr_contains_head(
      body, "Im",
    )
    && (min_val..min_val + 3).any(|i| {
      let sub = substitute_variable(body, &var_name, &Expr::Integer(i as i128));
      matches!(
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "N".to_string(),
          args: vec![sub].into(),
        }),
        Ok(v) if try_eval_to_f64(&v).is_none()
          && crate::functions::list_helpers_ast::expr_to_complex_parts(&v)
            .is_some_and(|(_, im)| im != 0.0)
      )
    })
  {
    let sum = if is_infinite_upper(&items[2]) {
      nsum_euler_maclaurin(body, &var_name, min_val)
    } else {
      let max_val = try_eval_to_f64(&items[2]).map(|v| v as i64);
      max_val.and_then(|max_val| {
        let mut terms = Vec::new();
        for i in min_val..=max_val {
          terms.push(complex_value_at(
            body,
            &var_name,
            &Expr::Integer(i as i128),
          )?);
        }
        Some(
          terms
            .iter()
            .fold((0.0, 0.0), |(a, b), (c, d)| (a + c, b + d)),
        )
      })
    };
    let Some((re, im)) = sum else {
      return Ok(unevaluated("NSum", args));
    };
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Real(re),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Identifier("I".to_string()), Expr::Real(im)].into(),
        },
      ]
      .into(),
    });
  }
  // Compute the summand at integer index `i`: a plain numeric evaluation of the
  // body, or a recursive inner NSum when there are further ranges.
  let term_at = |i: i64| -> Option<f64> {
    let sub_val = Expr::Integer(i as i128);
    let sub_body = substitute_variable(body, &var_name, &sub_val);
    if inner_ranges.is_empty() {
      let val = crate::evaluator::evaluate_expr_to_expr(&sub_body).ok()?;
      return try_eval_to_f64(&val);
    }
    let mut inner_args: Vec<Expr> =
      Vec::with_capacity(1 + inner_ranges.len() + inner_options.len());
    inner_args.push(sub_body);
    for r in &inner_ranges {
      inner_args.push(substitute_variable(r, &var_name, &sub_val));
    }
    inner_args.extend(inner_options.iter().cloned());
    let val = nsum_ast(&inner_args).ok()?;
    try_eval_to_f64(&val)
  };

  if is_infinite {
    // Numerical infinite sum using polynomial extrapolation in 1/n
    // 1. Compute partial sums at several checkpoint values of n
    // 2. Extrapolate to n → ∞ (i.e., 1/n → 0) using Neville's algorithm
    let checkpoints: Vec<i64> = vec![50, 100, 150, 200, 300, 400, 500];
    let mut running_sum = 0.0_f64;
    let mut checkpoint_idx = 0;
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    let max_n = *checkpoints.last().unwrap();
    for i in min_val..(min_val + max_n) {
      let Some(term) = term_at(i) else {
        return Ok(unevaluated("NSum", args));
      };

      if !term.is_finite() {
        break;
      }
      running_sum += term;

      let current_n = i - min_val + 1;
      if checkpoint_idx < checkpoints.len()
        && current_n == checkpoints[checkpoint_idx]
      {
        xs.push(1.0 / current_n as f64);
        ys.push(running_sum);
        checkpoint_idx += 1;
      }
    }

    if xs.is_empty() {
      return Ok(Expr::Real(running_sum));
    }

    let result = neville_extrapolation(&xs, &ys);
    return Ok(Expr::Real(result));
  }

  // Finite sum
  let max_val = match try_eval_to_f64(&items[2]) {
    Some(v) => v as i64,
    None => {
      return Ok(unevaluated("NSum", args));
    }
  };

  let mut sum = 0.0_f64;
  for i in min_val..=max_val {
    let Some(term) = term_at(i) else {
      return Ok(unevaluated("NSum", args));
    };
    sum += term;
  }

  Ok(Expr::Real(sum))
}

/// NProduct[f, {i, imin, imax}] - Numerical product
pub fn nproduct_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Ok(unevaluated("NProduct", args));
  }
  if let Some(unit_step) = unit_step_iteration(args) {
    return nproduct_ast(&unit_step);
  }

  let body = &args[0];
  let iter_spec = &args[1];

  let items = match iter_spec {
    Expr::List(items) if items.len() >= 2 => items,
    _ => {
      return Ok(unevaluated("NProduct", args));
    }
  };

  let var_name = match &items[0] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(unevaluated("NProduct", args));
    }
  };

  if items.len() < 3 {
    return Ok(unevaluated("NProduct", args));
  }

  let min_val = match try_eval_to_f64(&items[1]) {
    Some(v) => v as i64,
    None => {
      return Ok(unevaluated("NProduct", args));
    }
  };

  // Detect `Infinity` upper bound and switch to Wynn-epsilon acceleration
  // over partial products. The standard `try_eval_to_f64` doesn't recognise
  // the `Infinity` identifier, so check it explicitly first.
  let is_infinity_max =
    matches!(&items[2], Expr::Identifier(s) if s == "Infinity");

  // Iterated multi-range NProduct: any argument after args[1] that is itself a
  // range `{var, lo, hi}` (rather than an option Rule) is an inner product
  // variable. Each outer factor is then the inner NProduct over the remaining
  // ranges (with the outer variable substituted into the body and those ranges,
  // whose bounds may depend on it, e.g. {j, 1, i}).
  let is_range = |e: &Expr| {
    matches!(e, Expr::List(it)
      if it.len() >= 3 && matches!(&it[0], Expr::Identifier(_)))
  };
  let mut inner_ranges: Vec<Expr> = Vec::new();
  let mut inner_options: Vec<Expr> = Vec::new();
  for a in &args[2..] {
    if is_range(a) && inner_options.is_empty() {
      inner_ranges.push(a.clone());
    } else {
      inner_options.push(a.clone());
    }
  }

  let eval_term = |i: i64| -> Result<Option<f64>, InterpreterError> {
    let sub_val = Expr::Integer(i as i128);
    let sub_body = substitute_variable(body, &var_name, &sub_val);
    if inner_ranges.is_empty() {
      let val = crate::evaluator::evaluate_expr_to_expr(&sub_body)?;
      return Ok(try_eval_to_f64(&val));
    }
    let mut inner_args: Vec<Expr> =
      Vec::with_capacity(1 + inner_ranges.len() + inner_options.len());
    inner_args.push(sub_body);
    for r in &inner_ranges {
      inner_args.push(substitute_variable(r, &var_name, &sub_val));
    }
    inner_args.extend(inner_options.iter().cloned());
    let val = nproduct_ast(&inner_args)?;
    Ok(try_eval_to_f64(&val))
  };

  if is_infinity_max {
    // Iterate enough terms to drive the residual small; for products with
    // O(1/n^2) tail (the common case), 100 000 terms gives ~10 digits.
    // After collecting the log-partial-sums, apply Wynn-epsilon to the
    // remaining tail estimates to extract a near-machine-precision limit.
    let n_terms: usize = 20_000;
    let mut log_product = 0.0_f64;
    // Sample log-partial-sums on a geometric schedule so Wynn-epsilon has
    // enough room to extract the asymptotic constant.
    let mut samples: Vec<f64> = Vec::with_capacity(64);
    let mut next_sample = 1usize;
    let sample_factor = 1.3_f64;
    for k in 0..n_terms {
      let i = min_val + k as i64;
      let Some(term) = eval_term(i)? else {
        return Ok(unevaluated("NProduct", args));
      };
      if term <= 0.0 || !term.is_finite() {
        return Ok(unevaluated("NProduct", args));
      }
      log_product += term.ln();
      if k + 1 >= next_sample {
        samples.push(log_product);
        next_sample =
          ((next_sample as f64) * sample_factor).ceil() as usize + 1;
      }
    }
    if samples.is_empty() {
      samples.push(log_product);
    }

    // Apply Wynn's epsilon algorithm to the log-partial-sums.
    let result_log = {
      let n = samples.len();
      if n >= 3 {
        let mut eps_prev: Vec<f64> = vec![0.0; n + 1];
        let mut eps_curr: Vec<f64> = samples.clone();
        let mut best = *samples.last().unwrap();
        let mut k: usize = 0;
        while eps_curr.len() >= 2 {
          let len_next = eps_curr.len() - 1;
          let mut eps_next: Vec<f64> = vec![0.0; len_next];
          for j in 0..len_next {
            let diff = eps_curr[j + 1] - eps_curr[j];
            if diff.abs() < 1e-300 || !diff.is_finite() {
              eps_next[j] = f64::INFINITY;
            } else {
              eps_next[j] =
                eps_prev.get(j + 1).copied().unwrap_or(0.0) + 1.0 / diff;
            }
          }
          if k.is_multiple_of(2) && k > 0 {
            let candidate = eps_curr[0];
            if candidate.is_finite() {
              best = candidate;
            }
          }
          eps_prev = eps_curr;
          eps_curr = eps_next;
          k += 1;
        }
        best
      } else {
        log_product
      }
    };
    return Ok(Expr::Real(result_log.exp()));
  }

  let max_val = match try_eval_to_f64(&items[2]) {
    Some(v) => v as i64,
    None => {
      return Ok(unevaluated("NProduct", args));
    }
  };

  let mut product = 1.0_f64;
  for i in min_val..=max_val {
    let Some(term) = eval_term(i)? else {
      return Ok(unevaluated("NProduct", args));
    };
    product *= term;
  }

  Ok(Expr::Real(product))
}

/// ManhattanDistance[u, v] - Manhattan (L1) distance between two points
pub fn manhattan_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ManhattanDistance expects exactly 2 arguments".into(),
    ));
  }

  if let (Expr::List(u), Expr::List(v)) = (&args[0], &args[1]) {
    if u.len() != v.len() {
      return Err(InterpreterError::EvaluationError(
        "ManhattanDistance: vectors must have the same length".into(),
      ));
    }
    let mut abs_diffs = Vec::new();
    for (ui, vi) in u.iter().zip(v.iter()) {
      let diff = crate::evaluator::evaluate_function_call_ast(
        "Subtract",
        &[ui.clone(), vi.clone()],
      )?;
      let abs = crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
      abs_diffs.push(abs);
    }
    crate::evaluator::evaluate_function_call_ast("Plus", &abs_diffs)
  } else {
    // Scalar distance: Abs[u - v]
    let diff = crate::evaluator::evaluate_function_call_ast(
      "Subtract",
      &[args[0].clone(), args[1].clone()],
    )?;
    crate::evaluator::evaluate_function_call_ast("Abs", &[diff])
  }
}

/// Compute the error function erf(x) using BigFloat arithmetic.
/// Uses the Taylor series: erf(x) = (2/sqrt(π)) * Σ_{n=0}^{∞} (-1)^n * x^(2n+1) / (n! * (2n+1))
/// Compute erfc(x) for x > 0 using the continued fraction representation.
/// erfc(x) = exp(-x²) / (f * sqrt(π)) where f is computed via modified Lentz's method.
fn bigfloat_erfc_cf(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // Use extra guard bits for intermediate computation
  let work_bits = bits + 64;

  let half = BigFloat::from_i32(1, work_bits).div(
    &BigFloat::from_i32(2, work_bits),
    work_bits,
    rm,
  );

  // Modified Lentz's method for the continued fraction
  // erfc(x) = (exp(-x²)/sqrt(π)) * 1/(x + 1/(2x + 2/(x + 3/(2x + ...))))
  // Using: a_n = n * 0.5, b_n = x
  let mut f = x.clone();
  let mut c = x.clone();
  let mut d = BigFloat::from_i32(0, work_bits);

  let max_iterations = work_bits * 2 + 200;
  for n in 1..max_iterations {
    // a_n = n * 0.5
    let a_n = BigFloat::from_i32(n as i32, work_bits).mul(&half, work_bits, rm);

    // d = x + a_n * d
    d = x.add(&a_n.mul(&d, work_bits, rm), work_bits, rm);
    // Guard against zero
    if d.is_zero() {
      d = BigFloat::min_positive_normal(work_bits);
    }

    // c = x + a_n / c
    c = x.add(&a_n.div(&c, work_bits, rm), work_bits, rm);
    if c.is_zero() {
      c = BigFloat::min_positive_normal(work_bits);
    }

    // d = 1/d
    d = BigFloat::from_i32(1, work_bits).div(&d, work_bits, rm);
    let delta = c.mul(&d, work_bits, rm);
    f = f.mul(&delta, work_bits, rm);

    // Check convergence: |delta - 1| is negligible
    let one = BigFloat::from_i32(1, work_bits);
    let diff = delta.sub(&one, work_bits, rm).abs();
    if diff.is_zero() {
      break;
    }
    if let Some(diff_exp) = diff.exponent()
      && diff_exp < -(work_bits as i32)
    {
      break;
    }
  }

  // erfc(x) = exp(-x²) / (f * sqrt(π))
  let x2 = x.mul(x, work_bits, rm);
  let neg_x2 = x2.neg();
  let exp_neg_x2 = neg_x2.exp(work_bits, rm, cc);
  let pi = cc.pi(work_bits, rm);
  let sqrt_pi = pi.sqrt(work_bits, rm);
  let denom = f.mul(&sqrt_pi, work_bits, rm);
  exp_neg_x2.div(&denom, bits, rm)
}

fn bigfloat_erf(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if x.is_zero() {
    return BigFloat::from_i32(0, bits);
  }

  // erf is odd: erf(-x) = -erf(x)
  let is_negative = x.is_negative();
  let x_abs = x.abs();

  // For large |x|, use the continued fraction for erfc(x) and compute erf = 1 - erfc.
  // The Taylor series suffers from catastrophic cancellation for large arguments.
  let four = BigFloat::from_i32(4, bits);
  if x_abs.cmp(&four) == Some(1) {
    // |x| > 4: use continued fraction
    let erfc_val = bigfloat_erfc_cf(&x_abs, bits, rm, cc);
    let one = BigFloat::from_i32(1, bits);
    let result = one.sub(&erfc_val, bits, rm);
    return if is_negative { result.neg() } else { result };
  }

  // For small |x| (≤ 4), use the Taylor series with extra guard bits to handle cancellation.
  // With |x| ≤ 4, the peak term is ~exp(x²/2) ≈ exp(8) ≈ 2981, needing ~12 extra bits.
  // We use 64 guard bits for safety.
  let work_bits = bits + 64;

  // Taylor series: term_0 = x, term_n = term_{n-1} * x^2 / n
  // contribution_n = term_n / (2n+1), alternating sign
  // sum = Σ (-1)^n * contribution_n
  let x2 = x_abs.mul(&x_abs, work_bits, rm);
  let mut term = x_abs.clone();
  let mut sum = x_abs.clone();

  let max_iterations = work_bits * 2 + 100;
  for n in 1..max_iterations {
    term = term.mul(&x2, work_bits, rm);
    let n_bf = BigFloat::from_i32(n as i32, work_bits);
    term = term.div(&n_bf, work_bits, rm);

    let denom = BigFloat::from_i32((2 * n + 1) as i32, work_bits);
    let contribution = term.div(&denom, work_bits, rm);

    if n % 2 == 1 {
      sum = sum.sub(&contribution, work_bits, rm);
    } else {
      sum = sum.add(&contribution, work_bits, rm);
    }

    if contribution.is_zero() {
      break;
    }
    if let (Some(c_exp), Some(s_exp)) =
      (contribution.exponent(), sum.exponent())
      && s_exp - c_exp > (work_bits as i32)
    {
      break;
    }
  }

  // Multiply by 2/sqrt(π), round to final precision
  let two = BigFloat::from_i32(2, work_bits);
  let pi = cc.pi(work_bits, rm);
  let sqrt_pi = pi.sqrt(work_bits, rm);
  let factor = two.div(&sqrt_pi, work_bits, rm);
  let result = sum.mul(&factor, bits, rm);

  if is_negative { result.neg() } else { result }
}

/// Compute erfc(x) with arbitrary precision.
/// For large x, uses continued fraction directly. For small x, uses 1 - erf(x).
fn bigfloat_erfc(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if x.is_zero() {
    return BigFloat::from_i32(1, bits);
  }

  let is_negative = x.is_negative();
  let x_abs = x.abs();

  let four = BigFloat::from_i32(4, bits);
  let result = if x_abs.cmp(&four) == Some(1) {
    // |x| > 4: use continued fraction directly for best precision
    bigfloat_erfc_cf(&x_abs, bits, rm, cc)
  } else {
    // |x| <= 4: compute via 1 - erf(x)
    let erf_val = bigfloat_erf(&x_abs, bits, rm, cc);
    let one = BigFloat::from_i32(1, bits);
    one.sub(&erf_val, bits, rm)
  };

  // erfc(-x) = 2 - erfc(x)
  if is_negative {
    let two = BigFloat::from_i32(2, bits);
    two.sub(&result, bits, rm)
  } else {
    result
  }
}

/// Machine `Erfc[x]` for 0.25 ≤ x < 27, rounded from the arbitrary-precision
/// value: worked out at 160 bits, since below x = 4 it is 1 - erf(x) and
/// that subtraction cancels up to ~30 of them.
pub(crate) fn erfc_f64_accurate(x: f64) -> Option<f64> {
  use astro_float::{BigFloat, RoundingMode};
  with_thread_consts(|cc| {
    let z = BigFloat::from_f64(x, 160);
    let r = bigfloat_erfc(&z, 160, RoundingMode::ToEven, cc);
    bigfloat_to_f64_approx(&r)
  })
}

/// Compute erfi(x) with arbitrary precision.
/// erfi(x) = (2/sqrt(pi)) * sum_{n=0}^{inf} x^(2n+1) / (n! * (2n+1))
/// Unlike erf, the terms do NOT alternate in sign.
fn bigfloat_erfi(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if x.is_zero() {
    return BigFloat::from_i32(0, bits);
  }

  // erfi is odd: erfi(-x) = -erfi(x)
  let is_negative = x.is_negative();
  let x_abs = x.abs();

  let work_bits = bits + 64;

  // Taylor series: term_0 = x, term_n = term_{n-1} * x^2 / n
  // contribution_n = term_n / (2n+1), all terms positive (no alternating sign)
  let x2 = x_abs.mul(&x_abs, work_bits, rm);
  let mut term = x_abs.clone();
  let mut sum = x_abs.clone();

  let max_iterations = work_bits * 2 + 100;
  for n in 1..max_iterations {
    term = term.mul(&x2, work_bits, rm);
    let n_bf = BigFloat::from_i32(n as i32, work_bits);
    term = term.div(&n_bf, work_bits, rm);

    let denom = BigFloat::from_i32((2 * n + 1) as i32, work_bits);
    let contribution = term.div(&denom, work_bits, rm);

    sum = sum.add(&contribution, work_bits, rm);

    if contribution.is_zero() {
      break;
    }
    if let (Some(c_exp), Some(s_exp)) =
      (contribution.exponent(), sum.exponent())
      && s_exp - c_exp > (work_bits as i32)
    {
      break;
    }
  }

  // Multiply by 2/sqrt(π), round to final precision
  let two = BigFloat::from_i32(2, work_bits);
  let pi = cc.pi(work_bits, rm);
  let sqrt_pi = pi.sqrt(work_bits, rm);
  let factor = two.div(&sqrt_pi, work_bits, rm);
  let result = sum.mul(&factor, bits, rm);

  if is_negative { result.neg() } else { result }
}

/// `ExpIntegralE[ν, z]`, `LogIntegral[x]` and `Gamma[a, z]` for real
/// arguments, as (re, im). Gamma with one argument is the Gamma function,
/// which `expr_to_bigfloat` handles before reaching here.
fn exp_integral_family_bigfloat(
  name: &str,
  args: &[Expr],
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(astro_float::BigFloat, astro_float::BigFloat), InterpreterError> {
  use crate::functions::math_ast as m;
  let fail = || {
    InterpreterError::EvaluationError(format!(
      "N: cannot evaluate {name}[...] to arbitrary precision"
    ))
  };
  match (name, args) {
    ("ExpIntegralE", [nu, z]) => {
      let nu = expr_to_bigfloat(nu, bits, rm, cc)?;
      let z = expr_to_bigfloat(z, bits, rm, cc)?;
      m::exp_integral_e_bigfloat(&nu, &z, bits, rm, cc).ok_or_else(fail)
    }
    ("LogIntegral", [x]) => {
      let x = expr_to_bigfloat(x, bits, rm, cc)?;
      if x.is_positive() {
        let v = m::log_integral_bigfloat(&x, bits, rm, cc).ok_or_else(fail)?;
        return Ok((v, astro_float::BigFloat::from_i32(0, bits)));
      }
      if x.is_zero() {
        return Err(fail());
      }
      // li(x) = Ei(ln|x| + iπ) for x < 0.
      let t = x.abs().ln(bits + 128, rm, cc);
      let pi = cc.pi(bits + 128, rm);
      Ok(complex_exp_integral_ei(&t, &pi, bits, rm, cc))
    }
    ("Gamma", [a, z]) => {
      let a = expr_to_bigfloat(a, bits, rm, cc)?;
      let z = expr_to_bigfloat(z, bits, rm, cc)?;
      m::upper_incomplete_gamma_bigfloat(&a, &z, bits, rm, cc).ok_or_else(fail)
    }
    _ => Err(fail()),
  }
}

/// Compute the complex exponential integral Ei(z) using BigFloat arithmetic.
/// For complex z = a + bi: Ei(z) = γ + Log(z) + Σ_{n=1}^{∞} z^n / (n * n!)
/// where γ is the Euler-Mascheroni constant and Log is the complex logarithm.
pub(crate) fn complex_exp_integral_ei(
  a: &astro_float::BigFloat,
  b: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> (astro_float::BigFloat, astro_float::BigFloat) {
  use astro_float::BigFloat;

  // Use many extra guard bits to ensure the rounding to `bits` is correct
  let work_bits = bits + 256;

  // γ (Euler-Mascheroni constant) - purely real
  let euler_gamma = compute_euler_gamma(work_bits, rm, cc);

  // Complex Log(z) = ln|z| + i*arg(z)
  // |z| = sqrt(a^2 + b^2), arg(z) = atan2(b, a)
  let abs_sq =
    a.mul(a, work_bits, rm)
      .add(&b.mul(b, work_bits, rm), work_bits, rm);
  let abs_z = abs_sq.sqrt(work_bits, rm);
  let ln_abs_z = abs_z.ln(work_bits, rm, cc);
  // atan2(b, a) via atan and quadrant adjustment
  let arg_z = {
    let zero = BigFloat::from_i32(0, work_bits);
    let pi = cc.pi(work_bits, rm);
    if !a.is_zero() {
      let ratio = b.div(a, work_bits, rm);
      let atan_val = ratio.atan(work_bits, rm, cc);
      if a.is_positive() {
        atan_val
      } else if b.is_negative() {
        atan_val.sub(&pi, work_bits, rm)
      } else {
        atan_val.add(&pi, work_bits, rm)
      }
    } else if b.is_positive() {
      pi.div(&BigFloat::from_i32(2, work_bits), work_bits, rm)
    } else if b.is_negative() {
      pi.div(&BigFloat::from_i32(2, work_bits), work_bits, rm)
        .neg()
    } else {
      zero
    }
  };

  // Start sum: γ + ln|z| for real part, arg(z) for imaginary part
  let mut sum_re = euler_gamma.add(&ln_abs_z, work_bits, rm);
  let mut sum_im = arg_z;

  // Power series: Σ_{n=1}^{∞} z^n / (n * n!)
  // Track z^n as (pow_re, pow_im), start with z^1 = (a, b)
  let mut pow_re = a.clone();
  let mut pow_im = b.clone();
  let mut factorial = BigFloat::from_i32(1, work_bits); // n!

  let max_iterations = bits * 2 + 100;
  for n in 1..max_iterations {
    let n_bf = BigFloat::from_i32(n as i32, work_bits);
    if n > 1 {
      // z^n = z^(n-1) * z
      let new_re = pow_re.mul(a, work_bits, rm).sub(
        &pow_im.mul(b, work_bits, rm),
        work_bits,
        rm,
      );
      let new_im = pow_re.mul(b, work_bits, rm).add(
        &pow_im.mul(a, work_bits, rm),
        work_bits,
        rm,
      );
      pow_re = new_re;
      pow_im = new_im;
      factorial = factorial.mul(&n_bf, work_bits, rm);
    }
    // term = z^n / (n * n!)
    let denom = n_bf.mul(&factorial, work_bits, rm);
    let term_re = pow_re.div(&denom, work_bits, rm);
    let term_im = pow_im.div(&denom, work_bits, rm);

    sum_re = sum_re.add(&term_re, work_bits, rm);
    sum_im = sum_im.add(&term_im, work_bits, rm);

    // Check convergence
    let term_abs_sq = term_re.mul(&term_re, work_bits, rm).add(
      &term_im.mul(&term_im, work_bits, rm),
      work_bits,
      rm,
    );
    if term_abs_sq.is_zero() {
      break;
    }
    let sum_abs_sq = sum_re.mul(&sum_re, work_bits, rm).add(
      &sum_im.mul(&sum_im, work_bits, rm),
      work_bits,
      rm,
    );
    if let (Some(t_exp), Some(s_exp)) =
      (term_abs_sq.exponent(), sum_abs_sq.exponent())
    {
      // Compare squared magnitudes, so convergence is 2x the bit threshold
      if s_exp - t_exp > (2 * work_bits as i32) {
        break;
      }
    }
  }

  // Perform final addition at target `bits` precision to truncate mantissa
  let zero = BigFloat::from_i32(0, bits);
  let result_re = sum_re.add(&zero, bits, rm);
  let result_im = sum_im.add(&zero, bits, rm);

  (result_re, result_im)
}

/// `floor(x · 2^frac_bits)` of a BigFloat, exactly.
fn bigfloat_to_fixed(
  x: &astro_float::BigFloat,
  frac_bits: usize,
) -> num_bigint::BigInt {
  use num_bigint::{BigInt, BigUint, Sign};
  let Some((words, _, sign, exp, _)) = x.as_raw_parts() else {
    return BigInt::from(0);
  };
  let mut mantissa = BigUint::from(0u8);
  for &word in words.iter().rev() {
    mantissa = (mantissa << astro_float::Word::BITS) | BigUint::from(word);
  }
  let shift = exp as i64
    - (words.len() * astro_float::Word::BITS as usize) as i64
    + frac_bits as i64;
  let scaled = if shift >= 0 {
    mantissa << shift as usize
  } else {
    mantissa >> (-shift) as usize
  };
  let sign = if sign == astro_float::Sign::Neg {
    Sign::Minus
  } else {
    Sign::Plus
  };
  BigInt::from_biguint(sign, scaled)
}

/// The BigFloat `n · 2^-frac_bits` rounded to `bits` bits.
fn fixed_to_bigfloat(
  n: &num_bigint::BigInt,
  frac_bits: usize,
  bits: usize,
  rm: astro_float::RoundingMode,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;
  use num_bigint::Sign;
  let (sign, magnitude) = n.clone().into_parts();
  if sign == Sign::NoSign {
    return BigFloat::from_i32(0, bits);
  }
  const WORD_BYTES: usize = (astro_float::Word::BITS / 8) as usize;
  // Left-align the integer in whole words so the mantissa's top bit is set.
  let len = magnitude.bits() as usize;
  let total = len.div_ceil(astro_float::Word::BITS as usize)
    * astro_float::Word::BITS as usize;
  let bytes = (magnitude << (total - len)).to_bytes_le();
  let words: Vec<astro_float::Word> = bytes
    .chunks(WORD_BYTES)
    .map(|chunk| {
      let mut buf = [0u8; WORD_BYTES];
      buf[..chunk.len()].copy_from_slice(chunk);
      astro_float::Word::from_le_bytes(buf)
    })
    .collect();
  let sign = if sign == Sign::Minus {
    astro_float::Sign::Neg
  } else {
    astro_float::Sign::Pos
  };
  let exact = BigFloat::from_words(
    &words,
    sign,
    (len as i64 - frac_bits as i64) as astro_float::Exponent,
  );
  exact.add(&BigFloat::from_i32(0, bits), bits, rm)
}

/// The Euler–Mascheroni constant γ to `bits` bits: from a 105-digit literal
/// up to 340 bits, beyond that by Brent and McMillan's algorithm B1,
/// γ = U/V with U = Σ A_k, V = Σ B_k, B_k = B_{k−1} n²/k²,
/// A_k = (A_{k−1} n²/k + B_k)/k, A_0 = −ln n — whose error is O(e^{−4n}).
pub(crate) fn compute_euler_gamma(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if bits <= 340 {
    let gamma_str = "0.5772156649015328606065120900824024310421593359399235988057672348848677267776646709369470632917467495";
    return BigFloat::parse(gamma_str, astro_float::Radix::Dec, bits, rm, cc);
  }
  thread_local! {
    static CACHE: std::cell::RefCell<std::collections::HashMap<usize, BigFloat>> =
      std::cell::RefCell::new(std::collections::HashMap::new());
  }
  if let Some(v) = CACHE.with(|c| c.borrow().get(&bits).cloned()) {
    return v;
  }
  // Every term is a rational multiple of the previous one, so the series
  // runs in fixed point on integers (`w` fractional bits): a multiplication
  // and a division by machine integers per term instead of full-precision
  // BigFloat divisions.
  use num_traits::Zero;
  let w = bits + 64;
  let n = (bits as f64 * std::f64::consts::LN_2 / 4.0).ceil() as u64 + 2;
  let n2 = n * n;
  let mut a = -bigfloat_to_fixed(&BigFloat::from_u64(n, w).ln(w, rm, cc), w);
  let mut b = num_bigint::BigInt::from(1) << w;
  let mut u = a.clone();
  let mut v = b.clone();
  for k in 1u64.. {
    let k2 = k * k;
    b = b * n2 / k2;
    a = (a * n2 / k + &b) / k;
    u += &a;
    v += &b;
    if k > n && a.is_zero() && b.is_zero() {
      break;
    }
  }
  let gamma = fixed_to_bigfloat(&((u << w) / v), w, bits, rm);
  CACHE.with(|c| c.borrow_mut().insert(bits, gamma.clone()));
  gamma
}

/// Catalan's constant G to `bits` bits: a 105-digit literal up to 340 bits,
/// beyond that Ramanujan's series
/// G = π/8 ln(2 + √3) + 3/8 Σ_{k≥0} 1/((2k+1)² C(2k, k)), two bits a term.
fn compute_catalan(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if bits <= 340 {
    // Catalan's constant G (OEIS A006752, ~105 digits)
    let catalan_str = "0.91596559417721901505460351493238411077414937428167213426649811962176301977625476947935651292611510624857";
    return BigFloat::parse(catalan_str, astro_float::Radix::Dec, bits, rm, cc);
  }
  let w = bits + 64;
  let mut central = BigFloat::from_i32(1, w); // C(2k, k)
  let mut sum = BigFloat::from_i32(0, w);
  for k in 0u64.. {
    let odd = BigFloat::from_u64(2 * k + 1, w);
    let term = central.mul(&odd.mul(&odd, w, rm), w, rm).reciprocal(w, rm);
    sum = sum.add(&term, w, rm);
    if term.exponent().unwrap_or(0) < sum.exponent().unwrap_or(0) - w as i32 {
      break;
    }
    // C(2k+2, k+1) = C(2k, k) · 2(2k+1)/(k+1)
    central = central
      .mul(&BigFloat::from_u64(2 * (2 * k + 1), w), w, rm)
      .div(&BigFloat::from_u64(k + 1, w), w, rm);
  }
  let three = BigFloat::from_i32(3, w);
  let eight = BigFloat::from_i32(8, w);
  let log_term = three
    .sqrt(w, rm)
    .add(&BigFloat::from_i32(2, w), w, rm)
    .ln(w, rm, cc);
  cc.pi(w, rm)
    .mul(&log_term, w, rm)
    .add(&three.mul(&sum, w, rm), w, rm)
    .div(&eight, bits, rm)
}

/// The Glaisher–Kinkelin constant A to `bits` bits: a 102-digit literal up
/// to 330 bits, beyond that ln A = (γ + ln 2π)/12 − ζ′(2)/(2π²), with
/// ζ′(2) = −Σ ln k/k² summed directly below N and by Euler–Maclaurin above,
/// where f = ln x/x² has f⁽ᵐ⁾(x) = (−1)ᵐ (m+1)! x^(−m−2) (ln x − H_{m+1} + 1).
fn compute_glaisher(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if bits <= 330 {
    // Glaisher–Kinkelin constant A (OEIS A074962, ~102 digits).
    let glaisher_str = "1.28242712910062263687534256886979172776768892732500119206374002174040630883966455201507550549353290381";
    return BigFloat::parse(
      glaisher_str,
      astro_float::Radix::Dec,
      bits,
      rm,
      cc,
    );
  }
  let w = bits + 64;
  let one = BigFloat::from_i32(1, w);
  // The Euler–Maclaurin terms shrink like (2j)!/(2πN)^(2j) until j ≈ πN.
  let n = (w as f64 * std::f64::consts::LN_2 / (2.0 * std::f64::consts::PI))
    .ceil() as u64
    + 2;
  // −ζ′(2) = Σ_{k=2}^{N−1} ln k/k² + Σ_{k≥N} f(k).
  let mut sum = BigFloat::from_i32(0, w);
  for k in 2..n {
    let k_bf = BigFloat::from_u64(k, w);
    sum = sum.add(
      &k_bf.ln(w, rm, cc).div(&k_bf.mul(&k_bf, w, rm), w, rm),
      w,
      rm,
    );
  }
  let n_bf = BigFloat::from_u64(n, w);
  let ln_n = n_bf.ln(w, rm, cc);
  let n2 = n_bf.mul(&n_bf, w, rm);
  // ∫_N^∞ f = (ln N + 1)/N, plus f(N)/2.
  sum = sum
    .add(&ln_n.add(&one, w, rm).div(&n_bf, w, rm), w, rm)
    .add(&ln_n.div(&n2.add(&n2, w, rm), w, rm), w, rm);
  // − Σ_j B₂ⱼ/(2j)! f⁽²ʲ⁻¹⁾(N) = Σ_j B₂ⱼ N^(−2j−1) (ln N − H₂ⱼ + 1).
  let mut harmonic = BigFloat::from_i32(0, w); // H_{2j}
  let mut n_pow = n_bf.clone(); // N^(2j+1)
  for j in 1usize.. {
    let two_j = (2 * j) as u64;
    harmonic = harmonic
      .add(&one.div(&BigFloat::from_u64(two_j - 1, w), w, rm), w, rm)
      .add(&one.div(&BigFloat::from_u64(two_j, w), w, rm), w, rm);
    n_pow = n_pow.mul(&n2, w, rm);
    let b2j = bernoulli_over_index(j, w, rm, cc).mul(
      &BigFloat::from_u64(two_j, w),
      w,
      rm,
    );
    let term = b2j
      .mul(&ln_n.sub(&harmonic, w, rm).add(&one, w, rm), w, rm)
      .div(&n_pow, w, rm);
    sum = sum.add(&term, w, rm);
    if term.is_zero()
      || term.exponent().unwrap_or(0) < sum.exponent().unwrap_or(0) - w as i32
    {
      break;
    }
  }
  let zeta_prime_2 = sum.neg();
  let pi = cc.pi(w, rm);
  let two_pi = pi.add(&pi, w, rm);
  let gamma = compute_euler_gamma(w, rm, cc);
  let ln_a = gamma
    .add(&two_pi.ln(w, rm, cc), w, rm)
    .div(&BigFloat::from_i32(12, w), w, rm)
    .sub(&zeta_prime_2.div(&two_pi.mul(&pi, w, rm), w, rm), w, rm);
  ln_a.exp(bits, rm, cc)
}

/// Khinchin's constant K₀ to `bits` bits: a 102-digit literal up to 330 bits,
/// beyond that ln K₀ · ln 2 = Σ_{n≥1} ζ(2n, 2)/n · A_n with
/// A_n = Σ_{k=1}^{2n−1} (−1)^(k+1)/k. Since Σ_{n≥1} A_n x^(2n)/n =
/// −ln(1 − x) ln(1 + x), the terms k < N of ζ(2n, 2) sum in closed form
/// (Bailey, Borwein and Crandall):
/// ln K₀ · ln 2 = −Σ_{k=2}^{N−1} ln(1 − 1/k) ln(1 + 1/k) + Σ_{n≥1} ζ(2n, N)/n · A_n,
/// where ζ(2n, N) shrinks like N^(−2n), so it needs only about
/// bits/(2 log₂ N) terms, each to correspondingly fewer bits. N stays above
/// the Euler–Maclaurin shift of [`hurwitz_zeta_bigfloat`], so no ζ(2n, N)
/// sums any terms directly.
fn compute_khinchin(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if bits <= 330 {
    // Khinchin's constant K_0 (OEIS A002210, ~102 digits).
    let khinchin_str = "2.68545200106530644530971483548179569382038229399446295305115234555721885953715200280114117493184769799";
    return BigFloat::parse(
      khinchin_str,
      astro_float::Radix::Dec,
      bits,
      rm,
      cc,
    );
  }
  let w = bits + 64;
  let one = BigFloat::from_i32(1, w);
  let shift = (w / 4).max(64) as u64;
  let shift_bf = BigFloat::from_u64(shift, w);
  let log2_shift = (shift as f64).log2();
  // ln k for k ≤ N: only a prime needs a logarithm, a composite k = p·m
  // adds up ln p + ln m.
  let mut ln_table: Vec<BigFloat> = vec![BigFloat::from_i32(0, w); 2];
  for k in 2..=shift {
    let factor = (2..=k).take_while(|p| p * p <= k).find(|p| k % p == 0);
    let ln_k = match factor {
      Some(p) => ln_table[p as usize].add(&ln_table[(k / p) as usize], w, rm),
      None => BigFloat::from_u64(k, w).ln(w, rm, cc),
    };
    ln_table.push(ln_k);
  }
  // −Σ_{k=2}^{N−1} ln(1 − 1/k) ln(1 + 1/k), from ln(k − 1), ln k, ln(k + 1).
  let mut sum = BigFloat::from_i32(0, w);
  for k in 2..shift as usize {
    let product = ln_table[k].sub(&ln_table[k - 1], w, rm).mul(
      &ln_table[k + 1].sub(&ln_table[k], w, rm),
      w,
      rm,
    );
    sum = sum.add(&product, w, rm);
  }
  let mut alternating = BigFloat::from_i32(0, w); // A_n
  for n in 1u64.. {
    let k_odd = BigFloat::from_u64(2 * n - 1, w);
    if n > 1 {
      alternating = alternating.sub(
        &one.div(&BigFloat::from_u64(2 * n - 2, w), w, rm),
        w,
        rm,
      );
    }
    alternating = alternating.add(&one.div(&k_odd, w, rm), w, rm);
    let s = BigFloat::from_u64(2 * n, w);
    // ζ(2n, N) ≈ N^(−2n) only has to reach the 2^(−w) of the sum.
    let term_bits =
      (w as f64 - 2.0 * n as f64 * log2_shift).max(0.0) as usize + 64;
    let Some(zeta_tail) =
      hurwitz_zeta_bigfloat(&s, &shift_bf, false, term_bits, rm, cc)
    else {
      break;
    };
    let term =
      zeta_tail
        .mul(&alternating, w, rm)
        .div(&BigFloat::from_u64(n, w), w, rm);
    sum = sum.add(&term, w, rm);
    if term.exponent().unwrap_or(0) < sum.exponent().unwrap_or(0) - w as i32 {
      break;
    }
  }
  sum.div(&cc.ln_2(w, rm), w, rm).exp(bits, rm, cc)
}

/// The decimal expansion of the base-b Champernowne constant — the digits
/// of 1, 2, 3, … concatenated after the radix point in base b — truncated
/// to `decimal_digits` decimal places, as a "0.ddd…" string. Computed
/// exactly: enough base-b digits accumulate into a big integer M so that
/// M/b^K determines the requested decimal digits, then long-divided.
fn champernowne_decimal_digits(base: i128, decimal_digits: usize) -> String {
  let k_needed = ((decimal_digits as f64 + 2.0) * std::f64::consts::LN_10
    / (base as f64).ln())
  .ceil() as usize
    + 2;
  let big_base = BigInt::from(base);
  let mut m = BigInt::from(0);
  let mut count = 0usize;
  let mut n: i128 = 1;
  'outer: loop {
    let mut digits = Vec::new();
    let mut v = n;
    while v > 0 {
      digits.push(v % base);
      v /= base;
    }
    for &d in digits.iter().rev() {
      m = &m * &big_base + BigInt::from(d);
      count += 1;
      if count >= k_needed {
        break 'outer;
      }
    }
    n += 1;
  }
  let ten_p = BigInt::from(10).pow(decimal_digits as u32);
  let b_k = big_base.pow(count as u32);
  let d = (&m * &ten_p) / &b_k;
  let s = d.to_string();
  format!(
    "0.{}{}",
    "0".repeat(decimal_digits.saturating_sub(s.len())),
    s
  )
}

/// The machine-precision value of ChampernowneNumber[base].
pub(crate) fn champernowne_f64(base: i128) -> f64 {
  champernowne_decimal_digits(base, 20).parse().unwrap_or(0.0)
}

fn compute_champernowne(
  base: i128,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;
  // Generous guard digits: the parsed decimal string must exceed the
  // BigFloat's internal precision comfortably, like the ~105-digit
  // hardcoded constant strings, or the displayed digit tail diverges from
  // wolframscript.
  let decimal_digits =
    (bits as f64 * std::f64::consts::LOG10_2).ceil() as usize + 20;
  let s = champernowne_decimal_digits(base, decimal_digits);
  BigFloat::parse(&s, astro_float::Radix::Dec, bits, rm, cc)
}

/// ListFourierSequenceTransform[{u1, …, un}, ω(, k), FourierParameters ->
/// {a, b}] — discrete-time Fourier transform
/// |b/(2 Pi)|^((1 - a)/2) Sum[u_r E^(-I b (r + k - 1) ω), {r, n}], with
/// {a, b} = {1, 1} and k = 0 by default.
pub fn list_fourier_sequence_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated_result =
    || Ok(unevaluated("ListFourierSequenceTransform", args));
  let split = args
    .iter()
    .position(|a| matches!(a, Expr::Rule { .. }))
    .unwrap_or(args.len());
  let (positional, opts) = args.split_at(split);
  if positional.len() != 2 && positional.len() != 3 {
    return unevaluated_result();
  }
  let (mut a, mut b) = (Expr::Integer(1), Expr::Integer(1));
  for opt in opts {
    match crate::functions::expr_form::rule_parts(opt) {
      Some((Expr::Identifier(n), Expr::List(ab), _))
        if n == "FourierParameters" && ab.len() == 2 =>
      {
        a = ab[0].clone();
        b = ab[1].clone();
      }
      _ => return unevaluated_result(),
    }
  }
  // ListFourierSequenceTransform[list, ω, n0]: list starts at index n0.
  let shift: i128 = match positional.get(2) {
    None => 0,
    Some(Expr::Integer(n)) => *n,
    Some(_) => return unevaluated_result(),
  };

  let Expr::List(list) = &positional[0] else {
    return unevaluated_result();
  };

  if list.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  let omega = &positional[1];
  let default_parameters =
    matches!((&a, &b), (Expr::Integer(1), Expr::Integer(1)));

  // Sum[u_k E^(-I b ω k)]
  use crate::evaluator::evaluate_expr_to_expr;

  let mut terms: Vec<Expr> = Vec::new();
  for (k, coeff) in list.iter().enumerate() {
    if matches!(coeff, Expr::Integer(0)) {
      continue;
    }
    let k = k as i128 + shift;

    if k == 0 {
      terms.push(coeff.clone());
    } else {
      let exponent = times(vec![
        Expr::Integer(-k),
        id_expr("I"),
        b.clone(),
        omega.clone(),
      ]);
      terms.push(times2(coeff.clone(), pow2(id_expr("E"), exponent)));
    }
  }

  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }

  let mut sum = if terms.len() == 1 {
    terms.pop().unwrap()
  } else {
    plus(terms)
  };
  if !default_parameters {
    // |b/(2 Pi)|^((1 - a)/2)
    let scale = pow2(
      call1(
        "Abs",
        times2(
          b.clone(),
          pow2(times2(Expr::Integer(2), id_expr("Pi")), Expr::Integer(-1)),
        ),
      ),
      times2(
        plus(vec![Expr::Integer(1), times2(Expr::Integer(-1), a)]),
        make_rational(1, 2),
      ),
    );
    sum = times2(scale, evaluate_expr_to_expr(&sum)?);
  }

  evaluate_expr_to_expr(&sum)
}

/// FourierSequenceTransform[f, n, omega] — discrete-time Fourier transform of a
/// sequence with finite support, `Sum[f E^(I b n omega), n]` scaled by
/// `Sqrt[Abs[b]/(2 Pi)^(1 - a)]` for `FourierParameters -> {a, b}`.
/// Sequences whose support is not finite stay unevaluated.
pub fn fourier_sequence_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::evaluator::evaluate_expr_to_expr;
  use crate::syntax::substitute_variable;

  let unevaluated_result = || Ok(unevaluated("FourierSequenceTransform", args));
  if args.len() < 3 {
    return unevaluated_result();
  }
  let Expr::Identifier(var) = &args[1] else {
    return unevaluated_result();
  };
  let Ok((a, b)) = parse_fourier_parameters(&args[3..]) else {
    return unevaluated_result();
  };
  let (a, b) = if args.len() > 3 { (a, b) } else { (1.0, 1.0) };
  if a.fract() != 0.0 || b.fract() != 0.0 || b == 0.0 {
    return unevaluated_result();
  }
  let (a, b) = (a as i128, b as i128);

  // Probe a window of integers: the support must lie well inside it.
  const RADIUS: i128 = 64;
  const MARGIN: i128 = 16;
  let mut values: Vec<(i128, Expr)> = Vec::new();
  for k in -RADIUS..=RADIUS {
    let v = evaluate_expr_to_expr(&substitute_variable(
      &args[0],
      var,
      &Expr::Integer(k),
    ))?;
    if matches!(v, Expr::Integer(0)) {
      continue;
    }
    if k.abs() > RADIUS - MARGIN {
      return unevaluated_result();
    }
    values.push((k, v));
  }

  let omega = &args[2];
  let terms: Vec<Expr> = values
    .into_iter()
    .map(|(k, v)| {
      let exponent =
        times(vec![Expr::Integer(b * k), id_expr("I"), omega.clone()]);
      times2(v, pow2(id_expr("E"), exponent))
    })
    .collect();
  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }
  let mut sum = plus(terms);
  if a != 1 || b.abs() != 1 {
    // Sqrt[Abs[b]/(2 Pi)^(1 - a)]
    let scale = pow2(
      times2(
        Expr::Integer(b.abs()),
        pow2(
          times2(Expr::Integer(2), id_expr("Pi")),
          Expr::Integer(a - 1),
        ),
      ),
      make_rational(1, 2),
    );
    sum = times2(scale, sum);
  }
  evaluate_expr_to_expr(&sum)
}

/// The defining expression of a window function at `x`, built symbolically
/// so the evaluator's exact trig tables can fold it: `CosineWindow[1/4]`
/// becomes `Cos[Pi/4]` → `1/Sqrt[2]` and `LanczosWindow[1/4]` becomes
/// `Sinc[Pi/2]` → `2/Pi`, instead of being rounded back from a float.
fn window_expr(name: &str, x: &Expr) -> Option<Expr> {
  // `k Pi x`, the argument of the trigonometric terms.
  let pi_x = |k: i128| times(vec![Expr::Integer(k), id_expr("Pi"), x.clone()]);
  // Σ aₖ Cos[2πkx], the cosine-sum windows' common shape.
  let cosine_sum = |numerators: &[i128], denom: i128| {
    let terms: Vec<Expr> = numerators
      .iter()
      .enumerate()
      .map(|(k, &n)| {
        let coeff = make_rational(n, denom);
        if k == 0 {
          coeff
        } else {
          times(vec![coeff, call1("Cos", pi_x(2 * k as i128))])
        }
      })
      .collect();
    plus(terms)
  };
  // 1 - 4x², the parabolic windows' common shape.
  let parabola = || {
    plus(vec![
      Expr::Integer(1),
      times(vec![Expr::Integer(-4), pow(x.clone(), Expr::Integer(2))]),
    ])
  };
  Some(match name {
    "HammingWindow" => cosine_sum(&[25, 21], 46),
    "HannWindow" => cosine_sum(&[1, 1], 2),
    "BlackmanWindow" => cosine_sum(&[21, 25, 4], 50),
    "ExactBlackmanWindow" => cosine_sum(&[7938, 9240, 1430], 18608),
    "DirichletWindow" => Expr::Integer(1),
    "BartlettWindow" => plus(vec![
      Expr::Integer(1),
      times(vec![Expr::Integer(-2), call1("Abs", x.clone())]),
    ]),
    "WelchWindow" => parabola(),
    "ConnesWindow" => pow(parabola(), Expr::Integer(2)),
    "CosineWindow" => call1("Cos", pi_x(1)),
    "LanczosWindow" => call1("Sinc", pi_x(2)),
    _ => return None,
  })
}

/// Generic window function evaluator.
/// All window functions are defined on [-1/2, 1/2] and return 0 outside.
/// Machine reals evaluate in floating point; every other numeric argument
/// goes through the symbolic form, so exact input gives exact output.
pub fn window_function_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(unevaluated(name, args));
  }

  let Some(x) = try_eval_to_f64(&args[0]) else {
    return Ok(unevaluated(name, args));
  };
  let is_real = matches!(&args[0], Expr::Real(_));

  if x.abs() > 0.5 {
    return Ok(if is_real {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }

  if !is_real {
    return match window_expr(name, &args[0]) {
      Some(e) => crate::evaluator::evaluate_expr_to_expr(&e),
      None => Ok(unevaluated(name, args)),
    };
  }

  let pi = std::f64::consts::PI;
  let val = match name {
    "HammingWindow" => 25.0 / 46.0 + 21.0 / 46.0 * (2.0 * pi * x).cos(),
    "HannWindow" => f64::midpoint(1.0, (2.0 * pi * x).cos()),
    // Integer numerators summed and divided once at the end, like the
    // other cosine-sum windows: the 0.42 / 0.5 / 0.08 literals lose the
    // exact endpoints (BlackmanWindow[0.] came out 0.9999999999999999
    // and BlackmanWindow[0.5] came out -1.4*^-17).
    "BlackmanWindow" => {
      (21.0 + 25.0 * (2.0 * pi * x).cos() + 4.0 * (4.0 * pi * x).cos()) / 50.0
    }
    "DirichletWindow" => 1.0,
    "BartlettWindow" => 1.0 - 2.0 * x.abs(),
    "WelchWindow" => 1.0 - 4.0 * x * x,
    "CosineWindow" => (pi * x).cos(),
    "ConnesWindow" => {
      let t = 1.0 - 4.0 * x * x;
      t * t
    }
    "LanczosWindow" => {
      let arg = 2.0 * x;
      if arg.abs() < 1e-15 {
        1.0
      } else {
        (pi * arg).sin() / (pi * arg)
      }
    }
    // Blackman's exact coefficients {7938, 9240, 1430}/18608 — the
    // "exact" in the name is about these rationals replacing the rounded
    // 0.42 / 0.5 / 0.08 of BlackmanWindow.
    "ExactBlackmanWindow" => {
      (7938.0 + 9240.0 * (2.0 * pi * x).cos() + 1430.0 * (4.0 * pi * x).cos())
        / 18608.0
    }
    _ => {
      return Ok(unevaluated(name, args));
    }
  };

  Ok(Expr::Real(val))
}

/// TukeyWindow[x] / TukeyWindow[x, alpha] — Tukey (tapered-cosine) window.
/// Default alpha = 2/3. The window is 0 outside [-1/2, 1/2], a flat 1 for
/// |x| <= (1-alpha)/2, and a raised-cosine taper
/// (1 + Cos[Pi (2|x| - 1 + alpha)/alpha]) / 2 in between. Exact arguments are
/// evaluated symbolically (so Cos simplifies to wolframscript's radical forms,
/// e.g. TukeyWindow[1/4] -> (1 + 1/Sqrt[2])/2); Real arguments numericize.
pub fn tukey_window_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("TukeyWindow", args);
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated());
  }
  let x = &args[0];
  let alpha = args.get(1).cloned().unwrap_or_else(|| make_rational(2, 3));

  let (Some(xf), Some(af)) = (try_eval_to_f64(x), try_eval_to_f64(&alpha))
  else {
    return Ok(unevaluated());
  };

  fn contains_real(e: &Expr) -> bool {
    match e {
      Expr::Real(_) | Expr::BigFloat(_, _) => true,
      Expr::BinaryOp { left, right, .. } => {
        contains_real(left) || contains_real(right)
      }
      Expr::UnaryOp { operand, .. } => contains_real(operand),
      Expr::FunctionCall { args, .. } | Expr::List(args) => {
        args.iter().any(contains_real)
      }
      _ => false,
    }
  }
  let inexact = contains_real(x) || contains_real(&alpha);

  let ax = xf.abs();
  // Outside the window.
  if ax > 0.5 {
    return Ok(if inexact {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  // Flat top.
  if ax <= (1.0 - af) / 2.0 {
    return Ok(if inexact {
      Expr::Real(1.0)
    } else {
      Expr::Integer(1)
    });
  }
  // Raised-cosine taper.
  if inexact {
    let theta = (2.0 * ax - 1.0 + af) / af;
    return Ok(Expr::Real(f64::midpoint(
      1.0,
      (std::f64::consts::PI * theta).cos(),
    )));
  }
  // Exact: build (1 + Cos[Pi (2 Abs[x] - 1 + alpha)/alpha]) / 2 and evaluate
  // symbolically so Cos simplifies (matching wolframscript's radical forms).
  let abs_x = call1("Abs", x.clone());
  let theta_num = plus(vec![
    times(vec![Expr::Integer(2), abs_x]),
    Expr::Integer(-1),
    alpha.clone(),
  ]);
  let theta = div2(theta_num, alpha);
  let cos = call("Cos", vec![times2(id_expr("Pi"), theta)]);
  crate::evaluator::evaluate_expr_to_expr(&div2(
    plus(vec![Expr::Integer(1), cos]),
    Expr::Integer(2),
  ))
}

/// KaiserWindow[x] / KaiserWindow[x, alpha] — Kaiser (Kaiser–Bessel) window.
/// Default alpha = 3. The window is 0 outside [-1/2, 1/2] and equals
/// BesselI[0, alpha Sqrt[1 - 4 x^2]] / BesselI[0, alpha] within. Exact
/// arguments stay symbolic (matching wolframscript's radical forms, e.g.
/// KaiserWindow[1/4] -> BesselI[0, (3 Sqrt[3])/2]/BesselI[0, 3]); Real
/// arguments numericize.
pub fn kaiser_window_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("KaiserWindow", args);
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated());
  }
  let x = &args[0];
  let alpha = args.get(1).cloned().unwrap_or(Expr::Integer(3));

  let Some(xf) = try_eval_to_f64(x) else {
    return Ok(unevaluated());
  };
  if try_eval_to_f64(&alpha).is_none() {
    return Ok(unevaluated());
  }

  let inexact = window_arg_inexact(x) || window_arg_inexact(&alpha);

  // Outside the window support.
  if xf.abs() > 0.5 {
    return Ok(if inexact {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }

  // Build BesselI[0, alpha Sqrt[1 - 4 x^2]] / BesselI[0, alpha] and evaluate.
  // Real arguments numericize; exact arguments stay symbolic.
  let x_sq = pow(x.clone(), Expr::Integer(2));
  let one_minus =
    plus(vec![Expr::Integer(1), times(vec![Expr::Integer(-4), x_sq])]);
  let sqrt = call1("Sqrt", one_minus);
  let bessel_arg = times(vec![alpha.clone(), sqrt]);
  let numer = call("BesselI", vec![Expr::Integer(0), bessel_arg]);
  let denom = call("BesselI", vec![Expr::Integer(0), alpha]);
  crate::evaluator::evaluate_expr_to_expr(&div2(numer, denom))
}

/// True if `e` contains an inexact (machine) number, so a window function
/// numericizes; exact arguments stay symbolic.
fn window_arg_inexact(e: &Expr) -> bool {
  match e {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::BinaryOp { left, right, .. } => {
      window_arg_inexact(left) || window_arg_inexact(right)
    }
    Expr::UnaryOp { operand, .. } => window_arg_inexact(operand),
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(window_arg_inexact)
    }
    _ => false,
  }
}

/// ParzenWindow[x] — de la Vallée Poussin window: 0 outside [-1/2, 1/2],
/// 1 - 24 x^2 + 48 |x|^3 for |x| <= 1/4, and 2 (1 - 2 |x|)^3 for 1/4 < |x| <= 1/2.
/// Exact arguments give the exact polynomial value (e.g. ParzenWindow[1/3] ->
/// 2/27); Real arguments numericize.
pub fn parzen_window_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("ParzenWindow", args);
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  let x = &args[0];
  let Some(xf) = try_eval_to_f64(x) else {
    return Ok(unevaluated());
  };
  let inexact = window_arg_inexact(x);
  let ax = xf.abs();
  if ax > 0.5 {
    return Ok(if inexact {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  if inexact {
    let v = if ax <= 0.25 {
      1.0 - 24.0 * xf * xf + 48.0 * ax * ax * ax
    } else {
      let t = 1.0 - 2.0 * ax;
      2.0 * t * t * t
    };
    return Ok(Expr::Real(v));
  }
  // Exact polynomial.
  let abs_x = call1("Abs", x.clone());
  let pow = |base: Expr, n: i128| pow2(base, Expr::Integer(n));
  let times = |c: i128, e: Expr| times(vec![Expr::Integer(c), e]);
  let expr = if ax <= 0.25 {
    // 1 - 24 x^2 + 48 Abs[x]^3
    plus(vec![
      Expr::Integer(1),
      times(-24, pow(x.clone(), 2)),
      times(48, pow(abs_x, 3)),
    ])
  } else {
    // 2 (1 - 2 Abs[x])^3
    times(2, pow(plus(vec![Expr::Integer(1), times(-2, abs_x)]), 3))
  };
  crate::evaluator::evaluate_expr_to_expr(&expr)
}

/// GaussianWindow[x] / GaussianWindow[x, sigma] — Gaussian window
/// E^(-x^2/(2 sigma^2)) on [-1/2, 1/2] (0 outside), default sigma = 3/10.
/// Exact arguments give E^(rational) (e.g. GaussianWindow[1/4] -> E^(-25/72));
/// Real arguments numericize.
pub fn gaussian_window_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("GaussianWindow", args);
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated());
  }
  let x = &args[0];
  let sigma = args.get(1).cloned().unwrap_or_else(|| make_rational(3, 10));
  let (Some(xf), Some(sf)) = (try_eval_to_f64(x), try_eval_to_f64(&sigma))
  else {
    return Ok(unevaluated());
  };
  let inexact = window_arg_inexact(x) || window_arg_inexact(&sigma);
  let ax = xf.abs();
  if ax > 0.5 {
    return Ok(if inexact {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  if inexact {
    return Ok(Expr::Real((-xf * xf / (2.0 * sf * sf)).exp()));
  }
  // Exact: Exp[-x^2 / (2 sigma^2)].
  let pow = |base: Expr, n: i128| pow2(base, Expr::Integer(n));
  let numer = times(vec![Expr::Integer(-1), pow(x.clone(), 2)]);
  let denom = times(vec![Expr::Integer(2), pow(sigma, 2)]);
  let arg = div2(numer, denom);
  crate::evaluator::evaluate_expr_to_expr(&call1("Exp", arg))
}

/// BohmanWindow[x] — Bohman window: 0 outside [-1/2, 1/2], otherwise
/// (1 - 2|x|) Cos[2 Pi |x|] + Sin[2 Pi |x|] / Pi. Exact arguments evaluate the
/// symbolic form (e.g. BohmanWindow[1/4] -> 1/Pi); Real arguments numericize.
pub fn bohman_window_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("BohmanWindow", args);
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  let x = &args[0];
  let Some(xf) = try_eval_to_f64(x) else {
    return Ok(unevaluated());
  };
  let inexact = window_arg_inexact(x);
  let ax = xf.abs();
  if ax > 0.5 {
    return Ok(if inexact {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  if inexact {
    let arg = 2.0 * std::f64::consts::PI * ax;
    let v = (1.0 - 2.0 * ax) * arg.cos() + arg.sin() / std::f64::consts::PI;
    return Ok(Expr::Real(v));
  }
  // Exact: (1 - 2 Abs[x]) Cos[2 Pi Abs[x]] + Sin[2 Pi Abs[x]] / Pi.
  let abs_x = call1("Abs", x.clone());
  let two_pi_absx = times(vec![Expr::Integer(2), id_expr("Pi"), abs_x.clone()]);
  let one_minus_2ax = plus(vec![
    Expr::Integer(1),
    times(vec![Expr::Integer(-2), abs_x]),
  ]);
  let cos = call1("Cos", two_pi_absx.clone());
  let sin = call1("Sin", two_pi_absx);
  let term1 = times(vec![one_minus_2ax, cos]);
  let term2 = div2(sin, id_expr("Pi"));
  crate::evaluator::evaluate_expr_to_expr(&plus(vec![term1, term2]))
}

/// BandpassFilter[data, {omega1, omega2}] or BandpassFilter[data, {omega1, omega2}, n]
/// Applies a bandpass FIR filter using a windowed-sinc kernel with exact Hamming window.
/// Default order n = length of data. Default SampleRate = 1.
pub fn bandpass_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(unevaluated("BandpassFilter", args));
  }

  // Extract {omega1, omega2}
  let (omega1, omega2) = match &args[1] {
    Expr::List(freqs) if freqs.len() == 2 => {
      let Some(o1) = try_eval_to_f64(&freqs[0]) else {
        return Ok(unevaluated("BandpassFilter", args));
      };
      let Some(o2) = try_eval_to_f64(&freqs[1]) else {
        return Ok(unevaluated("BandpassFilter", args));
      };
      (o1, o2)
    }
    _ => {
      return Ok(unevaluated("BandpassFilter", args));
    }
  };

  let mut sample_rate = 1.0_f64;
  let mut explicit_order: Option<usize> = None;
  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        explicit_order = Some(*n as usize);
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  let w1 = omega1 / sample_rate;
  let w2 = omega2 / sample_rate;

  // Image input: separable 2D filter (row-then-column 1D passes).
  if let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
    let w = *width as usize;
    let h = *height as usize;
    let ch = *channels as usize;
    if w == 0 || h == 0 {
      return Ok(args[0].clone());
    }
    let row_order = explicit_order.unwrap_or(w);
    let col_order = explicit_order.unwrap_or(h);
    let row_kernel = bandpass_kernel(row_order, w1, w2);
    let col_kernel = bandpass_kernel(col_order, w1, w2);
    let mut out: Vec<f64> = vec![0.0; data.len()];
    for c_idx in 0..ch {
      let mut row_filtered: Vec<f64> = vec![0.0; w * h];
      for y in 0..h {
        let row: Vec<f64> =
          (0..w).map(|x| data[(y * w + x) * ch + c_idx]).collect();
        let filtered = convolve_edge_padded(&row, &row_kernel);
        for x in 0..w {
          row_filtered[y * w + x] = filtered[x];
        }
      }
      for x in 0..w {
        let col: Vec<f64> = (0..h).map(|y| row_filtered[y * w + x]).collect();
        let filtered = convolve_edge_padded(&col, &col_kernel);
        for y in 0..h {
          out[(y * w + x) * ch + c_idx] = filtered[y];
        }
      }
    }
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width: *width,
      height: *height,
      channels: *channels,
      data: std::sync::Arc::new(out),
      image_type: *image_type,
    });
  }

  // Extract data list
  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(unevaluated("BandpassFilter", args));
    }
  };

  let order = explicit_order.unwrap_or(items.len());
  let kernel = bandpass_kernel(order, w1, w2);

  // Try numeric path first
  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() == items.len() {
    // All numeric — fast path
    let result = convolve_edge_padded(&data, &kernel);
    return Ok(Expr::List(result.into_iter().map(Expr::Real).collect()));
  }

  // Symbolic path: convolve symbolically
  let result = convolve_edge_padded_symbolic(items, &kernel);
  Ok(Expr::List(result.into()))
}

/// Compute the bandpass FIR kernel of length n using windowed-sinc with exact Hamming window.
fn bandpass_kernel(n: usize, omega1: f64, omega2: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;

    // Exact Hamming window: alpha = 25/46
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };

    // Bandpass sinc kernel
    let sinc_bp = if t.abs() < 1e-15 {
      (omega2 - omega1) / std::f64::consts::PI
    } else {
      ((omega2 * t).sin() - (omega1 * t).sin()) / (std::f64::consts::PI * t)
    };

    kernel.push(w * sinc_bp);
  }
  kernel
}

/// Convolve data with kernel using edge-padding (repeat boundary values).
/// Returns output of same length as data.
pub fn convolve_edge_padded(data: &[f64], kernel: &[f64]) -> Vec<f64> {
  let n = kernel.len();
  let len = data.len();
  let left_pad = n / 2;

  let mut result = Vec::with_capacity(len);
  for m in 0..len {
    let mut sum = 0.0;
    for j in 0..n {
      let idx = m as isize + j as isize - left_pad as isize;
      let val = if idx < 0 {
        data[0]
      } else if idx >= len as isize {
        data[len - 1]
      } else {
        data[idx as usize]
      };
      sum += kernel[j] * val;
    }
    result.push(fourier_round(sum));
  }
  result
}

/// Symbolic convolution: for each output position, compute a symbolic sum of kernel[j] * data[idx].
/// Uses edge-padding (repeat boundary values) like the numeric version.
fn convolve_edge_padded_symbolic(data: &[Expr], kernel: &[f64]) -> Vec<Expr> {
  let n = kernel.len();
  let len = data.len();
  let left_pad = n / 2;

  let mut result = Vec::with_capacity(len);
  for m in 0..len {
    // Group kernel coefficients by data index
    let mut coeff_by_idx: Vec<f64> = vec![0.0; len];
    for j in 0..n {
      let idx = m as isize + j as isize - left_pad as isize;
      let data_idx = if idx < 0 {
        0
      } else if idx >= len as isize {
        len - 1
      } else {
        idx as usize
      };
      coeff_by_idx[data_idx] += kernel[j];
    }
    // Build symbolic sum
    let mut terms: Vec<Expr> = Vec::new();
    for (i, &coeff) in coeff_by_idx.iter().enumerate() {
      if coeff == 0.0 {
        continue;
      }
      terms.push(times(vec![Expr::Real(coeff), data[i].clone()]));
    }
    let expr = if terms.is_empty() {
      Expr::Integer(0)
    } else if terms.len() == 1 {
      terms.pop().unwrap()
    } else {
      plus(terms)
    };
    // Evaluate the symbolic expression to simplify
    match crate::evaluator::evaluate_expr_to_expr(&expr) {
      Ok(e) => result.push(e),
      Err(_) => result.push(expr),
    }
  }
  result
}

/// LowpassFilter[data, omega_c] or LowpassFilter[data, omega_c, n]
/// Applies a lowpass FIR filter using a windowed-sinc kernel with exact Hamming window.
pub fn lowpass_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(unevaluated("LowpassFilter", args));
  }

  let Some(omega_c) = try_eval_to_f64(&args[1]) else {
    return Ok(unevaluated("LowpassFilter", args));
  };

  let mut sample_rate = 1.0_f64;
  let mut explicit_order: Option<usize> = None;
  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        explicit_order = Some(*n as usize);
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  let wc = omega_c / sample_rate;

  // Image input: filter each channel row-by-row, then column-by-column
  // with the same 1D windowed-sinc kernel. Matches wolframscript's
  // separable 2D filter (`LowpassFilter[Image[…], ωc]`).
  if let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
    let w = *width as usize;
    let h = *height as usize;
    let ch = *channels as usize;
    if w == 0 || h == 0 {
      return Ok(args[0].clone());
    }
    let row_order = explicit_order.unwrap_or(w);
    let col_order = explicit_order.unwrap_or(h);
    let row_kernel = lowpass_kernel(row_order, wc);
    let col_kernel = lowpass_kernel(col_order, wc);
    let mut out: Vec<f64> = vec![0.0; data.len()];
    for c_idx in 0..ch {
      // Row-wise pass: into a (h × w) scratch buffer.
      let mut row_filtered: Vec<f64> = vec![0.0; w * h];
      for y in 0..h {
        let row: Vec<f64> =
          (0..w).map(|x| data[(y * w + x) * ch + c_idx]).collect();
        let filtered = convolve_edge_padded(&row, &row_kernel);
        for x in 0..w {
          row_filtered[y * w + x] = filtered[x];
        }
      }
      // Column-wise pass.
      for x in 0..w {
        let col: Vec<f64> = (0..h).map(|y| row_filtered[y * w + x]).collect();
        let filtered = convolve_edge_padded(&col, &col_kernel);
        for y in 0..h {
          out[(y * w + x) * ch + c_idx] = filtered[y];
        }
      }
    }
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width: *width,
      height: *height,
      channels: *channels,
      data: std::sync::Arc::new(out),
      image_type: *image_type,
    });
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(unevaluated("LowpassFilter", args));
    }
  };
  let order = explicit_order.unwrap_or(items.len());

  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() != items.len() {
    return Ok(unevaluated("LowpassFilter", args));
  }

  let kernel = lowpass_kernel(order, wc);
  let result = convolve_edge_padded(&data, &kernel);

  Ok(Expr::List(result.into_iter().map(Expr::Real).collect()))
}

/// Compute the lowpass FIR kernel of length n, normalized to sum to 1.
fn lowpass_kernel(n: usize, omega_c: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };
    let sinc_lp = if t.abs() < 1e-15 {
      omega_c / std::f64::consts::PI
    } else {
      (omega_c * t).sin() / (std::f64::consts::PI * t)
    };
    kernel.push(w * sinc_lp);
  }
  // Normalize to sum to 1 (unity DC gain)
  let sum: f64 = kernel.iter().sum();
  if sum.abs() > 1e-15 {
    for v in &mut kernel {
      *v /= sum;
    }
  }
  kernel
}

/// HighpassFilter[data, omega_c] or HighpassFilter[data, omega_c, n]
/// Applies a highpass FIR filter using a windowed-sinc kernel with exact Hamming window.
pub fn highpass_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(unevaluated("HighpassFilter", args));
  }

  let Some(omega_c) = try_eval_to_f64(&args[1]) else {
    return Ok(unevaluated("HighpassFilter", args));
  };

  let mut sample_rate = 1.0_f64;
  let mut explicit_order: Option<usize> = None;
  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        explicit_order = Some(*n as usize);
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  let wc = omega_c / sample_rate;

  // Image input: separable 2D filter (row-then-column 1D passes).
  if let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
    let w = *width as usize;
    let h = *height as usize;
    let ch = *channels as usize;
    if w == 0 || h == 0 {
      return Ok(args[0].clone());
    }
    let row_order = explicit_order.unwrap_or(w);
    let col_order = explicit_order.unwrap_or(h);
    let row_kernel = highpass_kernel(row_order, wc);
    let col_kernel = highpass_kernel(col_order, wc);
    let mut out: Vec<f64> = vec![0.0; data.len()];
    for c_idx in 0..ch {
      let mut row_filtered: Vec<f64> = vec![0.0; w * h];
      for y in 0..h {
        let row: Vec<f64> =
          (0..w).map(|x| data[(y * w + x) * ch + c_idx]).collect();
        let filtered = convolve_edge_padded(&row, &row_kernel);
        for x in 0..w {
          row_filtered[y * w + x] = filtered[x];
        }
      }
      for x in 0..w {
        let col: Vec<f64> = (0..h).map(|y| row_filtered[y * w + x]).collect();
        let filtered = convolve_edge_padded(&col, &col_kernel);
        for y in 0..h {
          out[(y * w + x) * ch + c_idx] = filtered[y];
        }
      }
    }
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width: *width,
      height: *height,
      channels: *channels,
      data: std::sync::Arc::new(out),
      image_type: *image_type,
    });
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(unevaluated("HighpassFilter", args));
    }
  };
  let order = explicit_order.unwrap_or(items.len());

  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() != items.len() {
    return Ok(unevaluated("HighpassFilter", args));
  }

  let kernel = highpass_kernel(order, wc);
  let result = convolve_edge_padded(&data, &kernel);

  Ok(Expr::List(result.into_iter().map(Expr::Real).collect()))
}

/// Compute the highpass FIR kernel of length n.
/// Highpass = delta - lowpass (spectral inversion).
fn highpass_kernel(n: usize, omega_c: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };
    // Ideal highpass: delta(t) - sinc_lp(t) for the windowed version
    let sinc_hp = if t.abs() < 1e-15 {
      1.0 - omega_c / std::f64::consts::PI
    } else {
      -(omega_c * t).sin() / (std::f64::consts::PI * t)
    };
    kernel.push(w * sinc_hp);
  }
  kernel
}

/// BandstopFilter[data, {omega1, omega2}] or BandstopFilter[data, {omega1, omega2}, n]
/// Applies a bandstop (notch) FIR filter.
pub fn bandstop_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(unevaluated("BandstopFilter", args));
  }

  let (omega1, omega2) = match &args[1] {
    Expr::List(freqs) if freqs.len() == 2 => {
      let Some(o1) = try_eval_to_f64(&freqs[0]) else {
        return Ok(unevaluated("BandstopFilter", args));
      };
      let Some(o2) = try_eval_to_f64(&freqs[1]) else {
        return Ok(unevaluated("BandstopFilter", args));
      };
      (o1, o2)
    }
    _ => {
      return Ok(unevaluated("BandstopFilter", args));
    }
  };

  let mut sample_rate = 1.0_f64;
  let mut explicit_order: Option<usize> = None;
  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        explicit_order = Some(*n as usize);
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  let w1 = omega1 / sample_rate;
  let w2 = omega2 / sample_rate;

  // Image input: separable 2D filter (row-then-column 1D passes).
  if let Expr::Image {
    color_space: _,
    display_size: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
    let w = *width as usize;
    let h = *height as usize;
    let ch = *channels as usize;
    if w == 0 || h == 0 {
      return Ok(args[0].clone());
    }
    let row_order = explicit_order.unwrap_or(w);
    let col_order = explicit_order.unwrap_or(h);
    let row_kernel = bandstop_kernel(row_order, w1, w2);
    let col_kernel = bandstop_kernel(col_order, w1, w2);
    let mut out: Vec<f64> = vec![0.0; data.len()];
    for c_idx in 0..ch {
      let mut row_filtered: Vec<f64> = vec![0.0; w * h];
      for y in 0..h {
        let row: Vec<f64> =
          (0..w).map(|x| data[(y * w + x) * ch + c_idx]).collect();
        let filtered = convolve_edge_padded(&row, &row_kernel);
        for x in 0..w {
          row_filtered[y * w + x] = filtered[x];
        }
      }
      for x in 0..w {
        let col: Vec<f64> = (0..h).map(|y| row_filtered[y * w + x]).collect();
        let filtered = convolve_edge_padded(&col, &col_kernel);
        for y in 0..h {
          out[(y * w + x) * ch + c_idx] = filtered[y];
        }
      }
    }
    return Ok(Expr::Image {
      color_space: None,
      display_size: None,
      width: *width,
      height: *height,
      channels: *channels,
      data: std::sync::Arc::new(out),
      image_type: *image_type,
    });
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(unevaluated("BandstopFilter", args));
    }
  };
  let order = explicit_order.unwrap_or(items.len());

  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() != items.len() {
    return Ok(unevaluated("BandstopFilter", args));
  }

  let kernel = bandstop_kernel(order, w1, w2);
  let result = convolve_edge_padded(&data, &kernel);

  Ok(Expr::List(result.into_iter().map(Expr::Real).collect()))
}

/// Compute the bandstop FIR kernel of length n.
/// Bandstop = delta - bandpass (spectral inversion).
fn bandstop_kernel(n: usize, omega1: f64, omega2: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };
    let sinc_bs = if t.abs() < 1e-15 {
      1.0 - (omega2 - omega1) / std::f64::consts::PI
    } else {
      -((omega2 * t).sin() - (omega1 * t).sin()) / (std::f64::consts::PI * t)
    };
    kernel.push(w * sinc_bs);
  }
  kernel
}

// ─── Numeric RootSum ────────────────────────────────────────────────

/// `N[RootSum[poly &, fn &]]`: find every (complex) root of `poly`, apply
/// `fn` to each, and sum. Returns `Some(Real)` (or a Plus[Real, Times[Real,
/// I]] for residual imaginary parts) on success and `None` for shapes that
/// don't fit — letting the caller fall back to the generic recursive `N`.
fn root_sum_n_eval(poly_arg: &Expr, fn_arg: &Expr) -> Option<Expr> {
  use crate::functions::polynomial_ast::extract_poly_coeffs_big as extract_poly_coeffs;
  // poly_arg should be a Function whose body is a polynomial in #1.
  let poly_body = match poly_arg {
    Expr::Function { body } => body.as_ref(),
    _ => return None,
  };
  // Substitute Slot(1) → __rs_x__ to get a polynomial in a named variable
  // (extract_poly_coeffs needs an identifier).
  let var = "__rs_x__";
  let poly_in_var = substitute_variable(poly_body, "#1", &id_expr(var));
  // Slot(1) inside Function bodies is stored differently from Identifier
  // "#1"; substitute that variant too.
  let poly_in_var = substitute_slot_with_identifier(&poly_in_var, 1, var);
  let expanded =
    crate::functions::polynomial_ast::expand_and_combine(&poly_in_var);
  let coeffs_i = extract_poly_coeffs(&expanded, var)?;
  if coeffs_i.len() < 2 {
    return None;
  }
  let coeffs_f = bigint_coeffs_to_f64(&coeffs_i)?;
  let roots = aberth_complex_roots(&coeffs_f);
  if roots.is_empty() {
    return None;
  }
  // Apply fn_arg (a Function or a builtin like Sin) to each complex root,
  // sum the results numerically.
  let mut sum_re = 0.0f64;
  let mut sum_im = 0.0f64;
  for (zr, zi) in &roots {
    let val = apply_fn_to_complex(fn_arg, *zr, *zi)?;
    sum_re += val.0;
    sum_im += val.1;
  }
  if sum_im.abs() < 1e-10 * sum_re.abs().max(1.0) {
    Some(Expr::Real(sum_re))
  } else {
    // Build Real + Real*I as Plus[..., Times[..., I]] so subsequent
    // Chop can drop the imaginary tail.
    Some(plus(vec![
      Expr::Real(sum_re),
      times(vec![Expr::Real(sum_im), id_expr("I")]),
    ]))
  }
}

/// Shared by the machine- and arbitrary-precision `Root[f, k]` evaluators:
/// extracts the integer coefficients of the polynomial `f` (in whichever
/// single variable it names — a Slot-bodied pure function or a named
/// symbol), finds every root at machine precision, sorts them the way
/// `Root`'s index `k` expects (real roots first in ascending order, then
/// complex roots ordered by real then imaginary part), and returns the
/// `k`-th one as a machine-precision seed for whichever precision the
/// caller ultimately wants. Both evaluators therefore agree on which root
/// `k` names — refining the same seed to more digits, not risking a
/// different root at high precision.
fn root_coeffs_and_seed(
  poly_arg: &Expr,
  k_arg: &Expr,
) -> Option<(Vec<num_bigint::BigInt>, bool, f64, f64)> {
  use crate::functions::polynomial_ast::extract_poly_coeffs_big as extract_poly_coeffs;
  let k = expr_to_i128(k_arg)?;
  if k < 1 {
    return None;
  }
  // Two accepted first-argument shapes for Root[f, k]:
  //   * pure function `#^3 - 2 &` (Slot/`#1` is the root variable), or
  //   * an ordinary polynomial expression in a single symbol, e.g. `x^3 - 2`.
  let (poly_in_var, var): (Expr, String) =
    if let Expr::Function { body } = poly_arg {
      let var = "__root_x__".to_string();
      let sub = crate::syntax::substitute_variable(
        body.as_ref(),
        "#1",
        &Expr::Identifier(var.clone()),
      );
      (substitute_slot_with_identifier(&sub, 1, &var), var)
    } else {
      let mut vars = std::collections::HashSet::new();
      collect_expr_vars(poly_arg, &mut vars);
      if vars.len() != 1 {
        return None;
      }
      let var = vars.into_iter().next().unwrap();
      (poly_arg.clone(), var)
    };
  let expanded =
    crate::functions::polynomial_ast::expand_and_combine(&poly_in_var);
  let coeffs_i = extract_poly_coeffs(&expanded, &var)?;
  if coeffs_i.len() < 2 {
    return None;
  }
  let coeffs_f = bigint_coeffs_to_f64(&coeffs_i)?;
  let mut roots = aberth_complex_roots(&coeffs_f);
  if roots.is_empty() || (k as usize) > roots.len() {
    return None;
  }
  let is_real = |re: f64, im: f64| im.abs() < 1e-8 * (1.0 + re.abs());
  roots.sort_by(|a, b| {
    use std::cmp::Ordering;
    let ar = is_real(a.0, a.1);
    let br = is_real(b.0, b.1);
    match (ar, br) {
      (true, false) => Ordering::Less,
      (false, true) => Ordering::Greater,
      _ => a
        .0
        .partial_cmp(&b.0)
        .unwrap_or(Ordering::Equal)
        .then(a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)),
    }
  });
  let (re, im) = roots[(k - 1) as usize];
  Some((coeffs_i, is_real(re, im), re, im))
}

/// Integer polynomial coefficients as machine numbers; `None` when one
/// overflows a double.
fn bigint_coeffs_to_f64(coeffs: &[num_bigint::BigInt]) -> Option<Vec<f64>> {
  use num_traits::ToPrimitive;
  coeffs
    .iter()
    .map(|c| c.to_f64().filter(|v| v.is_finite()))
    .collect()
}

/// An integer coefficient as a `bits`-precision BigFloat.
fn bigint_to_bigfloat(
  c: &num_bigint::BigInt,
  bits: usize,
  rm: astro_float::RoundingMode,
) -> astro_float::BigFloat {
  use num_traits::ToPrimitive;
  match c.to_i128() {
    Some(small) => astro_float::BigFloat::from_i128(small, bits),
    None => match astro_float::Consts::new() {
      Ok(mut cc) => astro_float::BigFloat::parse(
        &c.to_string(),
        astro_float::Radix::Dec,
        bits,
        rm,
        &mut cc,
      ),
      Err(_) => astro_float::BigFloat::nan(None),
    },
  }
}

/// `N[Root[poly &, k]]`: the k-th root of the polynomial `poly`, ordered as
/// wolframscript does — real roots first in increasing order, then the
/// non-real roots by increasing real part and then increasing imaginary part.
/// Returns the root as a `Real` (or `Plus[Real, Times[Real, I]]` for complex
/// roots), or `None` when the shape doesn't fit.
pub(crate) fn root_n_eval(poly_arg: &Expr, k_arg: &Expr) -> Option<Expr> {
  // A polyhedron's coordinates repeat the same few `Root[…]` numbers many
  // times, and each one costs a root finder and a BigFloat refinement.
  thread_local! {
    static ROOT_VALUES: std::cell::RefCell<
      std::collections::HashMap<(String, String), Option<Expr>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
  }
  let key = (expr_to_string(poly_arg), expr_to_string(k_arg));
  if let Some(hit) = ROOT_VALUES.with(|c| c.borrow().get(&key).cloned()) {
    return hit;
  }
  let value = root_n_eval_uncached(poly_arg, k_arg);
  ROOT_VALUES.with(|c| {
    let mut c = c.borrow_mut();
    if c.len() >= 4096 {
      c.clear();
    }
    c.insert(key, value.clone());
  });
  value
}

fn root_n_eval_uncached(poly_arg: &Expr, k_arg: &Expr) -> Option<Expr> {
  let (coeffs, is_real, re, im) = root_coeffs_and_seed(poly_arg, k_arg)?;
  if is_real {
    // The machine seed can be an ulp off; Wolfram's is correctly rounded.
    let refined = bigfloat_to_f64_approx(&refine_real_root(
      &coeffs,
      re,
      128,
      astro_float::RoundingMode::ToEven,
    ))
    .filter(|v| v.is_finite() && (v - re).abs() <= 1e-9 * re.abs().max(1.0));
    Some(Expr::Real(refined.unwrap_or(re)))
  } else {
    Some(plus(vec![
      Expr::Real(re),
      times(vec![Expr::Real(im), id_expr("I")]),
    ]))
  }
}

/// Arbitrary-precision `Root[f, k]` — Newton-refines the machine-precision
/// seed from `root_coeffs_and_seed` to `bits` of working precision using
/// `astro_float::BigFloat` arithmetic. Real roots only (a complex result
/// falls back to `expr_to_complex_bigfloat`'s caller, same as any other
/// expression it doesn't special-case); `N[Root[#^3 - # - 1 &, 1], 30]`
/// (the plastic constant) is the motivating case.
pub(crate) fn root_n_eval_arbitrary(
  poly_arg: &Expr,
  k_arg: &Expr,
  bits: usize,
  rm: astro_float::RoundingMode,
) -> Option<astro_float::BigFloat> {
  let (coeffs_i, is_real, re0, _im0) = root_coeffs_and_seed(poly_arg, k_arg)?;
  if !is_real {
    return None;
  }
  Some(refine_real_root(&coeffs_i, re0, bits, rm))
}

/// Newton-refine the real root seed `re0` of the integer polynomial
/// `coeffs_i` to `bits` of working precision.
fn refine_real_root(
  coeffs_i: &[num_bigint::BigInt],
  re0: f64,
  bits: usize,
  rm: astro_float::RoundingMode,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;
  // A couple of guard words above the target precision so the Newton
  // iterates (and the Horner evaluation of p/p' at each step) don't lose
  // the last few bits to rounding before the final result is rounded down
  // to `bits`.
  let wbits = bits + 64;
  let coeffs_bf: Vec<BigFloat> = coeffs_i
    .iter()
    .map(|c| bigint_to_bigfloat(c, wbits, rm))
    .collect();
  let n = coeffs_bf.len() - 1;
  let eval = |x: &BigFloat| -> (BigFloat, BigFloat) {
    // Horner's method for p(x) and, alongside it, p'(x) via the standard
    // simultaneous evaluation (b_n = a_n, b_{k} = a_k + x*b_{k+1}; p' is
    // built from the b's excluding the constant term).
    let mut p = coeffs_bf[n].clone();
    let mut dp = BigFloat::from_i32(0, wbits);
    for k in (0..n).rev() {
      dp = dp.mul(x, wbits, rm).add(&p, wbits, rm);
      p = p.mul(x, wbits, rm).add(&coeffs_bf[k], wbits, rm);
    }
    (p, dp)
  };
  let mut x = BigFloat::from_f64(re0, wbits);
  // Newton's method roughly doubles the number of correct digits per step;
  // starting from a machine-precision (~53-bit) seed, enough steps to
  // double past `wbits` covers any working precision this function is
  // called with.
  let iterations = (wbits as f64 / 40.0).log2().ceil().max(1.0) as usize + 4;
  for _ in 0..iterations {
    let (p, dp) = eval(&x);
    if dp.is_zero() {
      break;
    }
    x = x.sub(&p.div(&dp, wbits, rm), wbits, rm);
  }
  x
}

/// Arbitrary-precision complex `Root[f, k]`: Newton's method on the
/// complex plane from the machine-precision seed, in `astro_float`
/// arithmetic, the complex counterpart of [`root_n_eval_arbitrary`].
pub(crate) fn root_n_eval_arbitrary_complex(
  poly_arg: &Expr,
  k_arg: &Expr,
  bits: usize,
  rm: astro_float::RoundingMode,
) -> Option<(astro_float::BigFloat, astro_float::BigFloat)> {
  use astro_float::BigFloat;
  let (coeffs_i, _, re0, im0) = root_coeffs_and_seed(poly_arg, k_arg)?;
  let wbits = bits + 64;
  let coeffs: Vec<BigFloat> = coeffs_i
    .iter()
    .map(|c| bigint_to_bigfloat(c, wbits, rm))
    .collect();
  let n = coeffs.len() - 1;
  let zero = || BigFloat::from_i32(0, wbits);
  let mul = |(a, b): (&BigFloat, &BigFloat), (c, d): (&BigFloat, &BigFloat)| {
    (
      a.mul(c, wbits, rm).sub(&b.mul(d, wbits, rm), wbits, rm),
      a.mul(d, wbits, rm).add(&b.mul(c, wbits, rm), wbits, rm),
    )
  };
  let mut x = (
    BigFloat::from_f64(re0, wbits),
    BigFloat::from_f64(im0, wbits),
  );
  let iterations = (wbits as f64 / 40.0).log2().ceil().max(1.0) as usize + 4;
  for _ in 0..iterations {
    // Horner for p(x) and p'(x) together.
    let mut p = (coeffs[n].clone(), zero());
    let mut dp = (zero(), zero());
    for k in (0..n).rev() {
      let t = mul((&dp.0, &dp.1), (&x.0, &x.1));
      dp = (t.0.add(&p.0, wbits, rm), t.1.add(&p.1, wbits, rm));
      let t = mul((&p.0, &p.1), (&x.0, &x.1));
      p = (t.0.add(&coeffs[k], wbits, rm), t.1);
    }
    let denom =
      dp.0
        .mul(&dp.0, wbits, rm)
        .add(&dp.1.mul(&dp.1, wbits, rm), wbits, rm);
    if denom.is_zero() {
      break;
    }
    // p / p' = p · conj(p') / |p'|²
    let q = mul((&p.0, &p.1), (&dp.0, &dp.1.neg()));
    x = (
      x.0.sub(&q.0.div(&denom, wbits, rm), wbits, rm),
      x.1.sub(&q.1.div(&denom, wbits, rm), wbits, rm),
    );
  }
  Some(x)
}

/// Collect the free symbol names in `expr` (skipping the boolean/null
/// literals). Used to infer the root variable of an expression-form
/// `Root[x^3 - 2, k]`.
fn collect_expr_vars(
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
      collect_expr_vars(left, vars);
      collect_expr_vars(right, vars);
    }
    Expr::UnaryOp { operand, .. } => collect_expr_vars(operand, vars),
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_expr_vars(a, vars);
      }
    }
    Expr::List(items) => {
      for i in items {
        collect_expr_vars(i, vars);
      }
    }
    _ => {}
  }
}

/// Replace every `Slot(k)` in `expr` with `Identifier(name)`. A companion
/// to `substitute_variable` so `Function`-bodied polynomials in `#1` can
/// be passed through the named-variable polynomial helpers.
fn substitute_slot_with_identifier(expr: &Expr, k: usize, name: &str) -> Expr {
  match expr {
    Expr::Slot(n) if *n == k => id_expr(name),
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| substitute_slot_with_identifier(e, k, name))
        .collect(),
    ),
    Expr::FunctionCall { name: fname, args } => Expr::FunctionCall {
      name: fname.clone(),
      args: args
        .iter()
        .map(|e| substitute_slot_with_identifier(e, k, name))
        .collect(),
    },
    Expr::BinaryOp { op, left, right } => binop(
      *op,
      substitute_slot_with_identifier(left, k, name),
      substitute_slot_with_identifier(right, k, name),
    ),
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_slot_with_identifier(operand, k, name)),
    },
    _ => expr.clone(),
  }
}

/// Aberth's method for finding all complex roots of a real-coefficient
/// polynomial `p(x) = c_0 + c_1 x + … + c_n x^n`. Returns a vector of
/// `(re, im)` pairs of length `n` (degree of polynomial).
fn aberth_complex_roots(coeffs: &[f64]) -> Vec<(f64, f64)> {
  let n = coeffs.len() - 1;
  if n == 0 {
    return vec![];
  }
  if coeffs[n].abs() < 1e-300 {
    return vec![];
  }
  // Cauchy bound on |roots|: 1 + max_{i<n} |c_i / c_n|.
  let lead = coeffs[n].abs();
  let r = 1.0
    + coeffs[..n]
      .iter()
      .map(|c| c.abs() / lead)
      .fold(0.0f64, f64::max);
  let pi = std::f64::consts::PI;
  // Initial guesses on the Cauchy circle, slightly off-axis.
  let mut zs: Vec<(f64, f64)> = (0..n)
    .map(|k| {
      let theta = 2.0 * pi * (k as f64 + 0.5) / (n as f64) + 0.7;
      (r * theta.cos(), r * theta.sin())
    })
    .collect();
  for _ in 0..200 {
    let mut max_change: f64 = 0.0;
    let mut new_zs: Vec<(f64, f64)> = Vec::with_capacity(n);
    for k in 0..n {
      let (zr, zi) = zs[k];
      let (pr, pi_) = poly_eval_complex(coeffs, zr, zi);
      let (dr, di) = poly_deriv_eval_complex(coeffs, zr, zi);
      if dr == 0.0 && di == 0.0 {
        new_zs.push((zr, zi));
        continue;
      }
      // q = p(z) / p'(z). Smith's division avoids squaring |p'|, which
      // overflows for high-degree polynomials with large roots.
      let (qr, qi) = if dr.abs() >= di.abs() {
        let t = di / dr;
        let d = dr + di * t;
        ((pr + pi_ * t) / d, (pi_ - pr * t) / d)
      } else {
        let t = dr / di;
        let d = dr * t + di;
        ((pr * t + pi_) / d, (pi_ * t - pr) / d)
      };
      // sum_{j != k} 1 / (z_k - z_j)
      let mut sr = 0.0f64;
      let mut si = 0.0f64;
      for j in 0..n {
        if j == k {
          continue;
        }
        let dxr = zr - zs[j].0;
        let dxi = zi - zs[j].1;
        let d = dxr * dxr + dxi * dxi;
        if d < 1e-300 {
          continue;
        }
        sr += dxr / d;
        si += -dxi / d;
      }
      // Aberth: z_new = z - q / (1 - q*sum)
      let qsr = qr * sr - qi * si;
      let qsi = qr * si + qi * sr;
      let denr = 1.0 - qsr;
      let deni = -qsi;
      let dnorm = denr * denr + deni * deni;
      if dnorm < 1e-300 {
        new_zs.push((zr, zi));
        continue;
      }
      // q / (1 - q*sum)
      let dxr = (qr * denr + qi * deni) / dnorm;
      let dxi = (qi * denr - qr * deni) / dnorm;
      let new_zr = zr - dxr;
      let new_zi = zi - dxi;
      let change = (dxr * dxr + dxi * dxi).sqrt();
      if change > max_change {
        max_change = change;
      }
      new_zs.push((new_zr, new_zi));
    }
    zs = new_zs;
    if max_change < 1e-13 {
      break;
    }
  }
  polish_and_pair_roots(coeffs, &mut zs);
  zs
}

/// Shared final step of the numeric polynomial root finders: Newton-polish
/// every root, snap fp noise to exact zeros and turn near-conjugate pairs into
/// exact mirrors of each other.
///
/// Both root finders (Durand-Kerner in `polynomial_ast::solve` and Aberth
/// here) end with this so `NRoots`, `NSolve` and `N[Root[…]]` report the same
/// digits for the same polynomial. Exact conjugate pairs matter beyond
/// cosmetics: a quadrature that sums a conjugate-symmetric set of terms then
/// cancels its imaginary part exactly instead of leaving a residue that
/// `Chop` cannot remove.
///
/// `coeffs[i]` is the coefficient of `x^i`.
pub fn polish_and_pair_roots(coeffs: &[f64], roots: &mut [(f64, f64)]) {
  let n = coeffs.len().saturating_sub(1);
  if n == 0 || coeffs[n].abs() < 1e-300 {
    return;
  }
  // Monic form: divide by the leading coefficient so the residual |p(z)|
  // compared across iterates is on a fixed scale.
  let lc = coeffs[n];
  let monic: Vec<f64> = coeffs.iter().map(|c| c / lc).collect();

  // Horner with FMA — more accurate near a root, which lets the polish step
  // tell the correctly-rounded f64 from its neighbour.
  let eval_p = |zr: f64, zi: f64| -> (f64, f64) {
    let mut re = monic[n];
    let mut im = 0.0;
    for k in (0..n).rev() {
      let neg_im: f64 = -im;
      let nr = re.mul_add(zr, neg_im.mul_add(zi, monic[k]));
      let ni = re.mul_add(zi, im * zr);
      re = nr;
      im = ni;
    }
    (re, im)
  };
  let eval_dp = |zr: f64, zi: f64| -> (f64, f64) {
    let mut re = (n as f64) * monic[n];
    let mut im = 0.0;
    for k in (1..n).rev() {
      let nr = re * zr - im * zi + (k as f64) * monic[k];
      let ni = re * zi + im * zr;
      re = nr;
      im = ni;
    }
    (re, im)
  };

  // Newton steps, keeping the iterate with the smallest |p(z)|: near the root
  // Newton may oscillate between two adjacent f64 values (both equally good).
  for slot in roots.iter_mut() {
    let (mut zr, mut zi) = *slot;
    let (mut best_zr, mut best_zi) = (zr, zi);
    let (pr0, pi0) = eval_p(zr, zi);
    let mut best_err = pr0.hypot(pi0);
    for _ in 0..30 {
      let (pr, pi) = eval_p(zr, zi);
      let (dr, di) = eval_dp(zr, zi);
      let denom = dr * dr + di * di;
      if denom < 1e-300 {
        break;
      }
      zr -= (pr * dr + pi * di) / denom;
      zi -= (pi * dr - pr * di) / denom;
      let (npr, npi) = eval_p(zr, zi);
      let err = npr.hypot(npi);
      if err < best_err {
        best_err = err;
        best_zr = zr;
        best_zi = zi;
      }
      if err == 0.0 {
        break;
      }
    }
    *slot = (best_zr, best_zi);
  }

  // Snap to real / round trailing fp noise.
  for (r, i) in roots.iter_mut() {
    let mag = r.hypot(*i).max(1.0);
    if i.abs() < 1e-12 * mag {
      *i = 0.0;
    }
    if r.abs() < 1e-12 * mag {
      *r = 0.0;
    }
  }

  // Match conjugate pairs: two roots sharing a real part and carrying opposite
  // imaginary parts (within tolerance) become exact mirrors. Each root joins at
  // most one pair, and two real roots are never paired with each other — for
  // real coefficients a genuine pair always has a non-zero imaginary part.
  let snap_eps = 1e-9;
  let mut paired = vec![false; roots.len()];
  for k in 0..roots.len() {
    if paired[k] || roots[k].1 == 0.0 {
      continue;
    }
    for j in (k + 1)..roots.len() {
      if paired[j] || roots[j].1 == 0.0 {
        continue;
      }
      if (roots[k].0 - roots[j].0).abs() < snap_eps
        && (roots[k].1 + roots[j].1).abs() < snap_eps
      {
        let avg = f64::midpoint(roots[k].0, roots[j].0);
        let mag = f64::midpoint(roots[k].1.abs(), roots[j].1.abs());
        let k_negative = roots[k].1 < 0.0;
        roots[k] = (avg, if k_negative { -mag } else { mag });
        roots[j] = (avg, if k_negative { mag } else { -mag });
        paired[k] = true;
        paired[j] = true;
        break;
      }
    }
  }
}

/// Horner-style polynomial evaluation at a complex point `z = zr + i*zi`.
fn poly_eval_complex(coeffs: &[f64], zr: f64, zi: f64) -> (f64, f64) {
  let mut rr = 0.0f64;
  let mut ri = 0.0f64;
  for &c in coeffs.iter().rev() {
    // r = r * z + c
    let nr = rr * zr - ri * zi + c;
    let ni = rr * zi + ri * zr;
    rr = nr;
    ri = ni;
  }
  (rr, ri)
}

/// Evaluate `p'(z)` at a complex point `z`. `p'(x) = sum_{k>=1} k * c_k *
/// x^(k-1)`.
fn poly_deriv_eval_complex(coeffs: &[f64], zr: f64, zi: f64) -> (f64, f64) {
  if coeffs.len() < 2 {
    return (0.0, 0.0);
  }
  let dcoeffs: Vec<f64> =
    (1..coeffs.len()).map(|k| (k as f64) * coeffs[k]).collect();
  poly_eval_complex(&dcoeffs, zr, zi)
}

/// Apply a simple unary numeric function (or a `Function` body) to a
/// complex argument. Supports `Sin`, `Cos`, `Tan`, `Exp`, `Log`, `Identity`,
/// and arbitrary `Function`-bodied expressions in #1 by substituting the
/// complex value back through the symbolic evaluator.
fn apply_fn_to_complex(fnexpr: &Expr, zr: f64, zi: f64) -> Option<(f64, f64)> {
  match fnexpr {
    Expr::Identifier(s) => match s.as_str() {
      "Sin" => Some(complex_sin(zr, zi)),
      "Cos" => Some(complex_cos(zr, zi)),
      "Identity" => Some((zr, zi)),
      _ => None,
    },
    Expr::Function { body } => {
      // Substitute Slot(1) → Complex[zr, zi] and evaluate symbolically.
      let value = if zi.abs() < 1e-300 {
        Expr::Real(zr)
      } else {
        call("Complex", vec![Expr::Real(zr), Expr::Real(zi)])
      };
      let substituted = crate::syntax::substitute_slots(body, &[value]);
      let evaluated =
        crate::evaluator::evaluate_expr_to_expr(&substituted).ok()?;
      let n_form = n_eval(&evaluated).ok()?;
      complex_from_expr(&n_form)
        .or_else(|| super::numeric_utils::try_extract_complex_float(&n_form))
    }
    _ => None,
  }
}

/// Convert an evaluated numeric expression into a `(re, im)` pair when
/// possible. Recognises plain Reals, Complex[a, b], and Plus[real, im*I]
/// shapes.
fn complex_from_expr(e: &Expr) -> Option<(f64, f64)> {
  match e {
    Expr::Real(v) => Some((*v, 0.0)),
    Expr::Integer(n) => Some((*n as f64, 0.0)),
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      let r = match &args[0] {
        Expr::Real(v) => *v,
        Expr::Integer(n) => *n as f64,
        _ => return None,
      };
      let i = match &args[1] {
        Expr::Real(v) => *v,
        Expr::Integer(n) => *n as f64,
        _ => return None,
      };
      Some((r, i))
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut re = 0.0;
      let mut im = 0.0;
      for a in args {
        if let Some((r, i)) = complex_from_expr(a) {
          re += r;
          im += i;
        } else {
          im += imaginary_coeff(a)?;
        }
      }
      Some((re, im))
    }
    Expr::Identifier(s) if s == "I" => Some((0.0, 1.0)),
    // Times of (possibly complex) factors: multiply through using
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i, recursively. Enables shapes
    // like `Times[Plus[real, im*I], Complex[…, …]]` produced by `N[]`
    // on a rational of complex inputs to collapse to a single complex.
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      let mut re = 1.0f64;
      let mut im = 0.0f64;
      for a in args {
        let (ar, ai) = complex_from_expr(a)?;
        let (nr, ni) = (re * ar - im * ai, re * ai + im * ar);
        re = nr;
        im = ni;
      }
      Some((re, im))
    }
    _ => imaginary_coeff(e).map(|im| (0.0, im)),
  }
}

/// Recognise `Times[real, I]` (in either nesting) and return the real
/// coefficient.
fn imaginary_coeff(e: &Expr) -> Option<f64> {
  if let Expr::FunctionCall { name, args } = e
    && name == "Times"
  {
    let mut coef = 1.0f64;
    let mut has_i = false;
    for a in args {
      match a {
        Expr::Identifier(s) if s == "I" => has_i = true,
        Expr::Real(v) => coef *= *v,
        Expr::Integer(n) => coef *= *n as f64,
        _ => return None,
      }
    }
    if has_i {
      return Some(coef);
    }
  }
  None
}

fn complex_sin(zr: f64, zi: f64) -> (f64, f64) {
  // Sin(a + bi) = Sin(a) Cosh(b) + i Cos(a) Sinh(b)
  (zr.sin() * zi.cosh(), zr.cos() * zi.sinh())
}

fn complex_cos(zr: f64, zi: f64) -> (f64, f64) {
  // Cos(a + bi) = Cos(a) Cosh(b) - i Sin(a) Sinh(b)
  (zr.cos() * zi.cosh(), -zr.sin() * zi.sinh())
}

/// The cosine-sum windows: w(x) = Σ aₖ Cos[2πkx] on [-1/2, 1/2], zero
/// outside, with the exact rational coefficients wolframscript uses (all
/// with plus signs — negative lobes come from coefficient magnitudes, e.g.
/// FlatTopWindow[1/4] = -54736843/1000000000). Exact arguments build the
/// sum symbolically so the trig tables produce exact rationals and radical
/// forms; Real arguments evaluate in floating point. Symbolic arguments
/// stay unevaluated, like wolframscript.
pub fn cosine_sum_window_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated(name, args);
  if args.len() != 1 {
    return Ok(unevaluated());
  }
  // (numerators for a0, a1, …; common denominator)
  let (numerators, denom): (&[i128], i128) = match name {
    "BlackmanHarrisWindow" => (&[35875, 48829, 14128, 1168], 100000),
    "NuttallWindow" => (&[88942, 121849, 36058, 3151], 250000),
    "BlackmanNuttallWindow" => (&[3635819, 4891775, 1365995, 106411], 10000000),
    "FlatTopWindow" => (
      &[215578947, 416631580, 277263158, 83578947, 6947368],
      1000000000,
    ),
    // wolframscript's KaiserBesselWindow is this rational 4-term
    // cosine-sum approximation (fitted exactly from its outputs).
    "KaiserBesselWindow" => (&[402, 498, 99, 1], 1000),
    _ => return Ok(unevaluated()),
  };

  let Some(x) = try_eval_to_f64(&args[0]) else {
    return Ok(unevaluated());
  };
  let is_real = matches!(&args[0], Expr::Real(_) | Expr::BigFloat(..));
  if x.abs() > 0.5 {
    return Ok(if is_real {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  if is_real {
    // Integer-numerator terms summed in coefficient order, divided once at
    // the end. This matches wolframscript's machine values at most tested
    // points; the remaining last-ULP drift at some arguments is the known
    // summation-order/libm divergence class (no single ordering reproduces
    // wolframscript everywhere — reverse and value-sorted orders each fix
    // some points while breaking others).
    let pi = std::f64::consts::PI;
    let val: f64 = numerators
      .iter()
      .enumerate()
      .map(|(k, &n)| n as f64 * (2.0 * pi * k as f64 * x).cos())
      .sum::<f64>()
      / denom as f64;
    return Ok(Expr::Real(val));
  }

  // Exact argument: build Σ (nₖ/d) Cos[2πk x] and let the evaluator's exact
  // trig tables fold it (rational at the common grid points, radicals
  // elsewhere).
  let rational = |n: i128, d: i128| {
    call("Rational", vec![Expr::Integer(n), Expr::Integer(d)])
  };
  let terms: Vec<Expr> = numerators
    .iter()
    .enumerate()
    .map(|(k, &n)| {
      if k == 0 {
        rational(n, denom)
      } else {
        times(vec![
          rational(n, denom),
          call1(
            "Cos",
            times(vec![
              Expr::Integer(2 * k as i128),
              id_expr("Pi"),
              args[0].clone(),
            ]),
          ),
        ])
      }
    })
    .collect();
  crate::evaluator::evaluate_expr_to_expr(&plus(terms))
}

/// ListZTransform[{a0, a1, …}, z] — the finite Z transform
/// Sum[a_k z^(-k-n)] with the optional third argument n shifting the
/// starting index (default 0). An empty list gives {}; non-list first
/// arguments emit ::arg1 and echo, like wolframscript.
pub fn list_z_transform_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let uneval = || Ok(unevaluated("ListZTransform", args));
  let Expr::List(items) = &args[0] else {
    crate::emit_message(&format!(
      "ListZTransform::arg1: Expected a numeric array instead of {}.",
      expr_to_output(&args[0])
    ));
    return uneval();
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }
  let shift: i128 = match args.get(2) {
    None => 0,
    Some(Expr::Integer(n)) => *n,
    Some(_) => return uneval(),
  };
  let terms: Vec<Expr> = items
    .iter()
    .enumerate()
    .map(|(k, a)| {
      times(vec![
        a.clone(),
        pow(args[1].clone(), Expr::Integer(-(k as i128) - shift)),
      ])
    })
    .collect();
  let sum = plus(terms);
  crate::evaluator::evaluate_expr_to_expr(&sum)
}

/// DiscreteHadamardTransform[list] — the sequency-ordered Walsh–Hadamard
/// transform: the input is zero-padded to the next power of two N, each
/// output is a ±-signed sum scaled by 1/Sqrt[N], and the rows come in
/// sequency (Walsh) order — natural row bitreverse(grayEncode(s)) for
/// output slot s, matching wolframscript. Exact input stays exact (odd
/// log2 sizes give radical forms like 15/(2*Sqrt[2])); non-numeric data
/// emits ::data.
pub fn discrete_hadamard_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let uneval = || Ok(unevaluated("DiscreteHadamardTransform", args));
  if args.len() != 1 {
    return uneval();
  }
  let data_err = || {
    crate::emit_message(&format!(
      "DiscreteHadamardTransform::data: {} is not a numerical array.",
      expr_to_output(&args[0])
    ));
    Ok(unevaluated("DiscreteHadamardTransform", args))
  };
  let Expr::List(items) = &args[0] else {
    return data_err();
  };
  if items.is_empty()
    || !items
      .iter()
      .all(|e| try_eval_to_f64(e).is_some_and(f64::is_finite))
  {
    return data_err();
  }
  let is_real = items
    .iter()
    .any(|e| matches!(e, Expr::Real(_) | Expr::BigFloat(..)));
  let mut bits = 0u32;
  while (1usize << bits) < items.len() {
    bits += 1;
  }
  let n = 1usize << bits;
  if n > 1 << 16 {
    return uneval();
  }
  // Natural-order row for sequency slot s: bitreverse(gray(s)).
  let natural_row = |s: usize| -> usize {
    let g = s ^ (s >> 1);
    let mut r = 0usize;
    for b in 0..bits {
      if g & (1 << b) != 0 {
        r |= 1 << (bits - 1 - b);
      }
    }
    r
  };
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let mut out: Vec<Expr> = Vec::with_capacity(n);
  for s in 0..n {
    let r = natural_row(s);
    if is_real {
      let mut sum = 0.0f64;
      for (j, e) in items.iter().enumerate() {
        let v = try_eval_to_f64(e).unwrap_or(0.0);
        if (r & j).count_ones() % 2 == 0 {
          sum += v;
        } else {
          sum -= v;
        }
      }
      out.push(Expr::Real(sum / (n as f64).sqrt()));
    } else {
      let terms: Vec<Expr> = items
        .iter()
        .enumerate()
        .map(|(j, e)| {
          if (r & j).count_ones() % 2 == 0 {
            e.clone()
          } else {
            neg(e.clone())
          }
        })
        .collect();
      let scaled = times(vec![
        plus(terms),
        pow(
          Expr::Integer(n as i128),
          call("Rational", vec![Expr::Integer(-1), Expr::Integer(2)]),
        ),
      ]);
      out.push(eval(&scaled)?);
    }
  }
  Ok(Expr::List(out.into()))
}

/// CauchyWindow[x] / CauchyWindow[x, α] — 1/(1 + (2αx)²) on [-1/2, 1/2]
/// (default α = 3), zero outside; PoissonWindow[x] / [x, α] — E^(-2α|x|);
/// BartlettHannWindow[x] — 31/50 - (12/25)|x| + (19/50)Cos[2πx]. Exact
/// arguments evaluate symbolically (Poisson keeps wolframscript's
/// unfolded E^(-3/2) form); Reals numericize; symbolic arguments stay
/// unevaluated.
pub fn parametric_window_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let uneval = || Ok(unevaluated(name, args));
  let max_args = if name == "BartlettHannWindow" { 1 } else { 2 };
  if args.is_empty() || args.len() > max_args {
    return uneval();
  }
  let Some(x) = try_eval_to_f64(&args[0]) else {
    return uneval();
  };
  let is_real = matches!(&args[0], Expr::Real(_) | Expr::BigFloat(..))
    || matches!(args.get(1), Some(Expr::Real(_) | Expr::BigFloat(..)));
  if !x.is_finite() {
    return uneval();
  }
  if x.abs() > 0.5 {
    return Ok(if is_real {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }
  // The α parameter (Cauchy/Poisson only; default 3).
  let alpha_expr = args.get(1).cloned().unwrap_or(Expr::Integer(3));
  let Some(alpha) = try_eval_to_f64(&alpha_expr) else {
    return uneval();
  };
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let rational = |p: i128, q: i128| {
    call("Rational", vec![Expr::Integer(p), Expr::Integer(q)])
  };
  if is_real {
    let pi = std::f64::consts::PI;
    let v = match name {
      "CauchyWindow" => 1.0 / (1.0 + (2.0 * alpha * x).powi(2)),
      "PoissonWindow" => (-2.0 * alpha * x.abs()).exp(),
      _ => {
        31.0 / 50.0 - 12.0 / 25.0 * x.abs() + 19.0 / 50.0 * (2.0 * pi * x).cos()
      }
    };
    return Ok(Expr::Real(v));
  }
  let expr = match name {
    // 1/(1 + (2 α x)²)
    "CauchyWindow" => pow(
      plus(vec![
        Expr::Integer(1),
        pow(
          times(vec![Expr::Integer(2), alpha_expr, args[0].clone()]),
          Expr::Integer(2),
        ),
      ]),
      Expr::Integer(-1),
    ),
    // E^(-2 α |x|)
    "PoissonWindow" => pow(
      id_expr("E"),
      times(vec![
        Expr::Integer(-2),
        alpha_expr,
        call1("Abs", args[0].clone()),
      ]),
    ),
    // 31/50 - (12/25)|x| + (19/50)Cos[2 π x]
    _ => plus(vec![
      rational(31, 50),
      times(vec![rational(-12, 25), call1("Abs", args[0].clone())]),
      times(vec![
        rational(19, 50),
        call1(
          "Cos",
          times(vec![Expr::Integer(2), id_expr("Pi"), args[0].clone()]),
        ),
      ]),
    ]),
  };
  let _ = alpha;
  eval(&expr)
}

#[cfg(test)]
mod fourier_result_tests {
  use super::*;

  #[test]
  fn direct_complex_matches_the_evaluated_form() {
    for (re, im) in [(1.5, 2.5), (-1.5, 2.5), (1.5, -2.5), (-3.0, -0.25)] {
      let direct = fourier_result_to_expr(re, im, false);
      let evaluated = crate::evaluator::evaluate_function_call_ast(
        "Complex",
        &[Expr::Real(re), Expr::Real(im)],
      )
      .unwrap();
      assert_eq!(
        format!("{direct:?}"),
        format!("{evaluated:?}"),
        "{re} + {im} I"
      );
    }
  }
}
