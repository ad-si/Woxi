#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

// ─── Together ───────────────────────────────────────────────────────

/// Together[expr] - Combines fractions over a common denominator
/// Threads over List.
pub fn together_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Together expects exactly 1 argument".into(),
    ));
  }
  // Thread over List
  if let Expr::List(items) = &args[0] {
    let results: Vec<Expr> = items
      .iter()
      .map(|e| canonicalize_together_result(&together_expr(e)))
      .collect();
    return Ok(Expr::List(results.into()));
  }
  // An argument that evaluates to a plain number is left as that number.
  // In particular wolframscript keeps Together[1/3 + I/3] as the complex
  // scalar 1/3 + I/3 rather than the rebuilt quotient (1 + I)/3 (which is
  // also what the fraction-combining path spun into an infinite
  // Together/Cancel/Factor recursion on). Constant expressions that are
  // not number atoms still combine: Together[1 - E^(-2)] = (-1 + E^2)/E^2.
  {
    let mut vars = std::collections::HashSet::new();
    super::simplify::collect_variables(&args[0], &mut vars);
    vars.remove("I");
    if vars.is_empty()
      && let Ok(value) = crate::evaluator::evaluate_expr_to_expr(&args[0])
      && (matches!(
        &value,
        Expr::Integer(_)
          | Expr::Real(_)
          | Expr::BigInteger(_)
          | Expr::BigFloat(_, _)
      ) || matches!(&value, Expr::FunctionCall { name, .. } if name == "Rational")
        || crate::functions::predicate_ast::is_complex_number(&value))
    {
      return Ok(value);
    }
  }
  Ok(canonicalize_together_result(&together_expr(&args[0])))
}

/// Apply the wolframscript quotient-sign canonicalization to a Together
/// result. Only the user-facing `together_ast` does this — internal callers
/// (e.g. the Simplify pipeline, whose flip rule is weaker) keep the raw
/// quotient sign from `together_expr`.
fn canonicalize_together_result(expr: &Expr) -> Expr {
  // A complex-rational scalar is an atom: wolframscript's Together leaves
  // 1/3 + I/3 as-is rather than rebuilding it as the quotient (1 + I)/3
  // (extract_num_den would read the rational parts as a denominator).
  if matches!(expr, Expr::FunctionCall { name, .. } if name == "Complex") {
    return expr.clone();
  }
  let (num, den) = extract_num_den(expr);
  if matches!(&den, Expr::Integer(1)) {
    return expr.clone();
  }
  let result = canonicalize_quotient_sign(&num, &den, false)
    .unwrap_or_else(|| expr.clone());
  // Shared numeric content between numerator and denominator cancels:
  // Together[(4+2x)/(6x)] → (2+x)/(3x), (4+2x)/6 → (2+x)/3
  // (wolframscript-verified; differential fuzzer, seed
  // 1783631489573774000).
  let (rnum, rden) = extract_num_den(&result);
  if matches!(&rden, Expr::Integer(1)) {
    return result;
  }
  if let Some((rn, rd)) = reduce_shared_integer_content(&rnum, &rden) {
    if matches!(&rd, Expr::Integer(1)) {
      return rn;
    }
    return Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(rn),
      right: Box::new(rd),
    };
  }
  result
}

/// Cancel the gcd of a sum (or content-wrapped) numerator's integer
/// content with the denominator's integer factor: (4+2x)/(6x) →
/// (2+x)/(3x), (4+2x)/6 → (2+x)/3 (wolframscript Together/Cancel both
/// reduce). Returns None when nothing cancels.
fn reduce_shared_integer_content(
  num: &Expr,
  den: &Expr,
) -> Option<(Expr, Expr)> {
  fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
  }
  let mut den_factors = flatten_times_args(std::slice::from_ref(den));
  let (den_idx, den_int) =
    den_factors.iter().enumerate().find_map(|(i, f)| match f {
      Expr::Integer(k) if *k > 1 => Some((i, *k)),
      _ => None,
    })?;
  // A content-wrapped numerator (Times[c, …] from an earlier hoist)
  // cancels its coefficient against the denominator's integer factor.
  let num_factors = flatten_times_args(std::slice::from_ref(num));
  if num_factors.len() >= 2
    && let Some(c) = num_factors.iter().find_map(|f| match f {
      Expr::Integer(c) if c.abs() > 1 => Some(*c),
      _ => None,
    })
  {
    let g = gcd(c.abs(), den_int);
    if g <= 1 {
      return None;
    }
    let mut new_num_factors: Vec<Expr> = Vec::new();
    let mut used = false;
    for f in num_factors {
      match f {
        Expr::Integer(cf) if cf == c && !used => {
          used = true;
          if c / g != 1 {
            new_num_factors.push(Expr::Integer(c / g));
          }
        }
        // extract_num_den can leave a stray unit factor behind
        Expr::Integer(1) => {}
        other => new_num_factors.push(other),
      }
    }
    let new_num = match new_num_factors.len() {
      0 => Expr::Integer(1),
      1 => new_num_factors.remove(0),
      _ => build_product(new_num_factors),
    };
    if den_int / g == 1 {
      den_factors.remove(den_idx);
    } else {
      den_factors[den_idx] = Expr::Integer(den_int / g);
    }
    let new_den = match den_factors.len() {
      0 => Expr::Integer(1),
      1 => den_factors.remove(0),
      _ => build_product(den_factors),
    };
    return Some((new_num, new_den));
  }
  // A sum numerator divides its integer content termwise.
  let num_terms = collect_additive_terms(num);
  if num_terms.len() < 2 {
    return None;
  }
  let (n, d, _) = super::factor::rational_content(&num_terms)?;
  if d != 1 || n.abs() <= 1 {
    return None;
  }
  let g = gcd(n.abs(), den_int);
  if g <= 1 {
    return None;
  }
  let divided: Result<Vec<Expr>, _> = num_terms
    .iter()
    .map(|t| {
      crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(t.clone()),
        right: Box::new(Expr::Integer(g)),
      })
    })
    .collect();
  let mut divided = divided.ok()?;
  let new_num = if divided.len() == 1 {
    divided.remove(0)
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: divided.into(),
    }
  };
  if den_int / g == 1 {
    den_factors.remove(den_idx);
  } else {
    den_factors[den_idx] = Expr::Integer(den_int / g);
  }
  let new_den = match den_factors.len() {
    0 => Expr::Integer(1),
    1 => den_factors.remove(0),
    _ => build_product(den_factors),
  };
  Some((new_num, new_den))
}

/// Extract numerator and denominator from an expression.
/// Handles BinaryOp::Divide, Rational, Power[..., -1], and
/// Times[..., Power[..., -1]] forms.
pub fn extract_num_den(expr: &Expr) -> (Expr, Expr) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      (args[0].clone(), args[1].clone())
    }
    // Power[base, -n] => 1/base^n (FunctionCall form) — handles integer and rational exponents
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Some(pos_exp) = get_negative_exponent(&args[1]) {
        if matches!(&pos_exp, Expr::Integer(1)) {
          (Expr::Integer(1), args[0].clone())
        } else {
          (
            Expr::Integer(1),
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![args[0].clone(), pos_exp].into(),
            },
          )
        }
      } else {
        (expr.clone(), Expr::Integer(1))
      }
    }
    // Times[factors...] — split into numerator factors and denominator factors
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      // Flatten nested Times first
      let flat_args = flatten_times_args(args);
      let mut num_factors = Vec::new();
      let mut den_factors = Vec::new();
      for arg in &flat_args {
        match arg {
          Expr::FunctionCall {
            name: pname,
            args: pargs,
          } if pname == "Power" && pargs.len() == 2 => {
            if let Some(pos_exp) = get_negative_exponent(&pargs[1]) {
              if matches!(&pos_exp, Expr::Integer(1)) {
                den_factors.push(pargs[0].clone());
              } else {
                den_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![pargs[0].clone(), pos_exp].into(),
                });
              }
            } else {
              // Power[fraction, n] → split base into num/den
              let (base_num, base_den) = extract_num_den(&pargs[0]);
              if !matches!(&base_den, Expr::Integer(1)) {
                num_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![base_num, pargs[1].clone()].into(),
                });
                den_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![base_den, pargs[1].clone()].into(),
                });
              } else {
                num_factors.push(arg.clone());
              }
            }
          }
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } => {
            if let Some(pos_exp) = get_negative_exponent(right) {
              if matches!(&pos_exp, Expr::Integer(1)) {
                den_factors.push(*left.clone());
              } else {
                den_factors.push(Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![*left.clone(), pos_exp].into(),
                });
              }
            } else {
              // Power[fraction, n] → split base into num/den
              let (base_num, base_den) = extract_num_den(left);
              if !matches!(&base_den, Expr::Integer(1)) {
                num_factors.push(Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(base_num),
                  right: right.clone(),
                });
                den_factors.push(Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(base_den),
                  right: right.clone(),
                });
              } else {
                num_factors.push(arg.clone());
              }
            }
          }
          // BinaryOp::Divide inside Times: split into num/den
          Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left,
            right,
          } => {
            num_factors.push(*left.clone());
            den_factors.push(*right.clone());
          }
          // Rational[n,d] inside Times: split numerator n and denominator d
          Expr::FunctionCall {
            name: rname,
            args: rargs,
          } if rname == "Rational" && rargs.len() == 2 => {
            // Denominator is always positive after make_rational normalisation
            num_factors.push(rargs[0].clone());
            den_factors.push(rargs[1].clone());
          }
          _ => num_factors.push(arg.clone()),
        }
      }
      if den_factors.is_empty() {
        (expr.clone(), Expr::Integer(1))
      } else {
        let num = build_product(num_factors);
        let den = build_product(den_factors);
        (num, den)
      }
    }
    // BinaryOp::Power — handle negative exponents and Power[fraction, n]
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Some(pos_exp) = get_negative_exponent(right) {
        if matches!(&pos_exp, Expr::Integer(1)) {
          (Expr::Integer(1), *left.clone())
        } else {
          (
            Expr::Integer(1),
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: left.clone(),
              right: Box::new(pos_exp),
            },
          )
        }
      } else {
        // Power[num/den, n] → (num^n, den^n)
        let (base_num, base_den) = extract_num_den(left);
        if !matches!(&base_den, Expr::Integer(1)) {
          let num = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(base_num),
            right: right.clone(),
          };
          let den = Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(base_den),
            right: right.clone(),
          };
          (num, den)
        } else {
          (expr.clone(), Expr::Integer(1))
        }
      }
    }
    // BinaryOp::Times — split into numerator and denominator factors
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      // Flatten into a Times FunctionCall and recurse
      let flat = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![*left.clone(), *right.clone()].into(),
      };
      extract_num_den(&flat)
    }
    // UnaryOp::Minus — negate the numerator
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (num, den) = extract_num_den(operand);
      (negate_expr(&num), den)
    }
    _ => (expr.clone(), Expr::Integer(1)),
  }
}

/// Flatten nested Times args: Times[a, Times[b, c]] → [a, b, c]
pub fn flatten_times_args(args: &[Expr]) -> Vec<Expr> {
  let mut flat = Vec::new();
  for arg in args {
    match arg {
      Expr::FunctionCall { name, args: inner } if name == "Times" => {
        flat.extend(flatten_times_args(inner));
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        flat.extend(flatten_times_args(&[*left.clone(), *right.clone()]));
      }
      _ => flat.push(arg.clone()),
    }
  }
  flat
}

/// Check if an expression is a negative integer and return its absolute value
pub fn get_negative_integer(expr: &Expr) -> Option<i64> {
  match expr {
    Expr::Integer(n) if *n < 0 => Some((-*n) as i64),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => match operand.as_ref() {
      Expr::Integer(n) if *n > 0 => Some(*n as i64),
      _ => None,
    },
    _ => None,
  }
}

/// Check if an expression is a negative exponent (integer or rational)
/// and return the negated (positive) exponent.
pub fn get_negative_exponent(expr: &Expr) -> Option<Expr> {
  // Try integer first
  if let Some(neg) = get_negative_integer(expr) {
    return Some(Expr::Integer(neg as i128));
  }
  // Try Rational[-n, d]
  if let Expr::FunctionCall { name, args } = expr
    && name == "Rational"
    && args.len() == 2
    && let Expr::Integer(n) = &args[0]
    && *n < 0
  {
    return Some(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(-*n), args[1].clone()].into(),
    });
  }
  // Try Times[-1, Rational[p, q]] → negative rational
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
    && args.len() == 2
    && matches!(&args[0], Expr::Integer(-1))
    && let Expr::FunctionCall { name: rn, args: ra } = &args[1]
    && rn == "Rational"
    && ra.len() == 2
    && let Expr::Integer(n) = &ra[0]
    && *n > 0
  {
    return Some(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(*n), ra[1].clone()].into(),
    });
  }
  None
}

/// Negate an expression
pub(super) fn negate_expr(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(n) => Expr::Integer(-n),
    Expr::Real(f) => Expr::Real(-f),
    // Rational[a, b] is stored as a FunctionCall — negate the numerator
    // so `-Rational[-49, 8]` collapses to `Rational[49, 8]` rather than
    // surfacing a stale `Minus` wrapper that displays as `--49/8`.
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let Expr::Integer(num) = &args[0] {
        return Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-num), args[1].clone()].into(),
        };
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(expr.clone()),
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => *operand.clone(),
    _ => Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(expr.clone()),
    },
  }
}

/// Sign a quotient canonicalization treats an additive expression as
/// carrying. Univariate sums use the highest-degree term's coefficient
/// (Cancel[(2+x)/(4+x-4x^2)] → (-2-x)/(-4-x+4x^2)); multivariate sums use
/// the first non-numeric term in canonical order (Together[x/(b - a*c)]
/// and x/(a*c - d) keep their denominators, x/(d - a*c) flips to a*c - d,
/// (1+a)/(y-x) flips to x-y). None when a coefficient is non-rational.
fn additive_content_sign(expr: &Expr) -> Option<i128> {
  let terms = collect_additive_terms(expr);
  let mut vars = std::collections::HashSet::new();
  super::simplify::collect_variables(expr, &mut vars);
  if vars.len() <= 1 {
    return super::factor::rational_content(&terms)
      .map(|(n, _, _)| if n < 0 { -1 } else { 1 });
  }
  for t in &terms {
    let mut tv = std::collections::HashSet::new();
    super::simplify::collect_variables(t, &mut tv);
    if tv.is_empty() {
      continue;
    }
    return super::factor::rational_content(std::slice::from_ref(t))
      .map(|(n, _, _)| if n < 0 { -1 } else { 1 });
  }
  None
}

/// wolframscript canonicalizes the sign of evaluated quotients: every
/// variable-bearing sum factor of the denominator gets a positive content
/// sign — Together[x/((1-x)*(2+x))] → -(x/((-1+x)*(2+x))), including Power
/// bases (Together[1/(1-x) + 1/(1-x)^2] → (2-x)/(-1+x)^2) — and the
/// numerator absorbs the accumulated sign. A numerator of literal 1 cannot
/// absorb an odd flip: Cancel[2/(2-2x)] stays (1-x)^(-1).
///
/// With `require_negative_numerator` (Simplify's weaker rule) the flip only
/// happens when the numerator is constant or its own content sign is
/// negative: Simplify[(-1-5x)/(3-x)] → (1+5x)/(-3+x) and Simplify[3/(1-x)]
/// → -3/(-1+x), but Simplify[(1+x)/(1-x)] keeps its form.
///
/// Returns None when nothing changes.
pub(super) fn canonicalize_quotient_sign(
  num: &Expr,
  den: &Expr,
  require_negative_numerator: bool,
) -> Option<Expr> {
  // Slot-bearing quotients (e.g. the Möbius inverse (-b + #1*d)/(a - #1*c)
  // built by InverseFunction) keep the form their construction produced.
  fn contains_slot(e: &Expr) -> bool {
    match e {
      Expr::Slot(_) | Expr::SlotSequence(_) => true,
      Expr::FunctionCall { args, .. } => args.iter().any(contains_slot),
      Expr::List(items) => items.iter().any(contains_slot),
      Expr::BinaryOp { left, right, .. } => {
        contains_slot(left) || contains_slot(right)
      }
      Expr::UnaryOp { operand, .. } => contains_slot(operand),
      _ => false,
    }
  }
  if contains_slot(num) || contains_slot(den) {
    return None;
  }
  let negate = |e: &Expr| {
    expand_and_combine(&Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(e.clone()),
    })
  };

  let factors = flatten_times_args(std::slice::from_ref(den));
  // Together/Cancel split powered PRODUCT factors so numeric content can
  // hoist through the power: x/(-2+2x)^2 has den (2*(-1+x))^2 → 4,
  // (-1+x)^2 (wolframscript: x/(4*(-1+x)^2)); an odd power of a negative
  // numeric ((-2)^3 → -8) feeds the numeric flip below.
  let mut presplit_changed = false;
  let factors = if require_negative_numerator {
    factors
  } else {
    let mut split: Vec<Expr> = Vec::new();
    for f in factors {
      let (b, e) = match &f {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } => match right.as_ref() {
          Expr::Integer(e) => ((**left).clone(), *e),
          _ => {
            split.push(f);
            continue;
          }
        },
        Expr::FunctionCall { name, args }
          if name == "Power" && args.len() == 2 =>
        {
          match &args[1] {
            Expr::Integer(e) => (args[0].clone(), *e),
            _ => {
              split.push(f);
              continue;
            }
          }
        }
        _ => {
          split.push(f);
          continue;
        }
      };
      let subs = flatten_times_args(std::slice::from_ref(&b));
      if subs.len() < 2 || !(1..=40).contains(&e) {
        split.push(f);
        continue;
      }
      presplit_changed = true;
      for sub in subs {
        match &sub {
          Expr::Integer(n) => match n.checked_pow(e as u32) {
            Some(p) => split.push(Expr::Integer(p)),
            None => split.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(sub.clone()),
              right: Box::new(Expr::Integer(e)),
            }),
          },
          _ => {
            if e == 1 {
              split.push(sub);
            } else {
              split.push(Expr::BinaryOp {
                op: BinaryOperator::Power,
                left: Box::new(sub),
                right: Box::new(Expr::Integer(e)),
              });
            }
          }
        }
      }
    }
    split
  };
  let mut sign: i128 = 1;
  let mut flipped_any = false;
  // Structural denominator change without a sign flip (Together/Cancel
  // content hoist: x/(5+5x) → x/(5*(1+x))).
  let mut den_changed = presplit_changed;
  let mut any_flipped_content_one = false;
  // Positive magnitude of numeric denominator factors whose sign was
  // absorbed (Together pre-factors 2 - 2x into -2*(-1 + x); the -2 must
  // not stay in the denominator).
  let mut numeric_flip_content: i128 = 1;
  let mut new_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    // A negative numeric factor flips outright: Together[(3-5x)/(2-2x)]
    // is (-3+5x)/(2*(-1+x)) in wolframscript, never (3-5x)/(-2*(-1+x)).
    // The Simplify caller (require_negative_numerator) keeps its own
    // decoded rules and is left untouched.
    if !require_negative_numerator
      && let Expr::Integer(n) = f
      && *n < 0
    {
      sign = -sign;
      flipped_any = true;
      numeric_flip_content = numeric_flip_content.saturating_mul(-n);
      if *n != -1 {
        new_factors.push(Expr::Integer(-n));
      }
      continue;
    }
    let (base, exp) = match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => match right.as_ref() {
        Expr::Integer(e) => ((**left).clone(), *e),
        _ => {
          new_factors.push(f.clone());
          continue;
        }
      },
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        match &args[1] {
          Expr::Integer(e) => (args[0].clone(), *e),
          _ => {
            new_factors.push(f.clone());
            continue;
          }
        }
      }
      other => (other.clone(), 1),
    };
    // Complex-coefficient sums are never flipped: rational_content would
    // read `1 - I*x` as negative content (the -I coefficient), but
    // wolframscript keeps Simplify[1/(1-I x) + 1/(1+I x)] → 2/(1+x^2).
    let flippable_sum = collect_additive_terms(&base).len() >= 2
      && !super::reduce::contains_imaginary(&base)
      && {
        let mut vars = std::collections::HashSet::new();
        super::simplify::collect_variables(&base, &mut vars);
        !vars.is_empty()
      };
    if flippable_sum && additive_content_sign(&base) == Some(-1) {
      flipped_any = true;
      if exp % 2 != 0 {
        sign = -sign;
      }
      let base_content =
        super::factor::rational_content(&collect_additive_terms(&base))
          .map(|(n, d, _)| (n.abs(), d))
          .unwrap_or((1, 1));
      if base_content == (1, 1) {
        any_flipped_content_one = true;
      }
      let mut neg_base = negate(&base);
      // A flipped denominator factor leaves its integer content behind as
      // a plain numeric factor, raised to the factor's power (term-wise
      // division; the whole-sum quotient would stay an unreduced Divide):
      // Simplify[(2-x)/(5-5x)] → (-2+x)/(5*(-1+x)), Cancel[x/(2-2x)] →
      // -1/2*x/(-1+x), Cancel[x/(2-2x)^2] → x/(4*(-1+x)^2); all
      // wolframscript-verified.
      if base_content.1 == 1
        && base_content.0 > 1
        && exp >= 1
        && let Ok(exp_u32) = u32::try_from(exp)
        && let Some(content_pow) = base_content.0.checked_pow(exp_u32)
      {
        let divided: Result<Vec<Expr>, _> = collect_additive_terms(&neg_base)
          .iter()
          .map(|t| {
            crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
              op: BinaryOperator::Divide,
              left: Box::new(t.clone()),
              right: Box::new(Expr::Integer(base_content.0)),
            })
          })
          .collect();
        if let Ok(terms) = divided {
          neg_base = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: terms.into(),
          };
          new_factors.push(Expr::Integer(content_pow));
        }
      }
      new_factors.push(if exp == 1 {
        neg_base
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(neg_base),
          right: Box::new(Expr::Integer(exp)),
        }
      });
    } else if !require_negative_numerator
      && flippable_sum
      && (1..=40).contains(&exp)
      && let Some((n, 1, _)) =
        super::factor::rational_content(&collect_additive_terms(&base))
      && n > 1
      && let Some(content_pow) = n.checked_pow(exp as u32)
    {
      // Positive-content hoist without a flip: Together/Cancel display
      // x/(5+5x) as x/(5*(1+x)) (wolframscript-verified, incl. powers
      // and multivariate sums: (2+x*y)/(4+4x*y) → (2+x*y)/(4*(1+x*y))).
      let divided: Result<Vec<Expr>, _> = collect_additive_terms(&base)
        .iter()
        .map(|t| {
          crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(t.clone()),
            right: Box::new(Expr::Integer(n)),
          })
        })
        .collect();
      match divided {
        Ok(terms) => {
          den_changed = true;
          let prim = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: terms.into(),
          };
          new_factors.push(Expr::Integer(content_pow));
          new_factors.push(if exp == 1 {
            prim
          } else {
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(prim),
              right: Box::new(Expr::Integer(exp)),
            }
          });
        }
        Err(_) => new_factors.push(f.clone()),
      }
    } else {
      new_factors.push(f.clone());
    }
  }
  if !flipped_any {
    // Cancel and Together fold a bare unit-negative reciprocal into its
    // base: -1/(1+x) → (-1-x)^(-1) and -1/(-1+x) → (1-x)^(-1). (Simplify
    // keeps -(1+x)^(-1), so this is gated to the non-Simplify callers.)
    // Only fires when the numerator is exactly -1 and the denominator is a
    // single sum: a product denominator keeps a rational coefficient
    // (-1/(2+2x) → -1/2*1/(1+x)), and a negative-content sum was already
    // handled by the flip above.
    let den_is_bare_sum = matches!(
      den,
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        ..
      }
    ) || matches!(den, Expr::FunctionCall { name, .. } if name == "Plus");
    if !require_negative_numerator
      && matches!(num, Expr::Integer(-1))
      && den_is_bare_sum
    {
      return Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(negate(den)),
        right: Box::new(Expr::Integer(-1)),
      });
    }
    // No sign flip, but the denominator's structure changed (content
    // hoist / powered-product split): rebuild the quotient.
    if den_changed {
      let new_den = if new_factors.len() == 1 {
        new_factors.remove(0)
      } else {
        crate::functions::math_ast::sort_symbolic_factors(&mut new_factors);
        build_product(new_factors)
      };
      return Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(num.clone()),
        right: Box::new(new_den),
      });
    }
    return None;
  }
  // A flipped numeric content with a numerator that can't absorb the sign
  // termwise becomes a rational prefactor, with the content leaving the
  // denominator: Together[1/(2-2x)] → -1/2*1/(-1+x) and
  // Together[x/(6-3x)] → -1/3*x/(-2+x), matching wolframscript.
  let prefactor_form =
    |num_factor: Option<Expr>, factors: &[Expr], content: i128| -> Expr {
      let rest: Vec<Expr> = factors
        .iter()
        .filter(|f| !matches!(f, Expr::Integer(n) if *n == content))
        .cloned()
        .collect();
      let rest_den = if rest.len() == 1 {
        rest.into_iter().next().unwrap()
      } else {
        build_product(rest)
      };
      let mut parts: Vec<Expr> =
        vec![crate::functions::math_ast::make_rational_pub(-1, content)];
      if let Some(nf) = num_factor {
        parts.push(nf);
      }
      parts.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(rest_den),
        right: Box::new(Expr::Integer(-1)),
      });
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: parts.into(),
      }
    };
  if sign < 0 && matches!(num, Expr::Integer(1)) {
    if numeric_flip_content > 1 {
      return Some(prefactor_form(None, &new_factors, numeric_flip_content));
    }
    return None;
  }
  if require_negative_numerator {
    let constant_num = {
      let mut vars = std::collections::HashSet::new();
      super::simplify::collect_variables(num, &mut vars);
      vars.is_empty()
    };
    // The numerator must be able to absorb the flip: a constant, a sum
    // whose every term is nonpositive, or — for a mixed-sign numerator
    // with a negative leading coefficient — only when the flip is
    // "free": the flipped denominator factor has integer content 1, or
    // the numerator carries a bare unit-negative monomial (-x) whose
    // sign disappears. Simplify[(2-x)/(1-x)] → (-2+x)/(-1+x) and
    // Simplify[(2-x)/(5-5x)] → (-2+x)/(5*(-1+x)) flip, but
    // Simplify[(5-4x-3x^2)/(5-5x)] and Simplify[(5-4x)/(5-5x)] keep
    // their form (differential fuzzer, seed 1783537668073123846; all
    // wolframscript-verified).
    let num_terms = collect_additive_terms(num);
    let all_nonpositive_num = num_terms.iter().all(|t| {
      super::factor::rational_content(std::slice::from_ref(t))
        .is_some_and(|(n, _, _)| n <= 0)
    });
    let mixed_leading_neg = additive_content_sign(num) == Some(-1);
    let has_unit_neg_term = num_terms.iter().any(|t| {
      let mut vars = std::collections::HashSet::new();
      super::simplify::collect_variables(t, &mut vars);
      !vars.is_empty()
        && super::factor::rational_content(std::slice::from_ref(t))
          .is_some_and(|(_, _, coeffs)| coeffs.first() == Some(&(-1, 1)))
    });
    let mixed_flip_is_free =
      mixed_leading_neg && (any_flipped_content_one || has_unit_neg_term);
    if sign >= 0
      || (!constant_num && !all_nonpositive_num && !mixed_flip_is_free)
    {
      return None;
    }
  }

  let new_den = if new_factors.len() == 1 {
    new_factors.remove(0)
  } else {
    crate::functions::math_ast::sort_symbolic_factors(&mut new_factors);
    build_product(new_factors)
  };

  let new_num = if sign >= 0 {
    num.clone()
  } else if collect_additive_terms(num).len() >= 2
    || matches!(num, Expr::Integer(_) | Expr::Real(_) | Expr::UnaryOp { .. })
    || matches!(num, Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2)
    || additive_content_sign(num) == Some(-1)
  {
    negate(num)
  } else {
    // A monomial numerator with an integer coefficient absorbs the flip
    // into that coefficient, and any numeric denominator content stays
    // put: Together[(5*x)/(1-5*x)] → (-5*x)/(-1+5*x) and
    // Together[(3*x)/(2-2*x)] → (-3*x)/(2*(-1+x)), matching
    // wolframscript (differential fuzzer, seed 8887).
    let num_factors = flatten_times_args(std::slice::from_ref(num));
    if num_factors
      .iter()
      .any(|f| matches!(f, Expr::Integer(n) if n.abs() > 1))
    {
      // Negate the integer factor in place — expanding the negation
      // would distribute over Plus factors, but wolframscript keeps
      // Together[(5*(1+x))/((1-x)*(2+x))] → (-5*(1+x))/((-1+x)*(2+x)).
      let mut negated = false;
      let neg_factors: Vec<Expr> = num_factors
        .iter()
        .map(|f| match f {
          Expr::Integer(n) if !negated && n.abs() > 1 => {
            negated = true;
            Expr::Integer(-n)
          }
          other => other.clone(),
        })
        .collect();
      return Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(build_product(neg_factors)),
        right: Box::new(new_den),
      });
    }
    // A unit monomial can't absorb the flip. With flipped numeric
    // content, the sign and content become a rational prefactor
    // (-1/3*x/(-2+x)); remaining numeric denominator content hoists the
    // same way (Together[(x/2)/(1-x)] → -1/2*x/(-1+x)); only a
    // content-free denominator pulls the sign out of the whole
    // quotient: -(x/((-1+x)*(2+x))) and -((x*y)/(-1+x)).
    if numeric_flip_content > 1 {
      let factors = flatten_times_args(std::slice::from_ref(&new_den));
      return Some(prefactor_form(
        Some(num.clone()),
        &factors,
        numeric_flip_content,
      ));
    }
    let den_factors = flatten_times_args(std::slice::from_ref(&new_den));
    if let Some(k) = den_factors.iter().find_map(|f| match f {
      Expr::Integer(n) if *n > 1 => Some(*n),
      _ => None,
    }) {
      let stripped: Vec<Expr> = num_factors
        .iter()
        .filter(|f| !matches!(f, Expr::Integer(1)))
        .cloned()
        .collect();
      let num_clean = if stripped.is_empty() {
        Expr::Integer(1)
      } else if stripped.len() == 1 {
        stripped.into_iter().next().unwrap()
      } else {
        build_product(stripped)
      };
      return Some(prefactor_form(Some(num_clean), &den_factors, k));
    }
    return Some(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(num.clone()),
        right: Box::new(new_den),
      }),
    });
  };
  // A numerator flipped to exactly 1 displays as a reciprocal power:
  // Simplify[-1/(1-x)] → (-1+x)^(-1). A product denominator stays a
  // quotient (1/(2*(-1+2*x)), never (2*(-1+2*x))^(-1)) — the numeric
  // content must show as a rational coefficient, so exclude a Times
  // denominator in either representation (BinaryOp or FunctionCall).
  let den_is_product = matches!(
    &new_den,
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    }
  ) || matches!(&new_den, Expr::FunctionCall { name, .. } if name == "Times");
  if matches!(&new_num, Expr::Integer(1)) && !den_is_product {
    return Some(Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(new_den),
      right: Box::new(Expr::Integer(-1)),
    });
  }
  Some(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(new_num),
    right: Box::new(new_den),
  })
}

/// Wolfram's Simplify pulls a `-1` out in front of a quotient when the
/// numerator's content sign is negative (its highest-degree coefficient
/// for a univariate sum): Simplify[(-2-3x)/(4x+3x²)] →
/// -((2+3x)/(4x+3x²)) and Simplify[(1-5x-3x²-x³)/(-1-2x+4x²)] →
/// -((-1+5x+3x²+x³)/(-1-2x+4x²)). An entirely-nonpositive denominator
/// flips too — Simplify[(2+3x)/(-4x-3x²)] → -((2+3x)/(4x+3x²)) — while
/// a mixed-sign denominator like 1-x stays put (Simplify[x/(1-x)] keeps
/// its form; that case is canonicalize_quotient_sign's). Two flips
/// cancel. Returns None when nothing changes.
pub(super) fn extract_quotient_minus(num: &Expr, den: &Expr) -> Option<Expr> {
  let negate = |e: &Expr| {
    expand_and_combine(&Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(e.clone()),
    })
  };
  // Only UNIVARIATE polynomial sums participate: wolframscript keeps
  // (-b + d*y)/(a - c*y) and (1 - Cos[2x])/2 exactly as they are.
  let is_variable_sum = |e: &Expr| {
    collect_additive_terms(e).len() >= 2
      && !super::reduce::contains_imaginary(e)
      && super::simplify::polynomial_like(e)
      && {
        let mut vars = std::collections::HashSet::new();
        super::simplify::collect_variables(e, &mut vars);
        vars.len() == 1
      }
  };
  let all_terms_nonpositive = |e: &Expr| {
    collect_additive_terms(e).iter().all(|t| {
      super::factor::rational_content(std::slice::from_ref(t))
        .is_some_and(|(n, _, _)| n <= 0)
    })
  };

  // A numerator flip needs the denominator already in canonical
  // (positive-leading) form: Simplify[(1-5x-3x²-x³)/(-1-2x+4x²)] pulls
  // the minus out, but Simplify[(5-4x-3x²)/(5-5x)] keeps its form.
  let num_flip = is_variable_sum(num)
    && additive_content_sign(num) == Some(-1)
    && additive_content_sign(den) == Some(1);
  let den_flip = is_variable_sum(den) && all_terms_nonpositive(den);
  if !num_flip && !den_flip {
    return None;
  }
  let new_num = if num_flip { negate(num) } else { num.clone() };
  let new_den = if den_flip { negate(den) } else { den.clone() };
  let quotient = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(new_num),
    right: Box::new(new_den),
  };
  if num_flip != den_flip {
    Some(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(quotient),
    })
  } else {
    Some(quotient)
  }
}

/// Recursively run `together_expr` on sub-expressions so that nested fractions
/// inside Power bases, Divide operands, and Times factors are combined first.
/// Also rewrites `1 / (a/b)` → `b/a` (Power[Divide[a,b], -1]) so the outer
/// Together pass sees a clean rational form.
fn together_expr_preprocess(expr: &Expr) -> Expr {
  match expr {
    // Leaf
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::Constant(_)
    | Expr::Identifier(_) => expr.clone(),

    // a/b: Together each operand, then return as division — the outer
    // together_expr will see it as a single fraction via extract_num_den.
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let num = together_expr(left);
      let den = together_expr(right);
      // If the denominator itself is now a fraction p/q, flip: a / (p/q) → a*q/p
      let (d_num, d_den) = extract_num_den(&den);
      if !matches!(&d_den, Expr::Integer(1)) {
        let new_num = multiply_exprs(&num, &d_den);
        let new_den = d_num;
        return make_fraction(new_num, new_den);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(num),
        right: Box::new(den),
      }
    }

    // a^b: Together the base; for negative integer exponents, flip if base is a fraction
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let base = together_expr(left);
      let exp = together_expr_preprocess(right);
      // If exp is -1 and base is a fraction p/q, return q/p
      if matches!(&exp, Expr::Integer(-1)) {
        let (b_num, b_den) = extract_num_den(&base);
        if !matches!(&b_den, Expr::Integer(1)) {
          return make_fraction(b_den, b_num);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(exp),
      }
    }

    // Power[base, exp] as FunctionCall form
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      let base = together_expr(&args[0]);
      let exp = together_expr_preprocess(&args[1]);
      if matches!(&exp, Expr::Integer(-1)) {
        let (b_num, b_den) = extract_num_den(&base);
        if !matches!(&b_den, Expr::Integer(1)) {
          return make_fraction(b_den, b_num);
        }
      }
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![base, exp].into(),
      }
    }

    // Times (binary): each factor must be fully combined so that a Plus
    // factor like `(1/x + 1/y)` becomes a single fraction before the product
    // is taken. Otherwise `Together[3 (1/x + 1/y)]` leaves the inner sum
    // uncombined (the outer together_expr sees the whole product as one
    // additive term and returns it untouched).
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(together_expr(left)),
      right: Box::new(together_expr(right)),
    },

    // Binary plus/minus: recurse into each side. together_expr itself will
    // combine additive terms after preprocessing.
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(together_expr_preprocess(left)),
      right: Box::new(together_expr_preprocess(right)),
    },

    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(together_expr_preprocess(operand)),
    },

    // Plus as FunctionCall: recurse into each argument; the outer
    // together_expr combines the additive terms afterwards.
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let new_args: Vec<Expr> =
        args.iter().map(together_expr_preprocess).collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }

    // Times as FunctionCall: fully combine each factor (see the binary Times
    // case above) so a Plus factor becomes a single fraction.
    Expr::FunctionCall { name, args } if name == "Times" => {
      let new_args: Vec<Expr> = args.iter().map(together_expr).collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }

    // Other function calls — do not descend (e.g. Sin[...], f[x]); leave as-is.
    _ => expr.clone(),
  }
}

/// Build a Divide expression, collapsing trivial cases.
fn make_fraction(num: Expr, den: Expr) -> Expr {
  if matches!(&den, Expr::Integer(1)) {
    num
  } else if matches!(&num, Expr::Integer(0)) {
    Expr::Integer(0)
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(num),
      right: Box::new(den),
    }
  }
}

/// wolframscript's Together pulls the numeric content out of a polynomial
/// result (e.g. 2 + 2 x -> 2 (1 + x)), matching FactorTerms. Applied only to
/// the fully-cancelled polynomial output of a fraction.
fn factor_numeric_content(poly: &Expr) -> Expr {
  crate::evaluator::evaluate_function_call_ast("FactorTerms", &[poly.clone()])
    .unwrap_or_else(|_| poly.clone())
}

/// Count the leaf nodes (atoms) of an expression, used to bound work.
fn leaf_count(e: &Expr) -> usize {
  match e {
    Expr::BinaryOp { left, right, .. } => leaf_count(left) + leaf_count(right),
    Expr::UnaryOp { operand, .. } => leaf_count(operand),
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      1 + args.iter().map(leaf_count).sum::<usize>()
    }
    _ => 1,
  }
}

/// If the fraction `num/den` reduces to a polynomial (denominator divides the
/// numerator exactly), return that polynomial with its numeric content pulled
/// out (matching wolframscript's Together); otherwise None.
///
/// Guarded to small, few-variable fractions so the polynomial-GCD cancellation
/// never runs on the large multivariate intermediates that Solve /
/// InverseFunction route through Together (which would blow up).
fn try_reduce_to_polynomial(num: &Expr, den: &Expr) -> Option<Expr> {
  let frac = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(num.clone()),
    right: Box::new(den.clone()),
  };
  if leaf_count(&frac) > 40 {
    return None;
  }
  let mut vars = std::collections::HashSet::new();
  super::simplify::collect_variables(&frac, &mut vars);
  // The imaginary unit is not a polynomial variable.
  vars.remove("I");
  // A purely numeric fraction (e.g. Complex[1, 1]/3 from Together on a
  // complex-rational scalar) has nothing polynomial to reduce — and
  // running it through Cancel/Factor recurses forever: Factor calls
  // Together, which rebuilds the same fraction and calls back into Cancel.
  if vars.is_empty() {
    return None;
  }
  if vars.len() > 2 {
    return None;
  }
  let cancelled = cancel_expr_keep_quotient_sign(&frac);
  if matches!(extract_num_den(&cancelled).1, Expr::Integer(1)) {
    Some(factor_numeric_content(&cancelled))
  } else {
    None
  }
}

/// Whether folding a single fraction can actually cancel something: a
/// numerator factor and a denominator factor share an integer content or
/// an identical symbolic base. wolframscript folds x*(x+y)/(x*y) → (x+y)/y
/// and cancels the 2 in (2*x)/((2-2*x)*(2+x)), but keeps
/// (5*x)/((1-x)*(2+x)) and (5*x*(1+x))/((1-x)*(2+x)) factored
/// (differential fuzzer, seed 8887; all wolframscript-verified).
fn shares_cancelable_factor(num: &Expr, den: &Expr) -> bool {
  fn strip_power(f: &Expr) -> Expr {
    match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if matches!(right.as_ref(), Expr::Integer(_)) => (**left).clone(),
      Expr::FunctionCall { name, args }
        if name == "Power"
          && args.len() == 2
          && matches!(&args[1], Expr::Integer(_)) =>
      {
        args[0].clone()
      }
      other => other.clone(),
    }
  }
  // (integer content, symbolic factor bases) of a product.
  let side = |e: &Expr| -> (i128, Vec<String>) {
    let mut content: i128 = 1;
    let mut bases = Vec::new();
    for f in flatten_times_args(std::slice::from_ref(e)) {
      let b = strip_power(&f);
      match &b {
        Expr::Integer(n) => {
          content = content.saturating_mul(n.abs().max(1));
        }
        _ => {
          let terms = collect_additive_terms(&b);
          if terms.len() >= 2
            && let Some((n, _, _)) = super::factor::rational_content(&terms)
          {
            content = content.saturating_mul(n.abs().max(1));
          }
          bases.push(expr_to_string(&b));
        }
      }
    }
    (content, bases)
  };
  let (num_content, num_bases) = side(num);
  let (den_content, den_bases) = side(den);
  if super::factor::gcd_i128(num_content, den_content).abs() > 1 {
    return true;
  }
  num_bases.iter().any(|b| den_bases.contains(b))
}

pub fn together_expr(expr: &Expr) -> Expr {
  // First recursively apply Together to sub-expressions so that nested
  // fractions (e.g. `1/(1 + 1/x)` or continued-fraction-like forms) get
  // combined bottom-up. `together_expr_preprocess` leaves polynomial-shaped
  // expressions alone but pushes Together into Power bases, Divide operands,
  // and Times factors.
  let expr_rec = together_expr_preprocess(expr);

  // Collect additive terms and put them over a common denominator
  let terms = collect_additive_terms(&expr_rec);
  if terms.len() <= 1 {
    // A single additive term with no denominator has nothing to combine —
    // return the (preprocessed) form. A single term that carries a
    // *variable* denominator must still flow through the num/den logic below:
    // distributing a scalar into a combined fraction yields a single term
    // like `3 * (2 a)/((a-I x)(a+I x))` whose numerator needs folding to
    // `6 a`, and `x * (x+y)/(x y)` needs its common `x` cancelled.
    //
    // A purely *constant* denominator (e.g. `(-4 (-3+2 x))/13`) is left
    // alone: re-combining would expand the numerator, but wolframscript keeps
    // the common numeric content factored out (`-4 (-3+2 x)`), so don't touch
    // those.
    //
    // Only a genuine *product involving a combined fraction* is folded —
    // e.g. a scalar distributed onto a fraction (`3 * (2 a)/denom`,
    // `1/2 * (2 a)/denom`) or `x * (x+y)/(x y)`. These carry a `Divide` node
    // (produced when together_expr combined the inner sum) alongside at least
    // one other factor, or have two distinct numerator-contributing factors.
    //
    // A bare single fraction is left alone. In particular `(-m+x) * (1/s) *
    // (1/Sqrt[2])` is structurally a Times but is already a single fraction
    // (only inverse Power factors, one numerator factor); re-combining it
    // would only reorder its denominator (`Sqrt[2]*s` → `s*Sqrt[2]`). And a
    // purely constant denominator (`(-4 (-3+2 x))/13`) is left factored.
    let is_foldable_product =
      terms
        .first()
        .map(|t| {
          let den = extract_num_den(t).1;
          if matches!(den, Expr::Integer(1)) {
            return false;
          }
          let mut vars = std::collections::HashSet::new();
          super::simplify::collect_variables(&den, &mut vars);
          if vars.is_empty() {
            return false;
          }
          let factors = flatten_times_args(std::slice::from_ref(t));
          if factors.len() < 2 {
            return false;
          }
          let has_divide_node = factors.iter().any(|f| {
          matches!(f, Expr::BinaryOp { op: BinaryOperator::Divide, .. })
            || matches!(f, Expr::FunctionCall { name, .. } if name == "Divide")
        });
          let num_contributors = factors
            .iter()
            .filter(|f| !matches!(extract_num_den(f).0, Expr::Integer(1)))
            .count();
          // Multiple numerator factors alone don't justify combining:
          // wolframscript keeps (5*x*(1+x))/((1-x)*(2+x)) factored, so
          // only fold when something actually cancels.
          has_divide_node
            || (num_contributors >= 2 && {
              let (n, d) = extract_num_den(t);
              shares_cancelable_factor(&n, &d)
            })
        })
        .unwrap_or(false);
    if !is_foldable_product {
      // A lone fraction is still GCD-cancelled when it reduces to a polynomial
      // (Together[(x^2-1)/(x-1)] -> 1 + x). Only commit when a denominator was
      // present and fully divided out; otherwise keep the form unchanged so
      // factored/constant denominators are preserved.
      let (en, ed) = extract_num_den(&expr_rec);
      if !matches!(&ed, Expr::Integer(1))
        && let Some(reduced) = try_reduce_to_polynomial(&en, &ed)
      {
        return reduced;
      }
      // A lone fraction over a bare polynomial denominator may still share a
      // polynomial factor (e.g. (x^2+x)/(x^2-1) -> x/(x-1)); cancel the GCD.
      if is_plus_polynomial(&ed) && single_variable_fraction(&en, &ed) {
        let cancelled =
          super::cancel::cancel_expr_keep_quotient_sign(&expr_rec);
        if expr_to_string(&cancelled) != expr_to_string(&expr_rec) {
          return cancelled;
        }
      }
      // A nested purely numeric denominator folds to a single number:
      // wolframscript shows Together[(a + b/6)/2] as (6*a + b)/12,
      // never (6*a + b)/(2*6).
      if !matches!(&ed, Expr::Integer(1)) {
        let mut den_vars = std::collections::HashSet::new();
        super::simplify::collect_variables(&ed, &mut den_vars);
        den_vars.remove("I");
        if den_vars.is_empty()
          && let Ok(folded) = crate::evaluator::evaluate_expr_to_expr(&ed)
          && expr_to_string(&folded) != expr_to_string(&ed)
        {
          let mut num_factors: Vec<Expr> =
            flatten_times_args(std::slice::from_ref(&en))
              .into_iter()
              .filter(|f| !matches!(f, Expr::Integer(1)))
              .collect();
          let en = match num_factors.len() {
            0 => Expr::Integer(1),
            1 => num_factors.remove(0),
            _ => build_product(num_factors),
          };
          if matches!(&folded, Expr::Integer(1)) {
            return en;
          }
          return Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(en),
            right: Box::new(folded),
          };
        }
      }
      return expr_rec;
    }
  }

  // Extract numerator and denominator for each term
  let mut fractions: Vec<(Expr, Expr)> = Vec::new();
  for term in &terms {
    fractions.push(extract_num_den(term));
  }

  // Compute the common denominator (LCM of all denominators)
  // Decompose each denominator into base^exp pairs and take max exp for each base
  let mut base_exp_map: Vec<(String, Expr, RatExp)> = Vec::new();
  let mut fractional_int_exponent = false;
  for (_, den) in &fractions {
    if matches!(den, Expr::Integer(1)) {
      continue;
    }
    let den_factors = extract_den_factors(den);
    for (base, exp) in &den_factors {
      if matches!(base, Expr::Integer(_)) && exp.1 != 1 {
        fractional_int_exponent = true;
      }
      let key = expr_to_string(base);
      if let Some(entry) = base_exp_map.iter_mut().find(|(k, _, _)| *k == key) {
        entry.2 = rat_max(entry.2, *exp); // Take max exponent (LCM)
      } else {
        base_exp_map.push((key, base.clone(), *exp));
      }
    }
  }

  // Integer denominators combine by their integer LCM, not by product:
  // Together[a/2 + b/6] is (3*a + b)/6, never (2*(3*a + b))/12. Skipped
  // entirely when any integer base occurred with a fractional exponent
  // (a rationalized radical like 3/2^(1/2) alongside 3/2): the merged
  // entry would bypass the exponent arithmetic that produces the
  // fractional missing factor 2^(1/2).
  if !fractional_int_exponent {
    let mut int_lcm: i128 = 1;
    let mut merged = 0usize;
    base_exp_map.retain(|(_, base, exp)| {
      if let Expr::Integer(n) = base
        && *n > 1
        && exp.1 == 1
        && exp.0 > 0
        && let Ok(e) = u32::try_from(exp.0)
        && let Some(p) = n.checked_pow(e)
        && let Some(l) =
          int_lcm.checked_mul(p / super::factor::gcd_i128(int_lcm, p).abs())
      {
        int_lcm = l;
        merged += 1;
        return false;
      }
      true
    });
    if merged > 0 && int_lcm > 1 {
      base_exp_map
        .insert(0, (int_lcm.to_string(), Expr::Integer(int_lcm), (1, 1)));
    }
  }

  if base_exp_map.is_empty() {
    // No fractions to combine — but wolframscript still pulls the numeric
    // content out of a plain sum: Together[2 - 4x - 4x^2] →
    // -2*(-1 + 2x + 2x^2), Together[2 Sin[x] + 4 Sin[y]] →
    // 2*(Sin[x] + 2*Sin[y]).
    return super::factor::factor_terms_numeric(&expr_rec, &terms)
      .unwrap_or(expr_rec);
  }

  // Build numerator: for each term, multiply by (common_den / den_i)
  let mut new_num_terms = Vec::new();
  for (num, den) in &fractions {
    let missing_factor = compute_missing_factor(den, &base_exp_map);
    new_num_terms.push(multiply_exprs(num, &missing_factor));
  }

  let combined_num = if new_num_terms.len() == 1 {
    expand_and_combine(&new_num_terms.remove(0))
  } else {
    expand_and_combine(&build_sum(new_num_terms))
  };
  // Keep denominator in factored form (Wolfram behavior). For exponent 1
  // a base like `(y+1)` stays as `1 + y`; for exponent ≥ 2 we keep the
  // power literal `(1 + y)^2` rather than expanding, so e.g.
  // `Together[x/(y+1) + x/(y+1)^2]` lands on `(x*(2+y))/(1+y)^2`.
  let combined_den = {
    let mut canonical_dens: Vec<Expr> = base_exp_map
      .iter()
      .map(|(_, base, exp)| {
        if *exp == (1, 1) {
          expand_and_combine(base)
        } else {
          let canonical_base = expand_and_combine(base);
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(canonical_base),
            right: Box::new(rat_exp_to_expr(*exp)),
          }
        }
      })
      .collect();
    // Fold integer factors together so the numeric part of the common
    // denominator is a single number: Together[x/2 + y/3] → (3x + 2y)/6,
    // not (3x + 2y)/(2*3), and Together[x/2 + y/(3 z)] → (2y + 3xz)/(6z).
    // Accumulate in a BigInt: pathological intermediates (e.g. Simplify
    // iterating on complex-radical quotients) can carry integer factors
    // whose product overflows i128, which previously panicked here.
    let mut int_prod = num_bigint::BigInt::from(1);
    let mut split_dens: Vec<Expr> = Vec::new();
    for f in canonical_dens.drain(..) {
      match f {
        Expr::Integer(n) => int_prod *= n,
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          ..
        }
        | Expr::FunctionCall { .. }
          if matches!(&f, Expr::FunctionCall { name, .. } if name == "Times")
            || matches!(
              &f,
              Expr::BinaryOp {
                op: BinaryOperator::Times,
                ..
              }
            ) =>
        {
          // Pull integer factors out of a composite Times base.
          let mut rest: Vec<Expr> = Vec::new();
          for sub in flatten_times_args(std::slice::from_ref(&f)) {
            if let Expr::Integer(n) = sub {
              int_prod *= n;
            } else {
              rest.push(sub);
            }
          }
          match rest.len() {
            0 => {}
            1 => split_dens.push(rest.remove(0)),
            _ => split_dens.push(build_product(rest)),
          }
        }
        other => split_dens.push(other),
      }
    }
    canonical_dens = split_dens;
    if int_prod != num_bigint::BigInt::from(1) {
      let folded = match i128::try_from(&int_prod) {
        Ok(n) => Expr::Integer(n),
        Err(_) => Expr::BigInteger(int_prod),
      };
      canonical_dens.insert(0, folded);
    }
    crate::functions::math_ast::sort_symbolic_factors(&mut canonical_dens);
    if canonical_dens.len() == 1 {
      canonical_dens.remove(0)
    } else {
      build_product(canonical_dens)
    }
  };

  if matches!(&combined_num, Expr::Integer(0)) {
    Expr::Integer(0)
  } else if matches!(&combined_den, Expr::Integer(1)) {
    combined_num
  } else {
    // wolframscript's Together cancels the numerator/denominator GCD. When the
    // fraction reduces to a polynomial (the denominator divides the numerator
    // exactly), return that polynomial, e.g. Together[(x^2-1)/(x-1)] -> 1 + x.
    // If a denominator survives, keep the factored form below rather than the
    // expanded denominator that Cancel produces.
    if let Some(reduced) =
      try_reduce_to_polynomial(&combined_num, &combined_den)
    {
      return reduced;
    }
    // Try to cancel common monomial factors between numerator and denominator
    // without re-expanding the denominator (preserves factored form).
    let (mut simplified_num, simplified_den) =
      cancel_common_monomial_factors(&combined_num, &combined_den);
    // When the denominator carries a non-trivial Power factor (e.g.
    // `(y+1)^2`), wolframscript additionally factors any common monomial
    // out of the numerator — `Together[x/(y+1) + x/(y+1)^2]` lands on
    // `(x*(2+y))/(1+y)^2` rather than `(2*x + x*y)/(1+y)^2`. Apply that
    // same hoist; for purely linear denominators the numerator stays
    // expanded (matching wolframscript).
    if denominator_has_power_factor(&simplified_den) {
      simplified_num = factor_common_monomial_from_terms(&simplified_num);
    }
    // A single (unfactored) polynomial denominator may still share a polynomial
    // factor with the numerator — e.g. (x^2+x)/(x^2-1) reduces to x/(x-1).
    // Monomial cancellation above misses this, so fall back to the full GCD
    // cancellation (Cancel) when the denominator is a bare Plus polynomial.
    // Factored (Times/Power) denominators are left to the logic above so their
    // form is preserved, matching wolframscript.
    if is_plus_polynomial(&simplified_den)
      && single_variable_fraction(&simplified_num, &simplified_den)
    {
      let frac = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(simplified_num.clone()),
        right: Box::new(simplified_den.clone()),
      };
      let cancelled = super::cancel::cancel_expr_keep_quotient_sign(&frac);
      if expr_to_string(&cancelled) != expr_to_string(&frac) {
        return cancelled;
      }
    }
    if matches!(&simplified_den, Expr::Integer(1)) {
      simplified_num
    } else {
      // Over a pure integer denominator wolframscript factors the numeric
      // content out of the numerator: Together[3/2 - 3x/2] → (-3*(-1+x))/2,
      // Together[2/3 + 4x/3] → (2*(1 + 2*x))/3.
      if matches!(&simplified_den, Expr::Integer(_)) {
        let num_terms = collect_additive_terms(&simplified_num);
        if num_terms.len() > 1
          && let Ok(factored) =
            super::factor::factor_terms_numeric(&simplified_num, &num_terms)
        {
          simplified_num = factored;
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(simplified_num),
        right: Box::new(simplified_den),
      }
    }
  }
}

/// True when `num`/`den` together involve exactly one symbolic variable, the
/// only case the univariate GCD cancellation in `cancel_expr` can reduce.
/// Restricting to this keeps the fallback cheap (multivariate / Slot-bearing
/// rationals — e.g. a Möbius inverse `(-b+#1 d)/(a-#1 c)` — are skipped).
fn single_variable_fraction(num: &Expr, den: &Expr) -> bool {
  let mut vars = std::collections::HashSet::new();
  super::simplify::collect_variables(num, &mut vars);
  super::simplify::collect_variables(den, &mut vars);
  vars.len() == 1
}

/// Returns `true` when `expr` is a bare additive polynomial (a Plus / Minus),
/// as opposed to a factored product or power denominator.
fn is_plus_polynomial(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, .. } if name == "Plus")
    || matches!(
      expr,
      Expr::BinaryOp {
        op: BinaryOperator::Plus | BinaryOperator::Minus,
        ..
      }
    )
}

/// Returns `true` when `den` is a Power[base, n] with n ≥ 2 or a Times/Divide
/// containing such a factor anywhere.
fn denominator_has_power_factor(den: &Expr) -> bool {
  match den {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      right,
      ..
    } => matches!(right.as_ref(), Expr::Integer(n) if *n >= 2),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      matches!(&args[1], Expr::Integer(n) if *n >= 2)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().any(denominator_has_power_factor)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      denominator_has_power_factor(left) || denominator_has_power_factor(right)
    }
    _ => false,
  }
}

/// Factor the largest common monomial out of a sum-of-products numerator.
/// E.g. `x*y + 2*x` → `x*(2 + y)`. Returns the input unchanged when no
/// non-trivial common factor exists. The cofactor is left expanded (matches
/// wolframscript's surface form for the rare cases this is invoked).
fn factor_common_monomial_from_terms(num: &Expr) -> Expr {
  let terms = collect_additive_terms(num);
  if terms.len() < 2 {
    return num.clone();
  }

  fn term_factors(term: &Expr) -> Vec<(String, Expr, i128)> {
    let factors = flatten_times_args(&[term.clone()]);
    let mut map: Vec<(String, Expr, i128)> = Vec::new();
    for f in &factors {
      let (key, base, exp) = match f {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } => match right.as_ref() {
          Expr::Integer(n) if *n > 0 => {
            (expr_to_string(left), *left.clone(), *n)
          }
          _ => continue,
        },
        Expr::FunctionCall {
          name: pname,
          args: pargs,
        } if pname == "Power" && pargs.len() == 2 => match &pargs[1] {
          Expr::Integer(n) if *n > 0 => {
            (expr_to_string(&pargs[0]), pargs[0].clone(), *n)
          }
          _ => continue,
        },
        Expr::Integer(_) => continue,
        _ => (expr_to_string(f), f.clone(), 1),
      };
      if let Some(entry) = map.iter_mut().find(|(k, _, _)| *k == key) {
        entry.2 += exp;
      } else {
        map.push((key, base, exp));
      }
    }
    map
  }

  let mut common = term_factors(&terms[0]);
  for term in &terms[1..] {
    let tmap = term_factors(term);
    common.retain_mut(|(key, _, exp)| {
      if let Some(entry) = tmap.iter().find(|(k, _, _)| k == key) {
        *exp = (*exp).min(entry.2);
        *exp > 0
      } else {
        false
      }
    });
    if common.is_empty() {
      break;
    }
  }
  if common.is_empty() {
    return num.clone();
  }

  // Build cofactor: divide each term by the common monomial.
  let mut new_terms: Vec<Expr> = Vec::new();
  for term in &terms {
    let mut t_factors: Vec<Expr> = flatten_times_args(&[term.clone()]);
    for (cancel_key, cancel_exp) in common
      .iter()
      .map(|(k, _, e)| (k.clone(), *e))
      .collect::<Vec<_>>()
    {
      let mut remaining = cancel_exp;
      let mut new_factors = Vec::new();
      for f in t_factors {
        if remaining <= 0 {
          new_factors.push(f);
          continue;
        }
        let (key, base, exp) = match &f {
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } => match right.as_ref() {
            Expr::Integer(n) if *n > 0 => {
              (expr_to_string(left), *left.clone(), *n)
            }
            _ => {
              new_factors.push(f);
              continue;
            }
          },
          Expr::FunctionCall {
            name: pname,
            args: pargs,
          } if pname == "Power" && pargs.len() == 2 => match &pargs[1] {
            Expr::Integer(n) if *n > 0 => {
              (expr_to_string(&pargs[0]), pargs[0].clone(), *n)
            }
            _ => {
              new_factors.push(f);
              continue;
            }
          },
          Expr::Integer(_) => {
            new_factors.push(f);
            continue;
          }
          _ => (expr_to_string(&f), f.clone(), 1),
        };
        if key == cancel_key {
          let reduce = remaining.min(exp);
          remaining -= reduce;
          let new_exp = exp - reduce;
          if new_exp > 1 {
            new_factors.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(base),
              right: Box::new(Expr::Integer(new_exp)),
            });
          } else if new_exp == 1 {
            new_factors.push(base);
          }
        } else {
          new_factors.push(f);
        }
      }
      t_factors = new_factors;
    }
    new_terms.push(if t_factors.is_empty() {
      Expr::Integer(1)
    } else {
      build_product(t_factors)
    });
  }

  // Build the common monomial expression.
  let mut common_factors: Vec<Expr> = common
    .into_iter()
    .map(|(_, base, exp)| {
      if exp == 1 {
        base
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(base),
          right: Box::new(Expr::Integer(exp)),
        }
      }
    })
    .collect();
  let cofactor = if new_terms.len() == 1 {
    new_terms.into_iter().next().unwrap()
  } else {
    build_sum(new_terms)
  };
  common_factors.push(cofactor);
  build_product(common_factors)
}

/// A rational exponent represented as (numerator, denominator) with denominator > 0.
type RatExp = (i128, i128);

fn rat_exp_from_expr(exp: &Expr) -> RatExp {
  match exp {
    Expr::Integer(n) => (*n, 1),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        (*n, *d)
      } else {
        (1, 1)
      }
    }
    _ => (1, 1),
  }
}

fn rat_exp_to_expr((n, d): RatExp) -> Expr {
  if d == 1 {
    Expr::Integer(n)
  } else {
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
    }
  }
}

/// Compare two rational exponents: returns a > b
fn rat_gt((an, ad): RatExp, (bn, bd): RatExp) -> bool {
  an * bd > bn * ad
}

/// Compute max of two rational exponents
fn rat_max(a: RatExp, b: RatExp) -> RatExp {
  if rat_gt(a, b) { a } else { b }
}

/// Subtract two rational exponents: a - b
fn rat_sub((an, ad): RatExp, (bn, bd): RatExp) -> RatExp {
  let n = an * bd - bn * ad;
  let d = ad * bd;
  let g = crate::functions::math_ast::gcd(n, d);
  (n / g, d / g)
}

/// Extract base and exponent from a denominator expression.
/// Returns (base, exponent) pairs with rational exponents.
fn extract_den_factors(den: &Expr) -> Vec<(Expr, RatExp)> {
  match den {
    Expr::Integer(1) => vec![],
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      vec![(*left.clone(), rat_exp_from_expr(right))]
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      vec![(args[0].clone(), rat_exp_from_expr(&args[1]))]
    }
    // Times[a, b, ...] in denominator — split into factors
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      let mut result = Vec::new();
      for arg in args {
        result.extend(extract_den_factors(arg));
      }
      result
    }
    _ => vec![(den.clone(), (1, 1))],
  }
}

/// Compute the "missing factor" needed to bring a fraction's denominator up to the common
/// denominator. For each base in the LCM, compute base^(lcm_exp - den_exp).
fn compute_missing_factor(
  den: &Expr,
  base_exp_map: &[(String, Expr, RatExp)],
) -> Expr {
  let den_factors = extract_den_factors(den);
  let mut den_map: Vec<(String, RatExp)> = Vec::new();
  for (base, exp) in &den_factors {
    let key = expr_to_string(base);
    if let Some(entry) = den_map.iter_mut().find(|(k, _)| *k == key) {
      // Add exponents for same base
      entry.1 = (entry.1.0 * exp.1 + exp.0 * entry.1.1, entry.1.1 * exp.1);
      let g = crate::functions::math_ast::gcd(entry.1.0, entry.1.1);
      entry.1 = (entry.1.0 / g, entry.1.1 / g);
    } else {
      den_map.push((key, *exp));
    }
  }

  // This fraction's integer denominator content, for dividing out of the
  // map's merged integer-LCM entry (a/2 against an LCM of 6 needs 3).
  let den_int: i128 =
    den_factors
      .iter()
      .fold(1i128, |acc, (base, exp)| match (base, exp) {
        (Expr::Integer(n), (e, 1)) if *n > 1 && *e > 0 => u32::try_from(*e)
          .ok()
          .and_then(|e| n.checked_pow(e))
          .and_then(|p| acc.checked_mul(p))
          .unwrap_or(0),
        _ => acc,
      });
  // A fractional exponent on an integer base (2^(1/2) from a rationalized
  // radical) needs the generic exponent arithmetic — the fast integer
  // branch would produce a whole missing factor 2 instead of 2^(1/2).
  let den_has_frac_int = den_factors
    .iter()
    .any(|(base, exp)| matches!(base, Expr::Integer(_)) && exp.1 != 1);

  let mut missing_factors: Vec<Expr> = Vec::new();
  for (key, base, lcm_exp) in base_exp_map {
    if let Expr::Integer(l) = base
      && *l > 1
      && *lcm_exp == (1, 1)
      && !den_has_frac_int
      && den_int > 0
      && l % den_int == 0
    {
      if l / den_int > 1 {
        missing_factors.push(Expr::Integer(l / den_int));
      }
      continue;
    }
    let den_exp = den_map
      .iter()
      .find(|(k, _)| k == key)
      .map(|(_, e)| *e)
      .unwrap_or((0, 1));
    let diff = rat_sub(*lcm_exp, den_exp);
    if rat_gt(diff, (0, 1)) {
      if diff == (1, 1) {
        missing_factors.push(base.clone());
      } else {
        missing_factors.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(base.clone()),
          right: Box::new(rat_exp_to_expr(diff)),
        });
      }
    }
  }

  if missing_factors.is_empty() {
    Expr::Integer(1)
  } else {
    build_product(missing_factors)
  }
}

/// Cancel common monomial factors between a numerator (sum of terms) and a denominator (product).
/// E.g. (a^2*x + a*x^2) / a^2 → (a*x + x^2) / a
/// Does not expand the denominator, preserving its factored form.
fn cancel_common_monomial_factors(num: &Expr, den: &Expr) -> (Expr, Expr) {
  let terms = collect_additive_terms(num);
  if terms.len() < 2 {
    return (num.clone(), den.clone());
  }

  // For each term, extract base→exp map of its multiplicative factors (ignoring integers)
  fn term_base_exp(term: &Expr) -> Vec<(String, Expr, i128)> {
    let factors = flatten_times_args(&[term.clone()]);
    let mut map: Vec<(String, Expr, i128)> = Vec::new();
    for f in &factors {
      let (base_str, base, exp) = match f {
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left,
          right,
        } => {
          if let Expr::Integer(n) = right.as_ref() {
            if *n > 0 {
              (expr_to_string(left), *left.clone(), *n)
            } else {
              continue;
            }
          } else {
            continue;
          }
        }
        Expr::FunctionCall { name, args }
          if name == "Power" && args.len() == 2 =>
        {
          if let Expr::Integer(n) = &args[1] {
            if *n > 0 {
              (expr_to_string(&args[0]), args[0].clone(), *n)
            } else {
              continue;
            }
          } else {
            continue;
          }
        }
        Expr::Integer(_) => continue,
        _ => (expr_to_string(f), f.clone(), 1),
      };
      if let Some(entry) = map.iter_mut().find(|(k, _, _)| *k == base_str) {
        entry.2 += exp;
      } else {
        map.push((base_str, base, exp));
      }
    }
    map
  }

  // Find common base^exp across all terms (min exponent for each base)
  let mut common = term_base_exp(&terms[0]);
  for term in &terms[1..] {
    let tmap = term_base_exp(term);
    common.retain_mut(|(key, _, exp)| {
      if let Some(entry) = tmap.iter().find(|(k, _, _)| k == key) {
        *exp = (*exp).min(entry.2);
        *exp > 0
      } else {
        false
      }
    });
  }

  if common.is_empty() {
    return (num.clone(), den.clone());
  }

  // Check which common factors also appear in the denominator and can be cancelled
  let den_factors = flatten_times_args(&[den.clone()]);
  let mut den_map: Vec<(String, Expr, i128)> = Vec::new();
  for f in &den_factors {
    let (base_str, base, exp) = match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => {
        if let Expr::Integer(n) = right.as_ref() {
          (expr_to_string(left), *left.clone(), *n)
        } else {
          (expr_to_string(f), f.clone(), 1)
        }
      }
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        if let Expr::Integer(n) = &args[1] {
          (expr_to_string(&args[0]), args[0].clone(), *n)
        } else {
          (expr_to_string(f), f.clone(), 1)
        }
      }
      Expr::Integer(_) => continue,
      _ => (expr_to_string(f), f.clone(), 1),
    };
    if let Some(entry) = den_map.iter_mut().find(|(k, _, _)| *k == base_str) {
      entry.2 += exp;
    } else {
      den_map.push((base_str, base, exp));
    }
  }

  // Determine how much of each common factor can be cancelled with the denominator
  let mut cancel_map: Vec<(String, i128)> = Vec::new();
  for (key, _, num_exp) in &common {
    if let Some((_, _, den_exp)) = den_map.iter().find(|(k, _, _)| k == key) {
      let cancel_exp = (*num_exp).min(*den_exp);
      if cancel_exp > 0 {
        cancel_map.push((key.clone(), cancel_exp));
      }
    }
  }

  if cancel_map.is_empty() {
    return (num.clone(), den.clone());
  }

  // Divide each numerator term by the cancelled factors
  let mut new_terms = Vec::new();
  for term in &terms {
    let mut t_factors: Vec<Expr> = flatten_times_args(&[term.clone()]);
    for (cancel_key, cancel_exp) in &cancel_map {
      let mut remaining = *cancel_exp;
      let mut new_factors = Vec::new();
      for f in t_factors {
        if remaining <= 0 {
          new_factors.push(f);
          continue;
        }
        let (base_str, base, exp) = match &f {
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } => {
            if let Expr::Integer(n) = right.as_ref() {
              (expr_to_string(left), *left.clone(), *n)
            } else {
              new_factors.push(f);
              continue;
            }
          }
          Expr::FunctionCall { name, args }
            if name == "Power" && args.len() == 2 =>
          {
            if let Expr::Integer(n) = &args[1] {
              (expr_to_string(&args[0]), args[0].clone(), *n)
            } else {
              new_factors.push(f);
              continue;
            }
          }
          Expr::Integer(_) => {
            new_factors.push(f);
            continue;
          }
          _ => (expr_to_string(&f), f.clone(), 1),
        };
        if base_str == *cancel_key {
          let reduce = remaining.min(exp);
          remaining -= reduce;
          let new_exp = exp - reduce;
          if new_exp > 1 {
            new_factors.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(base),
              right: Box::new(Expr::Integer(new_exp)),
            });
          } else if new_exp == 1 {
            new_factors.push(base);
          }
          // new_exp == 0: factor removed
        } else {
          new_factors.push(f);
        }
      }
      t_factors = new_factors;
    }
    if t_factors.is_empty() {
      new_terms.push(Expr::Integer(1));
    } else {
      new_terms.push(build_product(t_factors));
    }
  }

  // Build new numerator
  let new_num = if new_terms.len() == 1 {
    expand_and_combine(&new_terms[0])
  } else {
    expand_and_combine(&build_sum(new_terms))
  };

  // Build new denominator: reduce exponents of cancelled factors
  let mut new_den_factors: Vec<Expr> = Vec::new();
  // Keep integer factors
  for f in &den_factors {
    if let Expr::Integer(n) = f {
      new_den_factors.push(Expr::Integer(*n));
    }
  }
  let mut den_map_remaining = den_map.clone();
  for (cancel_key, cancel_exp) in &cancel_map {
    if let Some(entry) = den_map_remaining
      .iter_mut()
      .find(|(k, _, _)| k == cancel_key)
    {
      entry.2 -= cancel_exp;
    }
  }
  for (_, base, exp) in &den_map_remaining {
    if *exp > 1 {
      new_den_factors.push(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base.clone()),
        right: Box::new(Expr::Integer(*exp)),
      });
    } else if *exp == 1 {
      new_den_factors.push(base.clone());
    }
  }
  // Sort non-numeric factors to match Wolfram canonical order
  let numeric_end = new_den_factors
    .iter()
    .position(|f| !matches!(f, Expr::Integer(_)))
    .unwrap_or(new_den_factors.len());
  crate::functions::math_ast::sort_symbolic_factors(
    &mut new_den_factors[numeric_end..],
  );
  let new_den = if new_den_factors.is_empty() {
    Expr::Integer(1)
  } else {
    build_product(new_den_factors)
  };

  (new_num, new_den)
}
