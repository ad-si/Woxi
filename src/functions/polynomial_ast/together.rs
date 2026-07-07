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
    let results: Vec<Expr> = items.iter().map(together_expr).collect();
    return Ok(Expr::List(results.into()));
  }
  Ok(together_expr(&args[0]))
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
  if vars.len() > 2 {
    return None;
  }
  let cancelled = cancel_expr(&frac);
  if matches!(extract_num_den(&cancelled).1, Expr::Integer(1)) {
    Some(factor_numeric_content(&cancelled))
  } else {
    None
  }
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
          has_divide_node || num_contributors >= 2
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
        let cancelled = super::cancel::cancel_expr(&expr_rec);
        if expr_to_string(&cancelled) != expr_to_string(&expr_rec) {
          return cancelled;
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
  for (_, den) in &fractions {
    if matches!(den, Expr::Integer(1)) {
      continue;
    }
    let den_factors = extract_den_factors(den);
    for (base, exp) in &den_factors {
      let key = expr_to_string(base);
      if let Some(entry) = base_exp_map.iter_mut().find(|(k, _, _)| *k == key) {
        entry.2 = rat_max(entry.2, *exp); // Take max exponent (LCM)
      } else {
        base_exp_map.push((key, base.clone(), *exp));
      }
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
    let mut int_prod: i128 = 1;
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
    if int_prod != 1 {
      canonical_dens.insert(0, Expr::Integer(int_prod));
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
      let cancelled = super::cancel::cancel_expr(&frac);
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

  let mut missing_factors: Vec<Expr> = Vec::new();
  for (key, base, lcm_exp) in base_exp_map {
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
