#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::functions::math_ast::{gcd_i128, rat_reduce};
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, unevaluated};

// ─── Apart ──────────────────────────────────────────────────────────

/// Flatten a (possibly nested) `Plus` chain into its individual summands,
/// leaving every other expression (including `UnaryOp::Minus` terms) intact.
fn flatten_plus(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      flatten_plus(left, out);
      flatten_plus(right, out);
    }
    other => out.push(other.clone()),
  }
}

/// Apart[expr] - Partial fraction decomposition
pub fn apart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Apart expects 1 or 2 arguments".into(),
    ));
  }

  // Thread over lists
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        if args.len() == 2 {
          apart_ast(&[item.clone(), args[1].clone()])
        } else {
          apart_ast(&[item.clone()])
        }
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  // Short-circuit: if the expression has no denominator, Apart is a no-op.
  let (_num, den_check) = super::together::extract_num_den(&args[0]);
  if matches!(&den_check, Expr::Integer(1)) {
    return Ok(args[0].clone());
  }

  let var = if args.len() == 2 {
    match &args[1] {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Ok(unevaluated("Apart", args));
      }
    }
  } else {
    // Find the variable automatically
    match find_single_variable(&args[0]) {
      Some(v) => v,
      None => {
        // A variable-free (numeric) argument is already apart:
        // Apart[Divide[1, 2]] → 1/2, not the unevaluated call.
        if crate::functions::predicate_ast::is_numeric_q(&args[0]) {
          return Ok(args[0].clone());
        }
        return Ok(unevaluated("Apart", args));
      }
    }
  };

  apart_expr(&args[0], &var)
}

fn apart_expr(expr: &Expr, var: &str) -> Result<Expr, InterpreterError> {
  // Apart's value is an ordinary expression: run the assembled sum
  // through the evaluator so each term takes its canonical form
  // (Apart[1/(-3 x)] prints -1/3*1/x, exactly like evaluating the
  // input directly — the hand-built Divide tree printed -1/(3*x)).
  let raw = apart_expr_raw(expr, var)?;
  crate::evaluator::evaluate_expr_to_expr(&raw)
}

fn apart_expr_raw(expr: &Expr, var: &str) -> Result<Expr, InterpreterError> {
  // Extract numerator and denominator using the general-purpose extractor
  // which handles BinaryOp::Divide, Times[..., Power[..., -1]], etc.
  let (num, den) = super::together::extract_num_den(expr);
  if matches!(&den, Expr::Integer(1)) {
    // Not a fraction — return as-is
    return Ok(expr.clone());
  }

  // Rebuild the expression in Divide form for downstream functions
  let divide_expr = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(num.clone()),
    right: Box::new(den.clone()),
  };

  let num_expanded = expand_and_combine(&num);
  let den_expanded = expand_and_combine(&den);

  // Try integer-coefficient approach first
  let num_coeffs = extract_poly_coeffs(&num_expanded, var);
  let den_coeffs = extract_poly_coeffs(&den_expanded, var);

  if let (Some(nc), Some(dc)) = (&num_coeffs, &den_coeffs) {
    // A constant denominator splits termwise: Apart[(4+2x)/5] →
    // 4/5 + (2*x)/5 (wolframscript).
    if dc.len() == 1 && dc[0] != 0 {
      let d = dc[0];
      let mut parts: Vec<Expr> = Vec::new();
      for (i, &c) in nc.iter().enumerate() {
        if c == 0 {
          continue;
        }
        let mut term_coeffs = vec![0i128; i + 1];
        term_coeffs[i] = c;
        let term = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(coeffs_to_expr(&term_coeffs, var)),
          right: Box::new(Expr::Integer(d)),
        };
        parts.push(crate::evaluator::evaluate_expr_to_expr(&term)?);
      }
      if parts.is_empty() {
        return Ok(Expr::Integer(0));
      }
      return Ok(build_sum(parts));
    }
    // If numerator degree >= denominator degree, do polynomial division first
    if nc.len() >= dc.len() {
      if let Some(quot_coeffs) = poly_exact_divide(nc, dc) {
        return Ok(coeffs_to_expr(&quot_coeffs, var));
      }
      // Rational-coefficient division, so a non-integer leading quotient
      // decomposes too: Apart[(2 - 4 x)/(2 - 5 x)] → 4/5 - 2/(5*(-2 + 5*x)).
      if let Some((quotient, remainder)) = poly_long_divide_rat(nc, dc) {
        if remainder.iter().all(|&(n, _)| n == 0) {
          return rat_coeffs_to_expr(&quotient, var);
        }
        let quot_expr = rat_coeffs_to_expr(&quotient, var)?;
        // Clear the remainder's coefficient denominators into the
        // fraction's denominator: r(x)/den = r_int(x)/(L*den), then let
        // the proper-fraction path hoist the content and normalize signs.
        let mut lcm: i128 = 1;
        for &(_, d) in &remainder {
          let g = gcd_i128(lcm, d);
          lcm = match (lcm / g).checked_mul(d) {
            Some(v) => v,
            None => {
              return apart_symbolic(&divide_expr, &num_expanded, &den, var);
            }
          };
        }
        let rem_int: Option<Vec<i128>> = remainder
          .iter()
          .map(|&(n, d)| n.checked_mul(lcm / d))
          .collect();
        let den_scaled: Option<Vec<i128>> =
          dc.iter().map(|&c| c.checked_mul(lcm)).collect();
        let (Some(rem_int), Some(den_scaled)) = (rem_int, den_scaled) else {
          return apart_symbolic(&divide_expr, &num_expanded, &den, var);
        };
        let frac = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(coeffs_to_expr(&rem_int, var)),
          right: Box::new(coeffs_to_expr(&den_scaled, var)),
        };
        let apart_remainder = apart_proper_fraction(&frac, var)?;
        // Splice the polynomial quotient in front of the partial-fraction
        // terms as one flat sum so the result reads `q + f1 + f2`, not the
        // parenthesized `q + (f1 + f2)`. A zero quotient is dropped rather
        // than emitting a spurious `0 + …` term.
        let mut parts = if matches!(quot_expr, Expr::Integer(0)) {
          Vec::new()
        } else {
          vec![quot_expr]
        };
        flatten_plus(&apart_remainder, &mut parts);
        return Ok(build_sum(parts));
      }
      // i128 overflow in the rational division: fall back to the
      // integer-only division (best effort, matching the old behavior).
      let (quotient, remainder) = poly_long_divide(nc, dc);
      if remainder.iter().all(|&c| c == 0) {
        return Ok(coeffs_to_expr(&quotient, var));
      }
      let quot_expr = coeffs_to_expr(&quotient, var);
      let rem_expr = coeffs_to_expr(&remainder, var);
      let frac = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(rem_expr),
        right: Box::new(den_expanded.clone()),
      };
      let apart_remainder = apart_proper_fraction(&frac, var)?;
      let mut parts = if matches!(quot_expr, Expr::Integer(0)) {
        Vec::new()
      } else {
        vec![quot_expr]
      };
      flatten_plus(&apart_remainder, &mut parts);
      return Ok(build_sum(parts));
    }

    return apart_proper_fraction(&divide_expr, var);
  }

  // Fall back to symbolic approach (multivariate case)
  apart_symbolic(&divide_expr, &num_expanded, &den, var)
}

/// Perform partial fraction decomposition for a proper fraction (deg(num) < deg(den))
fn apart_proper_fraction(
  expr: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  let (num, den) = match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    _ => return Ok(expr.clone()),
  };

  let den_expanded = expand_and_combine(&den);
  let den_coeffs = match extract_poly_coeffs(&den_expanded, var) {
    Some(c) => c,
    None => return Ok(expr.clone()),
  };

  // General decomposition when the denominator has an irreducible quadratic (or
  // higher-degree) factor — the residue formulas below only cover integer
  // linear roots. apart_general returns None for the all-linear case so the
  // existing (well-tested) linear path stays in effect there.
  {
    let num_expanded = expand_and_combine(&num);
    if let Some(nc) = extract_poly_coeffs(&num_expanded, var)
      && let Some(result) = apart_general(&nc, &den_coeffs, var)
    {
      return Ok(result);
    }
  }

  // Factor the denominator
  let gcd_coeff = den_coeffs
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_i128);
  if gcd_coeff == 0 {
    return Ok(expr.clone());
  }
  let reduced: Vec<i128> = den_coeffs.iter().map(|c| c / gcd_coeff).collect();
  let (sign, reduced) = if reduced.last().map(|&c| c < 0).unwrap_or(false) {
    (-1i128, reduced.iter().map(|c| -c).collect::<Vec<_>>())
  } else {
    (1, reduced)
  };
  let _overall = gcd_coeff * sign;

  // Find roots of denominator
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

  if roots.len() < 2 {
    // Can't decompose further — canonicalize the quotient display.
    let num_expanded = expand_and_combine(&num);
    if let Some(ncs) = extract_poly_coeffs(&num_expanded, var)
      && ncs.len() < den_coeffs.len()
      && let Some(norm) = normalize_irreducible_quotient(&ncs, &den_coeffs, var)
    {
      return Ok(norm);
    }
    return Ok(expr.clone());
  }

  // Sort roots descending so linear factors (-root + x) appear in ascending order
  roots.sort_by(|a, b| b.cmp(a));

  // Simple partial fraction for distinct linear factors:
  // N(x) / ((x-r1)(x-r2)...) = A1/(x-r1) + A2/(x-r2) + ...
  // where Ai = N(ri) / product of (ri - rj) for j != i
  let num_expanded = expand_and_combine(&num);
  let num_coeffs = match extract_poly_coeffs(&num_expanded, var) {
    Some(c) => c,
    None => return Ok(expr.clone()),
  };

  // If there's a remaining irreducible factor, we can't do simple partial fractions
  if !remaining.iter().all(|&c| c == 0) && remaining.len() > 1 {
    if num_coeffs.len() < den_coeffs.len()
      && let Some(norm) =
        normalize_irreducible_quotient(&num_coeffs, &den_coeffs, var)
    {
      return Ok(norm);
    }
    return Ok(expr.clone());
  }

  // If any root is repeated, the distinct-root residue formula below divides
  // by zero. Use the general linear-system decomposition, which handles
  // multiplicities (e.g. Apart[1/(x^2 (x + 1))]).
  let has_repeated = {
    let mut sorted = roots.clone();
    sorted.sort();
    sorted.windows(2).any(|w| w[0] == w[1])
  };
  if has_repeated {
    let overall_factor = gcd_coeff * sign;
    let rem_const = if remaining.len() == 1 && remaining[0] != 0 {
      remaining[0]
    } else {
      1
    };
    if let Some(result) =
      apart_repeated_roots(&num_coeffs, &roots, rem_const, overall_factor, var)
    {
      return Ok(result);
    }
    return Ok(expr.clone());
  }

  let mut result_terms = Vec::new();
  let overall_factor = gcd_coeff * sign;

  for (i, &root) in roots.iter().enumerate() {
    let num_at_root = evaluate_poly(&num_coeffs, root);
    let mut den_product = 1i128;
    for (j, &other_root) in roots.iter().enumerate() {
      if i != j {
        den_product *= root - other_root;
      }
    }
    // Handle the remaining factor
    if remaining.len() == 1 && remaining[0] != 0 {
      den_product *= remaining[0];
    }
    den_product *= overall_factor;

    if den_product == 0 {
      return Ok(expr.clone());
    }

    // A_i = num_at_root / den_product
    // Term = A_i / (x - root) which in canonical form is A_i / (-root + x)
    let (an, ad) = rat_reduce(num_at_root, den_product);

    // A zero residue contributes nothing — skip it rather than emitting a
    // spurious `0/(...)` term (e.g. Apart[(x + 1)/(x^2 + x)] is just 1/x).
    if an == 0 {
      continue;
    }

    // Build linear factor: when root is 0, just use the variable directly
    let linear_factor = if root == 0 {
      Expr::Identifier(var.to_string())
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-root)),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    };

    if ad == 1 && an.abs() == 1 {
      // Wolfram canonical form: (expr)^(-1), negated via UnaryOp so it
      // renders -(-1 + x)^(-1).
      let pow = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(linear_factor),
        right: Box::new(Expr::Integer(-1)),
      };
      result_terms.push(if an < 0 {
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(pow),
        }
      } else {
        pow
      });
    } else {
      // Signed numerator over the (possibly scaled) factor so negatives
      // render -2/(1 + x) / -1/(2*(-1 + x)), matching wolframscript.
      let denom = if ad == 1 {
        linear_factor
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(ad)),
          right: Box::new(linear_factor),
        }
      };
      result_terms.push(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(an)),
        right: Box::new(denom),
      });
    }
  }

  if result_terms.is_empty() {
    Ok(expr.clone())
  } else {
    Ok(build_sum(result_terms))
  }
}

/// Multiply two ascending-coefficient integer polynomials.
fn poly_mul_i128(a: &[i128], b: &[i128]) -> Vec<i128> {
  if a.is_empty() || b.is_empty() {
    return vec![];
  }
  let mut out = vec![0i128; a.len() + b.len() - 1];
  for (i, &ai) in a.iter().enumerate() {
    for (j, &bj) in b.iter().enumerate() {
      out[i + j] += ai * bj;
    }
  }
  out
}

/// Solve the square rational system `mat * x = rhs` by Gauss–Jordan
/// elimination. `mat` is row-major `n x n`. Returns None if singular.
fn solve_rat_system(
  mut mat: Vec<Vec<Rat>>,
  mut rhs: Vec<Rat>,
) -> Option<Vec<Rat>> {
  let n = rhs.len();
  for col in 0..n {
    let piv = (col..n).find(|&r| !mat[r][col].is_zero())?;
    mat.swap(col, piv);
    rhs.swap(col, piv);
    let pivot = mat[col][col];
    for r in 0..n {
      if r == col || mat[r][col].is_zero() {
        continue;
      }
      let factor = mat[r][col].div(pivot);
      for c in col..n {
        let v = mat[col][c].mul(factor);
        mat[r][c] = mat[r][c].sub(v);
      }
      let v = rhs[col].mul(factor);
      rhs[r] = rhs[r].sub(v);
    }
  }
  let mut sol = vec![Rat::int(0); n];
  for i in 0..n {
    if mat[i][i].is_zero() {
      return None;
    }
    sol[i] = rhs[i].div(mat[i][i]);
  }
  Some(sol)
}

/// gcd of the coefficients, signed by the highest-degree nonzero
/// coefficient (the FactorTerms sign convention). Returns 0 for the zero
/// polynomial.
fn signed_content(coeffs: &[i128]) -> i128 {
  let g = coeffs.iter().fold(0i128, |acc, &c| gcd_i128(acc, c.abs()));
  if coeffs
    .iter()
    .rev()
    .find(|&&c| c != 0)
    .is_some_and(|&c| c < 0)
  {
    -g
  } else {
    g
  }
}

/// Canonicalize a proper fraction that Apart cannot decompose further into
/// wolframscript's display form: the denominator's signed integer content is
/// divided out against the numerator's (Apart[(2+4x)/(-3+x-x^2)] →
/// (-2*(1+2*x))/(3-x+x^2), Apart[(3+6x)/(2+2x^2)] → (3*(1+2*x))/(2*(1+x^2))),
/// a multi-term numerator keeps its |content| > 1 factored out, and a -1
/// content distributes back into the sum ((-1-2*x)/(3-x+x^2)).
fn normalize_irreducible_quotient(
  num_coeffs: &[i128],
  den_coeffs: &[i128],
  var: &str,
) -> Option<Expr> {
  let cn = signed_content(num_coeffs);
  let cd = signed_content(den_coeffs);
  if cn == 0 || cd == 0 {
    return None;
  }
  let num_prim: Vec<i128> = num_coeffs.iter().map(|c| c / cn).collect();
  let den_prim: Vec<i128> = den_coeffs.iter().map(|c| c / cd).collect();
  let (n_c, d_c) = rat_reduce(cn, cd);
  // A ±1 numerator over an unscaled denominator displays as (den)^(-1) /
  // -(den)^(-1): Apart[2/(-2-2x-4x^2)] → -(1+x+2x^2)^(-1).
  if num_prim == [1] && n_c.abs() == 1 && d_c == 1 {
    let pow = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(coeffs_to_expr(&den_prim, var)),
      right: Box::new(Expr::Integer(-1)),
    };
    return Some(if n_c < 0 {
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(pow),
      }
    } else {
      pow
    });
  }
  let nonzero = num_prim.iter().filter(|&&c| c != 0).count();
  let num_expr = if nonzero >= 2 && n_c.abs() > 1 {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(n_c)),
      right: Box::new(coeffs_to_expr(&num_prim, var)),
    }
  } else {
    let scaled: Vec<i128> = num_prim.iter().map(|c| c * n_c).collect();
    coeffs_to_expr(&scaled, var)
  };
  let den_expr = if d_c == 1 {
    coeffs_to_expr(&den_prim, var)
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(d_c)),
      right: Box::new(coeffs_to_expr(&den_prim, var)),
    }
  };
  Some(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(num_expr),
    right: Box::new(den_expr),
  })
}

/// Build one partial-fraction term `numerator / (L * factor^k)` with integer
/// numerator coefficients (rational denominators cleared, content reduced
/// against L), in the natural shape the evaluator renders like wolframscript.
fn build_apart_term(
  pnum: &[Rat],
  factor: &[i128],
  k: usize,
  var: &str,
) -> Expr {
  let mut l: i128 = 1;
  for r in pnum {
    l = lcm_i128(l, r.d);
  }
  let mut inum: Vec<i128> = pnum.iter().map(|r| r.n * (l / r.d)).collect();
  let mut g = l.abs();
  for &c in &inum {
    g = gcd_i128(g, c.abs());
  }
  if g > 1 {
    for c in inum.iter_mut() {
      *c /= g;
    }
    l /= g;
  }
  // Hoist integer content out of multi-term numerators, signed by the
  // highest-degree coefficient (FactorTerms convention): (6+3x)/(3-x+x^2)
  // displays as (3*(2+x))/(3-x+x^2), (6-3x) as -3*(-2+x). Content-1
  // numerators stay expanded ((37+17x)/(5*(3-x+x^2))).
  let nonzero = inum.iter().filter(|&&c| c != 0).count();
  let content = signed_content(&inum);
  let num_expr = if nonzero >= 2 && content.abs() > 1 {
    let primitive: Vec<i128> = inum.iter().map(|c| c / content).collect();
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(content)),
      right: Box::new(coeffs_to_expr(&primitive, var)),
    }
  } else {
    coeffs_to_expr(&inum, var)
  };
  let factor_expr = coeffs_to_expr(factor, var);
  let factor_pow = if k == 1 {
    factor_expr
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(factor_expr),
      right: Box::new(Expr::Integer(k as i128)),
    }
  };
  let denom = if l == 1 {
    factor_pow
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(l)),
      right: Box::new(factor_pow),
    }
  };
  Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(num_expr),
    right: Box::new(denom),
  }
}

/// Partial-fraction decomposition for a proper fraction whose denominator has
/// at least one irreducible quadratic (or higher-degree) factor over the
/// integers — the linear-residue path above only covers integer linear roots.
/// Factors the denominator, sets up the standard ansatz (constant numerators
/// over linear factors, linear numerators `B x + C` over quadratic factors,
/// and the analogous higher-degree forms, repeated per factor multiplicity),
/// then solves the rational linear system by matching coefficients. Each term
/// is built in the natural `numerator / (scale * factor^k)` shape so the
/// evaluator's own rendering reproduces wolframscript's canonical output.
///
/// Returns None (deferring to the linear path) when every factor is linear,
/// when fewer than two factor instances are present, or when factoring / the
/// linear solve does not cleanly succeed.
fn apart_general(
  num_coeffs: &[i128],
  den_coeffs: &[i128],
  var: &str,
) -> Option<Expr> {
  let factor_exprs = factor_integer_poly(den_coeffs, var);
  let mut factors: Vec<Vec<i128>> = Vec::new();
  for f in &factor_exprs {
    if let Expr::Integer(_) = f {
      continue; // constant content — folded into the overall scale below
    }
    let mut c = extract_poly_coeffs(f, var)?;
    if c.len() >= 2 {
      // Normalize each factor to its primitive, positive-leading form —
      // wolframscript presents partial-fraction denominators content-hoisted
      // and leading-positive: Apart[(4x-3x^2-x^3)/(x+x^2-5x^3)] = 1/5 +
      // (-19 + 16*x)/(5*(-1 - x + 5*x^2)), not (19-16*x)/(5+5*x-25*x^2).
      // Content and sign are absorbed by `scale` (= den / prod of factors)
      // below, and the content resurfaces as the term's scalar multiplier.
      let content = c.iter().fold(0i128, |acc, &v| gcd_i128(acc, v.abs()));
      if content > 1 {
        for v in c.iter_mut() {
          *v /= content;
        }
      }
      if c.last().map(|&l| l < 0).unwrap_or(false) {
        for v in c.iter_mut() {
          *v = -*v;
        }
      }
      factors.push(c);
    }
  }
  // Two-or-more factor instances, and at least one the integer-root residue
  // path can't handle: a non-linear factor, or a non-monic linear factor
  // (rational root, e.g. 3*x + 1 — differential fuzzer, seed
  // 10107924694092248000). The all-monic-linear case stays on the existing
  // (well-tested) linear path.
  let needs_general = factors.iter().any(|c| {
    c.len() >= 3
      || (c.len() == 2 && c.last().map(|&l| l.abs() != 1) == Some(true))
  });
  if factors.len() < 2 || !needs_general {
    return None;
  }

  let mut prod_nc = vec![1i128];
  for f in &factors {
    prod_nc = poly_mul_i128(&prod_nc, f);
  }
  let scale = poly_exact_divide(den_coeffs, &prod_nc)?;
  if scale.len() != 1 || scale[0] == 0 {
    return None;
  }
  let scale = scale[0];

  // Distinct factors with multiplicity, preserving factor_integer_poly's order
  // (wolframscript's factor order).
  let mut groups: Vec<(Vec<i128>, usize)> = Vec::new();
  for f in &factors {
    if let Some(grp) = groups.iter_mut().find(|(c, _)| c == f) {
      grp.1 += 1;
    } else {
      groups.push((f.clone(), 1));
    }
  }

  let deg_d = prod_nc.len() - 1; // number of unknowns
  let mut basis: Vec<Vec<Rat>> = Vec::new();
  let mut meta: Vec<(usize, usize, usize)> = Vec::new(); // (group, k, t)
  for (gi, (f, e)) in groups.iter().enumerate() {
    let dfi = f.len() - 1;
    for k in 1..=*e {
      let mut fk = vec![1i128];
      for _ in 0..k {
        fk = poly_mul_i128(&fk, f);
      }
      let base = poly_exact_divide(&prod_nc, &fk)?;
      for t in 0..dfi {
        let mut col = vec![Rat::int(0); deg_d];
        for (i, &c) in base.iter().enumerate() {
          let idx = i + t;
          if idx >= deg_d {
            return None;
          }
          col[idx] = Rat::int(c);
        }
        basis.push(col);
        meta.push((gi, k, t));
      }
    }
  }
  if basis.len() != deg_d {
    return None;
  }

  let mut rhs = vec![Rat::int(0); deg_d];
  for (i, &c) in num_coeffs.iter().enumerate() {
    if i >= deg_d {
      return None; // not a proper fraction
    }
    rhs[i] = Rat::new(c, scale);
  }
  let mat: Vec<Vec<Rat>> = (0..deg_d)
    .map(|r| basis.iter().map(|col| col[r]).collect())
    .collect();
  let sol = solve_rat_system(mat, rhs)?;

  // Regroup solved coefficients into numerator polynomials P_{group,k}(x).
  let mut term_nums: Vec<((usize, usize), Vec<Rat>)> = Vec::new();
  for (u, &(gi, k, t)) in meta.iter().enumerate() {
    let dfi = groups[gi].0.len() - 1;
    if let Some((_, v)) = term_nums
      .iter_mut()
      .find(|((g, kk), _)| *g == gi && *kk == k)
    {
      v[t] = sol[u];
    } else {
      let mut v = vec![Rat::int(0); dfi];
      v[t] = sol[u];
      term_nums.push(((gi, k), v));
    }
  }

  // Emit terms in factor order, ascending power k.
  let mut terms: Vec<Expr> = Vec::new();
  for (gi, (f, _e)) in groups.iter().enumerate() {
    let mut ks: Vec<&((usize, usize), Vec<Rat>)> =
      term_nums.iter().filter(|((g, _), _)| *g == gi).collect();
    ks.sort_by_key(|((_, k), _)| *k);
    for ((_, k), pnum) in ks {
      if pnum.iter().all(|r| r.is_zero()) {
        continue;
      }
      terms.push(build_apart_term(pnum, f, *k, var));
    }
  }
  if terms.is_empty() {
    return None;
  }
  let sum = build_sum(terms);
  Some(crate::evaluator::evaluate_expr_to_expr(&sum).unwrap_or(sum))
}

/// A reduced rational number with a positive denominator. Used for the
/// repeated-root partial-fraction linear solve.
#[derive(Clone, Copy)]
struct Rat {
  n: i128,
  d: i128,
}

impl Rat {
  fn new(n: i128, d: i128) -> Rat {
    let (n, d) = rat_reduce(n, d);
    Rat { n: n, d: d }
  }
  fn int(n: i128) -> Rat {
    Rat { n, d: 1 }
  }
  fn is_zero(self) -> bool {
    self.n == 0
  }
  fn sub(self, o: Rat) -> Rat {
    Rat::new(self.n * o.d - o.n * self.d, self.d * o.d)
  }
  fn mul(self, o: Rat) -> Rat {
    Rat::new(self.n * o.n, self.d * o.d)
  }
  fn div(self, o: Rat) -> Rat {
    Rat::new(self.n * o.d, self.d * o.n)
  }
}

/// Partial-fraction decomposition for a proper fraction whose denominator
/// factors completely into integer linear factors, allowing repeated roots.
///
/// The denominator is `scale * prod_i (x - r_i)^{m_i}` where
/// `scale = overall_factor * rem_const`. We solve for the coefficients
/// `A_{i,k}` in `N(x)/D(x) = sum_{i,k} A_{i,k}/(x - r_i)^k` by matching
/// polynomial coefficients — a square linear system over the rationals.
/// Returns `None` if the system is singular (shouldn't happen for a genuine
/// proper fraction) so the caller can leave the input unevaluated.
fn apart_repeated_roots(
  num_coeffs: &[i128],
  roots: &[i128],
  rem_const: i128,
  overall_factor: i128,
  var: &str,
) -> Option<Expr> {
  let scale = overall_factor.checked_mul(rem_const)?;
  if scale == 0 {
    return None;
  }
  let degree = roots.len();
  if degree == 0 {
    return None;
  }

  // Distinct roots with multiplicity, ordered by descending root value so the
  // emitted factors (-root + x) appear in ascending order, matching
  // wolframscript.
  let mut distinct: Vec<i128> = roots.to_vec();
  distinct.sort_by(|a, b| b.cmp(a));
  distinct.dedup();
  let multiplicity =
    |r: i128| -> usize { roots.iter().filter(|&&x| x == r).count() };

  // Monic product M(x) = prod (x - r_i)^{m_i}.
  let mut m_poly = vec![1i128];
  for &r in roots {
    // Multiply by (x - r): coeffs are [-r, 1].
    let mut next = vec![0i128; m_poly.len() + 1];
    for (i, &c) in m_poly.iter().enumerate() {
      next[i] += c * (-r);
      next[i + 1] += c;
    }
    m_poly = next;
  }

  // Basis: for each distinct root (descending) and power k from its
  // multiplicity down to 1, the polynomial B = M(x)/(x - r)^k.
  struct Basis {
    root: i128,
    k: usize,
    coeffs: Vec<i128>,
  }
  let mut basis: Vec<Basis> = Vec::with_capacity(degree);
  for &r in &distinct {
    let m = multiplicity(r);
    // M / (x - r)^j for j = 1..=m, computed incrementally.
    let mut reduced = m_poly.clone();
    let mut by_power: Vec<Vec<i128>> = Vec::with_capacity(m);
    for _ in 0..m {
      reduced = divide_by_root(&reduced, r);
      by_power.push(reduced.clone());
    }
    // Emit powers high → low so higher powers print first.
    for k in (1..=m).rev() {
      basis.push(Basis {
        root: r,
        k,
        coeffs: by_power[k - 1].clone(),
      });
    }
  }
  if basis.len() != degree {
    return None;
  }

  // Build the d×d system  M_sys · a = b  with rows = coefficient of x^p.
  let mut mat: Vec<Vec<Rat>> = vec![vec![Rat::int(0); degree]; degree];
  for (j, b) in basis.iter().enumerate() {
    for (p, &c) in b.coeffs.iter().enumerate() {
      if p < degree {
        mat[p][j] = Rat::int(c);
      }
    }
  }
  let mut rhs: Vec<Rat> = (0..degree)
    .map(|p| Rat::new(*num_coeffs.get(p).unwrap_or(&0), scale))
    .collect();

  // Gaussian elimination over the rationals.
  for col in 0..degree {
    let pivot = (col..degree).find(|&r| !mat[r][col].is_zero())?;
    mat.swap(col, pivot);
    rhs.swap(col, pivot);
    let inv = mat[col][col];
    for j in col..degree {
      mat[col][j] = mat[col][j].div(inv);
    }
    rhs[col] = rhs[col].div(inv);
    for r in 0..degree {
      if r != col && !mat[r][col].is_zero() {
        let factor = mat[r][col];
        for j in col..degree {
          mat[r][j] = mat[r][j].sub(factor.mul(mat[col][j]));
        }
        rhs[r] = rhs[r].sub(factor.mul(rhs[col]));
      }
    }
  }

  // Assemble the result terms (already in the canonical output order).
  let mut terms: Vec<Expr> = Vec::new();
  for (j, b) in basis.iter().enumerate() {
    let a = rhs[j];
    if a.is_zero() {
      continue;
    }
    let neg = a.n < 0;
    let an = a.n.abs();
    let ad = a.d; // already positive
    let linear_factor = if b.root == 0 {
      Expr::Identifier(var.to_string())
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-b.root)),
        right: Box::new(Expr::Identifier(var.to_string())),
      }
    };
    // The factor raised to the term's power: f for k == 1, f^k otherwise.
    let den_factor = if b.k == 1 {
      linear_factor.clone()
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(linear_factor.clone()),
        right: Box::new(Expr::Integer(b.k as i128)),
      }
    };
    if ad == 1 && an == 1 {
      // (factor)^(-k), negated via UnaryOp so it renders -x^(-2).
      let pow = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(linear_factor),
        right: Box::new(Expr::Integer(-(b.k as i128))),
      };
      terms.push(if neg {
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(pow),
        }
      } else {
        pow
      });
    } else {
      // Signed numerator over the (possibly scaled) factor, so negatives
      // render -2/x^2 / -3/(2*x^2), matching wolframscript.
      let signed_an = if neg { -an } else { an };
      let denom = if ad == 1 {
        den_factor
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(ad)),
          right: Box::new(den_factor),
        }
      };
      terms.push(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(signed_an)),
        right: Box::new(denom),
      });
    }
  }
  if terms.is_empty() {
    return None;
  }
  Some(build_sum(terms))
}

/// a - b*c over i128 fractions, reduced; None on overflow.
fn rat_sub_mul(
  a: (i128, i128),
  b: (i128, i128),
  c: (i128, i128),
) -> Option<(i128, i128)> {
  let bn = b.0.checked_mul(c.0)?;
  let bd = b.1.checked_mul(c.1)?;
  let (bn, bd) = rat_reduce(bn, bd);
  let n = a.0.checked_mul(bd)?.checked_sub(bn.checked_mul(a.1)?)?;
  let d = a.1.checked_mul(bd)?;
  Some(rat_reduce(n, d))
}

/// Rational-coefficient polynomial long division: num/den → (quotient,
/// remainder) as reduced (numerator, denominator) fraction pairs. Unlike
/// `poly_long_divide` this never bails on a non-integer leading quotient
/// (Apart[(2 - 4 x)/(2 - 5 x)] needs the quotient 4/5). Returns None on
/// i128 overflow.
fn poly_long_divide_rat(
  num: &[i128],
  den: &[i128],
) -> Option<(Vec<(i128, i128)>, Vec<(i128, i128)>)> {
  let n_deg = num.len();
  let d_deg = den.len();
  let lead_den = den[d_deg - 1];
  if n_deg < d_deg || lead_den == 0 {
    return None;
  }
  let mut remainder: Vec<(i128, i128)> = num.iter().map(|&c| (c, 1)).collect();
  let mut quotient: Vec<(i128, i128)> = vec![(0, 1); n_deg - d_deg + 1];

  for i in (0..quotient.len()).rev() {
    let rem_idx = i + d_deg - 1;
    if rem_idx >= remainder.len() {
      continue;
    }
    let q = rat_reduce(remainder[rem_idx].0, remainder[rem_idx].1 * lead_den);
    quotient[i] = q;
    for j in 0..d_deg {
      remainder[i + j] = rat_sub_mul(remainder[i + j], q, (den[j], 1))?;
    }
  }
  Some((quotient, remainder))
}

/// Build the polynomial expression for rational coefficients (ascending
/// powers of `var`), evaluated so terms canonicalize.
fn rat_coeffs_to_expr(
  coeffs: &[(i128, i128)],
  var: &str,
) -> Result<Expr, InterpreterError> {
  let mut terms: Vec<Expr> = Vec::new();
  for (i, &(n, d)) in coeffs.iter().enumerate() {
    if n == 0 {
      continue;
    }
    let coeff = if d == 1 {
      Expr::Integer(n)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
      }
    };
    let term = match i {
      0 => coeff,
      _ => {
        let var_pow = if i == 1 {
          Expr::Identifier(var.to_string())
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::Identifier(var.to_string())),
            right: Box::new(Expr::Integer(i as i128)),
          }
        };
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(coeff),
          right: Box::new(var_pow),
        }
      }
    };
    terms.push(term);
  }
  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  })
}

pub fn poly_long_divide(num: &[i128], den: &[i128]) -> (Vec<i128>, Vec<i128>) {
  let n_deg = num.len();
  let d_deg = den.len();
  if n_deg < d_deg {
    return (vec![0], num.to_vec());
  }
  let mut remainder = num.to_vec();
  let mut quotient = vec![0i128; n_deg - d_deg + 1];
  let lead_den = den[d_deg - 1];
  if lead_den == 0 {
    return (vec![0], num.to_vec());
  }

  for i in (0..quotient.len()).rev() {
    let rem_idx = i + d_deg - 1;
    if rem_idx >= remainder.len() {
      continue;
    }
    if remainder[rem_idx] % lead_den != 0 {
      // Non-integer quotient - stop here
      return (vec![0], num.to_vec());
    }
    let q = remainder[rem_idx] / lead_den;
    quotient[i] = q;
    for j in 0..d_deg {
      remainder[i + j] -= q * den[j];
    }
  }
  (quotient, remainder)
}

// ─── Symbolic Apart (multivariate) ─────────────────────────────────

/// Flatten a product expression into its factors.
fn flatten_product_factors(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args {
        flatten_product_factors(a, out);
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      flatten_product_factors(left, out);
      flatten_product_factors(right, out);
    }
    _ => out.push(expr.clone()),
  }
}

/// Check if expr is a polynomial of degree exactly 1 in var.
/// Returns Some((coeff_of_var, constant_term)) if linear.
fn extract_linear_coeffs(expr: &Expr, var: &str) -> Option<(Expr, Expr)> {
  let expanded = expand_and_combine(expr);
  // Constant term: substitute var=0
  let constant =
    crate::syntax::substitute_variable(&expanded, var, &Expr::Integer(0));
  let constant =
    crate::evaluator::evaluate_expr_to_expr(&constant).unwrap_or(constant);
  // Value at var=1: substitute var=1
  let at_one =
    crate::syntax::substitute_variable(&expanded, var, &Expr::Integer(1));
  let at_one =
    crate::evaluator::evaluate_expr_to_expr(&at_one).unwrap_or(at_one);
  // linear_coeff = at_one - constant
  let coeff = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(at_one),
    right: Box::new(constant.clone()),
  };
  let coeff = crate::evaluator::evaluate_expr_to_expr(&coeff).unwrap_or(coeff);
  // Check that the coefficient is not zero (otherwise not linear)
  if matches!(&coeff, Expr::Integer(0)) {
    return None;
  }
  // Verify it's actually linear (degree 1): check that at var=2, value = 2*coeff + constant
  let at_two =
    crate::syntax::substitute_variable(&expanded, var, &Expr::Integer(2));
  let at_two =
    crate::evaluator::evaluate_expr_to_expr(&at_two).unwrap_or(at_two);
  // expected = 2*coeff + constant
  let expected = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(coeff.clone()),
    }),
    right: Box::new(constant.clone()),
  };
  let expected =
    crate::evaluator::evaluate_expr_to_expr(&expected).unwrap_or(expected);
  let diff = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(at_two),
    right: Box::new(expected),
  };
  let diff = crate::evaluator::evaluate_expr_to_expr(&diff).unwrap_or(diff);
  if !matches!(&diff, Expr::Integer(0)) {
    return None; // Not linear
  }
  Some((coeff, constant))
}

/// If `factor` is linear in `var` with a negative coefficient on `var`,
/// negate the factor (so the variable term has a positive coefficient)
/// and report `true`. Otherwise return the factor unchanged with `false`.
/// Used to match Wolfram's Apart output convention of writing each
/// linear denominator factor with the focus variable on the right and a
/// positive coefficient — the negation is folded into the term's sign.
fn negate_if_var_coeff_negative(factor: &Expr, var: &str) -> (Expr, bool) {
  let coeffs = match extract_linear_coeffs(factor, var) {
    Some(c) => c,
    None => return (factor.clone(), false),
  };
  let coeff_negative = match &coeffs.0 {
    Expr::Integer(n) => *n < 0,
    Expr::Real(f) => *f < 0.0,
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      ..
    } => true,
    _ => false,
  };
  if !coeff_negative {
    return (factor.clone(), false);
  }
  // Negate the factor: -(coeff*var + constant) = (-coeff)*var + (-constant)
  let negated = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(factor.clone()),
  };
  let negated =
    crate::evaluator::evaluate_expr_to_expr(&negated).unwrap_or(negated);
  (negated, true)
}

/// Extract leading sign from a Times expression.
/// Returns (sign, abs_expr) where sign is 1 or -1.
fn extract_leading_sign(expr: &Expr) -> (i128, Expr) {
  match expr {
    Expr::Integer(n) if *n < 0 => (-1, Expr::Integer(-n)),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => (-1, *operand.clone()),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let (lsign, labs) = extract_leading_sign(left);
      if lsign < 0 {
        let rebuilt =
          crate::functions::math_ast::times_ast(&[labs, *right.clone()])
            .unwrap_or(*right.clone());
        (-1, rebuilt)
      } else {
        (1, expr.clone())
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      let (lsign, labs) = extract_leading_sign(&args[0]);
      if lsign < 0 {
        let mut new_args = vec![labs];
        new_args.extend_from_slice(&args[1..]);
        let rebuilt = crate::functions::math_ast::times_ast(&new_args)
          .unwrap_or(expr.clone());
        (-1, rebuilt)
      } else {
        (1, expr.clone())
      }
    }
    _ => (1, expr.clone()),
  }
}

/// Symbolic partial fraction decomposition for multivariate expressions.
/// Factor the denominator, find linear factors in var, apply cover-up method.
fn apart_symbolic(
  expr: &Expr,
  num: &Expr,
  den: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  // Factor the denominator
  let den_factored = factor_ast(&[den.clone()])?;
  // Collect factors
  let mut factors = Vec::new();
  flatten_product_factors(&den_factored, &mut factors);

  // Separate linear (in var) factors from non-linear/constant factors
  let mut linear_factors: Vec<Expr> = Vec::new(); // the full factor expression
  let mut linear_roots: Vec<Expr> = Vec::new(); // root = -constant/coeff
  let mut other_factor = Expr::Integer(1);

  for f in &factors {
    if !crate::functions::polynomial_ast::contains_var(f, var) {
      // Constant factor (doesn't contain var)
      other_factor = crate::functions::math_ast::times_ast(&[
        other_factor.clone(),
        f.clone(),
      ])
      .unwrap_or(other_factor);
      continue;
    }
    if let Some((coeff, constant)) = extract_linear_coeffs(f, var) {
      // Root: var = -constant/coeff
      let neg_const = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(constant),
      };
      let root = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(neg_const),
        right: Box::new(coeff),
      };
      let root = crate::evaluator::evaluate_expr_to_expr(&root).unwrap_or(root);
      linear_factors.push(f.clone());
      linear_roots.push(root);
    } else {
      // Non-linear factor containing var — can't decompose
      return Ok(expr.clone());
    }
  }

  if linear_factors.len() < 2 {
    return Ok(expr.clone());
  }

  // Apply cover-up method for partial fractions
  let n = linear_roots.len();
  let mut result_terms = Vec::new();

  for i in 0..n {
    // Evaluate numerator at root_i
    let num_val =
      crate::syntax::substitute_variable(num, var, &linear_roots[i]);
    let num_val =
      crate::evaluator::evaluate_expr_to_expr(&num_val).unwrap_or(num_val);

    // Compute scalar = other_factor * product_{j!=i} factor_j(r_i)
    let mut scalar_parts: Vec<Expr> = vec![other_factor.clone()];
    for j in 0..n {
      if i != j {
        let fj_at_ri = crate::syntax::substitute_variable(
          &linear_factors[j],
          var,
          &linear_roots[i],
        );
        let fj_at_ri = crate::evaluator::evaluate_expr_to_expr(&fj_at_ri)
          .unwrap_or(fj_at_ri);
        scalar_parts.push(fj_at_ri);
      }
    }
    let scalar = crate::functions::math_ast::times_ast(&scalar_parts)
      .unwrap_or(Expr::Integer(1));

    // Extract sign from scalar: if leading coefficient is negative, flip sign
    let (sign, abs_scalar) = extract_leading_sign(&scalar);

    // Wolfram displays the focus variable with positive coefficient inside
    // each linear denominator factor: `(x - y)` (y-coeff = -1) is rewritten
    // as `-(-x + y)` and the leading minus is folded into the term sign.
    // This keeps the per-factor `var` term aligned with Wolfram's
    // canonical Plus order for that factor.
    let (factor_for_display, factor_negated) =
      negate_if_var_coeff_negative(&linear_factors[i], var);
    let mut term_sign = sign;
    if factor_negated {
      term_sign = -term_sign;
    }

    // Build denominator = abs_scalar * factor_i (always positive scalar)
    let mut denom_factors = Vec::new();
    flatten_product_factors(&abs_scalar, &mut denom_factors);
    denom_factors.push(factor_for_display);
    let full_denom = crate::functions::math_ast::times_ast(&denom_factors)
      .unwrap_or(Expr::Integer(1));

    // Build fraction. When the term is negative, fold the sign into the
    // numerator so the result evaluates as `-num_val / full_denom` and
    // gets factored by Times canonical ordering (e.g. `-1/(2*x*…)` is
    // displayed as `-1/2 * 1/(x*…)`), matching Wolfram's Apart format.
    let signed_num = if term_sign < 0 {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(num_val),
      }
    } else {
      num_val
    };
    let frac = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(signed_num),
      right: Box::new(full_denom),
    };
    let frac = crate::evaluator::evaluate_expr_to_expr(&frac).unwrap_or(frac);
    result_terms.push(frac);
  }

  if result_terms.is_empty() {
    return Ok(expr.clone());
  }

  // Defer to canonical Plus ordering so terms (positive and negative) sort
  // by Wolfram's rules. Falls back to a simple positive-then-negative
  // chain if plus_ast can't handle the inputs.
  let summed = crate::functions::math_ast::plus_ast(&result_terms)
    .unwrap_or_else(|_| {
      let mut positive: Vec<Expr> = Vec::new();
      let mut negative: Vec<Expr> = Vec::new();
      for t in &result_terms {
        match t {
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } => negative.push(*operand.clone()),
          _ => positive.push(t.clone()),
        }
      }
      let mut result = if positive.is_empty() {
        let first = negative.remove(0);
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(first),
        }
      } else {
        positive.remove(0)
      };
      for p in &positive {
        result = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(result),
          right: Box::new(p.clone()),
        };
      }
      for n in &negative {
        result = Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(result),
          right: Box::new(n.clone()),
        };
      }
      result
    });
  Ok(summed)
}

/// ApartSquareFree[expr] / ApartSquareFree[expr, var] — partial fraction
/// decomposition over the denominator's *syntactic* factorization: the
/// bases of the powers in the product are used exactly as given and never
/// factored further. So 1/(x^2 - 1) stays put (one square-free base) while
/// 1/((x - 1) (x + 2)^2) splits fully, and an expanded 1/(x^2 + x - 2)^2
/// keeps its quadratic base. Reuses Apart's rational linear solve; the
/// whole part of improper fractions is split off first.
pub fn apart_square_free_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated("ApartSquareFree", args));
  }
  // Thread over lists, like Apart.
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| {
        let mut sub = vec![item.clone()];
        sub.extend(args[1..].iter().cloned());
        apart_square_free_ast(&sub)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  let unchanged = || Ok(args[0].clone());
  let (num, den) = super::together::extract_num_den(&args[0]);
  if matches!(&den, Expr::Integer(1)) {
    return unchanged();
  }
  let var = if args.len() == 2 {
    match &args[1] {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Ok(unevaluated("ApartSquareFree", args));
      }
    }
  } else {
    match find_single_variable(&args[0]) {
      Some(v) => v,
      None => return unchanged(),
    }
  };

  // The denominator's syntactic factors: bases of integer powers in the
  // product, with constant factors folded into an overall scale.
  fn collect_syntactic_factors(e: &Expr, out: &mut Vec<(Expr, usize)>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        collect_syntactic_factors(left, out);
        collect_syntactic_factors(right, out);
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          collect_syntactic_factors(a, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => {
        if let Expr::Integer(k) = right.as_ref()
          && *k >= 1
        {
          out.push(((**left).clone(), *k as usize));
        } else {
          out.push((e.clone(), 1));
        }
      }
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        if let Expr::Integer(k) = &args[1]
          && *k >= 1
        {
          out.push((args[0].clone(), *k as usize));
        } else {
          out.push((e.clone(), 1));
        }
      }
      _ => out.push((e.clone(), 1)),
    }
  }
  let mut raw_factors: Vec<(Expr, usize)> = Vec::new();
  collect_syntactic_factors(&den, &mut raw_factors);

  let mut scale: i128 = 1;
  // Distinct bases by coefficient vector, multiplicities summed, in the
  // order given.
  let mut groups: Vec<(Vec<i128>, usize)> = Vec::new();
  for (base, mult) in &raw_factors {
    let coeffs = match extract_poly_coeffs(&expand_and_combine(base), &var) {
      Some(c) => c,
      None => return unchanged(),
    };
    if coeffs.len() <= 1 {
      // Constant factor: fold its mult-th power into the scale.
      let c = coeffs.first().copied().unwrap_or(0);
      if c == 0 {
        return unchanged();
      }
      for _ in 0..*mult {
        scale = match scale.checked_mul(c) {
          Some(s) => s,
          None => return unchanged(),
        };
      }
      continue;
    }
    if let Some(g) = groups.iter_mut().find(|(f, _)| *f == coeffs) {
      g.1 += mult;
    } else {
      groups.push((coeffs, *mult));
    }
  }
  if groups.is_empty() {
    return unchanged();
  }
  let num_coeffs = match extract_poly_coeffs(&expand_and_combine(&num), &var) {
    Some(c) => c,
    None => return unchanged(),
  };

  let mut prod_nc = vec![1i128];
  for (f, m) in &groups {
    for _ in 0..*m {
      prod_nc = poly_mul_i128(&prod_nc, f);
    }
  }
  let deg_d = prod_nc.len() - 1;

  // Split off the whole part with rational long division.
  let mut rem: Vec<Rat> =
    num_coeffs.iter().map(|&c| Rat::new(c, scale)).collect();
  let mut quot: Vec<Rat> = vec![Rat::int(0); rem.len().saturating_sub(deg_d)];
  let lead = Rat::int(*prod_nc.last().unwrap());
  while rem.len() > deg_d {
    let l = *rem.last().unwrap();
    let pos = rem.len() - 1 - deg_d;
    if !l.is_zero() {
      let c = l.div(lead);
      quot[pos] = c;
      for (i, &p) in prod_nc.iter().enumerate() {
        rem[pos + i] = rem[pos + i].sub(c.mul(Rat::int(p)));
      }
    }
    rem.pop();
  }

  // The same column basis Apart's general solver uses, but over the
  // syntactic groups: unknown numerators of degree < deg(base) for each
  // power of each base. A singular or ill-sized system (e.g. non-coprime
  // bases) leaves the input unchanged.
  let mut basis: Vec<Vec<Rat>> = Vec::new();
  let mut meta: Vec<(usize, usize, usize)> = Vec::new();
  for (gi, (f, e)) in groups.iter().enumerate() {
    let dfi = f.len() - 1;
    for k in 1..=*e {
      let mut fk = vec![1i128];
      for _ in 0..k {
        fk = poly_mul_i128(&fk, f);
      }
      let base = match crate::functions::polynomial_ast::poly_exact_divide(
        &prod_nc, &fk,
      ) {
        Some(b) => b,
        None => return unchanged(),
      };
      for t in 0..dfi {
        let mut col = vec![Rat::int(0); deg_d];
        for (i, &c) in base.iter().enumerate() {
          let idx = i + t;
          if idx >= deg_d {
            return unchanged();
          }
          col[idx] = Rat::int(c);
        }
        basis.push(col);
        meta.push((gi, k, t));
      }
    }
  }
  if basis.len() != deg_d {
    return unchanged();
  }
  let mut rhs = vec![Rat::int(0); deg_d];
  for (i, &c) in rem.iter().enumerate() {
    rhs[i] = c;
  }
  let mat: Vec<Vec<Rat>> = (0..deg_d)
    .map(|r| basis.iter().map(|col| col[r]).collect())
    .collect();
  let sol = match solve_rat_system(mat, rhs) {
    Some(s) => s,
    None => return unchanged(),
  };

  let mut term_nums: Vec<((usize, usize), Vec<Rat>)> = Vec::new();
  for (u, &(gi, k, t)) in meta.iter().enumerate() {
    let dfi = groups[gi].0.len() - 1;
    if let Some((_, v)) = term_nums
      .iter_mut()
      .find(|((g, kk), _)| *g == gi && *kk == k)
    {
      v[t] = sol[u];
    } else {
      let mut v = vec![Rat::int(0); dfi];
      v[t] = sol[u];
      term_nums.push(((gi, k), v));
    }
  }

  let mut terms: Vec<Expr> = Vec::new();
  // Whole part first (evaluation re-sorts the sum canonically anyway).
  for (p, &c) in quot.iter().enumerate() {
    if c.is_zero() {
      continue;
    }
    let coeff = if c.d == 1 {
      Expr::Integer(c.n)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(c.n), Expr::Integer(c.d)].into(),
      }
    };
    let term = match p {
      0 => coeff,
      _ => Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          coeff,
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![Expr::Identifier(var.clone()), Expr::Integer(p as i128)]
              .into(),
          },
        ]
        .into(),
      },
    };
    terms.push(term);
  }
  for (gi, (f, _e)) in groups.iter().enumerate() {
    let mut ks: Vec<&((usize, usize), Vec<Rat>)> =
      term_nums.iter().filter(|((g, _), _)| *g == gi).collect();
    ks.sort_by_key(|((_, k), _)| *k);
    for ((_, k), pnum) in ks {
      if pnum.iter().all(|r| r.is_zero()) {
        continue;
      }
      terms.push(build_apart_term(pnum, f, *k, &var));
    }
  }
  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }
  let sum = build_sum(terms);
  Ok(crate::evaluator::evaluate_expr_to_expr(&sum).unwrap_or(sum))
}
