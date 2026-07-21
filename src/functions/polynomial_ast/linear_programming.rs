//! LinearProgramming[c, m, b] — exact linear program solver.
//!
//! Minimizes `c · x` subject to the constraints encoded by `m` and `b` and
//! (by default) `x >= 0`. Each constraint row `m[[i]] · x` is compared to
//! `b[[i]]`; when `b[[i]]` is a bare number the relation is `>=`, and when it
//! is a pair `{value, sign}` the sign selects `>=` (1), `==` (0) or `<=` (-1).
//!
//! The solver is an exact two-phase simplex over `BigInt` rationals. It uses
//! Dantzig's entering rule — which reproduces the vertex wolframscript reports
//! when a problem has multiple optima — and falls back to Bland's rule after a
//! generous iteration budget so it can never cycle on degenerate problems.

use crate::InterpreterError;
use crate::functions::math_ast::gcd_bigint;
use crate::syntax::{Expr, unevaluated};
use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

/// An exact rational number with a `BigInt` numerator/denominator, kept in
/// lowest terms with a positive denominator.
#[derive(Clone, PartialEq, Eq)]
struct Rat {
  num: BigInt,
  den: BigInt,
}

impl Rat {
  fn new(num: BigInt, den: BigInt) -> Self {
    let mut r = Rat { num, den };
    r.reduce();
    r
  }
  fn from_int(n: BigInt) -> Self {
    Rat {
      num: n,
      den: BigInt::one(),
    }
  }
  fn zero() -> Self {
    Rat::from_int(BigInt::zero())
  }
  fn one() -> Self {
    Rat::from_int(BigInt::one())
  }
  fn reduce(&mut self) {
    if self.den.is_negative() {
      self.num = -&self.num;
      self.den = -&self.den;
    }
    let g = gcd_bigint(&self.num, &self.den);
    if !g.is_zero() && !g.is_one() {
      self.num /= &g;
      self.den /= &g;
    }
  }
  fn is_zero(&self) -> bool {
    self.num.is_zero()
  }
  fn is_negative(&self) -> bool {
    self.num.is_negative()
  }
  fn is_positive(&self) -> bool {
    self.num.is_positive()
  }
  fn neg(&self) -> Rat {
    Rat {
      num: -&self.num,
      den: self.den.clone(),
    }
  }
  fn add(&self, o: &Rat) -> Rat {
    Rat::new(&self.num * &o.den + &o.num * &self.den, &self.den * &o.den)
  }
  fn sub(&self, o: &Rat) -> Rat {
    Rat::new(&self.num * &o.den - &o.num * &self.den, &self.den * &o.den)
  }
  fn mul(&self, o: &Rat) -> Rat {
    Rat::new(&self.num * &o.num, &self.den * &o.den)
  }
  fn div(&self, o: &Rat) -> Rat {
    Rat::new(&self.num * &o.den, &self.den * &o.num)
  }
  fn cmp(&self, o: &Rat) -> std::cmp::Ordering {
    // den is always positive, so cross-multiplication preserves the order.
    (&self.num * &o.den).cmp(&(&o.num * &self.den))
  }
  fn to_expr(&self) -> Expr {
    if self.den.is_one() {
      bigint_to_expr(&self.num)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![bigint_to_expr(&self.num), bigint_to_expr(&self.den)].into(),
      }
    }
  }
}

fn bigint_to_expr(n: &BigInt) -> Expr {
  match i128::try_from(n.clone()) {
    Ok(small) => Expr::Integer(small),
    Err(_) => Expr::BigInteger(n.clone()),
  }
}

/// Convert a numeric expression to an exact rational, or None when it is not a
/// (real) exact number.
fn expr_to_rat(e: &Expr) -> Option<Rat> {
  match e {
    Expr::Integer(n) => Some(Rat::from_int(BigInt::from(*n))),
    Expr::BigInteger(n) => Some(Rat::from_int(n.clone())),
    Expr::Real(f) => rat_from_f64(*f),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => expr_to_rat(operand).map(|r| r.neg()),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      let n = expr_to_rat(&args[0])?;
      let d = expr_to_rat(&args[1])?;
      if d.is_zero() { None } else { Some(n.div(&d)) }
    }
    _ => {
      // Fall back to evaluating (handles e.g. 1/2 parsed as Times/Power).
      let ev = crate::evaluator::evaluate_expr_to_expr(e).ok()?;
      match &ev {
        Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => {
          expr_to_rat(&ev)
        }
        Expr::FunctionCall { name, .. } if name == "Rational" => {
          expr_to_rat(&ev)
        }
        _ => None,
      }
    }
  }
}

fn rat_from_f64(f: f64) -> Option<Rat> {
  if !f.is_finite() {
    return None;
  }
  if f == 0.0 {
    return Some(Rat::zero());
  }
  // Decompose the IEEE-754 double into mantissa * 2^exp exactly.
  let bits = f.to_bits();
  let sign = if bits >> 63 == 1 { -1i8 } else { 1 };
  let exponent = ((bits >> 52) & 0x7ff) as i64;
  let mantissa = if exponent == 0 {
    (bits & 0xf_ffff_ffff_ffff) << 1
  } else {
    (bits & 0xf_ffff_ffff_ffff) | 0x10_0000_0000_0000
  };
  let exp = exponent - 1075;
  let mut num = BigInt::from(mantissa);
  if sign < 0 {
    num = -num;
  }
  let mut den = BigInt::one();
  if exp >= 0 {
    num <<= exp as usize;
  } else {
    den <<= (-exp) as usize;
  }
  Some(Rat::new(num, den))
}

/// A single constraint relation.
#[derive(Clone, Copy, PartialEq)]
enum Rel {
  Ge,
  Eq,
  Le,
}

pub fn linear_programming_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Accept the 3-argument form LinearProgramming[c, m, b]. The 4-argument
  // bounds form is left unevaluated for now.
  if args.len() != 3 {
    return Ok(unevaluated("LinearProgramming", args));
  }
  let (Expr::List(c_items), Expr::List(m_rows), Expr::List(b_items)) =
    (&args[0], &args[1], &args[2])
  else {
    return Ok(unevaluated("LinearProgramming", args));
  };

  let n = c_items.len();
  let k = m_rows.len();
  if n == 0 || k == 0 || b_items.len() != k {
    return Ok(unevaluated("LinearProgramming", args));
  }

  // Objective coefficients.
  let mut c = Vec::with_capacity(n);
  for ci in c_items.iter() {
    match expr_to_rat(ci) {
      Some(r) => c.push(r),
      None => return Ok(unevaluated("LinearProgramming", args)),
    }
  }

  // Constraint matrix.
  let mut a: Vec<Vec<Rat>> = Vec::with_capacity(k);
  for row in m_rows.iter() {
    let Expr::List(row_items) = row else {
      return Ok(unevaluated("LinearProgramming", args));
    };
    if row_items.len() != n {
      return Ok(unevaluated("LinearProgramming", args));
    }
    let mut r = Vec::with_capacity(n);
    for e in row_items.iter() {
      match expr_to_rat(e) {
        Some(v) => r.push(v),
        None => return Ok(unevaluated("LinearProgramming", args)),
      }
    }
    a.push(r);
  }

  // Right-hand sides with their relation. A bare number means `>=`; a pair
  // {value, sign} selects the relation from the sign.
  let mut b = Vec::with_capacity(k);
  let mut rel = Vec::with_capacity(k);
  for item in b_items.iter() {
    match item {
      Expr::List(pair) if pair.len() == 2 => {
        let (Some(val), Some(sign)) =
          (expr_to_rat(&pair[0]), expr_to_rat(&pair[1]))
        else {
          return Ok(unevaluated("LinearProgramming", args));
        };
        rel.push(if sign.is_zero() {
          Rel::Eq
        } else if sign.is_negative() {
          Rel::Le
        } else {
          Rel::Ge
        });
        b.push(val);
      }
      _ => match expr_to_rat(item) {
        Some(v) => {
          b.push(v);
          rel.push(Rel::Ge);
        }
        None => return Ok(unevaluated("LinearProgramming", args)),
      },
    }
  }

  match solve_simplex(&c, &a, &b, &rel, n) {
    LpResult::Optimal(x) => {
      Ok(Expr::List(x.iter().map(|r| r.to_expr()).collect()))
    }
    LpResult::Unbounded => {
      crate::emit_message(
        "LinearProgramming::lpsub: This problem is unbounded.",
      );
      let inds: Vec<Expr> = (0..n)
        .map(|_| Expr::Identifier("Indeterminate".to_string()))
        .collect();
      Ok(Expr::List(inds.into()))
    }
    LpResult::Infeasible => {
      crate::emit_message(
        "LinearProgramming::lpsnf: No solution can be found that satisfies the constraints.",
      );
      Ok(unevaluated("LinearProgramming", args))
    }
  }
}

enum LpResult {
  Optimal(Vec<Rat>),
  Unbounded,
  Infeasible,
}

/// Two-phase simplex. Columns `0..n` are the structural (non-negative)
/// variables, followed by one slack/surplus per inequality and one artificial
/// per `>=`/`==` constraint. Each tableau row stores all column coefficients
/// plus a trailing right-hand-side entry.
fn solve_simplex(
  c: &[Rat],
  a: &[Vec<Rat>],
  b: &[Rat],
  rel: &[Rel],
  n: usize,
) -> LpResult {
  let k = a.len();

  // Normalize each constraint to a non-negative right-hand side, flipping the
  // row and relation when b_i < 0.
  let mut rows: Vec<Vec<Rat>> = Vec::with_capacity(k);
  let mut rhs: Vec<Rat> = Vec::with_capacity(k);
  let mut rels: Vec<Rel> = Vec::with_capacity(k);
  for i in 0..k {
    if b[i].is_negative() {
      rows.push(a[i].iter().map(|v| v.neg()).collect());
      rhs.push(b[i].neg());
      rels.push(match rel[i] {
        Rel::Ge => Rel::Le,
        Rel::Le => Rel::Ge,
        Rel::Eq => Rel::Eq,
      });
    } else {
      rows.push(a[i].clone());
      rhs.push(b[i].clone());
      rels.push(rel[i]);
    }
  }

  let n_slack = rels
    .iter()
    .filter(|r| matches!(r, Rel::Ge | Rel::Le))
    .count();
  let n_art = rels
    .iter()
    .filter(|r| matches!(r, Rel::Ge | Rel::Eq))
    .count();
  let slack_base = n;
  let art_base = n + n_slack;
  let total = n + n_slack + n_art;
  let rhs_col = total; // index of the RHS entry in each row

  let mut tab: Vec<Vec<Rat>> = Vec::with_capacity(k);
  let mut basis: Vec<usize> = Vec::with_capacity(k);
  let mut slack_i = 0usize;
  let mut art_i = 0usize;
  for i in 0..k {
    let mut row = vec![Rat::zero(); total + 1];
    for (j, v) in rows[i].iter().enumerate() {
      row[j] = v.clone();
    }
    row[rhs_col] = rhs[i].clone();
    match rels[i] {
      Rel::Le => {
        row[slack_base + slack_i] = Rat::one();
        basis.push(slack_base + slack_i);
        slack_i += 1;
      }
      Rel::Ge => {
        row[slack_base + slack_i] = Rat::one().neg();
        slack_i += 1;
        row[art_base + art_i] = Rat::one();
        basis.push(art_base + art_i);
        art_i += 1;
      }
      Rel::Eq => {
        row[art_base + art_i] = Rat::one();
        basis.push(art_base + art_i);
        art_i += 1;
      }
    }
    tab.push(row);
  }

  // Phase I: minimize the sum of the artificial variables.
  if n_art > 0 {
    let mut phase1_cost = vec![Rat::zero(); total];
    for cost in phase1_cost.iter_mut().take(total).skip(art_base) {
      *cost = Rat::one();
    }
    match run_simplex(&mut tab, &mut basis, &phase1_cost, total, rhs_col) {
      SimplexOutcome::Optimal(obj) => {
        if obj.is_positive() {
          return LpResult::Infeasible;
        }
      }
      // Phase I is bounded below by 0, so it can never be unbounded.
      SimplexOutcome::Unbounded => return LpResult::Infeasible,
    }
    // Drive any artificial still in the basis (at value 0) out of it.
    for i in 0..k {
      if basis[i] >= art_base {
        // Find a non-artificial, non-basic column with a nonzero pivot entry.
        let pivot_col = (0..art_base).find(|&j| !tab[i][j].is_zero());
        if let Some(j) = pivot_col {
          pivot(&mut tab, &mut basis, i, j, rhs_col);
        }
        // Otherwise the row is redundant; leaving the zero artificial basic is
        // harmless because Phase II ignores artificial columns.
      }
    }
  }

  // Phase II: minimize the true objective c·x. Artificial columns keep a large
  // implicit cost by being excluded from entering (their cost stays 0 but we
  // never let them re-enter — they are zeroed and pinned below).
  let mut cost = vec![Rat::zero(); total];
  for (j, cj) in c.iter().enumerate() {
    cost[j] = cj.clone();
  }
  // Forbid artificials from re-entering by giving them no reduced-cost appeal:
  // we restrict the entering search to columns `0..art_base`.
  match run_simplex_restricted(
    &mut tab, &mut basis, &cost, total, art_base, rhs_col,
  ) {
    SimplexOutcome::Optimal(_) => {}
    SimplexOutcome::Unbounded => return LpResult::Unbounded,
  }

  // Read the structural variable values from the final basis.
  let mut x = vec![Rat::zero(); n];
  for i in 0..k {
    if basis[i] < n {
      x[basis[i]] = tab[i][rhs_col].clone();
    }
  }
  LpResult::Optimal(x)
}

enum SimplexOutcome {
  Optimal(Rat),
  Unbounded,
}

fn run_simplex(
  tab: &mut [Vec<Rat>],
  basis: &mut [usize],
  cost: &[Rat],
  total: usize,
  rhs_col: usize,
) -> SimplexOutcome {
  run_simplex_restricted(tab, basis, cost, total, total, rhs_col)
}

/// Primal simplex with Bland's rule. Only columns `0..entering_limit` are
/// considered for entering the basis (used to exclude artificial columns in
/// Phase II). Returns the optimal objective value or reports unboundedness.
fn run_simplex_restricted(
  tab: &mut [Vec<Rat>],
  basis: &mut [usize],
  cost: &[Rat],
  total: usize,
  entering_limit: usize,
  rhs_col: usize,
) -> SimplexOutcome {
  let k = tab.len();
  // Dantzig's rule matches wolframscript's vertex choice among multiple
  // optima, but can cycle on degenerate problems; after a generous iteration
  // budget we switch to Bland's rule, which is guaranteed to terminate.
  let bland_after = 20 * (total + k + 1);
  let mut iter = 0usize;
  loop {
    let use_bland = iter >= bland_after;
    iter += 1;
    // Reduced cost of column j: cost[j] - sum_i cost[basis[i]] * tab[i][j].
    // Dantzig's rule: pick the column with the most negative reduced cost,
    // breaking ties toward the lowest index. This matches the vertex
    // wolframscript reports when a problem has multiple optima. Under Bland's
    // fallback we instead take the first column with a negative reduced cost.
    let mut entering = None;
    let mut best_rc: Option<Rat> = None;
    for j in 0..entering_limit {
      let mut rc = cost[j].clone();
      for i in 0..k {
        let cb = &cost[basis[i]];
        if !cb.is_zero() {
          rc = rc.sub(&cb.mul(&tab[i][j]));
        }
      }
      if rc.is_negative() {
        if use_bland {
          entering = Some(j);
          break;
        }
        if best_rc
          .as_ref()
          .is_none_or(|b| rc.cmp(b) == std::cmp::Ordering::Less)
        {
          best_rc = Some(rc);
          entering = Some(j);
        }
      }
    }
    let Some(col) = entering else {
      // Optimal: compute the objective value.
      let mut obj = Rat::zero();
      for i in 0..k {
        let cb = &cost[basis[i]];
        if !cb.is_zero() {
          obj = obj.add(&cb.mul(&tab[i][rhs_col]));
        }
      }
      return SimplexOutcome::Optimal(obj);
    };

    // Ratio test: minimize rhs_i / tab[i][col] over rows with tab[i][col] > 0.
    // Bland's rule breaks ties by the smallest leaving basic-variable index.
    let mut leaving: Option<usize> = None;
    let mut best_ratio: Option<Rat> = None;
    for i in 0..k {
      if tab[i][col].is_positive() {
        let ratio = tab[i][rhs_col].div(&tab[i][col]);
        let take = match &best_ratio {
          None => true,
          Some(br) => match ratio.cmp(br) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Equal => basis[i] < basis[leaving.unwrap()],
            std::cmp::Ordering::Greater => false,
          },
        };
        if take {
          best_ratio = Some(ratio);
          leaving = Some(i);
        }
      }
    }
    let Some(row) = leaving else {
      return SimplexOutcome::Unbounded;
    };
    pivot(tab, basis, row, col, rhs_col);
    let _ = total;
  }
}

/// Gauss-Jordan pivot on `tab[row][col]`, updating the basis.
fn pivot(
  tab: &mut [Vec<Rat>],
  basis: &mut [usize],
  row: usize,
  col: usize,
  rhs_col: usize,
) {
  let width = rhs_col + 1;
  let piv = tab[row][col].clone();
  for entry in tab[row].iter_mut().take(width) {
    *entry = entry.div(&piv);
  }
  let pivot_row = tab[row].clone();
  for (i, tab_i) in tab.iter_mut().enumerate() {
    if i == row {
      continue;
    }
    let factor = tab_i[col].clone();
    if factor.is_zero() {
      continue;
    }
    for (j, entry) in tab_i.iter_mut().enumerate().take(width) {
      *entry = entry.sub(&factor.mul(&pivot_row[j]));
    }
  }
  basis[row] = col;
}
