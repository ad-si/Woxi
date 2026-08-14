//! LinearProgramming[c, m, b] / LinearProgramming[c, m, b, l] — exact linear
//! program solver.
//!
//! Minimizes `c · x` subject to the constraints encoded by `m` and `b` and
//! (by default) `x >= 0`. Each constraint row `m[[i]] · x` is compared to
//! `b[[i]]`; when `b[[i]]` is a bare number the relation is `>=`, and when it
//! is a pair `{value, sign}` the sign selects `>=` (1), `==` (0) or `<=` (-1).
//!
//! `l` replaces the default `x >= 0` bounds: a scalar lower bound for every
//! variable, a vector of lower bounds, or a matrix of `{lower, upper}` pairs
//! whose entries may be `Infinity` or `-Infinity`. A finite lower bound is
//! removed by substituting `x = lower + y` with `y >= 0`, a variable with no
//! lower bound is split into the difference of two non-negative ones, and a
//! finite upper bound becomes one more `<=` constraint — so the simplex below
//! only ever sees non-negative variables.
//!
//! The solver is an exact two-phase simplex over `BigInt` rationals. It uses
//! Dantzig's entering rule — which reproduces the vertex wolframscript reports
//! when a problem has multiple optima — and falls back to Bland's rule after a
//! generous iteration budget so it can never cycle on degenerate problems.

#[allow(unused_imports)]
use super::*;
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
    let mut r = Self { num, den };
    r.reduce();
    r
  }
  fn from_int(n: BigInt) -> Self {
    Self {
      num: n,
      den: BigInt::one(),
    }
  }
  fn zero() -> Self {
    Self::from_int(BigInt::zero())
  }
  fn one() -> Self {
    Self::from_int(BigInt::one())
  }
  fn reduce(&mut self) {
    (self.num, self.den) = rat_reduce_bigint(&self.num, &self.den);
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
  fn neg(&self) -> Self {
    Self {
      num: -&self.num,
      den: self.den.clone(),
    }
  }
  fn add(&self, o: &Self) -> Self {
    Self::new(&self.num * &o.den + &o.num * &self.den, &self.den * &o.den)
  }
  fn sub(&self, o: &Self) -> Self {
    Self::new(&self.num * &o.den - &o.num * &self.den, &self.den * &o.den)
  }
  fn mul(&self, o: &Self) -> Self {
    Self::new(&self.num * &o.num, &self.den * &o.den)
  }
  fn div(&self, o: &Self) -> Self {
    Self::new(&self.num * &o.den, &self.den * &o.num)
  }
  fn cmp(&self, o: &Self) -> std::cmp::Ordering {
    // den is always positive, so cross-multiplication preserves the order.
    (&self.num * &o.den).cmp(&(&o.num * &self.den))
  }
  fn to_expr(&self) -> Expr {
    if self.den.is_one() {
      bigint_to_expr(&self.num)
    } else {
      call(
        "Rational",
        vec![bigint_to_expr(&self.num), bigint_to_expr(&self.den)],
      )
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

/// One variable's bound, which may be infinite in either direction.
#[derive(Clone)]
enum Bound {
  Finite(Rat),
  PosInf,
  NegInf,
}

/// Why a bound specification was refused, and which message wolframscript
/// emits for it.
enum BoundError {
  /// `::lprank012` — not a scalar, a vector, or a matrix with two columns.
  Rank,
  /// `::lpdim` — the right shape but the wrong number of variables.
  Dim,
  /// `::lpbd` — an entry that is neither a real number nor +/-Infinity.
  Value,
  /// `::lpsbnn` — a variable whose lower and upper bound are both Infinity.
  BothInfinite,
}

/// A single lower/upper bound entry. Infinities are recognised through the
/// printed form, which is where the several internal shapes of `-Infinity`
/// (`DirectedInfinity`, a negated symbol, `Times[-1, Infinity]`) are already
/// normalised to one token.
fn parse_bound(e: &Expr) -> Option<Bound> {
  if let Some(r) = expr_to_rat(e) {
    return Some(Bound::Finite(r));
  }
  match crate::syntax::format_expr(e, crate::syntax::ExprForm::Output).as_str()
  {
    "Infinity" => Some(Bound::PosInf),
    "-Infinity" => Some(Bound::NegInf),
    _ => None,
  }
}

/// Read the `l` argument of `LinearProgramming` into one `(lower, upper)` pair
/// per variable. Shape is checked before length, and length before the entries,
/// matching the order wolframscript reports the three problems in.
fn parse_bounds(l: &Expr, n: usize) -> Result<Vec<(Bound, Bound)>, BoundError> {
  let entries: Vec<(Bound, Bound)> = match l {
    Expr::List(items) => {
      let rows = items
        .iter()
        .filter(|it| matches!(it, Expr::List(_)))
        .count();
      if rows == items.len() && !items.is_empty() {
        // A matrix: every row must hold exactly a lower and an upper bound.
        if items
          .iter()
          .any(|it| !matches!(it, Expr::List(p) if p.len() == 2))
        {
          return Err(BoundError::Rank);
        }
        if items.len() != n {
          return Err(BoundError::Dim);
        }
        let mut out = Vec::with_capacity(n);
        for it in items {
          let Expr::List(pair) = it else { unreachable!() };
          let (Some(lo), Some(hi)) =
            (parse_bound(&pair[0]), parse_bound(&pair[1]))
          else {
            return Err(BoundError::Value);
          };
          out.push((lo, hi));
        }
        out
      } else if rows == 0 {
        // A vector of lower bounds, with no upper bounds.
        if items.len() != n {
          return Err(BoundError::Dim);
        }
        let mut out = Vec::with_capacity(n);
        for it in items {
          match parse_bound(it) {
            Some(lo) => out.push((lo, Bound::PosInf)),
            None => return Err(BoundError::Value),
          }
        }
        out
      } else {
        return Err(BoundError::Rank);
      }
    }
    // A scalar lower bound shared by every variable.
    scalar => match parse_bound(scalar) {
      Some(lo) => vec![(lo, Bound::PosInf); n],
      None => return Err(BoundError::Value),
    },
  };
  if entries
    .iter()
    .any(|(lo, hi)| matches!((lo, hi), (Bound::PosInf, Bound::PosInf)))
  {
    return Err(BoundError::BothInfinite);
  }
  Ok(entries)
}

/// How an original variable is expressed in the non-negative variables the
/// simplex works with.
enum VarMap {
  /// `x = offset + y[col]`, with `y[col] >= 0`.
  Shifted { col: usize, offset: Rat },
  /// `x = y[pos] - y[neg]`, both non-negative — a variable with no lower bound.
  Free { pos: usize, neg: usize },
}

pub fn linear_programming_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let call = || Ok(unevaluated("LinearProgramming", args));
  let lpdim = || {
    crate::emit_message(
      "LinearProgramming::lpdim: Invalid input: the dimensions of the input \
       vectors or matrices must match.",
    );
  };
  if args.len() != 3 && args.len() != 4 {
    return call();
  }
  let (Expr::List(c_items), Expr::List(m_rows), Expr::List(b_items)) =
    (&args[0], &args[1], &args[2])
  else {
    return call();
  };

  let n = c_items.len();
  let k = m_rows.len();
  if n == 0 || k == 0 {
    return call();
  }
  if b_items.len() != k {
    lpdim();
    return call();
  }

  // Objective coefficients.
  let mut c = Vec::with_capacity(n);
  for ci in c_items {
    match expr_to_rat(ci) {
      Some(r) => c.push(r),
      None => return call(),
    }
  }

  // Constraint matrix.
  let mut a: Vec<Vec<Rat>> = Vec::with_capacity(k);
  for row in m_rows {
    let Expr::List(row_items) = row else {
      return call();
    };
    if row_items.len() != n {
      lpdim();
      return call();
    }
    let mut r = Vec::with_capacity(n);
    for e in row_items {
      match expr_to_rat(e) {
        Some(v) => r.push(v),
        None => return call(),
      }
    }
    a.push(r);
  }

  // Right-hand sides with their relation. A bare number means `>=`; a pair
  // {value, sign} selects the relation from the sign.
  let mut b = Vec::with_capacity(k);
  let mut rel = Vec::with_capacity(k);
  for item in b_items {
    match item {
      Expr::List(pair) if pair.len() == 2 => {
        let (Some(val), Some(sign)) =
          (expr_to_rat(&pair[0]), expr_to_rat(&pair[1]))
        else {
          return call();
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
        None => return call(),
      },
    }
  }

  // Variable bounds. Without `l` every variable is simply non-negative, which
  // the substitution below reproduces as a zero shift.
  let bounds =
    match args.get(3) {
      None => vec![(Bound::Finite(Rat::zero()), Bound::PosInf); n],
      Some(l) => match parse_bounds(l, n) {
        Ok(bounds) => bounds,
        Err(err) => {
          let message = match err {
          BoundError::Rank => format!(
            "LinearProgramming::lprank012: {} must be a scalar, a vector or a \
             matrix with 2 columns.",
            crate::syntax::format_expr(l, crate::syntax::ExprForm::Output)
          ),
          BoundError::Dim => "LinearProgramming::lpdim: Invalid input: the \
                              dimensions of the input vectors or matrices must \
                              match."
            .to_string(),
          BoundError::Value =>
            "LinearProgramming::lpbd: The input that specifies lower/upper \
             bounds contains elements that are not real numbers, Infinity or \
             -Infinity."
              .to_string(),
          BoundError::BothInfinite =>
            "LinearProgramming::lpsbnn: Found lower bound and upper bound both \
             set at Infinity."
              .to_string(),
        };
          crate::emit_message(&message);
          return call();
        }
      },
    };

  // A lower bound of +Infinity, or an upper bound of -Infinity, excludes every
  // real value; there is nothing for the simplex to find.
  let unsatisfiable = bounds
    .iter()
    .any(|(lo, hi)| matches!(lo, Bound::PosInf) || matches!(hi, Bound::NegInf));

  // Substitute the bounds away: shift each finite lower bound to zero, split
  // each variable without one into a difference of two non-negative variables,
  // and turn each finite upper bound into one more `<=` row.
  let mut maps: Vec<VarMap> = Vec::with_capacity(n);
  let mut width = 0usize;
  for (lo, _) in &bounds {
    match lo {
      Bound::Finite(offset) => {
        maps.push(VarMap::Shifted {
          col: width,
          offset: offset.clone(),
        });
        width += 1;
      }
      // A +Infinity lower bound is already known to be unsatisfiable; treat it
      // as free here so the tableau still has consistent dimensions.
      Bound::NegInf | Bound::PosInf => {
        maps.push(VarMap::Free {
          pos: width,
          neg: width + 1,
        });
        width += 2;
      }
    }
  }
  let mut c2 = vec![Rat::zero(); width];
  for (j, map) in maps.iter().enumerate() {
    match map {
      VarMap::Shifted { col, .. } => c2[*col] = c[j].clone(),
      VarMap::Free { pos, neg } => {
        c2[*pos] = c[j].clone();
        c2[*neg] = c[j].neg();
      }
    }
  }
  let mut a2: Vec<Vec<Rat>> = Vec::with_capacity(k + n);
  let mut b2: Vec<Rat> = Vec::with_capacity(k + n);
  let mut rel2: Vec<Rel> = Vec::with_capacity(k + n);
  for i in 0..k {
    let mut row = vec![Rat::zero(); width];
    let mut rhs = b[i].clone();
    for (j, map) in maps.iter().enumerate() {
      match map {
        VarMap::Shifted { col, offset } => {
          row[*col] = a[i][j].clone();
          rhs = rhs.sub(&a[i][j].mul(offset));
        }
        VarMap::Free { pos, neg } => {
          row[*pos] = a[i][j].clone();
          row[*neg] = a[i][j].neg();
        }
      }
    }
    a2.push(row);
    b2.push(rhs);
    rel2.push(rel[i]);
  }
  for (j, map) in maps.iter().enumerate() {
    let Bound::Finite(hi) = &bounds[j].1 else {
      continue;
    };
    let mut row = vec![Rat::zero(); width];
    let rhs = match map {
      VarMap::Shifted { col, offset } => {
        row[*col] = Rat::one();
        hi.sub(offset)
      }
      VarMap::Free { pos, neg } => {
        row[*pos] = Rat::one();
        row[*neg] = Rat::one().neg();
        hi.clone()
      }
    };
    a2.push(row);
    b2.push(rhs);
    rel2.push(Rel::Le);
  }

  let outcome = if unsatisfiable {
    LpResult::Infeasible
  } else {
    solve_simplex(&c2, &a2, &b2, &rel2, width)
  };
  match outcome {
    LpResult::Optimal(y) => {
      let values: Vec<Expr> = maps
        .iter()
        .map(|map| match map {
          VarMap::Shifted { col, offset } => offset.add(&y[*col]).to_expr(),
          VarMap::Free { pos, neg } => y[*pos].sub(&y[*neg]).to_expr(),
        })
        .collect();
      Ok(Expr::List(values.into()))
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
      call()
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
      rows.push(a[i].iter().map(Rat::neg).collect());
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
