//! `PiecewiseExpand` of the step-like functions of an argument that is
//! linear in one variable:
//!
//!  * the rounding functions (`Floor`, `Ceiling`, `Round`, `IntegerPart`,
//!    `FractionalPart`, `Quotient`) and the periodic waves (`SawtoothWave`,
//!    `SquareWave`, `TriangleWave`), when the assumptions confine the
//!    variable to a bounded interval, so they take finitely many pieces;
//!  * `UnitBox` and `UnitTriangle`, with or without such an interval.
//!
//!   PiecewiseExpand[Floor[x, 2], 0 < x < 4]
//!     → Piecewise[{{2, x >= 2}}, 0]
//!
//! As in Wolfram, pieces with equal values are joined with `Or`, the pieces
//! are listed in canonical order of their values, a zero piece becomes the
//! default (otherwise the last piece does), and a bound that coincides with
//! the assumed interval is left out of a piece's condition.

use crate::InterpreterError;
use crate::helpers::call;
use crate::syntax::{ComparisonOp, Expr};

/// An exact rational number with a positive denominator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Q {
  n: i128,
  d: i128,
}

fn gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    (a, b) = (b, a % b);
  }
  a
}

impl Q {
  fn new(n: i128, d: i128) -> Option<Q> {
    if d == 0 {
      return None;
    }
    let g = gcd(n, d).max(1);
    let s = if d < 0 { -1 } else { 1 };
    Some(Q {
      n: s * n / g,
      d: s * d / g,
    })
  }
  fn int(n: i128) -> Q {
    Q { n, d: 1 }
  }
  fn add(self, o: Q) -> Option<Q> {
    Q::new(
      self
        .n
        .checked_mul(o.d)?
        .checked_add(o.n.checked_mul(self.d)?)?,
      self.d.checked_mul(o.d)?,
    )
  }
  fn sub(self, o: Q) -> Option<Q> {
    self.add(Q { n: -o.n, d: o.d })
  }
  fn mul(self, o: Q) -> Option<Q> {
    Q::new(self.n.checked_mul(o.n)?, self.d.checked_mul(o.d)?)
  }
  fn div(self, o: Q) -> Option<Q> {
    Q::new(self.n.checked_mul(o.d)?, self.d.checked_mul(o.n)?)
  }
  fn floor(self) -> i128 {
    self.n.div_euclid(self.d)
  }
  fn ceil(self) -> i128 {
    -(-self.n).div_euclid(self.d)
  }
  fn lt(self, o: Q) -> bool {
    self.n * o.d < o.n * self.d
  }
  fn to_expr(self) -> Expr {
    if self.d == 1 {
      Expr::Integer(self.n)
    } else {
      call(
        "Rational",
        vec![Expr::Integer(self.n), Expr::Integer(self.d)],
      )
    }
  }
  fn from_expr(e: &Expr) -> Option<Q> {
    match e {
      Expr::Integer(n) => Some(Q::int(*n)),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        match (&args[0], &args[1]) {
          (Expr::Integer(n), Expr::Integer(d)) => Q::new(*n, *d),
          _ => None,
        }
      }
      _ => None,
    }
  }
}

fn same(a: &Expr, b: &Expr) -> bool {
  crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
}

fn eval(e: &Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(e)
}

/// A bound of the assumed interval: its value and whether it is attained.
#[derive(Clone, Copy)]
struct Bound {
  v: Q,
  closed: bool,
}

/// The variable and the bounded interval the assumption confines it to.
fn interval_of(assum: &Expr) -> Option<(Expr, Bound, Bound)> {
  let mut var: Option<Expr> = None;
  let mut lo: Option<Bound> = None;
  let mut hi: Option<Bound> = None;
  let mut parts = vec![assum];
  let mut i = 0;
  while i < parts.len() {
    let p = parts[i];
    i += 1;
    match p {
      Expr::FunctionCall { name, args } if name == "And" => {
        parts.extend(args.iter());
      }
      Expr::FunctionCall { name, args }
        if name == "Element"
          && args.len() == 2
          && matches!(&args[1], Expr::Identifier(d) if d == "Reals") => {}
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() == operators.len() + 1 => {
        for (k, op) in operators.iter().enumerate() {
          let (a, b) = (&operands[k], &operands[k + 1]);
          // Normalise to `left < right` / `left <= right`.
          let (l, r, closed) = match op {
            ComparisonOp::Less => (a, b, false),
            ComparisonOp::LessEqual => (a, b, true),
            ComparisonOp::Greater => (b, a, false),
            ComparisonOp::GreaterEqual => (b, a, true),
            _ => return None,
          };
          if let Some(c) = Q::from_expr(r) {
            // l (<|<=) c: an upper bound on the variable l.
            if Q::from_expr(l).is_some() {
              continue;
            }
            if !same(var.get_or_insert_with(|| l.clone()), l) {
              return None;
            }
            hi = Some(Bound { v: c, closed });
          } else {
            let c = Q::from_expr(l)?;
            if !same(var.get_or_insert_with(|| r.clone()), r) {
              return None;
            }
            lo = Some(Bound { v: c, closed });
          }
        }
      }
      _ => return None,
    }
  }
  let (var, lo, hi) = (var?, lo?, hi?);
  if !matches!(var, Expr::Identifier(_)) || !lo.v.lt(hi.v) {
    return None;
  }
  Some((var, lo, hi))
}

/// `arg == a var + b` with rational a != 0 and b.
fn linear_coefficients(arg: &Expr, var: &Expr) -> Option<(Q, Q)> {
  let a = eval(&call("D", vec![arg.clone(), var.clone()])).ok()?;
  let a = Q::from_expr(&a)?;
  if a.n == 0 {
    return None;
  }
  let b = eval(&call(
    "ReplaceAll",
    vec![
      arg.clone(),
      Expr::Rule {
        pattern: Box::new(var.clone()),
        replacement: Box::new(Expr::Integer(0)),
      },
    ],
  ))
  .ok()?;
  Some((a, Q::from_expr(&b)?))
}

#[derive(Clone, Copy, PartialEq)]
enum Step {
  Floor,
  Ceiling,
  Round,
  IntegerPart,
}

impl Step {
  fn value(self, u: Q) -> i128 {
    match self {
      Step::Floor => u.floor(),
      Step::Ceiling => u.ceil(),
      Step::IntegerPart => {
        if u.n >= 0 {
          u.floor()
        } else {
          u.ceil()
        }
      }
      Step::Round => {
        let f = u.floor();
        let frac = u.sub(Q::int(f)).unwrap_or(Q::int(0));
        let half = Q { n: 1, d: 2 };
        if frac.lt(half) {
          f
        } else if half.lt(frac) {
          f + 1
        } else if f % 2 == 0 {
          f
        } else {
          f + 1
        }
      }
    }
  }
}

/// How a piece's value follows from its integer step `k`.
#[derive(Clone, Copy)]
enum Value {
  /// `k scale` (`Floor[x, a] == a Floor[x/a]`).
  Scaled,
  /// the bare `k` (`Quotient[m, n] == Floor[m/n]`).
  Integer,
  /// `arg - k` (FractionalPart, SawtoothWave).
  Fractional,
  /// `±1` by the parity of k (SquareWave).
  Square,
  /// `±4 (arg - k/2)` by the parity of k (TriangleWave).
  Triangle,
}

/// The step function as `value(k)` with `k = step(u)` and
/// `u = arg/scale + shift`.
struct Plan {
  step: Step,
  arg: Expr,
  scale: Q,
  shift: Q,
  value: Value,
}

fn plan_of(expr: &Expr) -> Option<Plan> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  let plan = |step, scale: Q, shift: Q, value| {
    (scale.n != 0).then(|| Plan {
      step,
      arg: args[0].clone(),
      scale,
      shift,
      value,
    })
  };
  let zero = Q::int(0);
  let half = Q { n: 1, d: 2 };
  let scale_arg = || match args.len() {
    1 => Some(Q::int(1)),
    2 => Q::from_expr(&args[1]),
    _ => None,
  };
  match name.as_str() {
    "Floor" => plan(Step::Floor, scale_arg()?, zero, Value::Scaled),
    "Ceiling" => plan(Step::Ceiling, scale_arg()?, zero, Value::Scaled),
    "Round" => plan(Step::Round, scale_arg()?, zero, Value::Scaled),
    "IntegerPart" if args.len() == 1 => {
      plan(Step::IntegerPart, Q::int(1), zero, Value::Scaled)
    }
    "FractionalPart" if args.len() == 1 => {
      plan(Step::IntegerPart, Q::int(1), zero, Value::Fractional)
    }
    // Quotient[m, n] == Floor[m/n]; the value is the integer itself.
    "Quotient" if args.len() == 2 => {
      plan(Step::Floor, Q::from_expr(&args[1])?, zero, Value::Integer)
    }
    // The unit-period waves: SawtoothWave[x] == x - Floor[x],
    // SquareWave[x] == (-1)^Floor[2 x], and TriangleWave[x] rising with
    // slope 4 through 0 on [-1/4, 1/4) and falling on [1/4, 3/4).
    "SawtoothWave" if args.len() == 1 => {
      plan(Step::Floor, Q::int(1), zero, Value::Fractional)
    }
    "SquareWave" if args.len() == 1 => {
      plan(Step::Floor, half, zero, Value::Square)
    }
    "TriangleWave" if args.len() == 1 => {
      plan(Step::Floor, half, half, Value::Triangle)
    }
    _ => None,
  }
}

/// `PiecewiseExpand[f[a x + b], lo < x < hi]` for the step-like functions.
pub fn expand_rounding(
  expr: &Expr,
  assum: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  if let Some(result) = expand_unit_shape(expr, Some(assum)) {
    return Ok(Some(eval(&result)?));
  }
  let Some(plan) = plan_of(expr) else {
    return Ok(None);
  };
  let Some((var, lo, hi)) = interval_of(assum) else {
    return Ok(None);
  };
  let Some((a, b)) = linear_coefficients(&plan.arg, &var) else {
    return Ok(None);
  };
  match build(&plan, &var, lo, hi, a, b) {
    Some(pw) => Ok(Some(eval(&pw)?)),
    None => Ok(None),
  }
}

/// `PiecewiseExpand[UnitBox[…]]` / `UnitTriangle[…]` without assumptions.
pub fn expand_unit_shape_plain(
  expr: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  match expand_unit_shape(expr, None) {
    Some(e) => Ok(Some(eval(&e)?)),
    None => Ok(None),
  }
}

/// A piece of a piecewise function of `x`: its value on the interval from
/// `lo` to `hi` (`None` for an infinite end; the flag tells whether the end
/// is attained).
struct Piece {
  value: Expr,
  lo: Option<(Q, bool)>,
  hi: Option<(Q, bool)>,
}

/// `UnitBox[t]` and `UnitTriangle[t]` for `t` linear in one variable, and
/// their products over several arguments.
fn expand_unit_shape(expr: &Expr, assum: Option<&Expr>) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if !matches!(name.as_str(), "UnitBox" | "UnitTriangle") || args.is_empty() {
    return None;
  }
  if args.len() > 1 {
    // The product of the one-dimensional shapes; only when each factor
    // collapses to a single expression.
    let mut factors = Vec::new();
    for a in args {
      let f = expand_unit_shape(&call(name, vec![a.clone()]), assum)?;
      let f = eval(&f).ok()?;
      if matches!(&f, Expr::FunctionCall { name, .. } if name == "Piecewise") {
        return None;
      }
      factors.push(f);
    }
    return Some(call("Factor", vec![call("Times", factors)]));
  }
  let arg = &args[0];
  let var = match assum.and_then(interval_of) {
    Some((v, _, _)) => v,
    None => Expr::Identifier(single_symbol(arg)?),
  };
  let (a, b) = linear_coefficients(arg, &var)?;
  let bounds = assum.and_then(interval_of).map(|(_, lo, hi)| (lo, hi));
  // Pieces in t, then mapped to x = (t - b)/a.
  let t_pieces: Vec<(Expr, Q, bool, Q, bool)> = if name == "UnitBox" {
    let h = Q { n: 1, d: 2 };
    vec![(Expr::Integer(1), Q { n: -1, d: 2 }, true, h, true)]
  } else {
    vec![
      (
        eval(&call(
          "Plus",
          vec![
            Expr::Integer(1),
            call("Times", vec![Expr::Integer(-1), arg.clone()]),
          ],
        ))
        .ok()?,
        Q::int(0),
        true,
        Q::int(1),
        true,
      ),
      (
        eval(&call("Plus", vec![Expr::Integer(1), arg.clone()])).ok()?,
        Q::int(-1),
        true,
        Q::int(0),
        false,
      ),
    ]
  };
  let x_of = |t: Q| t.sub(b)?.div(a);
  let mut pieces = Vec::new();
  for (value, tl, tlc, tr, trc) in t_pieces {
    let (l, lc, r, rc) = if a.n > 0 {
      (x_of(tl)?, tlc, x_of(tr)?, trc)
    } else {
      (x_of(tr)?, trc, x_of(tl)?, tlc)
    };
    pieces.push(Piece {
      value,
      lo: Some((l, lc)),
      hi: Some((r, rc)),
    });
  }
  // The zero regions outside.
  let first_lo = pieces
    .iter()
    .filter_map(|p| p.lo)
    .min_by(|x, y| cmp_q(x.0, y.0))?;
  let last_hi = pieces
    .iter()
    .filter_map(|p| p.hi)
    .max_by(|x, y| cmp_q(x.0, y.0))?;
  pieces.push(Piece {
    value: Expr::Integer(0),
    lo: None,
    hi: Some((first_lo.0, !first_lo.1)),
  });
  pieces.push(Piece {
    value: Expr::Integer(0),
    lo: Some((last_hi.0, !last_hi.1)),
    hi: None,
  });
  assemble(pieces, &var, bounds)
}

fn cmp_q(x: Q, y: Q) -> std::cmp::Ordering {
  (x.n * y.d).cmp(&(y.n * x.d))
}

/// The only symbol in `e`, if it has exactly one.
fn single_symbol(e: &Expr) -> Option<String> {
  fn walk(e: &Expr, out: &mut Vec<String>) {
    match e {
      Expr::Identifier(n) => {
        if !matches!(n.as_str(), "Pi" | "E" | "I") && !out.contains(n) {
          out.push(n.clone());
        }
      }
      Expr::FunctionCall { args, .. } | Expr::List(args) => {
        args.iter().for_each(|a| walk(a, out));
      }
      Expr::BinaryOp { left, right, .. } => {
        walk(left, out);
        walk(right, out);
      }
      Expr::UnaryOp { operand, .. } => walk(operand, out),
      _ => {}
    }
  }
  let mut out = Vec::new();
  walk(e, &mut out);
  (out.len() == 1).then(|| out.remove(0))
}

/// Clip the pieces to the assumed interval (if any), join equal values
/// with `Or`, and build the `Piecewise` (or the single value).
fn assemble(
  pieces: Vec<Piece>,
  var: &Expr,
  bounds: Option<(Bound, Bound)>,
) -> Option<Expr> {
  // (value, conditions) groups in order of first appearance.
  let mut groups: Vec<(Expr, Vec<Expr>)> = Vec::new();
  for p in pieces {
    let (mut lo, mut hi) = (p.lo, p.hi);
    if let Some((blo, bhi)) = bounds {
      // Intersect with the assumed interval; a bound that coincides with
      // the assumed one (or is cut by it) is implied and dropped.
      lo = match lo {
        Some((l, lc)) if cmp_q(l, blo.v).is_gt() => Some((l, lc)),
        Some((l, lc)) if l == blo.v && (lc != blo.closed) && !lc => {
          Some((l, lc))
        }
        _ => None,
      };
      hi = match hi {
        Some((r, rc)) if cmp_q(r, bhi.v).is_lt() => Some((r, rc)),
        Some((r, rc)) if r == bhi.v && (rc != bhi.closed) && !rc => {
          Some((r, rc))
        }
        _ => None,
      };
      // Empty after clipping?
      let eff_lo = lo.unwrap_or((blo.v, blo.closed));
      let eff_hi = hi.unwrap_or((bhi.v, bhi.closed));
      let order = cmp_q(eff_lo.0, eff_hi.0);
      if order.is_gt() || (order.is_eq() && !(eff_lo.1 && eff_hi.1)) {
        continue;
      }
    }
    let le = |c: bool| {
      if c {
        ComparisonOp::LessEqual
      } else {
        ComparisonOp::Less
      }
    };
    let cond = match (lo, hi) {
      (None, None) => Expr::Identifier("True".into()),
      (Some((l, lc)), None) => Expr::Comparison {
        operands: vec![var.clone(), l.to_expr()],
        operators: vec![if lc {
          ComparisonOp::GreaterEqual
        } else {
          ComparisonOp::Greater
        }],
      },
      (None, Some((r, rc))) => Expr::Comparison {
        operands: vec![var.clone(), r.to_expr()],
        operators: vec![le(rc)],
      },
      (Some((l, _)), Some((r, _))) if l == r => Expr::Comparison {
        operands: vec![var.clone(), l.to_expr()],
        operators: vec![ComparisonOp::Equal],
      },
      (Some((l, lc)), Some((r, rc))) => Expr::Comparison {
        operands: vec![l.to_expr(), var.clone(), r.to_expr()],
        operators: vec![le(lc), le(rc)],
      },
    };
    let key = crate::syntax::expr_to_string(&p.value);
    match groups
      .iter_mut()
      .find(|(v, _)| crate::syntax::expr_to_string(v) == key)
    {
      Some((_, conds)) => conds.push(cond),
      None => groups.push((p.value, vec![cond])),
    }
  }
  if groups.is_empty() {
    return None;
  }
  if groups.len() == 1 {
    return Some(groups.pop()?.0);
  }
  let mut pieces: Vec<(Expr, Expr)> = groups
    .into_iter()
    .map(|(v, mut conds)| {
      let cond = if conds.len() == 1 {
        conds.pop().unwrap()
      } else {
        call("Or", conds)
      };
      (v, cond)
    })
    .collect();
  // Canonical order of the values.
  let values = Expr::List(pieces.iter().map(|p| p.0.clone()).collect());
  let order = eval(&call("Ordering", vec![values])).ok()?;
  let Expr::List(order) = &order else {
    return None;
  };
  let mut sorted: Vec<(Expr, Expr)> = Vec::new();
  for o in order {
    let Expr::Integer(i) = o else { return None };
    sorted.push(pieces.get(usize::try_from(*i - 1).ok()?)?.clone());
  }
  pieces.clear();
  let default =
    match sorted.iter().position(|p| matches!(p.0, Expr::Integer(0))) {
      Some(i) => sorted.remove(i).0,
      None => sorted.pop()?.0,
    };
  let clauses = Expr::List(
    sorted
      .into_iter()
      .map(|(v, c)| Expr::List(vec![v, c].into()))
      .collect(),
  );
  Some(call("Piecewise", vec![clauses, default]))
}

fn build(
  plan: &Plan,
  var: &Expr,
  lo: Bound,
  hi: Bound,
  a: Q,
  b: Q,
) -> Option<Expr> {
  // u = arg / scale + shift = a' x + b'
  let a2 = a.div(plan.scale)?;
  let b2 = b.div(plan.scale)?.add(plan.shift)?;
  let u_of = |x: Q| a2.mul(x)?.add(b2);
  let x_of = |u: Q| u.sub(b2)?.div(a2);
  let (u_lo, u_hi, lo_closed, hi_closed) = if a2.n > 0 {
    (u_of(lo.v)?, u_of(hi.v)?, lo.closed, hi.closed)
  } else {
    (u_of(hi.v)?, u_of(lo.v)?, hi.closed, lo.closed)
  };
  // Jump points strictly inside (u_lo, u_hi): integers, or half-integers
  // for Round.
  let round = plan.step == Step::Round;
  let step_size = if round { 2 } else { 1 };
  let first = u_lo.mul(Q::int(step_size))?.floor() + 1;
  let last = u_hi.mul(Q::int(step_size))?.ceil() - 1;
  if last - first > 1000 {
    return None;
  }
  let mut points: Vec<Q> = Vec::new();
  for k in first..=last {
    if round && k % 2 == 0 {
      continue; // integers are not jumps of Round
    }
    points.push(Q::new(k, step_size)?);
  }
  // Elements in u order: closed points and open gaps between them.
  struct Elem {
    s: Q,
    sc: bool,
    e: Q,
    ec: bool,
    sample: Q,
  }
  let half = Q { n: 1, d: 2 };
  let point = |p: Q| Elem {
    s: p,
    sc: true,
    e: p,
    ec: true,
    sample: p,
  };
  let mut elems: Vec<Elem> = Vec::new();
  if lo_closed {
    elems.push(point(u_lo));
  }
  let mut prev = u_lo;
  for p in points.iter().copied().chain(std::iter::once(u_hi)) {
    elems.push(Elem {
      s: prev,
      sc: false,
      e: p,
      ec: false,
      sample: prev.add(p)?.mul(half)?,
    });
    if p != u_hi || hi_closed {
      elems.push(point(p));
    }
    prev = p;
  }
  // Merge runs with equal steps.
  let mut runs: Vec<(i128, Elem)> = Vec::new();
  for el in elems {
    let k = plan.step.value(el.sample);
    match runs.last_mut() {
      Some((rk, run)) if *rk == k => {
        run.e = el.e;
        run.ec = el.ec;
      }
      _ => runs.push((k, el)),
    }
  }

  let value_of = |k: i128| -> Option<Expr> {
    let arg_minus = |c: Q| {
      eval(&call(
        "Plus",
        vec![plan.arg.clone(), c.mul(Q::int(-1))?.to_expr()],
      ))
      .ok()
    };
    Some(match plan.value {
      Value::Scaled => Q::int(k).mul(plan.scale)?.to_expr(),
      Value::Integer => Expr::Integer(k),
      Value::Fractional => arg_minus(Q::int(k))?,
      Value::Square => Expr::Integer(if k.rem_euclid(2) == 0 { 1 } else { -1 }),
      Value::Triangle => {
        let sign = if k.rem_euclid(2) == 0 { 4 } else { -4 };
        let lin = arg_minus(Q::new(k, 2)?)?;
        eval(&call(
          "Factor",
          vec![call("Times", vec![Expr::Integer(sign), lin])],
        ))
        .ok()?
      }
    })
  };

  let mut pieces = Vec::new();
  for (k, run) in &runs {
    let (xl, xlc, xr, xrc) = if a2.n > 0 {
      (x_of(run.s)?, run.sc, x_of(run.e)?, run.ec)
    } else {
      (x_of(run.e)?, run.ec, x_of(run.s)?, run.sc)
    };
    pieces.push(Piece {
      value: value_of(*k)?,
      lo: Some((xl, xlc)),
      hi: Some((xr, xrc)),
    });
  }
  assemble(pieces, var, Some((lo, hi)))
}
