//! CoordinateBoundsArray / CoordinateBoundingBoxArray — grids of coordinate
//! tuples over a rectangular region, sharing one engine. The two functions
//! differ only in how the region is written: per-dimension `{min, max}`
//! pairs versus the two corner points `{mins, maxs}`.

use crate::InterpreterError;
use crate::syntax::{Expr, UnaryOperator, expr_to_output};

/// One coordinate value: an exact rational (p/q, q > 0) or a machine real.
/// Grids over exact bounds with exact steps stay exact (`Into[2]` of a unit
/// range gives 0, 1/2, 1); any real anywhere in a dimension makes that whole
/// dimension real, like wolframscript.
#[derive(Clone, Copy)]
enum Num {
  Exact(i128, i128),
  Real(f64),
}

impl Num {
  fn to_f64(self) -> f64 {
    match self {
      Num::Exact(p, q) => p as f64 / q as f64,
      Num::Real(v) => v,
    }
  }

  fn is_positive(self) -> bool {
    match self {
      Num::Exact(p, _) => p > 0,
      Num::Real(v) => v > 0.0,
    }
  }

  fn to_expr(self) -> Expr {
    match self {
      Num::Exact(0, _) => Expr::Integer(0),
      Num::Exact(p, q) => {
        let g = gcd(p.abs(), q);
        let (p, q) = (p / g, q / g);
        if q == 1 {
          Expr::Integer(p)
        } else {
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(p), Expr::Integer(q)].into(),
          }
        }
      }
      Num::Real(v) => Expr::Real(v),
    }
  }
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
  while b != 0 {
    (a, b) = (b, a % b);
  }
  a.max(1)
}

/// Combine two numbers with an exact rational operation, falling back to f64
/// when either side is real or the exact arithmetic overflows.
fn combine(
  a: Num,
  b: Num,
  exact: impl Fn(i128, i128, i128, i128) -> Option<(i128, i128)>,
  real: impl Fn(f64, f64) -> f64,
) -> Num {
  if let (Num::Exact(p1, q1), Num::Exact(p2, q2)) = (a, b)
    && let Some((p, q)) = exact(p1, q1, p2, q2)
  {
    return Num::Exact(p, q);
  }
  Num::Real(real(a.to_f64(), b.to_f64()))
}

fn add(a: Num, b: Num) -> Num {
  combine(
    a,
    b,
    |p1, q1, p2, q2| {
      Some((
        p1.checked_mul(q2)?.checked_add(p2.checked_mul(q1)?)?,
        q1 * q2,
      ))
    },
    |x, y| x + y,
  )
}

fn sub(a: Num, b: Num) -> Num {
  combine(
    a,
    b,
    |p1, q1, p2, q2| {
      Some((
        p1.checked_mul(q2)?.checked_sub(p2.checked_mul(q1)?)?,
        q1 * q2,
      ))
    },
    |x, y| x - y,
  )
}

fn div_int(a: Num, n: i128) -> Num {
  match a {
    Num::Exact(p, q) => match q.checked_mul(n) {
      Some(qn) if qn != 0 => {
        let (p, q) = if qn < 0 { (-p, -qn) } else { (p, qn) };
        Num::Exact(p, q)
      }
      _ => Num::Real(a.to_f64() / n as f64),
    },
    Num::Real(v) => Num::Real(v / n as f64),
  }
}

/// Euclidean remainder a mod b (b > 0): the offset an explicit third
/// argument shifts the grid by is taken modulo the step.
fn rem_euclid(a: Num, b: Num) -> Num {
  combine(
    a,
    b,
    |p1, q1, p2, q2| {
      let num = p1.checked_mul(q2)?;
      let den = p2.checked_mul(q1)?;
      if den <= 0 {
        return None;
      }
      Some((num.rem_euclid(den), q1 * q2))
    },
    |x, y| x.rem_euclid(y),
  )
}

fn leq(a: Num, b: Num) -> bool {
  if let (Num::Exact(p1, q1), Num::Exact(p2, q2)) = (a, b)
    && let (Some(l), Some(r)) = (p1.checked_mul(q2), p2.checked_mul(q1))
  {
    return l <= r;
  }
  a.to_f64() <= b.to_f64()
}

/// Parse a numeric scalar (Integer, Rational, Real) to a `Num`.
fn to_num(expr: &Expr) -> Option<Num> {
  match expr {
    Expr::Integer(n) => Some(Num::Exact(*n, 1)),
    Expr::Real(v) if v.is_finite() => Some(Num::Real(*v)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
          Some(Num::Exact(*p * q.signum(), q.abs()))
        }
        _ => None,
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => to_num(operand).map(|v| match v {
      Num::Exact(p, q) => Num::Exact(-p, q),
      Num::Real(x) => Num::Real(-x),
    }),
    _ => None,
  }
}

/// One dimension's division spec: an explicit step or `Into[n]`.
enum DimSpec {
  Step(Num),
  Into(i128),
}

fn parse_dim_spec(expr: &Expr) -> Option<DimSpec> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Into"
    && args.len() == 1
    && let Expr::Integer(n) = &args[0]
    && *n > 0
  {
    return Some(DimSpec::Into(*n));
  }
  to_num(expr).map(DimSpec::Step)
}

/// The shared engine. `name` selects the message vocabulary; `ranges` is the
/// per-dimension list of (min, max). Returns None when the spec/offset
/// arguments don't parse, which leaves the call silently unevaluated
/// (wolframscript emits no message for e.g. a string step).
fn build_array(
  ranges: &[(Num, Num)],
  spec: Option<&Expr>,
  offsets: Option<&Expr>,
  emit_offs: &dyn Fn(&Expr),
) -> Option<Expr> {
  let k = ranges.len();
  // Per-dimension specs: a single spec applies to every dimension.
  let specs: Vec<DimSpec> = match spec {
    None => (0..k).map(|_| DimSpec::Step(Num::Exact(1, 1))).collect(),
    Some(Expr::List(items)) if items.len() == k => items
      .iter()
      .map(parse_dim_spec)
      .collect::<Option<Vec<_>>>()?,
    Some(e) => {
      let s = parse_dim_spec(e)?;
      match s {
        DimSpec::Step(v) => (0..k).map(|_| DimSpec::Step(v)).collect(),
        DimSpec::Into(n) => (0..k).map(|_| DimSpec::Into(n)).collect(),
      }
    }
  };
  let steps: Vec<Num> = specs
    .iter()
    .zip(ranges)
    .map(|(s, &(lo, hi))| match s {
      DimSpec::Step(v) => *v,
      DimSpec::Into(n) => div_int(sub(hi, lo), *n),
    })
    .collect();

  // Offsets shift each dimension by (offset mod step), relative to the
  // lower bound. Invalid offsets emit ::offs and stay unevaluated.
  let offs: Vec<Num> = match offsets {
    None => (0..k).map(|_| Num::Exact(0, 1)).collect(),
    Some(Expr::List(items)) if items.len() == k => {
      match items.iter().map(to_num).collect::<Option<Vec<_>>>() {
        Some(v) => v,
        None => {
          emit_offs(offsets.unwrap());
          return None;
        }
      }
    }
    Some(e) => match to_num(e) {
      Some(v) => (0..k).map(|_| v).collect(),
      None => {
        emit_offs(e);
        return None;
      }
    },
  };

  // Values per dimension: min + (off mod step) + k*step, up to max.
  // A non-positive step gives an empty grid ({}), like wolframscript.
  let dim_values: Vec<Vec<Num>> = ranges
    .iter()
    .zip(steps.iter().zip(&offs))
    .map(|(&(lo, hi), (&step, &off))| {
      let mut vals = Vec::new();
      if step.is_positive() {
        let mut v = add(lo, rem_euclid(off, step));
        while leq(v, hi) {
          vals.push(v);
          v = add(v, step);
        }
      }
      vals
    })
    .collect();

  fn build(dim_values: &[Vec<Num>], prefix: &mut Vec<Num>) -> Expr {
    if prefix.len() == dim_values.len() {
      return Expr::List(
        prefix
          .iter()
          .map(|v| v.to_expr())
          .collect::<Vec<_>>()
          .into(),
      );
    }
    let items: Vec<Expr> = dim_values[prefix.len()]
      .iter()
      .map(|&v| {
        prefix.push(v);
        let e = build(dim_values, prefix);
        prefix.pop();
        e
      })
      .collect();
    Expr::List(items.into())
  }
  Some(build(&dim_values, &mut Vec::new()))
}

/// CoordinateBoundsArray[{{x1min, x1max}, …}, spec, offsets]
pub fn coordinate_bounds_array_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "CoordinateBoundsArray".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.is_empty() || args.len() > 3 {
    return unevaluated();
  }
  let ranges: Option<Vec<(Num, Num)>> = match &args[0] {
    Expr::List(pairs) if !pairs.is_empty() => pairs
      .iter()
      .map(|p| match p {
        Expr::List(mm) if mm.len() == 2 => {
          Some((to_num(&mm[0])?, to_num(&mm[1])?))
        }
        _ => None,
      })
      .collect(),
    _ => None,
  };
  let Some(ranges) = ranges else {
    crate::emit_message(&format!(
      "CoordinateBoundsArray::bound: Invalid bounds specification {}.",
      expr_to_output(&args[0])
    ));
    return unevaluated();
  };
  match build_array(&ranges, args.get(1), args.get(2), &|off| {
    crate::emit_message(&format!(
      "CoordinateBoundsArray::offs: Invalid offset specification {}.",
      expr_to_output(off)
    ));
  }) {
    Some(result) => Ok(result),
    None => unevaluated(),
  }
}

/// CoordinateBoundingBoxArray[{mins, maxs}, spec, offsets]
pub fn coordinate_bounding_box_array_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "CoordinateBoundingBoxArray".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.is_empty() || args.len() > 3 {
    return unevaluated();
  }
  let ranges: Option<Vec<(Num, Num)>> = match &args[0] {
    Expr::List(corners) if corners.len() == 2 => {
      match (&corners[0], &corners[1]) {
        (Expr::List(mins), Expr::List(maxs))
          if mins.len() == maxs.len() && !mins.is_empty() =>
        {
          mins
            .iter()
            .zip(maxs.iter())
            .map(|(lo, hi)| Some((to_num(lo)?, to_num(hi)?)))
            .collect()
        }
        _ => None,
      }
    }
    _ => None,
  };
  let Some(ranges) = ranges else {
    crate::emit_message(&format!(
      "CoordinateBoundingBoxArray::bbox: Invalid bounding box specification {}.",
      expr_to_output(&args[0])
    ));
    return unevaluated();
  };
  match build_array(&ranges, args.get(1), args.get(2), &|off| {
    crate::emit_message(&format!(
      "CoordinateBoundingBoxArray::offs: Invalid offset specification {}.",
      expr_to_output(off)
    ));
  }) {
    Some(result) => Ok(result),
    None => unevaluated(),
  }
}
