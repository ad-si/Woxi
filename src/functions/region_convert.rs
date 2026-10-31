//! `RegionConvert[reg, "Implicit"]` — a region as an `ImplicitRegion` in
//! the formal variables `\[FormalX]`, `\[FormalY]`, `\[FormalZ]`.

use super::*;

/// The formal coordinate symbols of a region of dimension `n` (at most 3).
fn formal_vars(n: usize) -> Option<Vec<Expr>> {
  let names = ["\\[FormalX]", "\\[FormalY]", "\\[FormalZ]"];
  if n == 0 || n > names.len() {
    return None;
  }
  names[..n]
    .iter()
    .map(|s| crate::syntax::string_to_expr(s).ok())
    .collect()
}

fn coords(e: &Expr) -> Option<Vec<Expr>> {
  match e {
    Expr::List(items) if !items.is_empty() => Some(items.to_vec()),
    _ => None,
  }
}

fn eval(e: &Expr) -> Option<Expr> {
  crate::evaluator::evaluate_expr_to_expr(e).ok()
}

fn cmp(operands: Vec<Expr>, op: ComparisonOp) -> Expr {
  let operators = vec![op; operands.len() - 1];
  Expr::Comparison {
    operands,
    operators,
  }
}

fn and_all(parts: Vec<Expr>) -> Expr {
  if parts.len() == 1 {
    return parts.into_iter().next().unwrap();
  }
  Expr::FunctionCall {
    name: "And".to_string(),
    args: parts.into(),
  }
}

fn implicit(condition: &Expr, vars: Vec<Expr>) -> Option<Expr> {
  Some(Expr::FunctionCall {
    name: "ImplicitRegion".to_string(),
    args: vec![eval(condition)?, Expr::List(vars.into())].into(),
  })
}

/// `lo_i <= x_i <= hi_i` for every axis of the box spanned by `p` and `q`.
fn box_region(p: &[Expr], q: &[Expr]) -> Option<Expr> {
  if p.len() != q.len() {
    return None;
  }
  let vars = formal_vars(p.len())?;
  let mut parts = Vec::new();
  for ((a, b), v) in p.iter().zip(q).zip(&vars) {
    let lo = eval(&call("Min", vec![a.clone(), b.clone()]))?;
    let hi = eval(&call("Max", vec![a.clone(), b.clone()]))?;
    parts.push(cmp(vec![lo, v.clone(), hi], ComparisonOp::LessEqual));
  }
  implicit(&and_all(parts), vars)
}

/// `Σ (x_i - c_i)^2 / r_i^2 <= 1` with the denominators cleared:
/// `Σ (x_i - c_i)^2 <= r^2` for a round ball.
fn ball_region(center: &[Expr], radii: &[Expr]) -> Option<Expr> {
  let vars = formal_vars(center.len())?;
  let square = |v: &Expr, c: &Expr| {
    call(
      "Power",
      vec![
        call(
          "Plus",
          vec![v.clone(), call("Times", vec![Expr::Integer(-1), c.clone()])],
        ),
        Expr::Integer(2),
      ],
    )
  };
  let round = radii.windows(2).all(|w| {
    crate::syntax::expr_to_string(&w[0]) == crate::syntax::expr_to_string(&w[1])
  });
  let condition = if round {
    let lhs = call(
      "Plus",
      vars.iter().zip(center).map(|(v, c)| square(v, c)).collect(),
    );
    let r2 = call("Power", vec![radii[0].clone(), Expr::Integer(2)]);
    cmp(vec![lhs, r2], ComparisonOp::LessEqual)
  } else {
    // Each axis term carries the product of the other axes' squared radii.
    let r2: Vec<Expr> = radii
      .iter()
      .map(|r| call("Power", vec![r.clone(), Expr::Integer(2)]))
      .collect();
    let terms: Vec<Expr> = vars
      .iter()
      .zip(center)
      .enumerate()
      .map(|(i, (v, c))| {
        let mut factors: Vec<Expr> = r2
          .iter()
          .enumerate()
          .filter(|(j, _)| *j != i)
          .map(|(_, r)| r.clone())
          .collect();
        factors.push(square(v, c));
        call("Times", factors)
      })
      .collect();
    cmp(
      vec![call("Plus", terms), call("Times", r2)],
      ComparisonOp::LessEqual,
    )
  };
  implicit(&condition, vars)
}

/// The parallelepiped `o + Σ t_i v_i`, `0 <= t_i <= 1`: the coefficients
/// `t = Inverse[{v_1, …}ᵀ] . (x - o)`, each bounded by 0 and 1.
fn parallelepiped_region(origin: &[Expr], vectors: &[Expr]) -> Option<Expr> {
  let n = origin.len();
  if vectors.len() != n {
    return None;
  }
  let vars = formal_vars(n)?;
  let columns = eval(&call(
    "Transpose",
    vec![Expr::List(vectors.to_vec().into())],
  ))?;
  let inverse = eval(&call("Inverse", vec![columns]))?;
  if !matches!(inverse, Expr::List(_)) {
    return None;
  }
  let shifted: Vec<Expr> = vars
    .iter()
    .zip(origin)
    .map(|(v, o)| {
      call(
        "Plus",
        vec![v.clone(), call("Times", vec![Expr::Integer(-1), o.clone()])],
      )
    })
    .collect();
  let t = eval(&call(
    "Expand",
    vec![call("Dot", vec![inverse, Expr::List(shifted.into())])],
  ))?;
  let Expr::List(ref ts) = t else {
    return None;
  };
  let parts = ts
    .iter()
    .map(|ti| {
      cmp(
        vec![Expr::Integer(0), ti.clone(), Expr::Integer(1)],
        ComparisonOp::LessEqual,
      )
    })
    .collect();
  implicit(&and_all(parts), vars)
}

fn implicit_form(region: &Expr) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = region else {
    return None;
  };
  let one = || Expr::Integer(1);
  match (name.as_str(), args.as_slice()) {
    ("ImplicitRegion", [_, _]) => Some(region.clone()),
    ("Rectangle", []) => {
      box_region(&[Expr::Integer(0), Expr::Integer(0)], &[one(), one()])
    }
    ("Rectangle", [p]) => {
      let p = coords(p)?;
      let q: Vec<Expr> = p
        .iter()
        .map(|c| call("Plus", vec![c.clone(), one()]))
        .collect();
      box_region(&p, &q)
    }
    ("Rectangle" | "Cuboid", [p, q]) => box_region(&coords(p)?, &coords(q)?),
    ("Cuboid", []) => box_region(
      &[Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)],
      &[one(), one(), one()],
    ),
    ("Cuboid", [p]) => {
      let p = coords(p)?;
      let q: Vec<Expr> = p
        .iter()
        .map(|c| call("Plus", vec![c.clone(), one()]))
        .collect();
      box_region(&p, &q)
    }
    ("Disk", []) => {
      ball_region(&[Expr::Integer(0), Expr::Integer(0)], &[one(), one()])
    }
    ("Ball", []) => ball_region(
      &[Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)],
      &[one(), one(), one()],
    ),
    ("Disk" | "Ball", [c]) => {
      let c = coords(c)?;
      let r = vec![one(); c.len()];
      ball_region(&c, &r)
    }
    ("Disk" | "Ball", [c, r]) => {
      let c = coords(c)?;
      let r = match r {
        Expr::List(rs) if rs.len() == c.len() => rs.to_vec(),
        Expr::List(_) => return None,
        r => vec![r.clone(); c.len()],
      };
      ball_region(&c, &r)
    }
    ("Parallelogram" | "Parallelepiped", [o, Expr::List(vs)]) => {
      parallelepiped_region(&coords(o)?, vs)
    }
    _ => None,
  }
}

/// RegionConvert[reg, "Implicit"]; anything else stays unevaluated.
pub fn region_convert_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let [region, Expr::String(form)] = args
    && form == "Implicit"
    && let Some(result) = implicit_form(region)
  {
    return Ok(result);
  }
  Ok(crate::helpers::unevaluated("RegionConvert", args))
}
