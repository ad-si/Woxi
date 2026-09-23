//! `KnotData[name]` and `KnotData[name, property]` for torus knots: knots
//! that lie on the surface of an (unknotted) torus, winding `p` times
//! around its tube and `q` times through its hole. Every named entity here
//! is a torus knot, and any coprime `{p, q}` is accepted directly through
//! `KnotData[{"TorusKnot", {p, q}}]` — the general family, not just the
//! handful of named members.
//!
//! The space curve is the standard textbook parametrization of a
//! `(p, q)`-torus knot on a torus of major radius 2 and tube radius 1
//! (see e.g. the "Torus knot" article on Wikipedia): an independently
//! derived formula, not Wolfram's own internal representation, so the
//! numeric coefficients Wolfram prints for e.g. `KnotData["Trefoil",
//! "SpaceCurve"]` will not match ours — only the shape of the knot does.
//! `"ImageData"` sweeps a tube mesh around that same curve; likewise only
//! its shape, not its exact mesh coordinates, matches real Wolfram.

#[allow(unused_imports)]
use super::*;

struct KnotInfo {
  name: &'static str,
  aliases: &'static [&'static str],
  alexander_briggs: &'static str,
  p: i128,
  q: i128,
}

static KNOTS: &[KnotInfo] = &[
  KnotInfo {
    name: "Trefoil",
    aliases: &["TrefoilKnot", "3_1"],
    alexander_briggs: "3_1",
    p: 2,
    q: 3,
  },
  KnotInfo {
    name: "CinquefoilKnot",
    aliases: &["SolomonsSealKnot", "Cinquefoil", "5_1"],
    alexander_briggs: "5_1",
    p: 2,
    q: 5,
  },
  KnotInfo {
    name: "SeptafoilKnot",
    aliases: &["Septafoil", "7_1"],
    alexander_briggs: "7_1",
    p: 2,
    q: 7,
  },
];

fn find_named_knot(name: &str) -> Option<&'static KnotInfo> {
  KNOTS
    .iter()
    .find(|k| k.name == name || k.aliases.contains(&name))
}

fn as_i128(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Integer(n) => Some(*n),
    _ => None,
  }
}

/// Parse a `{"TorusKnot", {p, q}}` specification into its two coprime
/// winding numbers.
fn torus_knot_spec(expr: &Expr) -> Option<(i128, i128)> {
  let Expr::List(items) = expr else {
    return None;
  };
  if items.len() != 2 {
    return None;
  }
  let Expr::String(tag) = &items[0] else {
    return None;
  };
  if tag != "TorusKnot" {
    return None;
  }
  let Expr::List(pq) = &items[1] else {
    return None;
  };
  if pq.len() != 2 {
    return None;
  }
  let p = as_i128(&pq[0])?;
  let q = as_i128(&pq[1])?;
  if p < 1 || q < 1 || crate::functions::math_ast::gcd_i128(p, q) != 1 {
    return None;
  }
  Some((p, q))
}

/// Resolve any accepted `KnotData` name spec to `(p, q, Alexander-Briggs
/// notation)`.
fn resolve(spec: &Expr) -> Option<(i128, i128, Option<&'static str>)> {
  match spec {
    Expr::String(name) => {
      find_named_knot(name).map(|k| (k.p, k.q, Some(k.alexander_briggs)))
    }
    Expr::List(_) => torus_knot_spec(spec).map(|(p, q)| (p, q, None)),
    _ => None,
  }
}

fn eval_wl(src: &str) -> Result<Expr, InterpreterError> {
  let parsed = crate::functions::string_ast::parse_program_to_expr(src)?;
  crate::evaluator::evaluate_expr_to_expr(&parsed)
}

/// The `(p, q)`-torus knot's space curve, as `Function[{t}, {x, y, z}]`.
fn space_curve(p: i128, q: i128) -> Result<Expr, InterpreterError> {
  eval_wl(&format!(
    "Function[{{t}}, {{(2+Cos[{q} t]) Cos[{p} t], \
     (2+Cos[{q} t]) Sin[{p} t], Sin[{q} t]}}]"
  ))
}

/// `(p, q)`-torus knot space curve, evaluated numerically at `t`.
fn curve_point(p: i128, q: i128, t: f64) -> (f64, f64, f64) {
  let p = p as f64;
  let q = q as f64;
  let radial = 2.0 + (q * t).cos();
  (
    radial * (p * t).cos(),
    radial * (p * t).sin(),
    (q * t).sin(),
  )
}

/// The 3D mesh for `KnotData[…, "ImageData"]`: a tube swept around the
/// `(p, q)`-torus knot's space curve, as `{GraphicsComplex[points,
/// {Polygon[faces]}]}` — the shape `KnotData[…, "ImageData"]` returns in
/// real Wolfram (see its docs: `GraphicsComplex` mesh data for the 3D
/// knot picture). The mesh itself — vertex count, triangulation, tube
/// radius — is our own choice, not Wolfram's internal one, so (like
/// `SpaceCurve`) only the swept shape matches, not the exact coordinates.
fn image_data(p: i128, q: i128) -> Expr {
  const N_ALONG: usize = 96;
  const N_AROUND: usize = 8;
  const TUBE_RADIUS: f64 = 0.2;
  const DT: f64 = 1e-4;

  let mut points: Vec<Expr> = Vec::with_capacity(N_ALONG * N_AROUND);
  for i in 0..N_ALONG {
    let t = 2.0 * std::f64::consts::PI * i as f64 / N_ALONG as f64;
    let (cx, cy, cz) = curve_point(p, q, t);

    // Tangent via central difference, then an arbitrary orthonormal
    // (normal, binormal) frame around it to place the tube's ring.
    let (px, py, pz) = curve_point(p, q, t - DT);
    let (nx, ny, nz) = curve_point(p, q, t + DT);
    let (tx, ty, tz) = normalize(nx - px, ny - py, nz - pz);
    let reference = if tx.abs() < 0.9 {
      (1.0, 0.0, 0.0)
    } else {
      (0.0, 1.0, 0.0)
    };
    let (ux, uy, uz) = normalize(
      ty * reference.2 - tz * reference.1,
      tz * reference.0 - tx * reference.2,
      tx * reference.1 - ty * reference.0,
    );
    let (vx, vy, vz) =
      (ty * uz - tz * uy, tz * ux - tx * uz, tx * uy - ty * ux);

    for j in 0..N_AROUND {
      let theta = 2.0 * std::f64::consts::PI * j as f64 / N_AROUND as f64;
      let (cos_t, sin_t) = (theta.cos(), theta.sin());
      let ox = TUBE_RADIUS * (cos_t * ux + sin_t * vx);
      let oy = TUBE_RADIUS * (cos_t * uy + sin_t * vy);
      let oz = TUBE_RADIUS * (cos_t * uz + sin_t * vz);
      points.push(Expr::List(
        vec![
          Expr::Real(cx + ox),
          Expr::Real(cy + oy),
          Expr::Real(cz + oz),
        ]
        .into(),
      ));
    }
  }

  let mut faces: Vec<Expr> = Vec::with_capacity(N_ALONG * N_AROUND);
  for i in 0..N_ALONG {
    let i_next = (i + 1) % N_ALONG;
    for j in 0..N_AROUND {
      let j_next = (j + 1) % N_AROUND;
      let a = (i * N_AROUND + j + 1) as i128;
      let b = (i * N_AROUND + j_next + 1) as i128;
      let c = (i_next * N_AROUND + j_next + 1) as i128;
      let d = (i_next * N_AROUND + j + 1) as i128;
      faces.push(Expr::List(
        vec![
          Expr::Integer(a),
          Expr::Integer(b),
          Expr::Integer(c),
          Expr::Integer(d),
        ]
        .into(),
      ));
    }
  }

  let polygon = Expr::FunctionCall {
    name: "Polygon".to_string(),
    args: vec![Expr::List(faces.into())].into(),
  };
  let complex = Expr::FunctionCall {
    name: "GraphicsComplex".to_string(),
    args: vec![Expr::List(points.into()), Expr::List(vec![polygon].into())]
      .into(),
  };
  Expr::List(vec![complex].into())
}

fn normalize(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
  let len = (x * x + y * y + z * z).sqrt();
  (x / len, y / len, z / len)
}

fn knot_graphics(p: i128, q: i128) -> Result<Expr, InterpreterError> {
  eval_wl(&format!(
    "ParametricPlot3D[{{(2+Cos[{q} t]) Cos[{p} t], \
     (2+Cos[{q} t]) Sin[{p} t], Sin[{q} t]}}, {{t, 0, 2 Pi}}]"
  ))
}

/// Crossing number of the `(p, q)`-torus knot: `min(p(q-1), q(p-1))`, a
/// proven theorem for torus knots (not specific to any single one).
fn crossing_number(p: i128, q: i128) -> Expr {
  Expr::Integer((p * (q - 1)).min(q * (p - 1)))
}

static PROPERTIES: &[&str] = &[
  "AlexanderBriggsNotation",
  "CrossingNumber",
  "ImageData",
  "SpaceCurve",
];

fn string_list(items: &[&str]) -> Expr {
  Expr::List(
    items
      .iter()
      .map(|s| Expr::String(s.to_string()))
      .collect::<Vec<_>>()
      .into(),
  )
}

pub fn knot_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("KnotData", args));

  // `KnotData[All]` — the list of known named entities.
  if let Some(Expr::Identifier(sym)) = args.first()
    && sym == "All"
    && args.len() == 1
  {
    let mut names: Vec<&str> = KNOTS.iter().map(|k| k.name).collect();
    names.sort_unstable();
    return Ok(string_list(&names));
  }

  // `KnotData["Properties"]` — handled before `resolve` so this reserved
  // string isn't reported as an unknown entity.
  if let Some(Expr::String(kind)) = args.first()
    && args.len() == 1
    && kind == "Properties"
  {
    return Ok(string_list(PROPERTIES));
  }

  let Some(spec) = args.first() else {
    return unevaluated();
  };
  let Some((p, q, alexander_briggs)) = resolve(spec) else {
    if let Expr::String(name) = spec {
      crate::emit_message(&format!(
        "KnotData::notent: {name} is not a known entity, class, or tag for \
         KnotData. Use KnotData[] for a list of entities."
      ));
    }
    return unevaluated();
  };

  match args.len() {
    1 => knot_graphics(p, q),
    2 => {
      let Expr::String(property) = &args[1] else {
        return unevaluated();
      };
      match property.as_str() {
        "SpaceCurve" => space_curve(p, q),
        "ImageData" => Ok(image_data(p, q)),
        "CrossingNumber" => Ok(crossing_number(p, q)),
        "AlexanderBriggsNotation" => match alexander_briggs {
          Some(s) => Ok(Expr::String(s.to_string())),
          None => unevaluated(),
        },
        _ => unevaluated(),
      }
    }
    _ => unevaluated(),
  }
}
