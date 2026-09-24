//! `KnotData[knot]` and `KnotData[knot, property]`.
//!
//! Knots are named by their place in the Rolfsen table of prime knots,
//! `{n, k}` — the `k`-th knot with `n` crossings, `{0, 1}` being the unknot
//! — by the standard names of the few famous ones (`"Trefoil"`,
//! `"FigureEight"`, …), or as a general torus knot `{"TorusKnot", {p, q}}`,
//! which winds `p` times around a torus's tube and `q` times through its
//! hole.
//!
//! Properties that follow from the name alone (crossing number, the
//! Alexander–Briggs label, the standard name) are known for every table
//! knot. A space curve is known for the torus knots — the textbook
//! parametrization on a torus of major radius 2 and tube radius 1 (see e.g.
//! the "Torus knot" article on Wikipedia) — and for the trefoil, whose
//! classic `{Sin[t] + 2 Sin[2 t], Cos[t] - 2 Cos[2 t], -Sin[3 t]}` form is
//! the one Wolfram uses too. Wolfram's space curves for other knots are
//! interpolated from its own curated data, which is not bundled.
//! `"ImageData"` sweeps a tube mesh around the space curve; only its shape,
//! not its exact mesh coordinates, matches real Wolfram.

#[allow(unused_imports)]
use super::*;

/// How many prime knots the Rolfsen table lists for each crossing number,
/// `{0, 1}` (the unknot) included.
const TABLE_COUNTS: &[(i128, i128)] = &[
  (0, 1),
  (3, 1),
  (4, 1),
  (5, 2),
  (6, 3),
  (7, 7),
  (8, 21),
  (9, 49),
  (10, 165),
];

/// The knots with a standard name, their table entry, and their lowercase
/// descriptive name.
const NAMED_KNOTS: &[(&str, (i128, i128), &str)] = &[
  ("Unknot", (0, 1), "unknot"),
  ("Trefoil", (3, 1), "trefoil"),
  ("FigureEight", (4, 1), "figure eight knot"),
  ("SolomonSeal", (5, 1), "Solomon seal knot"),
  ("Stevedore", (6, 1), "Stevedore knot"),
  ("PerkoPair", (10, 161), "Perko pair"),
];

/// The table knots that are torus knots, with their `(p, q)`.
const TABLE_TORUS_KNOTS: &[((i128, i128), (i128, i128))] = &[
  ((3, 1), (2, 3)),
  ((5, 1), (2, 5)),
  ((7, 1), (2, 7)),
  ((8, 19), (3, 4)),
  ((9, 1), (2, 9)),
  ((10, 124), (3, 5)),
];

/// A knot `KnotData` knows.
#[derive(Clone, Copy)]
enum Knot {
  /// `{n, k}` in the Rolfsen table.
  Table(i128, i128),
  /// `{"TorusKnot", {p, q}}`.
  Torus(i128, i128),
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

fn in_table(n: i128, k: i128) -> bool {
  TABLE_COUNTS
    .iter()
    .any(|&(cn, count)| cn == n && (1..=count).contains(&k))
}

/// Resolve a `KnotData` knot spec.
fn resolve(spec: &Expr) -> Option<Knot> {
  match spec {
    Expr::String(name) => NAMED_KNOTS
      .iter()
      .find(|(n, _, _)| n == name)
      .map(|&(_, (n, k), _)| Knot::Table(n, k)),
    Expr::List(items) if items.len() == 2 => {
      if let (Some(n), Some(k)) = (as_i128(&items[0]), as_i128(&items[1])) {
        return in_table(n, k).then_some(Knot::Table(n, k));
      }
      torus_knot_spec(spec).map(|(p, q)| Knot::Torus(p, q))
    }
    _ => None,
  }
}

impl Knot {
  /// `(p, q)` when the knot is a torus knot.
  fn torus(self) -> Option<(i128, i128)> {
    match self {
      Knot::Torus(p, q) => Some((p, q)),
      Knot::Table(n, k) => TABLE_TORUS_KNOTS
        .iter()
        .find(|(entry, _)| *entry == (n, k))
        .map(|&(_, pq)| pq),
    }
  }

  /// The space curve, as a `{x, y, z}` formula in `#1`, written the way
  /// Wolfram's (held) pure function body reads.
  fn curve_formula(self) -> Option<String> {
    if let Knot::Table(3, 1) = self {
      return Some(
        "{Sin[#1] + 2*Sin[2*#1], Cos[#1] - 2*Cos[2*#1], -Sin[3*#1]}"
          .to_string(),
      );
    }
    let (p, q) = self.torus()?;
    Some(format!(
      "{{(2 + Cos[{q}*#1])*Cos[{p}*#1], (2 + Cos[{q}*#1])*Sin[{p}*#1], \
       Sin[{q}*#1]}}"
    ))
  }

  /// The space curve, evaluated numerically at `t`.
  fn curve_point(self, t: f64) -> Option<(f64, f64, f64)> {
    if let Knot::Table(3, 1) = self {
      return Some((
        t.sin() + 2.0 * (2.0 * t).sin(),
        t.cos() - 2.0 * (2.0 * t).cos(),
        -(3.0 * t).sin(),
      ));
    }
    let (p, q) = self.torus()?;
    Some(curve_point(p, q, t))
  }
}

fn eval_wl(src: &str) -> Result<Expr, InterpreterError> {
  let parsed = crate::functions::string_ast::parse_program_to_expr(src)?;
  crate::evaluator::evaluate_expr_to_expr(&parsed)
}

/// The knot's space curve as a pure function, `{x, y, z} &`.
fn space_curve(knot: Knot) -> Option<Result<Expr, InterpreterError>> {
  Some(eval_wl(&format!("{} &", knot.curve_formula()?)))
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
fn image_data(knot: Knot) -> Option<Expr> {
  const N_ALONG: usize = 96;
  const N_AROUND: usize = 8;
  const TUBE_RADIUS: f64 = 0.2;
  const DT: f64 = 1e-4;

  let mut points: Vec<Expr> = Vec::with_capacity(N_ALONG * N_AROUND);
  for i in 0..N_ALONG {
    let t = 2.0 * std::f64::consts::PI * i as f64 / N_ALONG as f64;
    let (cx, cy, cz) = knot.curve_point(t)?;

    // Tangent via central difference, then an arbitrary orthonormal
    // (normal, binormal) frame around it to place the tube's ring.
    let (px, py, pz) = knot.curve_point(t - DT)?;
    let (nx, ny, nz) = knot.curve_point(t + DT)?;
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
  Some(Expr::List(vec![complex].into()))
}

fn normalize(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
  let len = (x * x + y * y + z * z).sqrt();
  (x / len, y / len, z / len)
}

fn knot_graphics(knot: Knot) -> Option<Result<Expr, InterpreterError>> {
  let curve = knot.curve_formula()?.replace("#1", "t");
  Some(eval_wl(&format!(
    "ParametricPlot3D[{curve}, {{t, 0, 2 Pi}}]"
  )))
}

/// Crossing number: the table's `n`, or for a `(p, q)`-torus knot
/// `min(p(q-1), q(p-1))`, a proven theorem for torus knots.
fn crossing_number(knot: Knot) -> Expr {
  match knot {
    Knot::Table(n, _) => Expr::Integer(n),
    Knot::Torus(p, q) => Expr::Integer((p * (q - 1)).min(q * (p - 1))),
  }
}

static PROPERTIES: &[&str] = &[
  "AlexanderBriggsList",
  "AlexanderBriggsNotation",
  "CrossingNumber",
  "ImageData",
  "Name",
  "SpaceCurve",
  "StandardName",
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

fn pair(a: i128, b: i128) -> Expr {
  Expr::List(vec![Expr::Integer(a), Expr::Integer(b)].into())
}

fn not_applicable() -> Expr {
  call1("Missing", Expr::String("NotApplicable".to_string()))
}

pub fn knot_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("KnotData", args));

  // `KnotData[]` — the knots with a standard name.
  if args.is_empty() {
    let names: Vec<&str> = NAMED_KNOTS.iter().map(|(n, _, _)| *n).collect();
    return Ok(string_list(&names));
  }

  // `KnotData[All]` — every knot of the table.
  if let Some(Expr::Identifier(sym)) = args.first()
    && sym == "All"
    && args.len() == 1
  {
    return Ok(Expr::List(
      TABLE_COUNTS
        .iter()
        .flat_map(|&(n, count)| (1..=count).map(move |k| pair(n, k)))
        .collect(),
    ));
  }

  // `KnotData["Properties"]` — handled before `resolve` so this reserved
  // string isn't reported as an unknown entity.
  if let Some(Expr::String(kind)) = args.first()
    && args.len() == 1
    && kind == "Properties"
  {
    return Ok(string_list(PROPERTIES));
  }

  let spec = &args[0];
  let Some(knot) = resolve(spec) else {
    if matches!(spec, Expr::String(_) | Expr::List(_)) {
      let shown =
        crate::syntax::format_expr(spec, crate::syntax::ExprForm::Output);
      crate::emit_message(&format!(
        "KnotData::notent: {shown} is not a known entity, class or tag for \
         KnotData. Use KnotData[] for a list of entities."
      ));
    }
    return unevaluated();
  };

  match args.len() {
    1 => knot_graphics(knot).unwrap_or_else(unevaluated),
    2 => {
      let Expr::String(property) = &args[1] else {
        return unevaluated();
      };
      let named = match knot {
        Knot::Table(n, k) => NAMED_KNOTS.iter().find(|(_, e, _)| *e == (n, k)),
        Knot::Torus(..) => None,
      };
      match (property.as_str(), knot) {
        ("SpaceCurve", _) => space_curve(knot).unwrap_or_else(unevaluated),
        ("ImageData", _) => image_data(knot).map_or_else(unevaluated, Ok),
        ("CrossingNumber", _) => Ok(crossing_number(knot)),
        ("AlexanderBriggsList", Knot::Table(n, k)) => Ok(pair(n, k)),
        ("AlexanderBriggsNotation", Knot::Table(n, k)) => {
          Ok(call("Subscript", vec![Expr::Integer(n), Expr::Integer(k)]))
        }
        (
          "AlexanderBriggsList" | "AlexanderBriggsNotation",
          Knot::Torus(..),
        ) => Ok(not_applicable()),
        ("StandardName", Knot::Table(n, k)) => Ok(match named {
          Some((name, _, _)) => Expr::String(name.to_string()),
          None => Expr::List(
            vec![Expr::String("Knot".to_string()), pair(n, k)].into(),
          ),
        }),
        ("StandardName", Knot::Torus(p, q)) => Ok(Expr::List(
          vec![Expr::String("TorusKnot".to_string()), pair(p, q)].into(),
        )),
        ("Name", Knot::Table(n, k)) => Ok(Expr::String(match named {
          Some((_, _, name)) => name.to_string(),
          None => format!("knot {n}-{k}"),
        })),
        ("Name", Knot::Torus(p, q)) => {
          Ok(Expr::String(format!("({p},{q})-torus knot")))
        }
        _ => unevaluated(),
      }
    }
    _ => unevaluated(),
  }
}
