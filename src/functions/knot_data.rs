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

static PROPERTIES: &[&str] =
  &["AlexanderBriggsNotation", "CrossingNumber", "SpaceCurve"];

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
