//! The coordinate charts of `CoordinateTransformData`: the mapping of each
//! chart to Cartesian coordinates, and the list `CoordinateTransformData[]`
//! of every transformation between charts of the same dimension.

use crate::syntax::Expr;

/// A coordinate chart of a Euclidean space: its name, dimension, default
/// (formal) parameters and its mapping to Cartesian coordinates, as Wolfram
/// Language source over the coordinates `ctd$1`, `ctd$2`, … (`None` for a
/// chart with no mapping here).
struct Chart {
  name: &'static str,
  dim: usize,
  params: &'static [&'static str],
  to_cartesian: Option<&'static str>,
}

const fn chart(
  name: &'static str,
  dim: usize,
  params: &'static [&'static str],
  to_cartesian: Option<&'static str>,
) -> Chart {
  Chart {
    name,
    dim,
    params,
    to_cartesian,
  }
}

/// The charts, grouped in the order `CoordinateTransformData[]` lists the
/// transformations from them (each group's charts are also the order of the
/// targets of a transformation in that dimension).
const CHARTS: &[Chart] = &[
  chart(
    "Bipolar",
    2,
    &["\\[FormalA]"],
    Some(
      "{(\\[FormalA] Sinh[ctd$2])/(Cosh[ctd$2] - Cos[ctd$1]), \
       (\\[FormalA] Sin[ctd$1])/(Cosh[ctd$2] - Cos[ctd$1])}",
    ),
  ),
  chart(
    "BipolarCylindrical",
    3,
    &["\\[FormalA]"],
    Some(
      "{(\\[FormalA] Sinh[ctd$2])/(Cosh[ctd$2] - Cos[ctd$1]), \
       (\\[FormalA] Sin[ctd$1])/(Cosh[ctd$2] - Cos[ctd$1]), ctd$3}",
    ),
  ),
  chart(
    "Bispherical",
    3,
    &["\\[FormalA]"],
    Some(
      "{(\\[FormalA] Cos[ctd$3] Sin[ctd$1])/(Cosh[ctd$2] - Cos[ctd$1]), \
       (\\[FormalA] Sin[ctd$1] Sin[ctd$3])/(Cosh[ctd$2] - Cos[ctd$1]), \
       (\\[FormalA] Sinh[ctd$2])/(Cosh[ctd$2] - Cos[ctd$1])}",
    ),
  ),
  chart("Cartesian", 1, &[], Some("{ctd$1}")),
  chart("Cartesian", 2, &[], Some("{ctd$1, ctd$2}")),
  chart("Cartesian", 3, &[], Some("{ctd$1, ctd$2, ctd$3}")),
  chart(
    "CircularParabolic",
    3,
    &[],
    Some(
      "{ctd$1 ctd$2 Cos[ctd$3], ctd$1 ctd$2 Sin[ctd$3], \
       (ctd$1^2 - ctd$2^2)/2}",
    ),
  ),
  chart(
    "Confocal",
    2,
    &["\\[FormalAlpha]", "\\[FormalBeta]"],
    Some(
      "{Sqrt[((\\[FormalAlpha] + ctd$1) (\\[FormalAlpha] + ctd$2))/\
       (\\[FormalAlpha] - \\[FormalBeta])], \
       Sqrt[((\\[FormalBeta] + ctd$1) (\\[FormalBeta] + ctd$2))/\
       (\\[FormalBeta] - \\[FormalAlpha])]}",
    ),
  ),
  chart(
    "Confocal",
    3,
    &["\\[FormalAlpha]", "\\[FormalBeta]", "\\[FormalGamma]"],
    Some(
      "{Sqrt[((\\[FormalAlpha] + ctd$1) (\\[FormalAlpha] + ctd$2) \
       (\\[FormalAlpha] + ctd$3))/((\\[FormalAlpha] - \\[FormalBeta]) \
       (\\[FormalAlpha] - \\[FormalGamma]))], \
       Sqrt[((\\[FormalBeta] + ctd$1) (\\[FormalBeta] + ctd$2) \
       (\\[FormalBeta] + ctd$3))/((\\[FormalBeta] - \\[FormalAlpha]) \
       (\\[FormalBeta] - \\[FormalGamma]))], \
       Sqrt[((\\[FormalGamma] + ctd$1) (\\[FormalGamma] + ctd$2) \
       (\\[FormalGamma] + ctd$3))/((\\[FormalGamma] - \\[FormalAlpha]) \
       (\\[FormalGamma] - \\[FormalBeta]))]}",
    ),
  ),
  chart(
    "Confocal",
    4,
    &[
      "Subscript[\\[FormalAlpha], 1]",
      "Subscript[\\[FormalAlpha], 2]",
      "Subscript[\\[FormalAlpha], 3]",
      "Subscript[\\[FormalAlpha], 4]",
    ],
    None,
  ),
  chart(
    "ConfocalParaboloidal",
    3,
    &["\\[FormalAlpha]", "\\[FormalBeta]"],
    Some(
      "{Sqrt[((\\[FormalAlpha] - ctd$1) (\\[FormalAlpha] - ctd$2) \
       (\\[FormalAlpha] - ctd$3))/(\\[FormalBeta] - \\[FormalAlpha])], \
       Sqrt[((\\[FormalBeta] - ctd$1) (\\[FormalBeta] - ctd$2) \
       (\\[FormalBeta] - ctd$3))/(\\[FormalAlpha] - \\[FormalBeta])], \
       (\\[FormalAlpha] + \\[FormalBeta] - ctd$1 - ctd$2 - ctd$3)/2}",
    ),
  ),
  chart(
    "Conical",
    3,
    &["\\[FormalB]", "\\[FormalC]"],
    Some(
      "{(ctd$1 Sqrt[((\\[FormalC]^2 - ctd$2^2) (\\[FormalC]^2 - ctd$3^2))/\
       (\\[FormalC]^2 - \\[FormalB]^2)])/\\[FormalC], \
       (ctd$1 Sqrt[((ctd$2^2 - \\[FormalB]^2) (\\[FormalB]^2 - ctd$3^2))/\
       (\\[FormalC]^2 - \\[FormalB]^2)])/\\[FormalB], \
       (ctd$1 ctd$2 ctd$3)/(\\[FormalB] \\[FormalC])}",
    ),
  ),
  chart(
    "Cylindrical",
    3,
    &[],
    Some("{ctd$1 Cos[ctd$2], ctd$1 Sin[ctd$2], ctd$3}"),
  ),
  chart(
    "Elliptic",
    2,
    &["\\[FormalA]"],
    Some(
      "{\\[FormalA] Cos[ctd$2] Cosh[ctd$1], \
       \\[FormalA] Sin[ctd$2] Sinh[ctd$1]}",
    ),
  ),
  chart(
    "EllipticCylindrical",
    3,
    &["\\[FormalA]"],
    Some(
      "{\\[FormalA] Cos[ctd$2] Cosh[ctd$1], \
       \\[FormalA] Sin[ctd$2] Sinh[ctd$1], ctd$3}",
    ),
  ),
  chart(
    "Polar",
    2,
    &[],
    Some("{ctd$1 Cos[ctd$2], ctd$1 Sin[ctd$2]}"),
  ),
  chart(
    "Hyperspherical",
    3,
    &[],
    Some(
      "{ctd$1 Cos[ctd$2], ctd$1 Cos[ctd$3] Sin[ctd$2], \
       ctd$1 Sin[ctd$2] Sin[ctd$3]}",
    ),
  ),
  chart("Hyperspherical", 4, &[], None),
  chart(
    "OblateSpheroidal",
    3,
    &["\\[FormalA]"],
    Some(
      "{\\[FormalA] Cos[ctd$3] Cosh[ctd$1] Sin[ctd$2], \
       \\[FormalA] Cosh[ctd$1] Sin[ctd$2] Sin[ctd$3], \
       \\[FormalA] Cos[ctd$2] Sinh[ctd$1]}",
    ),
  ),
  chart(
    "ParabolicCylindrical",
    3,
    &[],
    Some("{(ctd$1^2 - ctd$2^2)/2, ctd$1 ctd$2, ctd$3}"),
  ),
  chart(
    "PlanarParabolic",
    2,
    &[],
    Some("{(ctd$1^2 - ctd$2^2)/2, ctd$1 ctd$2}"),
  ),
  chart(
    "ProlateSpheroidal",
    3,
    &["\\[FormalA]"],
    Some(
      "{\\[FormalA] Cos[ctd$3] Sin[ctd$2] Sinh[ctd$1], \
       \\[FormalA] Sin[ctd$2] Sin[ctd$3] Sinh[ctd$1], \
       \\[FormalA] Cos[ctd$2] Cosh[ctd$1]}",
    ),
  ),
  chart(
    "Spherical",
    3,
    &[],
    Some(
      "{ctd$1 Cos[ctd$3] Sin[ctd$2], ctd$1 Sin[ctd$2] Sin[ctd$3], \
       ctd$1 Cos[ctd$2]}",
    ),
  ),
  chart(
    "Toroidal",
    3,
    &["\\[FormalA]"],
    Some(
      "{(\\[FormalA] Cos[ctd$3] Sinh[ctd$1])/(Cosh[ctd$1] - Cos[ctd$2]), \
       (\\[FormalA] Sin[ctd$3] Sinh[ctd$1])/(Cosh[ctd$1] - Cos[ctd$2]), \
       (\\[FormalA] Sin[ctd$2])/(Cosh[ctd$1] - Cos[ctd$2])}",
    ),
  ),
];

/// The transformations between the charts of a sphere, listed after the
/// Euclidean ones.
const SPHERE_ENTRIES: &[(&str, &str, usize)] = &[
  ("Standard", "Standard", 1),
  ("Standard", "Stereographic", 1),
  ("Standard", "Standard", 2),
  ("Standard", "Stereographic", 2),
  ("Standard", "Standard", 3),
  ("Standard", "Stereographic", 3),
  ("Stereographic", "Standard", 1),
  ("Stereographic", "Stereographic", 1),
  ("Stereographic", "Standard", 2),
  ("Stereographic", "Stereographic", 2),
  ("Stereographic", "Standard", 3),
  ("Stereographic", "Stereographic", 3),
];

fn parse(source: &str) -> Option<Expr> {
  crate::parse_to_expr(source).ok()
}

/// A chart's specification: `"Name"` or `{"Name", {params…}}`, each of its
/// parameters among `taken` (the source's, for a target) renamed `p[1]`.
fn chart_spec(c: &Chart, taken: &[&str]) -> Option<Expr> {
  let name = Expr::String(c.name.to_string());
  if c.params.is_empty() {
    return Some(name);
  }
  let params = c
    .params
    .iter()
    .map(|p| {
      if taken.contains(p) {
        parse(&format!("{p}[1]"))
      } else {
        parse(p)
      }
    })
    .collect::<Option<Vec<_>>>()?;
  Some(Expr::List(vec![name, Expr::List(params.into())].into()))
}

/// `CoordinateTransformData[]`: every transformation between two charts of
/// one dimension, `{from -> to, "Euclidean", n}`, and those between the
/// charts of a sphere.
pub fn all_transformations() -> Option<Expr> {
  let mut entries = Vec::new();
  for source in CHARTS {
    for target in CHARTS.iter().filter(|t| t.dim == source.dim) {
      entries.push(Expr::List(
        vec![
          Expr::Rule {
            pattern: Box::new(chart_spec(source, &[])?),
            replacement: Box::new(chart_spec(target, source.params)?),
          },
          Expr::String("Euclidean".to_string()),
          Expr::Integer(source.dim as i128),
        ]
        .into(),
      ));
    }
  }
  let sphere = parse("{\"Sphere\", {\\[FormalCapitalR]}}")?;
  for (from, to, dim) in SPHERE_ENTRIES {
    entries.push(Expr::List(
      vec![
        Expr::Rule {
          pattern: Box::new(Expr::String(from.to_string())),
          replacement: Box::new(Expr::String(to.to_string())),
        },
        sphere.clone(),
        Expr::Integer(*dim as i128),
      ]
      .into(),
    ));
  }
  Some(Expr::List(entries.into()))
}

/// The chart named by `spec` (`"Name"` or `{"Name", {params…}}`) in
/// dimension `dim` (any, when `None`), with the parameters it was given.
fn find_chart(
  spec: &Expr,
  dim: Option<usize>,
) -> Option<(&'static Chart, Vec<Expr>)> {
  let (name, given) = match spec {
    Expr::String(s) => (s.as_str(), Vec::new()),
    Expr::List(items) if items.len() == 2 => match (&items[0], &items[1]) {
      (Expr::String(s), Expr::List(ps)) => (s.as_str(), ps.to_vec()),
      _ => return None,
    },
    _ => return None,
  };
  let chart = CHARTS.iter().find(|c| {
    c.name == name
      && dim.is_none_or(|d| d == c.dim)
      && (given.is_empty() || given.len() == c.params.len())
  })?;
  Some((chart, given))
}

/// The dimension of the chart `spec` names, when it names one in a single
/// dimension or `dim_hint` picks it.
pub fn chart_dimension(spec: &Expr, dim_hint: Option<usize>) -> Option<usize> {
  let (chart, _) = find_chart(spec, dim_hint)?;
  // A name of charts in several dimensions ("Cartesian", "Confocal") picks
  // none without a hint.
  let ambiguous = dim_hint.is_none()
    && CHARTS
      .iter()
      .any(|c| c.name == chart.name && c.dim != chart.dim);
  (!ambiguous).then_some(chart.dim)
}

/// `spec`'s mapping to Cartesian coordinates at `point`, its parameters
/// those it was given (or its formal defaults). `None` for a chart without
/// one here.
pub fn to_cartesian(spec: &Expr, point: &[Expr]) -> Option<Expr> {
  let (chart, given) = find_chart(spec, Some(point.len()))?;
  let mut mapping = parse(chart.to_cartesian?)?;
  for (i, coordinate) in point.iter().enumerate() {
    mapping = crate::syntax::substitute_variable(
      &mapping,
      &format!("ctd${}", i + 1),
      coordinate,
    );
  }
  if !given.is_empty() {
    for (formal, value) in chart.params.iter().zip(&given) {
      let formal = parse(formal)?;
      mapping = crate::functions::polynomial_ast::solve::substitute_expr(
        &mapping, &formal, value,
      );
    }
  }
  crate::evaluator::evaluate_expr_to_expr(&mapping).ok()
}
