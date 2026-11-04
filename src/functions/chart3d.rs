//! The 3D charts drawn the way Wolfram builds them: as a `Graphics3D` of
//! shaded blocks (or spheres) with the chart's axes, box and view, rendered
//! by the one `Graphics3D` renderer.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Wolfram's default colours for the series of a 3D chart.
pub(crate) const CHART3D_COLORS: [(f64, f64, f64); 7] = [
  (0.95, 0.7125, 0.0),
  (0.0, 0.595, 0.85),
  (0.325, 0.65, 0.0),
  (0.9, 0.333, 0.09),
  (0.6525, 0.405, 0.9),
  (0.75, 0.45, 0.0),
  (0.0, 0.6, 0.6),
];

/// The `i`-th default series colour.
pub(crate) fn chart3d_color(i: usize) -> (f64, f64, f64) {
  CHART3D_COLORS[i % CHART3D_COLORS.len()]
}

fn real(x: f64) -> Expr {
  Expr::Real(x)
}

fn list(items: Vec<Expr>) -> Expr {
  Expr::List(items.into())
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn rule(name: &str, value: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(Expr::Identifier(name.to_string())),
    replacement: Box::new(value),
  }
}

fn ident(name: &str) -> Expr {
  Expr::Identifier(name.to_string())
}

fn rgb((r, g, b): (f64, f64, f64)) -> Expr {
  call("RGBColor", vec![real(r), real(g), real(b)])
}

/// A bar of a 3D bar or rectangle chart from corner `lo` to corner `hi`, as
/// Wolfram draws one: unlit faces in colour `c` blending towards a light
/// tint (`c` four fifths of the way to white) on one side and a dark shade
/// (`c / 2`) on the other, with faint edges.
pub(crate) fn chart3d_block(
  lo: [f64; 3],
  hi: [f64; 3],
  c: (f64, f64, f64),
) -> Expr {
  let light = (
    c.0 + (1.0 - c.0) * 0.8,
    c.1 + (1.0 - c.1) * 0.8,
    c.2 + (1.0 - c.2) * 0.8,
  );
  let dark = (c.0 / 2.0, c.1 / 2.0, c.2 / 2.0);
  let [x0, y0, z0] = lo;
  let [x1, y1, z1] = hi;
  let pts = [
    [x0, y0, z0],
    [x0, y1, z0],
    [x0, y1, z1],
    [x0, y0, z1],
    [x1, y0, z0],
    [x1, y1, z0],
    [x1, y1, z1],
    [x1, y0, z1],
  ];
  #[derive(Clone, Copy)]
  enum Tone {
    C,
    L,
    D,
  }
  use Tone::{C, D, L};
  let faces: [([i128; 4], [Tone; 4]); 6] = [
    ([4, 3, 2, 1], [C, C, D, D]),
    ([5, 6, 7, 8], [D, D, C, C]),
    ([2, 3, 7, 6], [L, C, C, L]),
    ([5, 8, 4, 1], [C, L, L, C]),
    ([4, 8, 7, 3], [C, C, L, L]),
    ([2, 6, 5, 1], [C, C, L, L]),
  ];
  let tone = |t: Tone| {
    rgb(match t {
      C => c,
      L => light,
      D => dark,
    })
  };
  let polygons: Vec<Expr> = faces
    .iter()
    .map(|(idx, tones)| Expr::FunctionCall {
      name: "Polygon".to_string(),
      args: vec![
        list(idx.iter().map(|&k| Expr::Integer(k)).collect()),
        rule(
          "VertexColors",
          list(tones.iter().map(|&t| tone(t)).collect()),
        ),
      ]
      .into(),
    })
    .collect();
  let complex = call(
    "GraphicsComplex",
    vec![
      list(
        pts
          .iter()
          .map(|p| list(p.iter().map(|&v| real(v)).collect()))
          .collect(),
      ),
      list(polygons),
    ],
  );
  list(vec![
    call(
      "EdgeForm",
      vec![call(
        "Directive",
        vec![
          call("GrayLevel", vec![Expr::Integer(0)]),
          call("Opacity", vec![real(0.1)]),
        ],
      )],
    ),
    rgb(c),
    call(
      "Style",
      vec![
        complex,
        rule(
          "Lighting",
          list(vec![list(vec![
            Expr::String("Ambient".to_string()),
            call("GrayLevel", vec![Expr::Integer(1)]),
          ])]),
        ),
      ],
    ),
  ])
}

/// The options a chart passes on to the `Graphics3D` it draws: the ones a
/// `Graphics3D` reads (`ImageSize`, `ViewPoint`, …), not the chart's own.
fn passed_options(args: &[Expr]) -> Vec<Expr> {
  const PASSED: [&str; 16] = [
    "ImageSize",
    "ViewPoint",
    "ViewAngle",
    "ViewVertical",
    "Boxed",
    "Axes",
    "AxesLabel",
    "AxesStyle",
    "AxesEdge",
    "PlotLabel",
    "Background",
    "BoxRatios",
    "Lighting",
    "PlotRange",
    "Ticks",
    "FaceGrids",
  ];
  args
    .iter()
    .filter(|a| {
      matches!(a, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if PASSED.contains(&n.as_str())))
    })
    .cloned()
    .collect()
}

/// Draws `content` as a `Graphics3D` with the chart's `defaults`, any of
/// them overridden by the same option among the chart's `args`.
pub(crate) fn render_chart3d(
  content: Vec<Expr>,
  defaults: Vec<Expr>,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let user = passed_options(args);
  let named = |e: &Expr| match e {
    Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
      match pattern.as_ref() {
        Expr::Identifier(n) => Some(n.clone()),
        _ => None,
      }
    }
    _ => None,
  };
  let mut g_args = vec![list(content)];
  for d in defaults {
    if !user
      .iter()
      .any(|u| named(u).is_some() && named(u) == named(&d))
    {
      g_args.push(d);
    }
  }
  g_args.extend(user);
  crate::functions::plot3d::graphics3d_ast(&g_args)
}

/// The options Wolfram gives a 3D bar-like chart whose blocks span
/// `x_extent` along x and `y_extent` deep: the box flattened to the bars'
/// depth, seen almost from the front, with a value axis on the left and
/// the bar positions as unlabelled ticks along the bottom.
pub(crate) fn bar_chart3d_options(
  x_ticks: &[(f64, f64)],
  x_extent: f64,
  y_extent: f64,
  z_axis_right: bool,
  y_ticks_automatic: bool,
) -> Vec<Expr> {
  let tick = |&(at, len): &(f64, f64)| {
    list(vec![
      real(at),
      Expr::String(String::new()),
      list(vec![real(len), Expr::Integer(0)]),
    ])
  };
  let xt: Vec<Expr> = x_ticks.iter().map(tick).collect();
  let ratio_y = if x_extent > 0.0 {
    (y_extent / x_extent).max(0.2)
  } else {
    1.0
  };
  vec![
    rule(
      "Axes",
      list(vec![ident("True"), ident("True"), ident("True")]),
    ),
    rule(
      "AxesEdge",
      list(vec![
        list(vec![Expr::Integer(-1), Expr::Integer(-1)]),
        list(vec![Expr::Integer(1), Expr::Integer(-1)]),
        list(vec![
          Expr::Integer(if z_axis_right { 1 } else { -1 }),
          Expr::Integer(1),
        ]),
      ]),
    ),
    rule("Boxed", ident("False")),
    rule(
      "BoxRatios",
      list(vec![
        real(1.0),
        real(ratio_y),
        real(2.0 / (1.0 + 5.0_f64.sqrt())),
      ]),
    ),
    rule(
      "FaceGrids",
      list(vec![list(vec![
        list(vec![Expr::Integer(0), Expr::Integer(1), Expr::Integer(0)]),
        list(vec![ident("None"), ident("Automatic")]),
      ])]),
    ),
    rule("Lighting", Expr::String("Neutral".to_string())),
    rule(
      "Ticks",
      list(vec![
        list(xt),
        if y_ticks_automatic {
          ident("Automatic")
        } else {
          list(
            [
              (0.0, 0.005),
              (y_extent, 0.005),
              (0.0, 0.01),
              (y_extent, 0.01),
            ]
            .iter()
            .map(tick)
            .collect(),
          )
        },
        ident("Automatic"),
      ]),
    ),
    rule(
      "ViewPoint",
      list(vec![Expr::Integer(2), Expr::Integer(-10), Expr::Integer(2)]),
    ),
  ]
}

/// The unlabelled ticks along a bar chart's x axis for bars starting at
/// `xs` (each one wide) in groups `group_of`: short ones between the bars
/// of a group and at the ends, long ones at the ends of every group.
pub(crate) fn bar_ticks(xs: &[f64], group_of: &[usize]) -> Vec<(f64, f64)> {
  let (Some(&first), Some(&last)) = (xs.first(), xs.last()) else {
    return Vec::new();
  };
  let mut ticks = vec![(first, 0.005), (last + 1.0, 0.005)];
  for i in 1..xs.len() {
    if group_of[i] == group_of[i - 1] {
      ticks.push(((xs[i - 1] + 1.0 + xs[i]) / 2.0, 0.005));
    }
  }
  for i in 0..xs.len() {
    if i == 0 || group_of[i] != group_of[i - 1] {
      ticks.push((xs[i], 0.01));
    }
    if i + 1 == xs.len() || group_of[i] != group_of[i + 1] {
      ticks.push((xs[i] + 1.0, 0.01));
    }
  }
  ticks
}

/// Where Wolfram puts `n` unit-wide bars along x: each one wide, separated
/// by a gap of `0.1 - n / 1000`; `group_of[i]` starts a new group (an
/// extra `0.2` gap) when it differs from the bar before.
pub(crate) fn bar_positions(group_of: &[usize]) -> Vec<f64> {
  let n = group_of.len();
  let gap = 0.1 - n as f64 / 1000.0;
  let mut x = 0.0;
  let mut out = Vec::with_capacity(n);
  for (i, &g) in group_of.iter().enumerate() {
    if i > 0 {
      x += 1.0 + if g == group_of[i - 1] { gap } else { 0.2 };
    }
    out.push(x);
  }
  out
}

/// `BubbleChart3D[{{x, y, z, s}, …}]` (or several such datasets): a sphere
/// at each point whose volume follows its size `s` — the largest a fifth
/// of the points' smallest extent across — in the dataset's colour, shiny
/// under neutral lights, with grids on the box's back faces.
pub fn bubble_chart_3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use crate::functions::math_ast::try_eval_to_f64;
  let data = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let quad = |e: &Expr| -> Option<[f64; 4]> {
    let Expr::List(c) = e else { return None };
    if c.len() != 4 {
      return None;
    }
    let v: Vec<f64> = c.iter().filter_map(try_eval_to_f64).collect();
    (v.len() == 4).then(|| [v[0], v[1], v[2], v[3]])
  };
  let Expr::List(items) = &data else {
    return Ok(crate::functions::chart::invalid_dataset(
      "BubbleChart3D",
      args,
    ));
  };
  let sets: Option<Vec<Vec<[f64; 4]>>> =
    if items.iter().all(|e| quad(e).is_some()) {
      Some(vec![items.iter().filter_map(quad).collect()])
    } else {
      items
        .iter()
        .map(|set| match set {
          Expr::List(ps) => ps.iter().map(quad).collect(),
          _ => None,
        })
        .collect()
    };
  let Some(sets) = sets.filter(|s| s.iter().any(|set| !set.is_empty())) else {
    return Ok(crate::functions::chart::invalid_dataset(
      "BubbleChart3D",
      args,
    ));
  };
  let all: Vec<[f64; 4]> = sets.iter().flatten().copied().collect();
  let extent = (0..3)
    .map(|d| {
      let (lo, hi) = all
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| {
          (a.min(p[d]), b.max(p[d]))
        });
      hi - lo
    })
    .filter(|e| *e > 0.0)
    .fold(f64::INFINITY, f64::min);
  let extent = if extent.is_finite() { extent } else { 1.0 };
  let s_max = all.iter().map(|p| p[3].abs()).fold(0.0, f64::max);
  let r_max = 0.2 * extent;
  let mut content = Vec::new();
  for (k, set) in sets.iter().enumerate() {
    let mut group = vec![
      rgb(chart3d_color(k)),
      call(
        "Specularity",
        vec![call("GrayLevel", vec![Expr::Integer(1)]), Expr::Integer(20)],
      ),
    ];
    for p in set {
      let r = if s_max > 0.0 {
        r_max * (p[3].abs() / s_max).cbrt()
      } else {
        r_max
      };
      group.push(call(
        "Sphere",
        vec![list(vec![real(p[0]), real(p[1]), real(p[2])]), real(r)],
      ));
    }
    content.push(list(group));
  }
  let back_faces = list(vec![
    list(vec![
      list(vec![Expr::Integer(0), Expr::Integer(1), Expr::Integer(0)]),
      list(vec![ident("Automatic"), ident("Automatic")]),
    ]),
    list(vec![
      list(vec![Expr::Integer(-1), Expr::Integer(0), Expr::Integer(0)]),
      list(vec![ident("Automatic"), ident("Automatic")]),
    ]),
    list(vec![
      list(vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(-1)]),
      list(vec![ident("Automatic"), ident("Automatic")]),
    ]),
  ]);
  let defaults = vec![
    rule("Axes", ident("True")),
    rule("BoxRatios", list(vec![real(1.0), real(1.0), real(1.0)])),
    rule("FaceGrids", back_faces),
    rule("Lighting", Expr::String("Neutral".to_string())),
  ];
  render_chart3d(content, defaults, &args[1..])
}
