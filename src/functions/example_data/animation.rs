//! `ExampleData[{"TestAnimation", …}]`: generated frame sequences.
//!
//! An animation's frames are entries of their own (`"WalterCronkite01"`,
//! …). They show one generated scene through a window that drifts from
//! frame to frame — a talking head that sways, a pan across a plant seen
//! from above, toy vehicles driving across — so frame differences, optical
//! flow and tracking demos have motion to find. `"Animation"` assembles the
//! frames into the recorded `Manipulate` of a `PaneSelector`.

use super::images::{cached_opt, graphics, image_property, shape_of};
use super::raster::{Canvas, Paint, rgb8};
use super::{Collection, Entry, scenes};
use crate::functions::*;

/// `"ToyVehicles07"` → (`"ToyVehicles"`, 7).
fn split_frame(name: &str) -> Option<(&str, usize)> {
  let digits =
    name.len() - name.trim_end_matches(|c: char| c.is_ascii_digit()).len();
  if digits == 0 {
    return None;
  }
  let (base, num) = name.split_at(name.len() - digits);
  Some((base, num.parse().ok()?))
}

/// The number of frames of the animation `base`.
fn frame_count(entry: &Entry, base: &str) -> usize {
  let Some(row) = entry.table.row(base) else {
    return 1;
  };
  match entry.table.raw(row, "Frames") {
    Some(raw) => match super::meta::Cell::from_raw(raw) {
      super::meta::Cell::Exact(Expr::List(ref f)) => f.len(),
      _ => 1,
    },
    None => 1,
  }
}

/// Frame `k` (1-based) of `n` of the scene `base`, as a `w × h` image.
fn frame(entry: &Entry, base: &str, k: usize, n: usize) -> Option<Expr> {
  let shape = shape_of(&entry.skeleton("Image")?)?;
  let (w, h) = (shape.width, shape.height);
  let t = if n > 1 {
    (k - 1) as f64 / (n - 1) as f64
  } else {
    0.0
  };
  let margin = 0.2;
  let (bw, bh) = (
    ((w as f64) * (1.0 + margin)) as usize,
    ((h as f64) * (1.0 + margin)) as usize,
  );
  let scene_kind = if base.starts_with("ChemicalPlant") {
    "ChemicalPlant"
  } else {
    base
  };
  let big = scenes::draw("TestAnimation", scene_kind, bw, bh);
  let (dx, dy) = match base {
    // A gentle sway for the talking head.
    "WalterCronkite" => (
      (0.5 + 0.4 * (t * std::f64::consts::TAU).sin()) * (bw - w) as f64,
      (0.5 + 0.2 * (t * std::f64::consts::PI).sin()) * (bh - h) as f64,
    ),
    _ => (t * (bw - w) as f64, 0.3 * t * (bh - h) as f64),
  };
  let mut c = Canvas::new(w, h, [0.0; 3]);
  for y in 0..h {
    for x in 0..w {
      c.set(
        x,
        y,
        big.get((x + dx as usize).min(bw - 1), (y + dy as usize).min(bh - 1)),
      );
    }
  }
  if base == "ToyVehicles" {
    // Two toy cars crossing in opposite directions.
    for (lane, dir) in [(0.35, 1.0), (0.7, -1.0)] {
      let x =
        if dir > 0.0 { t } else { 1.0 - t } * (w as f64 * 1.2) - 0.1 * w as f64;
      let y = lane * h as f64;
      c.rect(
        x - 0.08 * w as f64,
        y - 0.04 * h as f64,
        x + 0.08 * w as f64,
        y + 0.04 * h as f64,
        &Paint::Solid(rgb8(40, 40, 40)),
      );
      for wx in [-0.05, 0.05] {
        c.circle(
          x + wx * w as f64,
          y + 0.045 * h as f64,
          0.02 * w as f64,
          &Paint::Solid(rgb8(10, 10, 10)),
        );
      }
    }
  }
  super::images::apply_stats(&mut c, entry.stat("@Stats").as_ref(), true);
  Some(c.into_image(shape.channels, shape.image_type, shape.color_space))
}

fn frame_image(entry: &Entry) -> Option<Expr> {
  let (base, k) = split_frame(entry.name)?;
  let n = frame_count(entry, base);
  cached_opt(format!("TestAnimation/{}", entry.name), || {
    frame(entry, base, k, n)
  })
}

/// The `Graphics[Raster[…]]` of a frame, with the options a frame shows
/// with.
fn frame_graphics(entry: &Entry) -> Option<Expr> {
  let img = frame_image(entry)?;
  let (w, h) = match &img {
    Expr::Image { width, height, .. } => (*width as i128, *height as i128),
    _ => return None,
  };
  let raster_args = Expr::List(
    vec![
      Expr::List(
        vec![
          Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
          Expr::List(vec![Expr::Integer(w), Expr::Integer(h)].into()),
        ]
        .into(),
      ),
      Expr::List(vec![Expr::Integer(0), Expr::Integer(255)].into()),
      rule_expr(id_expr("ColorFunction"), id_expr("GrayLevel")),
    ]
    .into(),
  );
  let options = Expr::List(
    vec![Expr::List(
      vec![
        rule_expr(id_expr("ImageSize"), id_expr("Medium")),
        rule_expr(
          id_expr("PlotRange"),
          Expr::List(
            vec![
              Expr::List(vec![Expr::Integer(0), Expr::Integer(w)].into()),
              Expr::List(vec![Expr::Integer(0), Expr::Integer(h)].into()),
            ]
            .into(),
          ),
        ),
        rule_expr(
          id_expr("ImageSizeRaw"),
          Expr::List(vec![Expr::Real(w as f64), Expr::Real(h as f64)].into()),
        ),
      ]
      .into(),
    )]
    .into(),
  );
  Some(graphics(
    &img,
    Some(&Expr::List(vec![raster_args, options].into())),
  ))
}

/// The `Graphics[Raster[rows, …], options]` of one pane: the raster holds
/// the bottom-up rows as plain lists, and the options follow the recorded
/// skeleton (`ImageSize -> Medium`, `ImageSize -> {w, h}`, `PlotRange`).
fn pane_graphics(parent: &Entry, frame: &Entry) -> Option<Expr> {
  let img = frame_image(frame)?;
  let (w, h) = match &img {
    Expr::Image { width, height, .. } => (*width as i128, *height as i128),
    _ => return None,
  };
  let data = match super::images::integer_data(&img) {
    Expr::List(ref rows) => {
      Expr::List(rows.iter().rev().cloned().collect::<Vec<_>>().into())
    }
    other => other,
  };
  let pair = |a: i128, b: i128| {
    Expr::List(vec![Expr::Integer(a), Expr::Integer(b)].into())
  };
  let raster = call(
    "Raster",
    vec![
      data,
      Expr::List(vec![pair(0, 0), pair(w, h)].into()),
      pair(0, 255),
      rule_expr(id_expr("ColorFunction"), id_expr("GrayLevel")),
    ],
  );
  // Options of the first pane in the recorded skeleton.
  let mut options = Vec::new();
  if let Some(form) = parent.stat("@AnimationForm") {
    let text = crate::syntax::expr_to_input_form(&form);
    let medium = text.contains("ImageSize -> Medium");
    let size_int = text.contains("ImageSize -> Integer");
    if medium {
      options.push(rule_expr(id_expr("ImageSize"), id_expr("Medium")));
    }
    if size_int {
      options.push(rule_expr(id_expr("ImageSize"), Expr::Integer(w)));
    }
  }
  options.push(rule_expr(id_expr("ImageSize"), pair(w, h)));
  options.push(rule_expr(
    id_expr("PlotRange"),
    Expr::List(vec![pair(0, w), pair(0, h)].into()),
  ));
  let mut args = vec![raster];
  args.extend(options);
  Some(call("Graphics", args))
}

/// `Manipulate[PaneSelector[{1 -> frame1, …}, Dynamic[i], …], {{i, 1, ""},
/// 1, n, 1, …}, ControlType -> Animator, …]`.
fn animation(entry: &Entry) -> Option<Expr> {
  let frames = match entry.exact("Frames")? {
    Expr::List(ref f) => f.iter().cloned().collect::<Vec<_>>(),
    _ => return None,
  };
  let mut panes = Vec::with_capacity(frames.len());
  for (k, f) in frames.iter().enumerate() {
    let Expr::List(pair) = f else { continue };
    let Some(Expr::String(name)) = pair.get(1) else {
      continue;
    };
    let row = entry.table.row(name)?;
    let fe = Entry {
      collection: entry.collection,
      name: &row.name,
      table: entry.table,
      row,
    };
    panes.push(rule_expr(
      Expr::Integer(k as i128 + 1),
      pane_graphics(entry, &fe)?,
    ));
  }
  let n = panes.len() as i128;
  let i = id_expr("i");
  Some(call(
    "Manipulate",
    vec![
      call(
        "PaneSelector",
        vec![
          Expr::List(panes.into()),
          call1("Dynamic", i.clone()),
          rule_expr(id_expr("ImageSize"), id_expr("All")),
          rule_expr(id_expr("Alignment"), id_expr("Automatic")),
        ],
      ),
      Expr::List(
        vec![
          Expr::List(vec![i, Expr::Integer(1), str_expr("")].into()),
          Expr::Integer(1),
          Expr::Integer(n),
          Expr::Integer(1),
          rule_expr(id_expr("AnimationRate"), id_expr("Automatic")),
          rule_expr(id_expr("DisplayAllSteps"), id_expr("True")),
          rule_expr(
            id_expr("AppearanceElements"),
            super::string_list(&[
              "ProgressSlider",
              "PlayPauseButton",
              "FasterSlowerButtons",
              "DirectionButton",
            ]),
          ),
        ]
        .into(),
      ),
      rule_expr(id_expr("ControlType"), id_expr("Animator")),
      rule_expr(id_expr("AppearanceElements"), id_expr("None")),
      rule_expr(id_expr("SynchronousUpdating"), id_expr("True")),
      rule_expr(
        id_expr("Method"),
        Expr::List(
          vec![
            rule_expr(str_expr("DynamicCore"), id_expr("False")),
            rule_expr(str_expr("TemplateExpand"), id_expr("True")),
          ]
          .into(),
        ),
      ),
    ],
  ))
}

pub struct Animations;

impl Collection for Animations {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    match property {
      None
        if split_frame(entry.name).is_some()
          && entry.skeleton("Image").is_some() =>
      {
        frame_graphics(entry)
      }
      None | Some("Animation") => {
        cached_opt(format!("TestAnimation/{}/animation", entry.name), || {
          animation(entry)
        })
      }
      Some("Image") => frame_image(entry),
      Some("Data") => {
        frame_image(entry).map(|i| super::images::integer_data(&i))
      }
      Some(p) => image_property(entry, p),
    }
  }
}
