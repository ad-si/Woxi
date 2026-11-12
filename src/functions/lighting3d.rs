//! The lighting model every 3D renderer shades its facets with.
//!
//! Decoded against wolframscript by rasterizing single polygons at known
//! normals and reading back pixel values (see `conformance_gaps.md`,
//! "Graphics3D lighting"):
//!
//! - `color = glow + Σ ambient·c + Σ light·(c·max(0, n·l) + s·(n·h)^e)`,
//!   clipped to `[0, 1]` per channel, where `c` is the
//!   surface color, `s`/`e` its `Specularity`, `l` the unit vector toward
//!   the light and `h` the Blinn half vector between `l` and the viewer.
//!   The highlight only appears where the light reaches the face.
//! - Faces are lit two-sided: a normal pointing away from the viewer is
//!   turned round first.
//! - A `"Directional"` light at a single point `p` shines from `p` toward
//!   the origin of `p`'s own coordinate system, so `ImageScaled[{2, 0, 2}]`
//!   is the direction `{1, 0, 1}` in the camera frame (x right, y up, z
//!   toward the viewer) whatever the size of the picture.
//! - Arithmetic is on the sRGB channel values directly, no gamma.

use crate::functions::graphics::parse_color;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

pub(crate) type Rgb = [f64; 3];

/// Where a light sits: in the scene's own coordinates, or in the camera
/// frame `ImageScaled` places it in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LightPoint {
  World([f64; 3]),
  /// `ImageScaled[{x, y, z}]`: x across the picture, y up it, z toward
  /// the viewer; `{0.5, 0.5, 0.5}` is the middle of the scene.
  View([f64; 3]),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Light {
  Ambient(Rgb),
  /// Parallel light traveling from `from` toward `to`.
  Directional {
    color: Rgb,
    from: LightPoint,
    to: LightPoint,
  },
  /// A light at a point, shining every way; `attenuation` is `{c0, c1, c2}`
  /// of the intensity factor `1 / (c0 + c1 d + c2 d²)`.
  Point {
    color: Rgb,
    at: LightPoint,
    attenuation: [f64; 3],
  },
  /// A point light restricted to a cone of half angle `half_angle` about
  /// the direction toward `target`, fading as `cos^exponent` off its axis
  /// and with distance as a `Point` light does.
  Spot {
    color: Rgb,
    at: LightPoint,
    target: LightPoint,
    half_angle: f64,
    exponent: f64,
    attenuation: [f64; 3],
  },
}

/// A set of lights, as a `Lighting` option or directive describes it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Lighting {
  pub lights: Vec<Light>,
  /// The `Lighting -> "name"` this set was read from, which is how a plot
  /// writes it back into its style.
  pub name: Option<String>,
}

fn view_dir_light(color: Rgb, p: [f64; 3]) -> Light {
  Light::Directional {
    color,
    from: LightPoint::View(p),
    to: LightPoint::View([0.0; 3]),
  }
}

fn gray(g: f64) -> Rgb {
  [g, g, g]
}

impl Lighting {
  /// `Lighting -> Automatic` (also `"Standard"`): an ambient term and four
  /// colored directional lights fixed relative to the camera.
  pub(crate) fn standard() -> Self {
    Lighting {
      name: None,
      lights: vec![
        Light::Ambient([0.4, 0.2, 0.2]),
        view_dir_light([0.0, 0.18, 0.5], [2.0, 0.0, 2.0]),
        view_dir_light([0.18, 0.5, 0.18], [2.0, 2.0, 3.0]),
        view_dir_light([0.5, 0.18, 0.0], [0.0, 2.0, 2.0]),
        view_dir_light([0.0, 0.0, 0.18], [0.0, 0.0, 2.0]),
      ],
    }
  }

  /// `Lighting -> "Neutral"`: the same arrangement in colorless light.
  pub(crate) fn neutral() -> Self {
    Lighting {
      name: Some("Neutral".into()),
      lights: vec![
        Light::Ambient(gray(0.35)),
        view_dir_light(gray(0.37), [2.0, 0.0, 2.0]),
        view_dir_light(gray(0.37), [2.0, 2.0, 2.0]),
        view_dir_light(gray(0.37), [0.0, 2.0, 2.0]),
      ],
    }
  }

  /// The named light sets `Lighting -> "name"` accepts.
  pub(crate) fn named(name: &str) -> Option<Self> {
    let mut lighting = match name {
      "Standard" => Self::standard(),
      "Neutral" => Self::neutral(),
      "Classic" => Lighting {
        name: None,
        lights: vec![
          Light::Ambient([0.312, 0.188, 0.4]),
          view_dir_light([0.8, 0.0, 0.0], [2.0, 0.0, 2.0]),
          view_dir_light([0.0, 0.8, 0.0], [2.0, 2.0, 2.0]),
          view_dir_light([0.0, 0.0, 0.8], [0.0, 2.0, 2.0]),
        ],
      },
      "Accent" => Lighting {
        name: None,
        lights: vec![view_dir_light(gray(1.0), [1.0, 1.0, 4.0])],
      },
      "ThreePoint" => Lighting {
        name: None,
        lights: vec![
          view_dir_light(gray(0.85), [1.0, 1.0, 2.0]),
          view_dir_light(gray(0.4), [-1.0, -2.0, 2.0]),
          view_dir_light(gray(0.1), [0.5, 2.0, -1.0]),
        ],
      },
      _ => return None,
    };
    lighting.name = Some(name.to_string());
    Some(lighting)
  }

  /// Read a `Lighting` value. `Automatic` and anything unrecognized come
  /// back as `None` — the caller's default light set applies — as
  /// wolframscript ignores a malformed value (`Lighting -> Red` renders
  /// exactly like the default). `Lighting -> None` is the empty set: only
  /// `Glow` shows.
  pub(crate) fn parse(expr: &Expr) -> Option<Self> {
    match expr {
      Expr::Identifier(s) if s == "None" => Some(Lighting {
        name: None,
        lights: vec![],
      }),
      Expr::Identifier(s) | Expr::String(s) => Self::named(s),
      Expr::List(items) => {
        let lights: Option<Vec<Light>> =
          items.iter().map(parse_light).collect();
        lights.map(|lights| Lighting { name: None, lights })
      }
      _ => None,
    }
  }

  /// The light set of the `index`-th surface of a plot under the default
  /// theme (WL's `System`PlotThemeDump`$ThemeDefaultLighting`), with its
  /// color and specular exponent: the plot palette's color lit by a
  /// darkened copy of itself, cycling after ten surfaces.
  pub(crate) fn plot_theme(index: usize) -> (Rgb, f64, Lighting) {
    // (color, Specularity exponent, Darker amount, Lighter amount of the
    // ambient term, override of the middle light).
    const THEME: [(Rgb, f64, f64, f64, Option<Rgb>); 10] = [
      ([0.880722, 0.611041, 0.142051], 3.0, 0.7, 0.05, None),
      ([0.368417, 0.506779, 0.709798], 3.0, 0.58, 0.05, None),
      ([0.560181, 0.691569, 0.194885], 3.0, 0.75, 0.05, None),
      (
        [0.922526, 0.385626, 0.209179],
        6.0,
        0.75,
        0.1,
        Some([0.3; 3]),
      ),
      (
        [0.528488, 0.470624, 0.701351],
        6.0,
        0.5,
        0.2,
        Some([0.2, 0.0, 1.0]),
      ),
      (
        [0.772079, 0.431554, 0.102387],
        6.0,
        0.75,
        0.2,
        Some([0.3; 3]),
      ),
      ([0.363898, 0.618501, 0.782349], 6.0, 0.7, 0.2, None),
      ([1.0, 0.75, 0.0], 3.0, 0.74, 0.1, Some([0.1; 3])),
      ([0.647624, 0.37816, 0.614037], 6.0, 0.6, 0.2, Some([0.3; 3])),
      ([0.571589, 0.586483, 0.0], 6.0, 0.65, 0.2, None),
    ];
    let (color, exponent, darker, lighter, middle) = THEME[index % THEME.len()];
    let lc = color.map(|c| c * (1.0 - darker));
    let ambient = lc.map(|c| c + lighter * (1.0 - c));
    let lighting = Lighting {
      name: None,
      lights: vec![
        Light::Ambient(ambient),
        view_dir_light(lc, [0.0, 2.0, 2.0]),
        view_dir_light(middle.unwrap_or(lc), [2.0, 2.0, 2.0]),
        view_dir_light(lc, [2.0, 0.0, 2.0]),
      ],
    };
    (color, exponent, lighting)
  }

  /// The theme lighting as WL writes it into a plot's `Directive`, so a
  /// plot re-rendered by `Show` keeps its look.
  pub(crate) fn to_expr(&self) -> Expr {
    if let Some(name) = &self.name {
      return Expr::String(name.clone());
    }
    if self.lights.is_empty() {
      return Expr::Identifier("None".into());
    }
    let rgb = |c: Rgb| Expr::FunctionCall {
      name: "RGBColor".into(),
      args: c.iter().map(|v| Expr::Real(*v)).collect::<Vec<_>>().into(),
    };
    let point = |p: &LightPoint| {
      let (v, scaled) = match p {
        LightPoint::World(v) => (v, false),
        LightPoint::View(v) => (v, true),
      };
      let list = Expr::List(
        v.iter()
          .map(|c| {
            if c.fract() == 0.0 {
              Expr::Integer(*c as i128)
            } else {
              Expr::Real(*c)
            }
          })
          .collect::<Vec<_>>()
          .into(),
      );
      if scaled {
        Expr::FunctionCall {
          name: "ImageScaled".into(),
          args: vec![list].into(),
        }
      } else {
        list
      }
    };
    let s = |t: &str| Expr::String(t.into());
    Expr::List(
      self
        .lights
        .iter()
        .map(|l| {
          let items = match l {
            Light::Ambient(c) => vec![s("Ambient"), rgb(*c)],
            Light::Directional { color, from, to } => {
              let origin = matches!(to,
                LightPoint::View(v) | LightPoint::World(v) if *v == [0.0; 3]);
              let target = if origin {
                point(from)
              } else {
                Expr::List(vec![point(from), point(to)].into())
              };
              vec![s("Directional"), rgb(*color), target]
            }
            Light::Point { color, at, .. } => {
              vec![s("Point"), rgb(*color), point(at)]
            }
            Light::Spot {
              color,
              at,
              target,
              half_angle,
              ..
            } => vec![
              s("Spot"),
              rgb(*color),
              Expr::List(vec![point(at), point(target)].into()),
              Expr::Real(*half_angle),
            ],
          };
          Expr::List(items.into())
        })
        .collect::<Vec<_>>()
        .into(),
    )
  }
}

static STANDARD: std::sync::LazyLock<Lighting> =
  std::sync::LazyLock::new(Lighting::standard);
static NEUTRAL: std::sync::LazyLock<Lighting> =
  std::sync::LazyLock::new(Lighting::neutral);
static THEME: std::sync::LazyLock<Vec<(Rgb, f64, Lighting)>> =
  std::sync::LazyLock::new(|| (0..10).map(Lighting::plot_theme).collect());

/// `Lighting -> Automatic`, shared.
pub(crate) fn standard_lighting() -> &'static Lighting {
  &STANDARD
}

/// `Lighting -> "Neutral"`, shared.
pub(crate) fn neutral_lighting() -> &'static Lighting {
  &NEUTRAL
}

/// [`Lighting::plot_theme`], shared.
pub(crate) fn theme_surface(index: usize) -> &'static (Rgb, f64, Lighting) {
  &THEME[index % THEME.len()]
}

fn parse_rgb(expr: &Expr) -> Option<Rgb> {
  let c = parse_color(expr)?;
  Some([c.r, c.g, c.b])
}

fn parse_light_point(expr: &Expr) -> Option<LightPoint> {
  let vec3 = |e: &Expr| -> Option<[f64; 3]> {
    match e {
      Expr::List(v) if v.len() == 3 => Some([
        try_eval_to_f64(&v[0])?,
        try_eval_to_f64(&v[1])?,
        try_eval_to_f64(&v[2])?,
      ]),
      _ => None,
    }
  };
  match expr {
    Expr::FunctionCall { name, args }
      if name == "ImageScaled" && args.len() == 1 =>
    {
      vec3(&args[0]).map(LightPoint::View)
    }
    _ => vec3(expr).map(LightPoint::World),
  }
}

/// A `{pt}` or `{pt1, pt2}` light placement.
fn parse_light_pair(expr: &Expr) -> Option<(LightPoint, Option<LightPoint>)> {
  if let Some(p) = parse_light_point(expr) {
    return Some((p, None));
  }
  match expr {
    Expr::List(v) if v.len() == 2 => {
      Some((parse_light_point(&v[0])?, Some(parse_light_point(&v[1])?)))
    }
    _ => None,
  }
}

/// The `{c0, c1, c2}` of a light that keeps its strength at any distance.
const NO_ATTENUATION: [f64; 3] = [1.0, 0.0, 0.0];

/// The intensity factor `1 / (c0 + c1 d + c2 d²)` at distance `d`, or None
/// where the denominator is not positive.
fn attenuate(attenuation: &[f64; 3], dist: f64) -> Option<f64> {
  let falloff =
    attenuation[0] + attenuation[1] * dist + attenuation[2] * dist * dist;
  (falloff > 0.0).then(|| 1.0 / falloff)
}

fn parse_light(expr: &Expr) -> Option<Light> {
  let Expr::List(items) = expr else {
    return None;
  };
  let kind = match items.first()? {
    Expr::String(s) | Expr::Identifier(s) => s.as_str(),
    _ => return None,
  };
  let color = parse_rgb(items.get(1)?)?;
  match (kind, items.len()) {
    ("Ambient", 2) => Some(Light::Ambient(color)),
    ("Directional", 3) => {
      let (from, to) = parse_light_pair(&items[2])?;
      let to = to.unwrap_or(match from {
        LightPoint::World(_) => LightPoint::World([0.0; 3]),
        LightPoint::View(_) => LightPoint::View([0.0; 3]),
      });
      Some(Light::Directional { color, from, to })
    }
    ("Point", 3 | 4) => {
      let at = parse_light_point(&items[2])?;
      let attenuation = match items.get(3) {
        None => NO_ATTENUATION,
        Some(Expr::List(c)) if c.len() == 3 => [
          try_eval_to_f64(&c[0])?,
          try_eval_to_f64(&c[1])?,
          try_eval_to_f64(&c[2])?,
        ],
        Some(_) => return None,
      };
      Some(Light::Point {
        color,
        at,
        attenuation,
      })
    }
    ("Spot", 4 | 5) => {
      let (at, target) = parse_light_pair(&items[2])?;
      // `{c0, c1, c2}` attenuates with distance. Wolfram also accepts
      // other values in this place (a Demonstration passes a bare number)
      // and then draws the spot at full strength, as measured from its
      // rendered snapshot.
      let attenuation = match items.get(4) {
        Some(Expr::List(c)) if c.len() == 3 => [
          try_eval_to_f64(&c[0])?,
          try_eval_to_f64(&c[1])?,
          try_eval_to_f64(&c[2])?,
        ],
        _ => NO_ATTENUATION,
      };
      let (half_angle, exponent) = match &items[3] {
        Expr::List(v) if v.len() == 2 => {
          (try_eval_to_f64(&v[0])?, try_eval_to_f64(&v[1])?)
        }
        e => (try_eval_to_f64(e)?, 1.0),
      };
      Some(Light::Spot {
        color,
        at,
        target: target.unwrap_or(LightPoint::View([0.5; 3])),
        half_angle,
        exponent,
        attenuation,
      })
    }
    _ => None,
  }
}

/// The camera and scene placement the lights are positioned against.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewFrame {
  /// Screen right, screen up and the direction toward the viewer, as
  /// unit vectors in the renderer's coordinates.
  pub right: [f64; 3],
  pub up: [f64; 3],
  pub toward: [f64; 3],
  /// Middle of the scene and the length `ImageScaled` spans, so a point or
  /// spot light given in `ImageScaled` coordinates can be placed.
  pub center: [f64; 3],
  pub size: f64,
  /// How a scene coordinate maps into the renderer's: `p * scale + offset`
  /// per axis (plots draw in a normalized box).
  pub scale: [f64; 3],
  pub offset: [f64; 3],
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(a: [f64; 3]) -> Option<[f64; 3]> {
  let len = dot(a, a).sqrt();
  (len > 1e-15).then(|| [a[0] / len, a[1] / len, a[2] / len])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

impl ViewFrame {
  /// A light point in the renderer's coordinates.
  fn place(&self, p: LightPoint) -> [f64; 3] {
    match p {
      LightPoint::World(v) => [
        v[0] * self.scale[0] + self.offset[0],
        v[1] * self.scale[1] + self.offset[1],
        v[2] * self.scale[2] + self.offset[2],
      ],
      LightPoint::View(v) => {
        let o = [v[0] - 0.5, v[1] - 0.5, v[2] - 0.5];
        let mut out = self.center;
        for k in 0..3 {
          out[k] += self.size
            * (o[0] * self.right[k]
              + o[1] * self.up[k]
              + o[2] * self.toward[k]);
        }
        out
      }
    }
  }
}

/// What a facet is made of.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Material {
  pub color: Rgb,
  /// `Specularity[color, exponent]`.
  pub specular: Option<(Rgb, f64)>,
  /// `Glow[color]`.
  pub glow: Option<Rgb>,
}

/// The color of a facet with `normal` at `position` under `lighting`.
pub(crate) fn shade(
  material: &Material,
  normal: [f64; 3],
  position: [f64; 3],
  lighting: &Lighting,
  frame: &ViewFrame,
) -> (u8, u8, u8) {
  let mut n = normalize(normal).unwrap_or(frame.toward);
  if dot(n, frame.toward) < 0.0 {
    n = [-n[0], -n[1], -n[2]];
  }
  let c = material.color;
  let mut out = material.glow.unwrap_or([0.0; 3]);
  let add = |out: &mut Rgb, light: Rgb, intensity: f64, l: [f64; 3]| {
    let cos = dot(n, l);
    if cos <= 0.0 || intensity <= 0.0 {
      return;
    }
    let highlight = material.specular.and_then(|(s, e)| {
      let h = normalize([
        l[0] + frame.toward[0],
        l[1] + frame.toward[1],
        l[2] + frame.toward[2],
      ])?;
      Some((s, dot(n, h).max(0.0).powf(e)))
    });
    for k in 0..3 {
      let mut v = c[k] * cos;
      if let Some((s, f)) = highlight {
        v += s[k] * f;
      }
      out[k] += light[k] * intensity * v;
    }
  };
  for light in &lighting.lights {
    match light {
      Light::Ambient(a) => {
        for k in 0..3 {
          out[k] += a[k] * c[k];
        }
      }
      Light::Directional { color, from, to } => {
        let dir = match (from, to) {
          (LightPoint::View(f), LightPoint::View(t)) => {
            let d = sub(*f, *t);
            let mut w = [0.0; 3];
            for k in 0..3 {
              w[k] = d[0] * frame.right[k]
                + d[1] * frame.up[k]
                + d[2] * frame.toward[k];
            }
            w
          }
          _ => sub(frame.place(*from), frame.place(*to)),
        };
        if let Some(l) = normalize(dir) {
          add(&mut out, *color, 1.0, l);
        }
      }
      Light::Point {
        color,
        at,
        attenuation,
      } => {
        let d = sub(frame.place(*at), position);
        if let (Some(l), Some(k)) =
          (normalize(d), attenuate(attenuation, dot(d, d).sqrt()))
        {
          add(&mut out, *color, k, l);
        }
      }
      Light::Spot {
        color,
        at,
        target,
        half_angle,
        exponent,
        attenuation,
      } => {
        let src = frame.place(*at);
        let (Some(l), Some(axis)) = (
          normalize(sub(src, position)),
          normalize(sub(frame.place(*target), src)),
        ) else {
          continue;
        };
        let cos_off = -dot(l, axis);
        let d = sub(src, position);
        if cos_off >= half_angle.cos()
          && let Some(k) = attenuate(attenuation, dot(d, d).sqrt())
        {
          add(&mut out, *color, k * cos_off.max(0.0).powf(*exponent), l);
        }
      }
    }
  }
  let q = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
  (q(out[0]), q(out[1]), q(out[2]))
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Camera looking straight down -z with y up, the frame the probes were
  /// rendered in.
  fn frontal() -> ViewFrame {
    ViewFrame {
      right: [1.0, 0.0, 0.0],
      up: [0.0, 1.0, 0.0],
      toward: [0.0, 0.0, 1.0],
      center: [0.0; 3],
      size: 1.0,
      scale: [1.0; 3],
      offset: [0.0; 3],
    }
  }

  fn white() -> Material {
    Material {
      color: [1.0; 3],
      specular: None,
      glow: None,
    }
  }

  #[test]
  fn standard_lights_on_a_frontal_white_face() {
    // wolframscript: {0.886, 0.820, 0.863} ≈ (226, 209, 220).
    let c = shade(
      &white(),
      [0.0, 0.0, 1.0],
      [0.0; 3],
      &Lighting::standard(),
      &frontal(),
    );
    assert_eq!(c, (226, 209, 220));
  }

  #[test]
  fn back_faces_are_lit_like_front_faces() {
    let l = Lighting::standard();
    assert_eq!(
      shade(&white(), [0.0, 0.0, -1.0], [0.0; 3], &l, &frontal()),
      shade(&white(), [0.0, 0.0, 1.0], [0.0; 3], &l, &frontal())
    );
  }

  #[test]
  fn blinn_highlight() {
    // Black face, gray light along the view axis, normal tilted 0.4 rad:
    // wolframscript reads 0.392 = 0.5 cos(0.4)^3.
    let m = Material {
      color: [0.0; 3],
      specular: Some(([1.0; 3], 3.0)),
      glow: None,
    };
    let l = Lighting {
      name: None,
      lights: vec![view_dir_light(gray(0.5), [0.0, 0.0, 2.0])],
    };
    let c = shade(
      &m,
      [0.4f64.sin(), 0.0, 0.4f64.cos()],
      [0.0; 3],
      &l,
      &frontal(),
    );
    assert_eq!(c, (100, 100, 100));
  }
}
