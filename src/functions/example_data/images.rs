//! The image collections: `"TestImage"`, `"AerialImage"`, `"Texture"`,
//! `"ColorTexture"`, `"TestImageSet"` and the frames of `"TestAnimation"`.
//!
//! Each entry's picture is drawn once (see [`scenes`](super::scenes)) at the
//! recorded size, channel count and type, rescaled to the recorded
//! per-channel mean and spread, cached, and every property — `"Data"`,
//! `"GrayLevels"`, `"Graphics"`, `"Thumbnail"`, … — is derived from it.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use super::raster::{Canvas, luma};
use super::{Collection, Entry, scenes};
use crate::functions::*;
use crate::syntax::ImageType;

/// The decoded shape of an `Sk["Image", {w, h}, channels, type, space, …]`
/// skeleton.
#[derive(Clone, Copy)]
pub struct ImageShape {
  pub width: usize,
  pub height: usize,
  pub channels: usize,
  pub image_type: ImageType,
  pub color_space: Option<&'static str>,
}

pub fn image_type_named(name: &str) -> ImageType {
  match name {
    "Bit" => ImageType::Bit,
    "Byte" => ImageType::Byte,
    "Bit16" => ImageType::Bit16,
    "Real64" => ImageType::Real64,
    _ => ImageType::Real32,
  }
}

pub fn color_space_named(name: &str) -> Option<&'static str> {
  Some(match name {
    "Grayscale" => "Grayscale",
    "RGB" => "RGB",
    "CMYK" => "CMYK",
    "HSB" => "HSB",
    "LAB" => "LAB",
    _ => return None,
  })
}

/// Decode an image skeleton (`Sk["Image", …]` or `Sk["Image3D", …]`).
pub fn shape_of(skeleton: &Expr) -> Option<ImageShape> {
  let Expr::FunctionCall { name, args } = skeleton else {
    return None;
  };
  if name != "Sk" {
    return None;
  }
  let dims: Vec<usize> = match args.get(1)? {
    Expr::List(d) => d
      .iter()
      .filter_map(|v| match v {
        Expr::Integer(n) => Some(*n as usize),
        _ => None,
      })
      .collect(),
    _ => return None,
  };
  let channels = match args.get(2)? {
    Expr::Integer(n) => *n as usize,
    _ => 1,
  };
  let image_type = match args.get(3) {
    Some(Expr::String(t)) => image_type_named(t),
    _ => ImageType::Byte,
  };
  let color_space = match args.get(4) {
    Some(Expr::String(s)) => color_space_named(s),
    _ => None,
  };
  Some(ImageShape {
    width: *dims.first()?,
    height: *dims.get(1)?,
    channels,
    image_type,
    color_space,
  })
}

/// The numbers of a list expression.
pub fn numbers(e: &Expr) -> Vec<f64> {
  match e {
    Expr::List(items) => items.iter().flat_map(numbers).collect(),
    Expr::Integer(n) => vec![*n as f64],
    Expr::Real(r) => vec![*r],
    _ => match crate::functions::math_ast::try_eval_to_f64(e) {
      Some(v) => vec![v],
      None => Vec::new(),
    },
  }
}

static CACHE: LazyLock<Mutex<HashMap<String, Expr>>> =
  LazyLock::new(|| Mutex::new(HashMap::new()));

/// Memoize `make` under `key`.
pub fn cached(key: String, make: impl FnOnce() -> Expr) -> Expr {
  if let Some(v) = CACHE.lock().unwrap().get(&key) {
    return v.clone();
  }
  let v = make();
  CACHE.lock().unwrap().insert(key, v.clone());
  v
}

/// Memoize a value that may not exist (only existing values are kept).
pub fn cached_opt(
  key: String,
  make: impl FnOnce() -> Option<Expr>,
) -> Option<Expr> {
  if let Some(v) = CACHE.lock().unwrap().get(&key) {
    return Some(v.clone());
  }
  let v = make()?;
  CACHE.lock().unwrap().insert(key, v.clone());
  Some(v)
}

/// The scene behind an entry: frames of a set or an animation share their
/// parent's scene family through their name (`Oxford2011_03`).
pub fn render(entry: &Entry, shape: ImageShape) -> Expr {
  cached(format!("{}/{}/image", entry.collection, entry.name), || {
    let mut canvas =
      scenes::draw(entry.collection, entry.name, shape.width, shape.height);
    apply_stats(
      &mut canvas,
      entry.stat("@Stats").as_ref(),
      shape.channels == 1,
    );
    canvas.into_image(shape.channels, shape.image_type, shape.color_space)
  })
}

/// Rescale to the recorded `{mean, sd}` (gray) or `{{means}, {sds}}` (RGB).
pub fn apply_stats(canvas: &mut Canvas, stats: Option<&Expr>, gray: bool) {
  let Some(Expr::List(s)) = stats else {
    return;
  };
  if s.len() != 2 {
    return;
  }
  let (means, sds) = (numbers(&s[0]), numbers(&s[1]));
  if means.is_empty() || sds.is_empty() {
    return;
  }
  canvas.match_stats(&means, &sds, gray || means.len() == 1);
}

/// Image fields: width, height, channels, type, data.
pub fn image_parts(
  img: &Expr,
) -> Option<(usize, usize, usize, ImageType, Arc<Vec<f64>>)> {
  match img {
    Expr::Image {
      width,
      height,
      channels,
      data,
      image_type,
      ..
    } => Some((
      *width as usize,
      *height as usize,
      *channels as usize,
      *image_type,
      data.clone(),
    )),
    _ => None,
  }
}

fn levels(t: ImageType) -> f64 {
  match t {
    ImageType::Bit => 1.0,
    ImageType::Bit16 => 65535.0,
    _ => 255.0,
  }
}

/// `"Data"`: the pixel values as integers in the image's range, rows top
/// first, `{h, w}` or `{h, w, c}`.
pub fn integer_data(img: &Expr) -> Expr {
  let Some((w, h, ch, t, data)) = image_parts(img) else {
    return Expr::List(crate::ExprList::default());
  };
  let m = levels(t);
  let px = |v: f64| Expr::Integer((v * m).round() as i128);
  rows(w, h, ch, &data, px)
}

fn rows(
  w: usize,
  h: usize,
  ch: usize,
  data: &[f64],
  f: impl Fn(f64) -> Expr,
) -> Expr {
  Expr::List(
    (0..h)
      .map(|y| {
        Expr::List(
          (0..w)
            .map(|x| {
              let i = (y * w + x) * ch;
              if ch == 1 {
                f(data[i])
              } else {
                Expr::List(
                  (0..ch).map(|c| f(data[i + c])).collect::<Vec<_>>().into(),
                )
              }
            })
            .collect::<Vec<_>>()
            .into(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  )
}

/// Luma per pixel, quantized like the image (Wolfram reports the gray
/// levels of a byte image as multiples of 1/255).
fn gray_values(img: &Expr) -> Option<(usize, usize, Vec<f64>)> {
  let (w, h, ch, t, data) = image_parts(img)?;
  let m = levels(t);
  let g = (0..w * h)
    .map(|p| {
      let i = p * ch;
      let v = if ch >= 3 {
        luma([data[i], data[i + 1], data[i + 2]])
      } else {
        data[i]
      };
      (v * m).round() / m
    })
    .collect();
  Some((w, h, g))
}

pub fn gray_levels(img: &Expr) -> Expr {
  let Some((w, h, g)) = gray_values(img) else {
    return Expr::List(crate::ExprList::default());
  };
  rows(w, h, 1, &g, Expr::Real)
}

pub fn rgb_color_array(img: &Expr) -> Expr {
  let Some((w, h, ch, _, data)) = image_parts(img) else {
    return Expr::List(crate::ExprList::default());
  };
  Expr::List(
    (0..h)
      .map(|y| {
        Expr::List(
          (0..w)
            .map(|x| {
              let i = (y * w + x) * ch;
              let (r, g, b) = if ch >= 3 {
                (data[i], data[i + 1], data[i + 2])
              } else {
                (data[i], data[i], data[i])
              };
              call(
                "RGBColor",
                vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
              )
            })
            .collect::<Vec<_>>()
            .into(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  )
}

/// `"Graphics"`: a `Raster` of the image's bytes (bottom row first, as
/// `Raster` wants) with the recorded `Raster` arguments and `Graphics`
/// options.
pub fn graphics(img: &Expr, options: Option<&Expr>) -> Expr {
  let Some((w, h, ch, t, data)) = image_parts(img) else {
    return Expr::List(crate::ExprList::default());
  };
  let m = levels(t);
  let flipped: Vec<f64> = (0..h)
    .rev()
    .flat_map(|y| data[y * w * ch..(y + 1) * w * ch].iter().copied())
    .collect();
  let ints = rows(w, h, ch, &flipped, |v| {
    Expr::Integer((v * m).round() as i128)
  });
  let array = call("NumericArray", vec![ints, str_expr("UnsignedInteger8")]);
  let (raster_args, graphics_options) = match options {
    Some(Expr::List(o)) if o.len() == 2 => {
      let ra: Vec<Expr> = match &o[0] {
        Expr::List(a) => a.iter().cloned().collect(),
        _ => Vec::new(),
      };
      let go: Vec<Expr> = match &o[1] {
        Expr::List(a) => a.iter().cloned().collect(),
        _ => Vec::new(),
      };
      (ra, go)
    }
    _ => (Vec::new(), Vec::new()),
  };
  let mut raster = vec![array];
  raster.extend(raster_args);
  let mut g = vec![call("Raster", raster)];
  g.extend(graphics_options);
  call("Graphics", g)
}

/// A resized copy of a generated image (thumbnails).
pub fn resized(img: &Expr, width: usize, height: usize) -> Expr {
  let Some((w, h, ch, t, data)) = image_parts(img) else {
    return img.clone();
  };
  let cs = match img {
    Expr::Image { color_space, .. } => *color_space,
    _ => None,
  };
  let mut canvas = Canvas::new(w, h, [0.0; 3]);
  for y in 0..h {
    for x in 0..w {
      let i = (y * w + x) * ch;
      let c = if ch >= 3 {
        [data[i], data[i + 1], data[i + 2]]
      } else {
        [data[i]; 3]
      };
      canvas.set(x, y, c);
    }
  }
  canvas.resized(width, height).into_image(ch, t, cs)
}

/// Export `value` once to a file in Woxi's temporary directory and return
/// its path, so `"FilePath"` names a file `Import` can read.
pub fn file_path(
  entry: &Entry,
  extension: &str,
  value: impl FnOnce() -> Expr,
) -> Expr {
  let dir = std::env::temp_dir()
    .join("woxi-exampledata")
    .join(entry.collection);
  let path = dir.join(format!("{}.{extension}", entry.name));
  let path_str = path.to_string_lossy().to_string();
  if !path.exists() {
    let _ = std::fs::create_dir_all(&dir);
    let export = call("Export", vec![str_expr(&path_str), value()]);
    let _ = crate::evaluator::evaluate_expr_to_expr(&export);
  }
  str_expr(&path_str)
}

/// The image of an entry, if its skeleton says it has one.
pub fn entry_image(entry: &Entry, column: &str) -> Option<Expr> {
  let shape = shape_of(&entry.skeleton(column)?)?;
  Some(render(entry, shape))
}

/// Properties shared by every single-image entry.
pub fn image_property(entry: &Entry, property: &str) -> Option<Expr> {
  let image = || entry_image(entry, "Image");
  Some(match property {
    "Image" => image()?,
    "ImageList" => Expr::List(vec![image()?].into()),
    "Data" => integer_data(&image()?),
    "GrayLevels" => gray_levels(&image()?),
    "RGBColorArray" => rgb_color_array(&image()?),
    "Graphics" => graphics(&image()?, entry.stat("@GraphicsOptions").as_ref()),
    "Thumbnail" => {
      let s = shape_of(&entry.skeleton("Thumbnail")?)?;
      cached(format!("{}/{}/thumb", entry.collection, entry.name), || {
        resized(&image().unwrap_or(Expr::Integer(0)), s.width, s.height)
      })
    }
    "FilePath" => {
      file_path(entry, "png", || image().unwrap_or(Expr::Integer(0)))
    }
    _ => return None,
  })
}

pub struct Images;

impl Collection for Images {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    image_property(entry, property?)
  }
}

/// `"TestImageSet"`: sets of photographs, each frame its own entry.
pub struct ImageSets;

impl Collection for ImageSets {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    let frames = || -> Vec<Expr> {
      match entry.exact("Frames") {
        Some(Expr::List(ref f)) => f.iter().cloned().collect(),
        _ => Vec::new(),
      }
    };
    let frame_entry_image = |spec: &Expr, column: &str| -> Option<Expr> {
      let Expr::List(pair) = spec else { return None };
      let Some(Expr::String(name)) = pair.get(1) else {
        return None;
      };
      let row = entry.table.row(name)?;
      let e = Entry {
        collection: entry.collection,
        name: &row.name,
        table: entry.table,
        row,
      };
      image_property(&e, column)
    };
    match property? {
      "ThumbnailList" => Some(Expr::List(
        frames()
          .iter()
          .filter_map(|f| frame_entry_image(f, "Thumbnail"))
          .collect::<Vec<_>>()
          .into(),
      )),
      "ThumbnailCollage" => {
        let shape = shape_of(&entry.skeleton("ThumbnailCollage")?)?;
        let thumbs: Vec<Expr> = frames()
          .iter()
          .filter_map(|f| frame_entry_image(f, "Thumbnail"))
          .collect();
        Some(collage(&thumbs, shape))
      }
      p => image_property(entry, p),
    }
  }
}

/// Thumbnails laid out on a dark grid of `shape`'s size.
fn collage(thumbs: &[Expr], shape: ImageShape) -> Expr {
  let mut canvas = Canvas::new(shape.width, shape.height, [0.2; 3]);
  let n = thumbs.len().max(1);
  let cols = ((n as f64 * shape.width as f64 / shape.height as f64)
    .sqrt()
    .ceil() as usize)
    .max(1);
  let rows = n.div_ceil(cols);
  let (cw, ch) = (shape.width / cols, shape.height / rows.max(1));
  for (k, t) in thumbs.iter().enumerate() {
    let Some((w, h, c, _, data)) = image_parts(t) else {
      continue;
    };
    let (ox, oy) = (
      (k % cols) * cw + cw.saturating_sub(w) / 2,
      (k / cols) * ch + ch.saturating_sub(h) / 2,
    );
    for y in 0..h.min(ch) {
      for x in 0..w.min(cw) {
        let i = (y * w + x) * c;
        let px = if c >= 3 {
          [data[i], data[i + 1], data[i + 2]]
        } else {
          [data[i]; 3]
        };
        if ox + x < shape.width && oy + y < shape.height {
          canvas.set(ox + x, oy + y, px);
        }
      }
    }
  }
  canvas.into_image(shape.channels, shape.image_type, shape.color_space)
}
