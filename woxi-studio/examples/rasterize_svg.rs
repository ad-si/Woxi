//! Rasterize an SVG file to PNG (debugging aid).
//! Usage: cargo run --example rasterize_svg -- in.svg out.png [scale]

fn main() {
  let mut args = std::env::args().skip(1);
  let input = args.next().expect("input svg path");
  let output = args.next().expect("output png path");
  let scale: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2.0);

  let data = std::fs::read(&input).expect("read svg");
  let mut fontdb = resvg::usvg::fontdb::Database::new();
  fontdb.load_system_fonts();
  let opts = resvg::usvg::Options {
    fontdb: std::sync::Arc::new(fontdb),
    ..Default::default()
  };
  let tree = resvg::usvg::Tree::from_data(&data, &opts).expect("parse svg");
  let size = tree.size();
  let (w, h) = (
    (size.width() * scale).ceil() as u32,
    (size.height() * scale).ceil() as u32,
  );
  let mut pixmap = tiny_skia::Pixmap::new(w, h).expect("pixmap");
  pixmap.fill(tiny_skia::Color::WHITE);
  resvg::render(
    &tree,
    tiny_skia::Transform::from_scale(scale, scale),
    &mut pixmap.as_mut(),
  );
  pixmap.save_png(&output).expect("write png");
  println!("wrote {output} ({w}x{h})");
}
