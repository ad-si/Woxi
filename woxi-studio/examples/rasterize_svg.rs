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
  woxi::functions::image_ast::load_embedded_fonts(&mut fontdb);
  let data = woxi::functions::image_ast::synthesize_oblique_text(
    std::str::from_utf8(&data).expect("utf-8 svg"),
    &fontdb,
  )
  .into_owned();
  let opts =
    woxi::functions::image_ast::svg_render_options(std::sync::Arc::new(fontdb));
  let tree = resvg::usvg::Tree::from_str(&data, &opts).expect("parse svg");
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
