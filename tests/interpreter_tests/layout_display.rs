//! How layout and form wrappers (`Grid` options, `Style` decorations,
//! `Rotate`, `Magnify`, `Short`, `Shallow`, `PercentForm`, `MatrixForm`
//! headings, legends) are drawn in SVG.
use super::*;

mod layout_display {
  use super::*;

  fn svg_of(expr: &str) -> String {
    clear_state();
    interpret(&format!("ExportString[{expr}, \"SVG\"]")).unwrap()
  }

  /// The `<rect …>` elements of an SVG.
  fn rects(svg: &str) -> Vec<&str> {
    svg
      .match_indices("<rect")
      .map(|(i, _)| &svg[i..i + svg[i..].find('>').unwrap()])
      .collect()
  }

  /// The visible text of an SVG, all text nodes joined with whitespace
  /// dropped (box layouts set every token as its own `<text>`).
  fn flat_text(svg: &str) -> String {
    let body = match svg.find("</defs>") {
      Some(i) => &svg[i..],
      None => svg,
    };
    let mut out = String::new();
    let mut in_tag = false;
    for c in body.chars() {
      match c {
        '<' => in_tag = true,
        '>' => in_tag = false,
        c if !in_tag && !c.is_whitespace() => out.push(c),
        _ => {}
      }
    }
    out
  }

  /// The `<text …>…</text>` elements of an SVG.
  fn texts(svg: &str) -> Vec<&str> {
    svg
      .match_indices("<text")
      .map(|(i, _)| &svg[i..i + svg[i..].find("</text>").unwrap()])
      .collect()
  }

  /// Regression: a `Background` row list `{c1, …}` colored every row
  /// cyclically; like any explicit spec list it names only the first rows.
  #[test]
  fn grid_background_row_list_is_not_cyclic() {
    let svg =
      svg_of("Grid[{{1, 2}, {3, 4}, {5, 6}}, Background -> {None, {Red}}]");
    let red = rects(&svg)
      .iter()
      .filter(|r| r.contains("rgb(255,0,0)"))
      .count();
    assert_eq!(red, 1, "{svg}");
    // `{{c}}` is the cyclic form: every row.
    let svg =
      svg_of("Grid[{{1, 2}, {3, 4}, {5, 6}}, Background -> {None, {{Red}}}]");
    let red = rects(&svg)
      .iter()
      .filter(|r| r.contains("rgb(255,0,0)"))
      .count();
    assert_eq!(red, 3, "{svg}");
    // `{Red, Blue}`: rows 1 and 2, row 3 unpainted.
    let svg = svg_of(
      "Grid[{{1, 2}, {3, 4}, {5, 6}}, Background -> {None, {Red, Blue}}]",
    );
    let r = rects(&svg);
    assert_eq!(r.iter().filter(|r| r.contains("rgb(255,0,0)")).count(), 1);
    assert_eq!(r.iter().filter(|r| r.contains("rgb(0,0,255)")).count(), 1);
  }

  /// Regression: `ItemStyle -> {Automatic, {Directive[Bold, White]}}`
  /// styled every row instead of only the header row.
  #[test]
  fn grid_item_style_row_list_styles_only_named_rows() {
    let svg = svg_of(
      "Grid[{{\"h1\", \"h2\"}, {\"a\", \"b\"}, {\"c\", \"d\"}}, \
       ItemStyle -> {Automatic, {Directive[Bold, White]}}]",
    );
    let t = texts(&svg);
    let bold: Vec<_> = t.iter().filter(|t| t.contains("bold")).collect();
    assert_eq!(bold.len(), 2, "{svg}");
    assert!(bold.iter().all(|t| t.contains(">h")), "{svg}");
    // The cyclic `{{…}}` form styles every row.
    let svg = svg_of(
      "Grid[{{\"h1\", \"h2\"}, {\"a\", \"b\"}}, \
       ItemStyle -> {Automatic, {{Directive[Bold]}}}]",
    );
    assert_eq!(texts(&svg).iter().filter(|t| t.contains("bold")).count(), 4);
  }

  /// `Style[…, Background -> c]` paints a panel behind the text in a `Row`
  /// and a `Column`.
  #[test]
  fn style_background_paints_behind_text() {
    let svg = svg_of(
      "Row[{\"a\", Style[\"white on black\", White, Background -> Black]}]",
    );
    assert!(
      rects(&svg)
        .iter()
        .any(|r| r.contains("fill=\"rgb(0,0,0)\"")),
      "{svg}"
    );
    let svg = svg_of("Column[{Style[\"x\", Background -> Yellow], \"y\"}]");
    assert!(
      rects(&svg)
        .iter()
        .any(|r| r.contains("fill=\"rgb(255,255,0)\"")),
      "{svg}"
    );
  }

  /// Regression: a `Grid` cell's `Style[…, Background -> c]` (e.g. a
  /// "YES"/"NO" verdict label) dropped its panel; it is painted just behind
  /// the text, sized to it rather than filling the whole cell.
  #[test]
  fn grid_cell_style_background_paints_behind_text() {
    let svg = svg_of(
      "Grid[{{\"a\", Style[\"NO\", Large, Background -> Pink]}}, \
       ItemSize -> {{4, 10}, Automatic}]",
    );
    let rect = rects(&svg)
      .into_iter()
      .find(|r| r.contains("fill=\"rgb(255,128,128)\""))
      .unwrap_or_else(|| panic!("no pink panel: {svg}"));
    let width: f64 = rect
      .split("width=\"")
      .nth(1)
      .unwrap()
      .split('"')
      .next()
      .unwrap()
      .parse()
      .unwrap();
    // Two large glyphs wide, far narrower than the 10-em column.
    assert!(width > 15.0 && width < 60.0, "{rect}");
    // The panel sits under the text: it is drawn before it.
    let panel_at = svg.find(rect).unwrap();
    let text_at = svg.find(">NO</text>").unwrap();
    assert!(panel_at < text_at, "{svg}");
  }

  /// `Underlined`, `Struckthrough` and the `FontVariations` spellings draw
  /// their lines; "SmallCaps" sets the caps variant.
  #[test]
  fn style_text_decorations() {
    let svg = svg_of(
      "Column[{Style[\"u\", Underlined], Style[\"s\", Struckthrough], \
       Style[\"c\", FontVariations -> {\"CapsType\" -> \"SmallCaps\"}], \
       Style[\"v\", FontVariations -> {\"Underline\" -> True}], \
       Style[\"b\", Underlined, Struckthrough]}]",
    );
    let t = texts(&svg);
    let of = |label: &str| {
      *t.iter()
        .find(|t| t.ends_with(&format!(">{label}")))
        .unwrap_or_else(|| panic!("no {label}: {svg}"))
    };
    assert!(of("u").contains("text-decoration=\"underline\""));
    assert!(of("s").contains("text-decoration=\"line-through\""));
    assert!(svg.contains("font-variant=\"small-caps\""));
    assert!(svg.contains("font-size=\"80%\">C</tspan>"));
    assert!(of("v").contains("text-decoration=\"underline\""));
    assert!(of("b").contains("text-decoration=\"underline line-through\""));
    // Rows and grids draw them too.
    let svg = svg_of("Row[{Style[\"r\", Underlined]}]");
    assert!(svg.contains("text-decoration=\"underline\""), "{svg}");
    let svg = svg_of("Grid[{{Style[\"g\", Struckthrough]}}]");
    assert!(svg.contains("text-decoration=\"line-through\""), "{svg}");
  }

  /// `FontWeight` names beyond Bold map to their CSS weights.
  #[test]
  fn style_font_weight_names() {
    let svg = svg_of(
      "Grid[{{Style[\"t\", FontWeight -> \"Thin\"], \
       Style[\"k\", FontWeight -> \"Black\"], \
       Style[\"l\", FontWeight -> \"Light\"]}}]",
    );
    assert!(svg.contains("font-weight=\"100\""), "{svg}");
    assert!(svg.contains("font-weight=\"900\""), "{svg}");
    assert!(svg.contains("font-weight=\"300\""), "{svg}");
  }

  /// A bare `BarLegend` is drawn as a color bar with labeled ticks (it
  /// printed its own source); a contour list makes it discrete, and
  /// `LegendLabel` sits above the bar.
  #[test]
  fn bare_bar_legend_is_drawn() {
    let svg = svg_of("BarLegend[{\"Rainbow\", {0, 100}}]");
    assert!(svg.starts_with("<svg"), "{svg}");
    assert!(!svg.contains("BarLegend"), "{svg}");
    assert_eq!(rects(&svg).len(), 2, "{svg}");
    assert_eq!(svg.matches("<linearGradient ").count(), 1, "{svg}");
    assert!(svg.contains("fill=\"url(#woxiLegendGradient"), "{svg}");
    assert!(svg.contains("offset=\"0.000000\""), "{svg}");
    assert!(svg.contains("offset=\"1.000000\""), "{svg}");
    for label in [">0<", ">20<", ">100<"] {
      assert!(svg.contains(label), "missing {label}: {svg}");
    }
    let svg = svg_of("BarLegend[{\"BlueGreenYellow\", {0, 4}}, {1, 2, 3}]");
    // Four bands plus the outline.
    assert_eq!(rects(&svg).len(), 5, "{svg}");
    for label in [">1<", ">2<", ">3<"] {
      assert!(svg.contains(label), "missing {label}: {svg}");
    }
    assert!(!svg.contains(">0<") && !svg.contains(">4<"), "{svg}");
    // A list of colors is a gradient through them: a discrete bar with one
    // color per band shows exactly those, bottom to top.
    let svg =
      svg_of("BarLegend[{{Red, Green, Blue, Yellow}, {0, 4}}, {1, 2, 3}]");
    let fills: Vec<&str> = [
      "rgb(255,0,0)",
      "rgb(0,255,0)",
      "rgb(0,0,255)",
      "rgb(255,255,0)",
    ]
    .into_iter()
    .filter(|f| svg.contains(&format!("fill=\"{f}\"")))
    .collect();
    assert_eq!(fills.len(), 4, "{svg}");
    let svg =
      svg_of("BarLegend[{\"Rainbow\", {0, 1}}, LegendLabel -> \"Temp\"]");
    assert!(svg.contains(">Temp<"), "{svg}");
    // Inside a layout too.
    let svg = svg_of("Row[{BarLegend[{\"Rainbow\", {0, 1}}]}]");
    assert!(!svg.contains("BarLegend"), "{svg}");
  }

  #[test]
  fn svg_export_preserves_explicit_font_families() {
    let svg = svg_of(
      "Column[{Style[\"serif\", FontFamily -> \"Times\"], Style[\"mono\", FontFamily -> \"Courier\"]}]",
    );
    assert!(svg.contains("font-family=\"Times, serif\""), "{svg}");
    assert!(svg.contains("font-family=\"Courier, monospace\""), "{svg}");
    assert!(!svg.contains("svg text { font-family:"), "{svg}");
    assert!(!svg.contains("svg text[font-family*=\"Courier\"]"), "{svg}");
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn svg_axes_inherit_the_embedded_atkinson_default() {
    for expression in [
      "Plot[Sin[x],{x,0,1}]",
      "Plot3D[x+y,{x,0,1},{y,0,1},PlotPoints->3,MaxRecursion->0]",
    ] {
      let svg = svg_of(expression);
      assert!(svg.contains("<text "), "missing axis labels");
      assert!(
        svg
          .contains("@font-face { font-family: \"Atkinson Hyperlegible Next\"")
      );
      assert!(svg.contains(
        "svg { font-family: \"Atkinson Hyperlegible Next\", sans-serif; }"
      ));
      assert!(!svg.contains("text:not([font-family])"));
    }
    // Exercise the rasterizer used by the graphics review. An inherited
    // family must render identically to an explicit text family.
    // Export bakes the default family onto bare `<text>` elements;
    // stripping it again leaves them inheriting the stylesheet's.
    let explicit =
      svg_of("Plot3D[x+y,{x,0,1},{y,0,1},PlotPoints->3,MaxRecursion->0]");
    assert!(
      explicit
        .contains(" font-family=\"Atkinson Hyperlegible Next, sans-serif\">")
    );
    let svg = explicit.replace(
      " font-family=\"Atkinson Hyperlegible Next, sans-serif\">",
      ">",
    );
    let rasterize = woxi::functions::image_ast::rasterize_svg;
    let actual = rasterize(&svg, 72.0).unwrap();
    let expected = rasterize(&explicit, 72.0).unwrap();
    match (&actual, &expected) {
      (
        woxi::syntax::Expr::Image {
          width: aw,
          height: ah,
          channels: ac,
          data: a,
          ..
        },
        woxi::syntax::Expr::Image {
          width: bw,
          height: bh,
          channels: bc,
          data: b,
          ..
        },
      ) => {
        assert!(
          aw == bw && ah == bh && ac == bc && a == b,
          "axis font did not match explicit Atkinson rendering"
        );
      }
      _ => panic!("SVG rasterization did not produce images"),
    }
  }

  /// resvg never synthesizes an oblique face, so italic text (e.g.
  /// `Text[Style["a", Italic], …]` labels in `Graphics3D`) only renders
  /// slanted when a real italic face of the default family is embedded.
  /// Regression: italic and bold-italic labels rasterized pixel-identical
  /// to their upright forms.
  #[test]
  fn italic_text_rasterizes_slanted() {
    let raster = |style: &str| {
      let svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"120\" \
         height=\"40\"><text x=\"10\" y=\"30\" font-family=\"sans-serif\" \
         font-size=\"28\" {style}>lab</text></svg>"
      );
      match woxi::functions::image_ast::rasterize_svg(&svg, 72.0).unwrap() {
        woxi::syntax::Expr::Image { ref data, .. } => data.clone(),
        _ => panic!("SVG rasterization did not produce an image"),
      }
    };
    assert_ne!(raster(""), raster("font-style=\"italic\""));
    assert_ne!(
      raster("font-weight=\"bold\""),
      raster("font-weight=\"bold\" font-style=\"italic\"")
    );
    // The Graphics3D label path emits italic text for Style[…, Italic].
    let svg = svg_of("Graphics3D[{Text[Style[\"a\", 20, Italic], {0, 0, 0}]}]");
    assert!(svg.contains("font-style=\"italic\""));
  }

  /// Regression: a label with a character the requested face lacks (such
  /// as "ρ") was redrawn entirely in the first database face that had
  /// it, even when that face's weight or style differed (e.g. a bold
  /// oblique system font for a regular label).
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn svg_font_fallback_prefers_the_same_kind_of_face() {
    use resvg::usvg::fontdb::{Database, Weight};
    let mut db = Database::new();
    woxi::functions::image_ast::load_embedded_fonts(&mut db);
    let face = |family: &str, weight: u16| {
      db.faces()
        .find(|f| {
          f.weight == Weight(weight) && f.families.iter().any(|n| n.0 == family)
        })
        .unwrap_or_else(|| panic!("missing embedded {family} {weight}"))
        .id
    };
    let mono_bold = face("Atkinson Hyperlegible Mono", 700);
    let mono_regular = face("Atkinson Hyperlegible Mono", 400);
    let next_bold = face("Atkinson Hyperlegible Next", 700);
    let next_regular = face("Atkinson Hyperlegible Next", 400);
    assert!(db.face(mono_regular).unwrap().monospaced);
    assert!(!db.face(next_regular).unwrap().monospaced);
    let mut db = std::sync::Arc::new(db);
    let select = woxi::functions::image_ast::same_style_fallback_selector();
    // The closest weight wins over the kind of face: a regular base never
    // falls back to a bold face of its own kind.
    assert_eq!(select('a', &[next_bold], &mut db), Some(mono_bold));
    assert_eq!(select('a', &[mono_regular], &mut db), Some(next_regular));
    // Among faces of equal weight, the same kind wins over the database
    // order (the Mono faces come first): a proportional base falls back to
    // a proportional face and a monospaced to a monospaced.
    assert_eq!(
      select('a', &[next_bold, mono_bold], &mut db),
      Some(next_regular)
    );
    assert_eq!(
      select('a', &[mono_bold, next_bold], &mut db),
      Some(mono_regular)
    );
    // A character no face has yields no fallback.
    assert_eq!(select('\u{E000}', &[next_bold], &mut db), None);

    // The options used for rasterizing carry this selector.
    let opts = woxi::functions::image_ast::svg_render_options(db.clone());
    assert_eq!(
      (opts.font_resolver.select_fallback)('a', &[next_bold], &mut db),
      Some(mono_bold)
    );
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn serif_family_prefers_the_most_times_like_installed_face() {
    use woxi::functions::image_ast::preferred_serif_family;
    assert_eq!(
      preferred_serif_family(["DejaVu Serif", "Liberation Serif", "Arial"]),
      Some("Liberation Serif")
    );
    assert_eq!(
      preferred_serif_family(["liberation serif", "TIMES NEW ROMAN"]),
      Some("Times New Roman")
    );
    assert_eq!(
      preferred_serif_family(["Noto Serif", "DejaVu Sans"]),
      Some("Noto Serif")
    );
    assert_eq!(preferred_serif_family(["Arial", "DejaVu Sans"]), None);
    assert_eq!(preferred_serif_family([]), None);
  }

  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn generic_serif_text_uses_an_installed_serif_face() {
    use resvg::usvg::fontdb::{Database, Family, Query};
    let resolved_family = |db: &Database| {
      let id = db
        .query(&Query {
          families: &[Family::Serif],
          ..Query::default()
        })
        .expect("serif resolves");
      db.face(id).unwrap().families[0].0.clone()
    };

    // Without any installed serif face, "serif" falls back to the
    // embedded sans-serif instead of staying unresolved.
    let mut embedded_only = Database::new();
    woxi::functions::image_ast::load_embedded_fonts(&mut embedded_only);
    assert_eq!(
      resolved_family(&embedded_only),
      "Atkinson Hyperlegible Next"
    );

    // With system fonts, `FontFamily -> "Times"` text (measured with
    // Times-like advances) is drawn in the most Times-like installed
    // serif face, not in the wider sans-serif that would overrun it.
    let mut system = Database::new();
    system.load_system_fonts();
    woxi::functions::image_ast::load_embedded_fonts(&mut system);
    let expected = woxi::functions::image_ast::preferred_serif_family(
      system
        .faces()
        .flat_map(|f| f.families.iter().map(|(n, _)| n.as_str())),
    )
    .unwrap_or("Atkinson Hyperlegible Next");
    assert_eq!(
      resolved_family(&system).to_ascii_lowercase(),
      expected.to_ascii_lowercase()
    );
  }

  #[test]
  fn dataset_records_have_vertical_column_separators() {
    for expr in [
      "Dataset[{<|\"a\" -> 1, \"long header\" -> 2, \"c\" -> 3|>}]",
      "Dataset[{<|\"a\" -> 1, \"b\" -> 2|>, <|\"a\" -> 3, \"b\" -> 4|>}]",
    ] {
      let svg = svg_of(expr);
      let vertical = svg
        .match_indices("<line ")
        .filter(|(i, _)| {
          let line = &svg[*i..*i + svg[*i..].find('>').unwrap()];
          line.contains("y1=\"0\"")
            && !line.contains("y2=\"0.0\"")
            && !line.contains("y2=\"0\"")
        })
        .count();
      assert!(
        vertical >= if expr.contains("long header") { 4 } else { 3 },
        "missing column separators: {svg}"
      );
    }
  }

  /// `Rotate` turns text (not only pictures) inside a layout, and
  /// `Magnify` scales what it holds.
  #[test]
  fn rotate_and_magnify_text_in_row() {
    let svg = svg_of("Row[{Rotate[\"abc\", 45 Degree], \"x\"}]");
    assert!(svg.contains("rotate(-45"), "{svg}");
    assert!(!svg.contains("Rotate"), "{svg}");
    let svg = svg_of("Row[{Rotate[Framed[\"ab\"], Pi/2]}]");
    assert!(svg.contains("rotate(-90"), "{svg}");
    let small = svg_of("Row[{Magnify[\"abc\", 1]}]");
    let big = svg_of("Row[{Magnify[\"abc\", 3]}]");
    let width = |svg: &str| -> f64 {
      let i = svg.find("width=\"").unwrap() + 7;
      svg[i..i + svg[i..].find('"').unwrap()].parse().unwrap()
    };
    assert!(width(&big) > 2.5 * width(&small), "{small}\n{big}");
    assert!(!big.contains("Magnify"), "{big}");
  }

  /// `Shallow` is a display wrapper, not an unimplemented function: it
  /// stays as it is, and is drawn with skeletons below its depth and
  /// beyond its length.
  #[test]
  fn shallow_display() {
    clear_state();
    assert_eq!(
      interpret("Shallow[Nest[f, x, 5]]").unwrap(),
      "Shallow[f[f[f[f[f[x]]]]]]"
    );
    let svg = svg_of("Shallow[Nest[f, x, 20]]");
    assert_eq!(flat_text(&svg), "f[f[f[\u{226A}1\u{226B}]]]");
    let svg = svg_of("Shallow[Nest[{#, #} &, a, 6], 3]");
    assert_eq!(flat_text(&svg), "{{\u{226A}2\u{226B}},{\u{226A}2\u{226B}}}");
    let svg = svg_of("Shallow[Range[50], {2, 10}]");
    assert_eq!(flat_text(&svg), "{\u{226A}50\u{226B}}");
    let svg = svg_of("Shallow[Range[50]]");
    assert_eq!(flat_text(&svg), "{1,2,3,4,5,6,7,8,9,10,\u{226A}40\u{226B}}");
    // Atoms at the elided depth stay; compound parts there are skeletons.
    let svg = svg_of("Shallow[{1, {2, {3, 4}}}]");
    assert_eq!(flat_text(&svg), "{1,{2,{\u{226A}2\u{226B}}}}");
    // Inside a layout too.
    let svg = svg_of("Column[{Shallow[Nest[f, x, 20]]}]");
    assert_eq!(flat_text(&svg), "f[f[f[\u{226A}1\u{226B}]]]");
  }

  /// `Short` elides the middle of a long list, keeping parts from both
  /// ends (it drew all of them).
  #[test]
  fn short_elides_long_lists() {
    let svg = svg_of("Short[Range[1000]]");
    let text = flat_text(&svg);
    assert!(text.contains("10,11,\u{226A}978\u{226B},990,991"), "{text}");
    assert!(!text.contains("500"), "{text}");
    let svg = svg_of("Column[{Short[Range[1000]]}]");
    assert!(flat_text(&svg).contains("\u{226A}978\u{226B}"), "{svg}");
    // Something that fits is drawn whole.
    let svg = svg_of("Short[{1, 2, 3}]");
    assert_eq!(flat_text(&svg), "{1,2,3}");
  }

  /// `PercentForm` typesets numbers as percentages.
  #[test]
  fn percent_form_in_grid() {
    let svg = svg_of(
      "Grid[{{PercentForm[1234.5], PercentForm[-0.5], \
       PercentForm[98765.], PercentForm[2], PercentForm[x]}}]",
    );
    for s in [">123450%<", ">-50%<", ">9876500%<", ">200%<", ">x<"] {
      assert!(svg.contains(s), "missing {s}: {svg}");
    }
    assert!(!svg.contains("PercentForm"), "{svg}");
  }

  /// `MatrixForm[m, TableHeadings -> …]` draws the headings (it printed its
  /// own source).
  #[test]
  fn matrix_form_table_headings() {
    let svg = svg_of(
      "MatrixForm[{{1, 2}, {3, 4}}, \
       TableHeadings -> {{\"r1\", \"r2\"}, {\"c1\", \"c2\"}}]",
    );
    assert!(!svg.contains("MatrixForm"), "{svg}");
    for s in [">r1<", ">r2<", ">c1<", ">c2<", ">4<"] {
      assert!(svg.contains(s), "missing {s}: {svg}");
    }
  }

  /// `Framed[…, FrameStyle -> Dashed]` dashes its border.
  #[test]
  fn framed_dashed_frame_style() {
    let svg = svg_of("Row[{Framed[\"d\", FrameStyle -> Dashed]}]");
    assert!(svg.contains("stroke-dasharray=\"4.0,4.0\""), "{svg}");
    let svg = svg_of("Row[{Framed[\"d\"]}]");
    assert!(!svg.contains("stroke-dasharray"), "{svg}");
  }

  /// Regression: `Style[Framed[…], directives…]` was set as plain text and
  /// lost its frame; the directives now style the framed content instead.
  #[test]
  fn styled_framed_keeps_its_frame() {
    for expr in [
      "Style[Framed[\"ab\"], \"Label\", 14]",
      "Pane[Style[Framed[\"ab\"], 14], ImageSize -> {200, 50}]",
      "Row[{Style[Framed[\"ab\"], 14], \"c\"}]",
      "Column[{Style[Framed[\"ab\"], Red], \"c\"}]",
      "Grid[{{Style[Framed[\"ab\"], Red], \"c\"}}]",
    ] {
      let svg = svg_of(expr);
      assert_eq!(rects(&svg).len(), 1, "{expr}: {svg}");
      assert!(svg.contains(">ab</text>"), "{expr}: {svg}");
    }
    // The color reaches the framed text, not just the frame.
    let svg = svg_of("Grid[{{Style[Framed[\"ab\"], Red], \"c\"}}]");
    assert!(svg.contains("fill=\"rgb(255,0,0)\""), "{svg}");
    let svg = svg_of("Column[{Style[Framed[\"ab\"], Red], \"c\"}]");
    assert!(svg.contains("fill=\"rgb(255,0,0)\""), "{svg}");
  }

  /// The `font-family` of the `<text>` element that draws `text`.
  fn text_family<'a>(svg: &'a str, text: &str) -> &'a str {
    let end = svg
      .find(&format!(">{text}</text>"))
      .unwrap_or_else(|| panic!("no text {text}: {svg}"));
    let start = svg[..end].rfind("<text").unwrap();
    let tag = &svg[start..end];
    let from = tag.find("font-family=\"").unwrap() + "font-family=\"".len();
    &tag[from..from + tag[from..].find('"').unwrap()]
  }

  /// The font size of the `<text>` element that draws `text`.
  fn text_size(svg: &str, text: &str) -> f64 {
    let end = svg.find(&format!(">{text}</text>")).unwrap();
    let start = svg[..end].rfind("<text").unwrap();
    let tag = &svg[start..end];
    let from = tag.find("font-size=\"").unwrap() + "font-size=\"".len();
    tag[from..from + tag[from..].find('"').unwrap()]
      .parse()
      .unwrap()
  }

  /// Regression: the text of a `Framed`/`Panel` grid cell was set in the
  /// monospace face (and the default size) while the grid's other text
  /// cells use the grid's proportional face and size — also when the frame
  /// sits in a styled `Column` cell.
  #[test]
  fn framed_grid_cell_text_in_grid_face() {
    for (expr, framed) in [
      ("Grid[{{\"abc\", Framed[\"msg\"]}}]", "msg"),
      ("Grid[{{\"abc\", Panel[\"msg\"]}}]", "msg"),
      ("Grid[{{\"abc\", Framed[Style[\"msg\", 20]]}}]", "msg"),
      (
        "Grid[{{\"abc\"}, {Style[Column[{Grid[{{\"g\"}}], \
         Framed[\"msg\"]}], \"Label\", 16]}}]",
        "msg",
      ),
    ] {
      let svg = svg_of(expr);
      assert_eq!(
        text_family(&svg, framed),
        text_family(&svg, "abc"),
        "{expr}: {svg}"
      );
      assert!(!rects(&svg).is_empty(), "{expr}: {svg}");
    }
    // The grid's size reaches the framed text; the cell's own size wins.
    let svg = svg_of("Style[Grid[{{\"abc\", Framed[\"msg\"]}}], 16]");
    assert_eq!(text_size(&svg, "msg"), 16.0, "{svg}");
    let svg =
      svg_of("Style[Grid[{{\"abc\", Framed[Style[\"msg\", 20]]}}], 16]");
    assert_eq!(text_size(&svg, "msg"), 20.0, "{svg}");
    // A standalone frame keeps the monospace face.
    let svg = svg_of("Framed[\"msg\"]");
    assert!(text_family(&svg, "msg").contains("monospace"), "{svg}");
  }

  /// Regression: a `Row` grid cell holding a `Framed`/`Panel` was set as
  /// one line of text and lost the box.
  #[test]
  fn row_grid_cell_keeps_frame() {
    for expr in [
      "Grid[{{\"abc\", Row[{\"x\", Framed[\"y\"]}]}}]",
      "Grid[{{\"abc\", Row[{\"x\", Panel[\"y\"]}]}}]",
      "Grid[{{\"abc\", Style[Row[{\"x\", Framed[\"y\"]}], 16]}}]",
    ] {
      let svg = svg_of(expr);
      assert_eq!(rects(&svg).len(), 1, "{expr}: {svg}");
      assert!(svg.contains(">x</text>"), "{expr}: {svg}");
      assert!(svg.contains(">y</text>"), "{expr}: {svg}");
    }
  }

  /// Regression: `Framed[Style[text, FontFamily -> f]]` ignored the family;
  /// the text is now set (and the frame sized) in that face.
  #[test]
  fn framed_text_font_family() {
    let svg = svg_of("Framed[Style[\"fill\", FontFamily -> \"Times\"]]");
    assert!(text_family(&svg, "fill").starts_with("Times"), "{svg}");
    // The proportional text is narrower than four monospace cells.
    let mono = svg_of("Framed[\"fill\"]");
    let width = |svg: &str| -> f64 {
      let r = rects(svg)[0];
      let from = r.find("width=\"").unwrap() + "width=\"".len();
      r[from..from + r[from..].find('"').unwrap()]
        .parse()
        .unwrap()
    };
    assert!(width(&svg) < width(&mono), "{svg}\n{mono}");
  }

  /// Regression: a `Dividers` entry's thickness and dashing were dropped;
  /// only its color was drawn.
  #[test]
  fn grid_divider_thickness_and_dashing() {
    let svg = svg_of(
      "Grid[{{1, 2}, {3, 4}}, Dividers -> {{False, Directive[Red, Thick]}, \
       {False, Directive[Blue, Dashed]}}]",
    );
    assert!(
      svg.contains("stroke=\"rgb(255,0,0)\" stroke-width=\"2\""),
      "{svg}"
    );
    assert!(
      svg.contains(
        "stroke=\"rgb(0,0,255)\" stroke-width=\"1\" stroke-dasharray"
      ),
      "{svg}"
    );
  }

  /// `Item[…, Alignment -> Right]`, `Background` and `Frame` style their
  /// own cell.
  #[test]
  fn grid_item_options() {
    let svg = svg_of(
      "Grid[{{Item[\"r\", Alignment -> Right], \"a long cell\"}, \
       {\"another long cell\", Item[\"y\", Background -> Yellow]}, \
       {Item[\"f\", Frame -> True], \"d\"}}]",
    );
    assert!(
      texts(&svg)
        .iter()
        .any(|t| t.ends_with(">r") && t.contains("text-anchor=\"end\"")),
      "{svg}"
    );
    assert!(
      rects(&svg).iter().any(|r| r.contains("rgb(255,255,0)")),
      "{svg}"
    );
    assert!(
      rects(&svg).iter().any(|r| r.contains("fill=\"none\"")),
      "{svg}"
    );
  }

  /// `ItemSize -> {{w…}, {h…}}` fixes column widths (wrapping wider text)
  /// and gives rows their height.
  #[test]
  fn grid_item_size() {
    let svg = svg_of(
      "Grid[{{\"a\", \"b\"}, {\"wide content\", \"x\"}}, \
       ItemSize -> {{4, 12}, {2, 3}}]",
    );
    // "wide content" no longer fits 4 ems: it breaks onto two lines.
    assert!(svg.contains(">wide<") && svg.contains(">content<"), "{svg}");
    let wide = svg_of("Grid[{{\"a\", \"b\"}}, ItemSize -> {{4, 12}}]");
    let plain = svg_of("Grid[{{\"a\", \"b\"}}]");
    let width = |svg: &str| -> f64 {
      let i = svg.find("width=\"").unwrap() + 7;
      svg[i..i + svg[i..].find('"').unwrap()].parse().unwrap()
    };
    assert!(width(&wide) > 4.0 * width(&plain), "{wide}");
  }

  /// A flat `ItemSize -> {w, h, …}` with more than two entries still reads
  /// as `{w, h}`: every column is `w` ems wide, not just the first one.
  #[test]
  fn grid_item_size_flat_list_longer_than_two() {
    let cells = "{{\"ab\", \"c\", 1}, {2, 3, 4}}";
    let long = svg_of(&format!("Grid[{cells}, ItemSize -> {{6, 1, 1}}]"));
    let pair = svg_of(&format!("Grid[{cells}, ItemSize -> {{6, 1}}]"));
    let scalar = svg_of(&format!("Grid[{cells}, ItemSize -> 6]"));
    let width = |svg: &str| -> String {
      let i = svg.find("width=\"").unwrap() + 7;
      svg[i..i + svg[i..].find('"').unwrap()].to_string()
    };
    assert_eq!(width(&long), width(&pair), "{long}");
    assert_eq!(width(&long), width(&scalar), "{long}");
    // Same for a one-column `Column`.
    let col_long = svg_of("Column[{\"a\", \"b\"}, ItemSize -> {11, 2, 1}]");
    let col_pair = svg_of("Column[{\"a\", \"b\"}, ItemSize -> {11, 2}]");
    assert_eq!(width(&col_long), width(&col_pair), "{col_long}");
  }

  /// A heading merged across fixed-width columns with `SpanFromLeft` wraps
  /// to the width of the whole span, not to that of its first column.
  #[test]
  fn grid_item_size_span_wraps_to_span_width() {
    let svg = svg_of(
      "Grid[{{\"a heading of some words\", SpanFromLeft}, {\"x\", \"y\"}}, \
       ItemSize -> {{6, 10}}]",
    );
    assert!(svg.contains(">a heading of some words<"), "{svg}");
    assert!(!svg.contains("<tspan"), "{svg}");
    // Narrower than the text, the span still wraps it.
    let svg = svg_of(
      "Grid[{{\"a heading of some words\", SpanFromLeft}, {\"x\", \"y\"}}, \
       ItemSize -> {{3, 4}}]",
    );
    assert!(svg.contains("<tspan"), "{svg}");
    // A span over a content-sized column is not wrapped.
    let svg = svg_of(
      "Grid[{{\"a heading of some words\", SpanFromLeft}, {\"x\", \"y\"}}, \
       ItemSize -> {{3, Automatic}}]",
    );
    assert!(!svg.contains("<tspan"), "{svg}");
  }

  /// A short row ending in `SpanFromLeft` keeps spanning to the grid's
  /// right edge. Regression: the missing trailing columns stayed separate
  /// empty cells, so a framed heading row (`{"label", SpanFromLeft}` in a
  /// three-column grid) covered only two columns, was centered over them
  /// and had a divider before the last column.
  #[test]
  fn grid_trailing_span_from_left_reaches_last_column() {
    let short = svg_of(
      "Grid[{{\"aa\", \"bb\", \"cc\"}, {\"heading\", SpanFromLeft}, \
       {1, 2, 3}}, Frame -> All, ItemSize -> 4]",
    );
    let full = svg_of(
      "Grid[{{\"aa\", \"bb\", \"cc\"}, {\"heading\", SpanFromLeft, \
       SpanFromLeft}, {1, 2, 3}}, Frame -> All, ItemSize -> 4]",
    );
    assert_eq!(short, full);
    // The heading is centered across all three columns.
    let attr = |tag: &str, name: &str| -> f64 {
      let i = tag.find(&format!("{name}=\"")).unwrap() + name.len() + 2;
      tag[i..i + tag[i..].find('"').unwrap()].parse().unwrap()
    };
    let width = attr(&short, "width");
    let heading = short[..short.find(">heading<").unwrap()]
      .rsplit("<text")
      .next()
      .unwrap();
    assert!(
      (attr(heading, "x") - (width - 1.0) / 2.0).abs() < 1.0,
      "{short}"
    );
    // A short row without a trailing span is not merged.
    let ragged = svg_of(
      "Grid[{{\"aa\", \"bb\", \"cc\"}, {\"heading\", \"x\"}, {1, 2, 3}}, \
       Frame -> All, ItemSize -> 4]",
    );
    assert_ne!(ragged, short);
  }

  /// A string holding inline linear syntax (`\!\(…\)`) in a fixed-width
  /// `ItemSize` column is a typeset formula: it is not word-wrapped at its
  /// markup's spaces, which would split the box structure.
  #[test]
  fn grid_item_size_keeps_linear_syntax_whole() {
    let svg = svg_of(
      "Grid[{{\"\\!\\(TraditionalForm\\`cos \\[Phi]\\)\", \"x\"}}, \
       ItemSize -> {{8, All}, 2}]",
    );
    assert!(svg.contains(">cos \u{3d5}<"), "{svg}");
    assert!(!svg.contains("<tspan"), "{svg}");
  }

  /// `Highlighted` defaults to Wolfram's yellow and honors `Background`.
  #[test]
  fn highlighted_background_option() {
    let svg = svg_of("Row[{Highlighted[\"a\", Background -> LightGreen]}]");
    assert!(svg.contains("rgb(224,255,224)"), "{svg}");
    let svg = svg_of("Row[{Highlighted[\"a\"]}]");
    assert!(svg.contains("rgb(254,236,84)"), "{svg}");
  }

  /// `Pane[…, ImageSize -> {w, h}, Alignment -> {h, v}]` reserves the area
  /// and places its text inside it.
  #[test]
  fn pane_image_size_and_alignment() {
    let svg = svg_of(
      "Framed[Pane[\"text\", ImageSize -> {120, 60}, Alignment -> {Left, Top}]]",
    );
    assert!(svg.contains("width=\"130\" height=\"70\""), "{svg}");
    let t = texts(&svg);
    let x: f64 = {
      let s = t[0];
      let i = s.find("x=\"").unwrap() + 3;
      s[i..i + s[i..].find('"').unwrap()].parse().unwrap()
    };
    assert!(x < 40.0, "text should sit at the left: {svg}");
  }

  /// `Pane[text, w]` / `Pane[text, ImageSize -> w]` sets a width alone:
  /// the box is exactly `w` wide, its text wraps there, and the height
  /// follows the lines. Regression: a framed width-only pane was one line
  /// high and its text ran past the edge, cut off.
  #[test]
  fn pane_width_only_wraps_text() {
    let long = "Column[{Row[{12.5, \" units of the first kind in the sample\"}], Row[{7, \" units of the second kind left over at the end\"}]}]";
    for pane in [
      format!("Pane[{long}, 200]"),
      format!("Pane[{long}, ImageSize -> 200]"),
      format!("Text@Pane[{long}, ImageSize -> 200]"),
    ] {
      let svg = svg_of(&pane);
      assert!(svg.starts_with("<svg width=\"200\""), "{pane}: {svg}");
      // Four lines of text, not the two unwrapped ones.
      assert!(texts(&svg).len() >= 6, "{pane}: {svg}");
      let framed = svg_of(&format!("Framed[{pane}]"));
      assert!(framed.contains("width=\"210\""), "{pane}: {framed}");
      let i = framed.find("height=\"").unwrap() + 8;
      let h: f64 = framed[i..i + framed[i..].find('"').unwrap()]
        .parse()
        .unwrap();
      assert!(h > 4.0 * 14.0, "framed pane too short: {framed}");
    }
    // Short text keeps one line in a box of the full width.
    let svg = svg_of("Pane[\"abc\", 100]");
    assert!(svg.starts_with("<svg width=\"100\" height=\"15\""), "{svg}");
  }

  /// `Panel[expr, "Title"]` draws its title above the box, and a panel is
  /// the light-gray box at the top level too.
  #[test]
  fn panel_title_and_top_level() {
    let svg = svg_of("Row[{Panel[\"body\", \"Title\"]}]");
    let title = svg.find(">Title<").expect("title drawn");
    let body = svg.find(">body<").expect("body drawn");
    assert!(title < body, "{svg}");
    let svg = svg_of("Panel[\"x\"]");
    assert!(svg.contains("fill=\"rgb(245,245,245)\""), "{svg}");
    assert!(!svg.contains("Panel"), "{svg}");
  }

  /// `PointLegend`'s `LegendMarkers` glyphs, and `SwatchLegend`'s
  /// `LegendLabel` title.
  #[test]
  fn legend_markers_and_label() {
    let svg = svg_of(
      "PointLegend[{Red, Blue}, {\"a\", \"b\"}, \
       LegendMarkers -> {\"\\[FilledSquare]\", \"\\[FilledUpTriangle]\"}]",
    );
    assert!(
      svg.contains("\u{25A0}") && svg.contains("\u{25B2}"),
      "{svg}"
    );
    let svg = svg_of(
      "SwatchLegend[{Red, Blue}, {\"a\", \"b\"}, LegendLabel -> \"Colors\"]",
    );
    assert!(svg.contains(">Colors<"), "{svg}");
    assert!(svg.contains("stroke-opacity=\"0.3\""), "{svg}");
  }

  /// `Column[…, Frame -> All]` rules a line between its items, in black.
  #[test]
  fn column_frame_all_dividers() {
    let svg = svg_of("Column[{\"a\", \"b\", \"c\"}, Frame -> All]");
    assert_eq!(svg.matches("<line").count(), 2, "{svg}");
    assert!(svg.contains("stroke=\"black\""), "{svg}");
  }

  /// Regression: the rules between a `Column`'s items are drawn in its
  /// `FrameStyle`, like the outer frame, not always in black.
  #[test]
  fn column_frame_all_dividers_use_frame_style() {
    let svg =
      svg_of("Column[{\"a\", \"b\", \"c\"}, Frame -> All, FrameStyle -> Red]");
    let lines: Vec<&str> = svg
      .match_indices("<line")
      .map(|(i, _)| &svg[i..i + svg[i..].find('>').unwrap()])
      .collect();
    assert_eq!(lines.len(), 2, "{svg}");
    assert!(
      lines.iter().all(|l| l.contains("stroke=\"rgb(255,0,0)\"")),
      "{svg}"
    );
    assert!(!svg.contains("stroke=\"black\""), "{svg}");
  }

  /// Regression: `BaseStyle -> {…}` on a `Column` or `Row` styles all its
  /// items — nested grids included — as an enclosing `Style` would.
  #[test]
  fn column_and_row_base_style_reach_every_item() {
    for head in ["Column", "Row"] {
      let svg = svg_of(&format!(
        "{head}[{{\"a\", Grid[{{{{1, 2}}}}, Frame -> All]}}, \
         BaseStyle -> {{FontFamily -> \"Times\", FontSize -> 20}}]"
      ));
      let ts = texts(&svg);
      assert!(ts.len() >= 3, "{head}: {svg}");
      for t in ts {
        assert!(t.contains("font-family=\"Times"), "{head}: {t}");
        assert!(t.contains("font-size=\"20"), "{head}: {t}");
      }
    }
  }

  /// Regression: a power of a subscripted symbol typesets as a
  /// `SubsuperscriptBox` — the exponent stacked over the index, pulled
  /// back over the narrower script instead of following it.
  #[test]
  fn power_of_subscript_stacks_the_scripts() {
    let svg = svg_of("Row[{Subscript[x, 1]^k, \" + \", Subscript[x, 12]^2}]");
    let ts = texts(&svg);
    let first = ts.iter().find(|t| t.contains(">k<")).expect("x1^k");
    assert!(
      first.contains(
        "<tspan baseline-shift=\"super\" font-size=\"70%\">k</tspan>\
         <tspan dx=\"-0.5em\" baseline-shift=\"sub\" font-size=\"70%\">1</tspan>"
      ),
      "{first}"
    );
    let second = ts.iter().find(|t| t.contains(">12<")).expect("x12^2");
    // The narrower exponent goes first; the wider index is pulled back
    // under it by the exponent's width.
    assert!(
      second.contains(
        "<tspan baseline-shift=\"super\" font-size=\"70%\">2</tspan>\
         <tspan dx=\"-0.5em\" baseline-shift=\"sub\" font-size=\"70%\">12</tspan>"
      ),
      "{second}"
    );
  }

  /// Regression: a fixed `ItemSize` width narrower than a nested layout
  /// item widens the column instead of clipping the item at both sides.
  #[test]
  fn column_item_size_widens_for_a_nested_layout() {
    let svg = svg_of(
      "Column[{Column[{Row[{\"aaaaaaaaaaaaaaaaaaaa\"}], 1}]}, ItemSize -> 2]",
    );
    let outer_w: f64 = svg
      .split("width=\"")
      .nth(1)
      .and_then(|s| s.split('"').next())
      .and_then(|s| s.parse().ok())
      .expect("outer width");
    let inner = svg.match_indices("<svg x=\"").next().expect("nested svg").0;
    let inner_tag = &svg[inner..inner + svg[inner..].find('>').unwrap()];
    let x: f64 = inner_tag
      .split("x=\"")
      .nth(1)
      .and_then(|s| s.split('"').next())
      .and_then(|s| s.parse().ok())
      .unwrap();
    let w: f64 = inner_tag
      .split("width=\"")
      .nth(1)
      .and_then(|s| s.split('"').next())
      .and_then(|s| s.parse().ok())
      .unwrap();
    assert!(x >= 0.0, "{inner_tag}");
    assert!(x + w <= outer_w + 0.5, "{svg}");
  }

  /// A wider exponent is drawn after the index and pulled back over it.
  #[test]
  fn subsuperscript_wider_exponent_follows_the_index() {
    let svg = svg_of("Row[{Subsuperscript[a, 1, 23]}]");
    assert!(
      svg.contains(
        "<tspan baseline-shift=\"sub\" font-size=\"70%\">1</tspan>\
         <tspan dx=\"-0.5em\" baseline-shift=\"super\" font-size=\"70%\">23</tspan>"
      ),
      "{svg}"
    );
  }

  #[test]
  fn letter_overscripts_are_centered_above_the_base() {
    let svg = svg_of("Row[{Overscript[\"\\[LongRightArrow]\", \"f\"]}]");
    let f = texts(&svg)
      .into_iter()
      .find(|t| t.ends_with(">f"))
      .expect("overscript");
    assert!(f.contains("text-anchor=\"middle\""), "{f}");
    assert!(f.contains("font-size=\"10\""), "{f}");
    assert!(!f.contains("dx="), "{f}");
  }

  #[test]
  fn long_monospace_layouts_reserve_the_actual_glyph_advance() {
    for head in ["Column", "Row"] {
      let svg = svg_of(&format!(
        "{head}[{{Style[StringRepeat[\"W\", 100], Background -> Yellow]}}]"
      ));
      let rect = rects(&svg)
        .into_iter()
        .find(|r| r.contains("rgb(255,255,0)"))
        .expect("background");
      let width: f64 = rect
        .split("width=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .parse()
        .unwrap();
      assert!(
        width >= 100.0 * 14.0 * 0.632 - 0.1,
        "background clips glyphs: {rect}"
      );
    }
  }

  /// A text `Underoverscript` stacks its limits around the base.
  #[test]
  fn underoverscript_text_in_row() {
    let svg = svg_of("Row[{Underoverscript[\"S\", \"i=1\", \"n\"]}]");
    assert!(!svg.contains("Underoverscript"), "{svg}");
    let n = svg.find(">n<").expect("upper limit");
    let s = svg.find(">S<").expect("base");
    let i = svg.find(">i=1<").expect("lower limit");
    assert!(n < s && s < i, "{svg}");
  }
  #[test]
  fn held_standard_roots_and_matrix_baselines_typeset() {
    let svg = svg_of("StandardForm[HoldForm[Sqrt[x/(1+x)]]]");
    assert!(!svg.contains(">Sqrt<"), "root was printed as a call");
    let svg =
      svg_of("TraditionalForm[HoldForm[s == MatrixForm[{{a,b},{c,d}}]]]");
    assert!(svg.contains(">s</text>"));
  }

  #[test]
  fn bare_series_filling_rules_delegate_to_list_form() {
    for filling in ["1 -> {2}", "2 -> {1}"] {
      let expr = |f: &str| {
        format!(
          "ListPlot[{{Range[8], Range[8]/2}}, Joined -> {{True,False}}, Filling -> {f}]"
        )
      };
      assert_eq!(
        svg_of(&expr(filling)),
        svg_of(&expr(&format!("{{{filling}}}")))
      );
    }
  }

  #[test]
  fn spectral_images_keep_channels_while_display_uses_rgb_composite() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ColorConvert[Image[{{{1,0,0,1/2}}}], \"RGB\"]]")
        .unwrap(),
      "{{{1., 0., 0.5}}}"
    );
    assert_eq!(
      interpret("ImageChannels[Image[{{{1,0,0,1/2}}}]]").unwrap(),
      "4"
    );
    assert_eq!(
      interpret("ImageColorSpace[Image[{{{1,0,0,1/2}}}]]").unwrap(),
      "Automatic"
    );
    assert_eq!(interpret("ImageChannels[ColorConvert[Image[{{{1,0,0,1/2}}}, ColorSpace -> \"RGB\"], \"RGB\"]]").unwrap(), "4");
    assert_eq!(interpret("Table[With[{im=Image[{{Table[N[Mod[k,3]/2],{k,n}]}}]},ExportString[im,\"SVG\"]===ExportString[ColorConvert[im,\"RGB\"],\"SVG\"]],{n,4,9}]").unwrap(), "{True, True, True, True, True, True}");
  }

  #[test]
  fn zero_order_height_fields_are_horizontal_plates() {
    clear_state();
    assert_eq!(
      interpret(
        "Head[ListPlot3D[Table[x+y,{x,4},{y,4}],InterpolationOrder->0]]"
      )
      .unwrap(),
      "Graphics3D"
    );
    let svg = svg_of(
      "ListPlot3D[Table[x+y,{x,4},{y,4}],InterpolationOrder->0,Mesh->None]",
    );
    assert!(svg.contains("<polygon"));
  }

  #[test]
  fn spherical_surfaces_have_smooth_shading() {
    let svg = svg_of(
      "SphericalPlot3D[1+Sin[3p] Sin[t]/3,{t,0,Pi},{p,0,2Pi},PlotPoints->12,Mesh->None]",
    );
    assert!(
      svg.contains("<linearGradient"),
      "surface remained flat shaded"
    );
  }

  #[test]
  fn integration_monitor_uses_an_embedded_interior_rule() {
    clear_state();
    assert_eq!(interpret("Length[Reap[NIntegrate[Sin[x]^2,{x,0,2 Pi},EvaluationMonitor:>Sow[x]]][[2,1]]]").unwrap(), "77");
    let value: f64 = interpret("NIntegrate[Sin[x]^2,{x,0,2 Pi}]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((value - std::f64::consts::PI).abs() < 1e-10);
    for end in [1, 3, 5] {
      let value: f64 =
        interpret(&format!("NIntegrate[1/Sqrt[x],{{x,0,{end}}}]"))
          .unwrap()
          .parse()
          .unwrap();
      assert!((value - 2.0 * (end as f64).sqrt()).abs() < 1e-8);
    }
  }

  #[test]
  fn density_mesh_and_rounded_legend_endpoints_are_rendered() {
    let svg = svg_of(
      "ListDensityPlot[Table[Sin[x y],{x,0,3,1/4},{y,0,3,1/4}],Mesh->4]",
    );
    assert!(svg.contains("<polyline"), "missing density mesh");
    let svg = svg_of("BarLegend[{\"Rainbow\",{0.0003,0.999}}]");
    assert!(svg.contains(">0</text>") && svg.contains(">1.0</text>"));
  }

  #[test]
  fn small_caps_are_painted_in_all_text_layouts() {
    for head in ["Row", "Column"] {
      let svg = svg_of(&format!(
        "{head}[{{Style[\"Abc\",FontVariations->{{\"CapsType\"->\"SmallCaps\"}}]}}]"
      ));
      assert!(svg.contains(">B</tspan>") && svg.contains(">C</tspan>"));
    }
  }

  #[test]
  fn marketing_surfaces_include_rear_planes() {
    let svg = svg_of(
      "Plot3D[x+y,{x,-1,1},{y,-1,1},PlotTheme->\"Marketing\",PlotPoints->5,MaxRecursion->0]",
    );
    assert_eq!(svg.matches("fill=\"rgb(110,110,110)\"").count(), 3);
  }

  #[test]
  fn vertex_colors_keep_independent_rgb_gradients() {
    let svg = svg_of(
      "Graphics3D[Polygon[{{0,0,0},{1,0,0},{1,1,1},{0,1,1}},VertexColors->{Red,Green,Blue,Yellow}]]",
    );
    assert!(
      svg.contains("<pattern"),
      "vertex colors collapsed to a one-dimensional fit"
    );
  }

  #[test]
  fn finite_labeling_size_keeps_short_point_labels() {
    for size in [10, 30, 1000] {
      let svg = svg_of(&format!(
        "ListPlot[{{1,2,3}}->{{\"a\",\"b\",\"c\"}},LabelingSize->{size}]"
      ));
      assert!(svg.contains(">a</text>"));
    }
  }

  #[test]
  fn standard_form_grid_cells_keep_native_radicals() {
    let svg =
      svg_of("Grid[{{StandardForm[HoldForm[Sqrt[x/(1+x)]]]}},Frame->All]");
    assert!(
      !svg.contains("Sqrt["),
      "StandardForm radical became source text"
    );
    assert!(
      svg.contains("stroke-linejoin=\"round\""),
      "missing radical outline"
    );
  }

  #[test]
  fn mathematical_legends_use_native_fraction_bars_and_bounds() {
    for expression in [
      "DiscretePlot[{1/n,1/n^2,(-1)^n/n},{n,1,8},PlotLegends->\"Expressions\"]",
      "Plot[{x,x^2},{x,0,1},PlotLegends->{TraditionalForm[(a+b+c)/d],TraditionalForm[Sqrt[x]]}]",
    ] {
      let svg = svg_of(expression);
      assert!(
        svg.contains("<line x1=\"0\" y1="),
        "legend lost its native fraction bar"
      );
      assert!(
        !svg.contains("text-decoration=\"overline\""),
        "legend used a text-decoration fraction"
      );
      assert!(
        !svg.contains('\u{E000}'),
        "internal label marker leaked into the SVG"
      );
    }
  }

  #[test]
  fn binomial_plot_titles_reserve_their_native_height() {
    let svg = svg_of("ListPlot[Range[4],PlotLabel->Binomial[n,k]]");
    assert!(
      !svg.contains("dx=\"0.150ch\""),
      "binomial title used inline pair markup"
    );
    assert!(
      svg.contains("stroke-width=\"0.56\""),
      "binomial title lost its native brackets"
    );
  }
  #[test]
  fn global_adaptive_singularities_switch_to_double_exponential_after_refinement()
   {
    clear_state();
    assert_eq!(interpret("Length[Reap[NIntegrate[1/Abs[Sqrt[x]],{x,-1,0,1},EvaluationMonitor:>Sow[x]]][[2,1]]]").unwrap(),"262");
    assert_eq!(interpret("p=Reap[NIntegrate[1/Abs[Sqrt[x]],{x,-1,0,1},EvaluationMonitor:>Sow[x]]][[2,1]];Sign /@ p[[{23,45,67,89,111,133,155,177,199}]]").unwrap(),"{1, -1, -1, 1, 1, -1, -1, 1, 1}");
    for power in ["1/4", "1/2", "3/4"] {
      let value: f64 =
        interpret(&format!("NIntegrate[x^(-({power})),{{x,0,1}}]"))
          .unwrap()
          .parse()
          .unwrap();
      let expected: f64 = interpret(&format!("N[1/(1-({power}))]"))
        .unwrap()
        .parse()
        .unwrap();
      assert!(
        (value - expected).abs() < 1e-8,
        "failed singular exponent {power}: {value}"
      );
    }
  }

  #[test]
  fn calendar_and_surface_axes_have_minor_ticks() {
    let date = svg_of("DateListPlot[{{{2024,1,1},1},{{2024,7,1},2}}]");
    let date_none =
      svg_of("DateListPlot[{{{2024,1,1},1},{{2024,7,1},2}},FrameTicks->None]");
    assert!(
      date.matches("<line ").count() > date_none.matches("<line ").count() + 10
    );
    let surface = svg_of(
      "Plot3D[x+y,{x,0,1},{y,0,1},PlotPoints->3,MaxRecursion->0,Mesh->None]",
    );
    let no_ticks = svg_of(
      "Plot3D[x+y,{x,0,1},{y,0,1},PlotPoints->3,MaxRecursion->0,Mesh->None,Ticks->None]",
    );
    assert!(
      surface.matches("<line ").count()
        > no_ticks.matches("<line ").count() + 30
    );
  }

  #[test]
  fn custom_arrowhead_disks_fit_automatic_bounds() {
    let attr = |tag: &str, name: &str| -> f64 {
      let start = tag.find(&format!("{name}=\"")).unwrap() + name.len() + 2;
      tag[start..].split('"').next().unwrap().parse().unwrap()
    };
    for position in [0, 1] {
      let svg = svg_of(&format!(
        "Graphics[{{Arrowheads[{{{{0.15,{position},Graphics[{{Blue,Disk[{{0,0}},1/2]}}]}}}}],Arrow[{{{{0,0}},{{3,0}}}}]}}]"
      ));
      let root = svg.split('>').next().unwrap();
      let (width, height) = (attr(root, "width"), attr(root, "height"));
      for disk in svg.split("<ellipse ").skip(1) {
        let tag = disk.split('>').next().unwrap();
        let (cx, cy, rx, ry) = (
          attr(tag, "cx"),
          attr(tag, "cy"),
          attr(tag, "rx"),
          attr(tag, "ry"),
        );
        assert!(
          cx >= rx && cy >= ry && cx + rx <= width && cy + ry <= height,
          "custom head clipped: {tag}"
        );
      }
      assert!(svg.contains("<ellipse "), "custom disk missing");
    }
  }
  #[test]
  fn double_exponential_preserves_distances_from_shifted_endpoints() {
    for expression in ["1/Sqrt[1-x]", "1/Sqrt[1+x]"] {
      let bounds = if expression.contains("1-x") {
        "{x,0,1}"
      } else {
        "{x,-1,0}"
      };
      let value: f64 = interpret(&format!("NIntegrate[{expression},{bounds}]"))
        .unwrap()
        .parse()
        .unwrap();
      assert!(
        (value - 2.0).abs() < 1e-10,
        "lost endpoint distance for {expression}: {value}"
      );
    }
  }

  #[test]
  fn reviewed_custom_chart_elements_use_graphics_primitives() {
    for radius in ["1/10", "1/4"] {
      let svg = svg_of(&format!(
        "BarChart[{{2,5,-3}},ChartElementFunction->Function[{{bounds,value,meta}},{{EdgeForm[Black],Rectangle[{{bounds[[1,1]]+0.08,bounds[[2,1]]}},{{bounds[[1,2]]-0.08,bounds[[2,2]]}},RoundingRadius->{radius}]}}],ChartStyle->{{Blue,Orange,Red}}]"
      ));
      assert_eq!(
        svg.matches(" rx=\"").count(),
        3,
        "rounded bars missing: {svg}"
      );
      assert!(svg.contains("rgb(0,0,255)"), "custom bars lost chart color");
    }
    let svg =
      svg_of("BarChart[{2,5},ChartElementFunction->(Disk[Mean /@ #1,0.1]&)]");
    assert_eq!(svg.matches("<ellipse ").count(), 2);
  }

  #[test]
  fn reviewed_box_elements_keep_independent_styles_and_mean_intervals() {
    for origin in ["Bottom", "Left", "Top", "Right"] {
      let expr = format!(
        "BoxWhiskerChart[{{{{1,2,2,3,3,4,4,5,6,9}},{{2,3,4,5,5,6,6,7,8,12}}}},{{\"Basic\",{{\"Whiskers\",Dashed}},{{\"Fences\",0.5,Red}},{{\"MeanDiamond\",0.5,Gray}}}},BarOrigin->{origin}]"
      );
      let svg = svg_of(&expr);
      assert!(svg.contains("stroke-dasharray="), "whiskers not dashed");
      assert!(svg.contains("stroke=\"rgb(255,0,0)\""), "fences not red");
      assert_eq!(svg.matches("<polygon ").count(), 4, "mean diamonds missing");
      clear_state();
      let geometry =
        interpret(&format!("InputForm[Cases[{expr},_Polygon,Infinity]]"))
          .unwrap();
      assert!(
        geometry.contains("2.25624"),
        "incorrect mean confidence interval: {geometry}"
      );
    }
  }

  #[test]
  fn reviewed_projection_options_change_scene_geometry() {
    let scene = "{{LightBlue,Cuboid[]},{Orange,Sphere[{1.5,1.5,1.5},1/2]}}";
    let ortho = svg_of(&format!(
      "Graphics3D[{scene},ViewProjection->\"Orthographic\",ViewPoint->{{2,-3,1.5}}]"
    ));
    let perspective = svg_of(&format!(
      "Graphics3D[{scene},ViewProjection->\"Perspective\",ViewPoint->{{2,-3,1.5}}]"
    ));
    assert_ne!(ortho, perspective);
    assert_eq!(
      perspective,
      svg_of(&format!("Graphics3D[{scene},ViewPoint->{{2,-3,1.5}}]"))
    );
  }

  #[test]
  fn reviewed_edge_option_labels_unwrap_placed_and_keep_style() {
    for position in ["1/4", "3/4"] {
      let svg = svg_of(&format!(
        "Graph[{{1->2}},VertexCoordinates->{{{{0,0}},{{2,0}}}},EdgeLabels->{{DirectedEdge[1,2]->Placed[Style[\"rise\",Red],{position}]}}]"
      ));
      assert!(!svg.contains("Placed["), "{svg}");
      assert!(svg.contains("rise"));
      assert!(svg.contains("rgb(255,0,0)"), "label lost its style");
    }
  }

  #[test]
  fn reviewed_arrowheads_paint_after_their_shafts() {
    let svg = svg_of(
      "Graphics[{Arrowheads[{{0.1,0,Graphics[{Blue,Disk[]} ]},{0.1,1,Graphics[{Red,Disk[]} ]}}],Arrow[{{0,0},{3,0}}]}]",
    );
    let shaft = svg.find("<polyline ").unwrap();
    assert!(
      svg
        .match_indices("<ellipse ")
        .all(|(index, _)| index > shaft)
    );
  }

  #[test]
  fn reviewed_long_arrows_have_portable_vector_geometry() {
    for arrow in [
      "\\[LongRightArrow]",
      "\\[LongLeftArrow]",
      "\\[DoubleLongLeftRightArrow]",
    ] {
      let svg = svg_of(&format!("Row[{{Overscript[\"{arrow}\",\"f\"]}}]"));
      assert!(
        svg.contains("<path "),
        "arrow depends on a missing font glyph"
      );
      assert!(!svg.contains("⟶") && !svg.contains("⟵"));
    }
  }

  #[test]
  fn reviewed_finite_plot_label_size_truncates_instead_of_omitting() {
    for size in [10, 30, 1000] {
      let svg = svg_of(&format!(
        "Plot[{{Sin[x]+x,x Log[x]}},{{x,0,5}},PlotLabels->{{\"barcarole\",\"perilously\"}},LabelingSize->{size}]"
      ));
      if size < 1000 {
        assert!(svg.contains(">ba...</text>") && svg.contains(">pe...</text>"));
      } else {
        assert!(
          svg.contains(">barcarole</text>")
            && svg.contains(">perilously</text>")
        );
      }
      assert!(!svg.contains("FEPrivate"));
    }
  }

  #[test]
  fn reviewed_traditional_grid_arithmetic_uses_spaced_boxes() {
    let svg = svg_of(
      "Grid[{{TraditionalForm[HoldForm[(a+b)^2]],\"=\",TraditionalForm[HoldForm[a^2+2 a b+b^2]]}}]",
    );
    assert!(
      svg.contains(">+</text>"),
      "arithmetic was flattened into unspaced text"
    );
    assert!(!svg.contains("a+b"));
  }

  #[test]
  fn reviewed_padded_base_table_reserves_the_monospace_advance() {
    let svg = svg_of(
      "PaddedForm[BaseForm[TableForm[{{8,49},{123,5}}],2],8,NumberPadding->{\"0\",\"\"}]",
    );
    let root = svg.split('>').next().unwrap();
    let width: f64 = root
      .split("width=\"")
      .nth(1)
      .unwrap()
      .split('"')
      .next()
      .unwrap()
      .parse()
      .unwrap();
    for text in texts(&svg) {
      let x: f64 = text
        .split("x=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .parse()
        .unwrap();
      let right = x + 9.7 * 14.0 * 0.632;
      assert!(
        right <= width + 0.5,
        "table clips its base subscript: {text}, viewport {width}"
      );
    }
  }

  #[test]
  fn center_slice_density_preserves_exact_zeros_and_real_gradients() {
    let gradient = svg_of(
      "SliceDensityPlot3D[x+y+z,\"CenterPlanes\",{x,-1,1},{y,-1,1},{z,-1,1},PlotPoints->5]",
    );
    assert!(
      gradient.contains("<pattern"),
      "slice vertex colors lost their gradients"
    );
    clear_state();
    assert_eq!(
      interpret("Simplify[{x y z/.x->0,x y z/.y->0,x y z/.z->0}]").unwrap(),
      "{0, 0, 0}"
    );
  }

  #[test]
  fn nested_text_styles_inherit_outer_size_and_inner_color() {
    let svg = svg_of("Graphics[Text[Style[Style[\"label\",Red],18],{0,0}]]");
    assert!(svg.contains("fill=\"rgb(255,0,0)\""));
    assert!(svg.contains("font-size=\"18\""));
  }

  #[test]
  fn mean_diamond_confidence_intervals_follow_wolfram_rounding() {
    for (n, expected) in [
      (2, 2.15),
      (3, 1.8359738560230103),
      (5, 1.8172644276494268),
      (10, 2.1350624502966333),
    ] {
      clear_state();
      let error: f64 = interpret(&format!("g=BoxWhiskerChart[{{Range[{n}]}},{{\"Basic\",{{\"MeanDiamond\",0.5,Gray}}}}]; p=Last[Cases[g,Polygon[p_]:>p,Infinity]]; Max[p[[All,2]]]-Mean[Range[{n}]]")).unwrap().parse().unwrap();
      assert!(
        (error - expected).abs() < 1e-10,
        "{n}: {error} instead of {expected}"
      );
    }
  }

  #[test]
  fn custom_bar_callbacks_keep_exact_values_and_empty_metadata() {
    for data in ["{2,1/3}", "{{2,1/3},{3,1/2}}"] {
      let svg = svg_of(&format!(
        "BarChart[{data},ChartElementFunction->Function[{{b,v,m}},Text[If[ExactNumberQ[v]&&m==={{}},\"exact\",\"wrong\"],Mean /@ b]]]"
      ));
      assert!(svg.contains("exact"));
      assert!(!svg.contains("wrong"));
    }
  }

  /// `Factorial[n]` / `Factorial2[n]` box postfix (`n!`, `n!!`), with the
  /// operand bracketed when it binds no tighter than the factorial.
  #[test]
  fn factorial_boxes_postfix() {
    clear_state();
    for (code, boxes) in [
      ("ToBoxes[Factorial[n]]", "RowBox[{n, !}]"),
      ("ToBoxes[Factorial2[n]]", "RowBox[{n, !!}]"),
      ("ToBoxes[HoldForm[Factorial[7]]]", "RowBox[{7, !}]"),
      (
        "ToBoxes[HoldForm[(a + b)!]]",
        "RowBox[{RowBox[{(, RowBox[{a, +, b}], )}], !}]",
      ),
      (
        "ToBoxes[HoldForm[(-3)!]]",
        "RowBox[{RowBox[{(, RowBox[{-, 3}], )}], !}]",
      ),
      (
        "ToBoxes[HoldForm[(4!)!]]",
        "RowBox[{RowBox[{(, RowBox[{4, !}], )}], !}]",
      ),
      (
        "ToBoxes[HoldForm[f[x]!]]",
        "RowBox[{RowBox[{f, [, x, ]}], !}]",
      ),
    ] {
      assert_eq!(interpret(code).unwrap(), boxes, "{code}");
    }
  }

  /// Regression: a held factorial in a `Row` label printed as the call
  /// `Factorial[12]` instead of `12!`.
  #[test]
  fn held_factorial_in_row_draws_postfix() {
    for (code, text) in [
      ("Row[{HoldForm[Factorial[12]], \" = \", 479001600}]", "12!"),
      ("Style[Text[Row[{HoldForm[Factorial2[9]]}]], 20]", "9!!"),
      ("Row[{HoldForm[Factorial[a + b]]}]", "(a+b)!"),
    ] {
      let svg = svg_of(code);
      assert!(flat_text(&svg).contains(text), "{code}: {svg}");
      assert!(!svg.contains("Factorial"), "{code}: {svg}");
    }
  }

  /// A horizontal brace under or over a base (`Underscript[base,
  /// Underscript[\[UnderBrace], n]]`, the idiom labeling a digit count)
  /// is drawn as a brace path stretched across the base, with its label
  /// beyond it, rather than as a tiny brace glyph beside the base.
  #[test]
  fn stretchy_brace_scripts_span_their_base() {
    for code in [
      "Row[{Underscript[Row[{\"123\", \"\\[Ellipsis]\", 9}], Underscript[\\[UnderBrace], 41]]}]",
      "Row[{\"n = \", Overscript[\"000000\", Overscript[\\[OverBrace], 6]]}]",
      "Style[Row[{Underscript[\"abcdef\", \\[UnderBrace]]}], 20]",
      "Pane[Style[Text[Row[{x, \" = \", Underscript[Row[{1, 2, 3}], Underscript[\\[UnderBrace], 3]]}]], 20], {300, 80}]",
    ] {
      let svg = svg_of(code);
      assert!(
        !svg.contains('\u{FE38}') && !svg.contains('\u{FE37}'),
        "{code}: brace drawn as a glyph: {svg}"
      );
      let path = svg
        .match_indices("<path d=\"M")
        .map(|(i, _)| &svg[i..i + svg[i..].find("/>").unwrap()])
        .find(|p| p.matches(" Q ").count() == 4)
        .unwrap_or_else(|| panic!("{code}: no curly brace path: {svg}"));
      assert!(path.contains("stroke=\"currentColor\""), "{path}");
    }
    // The brace's label is set beside its own brace.
    let svg =
      svg_of("Row[{Underscript[\"0000\", Underscript[\\[UnderBrace], 4]]}]");
    assert!(flat_text(&svg).contains("00004"), "{svg}");
  }

  /// A brace-labeled base in a `Row` sits on the row's text line: the
  /// base's digits share the vertical position of the plain text before
  /// it instead of the whole stack being centered on the line.
  #[test]
  fn brace_labeled_row_item_keeps_the_text_line() {
    let svg = svg_of(
      "Row[{\"x = \", Underscript[\"12345\", Underscript[\\[UnderBrace], 5]]}]",
    );
    // The underscript stack hangs below its base line, so lining that
    // line up with the row's center puts the stack lower than plain
    // centering would.
    let attr = |s: &str, key: &str| -> f64 {
      let at = s.find(&format!(" {key}=\"")).unwrap() + key.len() + 3;
      s[at..at + s[at..].find('"').unwrap()].parse().unwrap()
    };
    let row_h = attr(&svg, "height");
    let nested = &svg[svg.find("<svg x=").expect("nested picture")..];
    let (y, h) = (attr(nested, "y"), attr(nested, "height"));
    assert!(y > (row_h - h) / 2.0 + 1.0, "stack centered: {svg}");
  }
}

#[test]
fn review_density_complex_preserves_vertex_colors() {
  clear_state();
  let graphic = "Graphics[GraphicsComplex[{{0,0},{1,0},{1,1},{0,1}}, Polygon[{1,2,3,4}], VertexColors->{White,White,Red,Red}]]";
  let svg = interpret(&format!("ExportString[{graphic}, \"SVG\"]")).unwrap();
  let normal = interpret(&format!(
    "ExportString[Graphics[Normal[First[{graphic}]]], \"SVG\"]"
  ))
  .unwrap();
  assert_eq!(
    svg, normal,
    "Rendering a complex must preserve Normal's vertex colors"
  );
}

#[test]
fn review_rectangle_chart_has_category_baseline_and_end_ticks() {
  clear_state();
  for data in ["{{1,2},{2,3}}", "{{2,4},{3,-1},{1,5}}"] {
    assert_eq!(
      interpret(&format!(
        "Length[Cases[RectangleChart[{data}], _Line, Infinity]]"
      ))
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret(&format!(
        "Length[Cases[RectangleChart[{data}], Offset[{{0,-6}}, _], Infinity]]"
      ))
      .unwrap(),
      "2"
    );
  }
  for option in ["Frame -> True", "Axes -> False", "Axes -> {False,False}"] {
    assert_eq!(
      interpret(&format!(
        "Cases[RectangleChart[{{{{1,2}},{{2,3}}}},{option}],_Line,Infinity]"
      ))
      .unwrap(),
      "{}"
    );
  }
}

#[test]
fn review_colored_background_keeps_black_tick_labels() {
  clear_state();
  for size in ["Full", "Scaled[0.5]"] {
    let svg = interpret(&format!("ExportString[Framed[Pane[Plot[Sin[x],{{x,0,10}},ImageSize->{size},Background->StandardOrange],{{200,100}}]],\"SVG\"]")).unwrap();
    assert!(!svg.contains("fill=\"#666\""));
    assert!(svg.contains("fill=\"#000000\""));
  }
}

/// `Style[Framed[…], …]` styles the framed content and keeps the frame:
/// typeset as styled text, the box border was dropped.
#[test]
fn styled_framed_keeps_its_frame() {
  clear_state();
  for code in [
    r#"ExportString[Style[Framed["degenerate"], 12, "Label"], "SVG"]"#,
    r#"ExportString[Style[Framed["degenerate"], Red], "SVG"]"#,
    r#"ExportString[Pane[Style[If[False, 1, Framed["degenerate"]], 12], ImageSize -> {200, 100}], "SVG"]"#,
  ] {
    let svg = interpret(code).unwrap();
    assert!(svg.contains("<rect"), "{code}: no frame in {svg}");
    assert!(svg.contains("degenerate"), "{code}: no text in {svg}");
    assert!(!svg.contains("Framed"), "{code}: raw call in {svg}");
  }
  // The directives still reach the content: 12 pt red text.
  let svg =
    interpret(r#"ExportString[Style[Framed["degenerate"], 12, Red], "SVG"]"#)
      .unwrap();
  assert!(svg.contains("font-size=\"12"), "{svg}");
  assert!(svg.contains("rgb(255,0,0)"), "{svg}");
}

/// `Style[Panel[…], …]` likewise keeps the panel box around the content.
#[test]
fn styled_panel_keeps_its_box() {
  clear_state();
  let svg =
    interpret(r#"ExportString[Style[Panel["inside"], Bold], "SVG"]"#).unwrap();
  assert!(svg.contains("<rect"), "{svg}");
  assert!(svg.contains("inside") && !svg.contains("Panel"), "{svg}");
}

/// The `font-family` values of an SVG's `<text>` elements.
fn text_families(svg: &str) -> Vec<&str> {
  svg
    .match_indices("<text ")
    .filter_map(|(i, _)| {
      let tag = &svg[i..i + svg[i..].find('>')?];
      let start = tag.find("font-family=\"")? + "font-family=\"".len();
      Some(&tag[start..start + tag[start..].find('"')?])
    })
    .collect()
}

#[test]
fn text_sets_layouts_in_proportional_face() {
  // `Text[…]` outside a `Graphics` shows its content in the proportional
  // text face — a `Row` inside too, alone, in a `Column` or styled.
  for expr in [
    "Text[Row[{\"Depth = \", 2.85, \" m\"}]]",
    "Text[Style[Row[{\"Depth = \", NumberForm[2.85, {2, 1}], \" m\"}], 15]]",
    "Column[{Text[Style[Row[{\"a = \", 1}], 15]], Text[Style[\"b\", 14]]}, Alignment -> Center]",
    "Column[{Text[Row[{\"a = \", 1}]], Graphics[Disk[]]}]",
    "Text[\"label\"]",
  ] {
    clear_state();
    let svg = interpret(&format!("ExportString[{expr}, \"SVG\"]")).unwrap();
    let families = text_families(&svg);
    assert!(!families.is_empty(), "{expr}: no text in {svg}");
    assert!(
      families
        .iter()
        .all(|f| *f == "Atkinson Hyperlegible Next, sans-serif"),
      "{expr}: {families:?}"
    );
  }
  // A face the text asks for itself still wins over the default.
  clear_state();
  let svg = interpret(
    "ExportString[Column[{Text[Style[Row[{\"a\", 1}], FontFamily -> \"Courier\"]]}], \"SVG\"]",
  )
  .unwrap();
  let families = text_families(&svg);
  assert!(
    !families.is_empty() && families.iter().all(|f| f.starts_with("Courier")),
    "{families:?}"
  );
  // A layout outside `Text` keeps the monospace output face.
  clear_state();
  let svg = interpret("ExportString[Row[{\"a = \", 1}], \"SVG\"]").unwrap();
  assert!(text_families(&svg).iter().all(|f| f.contains("Mono")));
}

#[test]
fn dynamic_text_item_keeps_proportional_face() {
  // A held `Dynamic[…]` whose current value is a `Text[…]` item (the
  // status line of a quiz: `Dynamic[If[ok, Text[…], Text[…]]]`) is set in
  // the proportional text face like a bare `Text[…]`, in a `Column` and a
  // `Row` alike.
  for layout in ["Column", "Row"] {
    clear_state();
    let svg = interpret(&format!(
      "a = 1; b = 2; ExportString[{layout}[{{Dynamic[If[a == b, \
       Text[Style[\"same\"]], Text[Style[\"different\"]]]], \
       Graphics[Circle[]]}}], \"SVG\"]"
    ))
    .unwrap();
    assert!(svg.contains(">different<"), "{layout}: {svg}");
    let families = text_families(&svg);
    assert!(
      !families.is_empty()
        && families
          .iter()
          .all(|f| *f == "Atkinson Hyperlegible Next, sans-serif"),
      "{layout}: {families:?}"
    );
  }
  // A notebook-style host (Woxi Studio) draws the layout result itself.
  for layout in ["Column", "Row"] {
    clear_state();
    let res = woxi::interpret_with_stdout(&format!(
      "a = 1; b = 2; {layout}[{{Dynamic[If[a == b, Text[Style[\"same\"]], \
       Text[Style[\"different\"]]]], Graphics[Circle[]]}}]"
    ))
    .unwrap();
    let svg = res.graphics.expect("layout drawn as a picture");
    assert!(svg.contains(">different<"), "{layout}: {svg}");
    let families = text_families(&svg);
    assert!(
      !families.is_empty() && families.iter().all(|f| f.contains("sans-serif")),
      "{layout}: {families:?}"
    );
  }
  // A `Dynamic` showing a bare string keeps the monospace output face.
  clear_state();
  let svg = interpret(
    "a = 1; ExportString[Column[{Dynamic[If[a == 1, \"yes\", \"no\"]], \
     Graphics[Circle[]]}], \"SVG\"]",
  )
  .unwrap();
  let families = text_families(&svg);
  assert!(
    !families.is_empty() && families.iter().all(|f| f.contains("Mono")),
    "{families:?}"
  );
}

#[test]
fn text_row_parts_sit_at_proportional_widths() {
  // The parts of a `Row` set in the default proportional face are spaced
  // by that face's glyph advances, not the monospace grid, so a short
  // number does not float in a wide gap after its label.
  clear_state();
  let svg = interpret(
    "ExportString[Text[Style[Row[{\"Depth of point = \", \"2.9\"}], 15]], \"SVG\"]",
  )
  .unwrap();
  let width: f64 = svg
    .split("width=\"")
    .nth(1)
    .and_then(|s| s.split('"').next())
    .and_then(|s| s.parse().ok())
    .unwrap();
  // 20 characters at 15pt: a monospace grid needs ~190pt, the proportional
  // face well under 150pt.
  assert!(width < 150.0, "width {width} in {svg}");
}

#[test]
fn styled_layout_keeps_picture_of_caption_product() {
  // Regression: a `Style` pushed onto a layout's items wrapped a product
  // with a picture (`"Label: " g`) whole, so the picture was typeset as
  // the `-Graphics-` placeholder.
  for layout in ["Column", "Row"] {
    clear_state();
    let svg = interpret(&format!(
      "ExportString[Style[{layout}[{{\"Label: \" Graphics[Disk[], ImageSize -> 15]}}], Red], \"SVG\"]"
    ))
    .unwrap();
    assert!(!svg.contains("-Graphics-"), "{layout}: {svg}");
    assert!(
      svg.contains("<ellipse") || svg.contains("<circle"),
      "{layout}"
    );
    // The caption still takes the style.
    assert!(
      svg.split("<text ").any(|t| t.contains("Label:")
        && t[..t.find('>').unwrap()].contains("fill=\"rgb(255,0,0)\"")),
      "{layout}: {svg}"
    );
  }
}

#[test]
fn styled_pane_keeps_its_width() {
  // Regression: `Style[Pane[text, w], …]` dropped the pane, so its text
  // ran on one long line instead of wrapping at the pane's width.
  let long = "Column[{Row[{12.5, \" units of the first kind in the sample\"}], Row[{7, \" units of the second kind left over at the end\"}]}]";
  for pane in [
    format!("Style[Pane[{long}, 200], 12]"),
    format!("Style[Pane[{long}, ImageSize -> 200], Red]"),
  ] {
    clear_state();
    let svg = interpret(&format!("ExportString[{pane}, \"SVG\"]")).unwrap();
    assert!(svg.starts_with("<svg width=\"200\""), "{pane}: {svg}");
  }
}

/// `Text[…]` sets its content in the proportional text face, a fixed-size
/// `Pane` keeps its box under `Style`/`Text` wrappers and when its content
/// is shrunk to fit, and a large fraction gets a proportionally heavier
/// rule.
mod text_in_fixed_panes {
  use super::*;

  fn svg_of(expr: &str) -> String {
    clear_state();
    interpret(&format!("ExportString[{expr}, \"SVG\"]")).unwrap()
  }

  /// The body of an SVG after its embedded stylesheet.
  fn body(svg: &str) -> &str {
    svg.find("</defs>").map_or(svg, |i| &svg[i..])
  }

  fn root_size(svg: &str) -> (String, String) {
    let attr = |name: &str| {
      let key = format!("{name}=\"");
      let start = svg.find(&key).unwrap() + key.len();
      svg[start..start + svg[start..].find('"').unwrap()].to_string()
    };
    (attr("width"), attr("height"))
  }

  #[test]
  fn text_of_a_styled_string_uses_the_sans_serif_face() {
    let svg = svg_of(r#"Text[Style["No, try again", 24, Red]]"#);
    let body = body(&svg);
    assert!(
      body.contains("Atkinson Hyperlegible Next, sans-serif"),
      "{svg}"
    );
    assert!(!body.contains("monospace"), "{svg}");
  }

  #[test]
  fn styled_text_of_a_fraction_uses_the_sans_serif_face() {
    let svg =
      svg_of(r#"Style[Text[Row[{HoldForm[4]/HoldForm[66], "?"}]], 80]"#);
    let body = body(&svg);
    assert!(
      body.contains("Atkinson Hyperlegible Next, sans-serif"),
      "{svg}"
    );
    assert!(!body.contains("monospace"), "{svg}");
  }

  #[test]
  fn text_keeps_an_explicit_font_family() {
    let svg = svg_of(r#"Text[Style["abc", FontFamily -> "Courier"]]"#);
    assert!(body(&svg).contains("Courier"), "{svg}");
  }

  #[test]
  fn plain_output_stays_monospace() {
    let svg = svg_of(r#"Style["abc", 24]"#);
    assert!(body(&svg).contains("monospace"), "{svg}");
  }

  #[test]
  fn styled_pane_keeps_its_box() {
    let svg = svg_of(
      r#"Style[Pane[Row[{47, "x", 87}, " "], {500, 25}, Alignment -> Center], Bold]"#,
    );
    assert_eq!(root_size(&svg), ("500".into(), "25".into()), "{svg}");
    assert!(svg.contains("font-weight=\"bold\""), "{svg}");
  }

  #[test]
  fn styled_text_pane_in_a_column_keeps_its_box() {
    let svg = svg_of(
      r#"Column[{Pane["", {300, 50}], Style[Text[Pane[Row[{47, "x", 87}, " "], {500, 25}, Alignment -> Center]], Bold]}, Alignment -> Center]"#,
    );
    // The two sized panes, stacked the default 6 pt apart.
    assert_eq!(root_size(&svg), ("500".into(), "81".into()), "{svg}");
    let row = svg.lines().find(|l| l.contains(">87</text>")).unwrap();
    assert!(
      row.contains("Atkinson Hyperlegible Next, sans-serif"),
      "{row}"
    );
    assert!(row.contains("font-weight=\"bold\""), "{row}");
  }

  #[test]
  fn shrink_to_fit_pane_keeps_its_box() {
    // The 80 pt fraction is taller than the 200 x 200 pane, so it is
    // shrunk — and centered in the pane, which stays 200 x 200.
    let svg = svg_of(
      r#"Pane[Style[HoldForm[4]/HoldForm[66], 80], {200, 200}, Alignment -> Center, ImageSizeAction -> "ShrinkToFit"]"#,
    );
    assert_eq!(root_size(&svg), ("200".into(), "200".into()), "{svg}");
  }

  /// The stroke widths of the `<line>` elements of an SVG.
  fn line_widths(svg: &str) -> Vec<f64> {
    svg
      .match_indices("<line")
      .filter_map(|(i, _)| {
        let tag = &svg[i..i + svg[i..].find('>').unwrap()];
        let start = tag.find("stroke-width=\"")? + "stroke-width=\"".len();
        tag[start..start + tag[start..].find('"')?].parse().ok()
      })
      .collect()
  }

  #[test]
  fn fraction_rule_scales_with_large_type() {
    assert_eq!(line_widths(&svg_of("HoldForm[4]/HoldForm[66]")), vec![0.8]);
    assert_eq!(
      line_widths(&svg_of("Style[HoldForm[4]/HoldForm[66], 70]")),
      vec![4.0]
    );
  }
}
