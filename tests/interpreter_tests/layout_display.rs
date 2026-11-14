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
