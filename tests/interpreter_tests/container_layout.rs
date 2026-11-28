//! Graphics shown inside a list, `Grid`, `Column` or `Row` are drawn at the
//! reduced size the front end gives them — half (a quarter in a `Row` of
//! several items) of their automatic size — with absolute line widths,
//! point sizes and font sizes, not drawn full size and shrunk.
use super::*;

mod container_layout {
  use super::*;

  fn export_svg(expr: &str) -> String {
    clear_state();
    interpret(&format!("ExportString[{expr}, \"SVG\"]")).unwrap()
  }

  /// The `width`/`height` of the root `<svg>` element.
  fn root_size(svg: &str) -> (String, String) {
    let header = &svg[..svg.find('>').unwrap()];
    let attr = |name: &str| {
      let key = format!(" {name}=\"");
      let start = header.find(&key).unwrap() + key.len();
      header[start..start + header[start..].find('"').unwrap()].to_string()
    };
    (attr("width"), attr("height"))
  }

  /// The `width` of every nested picture cell (`<svg x=… width=…>`).
  fn cell_widths(svg: &str) -> Vec<String> {
    svg
      .match_indices("<svg x=")
      .map(|(i, _)| {
        let tag = &svg[i..i + svg[i..].find('>').unwrap()];
        let start = tag.find(" width=\"").unwrap() + 8;
        tag[start..start + tag[start..].find('"').unwrap()].to_string()
      })
      .collect()
  }

  #[test]
  fn list_of_plots_draws_each_at_half_size() {
    // wolframscript: width="389pt" height="111pt", each plot 180 wide.
    let svg = export_svg("{Plot[x, {x, 0, 1}], Plot[x, {x, 0, 1}]}");
    assert_eq!(root_size(&svg), ("389".into(), "111".into()));
    assert_eq!(cell_widths(&svg), vec!["180.00", "180.00"]);
    // Drawn at 180 (a 1800-unit canvas at the plots' 10x resolution), so the
    // 2 pt curve and the 10 pt tick labels keep their absolute size.
    assert!(svg.contains("viewBox=\"0 0 1800 1110\""), "{svg}");
    assert!(svg.contains("stroke-width=\"20\""));
    assert!(svg.contains("font-size=\"100\""));
    assert!(!svg.contains("viewBox=\"0 0 3600"));
  }

  #[test]
  fn single_item_and_table_lists() {
    assert_eq!(
      root_size(&export_svg("{Plot[x, {x, 0, 1}]}")),
      ("196".into(), "111".into())
    );
    assert_eq!(
      root_size(&export_svg("Table[Plot[x, {x, 0, 1}], {4}]")),
      ("774".into(), "111".into())
    );
  }

  #[test]
  fn long_lists_wrap_onto_balanced_lines() {
    // Six 180-wide plots do not fit on one line: wolframscript sets them
    // three and three (582 wide), not five and one.
    let svg = export_svg("Table[Plot[x, {x, 0, 1}], {6}]");
    let (w, h) = root_size(&svg);
    assert_eq!(w, "582");
    assert!(h == "249" || h == "250", "{h}");
    assert_eq!(cell_widths(&svg).len(), 6);
  }

  #[test]
  fn nested_lists_stay_on_one_line() {
    // `{{p, p}, {p, p}}` is typeset with inner braces, not as a grid.
    let svg = export_svg("With[{p = Plot[x, {x, 0, 1}]}, {{p, p}, {p, p}}]");
    assert_eq!(root_size(&svg), ("805".into(), "111".into()));
  }

  #[test]
  fn explicit_image_size_is_kept() {
    let svg = export_svg("{Plot[x, {x, 0, 1}, ImageSize -> 300]}");
    assert_eq!(root_size(&svg), ("316".into(), "184".into()));
  }

  #[test]
  fn symbolic_graphics_in_a_list() {
    let svg = export_svg("{Graphics[Disk[]]}");
    assert_eq!(root_size(&svg), ("196".into(), "180".into()));
  }

  #[test]
  fn plots_without_redrawable_data_are_evaluated_again() {
    // A contour plot keeps no data the line renderer could redraw it from;
    // it is plotted again at the list's size.
    let svg = export_svg("{ContourPlot[x y, {x, 0, 1}, {y, 0, 1}]}");
    assert_eq!(root_size(&svg), ("196".into(), "176".into()));
    // The `Table` iterator is gone by the time the list is displayed; the
    // plot is remembered with the value it had.
    let svg =
      export_svg("Table[ContourPlot[x y + i, {x, 0, 1}, {y, 0, 1}], {i, 2}]");
    assert_eq!(root_size(&svg), ("389".into(), "176".into()));
  }

  #[test]
  fn plots_are_not_evaluated_again_for_their_side_effects() {
    clear_state();
    let out = interpret(
      "n = 0; p = Plot[(n++; x), {x, 0, 1}]; m = n; \
       ExportString[{p, p}, \"SVG\"]; n == m",
    )
    .unwrap();
    assert_eq!(out, "True");
  }

  #[test]
  fn grid_column_and_row_sizes() {
    // wolframscript: Grid 366x227, Column 180x227, Row 180x59 (two plots at
    // a quarter size, side by side), Row with a separator 373 wide.
    let p = "Plot[x, {x, 0, 1}]";
    let (w, _) = root_size(&export_svg(&format!("Grid[{{{{{p}, {p}}}}}]")));
    assert_eq!(w, "366");
    let (w, _) = root_size(&export_svg(&format!("Column[{{{p}, {p}}}]")));
    assert_eq!(w, "180");
    assert_eq!(
      root_size(&export_svg(&format!("Row[{{{p}, {p}}}]"))),
      ("180".into(), "59".into())
    );
    // A one-item `Row` shows its plot as it is shown alone.
    let (w, _) = root_size(&export_svg(&format!("Row[{{{p}}}]")));
    assert_eq!(w, "360");
    let svg = export_svg(&format!("Row[{{{p}, {p}}}, \",\"]"));
    let plots = cell_widths(&svg)
      .iter()
      .filter(|w| w.parse::<f64>() == Ok(180.0))
      .count();
    assert_eq!(plots, 2);
  }

  /// A `Column` given an alignment or options halves its automatically
  /// sized pictures like a bare `Column[{…}]`, also when it is a cell of a
  /// `Grid` shown as `Text`.
  #[test]
  fn column_with_options_draws_pictures_at_half_size() {
    let m = "ArrayPlot[{{1, 0, 1}, {0, 1, 0}}]";
    for column in [
      format!("Column[{{\"label\", {m}}}, Center]"),
      format!("Column[{{\"label\", {m}}}, Alignment -> Center]"),
      "Column[{\"label\", Plot[x, {x, 0, 1}]}, Alignment -> Right]".to_string(),
      format!("Text[Column[{{\"label\", {m}}}, Alignment -> Center]]"),
      format!(
        "Text[Grid[{{{{Column[{{\"label\", {m}}}, Alignment -> Center]}}}}]]"
      ),
    ] {
      let svg = export_svg(&column);
      let widths: Vec<f64> = cell_widths(&svg)
        .iter()
        .filter_map(|w| w.parse().ok())
        .collect();
      assert!(widths.contains(&180.0), "{column}: {widths:?}");
      assert!(widths.iter().all(|w| *w < 200.0), "{column}: {widths:?}");
    }
    // An explicit `ImageSize` is kept.
    let svg = export_svg(
      "Column[{\"label\", Plot[x, {x, 0, 1}, ImageSize -> 300]}, \
       Alignment -> Center]",
    );
    assert_eq!(root_size(&svg).0, "300");
  }

  /// The font family of the `<text>` element showing `label`.
  fn label_font(svg: &str, label: &str) -> String {
    let end = svg.find(&format!(">{label}<")).expect("label is drawn");
    let tag = &svg[svg[..end].rfind("<text").unwrap()..end];
    let start = tag.find("font-family=\"").unwrap() + 13;
    tag[start..start + tag[start..].find('"').unwrap()].to_string()
  }

  /// `Text[Grid[…]]` as a whole output sets the labels of the columns in
  /// its cells in the proportional text face, like every other cell.
  #[test]
  fn text_grid_output_sets_column_labels_in_the_text_face() {
    clear_state();
    let result = woxi::interpret_with_stdout(
      "Text[Grid[{{Column[{\"above\", ArrayPlot[{{1, 0}, {0, 1}}]}, \
       Alignment -> Center], \"beside\"}}]]",
    )
    .unwrap();
    let svg = result.graphics.expect("the grid is a graphic");
    let beside = label_font(&svg, "beside");
    assert!(!beside.contains("mono"), "{beside}");
    assert_eq!(label_font(&svg, "above"), beside, "{svg}");
  }

  /// `Text[Style[Row[{tables and pictures}], size]]` as a whole output
  /// sizes the tables in the proportional text face they are drawn in, the
  /// way the same expression exports: measured in the monospace face of a
  /// bare `Row`, a line of text took a third more room than it needs and
  /// pushed the last table out of a `Pane` the row fits in.
  #[test]
  fn text_row_output_sizes_its_tables_in_the_text_face() {
    for items in [
      "Grid[{{\"speed = 12.5 km / h\"}}]",
      "Grid[{{\"speed = 12.5 km / h\"}, {Row[{\"t = \", 3.25}]}}, \
       Alignment -> Left], Graphics[{Disk[]}, ImageSize -> {60, 90}], \
       Grid[{{\"distance = 40.625 km\"}}]",
    ] {
      let code = format!("Text[Style[Row[{{{items}}}], 17]]");
      clear_state();
      let shown = woxi::interpret_with_stdout(&code)
        .unwrap()
        .graphics
        .expect("the row is a graphic");
      assert_eq!(root_size(&shown), root_size(&export_svg(&code)), "{code}");
      assert!(!shown.contains("mono"), "{shown}");
    }
    // Both tables and the picture between them fit the pane, so nothing
    // is clipped and the row is centered in it.
    clear_state();
    let pane = woxi::interpret_with_stdout(
      "Pane[Text[Style[Row[{Grid[{{\"speed = 12.5 km / h\"}}], \
       Graphics[{Disk[]}, ImageSize -> {60, 90}], \
       Grid[{{\"distance = 40.625 km\"}}]}], 17]], {450, 200}, \
       Alignment -> {Center, Center}]",
    )
    .unwrap()
    .graphics
    .expect("the pane is a graphic");
    assert_eq!(root_size(&pane), ("450".to_string(), "200".to_string()));
    assert!(!pane.contains("clip-path=\"url(#paneClip"), "{pane}");
  }

  /// A `Style[Column[…], size]` cell of a grid is set in the grid's text
  /// face like a bare `Column` cell, at the size the cell asks for —
  /// whether the column is plain lines or holds typeset rows.
  #[test]
  fn styled_column_cell_of_a_grid_is_set_in_the_grid_face() {
    for column in [
      "Column[{\"first\", \"second\"}]",
      "Column[{Row[{\"first = \", Subscript[Style[\"q\", Italic],        \"in\"]}], Row[{\"second\", 2.5}]}]",
    ] {
      for grid in [
        format!("Grid[{{{{Style[{column}, 17]}}}}]"),
        format!("Text[Grid[{{{{Style[{column}, 17], \"beside\"}}}}]]"),
      ] {
        let svg = export_svg(&grid);
        assert!(!svg.contains("mono"), "{grid}: {svg}");
        assert!(svg.contains("font-size=\"17\""), "{grid}: {svg}");
        clear_state();
        let shown = woxi::interpret_with_stdout(&grid)
          .unwrap()
          .graphics
          .expect("the grid is a graphic");
        assert!(!shown.contains("mono"), "{grid}: {shown}");
      }
    }
    // The cell's own directives still win over the grid's.
    let svg = export_svg(
      "Style[Grid[{{Style[Column[{\"first\", \"second\"}], 17]}}], 11]",
    );
    assert!(svg.contains("font-size=\"17\""), "{svg}");
    assert!(!svg.contains("font-size=\"11\""), "{svg}");
  }

  /// A horizontal value in the row part of a grid's `Alignment`
  /// (`{Center, {Left}}`) sets the row's items against its top edge, the
  /// same as `Top`, instead of centering them.
  #[test]
  fn horizontal_value_as_a_row_alignment_sets_the_row_at_its_top() {
    let grid = |alignment: &str| {
      let svg = export_svg(&format!(
        "Grid[{{{{Graphics[Disk[], ImageSize -> {{20, 20}}],          Graphics[Disk[], ImageSize -> {{20, 80}}]}}}},          Alignment -> {alignment}]"
      ));
      svg
        .match_indices("<svg x=")
        .map(|(i, _)| {
          let tag = &svg[i..i + svg[i..].find('>').unwrap()];
          let start = tag.find(" y=\"").unwrap() + 4;
          tag[start..start + tag[start..].find('"').unwrap()].to_string()
        })
        .collect::<Vec<_>>()
    };
    let top = grid("{Center, Top}");
    assert_eq!(top.len(), 2);
    assert_eq!(top[0], top[1]);
    for alignment in [
      "{Center, {Left}}",
      "{Center, Right}",
      "{{Center, Center}, {Left, Left}}",
    ] {
      assert_eq!(grid(alignment), top, "{alignment}");
    }
    let centered = grid("{Center, Center}");
    assert_ne!(centered[0], centered[1]);
  }

  /// A `Row` in the text face is as wide as its text with the thin gaps
  /// between the digit groups of a long number, so the last digits are not
  /// cut off by the row's box.
  #[test]
  fn text_row_makes_room_for_digit_group_gaps() {
    let width = |n: &str| -> f64 {
      root_size(&export_svg(&format!("Text[Row[{{\"n: \", {n}}}]]")))
        .0
        .parse()
        .unwrap()
    };
    // Seven digits in three groups are wider than seven digits that are
    // not grouped (a four-digit and a three-digit number side by side
    // would be), by the two gaps of 0.18 em at 14 pt.
    let grouped = width("1234567");
    let per_digit = width("7777") - width("777");
    let ungrouped = width("777") + 4.0 * per_digit;
    assert!(
      grouped >= ungrouped + 2.0 * 0.18 * 14.0 - 1.5,
      "{grouped} {ungrouped}"
    );
    // Four digits are not grouped and take no gap.
    assert!((width("7777") - width("777") - per_digit).abs() < 1.5);
  }

  /// A picture centered in a wider grid column sits on a whole pixel, so
  /// the crisp-edged frame lines of a plot are not lost between pixels.
  #[test]
  fn centered_grid_pictures_sit_on_whole_pixels() {
    // Columns one pixel apart in width: one of the two pictures would be
    // centered at a half-pixel offset.
    for wide in [201, 202] {
      let svg = export_svg(&format!(
        "Grid[{{{{Graphics[Disk[], ImageSize -> 100]}}, \
         {{Graphics[Disk[], ImageSize -> {{{wide}, 20}}]}}}}]"
      ));
      let xs: Vec<f64> = svg
        .match_indices("<svg x=\"")
        .map(|(i, _)| {
          let rest = &svg[i + 8..];
          rest[..rest.find('"').unwrap()].parse().unwrap()
        })
        .collect();
      assert_eq!(xs.len(), 2, "{svg}");
      assert!(xs.iter().all(|x| x.fract() == 0.0), "{wide}: {xs:?}");
    }
  }

  #[test]
  fn grid_cells_are_a_fixed_gap_apart() {
    // wolframscript: 206x106 for 100x50 pictures in a 2x2 grid.
    let svg = export_svg(
      "With[{g = Graphics[{}, ImageSize -> {100, 50}]}, Grid[{{g, g}, {g, g}}]]",
    );
    assert_eq!(root_size(&svg), ("206".into(), "106".into()));
  }

  #[test]
  fn show_keeps_the_mesh_full_points() {
    // `Mesh -> Full` marks the initial sampling grid; a redraw from the
    // plot's data (`Show`, a list) marks the same points.
    clear_state();
    let out = interpret(
      "p = Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> Full]; \
       ExportString[Show[p, ImageSize -> 180], \"SVG\"] === \
       ExportString[Plot[Sin[x], {x, 0, 2 Pi}, Mesh -> Full, ImageSize -> 180], \"SVG\"]",
    )
    .unwrap();
    assert_eq!(out, "True");
  }

  /// The x positions of the vertical divider lines of a one-row grid.
  fn vertical_dividers(svg: &str) -> Vec<f64> {
    let attr = |line: &str, name: &str| -> f64 {
      let key = format!(" {name}=\"");
      let start = line.find(&key).unwrap() + key.len();
      line[start..start + line[start..].find('"').unwrap()]
        .parse()
        .unwrap()
    };
    let mut xs: Vec<f64> = svg
      .lines()
      .filter(|l| l.starts_with("<line") && attr(l, "x1") == attr(l, "x2"))
      .map(|l| attr(l, "x1"))
      .collect();
    xs.sort_by(f64::total_cmp);
    xs
  }

  /// The widths of the two columns of `Grid[{{"iiii", "MMMM"}}, …]` drawn
  /// under `wrap` (a format string with `{}` for the grid).
  fn narrow_and_wide_column(wrap: &str, opts: &str) -> (f64, f64) {
    let grid = format!(
      "Grid[{{{{\"iiii\", \"MMMM\"}}}}, Dividers -> All, Spacings -> 0{opts}]"
    );
    let xs = vertical_dividers(&export_svg(&wrap.replace("{}", &grid)));
    assert_eq!(xs.len(), 3, "{xs:?}");
    (xs[1] - xs[0], xs[2] - xs[1])
  }

  #[test]
  fn label_styled_grid_sizes_columns_in_the_proportional_face() {
    // Plain output text is laid out at a fixed advance per character.
    let (narrow, wide) = narrow_and_wide_column("{}", "");
    assert_eq!(narrow, wide);
    // `"Label"` text is proportional: four `i` take far less room than
    // four `M`, at the style's own size and at an explicit one.
    for wrap in ["Style[{}, \"Label\"]", "Style[{}, \"Label\", 16]"] {
      let (narrow, wide) = narrow_and_wide_column(wrap, "");
      assert!(narrow < wide * 0.5, "{wrap}: {narrow} vs {wide}");
    }
    // The same through `BaseStyle`, and for a grid nested in a styled one.
    let (narrow, wide) =
      narrow_and_wide_column("{}", ", BaseStyle -> \"Label\"");
    assert!(narrow < wide * 0.5, "{narrow} vs {wide}");
    let (narrow, wide) =
      narrow_and_wide_column("Style[Grid[{{{}}, {\"z\"}}], \"Label\", 16]", "");
    assert!(narrow < wide * 0.5, "{narrow} vs {wide}");
  }

  #[test]
  fn label_styled_grid_keeps_string_cells_evenly_spaced() {
    // With a 0.3 em gap between columns, a short string beside a long one
    // in the same column must not drift away from its neighbor: the gap
    // between the columns' text follows the glyphs, not a character count.
    let svg = export_svg(
      "Style[Grid[{{\"WWWW\", \"=\", \"1.500\"}}, \
       Spacings -> {{1 -> 1, 2 -> .3, 3 -> .3, 4 -> 1}, Automatic}], \"Label\", 16]",
    );
    let centers: Vec<f64> = svg
      .split("<text x=\"")
      .skip(1)
      .map(|t| t[..t.find('"').unwrap()].parse().unwrap())
      .collect();
    assert_eq!(centers.len(), 3, "{svg:.2000}");
    // Half of "WWWW" (about 3.4 em wide at 16 px) plus the gap and half of
    // "=" separate the first two centers.
    assert!(centers[1] - centers[0] > 30.0, "{centers:?}");
    assert!(centers[2] - centers[1] < 40.0, "{centers:?}");
  }

  #[test]
  fn string_beside_a_picture_keeps_the_letters_of_a_constant_name() {
    // A string that happens to spell a constant (or `Null`) is still text:
    // it must be drawn exactly like any other string of the same shape, not
    // as the constant's glyph (or as nothing).
    for (constant, plain) in [
      ("E", "F"),
      ("I", "J"),
      ("Pi", "Pj"),
      ("Infinity", "Infinitz"),
      ("Degree", "Degref"),
      ("Null", "Nulk"),
    ] {
      for template in [
        "Row[{Graphics[Circle[], ImageSize -> 20], \"{}\"}]",
        "Column[{Graphics[Circle[], ImageSize -> 20], \"{}\"}]",
        "Grid[{{Graphics[Circle[], ImageSize -> 20], \"{}\"}}]",
        "Labeled[Graphics[Circle[], ImageSize -> 20], \"{}\"]",
        "Row[{Graphics[Circle[], ImageSize -> 20], Style[\"{}\", Red]}]",
        "Text@Grid[{{Row[{Graphics[Circle[], ImageSize -> 20], \"{}\"}]}}]",
      ] {
        let svg = export_svg(&template.replace("{}", constant));
        assert!(svg.contains(&format!(">{constant}</text>")), "{svg:.3000}");
        let expected = export_svg(&template.replace("{}", plain))
          .replace(&format!(">{plain}</text>"), &format!(">{constant}</text>"));
        // Only the proportional faces advance differently per letter, so
        // compare the markup around the text positions.
        let strip = |s: &str| -> String {
          let mut out = String::new();
          let mut rest = s;
          while let Some(i) = rest.find("<text x=\"") {
            out.push_str(&rest[..i + 9]);
            rest = &rest[i + 9..];
            rest = &rest[rest.find('"').unwrap()..];
          }
          out + rest
        };
        if template.starts_with("Row[{Graphics[Circle[], ImageSize -> 20], \"")
        {
          assert_eq!(strip(&svg), strip(&expected), "{constant}");
        }
      }
    }
  }

  #[test]
  fn constant_symbol_beside_a_picture_is_still_typeset_as_its_glyph() {
    let svg = export_svg("Row[{Graphics[Circle[], ImageSize -> 20], Pi}]");
    assert!(svg.contains("\u{03C0}"), "{svg:.3000}");
  }
}
