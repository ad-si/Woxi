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
}
