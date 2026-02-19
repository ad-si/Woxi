use super::*;

/// Helper: wraps an expression with ExportString[..., "SVG"], evaluates it,
/// and returns the SVG string. Panics if evaluation fails or if the result
/// doesn't look like an SVG.
fn export_svg(expr: &str) -> String {
  let code = format!("ExportString[{}, \"SVG\"]", expr);
  let svg = interpret(&code).unwrap();
  assert!(
    svg.starts_with("<svg"),
    "Expected SVG output for {expr}, got: {}",
    &svg[..100.min(svg.len())]
  );
  assert!(svg.contains("</svg>"), "SVG should be complete for {expr}");
  svg
}

mod graphics {
  use super::*;

  mod basic {
    use super::*;

    #[test]
    fn circle() {
      insta::assert_snapshot!(export_svg("Graphics[{Circle[]}]"));
    }

    #[test]
    fn disk() {
      insta::assert_snapshot!(export_svg("Graphics[{Disk[]}]"));
    }

    #[test]
    fn empty_list_content() {
      insta::assert_snapshot!(export_svg("Graphics[{}]"));
    }
  }

  mod primitives {
    use super::*;

    #[test]
    fn point_single() {
      insta::assert_snapshot!(export_svg("Graphics[{Point[{0, 0}]}]"));
    }

    #[test]
    fn point_multi() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Point[{{0, 0}, {1, 1}, {2, 0}}]}]"
      ));
    }

    #[test]
    fn line() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Line[{{0, 0}, {1, 1}, {2, 0}}]}]"
      ));
    }

    #[test]
    fn circle_default() {
      insta::assert_snapshot!(export_svg("Graphics[{Circle[]}]"));
    }

    #[test]
    fn circle_with_center() {
      insta::assert_snapshot!(export_svg("Graphics[{Circle[{1, 2}]}]"));
    }

    #[test]
    fn circle_with_radius() {
      insta::assert_snapshot!(export_svg("Graphics[{Circle[{0, 0}, 2]}]"));
    }

    #[test]
    fn disk_default() {
      insta::assert_snapshot!(export_svg("Graphics[{Disk[]}]"));
    }

    #[test]
    fn disk_with_center_radius() {
      insta::assert_snapshot!(export_svg("Graphics[{Disk[{1, 0}, 0.5]}]"));
    }

    #[test]
    fn disk_sector() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Disk[{0, 0}, 1, {0, Pi}]}]"
      ));
    }

    #[test]
    fn disk_sector_yin_yang() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Red, Disk[{0, 0}, 2, {0, Pi}], Blue, Disk[{0, 0}, 2, {Pi, 2 Pi}], Red, Disk[{-1, 0}, 1], Blue, Disk[{1, 0}, 1]}, ImageSize -> 150]"
      ));
    }

    #[test]
    fn rectangle_default() {
      insta::assert_snapshot!(export_svg("Graphics[{Rectangle[]}]"));
    }

    #[test]
    fn rectangle_with_corners() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Rectangle[{0, 0}, {2, 3}]}]"
      ));
    }

    #[test]
    fn polygon() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Polygon[{{0, 0}, {1, 0}, {0.5, 1}}]}]"
      ));
    }

    #[test]
    fn arrow() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Arrow[{{0, 0}, {1, 1}}]}]"
      ));
    }

    #[test]
    fn text() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Text[\"Hello\", {0, 0}]}]"
      ));
    }

    #[test]
    fn bezier_curve() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{BezierCurve[{{0, 0}, {0.5, 1}, {1, 0}}]}]"
      ));
    }
  }

  mod styles {
    use super::*;

    #[test]
    fn named_color() {
      insta::assert_snapshot!(export_svg("Graphics[{Red, Disk[]}]"));
    }

    #[test]
    fn rgb_color() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{RGBColor[0.5, 0.3, 0.8], Circle[]}]"
      ));
    }

    #[test]
    fn opacity() {
      insta::assert_snapshot!(export_svg("Graphics[{Opacity[0.5], Disk[]}]"));
    }

    #[test]
    fn thickness() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Thickness[0.01], Line[{{0, 0}, {1, 1}}]}]"
      ));
    }

    #[test]
    fn dashing() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Dashing[{0.02, 0.02}], Line[{{0, 0}, {1, 1}}]}]"
      ));
    }

    #[test]
    fn point_size() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{PointSize[0.03], Point[{0, 0}]}]"
      ));
    }

    #[test]
    fn darker() {
      insta::assert_snapshot!(export_svg("Graphics[{Darker[Red], Disk[]}]"));
    }

    #[test]
    fn lighter() {
      insta::assert_snapshot!(export_svg("Graphics[{Lighter[Blue], Disk[]}]"));
    }

    #[test]
    fn style_scoping() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{{Red, Disk[]}, Circle[{2, 0}]}]"
      ));
    }

    #[test]
    fn edge_form() {
      insta::assert_snapshot!(export_svg("Graphics[{EdgeForm[Red], Disk[]}]"));
    }

    #[test]
    fn multiple_colors() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Red, Disk[], Blue, Circle[{2, 0}, 0.5]}]"
      ));
    }
  }

  mod text_styles {
    use super::*;

    #[test]
    fn text_with_style_bold() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Text[Style[\"Hello\", Bold], {0, 0}]}]"
      ));
    }

    #[test]
    fn text_with_style_italic() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Text[Style[\"World\", Italic], {0, 0}]}]"
      ));
    }

    #[test]
    fn text_with_font_size() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Text[Style[\"Big\", 20], {0, 0}]}]"
      ));
    }

    #[test]
    fn text_with_color() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Text[Style[\"Red Text\", Red], {0, 0}]}]"
      ));
    }

    #[test]
    fn text_with_multiple_style_directives() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Text[Style[\"Platten\", 12, Bold], {11.5, 3.5}]}]"
      ));
    }

    #[test]
    fn text_with_italic_and_size() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Text[Style[\"Schirm\", 11, Italic], {17, -5.0}]}]"
      ));
    }

    #[test]
    fn text_with_italic_and_color() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Text[Style[\"Strahl\", 11, Italic, Blue], {10, 0.7}]}]"
      ));
    }

    #[test]
    fn multiple_styled_texts_in_graphics() {
      insta::assert_snapshot!(export_svg(concat!(
        "Graphics[{",
        "Text[Style[\"Platten\", 12, Bold], {11.5, 3.5}], ",
        "Text[Style[\"Schirm\", 11, Italic], {17, -5.0}], ",
        "Text[Style[\"Strahl\", 11, Italic, Blue], {10, 0.7}]",
        "}]"
      )));
    }
  }

  mod options {
    use super::*;

    #[test]
    fn image_size_integer() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Circle[]}, ImageSize -> 200]"
      ));
    }

    #[test]
    fn image_size_pair() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Circle[]}, ImageSize -> {400, 300}]"
      ));
    }

    #[test]
    fn image_size_named() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Circle[]}, ImageSize -> Large]"
      ));
    }

    #[test]
    fn plot_range() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Circle[]}, PlotRange -> {{-2, 2}, {-2, 2}}]"
      ));
    }

    #[test]
    fn background() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Circle[]}, Background -> LightGray]"
      ));
    }
  }

  mod integration {
    use super::*;

    #[test]
    fn complex_diagram() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{
          Red, Disk[{0, 0}, 0.5],
          Blue, Circle[{2, 0}, 0.8],
          Green, Polygon[{{4, 0}, {5, 1}, {5, -1}}],
          Black, Line[{{0, 0}, {2, 0}, {4, 0}}],
          Orange, Arrow[{{-1, -1}, {-1, 1}}],
          Text[\"Origin\", {0, -0.8}]
        }, ImageSize -> 400]"
      ));
    }

    #[test]
    fn nested_style_scoping() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{
          {Red, Disk[{0, 0}]},
          {Blue, Disk[{3, 0}]},
          Circle[{1.5, 2}]
        }]"
      ));
    }

    #[test]
    fn graphics_primitives_are_symbolic() {
      assert_eq!(interpret("Circle[]").unwrap(), "Circle[]");
      assert_eq!(interpret("Disk[{1, 0}]").unwrap(), "Disk[{1, 0}]");
      assert_eq!(interpret("Point[{0, 0}]").unwrap(), "Point[{0, 0}]");
      assert_eq!(interpret("RGBColor[1, 0, 0]").unwrap(), "RGBColor[1, 0, 0]");
    }

    #[test]
    fn hue_color() {
      insta::assert_snapshot!(export_svg("Graphics[{Hue[0.6], Disk[]}]"));
    }

    #[test]
    fn hue_two_args() {
      insta::assert_snapshot!(export_svg(
        "Graphics[Table[{Hue[h, s], Disk[{12h, 8s}]}, {h, 0, 1, 1/6}, {s, 0, 1, 1/4}]]"
      ));
    }

    #[test]
    fn directive_compound() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Directive[Red, Thickness[0.01]], Line[{{0, 0}, {1, 1}}]}]"
      ));
    }

    #[test]
    fn graphics_evaluates_table_content() {
      insta::assert_snapshot!(export_svg(
        "Graphics[Table[Disk[{i, 0}, 0.5], {i, 3}]]"
      ));
    }

    #[test]
    fn graphics_nested_table() {
      insta::assert_snapshot!(export_svg(
        "Graphics[Table[Disk[{r*Cos[2 Pi q/4], r*Sin[2 Pi q/4]}, 0.3], {r, 1, 2}, {q, 4}]]"
      ));
    }

    #[test]
    fn graphics_table_with_symbolic_pi_step() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Yellow, Disk[{0, 0}, 0.3], Pink, Table[Disk[{Cos[θ], Sin[θ]}, 0.25], {θ, 0, 2 Pi - Pi/4, Pi/4}]}]"
      ));
    }

    #[test]
    fn edgeform_with_list_arg() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{EdgeForm[{GrayLevel[0, 0.5]}], Disk[]}]"
      ));
    }

    #[test]
    fn hue_with_alpha_fill_opacity() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Hue[0, 1, 1, 0.6], Disk[]}]"
      ));
    }

    #[test]
    fn separate_fill_and_stroke_opacity() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{EdgeForm[{GrayLevel[0, 0.5]}], Hue[0, 1, 1, 0.6], Disk[]}]"
      ));
    }

    #[test]
    fn square_aspect_ratio_for_symmetric_data() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Disk[{1, 0}, 0.5], Disk[{-1, 0}, 0.5], Disk[{0, 1}, 0.5], Disk[{0, -1}, 0.5]}]"
      ));
    }

    #[test]
    fn circle_has_equal_radii() {
      insta::assert_snapshot!(export_svg("Graphics[Circle[]]"));
    }

    #[test]
    fn circle_round_with_explicit_image_size() {
      insta::assert_snapshot!(export_svg(
        "Graphics[Circle[], ImageSize -> {400, 200}]"
      ));
    }

    #[test]
    fn preserves_aspect_ratio_attribute() {
      // Covered by the circle_has_equal_radii snapshot, but kept as explicit test
      let svg = export_svg("Graphics[Circle[]]");
      assert!(svg.contains("preserveAspectRatio=\"xMidYMid meet\""));
    }

    #[test]
    fn full_hue_rings_expression() {
      insta::assert_snapshot!(export_svg(
        "Graphics[Table[{EdgeForm[{GrayLevel[0, 0.5]}], Hue[(-11+q+10r)/72, 1, 1, 0.6], Disk[(8-r){Cos[2Pi q/12], Sin[2Pi q/12]}, (8-r)/3]}, {r, 6}, {q, 12}]]"
      ));
    }

    #[test]
    fn hue_accepts_leading_dot_real_literals() {
      insta::assert_snapshot!(export_svg(
        "Graphics[Table[{Hue[t/15, 1, .9, .3], Disk[{Cos[2 Pi t/15], Sin[2 Pi t/15]}]}, {t, 15}]]"
      ));
    }
  }

  mod graphicsbox_capture {
    #[test]
    fn captures_graphicsbox_for_disk() {
      woxi::interpret("Graphics[{Red, Disk[{0, 0}]}]").unwrap();
      let gbox = woxi::get_captured_graphicsbox();
      assert!(gbox.is_some(), "GraphicsBox should be captured");
      let gbox = gbox.unwrap();
      assert!(
        gbox.starts_with("GraphicsBox["),
        "Should start with GraphicsBox[, got: {}",
        &gbox[..40.min(gbox.len())]
      );
      assert!(
        gbox.contains("RGBColor[1, 0, 0]"),
        "Should contain RGBColor for Red"
      );
      assert!(gbox.contains("DiskBox[{0, 0}]"), "Should contain DiskBox");
    }

    #[test]
    fn captures_graphicsbox_via_interpret_with_stdout() {
      let result =
        woxi::interpret_with_stdout("Graphics[{Red, Disk[{0, 0}]}]").unwrap();
      assert_eq!(result.result, "-Graphics-");
      let gbox = woxi::get_captured_graphicsbox();
      assert!(
        gbox.is_some(),
        "GraphicsBox should survive after interpret_with_stdout"
      );
      let gbox = gbox.unwrap();
      assert!(
        gbox.contains("DiskBox"),
        "Should contain DiskBox, got: {}",
        gbox
      );
    }

    #[test]
    fn captures_graphicsbox_for_rectangle() {
      woxi::interpret("Graphics[{Blue, Rectangle[{0, 0}, {2, 3}]}]").unwrap();
      let gbox = woxi::get_captured_graphicsbox().unwrap();
      assert!(gbox.contains("RGBColor[0, 0, 1]"));
      assert!(gbox.contains("RectangleBox[{0, 0}, {2, 3}]"));
    }

    #[test]
    fn captures_graphicsbox_for_plot() {
      woxi::interpret("Plot[Sin[x], {x, -10, 10}]").unwrap();
      let gbox = woxi::get_captured_graphicsbox();
      assert!(gbox.is_some(), "GraphicsBox should be captured for Plot");
      let gbox = gbox.unwrap();
      assert!(gbox.starts_with("GraphicsBox["));
      assert!(
        gbox.contains("LineBox[CompressedData["),
        "Plot should use CompressedData for line points"
      );
    }
  }
}

mod plot3d {
  use super::*;

  mod basic {
    use super::*;

    #[test]
    fn paraboloid() {
      insta::assert_snapshot!(export_svg(
        "Plot3D[x^2 + y^2, {x, -2, 2}, {y, -2, 2}]"
      ));
    }

    #[test]
    fn trig_function() {
      insta::assert_snapshot!(export_svg(
        "Plot3D[Sin[x] * Cos[y], {x, -Pi, Pi}, {y, -Pi, Pi}]"
      ));
    }

    #[test]
    fn constant_function() {
      insta::assert_snapshot!(export_svg("Plot3D[5, {x, -1, 1}, {y, -1, 1}]"));
    }

    #[test]
    fn multiple_functions() {
      insta::assert_snapshot!(export_svg(
        "Plot3D[{x^2, -x^2}, {x, -1, 1}, {y, -1, 1}]"
      ));
    }

    #[test]
    fn nan_handling() {
      insta::assert_snapshot!(export_svg(
        "Plot3D[1/(x^2 + y^2), {x, -1, 1}, {y, -1, 1}]"
      ));
    }
  }

  mod options {
    use super::*;

    #[test]
    fn image_size_integer() {
      insta::assert_snapshot!(export_svg(
        "Plot3D[x + y, {x, -1, 1}, {y, -1, 1}, ImageSize -> 200]"
      ));
    }

    #[test]
    fn mesh_none() {
      insta::assert_snapshot!(export_svg(
        "Plot3D[x + y, {x, -1, 1}, {y, -1, 1}, Mesh -> None]"
      ));
    }

    #[test]
    fn plot_range() {
      insta::assert_snapshot!(export_svg(
        "Plot3D[x^2 + y^2, {x, -2, 2}, {y, -2, 2}, PlotRange -> {0, 4}]"
      ));
    }
  }

  mod errors {
    use super::*;

    #[test]
    fn too_few_args() {
      let result = interpret("Plot3D[x^2, {x, -1, 1}]").unwrap();
      assert!(
        result.contains("Plot3D"),
        "Should return unevaluated: {}",
        result
      );
    }

    #[test]
    fn invalid_iterator() {
      assert!(interpret("Plot3D[x^2, {x, -1, 1}, 5]").is_err());
    }
  }

  mod export {
    use super::*;

    #[test]
    fn export_svg() {
      let result = interpret(
        "Export[\"/tmp/test_plot3d.svg\", Plot3D[x + y, {x, -1, 1}, {y, -1, 1}]]",
      );
      assert!(result.is_ok());
      let content = std::fs::read_to_string("/tmp/test_plot3d.svg").unwrap();
      assert!(content.starts_with("<svg"));
      assert!(content.contains("<polygon"));
      std::fs::remove_file("/tmp/test_plot3d.svg").ok();
    }
  }

  mod graphics3d {
    use super::*;

    #[test]
    fn graphics3d_sphere() {
      insta::assert_snapshot!(export_svg("Graphics3D[Sphere[]]"));
    }

    #[test]
    fn graphics3d_arrow_with_background() {
      insta::assert_snapshot!(export_svg(
        "Graphics3D[{Arrow[{{0,0,0},{1,0,1}}]}, Background -> Red]"
      ));
    }

    #[test]
    fn graphics3d_polygon() {
      insta::assert_snapshot!(export_svg(
        "Graphics3D[Polygon[{{0,0,0}, {0,1,1}, {1,0,0}}]]"
      ));
    }
  }

  mod plot_misc {
    use super::*;

    #[test]
    fn plot_unevaluatable_returns_graphics() {
      // When the function can't be numerically evaluated, Plot still returns
      // -Graphics- (an empty plot). ExportString produces an empty SVG because
      // there are no plottable points, so we test the raw interpret output.
      assert_eq!(
        interpret("Plot[LucasL[1/2, x], {x, -5, 5}]").unwrap(),
        "-Graphics-"
      );
    }
  }

  mod list_plot {
    use super::*;

    #[test]
    fn list_plot_simple_y_values() {
      insta::assert_snapshot!(export_svg("ListPlot[{1, 4, 9}]"));
    }

    #[test]
    fn list_plot_explicit_xy() {
      insta::assert_snapshot!(export_svg("ListPlot[{{1, 2}, {3, 5}, {7, 1}}]"));
    }

    #[test]
    fn list_plot_joined() {
      insta::assert_snapshot!(export_svg(
        "ListPlot[{1, 4, 9}, Joined -> True]"
      ));
    }

    #[test]
    fn list_plot_image_size() {
      insta::assert_snapshot!(export_svg(
        "ListPlot[{1, 2, 3}, ImageSize -> 200]"
      ));
    }

    #[test]
    fn list_line_plot() {
      insta::assert_snapshot!(export_svg("ListLinePlot[{1, 2, 3, 2, 1}]"));
    }

    #[test]
    fn list_line_plot_filling_axis() {
      insta::assert_snapshot!(export_svg(
        "ListLinePlot[Table[Sin[x], {x, -5, 5, 0.2}], Filling -> Axis]"
      ));
    }

    #[test]
    fn list_step_plot() {
      insta::assert_snapshot!(export_svg("ListStepPlot[{1, 3, 2, 4}]"));
    }

    #[test]
    fn list_log_plot() {
      insta::assert_snapshot!(export_svg("ListLogPlot[{1, 10, 100, 1000}]"));
    }

    #[test]
    fn list_log_log_plot() {
      insta::assert_snapshot!(export_svg(
        "ListLogLogPlot[{{1, 10}, {10, 100}, {100, 1000}}]"
      ));
    }

    #[test]
    fn list_log_linear_plot() {
      insta::assert_snapshot!(export_svg(
        "ListLogLinearPlot[{{1, 2}, {10, 5}, {100, 8}}]"
      ));
    }

    #[test]
    fn list_polar_plot() {
      insta::assert_snapshot!(export_svg("ListPolarPlot[{1, 2, 3, 2, 1}]"));
    }

    #[test]
    fn list_plot_single_element() {
      insta::assert_snapshot!(export_svg("ListPlot[{5}]"));
    }
  }

  mod parametric_plot {
    use super::*;

    #[test]
    fn parametric_plot_circle() {
      insta::assert_snapshot!(export_svg(
        "ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2 Pi}]"
      ));
    }

    #[test]
    fn parametric_plot_lissajous() {
      insta::assert_snapshot!(export_svg(
        "ParametricPlot[{Sin[2 t], Sin[3 t]}, {t, 0, 2 Pi}]"
      ));
    }

    #[test]
    fn parametric_plot_image_size() {
      insta::assert_snapshot!(export_svg(
        "ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2 Pi}, ImageSize -> 200]"
      ));
    }

    #[test]
    fn polar_plot_cardioid() {
      insta::assert_snapshot!(export_svg(
        "PolarPlot[1 + Cos[t], {t, 0, 2 Pi}]"
      ));
    }

    #[test]
    fn polar_plot_rose() {
      insta::assert_snapshot!(export_svg("PolarPlot[Cos[3 t], {t, 0, 2 Pi}]"));
    }
  }

  mod charts {
    use super::*;

    #[test]
    fn bar_chart_basic() {
      insta::assert_snapshot!(export_svg("BarChart[{1, 2, 3}]"));
    }

    #[test]
    fn bar_chart_image_size() {
      insta::assert_snapshot!(export_svg(
        "BarChart[{1, 2, 3}, ImageSize -> 400]"
      ));
    }

    #[test]
    fn pie_chart_basic() {
      insta::assert_snapshot!(export_svg("PieChart[{30, 20, 10}]"));
    }

    #[test]
    fn pie_chart_single_slice() {
      insta::assert_snapshot!(export_svg("PieChart[{100}]"));
    }

    #[test]
    fn histogram_basic() {
      insta::assert_snapshot!(export_svg("Histogram[{1, 2, 2, 3, 3, 3}]"));
    }

    #[test]
    fn histogram_single_value() {
      insta::assert_snapshot!(export_svg("Histogram[{5}]"));
    }

    #[test]
    fn box_whisker_chart() {
      insta::assert_snapshot!(export_svg(
        "BoxWhiskerChart[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}]"
      ));
    }

    #[test]
    fn box_whisker_chart_multiple_datasets() {
      insta::assert_snapshot!(export_svg(
        "BoxWhiskerChart[{{12, 15, 18, 22}, {8, 10, 14, 16}, {5, 9, 12, 15}}]"
      ));
    }

    #[test]
    fn bubble_chart() {
      insta::assert_snapshot!(export_svg(
        "BubbleChart[{{1, 2, 3}, {4, 5, 1}, {2, 3, 5}}]"
      ));
    }

    #[test]
    fn sector_chart() {
      insta::assert_snapshot!(export_svg(
        "SectorChart[{{1, 2}, {2, 3}, {3, 1}}]"
      ));
    }

    #[test]
    fn sector_chart_simple_values() {
      insta::assert_snapshot!(export_svg("SectorChart[{1, 2, 3}]"));
    }

    #[test]
    fn date_list_plot() {
      insta::assert_snapshot!(export_svg("DateListPlot[{1, 3, 2, 5, 4}]"));
    }

    #[test]
    fn bar_chart_chart_labels() {
      insta::assert_snapshot!(export_svg(
        r#"BarChart[{5, 9, 24}, ChartLabels -> {"Anna", "Ben", "Carl"}]"#
      ));
    }

    #[test]
    fn bar_chart_plot_label() {
      insta::assert_snapshot!(export_svg(
        r#"BarChart[{1, 2, 3}, PlotLabel -> "My Title"]"#
      ));
    }

    #[test]
    fn bar_chart_axes_label() {
      insta::assert_snapshot!(export_svg(
        r#"BarChart[{1, 2, 3}, AxesLabel -> {"X Axis", "Y Axis"}]"#
      ));
    }

    #[test]
    fn bar_chart_all_labels() {
      insta::assert_snapshot!(export_svg(
        r#"BarChart[{5, 9, 24, 12, 11}, ChartLabels -> {"Anna", "Ben", "Carl", "Marc", "Sven"}, PlotLabel -> "Fruit Consumption", AxesLabel -> {"Person", "Fruits"}]"#
      ));
    }

    #[test]
    fn word_cloud_basic() {
      insta::assert_snapshot!(export_svg(
        r#"WordCloud[{"hello", "world", "hello", "foo", "hello", "world"}]"#
      ));
    }

    #[test]
    fn word_cloud_single_word() {
      insta::assert_snapshot!(export_svg(r#"WordCloud[{"only"}]"#));
    }

    #[test]
    fn word_cloud_many_words() {
      insta::assert_snapshot!(export_svg(
        r#"WordCloud[{"Lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "dolor", "Pellentesque", "dolor", "augue", "sit"}]"#
      ));
    }
  }

  mod field_plots {
    use super::*;

    #[test]
    fn density_plot_basic() {
      insta::assert_snapshot!(export_svg(
        "DensityPlot[x^2 + y^2, {x, -1, 1}, {y, -1, 1}]"
      ));
    }

    #[test]
    fn density_plot_image_size() {
      insta::assert_snapshot!(export_svg(
        "DensityPlot[Sin[x] * Cos[y], {x, -Pi, Pi}, {y, -Pi, Pi}, ImageSize -> 200]"
      ));
    }

    #[test]
    fn contour_plot_basic() {
      insta::assert_snapshot!(export_svg(
        "ContourPlot[x^2 + y^2, {x, -1, 1}, {y, -1, 1}]"
      ));
    }

    #[test]
    fn region_plot_basic() {
      insta::assert_snapshot!(export_svg(
        "RegionPlot[x^2 + y^2 < 1, {x, -2, 2}, {y, -2, 2}]"
      ));
    }

    #[test]
    fn vector_plot_basic() {
      insta::assert_snapshot!(export_svg(
        "VectorPlot[{-y, x}, {x, -2, 2}, {y, -2, 2}]"
      ));
    }

    #[test]
    fn stream_plot_basic() {
      insta::assert_snapshot!(export_svg(
        "StreamPlot[{-y, x}, {x, -2, 2}, {y, -2, 2}]"
      ));
    }

    #[test]
    fn stream_density_plot_basic() {
      insta::assert_snapshot!(export_svg(
        "StreamDensityPlot[{-y, x}, {x, -2, 2}, {y, -2, 2}]"
      ));
    }

    #[test]
    fn array_plot_basic() {
      insta::assert_snapshot!(export_svg(
        "ArrayPlot[{{0, 0.5, 1}, {1, 0.5, 0}, {0.5, 1, 0.5}}]"
      ));
    }

    #[test]
    fn matrix_plot_basic() {
      insta::assert_snapshot!(export_svg(
        "MatrixPlot[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]"
      ));
    }
  }

  mod graphics3d_primitives {
    use super::*;

    #[test]
    fn graphics3d_sphere() {
      insta::assert_snapshot!(export_svg("Graphics3D[Sphere[]]"));
    }

    #[test]
    fn graphics3d_cuboid() {
      insta::assert_snapshot!(export_svg(
        "Graphics3D[Cuboid[{0, 0, 0}, {1, 1, 1}]]"
      ));
    }

    #[test]
    fn graphics3d_polygon() {
      insta::assert_snapshot!(export_svg(
        "Graphics3D[Polygon[{{0,0,0}, {1,0,0}, {0,1,0}}]]"
      ));
    }

    #[test]
    fn graphics3d_line() {
      insta::assert_snapshot!(export_svg(
        "Graphics3D[Line[{{0,0,0}, {1,1,1}}]]"
      ));
    }

    #[test]
    fn graphics3d_point() {
      insta::assert_snapshot!(export_svg("Graphics3D[Point[{0, 0, 0}]]"));
    }

    #[test]
    fn graphics3d_arrow() {
      insta::assert_snapshot!(export_svg(
        "Graphics3D[Arrow[{{0,0,0},{1,0,1}}]]"
      ));
    }

    #[test]
    fn graphics3d_cylinder() {
      insta::assert_snapshot!(export_svg(
        "Graphics3D[Cylinder[{{0,0,0},{0,0,1}}, 0.5]]"
      ));
    }

    #[test]
    fn graphics3d_cone() {
      insta::assert_snapshot!(export_svg(
        "Graphics3D[Cone[{{0,0,0},{0,0,1}}, 0.5]]"
      ));
    }

    #[test]
    fn graphics3d_multiple_primitives() {
      insta::assert_snapshot!(export_svg(
        "Graphics3D[{Sphere[{0,0,0}, 0.5], Cuboid[{1,1,1}, {2,2,2}]}]"
      ));
    }

    #[test]
    fn graphics3d_image_size() {
      insta::assert_snapshot!(export_svg(
        "Graphics3D[Sphere[], ImageSize -> 200]"
      ));
    }

    #[test]
    fn graphics3d_background() {
      insta::assert_snapshot!(export_svg(
        "Graphics3D[{Arrow[{{0,0,0},{1,0,1}}]}, Background -> Red]"
      ));
    }
  }
}
