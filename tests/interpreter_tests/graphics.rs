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
    fn arrow_setback_zero() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Arrow[{{0, 0}, {1, 1}}, {0, 0}]}]"
      ));
    }

    #[test]
    fn arrow_setback_symmetric() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Arrow[{{0, 0}, {2, 0}}, {0.5, 0.5}]}]"
      ));
    }

    #[test]
    fn arrow_setback_scalar() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Arrow[{{0, 0}, {2, 0}}, 0.5]}]"
      ));
    }

    #[test]
    fn arrow_setback_exceeds_length() {
      // Setback exceeds total path length; should produce no arrow line
      insta::assert_snapshot!(export_svg(
        "Graphics[{Circle[], Arrow[{{0, 0}, {1, 0}}, {0.6, 0.6}]}]"
      ));
    }

    #[test]
    fn arrow_setback_multipoint() {
      insta::assert_snapshot!(export_svg(
        "Graphics[{Arrow[{{0, 0}, {1, 0}, {1, 1}}, {0.5, 0.5}]}]"
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

    #[test]
    fn axes_true() {
      let svg = export_svg("Graphics[Circle[], Axes -> True]");
      assert!(svg.contains(
        "<line x1=\"0.00\" y1=\"180.00\" x2=\"360.00\" y2=\"180.00\" stroke=\"#b3b3b3\" stroke-width=\"1\"/>"
      ));
      assert!(svg.contains(
        "<line x1=\"180.00\" y1=\"0.00\" x2=\"180.00\" y2=\"360.00\" stroke=\"#b3b3b3\" stroke-width=\"1\"/>"
      ));
      assert!(svg.matches("<line ").count() > 2);
      assert!(svg.contains("<text "));
      assert!(!svg.contains(">0</text>"));
    }

    #[test]
    fn axes_list_x_only() {
      let svg = export_svg("Graphics[Circle[], Axes -> {True, False}]");
      assert!(svg.contains(
        "<line x1=\"0.00\" y1=\"180.00\" x2=\"360.00\" y2=\"180.00\" stroke=\"#b3b3b3\" stroke-width=\"1\"/>"
      ));
      assert!(!svg.contains(
        "<line x1=\"180.00\" y1=\"0.00\" x2=\"180.00\" y2=\"360.00\" stroke=\"#b3b3b3\" stroke-width=\"1\"/>"
      ));
      assert!(svg.matches("<line ").count() > 1);
      assert!(svg.contains("<text "));
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

    #[test]
    fn trailing_semicolon_suppresses_graphics() {
      let result =
        woxi::interpret_with_stdout("Plot[Sin[x], {x, 0, 2 Pi}];").unwrap();
      assert_eq!(
        result.result, "Null",
        "Trailing semicolon should suppress result to Null"
      );
    }

    #[test]
    fn assignment_with_semicolon_suppresses_graphics() {
      let result =
        woxi::interpret_with_stdout("p = Plot[Sin[x], {x, 0, 2 Pi}];").unwrap();
      assert_eq!(
        result.result, "Null",
        "Assignment with trailing semicolon should suppress result to Null"
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

  mod plot_options {
    use super::*;

    #[test]
    fn plot_plot_label() {
      insta::assert_snapshot!(export_svg(
        r#"Plot[Sin[x], {x, 0, 2 Pi}, PlotLabel -> "Sine Wave"]"#
      ));
    }

    #[test]
    fn plot_plot_label_styled() {
      insta::assert_snapshot!(export_svg(
        r#"Plot[Sin[x], {x, 0, 2 Pi}, PlotLabel -> Style["Sine", Bold, Italic]]"#
      ));
    }

    #[test]
    fn plot_axes_label() {
      insta::assert_snapshot!(export_svg(
        r#"Plot[x^2, {x, -2, 2}, AxesLabel -> {"x", "y"}]"#
      ));
    }

    #[test]
    fn plot_plot_style_single() {
      insta::assert_snapshot!(export_svg(
        r#"Plot[Sin[x], {x, 0, 2 Pi}, PlotStyle -> Red]"#
      ));
    }

    #[test]
    fn plot_plot_style_multi() {
      insta::assert_snapshot!(export_svg(
        r#"Plot[{Sin[x], Cos[x]}, {x, 0, 2 Pi}, PlotStyle -> {Red, Blue}]"#
      ));
    }

    #[test]
    fn plot_all_options() {
      insta::assert_snapshot!(export_svg(
        r#"Plot[{Sin[x], Cos[x]}, {x, 0, 2 Pi}, PlotLabel -> "Trig Functions", AxesLabel -> {"x", "f(x)"}, PlotStyle -> {Red, Blue}]"#
      ));
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
    fn box_whisker_chart_plot_label() {
      insta::assert_snapshot!(export_svg(
        r#"BoxWhiskerChart[{1, 2, 3, 4, 5}, PlotLabel -> "My Title"]"#
      ));
    }

    #[test]
    fn box_whisker_chart_chart_labels() {
      insta::assert_snapshot!(export_svg(
        r#"BoxWhiskerChart[{{12, 15, 18, 22}, {8, 10, 14, 16}}, ChartLabels -> {"A", "B"}]"#
      ));
    }

    #[test]
    fn box_whisker_chart_frame_label() {
      insta::assert_snapshot!(export_svg(
        r#"BoxWhiskerChart[{{12, 15, 18, 22}, {8, 10, 14, 16}}, FrameLabel -> {"X Axis", "Y Axis"}]"#
      ));
    }

    #[test]
    fn box_whisker_chart_chart_style() {
      insta::assert_snapshot!(export_svg(
        r##"BoxWhiskerChart[{{12, 15, 18, 22}, {8, 10, 14, 16}}, ChartStyle -> RGBColor["#2a9d8f"]]"##
      ));
    }

    #[test]
    fn box_whisker_chart_all_options() {
      insta::assert_snapshot!(export_svg(
        r##"BoxWhiskerChart[{{12, 15, 18, 22}, {8, 10, 14, 16}}, PlotLabel -> Style["Test", Bold], ChartLabels -> {"A", "B"}, FrameLabel -> {"X", "Y"}, ChartStyle -> RGBColor["#2a9d8f"]]"##
      ));
    }

    #[test]
    fn box_whisker_chart_styled_plot_label() {
      insta::assert_snapshot!(export_svg(
        r#"BoxWhiskerChart[{1, 2, 3, 4, 5}, PlotLabel -> Style["Styled Title", Bold, Italic]]"#
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
    fn rgbcolor_hex_string() {
      insta::assert_snapshot!(export_svg(
        r##"Graphics[{RGBColor["#264653"], Disk[]}]"##
      ));
    }

    #[test]
    fn bar_chart_chart_style() {
      insta::assert_snapshot!(export_svg(
        r##"BarChart[{1, 2, 3}, ChartStyle -> {RGBColor["#264653"], RGBColor["#2a9d8f"], RGBColor["#e9c46a"]}]"##
      ));
    }

    #[test]
    fn bar_chart_plot_label_styled_bold() {
      insta::assert_snapshot!(export_svg(
        r#"BarChart[{1, 2, 3}, PlotLabel -> Style["Title", Bold]]"#
      ));
    }

    #[test]
    fn bar_chart_chart_style_and_styled_plot_label() {
      insta::assert_snapshot!(export_svg(
        r##"BarChart[{1, 2, 3}, ChartStyle -> {RGBColor["#264653"], RGBColor["#2a9d8f"], RGBColor["#e9c46a"]}, PlotLabel -> Style["My Chart", Bold, Italic]]"##
      ));
    }

    #[test]
    fn bar_chart_chart_style_cycling() {
      insta::assert_snapshot!(export_svg(
        r#"BarChart[{1, 2, 3, 4, 5}, ChartStyle -> {Red, Blue}]"#
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

    #[test]
    fn list_density_plot_matrix() {
      insta::assert_snapshot!(export_svg(
        "ListDensityPlot[{{1, 1, 1, 1}, {1, 2, 1, 2}, {1, 1, 3, 1}, {1, 2, 1, 4}}]"
      ));
    }

    #[test]
    fn list_density_plot_triples() {
      insta::assert_snapshot!(export_svg(
        "ListDensityPlot[{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}}]"
      ));
    }

    #[test]
    fn list_density_plot_image_size() {
      insta::assert_snapshot!(export_svg(
        "ListDensityPlot[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}, ImageSize -> 200]"
      ));
    }
  }

  mod list_contour_plots {
    use super::*;

    #[test]
    fn list_contour_plot_matrix() {
      insta::assert_snapshot!(export_svg(
        "ListContourPlot[{{1, 1, 1, 1}, {1, 2, 1, 2}, {1, 1, 3, 1}, {1, 2, 1, 4}}]"
      ));
    }

    #[test]
    fn list_contour_plot_triples() {
      insta::assert_snapshot!(export_svg(
        "ListContourPlot[{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}}]"
      ));
    }

    #[test]
    fn list_contour_plot_image_size() {
      insta::assert_snapshot!(export_svg(
        "ListContourPlot[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}, ImageSize -> 200]"
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

mod graphics_list {
  use super::*;

  #[test]
  fn single_graphics_still_works() {
    clear_state();
    let result = interpret_with_stdout("Graphics[{Circle[]}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains("<svg"));
    assert!(svg.contains("</svg>"));
  }

  #[test]
  fn table_1d_list_of_graphics() {
    clear_state();
    let result =
      interpret_with_stdout("Table[Graphics[{Circle[]}], {i, 1, 3}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    // Should be a combined SVG containing nested SVGs
    assert!(svg.contains("<svg"));
    // Should contain multiple nested SVG elements (one per cell)
    let nested_count = svg.matches("<svg x=").count();
    assert_eq!(nested_count, 3, "Expected 3 nested SVGs for 3-element list");
    // Should have list braces and comma separators
    assert!(svg.contains(">{</text>"));
    assert!(svg.contains(">}</text>"));
    assert!(svg.contains(">,</text>"));
  }

  #[test]
  fn table_2d_list_of_graphics() {
    clear_state();
    let result = interpret_with_stdout(
      "Table[Graphics[{RGBColor[r, g, 0], Rectangle[]}], {r, 0, 1, 0.5}, {g, 0, 1, 0.5}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    // 3 rows x 3 cols = 9 cells
    let nested_count = svg.matches("<svg x=").count();
    assert_eq!(nested_count, 9, "Expected 9 nested SVGs for 3x3 grid");
  }

  #[test]
  fn table_3d_list_with_tableform_mathmlform() {
    clear_state();
    let result = interpret_with_stdout(
      "Table[Graphics[{RGBColor[r, g, b], Rectangle[]}], {r, 0, 1, 1}, {g, 0, 1, 1}, {b, 0, 1, 1}] // TableForm // MathMLForm",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    // TableForm transposes: 2 blocks × (2 rows × 2 cols) = 4 rows × 2 cols = 8
    let nested_count = svg.matches("<svg x=").count();
    assert_eq!(
      nested_count, 8,
      "Expected 8 nested SVGs for transposed grid"
    );
  }

  #[test]
  fn table_3d_list_without_tableform() {
    clear_state();
    let result = interpret_with_stdout(
      "Table[Graphics[{RGBColor[r, g, b], Rectangle[]}], {r, 0, 1, 1}, {g, 0, 1, 1}, {b, 0, 1, 1}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    // Without TableForm: dim1=2 rows, dim2×dim3=2×2=4 cols per row = 8 total
    let nested_count = svg.matches("<svg x=").count();
    assert_eq!(nested_count, 8, "Expected 8 nested SVGs for 2x4 grid");
  }

  #[test]
  fn mathmlform_is_transparent() {
    clear_state();
    assert_eq!(interpret("MathMLForm[1 + 2]").unwrap(), "3");
    assert_eq!(interpret("StandardForm[3 * 4]").unwrap(), "12");
    assert_eq!(interpret("InputForm[{1, 2, 3}]").unwrap(), "{1, 2, 3}");
    assert_eq!(interpret("OutputForm[42]").unwrap(), "42");
  }

  #[test]
  fn table_with_style_wrapping_graphics() {
    clear_state();
    let result = interpret_with_stdout(
      "Table[Style[Graphics[{Rectangle[]}], ImageSizeMultipliers -> {0.2, 1}], {i, 1, 2}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    let nested_count = svg.matches("<svg x=").count();
    assert_eq!(nested_count, 2, "Expected 2 nested SVGs");
  }

  #[test]
  fn tableform_wrapping_1d_graphics_list() {
    clear_state();
    let result =
      interpret_with_stdout("TableForm[Table[Graphics[{Disk[]}], {i, 1, 4}]]")
        .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    let nested_count = svg.matches("<svg x=").count();
    assert_eq!(nested_count, 4, "Expected 4 nested SVGs for TableForm list");
  }

  #[test]
  fn tableform_2d_list() {
    clear_state();
    let result =
      interpret_with_stdout("TableForm[{{1, 2, 3}, {4, 5, 6}}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">6</text>"));
  }

  #[test]
  fn tableform_1d_list_as_column() {
    clear_state();
    let result = interpret_with_stdout("TableForm[{10, 20, 30}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains(">10</text>"));
    assert!(svg.contains(">20</text>"));
    assert!(svg.contains(">30</text>"));
  }

  #[test]
  fn tableform_3d_list_transposed_blocks() {
    // TableForm on 3D list: each block is transposed (sub-lists become columns),
    // blocks stacked vertically.
    // Input: {{{a,b,c},{d,e,f}}, {{g,h,i},{j,k,l}}}
    // Block 1 transposed: a d / b e / c f  (3 rows × 2 cols)
    // Block 2 transposed: g j / h k / i l  (3 rows × 2 cols)
    // Stacked: 6 rows × 2 cols
    clear_state();
    let result = interpret_with_stdout(
      "TableForm[{{{a, b, c}, {d, e, f}}, {{g, h, i}, {j, k, l}}}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    // All 12 elements should appear
    for ch in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"] {
      assert!(
        svg.contains(&format!(">{ch}</text>")),
        "Missing element {ch} in SVG"
      );
    }
  }
}

mod matrix_form {
  use super::*;

  #[test]
  fn matrix_form_non_visual() {
    clear_state();
    assert_eq!(
      interpret("MatrixForm[{{1,2},{3,4}}]").unwrap(),
      "MatrixForm[{{1, 2}, {3, 4}}]"
    );
  }

  #[test]
  fn matrix_form_visual_2d() {
    clear_state();
    let result = interpret_with_stdout("MatrixForm[{{1,2},{3,4}}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains("<svg"));
    for val in ["1", "2", "3", "4"] {
      assert!(
        svg.contains(&format!(">{val}</text>")),
        "Missing value {val} in MatrixForm SVG"
      );
    }
  }

  #[test]
  fn matrix_form_visual_1d_column() {
    clear_state();
    let result = interpret_with_stdout("MatrixForm[{a, b, c}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    for val in ["a", "b", "c"] {
      assert!(
        svg.contains(&format!(">{val}</text>")),
        "Missing value {val} in MatrixForm column SVG"
      );
    }
  }

  #[test]
  fn matrix_form_evaluates_args() {
    clear_state();
    assert_eq!(
      interpret("MatrixForm[{{1+1, 2+2},{3+3, 4+4}}]").unwrap(),
      "MatrixForm[{{2, 4}, {6, 8}}]"
    );
  }

  #[test]
  fn matrix_form_has_parentheses() {
    clear_state();
    let result = interpret_with_stdout("MatrixForm[{{1,2},{3,4}}]").unwrap();
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    // The SVG should contain path elements for the parentheses
    let path_count = svg.matches("<path ").count();
    assert_eq!(
      path_count, 2,
      "MatrixForm SVG should have exactly 2 parenthesis paths, got {path_count}"
    );
  }

  #[test]
  fn matrix_form_nested_3d() {
    clear_state();
    let result =
      interpret_with_stdout("MatrixForm[{{{1,2},{3,4}}, {{5,6},{7,8}}}]")
        .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains("<svg"));
    // Should have 10 parenthesis paths:
    // 2 outer parens + 4 cells × 2 sub-parens each = 10
    let path_count = svg.matches("<path ").count();
    assert_eq!(
      path_count, 10,
      "Nested MatrixForm should have 10 parenthesis paths, got {path_count}"
    );
    // All values should appear
    for val in ["1", "2", "3", "4", "5", "6", "7", "8"] {
      assert!(
        svg.contains(&format!(">{val}</text>")),
        "Missing value {val} in nested MatrixForm SVG"
      );
    }

    // Verify spatial layout: "1" and "2" should share the same x (stacked vertically)
    // and "1" should be above "2" (smaller y).
    // Parse text positions from SVG
    fn parse_text_pos(svg: &str, label: &str) -> Option<(f64, f64)> {
      let needle = format!(">{label}</text>");
      let idx = svg.find(&needle)?;
      let before = &svg[..idx];
      // Find the enclosing <text> tag
      let tag_start = before.rfind("<text ")?;
      let tag = &svg[tag_start..idx];
      let x_start = tag.find("x=\"")? + 3;
      let x_end = tag[x_start..].find('"')? + x_start;
      let y_start = tag.find("y=\"")? + 3;
      let y_end = tag[y_start..].find('"')? + y_start;
      let x: f64 = tag[x_start..x_end].parse().ok()?;
      let y: f64 = tag[y_start..y_end].parse().ok()?;
      Some((x, y))
    }

    let pos1 = parse_text_pos(&svg, "1").expect("1 not found");
    let pos2 = parse_text_pos(&svg, "2").expect("2 not found");
    let pos3 = parse_text_pos(&svg, "3").expect("3 not found");

    // 1 and 2 are in the same column vector → same x
    assert!(
      (pos1.0 - pos2.0).abs() < 1.0,
      "1 and 2 should have same x (stacked vertically), got x1={} x2={}",
      pos1.0,
      pos2.0
    );
    // 1 above 2 → smaller y
    assert!(
      pos1.1 < pos2.1,
      "1 should be above 2, got y1={} y2={}",
      pos1.1,
      pos2.1
    );
    // 3 is in a different column → different x from 1
    assert!(
      (pos3.0 - pos1.0).abs() > 10.0,
      "3 should be in a different column from 1, got x1={} x3={}",
      pos1.0,
      pos3.0
    );
  }
}

mod show {
  use super::*;

  #[test]
  fn show_two_graphics() {
    clear_state();
    assert_eq!(
      interpret(
        "Show[Graphics[{Red, Disk[]}], Graphics[{Blue, Circle[{1,0}]}]]"
      )
      .unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn show_svg_output() {
    clear_state();
    let result = interpret_with_stdout(
      "Show[Graphics[{Red, Disk[]}], Graphics[{Blue, Circle[{1,0}]}]]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains("<svg"));
    assert!(svg.contains("</svg>"));
  }

  #[test]
  fn show_single_graphics() {
    clear_state();
    assert_eq!(
      interpret("Show[Graphics[{Circle[]}]]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn show_no_graphics_returns_unevaluated() {
    clear_state();
    assert_eq!(interpret("Show[1, 2, 3]").unwrap(), "Show[1, 2, 3]");
  }
}

mod list_plot_3d {
  use super::*;

  #[test]
  fn list_plot3d_explicit_coords() {
    clear_state();
    assert_eq!(
      interpret("ListPlot3D[{{0,0,0},{1,0,1},{0,1,2},{1,1,3}}]").unwrap(),
      "-Graphics3D-"
    );
  }

  #[test]
  fn list_plot3d_matrix_format() {
    clear_state();
    assert_eq!(
      interpret("ListPlot3D[{{1,2,3},{4,5,6},{7,8,9}}]").unwrap(),
      "-Graphics3D-"
    );
  }

  #[test]
  fn list_plot3d_svg_output() {
    clear_state();
    let result =
      interpret_with_stdout("ListPlot3D[{{1,2,3},{4,5,6},{7,8,9}}]").unwrap();
    assert_eq!(result.result, "-Graphics3D-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains("<svg"));
    assert!(svg.contains("</svg>"));
  }

  #[test]
  fn list_plot3d_with_options() {
    clear_state();
    assert_eq!(
      interpret("ListPlot3D[{{1,2,3},{4,5,6}}, Mesh -> None]").unwrap(),
      "-Graphics3D-"
    );
  }
}

mod tree_form_graphics {
  use super::*;

  #[test]
  fn tree_form_produces_graphics() {
    clear_state();
    assert_eq!(interpret("TreeForm[f[x, y]]").unwrap(), "-Graphics-");
  }

  #[test]
  fn tree_form_svg_output() {
    clear_state();
    let result = interpret_with_stdout("TreeForm[f[x, y]]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
    let svg = result.graphics.unwrap();
    assert!(svg.contains("<svg"));
    assert!(svg.contains("</svg>"));
    // Should contain the function name and arguments as text
    assert!(svg.contains(">f<"), "SVG should contain node label 'f'");
    assert!(svg.contains(">x<"), "SVG should contain leaf label 'x'");
    assert!(svg.contains(">y<"), "SVG should contain leaf label 'y'");
  }

  #[test]
  fn tree_form_nested() {
    clear_state();
    assert_eq!(
      interpret("TreeForm[f[g[x], h[y, z]]]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn tree_form_with_depth_limit() {
    clear_state();
    assert_eq!(interpret("TreeForm[f[g[h[x]]], 2]").unwrap(), "-Graphics-");
  }

  #[test]
  fn tree_form_atom() {
    clear_state();
    assert_eq!(interpret("TreeForm[42]").unwrap(), "-Graphics-");
  }

  #[test]
  fn tree_form_no_args() {
    clear_state();
    assert_eq!(interpret("TreeForm[]").unwrap(), "TreeForm[]");
  }

  #[test]
  fn tree_form_minus_canonical() {
    // a - b should decompose as Plus[a, Times[-1, b]], not "Minus"
    clear_state();
    let result = interpret_with_stdout("TreeForm[Hold[a - b]]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains(">Plus<"),
      "SVG should contain 'Plus' for subtraction, got SVG without it"
    );
    assert!(
      svg.contains(">Times<"),
      "SVG should contain 'Times' for the -1 multiplication"
    );
    assert!(
      !svg.contains(">Minus<"),
      "SVG should NOT contain raw 'Minus' operator name"
    );
  }

  #[test]
  fn tree_form_divide_canonical() {
    // a / b should decompose as Times[a, Power[b, -1]], not "Divide"
    clear_state();
    let result = interpret_with_stdout("TreeForm[Hold[a / b]]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains(">Times<"),
      "SVG should contain 'Times' for division"
    );
    assert!(
      svg.contains(">Power<"),
      "SVG should contain 'Power' for the b^-1"
    );
  }

  #[test]
  fn tree_form_unary_minus_canonical() {
    // -x should decompose as Times[-1, x]
    clear_state();
    let result = interpret_with_stdout("TreeForm[Hold[-x]]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains(">Times<"),
      "SVG should contain 'Times' for unary minus"
    );
  }

  #[test]
  fn tree_form_comparison_canonical() {
    // a < b should decompose as Less[a, b]
    clear_state();
    let result = interpret_with_stdout("TreeForm[Hold[a < b]]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains(">Less<"),
      "SVG should contain 'Less' for comparison"
    );
  }

  #[test]
  fn tree_form_association() {
    clear_state();
    let result =
      interpret_with_stdout("TreeForm[<|\"a\" -> 1, \"b\" -> 2|>]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains(">Association<"),
      "SVG should contain 'Association' head"
    );
    assert!(
      svg.contains(">Rule<"),
      "SVG should contain 'Rule' for key-value pairs"
    );
  }

  #[test]
  fn tree_form_replace_all() {
    clear_state();
    let result = interpret_with_stdout("TreeForm[Hold[x /. x -> 1]]").unwrap();
    let svg = result.graphics.unwrap();
    assert!(
      svg.contains(">ReplaceAll<"),
      "SVG should contain 'ReplaceAll' head"
    );
  }
}

mod graphics_row {
  use super::*;

  #[test]
  fn basic_inline() {
    clear_state();
    let result = interpret(
      "GraphicsRow[{Plot[Sin[x], {x, 0, 2 Pi}], Plot[Cos[x], {x, 0, 2 Pi}]}]",
    )
    .unwrap();
    assert_eq!(result, "-Graphics-");
  }

  #[test]
  fn with_variables() {
    clear_state();
    let result = interpret(
      "p1 = Plot[Sin[x], {x, 0, 2 Pi}]; p2 = Plot[Cos[x], {x, 0, 2 Pi}]; GraphicsRow[{p1, p2}]",
    )
    .unwrap();
    assert_eq!(result, "-Graphics-");
  }

  #[test]
  fn three_plots_with_spacings() {
    clear_state();
    let result = interpret(
      "p1 = Plot[Sin[x], {x, 0, 2 Pi}]; p2 = Plot[Cos[x], {x, 0, 2 Pi}]; p3 = Plot[Sin[x] Cos[x], {x, 0, 2 Pi}]; GraphicsRow[{p1, p2, p3}, Spacings -> 0.5]",
    )
    .unwrap();
    assert_eq!(result, "-Graphics-");
  }

  #[test]
  fn produces_combined_svg() {
    clear_state();
    let result = interpret_with_stdout(
      "p1 = Plot[Sin[x], {x, 0, 2 Pi}]; p2 = Plot[Cos[x], {x, 0, 2 Pi}]; GraphicsRow[{p1, p2}]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    // Combined SVG should contain nested <svg> elements
    assert!(svg.starts_with("<svg"), "Should produce SVG output");
    // Count nested <svg> tags (should have at least 2 for the two plots)
    let nested_count = svg.matches("<svg ").count();
    assert!(
      nested_count >= 3,
      "Should have outer + 2 nested SVGs, got {}",
      nested_count
    );
  }

  #[test]
  fn single_element() {
    clear_state();
    let result =
      interpret("GraphicsRow[{Plot[Sin[x], {x, 0, 2 Pi}]}]").unwrap();
    assert_eq!(result, "-Graphics-");
  }

  #[test]
  fn empty_list() {
    clear_state();
    let result = interpret("GraphicsRow[{}]").unwrap();
    assert_eq!(result, "-Graphics-");
  }

  #[test]
  fn with_graphics_primitives() {
    clear_state();
    let result = interpret(
      "g1 = Graphics[{Circle[]}]; g2 = Graphics[{Disk[]}]; GraphicsRow[{g1, g2}]",
    )
    .unwrap();
    assert_eq!(result, "-Graphics-");
  }
}

mod graphics_column {
  use super::*;

  #[test]
  fn basic() {
    clear_state();
    let result = interpret(
      "GraphicsColumn[{Plot[Sin[x], {x, 0, 2 Pi}], Plot[Cos[x], {x, 0, 2 Pi}]}]",
    )
    .unwrap();
    assert_eq!(result, "-Graphics-");
  }

  #[test]
  fn with_variables() {
    clear_state();
    let result = interpret(
      "p1 = Plot[Sin[x], {x, 0, 2 Pi}]; p2 = Plot[Cos[x], {x, 0, 2 Pi}]; GraphicsColumn[{p1, p2}]",
    )
    .unwrap();
    assert_eq!(result, "-Graphics-");
  }

  #[test]
  fn produces_vertical_svg() {
    clear_state();
    let result = interpret_with_stdout(
      "p1 = Plot[Sin[x], {x, 0, 2 Pi}]; p2 = Plot[Cos[x], {x, 0, 2 Pi}]; GraphicsColumn[{p1, p2}]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.starts_with("<svg"), "Should produce SVG output");
    let nested_count = svg.matches("<svg ").count();
    assert!(
      nested_count >= 3,
      "Should have outer + 2 nested SVGs, got {}",
      nested_count
    );
  }
}

mod graphics_grid {
  use super::*;

  #[test]
  fn basic_2x2() {
    clear_state();
    let result = interpret(
      "p1 = Plot[Sin[x], {x, 0, 2 Pi}]; p2 = Plot[Cos[x], {x, 0, 2 Pi}]; \
       p3 = Plot[x^2, {x, -2, 2}]; p4 = Plot[x^3, {x, -2, 2}]; \
       GraphicsGrid[{{p1, p2}, {p3, p4}}]",
    )
    .unwrap();
    assert_eq!(result, "-Graphics-");
  }

  #[test]
  fn produces_grid_svg() {
    clear_state();
    let result = interpret_with_stdout(
      "p1 = Plot[Sin[x], {x, 0, 2 Pi}]; p2 = Plot[Cos[x], {x, 0, 2 Pi}]; \
       p3 = Plot[x^2, {x, -2, 2}]; p4 = Plot[x^3, {x, -2, 2}]; \
       GraphicsGrid[{{p1, p2}, {p3, p4}}]",
    )
    .unwrap();
    let svg = result.graphics.unwrap();
    assert!(svg.starts_with("<svg"), "Should produce SVG output");
    // 4 plots + 1 outer SVG = at least 5 <svg> tags
    let nested_count = svg.matches("<svg ").count();
    assert!(
      nested_count >= 5,
      "Should have outer + 4 nested SVGs, got {}",
      nested_count
    );
  }

  #[test]
  fn with_spacings() {
    clear_state();
    let result = interpret(
      "p1 = Graphics[{Circle[]}]; p2 = Graphics[{Disk[]}]; \
       GraphicsGrid[{{p1, p2}}, Spacings -> 1]",
    )
    .unwrap();
    assert_eq!(result, "-Graphics-");
  }
}

mod color_swatches {
  use super::*;

  #[test]
  fn rgbcolor_3_args() {
    clear_state();
    let result = interpret_with_stdout("RGBColor[1, 0, 0]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("fill=\"rgb(255,0,0)\""));
    assert!(svg.contains("width=\"16\""));
    assert!(svg.contains("height=\"16\""));
  }

  #[test]
  fn rgbcolor_hex() {
    clear_state();
    let result = interpret_with_stdout("RGBColor[\"#467396\"]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("fill=\"rgb(70,115,150)\""));
  }

  #[test]
  fn rgbcolor_gray() {
    clear_state();
    let result = interpret_with_stdout("RGBColor[0.5]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("fill=\"rgb(128,128,128)\""));
  }

  #[test]
  fn rgbcolor_with_alpha() {
    clear_state();
    let result = interpret_with_stdout("RGBColor[1, 0, 0, 0.5]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("fill=\"rgb(255,0,0)\""));
    assert!(svg.contains("opacity=\"0.5\""));
  }

  #[test]
  fn hue_swatch() {
    clear_state();
    let result = interpret_with_stdout("Hue[0.5]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn graylevel_swatch() {
    clear_state();
    let result = interpret_with_stdout("GrayLevel[0.3]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    assert!(svg.contains("fill=\"rgb(77,77,77)\""));
  }

  #[test]
  fn darker_swatch() {
    clear_state();
    let result = interpret_with_stdout("Darker[RGBColor[1, 0, 0]]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn lighter_swatch() {
    clear_state();
    let result = interpret_with_stdout("Lighter[RGBColor[0, 0, 1]]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    assert!(result.graphics.is_some());
  }

  #[test]
  fn list_of_colors_inline_swatches() {
    clear_state();
    let result = interpret_with_stdout(
      "{RGBColor[1, 0, 0], RGBColor[0, 1, 0], RGBColor[0, 0, 1]}",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    // Should contain braces as text elements
    assert!(svg.contains(">{</text>"));
    assert!(svg.contains(">}</text>"));
    // Should contain comma separators
    assert!(svg.contains(">,</text>"));
    // Should contain all three color swatches
    assert!(svg.contains("fill=\"rgb(255,0,0)\""));
    assert!(svg.contains("fill=\"rgb(0,255,0)\""));
    assert!(svg.contains("fill=\"rgb(0,0,255)\""));
  }

  #[test]
  fn mixed_list_no_swatch() {
    clear_state();
    let result = interpret_with_stdout("{RGBColor[1, 0, 0], 42}").unwrap();
    // Mixed list should NOT render as color swatches
    assert_ne!(result.result, "-Graphics-");
  }

  #[test]
  fn cli_mode_unchanged() {
    clear_state();
    let result = interpret("RGBColor[1, 0, 0]").unwrap();
    assert_eq!(result, "RGBColor[1, 0, 0]");
  }

  #[test]
  fn named_color_no_swatch() {
    clear_state();
    let result = interpret_with_stdout("Red").unwrap();
    // Named colors should NOT produce swatches
    assert_ne!(result.result, "-Graphics-");
  }
}
