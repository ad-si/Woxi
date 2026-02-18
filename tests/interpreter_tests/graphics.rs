use super::*;

mod graphics {
  use super::*;

  mod basic {
    use super::*;

    #[test]
    fn returns_graphics_placeholder() {
      assert_eq!(interpret("Graphics[{Circle[]}]").unwrap(), "-Graphics-");
    }

    #[test]
    fn returns_graphics_with_disk() {
      assert_eq!(interpret("Graphics[{Disk[]}]").unwrap(), "-Graphics-");
    }

    #[test]
    fn empty_list_content() {
      assert_eq!(interpret("Graphics[{}]").unwrap(), "-Graphics-");
    }
  }

  mod primitives {
    use super::*;

    #[test]
    fn point_single() {
      assert_eq!(
        interpret("Graphics[{Point[{0, 0}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn point_multi() {
      assert_eq!(
        interpret("Graphics[{Point[{{0, 0}, {1, 1}, {2, 0}}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn line() {
      assert_eq!(
        interpret("Graphics[{Line[{{0, 0}, {1, 1}, {2, 0}}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn circle_default() {
      assert_eq!(interpret("Graphics[{Circle[]}]").unwrap(), "-Graphics-");
    }

    #[test]
    fn circle_with_center() {
      assert_eq!(
        interpret("Graphics[{Circle[{1, 2}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn circle_with_radius() {
      assert_eq!(
        interpret("Graphics[{Circle[{0, 0}, 2]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn disk_default() {
      assert_eq!(interpret("Graphics[{Disk[]}]").unwrap(), "-Graphics-");
    }

    #[test]
    fn disk_with_center_radius() {
      assert_eq!(
        interpret("Graphics[{Disk[{1, 0}, 0.5]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn rectangle_default() {
      assert_eq!(interpret("Graphics[{Rectangle[]}]").unwrap(), "-Graphics-");
    }

    #[test]
    fn rectangle_with_corners() {
      assert_eq!(
        interpret("Graphics[{Rectangle[{0, 0}, {2, 3}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn polygon() {
      assert_eq!(
        interpret("Graphics[{Polygon[{{0, 0}, {1, 0}, {0.5, 1}}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn arrow() {
      assert_eq!(
        interpret("Graphics[{Arrow[{{0, 0}, {1, 1}}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn text() {
      assert_eq!(
        interpret("Graphics[{Text[\"Hello\", {0, 0}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn bezier_curve() {
      assert_eq!(
        interpret("Graphics[{BezierCurve[{{0, 0}, {0.5, 1}, {1, 0}}]}]")
          .unwrap(),
        "-Graphics-"
      );
    }
  }

  mod styles {
    use super::*;

    #[test]
    fn named_color() {
      assert_eq!(interpret("Graphics[{Red, Disk[]}]").unwrap(), "-Graphics-");
    }

    #[test]
    fn rgb_color() {
      assert_eq!(
        interpret("Graphics[{RGBColor[0.5, 0.3, 0.8], Circle[]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn opacity() {
      assert_eq!(
        interpret("Graphics[{Opacity[0.5], Disk[]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn thickness() {
      assert_eq!(
        interpret("Graphics[{Thickness[0.01], Line[{{0, 0}, {1, 1}}]}]")
          .unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn dashing() {
      assert_eq!(
        interpret("Graphics[{Dashing[{0.02, 0.02}], Line[{{0, 0}, {1, 1}}]}]")
          .unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn point_size() {
      assert_eq!(
        interpret("Graphics[{PointSize[0.03], Point[{0, 0}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn darker() {
      assert_eq!(
        interpret("Graphics[{Darker[Red], Disk[]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn lighter() {
      assert_eq!(
        interpret("Graphics[{Lighter[Blue], Disk[]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn style_scoping() {
      // Red only applies within the nested list; outer Circle is still black
      assert_eq!(
        interpret("Graphics[{{Red, Disk[]}, Circle[{2, 0}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn edge_form() {
      assert_eq!(
        interpret("Graphics[{EdgeForm[Red], Disk[]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn multiple_colors() {
      assert_eq!(
        interpret("Graphics[{Red, Disk[], Blue, Circle[{2, 0}, 0.5]}]")
          .unwrap(),
        "-Graphics-"
      );
    }
  }

  mod text_styles {
    use super::*;

    #[test]
    fn text_with_style_bold() {
      assert_eq!(
        interpret("Graphics[{Text[Style[\"Hello\", Bold], {0, 0}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn text_with_style_italic() {
      assert_eq!(
        interpret("Graphics[{Text[Style[\"World\", Italic], {0, 0}]}]")
          .unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn text_with_font_size() {
      assert_eq!(
        interpret("Graphics[{Text[Style[\"Big\", 20], {0, 0}]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn text_with_color() {
      assert_eq!(
        interpret("Graphics[{Text[Style[\"Red Text\", Red], {0, 0}]}]")
          .unwrap(),
        "-Graphics-"
      );
    }
  }

  mod options {
    use super::*;

    #[test]
    fn image_size_integer() {
      assert_eq!(
        interpret("Graphics[{Circle[]}, ImageSize -> 200]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn image_size_pair() {
      assert_eq!(
        interpret("Graphics[{Circle[]}, ImageSize -> {400, 300}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn image_size_named() {
      assert_eq!(
        interpret("Graphics[{Circle[]}, ImageSize -> Large]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn plot_range() {
      assert_eq!(
        interpret("Graphics[{Circle[]}, PlotRange -> {{-2, 2}, {-2, 2}}]")
          .unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn background() {
      assert_eq!(
        interpret("Graphics[{Circle[]}, Background -> LightGray]").unwrap(),
        "-Graphics-"
      );
    }
  }

  mod integration {
    use super::*;

    #[test]
    fn complex_diagram() {
      assert_eq!(
        interpret(
          "Graphics[{
            Red, Disk[{0, 0}, 0.5],
            Blue, Circle[{2, 0}, 0.8],
            Green, Polygon[{{4, 0}, {5, 1}, {5, -1}}],
            Black, Line[{{0, 0}, {2, 0}, {4, 0}}],
            Orange, Arrow[{{-1, -1}, {-1, 1}}],
            Text[\"Origin\", {0, -0.8}]
          }, ImageSize -> 400]"
        )
        .unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn nested_style_scoping() {
      assert_eq!(
        interpret(
          "Graphics[{
            {Red, Disk[{0, 0}]},
            {Blue, Disk[{3, 0}]},
            Circle[{1.5, 2}]
          }]"
        )
        .unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn graphics_primitives_are_symbolic() {
      // Graphics primitives should be returned as-is when not inside Graphics[]
      assert_eq!(interpret("Circle[]").unwrap(), "Circle[]");
      assert_eq!(interpret("Disk[{1, 0}]").unwrap(), "Disk[{1, 0}]");
      assert_eq!(interpret("Point[{0, 0}]").unwrap(), "Point[{0, 0}]");
      assert_eq!(interpret("RGBColor[1, 0, 0]").unwrap(), "RGBColor[1, 0, 0]");
    }

    #[test]
    fn hue_color() {
      assert_eq!(
        interpret("Graphics[{Hue[0.6], Disk[]}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn directive_compound() {
      assert_eq!(
        interpret(
          "Graphics[{Directive[Red, Thickness[0.01]], Line[{{0, 0}, {1, 1}}]}]"
        )
        .unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn graphics_evaluates_table_content() {
      // Graphics should evaluate its content (e.g. Table) before rendering
      assert_eq!(
        interpret("Graphics[Table[Disk[{i, 0}, 0.5], {i, 3}]]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      // Should have 3 disks
      assert_eq!(svg.matches("ellipse").count(), 3);
    }

    #[test]
    fn graphics_nested_table() {
      // 2D Table produces nested lists — Graphics should handle them
      assert_eq!(
        interpret("Graphics[Table[Disk[{r*Cos[2 Pi q/4], r*Sin[2 Pi q/4]}, 0.3], {r, 1, 2}, {q, 4}]]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert_eq!(svg.matches("ellipse").count(), 8);
    }

    #[test]
    fn graphics_table_with_symbolic_pi_step() {
      assert_eq!(
        interpret("Graphics[{Yellow, Disk[{0, 0}, 0.3], Pink, Table[Disk[{Cos[θ], Sin[θ]}, 0.25], {θ, 0, 2 Pi - Pi/4, Pi/4}]}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert_eq!(svg.matches("ellipse").count(), 9);
    }

    #[test]
    fn edgeform_with_list_arg() {
      // EdgeForm[{GrayLevel[0, 0.5]}] — list-wrapped directive
      assert_eq!(
        interpret("Graphics[{EdgeForm[{GrayLevel[0, 0.5]}], Disk[]}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("stroke=\"rgb(0,0,0)\""));
      assert!(svg.contains("stroke-opacity=\"0.5\""));
    }

    #[test]
    fn hue_with_alpha_fill_opacity() {
      // Hue[h, s, b, alpha] should produce fill-opacity, not opacity
      assert_eq!(
        interpret("Graphics[{Hue[0, 1, 1, 0.6], Disk[]}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("fill-opacity=\"0.6\""), "SVG: {}", svg);
      assert!(svg.contains("fill=\"rgb(255,0,0)\""), "SVG: {}", svg);
    }

    #[test]
    fn separate_fill_and_stroke_opacity() {
      // Fill and stroke should have separate opacities
      assert_eq!(
        interpret(
          "Graphics[{EdgeForm[{GrayLevel[0, 0.5]}], Hue[0, 1, 1, 0.6], Disk[]}]"
        )
        .unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("fill-opacity=\"0.6\""), "SVG: {}", svg);
      assert!(svg.contains("stroke-opacity=\"0.5\""), "SVG: {}", svg);
    }

    #[test]
    fn square_aspect_ratio_for_symmetric_data() {
      // Symmetric data should produce a square SVG
      assert_eq!(
        interpret("Graphics[{Disk[{1, 0}, 0.5], Disk[{-1, 0}, 0.5], Disk[{0, 1}, 0.5], Disk[{0, -1}, 0.5]}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      // Should have equal width and height
      assert!(
        svg.contains("width=\"360\" height=\"360\""),
        "SVG header: {}",
        &svg[..100]
      );
    }

    #[test]
    fn circle_has_equal_radii() {
      // Graphics[Circle[]] must produce a round circle (equal rx and ry)
      assert_eq!(interpret("Graphics[Circle[]]").unwrap(), "-Graphics-");
      let svg = woxi::get_captured_graphics().unwrap();
      // SVG should be square for symmetric data
      assert!(
        svg.contains("width=\"360\" height=\"360\""),
        "SVG should be square for Circle[]. Got: {}",
        &svg[..svg.find('>').unwrap_or(100) + 1]
      );
      // The ellipse rx and ry should be equal
      let rx_start = svg.find("rx=\"").unwrap() + 4;
      let rx_end = rx_start + svg[rx_start..].find('"').unwrap();
      let ry_start = svg.find("ry=\"").unwrap() + 4;
      let ry_end = ry_start + svg[ry_start..].find('"').unwrap();
      assert_eq!(
        &svg[rx_start..rx_end],
        &svg[ry_start..ry_end],
        "Circle rx and ry must be equal"
      );
    }

    #[test]
    fn circle_round_with_explicit_image_size() {
      // Even with non-square ImageSize, circles should be round
      assert_eq!(
        interpret("Graphics[Circle[], ImageSize -> {400, 200}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      // SVG should have the explicit dimensions
      assert!(
        svg.contains("width=\"400\" height=\"200\""),
        "SVG should use explicit ImageSize. Got: {}",
        &svg[..svg.find('>').unwrap_or(100) + 1]
      );
      let rx_start = svg.find("rx=\"").unwrap() + 4;
      let rx_end = rx_start + svg[rx_start..].find('"').unwrap();
      let ry_start = svg.find("ry=\"").unwrap() + 4;
      let ry_end = ry_start + svg[ry_start..].find('"').unwrap();
      assert_eq!(
        &svg[rx_start..rx_end],
        &svg[ry_start..ry_end],
        "Circle rx and ry must be equal even with explicit ImageSize. SVG: {}",
        svg
      );
    }

    #[test]
    fn preserves_aspect_ratio_attribute() {
      // SVG should include preserveAspectRatio for robust rendering
      assert_eq!(interpret("Graphics[Circle[]]").unwrap(), "-Graphics-");
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(
        svg.contains("preserveAspectRatio=\"xMidYMid meet\""),
        "SVG should have preserveAspectRatio. Got: {}",
        &svg[..svg.find('>').unwrap_or(100) + 1]
      );
    }

    #[test]
    fn full_hue_rings_expression() {
      // The target expression from the issue
      assert_eq!(
        interpret("Graphics[Table[{EdgeForm[{GrayLevel[0, 0.5]}], Hue[(-11+q+10r)/72, 1, 1, 0.6], Disk[(8-r){Cos[2Pi q/12], Sin[2Pi q/12]}, (8-r)/3]}, {r, 6}, {q, 12}]]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      // 6 rings × 12 disks = 72 ellipses
      assert_eq!(svg.matches("ellipse").count(), 72);
      // All should have stroke from EdgeForm
      assert_eq!(svg.matches("stroke=\"rgb(0,0,0)\"").count(), 72);
      // All should have fill-opacity from Hue alpha
      assert_eq!(svg.matches("fill-opacity=\"0.6\"").count(), 72);
    }

    #[test]
    fn hue_accepts_leading_dot_real_literals() {
      assert_eq!(
        interpret("Graphics[Table[{Hue[t/15, 1, .9, .3], Disk[{Cos[2 Pi t/15], Sin[2 Pi t/15]}]}, {t, 15}]]").unwrap(),
        "-Graphics-"
      );
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

    /// Mimics the exact WASM evaluate flow: interpret_with_stdout then get_captured_graphicsbox
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
    fn returns_graphics3d_placeholder() {
      assert_eq!(
        interpret("Plot3D[x^2 + y^2, {x, -2, 2}, {y, -2, 2}]").unwrap(),
        "-Graphics3D-"
      );
    }

    #[test]
    fn trig_function() {
      assert_eq!(
        interpret("Plot3D[Sin[x] * Cos[y], {x, -Pi, Pi}, {y, -Pi, Pi}]")
          .unwrap(),
        "-Graphics3D-"
      );
    }

    #[test]
    fn constant_function() {
      assert_eq!(
        interpret("Plot3D[5, {x, -1, 1}, {y, -1, 1}]").unwrap(),
        "-Graphics3D-"
      );
    }

    #[test]
    fn multiple_functions() {
      assert_eq!(
        interpret("Plot3D[{x^2, -x^2}, {x, -1, 1}, {y, -1, 1}]").unwrap(),
        "-Graphics3D-"
      );
    }

    #[test]
    fn nan_handling() {
      // 1/(x^2+y^2) has a singularity at origin but should still produce a plot
      assert_eq!(
        interpret("Plot3D[1/(x^2 + y^2), {x, -1, 1}, {y, -1, 1}]").unwrap(),
        "-Graphics3D-"
      );
    }
  }

  mod svg_capture {
    use super::*;

    #[test]
    fn captures_svg_with_polygons() {
      interpret("Plot3D[x^2 + y^2, {x, -2, 2}, {y, -2, 2}]").unwrap();
      let svg = woxi::get_captured_graphics();
      assert!(svg.is_some(), "SVG should be captured for Plot3D");
      let svg = svg.unwrap();
      assert!(svg.starts_with("<svg"), "Should be an SVG");
      assert!(svg.contains("<polygon"), "Should contain polygon elements");
      assert!(svg.contains("</svg>"), "Should be a complete SVG");
    }
  }

  mod options {
    use super::*;

    #[test]
    fn image_size_integer() {
      assert_eq!(
        interpret("Plot3D[x + y, {x, -1, 1}, {y, -1, 1}, ImageSize -> 200]")
          .unwrap(),
        "-Graphics3D-"
      );
    }

    #[test]
    fn mesh_none() {
      interpret("Plot3D[x + y, {x, -1, 1}, {y, -1, 1}, Mesh -> None]").unwrap();
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(
        svg.contains("stroke=\"none\""),
        "Mesh -> None should disable mesh stroke"
      );
    }

    #[test]
    fn plot_range() {
      assert_eq!(
        interpret(
          "Plot3D[x^2 + y^2, {x, -2, 2}, {y, -2, 2}, PlotRange -> {0, 4}]"
        )
        .unwrap(),
        "-Graphics3D-"
      );
    }
  }

  mod errors {
    use super::*;

    #[test]
    fn too_few_args() {
      // With fewer than 3 args, Plot3D returns unevaluated (not an error)
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
    fn graphics3d_returns_marker() {
      assert_eq!(interpret("Graphics3D[Sphere[]]").unwrap(), "-Graphics3D-");
      assert_eq!(
        interpret("Graphics3D[{Arrow[{{0,0,0},{1,0,1}}]}, Background -> Red]")
          .unwrap(),
        "-Graphics3D-"
      );
      assert_eq!(
        interpret("Graphics3D[Polygon[{{0,0,0}, {0,1,1}, {1,0,0}}]]").unwrap(),
        "-Graphics3D-"
      );
    }
  }

  mod plot_misc {
    use super::*;

    #[test]
    fn plot_unevaluatable_returns_graphics() {
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
      assert_eq!(interpret("ListPlot[{1, 4, 9}]").unwrap(), "-Graphics-");
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"), "Should produce SVG");
      assert!(svg.contains("<circle"), "Scatter plot should have circles");
    }

    #[test]
    fn list_plot_explicit_xy() {
      assert_eq!(
        interpret("ListPlot[{{1, 2}, {3, 5}, {7, 1}}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("<circle"), "Should have scatter circles");
    }

    #[test]
    fn list_plot_joined() {
      assert_eq!(
        interpret("ListPlot[{1, 4, 9}, Joined -> True]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("</svg>"), "Should be complete SVG");
    }

    #[test]
    fn list_plot_image_size() {
      assert_eq!(
        interpret("ListPlot[{1, 2, 3}, ImageSize -> 200]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn list_line_plot() {
      assert_eq!(
        interpret("ListLinePlot[{1, 2, 3, 2, 1}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(svg.contains("</svg>"));
    }

    #[test]
    fn list_step_plot() {
      assert_eq!(
        interpret("ListStepPlot[{1, 3, 2, 4}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn list_log_plot() {
      assert_eq!(
        interpret("ListLogPlot[{1, 10, 100, 1000}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn list_log_log_plot() {
      assert_eq!(
        interpret("ListLogLogPlot[{{1, 10}, {10, 100}, {100, 1000}}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn list_log_linear_plot() {
      assert_eq!(
        interpret("ListLogLinearPlot[{{1, 2}, {10, 5}, {100, 8}}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn list_polar_plot() {
      assert_eq!(
        interpret("ListPolarPlot[{1, 2, 3, 2, 1}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn list_plot_single_element() {
      assert_eq!(interpret("ListPlot[{5}]").unwrap(), "-Graphics-");
    }
  }

  mod parametric_plot {
    use super::*;

    #[test]
    fn parametric_plot_circle() {
      assert_eq!(
        interpret("ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2 Pi}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(svg.contains("</svg>"));
    }

    #[test]
    fn parametric_plot_lissajous() {
      assert_eq!(
        interpret("ParametricPlot[{Sin[2 t], Sin[3 t]}, {t, 0, 2 Pi}]")
          .unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn parametric_plot_image_size() {
      assert_eq!(
        interpret(
          "ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2 Pi}, ImageSize -> 200]"
        )
        .unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn polar_plot_cardioid() {
      assert_eq!(
        interpret("PolarPlot[1 + Cos[t], {t, 0, 2 Pi}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn polar_plot_rose() {
      assert_eq!(
        interpret("PolarPlot[Cos[3 t], {t, 0, 2 Pi}]").unwrap(),
        "-Graphics-"
      );
    }
  }

  mod charts {
    use super::*;

    #[test]
    fn bar_chart_basic() {
      assert_eq!(interpret("BarChart[{1, 2, 3}]").unwrap(), "-Graphics-");
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(svg.contains("<rect"), "BarChart should have rect elements");
    }

    #[test]
    fn bar_chart_image_size() {
      assert_eq!(
        interpret("BarChart[{1, 2, 3}, ImageSize -> 400]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn pie_chart_basic() {
      assert_eq!(interpret("PieChart[{30, 20, 10}]").unwrap(), "-Graphics-");
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(svg.contains("<path"), "PieChart should have arc paths");
    }

    #[test]
    fn pie_chart_single_slice() {
      assert_eq!(interpret("PieChart[{100}]").unwrap(), "-Graphics-");
    }

    #[test]
    fn histogram_basic() {
      assert_eq!(
        interpret("Histogram[{1, 2, 2, 3, 3, 3}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(svg.contains("<rect"), "Histogram should have rect elements");
    }

    #[test]
    fn histogram_single_value() {
      assert_eq!(interpret("Histogram[{5}]").unwrap(), "-Graphics-");
    }

    #[test]
    fn box_whisker_chart() {
      assert_eq!(
        interpret("BoxWhiskerChart[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(svg.contains("<rect"), "Should have box rectangle");
    }

    #[test]
    fn box_whisker_chart_multiple_datasets() {
      assert_eq!(
        interpret(
          "BoxWhiskerChart[{{12, 15, 18, 22}, {8, 10, 14, 16}, {5, 9, 12, 15}}]"
        )
        .unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(
        svg.contains("<rect"),
        "Should have box rectangles for each dataset"
      );
      // Should have 3 boxes (one per dataset)
      let rect_count = svg.matches("fill=\"rgb(").count();
      assert!(
        rect_count >= 3,
        "Should have at least 3 colored elements for 3 datasets, got {rect_count}"
      );
    }

    #[test]
    fn bubble_chart() {
      assert_eq!(
        interpret("BubbleChart[{{1, 2, 3}, {4, 5, 1}, {2, 3, 5}}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(svg.contains("<circle"), "BubbleChart should have circles");
    }

    #[test]
    fn sector_chart() {
      assert_eq!(
        interpret("SectorChart[{{1, 2}, {2, 3}, {3, 1}}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("<path"), "SectorChart should have arc paths");
    }

    #[test]
    fn sector_chart_simple_values() {
      assert_eq!(interpret("SectorChart[{1, 2, 3}]").unwrap(), "-Graphics-");
    }

    #[test]
    fn date_list_plot() {
      assert_eq!(
        interpret("DateListPlot[{1, 3, 2, 5, 4}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn bar_chart_chart_labels() {
      assert_eq!(
        interpret(
          r#"BarChart[{5, 9, 24}, ChartLabels -> {"Anna", "Ben", "Carl"}]"#
        )
        .unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("Anna"), "SVG should contain chart label Anna");
      assert!(svg.contains("Ben"), "SVG should contain chart label Ben");
      assert!(svg.contains("Carl"), "SVG should contain chart label Carl");
    }

    #[test]
    fn bar_chart_plot_label() {
      assert_eq!(
        interpret(r#"BarChart[{1, 2, 3}, PlotLabel -> "My Title"]"#).unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("My Title"), "SVG should contain plot label");
    }

    #[test]
    fn bar_chart_axes_label() {
      assert_eq!(
        interpret(r#"BarChart[{1, 2, 3}, AxesLabel -> {"X Axis", "Y Axis"}]"#)
          .unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("X Axis"), "SVG should contain x-axis label");
      assert!(svg.contains("Y Axis"), "SVG should contain y-axis label");
    }

    #[test]
    fn bar_chart_all_labels() {
      assert_eq!(
        interpret(
          r#"BarChart[{5, 9, 24, 12, 11}, ChartLabels -> {"Anna", "Ben", "Carl", "Marc", "Sven"}, PlotLabel -> "Fruit Consumption", AxesLabel -> {"Person", "Fruits"}]"#
        )
        .unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("Anna"), "ChartLabels");
      assert!(svg.contains("Sven"), "ChartLabels");
      assert!(svg.contains("Fruit Consumption"), "PlotLabel");
      assert!(svg.contains("Person"), "AxesLabel x");
      assert!(svg.contains("Fruits"), "AxesLabel y");
    }
  }

  mod field_plots {
    use super::*;

    #[test]
    fn density_plot_basic() {
      assert_eq!(
        interpret("DensityPlot[x^2 + y^2, {x, -1, 1}, {y, -1, 1}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(
        svg.contains("<rect"),
        "DensityPlot should have colored cells"
      );
    }

    #[test]
    fn density_plot_image_size() {
      assert_eq!(
        interpret("DensityPlot[Sin[x] * Cos[y], {x, -Pi, Pi}, {y, -Pi, Pi}, ImageSize -> 200]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn contour_plot_basic() {
      assert_eq!(
        interpret("ContourPlot[x^2 + y^2, {x, -1, 1}, {y, -1, 1}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      // Should have both colored background and contour lines
      assert!(svg.contains("<rect"), "Should have density background");
      assert!(svg.contains("<line"), "Should have contour lines");
    }

    #[test]
    fn region_plot_basic() {
      assert_eq!(
        interpret("RegionPlot[x^2 + y^2 < 1, {x, -2, 2}, {y, -2, 2}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(
        svg.contains("<rect"),
        "RegionPlot should have colored cells"
      );
    }

    #[test]
    fn vector_plot_basic() {
      assert_eq!(
        interpret("VectorPlot[{-y, x}, {x, -2, 2}, {y, -2, 2}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(svg.contains("<line"), "VectorPlot should have arrow lines");
      assert!(
        svg.contains("<polygon"),
        "VectorPlot should have arrowheads"
      );
    }

    #[test]
    fn stream_plot_basic() {
      assert_eq!(
        interpret("StreamPlot[{-y, x}, {x, -2, 2}, {y, -2, 2}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(
        svg.contains("<polyline"),
        "StreamPlot should have streamlines"
      );
    }

    #[test]
    fn stream_density_plot_basic() {
      assert_eq!(
        interpret("StreamDensityPlot[{-y, x}, {x, -2, 2}, {y, -2, 2}]")
          .unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(svg.contains("<rect"), "Should have density background");
      assert!(svg.contains("<polyline"), "Should have streamlines");
    }

    #[test]
    fn array_plot_basic() {
      assert_eq!(
        interpret("ArrayPlot[{{0, 0.5, 1}, {1, 0.5, 0}, {0.5, 1, 0.5}}]")
          .unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(svg.contains("<rect"), "ArrayPlot should have colored cells");
    }

    #[test]
    fn matrix_plot_basic() {
      assert_eq!(
        interpret("MatrixPlot[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
        "-Graphics-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(
        svg.contains("<rect"),
        "MatrixPlot should have colored cells"
      );
    }
  }

  mod graphics3d_primitives {
    use super::*;

    #[test]
    fn graphics3d_sphere() {
      assert_eq!(interpret("Graphics3D[Sphere[]]").unwrap(), "-Graphics3D-");
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(
        svg.contains("<polygon"),
        "Sphere should be tessellated into polygons"
      );
    }

    #[test]
    fn graphics3d_cuboid() {
      assert_eq!(
        interpret("Graphics3D[Cuboid[{0, 0, 0}, {1, 1, 1}]]").unwrap(),
        "-Graphics3D-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("<polygon"), "Cuboid should have polygon faces");
    }

    #[test]
    fn graphics3d_polygon() {
      assert_eq!(
        interpret("Graphics3D[Polygon[{{0,0,0}, {1,0,0}, {0,1,0}}]]").unwrap(),
        "-Graphics3D-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("<polygon"));
    }

    #[test]
    fn graphics3d_line() {
      assert_eq!(
        interpret("Graphics3D[Line[{{0,0,0}, {1,1,1}}]]").unwrap(),
        "-Graphics3D-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("<polyline"), "Line should produce polyline");
    }

    #[test]
    fn graphics3d_point() {
      assert_eq!(
        interpret("Graphics3D[Point[{0, 0, 0}]]").unwrap(),
        "-Graphics3D-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("<circle"), "Point should produce circle");
    }

    #[test]
    fn graphics3d_arrow() {
      assert_eq!(
        interpret("Graphics3D[Arrow[{{0,0,0},{1,0,1}}]]").unwrap(),
        "-Graphics3D-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("<polyline"), "Arrow should have shaft");
      assert!(svg.contains("<polygon"), "Arrow should have arrowhead");
    }

    #[test]
    fn graphics3d_cylinder() {
      assert_eq!(
        interpret("Graphics3D[Cylinder[{{0,0,0},{0,0,1}}, 0.5]]").unwrap(),
        "-Graphics3D-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("<polygon"), "Cylinder should be tessellated");
    }

    #[test]
    fn graphics3d_cone() {
      assert_eq!(
        interpret("Graphics3D[Cone[{{0,0,0},{0,0,1}}, 0.5]]").unwrap(),
        "-Graphics3D-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("<polygon"), "Cone should be tessellated");
    }

    #[test]
    fn graphics3d_multiple_primitives() {
      assert_eq!(
        interpret(
          "Graphics3D[{Sphere[{0,0,0}, 0.5], Cuboid[{1,1,1}, {2,2,2}]}]"
        )
        .unwrap(),
        "-Graphics3D-"
      );
      let svg = woxi::get_captured_graphics().unwrap();
      assert!(svg.contains("<polygon"));
    }

    #[test]
    fn graphics3d_image_size() {
      assert_eq!(
        interpret("Graphics3D[Sphere[], ImageSize -> 200]").unwrap(),
        "-Graphics3D-"
      );
    }

    #[test]
    fn graphics3d_background() {
      assert_eq!(
        interpret("Graphics3D[{Arrow[{{0,0,0},{1,0,1}}]}, Background -> Red]")
          .unwrap(),
        "-Graphics3D-"
      );
    }
  }
}
