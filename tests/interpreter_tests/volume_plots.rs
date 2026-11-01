use super::*;

/// The vertex list of the first `GraphicsComplex` in a plot.
const PTS: &str =
  "pts[g_] := Cases[g, GraphicsComplex[p_, ___] :> p, Infinity][[1]]; ";

mod density_plot3d {
  use super::*;

  // The values are mapped to [1/2, 1] and stored z-layer by y-row.
  #[test]
  fn raster_values_layout() {
    assert_eq!(
      interpret(
        "p = DensityPlot3D[x + 10 y + 100 z, {x, 0, 1}, {y, 0, 2}, {z, 0, 3}, PlotPoints -> 3]; Round[p[[1, 1, 1]], 0.0001]"
      )
      .unwrap(),
      "{{{0.5, 0.5008, 0.5016}, {0.5156000000000001, 0.5164, 0.5171}, {0.5312, 0.5319, 0.5327000000000001}}, {{0.7336, 0.7344, 0.7352000000000001}, {0.7492000000000001, 0.75, 0.7508}, {0.7648, 0.7656000000000001, 0.7664000000000001}}, {{0.9673, 0.9681000000000001, 0.9688}, {0.9829, 0.9836, 0.9844}, {0.9984000000000001, 0.9992000000000001, 1.}}}"
    );
  }

  #[test]
  fn raster_spans_the_box_with_51_points_by_default() {
    assert_eq!(
      interpret(
        "p = DensityPlot3D[x + 10 y + 100 z, {x, 0, 1}, {y, 0, 2}, {z, 0, 3}, PlotPoints -> 3]; {Head[p], Head[p[[1, 1]]], p[[1, 1, 2]], Dimensions[DensityPlot3D[x y z, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}][[1, 1, 1]]]}"
      )
      .unwrap(),
      "{Graphics3D, Raster3D, {{0., 0., 0.}, {1., 2., 3.}}, {51, 51, 51}}"
    );
  }

  #[test]
  fn renders_as_svg() {
    let svg = interpret(
      "ExportString[DensityPlot3D[x y z, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, PlotPoints -> 5], \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg") && svg.contains("<polygon"));
  }
}

mod slice_density_plot3d {
  use super::*;

  #[test]
  fn stacked_planes_are_evenly_spaced_inside_the_range() {
    assert_eq!(
      interpret(&format!(
        "{PTS}{{Union[Round[First /@ pts[SliceDensityPlot3D[x, {{\"XStackedPlanes\", 2}}, {{x, 0, 1}}, {{y, 0, 1}}, {{z, 0, 1}}, PlotPoints -> 3]], 0.01]], Union[Round[First /@ pts[SliceDensityPlot3D[x, \"XStackedPlanes\", {{x, 0, 1}}, {{y, 0, 1}}, {{z, 0, 1}}, PlotPoints -> 3]], 0.01]]}}"
      ))
      .unwrap(),
      "{{0.33, 0.67}, {0.25, 0.5, 0.75}}"
    );
  }

  // An equation that is not a coordinate plane is a surface through the
  // box: here the unit sphere.
  #[test]
  fn implicit_surface_slice() {
    assert_eq!(
      interpret(&format!(
        "{PTS}Max[Abs[Norm /@ pts[SliceDensityPlot3D[x, x^2 + y^2 + z^2 == 1, {{x, -2, 2}}, {{y, -2, 2}}, {{z, -2, 2}}]] - 1]] < 0.05"
      ))
      .unwrap(),
      "True"
    );
  }

  // A `ColorFunction` returning a directive (an opacity with the colour)
  // colours the slice with it, and the surface spec may be computed.
  #[test]
  fn directive_color_function_and_computed_surface() {
    assert_eq!(
      interpret(
        "g = SliceDensityPlot3D[x y, {If[True, {\"ZStackedPlanes\", {0.5}}, 0]}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, PlotPoints -> 3, ColorFunction -> (Directive[Opacity[#1], Red] &)]; {Head[g], MemberQ[g, Directive[Opacity[_], RGBColor[1, 0, 0]], Infinity]}"
      )
      .unwrap(),
      "{Graphics3D, True}"
    );
  }
}

mod slice_contour_plot3d {
  use super::*;

  #[test]
  fn coordinate_plane_slice() {
    assert_eq!(
      interpret(&format!(
        "{PTS}Union[Round[Last /@ pts[SliceContourPlot3D[x y, z == 0.5, {{x, 0, 1}}, {{y, 0, 1}}, {{z, 0, 1}}, PlotPoints -> 5]], 0.01]]"
      ))
      .unwrap(),
      "{0.5}"
    );
  }

  #[test]
  fn plot_legends_wraps_in_legended() {
    assert_eq!(
      interpret(
        "Head[SliceContourPlot3D[x y, z == 0.5, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, PlotLegends -> Automatic]]"
      )
      .unwrap(),
      "Legended"
    );
  }

  // `Contours -> n` splits the slice into n + 1 bands, band k of them
  // coloured by the colour function at k/(n + 2).
  #[test]
  fn contour_bands() {
    assert_eq!(
      interpret(
        "Take[Cases[SliceContourPlot3D[x, z == 0.5, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, Contours -> 3, ColorFunction -> Hue], _Hue, Infinity], 4]"
      )
      .unwrap(),
      "{Hue[1/5], Hue[2/5], Hue[3/5], Hue[4/5]}"
    );
  }
}
