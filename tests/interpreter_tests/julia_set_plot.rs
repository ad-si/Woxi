use super::*;

// The filled Julia set of z^2 + c as a Raster over the plot range: escape
// time coloured through wolframscript's palette, the set itself black.
#[test]
fn quadratic_raster_pixels() {
  assert_eq!(
    interpret(
      "g = JuliaSetPlot[-0.5 + 0.5 I, PlotRange -> 2]; {Head[g[[1]]], Dimensions[g[[1, 1]]], g[[1, 2]], g[[1, 1, 1, 1]], g[[1, 1, 251, 251]], g[[1, 1, 251, 60]], Options[g, Frame]}"
    )
    .unwrap(),
    "{Raster, {501, 501, 3}, {{-2., -2.}, {2., 2.}}, {0.4065135971340173, 0.5398078488845466, 0.7788308093144439}, {0., 0., 0.}, {0.4133528741247354, 0.5186060902133204, 0.7566031590946101}, {Frame -> True}}"
  );
}

// A polynomial in z; the raster's rows follow the range's aspect ratio.
#[test]
fn polynomial_and_aspect_ratio() {
  assert_eq!(
    interpret(
      "{Dimensions[JuliaSetPlot[z^3 + 0.5 z + 0.3 I, z, PlotRange -> 1][[1, 1]]], Dimensions[JuliaSetPlot[-0.5 + 0.5 I][[1, 1]]][[2]]}"
    )
    .unwrap(),
    "{{501, 501, 3}, 501}"
  );
}

// A map that is no polynomial stays unevaluated.
#[test]
fn non_polynomial_unevaluated() {
  assert_eq!(
    interpret("JuliaSetPlot[Sin[z], z]").unwrap(),
    "JuliaSetPlot[Sin[z], z]"
  );
}
