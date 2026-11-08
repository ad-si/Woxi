use super::*;

// A 240 × 240 RGB texture spanning the plot range, as in wolframscript.
#[test]
fn default_texture_raster() {
  assert_eq!(
    interpret(
      "g = LineIntegralConvolutionPlot[{-y, x}, {x, -1, 1}, {y, -1, 1}]; r = First[Cases[g, _Raster, Infinity]]; {Head[g], Dimensions[r[[1]]], r[[2]]}"
    )
    .unwrap(),
    "{Graphics, {240, 240, 3}, {{-1, -1}, {1, 1}}}"
  );
}

// `{"noise", m, …}` sets an m × m texture; a five-argument color function
// gets x, y, vx, vy and the field's norm.
#[test]
fn noise_size_and_color_function() {
  assert_eq!(
    interpret(
      "g = LineIntegralConvolutionPlot[{{-y, x}, {\"noise\", 50, 40}}, {x, -1, 1}, {y, -1, 1}, ColorFunctionScaling -> False, ColorFunction -> Function[{x, y, vx, vy, n}, Hue[n]]]; Dimensions[First[Cases[g, _Raster, Infinity]][[1]]]"
    )
    .unwrap(),
    "{50, 50, 3}"
  );
}
