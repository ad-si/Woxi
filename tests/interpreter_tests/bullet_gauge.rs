use super::*;

#[test]
fn bullet_gauge_is_graphics() {
  assert_eq!(
    interpret(
      "{Head[BulletGauge[7, 9, {0, 10}]], Head[BulletGauge[{1., 2., 3.}, {1.5}, {1, 1.5, 4}, ScaleDivisions -> {3, 1}, GaugeLabels -> \"pH\", GaugeStyle -> {Red, Green}]]}"
    )
    .unwrap(),
    "{Graphics, Graphics}"
  );
}

// The scale runs over {0, 1}: the only plain `Line[{{x, _}, _}]` is the
// scale line itself, and no bar starts at {0., 0.35}.
#[test]
fn bars_longest_first_and_reference_marker() {
  assert_eq!(
    interpret(
      "g = BulletGauge[{3, 8}, 5, {0, 10}]; {Cases[g, Rectangle[{0., 0.35}, {x_, _}] :> x, Infinity], Cases[g, Line[{{x_, _}, _}] :> x, Infinity]}"
    )
    .unwrap(),
    "{{}, {0.}}"
  );
}

// Bars are rotated onto the scale, longest first, with their exact
// fraction of the range; markers sit at the machine fraction.
#[test]
fn bars_markers_and_bands_as_fractions() {
  assert_eq!(
    interpret(
      "g = BulletGauge[{3, 8}, 5, {0, 10}]; {Cases[g, Tooltip[{_, Rectangle[_, {_, h_}, ___]}, v_] :> {h, v}, Infinity], Cases[g, GeometricTransformation[_, {_, t_}] :> t, Infinity]}"
    )
    .unwrap(),
    "{{{4/5, 8}, {3/10, 3}}, {{0.5, 0.05}}}"
  );
  assert_eq!(
    interpret(
      "Cases[BulletGauge[5, 7, {0, 3, 6, 10}], {Directive[_], Rectangle[{a_, 0}, {b_, 0.1}, ___]} :> {a, b}, Infinity]"
    )
    .unwrap(),
    "{{0, 3/10}, {3/10, 3/5}, {3/5, 1}}"
  );
  // Out-of-range values are clipped; ticks come from FindDivisions.
  assert_eq!(
    interpret(
      "g = BulletGauge[12, 7, {2, 9}]; {Cases[g, Tooltip[{_, Rectangle[_, {_, h_}, ___]}, v_] :> {h, v}, Infinity], Cases[g, Text[t_, ___] :> t, Infinity]}"
    )
    .unwrap(),
    "{{{1., 9.}}, {2, 4, 6, 8}}"
  );
  // ScaleDivisions labels a fractional step to its decimals.
  assert_eq!(
    interpret(
      "ToString[Cases[BulletGauge[5, 7, {0, 1}, ScaleDivisions -> 4], Text[t_, ___] :> t, Infinity], InputForm]"
    )
    .unwrap(),
    "{0, NumberForm[0.25, {Infinity, 2}], NumberForm[0.5, {Infinity, 2}], NumberForm[0.75, {Infinity, 2}], NumberForm[1., {Infinity, 2}]}"
  );
  // The reference marker takes the series colour after the bars.
  assert_eq!(
    interpret(
      "Cases[BulletGauge[{3, 8}, 5, {0, 10}], Directive[c_RGBColor, Opacity[0.5]] :> c, Infinity, 1]"
    )
    .unwrap(),
    "{RGBColor[0.5356156238679548, 0.73, 0.1971]}"
  );
}

/// Regression: the scale's ticks and labels sit at `Scaled`/`Offset`
/// positions, which the renderer dropped, so the SVG had no scale labels.
#[test]
fn bullet_gauge_draws_its_scale_labels() {
  let svg =
    interpret("ExportString[BulletGauge[70, 80, {0, 50, 75, 100}], \"SVG\"]")
      .unwrap();
  for label in ["0", "20", "40", "60", "80", "100"] {
    assert!(svg.contains(&format!(">{label}</text>")), "{label}: {svg}");
  }
}
