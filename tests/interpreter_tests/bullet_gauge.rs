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

// Every value is a bar from the scale's minimum, the longest drawn first so
// none hides another; the reference is a marker line.
#[test]
fn bars_longest_first_and_reference_marker() {
  assert_eq!(
    interpret(
      "g = BulletGauge[{3, 8}, 5, {0, 10}]; {Cases[g, Rectangle[{0., 0.35}, {x_, _}] :> x, Infinity], Cases[g, Line[{{x_, _}, _}] :> x, Infinity]}"
    )
    .unwrap(),
    "{{8., 3.}, {5.}}"
  );
}
