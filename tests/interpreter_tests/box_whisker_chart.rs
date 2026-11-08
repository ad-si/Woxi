use super::*;

#[test]
fn box_coordinates_preserve_rational_quartiles() {
  assert_eq!(
    interpret("Cases[BoxWhiskerChart[{1/3, 2/3, 1, 4/3, 5/3}], Polygon[p_] :> p[[All, 2]], Infinity]").unwrap(),
    "{{17/12, 17/12, 7/12, 7/12}}"
  );
}

// BoxWhiskerChart builds a framed symbolic Graphics: one box polygon per
// dataset, ticks under the boxes, no axes.
#[test]
fn framed_graphics_with_one_box_per_dataset() {
  assert_eq!(
    interpret(
      "g = BoxWhiskerChart[{{1, 2, 2, 3, 3, 3, 4, 4, 5}, {2, 3, 4, 4, 5, 5, 6, 7, 8}, {0, 1, 1, 2, 2, 2, 3, 6, 7}}]; \
       {Head[g], Frame /. Options[g], Axes /. Options[g], Length[Cases[g, _Polygon, Infinity]]}"
    )
    .unwrap(),
    "{Graphics, True, {False, False}, 3}"
  );
}

// The box spans the quartiles, whiskers reach the extremes, the median
// line is white.
#[test]
fn box_spans_quartiles_and_whiskers_reach_extremes() {
  assert_eq!(
    interpret(
      "g = BoxWhiskerChart[{1, 2, 2, 3, 3, 3, 4, 4, 5}]; \
       {MinMax[Cases[g, Polygon[p_] :> p[[All, 2]], Infinity]], \
        MinMax[Cases[g, {GrayLevel[0], CapForm[\"Butt\"], _, Line[l_]} :> l[[All, All, 2]], Infinity]], \
        Cases[g, {GrayLevel[0], GrayLevel[1], Line[l_]} :> l[[1, 2]], Infinity]}"
    )
    .unwrap(),
    "{{2, 4}, {1, 5}, {3}}"
  );
}

// "Outliers": whiskers stop at the last value inside the 1.5 IQR fences;
// the values beyond them are dots (far ones, past 3 IQR, in gray).
#[test]
fn outliers_are_dots_and_whiskers_stop_at_fences() {
  assert_eq!(
    interpret(
      "g = BoxWhiskerChart[{1, 2, 2, 3, 3, 3, 4, 4, 5, 9, 15}, \"Outliers\"]; \
       {Cases[g, Inset[Style[_, 10, GrayLevel[l_]], {_, y_}] :> {l, y}, Infinity], \
        MinMax[Cases[g, {GrayLevel[0], CapForm[\"Butt\"], _, Line[l_]} :> l[[All, All, 2]], Infinity]]}"
    )
    .unwrap(),
    "{{{0, 9}, {0.5, 15}}, {1, 5}}"
  );
}

// "Mean" adds a black line at the mean, half the box wide.
#[test]
fn mean_marker_line() {
  assert_eq!(
    interpret(
      "g = BoxWhiskerChart[{{1, 2, 2, 3, 3, 3, 4, 4, 12}}, \"Mean\"]; \
       Cases[g, {GrayLevel[0], GrayLevel[0], Line[{{x1_, y_}, {x2_, y_}}]} :> {Round[900 y], Round[(x1 - x2)/(1.1960569343063403 - 1), 0.5]}, Infinity]"
    )
    .unwrap(),
    "{{3400, 1.}}"
  );
}

// "Notched" cuts a notch into the box at the median, McGill's
// 1.7 * 1.25 IQR / (1.35 Sqrt[n]) deep, clipped to the box.
#[test]
fn notched_box() {
  assert_eq!(
    interpret(
      "g = BoxWhiskerChart[Range[20]^2, \"Notched\"]; \
       p = Cases[g, Polygon[p_] :> p, Infinity][[1]]; \
       {Length[p], Round[1000 p[[3, 2]]], p[[4, 2]]}"
    )
    .unwrap(),
    "{10, 184414, 221/2}"
  );
}

// BarOrigin -> Left lays the boxes along the y axis, labeled on the left.
#[test]
fn bar_origin_left_is_horizontal() {
  assert_eq!(
    interpret(
      "g = BoxWhiskerChart[{{1, 2, 2, 3, 3, 3, 4, 4, 5}}, BarOrigin -> Left, ChartLabels -> {\"a\"}]; \
       {MinMax[Cases[g, Polygon[p_] :> p[[All, 1]], Infinity]], \
        (FrameTicks /. Options[g])[[1, 1, 1, 2]], (FrameTicks /. Options[g])[[2]]}"
    )
    .unwrap(),
    "{{2, 4}, Spacer[0], {Automatic, Automatic}}"
  );
}

// BarOrigin -> Top hangs the boxes from the top: values are negated and the
// frame is labeled with their magnitudes.
#[test]
fn bar_origin_top_mirrors_values() {
  assert_eq!(
    interpret(
      "g = BoxWhiskerChart[{1, 2, 3, 4, 5}, BarOrigin -> Top]; \
       {MinMax[Cases[g, Polygon[p_] :> p[[All, 2]], Infinity]], \
        Cases[(FrameTicks /. Options[g])[[1, 1]], {-2., l_, _} :> l]}"
    )
    .unwrap(),
    "{{-17/4, -7/4}, {}}"
  );
}

// An association charts its values, one box per key.
#[test]
fn association_data() {
  assert_eq!(
    interpret(
      "g = BoxWhiskerChart[<|\"a\" -> {1, 2, 3, 4}, \"b\" -> {2, 3, 5, 9}|>]; \
       {Head[g], Length[Cases[g, _Polygon, Infinity]]}"
    )
    .unwrap(),
    "{Graphics, 2}"
  );
}

#[test]
fn unknown_bwspec_is_reported_and_ignored() {
  let r = interpret_with_stdout("Head[BoxWhiskerChart[{1, 2, 3}, \"Bogus\"]]")
    .unwrap();
  assert_eq!(r.result, "Graphics");
  assert!(
    r.warnings.iter().any(|w| w.contains(
      "BoxWhiskerChart::bwspec: Bogus cannot be used as a box-and-whisker symbol specification."
    )),
    "{:?}",
    r.warnings
  );
}

// The frame is drawn on all four sides of the exported SVG.
#[test]
fn exported_chart_has_a_frame() {
  let svg = interpret(
    "ExportString[BoxWhiskerChart[{{1, 2, 3, 4, 5}, {2, 3, 4, 5, 9}}], \"SVG\"]",
  )
  .unwrap();
  assert!(svg.contains("<svg"), "{svg}");
  // White median lines and the gray whiskers are both present.
  assert!(svg.contains("rgb(255,255,255)"), "{svg}");
  assert!(svg.contains("rgb(128,128,128)"), "{svg}");
}
