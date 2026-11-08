use super::*;

// One violin (a filled polygon) per dataset, in a symbolic Graphics.
#[test]
fn one_violin_per_dataset() {
  assert_eq!(
    interpret(
      "g = DistributionChart[{{1, 2, 2, 3, 3, 3, 4}, {2, 5, 6}}]; {Head[g], Length[Cases[g, _Polygon, Infinity]]}"
    )
    .unwrap(),
    "{Graphics, 2}"
  );
}

#[test]
fn chart_legends_and_smooth_density() {
  assert_eq!(
    interpret(
      "Head[DistributionChart[{{1, 2, 2, 3, 3, 3, 4}, {2, 5, 6}, {1, 1, 7}}, ChartElementFunction -> \"SmoothDensity\", ChartStyle -> \"Rainbow\", ChartLegends -> {\"a\", \"b\", \"c\"}]]"
    )
    .unwrap(),
    "Legended"
  );
}

// ChartElementFunction -> "Quantile": a min-to-max rectangle crossed by
// lines at the quartiles and median, in the fill darkened by 30%.
#[test]
fn quantile_element() {
  assert_eq!(
    interpret(
      "g = DistributionChart[{{1, 2, 2, 3, 3, 3, 4, 4, 5, 9}, {2, 3, 4, 4, 5, 5, 6, 7, 8}}, ChartElementFunction -> \"Quantile\"]; \
       {Cases[g, Rectangle[{_, a_}, {_, b_}, ___] :> {a, b}, Infinity], \
        Cases[g, {RGBColor[r_, _, _], Line[{{_, y_}, _}]} :> {r, y}, Infinity][[;; 3]]}"
    )
    .unwrap(),
    "{{{1., 9.}, {2., 8.}}, {{0.7, 2.}, {0.7, 3.}, {0.7, 4.}}}"
  );
}

// ChartElementFunction -> "Density": a bar over the data range whose
// color fades to white where the density estimate is lowest. Resolve WL's
// paired light/dark directives before inspecting the light palette.
#[test]
fn density_element() {
  assert_eq!(
    interpret(
      "g = DistributionChart[{{1, 2, 2, 3, 3, 3, 4, 4, 5, 9}}, ChartElementFunction -> \"Density\"]; \
       cs = Cases[g /. LightDarkSwitched[light_, ___] :> light, RGBColor[__], Infinity]; \
       {Length[Cases[g, _Rectangle, Infinity]] == 1, \
        Length[Cases[g, _GraphicsComplex, Infinity]] == 1, \
        Min[cs[[All, 3]]] == 0.25, Max[cs[[All, 3]]] == 1., \
        MinMax[Cases[g, Rectangle[{_, a_}, {_, b_}] :> {a, b}, Infinity]]}"
    )
    .unwrap(),
    "{True, True, True, True, {1., 9.}}"
  );
}

// `EdgeForm[Directive[Opacity[a], …]]` makes the edge translucent.
#[test]
fn edge_form_opacity_is_stroke_opacity() {
  let svg = interpret(
    "ExportString[Graphics[{EdgeForm[Directive[Opacity[0.5], AbsoluteThickness[1]]], Red, Rectangle[]}], \"SVG\"]",
  )
  .unwrap();
  assert!(svg.contains("stroke-opacity=\"0.5\""), "{svg}");
}

#[test]
fn non_list_data_is_ldata() {
  let r = interpret_with_stdout("DistributionChart[x]").unwrap();
  assert_eq!(r.result, "DistributionChart[x]");
  assert!(
    r.warnings
      .iter()
      .any(|w| w.contains("DistributionChart::ldata")),
    "{:?}",
    r.warnings
  );
}
