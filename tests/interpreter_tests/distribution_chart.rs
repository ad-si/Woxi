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
