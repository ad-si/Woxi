//! `ScalingFunctions` and the log plots, checked against wolframscript
//! (WL 15): scaled axes are drawn in scaled coordinates and labelled with
//! the ticks of `Charting`ScaledTicks`.
use super::*;

fn svg(expr: &str) -> String {
  interpret(&format!("ExportString[{expr}, \"SVG\"]")).unwrap()
}

/// The `(y, text)` of every right-aligned (y axis) tick label.
fn y_labels(svg: &str) -> Vec<(f64, String)> {
  svg
    .lines()
    .filter(|l| l.contains("<text") && l.contains("text-anchor=\"end\""))
    .filter_map(|l| {
      let y = l.split(" y=\"").nth(1)?.split('"').next()?.parse().ok()?;
      let text = l.split('>').nth(1)?.split('<').next()?.to_string();
      Some((y, text))
    })
    .collect()
}

/// The `(x, text)` of every centred (x axis) tick label.
fn x_labels(svg: &str) -> Vec<(f64, String)> {
  svg
    .lines()
    .filter(|l| l.contains("<text") && l.contains("text-anchor=\"middle\""))
    .filter_map(|l| {
      let x = l.split(" x=\"").nth(1)?.split('"').next()?.parse().ok()?;
      let text = l.split('>').nth(1)?.split('<').next()?.to_string();
      Some((x, text))
    })
    .collect()
}

fn texts(labels: &[(f64, String)]) -> Vec<&str> {
  labels.iter().map(|(_, t)| t.as_str()).collect()
}

#[test]
fn reverse_puts_large_values_at_the_bottom() {
  let s = svg("Plot[x^2, {x, 0, 10}, ScalingFunctions -> \"Reverse\"]");
  let y = y_labels(&s);
  assert_eq!(texts(&y), ["20", "40", "60", "80", "100"]);
  // Screen y grows downwards: 100 sits below 20.
  assert!(y[4].0 > y[0].0);
}

#[test]
fn reverse_x_runs_right_to_left() {
  let s = svg(
    "ListPlot[{1, 5, 10, 35, 60, 75, 140}, \
     ScalingFunctions -> {\"Reverse\", None}]",
  );
  let x = x_labels(&s);
  let six = x.iter().find(|(_, t)| t == "6").unwrap().0;
  let two = x.iter().find(|(_, t)| t == "2").unwrap().0;
  assert!(six < two);
}

#[test]
fn custom_scaling_pair_labels_original_values() {
  let s = svg(
    "Plot[x^2, {x, 0, 10}, ScalingFunctions -> {None, {-Log[#] &, Exp[-#] &}}]",
  );
  let y = y_labels(&s);
  let t = texts(&y);
  for label in ["1", "10", "100"] {
    assert!(t.contains(&label), "{t:?}");
  }
  // -Log reverses the axis: 100 below 1.
  let pos = |l: &str| y.iter().find(|(_, t)| t == l).unwrap().0;
  assert!(pos("100") > pos("1"));
}

#[test]
fn reciprocal_labels_padded_decimals() {
  let s = svg("Plot[x^2, {x, 1, 10}, ScalingFunctions -> \"Reciprocal\"]");
  let t: Vec<String> = texts(&y_labels(&s))
    .iter()
    .map(ToString::to_string)
    .collect();
  for label in ["3.0", "4.0", "5.0", "7.5", "10.0"] {
    assert!(t.iter().any(|x| x == label), "{t:?}");
  }
}

#[test]
fn log_axis_is_crossed_at_its_low_end() {
  // WL: the x axis runs along the padded bottom, below the curve's minimum
  // (about 0.25), not through 1.
  let s = svg("Plot[Abs[Gamma[z]], {z, -4, 8}, ScalingFunctions -> \"Log\"]");
  let y = y_labels(&s);
  let one = y.iter().find(|(_, t)| t == "1").expect("1 is labelled").0;
  // The x labels hang below the axis.
  let axis_y = s
    .lines()
    .filter(|l| l.contains("<text") && l.contains("text-anchor=\"middle\""))
    .filter_map(|l| {
      l.split(" y=\"")
        .nth(1)?
        .split('"')
        .next()?
        .parse::<f64>()
        .ok()
    })
    .fold(f64::MIN, f64::max);
  assert!(axis_y > one);
  assert!(texts(&y).contains(&"1000"));
}

#[test]
fn log_plot_is_plot_with_log_scaling() {
  let a = svg("LogPlot[x^2, {x, 1, 10}]");
  let b = svg("Plot[x^2, {x, 1, 10}, ScalingFunctions -> \"Log\"]");
  assert_eq!(a, b);
}

#[test]
fn list_log_plot_pads_below_its_lowest_point() {
  // The y = 1 point sits above the axis, and 1 keeps its label.
  let s =
    svg("ListPlot[{1, 5, 10, 35, 60, 75, 140}, ScalingFunctions -> \"Log\"]");
  assert_eq!(texts(&y_labels(&s)), ["1", "5", "10", "50", "100"]);
  assert_eq!(
    s,
    svg("ListLogPlot[{1, 5, 10, 35, 60, 75, 140}]"),
    "ListLogPlot is ListPlot with ScalingFunctions -> \"Log\""
  );
}

#[test]
fn list_log_plot_labels_every_other_decade() {
  let s =
    svg("ListLogPlot[Table[Binomial[25, k], {k, 0, 25}], Joined -> True]");
  let y = y_labels(&s);
  assert_eq!(y.len(), 4, "{y:?}");
  assert_eq!(y[0].1, "10");
  assert_eq!(y[1].1, "1000");
  // 10^5 and 10^7 as superscripts.
  assert!(s.contains(">5</tspan>") && s.contains(">7</tspan>"));
}

#[test]
fn log_axis_draws_explicit_grid_lines() {
  let s = svg(
    "Plot[x^2, {x, 1, 10}, ScalingFunctions -> \"Log\", \
     Ticks -> {Automatic, 2^Range[10]}, GridLines -> {None, 2^Range[10]}]",
  );
  // 2, 4, …, 128 lie in the range 1..100 (padded).
  let grid = s
    .lines()
    .filter(|l| {
      l.contains("opacity=\"0.4\"") && l.contains("stroke=\"#808080\"")
    })
    .count();
  assert_eq!(grid, 7, "{s}");
}

#[test]
fn find_divisions_takes_machine_reals() {
  assert_eq!(
    interpret("FindDivisions[{0.3, 17.2}, 6]").unwrap(),
    "{0, 5, 10, 15, 20}"
  );
}
