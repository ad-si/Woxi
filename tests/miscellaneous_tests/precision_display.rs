//! Tests for the notebook display-layer transform that truncates
//! arbitrary-precision reals to their precision in significant figures.
//!
//! The CLI / `eval` output keeps the full backtick InputForm (matching
//! `wolframscript -code`), but the Playground and Woxi Studio show the
//! notebook OutputForm where `N[Pi, 3]` reads as `3.14`.

use woxi::{
  clear_state, get_captured_output_svg, interpret, truncate_precision_reals,
};

mod tests {
  use super::*;

  /// The Playground/Studio result SVG for `code`, with all SVG tags removed so
  /// only the visible text remains (the renderer wraps runs in `<text>`/`<tspan>`).
  fn output_text(code: &str) -> String {
    clear_state();
    interpret(code).unwrap();
    let svg = get_captured_output_svg().expect("output SVG captured");
    // Strip tags, leaving just the rendered glyph runs.
    let mut text = String::new();
    let mut in_tag = false;
    for ch in svg.chars() {
      match ch {
        '<' => in_tag = true,
        '>' => in_tag = false,
        _ if !in_tag => text.push(ch),
        _ => {}
      }
    }
    text
  }

  #[test]
  fn playground_machine_reals_show_six_significant_figures() {
    // The notebook front end (emulated by the Playground/Studio SVG) shows
    // machine-precision reals at 6 significant figures, unlike the full-precision
    // CLI/`eval` output. Trailing zeros are dropped.
    // (List items are laid out by coordinate, so the extracted text has no
    // literal space after each comma.)
    assert_eq!(
      output_text("{N[Pi], N[E], N[Log[0.5]]}"),
      "{3.14159,2.71828,-0.693147}"
    );
    // A short machine real keeps its exact digits (no zero padding to 6 figs).
    assert_eq!(output_text("0.1 + 0.2"), "0.3");
    assert_eq!(output_text("N[1/3]"), "0.333333");
    assert_eq!(output_text("N[2/3]"), "0.666667");
  }

  #[test]
  fn tableform_machine_reals_show_six_significant_figures() {
    // Regression: `TableForm[RandomReal[...]]` (and any grid) previously dumped
    // machine reals at full round-trip precision in each cell. The notebook
    // front end shows them at 6 significant figures, matching wolframscript's
    // rendered TableForm (`4.086947450855356` → `4.08695`).
    let out = output_text(
      "TableForm[{{4.086947450855356, 0.5570980556561822}, \
       {1.2068048372882523, 0.32869379754390526}}]",
    );
    assert!(out.contains("4.08695"), "got: {out}");
    assert!(out.contains("0.557098"), "got: {out}");
    assert!(out.contains("1.2068"), "got: {out}");
    assert!(out.contains("0.328694"), "got: {out}");
    // The full-precision digits must not leak through.
    assert!(!out.contains("4.086947450855356"), "got: {out}");
    assert!(!out.contains("0.5570980556561822"), "got: {out}");
  }

  #[test]
  fn grid_machine_reals_switch_to_scientific_notation() {
    // Large (>= 1e6) and small (< 1e-5) magnitudes render in scientific
    // notation with a superscript exponent, exactly as wolframscript does:
    // `1234567.89` → `1.23457×10^6`, `0.000001234` → `1.234×10^-6`.
    let out = output_text("Grid[{{1234567.89, 0.000001234, 1000000.0}}]");
    // Tag stripping concatenates the mantissa, the "×10" run, and the
    // superscript exponent glyphs.
    assert!(out.contains("1.23457\u{00d7}106"), "got: {out}");
    assert!(out.contains("1.234\u{00d7}10-6"), "got: {out}");
    assert!(out.contains("1.\u{00d7}106"), "got: {out}");
  }

  #[test]
  fn exportstring_svg_shows_machine_reals_at_six_significant_figures() {
    // Regression: `Export[…, "SVG"]` of a bare real went through the box-form
    // text renderer, which kept the `MakeBoxes` backtick marker and the full
    // round-trip precision (`4.086947450855356\``). The typeset SVG must show
    // the notebook OutputForm — 6 significant figures, no backtick.
    clear_state();
    let svg = interpret("ExportString[4.086947450855356, \"SVG\"]").unwrap();
    assert!(svg.contains("4.08695"), "expected 6-sig-fig text in SVG");
    assert!(
      !svg.contains("4.086947450855356"),
      "full precision must not leak"
    );
    assert!(!svg.contains('`'), "backtick marker must be stripped");

    // A list of reals truncates every element too.
    clear_state();
    let svg = interpret(
      "ExportString[{4.086947450855356, 0.5570980556561822}, \"SVG\"]",
    )
    .unwrap();
    assert!(svg.contains("4.08695") && svg.contains("0.557098"));
    assert!(!svg.contains("0.5570980556561822"));

    // An arbitrary-precision real still shows all its requested figures.
    clear_state();
    let svg = interpret("ExportString[N[Pi, 20], \"SVG\"]").unwrap();
    assert!(svg.contains("3.1415926535897932385"));
  }

  #[test]
  fn truncates_n_pi_by_precision() {
    // Each precision keeps that many significant figures; precision 1 still
    // shows a trailing decimal point so it reads as an approximate real.
    assert_eq!(truncate_precision_reals("3.1415926535897932385`1."), "3.");
    assert_eq!(truncate_precision_reals("3.1415926535897932385`2."), "3.1");
    assert_eq!(truncate_precision_reals("3.1415926535897932385`3."), "3.14");
    assert_eq!(
      truncate_precision_reals("3.1415926535897932385`4."),
      "3.142"
    );
    assert_eq!(
      truncate_precision_reals("3.1415926535897932385`5."),
      "3.1416"
    );
    assert_eq!(
      truncate_precision_reals("3.1415926535897932385`6."),
      "3.14159"
    );
  }

  #[test]
  fn truncates_inside_a_list() {
    let input = "{3.1415926535897932385`1., 3.1415926535897932385`3., \
                 3.1415926535897932385`5.}";
    assert_eq!(truncate_precision_reals(input), "{3., 3.14, 3.1416}");
  }

  #[test]
  fn preserves_leading_zero_and_sub_one_values() {
    // Sin[1] to precision 3: leading fractional zeros are placeholders.
    assert_eq!(truncate_precision_reals("0.8414709848078965`3."), "0.841");
    assert_eq!(truncate_precision_reals("0.00123456`2."), "0.0012");
  }

  #[test]
  fn preserves_integer_magnitude_with_placeholder_zeros() {
    // Rounding within the integer part keeps magnitude via trailing zeros.
    assert_eq!(truncate_precision_reals("314.159`2."), "310.");
    assert_eq!(truncate_precision_reals("314.159`1."), "300.");
    // 4th sig fig is `1`, next digit `5` → rounds up to 314.2.
    assert_eq!(truncate_precision_reals("314.159`4."), "314.2");
  }

  #[test]
  fn rounding_carry_propagates_and_grows_magnitude() {
    // 9.99 to 2 sig figs rounds up through the nines to 10.
    assert_eq!(truncate_precision_reals("9.99`2."), "10.");
    // 0.996 to 2 sig figs → 1.0.
    assert_eq!(truncate_precision_reals("0.996`2."), "1.0");
    // 99.6 to 2 sig figs → 100.
    assert_eq!(truncate_precision_reals("99.6`2."), "100.");
  }

  #[test]
  fn negative_reals_keep_their_sign() {
    assert_eq!(
      truncate_precision_reals("-3.1415926535897932385`3."),
      "-3.14"
    );
  }

  #[test]
  fn scientific_suffix_renders_as_times_ten_power() {
    // The InputForm `*^` operator becomes the readable `×10^` form for plain-text
    // (Studio) display, and the mantissa is still truncated to its precision.
    assert_eq!(
      truncate_precision_reals("1.23456`3.*^6"),
      "1.23\u{00d7}10^6"
    );
    assert_eq!(
      truncate_precision_reals("1.23456`3.*^-6"),
      "1.23\u{00d7}10^-6"
    );
    // A machine-precision scientific real (no backtick) is converted too.
    assert_eq!(truncate_precision_reals("1.*^10"), "1.\u{00d7}10^10");
  }

  #[test]
  fn machine_reals_and_symbols_are_left_alone() {
    // No backtick → machine-real / plain text passes through unchanged.
    assert_eq!(
      truncate_precision_reals("{2.5, 3., -3.141592653589793}"),
      "{2.5, 3., -3.141592653589793}"
    );
    // A digit inside a symbol name is not a number token.
    assert_eq!(truncate_precision_reals("x2 + a1"), "x2 + a1");
    // Context backticks on symbols are untouched (start with a letter).
    assert_eq!(truncate_precision_reals("Global`x"), "Global`x");
  }

  #[test]
  fn precision_beyond_available_digits_shows_all_digits() {
    // When the requested precision exceeds the stored significant digits,
    // every available digit is shown (no padding beyond the mantissa).
    assert_eq!(truncate_precision_reals("3.14`8."), "3.14");
  }
}
