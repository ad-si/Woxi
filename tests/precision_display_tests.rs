//! Tests for the notebook display-layer transform that truncates
//! arbitrary-precision reals to their precision in significant figures.
//!
//! The CLI / `eval` output keeps the full backtick InputForm (matching
//! `wolframscript -code`), but the Playground and Woxi Studio show the
//! notebook OutputForm where `N[Pi, 3]` reads as `3.14`.

use woxi::truncate_precision_reals;

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
