use super::*;

mod notational_markup {
  use super::*;

  // Sub*/Super*/Over*/Under* markup heads are purely notational in WL: they
  // have no evaluation rules and stay unevaluated as their canonical form.
  // They must not be flagged as "not yet implemented".

  fn assert_inert(input: &str, expected: &str) {
    clear_state();
    let result = interpret_with_stdout(input).unwrap();
    assert_eq!(result.result, expected, "result mismatch for {input}");
    assert!(
      result
        .warnings
        .iter()
        .all(|w| !w.contains("not yet implemented")),
      "unexpected unimplemented warning for {input}: {:?}",
      result.warnings
    );
  }

  #[test]
  fn sub_plus_stays_symbolic() {
    assert_inert("SubPlus[5]", "SubPlus[5]");
  }

  #[test]
  fn sub_minus_stays_symbolic() {
    assert_inert("SubMinus[3]", "SubMinus[3]");
  }

  #[test]
  fn sub_star_stays_symbolic() {
    assert_inert("SubStar[x]", "SubStar[x]");
  }

  #[test]
  fn super_plus_stays_symbolic() {
    assert_inert("SuperPlus[a]", "SuperPlus[a]");
  }

  #[test]
  fn super_minus_stays_symbolic() {
    assert_inert("SuperMinus[b]", "SuperMinus[b]");
  }

  #[test]
  fn super_star_stays_symbolic() {
    assert_inert("SuperStar[z]", "SuperStar[z]");
  }

  #[test]
  fn super_dagger_stays_symbolic() {
    assert_inert("SuperDagger[m]", "SuperDagger[m]");
  }

  #[test]
  fn over_hat_stays_symbolic() {
    assert_inert("OverHat[x]", "OverHat[x]");
  }

  #[test]
  fn under_bar_stays_symbolic() {
    assert_inert("UnderBar[y]", "UnderBar[y]");
  }

  #[test]
  fn head_is_the_markup_symbol() {
    assert_inert("Head[SubMinus[3]]", "SubMinus");
  }

  #[test]
  fn combines_arithmetically() {
    assert_inert("SubPlus[x] + SubPlus[x]", "2*SubPlus[x]");
  }

  #[test]
  fn multi_argument_form_stays_symbolic() {
    assert_inert("SubPlus[a, b]", "SubPlus[a, b]");
  }
}

mod image_size {
  use super::*;

  #[test]
  fn image_size_is_symbol() {
    // ImageSize evaluates to itself as a symbol
    assert_eq!(interpret("ImageSize").unwrap(), "ImageSize");
  }

  #[test]
  fn image_size_attributes() {
    assert_eq!(interpret("Attributes[ImageSize]").unwrap(), "{Protected}");
  }

  #[test]
  fn image_size_head() {
    assert_eq!(interpret("Head[ImageSize]").unwrap(), "Symbol");
  }

  #[test]
  fn image_size_in_rule() {
    // ImageSize used as option name in a Rule
    assert_eq!(interpret("ImageSize -> 300").unwrap(), "ImageSize -> 300");
  }

  #[test]
  fn image_size_in_list_of_rules() {
    assert_eq!(
      interpret("{ImageSize -> 400, PlotRange -> All}").unwrap(),
      "{ImageSize -> 400, PlotRange -> All}"
    );
  }
}

mod font_size {
  use super::*;

  #[test]
  fn font_size_is_symbol() {
    assert_eq!(interpret("FontSize").unwrap(), "FontSize");
  }

  #[test]
  fn font_size_attributes() {
    assert_eq!(interpret("Attributes[FontSize]").unwrap(), "{Protected}");
  }

  #[test]
  fn font_size_head() {
    assert_eq!(interpret("Head[FontSize]").unwrap(), "Symbol");
  }

  #[test]
  fn font_size_in_rule() {
    assert_eq!(interpret("FontSize -> 14").unwrap(), "FontSize -> 14");
  }

  #[test]
  fn font_size_in_style() {
    // Style[expr, ...] unwraps to its content in OutputForm
    // (matching wolframscript).
    assert_eq!(
      interpret("Style[\"hello\", FontSize -> 24]").unwrap(),
      "hello"
    );
  }

  #[test]
  fn style_list_output_unwraps_to_content() {
    // Regression: {Style["green", Italic, Green], Style["red", Bold, Red]}
    // should render as {green, red}, matching wolframscript. Previously
    // Woxi kept the full Style[...] wrappers in OutputForm.
    assert_eq!(
      interpret("{Style[\"green\", Italic, Green], Style[\"red\", Bold, Red]}")
        .unwrap(),
      "{green, red}"
    );
  }

  #[test]
  fn style_single_output_unwraps_to_content() {
    assert_eq!(interpret("Style[\"hello\", Bold, Red]").unwrap(), "hello");
  }

  #[test]
  fn style_input_form_preserves_wrapper() {
    // ToString[..., InputForm] should still show the full Style[...] head.
    assert_eq!(
      interpret("ToString[Style[\"green\", Italic, Green], InputForm]")
        .unwrap(),
      "Style[\"green\", Italic, RGBColor[0, 1, 0]]"
    );
  }

  #[test]
  fn tostring_style_resolves_nested_number_form() {
    // Regression: under ToString a Style wrapper is display-only, so a nested
    // display wrapper such as NumberForm must still resolve to its formatted
    // text. Previously `ToString[Style[NumberForm[...], 18]]` echoed the raw
    // `NumberForm[50., {3, 1}]` head instead of "50.0".
    assert_eq!(
      interpret("ToString[Style[NumberForm[50., {3, 1}], 18, Bold]]").unwrap(),
      "50.0"
    );
  }

  #[test]
  fn tostring_row_concatenates_and_resolves_wrappers() {
    // A Row concatenates its parts under ToString, resolving nested Style /
    // NumberForm, matching wolframscript.
    assert_eq!(
      interpret(
        "ToString[Row[{Style[NumberForm[50., {3, 1}], 18, Bold], Style[\"% shaded\", 18, Bold]}]]"
      )
      .unwrap(),
      "50.0% shaded"
    );
    // Row with an explicit separator joins the (resolved) parts with it.
    assert_eq!(
      interpret("ToString[Row[{NumberForm[50., {3, 1}], \" pct\"}]]").unwrap(),
      "50.0 pct"
    );
  }
}

mod font_family {
  use super::*;

  #[test]
  fn font_family_evaluates_to_itself() {
    assert_eq!(interpret("FontFamily").unwrap(), "FontFamily");
  }

  #[test]
  fn font_family_head() {
    assert_eq!(interpret("Head[FontFamily]").unwrap(), "Symbol");
  }

  #[test]
  fn font_family_attributes() {
    assert_eq!(interpret("Attributes[FontFamily]").unwrap(), "{Protected}");
  }

  #[test]
  fn font_family_in_rule() {
    assert_eq!(
      interpret("FontFamily -> \"Helvetica\"").unwrap(),
      "FontFamily -> Helvetica"
    );
  }

  #[test]
  fn font_family_in_style() {
    // Style[expr, ...] unwraps to its content in OutputForm
    // (matching wolframscript).
    assert_eq!(
      interpret("Style[\"hello\", FontFamily -> \"Arial\"]").unwrap(),
      "hello"
    );
  }
}

mod thick {
  use super::*;

  #[test]
  fn thick_evaluates_to_itself() {
    assert_eq!(interpret("Thick").unwrap(), "Thickness[Large]");
  }

  #[test]
  fn thick_head() {
    assert_eq!(interpret("Head[Thick]").unwrap(), "Thickness");
  }

  #[test]
  fn thick_attributes() {
    assert_eq!(
      interpret("Attributes[Thick]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }

  #[test]
  fn thick_in_graphics_directive_list() {
    // Thick should be usable in a Graphics directive list
    assert_eq!(
      interpret("Graphics[{Thick, Line[{{0, 0}, {1, 1}}]}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn thick_in_graphics_input_form() {
    // Thick should evaluate to Thickness[Large] inside Graphics
    assert_eq!(
      interpret(
        "ToString[Graphics[{Thick, Line[{{0, 0}, {1, 1}}]}], InputForm]"
      )
      .unwrap(),
      "Graphics[{Thickness[Large], Line[{{0, 0}, {1, 1}}]}]"
    );
  }

  #[test]
  fn thick_in_plot_style() {
    // Thick can be used as a PlotStyle option value
    assert_eq!(
      interpret("Plot[Sin[x], {x, 0, 1}, PlotStyle -> Thick]").unwrap(),
      "-Graphics-"
    );
  }
}

mod dashed {
  use super::*;

  #[test]
  fn dashed_evaluates() {
    assert_eq!(interpret("Dashed").unwrap(), "Dashing[{Small, Small}]");
  }

  #[test]
  fn dotted_evaluates() {
    assert_eq!(interpret("Dotted").unwrap(), "Dashing[{0, Small}]");
  }

  #[test]
  fn dot_dashed_evaluates() {
    assert_eq!(
      interpret("DotDashed").unwrap(),
      "Dashing[{0, Small, Small, Small}]"
    );
  }

  #[test]
  fn dashed_head() {
    assert_eq!(interpret("Head[Dashed]").unwrap(), "Dashing");
  }

  #[test]
  fn dashed_in_graphics() {
    assert_eq!(
      interpret("Graphics[{Dashed, Line[{{0, 0}, {1, 1}}]}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn dashed_in_graphics_input_form() {
    assert_eq!(
      interpret(
        "ToString[Graphics[{Dashed, Line[{{0, 0}, {1, 1}}]}], InputForm]"
      )
      .unwrap(),
      "Graphics[{Dashing[{Small, Small}], Line[{{0, 0}, {1, 1}}]}]"
    );
  }

  #[test]
  fn dotted_in_graphics() {
    assert_eq!(
      interpret("Graphics[{Dotted, Line[{{0, 0}, {1, 1}}]}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn dotted_in_graphics_input_form() {
    assert_eq!(
      interpret(
        "ToString[Graphics[{Dotted, Line[{{0, 0}, {1, 1}}]}], InputForm]"
      )
      .unwrap(),
      "Graphics[{Dashing[{0, Small}], Line[{{0, 0}, {1, 1}}]}]"
    );
  }

  #[test]
  fn dot_dashed_in_graphics() {
    assert_eq!(
      interpret("Graphics[{DotDashed, Line[{{0, 0}, {1, 1}}]}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn dot_dashed_in_graphics_input_form() {
    assert_eq!(
      interpret(
        "ToString[Graphics[{DotDashed, Line[{{0, 0}, {1, 1}}]}], InputForm]"
      )
      .unwrap(),
      "Graphics[{Dashing[{0, Small, Small, Small}], Line[{{0, 0}, {1, 1}}]}]"
    );
  }

  #[test]
  fn dashing_with_named_sizes() {
    assert_eq!(
      interpret("Dashing[{Small, Small}]").unwrap(),
      "Dashing[{Small, Small}]"
    );
  }
}

mod base_style {
  use super::*;

  #[test]
  fn base_style_evaluates_to_itself() {
    assert_eq!(interpret("BaseStyle").unwrap(), "BaseStyle");
  }

  #[test]
  fn base_style_head() {
    assert_eq!(interpret("Head[BaseStyle]").unwrap(), "Symbol");
  }

  #[test]
  fn base_style_attributes() {
    assert_eq!(interpret("Attributes[BaseStyle]").unwrap(), "{Protected}");
  }

  #[test]
  fn base_style_in_rule() {
    assert_eq!(
      interpret("BaseStyle -> {FontSize -> 14}").unwrap(),
      "BaseStyle -> {FontSize -> 14}"
    );
  }

  #[test]
  fn base_style_in_graphics() {
    assert_eq!(
      interpret("Graphics[{Disk[]}, BaseStyle -> {Red}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn base_style_in_graphics_input_form() {
    // Disk[] → Disk[{0, 0}] and Red → RGBColor[1, 0, 0] inside Graphics
    assert_eq!(
      interpret("ToString[Graphics[{Disk[]}, BaseStyle -> {Red}], InputForm]")
        .unwrap(),
      "Graphics[{Disk[{0, 0}]}, BaseStyle -> {RGBColor[1, 0, 0]}]"
    );
  }
}

mod absolute_dashing {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("AbsoluteDashing[{1, 2}]").unwrap(),
      "AbsoluteDashing[{1, 2}]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[AbsoluteDashing]").unwrap(), "Symbol");
  }

  #[test]
  fn three_element_list() {
    assert_eq!(
      interpret("AbsoluteDashing[{1, 2, 3}]").unwrap(),
      "AbsoluteDashing[{1, 2, 3}]"
    );
  }
}

mod proportional {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("Proportional[a, b]").unwrap(), "a \u{221D} b");
  }

  #[test]
  fn multiple_args() {
    assert_eq!(
      interpret("Proportional[1, 2, 3]").unwrap(),
      "1 \u{221D} 2 \u{221D} 3"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[Proportional[a, b]]").unwrap(),
      "Proportional"
    );
  }
}

mod raster_size {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("RasterSize[1, 2]").unwrap(), "RasterSize[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[RasterSize]").unwrap(), "Symbol");
  }
}

mod padding {
  use super::*;

  #[test]
  fn symbol() {
    assert_eq!(interpret("Padding").unwrap(), "Padding");
  }

  #[test]
  fn with_args() {
    assert_eq!(interpret("Padding[1, 2, 3]").unwrap(), "Padding[1, 2, 3]");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Padding]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod data_reversed {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("DataReversed[1, 2]").unwrap(),
      "DataReversed[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[DataReversed]").unwrap(), "Symbol");
  }
}

mod axes_edge {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("AxesEdge[1, 2]").unwrap(), "AxesEdge[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[AxesEdge]").unwrap(), "Symbol");
  }
}

mod cmyk_color {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("CMYKColor[0.5, 0.2, 0.8, 0.1]").unwrap(),
      "CMYKColor[0.5, 0.2, 0.8, 0.1]"
    );
  }

  #[test]
  fn integer_args() {
    assert_eq!(
      interpret("CMYKColor[1, 0, 0, 0]").unwrap(),
      "CMYKColor[1, 0, 0, 0]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[CMYKColor[0.5, 0.2, 0.8, 0.1]]").unwrap(),
      "CMYKColor"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[CMYKColor]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod rgb_color_hex {
  use super::*;

  // A 6-digit hex string parses into the three channel values.
  #[test]
  fn six_digit() {
    assert_eq!(
      interpret("RGBColor[\"#FF0000\"]").unwrap(),
      "RGBColor[1., 0., 0.]"
    );
    assert_eq!(
      interpret("RGBColor[\"#336699\"]").unwrap(),
      "RGBColor[0.2, 0.4, 0.6]"
    );
  }

  // The 3-digit shorthand doubles each digit (#F00 -> #FF0000).
  #[test]
  fn three_digit_shorthand() {
    assert_eq!(
      interpret("RGBColor[\"#F00\"]").unwrap(),
      "RGBColor[1., 0., 0.]"
    );
  }

  // Lowercase hex digits are accepted.
  #[test]
  fn lowercase() {
    assert_eq!(
      interpret("RGBColor[\"#00ff00\"]").unwrap(),
      "RGBColor[0., 1., 0.]"
    );
  }

  // An 8-digit string adds an alpha channel.
  #[test]
  fn eight_digit_alpha() {
    assert_eq!(
      interpret("RGBColor[\"#FF000080\"]").unwrap(),
      "RGBColor[1., 0., 0., 0.5019607843137255]"
    );
  }

  // Unsupported forms (no `#`, 4 digits, non-hex, wrong length) stay symbolic.
  #[test]
  fn invalid_stays_symbolic() {
    assert_eq!(
      interpret("RGBColor[\"FF0000\"]").unwrap(),
      "RGBColor[FF0000]"
    );
    assert_eq!(interpret("RGBColor[\"#F008\"]").unwrap(), "RGBColor[#F008]");
    assert_eq!(interpret("RGBColor[\"#XYZ\"]").unwrap(), "RGBColor[#XYZ]");
    assert_eq!(interpret("RGBColor[\"#FF00\"]").unwrap(), "RGBColor[#FF00]");
  }
}

mod accounting_form {
  use super::*;

  #[test]
  fn unevaluated_real() {
    assert_eq!(
      interpret("AccountingForm[123.45]").unwrap(),
      "AccountingForm[123.45]"
    );
  }

  #[test]
  fn unevaluated_integer() {
    assert_eq!(
      interpret("AccountingForm[42]").unwrap(),
      "AccountingForm[42]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[AccountingForm[123.45]]").unwrap(),
      "AccountingForm"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[AccountingForm]").unwrap(),
      "{NHoldRest, Protected}"
    );
  }

  // ToString renders AccountingForm like NumberForm, with negatives in parens.
  #[test]
  fn to_string_positive() {
    assert_eq!(
      interpret("ToString[AccountingForm[1234.5]]").unwrap(),
      "1234.5"
    );
    assert_eq!(
      interpret("ToString[AccountingForm[1234567]]").unwrap(),
      "1234567"
    );
  }
  #[test]
  fn to_string_negative_in_parens() {
    assert_eq!(
      interpret("ToString[AccountingForm[-1234.5]]").unwrap(),
      "(1234.5)"
    );
    assert_eq!(interpret("ToString[AccountingForm[-5]]").unwrap(), "(5)");
  }
  #[test]
  fn to_string_with_precision() {
    assert_eq!(
      interpret("ToString[AccountingForm[1234.5678, 3]]").unwrap(),
      "1230."
    );
    assert_eq!(
      interpret("ToString[AccountingForm[-3.14159, 3]]").unwrap(),
      "(3.14)"
    );
  }
  // A DigitBlock option groups the digits (always in full decimal); negatives
  // keep their parentheses.
  #[test]
  fn to_string_digit_block() {
    assert_eq!(
      interpret("ToString[AccountingForm[-1234.5, DigitBlock -> 3]]").unwrap(),
      "(1,234.5)"
    );
    assert_eq!(
      interpret("ToString[AccountingForm[1234567.89, DigitBlock -> 3]]")
        .unwrap(),
      "1,234,568."
    );
    assert_eq!(
      interpret("ToString[AccountingForm[-1234567, DigitBlock -> 3]]").unwrap(),
      "(1,234,567)"
    );
    // A custom NumberSeparator overrides the comma.
    assert_eq!(
      interpret(
        "ToString[AccountingForm[1234567.89, DigitBlock -> 3, NumberSeparator -> \".\"]]"
      )
      .unwrap(),
      "1.234.568."
    );
  }
}

mod decimal_form {
  use super::*;

  #[test]
  fn unevaluated_real() {
    assert_eq!(
      interpret("DecimalForm[1234567.89]").unwrap(),
      "DecimalForm[1.23456789*^6]"
    );
  }

  #[test]
  fn unevaluated_with_digits() {
    assert_eq!(
      interpret("DecimalForm[1234567.89, 9]").unwrap(),
      "DecimalForm[1.23456789*^6, 9]"
    );
  }

  #[test]
  fn unevaluated_small_real() {
    assert_eq!(
      interpret("DecimalForm[0.000004]").unwrap(),
      "DecimalForm[4.*^-6]"
    );
  }

  #[test]
  fn unevaluated_integer() {
    assert_eq!(interpret("DecimalForm[42]").unwrap(), "DecimalForm[42]");
  }

  #[test]
  fn unevaluated_rational() {
    assert_eq!(interpret("DecimalForm[1/3]").unwrap(), "DecimalForm[1/3]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[DecimalForm[3.14]]").unwrap(), "DecimalForm");
  }

  // ToString renders DecimalForm in decimal (non-scientific) notation.
  #[test]
  fn to_string_basic() {
    assert_eq!(
      interpret("ToString[DecimalForm[3.14159]]").unwrap(),
      "3.14159"
    );
    assert_eq!(interpret("ToString[DecimalForm[-5.5]]").unwrap(), "-5.5");
    assert_eq!(interpret("ToString[DecimalForm[42]]").unwrap(), "42");
  }
  #[test]
  fn to_string_forces_decimal_for_large() {
    // The integer part is kept exact (not switched to scientific notation).
    assert_eq!(
      interpret("ToString[DecimalForm[1234567.89]]").unwrap(),
      "1234568"
    );
    assert_eq!(
      interpret("ToString[DecimalForm[123456789012.0]]").unwrap(),
      "123456789012"
    );
  }
  #[test]
  fn to_string_small() {
    assert_eq!(
      interpret("ToString[DecimalForm[0.00012345678]]").unwrap(),
      "0.000123457"
    );
  }
  #[test]
  fn to_string_with_precision() {
    // Two-argument form rounds to n significant figures.
    assert_eq!(
      interpret("ToString[DecimalForm[1234.5678, 6]]").unwrap(),
      "1234.57"
    );
    assert_eq!(
      interpret("ToString[DecimalForm[1234567.89, 3]]").unwrap(),
      "1230000"
    );
  }
}

mod thumbnail {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("Thumbnail[x]").unwrap(), "Thumbnail[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Thumbnail]").unwrap(), "Symbol");
  }
}

mod legend_function {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("LegendFunction[1, 2]").unwrap(),
      "LegendFunction[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[LegendFunction]").unwrap(), "Symbol");
  }
}

mod point_legend {
  use super::*;

  #[test]
  fn symbol() {
    assert_eq!(interpret("PointLegend").unwrap(), "PointLegend");
  }

  #[test]
  fn with_args() {
    assert_eq!(interpret("PointLegend[x, y]").unwrap(), "PointLegend[x, y]");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[PointLegend]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod legend_label {
  use super::*;

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(
      interpret("LegendLabel[\"test\"]").unwrap(),
      "LegendLabel[test]"
    );
  }

  #[test]
  fn with_none() {
    assert_eq!(interpret("LegendLabel[None]").unwrap(), "LegendLabel[None]");
  }

  #[test]
  fn bare_symbol() {
    assert_eq!(interpret("LegendLabel").unwrap(), "LegendLabel");
  }
}

mod long_right_arrow {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("LongRightArrow[x, y]").unwrap(), "x \u{27F6} y");
  }

  #[test]
  fn three_args() {
    assert_eq!(
      interpret("LongRightArrow[1, 2, 3]").unwrap(),
      "1 \u{27F6} 2 \u{27F6} 3"
    );
  }

  #[test]
  fn single_arg_unevaluated() {
    assert_eq!(interpret("LongRightArrow[x]").unwrap(), "LongRightArrow[x]");
  }

  #[test]
  fn no_args_unevaluated() {
    assert_eq!(interpret("LongRightArrow[]").unwrap(), "LongRightArrow[]");
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[LongRightArrow[x, y]]").unwrap(),
      "LongRightArrow"
    );
  }
}

mod hyperlink {
  use super::*;

  #[test]
  fn label_and_uri() {
    // Hyperlink[label, uri] keeps both arguments and renders them
    // unquoted in OutputForm, matching wolframscript.
    assert_eq!(
      interpret(r#"Hyperlink["Woxi", "https://woxi.ad-si.com"]"#).unwrap(),
      "Hyperlink[Woxi, https://woxi.ad-si.com]"
    );
  }

  #[test]
  fn uri_only() {
    // Hyperlink[uri] is the single-argument form.
    assert_eq!(
      interpret(r#"Hyperlink["https://woxi.ad-si.com"]"#).unwrap(),
      "Hyperlink[https://woxi.ad-si.com]"
    );
  }

  #[test]
  fn no_args_unevaluated() {
    assert_eq!(interpret("Hyperlink[]").unwrap(), "Hyperlink[]");
  }

  #[test]
  fn extra_args_unevaluated() {
    assert_eq!(
      interpret(r#"Hyperlink["a", "b", "c"]"#).unwrap(),
      "Hyperlink[a, b, c]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret(r#"Head[Hyperlink["Woxi", "https://woxi.ad-si.com"]]"#)
        .unwrap(),
      "Hyperlink"
    );
  }

  #[test]
  fn length_two_args() {
    assert_eq!(
      interpret(r#"Length[Hyperlink["Woxi", "https://woxi.ad-si.com"]]"#)
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn length_one_arg() {
    assert_eq!(
      interpret(r#"Length[Hyperlink["https://woxi.ad-si.com"]]"#).unwrap(),
      "1"
    );
  }

  #[test]
  fn part_extracts_label() {
    assert_eq!(
      interpret(r#"Hyperlink["Woxi", "https://woxi.ad-si.com"][[1]]"#).unwrap(),
      "Woxi"
    );
  }

  #[test]
  fn part_extracts_uri() {
    assert_eq!(
      interpret(r#"Hyperlink["Woxi", "https://woxi.ad-si.com"][[2]]"#).unwrap(),
      "https://woxi.ad-si.com"
    );
  }

  #[test]
  fn input_form_quotes_strings() {
    // ToString[..., InputForm] preserves the string quotes around
    // the label and URI, matching wolframscript.
    assert_eq!(
      interpret(
        r#"ToString[Hyperlink["Woxi", "https://woxi.ad-si.com"], InputForm]"#
      )
      .unwrap(),
      r#"Hyperlink["Woxi", "https://woxi.ad-si.com"]"#
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn grid_1() {
    assert_case(r#"Grid[{{a, b}, {c, d}}]"#, r#"Grid[{{a, b}, {c, d}}]"#);
  }
  #[test]
  fn grid_2() {
    assert_case(
      r#"Grid[{{a, b}, {c, d}}]; Grid[{a, b, c}]"#,
      r#"Grid[{a, b, c}]"#,
    );
  }
  #[test]
  fn grid_3() {
    assert_case(
      r#"Grid[{{a, b}, {c, d}}]; Grid[{a, b, c}]; Grid[{{"first", "second", "third"},{a},{1, 2, 3}}]"#,
      r#"Grid[{{"first", "second", "third"}, {a}, {1, 2, 3}}]"#,
    );
  }
  #[test]
  fn grid_4() {
    assert_case(
      r#"Grid[{{a, b}, {c, d}}]; Grid[{a, b, c}]; Grid[{{"first", "second", "third"},{a},{1, 2, 3}}]; Grid[{"This is a long title", {"first", "second", "third"},{a},{1, 2, 3}}]"#,
      r#"Grid[{"This is a long title", {"first", "second", "third"}, {a}, {1, 2, 3}}]"#,
    );
  }
  #[test]
  fn pane() {
    assert_case(
      r#"Pane[37!]"#,
      r#"Pane[13763753091226345046315979581580902400000000]"#,
    );
  }
  #[test]
  fn list_literal() {
    // mathics rendered the contents to LaTeX `\begin{array}…`;
    // wolframscript -code returns the unevaluated wrapper
    // `TeXForm[TableForm[{{Pane[a, 3], Pane[expt, 3]}}]]` verbatim.
    // Woxi matches.
    assert_case(
      r#"Pane[37!]; {{Pane[a,3], Pane[expt, 3]}}//TableForm//TeXForm"#,
      r#"TeXForm[TableForm[{{Pane[a, 3], Pane[expt, 3]}}]]"#,
    );
  }
  #[test]
  fn grid_5() {
    // mathics quoted the StringJoin args; wolframscript -code emits the
    // contents in OutputForm (no quotes around the held strings) — Woxi
    // matches.
    assert_case(
      r#"Grid[{{a,bc},{d,e}}, ColumnAlignments:>Symbol["Rig"<>"ht"]]"#,
      r#"Grid[{{a, bc}, {d, e}}, ColumnAlignments :> Symbol[StringJoin[Rig, ht]]]"#,
    );
  }
  #[test]
  fn greater() {
    // mathics rendered the contents to LaTeX `\begin{array}…`;
    // wolframscript -code returns the unevaluated wrapper
    // `TeXForm[Grid[{{a, bc}, {d, e}}, ColumnAlignments -> Left]]`
    // verbatim. Woxi matches.
    assert_case(
      r#"Grid[{{a,bc},{d,e}}, ColumnAlignments:>Symbol["Rig"<>"ht"]]; TeXForm@Grid[{{a,bc},{d,e}}, ColumnAlignments->Left]"#,
      r#"TeXForm[Grid[{{a, bc}, {d, e}}, ColumnAlignments -> Left]]"#,
    );
  }
  #[test]
  fn te_x_form() {
    // mathics rendered the contents to LaTeX `\begin{array}…`;
    // wolframscript -code returns the unevaluated wrapper
    // `TeXForm[TableForm[{{a, b}, {c, d}}]]` verbatim. Woxi matches.
    assert_case(
      r#"Grid[{{a,bc},{d,e}}, ColumnAlignments:>Symbol["Rig"<>"ht"]]; TeXForm@Grid[{{a,bc},{d,e}}, ColumnAlignments->Left]; TeXForm[TableForm[{{a,b},{c,d}}]]"#,
      r#"TeXForm[TableForm[{{a, b}, {c, d}}]]"#,
    );
  }
}
