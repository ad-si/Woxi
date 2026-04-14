use super::*;

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
