use super::*;

mod image_core {
  use super::*;

  #[test]
  fn image_constructor_grayscale() {
    clear_state();
    let result = interpret("Image[{{0, 0.5, 1}, {1, 0.5, 0}}]").unwrap();
    assert_eq!(result, "-Image-");
  }

  #[test]
  fn image_constructor_rgb() {
    clear_state();
    let result =
      interpret("Image[{{{1, 0, 0}, {0, 1, 0}}, {{0, 0, 1}, {1, 1, 0}}}]")
        .unwrap();
    assert_eq!(result, "-Image-");
  }

  #[test]
  fn image_q_true() {
    clear_state();
    let result = interpret("ImageQ[Image[{{0, 1}}]]").unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn image_q_false() {
    clear_state();
    let result = interpret("ImageQ[42]").unwrap();
    assert_eq!(result, "False");
  }

  #[test]
  fn image_q_false_list() {
    clear_state();
    let result = interpret("ImageQ[{1, 2, 3}]").unwrap();
    assert_eq!(result, "False");
  }

  #[test]
  fn image_dimensions_grayscale() {
    clear_state();
    let result = interpret(
      "img = Image[{{0, 0.5, 1}, {1, 0.5, 0}}]; ImageDimensions[img]",
    )
    .unwrap();
    assert_eq!(result, "{3, 2}");
  }

  #[test]
  fn image_dimensions_integer_bit_matrix_postfix() {
    clear_state();
    // Postfix form via //; 2 columns, 3 rows → {2, 3}.
    assert_eq!(
      interpret("Image[{{0, 1}, {1, 0}, {1, 1}}] // ImageDimensions").unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn image_dimensions_real_matrix_postfix() {
    clear_state();
    // 2-column, 3-row real matrix → ImageDimensions {2, 3}.
    assert_eq!(
      interpret(
        "Image[{{0.2, 0.4}, {0.9, 0.6}, {0.3, 0.8}}] // ImageDimensions"
      )
      .unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn image_aspect_ratio_from_small_matrix() {
    // 3 rows × 2 cols → {w, h} = {2, 3} → aspect h/w = 3/2.
    clear_state();
    assert_eq!(
      interpret("ImageAspectRatio[Image[{{0, 1}, {1, 0}, {1, 1}}]]").unwrap(),
      "3/2"
    );
  }

  #[test]
  fn image_aspect_ratio_square() {
    clear_state();
    assert_eq!(
      interpret("ImageAspectRatio[Image[{{0, 1}, {1, 0}}]]").unwrap(),
      "1"
    );
  }

  #[test]
  fn image_data_bit_form() {
    // ImageData[img, "Bit"] returns integer 0/1 values regardless of the
    // underlying storage.
    clear_state();
    assert_eq!(
      interpret("ImageData[Image[{{0, 1}, {1, 0}, {1, 1}}], \"Bit\"]").unwrap(),
      "{{0, 1}, {1, 0}, {1, 1}}"
    );
  }

  #[test]
  fn image_data_byte_form() {
    // ImageData[img, "Byte"] scales floats to 0..255 integers.
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[Image[{{0.2, 0.4}, {0.9, 0.6}, {0.5, 0.8}}], \"Byte\"]"
      )
      .unwrap(),
      "{{51, 102}, {230, 153}, {128, 204}}"
    );
  }

  #[test]
  fn image_q_on_invalid_ragged_nested_list() {
    // Invalid pixel data (ragged nested list) — Image[...] stays unevaluated
    // and ImageQ returns False (matches wolframscript).
    clear_state();
    assert_eq!(
      interpret("ImageQ[Image[{{{0, 0, 0}, {0, 1}}, {{0, 1, 0}, {0, 1, 1}}}]]")
        .unwrap(),
      "False"
    );
  }

  #[test]
  fn image_q_on_invalid_1d_list() {
    // 1D list is not a valid image — ImageQ returns False.
    clear_state();
    assert_eq!(interpret("ImageQ[Image[{1, 0, 1}]]").unwrap(), "False");
  }

  #[test]
  fn image_dimensions_rgb() {
    clear_state();
    let result = interpret(
      "img = Image[{{{1, 0, 0}, {0, 1, 0}}, {{0, 0, 1}, {1, 1, 0}}}]; ImageDimensions[img]",
    )
    .unwrap();
    assert_eq!(result, "{2, 2}");
  }

  #[test]
  fn image_channels_grayscale() {
    clear_state();
    let result = interpret("ImageChannels[Image[{{0, 1}}]]").unwrap();
    assert_eq!(result, "1");
  }

  #[test]
  fn image_channels_rgb() {
    clear_state();
    let result =
      interpret("ImageChannels[Image[{{{1, 0, 0}, {0, 1, 0}}}]]").unwrap();
    assert_eq!(result, "3");
  }

  #[test]
  fn image_type_default_real32() {
    clear_state();
    // Default image type is Real32 (matching wolframscript)
    let result = interpret("ImageType[Image[{{0, 0.5, 1}}]]").unwrap();
    assert_eq!(result, "Real32");
  }

  #[test]
  fn image_constructor_with_byte_type() {
    clear_state();
    let result =
      interpret("ImageType[Image[{{0, 0.5, 1}}, \"Byte\"]]").unwrap();
    assert_eq!(result, "Byte");
  }

  #[test]
  fn image_constructor_with_real32_type() {
    clear_state();
    let result =
      interpret("ImageType[Image[{{0, 0.5, 1}}, \"Real32\"]]").unwrap();
    assert_eq!(result, "Real32");
  }

  #[test]
  fn image_constructor_rgb_with_byte_type() {
    clear_state();
    let result = interpret(
      "img = Image[{{{1, 0, 0}, {0, 1, 0}}}, \"Byte\"]; {ImageType[img], ImageDimensions[img]}",
    )
    .unwrap();
    assert_eq!(result, "{Byte, {2, 1}}");
  }

  #[test]
  fn image_from_numeric_array_byte() {
    clear_state();
    // Audit case shape: Image[NumericArray[…, "Byte"]] should still
    // report ImageType -> "Byte" and have correct dimensions.
    assert_eq!(
      interpret(
        "ImageType[Image[NumericArray[{{{172, 150, 162}, {169, 147, 162}}}, \"Byte\"]]]"
      )
      .unwrap(),
      "Byte"
    );
    assert_eq!(
      interpret(
        "ImageDimensions[Image[NumericArray[{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}}, \"Byte\"]]]"
      )
      .unwrap(),
      "{3, 2}"
    );
  }

  // For a Byte-type image, ImageData defaults to Real64 with normalized
  // values (v/255), matching wolframscript.
  #[test]
  fn image_data_normalizes_byte_array() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[Image[NumericArray[{{{221, 139, 66}, {217, 135, 64}}}, \"Byte\"]]]"
      )
      .unwrap(),
      "{{{0.8666666666666667, 0.5450980392156862, 0.25882352941176473}, \
       {0.8509803921568627, 0.5294117647058824, 0.25098039215686274}}}"
    );
  }

  // Explicit "Byte" arg returns raw integers 0..255.
  #[test]
  fn image_data_byte_arg_returns_integers() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[Image[NumericArray[{{{221, 139, 66}}}, \"Byte\"]], \"Byte\"]"
      )
      .unwrap(),
      "{{{221, 139, 66}}}"
    );
  }

  // `Image[{{...}}, "Byte"]` treats input values as bytes (0..255 range).
  #[test]
  fn image_byte_input_values_are_raw_bytes() {
    clear_state();
    assert_eq!(
      interpret("ImageData[Image[{{255, 128, 0}}, \"Byte\"]]").unwrap(),
      "{{1., 0.5019607843137255, 0.}}"
    );
  }

  // Roundtripping through NumericArray preserves the original byte values
  // (no [0, 1] vs [0, 255] confusion that would cause overflow).
  #[test]
  fn image_byte_roundtrip_preserves_values() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[Image[NumericArray[{{{221, 139, 66}}}, \"Byte\"]], \"Byte\"]"
      )
      .unwrap(),
      "{{{221, 139, 66}}}"
    );
  }

  // ImageDimensions on a valid Image3D returns {width, height, depth}.
  #[test]
  fn image_dimensions_image3d_rank3_nested_list() {
    clear_state();
    assert_eq!(
      interpret("ImageDimensions[Image3D[{{{1.0, 0.0}, {0.0, 1.0}}}]]")
        .unwrap(),
      "{2, 2, 1}"
    );
    assert_eq!(
      interpret(
        "ImageDimensions[Image3D[{{{1.0, 0.0}, {0.0, 1.0}}, {{0.5, 0.5}, {0.5, 0.5}}}]]"
      )
      .unwrap(),
      "{2, 2, 2}"
    );
  }

  // Non-square slices and depths give correctly ordered {w, h, d}.
  #[test]
  fn image_dimensions_image3d_non_square() {
    clear_state();
    assert_eq!(
      interpret("ImageDimensions[Image3D[{{{1, 2, 3}}, {{4, 5, 6}}}]]")
        .unwrap(),
      "{3, 1, 2}"
    );
    assert_eq!(
      interpret(
        "ImageDimensions[Image3D[NumericArray[\
         {{{1, 2, 3, 4}, {5, 6, 7, 8}}, {{9, 10, 11, 12}, {13, 14, 15, 16}}, \
          {{17, 18, 19, 20}, {21, 22, 23, 24}}}, \"Byte\"]]]"
      )
      .unwrap(),
      "{4, 2, 3}"
    );
  }

  // Rank-4 (color) Image3D: innermost is channels and not part of dimensions.
  #[test]
  fn image_dimensions_image3d_color_rank4() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[Image3D[{{{{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, 1.0}}}}]]"
      )
      .unwrap(),
      "{3, 1, 1}"
    );
  }

  // ImageChannels for Image3D: rank-3 is grayscale (1 channel), rank-4
  // uses the innermost list length.
  #[test]
  fn image_channels_image3d_rank3_is_one() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageChannels[Image3D[NumericArray[{{{0, 1}, {2, 3}}}, \"Byte\"]]]"
      )
      .unwrap(),
      "1"
    );
    assert_eq!(
      interpret("ImageChannels[Image3D[{{{0.1, 0.2}, {0.3, 0.4}}}]]").unwrap(),
      "1"
    );
  }

  #[test]
  fn image_channels_image3d_rank4_uses_inner_size() {
    clear_state();
    assert_eq!(
      interpret("ImageChannels[Image3D[{{{{1.0, 0, 0}, {0, 1.0, 0}}}}]]")
        .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("ImageChannels[Image3D[{{{{1.0, 0, 0, 0.5}}}}]]").unwrap(),
      "4"
    );
  }

  // ImageType for Image3D: pull the type tag from the NumericArray, or
  // default to Real32 for a raw nested list.
  #[test]
  fn image_type_image3d_with_numeric_array() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageType[Image3D[NumericArray[{{{0, 1}, {2, 3}}}, \"Byte\"]]]"
      )
      .unwrap(),
      "Byte"
    );
    assert_eq!(
      interpret(
        "ImageType[Image3D[NumericArray[{{{1, 2}}}, \"UnsignedInteger16\"]]]"
      )
      .unwrap(),
      "Bit16"
    );
    assert_eq!(
      interpret("ImageType[Image3D[NumericArray[{{{0.5, 0.6}}}, \"Real32\"]]]")
        .unwrap(),
      "Real32"
    );
  }

  #[test]
  fn image_type_image3d_raw_list_defaults_to_real32() {
    clear_state();
    assert_eq!(
      interpret("ImageType[Image3D[{{{0.5, 0.6}}}]]").unwrap(),
      "Real32"
    );
  }

  // ImageAspectRatio for Image3D: height/width as an exact rational.
  #[test]
  fn image_aspect_ratio_image3d() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageAspectRatio[Image3D[NumericArray[{{{0, 1}, {2, 3}}}, \"Byte\"]]]"
      )
      .unwrap(),
      "1"
    );
    assert_eq!(
      interpret(
        "ImageAspectRatio[Image3D[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, \
         {0, 0, 1}, {0.5, 0.5, 0.5}}}]]"
      )
      .unwrap(),
      "4/3"
    );
  }

  // Image3D objects pass ImageQ, even though Image3D itself isn't fully
  // implemented. The shape check accepts a NumericArray of rank 3 or 4,
  // or a nested list of the same rank.
  #[test]
  fn image_q_image3d_numeric_array_byte() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageQ[Image3D[NumericArray[{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}}, \"Byte\"]]]"
      )
      .unwrap(),
      "True"
    );
  }

  // Nested-list rank-3 input also passes.
  #[test]
  fn image_q_image3d_nested_list_rank3() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageQ[Image3D[{{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}}]]"
      )
      .unwrap(),
      "True"
    );
  }

  // Rank-4 nested list (color 3D image) also passes.
  #[test]
  fn image_q_image3d_rank4_color() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageQ[Image3D[{{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}, \
         {{0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}}}}]]"
      )
      .unwrap(),
      "True"
    );
  }

  // Invalid Image3D arguments (wrong rank, scalar) return False.
  #[test]
  fn image_q_image3d_rejects_invalid_data() {
    clear_state();
    assert_eq!(interpret("ImageQ[Image3D[42]]").unwrap(), "False");
    assert_eq!(
      interpret("ImageQ[Image3D[{{0, 1}, {2, 3}}]]").unwrap(),
      "False"
    );
  }

  // Image3D[NumericArray[...]] with rank-2 inner data is invalid.
  #[test]
  fn image_q_image3d_rejects_low_rank_numeric_array() {
    clear_state();
    assert_eq!(
      interpret("ImageQ[Image3D[NumericArray[{{0, 1}}, \"Byte\"]]]").unwrap(),
      "False"
    );
  }

  // wolframscript's structured Image form takes a NumericArray, a type
  // tag, and options like ColorSpace and Interleaving. The options are
  // accepted (currently ignored) so ImageType can recover the type.
  #[test]
  fn image_constructor_accepts_options() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageType[Image[NumericArray[{{172, 150}}, \"UnsignedInteger8\"], \
         \"Byte\", ColorSpace -> Automatic, Interleaving -> None]]"
      )
      .unwrap(),
      "Byte"
    );
  }

  // Same for Bit16.
  #[test]
  fn image_constructor_bit16_with_options() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageType[Image[NumericArray[{{49, 246}}, \"UnsignedInteger16\"], \
         \"Bit16\", ColorSpace -> Automatic, Interleaving -> None]]"
      )
      .unwrap(),
      "Bit16"
    );
  }

  // Bit type is recognised when explicit, even though NumericArray uses
  // UnsignedInteger8 storage underneath.
  #[test]
  fn image_constructor_bit_with_options() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageType[Image[NumericArray[{{0, 1}}, \"UnsignedInteger8\"], \
         \"Bit\", ColorSpace -> Automatic, Interleaving -> None]]"
      )
      .unwrap(),
      "Bit"
    );
  }

  // The Image[data, type] short form (no NumericArray wrapper) also
  // takes options.
  #[test]
  fn image_constructor_short_form_with_options() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageType[Image[{{0.5, 0.6}}, \"Real64\", \
         ColorSpace -> Automatic, Interleaving -> None]]"
      )
      .unwrap(),
      "Real64"
    );
  }

  // ImageColorSpace still falls back to Automatic when the explicit
  // option isn't tracked through the constructor — at least the call
  // doesn't error on a 5-arg Image form (regression for the audit case).
  #[test]
  fn image_color_space_5arg_form_does_not_error() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageColorSpace[Image[NumericArray[{{{172, 150, 162}}}, \
         \"UnsignedInteger8\"], \"Byte\", \
         ColorSpace -> Automatic, Interleaving -> True]]"
      )
      .unwrap(),
      "Automatic"
    );
  }

  #[test]
  fn image_from_numeric_array_real64() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageType[Image[NumericArray[{{{0.5, 0.3, 0.1}, {0.4, 0.2, 0.0}}}, \"Real64\"]]]"
      )
      .unwrap(),
      "Real64"
    );
  }

  #[test]
  fn image_from_numeric_array_explicit_type_overrides() {
    clear_state();
    // An explicit second arg on Image overrides the NumericArray dtype.
    assert_eq!(
      interpret(
        "ImageType[Image[NumericArray[{{{0.5, 0.3, 0.1}}}, \"Real32\"], \"Real64\"]]"
      )
      .unwrap(),
      "Real64"
    );
  }

  #[test]
  fn image_constructor_constant_array() {
    clear_state();
    let result = interpret(
      "img = Image[ConstantArray[{1, 0, 0}, {10, 10}]]; {ImageDimensions[img], ImageChannels[img]}",
    )
    .unwrap();
    assert_eq!(result, "{{10, 10}, 3}");
  }

  #[test]
  fn image_data_grayscale() {
    clear_state();
    let result = interpret("ImageData[Image[{{0, 0.5, 1}}]]").unwrap();
    assert_eq!(result, "{{0., 0.5, 1.}}");
  }

  #[test]
  fn image_data_rgb() {
    clear_state();
    let result =
      interpret("ImageData[Image[{{{1, 0, 0}, {0, 1, 0}}}]]").unwrap();
    assert_eq!(result, "{{{1., 0., 0.}, {0., 1., 0.}}}");
  }

  #[test]
  fn image_color_space_grayscale() {
    clear_state();
    // Wolfram returns Automatic for ImageColorSpace
    let result = interpret("ImageColorSpace[Image[{{0, 1}}]]").unwrap();
    assert_eq!(result, "Automatic");
  }

  #[test]
  fn image_color_space_rgb() {
    clear_state();
    // Wolfram returns Automatic for ImageColorSpace
    let result = interpret("ImageColorSpace[Image[{{{1, 0, 0}}}]]").unwrap();
    assert_eq!(result, "Automatic");
  }

  #[test]
  fn image_color_space_non_image_returns_unevaluated() {
    // Matches wolframscript: non-image argument emits ImageColorSpace::imginv
    // and returns the expression unevaluated (no hard error).
    clear_state();
    assert_eq!(
      interpret("ImageColorSpace[img]").unwrap(),
      "ImageColorSpace[img]"
    );
  }

  #[test]
  fn image_color_space_image3d_returns_automatic() {
    // wolframscript treats Image3D as a valid image for ImageColorSpace
    // and returns Automatic. Previously Woxi rejected Image3D with the
    // generic ImageColorSpace::imginv message.
    clear_state();
    let result = interpret(
      "ImageColorSpace[Image3D[{{{0, 0, 0}, {1, 1, 1}}, \
       {{0.5, 0.5, 0.5}, {0.3, 0.3, 0.3}}}]]",
    )
    .unwrap();
    assert_eq!(result, "Automatic");
  }

  #[test]
  fn image_color_space_image3d_numericarray() {
    // wolframscript also accepts Image3D[NumericArray[...]] wrappers.
    clear_state();
    let result = interpret(
      "ImageColorSpace[Image3D[NumericArray[{{{0, 128}, {255, 64}}, \
       {{32, 16}, {200, 100}}}, \"UnsignedInteger8\"]]]",
    )
    .unwrap();
    assert_eq!(result, "Automatic");
  }

  #[test]
  fn image_data_roundtrip() {
    clear_state();
    let result = interpret(
      "img = Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}]; ImageData[img]",
    )
    .unwrap();
    // ImageData returns f32-precision values (matching wolframscript)
    assert_eq!(
      result,
      "{{0.10000000149011612, 0.20000000298023224, 0.30000001192092896}, {0.4000000059604645, 0.5, 0.6000000238418579}}"
    );
  }

  #[test]
  fn image_stored_in_variable() {
    clear_state();
    let result =
      interpret("img = Image[{{0, 0.5, 1}}]; ImageDimensions[img]").unwrap();
    assert_eq!(result, "{3, 1}");
  }
}

mod image_processing {
  use super::*;

  #[test]
  fn color_negate_grayscale() {
    clear_state();
    let result =
      interpret("ImageData[ColorNegate[Image[{{0, 0.5, 1}}]]]").unwrap();
    assert_eq!(result, "{{1., 0.5, 0.}}");
  }

  // Real32 image arithmetic: 1 - v computed in f64 then cast to f32 is
  // one f32 ulp off from 1f32 - v32 for values where the f64 result
  // rounds the other way. ColorNegate should match wolframscript by
  // doing the negation in f32 for Real32 images.
  #[test]
  fn color_negate_uses_f32_arithmetic() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ColorNegate[Image[{{0.6}}]]]").unwrap(),
      "{{0.3999999761581421}}"
    );
    assert_eq!(
      interpret("ImageData[ColorNegate[Image[{{0.8}}]]]").unwrap(),
      "{{0.19999998807907104}}"
    );
  }

  #[test]
  fn color_negate_rgb() {
    clear_state();
    let result =
      interpret("ImageData[ColorNegate[Image[{{{1, 0, 0.5}}}]]]").unwrap();
    assert_eq!(result, "{{{0., 1., 0.5}}}");
  }

  #[test]
  fn color_negate_yellow_is_blue() {
    // ColorNegate on a named color inverts its components. Integer
    // components stay integer so Yellow == Blue compares equal.
    clear_state();
    assert_eq!(interpret("ColorNegate[Yellow] == Blue").unwrap(), "True");
    assert_eq!(
      interpret("ColorNegate[RGBColor[0.2, 0.3, 0.4]]").unwrap(),
      "RGBColor[0.8, 0.7, 0.6]"
    );
  }

  #[test]
  fn color_negate_graylevel() {
    clear_state();
    assert_eq!(
      interpret("ColorNegate[GrayLevel[0.25]]").unwrap(),
      "GrayLevel[0.75]"
    );
  }

  // ColorNegate stays in the input's color space: a Hue stays a Hue and a
  // CMYKColor stays a CMYKColor (negation is computed via RGB).
  #[test]
  fn color_negate_preserves_hue_and_cmyk() {
    clear_state();
    assert_eq!(
      interpret("ColorNegate[Hue[0.5]]").unwrap(),
      "Hue[0., 1., 1.]"
    );
    assert_eq!(
      interpret("ColorNegate[Hue[0.2, 0.5, 0.8]]").unwrap(),
      "Hue[0.7000000000000001, 0.6666666666666667, 0.6]"
    );
    assert_eq!(
      interpret("ColorNegate[CMYKColor[0, 1, 1, 0]]").unwrap(),
      "CMYKColor[1., 0., 0., 0.]"
    );
  }

  // wolframscript on `ColorNegate[<non-image>]` emits
  // `ColorNegate::imginv` and leaves the call unevaluated.
  // Regression for mathics color_operations.py
  // `ColorNegate[Import["ExampleData/sunflowers.jpg"]]` row
  // (where Import fails and yields `$Failed`).
  #[test]
  fn color_negate_non_image_returns_unevaluated() {
    clear_state();
    assert_eq!(interpret("ColorNegate[42]").unwrap(), "ColorNegate[42]");
  }

  #[test]
  fn color_negate_failed_returns_unevaluated() {
    clear_state();
    assert_eq!(
      interpret("ColorNegate[$Failed]").unwrap(),
      "ColorNegate[$Failed]"
    );
  }

  // `Colorize[<integer-matrix>, …]` renders the matrix as an
  // Image (printed as `-Image-`). The `ColorFunction -> …` option
  // is accepted but currently the renderer maps each label to a
  // shade of gray; the displayed placeholder still matches
  // wolframscript. Regression for mathics image/colors.py
  // `Colorize[{{1, 2}, {2, 2}, {2, 3}}, …]` row.
  #[test]
  fn colorize_integer_matrix_returns_image() {
    clear_state();
    assert_eq!(
      interpret("Colorize[{{1, 2}, {2, 2}, {2, 3}}, ColorFunction -> (Blend[{White, Blue}, #]&)]").unwrap(),
      "-Image-"
    );
  }

  #[test]
  fn colorize_simple_matrix_returns_image() {
    clear_state();
    assert_eq!(interpret("Colorize[{{1, 2}, {3, 4}}]").unwrap(), "-Image-");
  }

  #[test]
  fn colorize_non_matrix_returns_unevaluated() {
    clear_state();
    assert_eq!(interpret("Colorize[42]").unwrap(), "Colorize[42]");
  }

  #[test]
  fn binarize_default_threshold() {
    clear_state();
    let result =
      interpret("ImageData[Binarize[Image[{{0.3, 0.5, 0.7}}]]]").unwrap();
    // Binarize produces Bit-type images, so ImageData returns integers
    assert_eq!(result, "{{0, 1, 1}}");
  }

  #[test]
  fn binarize_custom_threshold() {
    clear_state();
    let result =
      interpret("ImageData[Binarize[Image[{{0.3, 0.5, 0.7}}], 0.6]]").unwrap();
    assert_eq!(result, "{{0, 0, 1}}");
  }

  // The explicit-threshold form uses strict greater-than: a pixel
  // exactly equal to the threshold is binarized to 0 (matching
  // wolframscript). Previously Woxi used >= and turned 0.5 into 1.
  #[test]
  fn binarize_threshold_is_strict() {
    clear_state();
    assert_eq!(
      interpret("ImageData[Binarize[Image[{{0.5}}], 0.5]]").unwrap(),
      "{{0}}"
    );
    assert_eq!(
      interpret("ImageData[Binarize[Image[{{0.5}}], 0.4]]").unwrap(),
      "{{1}}"
    );
  }

  // Range threshold: Binarize[image, {t1, t2}] sets a pixel to 1 iff
  // t1 <= v <= t2 (both endpoints inclusive).
  #[test]
  fn binarize_range_threshold() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[Binarize[Image[{{0.0, 0.25, 0.5, 0.75, 1.0}}], {0.2, 0.7}]]"
      )
      .unwrap(),
      "{{0, 1, 1, 0, 0}}"
    );
    assert_eq!(
      interpret("ImageData[Binarize[Image[{{0.2, 0.7}}], {0.2, 0.7}]]")
        .unwrap(),
      "{{1, 1}}"
    );
  }

  #[test]
  fn image_adjust_rescale() {
    clear_state();
    let result =
      interpret("ImageData[ImageAdjust[Image[{{0.0, 0.5, 1.0}}]]]").unwrap();
    assert_eq!(result, "{{0., 0.5, 1.}}");
  }

  // ImageAdjust[image, c] applies the contrast curve
  // v' = 0.5 + (1 + c)*(v - 0.5), clamped to [0, 1]. No rescaling.
  // c = 0 is the identity, so pixel values pass through unchanged.
  #[test]
  fn image_adjust_contrast_zero_is_identity() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageAdjust[Image[{{0.1, 0.5, 0.9}}], 0]]").unwrap(),
      "{{0.10000000149011612, 0.5, 0.8999999761581421}}"
    );
  }

  // Negative contrast pulls values toward 0.5.
  #[test]
  fn image_adjust_negative_contrast() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageAdjust[Image[{{0.1, 0.5, 0.9}}], -0.5]]")
        .unwrap(),
      "{{0.30000001192092896, 0.5, 0.699999988079071}}"
    );
  }

  // Positive contrast pushes values away from 0.5 and clamps.
  #[test]
  fn image_adjust_positive_contrast_clamps() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageAdjust[Image[{{0.2, 0.5, 0.7}}], 0.5]]")
        .unwrap(),
      "{{0.05000000447034836, 0.5, 0.7999999523162842}}"
    );
  }

  // ImageAdjust[image, {c, b}] applies brightness (v * (1 + b)) before
  // the contrast curve. For c = 0 the contrast step is the identity.
  #[test]
  fn image_adjust_brightness_only() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageAdjust[Image[{{0.1, 0.5, 0.9}}], {0, 2}]]")
        .unwrap(),
      "{{0.30000001192092896, 1., 1.}}"
    );
  }

  // Combined brightness + contrast: brightness applied first.
  #[test]
  fn image_adjust_brightness_then_contrast() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageAdjust[Image[{{0.1, 0.5, 0.9}}], {0.5, 0.5}]]")
        .unwrap(),
      "{{0., 0.875, 1.}}"
    );
  }

  #[test]
  fn image_reflect_horizontal() {
    clear_state();
    let result =
      interpret("ImageDimensions[ImageReflect[Image[{{0, 0.5, 1}}]]]").unwrap();
    assert_eq!(result, "{3, 1}");
  }

  #[test]
  fn image_reflect_vertical() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageReflect[Image[{{0, 1}, {1, 0}}], Top -> Bottom]]",
    )
    .unwrap();
    assert_eq!(result, "{2, 2}");
  }

  // Default ImageReflect is top-bottom (vertical) flip — rows reversed.
  // Pixel precision is preserved (no Byte quantization).
  #[test]
  fn image_reflect_default_is_vertical_flip() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageReflect[Image[{{0.1, 0.2}, {0.3, 0.4}}]]]")
        .unwrap(),
      "{{0.30000001192092896, 0.4000000059604645}, \
       {0.10000000149011612, 0.20000000298023224}}"
    );
  }

  // ImageReflect[img, Left] / Right swaps left and right columns.
  #[test]
  fn image_reflect_left_right_flip() {
    clear_state();
    let expected = "{{0.20000000298023224, 0.10000000149011612}, \
                    {0.4000000059604645, 0.30000001192092896}}";
    assert_eq!(
      interpret(
        "ImageData[ImageReflect[Image[{{0.1, 0.2}, {0.3, 0.4}}], Left]]"
      )
      .unwrap(),
      expected
    );
    assert_eq!(
      interpret(
        "ImageData[ImageReflect[Image[{{0.1, 0.2}, {0.3, 0.4}}], Right]]"
      )
      .unwrap(),
      expected
    );
  }

  // Top/Bottom side argument is the same as the default vertical flip.
  #[test]
  fn image_reflect_top_bottom_flip() {
    clear_state();
    let expected = "{{0.30000001192092896, 0.4000000059604645}, \
                    {0.10000000149011612, 0.20000000298023224}}";
    assert_eq!(
      interpret(
        "ImageData[ImageReflect[Image[{{0.1, 0.2}, {0.3, 0.4}}], Top]]"
      )
      .unwrap(),
      expected
    );
    assert_eq!(
      interpret(
        "ImageData[ImageReflect[Image[{{0.1, 0.2}, {0.3, 0.4}}], Bottom]]"
      )
      .unwrap(),
      expected
    );
  }

  // Rule form: Top -> Bottom and Bottom -> Top both vertical flip.
  #[test]
  fn image_reflect_rule_vertical() {
    clear_state();
    let expected = "{{0.30000001192092896, 0.4000000059604645}, \
                    {0.10000000149011612, 0.20000000298023224}}";
    assert_eq!(
      interpret(
        "ImageData[ImageReflect[Image[{{0.1, 0.2}, {0.3, 0.4}}], Top -> Bottom]]"
      )
      .unwrap(),
      expected
    );
    assert_eq!(
      interpret(
        "ImageData[ImageReflect[Image[{{0.1, 0.2}, {0.3, 0.4}}], Bottom -> Top]]"
      )
      .unwrap(),
      expected
    );
  }

  // Rule form: Left -> Right and Right -> Left both horizontal flip.
  #[test]
  fn image_reflect_rule_horizontal() {
    clear_state();
    let expected = "{{0.20000000298023224, 0.10000000149011612}, \
                    {0.4000000059604645, 0.30000001192092896}}";
    assert_eq!(
      interpret(
        "ImageData[ImageReflect[Image[{{0.1, 0.2}, {0.3, 0.4}}], Left -> Right]]"
      )
      .unwrap(),
      expected
    );
    assert_eq!(
      interpret(
        "ImageData[ImageReflect[Image[{{0.1, 0.2}, {0.3, 0.4}}], Right -> Left]]"
      )
      .unwrap(),
      expected
    );
  }

  // Diagonal reflection: Left -> Top transposes the image; non-square
  // images have width and height swapped.
  #[test]
  fn image_reflect_main_diagonal() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageReflect[Image[{{0.1, 0.2}, {0.3, 0.4}}], Left -> Top]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.30000001192092896}, \
       {0.20000000298023224, 0.4000000059604645}}"
    );
    assert_eq!(
      interpret(
        "ImageDimensions[ImageReflect[Image[{{0, 0, 0}, {1, 1, 1}}], Left -> Top]]"
      )
      .unwrap(),
      "{2, 3}"
    );
  }

  // Anti-diagonal reflection: Left -> Bottom / Right -> Top.
  #[test]
  fn image_reflect_anti_diagonal() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageReflect[Image[{{0.1, 0.2}, {0.3, 0.4}}], Left -> Bottom]]"
      )
      .unwrap(),
      "{{0.4000000059604645, 0.20000000298023224}, \
       {0.30000001192092896, 0.10000000149011612}}"
    );
  }

  // RGB image keeps channel ordering; only spatial layout changes.
  #[test]
  fn image_reflect_rgb_preserves_channels() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageReflect[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}], Left]]"
      )
      .unwrap(),
      "{{{0., 1., 0.}, {1., 0., 0.}}}"
    );
  }

  #[test]
  fn image_rotate_90() {
    clear_state();
    // 3x1 image rotated 90° becomes 1x3
    let result =
      interpret("ImageDimensions[ImageRotate[Image[{{0, 0.5, 1}}], Pi/2]]")
        .unwrap();
    assert_eq!(result, "{1, 3}");
  }

  #[test]
  fn image_rotate_180() {
    clear_state();
    let result =
      interpret("ImageDimensions[ImageRotate[Image[{{0, 0.5, 1}}], Pi]]")
        .unwrap();
    assert_eq!(result, "{3, 1}");
  }

  // Blend on a list of images linearly interpolates per-pixel.
  #[test]
  fn blend_two_images_half() {
    clear_state();
    assert_eq!(
      interpret("ImageData[Blend[{Image[{{0.0}}], Image[{{1.0}}]}, 0.5]]")
        .unwrap(),
      "{{0.5}}"
    );
  }

  #[test]
  fn blend_two_images_with_inner_pixels() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[Blend[{Image[{{0.0, 0.5}}], Image[{{1.0, 0.5}}]}, 0.25]]"
      )
      .unwrap(),
      "{{0.25, 0.5}}"
    );
  }

  // No t argument → equal blend (average).
  #[test]
  fn blend_two_images_default_is_average() {
    clear_state();
    assert_eq!(
      interpret("ImageData[Blend[{Image[{{0.0}}], Image[{{1.0}}]}]]").unwrap(),
      "{{0.5}}"
    );
  }

  // Three images at t = 0.25: interpolation lands between images 1 and 2
  // (positions x = 0, 0.5, 1) at local_t = 0.5 → average of 0 and 0.5.
  #[test]
  fn blend_three_images_quarter() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[Blend[{Image[{{0.0}}], Image[{{0.5}}], Image[{{1.0}}]}, 0.25]]"
      )
      .unwrap(),
      "{{0.25}}"
    );
  }

  // Output dimensions match the inputs (all images must agree in shape).
  #[test]
  fn blend_image_preserves_dimensions() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[Blend[{Image[{{0.0, 0.5}, {0.5, 1.0}}], \
         Image[{{1.0, 0.5}, {0.5, 0.0}}]}, 0.5]]"
      )
      .unwrap(),
      "{2, 2}"
    );
  }

  // ColorSeparate splits an image into one single-channel image per
  // channel. The returned images preserve width, height and image type.
  #[test]
  fn color_separate_rgb_yields_three_grayscales() {
    clear_state();
    assert_eq!(
      interpret(
        "Length[ColorSeparate[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}]]]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret(
        "ImageData[ColorSeparate[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}]][[1]]]"
      )
      .unwrap(),
      "{{1., 0.}}"
    );
    assert_eq!(
      interpret(
        "ImageData[ColorSeparate[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}]][[2]]]"
      )
      .unwrap(),
      "{{0., 1.}}"
    );
    assert_eq!(
      interpret(
        "ImageData[ColorSeparate[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}]][[3]]]"
      )
      .unwrap(),
      "{{0., 0.}}"
    );
  }

  // RGBA → 4 images, the last being the alpha channel.
  #[test]
  fn color_separate_rgba_yields_four_channels() {
    clear_state();
    assert_eq!(
      interpret("Length[ColorSeparate[Image[{{{1.0, 0.0, 0.0, 0.5}}}]]]")
        .unwrap(),
      "4"
    );
    assert_eq!(
      interpret(
        "ImageData[ColorSeparate[Image[{{{1.0, 0.0, 0.0, 0.5}}}]][[4]]]"
      )
      .unwrap(),
      "{{0.5}}"
    );
  }

  // Grayscale → one image, the input itself (passes through unchanged).
  #[test]
  fn color_separate_grayscale_passthrough() {
    clear_state();
    assert_eq!(
      interpret("Length[ColorSeparate[Image[{{0.1, 0.2}}]]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("ImageData[ColorSeparate[Image[{{0.1, 0.2}}]][[1]]]").unwrap(),
      "{{0.10000000149011612, 0.20000000298023224}}"
    );
  }

  // Each separated image keeps the original spatial dimensions.
  #[test]
  fn color_separate_preserves_dimensions() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[ColorSeparate[Image[{{{1.0, 0.0, 0.0}, {0.5, 0.5, 0.5}}}]][[1]]]"
      )
      .unwrap(),
      "{2, 1}"
    );
  }

  // Dilation on an Image: replace each pixel with the max of its
  // (2r+1)×(2r+1) neighbourhood, clipped at the boundaries. Erosion is
  // the same with min; Opening / Closing compose the two.
  #[test]
  fn dilation_image_grayscale() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[Dilation[Image[{{0.1, 0.5, 0.2}, {0.8, 0.3, 0.9}, {0.4, 0.7, 0.6}}], 1]]"
      )
      .unwrap(),
      "{{0.800000011920929, 0.8999999761581421, 0.8999999761581421}, \
       {0.800000011920929, 0.8999999761581421, 0.8999999761581421}, \
       {0.800000011920929, 0.8999999761581421, 0.8999999761581421}}"
    );
  }

  #[test]
  fn erosion_image_grayscale() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[Erosion[Image[{{0.1, 0.5, 0.2}, {0.8, 0.3, 0.9}, {0.4, 0.7, 0.6}}], 1]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.10000000149011612, 0.20000000298023224}, \
       {0.10000000149011612, 0.10000000149011612, 0.20000000298023224}, \
       {0.30000001192092896, 0.30000001192092896, 0.30000001192092896}}"
    );
  }

  // Opening = Erosion then Dilation.
  #[test]
  fn opening_image_grayscale() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[Opening[Image[{{0.1, 0.5, 0.2}, {0.8, 0.3, 0.9}, {0.4, 0.7, 0.6}}], 1]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.20000000298023224, 0.20000000298023224}, \
       {0.30000001192092896, 0.30000001192092896, 0.30000001192092896}, \
       {0.30000001192092896, 0.30000001192092896, 0.30000001192092896}}"
    );
  }

  // Closing = Dilation then Erosion.
  #[test]
  fn closing_image_grayscale() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[Closing[Image[{{0.1, 0.5, 0.2}, {0.8, 0.3, 0.9}, {0.4, 0.7, 0.6}}], 1]]"
      )
      .unwrap(),
      "{{0.800000011920929, 0.800000011920929, 0.8999999761581421}, \
       {0.800000011920929, 0.800000011920929, 0.8999999761581421}, \
       {0.800000011920929, 0.800000011920929, 0.8999999761581421}}"
    );
  }

  // MedianFilter on an Image applies the 2D median filter per channel.
  #[test]
  fn median_filter_image_grayscale() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[MedianFilter[Image[{{0.1, 0.5, 0.2}, {0.8, 0.3, 0.9}, {0.4, 0.7, 0.6}}], 1]]"
      )
      .unwrap(),
      "{{0.5, 0.5, 0.5}, \
       {0.5, 0.5, 0.6000000238418579}, \
       {0.699999988079071, 0.699999988079071, 0.699999988079071}}"
    );
  }

  // MeanFilter[list, r] on a 1D list averages a clipped neighborhood,
  // keeping exact rational results.
  #[test]
  fn mean_filter_1d_rational() {
    clear_state();
    assert_eq!(
      interpret("MeanFilter[{1, 2, 3, 4, 5}, 1]").unwrap(),
      "{3/2, 2, 3, 4, 9/2}"
    );
    assert_eq!(
      interpret("MeanFilter[{1, 2, 3, 4, 5}, 2]").unwrap(),
      "{2, 5/2, 3, 7/2, 4}"
    );
    // r = 0 is the identity; large r averages everything.
    assert_eq!(
      interpret("MeanFilter[{1, 2, 3, 4, 5}, 0]").unwrap(),
      "{1, 2, 3, 4, 5}"
    );
    assert_eq!(
      interpret("MeanFilter[{1, 2, 3, 4, 5}, 10]").unwrap(),
      "{3, 3, 3, 3, 3}"
    );
  }

  // A negative integer radius is treated as its absolute value.
  #[test]
  fn mean_filter_negative_radius_is_abs() {
    clear_state();
    assert_eq!(
      interpret("MeanFilter[{1, 2, 3}, -1]").unwrap(),
      "{3/2, 2, 5/2}"
    );
  }

  // Symbolic and Real element handling.
  #[test]
  fn mean_filter_symbolic_and_real() {
    clear_state();
    assert_eq!(
      interpret("MeanFilter[{a, b, c}, 1]").unwrap(),
      "{(a + b)/2, (a + b + c)/3, (b + c)/2}"
    );
    // Real elements stay Real even when the mean is a whole number (2.).
    assert_eq!(
      interpret("MeanFilter[{1.0, 2.0, 3.0, 4.0, 5.0}, 1]").unwrap(),
      "{1.5, 2., 3., 4., 4.5}"
    );
    // Edge cases.
    assert_eq!(interpret("MeanFilter[{}, 1]").unwrap(), "{}");
    assert_eq!(interpret("MeanFilter[{7}, 1]").unwrap(), "{7}");
  }

  // MeanFilter on a 2D rectangular array uses a square clipped window.
  #[test]
  fn mean_filter_2d() {
    clear_state();
    assert_eq!(
      interpret("MeanFilter[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]").unwrap(),
      "{{3, 7/2, 4}, {9/2, 5, 11/2}, {6, 13/2, 7}}"
    );
    assert_eq!(
      interpret("MeanFilter[{{1, 2, 3, 4}, {5, 6, 7, 8}}, 1]").unwrap(),
      "{{7/2, 4, 5, 11/2}, {7/2, 4, 5, 11/2}}"
    );
  }

  // StandardDeviationFilter[list, r]: a moving (sample) standard deviation
  // over a clipped neighborhood. Exact for exact input.
  #[test]
  fn standard_deviation_filter_1d() {
    clear_state();
    assert_eq!(
      interpret("StandardDeviationFilter[{1, 2, 3, 4, 5}, 1]").unwrap(),
      "{1/Sqrt[2], 1, 1, 1, 1/Sqrt[2]}"
    );
    assert_eq!(
      interpret("StandardDeviationFilter[{1, 5, 2, 8, 3}, 1]").unwrap(),
      "{2*Sqrt[2], Sqrt[13/3], 3, Sqrt[31/3], 5/Sqrt[2]}"
    );
    assert_eq!(
      interpret("StandardDeviationFilter[{1, 2, 3, 4, 5}, 2]").unwrap(),
      "{1, Sqrt[5/3], Sqrt[5/2], Sqrt[5/3], 1}"
    );
    // A constant window has zero deviation.
    assert_eq!(
      interpret("StandardDeviationFilter[{4, 4, 4}, 1]").unwrap(),
      "{0, 0, 0}"
    );
  }

  // StandardDeviationFilter on a 2D rectangular array uses a square window.
  #[test]
  fn standard_deviation_filter_2d() {
    clear_state();
    assert_eq!(
      interpret(
        "StandardDeviationFilter[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]"
      )
      .unwrap(),
      "{{Sqrt[10/3], Sqrt[7/2], Sqrt[10/3]}, \
       {Sqrt[15/2], Sqrt[15/2], Sqrt[15/2]}, \
       {Sqrt[10/3], Sqrt[7/2], Sqrt[10/3]}}"
    );
  }

  // Regression: Mean of Real elements summing to a whole number must
  // display with a trailing dot, matching wolframscript.
  #[test]
  fn mean_real_whole_keeps_dot() {
    clear_state();
    assert_eq!(interpret("Mean[{1.0, 2.0, 3.0}]").unwrap(), "2.");
    assert_eq!(interpret("Mean[{1, 2, 3}]").unwrap(), "2");
  }

  // Default ImageRotate[image] is a Pi/2 counter-clockwise rotation.
  // For a 2x2 image, pixel data is reorganised but precision is
  // preserved (no Byte quantization round-trip).
  #[test]
  fn image_rotate_default_is_pi_over_two_ccw() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageRotate[Image[{{0.1, 0.2}, {0.3, 0.4}}]]]")
        .unwrap(),
      "{{0.20000000298023224, 0.4000000059604645}, \
       {0.10000000149011612, 0.30000001192092896}}"
    );
  }

  #[test]
  fn image_rotate_pi_over_two_square() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageRotate[Image[{{0.1, 0.2}, {0.3, 0.4}}], Pi/2]]"
      )
      .unwrap(),
      "{{0.20000000298023224, 0.4000000059604645}, \
       {0.10000000149011612, 0.30000001192092896}}"
    );
  }

  #[test]
  fn image_rotate_3_pi_over_two_square() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageRotate[Image[{{0.1, 0.2}, {0.3, 0.4}}], 3*Pi/2]]"
      )
      .unwrap(),
      "{{0.30000001192092896, 0.10000000149011612}, \
       {0.4000000059604645, 0.20000000298023224}}"
    );
  }

  #[test]
  fn image_rotate_pi_square() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageRotate[Image[{{0.1, 0.2}, {0.3, 0.4}}], Pi]]")
        .unwrap(),
      "{{0.4000000059604645, 0.30000001192092896}, \
       {0.20000000298023224, 0.10000000149011612}}"
    );
  }

  // For a non-square image, Pi/2 swaps width and height.
  #[test]
  fn image_rotate_pi_over_two_non_square() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageRotate[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], Pi/2]]"
      )
      .unwrap(),
      "{{0.30000001192092896, 0.6000000238418579}, \
       {0.20000000298023224, 0.5}, \
       {0.10000000149011612, 0.4000000059604645}}"
    );
    assert_eq!(
      interpret(
        "ImageDimensions[ImageRotate[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], Pi/2]]"
      )
      .unwrap(),
      "{2, 3}"
    );
  }

  // 0 and 2*Pi leave the image identical.
  #[test]
  fn image_rotate_zero_angle_is_identity() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageRotate[Image[{{0.1, 0.2}, {0.3, 0.4}}], 0]]")
        .unwrap(),
      "{{0.10000000149011612, 0.20000000298023224}, \
       {0.30000001192092896, 0.4000000059604645}}"
    );
  }

  // RGB images preserve channel ordering across rotation.
  #[test]
  fn image_rotate_rgb_preserves_channels() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageRotate[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}], Pi/2]]"
      )
      .unwrap(),
      "{{{0., 1., 0.}}, {{1., 0., 0.}}}"
    );
  }

  #[test]
  fn image_resize() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageResize[Image[{{0, 0.5, 1}, {1, 0.5, 0}}], {6, 4}]]",
    )
    .unwrap();
    assert_eq!(result, "{6, 4}");
  }

  #[test]
  fn image_resize_single_number_wide() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageResize[ConstantImage[0.5, {200, 100}], 128]]",
    )
    .unwrap();
    assert_eq!(result, "{128, 64}");
  }

  #[test]
  fn image_resize_single_number_tall() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageResize[ConstantImage[0.5, {100, 200}], 128]]",
    )
    .unwrap();
    assert_eq!(result, "{128, 256}");
  }

  #[test]
  fn image_resize_single_number_small() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageResize[ConstantImage[0.5, {200, 100}], 50]]",
    )
    .unwrap();
    assert_eq!(result, "{50, 25}");
  }

  #[test]
  fn image_trim_basic() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageTrim[ConstantImage[0.5, {10, 10}], {{0, 0}, {5, 5}}]]",
    )
    .unwrap();
    assert_eq!(result, "{6, 6}");
  }

  #[test]
  fn image_trim_full_image() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageTrim[ConstantImage[0.5, {10, 10}], {{0, 0}, {10, 10}}]]",
    )
    .unwrap();
    assert_eq!(result, "{10, 10}");
  }

  #[test]
  fn image_trim_inner_region() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageTrim[ConstantImage[0.5, {10, 10}], {{2, 3}, {7, 8}}]]",
    )
    .unwrap();
    assert_eq!(result, "{7, 7}");
  }

  #[test]
  fn image_trim_single_pixel() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageTrim[ConstantImage[0.5, {10, 10}], {{0, 0}, {0, 0}}]]",
    )
    .unwrap();
    assert_eq!(result, "{1, 1}");
  }

  #[test]
  fn image_trim_float_coords() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageTrim[ConstantImage[0.5, {10, 10}], {{0.9, 0.9}, {1.1, 1.1}}]]",
    )
    .unwrap();
    assert_eq!(result, "{2, 2}");
  }

  #[test]
  fn image_trim_clips_to_bounds() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageTrim[ConstantImage[0.5, {300, 200}], {{81., 10.}, {250., 240.}}]]",
    )
    .unwrap();
    assert_eq!(result, "{171, 191}");
  }

  #[test]
  fn image_trim_preserves_content() {
    clear_state();
    let result = interpret(
      "ImageQ[ImageTrim[ConstantImage[0.5, {10, 10}], {{2, 2}, {8, 8}}]]",
    )
    .unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn image_trim_non_image_returns_unevaluated() {
    clear_state();
    let result = interpret("ImageTrim[42, {{0, 0}, {5, 5}}]").unwrap();
    assert_eq!(result, "ImageTrim[42, {{0, 0}, {5, 5}}]");
  }

  #[test]
  fn image_crop_manual_returns_unevaluated() {
    clear_state();
    // Wolfram's ImageCrop does not accept {{x1,y1},{x2,y2}} as a region spec
    let result = interpret(
      "ImageDimensions[ImageCrop[Image[{{0, 0, 0, 0}, {0, 1, 1, 0}, {0, 0, 0, 0}}], {{1, 1}, {3, 2}}]]",
    )
    .unwrap();
    assert_eq!(
      result,
      "ImageDimensions[ImageCrop[-Image-, {{1, 1}, {3, 2}}]]"
    );
  }

  #[test]
  fn blur_preserves_dimensions() {
    clear_state();
    let result = interpret(
      "ImageDimensions[Blur[Image[{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}]]]",
    )
    .unwrap();
    assert_eq!(result, "{3, 3}");
  }

  // PixelValuePositions on a multi-channel image accepts a list of
  // target channel values. With no tolerance, only exact matches.
  #[test]
  fn pixel_value_positions_rgb_exact() {
    clear_state();
    assert_eq!(
      interpret(
        "PixelValuePositions[\
         Image[{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}], {1, 0, 0}]"
      )
      .unwrap(),
      "{{1, 1}}"
    );
  }

  // Multi-channel comparison uses L∞ (max-abs-channel-diff) for the
  // tolerance check. With tol=0.2 the (0.85, 0.05, 0.05) pixel matches
  // (max diff = 0.15 ≤ 0.2) but (0.5, 0.1, 0.1) does not.
  #[test]
  fn pixel_value_positions_rgb_with_tolerance() {
    clear_state();
    assert_eq!(
      interpret(
        "PixelValuePositions[\
         Image[{{{1.0, 0.0, 0.0}, {0.5, 0.1, 0.1}, \
         {0.85, 0.05, 0.05}, {0.0, 1.0, 0.0}}}], {1.0, 0.0, 0.0}, 0.2]"
      )
      .unwrap(),
      "{{1, 1}, {3, 1}}"
    );
  }

  // Multiple matching positions are returned in top-down, left-to-right
  // order; y uses bottom-up coordinates.
  #[test]
  fn pixel_value_positions_rgb_multiple_matches() {
    clear_state();
    assert_eq!(
      interpret(
        "PixelValuePositions[\
         Image[{{{1, 0, 0}, {1, 0, 0}, {0, 1, 0}}}], {1, 0, 0}]"
      )
      .unwrap(),
      "{{1, 1}, {2, 1}}"
    );
  }

  // A scalar target against a multi-channel image yields no matches
  // (target rank must match the pixel rank).
  #[test]
  fn pixel_value_positions_scalar_target_on_rgb() {
    clear_state();
    assert_eq!(
      interpret("PixelValuePositions[Image[{{{1.0, 0.0, 0.0}}}], 1.0]")
        .unwrap(),
      "{}"
    );
  }

  // ImageTrim crops to a region of interest defined by {{x1, y1}, {x2, y2}}.
  // Coordinates are 1-indexed with y measured from the bottom. Pixel
  // precision is preserved (no Byte round-trip).
  #[test]
  fn image_trim_preserves_pixels() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageTrim[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], \
         {{2, 1}, {3, 2}}]]"
      )
      .unwrap(),
      "{{0.20000000298023224, 0.30000001192092896}, \
       {0.5, 0.6000000238418579}}"
    );
    assert_eq!(
      interpret(
        "ImageDimensions[ImageTrim[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], \
         {{2, 1}, {3, 2}}]]"
      )
      .unwrap(),
      "{2, 2}"
    );
  }

  // Channels and image type pass through.
  #[test]
  fn image_trim_preserves_channels_and_type() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageChannels[ImageTrim[\
         Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}}], \
         {{2, 1}, {3, 1}}]]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("ImageType[ImageTrim[Image[{{0.1, 0.5}}], {{1, 1}, {2, 1}}]]")
        .unwrap(),
      "Real32"
    );
  }

  // RecurrenceFilter on a 1D image applies the IIR filter along the
  // row. With a = {1, -0.5}, b = {1}: y[n] = x[n] + 0.5*y[n-1].
  #[test]
  fn recurrence_filter_grayscale_row() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[RecurrenceFilter[{{1, -0.5}, {1}}, Image[{{0.1, 0.5, 0.9}}]]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.550000011920929, 1.1749999523162842}}"
    );
  }

  // On a 2D image the filter runs along rows, then along columns.
  // (Woxi computes the intermediate row pass in f64 before the column
  // pass; the result agrees with wolframscript to within one f32 ulp
  // for the off-diagonal pixel.)
  #[test]
  fn recurrence_filter_grayscale_2d() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[RecurrenceFilter[{{1, -0.5}, {1}}, \
         Image[{{0.1, 0.5, 0.9}, {0.2, 0.4, 0.6}}]]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.550000011920929, 1.1749999523162842}, \
       {0.25, 0.7749999761581421, 1.4375}}"
    );
  }

  // RGB: each channel filtered independently.
  #[test]
  fn recurrence_filter_rgb_per_channel() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[RecurrenceFilter[{{1, -0.5}, {1}}, \
         Image[{{{1.0, 0.0, 0.0}, {0.5, 0.5, 0.5}}}]]]"
      )
      .unwrap(),
      "{{{1., 0., 0.}, {1., 0.5, 0.5}}}"
    );
  }

  // Dimensions, channels and image type pass through.
  #[test]
  fn recurrence_filter_preserves_image_metadata() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[RecurrenceFilter[{{1, -0.5}, {1}}, \
         Image[{{0.1, 0.5, 0.9}, {0.2, 0.4, 0.6}}]]]"
      )
      .unwrap(),
      "{3, 2}"
    );
    assert_eq!(
      interpret(
        "ImageType[RecurrenceFilter[{{1, -0.5}, {1}}, Image[{{0.1, 0.5}}]]]"
      )
      .unwrap(),
      "Real32"
    );
  }

  // Image[image] is idempotent — wrapping an existing Image in another
  // Image[…] returns the inner image unchanged.
  #[test]
  fn image_of_image_returns_inner() {
    clear_state();
    assert_eq!(
      interpret("ImageData[Image[Image[{{0.1, 0.2}, {0.3, 0.4}}]]]").unwrap(),
      "{{0.10000000149011612, 0.20000000298023224}, \
       {0.30000001192092896, 0.4000000059604645}}"
    );
    assert_eq!(
      interpret("ImageDimensions[Image[Image[{{0.1, 0.2, 0.3}}]]]").unwrap(),
      "{3, 1}"
    );
  }

  // ImageConvolve applies a 2D kernel per pixel with replicated
  // boundary, kernel center at floor(size/2) for each axis.
  #[test]
  fn image_convolve_identity_kernel() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageConvolve[Image[{{0.1, 0.5, 0.9}}], {{1}}]]")
        .unwrap(),
      "{{0.10000000149011612, 0.5, 0.8999999761581421}}"
    );
  }

  // 1D box blur via a 1×3 kernel of {1/3, 1/3, 1/3}. Woxi computes
  // the convolution in f64 and casts to Real32 on the way out.
  #[test]
  fn image_convolve_1d_box_blur() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageConvolve[Image[{{0.1, 0.5, 0.9}}], {{1, 1, 1}}/3]]"
      )
      .unwrap(),
      "{{0.23333333432674408, 0.5, 0.7666666507720947}}"
    );
  }

  // Triangular 1D kernel exercises non-uniform weights.
  #[test]
  fn image_convolve_triangular_kernel() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageConvolve[Image[{{0.1, 0.5, 0.9, 0.5, 0.1}}], {{1, 2, 1}}/4]]"
      )
      .unwrap(),
      "{{0.20000000298023224, 0.5, 0.699999988079071, 0.5, 0.20000000298023224}}"
    );
  }

  // Even-size kernel: center at index size/2, so {a, b} reads
  // pixel[i-1] and pixel[i].
  #[test]
  fn image_convolve_even_size_kernel() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageConvolve[Image[{{0.1, 0.2, 0.3, 0.4, 0.5}}], {{0.5, 0.5}}]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.15000000596046448, 0.25, \
        0.3499999940395355, 0.44999998807907104}}"
    );
  }

  // 2D kernel: cross-pattern Laplacian-like with replicated boundary.
  #[test]
  fn image_convolve_2d_cross_kernel() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageConvolve[Image[{{0.1, 0.2}, {0.3, 0.4}}], \
         {{0, 1, 0}, {1, 0, 1}, {0, 1, 0}}/4]]"
      )
      .unwrap(),
      "{{0.17499999701976776, 0.22499999403953552}, \
       {0.2750000059604645, 0.32499998807907104}}"
    );
  }

  // Channels and image type pass through.
  #[test]
  fn image_convolve_preserves_channels_and_type() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageChannels[ImageConvolve[Image[{{{1.0, 0.0, 0.0}}}], {{1}}]]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("ImageType[ImageConvolve[Image[{{0.1, 0.5}}], {{1}}]]")
        .unwrap(),
      "Real32"
    );
  }

  // Thumbnail returns an Image; with no size argument it caps the
  // longer side at the default thumbnail size (currently 150), and
  // with an integer caps the longer side at that value.
  #[test]
  fn thumbnail_default_caps_longer_side() {
    clear_state();
    assert_eq!(
      interpret("ImageDimensions[Thumbnail[Image[Table[0.5, {200}, {200}]]]]")
        .unwrap(),
      "{150, 150}"
    );
  }

  #[test]
  fn thumbnail_explicit_size_caps_longer_side() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[Thumbnail[Image[Table[0.5, {200}, {200}]], 50]]"
      )
      .unwrap(),
      "{50, 50}"
    );
  }

  // A larger source preserves the aspect ratio when downsized.
  #[test]
  fn thumbnail_preserves_aspect_ratio() {
    clear_state();
    // 400 wide, 300 tall → longer side 400 → scale 150/400 = 0.375 →
    // 150 × round(0.375 * 300) = 150 × 113.
    assert_eq!(
      interpret("ImageDimensions[Thumbnail[Image[Table[0.5, {300}, {400}]]]]")
        .unwrap(),
      "{150, 113}"
    );
  }

  // Channels and image type are preserved.
  #[test]
  fn thumbnail_preserves_channels_and_type() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageChannels[Thumbnail[Image[Table[{0.5, 0.3, 0.7}, {50}, {50}]], 20]]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("ImageType[Thumbnail[Image[Table[0.5, {50}, {50}]], 20]]")
        .unwrap(),
      "Real32"
    );
  }

  // Sharpen preserves image dimensions, channels, and image type
  // (no Byte round-trip through the image crate).
  #[test]
  fn sharpen_preserves_metadata() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[Sharpen[Image[{{0.1, 0.5, 0.9}, {0.3, 0.7, 0.5}}], 1]]"
      )
      .unwrap(),
      "{3, 2}"
    );
    assert_eq!(
      interpret(
        "ImageChannels[Sharpen[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}], 1]]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("ImageType[Sharpen[Image[{{0.1, 0.5}}], 1]]").unwrap(),
      "Real32"
    );
  }

  // Radius 0 is the identity — the image is unchanged.
  #[test]
  fn sharpen_radius_zero_is_identity() {
    clear_state();
    assert_eq!(
      interpret("ImageData[Sharpen[Image[{{0.1, 0.5, 0.9}}], 0]]").unwrap(),
      "{{0.10000000149011612, 0.5, 0.8999999761581421}}"
    );
  }

  // A symmetric grayscale image leaves the center pixel anchored at
  // its original value (the unsharp-mask correction is zero there for
  // symmetric data).
  #[test]
  fn sharpen_center_pixel_invariant_for_symmetric_input() {
    clear_state();
    let out =
      interpret("ImageData[Sharpen[Image[{{0.1, 0.5, 0.9}}], 1]]").unwrap();
    let middle = out
      .trim_start_matches("{{")
      .trim_end_matches("}}")
      .split(", ")
      .nth(1)
      .unwrap();
    let v: f64 = middle.parse().unwrap();
    assert!((v - 0.5).abs() < 1e-6, "expected ~0.5, got {}", v);
  }

  // MinFilter / MaxFilter on Images apply a min/max kernel per channel,
  // with the (2r+1)×(2r+1) window clipped at image boundaries.
  #[test]
  fn min_filter_image_grayscale() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[MinFilter[Image[{{0.5, 0.2, 0.9}, {0.1, 0.6, 0.3}}], 1]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.10000000149011612, 0.20000000298023224}, \
       {0.10000000149011612, 0.10000000149011612, 0.20000000298023224}}"
    );
  }

  #[test]
  fn max_filter_image_grayscale() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[MaxFilter[Image[{{0.5, 0.2, 0.9}, {0.1, 0.6, 0.3}}], 1]]"
      )
      .unwrap(),
      "{{0.6000000238418579, 0.8999999761581421, 0.8999999761581421}, \
       {0.6000000238418579, 0.8999999761581421, 0.8999999761581421}}"
    );
  }

  // Channels and type pass through unchanged.
  #[test]
  fn min_filter_image_preserves_channels_and_type() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageChannels[MinFilter[Image[{{{1.0, 0.0, 0.5}, {0.0, 1.0, 0.5}}}], 1]]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("ImageType[MinFilter[Image[{{0.1, 0.5}}], 1]]").unwrap(),
      "Real32"
    );
  }

  // ImageResize accepts {Automatic, h} / {w, Automatic} to preserve
  // the aspect ratio on the auto-side. The audit case has Woxi
  // erroring on Automatic; this verifies the fix.
  #[test]
  fn image_resize_automatic_width() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[ImageResize[Image[{{0.0, 0.5, 1.0}}], {Automatic, 3}]]"
      )
      .unwrap(),
      "{9, 3}"
    );
  }

  #[test]
  fn image_resize_automatic_height() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[ImageResize[Image[{{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}], {9, Automatic}]]"
      )
      .unwrap(),
      "{9, 6}"
    );
  }

  // {w} (one-element list) caps the longer side at w, preserving aspect.
  #[test]
  fn image_resize_max_side() {
    clear_state();
    assert_eq!(
      interpret("ImageDimensions[ImageResize[Image[{{0.0, 0.5, 1.0}}], {6}]]")
        .unwrap(),
      "{6, 2}"
    );
  }

  // ImageType / ImageChannels survive a resize.
  #[test]
  fn image_resize_preserves_channels_and_type() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageChannels[ImageResize[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}], {4, 2}]]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("ImageType[ImageResize[Image[{{0.1, 0.5}}], {4, 2}]]").unwrap(),
      "Real32"
    );
  }

  // ImageCrop[image] with no size argument removes uniform borders.
  // Pixel precision is preserved (no Byte round-trip).
  #[test]
  fn image_crop_no_uniform_border_preserves_pixels() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageCrop[Image[\
         {{0.1, 0.2, 0.3}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.0}}]]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.20000000298023224, 0.30000001192092896}, \
       {0., 0.5, 0.}, \
       {0., 0., 0.}}"
    );
  }

  // Auto-crop trims a uniform border around the interior content.
  #[test]
  fn image_crop_trims_uniform_border() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[ImageCrop[Image[\
         {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.0}}]]]"
      )
      .unwrap(),
      "{1, 1}"
    );
    assert_eq!(
      interpret(
        "ImageData[ImageCrop[Image[\
         {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.0}}]]]"
      )
      .unwrap(),
      "{{0.5}}"
    );
  }

  // A larger central region gets cropped down to just the non-border
  // bounding box.
  #[test]
  fn image_crop_finds_interior_bounding_box() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[ImageCrop[Image[\
         {{0.0, 0.0, 0.0, 0.0}, \
         {0.0, 0.5, 0.7, 0.0}, \
         {0.0, 0.3, 0.4, 0.0}, \
         {0.0, 0.0, 0.0, 0.0}}]]]"
      )
      .unwrap(),
      "{2, 2}"
    );
  }

  // ImageApply on an RGB image with a function that returns a scalar
  // collapses each pixel's channel list to that scalar — the result
  // is a 1-channel (grayscale) image.
  #[test]
  fn image_apply_max_rgb_returns_grayscale() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageApply[Max, Image[{{{0.1, 0.5, 0.2}, {0.8, 0.3, 0.9}}}]]]"
      )
      .unwrap(),
      "{{0.5, 0.8999999761581421}}"
    );
    assert_eq!(
      interpret("ImageChannels[ImageApply[Max, Image[{{{0.1, 0.5, 0.2}}}]]]")
        .unwrap(),
      "1"
    );
  }

  // Min has the same channel-collapsing behaviour.
  #[test]
  fn image_apply_min_rgb_returns_grayscale() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageApply[Min, Image[{{{0.1, 0.5, 0.2}, {0.8, 0.3, 0.9}}}]]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.30000001192092896}}"
    );
  }

  // Grayscale passes the scalar pixel directly to f.
  #[test]
  fn image_apply_function_to_grayscale() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageApply[#^2 &, Image[{{0.5, 0.8}}]]]").unwrap(),
      "{{0.25, 0.64000004529953}}"
    );
  }

  // When f returns a list, the result keeps that channel count.
  #[test]
  fn image_apply_reverse_keeps_three_channels() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageApply[Reverse, Image[{{{0.1, 0.5, 0.2}}}]]]")
        .unwrap(),
      "{{{0.20000000298023224, 0.5, 0.10000000149011612}}}"
    );
  }

  // ImageCollage on same-shape images lays them out as a near-square
  // grid without resizing. The matching ws layouts for n=2 and n=4
  // happen to land on cols = ceil(sqrt(n)), rows = ceil(n/cols).
  #[test]
  fn image_collage_two_images_is_one_row() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageCollage[{Image[{{0.1, 0.2}}], Image[{{0.3, 0.4}}]}]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.20000000298023224, \
        0.30000001192092896, 0.4000000059604645}}"
    );
    assert_eq!(
      interpret(
        "ImageDimensions[ImageCollage[\
         {Image[{{0.1, 0.2}}], Image[{{0.3, 0.4}}]}]]"
      )
      .unwrap(),
      "{4, 1}"
    );
  }

  // Four images of the same shape pack into a 2x2 grid.
  #[test]
  fn image_collage_four_images_is_two_by_two() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageCollage[\
         {Image[{{0.1, 0.2}}], Image[{{0.3, 0.4}}], \
         Image[{{0.5, 0.6}}], Image[{{0.7, 0.8}}]}]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.20000000298023224, \
        0.30000001192092896, 0.4000000059604645}, \
       {0.5, 0.6000000238418579, \
        0.699999988079071, 0.800000011920929}}"
    );
  }

  // Single image: the result is the image itself (dimensions preserved).
  #[test]
  fn image_collage_single_image() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[ImageCollage[{Image[{{0.1, 0.2}, {0.3, 0.4}}]}]]"
      )
      .unwrap(),
      "{2, 2}"
    );
  }

  // Channels and image type are preserved.
  #[test]
  fn image_collage_preserves_channels_and_type() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageChannels[ImageCollage[\
         {Image[{{0.1, 0.2}}], Image[{{0.3, 0.4}}]}]]"
      )
      .unwrap(),
      "1"
    );
    assert_eq!(
      interpret(
        "ImageType[ImageCollage[\
         {Image[{{0.1, 0.2}}], Image[{{0.3, 0.4}}]}]]"
      )
      .unwrap(),
      "Real32"
    );
  }

  // ImageAssemble on a grid of same-shape images concatenates them
  // without resizing and preserves pixel precision (no Byte
  // round-trip).
  #[test]
  fn image_assemble_grid_preserves_pixels() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageAssemble[\
         {{Image[{{0.1, 0.2}}], Image[{{0.3, 0.4}}]}, \
         {Image[{{0.5, 0.6}}], Image[{{0.7, 0.8}}]}}]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.20000000298023224, \
        0.30000001192092896, 0.4000000059604645}, \
       {0.5, 0.6000000238418579, \
        0.699999988079071, 0.800000011920929}}"
    );
  }

  // Flat-list form assembles into a single row.
  #[test]
  fn image_assemble_flat_row() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageAssemble[{Image[{{0.1, 0.2}}], Image[{{0.3, 0.4}}]}]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.20000000298023224, \
        0.30000001192092896, 0.4000000059604645}}"
    );
  }

  // The output preserves channel count and image type for matching inputs.
  #[test]
  fn image_assemble_preserves_channels_and_type() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageChannels[ImageAssemble[\
         {{Image[{{0.1, 0.2}}], Image[{{0.3, 0.4}}]}}]]"
      )
      .unwrap(),
      "1"
    );
    assert_eq!(
      interpret(
        "ImageType[ImageAssemble[\
         {{Image[{{0.1, 0.2}}], Image[{{0.3, 0.4}}]}}]]"
      )
      .unwrap(),
      "Real32"
    );
  }

  // ImageDimensions stays additive across cells (no resampling).
  #[test]
  fn image_assemble_dimensions_are_additive() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[ImageAssemble[\
         {{Image[{{0.1, 0.2}}], Image[{{0.3, 0.4}}]}, \
         {Image[{{0.5, 0.6}}], Image[{{0.7, 0.8}}]}}]]"
      )
      .unwrap(),
      "{4, 2}"
    );
  }

  // ImageCompose on same-size images replaces the background with the
  // overlay (no blending unless explicit). The result preserves the
  // background's channel count.
  #[test]
  fn image_compose_same_size_replaces() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageCompose[Image[{{0.1, 0.2}, {0.3, 0.4}}], \
         Image[{{0.9, 0.8}, {0.7, 0.6}}]]]"
      )
      .unwrap(),
      "{{0.8999999761581421, 0.800000011920929}, \
       {0.699999988079071, 0.6000000238418579}}"
    );
  }

  // Smaller overlay is centered on the background; non-overlapped
  // pixels keep their original value.
  #[test]
  fn image_compose_smaller_overlay_is_centered() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageCompose[Image[{{0.1, 0.2, 0.3, 0.4}, \
         {0.5, 0.6, 0.7, 0.8}}], Image[{{0.9}}]]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.20000000298023224, 0.8999999761581421, \
       0.4000000059604645}, \
       {0.5, 0.6000000238418579, 0.699999988079071, 0.800000011920929}}"
    );
  }

  // ImageCompose[bg, {overlay, α}] alpha-blends: (1-α)*bg + α*overlay
  // for the overlapping region.
  #[test]
  fn image_compose_alpha_blend() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageCompose[Image[{{0.1, 0.2}, {0.3, 0.4}}], \
         {Image[{{0.9, 0.8}, {0.7, 0.6}}], 0.5}]]"
      )
      .unwrap(),
      "{{0.5, 0.5}, {0.5, 0.5}}"
    );
  }

  // Output dimensions and channel count match the background.
  #[test]
  fn image_compose_preserves_background_shape() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageDimensions[ImageCompose[Image[{{0.1, 0.2, 0.3, 0.4}, \
         {0.5, 0.6, 0.7, 0.8}}], Image[{{0.9}}]]]"
      )
      .unwrap(),
      "{4, 2}"
    );
    assert_eq!(
      interpret(
        "ImageChannels[ImageCompose[Image[{{0.1, 0.2}, {0.3, 0.4}}], \
         Image[{{0.9, 0.8}, {0.7, 0.6}}]]]"
      )
      .unwrap(),
      "1"
    );
  }

  // ImageAdd accepts a variadic list of extra terms: each is added in
  // turn (scalar or image), threading through the partial sums.
  #[test]
  fn image_add_three_args() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageAdd[Image[{{0.1, 0.5}}], 0.2, 0.1]]").unwrap(),
      "{{0.4000000059604645, 0.800000011920929}}"
    );
  }

  #[test]
  fn image_add_image_and_scalar() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageAdd[Image[{{0.1, 0.5}}], Image[{{0.4, 0.3}}], 0.1]]"
      )
      .unwrap(),
      "{{0.6000000238418579, 0.9000000357627869}}"
    );
  }

  // ImageSubtract chains: image - 0.1 - 0.05
  #[test]
  fn image_subtract_three_args() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageSubtract[Image[{{0.5, 0.8}}], 0.1, 0.05]]")
        .unwrap(),
      "{{0.3499999940395355, 0.6499999761581421}}"
    );
  }

  // ImageMultiply chains: image * 0.5 * 0.5 = image * 0.25
  #[test]
  fn image_multiply_three_args() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageMultiply[Image[{{1.0, 0.5}}], 0.5, 0.5]]")
        .unwrap(),
      "{{0.25, 0.125}}"
    );
  }

  // Blur preserves the channel count and image type (no quantisation
  // round-trip through the image crate's u8 buffer).
  #[test]
  fn blur_preserves_channels_and_type() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageChannels[Blur[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}], 1]]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("ImageType[Blur[Image[{{0.1, 0.5, 0.9}}], 1]]").unwrap(),
      "Real32"
    );
  }

  // Symmetry: the center of a symmetric grayscale input stays at its
  // central value after blurring.
  #[test]
  fn blur_preserves_symmetric_center() {
    clear_state();
    let out =
      interpret("ImageData[Blur[Image[{{0.1, 0.5, 0.9}}], 1]]").unwrap();
    // Output is { {p0, p1, p2} } — extract the middle pixel.
    let middle = out
      .trim_start_matches("{{")
      .trim_end_matches("}}")
      .split(", ")
      .nth(1)
      .unwrap();
    let v: f64 = middle.parse().unwrap();
    assert!((v - 0.5).abs() < 1e-6, "expected ~0.5, got {}", v);
  }

  // Edge pixels should not be unchanged (a uniform interior + edge gives
  // a non-trivial blur).
  #[test]
  fn blur_changes_edge_pixels() {
    clear_state();
    let out =
      interpret("ImageData[Blur[Image[{{0.0, 0.5, 1.0}}], 1]]").unwrap();
    let parts: Vec<f64> = out
      .trim_start_matches("{{")
      .trim_end_matches("}}")
      .split(", ")
      .map(|s| s.parse().unwrap())
      .collect();
    // Left edge should be strictly between 0.0 and 0.5.
    assert!(
      parts[0] > 0.0 && parts[0] < 0.5,
      "edge pixel out of expected range: {}",
      parts[0]
    );
    // Right edge symmetric.
    assert!(
      parts[2] > 0.5 && parts[2] < 1.0,
      "edge pixel out of expected range: {}",
      parts[2]
    );
  }

  // A zero-radius blur is the identity (no convolution applied).
  #[test]
  fn blur_radius_zero_is_identity() {
    clear_state();
    assert_eq!(
      interpret("ImageData[Blur[Image[{{0.1, 0.5, 0.9}}], 0]]").unwrap(),
      "{{0.10000000149011612, 0.5, 0.8999999761581421}}"
    );
  }

  #[test]
  fn sharpen_preserves_dimensions() {
    clear_state();
    let result = interpret(
      "ImageDimensions[Sharpen[Image[{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}]]]",
    )
    .unwrap();
    assert_eq!(result, "{3, 3}");
  }

  #[test]
  fn image_take_n_rows() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageTake[Image[{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {1, 1, 1}}], 2]]",
    )
    .unwrap();
    assert_eq!(result, "{3, 2}");
  }

  #[test]
  fn image_take_row_range() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageTake[Image[{{0, 0}, {1, 1}, {0, 0}, {1, 1}}], {2, 3}]]",
    )
    .unwrap();
    assert_eq!(result, "{2, 2}");
  }

  #[test]
  fn image_take_row_and_col_range() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageTake[Image[{{0, 0, 0, 0}, {0, 1, 1, 0}, {0, 0, 0, 0}}], {1, 2}, {2, 3}]]",
    )
    .unwrap();
    assert_eq!(result, "{2, 2}");
  }

  #[test]
  fn image_take_preserves_data() {
    clear_state();
    let result = interpret(
      "ImageData[ImageTake[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], 1]]",
    )
    .unwrap();
    // f32 precision values (matching wolframscript)
    assert_eq!(
      result,
      "{{0.10000000149011612, 0.20000000298023224, 0.30000001192092896}}"
    );
  }

  // Negative n means take the last |n| rows.
  #[test]
  fn image_take_negative_n_takes_last_rows() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageTake[Image[{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}], -2]]"
      )
      .unwrap(),
      "{{0.30000001192092896, 0.4000000059604645}, \
       {0.5, 0.6000000238418579}}"
    );
  }

  // {-n} is a single row indexed from the end.
  #[test]
  fn image_take_single_negative_row() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageTake[Image[{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}], {-2}]]"
      )
      .unwrap(),
      "{{0.30000001192092896, 0.4000000059604645}}"
    );
  }

  // {r1, r2} with negative endpoints counts from the end.
  #[test]
  fn image_take_negative_row_range() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageTake[Image[{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}], {-2, -1}]]"
      )
      .unwrap(),
      "{{0.30000001192092896, 0.4000000059604645}, \
       {0.5, 0.6000000238418579}}"
    );
  }

  // All selects every row/column in that axis.
  #[test]
  fn image_take_all_rows_and_col_range() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageTake[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], All, {1, 2}]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.20000000298023224}, \
       {0.4000000059604645, 0.5}}"
    );
  }

  // Column spec accepts integers (first/last n) just like the row spec.
  #[test]
  fn image_take_all_rows_negative_col_count() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageTake[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], All, -2]]"
      )
      .unwrap(),
      "{{0.20000000298023224, 0.30000001192092896}, \
       {0.5, 0.6000000238418579}}"
    );
  }

  #[test]
  fn image_take_all_rows_positive_col_count() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageTake[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}], All, 2]]"
      )
      .unwrap(),
      "{{0.10000000149011612, 0.20000000298023224}, \
       {0.4000000059604645, 0.5}}"
    );
  }

  // Integer column spec with integer row spec.
  #[test]
  fn image_take_int_row_int_col() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageTake[Image[{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}], 2, 1]]"
      )
      .unwrap(),
      "{{0.10000000149011612}, {0.30000001192092896}}"
    );
    assert_eq!(
      interpret(
        "ImageData[ImageTake[Image[{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}], 2, -1]]"
      )
      .unwrap(),
      "{{0.20000000298023224}, {0.4000000059604645}}"
    );
  }
}

mod image_advanced {
  use super::*;

  #[test]
  fn edge_detect_produces_grayscale() {
    clear_state();
    let result = interpret(
      "ImageChannels[EdgeDetect[Image[{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}]]]",
    )
    .unwrap();
    assert_eq!(result, "1");
  }

  #[test]
  fn edge_detect_preserves_dimensions() {
    clear_state();
    let result = interpret(
      "ImageDimensions[EdgeDetect[Image[{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}]]]",
    )
    .unwrap();
    assert_eq!(result, "{3, 3}");
  }

  #[test]
  fn edge_detect_returns_bit_type() {
    clear_state();
    let result = interpret(
      "ImageType[EdgeDetect[Image[{{0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 0, 0, 0, 0}}]]]",
    )
    .unwrap();
    assert_eq!(result, "Bit");
  }

  #[test]
  fn edge_detect_binary_output() {
    // All pixel values should be 0 or 1
    clear_state();
    let result = interpret(
      "Min[Flatten[ImageData[EdgeDetect[Image[{{0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 0, 0, 0, 0}}]]]]]",
    )
    .unwrap();
    assert_eq!(result, "0");
  }

  #[test]
  fn edge_detect_center_is_zero() {
    // The center of a uniform region should have no edge
    clear_state();
    let result = interpret(
      "ImageData[EdgeDetect[Image[{{0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 0, 0, 0, 0}}]]][[3, 3]]",
    )
    .unwrap();
    assert_eq!(result, "0");
  }

  #[test]
  fn edge_detect_inner_ring_detected() {
    // The 8 inner ring pixels around the center should all be edges
    clear_state();
    let result = interpret(
      "Total[Flatten[ImageData[EdgeDetect[Image[{{0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 0, 0, 0, 0}}]]]]]",
    )
    .unwrap();
    // Should detect at least 8 edge pixels (the inner ring)
    let total: f64 = result.parse().unwrap();
    assert!(
      total >= 8.0,
      "Expected at least 8 edge pixels, got {}",
      total
    );
  }

  #[test]
  fn edge_detect_uniform_is_zero() {
    // A completely uniform image should have no edges
    clear_state();
    let result = interpret(
      "Total[Flatten[ImageData[EdgeDetect[Image[{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}]]]]]",
    )
    .unwrap();
    assert_eq!(result, "0");
  }

  #[test]
  fn edge_detect_10x10_rectangle() {
    // Rectangular edges should be detected correctly.
    // Use multi-statement approach to avoid PEG parser slowdown with large inline matrices.
    clear_state();
    interpret(
      "img10 = Image[{{0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0},{0,0,1,1,1,1,1,1,0,0},{0,0,1,1,1,1,1,1,0,0},{0,0,1,1,1,1,1,1,0,0},{0,0,1,1,1,1,1,1,0,0},{0,0,1,1,1,1,1,1,0,0},{0,0,1,1,1,1,1,1,0,0},{0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0}}]",
    ).unwrap();
    let result = interpret("ImageData[EdgeDetect[img10]]").unwrap();
    // Interior should be 0 (no edge in uniform region)
    assert!(result.contains("{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}"));
    // Edge boundary rows should have some 1s
    let total: f64 = interpret("Total[Flatten[ImageData[EdgeDetect[img10]]]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!(
      total >= 16.0 && total <= 30.0,
      "Expected 16-30 edge pixels for rectangle, got {}",
      total
    );
  }

  #[test]
  fn edge_detect_with_radius() {
    // EdgeDetect[img, r] should accept a radius parameter
    clear_state();
    let result = interpret(
      "ImageType[EdgeDetect[Image[{{0,0,0,0,0},{0,1,1,1,0},{0,1,1,1,0},{0,1,1,1,0},{0,0,0,0,0}}], 1]]",
    )
    .unwrap();
    assert_eq!(result, "Bit");
  }

  #[test]
  fn edge_detect_with_radius_and_threshold() {
    // EdgeDetect[img, r, t] should accept radius and threshold
    clear_state();
    let result = interpret(
      "ImageType[EdgeDetect[Image[{{0,0,0,0,0},{0,1,1,1,0},{0,1,1,1,0},{0,1,1,1,0},{0,0,0,0,0}}], 1, 0.5]]",
    )
    .unwrap();
    assert_eq!(result, "Bit");
  }

  #[test]
  fn edge_detect_high_threshold_suppresses_all() {
    // Very high threshold should suppress all or nearly all edges
    clear_state();
    let result = interpret(
      "Total[Flatten[ImageData[EdgeDetect[Image[{{0,0,0,0,0},{0,1,1,1,0},{0,1,1,1,0},{0,1,1,1,0},{0,0,0,0,0}}], 2, 0.99]]]]",
    )
    .unwrap();
    let total: f64 = result.parse().unwrap();
    assert!(
      total <= 4.0,
      "High threshold should suppress most edges, got {}",
      total
    );
  }

  #[test]
  fn edge_detect_rgb_image() {
    // EdgeDetect on RGB image should produce single-channel output
    clear_state();
    let result = interpret(
      "ImageChannels[EdgeDetect[Image[{{{1,0,0},{0,0,0},{0,0,0}},{{0,0,0},{0,1,0},{0,0,0}},{{0,0,0},{0,0,0},{0,0,1}}}]]]",
    )
    .unwrap();
    assert_eq!(result, "1");
  }

  #[test]
  fn edge_detect_larger_radius_smoother() {
    // Larger radius should generally produce fewer edge pixels
    // (more smoothing = less detail).
    // Use multi-statement approach to avoid PEG parser slowdown with inline matrices.
    clear_state();
    interpret(
      "imgSmooth = Image[{{0,0,0,0,0},{0,1,1,1,0},{0,1,1,1,0},{0,1,1,1,0},{0,0,0,0,0}}]",
    ).unwrap();
    let r1: f64 =
      interpret("Total[Flatten[ImageData[EdgeDetect[imgSmooth, 1]]]]")
        .unwrap()
        .parse()
        .unwrap();
    let r3: f64 =
      interpret("Total[Flatten[ImageData[EdgeDetect[imgSmooth, 3]]]]")
        .unwrap()
        .parse()
        .unwrap();
    // Both should detect edges (non-zero)
    assert!(r1 > 0.0, "r=1 should detect some edges");
    assert!(r3 > 0.0, "r=3 should detect some edges");
  }

  #[test]
  fn color_convert_rgb_to_grayscale() {
    clear_state();
    let result = interpret(
      "ImageChannels[ColorConvert[Image[{{{1, 0, 0}, {0, 1, 0}}}], \"Grayscale\"]]",
    )
    .unwrap();
    assert_eq!(result, "1");
  }

  #[test]
  fn color_convert_grayscale_to_rgb() {
    clear_state();
    let result =
      interpret("ImageChannels[ColorConvert[Image[{{0.5, 1.0}}], \"RGB\"]]")
        .unwrap();
    assert_eq!(result, "3");
  }

  // ColorConvert on color directives: RGBColor → GrayLevel uses the
  // standard luminance weights (0.299 R + 0.587 G + 0.114 B).
  #[test]
  fn color_convert_rgbcolor_to_grayscale() {
    clear_state();
    assert_eq!(
      interpret("ColorConvert[RGBColor[1, 0, 0], \"Grayscale\"]").unwrap(),
      "GrayLevel[0.299]"
    );
    assert_eq!(
      interpret("ColorConvert[RGBColor[1.0, 0.5, 0.0], \"Grayscale\"]")
        .unwrap(),
      "GrayLevel[0.5925]"
    );
  }

  // ColorConvert from a named color: Red is RGBColor[1, 0, 0]; the
  // luminance evaluates to a Real.
  #[test]
  fn color_convert_red_to_grayscale() {
    clear_state();
    assert_eq!(
      interpret("ColorConvert[Red, \"Grayscale\"]").unwrap(),
      "GrayLevel[0.299]"
    );
  }

  // ColorConvert[RGBColor, "RGB"] forces components to Real.
  #[test]
  fn color_convert_rgbcolor_to_rgb_coerces_to_real() {
    clear_state();
    assert_eq!(
      interpret("ColorConvert[RGBColor[1, 0, 0], \"RGB\"]").unwrap(),
      "RGBColor[1., 0., 0.]"
    );
  }

  // GrayLevel → RGB broadcasts to all three channels.
  #[test]
  fn color_convert_graylevel_to_rgb() {
    clear_state();
    assert_eq!(
      interpret("ColorConvert[GrayLevel[0.5], \"RGB\"]").unwrap(),
      "RGBColor[0.5, 0.5, 0.5]"
    );
  }

  // GrayLevel → Grayscale is the identity (with Real coercion).
  #[test]
  fn color_convert_graylevel_identity() {
    clear_state();
    assert_eq!(
      interpret("ColorConvert[GrayLevel[0.5], \"Grayscale\"]").unwrap(),
      "GrayLevel[0.5]"
    );
  }

  // RGB <-> CMYK and RGB <-> HSB conversions on color directives.
  #[test]
  fn color_convert_cmyk_and_hsb() {
    clear_state();
    assert_eq!(
      interpret("ColorConvert[RGBColor[1, 0, 0], \"CMYK\"]").unwrap(),
      "CMYKColor[0., 1., 1., 0.]"
    );
    assert_eq!(
      interpret("ColorConvert[RGBColor[1, 0, 0], \"HSB\"]").unwrap(),
      "Hue[0., 1., 1.]"
    );
    // Hue and CMYKColor are accepted as inputs too.
    assert_eq!(
      interpret("ColorConvert[Hue[0.5], \"RGB\"]").unwrap(),
      "RGBColor[0., 1., 1.]"
    );
    assert_eq!(
      interpret("ColorConvert[CMYKColor[0, 1, 1, 0], \"RGB\"]").unwrap(),
      "RGBColor[1., 0., 0.]"
    );
    assert_eq!(
      interpret("ColorConvert[CMYKColor[0, 1, 1, 0], \"HSB\"]").unwrap(),
      "Hue[0., 1., 1.]"
    );
  }

  #[test]
  fn image_add() {
    clear_state();
    let result = interpret(
      "ImageData[ImageAdd[Image[{{0.25, 0.25}}], Image[{{0.25, 0.5}}]]]",
    )
    .unwrap();
    assert_eq!(result, "{{0.5, 0.75}}");
  }

  #[test]
  fn image_add_overflow() {
    clear_state();
    // Wolfram does NOT clamp ImageAdd results to [0,1]
    let result =
      interpret("ImageData[ImageAdd[Image[{{0.8}}], Image[{{0.5}}]]]").unwrap();
    assert_eq!(result, "{{1.2999999523162842}}");
  }

  #[test]
  fn image_subtract() {
    clear_state();
    let result = interpret(
      "ImageData[ImageSubtract[Image[{{0.5, 0.8}}], Image[{{0.2, 0.3}}]]]",
    )
    .unwrap();
    // f32 precision: 0.5-0.2 = 0.30000001192092896 in f32
    assert_eq!(result, "{{0.30000001192092896, 0.5}}");
  }

  #[test]
  fn image_subtract_negative() {
    clear_state();
    // Wolfram does NOT clamp ImageSubtract results to [0,1]
    let result =
      interpret("ImageData[ImageSubtract[Image[{{0.2}}], Image[{{0.5}}]]]")
        .unwrap();
    assert_eq!(result, "{{-0.30000001192092896}}");
  }

  // For Real32 images, arithmetic must be done in f32 — not f64-then-cast.
  // 0.6 - 0.2 = 0.4 in f64, but 0.6f32 - 0.2f32 ≈ 0.40000003576278687
  // (one ulp above 0.4f32); wolframscript reports the latter.
  #[test]
  fn image_subtract_uses_f32_arithmetic() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ImageSubtract[Image[{{0.5, 0.6}, {0.7, 0.8}}], 0.2]]"
      )
      .unwrap(),
      "{{0.30000001192092896, 0.40000003576278687}, \
       {0.5, 0.6000000238418579}}"
    );
  }

  // Same precision rule for the (Image, Image) form.
  #[test]
  fn image_subtract_image_image_uses_f32() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ImageSubtract[Image[{{0.6}}], Image[{{0.2}}]]]")
        .unwrap(),
      "{{0.40000003576278687}}"
    );
  }

  // ImageAdd and ImageMultiply share the precision path.
  #[test]
  fn image_add_uses_f32_arithmetic() {
    clear_state();
    // 0.1 + 0.2 = 0.30000001 in f32 (vs 0.30000000000000004 in f64).
    assert_eq!(
      interpret("ImageData[ImageAdd[Image[{{0.1}}], 0.2]]").unwrap(),
      "{{0.30000001192092896}}"
    );
  }

  #[test]
  fn image_multiply_uses_f32_arithmetic() {
    clear_state();
    // 0.1 * 3 = 0.30000001 in f32 (vs 0.30000000000000004 in f64).
    assert_eq!(
      interpret("ImageData[ImageMultiply[Image[{{0.1}}], 3]]").unwrap(),
      "{{0.30000001192092896}}"
    );
  }

  #[test]
  fn image_multiply() {
    clear_state();
    let result = interpret(
      "ImageData[ImageMultiply[Image[{{0.5, 1.0}}], Image[{{0.5, 0.5}}]]]",
    )
    .unwrap();
    assert_eq!(result, "{{0.25, 0.5}}");
  }

  #[test]
  fn image_compose_dimensions() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageCompose[Image[{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}], Image[{{1}}]]]",
    )
    .unwrap();
    assert_eq!(result, "{3, 3}");
  }

  #[test]
  fn random_image_default() {
    clear_state();
    let result = interpret("ImageDimensions[RandomImage[]]").unwrap();
    assert_eq!(result, "{150, 150}");
  }

  #[test]
  fn random_image_default_type() {
    clear_state();
    let result = interpret("ImageType[RandomImage[]]").unwrap();
    assert_eq!(result, "Real32");
  }

  #[test]
  fn random_image_with_size() {
    clear_state();
    let result = interpret("ImageDimensions[RandomImage[1, {10, 5}]]").unwrap();
    assert_eq!(result, "{10, 5}");
  }

  #[test]
  fn random_image_channels() {
    clear_state();
    let result = interpret("ImageChannels[RandomImage[1, {10, 5}]]").unwrap();
    assert_eq!(result, "1");
  }

  #[test]
  fn dominant_colors_returns_list() {
    clear_state();
    let result = interpret(
      "Length[DominantColors[Image[{{{1, 0, 0}, {1, 0, 0}, {0, 0, 1}}}], 2]]",
    )
    .unwrap();
    assert_eq!(result, "2");
  }

  // Grayscale (single-channel) input: wolframscript reports the dominant
  // shades as `GrayLevel[v]`. The exact centre values come from f64
  // averaging so we tolerate the standard rounding drift.
  #[test]
  fn dominant_colors_grayscale_two() {
    clear_state();
    assert_eq!(
      interpret("DominantColors[Image[{{0.1, 0.2, 0.15, 0.8, 0.9}}], 2]")
        .unwrap(),
      "{GrayLevel[0.15000000000000002], GrayLevel[0.8500000000000001]}"
    );
  }

  #[test]
  fn dominant_colors_grayscale_returns_gray_level_head() {
    // Audit regression: a single-channel image should produce GrayLevel
    // colors, not an error. Check the result is a List of length ≥ 1
    // whose first element has head GrayLevel.
    clear_state();
    let result = interpret(
      "Module[{dc = DominantColors[Image[{{0.1, 0.2, 0.1}, {0.8, 0.9, 0.85}}], 2]}, {Head[dc], Head[dc[[1]]]}]",
    )
    .unwrap();
    assert_eq!(result, "{List, GrayLevel}");
  }

  #[test]
  fn head_of_image() {
    clear_state();
    let result = interpret("Head[Image[{{0, 1}}]]").unwrap();
    assert_eq!(result, "Image");
  }
}

mod image_io {
  use super::*;

  #[test]
  fn import_jpeg() {
    clear_state();
    let result = interpret("ImageQ[Import[\"images/parrot.jpeg\"]]").unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn import_jpeg_dimensions() {
    clear_state();
    let result =
      interpret("ImageDimensions[Import[\"images/parrot.jpeg\"]]").unwrap();
    // Check that dimensions are non-zero
    assert!(result.starts_with("{"));
    assert!(result.contains(", "));
    assert!(result.ends_with("}"));
  }

  #[test]
  fn import_color_negate() {
    clear_state();
    let result =
      interpret("ImageQ[ColorNegate[Import[\"images/parrot.jpeg\"]]]").unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn export_image() {
    clear_state();
    let result = interpret(
      "Export[\"/tmp/woxi_test_export.png\", Image[{{0, 0.5, 1}, {1, 0.5, 0}}]]",
    )
    .unwrap();
    assert_eq!(result, "/tmp/woxi_test_export.png");
    // Verify file was created
    assert!(std::path::Path::new("/tmp/woxi_test_export.png").exists());
    // Clean up
    std::fs::remove_file("/tmp/woxi_test_export.png").ok();
  }

  #[test]
  fn import_local_jpeg() {
    clear_state();
    let result =
      interpret(r#"ImageQ[Import["tests/cli/images/logo.jpeg"]]"#).unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn import_local_dimensions() {
    clear_state();
    let result =
      interpret(r#"ImageDimensions[Import["tests/cli/images/logo.jpeg"]]"#)
        .unwrap();
    assert_eq!(result, "{887, 265}");
  }

  #[test]
  fn export_and_reimport() {
    clear_state();
    let _ = interpret(
      "Export[\"/tmp/woxi_test_roundtrip.png\", Image[{{0, 0.5, 1}}]]",
    )
    .unwrap();
    let result =
      interpret("ImageDimensions[Import[\"/tmp/woxi_test_roundtrip.png\"]]")
        .unwrap();
    assert_eq!(result, "{3, 1}");
    // Clean up
    std::fs::remove_file("/tmp/woxi_test_roundtrip.png").ok();
  }
}

mod image_display {
  use super::*;

  #[test]
  fn image_produces_graphics_output() {
    clear_state();
    let result = interpret_with_stdout("Image[{{0, 0.5, 1}}]").unwrap();
    assert_eq!(result.result, "-Image-");
    assert!(result.graphics.is_some());
    let html = result.graphics.unwrap();
    assert!(html.contains("<image href='data:image/png;base64,"));
  }

  #[test]
  fn image_text_result_is_placeholder() {
    clear_state();
    let result = interpret("Image[{{0, 0.5, 1}}]").unwrap();
    assert_eq!(result, "-Image-");
  }

  #[test]
  fn image_input_form_grayscale() {
    clear_state();
    assert_eq!(
      interpret("ToString[(Image[{{0, 0.5, 1}, {1, 0.5, 0}}]), InputForm]")
        .unwrap(),
      "Image[NumericArray[{{0., 0.5, 1.}, {1., 0.5, 0.}}, \"Real32\"], \"Real32\", ColorSpace -> Automatic, Interleaving -> None]"
    );
  }

  #[test]
  fn image_input_form_rgb() {
    clear_state();
    assert_eq!(
      interpret("ToString[(Image[{{{1, 0, 0}, {0, 1, 0}}, {{0, 0, 1}, {1, 1, 0}}}]), InputForm]").unwrap(),
      "Image[NumericArray[{{{1., 0., 0.}, {0., 1., 0.}}, {{0., 0., 1.}, {1., 1., 0.}}}, \"Real32\"], \"Real32\", ColorSpace -> Automatic, Interleaving -> True]"
    );
  }
}

mod image_collage {
  use super::*;

  #[test]
  fn collage_basic_returns_image() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{1, 0}, {0, 1}}]; img2 = Image[{{0, 1}, {1, 0}}]; ImageQ[ImageCollage[{img1, img2}]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn collage_single_image() {
    clear_state();
    assert_eq!(
      interpret(
        "img = Image[{{1, 0, 0}, {0, 1, 0}}]; ImageQ[ImageCollage[{img}]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn collage_weighted_images() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{1, 0}, {0, 1}}]; img2 = Image[{{0, 1}, {1, 0}}]; ImageQ[ImageCollage[{3*img1, img2}]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn collage_paired_format() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{1, 0}, {0, 1}}]; img2 = Image[{{0, 1}, {1, 0}}]; ImageQ[ImageCollage[{{img1, 2}, {img2, 1}}]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn collage_explicit_size() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{1, 0}, {0, 1}}]; img2 = Image[{{0, 1}, {1, 0}}]; ImageDimensions[ImageCollage[{img1, img2}, \"Fit\", {100, 80}]]"
      )
      .unwrap(),
      "{100, 80}"
    );
  }

  #[test]
  fn collage_with_fitting_and_size() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{1, 0}, {0, 1}}]; img2 = Image[{{0, 1}, {1, 0}}]; ImageDimensions[ImageCollage[{img1, img2}, \"Fit\", {200, 150}]]"
      )
      .unwrap(),
      "{200, 150}"
    );
  }

  #[test]
  fn collage_stretch_fitting() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{1, 0}, {0, 1}}]; img2 = Image[{{0, 1}, {1, 0}}]; ImageDimensions[ImageCollage[{img1, img2}, \"Stretch\", {120, 90}]]"
      )
      .unwrap(),
      "{120, 90}"
    );
  }

  #[test]
  fn collage_fill_fitting() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{1, 0}, {0, 1}}]; img2 = Image[{{0, 1}, {1, 0}}]; ImageDimensions[ImageCollage[{img1, img2}, \"Fill\", {120, 90}]]"
      )
      .unwrap(),
      "{120, 90}"
    );
  }

  #[test]
  fn collage_rgb_images() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{{1,0,0},{0,1,0}},{{0,0,1},{1,1,0}}}]; img2 = Image[{{{0,1,1},{1,0,1}},{{1,1,1},{0,0,0}}}]; ImageQ[ImageCollage[{img1, img2}]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn collage_many_images() {
    clear_state();
    assert_eq!(
      interpret(
        "imgs = Table[Image[{{i/5, 0}, {0, i/5}}], {i, 5}]; ImageQ[ImageCollage[imgs]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn collage_preserves_channel_count() {
    clear_state();
    // Grayscale inputs → grayscale output
    assert_eq!(
      interpret(
        "img1 = Image[{{1, 0}, {0, 1}}]; img2 = Image[{{0, 1}, {1, 0}}]; ImageChannels[ImageCollage[{img1, img2}]]"
      )
      .unwrap(),
      "1"
    );
  }

  #[test]
  fn collage_width_only_size() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{1, 0}, {0, 1}}]; img2 = Image[{{0, 1}, {1, 0}}]; dims = ImageDimensions[ImageCollage[{img1, img2}, \"Fit\", 200]]; dims[[1]]"
      )
      .unwrap(),
      "200"
    );
  }
}

mod image_assemble {
  use super::*;

  #[test]
  fn assemble_flat_list_single_row() {
    clear_state();
    // Two 2x2 images side by side → 4x2
    assert_eq!(
      interpret(
        "img1 = Image[{{1, 0}, {0, 1}}]; img2 = Image[{{0, 1}, {1, 0}}]; ImageDimensions[ImageAssemble[{img1, img2}]]"
      )
      .unwrap(),
      "{4, 2}"
    );
  }

  #[test]
  fn assemble_2x2_grid() {
    clear_state();
    // 2x2 grid of 2x2 images → 4x4
    assert_eq!(
      interpret(
        "a = Image[{{1,0},{0,1}}]; b = Image[{{0,1},{1,0}}]; c = Image[{{0.5,0.5},{0.5,0.5}}]; d = Image[{{0,0},{1,1}}]; ImageDimensions[ImageAssemble[{{a,b},{c,d}}]]"
      )
      .unwrap(),
      "{4, 4}"
    );
  }

  #[test]
  fn assemble_returns_image() {
    clear_state();
    assert_eq!(
      interpret(
        "img = Image[{{1, 0}, {0, 1}}]; ImageQ[ImageAssemble[{img, img}]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn assemble_single_image() {
    clear_state();
    assert_eq!(
      interpret(
        "img = Image[{{1, 0, 0}, {0, 1, 0}}]; ImageDimensions[ImageAssemble[{img}]]"
      )
      .unwrap(),
      "{3, 2}"
    );
  }

  #[test]
  fn assemble_mismatched_heights_unevaluated() {
    clear_state();
    // Without fitting, mismatched heights return unevaluated
    assert_eq!(
      interpret(
        "img1 = Image[{{1,0,0},{0,1,0},{0,0,1}}]; img2 = Image[{{0,1},{1,0}}]; ImageQ[ImageAssemble[{img1, img2}]]"
      )
      .unwrap(),
      "False"
    );
  }

  #[test]
  fn assemble_mismatched_with_fit() {
    clear_state();
    // With "Fit", mismatched sizes are handled
    assert_eq!(
      interpret(
        "img1 = Image[{{1,0,0},{0,1,0},{0,0,1}}]; img2 = Image[{{0,1},{1,0}}]; ImageDimensions[ImageAssemble[{img1, img2}, \"Fit\"]]"
      )
      .unwrap(),
      "{5, 3}"
    );
  }

  #[test]
  fn assemble_mismatched_with_stretch() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{1,0,0},{0,1,0},{0,0,1}}]; img2 = Image[{{0,1},{1,0}}]; ImageDimensions[ImageAssemble[{img1, img2}, \"Stretch\"]]"
      )
      .unwrap(),
      "{5, 3}"
    );
  }

  #[test]
  fn assemble_mismatched_with_fill() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{1,0,0},{0,1,0},{0,0,1}}]; img2 = Image[{{0,1},{1,0}}]; ImageDimensions[ImageAssemble[{img1, img2}, \"Fill\"]]"
      )
      .unwrap(),
      "{5, 3}"
    );
  }

  #[test]
  fn assemble_rgb_images() {
    clear_state();
    assert_eq!(
      interpret(
        "img1 = Image[{{{1,0,0},{0,1,0}},{{0,0,1},{1,1,0}}}]; img2 = Image[{{{0,1,1},{1,0,1}},{{1,1,1},{0,0,0}}}]; ImageDimensions[ImageAssemble[{img1, img2}]]"
      )
      .unwrap(),
      "{4, 2}"
    );
  }

  #[test]
  fn assemble_3x1_grid() {
    clear_state();
    // Three images in a column (3 rows, 1 col)
    assert_eq!(
      interpret(
        "a = Image[{{1,0},{0,1}}]; b = Image[{{0,1},{1,0}}]; c = Image[{{0.5,0.5},{0.5,0.5}}]; ImageDimensions[ImageAssemble[{{a},{b},{c}}]]"
      )
      .unwrap(),
      "{2, 6}"
    );
  }

  #[test]
  fn assemble_1x3_grid() {
    clear_state();
    // Single row with 3 images
    assert_eq!(
      interpret(
        "a = Image[{{1,0},{0,1}}]; b = Image[{{0,1},{1,0}}]; c = Image[{{0.5,0.5},{0.5,0.5}}]; ImageDimensions[ImageAssemble[{{a, b, c}}]]"
      )
      .unwrap(),
      "{6, 2}"
    );
  }

  #[test]
  fn assemble_preserves_channel_count() {
    clear_state();
    // Grayscale inputs → grayscale output
    assert_eq!(
      interpret(
        "img = Image[{{1, 0}, {0, 1}}]; ImageChannels[ImageAssemble[{img, img}]]"
      )
      .unwrap(),
      "1"
    );
  }

  #[test]
  fn assemble_missing_cell() {
    clear_state();
    // Missing[] in grid should produce background (still works)
    assert_eq!(
      interpret(
        "img = Image[{{1, 0}, {0, 1}}]; ImageDimensions[ImageAssemble[{{img, img}, {img, Missing[]}}, \"Fit\"]]"
      )
      .unwrap(),
      "{4, 4}"
    );
  }
}

mod constant_image {
  use super::*;

  #[test]
  fn grayscale_dimensions() {
    clear_state();
    assert_eq!(
      interpret("ImageDimensions[ConstantImage[0.5, {10, 20}]]").unwrap(),
      "{10, 20}"
    );
  }

  #[test]
  fn grayscale_channels() {
    clear_state();
    assert_eq!(
      interpret("ImageChannels[ConstantImage[0.5, {5, 5}]]").unwrap(),
      "1"
    );
  }

  #[test]
  fn grayscale_type() {
    clear_state();
    assert_eq!(
      interpret("ImageType[ConstantImage[0.5, {5, 5}]]").unwrap(),
      "Real32"
    );
  }

  #[test]
  fn named_color_rgb() {
    clear_state();
    assert_eq!(
      interpret("ImageChannels[ConstantImage[Red, {5, 5}]]").unwrap(),
      "3"
    );
  }

  #[test]
  fn rgb_color_channels() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageChannels[ConstantImage[RGBColor[0.5, 0.3, 0.1], {5, 5}]]"
      )
      .unwrap(),
      "3"
    );
  }

  #[test]
  fn default_size() {
    clear_state();
    assert_eq!(
      interpret("ImageDimensions[ConstantImage[0.5]]").unwrap(),
      "{150, 150}"
    );
  }

  #[test]
  fn image_q() {
    clear_state();
    assert_eq!(
      interpret("ImageQ[ConstantImage[0.5, {5, 5}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn red_image_data() {
    clear_state();
    assert_eq!(
      interpret("ImageData[ConstantImage[Red, {2, 1}]]").unwrap(),
      "{{{1., 0., 0.}, {1., 0., 0.}}}"
    );
  }

  #[test]
  fn dimensions_image_data_grayscale() {
    clear_state();
    assert_eq!(
      interpret("Dimensions[ImageData[Image[{{0, 0.5, 1}, {0.2, 0.4, 0.6}}]]]")
        .unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn dimensions_image_data_rgb() {
    clear_state();
    assert_eq!(
      interpret(
        "Dimensions[ImageData[Image[{{{1, 0, 0}, {0, 1, 0}}, {{0, 0, 1}, {1, 1, 0}}}]]]"
      )
      .unwrap(),
      "{2, 2, 3}"
    );
  }

  #[test]
  fn dimensions_image_directly() {
    clear_state();
    // Dimensions returns {} for Image objects — use ImageDimensions instead
    assert_eq!(
      interpret("Dimensions[Image[{{0, 0.5, 1}, {0.2, 0.4, 0.6}}]]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn dimensions_rgb_image_directly() {
    clear_state();
    assert_eq!(
      interpret(
        "Dimensions[Image[{{{1, 0, 0}, {0, 1, 0}}, {{0, 0, 1}, {1, 1, 0}}}]]"
      )
      .unwrap(),
      "{}"
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn random_image() {
    assert_case(
      r#"RandomImage[1, {100, 100}]"#,
      r#"Image[NumericArray[{{0.8504230380058289, 0.12020712345838547, 0.2001333087682724, 0.13625968992710114, 0.22991353273391724, 0.8094552755355835, 0.630945086479187, 0.45021140575408936, 0.046303439885377884, 0.21750669181346893, 0.6550355553627014, 0.6750809550285339, 0.05482448637485504, 0.9678640961647034, 0.7680334448814392, 0.5652081966400146, 0.0876629576086998, 0.013387036509811878, 0.009566417895257473, 0.7567000985145569, 0.5848711729049683, 0.5453628301620483, 0.49688491225242615, 0.881269097328186, 0.041187386959791183, 0.27829426527023315, 0.003098408691585064, 0.8509334325790405, 0.7409918904304504, 0.8034991025924683, 0.9513359069824219, 0.47878238558769226, 0.2630794644355774, 0.5559860467910767, 0.9727618098258972, 0.2005535364151001, 0.9447923302650452, 0.13690201938152313, 0.7264322638511658, 0.4410298466682434, 0.7586790919303894, 0.5235280990600586, 0.9526119232177734, 0.3692297339439392, 0.9476917386054993, 0.8195067644119263, 0.8640320301055908, 0.08339709788560867, 0.8497284650802612, 0.7446700930595398, 0.8906925320625305, 0.8050035834312439, 0.8524354696273804, 0.6810884475708008, 0.448053240776062, 0.8000140190124512, 0.3045911490917206, 0.3815711438655853, 0.7177073359489441, 0.6439903974533081, 0.24866555631160736, 0.8159181475639343, 0.5097633004188538, 0.575752854347229, 0.7792307734489441, 0.2881341874599457, 0.22367890179157257, 0.9723765850067139, 0.13180676102638245, 0.7629334926605225, 0.5182669162750244, 0.24940919876098633, 0.920589804649353, 0.0953938215970993, 0.15543662011623383, 0.6387333869934082, 0.3784691095352173, 0.13278774917125702, 0.7587009072303772, 0.5609787702560425, 0.9260031580924988, 0.5354996919631958, 0.9821401834487915, 0.22016966342926025, 0.8138998746871948, 0.7694128751754761, 0.6813184022903442, 0.8111280202865601, 0.4491408169269562, 0.2648775279521942, 0.9294854402542114, 0.5160025358200073, 0.8103695511817932, 0.3906894624233246, 0.21516236662864685, 0.5448493361473083, 0.271642804145813, 0.9791553616523743, 0.24226327240467072, 0.3658221662044525}, {0.27822810411453247, 0.3977767825126648, 0.2076532542705536, 0.6730566620826721, 0.6637044548988342, 0.5786892771720886, 0.17390252649784088, 0.9074671864509583, 0.8004252910614014, 0.05197066813707352, 0.08160579204559326, 0.6091148257255554, 0.5557429194450378, 0.3979879319667816, 0.7800790667533875, 0.39488187432289124, 0.7524593472480774, 0.45776379108428955, 0.5314825177192688, 0.5234230160713196, 0.2163933664560318, 0.9552223086357117, 0.7792617082595825, 0.29054608941078186, 0.7557146549224854, 0.6246825456619263, 0.5233676433563232, 0.2516992390155792, 0.6929445266723633, 0.6881045699119568, 0.8985528349876404, 0.46896159648895264, 0.8634102940559387, 0.5724959373474121, 0.4124239385128021, 0.32021576166152954, 0.46019259095191956, 0.4613141417503357, 0.12659643590450287, 0.2629024088382721, 0.31308501958847046, 0.21706698834896088, 0.21628046035766602, 0.04090302065014839, 0.18592612445354462, 0.5217729210853577, 0.37971851229667664, 0.4880172908306122, 0.1113363653421402, 0.11821216344833374, 0.42128700017929077, 0.11649508774280548, 0.38558220863342285, 0.8851243853569031, 0.1640923172235489, 0.14657633006572723, 0.9436848759651184, 0.2067292332649231, 0.8834922909736633, 0.4559359848499298, 0.8702185153961182, 0.7394294738769531, 0.9668525457382202, 0.6381635069847107, 0.9049034714698792, 0.6067693829536438, 0.21463286876678467, 0.14890679717063904, 0.08572885394096375, 0.4590650796890259, 0.6540353894233704, 0.9012057781219482, 0.805749237537384, 0.9170213341712952, 0.12934982776641846, 0.8770532608032227, 0.3329417109489441, 0.03898335248231888, 0.10276661813259125, 0.6931241154670715, 0.3592422604560852, 0.9003793001174927, 0.8707792162895203, 0.7005295753479004, 0.6406556963920593, 0.9654063582420349, 0.8056381940841675, 0.5579832792282104, 0.11490759998559952, 0.8067538738250732, 0.8642386198043823, 0.38583460450172424, 0.7894561290740967, 0.8906769156455994, 0.8466706275939941, 0.07299995422363281, 0.9149901270866394, 0.5507197976112366, 0.6564015746116638, 0.35129162669181824}, {0.38936927914619446, 0.2770198583602905, 0.158053457736969, 0.3571324050426483, 0.4928380846977234, 0.06636106967926025, 0.618083655834198, 0.22849787771701813, 0.15798109769821167, 0.16823235154151917, 0.06651389598846436, 0.14651887118816376, 0.4307006597518921, 0.3663787245750427, 0.2194928228855133, 0.9799826741218567, 0.9192975163459778, 0.29322928190231323, 0.12362945824861526, 0.6033345460891724, 0.8109371662139893, 0.3099300265312195, 0.13493743538856506, 0.028391821309924126, 0.5772712826728821, 0.1930772215127945, 0.30250459909439087, 0.9685661792755127, 0.3415383994579315, 0.5016332864761353, 0.8291964530944824, 0.7695356607437134, 0.11505492031574249, 0.20651113986968994, 0.3556717336177826, 0.6231397986412048, 0.4897788465023041, 0.5602150559425354, 0.05624352768063545, 0.5733413696289062, 0.1900406777858734, 0.5272610187530518, 0.05906716734170914, 0.5487231612205505, 0.023528052493929863, 0.9963155388832092, 0.6770263314247131, 0.11823855340480804, 0.3596782386302948, 0.667708694934845, 0.3459172546863556, 0.09462270885705948, 0.7225035429000854, 0.2471286803483963, 0.8387262225151062, 0.0854233130812645, 0.7724209427833557, 0.6535384654998779, 0.13458438217639923, 0.645948052406311, 0.47119346261024475, 0.7993506193161011, 0.8563597202301025, 0.6594855189323425, 0.011047173291444778, 0.6665021181106567, 0.026599176228046417, 0.447350412607193, 0.03600447252392769, 0.13292214274406433, 0.7308521866798401, 0.17266923189163208, 0.9985589385032654, 0.20662041008472443, 0.4423755705356598, 0.1478460282087326, 0.08967046439647675, 0.6187492609024048, 0.2385541796684265, 0.7725605368614197, 0.7386065125465393, 0.43107226490974426, 0.5539290904998779, 0.12025513499975204, 0.10875170677900314, 0.6220133304595947, 0.5777755975723267, 0.8244478106498718, 0.6451315879821777, 0.5755798816680908, 0.9825515747070312, 0.39420488476753235, 0.5158816576004028, 0.33514904975891113, 0.3467501401901245, 0.47038111090660095, 0.24390870332717896, 0.9069490432739258, 0.10582105070352554, 0.8295518159866333}, {0.5033634901046753, 0.05398034676909447, 0.08171738684177399, 0.3019554018974304, 0.7613180875778198, 0.9632248878479004, 0.2428758591413498, 0.07016821205615997, 0.10967796295881271, 0.6110402345657349, 0.606830358505249, 0.17710916697978973, 0.017184074968099594, 0.6090682744979858, 0.14066337049007416, 0.7285853624343872, 0.4945901036262512, 0.787074625492096, 0.8390697240829468, 0.7119424939155579, 0.029191158711910248, 0.2572312355041504, 0.4372461140155792, 0.15272478759288788, 0.7952682971954346, 0.9718928933143616, 0.004589525517076254, 0.9242216348648071, 0.3803531527519226, 0.1618005484342575, 0.11936024576425552, 0.2822287976741791, 0.17459186911582947, 0.4626694619655609, 0.8967418074607849, 0.3139055073261261, 0.41578468680381775, 0.07642368227243423, 0.3952617049217224, 0.27106207609176636, 0.9792140126228333, 0.5947346091270447, 0.11030547320842743, 0.6144946813583374, 0.4918906092643738, 0.8190739154815674, 0.2044394612312317, 0.1860286146402359, 0.3682728707790375, 0.437425822019577, 0.3156544268131256, 0.24290087819099426, 0.5642048716545105, 0.25159168243408203, 0.1479617953300476, 0.3889099955558777, 0.731597900390625, 0.5560135245323181, 0.5114867091178894, 0.4812278747558594, 0.46278610825538635, 0.3861982524394989, 0.8254769444465637, 0.805984616279602, 0.3466871976852417, 0.6756511926651001, 0.9198344349861145, 0.8236473202705383, 0.915949821472168, 0.9010955095291138, 0.853485643863678, 0.8938629627227783, 0.7839573621749878, 0.7410810589790344, 0.09453754127025604, 0.16347363591194153, 0.26122385263442993, 0.6946619153022766, 0.9516631364822388, 0.1091870367527008, 0.42668235301971436, 0.3387730121612549, 0.0381038673222065, 0.49962595105171204, 0.36127498745918274, 0.8496476411819458, 0.05600585415959358, 0.9819904565811157, 0.05238396301865578, 0.7534030079841614, 0.6775622963905334, 0.7501651644706726, 0.2700348496437073, 0.9657537937164307, 0.26473525166511536, 0.30205413699150085, 0.41777053475379944, 0.18169155716896057, 0.9557178020477295, 0.04283306747674942}, {0.8884923458099365, 0.2648736834526062, 0.7334069609642029, 0.32573091983795166, 0.8760970234870911, 0.21109287440776825, 0.11071576923131943, 0.5452114939689636, 0.8821389675140381, 0.8220111727714539, 0.7171516418457031, 0.9596377611160278, 0.9300462603569031, 0.4120369851589203, 0.7444896101951599, 0.1832609623670578, 0.37102487683296204, 0.21827977895736694, 0.8561930060386658, 0.5347576141357422, 0.2689031958580017, 0.9933080673217773, 0.9740462303161621, 0.13301730155944824, 0.056308191269636154, 0.09314790368080139, 0.2676998972892761, 0.739209771156311, 0.40993016958236694, 0.020919466391205788, 0.9333642721176147, 0.3632168769836426, 0.7456923127174377, 0.6463685631752014, 0.7677087187767029, 0.7414551377296448, 0.39254671335220337, 0.5985624194145203, 0.18972286581993103, 0.1567511409521103, 0.02134268917143345, 0.7758033275604248, 0.8002888560295105, 0.11113018542528152, 0.0991317629814148, 0.24106422066688538, 0.4797864854335785, 0.0792129710316658, 0.6120710372924805, 0.10695433616638184, 0.26131823658943176, 0.006592991761863232, 0.7081231474876404, 0.44281205534935, 0.6266566514968872, 0.8266412615776062, 0.2662794589996338, 0.8720853924751282, 0.3240280747413635, 0.044687483459711075, 0.7788375616073608, 0.35153159499168396, 0.20386281609535217, 0.6551347374916077, 0.7954999804496765, 0.9480498433113098, 0.4658662676811218, 0.5387143492698669, 0.09911750257015228, 0.093500517308712, 0.4332532286643982, 0.9043477773666382, 0.4115818738937378, 0.7888302206993103, 0.4612281620502472, 0.9569956064224243, 0.9451777338981628, 0.005224561784416437, 0.9894824028015137, 0.22334492206573486, 0.4046700596809387, 0.6551423072814941, 0.5601983666419983, 0.6160843372344971, 0.6159529685974121, 0.7265438437461853, 0.07679079473018646, 0.7332929372787476, 0.17657087743282318, 0.47440335154533386, 0.8296063542366028, 0.45738527178764343, 0.359485924243927, 0.2804205119609833, 0.4411099851131439, 0.08989464491605759, 0.8441183567047119, 0.3839125335216522, 0.5675407648086548, 0.19580668210983276}, {0.9394739270210266, 0.8662317395210266, 0.3374239504337311, 0.9065658450126648, 0.9569792151451111, 0.9647682905197144, 0.7642686367034912, 0.9992070198059082, 0.6080642342567444, 0.6953255534172058, 0.39816373586654663, 0.4809184670448303, 0.5846645832061768, 0.7031620144844055, 0.689384400844574, 0.3761059045791626, 0.12712515890598297, 0.7785769701004028, 0.7241586446762085, 0.11796797811985016, 0.9053844809532166, 0.2513693571090698, 0.9193971157073975, 0.3125288486480713, 0.3479987382888794, 0.8037990927696228, 0.46641772985458374, 0.7331066727638245, 0.22100357711315155, 0.8002514243125916, 0.9555689692497253, 0.5955371856689453, 0.17133201658725739, 0.14439071714878082, 0.4493861198425293, 0.4098396301269531, 0.8065125942230225, 0.7457917332649231, 0.5251893997192383, 0.00507045304402709, 0.6708433628082275, 0.416336327791214, 0.9671346545219421, 0.4511932134628296, 0.4217849373817444, 0.6300014853477478, 0.6541732549667358, 0.22489576041698456, 0.27341875433921814, 0.5276232957839966, 0.5330420136451721, 0.23154990375041962, 0.5355644822120667, 0.17778903245925903, 0.12542948126792908, 0.29387885332107544, 0.8083894848823547, 0.21325905621051788, 0.7580370306968689, 0.8101750016212463, 0.20892992615699768, 0.07000412791967392, 0.9946861267089844, 0.2853946089744568, 0.40442201495170593, 0.45834478735923767, 0.048893075436353683, 0.0047622742131352425, 0.5385808944702148, 0.6202524900436401, 0.044551603496074677, 0.5416032075881958, 0.07892622798681259, 0.7203118801116943, 0.20766741037368774, 0.39028632640838623, 0.6972414255142212, 0.005038440227508545, 0.3608587682247162, 0.6454934477806091, 0.6067338585853577, 0.23669306933879852, 0.9447422027587891, 0.16202621161937714, 0.677869975566864, 0.5553561449050903, 0.44124704599380493, 0.10846634209156036, 0.3882180154323578, 0.6205949783325195, 0.2370886206626892, 0.7887666821479797, 0.6892134547233582, 0.8803793787956238, 0.7159149646759033, 0.2291385382413864, 0.423196017742157, 0.22604705393314362, 0.08595598489046097, 0.4638025462627411}, {0.07897908985614777, 0.9845580458641052, 0.3096880316734314, 0.5996499061584473, 0.804765522480011, 0.8112592697143555, 0.6775122284889221, 0.8923686146736145, 0.1330523043870926, 0.9732910990715027, 0.09806454926729202, 0.01898907497525215, 0.7238990068435669, 0.21605481207370758, 0.5497144460678101, 0.13844317197799683, 0.32567915320396423, 0.008058831095695496, 0.069766566157341, 0.8276932239532471, 0.714502215385437, 0.7960177063941956, 0.09258463978767395, 0.9444786906242371, 0.5062655806541443, 0.9950422048568726, 0.5095604658126831, 0.48539355397224426, 0.9809320569038391, 0.11259230971336365, 0.3017212450504303, 0.2526201605796814, 0.6917288899421692, 0.8774569630622864, 0.82963627576828, 0.9353206157684326, 0.6209621429443359, 0.8200883865356445, 0.29752394556999207, 0.2926337718963623, 0.8217452764511108, 0.10389402508735657, 0.27178269624710083, 0.28975823521614075, 0.38834041357040405, 0.7283012270927429, 0.32330867648124695, 0.2474859356880188, 0.9203870892524719, 0.4713290333747864, 0.06235341727733612, 0.8015143275260925, 0.2244453728199005, 0.7394166588783264, 0.011807703413069248, 0.11507527530193329, 0.40563082695007324, 0.7698266506195068, 0.426132470369339, 0.022014165297150612, 0.10221531987190247, 0.9898159503936768, 0.7444547414779663, 0.45995715260505676, 0.8503733277320862, 0.955604076385498, 0.7090998888015747, 0.3009747266769409, 0.6615327000617981, 0.9729999303817749, 0.7126728296279907, 0.9798617959022522, 0.5668312907218933, 0.40645280480384827, 0.4871188998222351, 0.2867446839809418, 0.8197424411773682, 0.7390077114105225, 0.4742147922515869, 0.022739548236131668, 0.2770746946334839, 0.608462929725647, 0.6296801567077637, 0.631210446357727, 0.14132730662822723, 0.3723589777946472, 0.29467499256134033, 0.8514028191566467, 0.2281835675239563, 0.5445646643638611, 0.12813565135002136, 0.8001298308372498, 0.6580221652984619, 0.6417206525802612, 0.1781056672334671, 0.6032388806343079, 0.2602071166038513, 0.8711204528808594, 0.5718010663986206, 0.8741961121559143}, {0.16854940354824066, 0.749336838722229, 0.7728888392448425, 0.18423610925674438, 0.4172949194908142, 0.9485779404640198, 0.5074175000190735, 0.12557850778102875, 0.422415554523468, 0.1768399477005005, 0.3758013844490051, 0.21316398680210114, 0.7662849426269531, 0.892800509929657, 0.8156951665878296, 0.7488518357276917, 0.901447057723999, 0.7380874752998352, 0.12263242900371552, 0.8579025268554688, 0.24236054718494415, 0.2595860958099365, 0.5110389590263367, 0.7567280530929565, 0.7985833883285522, 0.08752476423978806, 0.21233637630939484, 0.24328801035881042, 0.004258256405591965, 0.3918575942516327, 0.12437480688095093, 0.1554863303899765, 0.8505492806434631, 0.49491214752197266, 0.9310325384140015, 0.17742812633514404, 0.31461986899375916, 0.935811460018158, 0.8738961219787598, 0.16386114060878754, 0.48624274134635925, 0.18386319279670715, 0.48816925287246704, 0.2677495777606964, 0.4213097393512726, 0.1288798302412033, 0.8174855709075928, 0.7745636701583862, 0.3719898462295532, 0.268341988325119, 0.7256401777267456, 0.714263916015625, 0.03820566460490227, 0.5224220156669617, 0.7092438340187073, 0.8248240351676941, 0.571763813495636, 0.6627675294876099, 0.5176861882209778, 0.009824661538004875, 0.8644697666168213, 0.9183409214019775, 0.5708417296409607, 0.8340622782707214, 0.7665989995002747, 0.5426353812217712, 0.16529107093811035, 0.18011850118637085, 0.12082867324352264, 0.6578594446182251, 0.13741232454776764, 0.2797444760799408, 0.5192334055900574, 0.22881701588630676, 0.41946983337402344, 0.9745871424674988, 0.7779221534729004, 0.020958680659532547, 0.23361290991306305, 0.1712418645620346, 0.24324673414230347, 0.09375084936618805, 0.7774490714073181, 0.07304903119802475, 0.666466474533081, 0.42657020688056946, 0.4676324129104614, 0.024679668247699738, 0.2501203417778015, 0.7619205713272095, 0.7659842371940613, 0.7032923698425293, 0.2493852823972702, 0.17793777585029602, 0.37795913219451904, 0.3973761796951294, 0.8380632400512695, 0.39157670736312866, 0.8357030153274536, 0.46949490904808044}, {0.6675028204917908, 0.09278228878974915, 0.5101723670959473, 0.7067201733589172, 0.08266005665063858, 0.7348914742469788, 0.3131669759750366, 0.399060994386673, 0.5350664258003235, 0.5288364887237549, 0.6527334451675415, 0.0011672171531245112, 0.10111138224601746, 0.08603248000144958, 0.3099416196346283, 0.08609248697757721, 0.7857266664505005, 0.6068125367164612, 0.06409859657287598, 0.5429248809814453, 0.2052062749862671, 0.14107556641101837, 0.6269888281822205, 0.9172220826148987, 0.9677394032478333, 0.5302925705909729, 0.9578779339790344, 0.15723657608032227, 0.39474058151245117, 0.7549474835395813, 0.6741874814033508, 0.7250961661338806, 0.2266473025083542, 0.6466544270515442, 0.12777183949947357, 0.707541286945343, 0.7641687393188477, 0.20217859745025635, 0.6490570306777954, 0.2304195612668991, 0.15656843781471252, 0.3682672083377838, 0.7923011779785156, 0.8916081786155701, 0.8815463781356812, 0.2878151834011078, 0.6032502055168152, 0.8627312779426575, 0.27116265892982483, 0.6312558650970459, 0.5185495615005493, 0.9800235629081726, 0.5195280313491821, 0.2736937403678894, 0.5102943181991577, 0.5416578650474548, 0.16936735808849335, 0.7369903922080994, 0.6248253583908081, 0.2803938686847687, 0.8653063774108887, 0.04828588292002678, 0.7930660843849182, 0.6396650075912476, 0.06226520985364914, 0.7603341937065125, 0.7538713216781616, 0.8353912830352783, 0.8280989527702332, 0.16439296305179596, 0.48722559213638306, 0.7365726232528687, 0.575480043888092, 0.38565540313720703, 0.23215919733047485, 0.64857017993927, 0.2606777250766754, 0.007795956917107105, 0.6899186968803406, 0.05936511233448982, 0.04750022664666176, 0.7488878965377808, 0.20466987788677216, 0.786125123500824, 0.4258389174938202, 0.17847278714179993, 0.25677070021629333, 0.8774675726890564, 0.8321441411972046, 0.80070960521698, 0.13697803020477295, 0.12041258066892624, 0.9935291409492493, 0.784259557723999, 0.5692885518074036, 0.9855568408966064, 0.019859429448843002, 0.9407095313072205, 0.6554074883460999, 0.33654335141181946}, {0.8688483238220215, 0.1260657161474228, 0.16036146879196167, 0.42628490924835205, 0.23288720846176147, 0.23533228039741516, 0.29684337973594666, 0.02562396414577961, 0.9238227009773254, 0.7465649843215942, 0.5921811461448669, 0.31911179423332214, 0.560141384601593, 0.7606337070465088, 0.36077767610549927, 0.5199529528617859, 0.44940000772476196, 0.7086085081100464, 0.3663721978664398, 0.9167582392692566, 0.024725766852498055, 0.23133310675621033, 0.5953042507171631, 0.6588608026504517, 0.03071143850684166, 0.9108402132987976, 0.7466799020767212, 0.8360960483551025, 0.37760722637176514, 0.6794567704200745, 0.5538642406463623, 0.04351053759455681, 0.05298018082976341, 0.8393434882164001, 0.21733254194259644, 0.060363080352544785, 0.21924841403961182, 0.3431716561317444, 0.6346892714500427, 0.8441552519798279, 0.2709071934223175, 0.005680169444531202, 0.766889214515686, 0.10359594970941544, 0.38446739315986633, 0.034509677439928055, 0.005657636094838381, 0.776965320110321, 0.18540774285793304, 0.1348203420639038, 0.2413601577281952, 0.24741138517856598, 0.039704591035842896, 0.876168429851532, 0.9803748726844788, 0.4039488136768341, 0.08162526041269302, 0.5845034718513489, 0.9172362089157104, 0.24737252295017242, 0.19643370807170868, 0.6396631002426147, 0.6970326900482178, 0.25630322098731995, 0.01393718272447586, 0.013744496740400791, 0.9744455814361572, 0.9158830642700195, 0.20802314579486847, 0.13022471964359283, 0.3710119128227234, 0.24312148988246918, 0.3156474232673645, 0.1852654069662094, 0.8970090746879578, 0.9307869672775269, 0.626654863357544, 0.9679094552993774, 0.5084463357925415, 0.48834583163261414, 0.20011378824710846, 0.8772273659706116, 0.19941772520542145, 0.1417914628982544, 0.28417161107063293, 0.04483281448483467, 0.8212049603462219, 0.010440378449857235, 0.38376250863075256, 0.8105820417404175, 0.45553359389305115, 0.3654700219631195, 0.16129045188426971, 0.9340004920959473, 0.6943865418434143, 0.5472015142440796, 0.03561766818165779, 0.12006395310163498, 0.022711148485541344, 0.3905426561832428}, {0.8900226354598999, 0.9608217477798462, 0.009123251773416996, 0.5365995764732361, 0.11460002511739731, 0.06047365069389343, 0.735691487789154, 0.30835095047950745, 0.661583662033081, 0.06429445743560791, 0.6829100251197815, 0.4204856753349304, 0.24667774140834808, 0.013861020095646381, 0.5529398918151855, 0.8783172369003296, 0.1694900095462799, 0.7664071321487427, 0.13507239520549774, 0.8196008801460266, 0.81497722864151, 0.1265181601047516, 0.6046032905578613, 0.396684855222702, 0.28377723693847656, 0.16888394951820374, 0.704555869102478, 0.7722216248512268, 0.24818702042102814, 0.10965767502784729, 0.46699947118759155, 0.7059521675109863, 0.3618009388446808, 0.6949279308319092, 0.7245088219642639, 0.6323319673538208, 0.03243238851428032, 0.21985073387622833, 0.6123069524765015, 0.9920046925544739, 0.015361050143837929, 0.7109373807907104, 0.8563105463981628, 0.3857102394104004, 0.9797226786613464, 0.19301792979240417, 0.7167463898658752, 0.7596962451934814, 0.4888037443161011, 0.4129808247089386, 0.6488394141197205, 0.08217353373765945, 0.599470317363739, 0.6739674806594849, 0.2246886044740677, 0.7354762554168701, 0.07887809723615646, 0.6519911289215088, 0.4095950722694397, 0.37587088346481323, 0.48553285002708435, 0.9790326356887817, 0.8670355677604675, 0.46638283133506775, 0.65203857421875, 0.7578762173652649, 0.02483280934393406, 0.39370158314704895, 0.34024545550346375, 0.025823358446359634, 0.8629189133644104, 0.6340116858482361, 0.48700806498527527, 0.6485847234725952, 0.5432604551315308, 0.02963644452393055, 0.4924508333206177, 0.22360306978225708, 0.1488075852394104, 0.40361690521240234, 0.17902947962284088, 0.02703474462032318, 0.5128759145736694, 0.047779250890016556, 0.056294769048690796, 0.32499074935913086, 0.5103417634963989, 0.6826861500740051, 0.2130354940891266, 0.07825129479169846, 0.6696047186851501, 0.626640260219574, 0.606048047542572, 0.9366344213485718, 0.030306871980428696, 0.6213101744651794, 0.37448573112487793, 0.20203983783721924, 0.29927951097488403, 0.641312301158905}, {0.10643862932920456, 0.4501097798347473, 0.799071192741394, 0.6301440000534058, 0.5409406423568726, 0.4345757067203522, 0.49876806139945984, 0.18360449373722076, 0.6152886748313904, 0.1645413041114807, 0.8641584515571594, 0.6687875390052795, 0.4685857892036438, 0.9592210650444031, 0.6847468614578247, 0.9733351469039917, 0.7775821089744568, 0.6012861132621765, 0.5597612261772156, 0.957773745059967, 0.5210644602775574, 0.5135582685470581, 0.10863679647445679, 0.7449062466621399, 0.3385239541530609, 0.7820227146148682, 0.17292241752147675, 0.5285668969154358, 0.09851248562335968, 0.3194827437400818, 0.7747154831886292, 0.11118707060813904, 0.868135392665863, 0.5745067000389099, 0.8489765524864197, 0.7773573398590088, 0.32336121797561646, 0.13974839448928833, 0.2083311825990677, 0.10754530876874924, 0.7992637753486633, 0.9283109903335571, 0.4613514840602875, 0.05974713712930679, 0.1884182095527649, 0.9687412977218628, 0.7234125137329102, 0.46099045872688293, 0.9191815853118896, 0.12023615837097168, 0.04943128675222397, 0.8281651735305786, 0.3244445323944092, 0.36550983786582947, 0.2901727259159088, 0.24492423236370087, 0.10613472014665604, 0.9019297957420349, 0.4471333920955658, 0.005113463383167982, 0.040018826723098755, 0.5034040808677673, 0.938608705997467, 0.35975468158721924, 0.8698610663414001, 0.8555201292037964, 0.3543303906917572, 0.9313591718673706, 0.6828713417053223, 0.6645790338516235, 0.07501120865345001, 0.22898298501968384, 0.6241204142570496, 0.9743528962135315, 0.7529717087745667, 0.6209596395492554, 0.36091363430023193, 0.6369472742080688, 0.9136396050453186, 0.3442859947681427, 0.7900457382202148, 0.601597785949707, 0.5817118883132935, 0.17958597838878632, 0.9788990616798401, 0.24221935868263245, 0.7284110188484192, 0.841484785079956, 0.9731760025024414, 0.24861156940460205, 0.2406127154827118, 0.47269994020462036, 0.6757106184959412, 0.8208947777748108, 0.24368228018283844, 0.6349794268608093, 0.47637656331062317, 0.9946995973587036, 0.34269943833351135, 0.057123880833387375}, {0.5635218620300293, 0.46889013051986694, 0.1595819741487503, 0.5968023538589478, 0.23188921809196472, 0.8149820566177368, 0.356514573097229, 0.5625863671302795, 0.4925236701965332, 0.8330838084220886, 0.7849994897842407, 0.38622596859931946, 0.685033917427063, 0.13210241496562958, 0.7534232139587402, 0.5499144792556763, 0.6110417246818542, 0.9364302158355713, 0.5362139940261841, 0.9434849619865417, 0.5658993721008301, 0.4476253092288971, 0.9963759183883667, 0.41913434863090515, 0.23658612370491028, 0.8030601739883423, 0.10782881081104279, 0.9182209968566895, 0.7399726510047913, 0.8254274725914001, 0.9217079281806946, 0.39442697167396545, 0.9368327856063843, 0.9609898924827576, 0.7106180787086487, 0.2902466654777527, 0.5227506756782532, 0.9660990834236145, 0.4405941069126129, 0.5869877934455872, 0.7377188801765442, 0.9201322197914124, 0.5264474749565125, 0.9330933690071106, 0.3453775644302368, 0.9299194812774658, 0.58601313829422, 0.3625839650630951, 0.006667715031653643, 0.32737281918525696, 0.3077826201915741, 0.8890206813812256, 0.496287077665329, 0.09030987322330475, 0.3707504868507385, 0.2292478233575821, 0.29350021481513977, 0.8084602952003479, 0.8265685439109802, 0.5264289975166321, 0.7884281277656555, 0.024123849347233772, 0.8410730361938477, 0.866203248500824, 0.5006245374679565, 0.5940152406692505, 0.5401377081871033, 0.4256168305873871, 0.938480019569397, 0.3070540428161621, 0.44665852189064026, 0.01023387536406517, 0.1976572424173355, 0.8449748158454895, 0.044535499066114426, 0.2349843531847, 0.49332207441329956, 0.3096795380115509, 0.3739027976989746, 0.9310874342918396, 0.2524011731147766, 0.8690001964569092, 0.7679184079170227, 0.852203369140625, 0.5139123201370239, 0.9476092457771301, 0.9210686087608337, 0.5206603407859802, 0.7058224081993103, 0.16600607335567474, 0.50136399269104, 0.6325134634971619, 0.8626760840415955, 0.8547446131706238, 0.029168609529733658, 0.8477604985237122, 0.02574373595416546, 0.7673045992851257, 0.4935978353023529, 0.5911397933959961}, {0.7270059585571289, 0.8379987478256226, 0.9853413105010986, 0.5067370533943176, 0.870658278465271, 0.6076249480247498, 0.9428158402442932, 0.08190908282995224, 0.5001099705696106, 0.5914739370346069, 0.990967333316803, 0.9617705345153809, 0.9847176671028137, 0.9195536971092224, 0.9731589555740356, 0.43165940046310425, 0.7858155965805054, 0.3813808560371399, 0.688999354839325, 0.2502763867378235, 0.30668768286705017, 0.7980754375457764, 0.915744960308075, 0.24087195098400116, 0.6028282046318054, 0.6516447067260742, 0.22198601067066193, 0.7882356643676758, 0.623367428779602, 0.34220242500305176, 0.7805476188659668, 0.26918044686317444, 0.4688820540904999, 0.07528941333293915, 0.7122516632080078, 0.16826798021793365, 0.21760332584381104, 0.6101091504096985, 0.9772830009460449, 0.9199604988098145, 0.9378072023391724, 0.36893028020858765, 0.6190387606620789, 0.4489949345588684, 0.9280872941017151, 0.7372064590454102, 0.408719927072525, 0.11955944448709488, 0.18372023105621338, 0.08810875564813614, 0.23536206781864166, 0.8560143113136292, 0.3371707499027252, 0.9235560297966003, 0.7479472756385803, 0.7734896540641785, 0.09209994971752167, 0.8384816646575928, 0.45809081196784973, 0.10489118844270706, 0.8521009683609009, 0.7978577017784119, 0.4630555212497711, 0.1005406528711319, 0.420920193195343, 0.9481164813041687, 0.5804983973503113, 0.06505608558654785, 0.5707130432128906, 0.7889280319213867, 0.3613301217556, 0.027797115966677666, 0.22214724123477936, 0.6546949744224548, 0.9389461278915405, 0.6606898903846741, 0.6052539348602295, 0.5760915875434875, 0.868562638759613, 0.39556023478507996, 0.6857859492301941, 0.7915783524513245, 0.8454707860946655, 0.5484679341316223, 0.4667079746723175, 0.6977353692054749, 0.9506816864013672, 0.4093143939971924, 0.8057969212532043, 0.2843318283557892, 0.31352269649505615, 0.71426922082901, 0.5940852761268616, 0.793400228023529, 0.9762995839118958, 0.5884966850280762, 0.5900006890296936, 0.257936954498291, 0.6617192625999451, 0.8702465891838074}, {0.7300350666046143, 0.05979108810424805, 0.12685909867286682, 0.48462221026420593, 0.3481845259666443, 0.36582595109939575, 0.6130593419075012, 0.8454601168632507, 0.28036758303642273, 0.11597942560911179, 0.8003000617027283, 0.8288113474845886, 0.31440219283103943, 0.053066227585077286, 0.9534893035888672, 0.10304219275712967, 0.40747544169425964, 0.28615179657936096, 0.5282697677612305, 0.4271649718284607, 0.5307804346084595, 0.40516048669815063, 0.08176219463348389, 0.32375431060791016, 0.5736621618270874, 0.7345119118690491, 0.047744203358888626, 0.060134559869766235, 0.4612036645412445, 0.8823814988136292, 0.20694565773010254, 0.47147229313850403, 0.5546632409095764, 0.3224206268787384, 0.9443755745887756, 0.6086987853050232, 0.48504990339279175, 0.05502541735768318, 0.30972424149513245, 0.2061537504196167, 0.15559133887290955, 0.18131262063980103, 0.34889575839042664, 0.36499521136283875, 0.2895031273365021, 0.7176059484481812, 0.6956460475921631, 0.1399722844362259, 0.3372679054737091, 0.9384087920188904, 0.31908315420150757, 0.12104356288909912, 0.7386629581451416, 0.18833741545677185, 0.49205297231674194, 0.11759095638990402, 0.49601805210113525, 0.5708766579627991, 0.7814540266990662, 0.08130314201116562, 0.43361353874206543, 0.4004920423030853, 0.8840382695198059, 0.8251399397850037, 0.6755180358886719, 0.7844743728637695, 0.9782652258872986, 0.03205987066030502, 0.14959672093391418, 0.6543171405792236, 0.38367608189582825, 0.3463338017463684, 0.5958032011985779, 0.4919665455818176, 0.3263886868953705, 0.4545809328556061, 0.06284308433532715, 0.2246263027191162, 0.8611940145492554, 0.3429340124130249, 0.6916936635971069, 0.7544829249382019, 0.5733326077461243, 0.8932832479476929, 0.9347665309906006, 0.1818930059671402, 0.11779823899269104, 0.5875250101089478, 0.5925485491752625, 0.46567803621292114, 0.7320052981376648, 0.003646255936473608, 0.6102949976921082, 0.80079185962677, 0.8769616484642029, 0.8395859599113464, 0.3761598765850067, 0.9344877600669861, 0.1296999305486679, 0.6266553401947021}, {0.763456404209137, 0.5750162601470947, 0.48273423314094543, 0.8420974612236023, 0.3351898789405823, 0.16232098639011383, 0.7596231698989868, 0.6083659529685974, 0.6547523140907288, 0.7354391813278198, 0.6113293766975403, 0.5764157772064209, 0.2777544856071472, 0.1628369688987732, 0.058895789086818695, 0.18492841720581055, 0.025918900966644287, 0.2417554259300232, 0.574865996837616, 0.22176705300807953, 0.8160465359687805, 0.28164833784103394, 0.9862816333770752, 0.9632071256637573, 0.845672070980072, 0.31744831800460815, 0.7389929294586182, 0.2550615668296814, 0.565090000629425, 0.07969988137483597, 0.13219770789146423, 0.4281499683856964, 0.7871010303497314, 0.1751604825258255, 0.8382125496864319, 0.5454674959182739, 0.564136266708374, 0.18870435655117035, 0.9624302387237549, 0.3739798367023468, 0.6561686992645264, 0.8315421938896179, 0.018496353179216385, 0.56428462266922, 0.6075593829154968, 0.434848815202713, 0.7770899534225464, 0.9838429689407349, 0.28010520339012146, 0.7209224104881287, 0.12220663577318192, 0.400134801864624, 0.7889910340309143, 0.03334083408117294, 0.21939140558242798, 0.6221253275871277, 0.7015204429626465, 0.4504122734069824, 0.5294076204299927, 0.03655349090695381, 0.8199601769447327, 0.16520263254642487, 0.914979100227356, 0.24915306270122528, 0.9913619160652161, 0.9734119176864624, 0.4080621302127838, 0.7860210537910461, 0.3087259829044342, 0.9258151054382324, 0.5166798233985901, 0.7909741997718811, 0.4805918335914612, 0.03026416152715683, 0.9217251539230347, 0.023868853226304054, 0.17231355607509613, 0.8772484660148621, 0.3138962984085083, 0.40452152490615845, 0.7180010080337524, 0.3999526798725128, 0.9012634754180908, 0.3382173478603363, 0.44146421551704407, 0.27770379185676575, 0.5278270840644836, 0.3844150900840759, 0.4607500433921814, 0.653739333152771, 0.3511106073856354, 0.26731711626052856, 0.15499290823936462, 0.570773184299469, 0.02606399543583393, 0.6754699945449829, 0.4489438235759735, 0.04412337392568588, 0.7722099423408508, 0.4377506375312805}, {0.21666093170642853, 0.027085157111287117, 0.7094085812568665, 0.4794512987136841, 0.3739895224571228, 0.5351520776748657, 0.6086593866348267, 0.5813124179840088, 0.04555770754814148, 0.5858076810836792, 0.4919147193431854, 0.5727692246437073, 0.3153899908065796, 0.19938349723815918, 0.10233920812606812, 0.012478873133659363, 0.8724101781845093, 0.052187010645866394, 0.8018388152122498, 0.6218363046646118, 0.4876927435398102, 0.009107783436775208, 0.9319734573364258, 0.23984062671661377, 0.4376002848148346, 0.4574201703071594, 0.4344885051250458, 0.5643821358680725, 0.41014721989631653, 0.2467443197965622, 0.8545859456062317, 0.14801976084709167, 0.3081483542919159, 0.6722511053085327, 0.023561693727970123, 0.2372431457042694, 0.6904196739196777, 0.9910569787025452, 0.9147743582725525, 0.4939208924770355, 0.8289467096328735, 0.6628316640853882, 0.7397707104682922, 0.6285590529441833, 0.18586036562919617, 0.2249174565076828, 0.8295791149139404, 0.9315932393074036, 0.002135790651664138, 0.43508777022361755, 0.9072669148445129, 0.9133219122886658, 0.6306719183921814, 0.11399323493242264, 0.6563757061958313, 0.42974209785461426, 0.09274277836084366, 0.8497764468193054, 0.9418402910232544, 0.05464553460478783, 0.7233766913414001, 0.6941684484481812, 0.1601683497428894, 0.7648065686225891, 0.48572424054145813, 0.8637267351150513, 0.7931110858917236, 0.7019190788269043, 0.17810305953025818, 0.4032318890094757, 0.13908575475215912, 0.9231674671173096, 0.3832220733165741, 0.09897653758525848, 0.33658403158187866, 0.8824637532234192, 0.2774578034877777, 0.7967892289161682, 0.02913210354745388, 0.5125642418861389, 0.4399191737174988, 0.7484796643257141, 0.0885380357503891, 0.5314390659332275, 0.03991128131747246, 0.02887752652168274, 0.6961225867271423, 0.8708542585372925, 0.6528164744377136, 0.738190770149231, 0.7879778742790222, 0.2542860209941864, 0.608988344669342, 0.2813737392425537, 0.989391565322876, 0.3340105712413788, 0.24562455713748932, 0.40724384784698486, 0.3795059621334076, 0.05817630514502525}, {0.2736530900001526, 0.74940025806427, 0.8833975791931152, 0.37147876620292664, 0.2606430947780609, 0.4405718445777893, 0.5667819380760193, 0.3495609760284424, 0.5916213989257812, 0.5502309799194336, 0.6614100933074951, 0.5147643685340881, 0.019180264323949814, 0.6817227005958557, 0.10801156610250473, 0.7426266074180603, 0.1686597466468811, 0.28251346945762634, 0.584014356136322, 0.07660194486379623, 0.3387296497821808, 0.3741713762283325, 0.7791841626167297, 0.754912793636322, 0.523711621761322, 0.06393536925315857, 0.20108427107334137, 0.20230399072170258, 0.6780382394790649, 0.7282348871231079, 0.2960493266582489, 0.16793034970760345, 0.9642715454101562, 0.9782564640045166, 0.2700112760066986, 0.22686640918254852, 0.48000165820121765, 0.0832073837518692, 0.20107108354568481, 0.8744726777076721, 0.15710517764091492, 0.6045858263969421, 0.016971349716186523, 0.4691143035888672, 0.8883286714553833, 0.7410107851028442, 0.2494325041770935, 0.37364524602890015, 0.5792900323867798, 0.7119336128234863, 0.6192927360534668, 0.7025358080863953, 0.043412938714027405, 0.9846843481063843, 0.13686369359493256, 0.8981735706329346, 0.5564923286437988, 0.981383740901947, 0.4948456287384033, 0.6801400184631348, 0.6550004482269287, 0.6129590272903442, 0.36058396100997925, 0.4882773756980896, 0.7772408723831177, 0.9978392720222473, 0.1729879081249237, 0.7048876881599426, 0.07201404869556427, 0.6603001356124878, 0.7333595156669617, 0.07883723825216293, 0.40512731671333313, 0.8227922320365906, 0.6795493960380554, 0.23358337581157684, 0.8663985133171082, 0.5904356241226196, 0.837787926197052, 0.3662526309490204, 0.32065558433532715, 0.21729937195777893, 0.4594438374042511, 0.6063658595085144, 0.7002711296081543, 0.42902272939682007, 0.4323693513870239, 0.5834606289863586, 0.3538118600845337, 0.899100661277771, 0.8115540146827698, 0.6315318942070007, 0.5097130537033081, 0.8759744167327881, 0.7732481956481934, 0.2256544679403305, 0.438350111246109, 0.24979625642299652, 0.2707109749317169, 0.623473048210144}, {0.7182823419570923, 0.7681992053985596, 0.19514870643615723, 0.3796467185020447, 0.2339545339345932, 0.7653937935829163, 0.07142779231071472, 0.2437843680381775, 0.6785507798194885, 0.4016217291355133, 0.0889270007610321, 0.2859765589237213, 0.37336331605911255, 0.6985871195793152, 0.6675858497619629, 0.2449357956647873, 0.3256365656852722, 0.6354649066925049, 0.27423128485679626, 0.24948862195014954, 0.47227898240089417, 0.11499688029289246, 0.9712839722633362, 0.8941547870635986, 0.9698910117149353, 0.5924094319343567, 0.7768167853355408, 0.6038021445274353, 0.977059006690979, 0.1924286037683487, 0.546707808971405, 0.17261053621768951, 0.8796670436859131, 0.49442431330680847, 0.5276975631713867, 0.2665858864784241, 0.11259905993938446, 0.6614609956741333, 0.3535640239715576, 0.20341463387012482, 0.09094312787055969, 0.6338856220245361, 0.4853556752204895, 0.12712423503398895, 0.25529417395591736, 0.0706147700548172, 0.4149570167064667, 0.6864685416221619, 0.06199566274881363, 0.2181474268436432, 0.9472142457962036, 0.7987524271011353, 0.2195722460746765, 0.7095844745635986, 0.011792682111263275, 0.11370665580034256, 0.9629700779914856, 0.2854176461696625, 0.7161007523536682, 0.06657442450523376, 0.8244187235832214, 0.7201775908470154, 0.7213181853294373, 0.5365610718727112, 0.18583744764328003, 0.11753382533788681, 0.6172783374786377, 0.09469573199748993, 0.19625481963157654, 0.5693343281745911, 0.9064959287643433, 0.1342131644487381, 0.02910301461815834, 0.4971165657043457, 0.0340285561978817, 0.055474959313869476, 0.2542687654495239, 0.37236398458480835, 0.650881290435791, 0.3630831241607666, 0.1327185034751892, 0.44257238507270813, 0.47671589255332947, 0.28895989060401917, 0.19760704040527344, 0.49311625957489014, 0.6049328446388245, 0.8556302189826965, 0.7428467869758606, 0.9883568286895752, 0.25584539771080017, 0.48843997716903687, 0.2723716199398041, 0.03239903226494789, 0.9147620797157288, 0.24411997199058533, 0.3418821692466736, 0.28533419966697693, 0.8859678506851196, 0.4378410577774048}, {0.359172523021698, 0.5971083045005798, 0.03567606955766678, 0.22672155499458313, 0.23911382257938385, 0.46300601959228516, 0.7549265027046204, 0.4833844006061554, 0.3141310214996338, 0.7591236233711243, 0.4124397039413452, 0.09020595252513885, 0.7278786897659302, 0.9280110001564026, 0.07976622879505157, 0.22281964123249054, 0.3753165304660797, 0.07813090085983276, 0.15160156786441803, 0.16457435488700867, 0.9595357775688171, 0.8776809573173523, 0.2942555546760559, 0.3935529291629791, 0.19934336841106415, 0.1700851321220398, 0.6044939756393433, 0.4502418339252472, 0.25488564372062683, 0.7912400364875793, 0.6225833892822266, 0.7386006712913513, 0.04136110097169876, 0.42723843455314636, 0.6968786120414734, 0.33935147523880005, 0.012772414833307266, 0.0232376866042614, 0.9676304459571838, 0.9498108625411987, 0.001850143657065928, 0.369880348443985, 0.6461575627326965, 0.6664638519287109, 0.07429470121860504, 0.42758604884147644, 0.8075672388076782, 0.6347774863243103, 0.6741223931312561, 0.5939666628837585, 0.11475588381290436, 0.6013176441192627, 0.127328023314476, 0.42246395349502563, 0.40083715319633484, 0.8622366786003113, 0.5025070905685425, 0.6023308634757996, 0.782880425453186, 0.3955119550228119, 0.2433764487504959, 0.7671778202056885, 0.988874077796936, 0.46677395701408386, 0.94808429479599, 0.6112368702888489, 0.3753332495689392, 0.28840336203575134, 0.10790538787841797, 0.656324565410614, 0.17664045095443726, 0.7588958740234375, 0.6600516438484192, 0.5342243909835815, 0.7209420204162598, 0.015472977422177792, 0.6443530917167664, 0.35031116008758545, 0.2602507174015045, 0.7435663938522339, 0.06061115488409996, 0.7847903370857239, 0.8093681931495667, 0.9331741333007812, 0.1604287177324295, 0.5533326268196106, 0.39023303985595703, 0.5813277959823608, 0.8404349684715271, 0.9715670943260193, 0.5581678748130798, 0.8354477286338806, 0.8592302799224854, 0.9709701538085938, 0.37791427969932556, 0.8382391929626465, 0.437987357378006, 0.7995947003364563, 0.3282906711101532, 0.12830038368701935}, {0.6854858994483948, 0.15247459709644318, 0.9803310632705688, 0.07769959419965744, 0.9665688872337341, 0.3611457645893097, 0.4057995080947876, 0.9173739552497864, 0.323675274848938, 0.20616310834884644, 0.9355189204216003, 0.2844453454017639, 0.4756021797657013, 0.35417985916137695, 0.9552919268608093, 0.14128482341766357, 0.4959349036216736, 0.49289801716804504, 0.07425759732723236, 0.015339231118559837, 0.5056421160697937, 0.3059806227684021, 0.20932768285274506, 0.5800616145133972, 0.9107903242111206, 0.9350030422210693, 0.4303681254386902, 0.7596822381019592, 0.31261876225471497, 0.069813571870327, 0.7612990140914917, 0.586561381816864, 0.3553980886936188, 0.252246230840683, 0.6747890114784241, 0.9459574818611145, 0.5060237646102905, 0.269062340259552, 0.7270051836967468, 0.4041003882884979, 0.10657442361116409, 0.27977389097213745, 0.6864308714866638, 0.8746102452278137, 0.5774561762809753, 0.19065654277801514, 0.2824159860610962, 0.9198304414749146, 0.4102349579334259, 0.25106313824653625, 0.28058770298957825, 0.9321912527084351, 0.8621768951416016, 0.9885106682777405, 0.2109600156545639, 0.2750416696071625, 0.18623800575733185, 0.18834009766578674, 0.6797000169754028, 0.0887705460190773, 0.9898660182952881, 0.30081406235694885, 0.560349702835083, 0.7470682263374329, 0.38775137066841125, 0.04143296182155609, 0.5869919061660767, 0.23379629850387573, 0.12411152571439743, 0.39198487997055054, 0.7656592726707458, 0.30678901076316833, 0.030543530359864235, 0.5894366502761841, 0.6635870933532715, 0.8700049519538879, 0.5039341449737549, 0.7231231927871704, 0.6194799542427063, 0.29525259137153625, 0.2090597152709961, 0.007172367535531521, 0.5678648948669434, 0.48052358627319336, 0.24343715608119965, 0.5150649547576904, 0.9927526116371155, 0.7122169137001038, 0.960327684879303, 0.0519484207034111, 0.07291058450937271, 0.9835347533226013, 0.8310365080833435, 0.4592995345592499, 0.7993347644805908, 0.17022162675857544, 0.7843629121780396, 0.6596028804779053, 0.08215461671352386, 0.06531430035829544}, {0.11942038685083389, 0.06405040621757507, 0.1549472212791443, 0.3199266493320465, 0.39649805426597595, 0.4573734402656555, 0.3269072473049164, 0.015371534042060375, 0.842391848564148, 0.4191659986972809, 0.05483177304267883, 0.2848299443721771, 0.40675652027130127, 0.30059614777565, 0.17930832505226135, 0.7359824776649475, 0.3701893389225006, 0.06176642328500748, 0.4226345419883728, 0.7290066480636597, 0.07452072948217392, 0.6214810013771057, 0.9361134767532349, 0.6424502730369568, 0.2890843152999878, 0.5864616632461548, 0.3865981101989746, 0.322456419467926, 0.024638867005705833, 0.9048427939414978, 0.07348355650901794, 0.8500868082046509, 0.0028821052983403206, 0.5723272562026978, 0.6005833745002747, 0.5833739042282104, 0.9964674711227417, 0.8800317049026489, 0.34710073471069336, 0.7488723397254944, 0.612002968788147, 0.44605547189712524, 0.0033349674195051193, 0.7013749480247498, 0.5990187525749207, 0.4978393614292145, 0.053098443895578384, 0.71363365650177, 0.6244479417800903, 0.6125363707542419, 0.7005191445350647, 0.3072779178619385, 0.03871360421180725, 0.22007469832897186, 0.9792734980583191, 0.7651551365852356, 0.734717845916748, 0.17870531976222992, 0.44417932629585266, 0.2179657369852066, 0.9591480493545532, 0.2803013026714325, 0.1185930147767067, 0.9058464765548706, 0.23474544286727905, 0.2004777193069458, 0.14858026802539825, 0.4179771840572357, 0.6740628480911255, 0.3323284685611725, 0.20192724466323853, 0.5708499550819397, 0.13826406002044678, 0.7149617671966553, 0.7548125386238098, 0.6021342277526855, 0.3239288330078125, 0.2019103616476059, 0.49418991804122925, 0.7900963425636292, 0.2114921510219574, 0.17470872402191162, 0.8754268884658813, 0.44253864884376526, 0.3208003342151642, 0.9398529529571533, 0.1835707128047943, 0.505932629108429, 0.43236038088798523, 0.03545856103301048, 0.1949261724948883, 0.5602925419807434, 0.8662565350532532, 0.6540554761886597, 0.3518853485584259, 0.9023789167404175, 0.7323793172836304, 0.7479748129844666, 0.6063048243522644, 0.26490920782089233}, {0.4805925488471985, 0.6176995038986206, 0.37009549140930176, 0.9717304706573486, 0.5992392301559448, 0.9299008250236511, 0.22602881491184235, 0.6527561545372009, 0.8279484510421753, 0.3949808180332184, 0.799735426902771, 0.7993429899215698, 0.2030566930770874, 0.9609346389770508, 0.3323745131492615, 0.9537368416786194, 0.15895314514636993, 0.1036812961101532, 0.19629165530204773, 0.049973100423812866, 0.7934573292732239, 0.4689372777938843, 0.23424474895000458, 0.5163378119468689, 0.3026841878890991, 0.9125207662582397, 0.8587938547134399, 0.6908925175666809, 0.26192760467529297, 0.6527308225631714, 0.9468443393707275, 0.8915044069290161, 0.24860014021396637, 0.08161803334951401, 0.5765559077262878, 0.3465227782726288, 0.5794121623039246, 0.3670407235622406, 0.5527188181877136, 0.25442591309547424, 0.09090276807546616, 0.49946942925453186, 0.47889670729637146, 0.7662227153778076, 0.18683108687400818, 0.7806288003921509, 0.5415175557136536, 0.940411388874054, 0.7263017296791077, 0.15430104732513428, 0.8942121267318726, 0.9738821387290955, 0.8821706771850586, 0.5083155632019043, 0.03350749984383583, 0.3810443580150604, 0.14233805239200592, 0.22209307551383972, 0.24576959013938904, 0.2402379810810089, 0.5460731983184814, 0.4424227774143219, 0.9715291261672974, 0.7175574898719788, 0.06994439661502838, 0.8732144832611084, 0.3041429817676544, 0.7328984141349792, 0.8363993763923645, 0.8405565619468689, 0.6568422913551331, 0.018616819754242897, 0.22845913469791412, 0.7127190232276917, 0.9507973194122314, 0.5990082025527954, 0.47518813610076904, 0.23598037660121918, 0.3314153254032135, 0.7921792268753052, 0.8234051465988159, 0.4924108386039734, 0.5986899137496948, 0.6458475589752197, 0.6401792764663696, 0.6197957396507263, 0.26491469144821167, 0.02457100711762905, 0.5389930605888367, 0.997097373008728, 0.16139596700668335, 0.09975637495517731, 0.14922624826431274, 0.10803976655006409, 0.5559287667274475, 0.7116973996162415, 0.17416150867938995, 0.10222446173429489, 0.0856432169675827, 0.6851887702941895}, {0.5608122944831848, 0.8228111267089844, 0.14694276452064514, 0.07198735326528549, 0.9314648509025574, 0.5973130464553833, 0.4540582001209259, 0.3352218568325043, 0.7127059698104858, 0.2344377338886261, 0.8749989867210388, 0.6520208120346069, 0.7861739993095398, 0.1904706507921219, 0.19710084795951843, 0.9783971905708313, 0.3401285707950592, 0.5163303017616272, 0.9383383393287659, 0.6432634592056274, 0.6097255945205688, 0.07491613179445267, 0.7072497606277466, 0.3093087375164032, 0.6535428166389465, 0.2605331838130951, 0.6253921389579773, 0.10642117261886597, 0.42923516035079956, 0.6420450210571289, 0.4382359981536865, 0.83758145570755, 0.2876797020435333, 0.3853752017021179, 0.10164391994476318, 0.3182119131088257, 0.7899256348609924, 0.634350597858429, 0.784081220626831, 0.5607271790504456, 0.30407869815826416, 0.9104309678077698, 0.49869996309280396, 0.09568176418542862, 0.38752129673957825, 0.09772376716136932, 0.16768932342529297, 0.09425563365221024, 0.8730323314666748, 0.17890675365924835, 0.12650254368782043, 0.25726062059402466, 0.803280770778656, 0.6589453816413879, 0.40022552013397217, 0.9641976356506348, 0.9841630458831787, 0.08615507185459137, 0.7960510849952698, 0.9494923949241638, 0.780400812625885, 0.6422445774078369, 0.43292874097824097, 0.027832472696900368, 0.8525819182395935, 0.046310339123010635, 0.6092625856399536, 0.12027700990438461, 0.16852040588855743, 0.7651022672653198, 0.0010202820412814617, 0.04971319064497948, 0.8596469163894653, 0.39432570338249207, 0.685102641582489, 0.008679522201418877, 0.3755892515182495, 0.9015069007873535, 0.03400658816099167, 0.7584099769592285, 0.9491848349571228, 0.7532780766487122, 0.37673473358154297, 0.980836033821106, 0.1814945936203003, 0.12652470171451569, 0.039899252355098724, 0.21517975628376007, 0.46724990010261536, 0.4699389636516571, 0.10227935761213303, 0.4348820447921753, 0.5499769449234009, 0.30904197692871094, 0.30609792470932007, 0.6139881014823914, 0.1548728197813034, 0.47339656949043274, 0.9338932633399963, 0.9768468141555786}, {0.2019481062889099, 0.3612610995769501, 0.06337283551692963, 0.26793697476387024, 0.6012983918190002, 0.7411116361618042, 0.4763004183769226, 0.8752572536468506, 0.17575831711292267, 0.8681660294532776, 0.5122919678688049, 0.4764805734157562, 0.756833553314209, 0.4697604775428772, 0.1558225303888321, 0.7100810408592224, 0.17567360401153564, 0.46066412329673767, 0.04013775661587715, 0.3975195288658142, 0.3185579478740692, 0.5660365223884583, 0.4977181553840637, 0.4050396978855133, 0.48822256922721863, 0.7618896961212158, 0.1936318725347519, 0.9157942533493042, 0.518952488899231, 0.016598857939243317, 0.24965247511863708, 0.12109313160181046, 0.634843111038208, 0.0012873758096247911, 0.4109083414077759, 0.6817615628242493, 0.21616137027740479, 0.5353949666023254, 0.6749654412269592, 0.2961445450782776, 0.6870783567428589, 0.2198120504617691, 0.23901855945587158, 0.029377739876508713, 0.718895435333252, 0.8108409643173218, 0.54631507396698, 0.6860823035240173, 0.3155878484249115, 0.1358986496925354, 0.5107421278953552, 0.909948468208313, 0.14473415911197662, 0.9265945553779602, 0.6827569007873535, 0.6967703104019165, 0.13836409151554108, 0.022223051637411118, 0.31734412908554077, 0.9180434942245483, 0.9870057106018066, 0.19377073645591736, 0.9369344115257263, 0.36083123087882996, 0.8716601133346558, 0.20369328558444977, 0.8964509963989258, 0.6264285445213318, 0.5613856911659241, 0.9419177770614624, 0.9745017290115356, 0.8677825331687927, 0.09700331836938858, 0.6203422546386719, 0.1371716856956482, 0.8134379982948303, 0.3961210250854492, 0.972701370716095, 0.9556578993797302, 0.33274558186531067, 0.29451197385787964, 0.20037347078323364, 0.6884726881980896, 0.9186597466468811, 0.9221906661987305, 0.13554206490516663, 0.7684119939804077, 0.596330463886261, 0.7598798274993896, 0.7603902816772461, 0.6220723390579224, 0.04281537979841232, 0.24082458019256592, 0.9018418788909912, 0.15737873315811157, 0.004464215599000454, 0.9255144000053406, 0.1585625559091568, 0.7041997909545898, 0.6779392957687378}, {0.38707754015922546, 0.6991764307022095, 0.031220946460962296, 0.9355536699295044, 0.07060004025697708, 0.4526466429233551, 0.21514780819416046, 0.07790858298540115, 0.39214614033699036, 0.7785907983779907, 0.07685552537441254, 0.6611282229423523, 0.6850208044052124, 0.1479388177394867, 0.6634746789932251, 0.22096681594848633, 0.015481440350413322, 0.9248581528663635, 0.5186747312545776, 0.29213985800743103, 0.04399670287966728, 0.18899346888065338, 0.02069493755698204, 0.8142536878585815, 0.4458308517932892, 0.12463606894016266, 0.7646692991256714, 0.5129691362380981, 0.9444654583930969, 0.44578683376312256, 0.9436551928520203, 0.7984576225280762, 0.5835269093513489, 0.8413272500038147, 0.20059289038181305, 0.07949797064065933, 0.48891010880470276, 0.9929640293121338, 0.04019181802868843, 0.8253743052482605, 0.7323775887489319, 0.1926773190498352, 0.5838334560394287, 0.6296576261520386, 0.7467824816703796, 0.16347666084766388, 0.6828420162200928, 0.9079146385192871, 0.659942090511322, 0.3501951992511749, 0.4288601279258728, 0.4981471598148346, 0.8540468215942383, 0.5130351781845093, 0.8099651336669922, 0.13951286673545837, 0.6590705513954163, 0.04272540286183357, 0.826856791973114, 0.19918961822986603, 0.15461137890815735, 0.05439728498458862, 0.4101182818412781, 0.8893437385559082, 0.7213159799575806, 0.4566374719142914, 0.41378799080848694, 0.38929158449172974, 0.8032532930374146, 0.5293999314308167, 0.11181586235761642, 0.5945460200309753, 0.3936794698238373, 0.8112318515777588, 0.7237597107887268, 0.43651753664016724, 0.6902892589569092, 0.02596358023583889, 0.0881131961941719, 0.711247444152832, 0.3556199073791504, 0.48205217719078064, 0.4355510175228119, 0.46016043424606323, 0.7328281402587891, 0.8578259944915771, 0.5417117476463318, 0.5670450329780579, 0.6618067622184753, 0.14057525992393494, 0.3603448271751404, 0.6784566640853882, 0.6657304763793945, 0.3449358344078064, 0.24346433579921722, 0.06464795768260956, 0.004207239486277103, 0.5994012355804443, 0.18101266026496887, 0.022362081333994865}, {0.1310233622789383, 0.32116150856018066, 0.8067367672920227, 0.986373245716095, 0.563761830329895, 0.08401428163051605, 0.8559728264808655, 0.46949824690818787, 0.251324325799942, 0.6799377202987671, 0.65970379114151, 0.4313027858734131, 0.38924190402030945, 0.02077053114771843, 0.7115076780319214, 0.8797368407249451, 0.9951778054237366, 0.861023485660553, 0.17400728166103363, 0.8873289823532104, 0.15333394706249237, 0.5469309687614441, 0.14305591583251953, 0.824630618095398, 0.9417279958724976, 0.06884448230266571, 0.9194891452789307, 0.5292588472366333, 0.17224907875061035, 0.35078275203704834, 0.12392814457416534, 0.049803782254457474, 0.0790693536400795, 0.7186753153800964, 0.4145142734050751, 0.14954763650894165, 0.9658435583114624, 0.08718273788690567, 0.9745304584503174, 0.25722596049308777, 0.5471940040588379, 0.11352989822626114, 0.5702816247940063, 0.9978790283203125, 0.04111994802951813, 0.08293822407722473, 0.7085795402526855, 0.16821707785129547, 0.26730599999427795, 0.37599608302116394, 0.8388813734054565, 0.3996025025844574, 0.9773322939872742, 0.020797239616513252, 0.9501233100891113, 0.25973936915397644, 0.8357649445533752, 0.01221761666238308, 0.5313512086868286, 0.4638078212738037, 0.0859764888882637, 0.4944900572299957, 0.864128589630127, 0.9032285809516907, 0.031720466911792755, 0.20819979906082153, 0.34166863560676575, 0.25629448890686035, 0.5056937336921692, 0.045619335025548935, 0.4861341416835785, 0.36317530274391174, 0.8792020082473755, 0.2591661214828491, 0.8309213519096375, 0.8955357074737549, 0.1821553111076355, 0.061361491680145264, 0.27760568261146545, 0.26821252703666687, 0.5457864999771118, 0.5487280488014221, 0.5656822919845581, 0.16751258075237274, 0.8664087653160095, 0.6316655278205872, 0.18921779096126556, 0.3262229561805725, 0.22105814516544342, 0.37535524368286133, 0.333699494600296, 0.2811618149280548, 0.3263235092163086, 0.7141526341438293, 0.8820355534553528, 0.702420711517334, 0.4613471031188965, 0.30873891711235046, 0.18047355115413666, 0.1084747239947319}, {0.203713059425354, 0.030051078647375107, 0.36720526218414307, 0.9049127101898193, 0.9261201620101929, 0.3897867202758789, 0.15351329743862152, 0.613680899143219, 0.889512836933136, 0.55252605676651, 0.14018507301807404, 0.20016774535179138, 0.04290188103914261, 0.0434691347181797, 0.5261167287826538, 0.6089645624160767, 0.8200280070304871, 0.277468204498291, 0.948272168636322, 0.5940170884132385, 0.5594348311424255, 0.5818033814430237, 0.449790358543396, 0.1974637657403946, 0.8265315294265747, 0.18733736872673035, 0.23212188482284546, 0.17898690700531006, 0.355380654335022, 0.004583743866533041, 0.1544857621192932, 0.5947272181510925, 0.708358883857727, 0.8491721749305725, 0.5993586182594299, 0.9591673612594604, 0.5861018896102905, 0.28725722432136536, 0.0001923816162161529, 0.4088585078716278, 0.04509885236620903, 0.5059739947319031, 0.5610738396644592, 0.8253386616706848, 0.26311907172203064, 0.514350950717926, 0.4414673149585724, 0.9467038512229919, 0.13057789206504822, 0.7574869990348816, 0.08775073289871216, 0.5135769844055176, 0.9481356143951416, 0.9145607948303223, 0.4730209410190582, 0.311116486787796, 0.1378886103630066, 0.020107632502913475, 0.7017074823379517, 0.9397321343421936, 0.614876389503479, 0.8921381235122681, 0.04278920963406563, 0.5460952520370483, 0.8098261952400208, 0.7062510251998901, 0.6463128924369812, 0.5050660371780396, 0.912402331829071, 0.11059024184942245, 0.10701595991849899, 0.7958484888076782, 0.06697463244199753, 0.750793993473053, 0.8012866973876953, 0.8350779414176941, 0.6249364614486694, 0.14943622052669525, 0.22321774065494537, 0.9642209410667419, 0.3090784251689911, 0.8313511610031128, 0.2526096999645233, 0.6650582551956177, 0.4419698417186737, 0.6579383015632629, 0.583787202835083, 0.013613412156701088, 0.916519045829773, 0.21890701353549957, 0.7847070097923279, 0.8841129541397095, 0.6995048522949219, 0.7221832871437073, 0.22751252353191376, 0.5702723264694214, 0.6043204069137573, 0.24272197484970093, 0.6732351183891296, 0.5729366540908813}, {0.1987040936946869, 0.18531407415866852, 0.06108985096216202, 0.7594457268714905, 0.35179951786994934, 0.8296968340873718, 0.24789772927761078, 0.3738710880279541, 0.4625121057033539, 0.19801846146583557, 0.45554205775260925, 0.12478072196245193, 0.006891844794154167, 0.15120930969715118, 0.8624886870384216, 0.3762774169445038, 0.8891832232475281, 0.5895211696624756, 0.12572361528873444, 0.09045357257127762, 0.16200019419193268, 0.004660500213503838, 0.3611336946487427, 0.22832611203193665, 0.37350648641586304, 0.38875311613082886, 0.2688930630683899, 0.2020975798368454, 0.5373716950416565, 0.46593931317329407, 0.3486972153186798, 0.4783938527107239, 0.8081241250038147, 0.3965831398963928, 0.6812594532966614, 0.8458773493766785, 0.3793649971485138, 0.5234466791152954, 0.2740677297115326, 0.5145995020866394, 0.4664902091026306, 0.38570836186408997, 0.3381737768650055, 0.9139106869697571, 0.7632412910461426, 0.8927696347236633, 0.1772886961698532, 0.6211885809898376, 0.6213788390159607, 0.4973439872264862, 0.0875844731926918, 0.06106642261147499, 0.5354315042495728, 0.9884769320487976, 0.6288526654243469, 0.28059765696525574, 0.5088022947311401, 0.07217085361480713, 0.020917564630508423, 0.9805091023445129, 0.10649102926254272, 0.25381478667259216, 0.6933073997497559, 0.036070775240659714, 0.06884606182575226, 0.35679954290390015, 0.5128742456436157, 0.41802507638931274, 0.8848441243171692, 0.946773886680603, 0.2429681271314621, 0.7762474417686462, 0.7003348469734192, 0.5547160506248474, 0.2989194691181183, 0.5027933716773987, 0.5709971189498901, 0.060202986001968384, 0.5789755582809448, 0.7767617106437683, 0.8995673060417175, 0.1900729387998581, 0.3587094247341156, 0.1555669754743576, 0.19232459366321564, 0.6928114891052246, 0.8743404746055603, 0.05820148065686226, 0.3667401075363159, 0.5000573396682739, 0.7757750749588013, 0.33597514033317566, 0.8901263475418091, 0.6660892367362976, 0.05394680052995682, 0.5871441960334778, 0.9167342782020569, 0.7135505080223083, 0.975352942943573, 0.1404605358839035}, {0.1910235583782196, 0.27171382308006287, 0.2207345813512802, 0.2660493552684784, 0.8301040530204773, 0.03579730540513992, 0.4777711033821106, 0.5072410106658936, 0.6387141346931458, 0.5669946074485779, 0.5531414747238159, 0.2644716799259186, 0.48109161853790283, 0.6652960777282715, 0.6233681440353394, 0.16012367606163025, 0.022944340482354164, 0.5525051951408386, 0.012449628673493862, 0.6420469284057617, 0.9395368695259094, 0.10584921389818192, 0.7801415920257568, 0.6606438159942627, 0.8295783400535583, 0.13487504422664642, 0.3945915102958679, 0.8002373576164246, 0.16747435927391052, 0.5082407593727112, 0.025640178471803665, 0.5096614956855774, 0.9918441772460938, 0.3275793492794037, 0.25617313385009766, 0.5753445029258728, 0.39321625232696533, 0.5457677245140076, 0.9548004269599915, 0.43658554553985596, 0.4019368290901184, 0.0646141842007637, 0.8713576197624207, 0.6788461208343506, 0.47440963983535767, 0.8135831356048584, 0.7160353660583496, 0.5865498781204224, 0.7480540871620178, 0.3897136151790619, 0.1520293802022934, 0.9273821115493774, 0.9158889651298523, 0.40813806653022766, 0.19725137948989868, 0.41660434007644653, 0.228731170296669, 0.9208574295043945, 0.2014274150133133, 0.3045908212661743, 0.22417908906936646, 0.0849263146519661, 0.24648630619049072, 0.017718801274895668, 0.590219259262085, 0.9568892121315002, 0.3025051951408386, 0.7171180248260498, 0.3990548551082611, 0.4693787395954132, 0.7672533988952637, 0.10872948914766312, 0.6158433556556702, 0.6344337463378906, 0.2672008275985718, 0.33576500415802, 0.8073689937591553, 0.3686894476413727, 0.8109161257743835, 0.8019362688064575, 0.6058884859085083, 0.8295654654502869, 0.433605819940567, 0.7302549481391907, 0.39536866545677185, 0.09723778069019318, 0.6966425776481628, 0.12999221682548523, 0.6061531901359558, 0.022503137588500977, 0.06316817551851273, 0.8068516850471497, 0.44912657141685486, 0.5415083169937134, 0.32099446654319763, 0.4920940697193146, 0.8401767611503601, 0.7211849689483643, 0.05808252468705177, 0.007888281717896461}, {0.05530144274234772, 0.7271120548248291, 0.910595178604126, 0.7126938700675964, 0.9176290035247803, 0.4507140815258026, 0.946645200252533, 0.12441853433847427, 0.23799976706504822, 0.6732068061828613, 0.6080780029296875, 0.8927719593048096, 0.21341562271118164, 0.7653282880783081, 0.02870149537920952, 0.7018741965293884, 0.9488718509674072, 0.7294560670852661, 0.19314351677894592, 0.5649876594543457, 0.4062367379665375, 0.7197444438934326, 0.2676071524620056, 0.0185216274112463, 0.007385922595858574, 0.011141958646476269, 0.8556477427482605, 0.8906378149986267, 0.7290172576904297, 0.56092369556427, 0.11234193295240402, 0.935699999332428, 0.9588406682014465, 0.8995216488838196, 0.7681945562362671, 0.1861700415611267, 0.8338334560394287, 0.23921741545200348, 0.27481523156166077, 0.17023372650146484, 0.541327953338623, 0.49500033259391785, 0.662282407283783, 0.005170723423361778, 0.5431965589523315, 0.3358807861804962, 0.19193926453590393, 0.17888064682483673, 0.5844208002090454, 0.8247349858283997, 0.8671711683273315, 0.29567989706993103, 0.48915377259254456, 0.1886328011751175, 0.13130146265029907, 0.06852158904075623, 0.5738430619239807, 0.2989768087863922, 0.8058149218559265, 0.22384682297706604, 0.8285248279571533, 0.5055390000343323, 0.7367054224014282, 0.2924460768699646, 0.8824690580368042, 0.42651689052581787, 0.5388357043266296, 0.8352993726730347, 0.9080399870872498, 0.9501482844352722, 0.7774039506912231, 0.7358170747756958, 0.6069437861442566, 0.989478349685669, 0.24323606491088867, 0.9239493012428284, 0.10819829255342484, 0.8305680155754089, 0.5840575098991394, 0.21056731045246124, 0.026024898514151573, 0.5529457330703735, 0.3293914496898651, 0.5305477380752563, 0.28101596236228943, 0.31148067116737366, 0.6753860116004944, 0.42048758268356323, 0.6028977632522583, 0.6091955900192261, 0.3396799564361572, 0.6811777949333191, 0.4548291862010956, 0.22226250171661377, 0.7574658393859863, 0.9758834838867188, 0.5849416851997375, 0.9100722670555115, 0.22811074554920197, 0.24126197397708893}, {0.447185218334198, 0.957986056804657, 0.5456486344337463, 0.03319203481078148, 0.6595658659934998, 0.2362939566373825, 0.9715515971183777, 0.8207545280456543, 0.4438553750514984, 0.73015296459198, 0.06409814208745956, 0.8426901698112488, 0.7818371057510376, 0.8705470561981201, 0.4739093780517578, 0.13068635761737823, 0.9807338714599609, 0.4518522024154663, 0.06876850128173828, 0.6146238446235657, 0.7723082900047302, 0.4840191900730133, 0.23571562767028809, 0.9847822189331055, 0.5658852458000183, 0.16956977546215057, 0.7540052533149719, 0.09523716568946838, 0.100498266518116, 0.11585643887519836, 0.5676707029342651, 0.6118067502975464, 0.2054060399532318, 0.5068644881248474, 0.8984256386756897, 0.7407627701759338, 0.8705689907073975, 0.5961953997612, 0.45879602432250977, 0.839585542678833, 0.5176472067832947, 0.9025437831878662, 0.3228546977043152, 0.2528785765171051, 0.7824758291244507, 0.6845203042030334, 0.91788250207901, 0.19246806204319, 0.9599406123161316, 0.9317463040351868, 0.26905927062034607, 0.20349858701229095, 0.27800554037094116, 0.2949255704879761, 0.1804707944393158, 0.484855055809021, 0.7410919070243835, 0.7312283515930176, 0.18344999849796295, 0.5221772193908691, 0.0015268897404894233, 0.20145250856876373, 0.9656309485435486, 0.12836797535419464, 0.8234991431236267, 0.8536091446876526, 0.9476038217544556, 0.3024127781391144, 0.31061094999313354, 0.7690089344978333, 0.4036925137042999, 0.8964243531227112, 0.8463389873504639, 0.30952340364456177, 0.5921138525009155, 0.8483530879020691, 0.8211317658424377, 0.3442842662334442, 0.40748685598373413, 0.8742014169692993, 0.23828385770320892, 0.8750678896903992, 0.5871320962905884, 0.23322246968746185, 0.9238437414169312, 0.6704738140106201, 0.21345862746238708, 0.1249181479215622, 0.32391881942749023, 0.8935784697532654, 0.5688027739524841, 0.20255951583385468, 0.06666794419288635, 0.7224587798118591, 0.9089962840080261, 0.16131290793418884, 0.08679793775081635, 0.2143070101737976, 0.6142333745956421, 0.1339259147644043}, {0.5931437015533447, 0.7102881669998169, 0.2997100055217743, 0.1278282254934311, 0.6628331542015076, 0.6566808223724365, 0.9356547594070435, 0.05888731777667999, 0.9917028546333313, 0.8875743746757507, 0.7484702467918396, 0.11099874973297119, 0.26545482873916626, 0.1796611100435257, 0.04132366180419922, 0.46221834421157837, 0.10009418427944183, 0.8493369221687317, 0.4213295876979828, 0.4360937774181366, 0.05149995535612106, 0.27588990330696106, 0.5770769715309143, 0.9145933389663696, 0.6688590049743652, 0.14958056807518005, 0.27238109707832336, 0.6620211005210876, 0.6155856251716614, 0.739147424697876, 0.3647395074367523, 0.9652958512306213, 0.15960802137851715, 0.31898608803749084, 0.8144330382347107, 0.5843861103057861, 0.506235659122467, 0.6592332720756531, 0.17141638696193695, 0.19410839676856995, 0.36198532581329346, 0.7649742960929871, 0.4261573851108551, 0.4180987775325775, 0.40882259607315063, 0.698726236820221, 0.6725974678993225, 0.9205314517021179, 0.753373384475708, 0.9454559683799744, 0.31683772802352905, 0.15815025568008423, 0.7707591652870178, 0.6543434262275696, 0.21403588354587555, 0.5322384238243103, 0.14143285155296326, 0.7182567119598389, 0.691658079624176, 0.6435924768447876, 0.863362193107605, 0.85633385181427, 0.49519652128219604, 0.9740386009216309, 0.6119422912597656, 0.6034422516822815, 0.2146466225385666, 0.403240442276001, 0.7603680491447449, 0.6544845104217529, 0.5782191753387451, 0.1604614406824112, 0.7985689640045166, 0.29622215032577515, 0.5443063974380493, 0.6332200765609741, 0.684611439704895, 0.25827690958976746, 0.311622679233551, 0.9165754914283752, 0.34690767526626587, 0.7574020028114319, 0.4501175582408905, 0.598983108997345, 0.8583037853240967, 0.5653478503227234, 0.5747735500335693, 0.42925775051116943, 0.250840961933136, 0.2398804873228073, 0.8277305960655212, 0.6433431506156921, 0.6899714469909668, 0.32874372601509094, 0.7466050386428833, 0.7633476853370667, 0.3711039423942566, 0.5082842111587524, 0.12174875289201736, 0.4646325707435608}, {0.511006772518158, 0.06876721978187561, 0.5120307207107544, 0.19308999180793762, 0.04775792360305786, 0.8294664621353149, 0.7310768961906433, 0.4645040035247803, 0.43267521262168884, 0.5331487059593201, 0.8795285820960999, 0.9096544981002808, 0.8989311456680298, 0.45550698041915894, 0.3341532051563263, 0.08541560173034668, 0.43655887246131897, 0.4735562801361084, 0.081952303647995, 0.5540134310722351, 0.19468000531196594, 0.047835834324359894, 0.23927436769008636, 0.5960198640823364, 0.20740114152431488, 0.18130819499492645, 0.9297575950622559, 0.5557599663734436, 0.6638069152832031, 0.22299134731292725, 0.5820881724357605, 0.28437623381614685, 0.4407089054584503, 0.7755880355834961, 0.4496702551841736, 0.15550586581230164, 0.8859315514564514, 0.22687743604183197, 0.8726135492324829, 0.31798961758613586, 0.31993770599365234, 0.5953575372695923, 0.6081650853157043, 0.0792054533958435, 0.0963219627737999, 0.1704714298248291, 0.7296199798583984, 0.29563283920288086, 0.8339181542396545, 0.6315314173698425, 0.7509730458259583, 0.21552105247974396, 0.5461100339889526, 0.8474245071411133, 0.47937050461769104, 0.5002043843269348, 0.06804344803094864, 0.6513932943344116, 0.930313766002655, 0.9393244981765747, 0.33756351470947266, 0.7019335031509399, 0.6185816526412964, 0.7481699585914612, 0.37594515085220337, 0.8948655724525452, 0.4101434350013733, 0.13552922010421753, 0.06342599540948868, 0.3624424338340759, 0.6125749945640564, 0.714440107345581, 0.05454593524336815, 0.4551225006580353, 0.11941055953502655, 0.690115213394165, 0.10527724027633667, 0.42748522758483887, 0.4321853518486023, 0.6142588257789612, 0.4039510190486908, 0.22808803617954254, 0.1006213054060936, 0.582197368144989, 0.12242820858955383, 0.6778723001480103, 0.3836289048194885, 0.8604351878166199, 0.45029130578041077, 0.6085888743400574, 0.047162219882011414, 0.30546632409095764, 0.158091738820076, 0.13270646333694458, 0.7709771990776062, 0.5806575417518616, 0.23374326527118683, 0.5859432816505432, 0.12486758828163147, 0.750441312789917}, {0.9988315105438232, 0.7645962834358215, 0.29095518589019775, 0.3068690001964569, 0.2840743362903595, 0.061836689710617065, 0.8479582071304321, 0.8779636025428772, 0.591180682182312, 0.39144906401634216, 0.19660882651805878, 0.9820630550384521, 0.16209790110588074, 0.06713416427373886, 0.6056057810783386, 0.5698900818824768, 0.9569679498672485, 0.04859880730509758, 0.007341356482356787, 0.719523012638092, 0.3127129077911377, 0.714714527130127, 0.04726765677332878, 0.44743621349334717, 0.9877123832702637, 0.3144803047180176, 0.0024519478902220726, 0.22409039735794067, 0.19514596462249756, 0.08934712409973145, 0.44314226508140564, 0.9854488968849182, 0.6654446125030518, 0.7124336361885071, 0.23100417852401733, 0.30378401279449463, 0.22792837023735046, 0.9941279888153076, 0.13175636529922485, 0.5498566627502441, 0.23494933545589447, 0.48947685956954956, 0.527993381023407, 0.7831432223320007, 0.9714057445526123, 0.9625798463821411, 0.6832454204559326, 0.46035274863243103, 0.34260764718055725, 0.3172914981842041, 0.7071313858032227, 0.5399917364120483, 0.1051691398024559, 0.17852051556110382, 0.6098248958587646, 0.5804650187492371, 0.7075319290161133, 0.8497997522354126, 0.9386456608772278, 0.7060829401016235, 0.4914359748363495, 0.24962520599365234, 0.4933547377586365, 0.06286262720823288, 0.8145366907119751, 0.6863649487495422, 0.16260036826133728, 0.5568529963493347, 0.18505072593688965, 0.4034208655357361, 0.07669398188591003, 0.7261297106742859, 0.6327236294746399, 0.4253079891204834, 0.43250301480293274, 0.6938682198524475, 0.6287171840667725, 0.17718464136123657, 0.8023736476898193, 0.6730828881263733, 0.4922649562358856, 0.8972353935241699, 0.6910529136657715, 0.3996196389198303, 0.7425448894500732, 0.28542134165763855, 0.7083870768547058, 0.9911081790924072, 0.5262545347213745, 0.11447422206401825, 0.958534836769104, 0.21692699193954468, 0.4770444631576538, 0.7886916995048523, 0.7913968563079834, 0.5726975798606873, 0.45508959889411926, 0.7106649279594421, 0.140527606010437, 0.08133057504892349}, {0.6497355699539185, 0.523222804069519, 0.11174159497022629, 0.4365151524543762, 0.015144011937081814, 0.3141362965106964, 0.6355296969413757, 0.4035469591617584, 0.8885217905044556, 0.35782432556152344, 0.23728586733341217, 0.8788866400718689, 0.7956506609916687, 0.004878146108239889, 0.9103752374649048, 0.16182348132133484, 0.819269061088562, 0.9215672016143799, 0.3665347397327423, 0.6633359789848328, 0.16911831498146057, 0.7929551601409912, 0.35382285714149475, 0.004415234550833702, 0.9507476687431335, 0.5443639755249023, 0.972912073135376, 0.3348185122013092, 0.26655182242393494, 0.8438148498535156, 0.7916200757026672, 0.645005464553833, 0.12733668088912964, 0.16621236503124237, 0.20611517131328583, 0.7128918170928955, 0.6900288462638855, 0.7115622758865356, 0.069595567882061, 0.603028416633606, 0.3692212700843811, 0.870029866695404, 0.3437451124191284, 0.6338562369346619, 0.5532525181770325, 0.11580723524093628, 0.9046177864074707, 0.5505442023277283, 0.10504744201898575, 0.5683056712150574, 0.20082493126392365, 0.6935276985168457, 0.8300307989120483, 0.5855787396430969, 0.03335832431912422, 0.5463894009590149, 0.81647789478302, 0.6274303197860718, 0.8885874152183533, 0.7137773036956787, 0.4510653018951416, 0.2514042854309082, 0.39674636721611023, 0.7088550329208374, 0.6536989212036133, 0.5736684203147888, 0.9806886315345764, 0.8380329012870789, 0.5687680244445801, 0.1519341766834259, 0.04282158613204956, 0.5942060351371765, 0.05551864579319954, 0.0008176119299605489, 0.038697462528944016, 0.8030350804328918, 0.8788918852806091, 0.08015654981136322, 0.9049162268638611, 0.1257176548242569, 0.9851559400558472, 0.9050185680389404, 0.9272173047065735, 0.6037102341651917, 0.24509188532829285, 0.1833731234073639, 0.6245072484016418, 0.3499372601509094, 0.9330610036849976, 0.3702946603298187, 0.08171267062425613, 0.13787773251533508, 0.5887185335159302, 0.159505233168602, 0.09714299440383911, 0.6671529412269592, 0.4307168424129486, 0.2915897071361542, 0.01957302913069725, 0.6595460772514343}, {0.7560347318649292, 0.6349931955337524, 0.698276162147522, 0.4811500012874603, 0.9368686676025391, 0.3337898552417755, 0.3737758994102478, 0.37011492252349854, 0.6650868058204651, 0.5157383680343628, 0.8076391220092773, 0.7876760959625244, 0.45995089411735535, 0.7804495096206665, 0.2216019332408905, 0.9526638388633728, 0.026992609724402428, 0.7910482883453369, 0.12384802848100662, 0.0189931970089674, 0.803188681602478, 0.62701815366745, 0.1864989548921585, 0.2073862999677658, 0.43881967663764954, 0.048489779233932495, 0.4911876618862152, 0.8582486510276794, 0.9862342476844788, 0.47482243180274963, 0.8895458579063416, 0.3652207553386688, 0.18278633058071136, 0.1734907329082489, 0.5646872520446777, 0.969428539276123, 0.8380942940711975, 0.4695698618888855, 0.39354369044303894, 0.1622227430343628, 0.8541335463523865, 0.29929301142692566, 0.9151666164398193, 0.4771066904067993, 0.1883545070886612, 0.5839971899986267, 0.4372914731502533, 0.4043077826499939, 0.4942218065261841, 0.7141488194465637, 0.44265633821487427, 0.7667447328567505, 0.8949857950210571, 0.5365610718727112, 0.1674881875514984, 0.30933263897895813, 0.07782074809074402, 0.7296300530433655, 0.45989277958869934, 0.9145051836967468, 0.5992981791496277, 0.2162010222673416, 0.8562962412834167, 0.006777230184525251, 0.40916070342063904, 0.10637838393449783, 0.6111181378364563, 0.4018363058567047, 0.80140620470047, 0.6737348437309265, 0.4621225595474243, 0.5497329831123352, 0.41112127900123596, 0.9272579550743103, 0.3433573544025421, 0.41379499435424805, 0.21402107179164886, 0.6806241273880005, 0.978752076625824, 0.13956348598003387, 0.04763571918010712, 0.3437883257865906, 0.7198847532272339, 0.5345651507377625, 0.11957389861345291, 0.4075027108192444, 0.878709077835083, 0.44298291206359863, 0.6240983009338379, 0.5807770490646362, 0.4813080132007599, 0.00008291609992738813, 0.011496546678245068, 0.787674069404602, 0.808801531791687, 0.6182231903076172, 0.7958360314369202, 0.920871913433075, 0.2229558825492859, 0.7620885372161865}, {0.5273054838180542, 0.23880575597286224, 0.6495642066001892, 0.5667009353637695, 0.7504971027374268, 0.4803762435913086, 0.647822380065918, 0.5951032042503357, 0.5999670624732971, 0.17980647087097168, 0.7345944046974182, 0.7523949146270752, 0.5415397882461548, 0.8367573022842407, 0.7353087663650513, 0.010968959890305996, 0.5874217748641968, 0.7901731133460999, 0.4621632993221283, 0.10656796395778656, 0.520067036151886, 0.013465840369462967, 0.6416769623756409, 0.2567861080169678, 0.9953027963638306, 0.8595380187034607, 0.691215991973877, 0.26950615644454956, 0.8738836050033569, 0.8941329121589661, 0.601468563079834, 0.8844290375709534, 0.5694302320480347, 0.19314493238925934, 0.7185813188552856, 0.012578734196722507, 0.5329726338386536, 0.5097934603691101, 0.4493393898010254, 0.5752313733100891, 0.049286291003227234, 0.8363551497459412, 0.14202024042606354, 0.4279206693172455, 0.5686853528022766, 0.12315385043621063, 0.5251691937446594, 0.5090492367744446, 0.1791701316833496, 0.5312484502792358, 0.34833285212516785, 0.3756336271762848, 0.9519469738006592, 0.5719631910324097, 0.06247405707836151, 0.08952147513628006, 0.5930898189544678, 0.04614151269197464, 0.7298902273178101, 0.38375791907310486, 0.8093978762626648, 0.9253642559051514, 0.4020037353038788, 0.2923700213432312, 0.38838300108909607, 0.061429478228092194, 0.9350501298904419, 0.47799089550971985, 0.6438892483711243, 0.437648743391037, 0.8375800251960754, 0.1286148726940155, 0.38503339886665344, 0.2700354754924774, 0.3749243915081024, 0.2421485334634781, 0.6444838047027588, 0.7084568738937378, 0.9807375073432922, 0.06300947815179825, 0.7218493223190308, 0.7436344027519226, 0.6353691816329956, 0.730961263179779, 0.7119470238685608, 0.03455229103565216, 0.027255814522504807, 0.10420186817646027, 0.9755312204360962, 0.6131666898727417, 0.8025416731834412, 0.26613759994506836, 0.5842165350914001, 0.6602858304977417, 0.7811252474784851, 0.31027156114578247, 0.5511500835418701, 0.20083032548427582, 0.22479400038719177, 0.9208678603172302}, {0.1694798618555069, 0.4639839828014374, 0.5311039090156555, 0.09139436483383179, 0.4215473532676697, 0.6142869591712952, 0.23828087747097015, 0.7615440487861633, 0.07181668281555176, 0.58255535364151, 0.1679871678352356, 0.7580776810646057, 0.2894994616508484, 0.048011414706707, 0.8445659279823303, 0.07269074767827988, 0.767451286315918, 0.664056658744812, 0.7978405952453613, 0.8412659168243408, 0.600256085395813, 0.8767250776290894, 0.2737347483634949, 0.44783440232276917, 0.47018808126449585, 0.8643421530723572, 0.07629325240850449, 0.16661445796489716, 0.5507834553718567, 0.1358954757452011, 0.08032883703708649, 0.5666871666908264, 0.8205505609512329, 0.7035539746284485, 0.5043708086013794, 0.4953373968601227, 0.66404128074646, 0.2898591458797455, 0.7813056707382202, 0.6646125912666321, 0.8486656546592712, 0.5804712176322937, 0.6387900710105896, 0.9810224175453186, 0.8913968205451965, 0.3320448398590088, 0.4666062295436859, 0.7446502447128296, 0.8582072257995605, 0.9254496693611145, 0.18810692429542542, 0.9431211352348328, 0.7100058197975159, 0.8411544561386108, 0.9638139605522156, 0.6065285205841064, 0.44948840141296387, 0.38291609287261963, 0.5491603016853333, 0.7593693137168884, 0.6973544955253601, 0.15274909138679504, 0.6355476975440979, 0.2427242547273636, 0.6790069341659546, 0.6819877028465271, 0.02739827334880829, 0.32695579528808594, 0.7938126921653748, 0.8019931316375732, 0.1725137084722519, 0.9285465478897095, 0.2878226041793823, 0.6499788761138916, 0.1920720338821411, 0.08280213177204132, 0.543632984161377, 0.9566694498062134, 0.35402026772499084, 0.4624477028846741, 0.4455188810825348, 0.7415528297424316, 0.04775431007146835, 0.40454405546188354, 0.23080308735370636, 0.014125424437224865, 0.13622862100601196, 0.8711129426956177, 0.590283215045929, 0.045783888548612595, 0.6064056754112244, 0.5285868644714355, 0.3509146571159363, 0.13000427186489105, 0.06132850795984268, 0.585504949092865, 0.1195906400680542, 0.5189818739891052, 0.45520785450935364, 0.47380176186561584}, {0.6885837316513062, 0.49500536918640137, 0.9933855533599854, 0.678404688835144, 0.47547584772109985, 0.04684554040431976, 0.0630081295967102, 0.9649627208709717, 0.6481660008430481, 0.5126325488090515, 0.9241519570350647, 0.8140473961830139, 0.6672980189323425, 0.474899023771286, 0.934022843837738, 0.8461536169052124, 0.048001017421483994, 0.8318262696266174, 0.8628321886062622, 0.19735611975193024, 0.9445045590400696, 0.4182133674621582, 0.4789014458656311, 0.9368205666542053, 0.8141730427742004, 0.8270572423934937, 0.9521230459213257, 0.5313376188278198, 0.7636146545410156, 0.35163217782974243, 0.9014521837234497, 0.7603784799575806, 0.9074994325637817, 0.1114935576915741, 0.8095224499702454, 0.8628612160682678, 0.502771258354187, 0.671380341053009, 0.7994767427444458, 0.7073444724082947, 0.5566127896308899, 0.10298504680395126, 0.023168563842773438, 0.835052490234375, 0.7200594544410706, 0.5878618955612183, 0.46974116563796997, 0.9147006869316101, 0.8438852429389954, 0.7512280941009521, 0.43468841910362244, 0.648441731929779, 0.0002594148681964725, 0.8140532374382019, 0.06415899097919464, 0.17589014768600464, 0.4935045838356018, 0.8740968704223633, 0.6405996084213257, 0.04556046053767204, 0.8428670763969421, 0.23536919057369232, 0.008623150177299976, 0.51240074634552, 0.9799628853797913, 0.342094361782074, 0.01959780976176262, 0.06927702575922012, 0.8516906499862671, 0.9736594557762146, 0.9174665808677673, 0.07575518637895584, 0.5601450204849243, 0.8869425058364868, 0.029856162145733833, 0.02600720338523388, 0.4817926585674286, 0.4268462657928467, 0.15033085644245148, 0.5897210836410522, 0.5672061443328857, 0.622981071472168, 0.9132732152938843, 0.5371285080909729, 0.9309543371200562, 0.192472904920578, 0.5543726086616516, 0.0025015606079250574, 0.9064236283302307, 0.604295015335083, 0.5412107110023499, 0.45053359866142273, 0.8858124017715454, 0.4232361614704132, 0.6188766956329346, 0.6520261168479919, 0.14339035749435425, 0.34920892119407654, 0.8894157409667969, 0.24864481389522552}, {0.11400621384382248, 0.145504891872406, 0.3506294786930084, 0.35885071754455566, 0.28297290205955505, 0.7491649389266968, 0.1476455181837082, 0.40177977085113525, 0.4848446547985077, 0.3328503966331482, 0.8112652897834778, 0.376180499792099, 0.04910377785563469, 0.33905404806137085, 0.9867806434631348, 0.6937202215194702, 0.5961165428161621, 0.7211424708366394, 0.24016132950782776, 0.9386672973632812, 0.6127721667289734, 0.6037468910217285, 0.0008842581883072853, 0.4452945590019226, 0.631016194820404, 0.5557343363761902, 0.09218093752861023, 0.6985546946525574, 0.2313876748085022, 0.5821072459220886, 0.33775225281715393, 0.24016128480434418, 0.7678771615028381, 0.8450220227241516, 0.4210148751735687, 0.8870720267295837, 0.5874729156494141, 0.9188179969787598, 0.5137609839439392, 0.5629927515983582, 0.7715153694152832, 0.7905055284500122, 0.653403103351593, 0.4458867013454437, 0.16394633054733276, 0.6759013533592224, 0.5199586749076843, 0.2791358530521393, 0.956183135509491, 0.9257779121398926, 0.4209250807762146, 0.6497026681900024, 0.9800785183906555, 0.19670267403125763, 0.7832313776016235, 0.07144254446029663, 0.11271888017654419, 0.25429484248161316, 0.8279443383216858, 0.018443206325173378, 0.7258605360984802, 0.6219306588172913, 0.6804946064949036, 0.8472003936767578, 0.7278024554252625, 0.4021965563297272, 0.42144715785980225, 0.23739072680473328, 0.06564025580883026, 0.23421497642993927, 0.28779107332229614, 0.8166157603263855, 0.6625746488571167, 0.8208054304122925, 0.45254313945770264, 0.1088947281241417, 0.7825917601585388, 0.5923108458518982, 0.15123462677001953, 0.17247864603996277, 0.7228696942329407, 0.11185480654239655, 0.11677075177431107, 0.7849201560020447, 0.25699377059936523, 0.517275333404541, 0.9185616374015808, 0.3170076012611389, 0.7146527767181396, 0.9827267527580261, 0.41587862372398376, 0.8693369030952454, 0.3517988324165344, 0.41305872797966003, 0.2180488556623459, 0.3886841833591461, 0.7888103127479553, 0.058192573487758636, 0.6596737504005432, 0.16645175218582153}, {0.025417232885956764, 0.620983362197876, 0.5539821982383728, 0.6788197159767151, 0.8417611718177795, 0.947536289691925, 0.3622831702232361, 0.7387121319770813, 0.5905662178993225, 0.575042724609375, 0.8260439038276672, 0.8418689966201782, 0.3870845437049866, 0.8085604310035706, 0.07659320533275604, 0.9472149014472961, 0.9814899563789368, 0.5115296840667725, 0.15330781042575836, 0.7132315039634705, 0.44982948899269104, 0.6160332560539246, 0.7025760412216187, 0.3304601311683655, 0.5147131681442261, 0.9439350962638855, 0.9362180829048157, 0.58160001039505, 0.7544974088668823, 0.6659405827522278, 0.18788939714431763, 0.3450984060764313, 0.42658624053001404, 0.30595386028289795, 0.8552803993225098, 0.915450394153595, 0.7433164119720459, 0.3979160487651825, 0.9576917290687561, 0.002380070509389043, 0.8707732558250427, 0.7106258273124695, 0.7194613218307495, 0.6676790118217468, 0.6916623711585999, 0.06985928118228912, 0.05796324461698532, 0.9942561984062195, 0.021905040368437767, 0.0230819471180439, 0.8873934745788574, 0.12038577347993851, 0.08967164903879166, 0.8962299823760986, 0.2346850484609604, 0.5013400912284851, 0.5983027815818787, 0.6462899446487427, 0.6020112037658691, 0.055368997156620026, 0.1754530370235443, 0.7872975468635559, 0.6937152147293091, 0.2636842131614685, 0.29585814476013184, 0.04457187280058861, 0.9806501865386963, 0.064211405813694, 0.21424801647663116, 0.8112451434135437, 0.47328057885169983, 0.33983299136161804, 0.048429224640131, 0.7746549248695374, 0.35551753640174866, 0.7162760496139526, 0.47150009870529175, 0.7885903716087341, 0.8403775095939636, 0.41785773634910583, 0.6202826499938965, 0.3563929498195648, 0.16573019325733185, 0.8369887471199036, 0.7358266115188599, 0.5599601864814758, 0.652204155921936, 0.5501561164855957, 0.8377059698104858, 0.7697288393974304, 0.6740833520889282, 0.7578181624412537, 0.9222432374954224, 0.7743422389030457, 0.6786575317382812, 0.9433718919754028, 0.957919180393219, 0.7003893256187439, 0.7308017611503601, 0.14502601325511932}, {0.11834470927715302, 0.33884039521217346, 0.7150226831436157, 0.4252205193042755, 0.7127344012260437, 0.16733025014400482, 0.7935177683830261, 0.9956992268562317, 0.8512057662010193, 0.01830049604177475, 0.8651475310325623, 0.9442574977874756, 0.09565708041191101, 0.4986472725868225, 0.7491505146026611, 0.8789665699005127, 0.300680011510849, 0.023247821256518364, 0.12367037683725357, 0.4577449560165405, 0.8657178282737732, 0.9328337907791138, 0.1348627209663391, 0.41244617104530334, 0.10376254469156265, 0.7275498509407043, 0.8368779420852661, 0.9521287083625793, 0.5530375242233276, 0.7373003363609314, 0.9496724605560303, 0.6888864636421204, 0.3370972275733948, 0.8342723250389099, 0.9196969866752625, 0.5205362439155579, 0.7222334146499634, 0.73009192943573, 0.4528106451034546, 0.8620093464851379, 0.4412810802459717, 0.7260866165161133, 0.7257246375083923, 0.5653448104858398, 0.43026623129844666, 0.7545133233070374, 0.1533631831407547, 0.4149850308895111, 0.31938403844833374, 0.48351356387138367, 0.15569597482681274, 0.21354861557483673, 0.7573505640029907, 0.9974389672279358, 0.7995071411132812, 0.6573175191879272, 0.43882808089256287, 0.5964236855506897, 0.25414687395095825, 0.624243438243866, 0.2743018567562103, 0.6703637838363647, 0.7671326994895935, 0.759296178817749, 0.005682635121047497, 0.3792482912540436, 0.04774241894483566, 0.27707815170288086, 0.1845826804637909, 0.015701638534665108, 0.7268204689025879, 0.2747085690498352, 0.8155105710029602, 0.7939951419830322, 0.051854051649570465, 0.8016386032104492, 0.08714408427476883, 0.13208633661270142, 0.38134634494781494, 0.29484036564826965, 0.2836010158061981, 0.5799383521080017, 0.07791610062122345, 0.8772507905960083, 0.20511990785598755, 0.7294049859046936, 0.10061013698577881, 0.41378822922706604, 0.9407204389572144, 0.8679502010345459, 0.183552086353302, 0.7522937655448914, 0.5385565161705017, 0.6447160840034485, 0.3114221692085266, 0.3420221507549286, 0.5127866864204407, 0.271940678358078, 0.06522350013256073, 0.06615909934043884}, {0.31043410301208496, 0.3979641795158386, 0.8381514549255371, 0.6936491131782532, 0.4956310987472534, 0.9755268692970276, 0.5333825945854187, 0.7012447118759155, 0.9838182926177979, 0.2280806601047516, 0.6239445209503174, 0.8963527679443359, 0.6536149978637695, 0.13083121180534363, 0.17780020833015442, 0.7544528841972351, 0.5662034749984741, 0.037671420723199844, 0.33517971634864807, 0.9823244214057922, 0.1606079488992691, 0.8428531885147095, 0.3465433716773987, 0.043940410017967224, 0.7935824990272522, 0.5842010974884033, 0.7608495950698853, 0.4579290449619293, 0.31424596905708313, 0.4369213283061981, 0.40148910880088806, 0.9363804459571838, 0.43713974952697754, 0.1384768933057785, 0.8222399353981018, 0.5572617650032043, 0.9878867864608765, 0.7751359939575195, 0.9268811345100403, 0.9593201875686646, 0.6339653134346008, 0.41788628697395325, 0.21475651860237122, 0.4751668870449066, 0.5075746774673462, 0.6132748126983643, 0.7060683369636536, 0.5913534164428711, 0.06648284941911697, 0.17000555992126465, 0.5755090117454529, 0.45134639739990234, 0.46642258763313293, 0.5027194619178772, 0.13167083263397217, 0.9587186574935913, 0.8007139563560486, 0.1310611367225647, 0.6394619941711426, 0.0895518884062767, 0.6560859680175781, 0.20181317627429962, 0.3700504004955292, 0.873408854007721, 0.30875536799430847, 0.888131320476532, 0.4802393615245819, 0.8955504298210144, 0.6738505363464355, 0.7800882458686829, 0.7955507636070251, 0.7022992968559265, 0.7215733528137207, 0.40134745836257935, 0.7012724280357361, 0.7878970503807068, 0.4984067976474762, 0.09573663026094437, 0.8556573390960693, 0.39252063632011414, 0.19656290113925934, 0.21537716686725616, 0.34260010719299316, 0.42762434482574463, 0.44015035033226013, 0.37124010920524597, 0.9699124693870544, 0.6331822872161865, 0.39444679021835327, 0.791615903377533, 0.30543795228004456, 0.692534327507019, 0.6845219135284424, 0.22904038429260254, 0.6432438492774963, 0.2104688137769699, 0.11120326071977615, 0.6700856685638428, 0.32053887844085693, 0.8069648742675781}, {0.5174455046653748, 0.959958016872406, 0.5215731859207153, 0.8625986576080322, 0.2700743079185486, 0.085242860019207, 0.47265711426734924, 0.1399795264005661, 0.33746978640556335, 0.21726270020008087, 0.024815842509269714, 0.3752287030220032, 0.30032673478126526, 0.8028284311294556, 0.7124292254447937, 0.921504557132721, 0.48007485270500183, 0.4643864333629608, 0.57660311460495, 0.9113354682922363, 0.34430381655693054, 0.4753338694572449, 0.7028676867485046, 0.28038495779037476, 0.955989420413971, 0.406637042760849, 0.17824792861938477, 0.2650201916694641, 0.643518328666687, 0.4002816081047058, 0.9151740074157715, 0.6411314010620117, 0.14411519467830658, 0.9862844944000244, 0.6989493370056152, 0.41837364435195923, 0.4803382158279419, 0.6144672632217407, 0.04748142510652542, 0.6609766483306885, 0.44573795795440674, 0.212447389960289, 0.5122035145759583, 0.7148959636688232, 0.8331843614578247, 0.25709599256515503, 0.9856310486793518, 0.6925711631774902, 0.9833210706710815, 0.920362114906311, 0.018493346869945526, 0.1544635146856308, 0.13819515705108643, 0.5039133429527283, 0.8253834843635559, 0.9114052057266235, 0.014856401830911636, 0.35634303092956543, 0.4020405113697052, 0.821709394454956, 0.49534162878990173, 0.2664909362792969, 0.9573602080345154, 0.6255002021789551, 0.06658537685871124, 0.9135561585426331, 0.7856625914573669, 0.9124006032943726, 0.3228016197681427, 0.6919296979904175, 0.6377193927764893, 0.17703138291835785, 0.9906007647514343, 0.3098744750022888, 0.9029309153556824, 0.9437844157218933, 0.5588991641998291, 0.2724247872829437, 0.5555070638656616, 0.06931928545236588, 0.11254611611366272, 0.04924456402659416, 0.825491189956665, 0.7936512231826782, 0.7771626114845276, 0.07551362365484238, 0.3153068423271179, 0.7892391681671143, 0.7196085453033447, 0.2690378427505493, 0.20096290111541748, 0.10474888980388641, 0.6456130743026733, 0.2849133610725403, 0.24812859296798706, 0.3896292448043823, 0.027382178232073784, 0.9813103675842285, 0.2692278325557709, 0.24578799307346344}, {0.796266496181488, 0.9802514910697937, 0.38952866196632385, 0.0991261675953865, 0.7470022439956665, 0.11544182896614075, 0.49765047430992126, 0.1016063243150711, 0.3498392105102539, 0.03288369998335838, 0.23715488612651825, 0.7111051082611084, 0.006095172371715307, 0.9696851372718811, 0.8586773872375488, 0.001133460900746286, 0.12436748296022415, 0.10649237781763077, 0.8473679423332214, 0.22656944394111633, 0.34731173515319824, 0.8464759588241577, 0.1603446900844574, 0.16120021045207977, 0.9091988205909729, 0.4977646768093109, 0.540672242641449, 0.5412381887435913, 0.3234426975250244, 0.027903271839022636, 0.8114923238754272, 0.6626760363578796, 0.8161131739616394, 0.15129749476909637, 0.03553321957588196, 0.43381819128990173, 0.6526907682418823, 0.12387150526046753, 0.2694791257381439, 0.8115826845169067, 0.486405611038208, 0.5221582055091858, 0.4974115490913391, 0.10131438076496124, 0.23662284016609192, 0.585547149181366, 0.6869416832923889, 0.5253491997718811, 0.5322105288505554, 0.6853898763656616, 0.5060189366340637, 0.8621134757995605, 0.14151647686958313, 0.9139993786811829, 0.5527997016906738, 0.9636402726173401, 0.6162083148956299, 0.04332923889160156, 0.3020201623439789, 0.20074042677879333, 0.1865852028131485, 0.6997145414352417, 0.2937614917755127, 0.09290372580289841, 0.429829865694046, 0.03497288376092911, 0.19787350296974182, 0.7957434058189392, 0.38434067368507385, 0.059338394552469254, 0.6539903283119202, 0.653531551361084, 0.3480033576488495, 0.389704167842865, 0.03290458396077156, 0.6040077209472656, 0.7800953984260559, 0.25321611762046814, 0.8801187872886658, 0.14883477985858917, 0.0464131236076355, 0.6077160239219666, 0.896359920501709, 0.6455170512199402, 0.006741800345480442, 0.3767032325267792, 0.240129292011261, 0.8458374738693237, 0.13474610447883606, 0.6323390603065491, 0.348861962556839, 0.03627726435661316, 0.4541807174682617, 0.6668374538421631, 0.2661376893520355, 0.28864729404449463, 0.19447563588619232, 0.8601717352867126, 0.8337750434875488, 0.39518046379089355}, {0.7683135867118835, 0.4467967450618744, 0.7209311723709106, 0.08886203914880753, 0.8204885125160217, 0.7900793552398682, 0.8037480115890503, 0.6542628407478333, 0.8418726325035095, 0.0693759098649025, 0.739061713218689, 0.7672305703163147, 0.9343883395195007, 0.944317638874054, 0.19015145301818848, 0.798242449760437, 0.7429177165031433, 0.745745062828064, 0.1687740832567215, 0.7809441685676575, 0.598270058631897, 0.6382607817649841, 0.2800266444683075, 0.318792462348938, 0.5369765758514404, 0.3807527720928192, 0.2783655822277069, 0.4366516172885895, 0.08452113717794418, 0.8312243223190308, 0.6509167551994324, 0.9336540699005127, 0.5139922499656677, 0.26961904764175415, 0.15827058255672455, 0.38791680335998535, 0.5689923763275146, 0.5250937342643738, 0.7858083844184875, 0.8717655539512634, 0.6060531735420227, 0.6324065327644348, 0.4510239064693451, 0.9915573596954346, 0.06684329360723495, 0.2996287941932678, 0.9704886674880981, 0.2228364497423172, 0.9837080240249634, 0.21899059414863586, 0.6341888904571533, 0.7528236508369446, 0.8545634746551514, 0.13432016968727112, 0.6012569665908813, 0.013611915521323681, 0.7059687972068787, 0.9609417915344238, 0.8825558423995972, 0.8066909313201904, 0.9231472611427307, 0.21332837641239166, 0.3055015802383423, 0.07858734577894211, 0.3855588138103485, 0.33346304297447205, 0.7361993193626404, 0.3598192632198334, 0.3940706253051758, 0.09708789736032486, 0.4093974828720093, 0.08773148059844971, 0.32596462965011597, 0.5549593567848206, 0.2499341517686844, 0.8993669748306274, 0.6234959363937378, 0.4983392655849457, 0.7595569491386414, 0.8138284087181091, 0.6554362177848816, 0.11748315393924713, 0.22216428816318512, 0.7098351716995239, 0.9487951993942261, 0.7955620288848877, 0.1487574577331543, 0.13407090306282043, 0.1489202380180359, 0.046169959008693695, 0.5138800144195557, 0.5663359761238098, 0.6525959372520447, 0.22647804021835327, 0.536349892616272, 0.7874571681022644, 0.5020016431808472, 0.8985573649406433, 0.4502190351486206, 0.5363384485244751}, {0.6914162039756775, 0.4162414073944092, 0.46406030654907227, 0.026196379214525223, 0.8365702033042908, 0.9496849179267883, 0.4612734019756317, 0.5025965571403503, 0.9334456324577332, 0.14071136713027954, 0.06000338867306709, 0.4881596565246582, 0.4450588524341583, 0.8490552306175232, 0.7211502194404602, 0.41033196449279785, 0.8786165714263916, 0.13400442898273468, 0.9787914156913757, 0.6659945249557495, 0.6068630814552307, 0.4576738178730011, 0.08295581489801407, 0.32120630145072937, 0.7498784065246582, 0.1641625165939331, 0.18816596269607544, 0.34818020462989807, 0.05627519264817238, 0.8882521390914917, 0.43160369992256165, 0.012078835628926754, 0.33241963386535645, 0.18199630081653595, 0.5027449727058411, 0.9779037237167358, 0.9167682528495789, 0.04165055230259895, 0.18150624632835388, 0.4472573399543762, 0.763591468334198, 0.6393184065818787, 0.29149600863456726, 0.13403426110744476, 0.427232950925827, 0.2079927772283554, 0.2586407959461212, 0.3497679531574249, 0.9131360650062561, 0.39036986231803894, 0.0536833219230175, 0.8608385324478149, 0.5237226486206055, 0.03103596717119217, 0.6338827610015869, 0.7326480150222778, 0.43017756938934326, 0.3713567852973938, 0.1512337327003479, 0.8042488098144531, 0.07318507879972458, 0.31232088804244995, 0.2704313099384308, 0.08631569147109985, 0.8041930198669434, 0.7995191812515259, 0.33323559165000916, 0.9133736491203308, 0.11849309504032135, 0.48246103525161743, 0.6253606081008911, 0.9090041518211365, 0.9141589999198914, 0.5523567795753479, 0.03820831701159477, 0.17171168327331543, 0.23286132514476776, 0.34142231941223145, 0.3311779797077179, 0.31404900550842285, 0.13508696854114532, 0.8317050933837891, 0.12283110618591309, 0.5692837834358215, 0.2048599272966385, 0.9997655749320984, 0.6596394777297974, 0.7778358459472656, 0.3293771743774414, 0.8738430142402649, 0.544151782989502, 0.6942222714424133, 0.965241014957428, 0.4924386739730835, 0.042553387582302094, 0.6573638319969177, 0.7986736297607422, 0.004743683151900768, 0.501398503780365, 0.3270946443080902}, {0.2100549042224884, 0.9260525703430176, 0.8333266973495483, 0.7961695194244385, 0.7146905660629272, 0.21689893305301666, 0.7216064929962158, 0.8455479741096497, 0.36019188165664673, 0.6127209067344666, 0.49685171246528625, 0.8336087465286255, 0.792407751083374, 0.28876376152038574, 0.10698619484901428, 0.25959980487823486, 0.7765545845031738, 0.47729822993278503, 0.715366780757904, 0.8217611908912659, 0.2687438726425171, 0.8425638675689697, 0.3524155914783478, 0.4156681299209595, 0.5174520611763, 0.6980339884757996, 0.47441619634628296, 0.24193048477172852, 0.6685543656349182, 0.0753251314163208, 0.0874369740486145, 0.8231339454650879, 0.2802465558052063, 0.8855010867118835, 0.08742929995059967, 0.36416125297546387, 0.9225411415100098, 0.8414165377616882, 0.026388300582766533, 0.9943731427192688, 0.4735241234302521, 0.09553007781505585, 0.9647760391235352, 0.23515576124191284, 0.34979912638664246, 0.5925520062446594, 0.3534857928752899, 0.5194159150123596, 0.6150535941123962, 0.48131054639816284, 0.35889580845832825, 0.3142683506011963, 0.29952529072761536, 0.3773176968097687, 0.0014566525351256132, 0.5821366310119629, 0.9116823077201843, 0.6854826807975769, 0.6255808472633362, 0.46910086274147034, 0.6303512454032898, 0.19800496101379395, 0.40521249175071716, 0.18035747110843658, 0.43357762694358826, 0.11006312817335129, 0.3843478262424469, 0.6548348069190979, 0.4920342266559601, 0.2862855792045593, 0.27325373888015747, 0.14805056154727936, 0.2282734513282776, 0.563881516456604, 0.051563065499067307, 0.6041623950004578, 0.4168775975704193, 0.23466169834136963, 0.5882894992828369, 0.03417642042040825, 0.542110800743103, 0.7279455065727234, 0.7285488247871399, 0.21956032514572144, 0.06662648916244507, 0.709381639957428, 0.8932725191116333, 0.44523364305496216, 0.2955029308795929, 0.4960254430770874, 0.9935190081596375, 0.20228590071201324, 0.1160857155919075, 0.9004771113395691, 0.7021127939224243, 0.09534712880849838, 0.6054244041442871, 0.19659627974033356, 0.086133673787117, 0.1884736269712448}, {0.836663544178009, 0.4405445456504822, 0.0901079997420311, 0.44957295060157776, 0.7708223462104797, 0.6340857148170471, 0.9327208995819092, 0.9040667414665222, 0.9958588480949402, 0.8999373912811279, 0.6019712090492249, 0.5551685690879822, 0.5252341628074646, 0.8982097506523132, 0.2814246118068695, 0.28044605255126953, 0.9627223014831543, 0.4819345474243164, 0.8926771283149719, 0.5378909111022949, 0.3173401355743408, 0.954271137714386, 0.011017524637281895, 0.9562666416168213, 0.31172940135002136, 0.5733271837234497, 0.5067759156227112, 0.08074730634689331, 0.5195608139038086, 0.580113410949707, 0.8019164800643921, 0.33395907282829285, 0.7984516620635986, 0.09458810836076736, 0.13778303563594818, 0.6733376383781433, 0.2023189812898636, 0.20247595012187958, 0.4073857367038727, 0.5209575891494751, 0.9972323775291443, 0.8387810587882996, 0.16482529044151306, 0.9318526387214661, 0.8852670788764954, 0.7199199199676514, 0.27084672451019287, 0.28411269187927246, 0.5580671429634094, 0.8631928563117981, 0.5795823335647583, 0.14356261491775513, 0.5871624946594238, 0.3972599506378174, 0.2315228134393692, 0.7890099883079529, 0.411332905292511, 0.1832154095172882, 0.9916802048683167, 0.2590462267398834, 0.3631932735443115, 0.4793214201927185, 0.5352311730384827, 0.9234924912452698, 0.9890021681785583, 0.16355949640274048, 0.942465603351593, 0.9141693711280823, 0.5551697611808777, 0.2865661084651947, 0.32520920038223267, 0.14185631275177002, 0.6782444715499878, 0.08530837297439575, 0.7040428519248962, 0.545569896697998, 0.7452887892723083, 0.5764236450195312, 0.5267749428749084, 0.39537325501441956, 0.39574605226516724, 0.5987535715103149, 0.40221503376960754, 0.6093694567680359, 0.3457992672920227, 0.11639995127916336, 0.3334348201751709, 0.6510035991668701, 0.859326183795929, 0.2717879116535187, 0.5536321997642517, 0.2863025367259979, 0.6848488450050354, 0.9474583864212036, 0.98430335521698, 0.6843729615211487, 0.6079942584037781, 0.5413530468940735, 0.008251667022705078, 0.8240056037902832}, {0.556083619594574, 0.14566338062286377, 0.3043536841869354, 0.7234845757484436, 0.8915777206420898, 0.7296901941299438, 0.7679494619369507, 0.48546719551086426, 0.9666142463684082, 0.7742387652397156, 0.7836774587631226, 0.007376166060566902, 0.5703810453414917, 0.8490275740623474, 0.9485040307044983, 0.867432713508606, 0.8129174709320068, 0.802193820476532, 0.7400637269020081, 0.7047954201698303, 0.8371273875236511, 0.13279476761817932, 0.2154397964477539, 0.8435242176055908, 0.975001871585846, 0.8482493758201599, 0.4922219216823578, 0.04508412256836891, 0.6217865943908691, 0.19077813625335693, 0.5837372541427612, 0.40514233708381653, 0.5511481761932373, 0.5649366974830627, 0.614956259727478, 0.6610972285270691, 0.08914285153150558, 0.34535571932792664, 0.06938028335571289, 0.22592614591121674, 0.10648037493228912, 0.7924330234527588, 0.16248680651187897, 0.4380905032157898, 0.6775497198104858, 0.6534051299095154, 0.9079463481903076, 0.5694157481193542, 0.8870503306388855, 0.7869403958320618, 0.5697813630104065, 0.009897342883050442, 0.8766713738441467, 0.07199881970882416, 0.9201414585113525, 0.4857734441757202, 0.9647899866104126, 0.11348306387662888, 0.32453399896621704, 0.3238021731376648, 0.4145611524581909, 0.04152785241603851, 0.49523410201072693, 0.5433606505393982, 0.9012097716331482, 0.3255856931209564, 0.6495660543441772, 0.21126937866210938, 0.4318779706954956, 0.6442093253135681, 0.07226289808750153, 0.5253344178199768, 0.750930905342102, 0.7875571846961975, 0.1313270777463913, 0.5584706664085388, 0.495930016040802, 0.10054633766412735, 0.6495108604431152, 0.6735780835151672, 0.6421399116516113, 0.7743564248085022, 0.1716993898153305, 0.09732035547494888, 0.11913185566663742, 0.34496280550956726, 0.1607237458229065, 0.732466459274292, 0.1104101836681366, 0.25708791613578796, 0.7216579914093018, 0.14130637049674988, 0.6526004672050476, 0.13451255857944489, 0.20373794436454773, 0.5061658024787903, 0.030580295249819756, 0.30172643065452576, 0.8931977152824402, 0.6881164908409119}, {0.787592887878418, 0.7188746333122253, 0.1996520459651947, 0.5352486968040466, 0.9500007629394531, 0.17431139945983887, 0.3950878977775574, 0.482393741607666, 0.22119194269180298, 0.7763543128967285, 0.5568438172340393, 0.6378023624420166, 0.18176215887069702, 0.9049457311630249, 0.5808089375495911, 0.12448780238628387, 0.7072580456733704, 0.804695188999176, 0.9411383867263794, 0.569033145904541, 0.3749973773956299, 0.9076200127601624, 0.7035728693008423, 0.8510226607322693, 0.576229453086853, 0.03629384934902191, 0.8663265109062195, 0.27075180411338806, 0.622336208820343, 0.9394669532775879, 0.5193740725517273, 0.29889634251594543, 0.028451835736632347, 0.866653561592102, 0.7943951487541199, 0.5465312600135803, 0.584577739238739, 0.2996719181537628, 0.06246146932244301, 0.04328402131795883, 0.733457624912262, 0.6029285788536072, 0.6423769593238831, 0.13315127789974213, 0.8401905298233032, 0.10762294381856918, 0.6236782670021057, 0.49757474660873413, 0.8764604926109314, 0.7052114009857178, 0.5133391618728638, 0.6323850750923157, 0.7270814776420593, 0.11370887607336044, 0.39568009972572327, 0.21864262223243713, 0.2604110836982727, 0.16194495558738708, 0.13752451539039612, 0.8511731624603271, 0.38754183053970337, 0.42854589223861694, 0.667555570602417, 0.3131106495857239, 0.024478476494550705, 0.9207180738449097, 0.9964471459388733, 0.9381166696548462, 0.2357109785079956, 0.5101838707923889, 0.3753202557563782, 0.27003273367881775, 0.8363941311836243, 0.211525559425354, 0.45282191038131714, 0.9728847742080688, 0.4503418207168579, 0.6728585362434387, 0.38597121834754944, 0.9775869250297546, 0.46641525626182556, 0.021836090832948685, 0.8383750319480896, 0.7679964900016785, 0.5566791296005249, 0.02524004876613617, 0.8046467900276184, 0.9724910259246826, 0.9631719589233398, 0.907706618309021, 0.16772551834583282, 0.1432197242975235, 0.34005072712898254, 0.8244016766548157, 0.955520749092102, 0.8306767344474792, 0.7648820877075195, 0.36542513966560364, 0.29360711574554443, 0.7226501107215881}, {0.14155592024326324, 0.06060977280139923, 0.9815828204154968, 0.6861420273780823, 0.28286072611808777, 0.3392033278942108, 0.12353997677564621, 0.4324064254760742, 0.7871613502502441, 0.19919036328792572, 0.22222083806991577, 0.44866713881492615, 0.2512964606285095, 0.9574357867240906, 0.3688198924064636, 0.0563482865691185, 0.5640382766723633, 0.9549880027770996, 0.39817243814468384, 0.5808753967285156, 0.7199984788894653, 0.6903790831565857, 0.8588517904281616, 0.9115021824836731, 0.45580971240997314, 0.5666255354881287, 0.9116129279136658, 0.24773597717285156, 0.42333945631980896, 0.9276114106178284, 0.6371695399284363, 0.9561874270439148, 0.4314892888069153, 0.9086529612541199, 0.9894225597381592, 0.36580899357795715, 0.03203289955854416, 0.07419649511575699, 0.16224460303783417, 0.5870652198791504, 0.45294705033302307, 0.4342857599258423, 0.77982097864151, 0.8371729254722595, 0.1681458055973053, 0.2563546597957611, 0.518419623374939, 0.05737947300076485, 0.7626123428344727, 0.18394263088703156, 0.453747421503067, 0.48749008774757385, 0.009302228689193726, 0.6386236548423767, 0.40229129791259766, 0.9201799035072327, 0.34251487255096436, 0.8623901009559631, 0.006721556186676025, 0.3502238988876343, 0.719072699546814, 0.6003347635269165, 0.9111028909683228, 0.7421655654907227, 0.7875926494598389, 0.4295656085014343, 0.7101811170578003, 0.5574104189872742, 0.12871800363063812, 0.9680020213127136, 0.47811359167099, 0.5658937692642212, 0.36349016427993774, 0.9945728778839111, 0.5872743129730225, 0.5629680156707764, 0.45220041275024414, 0.20780996978282928, 0.7555148601531982, 0.7541857957839966, 0.6977891325950623, 0.00488283159211278, 0.31715813279151917, 0.04101909324526787, 0.7069714665412903, 0.5173729658126831, 0.49568837881088257, 0.6628625392913818, 0.5995065569877625, 0.4995691776275635, 0.059222374111413956, 0.4878484606742859, 0.9685561060905457, 0.8820717930793762, 0.8904341459274292, 0.6445629000663757, 0.9393415451049805, 0.34687334299087524, 0.3420271575450897, 0.41719090938568115}, {0.1413537561893463, 0.9056728482246399, 0.4898204803466797, 0.04384532570838928, 0.879233717918396, 0.4859065115451813, 0.43591928482055664, 0.4350864589214325, 0.46110042929649353, 0.18104007840156555, 0.6140700578689575, 0.1164470836520195, 0.20372726023197174, 0.5201537013053894, 0.5599218010902405, 0.7059187293052673, 0.3528705835342407, 0.5384339690208435, 0.37223801016807556, 0.8965438008308411, 0.3629581332206726, 0.8819565773010254, 0.24938364326953888, 0.5563995838165283, 0.2200939953327179, 0.9875203967094421, 0.3498041331768036, 0.5672976970672607, 0.829636812210083, 0.6193680167198181, 0.3321326673030853, 0.10807505249977112, 0.889105498790741, 0.37436339259147644, 0.670508086681366, 0.5686106085777283, 0.4539405107498169, 0.09951566159725189, 0.8557170033454895, 0.9920673966407776, 0.39211171865463257, 0.995987057685852, 0.9571957588195801, 0.020088674500584602, 0.17990683019161224, 0.5132585167884827, 0.2997666299343109, 0.5160889029502869, 0.08232139050960541, 0.8163530826568604, 0.3688730299472809, 0.8003644943237305, 0.39237114787101746, 0.3672555983066559, 0.03338511660695076, 0.1302611231803894, 0.4016442596912384, 0.4369523525238037, 0.12625911831855774, 0.6808580756187439, 0.7376728653907776, 0.11185403168201447, 0.39602628350257874, 0.9941487908363342, 0.4976791441440582, 0.172040656208992, 0.9130340814590454, 0.24900420010089874, 0.7946408987045288, 0.9522771239280701, 0.13824668526649475, 0.6259419918060303, 0.8358351588249207, 0.35349735617637634, 0.45006176829338074, 0.9800636172294617, 0.37188073992729187, 0.5291212797164917, 0.832054853439331, 0.19352248311042786, 0.7681031823158264, 0.11514360457658768, 0.8189542889595032, 0.6707820296287537, 0.8546247482299805, 0.12335191667079926, 0.8826188445091248, 0.16526129841804504, 0.31078705191612244, 0.06967891752719879, 0.47689858078956604, 0.276754230260849, 0.43297111988067627, 0.1413356065750122, 0.2269197404384613, 0.3861427307128906, 0.602509617805481, 0.13808652758598328, 0.042964085936546326, 0.21165715157985687}, {0.8243165612220764, 0.4040775001049042, 0.3653409779071808, 0.8738712668418884, 0.5655478835105896, 0.6542575359344482, 0.8297359347343445, 0.4631984829902649, 0.8695665001869202, 0.9575228095054626, 0.9976337552070618, 0.24881528317928314, 0.7529560327529907, 0.8959143161773682, 0.5882912874221802, 0.8363702893257141, 0.3358704447746277, 0.6614288091659546, 0.3089337944984436, 0.06103619933128357, 0.8106440305709839, 0.13986100256443024, 0.36281079053878784, 0.437237411737442, 0.15410122275352478, 0.511448323726654, 0.03404289484024048, 0.8913771510124207, 0.7037756443023682, 0.29300764203071594, 0.2915967106819153, 0.8599153161048889, 0.8498225212097168, 0.2640744149684906, 0.11695009469985962, 0.704507052898407, 0.2992148995399475, 0.7396494746208191, 0.38494524359703064, 0.7189334034919739, 0.684609055519104, 0.4110352694988251, 0.14437265694141388, 0.764681339263916, 0.32006508111953735, 0.5430744886398315, 0.5239596962928772, 0.9549211263656616, 0.7713897228240967, 0.8313105702400208, 0.7905556559562683, 0.06283167749643326, 0.7561139464378357, 0.4384011924266815, 0.5607253909111023, 0.07881417125463486, 0.8125118017196655, 0.324238121509552, 0.20758861303329468, 0.1255612075328827, 0.3493596911430359, 0.7048447728157043, 0.09806827455759048, 0.4159861207008362, 0.010237818583846092, 0.17666731774806976, 0.08807570487260818, 0.04632222652435303, 0.4148326516151428, 0.5588500499725342, 0.12715817987918854, 0.2287842482328415, 0.46517592668533325, 0.5985735058784485, 0.8841381669044495, 0.56207674741745, 0.06185440346598625, 0.31349167227745056, 0.3813042640686035, 0.3810180425643921, 0.6242833733558655, 0.8339076638221741, 0.9130973219871521, 0.6353113651275635, 0.4211704730987549, 0.23803205788135529, 0.5795015096664429, 0.09256013482809067, 0.11524529755115509, 0.5795172452926636, 0.37286534905433655, 0.8020954728126526, 0.49902820587158203, 0.4980633556842804, 0.6152986288070679, 0.5136394500732422, 0.2900548279285431, 0.5090078115463257, 0.0528903603553772, 0.7776461243629456}, {0.26759907603263855, 0.36111336946487427, 0.6326896548271179, 0.7551198601722717, 0.039769247174263, 0.9177481532096863, 0.30173543095588684, 0.30074191093444824, 0.18547208607196808, 0.8386496901512146, 0.851697564125061, 0.6179203987121582, 0.2849622368812561, 0.9217984080314636, 0.9764697551727295, 0.4563383460044861, 0.853249192237854, 0.6191803812980652, 0.5948131680488586, 0.9008748531341553, 0.6585341691970825, 0.4183898866176605, 0.9973099827766418, 0.5841593146324158, 0.40708568692207336, 0.9597992897033691, 0.739307165145874, 0.5140644311904907, 0.024012679234147072, 0.6770198345184326, 0.5159120559692383, 0.9086056351661682, 0.23652881383895874, 0.25204524397850037, 0.6830810904502869, 0.2777850031852722, 0.5036585330963135, 0.9118419885635376, 0.9214751124382019, 0.35666152834892273, 0.342715859413147, 0.2324320524930954, 0.9888944029808044, 0.9812262654304504, 0.9126930832862854, 0.7062551975250244, 0.11765933036804199, 0.7413220405578613, 0.01482295524328947, 0.30056455731391907, 0.5297303199768066, 0.18459278345108032, 0.44405484199523926, 0.3991678059101105, 0.3904634416103363, 0.4590246379375458, 0.4684603810310364, 0.4277491867542267, 0.5287622809410095, 0.3647441864013672, 0.7478206753730774, 0.18151719868183136, 0.5367976427078247, 0.42587584257125854, 0.9322801232337952, 0.51788330078125, 0.5746427774429321, 0.5190488696098328, 0.3044835329055786, 0.4787488877773285, 0.4717384874820709, 0.8280866742134094, 0.24351812899112701, 0.5611842274665833, 0.30979788303375244, 0.1423601508140564, 0.8501372337341309, 0.7566111087799072, 0.7986920475959778, 0.06837454438209534, 0.9672983884811401, 0.6308451294898987, 0.6842554211616516, 0.7736214995384216, 0.670595109462738, 0.573204755783081, 0.9798648953437805, 0.8169849514961243, 0.7792700529098511, 0.9362886548042297, 0.559410572052002, 0.31632012128829956, 0.8965279459953308, 0.7420526146888733, 0.7666410207748413, 0.8200274109840393, 0.7755001783370972, 0.5139145255088806, 0.6050055027008057, 0.08579540252685547}, {0.17199008166790009, 0.2729889452457428, 0.4003038704395294, 0.848581075668335, 0.29590722918510437, 0.8397247791290283, 0.08492492139339447, 0.5994902849197388, 0.7316491603851318, 0.6829419136047363, 0.5501359701156616, 0.9514467716217041, 0.29280388355255127, 0.8935670852661133, 0.2818497121334076, 0.4061644673347473, 0.5433705449104309, 0.8495242595672607, 0.5501057505607605, 0.28008896112442017, 0.03345249965786934, 0.6579172015190125, 0.2854972779750824, 0.5052046179771423, 0.002221782924607396, 0.8880664706230164, 0.20059673488140106, 0.5017002820968628, 0.8379815220832825, 0.08396245539188385, 0.8935261964797974, 0.4195918142795563, 0.9237565994262695, 0.8227137327194214, 0.29137328267097473, 0.9892493486404419, 0.36760130524635315, 0.4056285321712494, 0.6931344270706177, 0.9744377732276917, 0.6119393110275269, 0.43800708651542664, 0.19773706793785095, 0.016079794615507126, 0.732807993888855, 0.40373459458351135, 0.6137261986732483, 0.06648702174425125, 0.5830211639404297, 0.07810509204864502, 0.9147099852561951, 0.7862226366996765, 0.03423835337162018, 0.8333142995834351, 0.749413788318634, 0.7890707850456238, 0.42947742342948914, 0.1441902071237564, 0.42301666736602783, 0.23161636292934418, 0.18918168544769287, 0.318136602640152, 0.8399116396903992, 0.487649142742157, 0.3498924672603607, 0.28173893690109253, 0.27071496844291687, 0.9316331148147583, 0.9875667691230774, 0.27020829916000366, 0.437795490026474, 0.6519505977630615, 0.13711489737033844, 0.6060636043548584, 0.2594422399997711, 0.3663896918296814, 0.719039261341095, 0.6239022612571716, 0.6108048558235168, 0.2507883608341217, 0.30337393283843994, 0.23432131111621857, 0.10030467808246613, 0.8293048143386841, 0.8410565853118896, 0.47624632716178894, 0.5126296877861023, 0.411122590303421, 0.7904766798019409, 0.06415171176195145, 0.18904651701450348, 0.11840229481458664, 0.7440508008003235, 0.4923805892467499, 0.7604817152023315, 0.7779489755630493, 0.5150029063224792, 0.8093478679656982, 0.8628500699996948, 0.4537159502506256}, {0.19606363773345947, 0.3374220132827759, 0.9775075912475586, 0.8947319388389587, 0.39380207657814026, 0.6819313764572144, 0.7483308911323547, 0.2794169783592224, 0.5689557194709778, 0.2529100179672241, 0.9019365310668945, 0.6747399568557739, 0.09853711724281311, 0.9677171111106873, 0.6734975576400757, 0.8641301393508911, 0.617077648639679, 0.06494744122028351, 0.16896501183509827, 0.535082221031189, 0.3341187536716461, 0.23184850811958313, 0.8398587107658386, 0.028849568217992783, 0.12132741510868073, 0.8314917683601379, 0.0010152936447411776, 0.41824471950531006, 0.9531431794166565, 0.10758047550916672, 0.23539197444915771, 0.4328833520412445, 0.7436730265617371, 0.17113298177719116, 0.3570489287376404, 0.6158034801483154, 0.4558212459087372, 0.4489699900150299, 0.7624384164810181, 0.637627899646759, 0.6096276640892029, 0.25730621814727783, 0.21178671717643738, 0.5151058435440063, 0.13386797904968262, 0.711639404296875, 0.5537627935409546, 0.18573226034641266, 0.13935692608356476, 0.39088675379753113, 0.14853590726852417, 0.07288725674152374, 0.33274203538894653, 0.421047180891037, 0.6596972346305847, 0.7536218762397766, 0.027018915861845016, 0.8326999545097351, 0.2580997049808502, 0.24609148502349854, 0.09144620597362518, 0.396666020154953, 0.06260095536708832, 0.9196990132331848, 0.9592710733413696, 0.7855684161186218, 0.7069084644317627, 0.4602094888687134, 0.17087717354297638, 0.28414419293403625, 0.097154401242733, 0.60986328125, 0.5125375390052795, 0.7934327721595764, 0.6886681318283081, 0.9007153511047363, 0.7580180168151855, 0.5372734665870667, 0.6454928517341614, 0.2539330720901489, 0.34985536336898804, 0.9613982439041138, 0.28449296951293945, 0.7219562530517578, 0.11190818250179291, 0.951907217502594, 0.8941603302955627, 0.7137942314147949, 0.9682047367095947, 0.7008830308914185, 0.24655094742774963, 0.2589736878871918, 0.8274102807044983, 0.37742140889167786, 0.684840202331543, 0.06633323431015015, 0.4981558322906494, 0.6688414812088013, 0.6621981263160706, 0.49518635869026184}, {0.09236115962266922, 0.6239190697669983, 0.7386054396629333, 0.21533632278442383, 0.028824569657444954, 0.031671494245529175, 0.12590019404888153, 0.5987670421600342, 0.23268969357013702, 0.43795257806777954, 0.17523504793643951, 0.15036529302597046, 0.45161473751068115, 0.10790638625621796, 0.5158116817474365, 0.8016271591186523, 0.21852929890155792, 0.3214733600616455, 0.2580438554286957, 0.2637481689453125, 0.22245974838733673, 0.5865628123283386, 0.6533939838409424, 0.8526061177253723, 0.0624411515891552, 0.5613331198692322, 0.9489996433258057, 0.7099622488021851, 0.38454338908195496, 0.46459564566612244, 0.8497446179389954, 0.7837724685668945, 0.9123365879058838, 0.8369404077529907, 0.11505761742591858, 0.06011728197336197, 0.30725857615470886, 0.42742571234703064, 0.8794317245483398, 0.059186920523643494, 0.12195423990488052, 0.7575886249542236, 0.09155698120594025, 0.9234821796417236, 0.09405463188886642, 0.22374989092350006, 0.16621364653110504, 0.7455990314483643, 0.00503816781565547, 0.6825342178344727, 0.012163662351667881, 0.9669622182846069, 0.8248226642608643, 0.9647466540336609, 0.37282800674438477, 0.764762282371521, 0.9717637300491333, 0.9539538025856018, 0.4023198187351227, 0.30286985635757446, 0.5786043405532837, 0.532256007194519, 0.9057134985923767, 0.5583181977272034, 0.2889764904975891, 0.5511046051979065, 0.6587949991226196, 0.01286377478390932, 0.9826763868331909, 0.016945011913776398, 0.13358740508556366, 0.14620831608772278, 0.758479118347168, 0.6013891696929932, 0.2611987888813019, 0.9868791699409485, 0.8204458355903625, 0.052397970110177994, 0.774472177028656, 0.46547970175743103, 0.7860342860221863, 0.721764862537384, 0.2747557461261749, 0.21310478448867798, 0.6029403209686279, 0.2479294091463089, 0.3824285566806793, 0.7821750044822693, 0.8894227147102356, 0.7072851061820984, 0.6873480677604675, 0.7266681790351868, 0.6359025239944458, 0.6439433097839355, 0.655292272567749, 0.3502329885959625, 0.7407116293907166, 0.5088387131690979, 0.7302759885787964, 0.9014037847518921}, {0.731414258480072, 0.9334530234336853, 0.2359258532524109, 0.9440563917160034, 0.9708468317985535, 0.5970478653907776, 0.23398122191429138, 0.6469970941543579, 0.4251619875431061, 0.4002675712108612, 0.4102783501148224, 0.3512488901615143, 0.6871764659881592, 0.6774683594703674, 0.6036772727966309, 0.17552420496940613, 0.39100441336631775, 0.7662838101387024, 0.654205858707428, 0.8470414876937866, 0.07201942056417465, 0.6923518180847168, 0.24811747670173645, 0.8304961919784546, 0.8340694308280945, 0.13809488713741302, 0.21692833304405212, 0.8806612491607666, 0.23853859305381775, 0.6357868909835815, 0.13814347982406616, 0.14993314445018768, 0.40925902128219604, 0.9182149171829224, 0.4497792422771454, 0.4541441798210144, 0.04981611669063568, 0.7501295208930969, 0.9127313494682312, 0.8952535390853882, 0.30498453974723816, 0.161485955119133, 0.18883058428764343, 0.20248974859714508, 0.9022048115730286, 0.7268747091293335, 0.04948447644710541, 0.9568425416946411, 0.6254112720489502, 0.7366415858268738, 0.9605888724327087, 0.04484103247523308, 0.3473818600177765, 0.13557815551757812, 0.06534092873334885, 0.2732900083065033, 0.24419543147087097, 0.20981214940547943, 0.966583251953125, 0.7847681045532227, 0.3770136833190918, 0.6922388076782227, 0.14030416309833527, 0.3221625089645386, 0.7696771025657654, 0.8997540473937988, 0.25558584928512573, 0.8662858605384827, 0.49565625190734863, 0.5952226519584656, 0.4211944341659546, 0.44690847396850586, 0.2636381983757019, 0.1586342304944992, 0.4007141590118408, 0.1362641602754593, 0.20565538108348846, 0.3360057473182678, 0.6976172924041748, 0.46261727809906006, 0.5617144107818604, 0.9924648404121399, 0.5779439210891724, 0.43572497367858887, 0.32728320360183716, 0.5077146887779236, 0.9095508456230164, 0.05315050855278969, 0.020695816725492477, 0.09200097620487213, 0.7213810086250305, 0.17609694600105286, 0.29534444212913513, 0.6199882626533508, 0.5877922773361206, 0.8273991942405701, 0.7197951674461365, 0.16267013549804688, 0.15695373713970184, 0.13835649192333221}, {0.8660687208175659, 0.918748140335083, 0.37107211351394653, 0.6864343285560608, 0.145445317029953, 0.031184328719973564, 0.23590289056301117, 0.5852715969085693, 0.347278356552124, 0.43163806200027466, 0.04095000773668289, 0.12495334446430206, 0.9722039103507996, 0.6413149237632751, 0.897767186164856, 0.9626724720001221, 0.5297466516494751, 0.0873122587800026, 0.2756384313106537, 0.1432923823595047, 0.8019281029701233, 0.7801145315170288, 0.708441972732544, 0.4676275849342346, 0.8036195039749146, 0.9541142582893372, 0.25765693187713623, 0.7689487934112549, 0.10653459280729294, 0.7031818628311157, 0.5669011473655701, 0.8241578936576843, 0.6773345470428467, 0.9346283674240112, 0.9132937788963318, 0.5635434985160828, 0.9537054300308228, 0.8414809107780457, 0.0023338713217526674, 0.037152279168367386, 0.9083033800125122, 0.29095327854156494, 0.6257956624031067, 0.36338239908218384, 0.6570227146148682, 0.0925031453371048, 0.06956884264945984, 0.8004862666130066, 0.2725171744823456, 0.9012083411216736, 0.6137814521789551, 0.9179283976554871, 0.5226031541824341, 0.730994701385498, 0.16327881813049316, 0.5271397233009338, 0.2727085053920746, 0.13623249530792236, 0.9252714514732361, 0.3899156153202057, 0.9529449939727783, 0.36465296149253845, 0.0957120954990387, 0.4854334890842438, 0.7046589255332947, 0.4825678765773773, 0.3845832347869873, 0.3337637484073639, 0.6539276242256165, 0.15191441774368286, 0.6240987181663513, 0.947928786277771, 0.25220203399658203, 0.7671258449554443, 0.2568807303905487, 0.46913522481918335, 0.1309828907251358, 0.1839682012796402, 0.5514726638793945, 0.720498263835907, 0.5829481482505798, 0.49384254217147827, 0.9516421556472778, 0.5867346525192261, 0.23322616517543793, 0.3664129674434662, 0.9541621804237366, 0.20923230051994324, 0.5321515202522278, 0.8878437280654907, 0.16849198937416077, 0.41307690739631653, 0.4596122205257416, 0.060531970113515854, 0.1431862711906433, 0.034839384257793427, 0.3416479825973511, 0.8061874508857727, 0.5108311772346497, 0.31919246912002563}, {0.907950758934021, 0.6422431468963623, 0.36014339327812195, 0.3192790150642395, 0.1545218974351883, 0.5831043124198914, 0.5158838629722595, 0.4825049638748169, 0.5793740749359131, 0.15120573341846466, 0.21666200459003448, 0.9189265966415405, 0.7497401237487793, 0.2854611575603485, 0.2599989175796509, 0.5175271034240723, 0.8232319951057434, 0.49642041325569153, 0.04498201981186867, 0.2081480473279953, 0.7542617321014404, 0.8051735162734985, 0.8779100179672241, 0.29033225774765015, 0.7455236315727234, 0.5513888001441956, 0.33955174684524536, 0.4942331612110138, 0.19852390885353088, 0.6396212577819824, 0.8535749316215515, 0.29115140438079834, 0.9211691617965698, 0.866234540939331, 0.26731371879577637, 0.1153823658823967, 0.2408566027879715, 0.4949384033679962, 0.6470156311988831, 0.8174494504928589, 0.08145632594823837, 0.485723078250885, 0.9354276657104492, 0.45567819476127625, 0.5905316472053528, 0.7002815008163452, 0.4234677255153656, 0.09844592213630676, 0.5310843586921692, 0.18418294191360474, 0.8564209342002869, 0.44862934947013855, 0.23495544493198395, 0.6761331558227539, 0.9086601734161377, 0.61086106300354, 0.16527682542800903, 0.6115967035293579, 0.3902755379676819, 0.747016191482544, 0.35608118772506714, 0.8915562033653259, 0.12424710392951965, 0.637059211730957, 0.31705546379089355, 0.9675114750862122, 0.192810520529747, 0.6598767042160034, 0.7885512709617615, 0.18027235567569733, 0.6198890209197998, 0.7667322158813477, 0.027107561007142067, 0.24900051951408386, 0.9765522480010986, 0.45496490597724915, 0.2926212251186371, 0.9055909514427185, 0.6193610429763794, 0.22553233802318573, 0.05768685042858124, 0.769480288028717, 0.7287275791168213, 0.3904435336589813, 0.6326232552528381, 0.9135189652442932, 0.20752742886543274, 0.03089716285467148, 0.7085281014442444, 0.22633560001850128, 0.5176387429237366, 0.8787021040916443, 0.09656428545713425, 0.07974360883235931, 0.6480170488357544, 0.7875801920890808, 0.4030035138130188, 0.2153608500957489, 0.604783833026886, 0.5721915364265442}, {0.42053353786468506, 0.8084112405776978, 0.20570655167102814, 0.13526640832424164, 0.2688467800617218, 0.030300764366984367, 0.9695678949356079, 0.8221189379692078, 0.5965505838394165, 0.791346549987793, 0.7562409043312073, 0.170631542801857, 0.40043431520462036, 0.916049599647522, 0.5999785661697388, 0.7427932620048523, 0.01572909578680992, 0.03314851596951485, 0.4008428752422333, 0.6835272312164307, 0.27410921454429626, 0.43769413232803345, 0.894386887550354, 0.3827209770679474, 0.48090580105781555, 0.7187351584434509, 0.45293182134628296, 0.02563261240720749, 0.6209564805030823, 0.4228573441505432, 0.5340514779090881, 0.7385646104812622, 0.16089440882205963, 0.21175852417945862, 0.2353600114583969, 0.09861383587121964, 0.4939865171909332, 0.2562662661075592, 0.48615458607673645, 0.1457120031118393, 0.2781287729740143, 0.025579208508133888, 0.9816099405288696, 0.39923450350761414, 0.7527656555175781, 0.12505950033664703, 0.7481358647346497, 0.7554581761360168, 0.28765052556991577, 0.5388317704200745, 0.5438243746757507, 0.6631920337677002, 0.26208335161209106, 0.5382667183876038, 0.2640339434146881, 0.7051363587379456, 0.1919122040271759, 0.9486129879951477, 0.6202402710914612, 0.3797900378704071, 0.8922446370124817, 0.19977465271949768, 0.4146985709667206, 0.8843743205070496, 0.6263867020606995, 0.8988282680511475, 0.22511833906173706, 0.9974760413169861, 0.12359124422073364, 0.7785593867301941, 0.7938812971115112, 0.130552276968956, 0.6759050488471985, 0.4976223111152649, 0.14448325335979462, 0.9872263073921204, 0.5046866536140442, 0.6457939147949219, 0.9634549617767334, 0.09804365038871765, 0.6183505058288574, 0.8308549523353577, 0.10203584283590317, 0.6947256326675415, 0.3929959833621979, 0.2390323430299759, 0.6856335997581482, 0.37777286767959595, 0.4010956883430481, 0.3197437822818756, 0.4982386827468872, 0.689702570438385, 0.04682879149913788, 0.4161107540130615, 0.6562808156013489, 0.7936096787452698, 0.7261994481086731, 0.7215670347213745, 0.016340717673301697, 0.2764054834842682}, {0.5517759323120117, 0.8822575807571411, 0.5526543259620667, 0.7127084136009216, 0.24681976437568665, 0.30030885338783264, 0.5392318964004517, 0.052726954221725464, 0.8867172002792358, 0.2266407310962677, 0.0443720668554306, 0.4485550820827484, 0.055135756731033325, 0.676204264163971, 0.9661349654197693, 0.750123918056488, 0.5420811772346497, 0.51889568567276, 0.3909114599227905, 0.3090822994709015, 0.652082622051239, 0.346539169549942, 0.4617282748222351, 0.7097111344337463, 0.7420846819877625, 0.06804408878087997, 0.5926237106323242, 0.9932708740234375, 0.8044612407684326, 0.8415874242782593, 0.25591525435447693, 0.3296888768672943, 0.5885347127914429, 0.5763828754425049, 0.33100810647010803, 0.0010313885286450386, 0.5339469909667969, 0.33208802342414856, 0.9460894465446472, 0.5192720890045166, 0.06551248580217361, 0.5759068131446838, 0.6758350133895874, 0.5037274956703186, 0.068203404545784, 0.57639479637146, 0.6013650298118591, 0.6637954711914062, 0.5845447778701782, 0.23955866694450378, 0.3394681513309479, 0.4308929145336151, 0.9481370449066162, 0.06721574813127518, 0.6690329909324646, 0.6306695342063904, 0.11453807353973389, 0.08018035441637039, 0.2950950860977173, 0.8879756331443787, 0.7392510771751404, 0.9955528378486633, 0.34214502573013306, 0.17228998243808746, 0.9472959041595459, 0.9472766518592834, 0.0886324793100357, 0.5251675248146057, 0.1814483404159546, 0.6176595091819763, 0.08912880718708038, 0.461805522441864, 0.6061358451843262, 0.5035189390182495, 0.32800665497779846, 0.48918208479881287, 0.3334099352359772, 0.46587935090065, 0.5912353992462158, 0.5342962741851807, 0.6306884288787842, 0.39666256308555603, 0.20221200585365295, 0.4923194348812103, 0.8759295344352722, 0.368100643157959, 0.13729818165302277, 0.42580053210258484, 0.371592253446579, 0.4581467807292938, 0.026438534259796143, 0.2600988745689392, 0.901142418384552, 0.5947840809822083, 0.7392715811729431, 0.600178599357605, 0.7488950490951538, 0.7574044466018677, 0.39822760224342346, 0.9609776735305786}, {0.06264375150203705, 0.7728891372680664, 0.04108922556042671, 0.04515291005373001, 0.4308415949344635, 0.25057733058929443, 0.6445152163505554, 0.7265562415122986, 0.4380839169025421, 0.6482874155044556, 0.058079250156879425, 0.14164648950099945, 0.19751697778701782, 0.00032534159254282713, 0.006948444992303848, 0.21523916721343994, 0.784772515296936, 0.3562258780002594, 0.09869538992643356, 0.04396801441907883, 0.03274622559547424, 0.15929317474365234, 0.037826258689165115, 0.9342238903045654, 0.9974027276039124, 0.7705681920051575, 0.1474609375, 0.5153565406799316, 0.9105399250984192, 0.8208484649658203, 0.6173052787780762, 0.3544115722179413, 0.6676075458526611, 0.3958095908164978, 0.8321279287338257, 0.27195045351982117, 0.9992734789848328, 0.207072451710701, 0.9561100602149963, 0.7991288900375366, 0.6554350256919861, 0.8238354325294495, 0.8957266211509705, 0.4743903875350952, 0.8359169960021973, 0.3877749443054199, 0.14239619672298431, 0.12571963667869568, 0.23482394218444824, 0.7711678743362427, 0.6086710691452026, 0.49118906259536743, 0.9916663765907288, 0.5697830319404602, 0.0695405900478363, 0.8407790064811707, 0.8253714442253113, 0.5615211129188538, 0.6938133239746094, 0.6075434684753418, 0.7269778251647949, 0.27214646339416504, 0.30000370740890503, 0.13021965324878693, 0.163510262966156, 0.28852128982543945, 0.5603885650634766, 0.09263566881418228, 0.6848005652427673, 0.09470637142658234, 0.6242619752883911, 0.7986608743667603, 0.6427004933357239, 0.9875348806381226, 0.30064570903778076, 0.38229984045028687, 0.36319541931152344, 0.5913324952125549, 0.4390281140804291, 0.09510602802038193, 0.009203790687024593, 0.4262103736400604, 0.2049618661403656, 0.6920801401138306, 0.8446701765060425, 0.165915846824646, 0.6586934924125671, 0.903266429901123, 0.17165957391262054, 0.8818023800849915, 0.4819094240665436, 0.2797207832336426, 0.7415966391563416, 0.33032089471817017, 0.4058106243610382, 0.7497352957725525, 0.5239899754524231, 0.633293092250824, 0.9846329689025879, 0.5811689496040344}, {0.1354474127292633, 0.6211629509925842, 0.017612943425774574, 0.13988344371318817, 0.35316380858421326, 0.6439041495323181, 0.8322182893753052, 0.20712216198444366, 0.1643853634595871, 0.23700621724128723, 0.6383283734321594, 0.7348405718803406, 0.37648332118988037, 0.4354073703289032, 0.26736873388290405, 0.4053422510623932, 0.16886618733406067, 0.8158379197120667, 0.6813479661941528, 0.8566135168075562, 0.6522442102432251, 0.7514921426773071, 0.6840652227401733, 0.6801038980484009, 0.305830180644989, 0.6456333994865417, 0.14756149053573608, 0.3015283942222595, 0.29380476474761963, 0.938124418258667, 0.01689416728913784, 0.8544921278953552, 0.7668965458869934, 0.007422712631523609, 0.13386273384094238, 0.24014216661453247, 0.06845109164714813, 0.3864147961139679, 0.3342667818069458, 0.08267218619585037, 0.6434180736541748, 0.18731403350830078, 0.09902495890855789, 0.7924726009368896, 0.4158318340778351, 0.6259189248085022, 0.23026999831199646, 0.14854857325553894, 0.04295334592461586, 0.6160602569580078, 0.2761092782020569, 0.6515311598777771, 0.036293964833021164, 0.4549230635166168, 0.6117811799049377, 0.22450342774391174, 0.9584752321243286, 0.8230636715888977, 0.965049147605896, 0.5800237655639648, 0.6534284353256226, 0.8074710369110107, 0.15586026012897491, 0.7208128571510315, 0.6552152037620544, 0.8067886233329773, 0.41305190324783325, 0.7031393051147461, 0.9408276677131653, 0.8071886301040649, 0.5066784024238586, 0.24236175417900085, 0.09085848182439804, 0.10273554176092148, 0.9637274742126465, 0.08919920772314072, 0.366847962141037, 0.5041314959526062, 0.5432356595993042, 0.6192861795425415, 0.16408437490463257, 0.6856434345245361, 0.29620426893234253, 0.1514279693365097, 0.943136990070343, 0.7252128720283508, 0.09095288813114166, 0.046275943517684937, 0.2635469138622284, 0.9770867228507996, 0.9314616918563843, 0.05986694246530533, 0.4560070335865021, 0.8493226766586304, 0.22260409593582153, 0.3624859154224396, 0.6683400869369507, 0.6968082785606384, 0.24441160261631012, 0.23865845799446106}, {0.43557286262512207, 0.7925032377243042, 0.27140647172927856, 0.3310733735561371, 0.8176286220550537, 0.7163587212562561, 0.8689660429954529, 0.4152315855026245, 0.9753678441047668, 0.706319272518158, 0.5092378854751587, 0.579987108707428, 0.6149289608001709, 0.11766036599874496, 0.5019949078559875, 0.5039321780204773, 0.9391781687736511, 0.8835844397544861, 0.25447791814804077, 0.5623159408569336, 0.4309598207473755, 0.4877489507198334, 0.873827338218689, 0.732757568359375, 0.48561516404151917, 0.052531544119119644, 0.058233048766851425, 0.610123336315155, 0.18287144601345062, 0.30010294914245605, 0.8540666699409485, 0.6361896395683289, 0.22366051375865936, 0.7912817597389221, 0.5303614735603333, 0.24470849335193634, 0.2767748236656189, 0.5834924578666687, 0.5453978776931763, 0.48084381222724915, 0.2811121344566345, 0.7280074954032898, 0.7475252151489258, 0.4879455864429474, 0.4126388728618622, 0.06911468505859375, 0.5713120698928833, 0.6902819275856018, 0.7407330870628357, 0.6664143800735474, 0.6961774826049805, 0.2172655463218689, 0.897271454334259, 0.8672818541526794, 0.15852028131484985, 0.06961327791213989, 0.6688588857650757, 0.15983140468597412, 0.12536154687404633, 0.8513672351837158, 0.51286381483078, 0.39484331011772156, 0.9948205947875977, 0.925597608089447, 0.030253566801548004, 0.13865160942077637, 0.6319965720176697, 0.6378791332244873, 0.6711099147796631, 0.24462050199508667, 0.24701619148254395, 0.2624414563179016, 0.8172388076782227, 0.9553409814834595, 0.3352615535259247, 0.3044953942298889, 0.595671534538269, 0.44374552369117737, 0.6048862934112549, 0.1363350749015808, 0.29690587520599365, 0.8425527811050415, 0.8210467100143433, 0.7289726138114929, 0.7295643091201782, 0.6030838489532471, 0.885836660861969, 0.24282710254192352, 0.9724904894828796, 0.29112935066223145, 0.7700057625770569, 0.872704803943634, 0.3802907168865204, 0.9217389225959778, 0.40098193287849426, 0.2660321295261383, 0.7172657251358032, 0.34736254811286926, 0.8613914847373962, 0.8119197487831116}, {0.4552553594112396, 0.1661178171634674, 0.8114076256752014, 0.21083585917949677, 0.5416946411132812, 0.32568803429603577, 0.17529775202274323, 0.5023009777069092, 0.1524454951286316, 0.005476151593029499, 0.245033398270607, 0.31379932165145874, 0.22316549718379974, 0.26764172315597534, 0.9794679284095764, 0.061675090342760086, 0.42584362626075745, 0.5219647288322449, 0.9739484190940857, 0.6517491936683655, 0.6461368799209595, 0.04621903970837593, 0.7733979821205139, 0.8114979267120361, 0.4494691789150238, 0.44965049624443054, 0.7996225357055664, 0.24331311881542206, 0.9877908229827881, 0.09582604467868805, 0.43094176054000854, 0.3363691568374634, 0.83524489402771, 0.5013229250907898, 0.4506050944328308, 0.32478174567222595, 0.6010932922363281, 0.8873777985572815, 0.6925675272941589, 0.12292970716953278, 0.19133655726909637, 0.3538194000720978, 0.6901763081550598, 0.17929679155349731, 0.21081602573394775, 0.25515714287757874, 0.8826594948768616, 0.18263228237628937, 0.815261721611023, 0.8516287207603455, 0.8030432462692261, 0.5426782369613647, 0.7578899264335632, 0.589036226272583, 0.2644183039665222, 0.3041805028915405, 0.02321345917880535, 0.24395953118801117, 0.444378137588501, 0.5716236233711243, 0.2721456289291382, 0.577882707118988, 0.377284973859787, 0.5100464224815369, 0.5381632447242737, 0.7789450287818909, 0.42055565118789673, 0.014845399186015129, 0.6883514523506165, 0.46857836842536926, 0.1085972711443901, 0.03752099350094795, 0.14264149963855743, 0.43474137783050537, 0.9748798608779907, 0.5184578895568848, 0.16881734132766724, 0.4976704716682434, 0.39957863092422485, 0.773994505405426, 0.7588335275650024, 0.2672705352306366, 0.3720491826534271, 0.4695829153060913, 0.780307948589325, 0.34069791436195374, 0.4351004660129547, 0.6489077210426331, 0.5098023414611816, 0.7804304957389832, 0.25552329421043396, 0.09696628153324127, 0.2802857756614685, 0.08892328292131424, 0.01110153179615736, 0.9993274807929993, 0.3561723232269287, 0.13015443086624146, 0.6804819703102112, 0.9804860353469849}, {0.8402237296104431, 0.8611384034156799, 0.31522974371910095, 0.08656805008649826, 0.8762056231498718, 0.7813288569450378, 0.5226237773895264, 0.4821512997150421, 0.9808861017227173, 0.9330892562866211, 0.32146745920181274, 0.0026168629992753267, 0.7837129831314087, 0.11648081243038177, 0.5936108827590942, 0.19298048317432404, 0.3081343173980713, 0.06886819005012512, 0.40853869915008545, 0.6873350739479065, 0.6792213320732117, 0.9807753562927246, 0.9070724844932556, 0.8444690108299255, 0.340597003698349, 0.029081959277391434, 0.05149533227086067, 0.6413372755050659, 0.12826472520828247, 0.2553364336490631, 0.9220461249351501, 0.35635697841644287, 0.13933219015598297, 0.23812714219093323, 0.10186641663312912, 0.38818585872650146, 0.12444271147251129, 0.13536806404590607, 0.030404971912503242, 0.048339929431676865, 0.0056882696226239204, 0.9615370631217957, 0.010911130346357822, 0.8403884172439575, 0.9385703206062317, 0.11888832598924637, 0.9418313503265381, 0.7644728422164917, 0.32659128308296204, 0.515304446220398, 0.23634344339370728, 0.5438627004623413, 0.7006506323814392, 0.9835819005966187, 0.11174922436475754, 0.04173906892538071, 0.43233829736709595, 0.2511581778526306, 0.7419435977935791, 0.13808585703372955, 0.8583317995071411, 0.303245484828949, 0.9492394924163818, 0.46482157707214355, 0.4047127664089203, 0.7797443270683289, 0.33496329188346863, 0.2639813721179962, 0.30860012769699097, 0.25873804092407227, 0.5559571385383606, 0.6156843900680542, 0.6782705187797546, 0.3137730658054352, 0.07365602999925613, 0.1544950008392334, 0.4330737590789795, 0.04044949635863304, 0.686306357383728, 0.05455522611737251, 0.4902344048023224, 0.9187493324279785, 0.2620585560798645, 0.1982588768005371, 0.6245680451393127, 0.49132040143013, 0.8002579212188721, 0.06730710715055466, 0.287356972694397, 0.261200875043869, 0.35219132900238037, 0.16952456533908844, 0.7133945226669312, 0.8661649227142334, 0.08393070846796036, 0.229508176445961, 0.621868371963501, 0.05560971051454544, 0.24686948955059052, 0.8899211287498474}, {0.05852551758289337, 0.8883979916572571, 0.8800884485244751, 0.06013079732656479, 0.4416556656360626, 0.3336174190044403, 0.5175039768218994, 0.16573120653629303, 0.392486035823822, 0.7209242582321167, 0.04253394901752472, 0.4582321047782898, 0.36233991384506226, 0.611237108707428, 0.9664813876152039, 0.7443915605545044, 0.4865715205669403, 0.8926457762718201, 0.9104334115982056, 0.5328962206840515, 0.32188016176223755, 0.9165946245193481, 0.9036336541175842, 0.844607412815094, 0.5301212668418884, 0.709895133972168, 0.07309422641992569, 0.6197893023490906, 0.0025620381347835064, 0.8719336986541748, 0.14925681054592133, 0.25876936316490173, 0.9892578125, 0.9973140954971313, 0.6699689626693726, 0.8409379124641418, 0.2691977918148041, 0.8412842750549316, 0.19436340034008026, 0.3996350169181824, 0.2548372149467468, 0.7810267806053162, 0.4159158170223236, 0.5740999579429626, 0.6749483346939087, 0.7361810803413391, 0.8977506756782532, 0.3073590099811554, 0.11623681336641312, 0.39948853850364685, 0.8604098558425903, 0.17359498143196106, 0.6579459309577942, 0.8050484657287598, 0.38531604409217834, 0.38123106956481934, 0.7380885481834412, 0.03063388541340828, 0.3339988887310028, 0.10872191190719604, 0.24804134666919708, 0.7410774230957031, 0.09344667941331863, 0.5626105070114136, 0.40961137413978577, 0.6694551110267639, 0.05428772419691086, 0.25117620825767517, 0.48213478922843933, 0.32614728808403015, 0.4612446427345276, 0.686302125453949, 0.8950322270393372, 0.6010268926620483, 0.4940398335456848, 0.55418860912323, 0.3789658844470978, 0.9649243950843811, 0.529045581817627, 0.44321438670158386, 0.25624436140060425, 0.30409589409828186, 0.29231706261634827, 0.6863848567008972, 0.08934614807367325, 0.5956682562828064, 0.3231692910194397, 0.2385755330324173, 0.3515399992465973, 0.5955028533935547, 0.19806240499019623, 0.5217381119728088, 0.18214546144008636, 0.09110108762979507, 0.2618458569049835, 0.7227481007575989, 0.8811311721801758, 0.3814971446990967, 0.7912336587905884, 0.36322078108787537}, {0.47925567626953125, 0.3921910524368286, 0.4015915095806122, 0.5347911715507507, 0.4361729919910431, 0.6731534600257874, 0.5717144012451172, 0.9669167399406433, 0.8262593150138855, 0.46777835488319397, 0.06669334322214127, 0.8941019177436829, 0.21729382872581482, 0.8545233011245728, 0.7028560638427734, 0.7385834455490112, 0.36988452076911926, 0.5934886932373047, 0.9300591945648193, 0.9709665775299072, 0.10681111365556717, 0.7298406958580017, 0.5492857694625854, 0.5608676671981812, 0.8248620629310608, 0.9303900599479675, 0.028884800150990486, 0.5065717101097107, 0.35087594389915466, 0.3056774437427521, 0.21648795902729034, 0.8779768347740173, 0.8106526732444763, 0.09412425756454468, 0.37714797258377075, 0.07858803868293762, 0.1953706592321396, 0.9938649535179138, 0.0005299206241033971, 0.9539291262626648, 0.9807761311531067, 0.8802284002304077, 0.1392282396554947, 0.7358031272888184, 0.36395734548568726, 0.9202728867530823, 0.5638365745544434, 0.19455686211585999, 0.6585793495178223, 0.9038268327713013, 0.8515844345092773, 0.3665207624435425, 0.9583156704902649, 0.2154608815908432, 0.26287978887557983, 0.2482873648405075, 0.12288626283407211, 0.7489870190620422, 0.10381319373846054, 0.4774388074874878, 0.9898713827133179, 0.8344538807868958, 0.2855609655380249, 0.19408899545669556, 0.59323650598526, 0.6309822201728821, 0.6746847629547119, 0.09561944752931595, 0.98046875, 0.6881959438323975, 0.06548916548490524, 0.24289701879024506, 0.2629488706588745, 0.39749670028686523, 0.2546209692955017, 0.9638678431510925, 0.3314839005470276, 0.4532093405723572, 0.19304130971431732, 0.7832087874412537, 0.602542519569397, 0.05199676379561424, 0.3314238488674164, 0.412069708108902, 0.02117936499416828, 0.7988429069519043, 0.6894753575325012, 0.6189761161804199, 0.12117285281419754, 0.8214765787124634, 0.7924765944480896, 0.6790727376937866, 0.4145975112915039, 0.12181728333234787, 0.056308429688215256, 0.8483636975288391, 0.20380575954914093, 0.10193013399839401, 0.49512720108032227, 0.020031288266181946}, {0.9074591994285583, 0.10210216790437698, 0.8301611542701721, 0.36506742238998413, 0.8069852590560913, 0.41221559047698975, 0.839820921421051, 0.8836753368377686, 0.02088029868900776, 0.8388499021530151, 0.3087150752544403, 0.5785909295082092, 0.1379186064004898, 0.2743373215198517, 0.8661473393440247, 0.8207105994224548, 0.2341110110282898, 0.06026596203446388, 0.49589240550994873, 0.37335601449012756, 0.48448610305786133, 0.11118728667497635, 0.4515094757080078, 0.6317046284675598, 0.6939470767974854, 0.6076922416687012, 0.8180950284004211, 0.0058302548713982105, 0.01666974648833275, 0.7943288683891296, 0.018868504092097282, 0.5545187592506409, 0.5686479806900024, 0.2972721755504608, 0.1942378580570221, 0.7525373101234436, 0.6713894605636597, 0.6923412680625916, 0.439175009727478, 0.9495701789855957, 0.41976073384284973, 0.49886101484298706, 0.44871482253074646, 0.8399158716201782, 0.6219837665557861, 0.5404449105262756, 0.49307265877723694, 0.2696089446544647, 0.2726225256919861, 0.7783411145210266, 0.5277116894721985, 0.1684005856513977, 0.910925030708313, 0.2103903740644455, 0.8198983669281006, 0.958075225353241, 0.28930819034576416, 0.35660213232040405, 0.23013441264629364, 0.7744404077529907, 0.0006704223924316466, 0.6250317692756653, 0.6006023287773132, 0.39465340971946716, 0.427537739276886, 0.5767065286636353, 0.5970850586891174, 0.04595537111163139, 0.5191332697868347, 0.14594298601150513, 0.6950914263725281, 0.09510983526706696, 0.2568252682685852, 0.2002691924571991, 0.5583091974258423, 0.44300657510757446, 0.6614370942115784, 0.05150781199336052, 0.8176923990249634, 0.44715577363967896, 0.06468495726585388, 0.8750038146972656, 0.9904181361198425, 0.24810339510440826, 0.21289722621440887, 0.39489153027534485, 0.6723510026931763, 0.3043542206287384, 0.640810489654541, 0.312176913022995, 0.7299783229827881, 0.2518060505390167, 0.6233085989952087, 0.7836226224899292, 0.3069470226764679, 0.6259698867797852, 0.9776831865310669, 0.04925008863210678, 0.9422520399093628, 0.953279435634613}, {0.3484022617340088, 0.6435733437538147, 0.9665765762329102, 0.9112539887428284, 0.5011199116706848, 0.7630560994148254, 0.8647421598434448, 0.5154312252998352, 0.7977951169013977, 0.7136216759681702, 0.46816766262054443, 0.7353894114494324, 0.8825540542602539, 0.3773531913757324, 0.08465512841939926, 0.3186142146587372, 0.13342900574207306, 0.9909791350364685, 0.3336386978626251, 0.4416022300720215, 0.18156825006008148, 0.8898938894271851, 0.6056728959083557, 0.6774377822875977, 0.013569277711212635, 0.8883867859840393, 0.8605535626411438, 0.46472787857055664, 0.4323558509349823, 0.23604613542556763, 0.41065382957458496, 0.03914114832878113, 0.9614832401275635, 0.9094648957252502, 0.14325198531150818, 0.5214378237724304, 0.5235615968704224, 0.20302611589431763, 0.6525174379348755, 0.709277868270874, 0.45435091853141785, 0.6796265840530396, 0.22076429426670074, 0.36482059955596924, 0.8094600439071655, 0.25151556730270386, 0.9701110124588013, 0.8551803231239319, 0.09921519458293915, 0.13496866822242737, 0.6813306212425232, 0.6739909052848816, 0.2532602846622467, 0.08759588748216629, 0.350495845079422, 0.08821450918912888, 0.907189667224884, 0.4813413619995117, 0.19937290251255035, 0.05765793099999428, 0.5426404476165771, 0.1650620400905609, 0.8373223543167114, 0.3472330570220947, 0.6433541178703308, 0.7699923515319824, 0.4839698374271393, 0.3971758186817169, 0.9157699942588806, 0.41327473521232605, 0.10077745467424393, 0.7040106058120728, 0.3228447437286377, 0.9317010641098022, 0.19265228509902954, 0.2949785590171814, 0.6080365777015686, 0.5110632181167603, 0.9383640289306641, 0.7694316506385803, 0.2175750881433487, 0.13334859907627106, 0.4103875160217285, 0.944521963596344, 0.22704316675662994, 0.8383178114891052, 0.8814414739608765, 0.6296427249908447, 0.8592730760574341, 0.6386082172393799, 0.727233350276947, 0.8254384398460388, 0.5148025155067444, 0.3368074297904968, 0.14689995348453522, 0.7826665043830872, 0.26673585176467896, 0.09217601269483566, 0.21014954149723053, 0.7852442860603333}, {0.8189944624900818, 0.9028928279876709, 0.8293606638908386, 0.9023454785346985, 0.09886778891086578, 0.7489389181137085, 0.9350311160087585, 0.887177586555481, 0.8433471322059631, 0.8897603750228882, 0.060364071279764175, 0.7845578789710999, 0.39375540614128113, 0.09795504063367844, 0.04371175169944763, 0.14605015516281128, 0.7497039437294006, 0.8875212669372559, 0.008382448926568031, 0.23054100573062897, 0.8921884894371033, 0.5540662407875061, 0.8756552338600159, 0.9450830221176147, 0.5331270098686218, 0.21531297266483307, 0.0076604438945651054, 0.6308374404907227, 0.3554995357990265, 0.4412148594856262, 0.8471889495849609, 0.6236358880996704, 0.3475959002971649, 0.9495395421981812, 0.531904399394989, 0.598142147064209, 0.2797030806541443, 0.4551677703857422, 0.5864138007164001, 0.6346940398216248, 0.2301812618970871, 0.5715111494064331, 0.8611293435096741, 0.7026569843292236, 0.703528106212616, 0.6895046830177307, 0.8342430591583252, 0.5531585216522217, 0.8356319665908813, 0.30414026975631714, 0.3431921601295471, 0.8251692652702332, 0.12168341130018234, 0.5602235198020935, 0.658156156539917, 0.1306910514831543, 0.722220778465271, 0.8389914035797119, 0.2648352086544037, 0.3138195276260376, 0.796008288860321, 0.34822550415992737, 0.19185727834701538, 0.8221265077590942, 0.08977586030960083, 0.6707620620727539, 0.7504134774208069, 0.5417088270187378, 0.8192015290260315, 0.8347575664520264, 0.3525053560733795, 0.14633648097515106, 0.545773983001709, 0.46522626280784607, 0.8408788442611694, 0.3105728030204773, 0.5294788479804993, 0.082206130027771, 0.7735927700996399, 0.7730347514152527, 0.5756844282150269, 0.6399381160736084, 0.6059135794639587, 0.19867394864559174, 0.33773893117904663, 0.019607778638601303, 0.19371500611305237, 0.6600409150123596, 0.3286338150501251, 0.6345043182373047, 0.6921296715736389, 0.7964634299278259, 0.4057987630367279, 0.5989806652069092, 0.01821945048868656, 0.2950421869754791, 0.7221066355705261, 0.6840521693229675, 0.9730800986289978, 0.8209878206253052}, {0.9113158583641052, 0.03303300589323044, 0.16346436738967896, 0.6295239925384521, 0.08134070038795471, 0.6216850876808167, 0.24657903611660004, 0.9265550374984741, 0.0888233408331871, 0.011845413595438004, 0.5560207366943359, 0.6588391661643982, 0.8197553157806396, 0.11883176863193512, 0.317251980304718, 0.85569828748703, 0.7699646949768066, 0.5330391526222229, 0.6524525880813599, 0.42455238103866577, 0.77019864320755, 0.20191465318202972, 0.3134302794933319, 0.7957846522331238, 0.2947635352611542, 0.3977154493331909, 0.9017494320869446, 0.5481663346290588, 0.9603252410888672, 0.05425490811467171, 0.09087088704109192, 0.7033517360687256, 0.7043488025665283, 0.4943332076072693, 0.42459550499916077, 0.736396849155426, 0.5554577112197876, 0.5908530354499817, 0.8219396471977234, 0.00563237676396966, 0.06010470166802406, 0.20320096611976624, 0.28814154863357544, 0.3394387364387512, 0.5433846712112427, 0.9486932158470154, 0.6184802651405334, 0.5462417006492615, 0.4699713885784149, 0.37845802307128906, 0.16429878771305084, 0.12727312743663788, 0.3444094657897949, 0.8044461607933044, 0.7124602198600769, 0.9577705264091492, 0.06955336034297943, 0.013530458323657513, 0.020312363281846046, 0.7991516590118408, 0.21085238456726074, 0.9467102885246277, 0.05346772074699402, 0.6664749383926392, 0.4677586853504181, 0.18701599538326263, 0.28422868251800537, 0.3647591471672058, 0.6489128470420837, 0.6707139015197754, 0.5806735754013062, 0.6282256841659546, 0.31956803798675537, 0.7300606966018677, 0.8879443407058716, 0.6467981338500977, 0.10856598615646362, 0.3467521369457245, 0.02774311602115631, 0.6504606604576111, 0.2719123363494873, 0.3877435028553009, 0.15711486339569092, 0.8824304342269897, 0.8914259076118469, 0.08243262022733688, 0.5376684665679932, 0.4718901515007019, 0.7697537541389465, 0.047789957374334335, 0.7337154746055603, 0.2892409861087799, 0.5637979507446289, 0.5118345022201538, 0.8679958581924438, 0.5702959299087524, 0.41040998697280884, 0.6456342339515686, 0.8226571083068848, 0.9500705599784851}, {0.04061208292841911, 0.5519744157791138, 0.36765626072883606, 0.8053485155105591, 0.2116047590970993, 0.234166219830513, 0.5736017227172852, 0.7013214230537415, 0.603331983089447, 0.5608437657356262, 0.8450901508331299, 0.023315701633691788, 0.42120668292045593, 0.928613543510437, 0.7607690691947937, 0.7409753799438477, 0.22780123353004456, 0.14806610345840454, 0.790848970413208, 0.7185989022254944, 0.8284046649932861, 0.9267550706863403, 0.4869634807109833, 0.9548274874687195, 0.8521997332572937, 0.2659350633621216, 0.5482759475708008, 0.07540348917245865, 0.5650928020477295, 0.9378625154495239, 0.010987019166350365, 0.6818116903305054, 0.16105987131595612, 0.6550353765487671, 0.004317395854741335, 0.08926012367010117, 0.5640782713890076, 0.5576744079589844, 0.6480452418327332, 0.581721842288971, 0.7562816739082336, 0.9000746011734009, 0.005926780868321657, 0.24205505847930908, 0.5305302143096924, 0.8957642912864685, 0.32400327920913696, 0.6917314529418945, 0.9067240357398987, 0.08149099349975586, 0.4424077868461609, 0.1633591204881668, 0.6274644732475281, 0.06277057528495789, 0.23581282794475555, 0.44774070382118225, 0.3944564759731293, 0.4677185118198395, 0.4144227206707001, 0.40561479330062866, 0.6965122818946838, 0.6673185229301453, 0.05432839319109917, 0.653428316116333, 0.9803332686424255, 0.25439152121543884, 0.19831717014312744, 0.07903633266687393, 0.7311198115348816, 0.45707008242607117, 0.9335768222808838, 0.8312981128692627, 0.4827239215373993, 0.8411898612976074, 0.2836097776889801, 0.5551427602767944, 0.11811797320842743, 0.9658834934234619, 0.593818724155426, 0.6972563862800598, 0.1454945206642151, 0.41518503427505493, 0.38077130913734436, 0.9640500545501709, 0.22192083299160004, 0.1268392652273178, 0.7716476321220398, 0.9091143012046814, 0.09614881128072739, 0.4784274399280548, 0.7823416590690613, 0.5030183792114258, 0.7281538844108582, 0.8329057097434998, 0.34024307131767273, 0.14167097210884094, 0.9127330780029297, 0.9695451259613037, 0.004037568345665932, 0.05261585861444473}, {0.006035769358277321, 0.3068501353263855, 0.6925891637802124, 0.7954465746879578, 0.6778050065040588, 0.6662754416465759, 0.916657030582428, 0.31130167841911316, 0.8442315459251404, 0.605959951877594, 0.20496433973312378, 0.5659388899803162, 0.49216949939727783, 0.710157573223114, 0.522868812084198, 0.022823205217719078, 0.5087695121765137, 0.7504661083221436, 0.6718510985374451, 0.7381145358085632, 0.8838472366333008, 0.19017283618450165, 0.9178786277770996, 0.9241194725036621, 0.7077597975730896, 0.7602375149726868, 0.5979136824607849, 0.02855638787150383, 0.4313598871231079, 0.787352979183197, 0.8816033601760864, 0.6403058767318726, 0.7554517984390259, 0.041772183030843735, 0.1534913331270218, 0.8407299518585205, 0.7689167857170105, 0.8446004986763, 0.5775453448295593, 0.5164831280708313, 0.06949665397405624, 0.5229800939559937, 0.42809873819351196, 0.5840622782707214, 0.5636864900588989, 0.11240039765834808, 0.05597823113203049, 0.24722279608249664, 0.24724037945270538, 0.9302054047584534, 0.20924708247184753, 0.696522057056427, 0.3254472315311432, 0.6427315473556519, 0.0710376724600792, 0.6610140204429626, 0.44695237278938293, 0.019662706181406975, 0.2521986961364746, 0.8582349419593811, 0.3200477063655853, 0.38097497820854187, 0.4051959812641144, 0.12272492796182632, 0.39074328541755676, 0.7350073456764221, 0.13971109688282013, 0.36244285106658936, 0.7627640962600708, 0.009463312104344368, 0.4915304183959961, 0.040185119956731796, 0.5950431227684021, 0.04610391706228256, 0.06154625490307808, 0.9136227965354919, 0.9430335164070129, 0.7765167951583862, 0.2657569646835327, 0.5505746006965637, 0.3227408230304718, 0.7311307787895203, 0.3872247040271759, 0.16031469404697418, 0.6718116998672485, 0.46042221784591675, 0.3129000961780548, 0.020554792135953903, 0.7401023507118225, 0.014347991906106472, 0.535926878452301, 0.3101126253604889, 0.3165009319782257, 0.33643069863319397, 0.46402454376220703, 0.40195685625076294, 0.9795850515365601, 0.8683503866195679, 0.12975747883319855, 0.7767803072929382}, {0.1491720974445343, 0.08714767545461655, 0.8324971199035645, 0.24001440405845642, 0.8072130084037781, 0.4327245056629181, 0.7070329189300537, 0.3274371027946472, 0.6411504149436951, 0.5027778744697571, 0.21133241057395935, 0.679325520992279, 0.7553214430809021, 0.20143552124500275, 0.5349628925323486, 0.8970568776130676, 0.9737607836723328, 0.5243906378746033, 0.6591758728027344, 0.5305814146995544, 0.04789746552705765, 0.20359693467617035, 0.544765293598175, 0.5472044944763184, 0.299728661775589, 0.7491304278373718, 0.9672799110412598, 0.1797298789024353, 0.29165026545524597, 0.6498312950134277, 0.19569174945354462, 0.5768833160400391, 0.11289068311452866, 0.21769018471240997, 0.6811168193817139, 0.2655535042285919, 0.44621044397354126, 0.06212699040770531, 0.4431222379207611, 0.7576354742050171, 0.18854966759681702, 0.9049146175384521, 0.9818912148475647, 0.1248360425233841, 0.6102441549301147, 0.8307822346687317, 0.7581571936607361, 0.4892926514148712, 0.22386638820171356, 0.5921186804771423, 0.14208559691905975, 0.08325597643852234, 0.6582669615745544, 0.4601326882839203, 0.3643796443939209, 0.4483729898929596, 0.8385757207870483, 0.9474970698356628, 0.8322780728340149, 0.1875067800283432, 0.9461556077003479, 0.06904849410057068, 0.4068793058395386, 0.9454387426376343, 0.9475706815719604, 0.5705606937408447, 0.9877366423606873, 0.4062231481075287, 0.04414139315485954, 0.5475702881813049, 0.3035126030445099, 0.4637037217617035, 0.9599981904029846, 0.5950507521629333, 0.4852951169013977, 0.8750328421592712, 0.23425759375095367, 0.8033562302589417, 0.4295336902141571, 0.4071653485298157, 0.26615312695503235, 0.9065108299255371, 0.8474556803703308, 0.16568180918693542, 0.9140651822090149, 0.8904607892036438, 0.35369622707366943, 0.6902005076408386, 0.22997459769248962, 0.4009835124015808, 0.6540864109992981, 0.13610152900218964, 0.16596780717372894, 0.7601760625839233, 0.3744350075721741, 0.22286666929721832, 0.9074132442474365, 0.9339225888252258, 0.15557743608951569, 0.6415574550628662}, {0.8952000141143799, 0.07530204951763153, 0.24263103306293488, 0.22494781017303467, 0.6510990262031555, 0.3045230805873871, 0.6497910022735596, 0.7423408627510071, 0.19968514144420624, 0.6455883979797363, 0.7447708249092102, 0.6969156265258789, 0.776491105556488, 0.18187955021858215, 0.14224651455879211, 0.9496119618415833, 0.8903186321258545, 0.8909396529197693, 0.2185210883617401, 0.4110402762889862, 0.31854677200317383, 0.9028329849243164, 0.2481173872947693, 0.9476070404052734, 0.5877996683120728, 0.561368465423584, 0.47301071882247925, 0.7723983526229858, 0.009567402303218842, 0.9404897689819336, 0.36878156661987305, 0.04593360051512718, 0.4648720622062683, 0.30916643142700195, 0.8356844186782837, 0.6403574347496033, 0.8644728064537048, 0.2642194628715515, 0.224150151014328, 0.7276844382286072, 0.7441442608833313, 0.8045720458030701, 0.8102662563323975, 0.5142454504966736, 0.39067256450653076, 0.8456345796585083, 0.8723730444908142, 0.5316952466964722, 0.6028523445129395, 0.008912691846489906, 0.9983949065208435, 0.5037056803703308, 0.4077785313129425, 0.4914547801017761, 0.3759050965309143, 0.7218430638313293, 0.24165862798690796, 0.9493292570114136, 0.2268308699131012, 0.028291799128055573, 0.6431116461753845, 0.4067841172218323, 0.9347921013832092, 0.5330950021743774, 0.3285231590270996, 0.5425398349761963, 0.45294681191444397, 0.6029247045516968, 0.9339438080787659, 0.8314186334609985, 0.8863717317581177, 0.9047900438308716, 0.6857151985168457, 0.7510492205619812, 0.5495141744613647, 0.9736596941947937, 0.408115029335022, 0.7583218216896057, 0.788491427898407, 0.3487616181373596, 0.2537163496017456, 0.6793434619903564, 0.024543223902583122, 0.6468079686164856, 0.7014679908752441, 0.4572707712650299, 0.11016565561294556, 0.9177167415618896, 0.4403429329395294, 0.04050575941801071, 0.38113483786582947, 0.40034809708595276, 0.19765199720859528, 0.20973606407642365, 0.2719053328037262, 0.9334747791290283, 0.42401647567749023, 0.33920523524284363, 0.3091996908187866, 0.5619879961013794}, {0.22495825588703156, 0.7233931422233582, 0.17173795402050018, 0.9614008665084839, 0.12338807433843613, 0.23734091222286224, 0.10992611199617386, 0.2791338562965393, 0.4005417823791504, 0.8996765613555908, 0.9579362273216248, 0.2768041491508484, 0.06208202242851257, 0.023962564766407013, 0.6354846954345703, 0.3687414526939392, 0.18579654395580292, 0.11245658993721008, 0.2971457839012146, 0.8402077555656433, 0.8815566301345825, 0.31600964069366455, 0.8847648501396179, 0.8664551973342896, 0.6878447532653809, 0.5965172648429871, 0.3447803258895874, 0.6560479998588562, 0.49100279808044434, 0.11374741047620773, 0.9334014058113098, 0.4031374454498291, 0.3938629627227783, 0.7121621370315552, 0.8935474157333374, 0.6911141276359558, 0.8091639280319214, 0.7066738605499268, 0.5846225023269653, 0.3373732566833496, 0.47607752680778503, 0.15420697629451752, 0.9980860352516174, 0.7990942597389221, 0.7327336668968201, 0.6684741973876953, 0.7843202352523804, 0.3940640687942505, 0.6320725679397583, 0.8294348120689392, 0.5098975300788879, 0.015174906700849533, 0.375916451215744, 0.7935141324996948, 0.17422276735305786, 0.9394256472587585, 0.2869279980659485, 0.843640148639679, 0.0558747760951519, 0.7204540967941284, 0.9355147480964661, 0.6412667036056519, 0.14473146200180054, 0.6341840028762817, 0.035933513194322586, 0.26953795552253723, 0.5807817578315735, 0.6961801648139954, 0.2357986867427826, 0.5299912095069885, 0.12352784723043442, 0.7640219926834106, 0.3730961084365845, 0.6941651105880737, 0.23812679946422577, 0.3511945605278015, 0.9305516481399536, 0.42494097352027893, 0.9385840892791748, 0.3893406391143799, 0.2676604986190796, 0.313996821641922, 0.9126774668693542, 0.23537950217723846, 0.22854934632778168, 0.46167275309562683, 0.2679319977760315, 0.2853049337863922, 0.8496396541595459, 0.15042266249656677, 0.41636037826538086, 0.15739843249320984, 0.7633829712867737, 0.1973908692598343, 0.553594172000885, 0.3320282995700836, 0.4564765393733978, 0.4801904857158661, 0.9156542420387268, 0.9376786351203918}, {0.8316908478736877, 0.5957370400428772, 0.05176643282175064, 0.09167686104774475, 0.6485790014266968, 0.1299881935119629, 0.8807981014251709, 0.40113770961761475, 0.1610281765460968, 0.07784286886453629, 0.8460410833358765, 0.811419665813446, 0.8264185786247253, 0.570772111415863, 0.31569862365722656, 0.06691025197505951, 0.07535906136035919, 0.5371163487434387, 0.7205058932304382, 0.722851037979126, 0.7143406867980957, 0.5789945125579834, 0.27800601720809937, 0.3259762227535248, 0.1148572564125061, 0.5359200835227966, 0.6523796319961548, 0.535220742225647, 0.20862242579460144, 0.5829071402549744, 0.6411120295524597, 0.5045971870422363, 0.36551159620285034, 0.8123978972434998, 0.267865926027298, 0.9340555667877197, 0.007546798791736364, 0.21356265246868134, 0.09138397127389908, 0.4364207983016968, 0.17587924003601074, 0.955531895160675, 0.5498132705688477, 0.47069668769836426, 0.43409648537635803, 0.7344918847084045, 0.9993519186973572, 0.43688398599624634, 0.3649560213088989, 0.0038374329451471567, 0.5098839402198792, 0.3978392481803894, 0.5737007260322571, 0.7823734879493713, 0.3536190688610077, 0.09431968629360199, 0.29971837997436523, 0.8190881609916687, 0.21545417606830597, 0.5706321001052856, 0.49496886134147644, 0.5249814987182617, 0.055582813918590546, 0.8132399320602417, 0.011691183783113956, 0.9429434537887573, 0.029400750994682312, 0.026570923626422882, 0.36654672026634216, 0.7496401071548462, 0.4221831262111664, 0.7084968090057373, 0.16090571880340576, 0.3197445869445801, 0.22315514087677002, 0.45995035767555237, 0.04096997529268265, 0.0006743260310031474, 0.20234546065330505, 0.5234204530715942, 0.15197665989398956, 0.28659892082214355, 0.9992781281471252, 0.8470747470855713, 0.29927679896354675, 0.9932371973991394, 0.36112168431282043, 0.5060992240905762, 0.05629611015319824, 0.5830678939819336, 0.9145530462265015, 0.007516122423112392, 0.1285899430513382, 0.00028003938496112823, 0.019924813881516457, 0.13715411722660065, 0.13939186930656433, 0.8650749325752258, 0.4575660526752472, 0.5677762031555176}, {0.05110647901892662, 0.9482160806655884, 0.35332512855529785, 0.8906146883964539, 0.3445499837398529, 0.9581645131111145, 0.16981269419193268, 0.9987477660179138, 0.27426275610923767, 0.2597987651824951, 0.10628439486026764, 0.1394391506910324, 0.8721075654029846, 0.43795251846313477, 0.8360723257064819, 0.6764670014381409, 0.24867714941501617, 0.4902766942977905, 0.7213387489318848, 0.8932689428329468, 0.2818518280982971, 0.6747655868530273, 0.11946128308773041, 0.10662595927715302, 0.9668338298797607, 0.5814071297645569, 0.31033578515052795, 0.7658069729804993, 0.818014919757843, 0.6741768717765808, 0.02881946600973606, 0.6683024764060974, 0.41879570484161377, 0.5205317139625549, 0.13901981711387634, 0.7079725861549377, 0.01897711493074894, 0.16192497313022614, 0.6224945783615112, 0.6191579699516296, 0.27359649538993835, 0.5828726887702942, 0.1647348552942276, 0.9199710488319397, 0.5590274333953857, 0.7275839447975159, 0.4079855978488922, 0.5435316562652588, 0.43795183300971985, 0.7435017228126526, 0.5893195867538452, 0.7660356163978577, 0.05450587347149849, 0.2046288251876831, 0.9522882103919983, 0.29788464307785034, 0.5818580389022827, 0.557925283908844, 0.24107851088047028, 0.5034967660903931, 0.13061249256134033, 0.6600267291069031, 0.4422164559364319, 0.9921908974647522, 0.48557233810424805, 0.48465079069137573, 0.19311054050922394, 0.09980075061321259, 0.8164364695549011, 0.10685190558433533, 0.29029589891433716, 0.24303488433361053, 0.052375905215740204, 0.3880592882633209, 0.3600733280181885, 0.8239871859550476, 0.1842929571866989, 0.16341440379619598, 0.021580688655376434, 0.6665188074111938, 0.10335804522037506, 0.32363414764404297, 0.8606361150741577, 0.6699826121330261, 0.7816174626350403, 0.3545445203781128, 0.9544371366500854, 0.2959623634815216, 0.11119309812784195, 0.7254553437232971, 0.3150428831577301, 0.4186437726020813, 0.32492953538894653, 0.704153835773468, 0.2598004639148712, 0.38517072796821594, 0.4918532073497772, 0.18373200297355652, 0.17442676424980164, 0.09650448709726334}, {0.8551186323165894, 0.13993720710277557, 0.8933914303779602, 0.9670671224594116, 0.616374135017395, 0.5654271841049194, 0.27407243847846985, 0.1501241773366928, 0.5520337224006653, 0.42835894227027893, 0.695405900478363, 0.7374696135520935, 0.08909634500741959, 0.9334926009178162, 0.7586959004402161, 0.244766965508461, 0.9182467460632324, 0.22379851341247559, 0.18668359518051147, 0.7618098855018616, 0.9803680777549744, 0.2804919481277466, 0.3973006010055542, 0.017400002107024193, 0.6087719202041626, 0.6122645139694214, 0.6380370855331421, 0.8171980977058411, 0.8601229190826416, 0.7920705080032349, 0.2346896231174469, 0.024059750139713287, 0.7890833616256714, 0.6244000792503357, 0.41996049880981445, 0.3841061592102051, 0.551662266254425, 0.24641364812850952, 0.3340591788291931, 0.05807242915034294, 0.996204674243927, 0.742794394493103, 0.21966294944286346, 0.7274774312973022, 0.27822360396385193, 0.21764631569385529, 0.8769686222076416, 0.3039292097091675, 0.4701942205429077, 0.5305324196815491, 0.8729200959205627, 0.39186251163482666, 0.5238475799560547, 0.844597578048706, 0.49475160241127014, 0.02457154542207718, 0.2724311947822571, 0.7376030683517456, 0.342506468296051, 0.40903764963150024, 0.7520844340324402, 0.3961363434791565, 0.8676404356956482, 0.4139351546764374, 0.8903392553329468, 0.4260901212692261, 0.8480985760688782, 0.6621105670928955, 0.87736576795578, 0.017619458958506584, 0.09739384055137634, 0.8149299621582031, 0.28561195731163025, 0.9710085988044739, 0.1866110861301422, 0.6101831793785095, 0.97881019115448, 0.9290933609008789, 0.48619580268859863, 0.9346242547035217, 0.7815516591072083, 0.908316969871521, 0.6679338216781616, 0.5633054971694946, 0.8659871220588684, 0.4013642370700836, 0.037920866161584854, 0.9371882081031799, 0.8216214179992676, 0.8936187028884888, 0.9870263934135437, 0.8770602345466614, 0.21468433737754822, 0.16225707530975342, 0.6746436357498169, 0.6745378971099854, 0.933346688747406, 0.3708225190639496, 0.324000746011734, 0.10612349212169647}, {0.0923532173037529, 0.5470149517059326, 0.6503274440765381, 0.17571641504764557, 0.778184711933136, 0.9659737348556519, 0.10094889253377914, 0.5850347280502319, 0.06141708791255951, 0.2019464522600174, 0.04432704672217369, 0.23722131550312042, 0.5697326064109802, 0.7631543874740601, 0.4961124360561371, 0.05510909855365753, 0.661267101764679, 0.2989761531352997, 0.6699420213699341, 0.6376724243164062, 0.9799252152442932, 0.668156087398529, 0.8488633632659912, 0.6980870366096497, 0.2507268786430359, 0.17829011380672455, 0.4966966509819031, 0.24931460618972778, 0.9856876730918884, 0.3579941987991333, 0.6631388664245605, 0.5664016604423523, 0.028787970542907715, 0.9204193949699402, 0.8569085001945496, 0.12152162939310074, 0.9103517532348633, 0.030269956216216087, 0.8614793419837952, 0.784641444683075, 0.9986774921417236, 0.35180383920669556, 0.5935757160186768, 0.74024498462677, 0.40679845213890076, 0.6529338955879211, 0.9494090676307678, 0.5117683410644531, 0.9162785410881042, 0.42101603746414185, 0.3164042532444, 0.20131070911884308, 0.037977565079927444, 0.4345652759075165, 0.3824580907821655, 0.21502478420734406, 0.7753006815910339, 0.19948330521583557, 0.19519704580307007, 0.4831278920173645, 0.4237612187862396, 0.7723475694656372, 0.24212181568145752, 0.10194042325019836, 0.41972625255584717, 0.009055686183273792, 0.7472924590110779, 0.534698486328125, 0.14234189689159393, 0.6726483106613159, 0.6398240327835083, 0.4056609272956848, 0.6466966867446899, 0.6018279790878296, 0.9378615617752075, 0.7239490151405334, 0.3030720353126526, 0.8630988597869873, 0.34640127420425415, 0.4950138032436371, 0.11804548650979996, 0.7902100086212158, 0.28515198826789856, 0.622382640838623, 0.12219060212373734, 0.8557130098342896, 0.6198441982269287, 0.2724396586418152, 0.18353797495365143, 0.3697238266468048, 0.4391362965106964, 0.09687422960996628, 0.07682891190052032, 0.9360700845718384, 0.9265010356903076, 0.1422843635082245, 0.2661629915237427, 0.8947346806526184, 0.3167434334754944, 0.18447281420230865}, {0.8669211268424988, 0.36815768480300903, 0.165858656167984, 0.771864116191864, 0.13252472877502441, 0.4090457558631897, 0.18682244420051575, 0.4469970762729645, 0.5128973722457886, 0.3497181534767151, 0.6372016668319702, 0.9375365376472473, 0.8951495289802551, 0.9061537981033325, 0.14495849609375, 0.2018672227859497, 0.1352022886276245, 0.8865365386009216, 0.024942081421613693, 0.19941018521785736, 0.6342405676841736, 0.27409008145332336, 0.9679269790649414, 0.17646653950214386, 0.3501144051551819, 0.16721701622009277, 0.38787364959716797, 0.008795952424407005, 0.2558860778808594, 0.6578138470649719, 0.32605475187301636, 0.17369753122329712, 0.6777428388595581, 0.5382776260375977, 0.5279703140258789, 0.43300917744636536, 0.9738220572471619, 0.06835581362247467, 0.458285927772522, 0.46604636311531067, 0.44536107778549194, 0.6808859705924988, 0.6535545587539673, 0.39916473627090454, 0.4663047790527344, 0.38719797134399414, 0.06458307802677155, 0.8341341018676758, 0.13558466732501984, 0.5822280645370483, 0.7870935797691345, 0.4501533806324005, 0.662481963634491, 0.5488693714141846, 0.10891830921173096, 0.6797416806221008, 0.3303300738334656, 0.0600782111287117, 0.8045548796653748, 0.9862732291221619, 0.7400239109992981, 0.7788013815879822, 0.9494141340255737, 0.9363197088241577, 0.5432246923446655, 0.17500676214694977, 0.6431481242179871, 0.36360469460487366, 0.40133267641067505, 0.6570122241973877, 0.09881211072206497, 0.4105868339538574, 0.5091329216957092, 0.8610162138938904, 0.7739984393119812, 0.46392321586608887, 0.4080716073513031, 0.7516079545021057, 0.3791302740573883, 0.49366044998168945, 0.4132436513900757, 0.7174736857414246, 0.886665403842926, 0.5629512071609497, 0.9721388220787048, 0.22872532904148102, 0.03433385491371155, 0.00943799875676632, 0.5798096656799316, 0.1909264624118805, 0.8222677707672119, 0.016745448112487793, 0.511084258556366, 0.06495741009712219, 0.14120955765247345, 0.9629514813423157, 0.6830647587776184, 0.8176639676094055, 0.725540816783905, 0.04571941867470741}, {0.2592734098434448, 0.04632510989904404, 0.8343892693519592, 0.43122243881225586, 0.9194911122322083, 0.8160827159881592, 0.49925780296325684, 0.7379059195518494, 0.30970555543899536, 0.02304794080555439, 0.18669413030147552, 0.8321754336357117, 0.2305055856704712, 0.7587600946426392, 0.9995564222335815, 0.09588257223367691, 0.20216749608516693, 0.8587157130241394, 0.06444183737039566, 0.27317488193511963, 0.49506303668022156, 0.6719024777412415, 0.010022297501564026, 0.4611927270889282, 0.8418670296669006, 0.010824887081980705, 0.1793198585510254, 0.6050204634666443, 0.3539857566356659, 0.16390058398246765, 0.14814507961273193, 0.7703405022621155, 0.23288722336292267, 0.6592749357223511, 0.20937465131282806, 0.9499530792236328, 0.49285951256752014, 0.9373959302902222, 0.929985761642456, 0.1338324397802353, 0.11756031960248947, 0.9668318629264832, 0.6391550898551941, 0.7152281999588013, 0.6448414921760559, 0.3353985548019409, 0.3650469183921814, 0.06940542906522751, 0.5569133758544922, 0.04433688893914223, 0.834111750125885, 0.6348602175712585, 0.017215238884091377, 0.9738938808441162, 0.4132228493690491, 0.31833478808403015, 0.38572341203689575, 0.925649881362915, 0.12695209681987762, 0.7657207250595093, 0.7341939806938171, 0.1057746633887291, 0.13812611997127533, 0.1898823231458664, 0.2819012701511383, 0.5438086986541748, 0.7307684421539307, 0.9348100423812866, 0.10948529094457626, 0.5080608129501343, 0.033449266105890274, 0.06266193091869354, 0.6549425721168518, 0.34930405020713806, 0.7014538645744324, 0.9515066146850586, 0.049955178052186966, 0.32338470220565796, 0.034052763134241104, 0.23803593218326569, 0.1845509558916092, 0.2222607284784317, 0.37748515605926514, 0.3988100588321686, 0.7491109371185303, 0.8967569470405579, 0.21727199852466583, 0.4120529592037201, 0.3434588611125946, 0.21190421283245087, 0.5798752903938293, 0.742343544960022, 0.6971207857131958, 0.507460355758667, 0.9264171123504639, 0.19548451900482178, 0.9191358685493469, 0.44433897733688354, 0.21313537657260895, 0.8219811320304871}, {0.6935059428215027, 0.6472840905189514, 0.24204455316066742, 0.5298870801925659, 0.9852730631828308, 0.5567464232444763, 0.08702579885721207, 0.7994489669799805, 0.7130714654922485, 0.8322762250900269, 0.5899090766906738, 0.35925132036209106, 0.2759258449077606, 0.45516133308410645, 0.41782546043395996, 0.28903108835220337, 0.9233568906784058, 0.6634562015533447, 0.730734646320343, 0.5905724763870239, 0.2888011038303375, 0.5268843173980713, 0.020351629704236984, 0.10257197171449661, 0.8517123460769653, 0.5483375191688538, 0.3507947623729706, 0.36956602334976196, 0.04342710226774216, 0.2813534438610077, 0.6321601867675781, 0.9298556447029114, 0.6060380339622498, 0.37097829580307007, 0.7731184959411621, 0.2763359844684601, 0.9814321994781494, 0.1662123054265976, 0.5191299319267273, 0.36200520396232605, 0.6453766226768494, 0.9104200005531311, 0.08051437884569168, 0.9306401014328003, 0.9926319122314453, 0.8162986636161804, 0.25144654512405396, 0.7773726582527161, 0.07945767790079117, 0.4907928705215454, 0.42655429244041443, 0.06783817708492279, 0.4575654864311218, 0.8426811099052429, 0.6190935969352722, 0.3046984076499939, 0.4024951457977295, 0.39265769720077515, 0.8590608239173889, 0.6592487096786499, 0.3159739077091217, 0.3740595281124115, 0.8442128896713257, 0.19860094785690308, 0.45670363306999207, 0.5315158367156982, 0.8086910247802734, 0.464856892824173, 0.9017688632011414, 0.15852990746498108, 0.23512811958789825, 0.10716560482978821, 0.9402145147323608, 0.43035152554512024, 0.13473522663116455, 0.32047125697135925, 0.5253759622573853, 0.8990294933319092, 0.9786255359649658, 0.8848042488098145, 0.15165084600448608, 0.8883214592933655, 0.4925174117088318, 0.9594799876213074, 0.7710146903991699, 0.7246239185333252, 0.10724237561225891, 0.07247430831193924, 0.42829570174217224, 0.40271010994911194, 0.8700668215751648, 0.8940035700798035, 0.33660250902175903, 0.45800143480300903, 0.2321534901857376, 0.567036509513855, 0.24753455817699432, 0.8256025910377502, 0.6689889430999756, 0.9486442804336548}, {0.1126994639635086, 0.09307945519685745, 0.24374200403690338, 0.43929600715637207, 0.5129638314247131, 0.5905040502548218, 0.12355204671621323, 0.7569888234138489, 0.6279537081718445, 0.024163421243429184, 0.5078852772712708, 0.16828133165836334, 0.012009359896183014, 0.8227868676185608, 0.9810967445373535, 0.740993857383728, 0.02556086890399456, 0.9145419597625732, 0.7829480767250061, 0.509748101234436, 0.5742030143737793, 0.034293051809072495, 0.26596397161483765, 0.9773479700088501, 0.16163292527198792, 0.6027300357818604, 0.22914834320545197, 0.3433150351047516, 0.25914815068244934, 0.07872816920280457, 0.8970381021499634, 0.41592657566070557, 0.2865390181541443, 0.707776665687561, 0.09151151031255722, 0.15390628576278687, 0.2522433400154114, 0.09674494713544846, 0.18694327771663666, 0.5360972285270691, 0.12006925791501999, 0.5204224586486816, 0.7448114156723022, 0.09360143542289734, 0.6951050758361816, 0.4593958854675293, 0.7578353881835938, 0.9961994886398315, 0.41636013984680176, 0.7170155644416809, 0.19536522030830383, 0.7406581044197083, 0.4389572739601135, 0.9762611389160156, 0.006313467398285866, 0.8740648627281189, 0.028881561011075974, 0.7999874353408813, 0.6377642750740051, 0.34482789039611816, 0.9883281588554382, 0.4887737035751343, 0.8276399374008179, 0.9943976998329163, 0.3962274491786957, 0.5572819113731384, 0.8681337237358093, 0.3716067671775818, 0.740996241569519, 0.6249212026596069, 0.6751906871795654, 0.13406293094158173, 0.6317510008811951, 0.8962599635124207, 0.8116723895072937, 0.7708475589752197, 0.05562485754489899, 0.8131631016731262, 0.44498753547668457, 0.6091771125793457, 0.9350759387016296, 0.10262730717658997, 0.13827718794345856, 0.50814288854599, 0.813393235206604, 0.40026599168777466, 0.03298310190439224, 0.8046830296516418, 0.965013325214386, 0.29781803488731384, 0.18795500695705414, 0.5700693726539612, 0.8742364645004272, 0.4824490249156952, 0.7952057719230652, 0.03424303978681564, 0.3990592062473297, 0.5142384171485901, 0.7446214556694031, 0.2845999598503113}, {0.14158591628074646, 0.9894388318061829, 0.8644989132881165, 0.23112621903419495, 0.8954099416732788, 0.09410203993320465, 0.00818351935595274, 0.9935681819915771, 0.3863866627216339, 0.2698362171649933, 0.5548526644706726, 0.5462622046470642, 0.14751265943050385, 0.7150067090988159, 0.673098087310791, 0.7627561688423157, 0.3078043758869171, 0.5348944664001465, 0.5744673013687134, 0.06205951049923897, 0.49053388833999634, 0.8166218996047974, 0.7295191287994385, 0.6888973712921143, 0.30330491065979004, 0.3205706775188446, 0.6547218561172485, 0.7440839409828186, 0.8640329241752625, 0.2339021861553192, 0.1651260256767273, 0.26629579067230225, 0.16773131489753723, 0.2538740038871765, 0.30454808473587036, 0.8603765368461609, 0.9322543144226074, 0.43935447931289673, 0.4357442855834961, 0.3395636975765228, 0.03504533693194389, 0.8779190182685852, 0.5221381783485413, 0.977379560470581, 0.1824839860200882, 0.8270644545555115, 0.9333474636077881, 0.9866368174552917, 0.4136204421520233, 0.24401403963565826, 0.7425355911254883, 0.7411152720451355, 0.5204696655273438, 0.2806698679924011, 0.7594391703605652, 0.8240419626235962, 0.3844304084777832, 0.9233793616294861, 0.40613025426864624, 0.12321107089519501, 0.6463177800178528, 0.2665634751319885, 0.33420130610466003, 0.3219476044178009, 0.726912260055542, 0.5167638063430786, 0.9868592023849487, 0.44895315170288086, 0.15525013208389282, 0.5856553316116333, 0.12482081353664398, 0.627717137336731, 0.1899442821741104, 0.11080195009708405, 0.2993833124637604, 0.3186897039413452, 0.8265529274940491, 0.8382769227027893, 0.8662523627281189, 0.010517233982682228, 0.06534498184919357, 0.6667439937591553, 0.6650167107582092, 0.5197287201881409, 0.04286584630608559, 0.03028719685971737, 0.888602614402771, 0.08016955852508545, 0.6295958161354065, 0.8441973924636841, 0.3635384142398834, 0.6432298421859741, 0.05401994660496712, 0.48843392729759216, 0.3045792579650879, 0.9265453219413757, 0.5385898947715759, 0.024823760613799095, 0.05654551088809967, 0.9618955254554749}, {0.3595246374607086, 0.015644406899809837, 0.8237012028694153, 0.3171856701374054, 0.47728046774864197, 0.793558657169342, 0.1312631517648697, 0.8128612637519836, 0.13610854744911194, 0.44374918937683105, 0.2581455409526825, 0.4561651349067688, 0.058117154985666275, 0.6197424530982971, 0.5788138508796692, 0.17560015618801117, 0.4955167770385742, 0.9562581181526184, 0.8244495987892151, 0.9630988836288452, 0.9635026454925537, 0.7377224564552307, 0.2742028534412384, 0.7663435339927673, 0.010839897207915783, 0.48168063163757324, 0.6944452524185181, 0.801559329032898, 0.6697244048118591, 0.8523672223091125, 0.5463743805885315, 0.051361288875341415, 0.7215512990951538, 0.5106332302093506, 0.46963614225387573, 0.6879914999008179, 0.5935148000717163, 0.1730358600616455, 0.886075496673584, 0.7450346350669861, 0.842242419719696, 0.6100168228149414, 0.03140519559383392, 0.8992104530334473, 0.947533905506134, 0.49097713828086853, 0.45751675963401794, 0.6938045620918274, 0.7285463809967041, 0.9250038266181946, 0.7145497798919678, 0.8375188708305359, 0.08250920474529266, 0.4446885585784912, 0.7407788038253784, 0.04463716596364975, 0.944030225276947, 0.06727632880210876, 0.04115300625562668, 0.21912550926208496, 0.39348578453063965, 0.07215296477079391, 0.16378739476203918, 0.03033575415611267, 0.5011312365531921, 0.14208289980888367, 0.73177170753479, 0.040021274238824844, 0.9784971475601196, 0.571783721446991, 0.015126371756196022, 0.04897293820977211, 0.759748101234436, 0.3378591239452362, 0.48531973361968994, 0.9508386850357056, 0.15752390027046204, 0.5589243173599243, 0.48387959599494934, 0.7000361680984497, 0.15199284255504608, 0.7405045628547668, 0.41831982135772705, 0.018990227952599525, 0.1590518057346344, 0.6787817478179932, 0.8603785634040833, 0.1799589842557907, 0.33337870240211487, 0.7933233976364136, 0.9938428997993469, 0.9346528649330139, 0.2233249694108963, 0.6874666810035706, 0.2857286334037781, 0.07542283833026886, 0.7063985466957092, 0.6530174612998962, 0.344872385263443, 0.9702763557434082}, {0.32021364569664, 0.44930484890937805, 0.27076277136802673, 0.33862707018852234, 0.5772673487663269, 0.49256864190101624, 0.9040158987045288, 0.08509315550327301, 0.6386123895645142, 0.6781555414199829, 0.4715403616428375, 0.2540370225906372, 0.47182294726371765, 0.6612448692321777, 0.880096971988678, 0.6666243672370911, 0.8209695219993591, 0.891534686088562, 0.919111430644989, 0.453313946723938, 0.15196162462234497, 0.7040517926216125, 0.13546425104141235, 0.0750289112329483, 0.49951088428497314, 0.1427038311958313, 0.33116617798805237, 0.9474323987960815, 0.7679321765899658, 0.09538254141807556, 0.2488974630832672, 0.8327236175537109, 0.9116449356079102, 0.5052281618118286, 0.25076186656951904, 0.22882401943206787, 0.6882458925247192, 0.93309485912323, 0.049995485693216324, 0.3328893780708313, 0.717863917350769, 0.570625901222229, 0.7134110927581787, 0.06788928061723709, 0.06336144357919693, 0.5123299360275269, 0.0482330359518528, 0.4306398332118988, 0.09677506238222122, 0.6950913667678833, 0.9430964589118958, 0.766758918762207, 0.9465560913085938, 0.11598630249500275, 0.20450647175312042, 0.3391510844230652, 0.2684715688228607, 0.3533512055873871, 0.46350783109664917, 0.48215582966804504, 0.10398940742015839, 0.25608471035957336, 0.7984510064125061, 0.8777235746383667, 0.20877566933631897, 0.4256763458251953, 0.38566818833351135, 0.07916337251663208, 0.49968740344047546, 0.4075824022293091, 0.8555209636688232, 0.08855324983596802, 0.520043671131134, 0.15258978307247162, 0.3745286166667938, 0.04086890071630478, 0.18167316913604736, 0.10228327661752701, 0.36201003193855286, 0.5461273789405823, 0.16417928040027618, 0.31119439005851746, 0.6954352855682373, 0.8892318606376648, 0.6824262738227844, 0.8725188374519348, 0.04712500795722008, 0.18759599328041077, 0.9656315445899963, 0.8155597448348999, 0.405221164226532, 0.9473857879638672, 0.9095763564109802, 0.8166155219078064, 0.9609527587890625, 0.6868627667427063, 0.6195418834686279, 0.7327517867088318, 0.49306511878967285, 0.3258731961250305}, {0.4351404905319214, 0.5506219863891602, 0.8307152986526489, 0.06223570182919502, 0.0811542496085167, 0.43117445707321167, 0.595363199710846, 0.750495433807373, 0.8914264440536499, 0.31217190623283386, 0.20698808133602142, 0.8369845747947693, 0.9819018840789795, 0.06616023927927017, 0.44070136547088623, 0.030229749158024788, 0.8662369251251221, 0.9149036407470703, 0.46458861231803894, 0.42022594809532166, 0.49681800603866577, 0.7525603175163269, 0.17552031576633453, 0.9293813705444336, 0.7439526915550232, 0.7629372477531433, 0.9511502385139465, 0.5883586406707764, 0.04850391298532486, 0.6473788619041443, 0.8591592311859131, 0.3141445517539978, 0.9088594317436218, 0.3431289494037628, 0.7206647992134094, 0.4765496850013733, 0.22006188333034515, 0.7668817639350891, 0.2422538697719574, 0.024131987243890762, 0.9498220682144165, 0.7842720746994019, 0.6519882082939148, 0.5120458602905273, 0.9221348166465759, 0.3656293451786041, 0.9926120042800903, 0.6143932938575745, 0.7232574820518494, 0.034716177731752396, 0.12461952865123749, 0.13154804706573486, 0.7857484817504883, 0.22531630098819733, 0.7271441221237183, 0.36332130432128906, 0.25180351734161377, 0.8850919008255005, 0.45936858654022217, 0.045391619205474854, 0.09830622375011444, 0.9311406016349792, 0.19092856347560883, 0.6922482252120972, 0.711418092250824, 0.02892356924712658, 0.06930160522460938, 0.8049849271774292, 0.21808524429798126, 0.6552045941352844, 0.8318417072296143, 0.0750291496515274, 0.8599295020103455, 0.5637354850769043, 0.2657473683357239, 0.8881675004959106, 0.49890220165252686, 0.001465430948883295, 0.6516809463500977, 0.5045510530471802, 0.23443301022052765, 0.07456964999437332, 0.6963704228401184, 0.8201296329498291, 0.5359516739845276, 0.02441943623125553, 0.1234341561794281, 0.9009141325950623, 0.10457517206668854, 0.9447148442268372, 0.136751189827919, 0.8373143076896667, 0.6660463213920593, 0.5491982102394104, 0.9416305422782898, 0.8952114582061768, 0.690021276473999, 0.4592910408973694, 0.2321869283914566, 0.3465687334537506}, {0.43018901348114014, 0.7830695509910583, 0.9977756142616272, 0.31792977452278137, 0.49298617243766785, 0.7321251034736633, 0.05042863264679909, 0.7021380066871643, 0.180597722530365, 0.11701752245426178, 0.17014488577842712, 0.6318798661231995, 0.8803346753120422, 0.18122898042201996, 0.11292793601751328, 0.3437618613243103, 0.3424909710884094, 0.6615437865257263, 0.9608468413352966, 0.9792557954788208, 0.4787079989910126, 0.034063417464494705, 0.7846269607543945, 0.4964370131492615, 0.7377758622169495, 0.21936620771884918, 0.15089742839336395, 0.8884356617927551, 0.21474258601665497, 0.3886024057865143, 0.6803416609764099, 0.32311931252479553, 0.9886201024055481, 0.8960013389587402, 0.6851202845573425, 0.5071201324462891, 0.3080689013004303, 0.25975945591926575, 0.4083215594291687, 0.47300854325294495, 0.8647559881210327, 0.6378809213638306, 0.04654626548290253, 0.07028163969516754, 0.6718665957450867, 0.9845177531242371, 0.1648453027009964, 0.653453528881073, 0.8382250666618347, 0.8219540119171143, 0.2269212156534195, 0.3702545166015625, 0.31592532992362976, 0.7154615521430969, 0.4395044445991516, 0.3965953290462494, 0.9705113768577576, 0.24000142514705658, 0.3248765468597412, 0.801903486251831, 0.7383080124855042, 0.30669984221458435, 0.19016465544700623, 0.8216519951820374, 0.7211742401123047, 0.4930526912212372, 0.8239443302154541, 0.6447453498840332, 0.8150322437286377, 0.14202077686786652, 0.8286712169647217, 0.8150535225868225, 0.11639901250600815, 0.4488459825515747, 0.5242099165916443, 0.27179455757141113, 0.42299309372901917, 0.5953565239906311, 0.16129250824451447, 0.5794046521186829, 0.582732081413269, 0.622740626335144, 0.36460158228874207, 0.29250600934028625, 0.010794933885335922, 0.7545866966247559, 0.779164731502533, 0.6945501565933228, 0.07534100860357285, 0.3721180260181427, 0.8236485123634338, 0.8521074056625366, 0.7153415679931641, 0.4089626967906952, 0.4831121563911438, 0.7070133686065674, 0.19906845688819885, 0.8759191632270813, 0.3923948407173157, 0.44580286741256714}, {0.6022211313247681, 0.8717324137687683, 0.40150904655456543, 0.8615702986717224, 0.9145747423171997, 0.32931971549987793, 0.2204192727804184, 0.8611003160476685, 0.11881495267152786, 0.008585409261286259, 0.7386065721511841, 0.7708781957626343, 0.6446359753608704, 0.23404276371002197, 0.6192210912704468, 0.9154004454612732, 0.10697920620441437, 0.3552491068840027, 0.08226881176233292, 0.18242132663726807, 0.7501086592674255, 0.34912997484207153, 0.5982781052589417, 0.6384850144386292, 0.6549445986747742, 0.9506505131721497, 0.9831375479698181, 0.9896630048751831, 0.8070104718208313, 0.7985230088233948, 0.34897735714912415, 0.4380933344364166, 0.42824575304985046, 0.8997424840927124, 0.24080654978752136, 0.5674653053283691, 0.9652913212776184, 0.8440278172492981, 0.6234805583953857, 0.8554086685180664, 0.37702804803848267, 0.11983590573072433, 0.9987509846687317, 0.826131284236908, 0.567003071308136, 0.40335702896118164, 0.09330299496650696, 0.4209337830543518, 0.15667615830898285, 0.7390708923339844, 0.49877846240997314, 0.3887925446033478, 0.6097532510757446, 0.48456206917762756, 0.4037645757198334, 0.486393541097641, 0.9730862379074097, 0.5711051821708679, 0.5458903312683105, 0.2930813729763031, 0.8434167504310608, 0.155650332570076, 0.5650189518928528, 0.46644920110702515, 0.02278777025640011, 0.2773032784461975, 0.29599082469940186, 0.3606446385383606, 0.034449253231287, 0.1348293274641037, 0.13389922678470612, 0.6612080335617065, 0.6544014811515808, 0.055441830307245255, 0.5471386909484863, 0.34265565872192383, 0.10714337229728699, 0.3144555687904358, 0.06411606073379517, 0.3917570412158966, 0.18407142162322998, 0.8866783380508423, 0.931183934211731, 0.1577170342206955, 0.37082183361053467, 0.21537823975086212, 0.20158551633358002, 0.633293628692627, 0.6784362196922302, 0.8033651113510132, 0.4617955684661865, 0.1547943353652954, 0.7852190136909485, 0.1814451366662979, 0.47201624512672424, 0.07658195495605469, 0.7220648527145386, 0.9201490879058838, 0.7735423445701599, 0.09301544725894928}, {0.18666817247867584, 0.8938629627227783, 0.046912435442209244, 0.8831529021263123, 0.6331491470336914, 0.494606614112854, 0.5524553656578064, 0.20058882236480713, 0.10561631619930267, 0.5519490242004395, 0.861983597278595, 0.7959479093551636, 0.1448749601840973, 0.2608501613140106, 0.22717420756816864, 0.20957359671592712, 0.34194278717041016, 0.17793183028697968, 0.8966529965400696, 0.9994150996208191, 0.35587939620018005, 0.6836534738540649, 0.1464490294456482, 0.9472026824951172, 0.2703109085559845, 0.24830301105976105, 0.9711320996284485, 0.6860600113868713, 0.35990819334983826, 0.920881986618042, 0.6852526664733887, 0.23666028678417206, 0.8915047645568848, 0.3064103424549103, 0.05006742104887962, 0.4695990979671478, 0.06462880969047546, 0.823455810546875, 0.7255615592002869, 0.287970632314682, 0.6815269589424133, 0.5462414622306824, 0.4528099298477173, 0.3380372226238251, 0.42895570397377014, 0.0932217612862587, 0.47283923625946045, 0.42539116740226746, 0.2887956500053406, 0.1812736839056015, 0.8766398429870605, 0.3177809715270996, 0.4954701066017151, 0.3167022466659546, 0.9117782115936279, 0.3639521896839142, 0.4617667496204376, 0.6371231079101562, 0.9837498068809509, 0.8428800702095032, 0.4638303518295288, 0.7747945189476013, 0.06453697383403778, 0.05748649314045906, 0.4836210012435913, 0.23087887465953827, 0.3623667061328888, 0.0014921589754521847, 0.5073906779289246, 0.27849456667900085, 0.6195497512817383, 0.6027819514274597, 0.23804278671741486, 0.44495314359664917, 0.02179589867591858, 0.48403096199035645, 0.00586031936109066, 0.3934267461299896, 0.7958403825759888, 0.17689508199691772, 0.8157885074615479, 0.3438055217266083, 0.6088124513626099, 0.36535269021987915, 0.5661991238594055, 0.6532983779907227, 0.01380864530801773, 0.5753710865974426, 0.027166040614247322, 0.6432704329490662, 0.1245172917842865, 0.9205299615859985, 0.21685412526130676, 0.5220300555229187, 0.8666890859603882, 0.08413581550121307, 0.8828323483467102, 0.5586241483688354, 0.621701180934906, 0.34595799446105957}, {0.5279501676559448, 0.26292452216148376, 0.5756978392601013, 0.5332962274551392, 0.6263255476951599, 0.38161158561706543, 0.869046688079834, 0.8477224111557007, 0.21321909129619598, 0.14645245671272278, 0.5704102516174316, 0.06488068401813507, 0.10423962771892548, 0.9167599678039551, 0.5074717402458191, 0.22106947004795074, 0.7781272530555725, 0.44959530234336853, 0.29535526037216187, 0.9125298261642456, 0.005682636052370071, 0.016484638676047325, 0.5024005770683289, 0.12297473847866058, 0.7850502133369446, 0.2936021387577057, 0.7435901761054993, 0.24165701866149902, 0.19789820909500122, 0.7897587418556213, 0.9576032161712646, 0.25445207953453064, 0.13488857448101044, 0.14840394258499146, 0.5007365942001343, 0.2492176592350006, 0.07542450726032257, 0.517340898513794, 0.1221347227692604, 0.04352088272571564, 0.5055527687072754, 0.9187526106834412, 0.5246134996414185, 0.6450484991073608, 0.6851782202720642, 0.639417290687561, 0.0596623495221138, 0.538537859916687, 0.026326538994908333, 0.9404393434524536, 0.8802086710929871, 0.9711467623710632, 0.8011273145675659, 0.4830455482006073, 0.9548187255859375, 0.4025169312953949, 0.38467955589294434, 0.8207676410675049, 0.279538094997406, 0.3540470004081726, 0.1609068214893341, 0.4757477641105652, 0.8835775256156921, 0.03789873048663139, 0.02105797454714775, 0.04606904089450836, 0.6578555703163147, 0.17478704452514648, 0.4034000635147095, 0.7153802514076233, 0.46447283029556274, 0.5295059084892273, 0.7785154581069946, 0.6107467412948608, 0.8739664554595947, 0.929443359375, 0.6919869780540466, 0.1196838915348053, 0.3423672020435333, 0.787612795829773, 0.4260473847389221, 0.9418649673461914, 0.014774108305573463, 0.7269272804260254, 0.8960112929344177, 0.6954079866409302, 0.565382182598114, 0.4481039345264435, 0.38698136806488037, 0.18950942158699036, 0.22779403626918793, 0.6071166396141052, 0.7046706676483154, 0.49018916487693787, 0.9144384264945984, 0.5308683514595032, 0.22153377532958984, 0.9957075119018555, 0.04819885268807411, 0.6899130940437317}, {0.7157199382781982, 0.17678797245025635, 0.04299233853816986, 0.03857436776161194, 0.696549654006958, 0.34543168544769287, 0.5075585842132568, 0.5138590931892395, 0.7541447877883911, 0.04257521033287048, 0.3524869382381439, 0.8486941456794739, 0.9805752635002136, 0.18344880640506744, 0.6052567958831787, 0.6819365620613098, 0.5331070423126221, 0.19264550507068634, 0.7623668909072876, 0.10393202304840088, 0.8551783561706543, 0.3586246967315674, 0.3455011248588562, 0.14770762622356415, 0.8843615055084229, 0.00994262658059597, 0.38593247532844543, 0.3586876094341278, 0.40980857610702515, 0.23181095719337463, 0.5017972588539124, 0.8997052311897278, 0.5702056884765625, 0.1241864338517189, 0.3236975371837616, 0.7691689729690552, 0.9488232731819153, 0.4535582959651947, 0.8736552596092224, 0.3682936429977417, 0.8187716007232666, 0.509346604347229, 0.540213406085968, 0.9670263528823853, 0.4979300796985626, 0.3881174623966217, 0.7118430733680725, 0.5073408484458923, 0.6416149735450745, 0.31624358892440796, 0.004471204709261656, 0.030396098271012306, 0.4939144551753998, 0.6962704658508301, 0.7234509587287903, 0.7059664726257324, 0.9726656675338745, 0.18495839834213257, 0.3166622221469879, 0.8463168144226074, 0.3311848044395447, 0.008518347516655922, 0.3965624272823334, 0.4900347590446472, 0.9495696425437927, 0.13713979721069336, 0.9907256960868835, 0.7952319383621216, 0.10885169357061386, 0.21463513374328613, 0.7491037249565125, 0.25543466210365295, 0.006171686574816704, 0.3704662322998047, 0.1606699824333191, 0.679152250289917, 0.09670611470937729, 0.5810272097587585, 0.20748290419578552, 0.6018441915512085, 0.047192107886075974, 0.30661946535110474, 0.04832574725151062, 0.0690251961350441, 0.4483832120895386, 0.9394779801368713, 0.8571441173553467, 0.26169535517692566, 0.13485898077487946, 0.9908509254455566, 0.8544480204582214, 0.7704305052757263, 0.5125952959060669, 0.9113317131996155, 0.09749014675617218, 0.9918539524078369, 0.5718641877174377, 0.8855401277542114, 0.057056576013565063, 0.771869957447052}, {0.3738524615764618, 0.4328654706478119, 0.09337517619132996, 0.6214812994003296, 0.04829534888267517, 0.585369884967804, 0.01631159894168377, 0.619729220867157, 0.671230137348175, 0.2680676281452179, 0.505603551864624, 0.6149173378944397, 0.4130804240703583, 0.8037636280059814, 0.4398295283317566, 0.6819040775299072, 0.6298449635505676, 0.12352900207042694, 0.8823726773262024, 0.09301440417766571, 0.795400857925415, 0.11199989169836044, 0.24745459854602814, 0.3230143189430237, 0.9522770643234253, 0.8316348791122437, 0.43988585472106934, 0.7292057275772095, 0.4457238018512726, 0.8187711238861084, 0.9972192049026489, 0.9769166111946106, 0.10855907201766968, 0.8028360605239868, 0.5668151378631592, 0.4367067813873291, 0.763792097568512, 0.7878431677818298, 0.7386108636856079, 0.5496728420257568, 0.7911450266838074, 0.514846682548523, 0.8668429851531982, 0.4827370047569275, 0.34553995728492737, 0.26912322640419006, 0.376865416765213, 0.4011807143688202, 0.015924064442515373, 0.6388203501701355, 0.8471598029136658, 0.8738680481910706, 0.7566604614257812, 0.006441984325647354, 0.28505176305770874, 0.2823806405067444, 0.6788814067840576, 0.18796642124652863, 0.7789177894592285, 0.5399688482284546, 0.8560553193092346, 0.9435098171234131, 0.010785899125039577, 0.9764243960380554, 0.7728333473205566, 0.6009365320205688, 0.6924855709075928, 0.6185959577560425, 0.5724595189094543, 0.3580794632434845, 0.05894007906317711, 0.17560307681560516, 0.13459229469299316, 0.5904291272163391, 0.6251096129417419, 0.5468416810035706, 0.7070238590240479, 0.9533355832099915, 0.8963091373443604, 0.3086216449737549, 0.4817936420440674, 0.22774957120418549, 0.6430849432945251, 0.6897240281105042, 0.7436901330947876, 0.584242582321167, 0.297488272190094, 0.49521180987358093, 0.302665114402771, 0.16669973731040955, 0.45860782265663147, 0.4926176071166992, 0.5248021483421326, 0.6845218539237976, 0.5778385400772095, 0.3659607470035553, 0.4750555157661438, 0.05384783446788788, 0.1254538595676422, 0.481513649225235}, {0.3797779679298401, 0.9129850268363953, 0.18700124323368073, 0.13949862122535706, 0.7569432258605957, 0.7587239146232605, 0.28881871700286865, 0.5259809494018555, 0.13347406685352325, 0.8535099029541016, 0.4080195724964142, 0.9996575713157654, 0.6974131464958191, 0.35759735107421875, 0.5987687706947327, 0.7392566204071045, 0.9826719164848328, 0.855799674987793, 0.28274068236351013, 0.43825897574424744, 0.4524584710597992, 0.77276211977005, 0.1072235107421875, 0.5910853147506714, 0.8182784914970398, 0.18967270851135254, 0.5017687082290649, 0.3858591616153717, 0.7335191965103149, 0.5307231545448303, 0.12065862119197845, 0.457227885723114, 0.8284981846809387, 0.05641675740480423, 0.16375885903835297, 0.8891246318817139, 0.9399895071983337, 0.8229392170906067, 0.8962270617485046, 0.3699582815170288, 0.03596211224794388, 0.4606669843196869, 0.7988042831420898, 0.7769170999526978, 0.020026616752147675, 0.46290069818496704, 0.6554597616195679, 0.4445924460887909, 0.2075432687997818, 0.9146659970283508, 0.1465786099433899, 0.5026050209999084, 0.1370447874069214, 0.2265583574771881, 0.48773133754730225, 0.4472971558570862, 0.07203295081853867, 0.11043624579906464, 0.8967539072036743, 0.11624301970005035, 0.0655287578701973, 0.0977732315659523, 0.5453234314918518, 0.2907891273498535, 0.3325314223766327, 0.3079296052455902, 0.867755651473999, 0.6474264860153198, 0.8172875642776489, 0.33732596039772034, 0.5649465322494507, 0.3426124155521393, 0.06708774715662003, 0.059609491378068924, 0.8410245180130005, 0.7629515528678894, 0.059260107576847076, 0.7794330716133118, 0.9500747919082642, 0.2054639756679535, 0.8235400319099426, 0.33569416403770447, 0.9989604353904724, 0.3921348750591278, 0.23730550706386566, 0.85655677318573, 0.9786080718040466, 0.6139028668403625, 0.03372771292924881, 0.6006152629852295, 0.19340921938419342, 0.4982711374759674, 0.9177770614624023, 0.9826765060424805, 0.7236142158508301, 0.37605446577072144, 0.09977801889181137, 0.897087037563324, 0.0011175506515428424, 0.8750755786895752}, {0.19865216314792633, 0.04889310151338577, 0.3601321876049042, 0.4769631326198578, 0.9660608768463135, 0.21001704037189484, 0.2593137323856354, 0.49200260639190674, 0.6870127320289612, 0.5789706110954285, 0.9304196238517761, 0.26922106742858887, 0.5375803112983704, 0.9817799925804138, 0.09287695586681366, 0.8801824450492859, 0.9447261095046997, 0.08705510944128036, 0.6260132789611816, 0.2366791069507599, 0.6909874677658081, 0.5207390785217285, 0.1979919672012329, 0.07157519459724426, 0.3578031659126282, 0.9586310982704163, 0.03938094899058342, 0.4804525673389435, 0.4427616596221924, 0.09676625579595566, 0.5622568726539612, 0.03465487062931061, 0.21176689863204956, 0.9897665977478027, 0.8143267035484314, 0.7044436931610107, 0.4508053958415985, 0.9452852606773376, 0.49850907921791077, 0.6225669384002686, 0.7254256010055542, 0.08617766946554184, 0.8286450505256653, 0.8665683269500732, 0.6690223813056946, 0.38204649090766907, 0.12395774573087692, 0.18397384881973267, 0.5388907194137573, 0.8403496742248535, 0.7080838084220886, 0.8074619770050049, 0.029622908681631088, 0.7091720700263977, 0.4172583520412445, 0.053847335278987885, 0.4903284013271332, 0.040840648114681244, 0.6093703508377075, 0.7829938530921936, 0.7031399607658386, 0.5867789387702942, 0.5813472867012024, 0.3948761522769928, 0.10873691737651825, 0.2091619074344635, 0.23747855424880981, 0.8436508774757385, 0.33412808179855347, 0.6547412276268005, 0.9463117122650146, 0.9994385242462158, 0.3569124639034271, 0.5363837480545044, 0.5815562605857849, 0.578728199005127, 0.10045459121465683, 0.06303480267524719, 0.4395662844181061, 0.49375730752944946, 0.437437504529953, 0.0758853480219841, 0.5481367111206055, 0.5292810201644897, 0.4687748849391937, 0.09054355323314667, 0.9251340627670288, 0.5297459363937378, 0.9570770263671875, 0.566440761089325, 0.2789801359176636, 0.9996487498283386, 0.668342113494873, 0.7737460732460022, 0.9612116813659668, 0.17626173794269562, 0.45327550172805786, 0.744842529296875, 0.43035730719566345, 0.7691805958747864}}, "Real32"], "Real32", ColorSpace -> Automatic, Interleaving -> None]"#,
    );
  }
  #[test]
  fn image_channels() {
    assert_case(r#"ImageChannels[Image[{{0, 1}, {1, 0}}]]"#, r#"1"#);
  }
  #[test]
  fn image_data_1() {
    assert_case(
      r#"img = Image[{{0.2, 0.4}, {0.9, 0.6}, {0.5, 0.8}}]; ImageData[img]"#,
      r#"{{0.20000000298023224, 0.4000000059604645}, {0.8999999761581421, 0.6000000238418579}, {0.5, 0.800000011920929}}"#,
    );
  }
  #[test]
  fn image_data_2() {
    assert_case(
      r#"img = Image[{{0.2, 0.4}, {0.9, 0.6}, {0.5, 0.8}}]; ImageData[img]; ImageData[img, "Byte"]"#,
      r#"{{51, 102}, {230, 153}, {128, 204}}"#,
    );
  }
  #[test]
  fn image_data_3() {
    assert_case(
      r#"img = Image[{{0.2, 0.4}, {0.9, 0.6}, {0.5, 0.8}}]; ImageData[img]; ImageData[img, "Byte"]; ImageData[Image[{{0, 1}, {1, 0}, {1, 1}}], "Bit"]"#,
      r#"{{0, 1}, {1, 0}, {1, 1}}"#,
    );
  }
  #[test]
  fn image_q_1() {
    assert_case(r#"ImageQ[Image[{{0, 1}, {1, 0}}]]"#, r#"True"#);
  }
  #[test]
  fn image_q_2() {
    assert_case(
      r#"ImageQ[Image[{{0, 1}, {1, 0}}]]; ImageQ[Image[{{{0, 0, 0}, {0, 1, 0}}, {{0, 1, 0}, {0, 1, 1}}}]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn image_add_1() {
    assert_case(
      r#"i = Image[{{0, 0.5, 0.2, 0.1, 0.9}, {1.0, 0.1, 0.3, 0.8, 0.6}}]; ImageAdd[i, 0.5]"#,
      r#"Image[NumericArray[{{0.5, 1., 0.699999988079071, 0.6000000238418579, 1.399999976158142}, {1.5, 0.6000000238418579, 0.800000011920929, 1.2999999523162842, 1.100000023841858}}, "Real32"], "Real32", ColorSpace -> Automatic, Interleaving -> None]"#,
    );
  }
  #[test]
  fn image_add_2() {
    assert_case(
      r#"i = Image[{{0, 0.5, 0.2, 0.1, 0.9}, {1.0, 0.1, 0.3, 0.8, 0.6}}]; ImageAdd[i, 0.5]; ImageAdd[i, i]"#,
      r#"Image[NumericArray[{{0., 1., 0.4000000059604645, 0.20000000298023224, 1.7999999523162842}, {2., 0.20000000298023224, 0.6000000238418579, 1.600000023841858, 1.2000000476837158}}, "Real32"], "Real32", ColorSpace -> Automatic, Interleaving -> None]"#,
    );
  }
  #[test]
  fn image_multiply_1() {
    assert_case(
      r#"i = Image[{{0, 0.5, 0.2, 0.1, 0.9}, {1.0, 0.1, 0.3, 0.8, 0.6}}]; ImageMultiply[i, 0.2]"#,
      r#"Image[NumericArray[{{0., 0.10000000149011612, 0.03999999910593033, 0.019999999552965164, 0.17999999225139618}, {0.20000000298023224, 0.019999999552965164, 0.06000000238418579, 0.1599999964237213, 0.12000000476837158}}, "Real32"], "Real32", ColorSpace -> Automatic, Interleaving -> None]"#,
    );
  }
  #[test]
  fn image_multiply_2() {
    assert_case(
      r#"i = Image[{{0, 0.5, 0.2, 0.1, 0.9}, {1.0, 0.1, 0.3, 0.8, 0.6}}]; ImageMultiply[i, 0.2]; ImageMultiply[i, i]"#,
      r#"Image[NumericArray[{{0., 0.25, 0.04000000283122063, 0.010000000707805157, 0.809999942779541}, {1., 0.010000000707805157, 0.09000000357627869, 0.64000004529953, 0.36000001430511475}}, "Real32"], "Real32", ColorSpace -> Automatic, Interleaving -> None]"#,
    );
  }
  #[test]
  fn image_subtract_1() {
    assert_case(
      r#"i = Image[{{0, 0.5, 0.2, 0.1, 0.9}, {1.0, 0.1, 0.3, 0.8, 0.6}}]; ImageSubtract[i, 0.2]"#,
      r#"Image[NumericArray[{{-0.20000000298023224, 0.30000001192092896, 2.9802322831784522*^-9, -0.10000000149011612, 0.699999988079071}, {0.800000011920929, -0.10000000149011612, 0.10000000894069672, 0.6000000238418579, 0.40000003576278687}}, "Real32"], "Real32", ColorSpace -> Automatic, Interleaving -> None]"#,
    );
  }
  #[test]
  fn image_subtract_2() {
    assert_case(
      r#"i = Image[{{0, 0.5, 0.2, 0.1, 0.9}, {1.0, 0.1, 0.3, 0.8, 0.6}}]; ImageSubtract[i, 0.2]; ImageSubtract[i, i]"#,
      r#"Image[NumericArray[{{0., 0., 0., 0., 0.}, {0., 0., 0., 0., 0.}}, "Real32"], "Real32", ColorSpace -> Automatic, Interleaving -> None]"#,
    );
  }
  // PixelValue[img, {x, y}] uses 1-indexed (x=column from left, y=row from
  // bottom); out-of-bounds positions return 0 (or {0,...} for multi-channel).
  #[test]
  fn pixel_value_grayscale_single_row() {
    assert_case(
      r#"PixelValue[Image[{{0.2, 0.5, 0.8}}], {1, 1}]"#,
      r#"0.20000000298023224"#,
    );
    assert_case(r#"PixelValue[Image[{{0.2, 0.5, 0.8}}], {2, 1}]"#, r#"0.5"#);
    assert_case(
      r#"PixelValue[Image[{{0.2, 0.5, 0.8}}], {3, 1}]"#,
      r#"0.800000011920929"#,
    );
  }

  // y is row counted from the BOTTOM (1-indexed). For a 2x2 image stored
  // top-row-first, position {1,1} reads the bottom-left pixel.
  #[test]
  fn pixel_value_grayscale_y_axis_orientation() {
    assert_case(
      r#"PixelValue[Image[{{0.1, 0.2}, {0.3, 0.4}}], {1, 1}]"#,
      r#"0.30000001192092896"#,
    );
    assert_case(
      r#"PixelValue[Image[{{0.1, 0.2}, {0.3, 0.4}}], {1, 2}]"#,
      r#"0.10000000149011612"#,
    );
    assert_case(
      r#"PixelValue[Image[{{0.1, 0.2}, {0.3, 0.4}}], {2, 1}]"#,
      r#"0.4000000059604645"#,
    );
    assert_case(
      r#"PixelValue[Image[{{0.1, 0.2}, {0.3, 0.4}}], {2, 2}]"#,
      r#"0.20000000298023224"#,
    );
  }

  #[test]
  fn pixel_value_rgb_returns_channel_list() {
    assert_case(
      r#"PixelValue[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}], {1, 1}]"#,
      r#"{1., 0., 0.}"#,
    );
    assert_case(
      r#"PixelValue[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}], {2, 1}]"#,
      r#"{0., 1., 0.}"#,
    );
  }

  #[test]
  fn pixel_value_out_of_bounds_grayscale() {
    assert_case(
      r#"PixelValue[Image[{{0.1, 0.2}, {0.3, 0.4}}], {0, 0}]"#,
      r#"0."#,
    );
    assert_case(
      r#"PixelValue[Image[{{0.1, 0.2}, {0.3, 0.4}}], {3, 1}]"#,
      r#"0."#,
    );
  }

  #[test]
  fn pixel_value_out_of_bounds_rgb() {
    assert_case(
      r#"PixelValue[Image[{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}}], {3, 1}]"#,
      r#"{0., 0., 0.}"#,
    );
  }

  // Non-image argument: matches wolframscript's PixelValue::imginv warning,
  // returns the call unevaluated.
  #[test]
  fn pixel_value_non_image_unevaluated() {
    assert_case(r#"PixelValue[42, {1, 1}]"#, r#"PixelValue[42, {1, 1}]"#);
  }

  #[test]
  fn pixel_value_positions_1() {
    assert_case(
      r#"PixelValuePositions[Image[{{0, 1}, {1, 0}, {1, 1}}], 1]"#,
      r#"{{2, 3}, {1, 2}, {1, 1}, {2, 1}}"#,
    );
  }
  #[test]
  fn pixel_value_positions_2() {
    assert_case(
      r#"PixelValuePositions[Image[{{0, 1}, {1, 0}, {1, 1}}], 1]; PixelValuePositions[Image[{{0.2, 0.4}, {0.9, 0.6}, {0.3, 0.8}}], 0.5, 0.15]"#,
      r#"{{2, 3}, {2, 2}}"#,
    );
  }
  #[test]
  fn color_negate() {
    assert_case(r#"ColorNegate[Yellow] == Blue"#, r#"True"#);
  }
  #[test]
  fn image_1() {
    assert_case(
      r#"Image[{{{1,1,0},{0,1,1}}, {{1,0,1},{1,1,0}}}]"#,
      r#"Image[NumericArray[{{{1., 1., 0.}, {0., 1., 1.}}, {{1., 0., 1.}, {1., 1., 0.}}}, "Real32"], "Real32", ColorSpace -> Automatic, Interleaving -> True]"#,
    );
  }
  #[test]
  fn image_2() {
    assert_case(
      r#"Image[{{{1,1,0},{0,1,1}}, {{1,0,1},{1,1,0}}}]; Image[{{{0,0,0,0.25},{0,0,0,0.5}}, {{0,0,0,0.5},{0,0,0,0.75}}}]"#,
      r#"Image[NumericArray[{{{0., 0., 0., 0.25}, {0., 0., 0., 0.5}}, {{0., 0., 0., 0.5}, {0., 0., 0., 0.75}}}, "Real32"], "Real32", ColorSpace -> Automatic, Interleaving -> True]"#,
    );
  }
}

mod color_negate_numericizes {
  use super::*;

  // ColorNegate numericizes components, so an exact 1 becomes the machine
  // real 0. (matching wolframscript's RGBColor[0., 1., 1.]).
  #[test]
  fn integer_components_become_reals() {
    clear_state();
    assert_eq!(
      interpret("ColorNegate[RGBColor[1, 0, 0]]").unwrap(),
      "RGBColor[0., 1., 1.]"
    );
    assert_eq!(
      interpret("ColorNegate[Red]").unwrap(),
      "RGBColor[0., 1., 1.]"
    );
    assert_eq!(interpret("ColorNegate[White]").unwrap(), "GrayLevel[0.]");
  }

  #[test]
  fn alpha_channel_preserved() {
    clear_state();
    assert_eq!(
      interpret("ColorNegate[RGBColor[1, 0, 0, 0.5]]").unwrap(),
      "RGBColor[0., 1., 1., 0.5]"
    );
  }
}

mod structural_numeric_equal {
  use super::*;

  // Equal compares same-head expressions component-wise, treating an exact
  // integer and the equal machine real as equal.
  #[test]
  fn color_directives_compare_numerically() {
    assert_eq!(
      interpret("RGBColor[0., 0., 1.] == RGBColor[0, 0, 1]").unwrap(),
      "True"
    );
    assert_eq!(interpret("ColorNegate[Yellow] == Blue").unwrap(), "True");
  }

  #[test]
  fn generic_heads_and_lists() {
    assert_eq!(interpret("f[1.] == f[1]").unwrap(), "True");
    assert_eq!(interpret("f[1., x] == f[1, x]").unwrap(), "True");
    assert_eq!(
      interpret("point[{0., 0.}] == point[{0, 0}]").unwrap(),
      "True"
    );
  }

  // A determinably-different component leaves the comparison unevaluated
  // (not False) for a symbolic head, matching Wolfram.
  #[test]
  fn mismatch_stays_symbolic() {
    assert_eq!(interpret("f[1.] == f[2]").unwrap(), "f[1.] == f[2]");
  }
}

mod structural_numeric_unequal {
  use super::*;

  // `!=` is the negation of structural-numeric Equal: determinably-equal
  // operands are not unequal.
  #[test]
  fn determinably_equal_operands_are_not_unequal() {
    assert_eq!(
      interpret("RGBColor[0., 0., 1.] != RGBColor[0, 0, 1]").unwrap(),
      "False"
    );
    assert_eq!(interpret("f[1.] != f[1]").unwrap(), "False");
    assert_eq!(
      interpret("point[{0., 0.}] != point[{0, 0}]").unwrap(),
      "False"
    );
    assert_eq!(interpret("Unequal[f[1.], f[1]]").unwrap(), "False");
  }

  // A determinably-different component leaves the operator comparison
  // unevaluated, matching Wolfram.
  #[test]
  fn mismatch_stays_symbolic() {
    assert_eq!(interpret("f[1.] != f[2]").unwrap(), "f[1.] != f[2]");
  }

  // SameQ stays structurally strict: 1. and 1 are not identical.
  #[test]
  fn sameq_remains_strict() {
    assert_eq!(
      interpret("RGBColor[0., 0., 1.] === RGBColor[0, 0, 1]").unwrap(),
      "False"
    );
  }
}

// CrossingDetect — zero crossings of arrays and images. All outputs
// verified against wolframscript.
mod crossing_detect {
  use super::*;

  #[test]
  fn one_dimensional_crossings() {
    assert_eq!(
      interpret("Normal[CrossingDetect[{4, 0, 1, -2, 1, -2, -3, -1, 3}]]")
        .unwrap(),
      "{0, 0, 1, 0, 1, 0, 0, 0, 1}"
    );
    // A positive element next to a negative one marks the positive side.
    assert_eq!(
      interpret("Normal[CrossingDetect[{1, -1}]]").unwrap(),
      "{1, 0}"
    );
    assert_eq!(
      interpret("Normal[CrossingDetect[{-1, 1}]]").unwrap(),
      "{0, 1}"
    );
    assert_eq!(
      interpret("Normal[CrossingDetect[{1, -1, 1, -1}]]").unwrap(),
      "{1, 0, 1, 0}"
    );
    // No crossings without a sign change; zeros neither mark nor
    // trigger.
    assert_eq!(
      interpret("Normal[CrossingDetect[{1, 2, 3}]]").unwrap(),
      "{0, 0, 0}"
    );
    assert_eq!(
      interpret("Normal[CrossingDetect[{-1, 0, 1}]]").unwrap(),
      "{0, 0, 0}"
    );
    assert_eq!(
      interpret("Normal[CrossingDetect[{0, -1}]]").unwrap(),
      "{0, 0}"
    );
  }

  #[test]
  fn delta_zeroes_small_values() {
    // |v| < delta is treated as zero — strictly, so delta 1 keeps the
    // plus-or-minus-1 values alive.
    assert_eq!(
      interpret("Normal[CrossingDetect[{4, 0, 1, -2, 1, -2, -3, -1, 3}, 1]]")
        .unwrap(),
      "{0, 0, 1, 0, 1, 0, 0, 0, 1}"
    );
    assert_eq!(
      interpret("Normal[CrossingDetect[{4, 0, 1, -2, 1, -2, -3, -1, 3}, 3/2]]")
        .unwrap(),
      "{0, 0, 0, 0, 0, 0, 0, 0, 0}"
    );
  }

  #[test]
  fn matrices_use_the_eight_neighborhood() {
    assert_eq!(
      interpret("Normal[CrossingDetect[{{1, -1}, {1, 1}}]]").unwrap(),
      "{{1, 0}, {1, 1}}"
    );
    // Diagonal neighbors count: every corner sees the negative center.
    assert_eq!(
      interpret("Normal[CrossingDetect[{{1, 1, 1}, {1, -5, 1}, {1, 1, 1}}]]")
        .unwrap(),
      "{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}}"
    );
    assert_eq!(
      interpret(
        "Normal[CrossingDetect[{{-1, -1, -1}, {-1, 5, -1}, {-1, -1, -1}}]]"
      )
      .unwrap(),
      "{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}"
    );
  }

  #[test]
  fn array_input_returns_sparse_arrays() {
    assert_eq!(
      interpret("CrossingDetect[{1, -1}]").unwrap(),
      "SparseArray[Automatic, {2}, 0, {1, {{0, 1}, {{1}}}, {1}}]"
    );
    assert_eq!(
      interpret("CrossingDetect[{{1, -1}, {-1, -1}}]").unwrap(),
      "SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 1}, {{1}}}, {1}}]"
    );
  }

  #[test]
  fn image_input_returns_a_bit_image() {
    assert_eq!(
      interpret("ImageData[CrossingDetect[Image[{{0.5, -0.5}, {-0.5, 0.5}}]]]")
        .unwrap(),
      "{{1, 0}, {0, 1}}"
    );
    assert_eq!(
      interpret("ImageType[CrossingDetect[Image[{{0.5, -0.5}, {-0.5, 0.5}}]]]")
        .unwrap(),
      "Bit"
    );
  }
}

// ImagePartition semantics decoded from wolframscript probes: plain sizes
// keep only complete top-left-anchored blocks (sizes clamped to the image),
// {n} sizes use a centered grid keeping clipped partial blocks, sizes and
// offsets are floored, offsets are clamped to >= 1.
mod image_partition {
  use super::*;

  const IMG: &str = "img = Image[Table[(10 r + c)/100., {r, 4}, {c, 5}]]; ";

  #[test]
  fn full_blocks_and_data() {
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, 2], {{2}}]"
      ))
      .unwrap(),
      "{{{2, 2}, {2, 2}}, {{2, 2}, {2, 2}}}"
    );
    // Byte images keep their type and exact pixel values, row-major.
    clear_state();
    assert_eq!(
      interpret(
        "bimg = Image[Table[Mod[10 r + c, 256], {r, 4}, {c, 6}], \"Byte\"]; \
         Map[ImageData[#, \"Byte\"] &, ImagePartition[bimg, 3], {2}]"
      )
      .unwrap(),
      "{{{{11, 12, 13}, {21, 22, 23}, {31, 32, 33}}, \
         {{14, 15, 16}, {24, 25, 26}, {34, 35, 36}}}}"
    );
    clear_state();
    assert_eq!(
      interpret(
        "bimg = Image[Table[Mod[10 r + c, 256], {r, 4}, {c, 6}], \"Byte\"]; \
         Map[ImageType, ImagePartition[bimg, 3], {2}]"
      )
      .unwrap(),
      "{{Byte, Byte}}"
    );
  }

  #[test]
  fn rectangular_sizes_and_offsets() {
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, {{3, 2}}], {{2}}]"
      ))
      .unwrap(),
      "{{{3, 2}}, {{3, 2}}}"
    );
    // Overlapping blocks via explicit offsets.
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, 2, {{1, 2}}], {{2}}]"
      ))
      .unwrap(),
      "{{{2, 2}, {2, 2}, {2, 2}, {2, 2}}, {{2, 2}, {2, 2}, {2, 2}, {2, 2}}}"
    );
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, 2, 1], {{2}}]"
      ))
      .unwrap(),
      "{{{2, 2}, {2, 2}, {2, 2}, {2, 2}}, {{2, 2}, {2, 2}, {2, 2}, {2, 2}}, \
        {{2, 2}, {2, 2}, {2, 2}, {2, 2}}}"
    );
  }

  #[test]
  fn sizes_are_floored_and_clamped() {
    // 2.7 floors to 2; block sizes larger than the image are clamped.
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, 2.7], {{2}}]"
      ))
      .unwrap(),
      "{{{2, 2}, {2, 2}}, {{2, 2}, {2, 2}}}"
    );
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, 6], {{2}}]"
      ))
      .unwrap(),
      "{{{5, 4}}}"
    );
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, {{6, 2}}], {{2}}]"
      ))
      .unwrap(),
      "{{{5, 2}}, {{5, 2}}}"
    );
    // Fractional offsets are floored with a minimum step of 1.
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, 2, 0.5], {{2}}]"
      ))
      .unwrap(),
      "{{{2, 2}, {2, 2}, {2, 2}, {2, 2}}, {{2, 2}, {2, 2}, {2, 2}, {2, 2}}, \
        {{2, 2}, {2, 2}, {2, 2}, {2, 2}}}"
    );
  }

  #[test]
  fn clipped_mode_keeps_partial_blocks() {
    // {s} centers the grid and keeps clipped partial edge blocks.
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, {{2}}], {{2}}]"
      ))
      .unwrap(),
      "{{{1, 2}, {2, 2}, {2, 2}}, {{1, 2}, {2, 2}, {2, 2}}}"
    );
    clear_state();
    assert_eq!(
      interpret(
        "img7 = Image[Table[c/10., {r, 2}, {c, 7}]]; \
         Map[ImageDimensions, ImagePartition[img7, {3}], {2}]"
      )
      .unwrap(),
      "{{{2, 2}, {3, 2}, {2, 2}}}"
    );
    // Odd overhang goes to the leading (top/left) edge.
    clear_state();
    assert_eq!(
      interpret(
        "img7 = Image[Table[c/10., {r, 2}, {c, 7}]]; \
         Map[ImageDimensions, ImagePartition[img7, {5}], {2}]"
      )
      .unwrap(),
      "{{{3, 2}, {4, 2}}}"
    );
    // Per-axis mixing of full and clipped modes.
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, {{{{2}}, 2}}], {{2}}]"
      ))
      .unwrap(),
      "{{{1, 2}, {2, 2}, {2, 2}}, {{1, 2}, {2, 2}, {2, 2}}}"
    );
    // Clipped mode with an explicit offset keeps every grid block whose
    // center falls within the image, including duplicates.
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, {{2}}, 3], {{2}}]"
      ))
      .unwrap(),
      "{{{2, 1}, {2, 1}}, {{2, 2}, {2, 2}}}"
    );
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}Map[ImageDimensions, ImagePartition[img, {{10}}, 2], {{2}}]"
      ))
      .unwrap(),
      "{{{5, 4}, {5, 4}, {5, 4}}, {{5, 4}, {5, 4}, {5, 4}}}"
    );
  }

  #[test]
  fn invalid_arguments_emit_messages() {
    clear_state();
    let r = interpret_with_stdout("ImagePartition[Image[{{0.5}}], 0]").unwrap();
    assert_eq!(r.result, "ImagePartition[-Image-, 0]");
    assert!(r.warnings[0].contains(
      "ImagePartition::arg2: 0 is not a valid size specification for image partitions."
    ));

    clear_state();
    let r = interpret_with_stdout("ImagePartition[Image[{{0.5}}], {2, All}]")
      .unwrap();
    assert!(r.warnings[0].contains(
      "ImagePartition::arg2: {2, All} is not a valid size specification"
    ));

    // Invalid scalar offsets are shown normalized to a pair.
    clear_state();
    let r =
      interpret_with_stdout("ImagePartition[Image[{{0.5}}], 2, -1]").unwrap();
    assert!(r.warnings[0].contains(
      "ImagePartition::arg3: {-1, -1} is not a positive number or a pair of positive numbers."
    ));

    clear_state();
    let r =
      interpret_with_stdout("ImagePartition[Image[{{0.5}}], 2, {1.5, x}]")
        .unwrap();
    assert!(r.warnings[0].contains(
      "ImagePartition::arg3: {1.5, x} is not a positive number or a pair of positive numbers."
    ));

    clear_state();
    let r = interpret_with_stdout("ImagePartition[5, 0]").unwrap();
    assert_eq!(r.result, "ImagePartition[5, 0]");
    assert!(r.warnings[0].contains(
      "ImagePartition::imginv: Expecting an image or graphics instead of 5."
    ));

    clear_state();
    let r = interpret_with_stdout("ImagePartition[Image[{{0.5}}]]").unwrap();
    assert_eq!(r.result, "ImagePartition[-Image-]");
    assert!(r.warnings[0].contains(
      "ImagePartition::argtu: ImagePartition called with 1 argument; 2 or 3 arguments are expected."
    ));
  }
}

// ColorCombine semantics decoded from wolframscript probes: channels of the
// inputs interleave (multichannel inputs concatenate), the optional
// colorspace argument only tags the result and must match the channel
// count, and the result type is the highest input type.
mod color_combine {
  use super::*;

  #[test]
  fn combines_grayscale_channels() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ColorCombine[{Image[{{0.1, 0.2}, {0.3, 0.4}}], \
           Image[{{0.5, 0.6}, {0.7, 0.8}}], Image[{{0.9, 1.0}, {0.15, 0.25}}]}]]"
      )
      .unwrap(),
      "{{{0.10000000149011612, 0.5, 0.8999999761581421}, \
         {0.20000000298023224, 0.6000000238418579, 1.}}, \
        {{0.30000001192092896, 0.699999988079071, 0.15000000596046448}, \
         {0.4000000059604645, 0.800000011920929, 0.25}}}"
    );
    // A single image gives a single-channel (grayscale) result; two give
    // a plain 2-channel image.
    clear_state();
    assert_eq!(
      interpret(
        "{ImageChannels[ColorCombine[{Image[{{0.1}}]}]], \
          ImageChannels[ColorCombine[{Image[{{0.1}}], Image[{{0.2}}]}]], \
          ImageColorSpace[ColorCombine[{Image[{{0.1}}], Image[{{0.2}}]}]]}"
      )
      .unwrap(),
      "{1, 2, Automatic}"
    );
    // Multichannel inputs concatenate their channels.
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ColorCombine[{Image[{{{0.1, 0.2, 0.3}}}], Image[{{0.4}}]}]]"
      )
      .unwrap(),
      "{{{0.10000000149011612, 0.20000000298023224, 0.30000001192092896, \
          0.4000000059604645}}}"
    );
  }

  #[test]
  fn colorspace_argument_tags_the_result() {
    clear_state();
    assert_eq!(
      interpret(
        "g = {Image[{{0.1}}], Image[{{0.2}}], Image[{{0.3}}]}; \
         {ImageColorSpace[ColorCombine[g, \"HSB\"]], \
          ImageColorSpace[ColorCombine[g, \"RGB\"]], \
          ImageColorSpace[ColorCombine[{Image[{{0.1}}]}, \"Grayscale\"]]}"
      )
      .unwrap(),
      "{HSB, RGB, Grayscale}"
    );
    // The data itself is not converted by the tag.
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[ColorCombine[{Image[{{0.1}}], Image[{{0.2}}], \
           Image[{{0.3}}]}, \"HSB\"]]"
      )
      .unwrap(),
      "{{{0.10000000149011612, 0.20000000298023224, 0.30000001192092896}}}"
    );
    // The tag survives assignment, ImageTake, and ImagePartition, and
    // ColorSeparate results are untagged again.
    clear_state();
    assert_eq!(
      interpret(
        "h = ColorCombine[{Image[{{0.1, 0.2}}], Image[{{0.3, 0.4}}], \
           Image[{{0.5, 0.6}}]}, \"HSB\"]; \
         {ImageColorSpace[h], ImageColorSpace[ImageTake[h, 1]], \
          ImageColorSpace[ImagePartition[h, 1][[1, 1]]], \
          Map[ImageColorSpace, ColorSeparate[h]]}"
      )
      .unwrap(),
      "{HSB, HSB, HSB, {Automatic, Automatic, Automatic}}"
    );
  }

  #[test]
  fn type_promotion() {
    // Byte + Byte stays Byte with exact values.
    clear_state();
    assert_eq!(
      interpret(
        "c = ColorCombine[{Image[{{10, 20}, {30, 40}}, \"Byte\"], \
           Image[{{50, 60}, {70, 80}}, \"Byte\"]}]; \
         {ImageType[c], ImageData[c, \"Byte\"]}"
      )
      .unwrap(),
      "{Byte, {{{10, 50}, {20, 60}}, {{30, 70}, {40, 80}}}}"
    );
    // Real32 channels are snapped to single precision inside a Real64
    // result; Byte channels convert exactly.
    clear_state();
    assert_eq!(
      interpret(
        "c = ColorCombine[{Image[{{0.1}}], Image[{{0.1}}, \"Real64\"]}]; \
         {ImageType[c], ImageData[c]}"
      )
      .unwrap(),
      "{Real64, {{{0.10000000149011612, 0.1}}}}"
    );
    clear_state();
    assert_eq!(
      interpret(
        "c = ColorCombine[{Image[{{10}}, \"Byte\"], \
           Image[{{0.1}}, \"Real64\"]}]; \
         {ImageType[c], ImageData[c]}"
      )
      .unwrap(),
      "{Real64, {{{0.0392156862745098, 0.1}}}}"
    );
    clear_state();
    assert_eq!(
      interpret(
        "c = ColorCombine[{Image[{{0, 1}}, \"Bit\"], \
           Image[{{50, 60}}, \"Byte\"]}]; \
         {ImageType[c], ImageData[c]}"
      )
      .unwrap(),
      "{Byte, {{{0., 0.19607843137254902}, {1., 0.23529411764705882}}}}"
    );
  }

  #[test]
  fn invalid_arguments_emit_messages() {
    // Mismatched dimensions, non-image entries, empty or non-list input.
    clear_state();
    let r = interpret_with_stdout(
      "ColorCombine[{Image[{{0.1}}], Image[{{0.1, 0.2}}]}]",
    )
    .unwrap();
    assert_eq!(r.result, "ColorCombine[{-Image-, -Image-}]");
    assert!(r.warnings[0].contains(
      "ColorCombine::ccbinput: {-Image-, -Image-} should be a list of images \
       with the same image dimensions."
    ));

    clear_state();
    let r = interpret_with_stdout("ColorCombine[{Image[{{0.1}}], 5}]").unwrap();
    assert!(r.warnings[0].contains(
      "ColorCombine::ccbinput: {-Image-, 5} should be a list of images"
    ));

    clear_state();
    let r = interpret_with_stdout("ColorCombine[5]").unwrap();
    assert!(
      r.warnings[0]
        .contains("ColorCombine::ccbinput: 5 should be a list of images")
    );

    // Invalid colorspace fires before the list check; strings render bare.
    clear_state();
    let r = interpret_with_stdout("ColorCombine[5, \"Foo\"]").unwrap();
    assert_eq!(r.result, "ColorCombine[5, Foo]");
    assert!(r.warnings[0].contains(
      "ColorCombine::imgcstype: Foo is an invalid color space specification."
    ));

    // A symbol is not a valid colorspace either.
    clear_state();
    let r = interpret_with_stdout(
      "ColorCombine[{Image[{{0.1}}], Image[{{0.2}}], Image[{{0.3}}]}, RGB]",
    )
    .unwrap();
    assert!(r.warnings[0].contains(
      "ColorCombine::imgcstype: RGB is an invalid color space specification."
    ));

    // Channel-count mismatch with the requested colorspace.
    clear_state();
    let r = interpret_with_stdout(
      "ColorCombine[{Image[{{0.1}}], Image[{{0.2}}]}, \"RGB\"]",
    )
    .unwrap();
    assert!(r.warnings[0].contains(
      "ColorCombine::imgcsmis: The specified color space RGB and the number \
       of channels 2 are not compatible."
    ));

    clear_state();
    let r = interpret_with_stdout("ColorCombine[{Image[{{0.1}}]}, \"RGB\", 3]")
      .unwrap();
    assert!(r.warnings[0].contains(
      "ColorCombine::argt: ColorCombine called with 3 arguments; 1 or 2 \
       arguments are expected."
    ));
  }
}

// DistanceTransform semantics decoded from wolframscript probes: exact
// Euclidean distance to the nearest background pixel (borders are not
// background), foreground = f32-snapped luminance strictly above t
// (default 0, f64), Real32 single-channel result, and the all-foreground
// quirk returning 1 everywhere. Exact non-machine thresholds trigger
// image-dependent garbage in wolframscript (WS-internal UB) and get sane
// numeric semantics instead.
mod distance_transform {
  use super::*;

  #[test]
  fn euclidean_distances() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[DistanceTransform[Image[{{0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, \
         {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 0, 0, 0, 0}}]]]"
      )
      .unwrap(),
      "{{0., 0., 0., 0., 0.}, {0., 1., 1., 1., 0.}, {0., 1., 2., 1., 0.}, \
        {0., 1., 1., 1., 0.}, {0., 0., 0., 0., 0.}}"
    );
    // Diagonal neighbors give sqrt(2) (f32-rounded); the image border
    // does not count as background.
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[DistanceTransform[Image[{{0, 0, 0, 0}, {0, 1, 1, 0}, \
         {0, 1, 1, 0}, {1, 1, 1, 1}}]]]"
      )
      .unwrap(),
      "{{0., 0., 0., 0.}, {0., 1., 1., 0.}, {0., 1., 1., 0.}, \
        {1., 1.4142135381698608, 1.4142135381698608, 1.}}"
    );
    // Distances are not clipped.
    clear_state();
    assert_eq!(
      interpret("ImageData[DistanceTransform[Image[{{0, 1, 1, 1, 1, 1, 1}}]]]")
        .unwrap(),
      "{{0., 1., 2., 3., 4., 5., 6.}}"
    );
    clear_state();
    assert_eq!(
      interpret(
        "d = DistanceTransform[Image[{{0, 1}}]]; \
         {ImageType[d], ImageChannels[d], ImageColorSpace[d]}"
      )
      .unwrap(),
      "{Real32, 1, Automatic}"
    );
  }

  #[test]
  fn all_foreground_gives_ones() {
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[DistanceTransform[Image[ConstantArray[1, {5, 5}]]]]"
      )
      .unwrap(),
      "{{1., 1., 1., 1., 1.}, {1., 1., 1., 1., 1.}, {1., 1., 1., 1., 1.}, \
        {1., 1., 1., 1., 1.}, {1., 1., 1., 1., 1.}}"
    );
    // A negative threshold makes even 0-pixels foreground.
    clear_state();
    assert_eq!(
      interpret("ImageData[DistanceTransform[Image[{{0, 1, 1}}], -1]]")
        .unwrap(),
      "{{1., 1., 1.}}"
    );
  }

  #[test]
  fn thresholds() {
    // Strictly above t, on normalized values.
    clear_state();
    assert_eq!(
      interpret("ImageData[DistanceTransform[Image[{{0.2, 0.5, 0.9}}], 0.5]]")
        .unwrap(),
      "{{0., 0., 1.}}"
    );
    // Byte pixels compare on the normalized scale.
    clear_state();
    assert_eq!(
      interpret(
        "b = Image[{{0, 100, 200}}, \"Byte\"]; \
         {ImageData[DistanceTransform[b]], \
          ImageData[DistanceTransform[b, 150]], \
          ImageData[DistanceTransform[b, 0.5]]}"
      )
      .unwrap(),
      "{{{0., 1., 2.}}, {{0., 0., 0.}}, {{0., 0., 1.}}}"
    );
    // Pixel values are f32-snapped before the compare, so a stored 0.3
    // is strictly above the threshold 0.3.
    clear_state();
    assert_eq!(
      interpret("ImageData[DistanceTransform[Image[{{0.3, 0.1}}], 0.3]]")
        .unwrap(),
      "{{1., 0.}}"
    );
    clear_state();
    assert_eq!(
      interpret("ImageData[DistanceTransform[Image[{{0.5, 0.25}}], 0.5]]")
        .unwrap(),
      "{{0., 0.}}"
    );
  }

  #[test]
  fn rgb_uses_luminance() {
    // 0.299 R + 0.587 G + 0.114 B: pure red is background at t = 0.3
    // but foreground at t = 0.29.
    clear_state();
    assert_eq!(
      interpret(
        "{ImageData[DistanceTransform[Image[{{{1, 0, 0}, {0, 0, 0}}}], 0.29]], \
          ImageData[DistanceTransform[Image[{{{1, 0, 0}, {0, 0, 0}}}], 0.3]]}"
      )
      .unwrap(),
      "{{{1., 0.}}, {{0., 0.}}}"
    );
  }

  #[test]
  fn invalid_arguments_emit_messages() {
    clear_state();
    let r = interpret_with_stdout("DistanceTransform[5]").unwrap();
    assert_eq!(r.result, "DistanceTransform[5]");
    assert!(r.warnings[0].contains(
      "DistanceTransform::imginv: Expecting an image or graphics instead of 5."
    ));

    clear_state();
    let r =
      interpret_with_stdout("DistanceTransform[Image[{{1}}], x]").unwrap();
    assert_eq!(r.result, "DistanceTransform[-Image-, x]");
    assert!(r.warnings[0].contains(
      "DistanceTransform::rthres: The specified threshold value x should \
       represent a real number."
    ));
  }
}

// FillingTransform semantics decoded from wolframscript probes: the plain
// form floods from the border with 4-connectivity; the depth form is a
// reconstruction with interior marker I+h and 8-connectivity; the marker
// form interpolates per pixel between the image and its plain fill by the
// largest marker value in each 4-connected basin (f32 lerp F*m + I*(1-m)).
mod filling_transform {
  use super::*;

  #[test]
  fn binary_hole_filling() {
    // Enclosed zeros fill; zeros touching the border stay.
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[FillingTransform[Image[{{1, 1, 1, 1, 1}, {1, 0, 0, 1, 0}, \
         {1, 0, 0, 1, 1}, {1, 1, 1, 1, 1}}]]]"
      )
      .unwrap(),
      "{{1., 1., 1., 1., 1.}, {1., 1., 1., 1., 0.}, {1., 1., 1., 1., 1.}, \
        {1., 1., 1., 1., 1.}}"
    );
    // Diagonal contact does not connect background to the border
    // (4-connectivity), so this hole still fills.
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[FillingTransform[Image[{{1, 1, 1}, {1, 0, 1}, {1, 1, 0}}]]]"
      )
      .unwrap(),
      "{{1., 1., 1.}, {1., 1., 1.}, {1., 1., 0.}}"
    );
  }

  #[test]
  fn grayscale_fill_and_types() {
    // Basins rise to their 4-connected pour-out level.
    clear_state();
    assert_eq!(
      interpret(
        "ImageData[FillingTransform[Image[{{0.5, 0.6, 0.7, 0.8}, \
         {0.6, 0.2, 0.3, 0.7}, {0.7, 0.4, 0.2, 0.6}, {0.8, 0.7, 0.6, 0.5}}]]]"
      )
      .unwrap(),
      "{{0.5, 0.6000000238418579, 0.699999988079071, 0.800000011920929}, \
        {0.6000000238418579, 0.6000000238418579, 0.6000000238418579, 0.699999988079071}, \
        {0.699999988079071, 0.6000000238418579, 0.6000000238418579, 0.6000000238418579}, \
        {0.800000011920929, 0.699999988079071, 0.6000000238418579, 0.5}}"
    );
    // Per-channel independent filling for multichannel images.
    clear_state();
    assert_eq!(
      interpret(
        "rgb = Image[Table[If[r == 2 && c == 2, {0.2, 0.1, 0.4}, \
         {0.9, 0.5, 0.9}], {r, 3}, {c, 3}]]; \
         ImageData[FillingTransform[rgb]][[2, 2]]"
      )
      .unwrap(),
      "{0.8999999761581421, 0.5, 0.8999999761581421}"
    );
    // The image type is preserved (with quantization for Byte).
    clear_state();
    assert_eq!(
      interpret(
        "b = FillingTransform[Image[{{200, 200, 200}, {200, 50, 200}, \
         {200, 200, 200}}, \"Byte\"]]; {ImageType[b], ImageData[b, \"Byte\"]}"
      )
      .unwrap(),
      "{Byte, {{200, 200, 200}, {200, 200, 200}, {200, 200, 200}}}"
    );
  }

  #[test]
  fn depth_form() {
    // Fill level is basin minimum + h, capped by the 8-connected
    // pour-out level (the corner pixel 0.5 caps this basin).
    clear_state();
    assert_eq!(
      interpret(
        "gray2 = Image[{{0.5, 0.6, 0.7, 0.8}, {0.6, 0.2, 0.3, 0.7}, \
         {0.7, 0.4, 0.2, 0.6}, {0.8, 0.7, 0.6, 0.5}}]; \
         {ImageData[FillingTransform[gray2, 0.2]][[2]], \
          ImageData[FillingTransform[gray2, 1.]][[2]]}"
      )
      .unwrap(),
      "{{0.6000000238418579, 0.4000000059604645, 0.4000000059604645, \
         0.699999988079071}, \
        {0.6000000238418579, 0.5, 0.5, 0.699999988079071}}"
    );
    // A diagonal border pixel caps via 8-connectivity in the depth form
    // even though the plain form (4-connectivity) fills fully.
    clear_state();
    assert_eq!(
      interpret(
        "i = Image[{{0.9, 0.9, 0.9}, {0.9, 0.4, 0.9}, {0.7, 0.9, 0.9}}]; \
         {ImageData[FillingTransform[i, 2.]][[2, 2]], \
          ImageData[FillingTransform[i]][[2, 2]]}"
      )
      .unwrap(),
      "{0.699999988079071, 0.8999999761581421}"
    );
    // Bit input becomes Real32; Byte results stay quantized; h = 0 is a
    // valid no-op.
    clear_state();
    assert_eq!(
      interpret(
        "bit = Image[{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}}, \"Bit\"]; \
         d = FillingTransform[bit, 0.4]; \
         by = FillingTransform[Image[{{200, 200, 200}, {200, 50, 200}, \
          {200, 200, 200}}, \"Byte\"], 0.2]; \
         {ImageType[d], ImageData[d][[2, 2]], ImageType[by], \
          ImageData[by][[2, 2]], \
          ImageData[FillingTransform[Image[{{0.9, 0.9, 0.9}, {0.9, 0.2, 0.9}, \
           {0.9, 0.9, 0.9}}], 0]][[2, 2]]}"
      )
      .unwrap(),
      "{Real32, 0.4000000059604645, Byte, 0.396078431372549, \
        0.20000000298023224}"
    );
  }

  #[test]
  fn marker_form() {
    // Only basins containing a nonzero marker pixel fill; a fractional
    // marker interpolates between image and fill (f32 lerp).
    clear_state();
    assert_eq!(
      interpret(
        "g = Image[{{0.9, 0.9, 0.9}, {0.9, 0.2, 0.9}, {0.9, 0.9, 0.9}}]; \
         Table[ImageData[FillingTransform[g, Image[{{0, 0, 0}, {0, mv, 0}, \
          {0, 0, 0}}]]][[2, 2]], {mv, {0.1, 0.5, 0.9, 1., 2.}}]"
      )
      .unwrap(),
      "{0.26999998092651367, 0.550000011920929, 0.8299999237060547, \
        0.8999999761581421, 0.8999999761581421}"
    );
    // The largest marker value in the basin applies per pixel:
    // I + (F - I) scales each pixel individually.
    clear_state();
    assert_eq!(
      interpret(
        "i5 = Image[{{0.9, 0.9, 0.9, 0.9, 0.9}, {0.9, 0.2, 0.3, 0.2, 0.9}, \
         {0.9, 0.9, 0.9, 0.9, 0.9}}]; \
         ImageData[FillingTransform[i5, Image[{{0, 0, 0, 0, 0}, \
          {0, 0.3, 0, 0.8, 0}, {0, 0, 0, 0, 0}}]]][[2]]"
      )
      .unwrap(),
      "{0.8999999761581421, 0.7599999904632568, 0.7799999713897705, \
        0.7599999904632568, 0.8999999761581421}"
    );
    // Result type: Real64 stays Real64, everything else becomes Real32.
    clear_state();
    assert_eq!(
      interpret(
        "m3 = Image[{{0, 0, 0}, {0, 0.3, 0}, {0, 0, 0}}]; \
         by = FillingTransform[Image[{{230, 230, 230}, {230, 50, 230}, \
          {230, 230, 230}}, \"Byte\"], m3]; \
         r = FillingTransform[Image[{{0.9, 0.9, 0.9}, {0.9, 0.2, 0.9}, \
          {0.9, 0.9, 0.9}}, \"Real64\"], m3]; \
         {ImageType[by], ImageData[by][[2, 2]], ImageType[r], \
          ImageData[r][[2, 2]]}"
      )
      .unwrap(),
      "{Real32, 0.4078431725502014, Real64, 0.4100000083446503}"
    );
    // A marker with mismatched dimensions marks nothing; a multichannel
    // marker echoes unevaluated.
    clear_state();
    assert_eq!(
      interpret(
        "g = Image[{{0.9, 0.9, 0.9}, {0.9, 0.2, 0.9}, {0.9, 0.9, 0.9}}]; \
         {ImageData[FillingTransform[g, Image[{{1, 0}}]]][[2, 2]], \
          FillingTransform[g, Image[{{{1, 0, 0}}}]]}"
      )
      .unwrap(),
      "{0.20000000298023224, FillingTransform[-Image-, -Image-]}"
    );
  }

  #[test]
  fn invalid_arguments_emit_messages() {
    clear_state();
    let r = interpret_with_stdout("FillingTransform[5]").unwrap();
    assert_eq!(r.result, "FillingTransform[5]");
    assert!(r.warnings[0].contains(
      "FillingTransform::imginv: Expecting an image or graphics instead of 5."
    ));

    clear_state();
    let r = interpret_with_stdout("FillingTransform[Image[{{1}}], x]").unwrap();
    assert!(r.warnings[0].contains(
      "FillingTransform::arg2: Expecting either a marker or depth \
       specification as the second argument instead of x."
    ));

    clear_state();
    let r =
      interpret_with_stdout("FillingTransform[Image[{{1}}], -0.5]").unwrap();
    assert!(r.warnings[0].contains(
      "FillingTransform::invh: The height specification -0.5 must be positive."
    ));
  }
}

// Image constructor conformance decoded from wolframscript probes: any
// channel count is accepted (single-element pixels collapse to
// grayscale), integer types read raw values on their own scale with
// half-even rounding and clamping, Image[image, type] re-quantizes the
// normalized data, and malformed arrays emit imgarray with an
// unevaluated echo.
mod image_constructor_conformance {
  use super::*;

  #[test]
  fn arbitrary_channel_counts() {
    clear_state();
    assert_eq!(
      interpret(
        "i2 = Image[{{{0.9, 0.5}, {0.2, 0.1}}, {{0.3, 0.7}, {0.4, 0.6}}}]; \
         {ImageChannels[i2], ImageColorSpace[i2], ImageData[i2][[1, 1]]}"
      )
      .unwrap(),
      "{2, Automatic, {0.8999999761581421, 0.5}}"
    );
    // Single-element pixels collapse to a plain grayscale image.
    clear_state();
    assert_eq!(
      interpret(
        "i1 = Image[{{{0.9}, {0.2}}, {{0.3}, {0.4}}}]; \
         {ImageChannels[i1], ImageData[i1]}"
      )
      .unwrap(),
      "{1, {{0.8999999761581421, 0.20000000298023224}, \
        {0.30000001192092896, 0.4000000059604645}}}"
    );
    clear_state();
    assert_eq!(
      interpret("ImageChannels[Image[{{{0.1, 0.2, 0.3, 0.4, 0.5}}}]]").unwrap(),
      "5"
    );
    // ColorSeparate splits 2-channel images.
    clear_state();
    assert_eq!(
      interpret(
        "Map[ImageData, ColorSeparate[Image[{{{0.9, 0.5}, {0.2, 0.1}}}]]]"
      )
      .unwrap(),
      "{{{0.8999999761581421, 0.20000000298023224}}, \
        {{0.5, 0.10000000149011612}}}"
    );
  }

  #[test]
  fn integer_types_quantize_raw_values() {
    // Values are read on the type's own scale: reals round half-even
    // and clamp into range.
    clear_state();
    assert_eq!(
      interpret(
        "{ImageData[Image[{{0.5, 0.998}}, \"Byte\"], \"Byte\"], \
          ImageData[Image[{{2.5, 3.5, -0.5}}, \"Byte\"], \"Byte\"], \
          ImageData[Image[{{300, -5}}, \"Byte\"], \"Byte\"], \
          ImageData[Image[{{0.3, 0.8}}, \"Bit\"]], \
          ImageData[Image[{{2, -1}}, \"Bit\"]], \
          ImageData[Image[{{0.5, 70000}}, \"Bit16\"], \"Bit16\"]}"
      )
      .unwrap(),
      "{{{0, 1}}, {{2, 4, 0}}, {{255, 0}}, {{0, 1}}, {{1, 0}}, {{0, 65535}}}"
    );
    // Untyped real input stores raw values without clamping.
    clear_state();
    assert_eq!(
      interpret("ImageData[Image[{{0.5, 2.5, -0.5}}]]").unwrap(),
      "{{0.5, 2.5, -0.5}}"
    );
    // Image[image, type] re-quantizes the NORMALIZED data instead.
    clear_state();
    assert_eq!(
      interpret("ImageData[Image[Image[{{0.5}}], \"Byte\"], \"Byte\"]")
        .unwrap(),
      "{{128}}"
    );
  }

  #[test]
  fn malformed_arrays_emit_imgarray() {
    clear_state();
    let r = interpret_with_stdout("Image[{{x}}]").unwrap();
    assert_eq!(r.result, "Image[{{x}}]");
    assert!(r.warnings[0].contains(
      "Image::imgarray: The specified argument {{x}} should be an array of \
       rank 2 or 3 with machine-sized numbers."
    ));

    clear_state();
    let r = interpret_with_stdout("Image[{{{0.1, 0.2}, {0.3}}}]").unwrap();
    assert!(r.warnings[0].contains("Image::imgarray"));

    clear_state();
    let r = interpret_with_stdout("Image[{}]").unwrap();
    assert!(r.warnings[0].contains(
      "Image::imgarray: The specified argument {} should be an array"
    ));

    clear_state();
    let r = interpret_with_stdout("Image[{{0.5}}, \"Foo\"]").unwrap();
    assert_eq!(r.result, "Image[{{0.5}}, Foo]");
    assert!(r.warnings[0].contains(
      "Image::imgdtype: The specified data type Foo should be \"Bit\", \
       \"Byte\", \"Bit16\", \"Real32\" or \"Real64\"."
    ));
  }
}

// ImageValue semantics decoded from wolframscript probes: bilinear
// tensor-product sampling in the image coordinate system (x from the
// left, y up from the bottom, pixel centers at half-integers) with zero
// padding outside; Real32 images snap pixels to f32 and round the
// result to f32, all other types compute in plain f64.
mod image_value {
  use super::*;

  const IMG: &str = "img = Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}]; ";

  #[test]
  fn bilinear_sampling() {
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}{{ImageValue[img, {{1, 1}}], ImageValue[img, {{1.5, 1.5}}], \
          ImageValue[img, {{1.2, 1.8}}], \
          ImageValue[img, {{1.2345678, 1.7654321}}]}}"
      ))
      .unwrap(),
      "{0.30000001192092896, 0.20000000298023224, 0.11900000274181366, \
        0.12741579115390778}"
    );
    // Positions outside blend with zero padding; fully outside gives 0.
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}{{ImageValue[img, {{3, 2}}], ImageValue[img, {{0, 0}}], \
          ImageValue[img, {{5, 5}}]}}"
      ))
      .unwrap(),
      "{0.07500000298023224, 0.10000000149011612, 0.}"
    );
    // A list of positions gives a list of values; multichannel images
    // give channel-value lists.
    clear_state();
    assert_eq!(
      interpret(&format!(
        "{IMG}ImageValue[img, {{{{1, 1}}, {{3, 2}}, {{1.5, 1.5}}}}]"
      ))
      .unwrap(),
      "{0.30000001192092896, 0.07500000298023224, 0.20000000298023224}"
    );
    clear_state();
    assert_eq!(
      interpret(
        "ImageValue[Image[{{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}}], {2, 1}]"
      )
      .unwrap(),
      "{0.10000000149011612, 0.125, 0.15000000596046448}"
    );
  }

  #[test]
  fn non_real32_types_compute_in_f64() {
    clear_state();
    assert_eq!(
      interpret("ImageValue[Image[{{10, 200}}, \"Byte\"], {1, 0.5}]").unwrap(),
      "0.4117647058823529"
    );
    clear_state();
    assert_eq!(
      interpret(
        "ImageValue[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}, \"Real64\"], \
         {1.2, 1.8}]"
      )
      .unwrap(),
      "0.119"
    );
  }

  #[test]
  fn invalid_arguments_emit_messages() {
    clear_state();
    let r =
      interpret_with_stdout("ImageValue[Image[{{0.5}}], {x, 1}]").unwrap();
    assert_eq!(r.result, "ImageValue[-Image-, {x, 1}]");
    assert!(r.warnings[0].contains(
      "ImageValue::imgrng: The specified argument {x, 1} should be an \
       image, a graphics object or a list of coordinates."
    ));

    clear_state();
    let r = interpret_with_stdout("ImageValue[5, {1, 1}]").unwrap();
    assert!(r.warnings[0].contains(
      "ImageValue::imginv: Expecting an image or graphics instead of 5."
    ));

    clear_state();
    let r = interpret_with_stdout("ImageValue[Image[{{0.5}}]]").unwrap();
    assert!(r.warnings[0].contains(
      "ImageValue::argtu: ImageValue called with 1 argument; 2 or 3 \
       arguments are expected."
    ));
  }
}
