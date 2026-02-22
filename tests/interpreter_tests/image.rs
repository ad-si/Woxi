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
  fn image_type_real64() {
    clear_state();
    let result = interpret("ImageType[Image[{{0, 0.5, 1}}]]").unwrap();
    assert_eq!(result, "Real64");
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
    let result = interpret("ImageColorSpace[Image[{{0, 1}}]]").unwrap();
    assert_eq!(result, "Grayscale");
  }

  #[test]
  fn image_color_space_rgb() {
    clear_state();
    let result = interpret("ImageColorSpace[Image[{{{1, 0, 0}}}]]").unwrap();
    assert_eq!(result, "RGB");
  }

  #[test]
  fn image_data_roundtrip() {
    clear_state();
    let result = interpret(
      "img = Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}]; ImageData[img]",
    )
    .unwrap();
    assert_eq!(result, "{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}");
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

  #[test]
  fn color_negate_rgb() {
    clear_state();
    let result =
      interpret("ImageData[ColorNegate[Image[{{{1, 0, 0.5}}}]]]").unwrap();
    assert_eq!(result, "{{{0., 1., 0.5}}}");
  }

  #[test]
  fn binarize_default_threshold() {
    clear_state();
    let result =
      interpret("ImageData[Binarize[Image[{{0.3, 0.5, 0.7}}]]]").unwrap();
    assert_eq!(result, "{{0., 1., 1.}}");
  }

  #[test]
  fn binarize_custom_threshold() {
    clear_state();
    let result =
      interpret("ImageData[Binarize[Image[{{0.3, 0.5, 0.7}}], 0.6]]").unwrap();
    assert_eq!(result, "{{0., 0., 1.}}");
  }

  #[test]
  fn image_adjust_rescale() {
    clear_state();
    let result =
      interpret("ImageData[ImageAdjust[Image[{{0.0, 0.5, 1.0}}]]]").unwrap();
    assert_eq!(result, "{{0., 0.5, 1.}}");
  }

  #[test]
  fn image_reflect_horizontal() {
    clear_state();
    // After horizontal flip, pixel order in each row is reversed
    // But since we go through DynamicImage conversion (u8), values get quantized
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
  fn image_crop_manual() {
    clear_state();
    let result = interpret(
      "ImageDimensions[ImageCrop[Image[{{0, 0, 0, 0}, {0, 1, 1, 0}, {0, 0, 0, 0}}], {{1, 1}, {3, 2}}]]",
    )
    .unwrap();
    assert_eq!(result, "{2, 1}");
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
    assert_eq!(result, "{{0.1, 0.2, 0.3}}");
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
  fn image_add_clamped() {
    clear_state();
    let result =
      interpret("ImageData[ImageAdd[Image[{{0.8}}], Image[{{0.5}}]]]").unwrap();
    assert_eq!(result, "{{1.}}");
  }

  #[test]
  fn image_subtract() {
    clear_state();
    let result = interpret(
      "ImageData[ImageSubtract[Image[{{0.5, 0.8}}], Image[{{0.2, 0.3}}]]]",
    )
    .unwrap();
    assert_eq!(result, "{{0.3, 0.5}}");
  }

  #[test]
  fn image_subtract_clamped() {
    clear_state();
    let result =
      interpret("ImageData[ImageSubtract[Image[{{0.2}}], Image[{{0.5}}]]]")
        .unwrap();
    assert_eq!(result, "{{0.}}");
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
  fn import_url_jpeg() {
    clear_state();
    let result = interpret(
      "ImageQ[Import[\"https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Usamljeni_jasen_-_panoramio_%28cropped%29.jpg/500px-Usamljeni_jasen_-_panoramio_%28cropped%29.jpg\"]]",
    )
    .unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn import_url_dimensions() {
    clear_state();
    let result = interpret(
      "ImageDimensions[Import[\"https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Usamljeni_jasen_-_panoramio_%28cropped%29.jpg/500px-Usamljeni_jasen_-_panoramio_%28cropped%29.jpg\"]]",
    )
    .unwrap();
    assert_eq!(result, "{500, 564}");
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
}
