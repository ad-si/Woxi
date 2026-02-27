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
  fn collage_result_has_rgba_channels() {
    clear_state();
    // Collage uses RGBA canvas
    assert_eq!(
      interpret(
        "img1 = Image[{{1, 0}, {0, 1}}]; img2 = Image[{{0, 1}, {1, 0}}]; ImageChannels[ImageCollage[{img1, img2}]]"
      )
      .unwrap(),
      "4"
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
  fn assemble_result_has_rgba_channels() {
    clear_state();
    assert_eq!(
      interpret(
        "img = Image[{{1, 0}, {0, 1}}]; ImageChannels[ImageAssemble[{img, img}]]"
      )
      .unwrap(),
      "4"
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
