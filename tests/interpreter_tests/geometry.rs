use super::*;

mod area {
  use super::*;

  #[test]
  fn unit_disk() {
    assert_eq!(interpret("Area[Disk[]]").unwrap(), "Pi");
  }

  #[test]
  fn disk_with_radius() {
    assert_eq!(interpret("Area[Disk[{0, 0}, 5]]").unwrap(), "25*Pi");
  }

  #[test]
  fn elliptical_disk() {
    assert_eq!(interpret("Area[Disk[{0, 0}, {3, 2}]]").unwrap(), "6*Pi");
  }

  #[test]
  fn unit_rectangle() {
    assert_eq!(interpret("Area[Rectangle[]]").unwrap(), "1");
  }

  #[test]
  fn rectangle_with_bounds() {
    assert_eq!(interpret("Area[Rectangle[{0, 0}, {3, 4}]]").unwrap(), "12");
  }

  #[test]
  fn triangle() {
    assert_eq!(
      interpret("Area[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn polygon() {
    assert_eq!(
      interpret("Area[Polygon[{{0, 0}, {4, 0}, {4, 3}, {0, 3}}]]").unwrap(),
      "12"
    );
  }

  #[test]
  fn circle_undefined() {
    assert_eq!(interpret("Area[Circle[]]").unwrap(), "Undefined");
  }

  #[test]
  fn symbolic_radius() {
    assert_eq!(interpret("Area[Disk[{0, 0}, r]]").unwrap(), "Pi*r^2");
  }

  #[test]
  fn default_triangle() {
    assert_eq!(interpret("Area[Triangle[]]").unwrap(), "1/2");
  }
}

mod arc_length {
  use super::*;

  #[test]
  fn unit_circle() {
    assert_eq!(interpret("ArcLength[Circle[]]").unwrap(), "2*Pi");
  }

  #[test]
  fn circle_with_radius() {
    assert_eq!(interpret("ArcLength[Circle[{0, 0}, r]]").unwrap(), "2*Pi*r");
  }

  #[test]
  fn circle_numeric_radius() {
    assert_eq!(interpret("ArcLength[Circle[{0, 0}, 5]]").unwrap(), "10*Pi");
  }

  #[test]
  fn line_two_points() {
    assert_eq!(interpret("ArcLength[Line[{{0, 0}, {3, 4}}]]").unwrap(), "5");
  }

  #[test]
  fn line_multi_segment() {
    assert_eq!(
      interpret("ArcLength[Line[{{0, 0}, {1, 0}, {1, 1}}]]").unwrap(),
      "2"
    );
  }

  #[test]
  fn polygon_undefined() {
    assert_eq!(
      interpret("ArcLength[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]")
        .unwrap(),
      "Undefined"
    );
  }

  #[test]
  fn disk_undefined() {
    assert_eq!(interpret("ArcLength[Disk[]]").unwrap(), "Undefined");
  }

  #[test]
  fn triangle_undefined() {
    assert_eq!(
      interpret("ArcLength[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "Undefined"
    );
  }
}

mod perimeter {
  use super::*;

  #[test]
  fn unit_square_polygon() {
    assert_eq!(
      interpret("Perimeter[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]")
        .unwrap(),
      "4"
    );
  }

  #[test]
  fn rectangle() {
    assert_eq!(
      interpret("Perimeter[Rectangle[{0, 0}, {3, 4}]]").unwrap(),
      "14"
    );
  }

  #[test]
  fn unit_rectangle() {
    assert_eq!(interpret("Perimeter[Rectangle[]]").unwrap(), "4");
  }

  #[test]
  fn triangle() {
    assert_eq!(
      interpret("Perimeter[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "2 + Sqrt[2]"
    );
  }

  #[test]
  fn disk() {
    assert_eq!(interpret("Perimeter[Disk[{0, 0}, r]]").unwrap(), "2*Pi*r");
  }

  #[test]
  fn unit_disk() {
    assert_eq!(interpret("Perimeter[Disk[]]").unwrap(), "2*Pi");
  }

  #[test]
  fn circle() {
    // Circle is a 1D curve, Perimeter is Undefined (use ArcLength instead)
    assert_eq!(
      interpret("Perimeter[Circle[{0, 0}, 3]]").unwrap(),
      "Undefined"
    );
  }

  #[test]
  fn unit_circle() {
    assert_eq!(interpret("Perimeter[Circle[]]").unwrap(), "Undefined");
  }
}

mod region_centroid {
  use super::*;

  #[test]
  fn point() {
    assert_eq!(
      interpret("RegionCentroid[Point[{3, 4}]]").unwrap(),
      "{3, 4}"
    );
  }

  #[test]
  fn unit_disk() {
    assert_eq!(interpret("RegionCentroid[Disk[]]").unwrap(), "{0, 0}");
  }

  #[test]
  fn disk_with_center() {
    assert_eq!(
      interpret("RegionCentroid[Disk[{3, 4}, 2]]").unwrap(),
      "{3, 4}"
    );
  }

  #[test]
  fn disk_symbolic() {
    assert_eq!(
      interpret("RegionCentroid[Disk[{a, b}, r]]").unwrap(),
      "{a, b}"
    );
  }

  #[test]
  fn unit_rectangle() {
    assert_eq!(
      interpret("RegionCentroid[Rectangle[]]").unwrap(),
      "{1/2, 1/2}"
    );
  }

  #[test]
  fn rectangle_with_bounds() {
    assert_eq!(
      interpret("RegionCentroid[Rectangle[{0, 0}, {2, 3}]]").unwrap(),
      "{1, 3/2}"
    );
  }

  #[test]
  fn rectangle_symbolic() {
    assert_eq!(
      interpret("RegionCentroid[Rectangle[{a, b}, {c, d}]]").unwrap(),
      "{(a + c)/2, (b + d)/2}"
    );
  }

  #[test]
  fn triangle_basic() {
    assert_eq!(
      interpret("RegionCentroid[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "{1/3, 1/3}"
    );
  }

  #[test]
  fn polygon_square() {
    assert_eq!(
      interpret("RegionCentroid[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]")
        .unwrap(),
      "{1/2, 1/2}"
    );
  }

  #[test]
  fn polygon_trapezoid() {
    assert_eq!(
      interpret("RegionCentroid[Polygon[{{0,0},{2,0},{3,1},{1,1}}]]").unwrap(),
      "{3/2, 1/2}"
    );
  }

  #[test]
  fn line_two_points() {
    assert_eq!(
      interpret("RegionCentroid[Line[{{0, 0}, {1, 1}}]]").unwrap(),
      "{1/2, 1/2}"
    );
  }

  #[test]
  fn ball_3d() {
    assert_eq!(
      interpret("RegionCentroid[Ball[{1, 2, 3}, 5]]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn circle_center() {
    assert_eq!(
      interpret("RegionCentroid[Circle[{2, 3}, 1]]").unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn unevaluated_unknown() {
    assert_eq!(
      interpret("RegionCentroid[foo]").unwrap(),
      "RegionCentroid[foo]"
    );
  }
}

mod triangle {
  use super::*;

  #[test]
  fn default_form() {
    assert_eq!(
      interpret("Triangle[]").unwrap(),
      "Triangle[{{0, 0}, {1, 0}, {0, 1}}]"
    );
  }

  #[test]
  fn explicit_vertices() {
    assert_eq!(
      interpret("Triangle[{{0, 0}, {3, 0}, {0, 4}}]").unwrap(),
      "Triangle[{{0, 0}, {3, 0}, {0, 4}}]"
    );
  }

  #[test]
  fn area() {
    assert_eq!(
      interpret("Area[Triangle[{{0, 0}, {3, 0}, {0, 4}}]]").unwrap(),
      "6"
    );
  }
}

mod circle_points {
  use super::*;

  #[test]
  fn single_point() {
    assert_eq!(interpret("CirclePoints[1]").unwrap(), "{{0, 1}}");
  }

  #[test]
  fn two_points() {
    assert_eq!(interpret("CirclePoints[2]").unwrap(), "{{1, 0}, {-1, 0}}");
  }

  #[test]
  fn three_points() {
    let result = interpret("CirclePoints[3]").unwrap();
    assert!(result.contains("Sqrt[3]/2") && result.contains("-1/2"));
  }

  #[test]
  fn four_points() {
    let result = interpret("CirclePoints[4]").unwrap();
    assert!(result.contains("1/Sqrt[2]"));
  }

  #[test]
  fn six_points() {
    let result = interpret("CirclePoints[6]").unwrap();
    assert!(result.contains("{1, 0}") && result.contains("{-1, 0}"));
  }
}

mod bezier_function {
  use super::*;

  #[test]
  fn quadratic_at_half() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0}}][0.5]").unwrap(),
      "{1., 0.5}"
    );
  }

  #[test]
  fn at_zero() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0}}][0]").unwrap(),
      "{0., 0.}"
    );
  }

  #[test]
  fn at_one() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0}}][1]").unwrap(),
      "{2., 0.}"
    );
  }

  #[test]
  fn cubic() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0},{3,2}}][0.25]").unwrap(),
      "{0.75, 0.453125}"
    );
  }

  #[test]
  fn cubic_midpoint_with_saved_binding() {
    // Save the BezierFunction to a variable and evaluate at 0.5 — the
    // midpoint of the control polygon's x-coords (1.5) and interpolated y.
    assert_eq!(
      interpret(
        "f = BezierFunction[{{0, 0}, {1, 1}, {2, 0}, {3, 2}}]; f[.5]"
      )
      .unwrap(),
      "{1.5, 0.625}"
    );
  }

  #[test]
  fn one_dimensional() {
    assert_eq!(
      interpret("BezierFunction[{{0},{1},{4}}][0.5]").unwrap(),
      "{1.5}"
    );
  }

  #[test]
  fn with_rational_parameter() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0}}][1/3]").unwrap(),
      "{0.6666666666666667, 0.4444444444444445}"
    );
  }

  #[test]
  fn unevaluated_symbolic() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0}}]").unwrap(),
      "BezierFunction[{{0, 0}, {1, 1}, {2, 0}}]"
    );
  }

  #[test]
  fn non_numeric_argument() {
    // With symbolic argument, returns unevaluated
    let result = interpret("BezierFunction[{{0,0},{1,1},{2,0}}][t]").unwrap();
    assert!(result.contains("BezierFunction"));
  }
}

mod region_bounds {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("RegionBounds[x]").unwrap(), "RegionBounds[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[RegionBounds]").unwrap(), "Symbol");
  }
}

mod region_nearest {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("RegionNearest[x, y]").unwrap(),
      "RegionNearest[x, y]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[RegionNearest]").unwrap(), "Symbol");
  }
}

mod region_equal {
  use super::*;

  #[test]
  fn no_args() {
    assert_eq!(interpret("RegionEqual[]").unwrap(), "True");
  }

  #[test]
  fn single_arg() {
    assert_eq!(interpret("RegionEqual[Disk[]]").unwrap(), "True");
  }

  #[test]
  fn same_disk() {
    assert_eq!(interpret("RegionEqual[Disk[], Disk[]]").unwrap(), "True");
  }

  #[test]
  fn disk_with_defaults() {
    assert_eq!(
      interpret("RegionEqual[Disk[{0, 0}], Disk[]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn disk_explicit_defaults() {
    assert_eq!(
      interpret("RegionEqual[Disk[{0, 0}, 1], Disk[]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn different_disks() {
    assert_eq!(
      interpret("RegionEqual[Disk[], Disk[{0, 0}, 2]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn rectangle_defaults() {
    assert_eq!(
      interpret("RegionEqual[Rectangle[{0, 0}, {1, 1}], Rectangle[]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn polygon_equals_rectangle() {
    assert_eq!(
      interpret(
        "RegionEqual[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}], Rectangle[]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn polygon_equals_triangle() {
    assert_eq!(
      interpret("RegionEqual[Polygon[{{0, 0}, {1, 0}, {0, 1}}], Triangle[]]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn triangle_defaults() {
    assert_eq!(
      interpret("RegionEqual[Triangle[], Triangle[{{0, 0}, {1, 0}, {0, 1}}]]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn circle_defaults() {
    assert_eq!(
      interpret("RegionEqual[Circle[], Circle[{0, 0}, 1]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn ball_equals_disk_2d() {
    assert_eq!(
      interpret("RegionEqual[Disk[], Ball[{0, 0}, 1]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn sphere_equals_circle_2d() {
    assert_eq!(
      interpret("RegionEqual[Circle[], Sphere[{0, 0}, 1]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn ball_defaults_3d() {
    assert_eq!(
      interpret("RegionEqual[Ball[], Ball[{0, 0, 0}, 1]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn sphere_defaults_3d() {
    assert_eq!(
      interpret("RegionEqual[Sphere[], Sphere[{0, 0, 0}, 1]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn different_points() {
    assert_eq!(
      interpret("RegionEqual[Point[{0, 0}], Point[{1, 1}]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn same_points() {
    assert_eq!(
      interpret("RegionEqual[Point[{1, 2}], Point[{1, 2}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn same_lines() {
    assert_eq!(
      interpret("RegionEqual[Line[{{0, 0}, {1, 1}}], Line[{{0, 0}, {1, 1}}]]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn reversed_lines() {
    assert_eq!(
      interpret("RegionEqual[Line[{{0, 0}, {1, 1}}], Line[{{1, 1}, {0, 0}}]]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn same_intervals() {
    assert_eq!(
      interpret("RegionEqual[Interval[{0, 1}], Interval[{0, 1}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn different_intervals() {
    assert_eq!(
      interpret("RegionEqual[Interval[{0, 1}], Interval[{0, 2}]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn three_equal_regions() {
    assert_eq!(
      interpret("RegionEqual[Disk[], Disk[], Disk[]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn three_regions_one_different() {
    assert_eq!(
      interpret("RegionEqual[Disk[], Disk[], Disk[{0, 0}, 2]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn polygon_rotated_vertices() {
    assert_eq!(
      interpret("RegionEqual[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}], Polygon[{{1, 0}, {1, 1}, {0, 1}, {0, 0}}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("RegionEqual[x, y]").unwrap(), "RegionEqual[x, y]");
  }
}

mod transformed_region {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("TransformedRegion[x, y]").unwrap(),
      "TransformedRegion[x, y]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[TransformedRegion]").unwrap(), "Symbol");
  }
}

mod delaunay_mesh {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("DelaunayMesh[x]").unwrap(), "DelaunayMesh[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[DelaunayMesh]").unwrap(), "Symbol");
  }
}

mod cantor_mesh {
  use super::*;

  #[test]
  fn level_0() {
    assert_eq!(
      interpret("CantorMesh[0]").unwrap(),
      "MeshRegion[{{0.}, {1.}}, {Line[{{1, 2}}]}]"
    );
  }

  #[test]
  fn level_1() {
    assert_eq!(
      interpret("CantorMesh[1]").unwrap(),
      "MeshRegion[{{0.}, {0.3333333333333333}, {0.6666666666666666}, {1.}}, {Line[{{1, 2}, {3, 4}}]}]"
    );
  }

  #[test]
  fn level_2() {
    assert_eq!(
      interpret("CantorMesh[2]").unwrap(),
      "MeshRegion[{{0.}, {0.1111111111111111}, {0.2222222222222222}, {0.3333333333333333}, {0.6666666666666666}, {0.7777777777777778}, {0.8888888888888888}, {1.}}, {Line[{{1, 2}, {3, 4}, {5, 6}, {7, 8}}]}]"
    );
  }
}

mod planar_angle {
  use super::*;

  #[test]
  fn right_angle() {
    assert_eq!(
      interpret("PlanarAngle[{{1, 0}, {0, 0}, {0, 1}}]").unwrap(),
      "Pi/2"
    );
  }

  #[test]
  fn straight_angle() {
    assert_eq!(
      interpret("PlanarAngle[{{1, 0}, {0, 0}, {-1, 0}}]").unwrap(),
      "Pi"
    );
  }

  #[test]
  fn zero_angle() {
    assert_eq!(
      interpret("PlanarAngle[{{1, 0}, {0, 0}, {1, 0}}]").unwrap(),
      "0"
    );
  }

  #[test]
  fn coincident_points() {
    assert_eq!(
      interpret("PlanarAngle[{{0, 0}, {0, 0}, {0, 1}}]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn pi_over_four() {
    assert_eq!(
      interpret("PlanarAngle[{{1, 0}, {0, 0}, {1, 1}}]").unwrap(),
      "Pi/4"
    );
  }

  #[test]
  fn invalid_input() {
    assert_eq!(interpret("PlanarAngle[0]").unwrap(), "PlanarAngle[0]");
  }
}

mod face_grids {
  use super::*;

  #[test]
  fn symbol() {
    assert_eq!(interpret("FaceGrids").unwrap(), "FaceGrids");
  }

  #[test]
  fn with_args() {
    assert_eq!(
      interpret("FaceGrids[1, 2, 3]").unwrap(),
      "FaceGrids[1, 2, 3]"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(interpret("Attributes[FaceGrids]").unwrap(), "{Protected}");
  }

  #[test]
  fn face_grids_style_symbol() {
    assert_eq!(interpret("FaceGridsStyle").unwrap(), "FaceGridsStyle");
  }
}

mod ellipsoid_function {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("Ellipsoid[{0, 0}, {1, 2}]").unwrap(),
      "Ellipsoid[{0, 0}, {1, 2}]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Ellipsoid]").unwrap(), "Symbol");
  }
}

mod complex_region_plot {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("ComplexRegionPlot[x]").unwrap(),
      "ComplexRegionPlot[x]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[ComplexRegionPlot]").unwrap(), "Symbol");
  }
}

mod array_mesh {
  use super::*;

  #[test]
  fn single_cell() {
    // MeshRegion with Polygon is now rendered to SVG
    assert_eq!(interpret("ArrayMesh[{{1}}]").unwrap(), "-Graphics-");
  }

  #[test]
  fn single_cell_head() {
    assert_eq!(interpret("Head[ArrayMesh[{{1}}]]").unwrap(), "MeshRegion");
  }

  #[test]
  fn horizontal_strip() {
    assert_eq!(interpret("ArrayMesh[{{1, 1}}]").unwrap(), "-Graphics-");
  }

  #[test]
  fn vertical_strip() {
    assert_eq!(interpret("ArrayMesh[{{1}, {1}}]").unwrap(), "-Graphics-");
  }

  #[test]
  fn diagonal_pattern() {
    assert_eq!(
      interpret("ArrayMesh[{{1, 0}, {0, 1}}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn complex_pattern() {
    assert_eq!(
      interpret("ArrayMesh[{{1, 1, 0}, {1, 1, 1}, {0, 1, 0}}]").unwrap(),
      "-Graphics-"
    );
  }
}

mod scaling_transform {
  use super::*;

  #[test]
  fn basic_2d() {
    assert_eq!(
      interpret("ScalingTransform[{2, 3}]").unwrap(),
      "TransformationFunction[{{2, 0, 0}, {0, 3, 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn basic_1d() {
    assert_eq!(
      interpret("ScalingTransform[{2}]").unwrap(),
      "TransformationFunction[{{2, 0}, {0, 1}}]"
    );
  }

  #[test]
  fn basic_3d() {
    assert_eq!(
      interpret("ScalingTransform[{2, 3, 4}]").unwrap(),
      "TransformationFunction[{{2, 0, 0, 0}, {0, 3, 0, 0}, {0, 0, 4, 0}, {0, 0, 0, 1}}]"
    );
  }

  #[test]
  fn with_center() {
    assert_eq!(
      interpret("ScalingTransform[{2, 3}, {1, 1}]").unwrap(),
      "TransformationFunction[{{2, 0, -1}, {0, 3, -2}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn apply_basic() {
    assert_eq!(
      interpret("ScalingTransform[{2, 3}][{1, 1}]").unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn apply_1d() {
    assert_eq!(interpret("ScalingTransform[{2}][{5}]").unwrap(), "{10}");
  }

  #[test]
  fn apply_with_center() {
    assert_eq!(
      interpret("ScalingTransform[{2, 3}, {1, 1}][{2, 2}]").unwrap(),
      "{3, 4}"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("ScalingTransform[{sx, sy}]").unwrap(),
      "TransformationFunction[{{sx, 0, 0}, {0, sy, 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[ScalingTransform]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}
