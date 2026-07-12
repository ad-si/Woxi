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
  fn ellipsoid_2d() {
    assert_eq!(
      interpret("Area[Ellipsoid[{0, 0}, {2, 3}]]").unwrap(),
      "6*Pi"
    );
  }

  #[test]
  fn ellipsoid_2d_offset_center() {
    assert_eq!(
      interpret("Area[Ellipsoid[{1, 2}, {3, 5}]]").unwrap(),
      "15*Pi"
    );
  }

  #[test]
  fn ellipsoid_circular() {
    assert_eq!(
      interpret("Area[Ellipsoid[{1, 1}, {4, 4}]]").unwrap(),
      "16*Pi"
    );
  }

  #[test]
  fn ellipsoid_3d_undefined() {
    // Area is the 2-dimensional measure; a 3D solid has no area.
    assert_eq!(
      interpret("Area[Ellipsoid[{0, 0, 0}, {2, 3, 4}]]").unwrap(),
      "Undefined"
    );
  }

  #[test]
  fn symbolic_radius() {
    assert_eq!(interpret("Area[Disk[{0, 0}, r]]").unwrap(), "Pi*r^2");
  }

  #[test]
  fn default_triangle() {
    assert_eq!(interpret("Area[Triangle[]]").unwrap(), "1/2");
  }

  // Area of a Parallelogram[p, {v1, v2}] is |Det[{v1, v2}]|, independent of the
  // base point p; it must agree with RegionMeasure. Verified against
  // wolframscript.
  #[test]
  fn parallelogram() {
    assert_eq!(
      interpret("Area[Parallelogram[{0, 0}, {{1, 0}, {0, 1}}]]").unwrap(),
      "1"
    );
    // Base point does not affect the area.
    assert_eq!(
      interpret("Area[Parallelogram[{1, 1}, {{3, 0}, {0, 2}}]]").unwrap(),
      "6"
    );
    // Sheared parallelogram.
    assert_eq!(
      interpret("Area[Parallelogram[{0, 0}, {{2, 0}, {1, 3}}]]").unwrap(),
      "6"
    );
    // Parallelogram[] is the unit square.
    assert_eq!(interpret("Area[Parallelogram[]]").unwrap(), "1");
  }

  #[test]
  fn regular_polygon_hexagon() {
    // Area[RegularPolygon[n]] = n/2 * Sin[2 Pi/n]; unit circumradius.
    assert_eq!(
      interpret("Area[RegularPolygon[6]]").unwrap(),
      "(3*Sqrt[3])/2"
    );
  }

  #[test]
  fn regular_polygon_triangle() {
    assert_eq!(
      interpret("Area[RegularPolygon[3]]").unwrap(),
      "(3*Sqrt[3])/4"
    );
  }

  #[test]
  fn regular_polygon_square() {
    assert_eq!(interpret("Area[RegularPolygon[4]]").unwrap(), "2");
  }

  #[test]
  fn regular_polygon_with_radius() {
    // RegularPolygon[r, n] scales the area by r^2.
    assert_eq!(
      interpret("Area[RegularPolygon[r, 6]]").unwrap(),
      "(3*Sqrt[3]*r^2)/2"
    );
    assert_eq!(interpret("Area[RegularPolygon[2, 4]]").unwrap(), "8");
  }

  #[test]
  fn regular_polygon_with_rotation() {
    // RegularPolygon[{r, theta}, n] — rotation doesn't change area.
    assert_eq!(
      interpret("Area[RegularPolygon[{2, Pi/4}, 4]]").unwrap(),
      "8"
    );
  }

  #[test]
  fn regular_polygon_with_center() {
    // RegularPolygon[{x, y}, rspec, n] — translation doesn't change area.
    assert_eq!(
      interpret("Area[RegularPolygon[{1, 2}, 2, 4]]").unwrap(),
      "8"
    );
  }

  #[test]
  fn unit_sphere() {
    // Sphere[] is the unit sphere at the origin; surface area = 4*Pi.
    assert_eq!(interpret("Area[Sphere[]]").unwrap(), "4*Pi");
  }

  #[test]
  fn sphere_with_center_and_radius() {
    // Area[Sphere[{c1, c2, c3}, r]] = 4*Pi*r^2.
    assert_eq!(
      interpret("Area[Sphere[{c1, c2, c3}, r]]").unwrap(),
      "4*Pi*r^2"
    );
    // Concrete radius collapses to 4*Pi*r^2 simplified.
    assert_eq!(interpret("Area[Sphere[{0, 0, 0}, 2]]").unwrap(), "16*Pi");
  }

  #[test]
  fn sphere_origin_default_radius() {
    // Sphere[p] with a 3-D center defaults to unit radius.
    assert_eq!(interpret("Area[Sphere[{1, 2, 3}]]").unwrap(), "4*Pi");
  }
}

mod arc_length {
  use super::*;

  #[test]
  fn unit_circle() {
    assert_eq!(interpret("ArcLength[Circle[]]").unwrap(), "2*Pi");
  }

  // ArcLength[curve, {t, a, b}] — parameterized curves. Integrand = speed
  // Sqrt[Sum of squared derivatives]; for a scalar f it is Sqrt[1 + f'^2].
  #[test]
  fn parametric_unit_circle() {
    assert_eq!(
      interpret("ArcLength[{Sin[t], Cos[t]}, {t, 0, 2 Pi}]").unwrap(),
      "2*Pi"
    );
  }

  #[test]
  fn parametric_radius_three_semicircle() {
    assert_eq!(
      interpret("ArcLength[{3 Cos[t], 3 Sin[t]}, {t, 0, Pi}]").unwrap(),
      "3*Pi"
    );
  }

  #[test]
  fn parametric_helix() {
    // One turn of a helix {Cos[t], Sin[t], t}: speed Sqrt[2].
    assert_eq!(
      interpret("ArcLength[{Cos[t], Sin[t], t}, {t, 0, 2 Pi}]").unwrap(),
      "2*Sqrt[2]*Pi"
    );
  }

  // Unbounded 1-D regions have infinite arc length.
  #[test]
  fn unbounded_regions_are_infinite() {
    assert_eq!(
      interpret("ArcLength[HalfLine[{0, 0}, {1, 1}]]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("ArcLength[InfiniteLine[{0, 0}, {1, 1}]]").unwrap(),
      "Infinity"
    );
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

  #[test]
  fn ellipsoid_undefined() {
    // A filled ellipse/ellipsoid is not a curve, so its arc length is
    // Undefined (use Perimeter for the boundary length of a 2D ellipse).
    assert_eq!(
      interpret("ArcLength[Ellipsoid[{0, 0}, {2, 3}]]").unwrap(),
      "Undefined"
    );
    assert_eq!(
      interpret("ArcLength[Ellipsoid[{0, 0, 0}, {1, 2, 3}]]").unwrap(),
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

  // The perimeter of a 2D Ellipsoid (filled ellipse) is the ellipse
  // circumference 4*r2*EllipticE[1 - (r1/r2)^2], using the second semi-axis
  // as the reference (WL's convention).
  #[test]
  fn ellipsoid() {
    assert_eq!(
      interpret("Perimeter[Ellipsoid[{0, 0}, {2, 3}]]").unwrap(),
      "12*EllipticE[5/9]"
    );
    assert_eq!(
      interpret("Perimeter[Ellipsoid[{0, 0}, {3, 2}]]").unwrap(),
      "8*EllipticE[-5/4]"
    );
  }

  // A circular Ellipsoid (equal semi-axes) reduces to 2*Pi*r since
  // EllipticE[0] = Pi/2.
  #[test]
  fn ellipsoid_circular() {
    assert_eq!(
      interpret("Perimeter[Ellipsoid[{0, 0}, {2, 2}]]").unwrap(),
      "4*Pi"
    );
  }

  // The center does not affect the perimeter.
  #[test]
  fn ellipsoid_offset_center() {
    assert_eq!(
      interpret("Perimeter[Ellipsoid[{1, 2}, {2, 3}]]").unwrap(),
      "12*EllipticE[5/9]"
    );
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

  // An ellipsoid is symmetric about its center, so its centroid is the center
  // (in any dimension).
  #[test]
  fn ellipsoid_center() {
    assert_eq!(
      interpret("RegionCentroid[Ellipsoid[{1, 2}, {3, 4}]]").unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret("RegionCentroid[Ellipsoid[{0, 0, 0}, {1, 2, 3}]]").unwrap(),
      "{0, 0, 0}"
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

mod volume {
  use super::*;

  #[test]
  fn unit_cuboid() {
    // Cuboid[] is the unit hypercube — volume 1.
    assert_eq!(interpret("Volume[Cuboid[]]").unwrap(), "1");
  }

  #[test]
  fn cuboid_with_single_corner() {
    // Cuboid[pmin] is also a unit hypercube.
    assert_eq!(interpret("Volume[Cuboid[{1, 2, 3}]]").unwrap(), "1");
  }

  #[test]
  fn cuboid_with_bounds_numeric() {
    // 2×3×4 brick.
    assert_eq!(
      interpret("Volume[Cuboid[{0, 0, 0}, {2, 3, 4}]]").unwrap(),
      "24"
    );
  }

  #[test]
  fn cuboid_with_bounds_symbolic() {
    // General 3-D cuboid: Abs[(d - a) * (e - b) * (f - c)].
    assert_eq!(
      interpret("Volume[Cuboid[{a, b, c}, {d, e, f}]]").unwrap(),
      "Abs[(-a + d)*(-b + e)*(-c + f)]"
    );
  }

  #[test]
  fn cuboid_2d() {
    // Volume[Cuboid[2D]] is Undefined — Volume is only the n-dim measure
    // when n == 3. Use Area or RegionMeasure for 2D Cuboids.
    assert_eq!(
      interpret("Volume[Cuboid[{0, 0}, {2, 3}]]").unwrap(),
      "Undefined"
    );
  }

  #[test]
  fn cylinder_default() {
    // Cylinder[] is a unit cylinder from {0, 0, -1} to {0, 0, 1}:
    // radius 1, length 2 -> volume 2 Pi.
    assert_eq!(interpret("Volume[Cylinder[]]").unwrap(), "2*Pi");
  }

  #[test]
  fn cylinder_with_radius() {
    // Pi * 2^2 * 5 = 20 Pi.
    assert_eq!(
      interpret("Volume[Cylinder[{{0, 0, 0}, {0, 0, 5}}, 2]]").unwrap(),
      "20*Pi"
    );
  }

  #[test]
  fn cylinder_default_radius() {
    // Default radius is 1: Pi * 1^2 * 5 = 5 Pi.
    assert_eq!(
      interpret("Volume[Cylinder[{{0, 0, 0}, {0, 0, 5}}]]").unwrap(),
      "5*Pi"
    );
  }

  #[test]
  fn cylinder_symbolic() {
    // Pi * r^2 * Sqrt[Sum[(x2_i - x1_i)^2]] in any orientation.
    assert_eq!(
      interpret(
        "Volume[Cylinder[{{Subscript[x, 1], Subscript[y, 1], Subscript[z, 1]}, {Subscript[x, 2], Subscript[y, 2], Subscript[z, 2]}}, r]]"
      )
      .unwrap(),
      "Pi*r^2*Sqrt[(Subscript[x, 1] - Subscript[x, 2])^2 + (Subscript[y, 1] - Subscript[y, 2])^2 + (Subscript[z, 1] - Subscript[z, 2])^2]"
    );
  }

  #[test]
  fn cone_default() {
    // Cone[] is a unit cone from {0, 0, -1} to {0, 0, 1}: r = 1, length 2
    // -> volume (1/3) * Pi * 1 * 2 = 2 Pi / 3.
    assert_eq!(interpret("Volume[Cone[]]").unwrap(), "(2*Pi)/3");
  }

  #[test]
  fn cone_with_radius() {
    // (1/3) * Pi * 2^2 * 5 = 20 Pi / 3.
    assert_eq!(
      interpret("Volume[Cone[{{0, 0, 0}, {0, 0, 5}}, 2]]").unwrap(),
      "(20*Pi)/3"
    );
  }

  #[test]
  fn cone_default_radius() {
    // Default radius 1: (1/3) * Pi * 5 = 5 Pi / 3.
    assert_eq!(
      interpret("Volume[Cone[{{0, 0, 0}, {0, 0, 5}}]]").unwrap(),
      "(5*Pi)/3"
    );
  }

  #[test]
  fn cone_symbolic() {
    // (Pi * r^2 * Sqrt[Sum[(x2_i - x1_i)^2]]) / 3.
    assert_eq!(
      interpret(
        "Volume[Cone[{{Subscript[x, 1], Subscript[y, 1], Subscript[z, 1]}, {Subscript[x, 2], Subscript[y, 2], Subscript[z, 2]}}, r]]"
      )
      .unwrap(),
      "(Pi*r^2*Sqrt[(Subscript[x, 1] - Subscript[x, 2])^2 + (Subscript[y, 1] - Subscript[y, 2])^2 + (Subscript[z, 1] - Subscript[z, 2])^2])/3"
    );
  }

  // A solid 3-D ball has volume (4/3) Pi r^3 (default unit ball is 3-D).
  #[test]
  fn ball_3d() {
    assert_eq!(interpret("Volume[Ball[{0, 0, 0}, 3]]").unwrap(), "36*Pi");
    assert_eq!(interpret("Volume[Ball[]]").unwrap(), "(4*Pi)/3");
    assert_eq!(interpret("Volume[Ball[{1, 2, 3}]]").unwrap(), "(4*Pi)/3");
    assert_eq!(
      interpret("Volume[Ball[{0, 0, 3}, r]]").unwrap(),
      "(4*Pi*r^3)/3"
    );
  }

  // A ball that is not 3-dimensional has no 3-volume.
  #[test]
  fn ball_non_3d_undefined() {
    assert_eq!(interpret("Volume[Ball[{0, 0}, 2]]").unwrap(), "Undefined");
  }

  // A solid 3-D ellipsoid has volume (4/3) Pi r1 r2 r3.
  #[test]
  fn ellipsoid_3d() {
    assert_eq!(
      interpret("Volume[Ellipsoid[{0, 0, 0}, {1, 2, 3}]]").unwrap(),
      "8*Pi"
    );
    assert_eq!(
      interpret("Volume[Ellipsoid[{0, 0, 0}, {r1, r2, r3}]]").unwrap(),
      "(4*Pi*r1*r2*r3)/3"
    );
    // A 2-D ellipse has no 3-volume.
    assert_eq!(
      interpret("Volume[Ellipsoid[{0, 0}, {2, 3}]]").unwrap(),
      "Undefined"
    );
  }

  // Volume is the 3-dimensional measure, so regions of lower intrinsic
  // dimension (and surfaces like Sphere) are Undefined.
  #[test]
  fn lower_dimensional_regions_undefined() {
    assert_eq!(
      interpret("Volume[Sphere[{0, 0, 0}, 2]]").unwrap(),
      "Undefined"
    );
    assert_eq!(interpret("Volume[Disk[]]").unwrap(), "Undefined");
    assert_eq!(interpret("Volume[Rectangle[]]").unwrap(), "Undefined");
    assert_eq!(
      interpret("Volume[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "Undefined"
    );
    assert_eq!(
      interpret("Volume[Polygon[{{0, 0}, {1, 0}, {1, 1}}]]").unwrap(),
      "Undefined"
    );
    assert_eq!(interpret("Volume[Circle[]]").unwrap(), "Undefined");
    assert_eq!(
      interpret("Volume[Line[{{0, 0}, {1, 1}}]]").unwrap(),
      "Undefined"
    );
    assert_eq!(interpret("Volume[Point[{1, 2, 3}]]").unwrap(), "Undefined");
  }
}

mod region_measure {
  use super::*;

  #[test]
  fn ellipsoid_3d_numeric() {
    // 3-D ellipsoid volume = 4 Pi r1 r2 r3 / 3.
    assert_eq!(
      interpret("RegionMeasure[Ellipsoid[{0, 0, 0}, {1, 2, 3}]]").unwrap(),
      "8*Pi"
    );
  }

  #[test]
  fn ellipsoid_3d_symbolic() {
    assert_eq!(
      interpret(
        "RegionMeasure[Ellipsoid[{Subscript[c, 1], Subscript[c, 2], Subscript[c, 3]}, {Subscript[r, 1], Subscript[r, 2], Subscript[r, 3]}]]"
      )
      .unwrap(),
      "(4*Pi*Subscript[r, 1]*Subscript[r, 2]*Subscript[r, 3])/3"
    );
  }

  #[test]
  fn ellipsoid_2d() {
    // 2-D ellipse area = Pi r1 r2.
    assert_eq!(
      interpret("RegionMeasure[Ellipsoid[{0, 0}, {2, 3}]]").unwrap(),
      "6*Pi"
    );
  }

  // RegionMeasure returns the measure of a region's intrinsic dimension:
  // 2D regions → area, 3D solids → volume, curves → length.
  #[test]
  fn disk_area() {
    assert_eq!(interpret("RegionMeasure[Disk[{0, 0}, 2]]").unwrap(), "4*Pi");
    assert_eq!(interpret("RegionMeasure[Disk[]]").unwrap(), "Pi");
    assert_eq!(
      interpret("RegionMeasure[Disk[{0, 0}, {2, 3}]]").unwrap(),
      "6*Pi"
    );
  }

  #[test]
  fn rectangle_area() {
    assert_eq!(
      interpret("RegionMeasure[Rectangle[{0, 0}, {2, 3}]]").unwrap(),
      "6"
    );
    assert_eq!(interpret("RegionMeasure[Rectangle[]]").unwrap(), "1");
  }

  #[test]
  fn triangle_and_polygon_area() {
    assert_eq!(
      interpret("RegionMeasure[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("RegionMeasure[Polygon[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}]]")
        .unwrap(),
      "4"
    );
  }

  #[test]
  fn cuboid_cylinder_cone_volume() {
    assert_eq!(
      interpret("RegionMeasure[Cuboid[{0, 0, 0}, {1, 2, 3}]]").unwrap(),
      "6"
    );
    assert_eq!(interpret("RegionMeasure[Cylinder[]]").unwrap(), "2*Pi");
    assert_eq!(interpret("RegionMeasure[Cone[]]").unwrap(), "(2*Pi)/3");
  }

  // Volume of a tetrahedron / 3-simplex / parallelepiped via the determinant.
  #[test]
  fn tetrahedron_simplex_parallelepiped_volume() {
    assert_eq!(
      interpret("Volume[Tetrahedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}]]")
        .unwrap(),
      "1/6"
    );
    assert_eq!(
      interpret("Volume[Tetrahedron[{{1,1,1},{2,3,1},{4,1,2},{1,2,5}}]]")
        .unwrap(),
      "25/6"
    );
    assert_eq!(
      interpret("Volume[Simplex[{{0,0,0},{2,0,0},{0,3,0},{0,0,6}}]]").unwrap(),
      "6"
    );
    assert_eq!(
      interpret("Volume[Parallelepiped[{0,0,0},{{1,0,0},{0,2,0},{0,0,3}}]]")
        .unwrap(),
      "6"
    );
  }

  // The same shapes via RegionMeasure (intrinsic dimension); a 2-simplex
  // and Area give the triangle area.
  #[test]
  fn simplex_region_measure_and_area() {
    assert_eq!(
      interpret(
        "RegionMeasure[Tetrahedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}]]"
      )
      .unwrap(),
      "1/6"
    );
    assert_eq!(
      interpret("RegionMeasure[Simplex[{{0,0},{2,0},{0,2}}]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Area[Simplex[{{0,0},{2,0},{0,2}}]]").unwrap(),
      "2"
    );
    // A tetrahedron is a 3-D solid, so its 2-area is Undefined.
    assert_eq!(
      interpret("Area[Tetrahedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}]]")
        .unwrap(),
      "Undefined"
    );
  }

  // Centroid of a tetrahedron / simplex is the mean of its vertices.
  #[test]
  fn tetrahedron_simplex_centroid() {
    assert_eq!(
      interpret(
        "RegionCentroid[Tetrahedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}]]"
      )
      .unwrap(),
      "{1/4, 1/4, 1/4}"
    );
    assert_eq!(
      interpret("RegionCentroid[Simplex[{{0,0,0},{3,0,0},{0,3,0},{0,0,3}}]]")
        .unwrap(),
      "{3/4, 3/4, 3/4}"
    );
  }

  // An n-ball's measure is its closed-form volume Pi^(n/2) r^n / Gamma[n/2+1];
  // the dimension is the length of the center vector (default 3D).
  #[test]
  fn ball_volume_by_dimension() {
    assert_eq!(interpret("RegionMeasure[Ball[]]").unwrap(), "(4*Pi)/3");
    assert_eq!(interpret("RegionMeasure[Ball[{1, 1}]]").unwrap(), "Pi");
    assert_eq!(
      interpret("RegionMeasure[Ball[{0, 0, 0}, 3]]").unwrap(),
      "36*Pi"
    );
    // 4-ball of radius 2: Pi^2/2 * 2^4 = 8 Pi^2.
    assert_eq!(
      interpret("RegionMeasure[Ball[{0, 0, 0, 0}, 2]]").unwrap(),
      "8*Pi^2"
    );
  }

  // A Sphere is a 2D surface → surface area; Circle/Line are curves → length.
  #[test]
  fn surface_and_curve_measures() {
    assert_eq!(
      interpret("RegionMeasure[Sphere[{0, 0, 0}, 2]]").unwrap(),
      "16*Pi"
    );
    assert_eq!(
      interpret("RegionMeasure[Circle[{0, 0}, 2]]").unwrap(),
      "4*Pi"
    );
    assert_eq!(
      interpret("RegionMeasure[Line[{{0, 0}, {3, 4}}]]").unwrap(),
      "5"
    );
  }

  // A Point is 0-dimensional, so its measure is the counting measure.
  #[test]
  fn point_counting_measure() {
    assert_eq!(interpret("RegionMeasure[Point[{1, 2}]]").unwrap(), "1");
    assert_eq!(
      interpret("RegionMeasure[Point[{{1, 2}, {3, 4}}]]").unwrap(),
      "2"
    );
  }
}

mod region_dimension {
  use super::*;

  // RegionDimension gives the intrinsic (manifold) dimension of a region,
  // independent of the embedding dimension.
  #[test]
  fn fixed_dimension_regions() {
    assert_eq!(interpret("RegionDimension[Point[{1, 2}]]").unwrap(), "0");
    assert_eq!(
      interpret("RegionDimension[Line[{{0, 0}, {1, 1}}]]").unwrap(),
      "1"
    );
    assert_eq!(interpret("RegionDimension[Circle[]]").unwrap(), "1");
    assert_eq!(interpret("RegionDimension[Disk[]]").unwrap(), "2");
    assert_eq!(interpret("RegionDimension[Rectangle[]]").unwrap(), "2");
    assert_eq!(
      interpret("RegionDimension[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("RegionDimension[Polygon[{{0, 0}, {1, 0}, {1, 1}}]]").unwrap(),
      "2"
    );
    assert_eq!(interpret("RegionDimension[Cylinder[]]").unwrap(), "3");
    assert_eq!(interpret("RegionDimension[Cone[]]").unwrap(), "3");
  }

  // A Ball / Cuboid takes the dimension of its defining coordinate vector
  // (default 3); they generalize to any dimension.
  #[test]
  fn ball_and_cuboid_by_coordinates() {
    assert_eq!(interpret("RegionDimension[Ball[]]").unwrap(), "3");
    assert_eq!(interpret("RegionDimension[Ball[{0, 0}, 2]]").unwrap(), "2");
    assert_eq!(
      interpret("RegionDimension[Ball[{0, 0, 0, 0}, 2]]").unwrap(),
      "4"
    );
    assert_eq!(interpret("RegionDimension[Cuboid[]]").unwrap(), "3");
    assert_eq!(
      interpret("RegionDimension[Cuboid[{0, 0}, {1, 2}]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("RegionDimension[Ellipsoid[{0, 0, 0}, {1, 2, 3}]]").unwrap(),
      "3"
    );
  }

  // A Sphere is the (n-1)-dimensional surface of an n-ball.
  #[test]
  fn sphere_is_surface() {
    assert_eq!(interpret("RegionDimension[Sphere[]]").unwrap(), "2");
    assert_eq!(
      interpret("RegionDimension[Sphere[{0, 0, 0}, 2]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("RegionDimension[Sphere[{0, 0}, 2]]").unwrap(),
      "1"
    );
  }

  // Simplex[n] is an n-simplex; Simplex[{p0,…,pk}] is a k-simplex.
  // Parallelepiped is spanned by its list of vectors.
  #[test]
  fn simplex_and_parallelepiped() {
    assert_eq!(interpret("RegionDimension[Simplex[2]]").unwrap(), "2");
    assert_eq!(
      interpret("RegionDimension[Simplex[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("RegionDimension[Parallelepiped[{0, 0}, {{1, 0}, {0, 1}}]]")
        .unwrap(),
      "2"
    );
    assert_eq!(
      interpret("RegionDimension[Annulus[{0, 0}, {1, 2}]]").unwrap(),
      "2"
    );
  }
}

mod region_embedding_dimension {
  use super::*;

  // RegionEmbeddingDimension gives the ambient (coordinate) dimension,
  // i.e. the number of {min, max} pairs in the bounding box.
  #[test]
  fn embedding_dimension_regions() {
    assert_eq!(interpret("RegionEmbeddingDimension[Disk[]]").unwrap(), "2");
    // Circle is a 1-D curve embedded in the plane.
    assert_eq!(
      interpret("RegionEmbeddingDimension[Circle[{1, 1}, 2]]").unwrap(),
      "2"
    );
    assert_eq!(interpret("RegionEmbeddingDimension[Ball[]]").unwrap(), "3");
    assert_eq!(
      interpret("RegionEmbeddingDimension[Cuboid[]]").unwrap(),
      "3"
    );
    // Sphere is a 2-D surface embedded in 3-space.
    assert_eq!(
      interpret("RegionEmbeddingDimension[Sphere[{0, 0, 0}, 1]]").unwrap(),
      "3"
    );
    // Line embedding dimension follows its point coordinates.
    assert_eq!(
      interpret("RegionEmbeddingDimension[Line[{{0, 0, 0}, {1, 1, 1}}]]")
        .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("RegionEmbeddingDimension[Point[{1, 2, 3}]]").unwrap(),
      "3"
    );
  }

  #[test]
  fn embedding_dimension_unevaluated() {
    assert_eq!(
      interpret("RegionEmbeddingDimension[x]").unwrap(),
      "RegionEmbeddingDimension[x]"
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

mod triangle_center {
  use super::*;

  // TriangleCenter[tri] defaults to the centroid.
  #[test]
  fn default_is_centroid() {
    assert_eq!(
      interpret("TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}]]").unwrap(),
      "{4/3, 1}"
    );
    assert_eq!(
      interpret("TriangleCenter[Triangle[]]").unwrap(),
      "{1/3, 1/3}"
    );
  }

  #[test]
  fn centroid() {
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}], \"Centroid\"]"
      )
      .unwrap(),
      "{4/3, 1}"
    );
    // Symbolic vertices: the mean of the coordinates.
    assert_eq!(
      interpret("TriangleCenter[Triangle[{{a, b}, {c, d}, {e, f}}]]").unwrap(),
      "{(a + c + e)/3, (b + d + f)/3}"
    );
  }

  #[test]
  fn incenter() {
    // 3-4-5 right triangle: inradius 1, incenter {1, 1}.
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}], \"Incenter\"]"
      )
      .unwrap(),
      "{1, 1}"
    );
    // Matches the center of Insphere for the unit right triangle.
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{0, 0}, {1, 0}, {0, 1}}], \"Incenter\"]"
      )
      .unwrap(),
      "{(2 + Sqrt[2])^(-1), (2 + Sqrt[2])^(-1)}"
    );
  }

  #[test]
  fn circumcenter() {
    // Right triangle: circumcenter is the hypotenuse midpoint.
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}], \"Circumcenter\"]"
      )
      .unwrap(),
      "{2, 3/2}"
    );
    // Scalene triangle with a rational circumcenter; matches the center of
    // Circumsphere[{{-1, 0}, {5, 1}, {2, 4}}].
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{-1, 0}, {5, 1}, {2, 4}}], \"Circumcenter\"]"
      )
      .unwrap(),
      "{27/14, 13/14}"
    );
  }

  #[test]
  fn orthocenter() {
    // Right triangle: orthocenter is the right-angle vertex.
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}], \"Orthocenter\"]"
      )
      .unwrap(),
      "{0, 0}"
    );
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{-1, 0}, {5, 1}, {2, 4}}], \"Orthocenter\"]"
      )
      .unwrap(),
      "{15/7, 22/7}"
    );
  }

  #[test]
  fn nine_point_center() {
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}], \
         \"NinePointCenter\"]"
      )
      .unwrap(),
      "{1, 3/4}"
    );
    // Midpoint of the circumcenter {27/14, 13/14} and orthocenter
    // {15/7, 22/7}.
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{-1, 0}, {5, 1}, {2, 4}}], \
         \"NinePointCenter\"]"
      )
      .unwrap(),
      "{57/28, 57/28}"
    );
  }

  #[test]
  fn symmedian_point() {
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}], \
         \"SymmedianPoint\"]"
      )
      .unwrap(),
      "{18/25, 24/25}"
    );
  }

  // In an equilateral triangle every center coincides.
  #[test]
  fn equilateral_centers_coincide() {
    let tri = "Triangle[{{0, 0}, {1, 0}, {1/2, Sqrt[3]/2}}]";
    for ctype in [
      "Incenter",
      "Circumcenter",
      "Orthocenter",
      "NinePointCenter",
      "SymmedianPoint",
    ] {
      assert_eq!(
        interpret(&format!("TriangleCenter[{tri}, \"{ctype}\"]")).unwrap(),
        "{1/2, 1/(2*Sqrt[3])}",
        "center type {ctype}"
      );
    }
  }

  // A bare vertex list works like Triangle[{p1, p2, p3}].
  #[test]
  fn bare_vertex_list() {
    assert_eq!(
      interpret("TriangleCenter[{{0, 0}, {4, 0}, {0, 3}}, \"Incenter\"]")
        .unwrap(),
      "{1, 1}"
    );
  }

  // Triangles in 3D are supported.
  #[test]
  fn triangle_3d() {
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{0, 0, 0}, {4, 0, 0}, {0, 3, 0}}], \
         \"Circumcenter\"]"
      )
      .unwrap(),
      "{2, 3/2, 0}"
    );
  }

  #[test]
  fn float_vertices() {
    assert_eq!(
      interpret(
        "TriangleCenter[Triangle[{{0., 0.}, {4., 0.}, {0., 3.}}], \
         \"Circumcenter\"]"
      )
      .unwrap(),
      "{2., 1.5}"
    );
  }

  // Unknown center types and non-triangle input stay unevaluated.
  #[test]
  fn unevaluated_cases() {
    assert_eq!(
      interpret("TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}], \"Foo\"]")
        .unwrap(),
      "TriangleCenter[Triangle[{{0, 0}, {4, 0}, {0, 3}}], Foo]"
    );
    assert_eq!(
      interpret("TriangleCenter[foo]").unwrap(),
      "TriangleCenter[foo]"
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

  // CirclePoints[{r, theta}, n] starts at angle theta with radius r.
  #[test]
  fn radius_and_angle() {
    assert_eq!(
      interpret("CirclePoints[{2, 0}, 4]").unwrap(),
      "{{2, 0}, {0, 2}, {-2, 0}, {0, -2}}"
    );
    assert_eq!(
      interpret("CirclePoints[{1, Pi/2}, 4]").unwrap(),
      "{{0, 1}, {-1, 0}, {0, -1}, {1, 0}}"
    );
  }

  // CirclePoints[r, n] uses the default starting angle, scaled by r.
  #[test]
  fn radius_only() {
    assert_eq!(
      interpret("CirclePoints[2, 4]").unwrap(),
      "{{Sqrt[2], -Sqrt[2]}, {Sqrt[2], Sqrt[2]}, {-Sqrt[2], Sqrt[2]}, \
       {-Sqrt[2], -Sqrt[2]}}"
    );
  }

  // CirclePoints[{cx, cy}, {r, theta}, n] translates the points to a center.
  #[test]
  fn center_form() {
    assert_eq!(
      interpret("CirclePoints[{1, 1}, {2, 0}, 4]").unwrap(),
      "{{3, 1}, {1, 3}, {-1, 1}, {1, -1}}"
    );
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
      interpret("f = BezierFunction[{{0, 0}, {1, 1}, {2, 0}, {3, 2}}]; f[.5]")
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
    // wolframscript expands `BezierFunction[points]` into its 7-arg
    // structured form: BezierFunction[degree, knots, {n}, {points, {}},
    // {0}, MachinePrecision, Unevaluated].
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0}}]").unwrap(),
      "BezierFunction[1, {{0., 1.}}, {2}, {{{0., 0.}, {1., 1.}, {2., 0.}}, {}}, {0}, MachinePrecision, Unevaluated]"
    );
  }

  #[test]
  fn non_numeric_argument() {
    // With symbolic argument, returns unevaluated
    let result = interpret("BezierFunction[{{0,0},{1,1},{2,0}}][t]").unwrap();
    assert!(result.contains("BezierFunction"));
  }
}

mod bspline_function {
  use super::*;

  #[test]
  fn curve_interior_quarter() {
    assert_eq!(
      interpret("BSplineFunction[{{1,1},{2,3},{3,-1},{4,1},{5,0}}][0.25]")
        .unwrap(),
      "{2.1875, 1.6875}"
    );
  }

  #[test]
  fn curve_interior_three_quarter() {
    assert_eq!(
      interpret("BSplineFunction[{{1,1},{2,3},{3,-1},{4,1},{5,0}}][0.75]")
        .unwrap(),
      "{3.8125, 0.4375}"
    );
  }

  #[test]
  fn curve_midpoint_rational_parameter() {
    assert_eq!(
      interpret("BSplineFunction[{{1,1},{2,3},{3,-1},{4,1},{5,0}}][1/2]")
        .unwrap(),
      "{3., 0.5}"
    );
  }

  #[test]
  fn curve_endpoints_are_clamped() {
    // Clamped knot vector ⇒ the curve interpolates its first/last control point.
    assert_eq!(
      interpret("BSplineFunction[{{1,1},{2,3},{3,-1},{4,1},{5,0}}][0]")
        .unwrap(),
      "{1., 1.}"
    );
    assert_eq!(
      interpret("BSplineFunction[{{1,1},{2,3},{3,-1},{4,1},{5,0}}][1]")
        .unwrap(),
      "{5., 0.}"
    );
  }

  #[test]
  fn quadratic_three_points() {
    // n = 3 control points ⇒ degree 2 (a quadratic Bézier on a clamped knot
    // vector with no interior knots).
    assert_eq!(
      interpret("BSplineFunction[{{0,0},{1,1},{2,0}}][0.25]").unwrap(),
      "{0.5, 0.375}"
    );
  }

  #[test]
  fn one_dimensional_points() {
    assert_eq!(
      interpret("BSplineFunction[{{0},{1},{4},{9}}][0.5]").unwrap(),
      "{3.}"
    );
  }

  #[test]
  fn saved_binding() {
    assert_eq!(
      interpret(
        "f = BSplineFunction[{{1,1},{2,3},{3,-1},{4,1},{5,0}}]; f[0.25]"
      )
      .unwrap(),
      "{2.1875, 1.6875}"
    );
  }

  #[test]
  fn surface_tensor_product() {
    // A bilinear B-spline surface (2×2 control net, degree {1, 1}).
    assert_eq!(
      interpret(
        "BSplineFunction[{{{0,0,0},{1,0,1}},{{0,1,1},{1,1,0}}}][0.25,0.75]"
      )
      .unwrap(),
      "{0.75, 0.25, 0.625}"
    );
  }

  #[test]
  fn unevaluated_curve_structured_form() {
    // wolframscript expands `BSplineFunction[points]` into its structured
    // 9-arg form.
    assert_eq!(
      interpret("BSplineFunction[{{1,1},{2,3},{3,-1},{4,1},{5,0}}]").unwrap(),
      "BSplineFunction[1, {{0., 1.}}, {3}, {False}, \
{{{1., 1.}, {2., 3.}, {3., -1.}, {4., 1.}, {5., 0.}}, Automatic}, \
{{0., 0., 0., 0., 0.5, 1., 1., 1., 1.}}, {0}, MachinePrecision, Unevaluated]"
    );
  }

  #[test]
  fn unevaluated_symbolic_argument() {
    let result = interpret("BSplineFunction[{{0,0},{1,1},{2,0}}][t]").unwrap();
    assert!(result.contains("BSplineFunction"));
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

  #[test]
  fn half_line_positive_direction() {
    // wolframscript: {{0, Infinity}, {0, Infinity}}.
    assert_eq!(
      interpret("RegionBounds[HalfLine[{{0, 0}, {1, 1}}]]").unwrap(),
      "{{0, Infinity}, {0, Infinity}}"
    );
  }

  #[test]
  fn half_line_negative_direction() {
    assert_eq!(
      interpret("RegionBounds[HalfLine[{{0, 0}, {-1, -1}}]]").unwrap(),
      "{{-Infinity, 0}, {-Infinity, 0}}"
    );
  }

  #[test]
  fn half_line_axis_aligned() {
    // y direction is zero — that dimension stays at the start.
    assert_eq!(
      interpret("RegionBounds[HalfLine[{{0, 0}, {1, 0}}]]").unwrap(),
      "{{0, Infinity}, {0, 0}}"
    );
  }

  #[test]
  fn half_line_offset_start() {
    assert_eq!(
      interpret("RegionBounds[HalfLine[{{1, 2}, {3, 5}}]]").unwrap(),
      "{{1, Infinity}, {2, Infinity}}"
    );
  }

  #[test]
  fn line_two_points() {
    assert_eq!(
      interpret("RegionBounds[Line[{{0, 0}, {1, 1}}]]").unwrap(),
      "{{0, 1}, {0, 1}}"
    );
  }

  #[test]
  fn line_polyline() {
    // Multi-vertex line: bounds are componentwise min/max.
    assert_eq!(
      interpret("RegionBounds[Line[{{0, 0}, {1, 1}, {2, -1}}]]").unwrap(),
      "{{0, 2}, {-1, 1}}"
    );
  }

  // Disk/Circle bounds are center +/- radius.
  #[test]
  fn disk_and_circle() {
    assert_eq!(
      interpret("RegionBounds[Disk[]]").unwrap(),
      "{{-1, 1}, {-1, 1}}"
    );
    assert_eq!(
      interpret("RegionBounds[Disk[{1, 2}, 3]]").unwrap(),
      "{{-2, 4}, {-1, 5}}"
    );
    assert_eq!(
      interpret("RegionBounds[Circle[{1, 1}, 2]]").unwrap(),
      "{{-1, 3}, {-1, 3}}"
    );
    // Elliptical disk with {rx, ry} semi-axes.
    assert_eq!(
      interpret("RegionBounds[Disk[{0, 0}, {2, 3}]]").unwrap(),
      "{{-2, 2}, {-3, 3}}"
    );
  }

  // Ball/Sphere are 3D; bounds are center +/- radius per axis.
  #[test]
  fn ball_and_sphere() {
    assert_eq!(
      interpret("RegionBounds[Ball[{0, 0, 0}, 2]]").unwrap(),
      "{{-2, 2}, {-2, 2}, {-2, 2}}"
    );
    assert_eq!(
      interpret("RegionBounds[Sphere[{0, 0, 0}, 2]]").unwrap(),
      "{{-2, 2}, {-2, 2}, {-2, 2}}"
    );
  }

  #[test]
  fn rectangle_cuboid_default() {
    assert_eq!(
      interpret("RegionBounds[Rectangle[{0, 0}, {2, 3}]]").unwrap(),
      "{{0, 2}, {0, 3}}"
    );
    assert_eq!(
      interpret("RegionBounds[Rectangle[]]").unwrap(),
      "{{0, 1}, {0, 1}}"
    );
    assert_eq!(
      interpret("RegionBounds[Cuboid[{0, 0, 0}, {1, 2, 3}]]").unwrap(),
      "{{0, 1}, {0, 2}, {0, 3}}"
    );
  }

  #[test]
  fn triangle_polygon_point() {
    assert_eq!(
      interpret("RegionBounds[Triangle[{{0, 0}, {4, 0}, {1, 3}}]]").unwrap(),
      "{{0, 4}, {0, 3}}"
    );
    assert_eq!(
      interpret("RegionBounds[Polygon[{{0, 0}, {2, 0}, {1, 1}}]]").unwrap(),
      "{{0, 2}, {0, 1}}"
    );
    assert_eq!(
      interpret("RegionBounds[Point[{3, 4}]]").unwrap(),
      "{{3, 3}, {4, 4}}"
    );
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

  // For a solid Disk/Ball an exterior point projects onto the boundary; an
  // interior point maps to itself.
  #[test]
  fn disk_projection_and_interior() {
    assert_eq!(
      interpret("RegionNearest[Disk[{0, 0}, 1], {3, 0}]").unwrap(),
      "{1, 0}"
    );
    assert_eq!(
      interpret("RegionNearest[Disk[{0, 0}, 2], {3, 4}]").unwrap(),
      "{6/5, 8/5}"
    );
    assert_eq!(
      interpret("RegionNearest[Disk[{0, 0}, 1], {0.5, 0}]").unwrap(),
      "{0.5, 0}"
    );
  }

  #[test]
  fn point_and_ball() {
    assert_eq!(
      interpret("RegionNearest[Point[{2, 3}], {5, 7}]").unwrap(),
      "{2, 3}"
    );
    assert_eq!(
      interpret("RegionNearest[Ball[{0, 0, 0}, 1], {0, 0, 3}]").unwrap(),
      "{0, 0, 1}"
    );
  }

  // Circle is the boundary, so even an interior point projects onto it.
  #[test]
  fn circle_always_projects() {
    assert_eq!(
      interpret("RegionNearest[Circle[{0, 0}, 1], {3, 4}]").unwrap(),
      "{3/5, 4/5}"
    );
  }

  #[test]
  fn rectangle_clamps() {
    assert_eq!(
      interpret("RegionNearest[Rectangle[{0, 0}, {2, 2}], {3, 1}]").unwrap(),
      "{2, 1}"
    );
    assert_eq!(
      interpret("RegionNearest[Rectangle[{0, 0}, {2, 2}], {3, 3}]").unwrap(),
      "{2, 2}"
    );
    // Already inside.
    assert_eq!(
      interpret("RegionNearest[Rectangle[{0, 0}, {2, 2}], {1, 1}]").unwrap(),
      "{1, 1}"
    );
  }

  // A Line is a segment (or polyline): the nearest point is the clamped
  // projection onto the closest segment, exact for exact inputs.
  #[test]
  fn line_projection() {
    assert_eq!(
      interpret("RegionNearest[Line[{{0, 0}, {2, 2}}], {2, 0}]").unwrap(),
      "{1, 1}"
    );
    // Projection lands inside the segment.
    assert_eq!(
      interpret("RegionNearest[Line[{{0, 0}, {4, 0}}], {2, 5}]").unwrap(),
      "{2, 0}"
    );
    // Beyond an endpoint clamps to that endpoint.
    assert_eq!(
      interpret("RegionNearest[Line[{{0, 0}, {4, 0}}], {-1, 3}]").unwrap(),
      "{0, 0}"
    );
    // Exact rational projection.
    assert_eq!(
      interpret("RegionNearest[Line[{{0, 0}, {3, 3}}], {3, 0}]").unwrap(),
      "{3/2, 3/2}"
    );
    // A 3D segment.
    assert_eq!(
      interpret("RegionNearest[Line[{{0, 0, 0}, {2, 2, 2}}], {2, 0, 0}]")
        .unwrap(),
      "{2/3, 2/3, 2/3}"
    );
  }

  // A multi-vertex polyline picks the closest segment.
  #[test]
  fn line_polyline() {
    assert_eq!(
      interpret("RegionNearest[Line[{{0, 0}, {2, 0}, {2, 2}}], {3, 1}]")
        .unwrap(),
      "{2, 1}"
    );
  }

  // Triangle/Polygon: an interior point maps to itself; an exterior point
  // projects onto the closest boundary edge (exact for exact inputs).
  #[test]
  fn triangle_and_polygon() {
    let t = "Triangle[{{0, 0}, {4, 0}, {0, 3}}]";
    // Inside → the point itself.
    assert_eq!(
      interpret(&format!("RegionNearest[{t}, {{1, 1}}]")).unwrap(),
      "{1, 1}"
    );
    // Outside → exact projection onto the hypotenuse.
    assert_eq!(
      interpret(&format!("RegionNearest[{t}, {{5, 5}}]")).unwrap(),
      "{56/25, 33/25}"
    );
    // Past a vertex → that vertex.
    assert_eq!(
      interpret(&format!("RegionNearest[{t}, {{-1, -1}}]")).unwrap(),
      "{0, 0}"
    );
    // A square polygon.
    assert_eq!(
      interpret("RegionNearest[Polygon[{{0,0},{2,0},{2,2},{0,2}}], {3, 1}]")
        .unwrap(),
      "{2, 1}"
    );
  }
}

mod region_distance_line {
  use super::*;

  // RegionDistance to a Line is the distance to its nearest point.
  #[test]
  fn segment_distance() {
    assert_eq!(
      interpret("RegionDistance[Line[{{0, 0}, {4, 0}}], {2, 5}]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("RegionDistance[Line[{{0, 0}, {2, 2}}], {2, 0}]").unwrap(),
      "Sqrt[2]"
    );
  }

  #[test]
  fn polyline_distance() {
    assert_eq!(
      interpret("RegionDistance[Line[{{0, 0}, {2, 0}, {2, 2}}], {3, 1}]")
        .unwrap(),
      "1"
    );
  }

  // RegionDistance to a Triangle/Polygon is 0 inside and the boundary
  // distance outside.
  #[test]
  fn triangle_distance() {
    let t = "Triangle[{{0, 0}, {4, 0}, {0, 3}}]";
    assert_eq!(
      interpret(&format!("RegionDistance[{t}, {{1, 1}}]")).unwrap(),
      "0"
    );
    assert_eq!(
      interpret(&format!("RegionDistance[{t}, {{5, 5}}]")).unwrap(),
      "23/5"
    );
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

  #[test]
  fn level_0_dim_2() {
    // 2D MeshRegion renders as -Graphics-, so inspect the structure.
    assert_eq!(interpret("Head[CantorMesh[0, 2]]").unwrap(), "MeshRegion");
    assert_eq!(
      interpret("CantorMesh[0, 2][[1]]").unwrap(),
      "{{0., 0.}, {0., 1.}, {1., 0.}, {1., 1.}}"
    );
    assert_eq!(
      interpret("CantorMesh[0, 2][[2]]").unwrap(),
      "{Polygon[{{1, 3, 4, 2}}]}"
    );
  }

  #[test]
  fn level_1_dim_2() {
    assert_eq!(interpret("Head[CantorMesh[1, 2]]").unwrap(), "MeshRegion");
    assert_eq!(interpret("Length[CantorMesh[1, 2][[1]]]").unwrap(), "16");
    assert_eq!(
      interpret("CantorMesh[1, 2][[2]]").unwrap(),
      "{Polygon[{{1, 5, 6, 2}, {3, 7, 8, 4}, {9, 13, 14, 10}, {11, 15, 16, 12}}]}"
    );
    // First and last vertices.
    assert_eq!(interpret("CantorMesh[1, 2][[1, 1]]").unwrap(), "{0., 0.}");
    assert_eq!(interpret("CantorMesh[1, 2][[1, -1]]").unwrap(), "{1., 1.}");
  }

  #[test]
  fn level_1_dim_1_via_two_arg() {
    // CantorMesh[n, 1] should match CantorMesh[n]
    assert_eq!(
      interpret("CantorMesh[1, 1]").unwrap(),
      "MeshRegion[{{0.}, {0.3333333333333333}, {0.6666666666666666}, {1.}}, {Line[{{1, 2}, {3, 4}}]}]"
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

mod polygon_angle {
  use super::*;

  // PolygonAngle[poly] gives the interior angle at each vertex.
  #[test]
  fn square_all_right_angles() {
    assert_eq!(
      interpret("PolygonAngle[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]")
        .unwrap(),
      "{Pi/2, Pi/2, Pi/2, Pi/2}"
    );
  }

  #[test]
  fn triangle_via_polygon_and_triangle_head() {
    assert_eq!(
      interpret("PolygonAngle[Polygon[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "{Pi/2, Pi/4, Pi/4}"
    );
    assert_eq!(
      interpret("PolygonAngle[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "{Pi/2, Pi/4, Pi/4}"
    );
  }

  #[test]
  fn scalene_right_triangle() {
    // 3-4-5 triangle: angles ArcCos[4/5], Pi/2, ArcCos[3/5].
    assert_eq!(
      interpret("PolygonAngle[Polygon[{{0, 0}, {4, 0}, {4, 3}}]]").unwrap(),
      "{ArcCos[4/5], Pi/2, ArcCos[3/5]}"
    );
  }

  // A reflex (concave) vertex has an interior angle greater than Pi.
  #[test]
  fn non_convex_has_reflex_angle() {
    assert_eq!(
      interpret(
        "PolygonAngle[Polygon[{{0, 0}, {2, 0}, {2, 2}, {1, 1}, {0, 2}}]]"
      )
      .unwrap(),
      "{Pi/2, Pi/2, Pi/4, (3*Pi)/2, Pi/4}"
    );
  }

  // The list form starts at the vertex with minimum x (ties: minimum y) and
  // then follows the polygon's cyclic order, matching wolframscript.
  #[test]
  fn list_starts_at_min_x_vertex() {
    // Min-x vertex is {-1, 2} (the last), so the list is rotated to start
    // there rather than at {0, 0}.
    assert_eq!(
      interpret(
        "PolygonAngle[Polygon[{{0, 0}, {2, 0}, {3, 2}, {1, 3}, {-1, 2}}]]"
      )
      .unwrap(),
      "{Pi/2, ArcCos[-(1/Sqrt[5])], ArcCos[-(1/Sqrt[5])], Pi/2, ArcCos[-3/5]}"
    );
  }

  // PolygonAngle[poly, vertex] gives the interior angle at one vertex.
  #[test]
  fn single_vertex_angle() {
    assert_eq!(
      interpret(
        "PolygonAngle[Polygon[{{0, 0}, {2, 0}, {2, 1}, {0, 1}}], {0, 0}]"
      )
      .unwrap(),
      "Pi/2"
    );
    assert_eq!(
      interpret("PolygonAngle[Polygon[{{0, 0}, {4, 0}, {4, 3}}], {4, 3}]")
        .unwrap(),
      "ArcCos[3/5]"
    );
  }

  // The interior angles of a simple n-gon sum to (n-2) Pi.
  #[test]
  fn angles_sum_to_n_minus_2_pi() {
    assert_eq!(
      interpret(
        "Total[PolygonAngle[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]]"
      )
      .unwrap(),
      "2*Pi"
    );
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

  #[test]
  fn one_d_with_gaps() {
    assert_eq!(
      interpret("ArrayMesh[{1, 0, 1, 1, 0, 1}]").unwrap(),
      "MeshRegion[{{0.}, {1.}, {2.}, {3.}, {4.}, {5.}, {6.}}, \
       {Line[{{1, 2}, {3, 4}, {4, 5}, {6, 7}}]}]"
    );
  }

  #[test]
  fn one_d_solid() {
    assert_eq!(
      interpret("ArrayMesh[{1, 1, 1}]").unwrap(),
      "MeshRegion[{{0.}, {1.}, {2.}, {3.}}, {Line[{{1, 2}, {2, 3}, {3, 4}}]}]"
    );
  }

  #[test]
  fn one_d_isolated_endpoints() {
    assert_eq!(
      interpret("ArrayMesh[{1, 0, 0, 1}]").unwrap(),
      "MeshRegion[{{0.}, {1.}, {3.}, {4.}}, {Line[{{1, 2}, {3, 4}}]}]"
    );
  }

  #[test]
  fn one_d_all_zeros() {
    assert_eq!(interpret("ArrayMesh[{0, 0, 0}]").unwrap(), "EmptyRegion[1]");
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

mod transformation_function_compose {
  use super::*;

  #[test]
  fn rotation_then_translation() {
    assert_eq!(
      interpret("RotationTransform[Pi] . TranslationTransform[{1, -1}]")
        .unwrap(),
      "TransformationFunction[{{-1, 0, -1}, {0, -1, 1}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn translation_then_rotation() {
    assert_eq!(
      interpret("TranslationTransform[{1, -1}] . RotationTransform[Pi]")
        .unwrap(),
      "TransformationFunction[{{-1, 0, 1}, {0, -1, -1}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn two_translations_add() {
    assert_eq!(
      interpret("TranslationTransform[{a, b}] . TranslationTransform[{c, d}]")
        .unwrap(),
      "TransformationFunction[{{1, 0, a + c}, {0, 1, b + d}, {0, 0, 1}}]"
    );
  }
}

mod region_member {
  use super::*;

  // RegionMember tests whether a point lies in a (closed) region.
  #[test]
  fn disk_interior_boundary_exterior() {
    assert_eq!(
      interpret("RegionMember[Disk[{0, 0}, 1], {0.5, 0.5}]").unwrap(),
      "True"
    );
    // The boundary is included (closed disk).
    assert_eq!(
      interpret("RegionMember[Disk[{0, 0}, 1], {1, 0}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("RegionMember[Disk[{0, 0}, 1], {2, 0}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn disk_defaults_and_offset_center() {
    // Disk[] is the unit disk at the origin.
    assert_eq!(
      interpret("RegionMember[Disk[], {0.3, 0.3}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("RegionMember[Disk[{1, 1}, 2], {2, 2}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn ball_three_dimensional() {
    assert_eq!(
      interpret("RegionMember[Ball[{0, 0, 0}, 1], {0.5, 0.5, 0.5}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn rectangle_inside_outside() {
    assert_eq!(
      interpret("RegionMember[Rectangle[{0, 0}, {2, 3}], {1, 1}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("RegionMember[Rectangle[{0, 0}, {2, 3}], {3, 1}]").unwrap(),
      "False"
    );
  }

  // Circle is the boundary curve, not the solid disk.
  #[test]
  fn circle_is_boundary() {
    assert_eq!(
      interpret("RegionMember[Circle[{0, 0}, 1], {1, 0}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("RegionMember[Circle[{0, 0}, 1], {0.5, 0.5}]").unwrap(),
      "False"
    );
  }

  // A Triangle/Polygon is a closed region: interior and boundary points
  // (edges and vertices) are members, exterior points are not.
  #[test]
  fn triangle_and_polygon() {
    let t = "Triangle[{{0, 0}, {4, 0}, {0, 3}}]";
    assert_eq!(
      interpret(&format!("RegionMember[{t}, {{1, 1}}]")).unwrap(),
      "True"
    );
    // On an edge and on a vertex.
    assert_eq!(
      interpret(&format!("RegionMember[{t}, {{2, 0}}]")).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(&format!("RegionMember[{t}, {{0, 0}}]")).unwrap(),
      "True"
    );
    // Outside.
    assert_eq!(
      interpret(&format!("RegionMember[{t}, {{3, 3}}]")).unwrap(),
      "False"
    );
    assert_eq!(
      interpret(&format!("RegionMember[{t}, {{-1, -1}}]")).unwrap(),
      "False"
    );
    // A square polygon.
    assert_eq!(
      interpret("RegionMember[Polygon[{{0,0},{2,0},{2,2},{0,2}}], {1, 1}]")
        .unwrap(),
      "True"
    );
    // A non-convex polygon: the concave notch excludes the point.
    assert_eq!(
      interpret(
        "RegionMember[Polygon[{{0,0},{4,0},{4,4},{2,1},{0,4}}], {2, 3}]"
      )
      .unwrap(),
      "False"
    );
  }
}

mod region_distance {
  use super::*;

  // RegionDistance is 0 inside a solid region and the boundary distance
  // outside; exact inputs give exact results.
  #[test]
  fn disk_solid() {
    assert_eq!(
      interpret("RegionDistance[Disk[{0, 0}, 1], {3, 0}]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("RegionDistance[Disk[{0, 0}, 2], {3, 4}]").unwrap(),
      "3"
    );
    // A point inside an exact region has distance 0; a machine point gives 0.
    assert_eq!(
      interpret("RegionDistance[Disk[{0, 0}, 1], {0.5, 0}]").unwrap(),
      "0."
    );
  }

  #[test]
  fn point_and_ball() {
    assert_eq!(
      interpret("RegionDistance[Point[{0, 0}], {3, 4}]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("RegionDistance[Ball[{0, 0, 0}, 1], {0, 0, 3}]").unwrap(),
      "2"
    );
  }

  // Circle is the boundary, so the distance to an interior point is nonzero.
  #[test]
  fn circle_boundary() {
    assert_eq!(
      interpret("RegionDistance[Circle[{0, 0}, 1], {0.5, 0}]").unwrap(),
      "0.5"
    );
    assert_eq!(
      interpret("RegionDistance[Circle[{0, 0}, 1], {3, 0}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn rectangle_edge_and_corner() {
    // Distance to the nearest edge.
    assert_eq!(
      interpret("RegionDistance[Rectangle[{0, 0}, {2, 2}], {3, 1}]").unwrap(),
      "1"
    );
    // Distance to a corner.
    assert_eq!(
      interpret("RegionDistance[Rectangle[{0, 0}, {2, 2}], {3, 3}]").unwrap(),
      "Sqrt[2]"
    );
    // Inside the box.
    assert_eq!(
      interpret("RegionDistance[Rectangle[{0, 0}, {2, 2}], {1, 1}]").unwrap(),
      "0"
    );
  }
}

mod signed_region_distance {
  use super::*;

  // Like RegionDistance but negative inside a solid region.
  #[test]
  fn disk_inside_negative_outside_positive() {
    assert_eq!(
      interpret("SignedRegionDistance[Disk[{0, 0}, 1], {3, 0}]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("SignedRegionDistance[Disk[{0, 0}, 1], {0.5, 0}]").unwrap(),
      "-0.5"
    );
    assert_eq!(
      interpret("SignedRegionDistance[Disk[{0, 0}, 1], {0, 0}]").unwrap(),
      "-1"
    );
    // The boundary is exactly zero.
    assert_eq!(
      interpret("SignedRegionDistance[Disk[{0, 0}, 1], {1, 0}]").unwrap(),
      "0"
    );
  }

  // Point/Circle have no interior, so the signed distance is the ordinary one.
  #[test]
  fn point_and_circle_nonnegative() {
    assert_eq!(
      interpret("SignedRegionDistance[Point[{0, 0}], {3, 4}]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("SignedRegionDistance[Circle[{0, 0}, 1], {0.5, 0}]").unwrap(),
      "0.5"
    );
  }

  // Axis-aligned box signed distance field.
  #[test]
  fn rectangle_signed() {
    // Inside: negative distance to the nearest edge.
    assert_eq!(
      interpret("SignedRegionDistance[Rectangle[{0, 0}, {4, 2}], {1, 1}]")
        .unwrap(),
      "-1"
    );
    // Outside, nearest a corner.
    assert_eq!(
      interpret("SignedRegionDistance[Rectangle[{0, 0}, {2, 2}], {3, 3}]")
        .unwrap(),
      "Sqrt[2]"
    );
    assert_eq!(
      interpret(
        "SignedRegionDistance[Cuboid[{0, 0, 0}, {2, 2, 2}], {1, 1, 1}]"
      )
      .unwrap(),
      "-1"
    );
  }

  // A Line is measure-zero, so its signed distance equals the ordinary one.
  #[test]
  fn line_is_unsigned() {
    assert_eq!(
      interpret("SignedRegionDistance[Line[{{0, 0}, {4, 0}}], {2, 5}]")
        .unwrap(),
      "5"
    );
  }

  // A solid Triangle/Polygon: positive outside, negative inside, zero on the
  // boundary.
  #[test]
  fn triangle_and_polygon() {
    let t = "Triangle[{{0, 0}, {4, 0}, {0, 3}}]";
    assert_eq!(
      interpret(&format!("SignedRegionDistance[{t}, {{5, 5}}]")).unwrap(),
      "23/5"
    );
    assert_eq!(
      interpret(&format!("SignedRegionDistance[{t}, {{1, 1}}]")).unwrap(),
      "-1"
    );
    // On the boundary.
    assert_eq!(
      interpret(&format!("SignedRegionDistance[{t}, {{2, 0}}]")).unwrap(),
      "0"
    );
    // Deep inside a square polygon: distance to the nearest edge, negated.
    assert_eq!(
      interpret(
        "SignedRegionDistance[Polygon[{{0,0},{4,0},{4,4},{0,4}}], {1, 2}]"
      )
      .unwrap(),
      "-1"
    );
  }
}

mod circumsphere {
  use super::*;

  // The unique sphere through d+1 points in d dimensions, as
  // Sphere[center, radius]. Exact rational inputs stay exact.
  #[test]
  fn triangle_2d() {
    assert_eq!(
      interpret("Circumsphere[{{0, 0}, {2, 0}, {0, 2}}]").unwrap(),
      "Sphere[{1, 1}, Sqrt[2]]"
    );
  }

  #[test]
  fn right_triangle_rational_center() {
    assert_eq!(
      interpret("Circumsphere[{{0, 0}, {4, 0}, {0, 3}}]").unwrap(),
      "Sphere[{2, 3/2}, 5/2]"
    );
  }

  #[test]
  fn tetrahedron_3d() {
    assert_eq!(
      interpret("Circumsphere[{{0,0,0},{2,0,0},{0,2,0},{0,0,2}}]").unwrap(),
      "Sphere[{1, 1, 1}, Sqrt[3]]"
    );
  }

  #[test]
  fn float_input() {
    assert_eq!(
      interpret("Circumsphere[{{0., 0.}, {2., 0.}, {0., 2.}}]").unwrap(),
      "Sphere[{1., 1.}, 1.4142135623730951]"
    );
  }

  // Wrong point count (need d+1) stays unevaluated, like wolframscript.
  #[test]
  fn wrong_point_count_unevaluated() {
    assert_eq!(
      interpret("Circumsphere[{{0, 0}, {4, 0}}]").unwrap(),
      "Circumsphere[{{0, 0}, {4, 0}}]"
    );
  }

  // Collinear points give a singular system and stay unevaluated.
  #[test]
  fn degenerate_collinear_unevaluated() {
    assert_eq!(
      interpret("Circumsphere[{{0, 0}, {1, 1}, {2, 2}}]").unwrap(),
      "Circumsphere[{{0, 0}, {1, 1}, {2, 2}}]"
    );
  }
}

// Insphere[{p1, …, p_{n+1}}] — the sphere inscribed in the simplex spanned by
// the points, given as Sphere[incenter, inradius]. Exact rational inputs stay
// exact. (The Triangle[…]/Tetrahedron[…] wrapper forms share the same code.)
mod insphere {
  use super::*;

  #[test]
  fn right_triangle_3_4_5() {
    // Incircle of the 3-4-5 triangle: incenter {1, 1}, inradius 1.
    assert_eq!(
      interpret("Insphere[{{0, 0}, {4, 0}, {0, 3}}]").unwrap(),
      "Sphere[{1, 1}, 1]"
    );
  }

  #[test]
  fn right_triangle_6_8_10() {
    assert_eq!(
      interpret("Insphere[{{0, 0}, {6, 0}, {0, 8}}]").unwrap(),
      "Sphere[{2, 2}, 2]"
    );
  }

  #[test]
  fn translated_triangle() {
    assert_eq!(
      interpret("Insphere[{{1, 1}, {5, 1}, {1, 4}}]").unwrap(),
      "Sphere[{2, 2}, 1]"
    );
  }

  #[test]
  fn matches_triangle_wrapper() {
    // The raw point-list form and the Triangle[…] wrapper agree.
    let raw = interpret("Insphere[{{0, 0}, {4, 0}, {0, 3}}]").unwrap();
    let wrapped =
      interpret("Insphere[Triangle[{{0, 0}, {4, 0}, {0, 3}}]]").unwrap();
    assert_eq!(raw, wrapped);
  }
}

// AngleBisector[{q1, p, q2}] — the interior-angle bisector at p, returned as
// InfiniteLine[p, Normalize[q1 - p] + Normalize[q2 - p]]. Verified against
// wolframscript.
mod angle_bisector {
  use super::*;

  #[test]
  fn right_angle_at_origin() {
    // Two axis-aligned unit legs bisect along {1, 1}.
    assert_eq!(
      interpret("AngleBisector[{{1, 0}, {0, 0}, {0, 1}}]").unwrap(),
      "InfiniteLine[{0, 0}, {1, 1}]"
    );
    // Direction is normalized per leg, so leg lengths don't matter.
    assert_eq!(
      interpret("AngleBisector[{{4, 0}, {0, 0}, {0, 3}}]").unwrap(),
      "InfiniteLine[{0, 0}, {1, 1}]"
    );
  }

  #[test]
  fn irrational_direction_is_simplified() {
    // 60° leg: bisector direction {3/2, Sqrt[3]/2}.
    assert_eq!(
      interpret("AngleBisector[{{1, 0}, {0, 0}, {1, Sqrt[3]}}]").unwrap(),
      "InfiniteLine[{0, 0}, {3/2, Sqrt[3]/2}]"
    );
    // 1/Sqrt[2] + 1/Sqrt[2] must collapse to Sqrt[2], not 2/Sqrt[2].
    assert_eq!(
      interpret("AngleBisector[{{2, 0}, {1, 1}, {2, 2}}]").unwrap(),
      "InfiniteLine[{1, 1}, {Sqrt[2], 0}]"
    );
  }

  #[test]
  fn non_origin_vertex() {
    assert_eq!(
      interpret("AngleBisector[{{5, 5}, {0, 0}, {5, -5}}]").unwrap(),
      "InfiniteLine[{0, 0}, {Sqrt[2], 0}]"
    );
  }

  #[test]
  fn non_2d_or_malformed_stays_unevaluated() {
    // 3-D points are not handled (matches wolframscript).
    assert_eq!(
      interpret("AngleBisector[{{1, 0, 0}, {0, 0, 0}, {0, 1, 0}}]").unwrap(),
      "AngleBisector[{{1, 0, 0}, {0, 0, 0}, {0, 1, 0}}]"
    );
    // A two-point list is not the {q1, p, q2} form.
    assert_eq!(
      interpret("AngleBisector[{{1, 0}, {0, 0}}]").unwrap(),
      "AngleBisector[{{1, 0}, {0, 0}}]"
    );
  }
}

// PerpendicularBisector[{p1, p2}] — the segment's perpendicular bisector as
// InfiniteLine[midpoint, {dy, -dx}] with {dx, dy} = p2 - p1. Verified against
// wolframscript.
mod perpendicular_bisector {
  use super::*;

  #[test]
  fn horizontal_and_vertical_segments() {
    assert_eq!(
      interpret("PerpendicularBisector[{{0, 0}, {2, 0}}]").unwrap(),
      "InfiniteLine[{1, 0}, {0, -2}]"
    );
    // A rational midpoint stays exact.
    assert_eq!(
      interpret("PerpendicularBisector[{{0, 0}, {0, 1}}]").unwrap(),
      "InfiniteLine[{0, 1/2}, {1, 0}]"
    );
  }

  #[test]
  fn oblique_segments() {
    assert_eq!(
      interpret("PerpendicularBisector[{{0, 0}, {4, 2}}]").unwrap(),
      "InfiniteLine[{2, 1}, {2, -4}]"
    );
    assert_eq!(
      interpret("PerpendicularBisector[{{1, 1}, {3, 5}}]").unwrap(),
      "InfiniteLine[{2, 3}, {4, -2}]"
    );
  }

  #[test]
  fn line_wrapper_form() {
    assert_eq!(
      interpret("PerpendicularBisector[Line[{{-1, -1}, {1, 1}}]]").unwrap(),
      "InfiniteLine[{0, 0}, {2, -2}]"
    );
  }

  #[test]
  fn non_2d_or_malformed_stays_unevaluated() {
    assert_eq!(
      interpret("PerpendicularBisector[{{0, 0, 0}, {2, 0, 0}}]").unwrap(),
      "PerpendicularBisector[{{0, 0, 0}, {2, 0, 0}}]"
    );
    assert_eq!(
      interpret("PerpendicularBisector[{{2, 3}}]").unwrap(),
      "PerpendicularBisector[{{2, 3}}]"
    );
  }
}

// BoundingRegion[pts] — the smallest axis-aligned box: Rectangle for 2D points,
// Cuboid for 1D or >=3D. Min/Max are exact and stay symbolic when needed.
mod bounding_region {
  use super::*;

  #[test]
  fn points_2d_rectangle() {
    assert_eq!(
      interpret("BoundingRegion[{{0, 0}, {1, 1}}]").unwrap(),
      "Rectangle[{0, 0}, {1, 1}]"
    );
    assert_eq!(
      interpret("BoundingRegion[{{0, 0}, {2, 3}, {1, -1}}]").unwrap(),
      "Rectangle[{0, -1}, {2, 3}]"
    );
  }

  #[test]
  fn points_3d_and_4d_cuboid() {
    assert_eq!(
      interpret("BoundingRegion[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "Cuboid[{1, 2, 3}, {4, 5, 6}]"
    );
    assert_eq!(
      interpret("BoundingRegion[{{1, 2, 3, 4}, {5, 6, 7, 8}}]").unwrap(),
      "Cuboid[{1, 2, 3, 4}, {5, 6, 7, 8}]"
    );
  }

  // 1D points give a Cuboid (not a Rectangle), matching wolframscript.
  #[test]
  fn points_1d_cuboid() {
    assert_eq!(
      interpret("BoundingRegion[{{1}, {5}, {3}}]").unwrap(),
      "Cuboid[{1}, {5}]"
    );
  }

  #[test]
  fn rational_and_real_preserved() {
    assert_eq!(
      interpret("BoundingRegion[{{1/2, 3}, {2, 1/4}}]").unwrap(),
      "Rectangle[{1/2, 1/4}, {2, 3}]"
    );
    assert_eq!(
      interpret("BoundingRegion[{{1.5, 2.5}, {3.5, 0.5}}]").unwrap(),
      "Rectangle[{1.5, 0.5}, {3.5, 2.5}]"
    );
  }

  // A single point gives a degenerate box with min == max.
  #[test]
  fn single_point() {
    assert_eq!(
      interpret("BoundingRegion[{{0, 0}}]").unwrap(),
      "Rectangle[{0, 0}, {0, 0}]"
    );
  }

  // Symbolic coordinates stay as Min[…]/Max[…].
  #[test]
  fn symbolic_coordinates() {
    assert_eq!(
      interpret("BoundingRegion[{{a, b}, {c, d}}]").unwrap(),
      "Rectangle[{Min[a, c], Min[b, d]}, {Max[a, c], Max[b, d]}]"
    );
  }

  // Structurally-invalid input stays unevaluated (with a regl message).
  #[test]
  fn malformed_unevaluated() {
    assert_eq!(
      interpret("BoundingRegion[{1, 2, 3}]").unwrap(),
      "BoundingRegion[{1, 2, 3}]"
    );
    assert_eq!(
      interpret("BoundingRegion[{{1, 2}, {3, 4, 5}}]").unwrap(),
      "BoundingRegion[{{1, 2}, {3, 4, 5}}]"
    );
  }
}

mod find_shortest_curve {
  use super::*;

  // The shortest curve between two points on a circle is the shorter arc,
  // returned as Circle[c, r, {θ1, θ2}] with exact angles.
  #[test]
  fn circle_exact_arc() {
    assert_eq!(
      interpret("FindShortestCurve[Circle[], {1, 0}, {0, 1}]").unwrap(),
      "Circle[{0, 0}, 1, {0, Pi/2}]"
    );
    assert_eq!(
      interpret("FindShortestCurve[Circle[], {-1, 0}, {0, 1}]").unwrap(),
      "Circle[{0, 0}, 1, {Pi/2, Pi}]"
    );
  }

  // When the shorter arc crosses the ±π branch cut of ArcTan, the spec
  // continues past π instead of taking the long way around.
  #[test]
  fn circle_arc_across_branch_cut() {
    assert_eq!(
      interpret(
        "FindShortestCurve[Circle[], {-Sqrt[2]/2, Sqrt[2]/2}, \
         {-Sqrt[2]/2, -Sqrt[2]/2}]"
      )
      .unwrap(),
      "Circle[{0, 0}, 1, {(3*Pi)/4, (5*Pi)/4}]"
    );
  }

  #[test]
  fn circle_translated_and_scaled() {
    assert_eq!(
      interpret("FindShortestCurve[Circle[{1, 2}, 3], {4, 2}, {1, 5}]")
        .unwrap(),
      "Circle[{1, 2}, 3, {0, Pi/2}]"
    );
  }

  // Antipodal points: both arcs are geodesics; the spec starts at the
  // smaller angle.
  #[test]
  fn circle_antipodal_points() {
    assert_eq!(
      interpret("FindShortestCurve[Circle[], {1, 0}, {-1, 0}]").unwrap(),
      "Circle[{0, 0}, 1, {0, Pi}]"
    );
  }

  // Machine-precision coordinates give machine-precision angles.
  #[test]
  fn circle_machine_precision() {
    assert_eq!(
      interpret("FindShortestCurve[Circle[], {1, 0}, {0.6, 0.8}]").unwrap(),
      "Circle[{0, 0}, 1, {0, 0.9272952180016123}]"
    );
  }

  // A point that is not on the circle leaves the call unevaluated.
  #[test]
  fn circle_point_off_region_unevaluated() {
    assert_eq!(
      interpret("FindShortestCurve[Circle[], {2, 0}, {0, 1}]").unwrap(),
      "FindShortestCurve[Circle[{0, 0}], {2, 0}, {0, 1}]"
    );
  }

  // ArcLength closes over the returned arc.
  #[test]
  fn arc_length_of_result() {
    assert_eq!(
      interpret("ArcLength[FindShortestCurve[Circle[], {1, 0}, {0, 1}]]")
        .unwrap(),
      "Pi/2"
    );
  }

  // In convex solids the geodesic is the straight segment, kept exact.
  #[test]
  fn convex_solids_straight_segment() {
    assert_eq!(
      interpret("FindShortestCurve[Disk[], {1/10, 4/5}, {-1/2, 0}]").unwrap(),
      "Line[{{1/10, 4/5}, {-1/2, 0}}]"
    );
    assert_eq!(
      interpret(
        "FindShortestCurve[Triangle[{{0, 0}, {4, 0}, {0, 3}}], {1, 1}, {2, 0}]"
      )
      .unwrap(),
      "Line[{{1, 1}, {2, 0}}]"
    );
    assert_eq!(
      interpret("FindShortestCurve[Cuboid[], {0, 0, 0}, {1, 1, 1}]").unwrap(),
      "Line[{{0, 0, 0}, {1, 1, 1}}]"
    );
  }

  #[test]
  fn convex_solid_point_outside_unevaluated() {
    assert_eq!(
      interpret("FindShortestCurve[Disk[], {2, 0}, {0, 0}]").unwrap(),
      "FindShortestCurve[Disk[{0, 0}], {2, 0}, {0, 0}]"
    );
  }

  // Curve regions are treated as meshes: the sub-path along the polyline,
  // at machine precision, starting at the first query point.
  #[test]
  fn polyline_path() {
    assert_eq!(
      interpret(
        "FindShortestCurve[Line[{{1, 0}, {2, 1}, {3, 0}, {4, 1}}], \
         {1, 0}, {3, 0}]"
      )
      .unwrap(),
      "Line[{{1., 0.}, {2., 1.}, {3., 0.}}]"
    );
    assert_eq!(
      interpret(
        "FindShortestCurve[Line[{{1, 0}, {2, 1}, {3, 0}, {4, 1}}], \
         {3, 0}, {1, 0}]"
      )
      .unwrap(),
      "Line[{{3., 0.}, {2., 1.}, {1., 0.}}]"
    );
  }

  // Points in the middle of segments work too.
  #[test]
  fn polyline_mid_segment_points() {
    assert_eq!(
      interpret("FindShortestCurve[Line[{{0, 0}, {4, 0}}], {1, 0}, {3, 0}]")
        .unwrap(),
      "Line[{{1., 0.}, {3., 0.}}]"
    );
  }

  // On a closed chain the shorter way around is taken — here through the
  // shared first/last vertex rather than the three other corners.
  #[test]
  fn closed_polyline_takes_short_way() {
    assert_eq!(
      interpret(
        "FindShortestCurve[Line[{{0, 0}, {2, 0}, {2, 2}, {0, 2}, {0, 0}}], \
         {1, 0}, {0, 1}]"
      )
      .unwrap(),
      "Line[{{1., 0.}, {0., 0.}, {0., 1.}}]"
    );
  }

  // One-dimensional mesh-style chains match wolframscript's Line[{{0.}, {1.}}].
  #[test]
  fn one_dimensional_chain() {
    assert_eq!(
      interpret("FindShortestCurve[Line[{{0}, {1}}], {0}, {1}]").unwrap(),
      "Line[{{0.}, {1.}}]"
    );
  }

  #[test]
  fn point_not_on_polyline_unevaluated() {
    assert_eq!(
      interpret("FindShortestCurve[Line[{{0, 0}, {1, 0}}], {5, 5}, {0, 0}]")
        .unwrap(),
      "FindShortestCurve[Line[{{0, 0}, {1, 0}}], {5, 5}, {0, 0}]"
    );
  }

  // Unsupported regions stay unevaluated.
  #[test]
  fn unsupported_region_unevaluated() {
    assert_eq!(
      interpret("FindShortestCurve[Annulus[], {1, 0}, {-0.8, 0.4}]").unwrap(),
      "FindShortestCurve[Annulus[], {1, 0}, {-0.8, 0.4}]"
    );
  }
}

mod shortest_curve_distance {
  use super::*;

  // Geodesic distance on a circle: r times the central angle, exact.
  #[test]
  fn circle_exact() {
    assert_eq!(
      interpret("ShortestCurveDistance[Circle[], {1, 0}, {0, 1}]").unwrap(),
      "Pi/2"
    );
    assert_eq!(
      interpret("ShortestCurveDistance[Circle[{0, 0}, 2], {2, 0}, {-2, 0}]")
        .unwrap(),
      "2*Pi"
    );
  }

  // Consistent with ArcLength[FindShortestCurve[…]] (Properties & Relations
  // example from the reference page).
  #[test]
  fn equals_arc_length_of_shortest_curve() {
    assert_eq!(
      interpret(
        "ArcLength[FindShortestCurve[Circle[], {-1, 0}, {0, 1}]] == \
         ShortestCurveDistance[Circle[], {-1, 0}, {0, 1}]"
      )
      .unwrap(),
      "True"
    );
  }

  // Great-circle distance on a sphere stays symbolic for symbolic points:
  // ShortestCurveDistance[Sphere[], {1, 0, 0}, {x, y, z}] is ArcCos[x].
  #[test]
  fn sphere_symbolic() {
    assert_eq!(
      interpret("ShortestCurveDistance[Sphere[], {1, 0, 0}, {x, y, z}]")
        .unwrap(),
      "ArcCos[x]"
    );
  }

  #[test]
  fn sphere_exact() {
    assert_eq!(
      interpret("ShortestCurveDistance[Sphere[], {0, 0, 1}, {0, 1, 0}]")
        .unwrap(),
      "Pi/2"
    );
    // Radius scales the distance: 2 ArcCos[0] = Pi.
    assert_eq!(
      interpret(
        "ShortestCurveDistance[Sphere[{0, 0, 0}, 2], {0, 0, 2}, {0, 2, 0}]"
      )
      .unwrap(),
      "Pi"
    );
  }

  // Convex solids: the Euclidean distance, without Abs for symbolic
  // coordinates (matching wolframscript's Sqrt[x^2 + (-1 + y)^2]).
  #[test]
  fn disk_symbolic() {
    assert_eq!(
      interpret("ShortestCurveDistance[Disk[], {0, 1}, {x, y}]").unwrap(),
      "Sqrt[x^2 + (-1 + y)^2]"
    );
  }

  #[test]
  fn convex_solids_numeric_and_exact() {
    assert_eq!(
      interpret(
        "ShortestCurveDistance[Ball[], {-0.2, 0.1, 0.3}, {-0.8, 0.19, -0.01}]"
      )
      .unwrap(),
      "0.6813222438758331"
    );
    assert_eq!(
      interpret(
        "ShortestCurveDistance[Rectangle[{0, 0}, {4, 3}], {1, 1}, {3, 2}]"
      )
      .unwrap(),
      "Sqrt[5]"
    );
    assert_eq!(
      interpret("ShortestCurveDistance[Cuboid[], {0, 0, 0}, {1, 1, 1}]")
        .unwrap(),
      "Sqrt[3]"
    );
  }

  // Curve regions: the machine-precision length along the polyline.
  #[test]
  fn polyline_length() {
    assert_eq!(
      interpret(
        "ShortestCurveDistance[Line[{{1, 0}, {2, 1}, {3, 0}, {4, 1}}], \
         {1, 0}, {3, 0}]"
      )
      .unwrap(),
      "2.8284271247461903"
    );
    assert_eq!(
      interpret("ShortestCurveDistance[Line[{{0}, {1}}], {0}, {1}]").unwrap(),
      "1."
    );
  }

  // Off-region points leave the call unevaluated.
  #[test]
  fn off_region_unevaluated() {
    assert_eq!(
      interpret("ShortestCurveDistance[Circle[], {2, 0}, {0, 1}]").unwrap(),
      "ShortestCurveDistance[Circle[{0, 0}], {2, 0}, {0, 1}]"
    );
    assert_eq!(
      interpret("ShortestCurveDistance[Disk[], {5, 5}, {0, 0}]").unwrap(),
      "ShortestCurveDistance[Disk[{0, 0}], {5, 5}, {0, 0}]"
    );
  }
}

mod arc_length_circular_arc {
  use super::*;

  // Circle[c, r, {θ1, θ2}] is a circular arc of length r (θ2 - θ1).
  #[test]
  fn quarter_arc() {
    assert_eq!(
      interpret("ArcLength[Circle[{0, 0}, 1, {0, Pi/2}]]").unwrap(),
      "Pi/2"
    );
  }

  #[test]
  fn scaled_arc() {
    assert_eq!(
      interpret("ArcLength[Circle[{0, 0}, 3, {Pi/4, Pi}]]").unwrap(),
      "(9*Pi)/4"
    );
  }
}

// Region[reg] — displays a geometric region as a plot (2D/3D graphics).
// Unsupported heads and symbolic coordinates leave the call unevaluated.
mod region {
  use super::*;

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Region[Disk[]]]").unwrap(), "Region");
  }

  #[test]
  fn areas_render_as_2d_graphics() {
    for reg in [
      "Disk[]",
      "Disk[{1, 1}]",
      "Disk[{0, 0}, 2]",
      "Disk[{0, 0}, {2, 1}]",
      "Disk[{0, 0}, 1, {0, Pi/2}]",
      "Rectangle[]",
      "Rectangle[{1, 2}]",
      "Rectangle[{0, 0}, {2, 1}]",
      "Triangle[]",
      "Triangle[{{0, 0}, {2, 0}, {1, 2}}]",
      "Polygon[{{0, 0}, {2, 0}, {1, 2}}]",
      "RegularPolygon[5]",
      "RegularPolygon[{1, 1}, 2, 6]",
      "Ball[{0, 0}, 2]",
    ] {
      assert_eq!(
        interpret(&format!("Region[{reg}]")).unwrap(),
        "-Graphics-",
        "Region[{reg}]"
      );
    }
  }

  #[test]
  fn curves_and_points_render_as_2d_graphics() {
    for reg in [
      "Circle[]",
      "Circle[{0, 0}, 2]",
      "Circle[{0, 0}, {2, 1}]",
      "Sphere[{1, 2}, 3]",
      "Line[{{0, 0}, {1, 1}, {2, 0}}]",
      "Line[{{{0, 0}, {1, 1}}, {{2, 0}, {3, 1}}}]",
      "Point[{1, 2}]",
      "Point[{{0, 0}, {1, 1}}]",
    ] {
      assert_eq!(
        interpret(&format!("Region[{reg}]")).unwrap(),
        "-Graphics-",
        "Region[{reg}]"
      );
    }
  }

  #[test]
  fn solids_render_as_3d_graphics() {
    for reg in [
      "Ball[]",
      "Ball[{0, 0, 0}, 2]",
      "Sphere[]",
      "Sphere[{1, 2, 3}, 2]",
      "Cuboid[]",
      "Cuboid[{0, 0, 0}, {2, 1, 1}]",
      "Cylinder[]",
      "Cylinder[{{0, 0, 0}, {0, 0, 2}}, 1]",
      "Cone[]",
      "Line[{{0, 0, 0}, {1, 1, 1}}]",
      "Point[{1, 2, 3}]",
      "Polygon[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]",
      "Triangle[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]",
    ] {
      assert_eq!(
        interpret(&format!("Region[{reg}]")).unwrap(),
        "-Graphics3D-",
        "Region[{reg}]"
      );
    }
  }

  #[test]
  fn accepts_graphics_options() {
    assert_eq!(
      interpret("Region[Disk[], ImageSize -> 200]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn export_string_svg() {
    let svg = interpret("ExportString[Region[Disk[]], \"SVG\"]").unwrap();
    assert!(svg.starts_with("<svg"), "Got: {svg}");
    assert!(svg.contains("ellipse"), "Got: {svg}");
  }

  // Region[Style[reg, directives…]] renders the region with the given
  // directives overriding the default region color.
  #[test]
  fn styled_regions_render() {
    for reg in [
      "Style[Disk[], Green]",
      "Style[Disk[], Red, Opacity[0.5]]",
      "Style[Circle[], Orange]",
      "Style[Rectangle[], Purple]",
      "Style[Triangle[], Brown]",
    ] {
      assert_eq!(
        interpret(&format!("Region[{reg}]")).unwrap(),
        "-Graphics-",
        "Region[{reg}]"
      );
    }
    assert_eq!(
      interpret("Region[Style[Ball[], Yellow]]").unwrap(),
      "-Graphics3D-"
    );
  }

  #[test]
  fn styled_region_applies_color() {
    let svg =
      interpret("ExportString[Region[Style[Disk[], Green]], \"SVG\"]").unwrap();
    assert!(
      svg.contains("rgb(0,255,0)"),
      "Style color should override the default region color, got: {svg}"
    );
  }

  // Opacity alone keeps the default region fill color.
  #[test]
  fn styled_region_without_color_keeps_default() {
    let svg =
      interpret("ExportString[Region[Style[Disk[], Opacity[0.5]]], \"SVG\"]")
        .unwrap();
    assert!(
      svg.contains("opacity"),
      "Opacity directive should apply, got: {svg}"
    );
    assert!(
      svg.contains("rgb(160,213,234)"),
      "Default region color should remain, got: {svg}"
    );
  }

  // A Style wrapping an undrawable region stays unevaluated (Style is
  // display-stripped in the textual echo, like elsewhere).
  #[test]
  fn styled_invalid_region_unevaluated() {
    assert_eq!(
      interpret("Region[Style[Disk[{a, b}], Green]]").unwrap(),
      "Region[Disk[{a, b}]]"
    );
  }

  // The SVG must be captured for graphical front ends (playground,
  // JupyterLite, Woxi Studio all read InterpretResult::graphics).
  #[test]
  fn graphics_captured_for_frontends() {
    let result = woxi::interpret_with_stdout("Region[Disk[]]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.expect("no captured graphics");
    assert!(svg.starts_with("<svg"), "Got: {svg}");
  }

  // Symbolic coordinates or non-region arguments stay unevaluated.
  #[test]
  fn symbolic_or_invalid_unevaluated() {
    assert_eq!(
      interpret("Region[Disk[{a, b}]]").unwrap(),
      "Region[Disk[{a, b}]]"
    );
    assert_eq!(interpret("Region[5]").unwrap(), "Region[5]");
    assert_eq!(interpret("Region[{1, 2}]").unwrap(), "Region[{1, 2}]");
    assert_eq!(interpret("Region[foo]").unwrap(), "Region[foo]");
  }

  // The renderer draws full circles only, so an arc spec must not be
  // silently drawn as a full circle.
  #[test]
  fn circle_arc_unevaluated() {
    assert_eq!(
      interpret("Region[Circle[{0, 0}, 1, {0, Pi/2}]]").unwrap(),
      "Region[Circle[{0, 0}, 1, {0, Pi/2}]]"
    );
  }

  // Region functions unwrap the Region display wrapper.
  #[test]
  fn region_functions_unwrap_wrapper() {
    assert_eq!(interpret("RegionMeasure[Region[Disk[]]]").unwrap(), "Pi");
    assert_eq!(
      interpret("Area[Region[Rectangle[{0, 0}, {2, 3}]]]").unwrap(),
      "6"
    );
    assert_eq!(interpret("RegionDimension[Region[Disk[]]]").unwrap(), "2");
    assert_eq!(
      interpret("RegionEmbeddingDimension[Region[Ball[]]]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("RegionMember[Region[Disk[]], {0, 0}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("RegionCentroid[Region[Disk[{1, 2}]]]").unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret("RegionBounds[Region[Disk[]]]").unwrap(),
      "{{-1, 1}, {-1, 1}}"
    );
    assert_eq!(
      interpret("RegionDistance[Region[Disk[]], {2, 0}]").unwrap(),
      "1"
    );
    assert_eq!(interpret("ArcLength[Region[Circle[]]]").unwrap(), "2*Pi");
    assert_eq!(interpret("Volume[Region[Ball[]]]").unwrap(), "(4*Pi)/3");
    assert_eq!(
      interpret(
        "RegionWithin[Region[Disk[{0, 0}, 2]], Region[Disk[{0, 0}, 1]]]"
      )
      .unwrap(),
      "True"
    );
  }
}

// Graphics-object regions from the GraphicsObjects guide: Torus, FilledTorus,
// Parallelogram, HalfPlane, InfinitePlane.
mod graphics_object_regions {
  use super::*;

  #[test]
  fn torus_stays_symbolic() {
    assert_eq!(interpret("Torus[]").unwrap(), "Torus[]");
    assert_eq!(
      interpret("FilledTorus[{0, 0, 0}, {1, 2}]").unwrap(),
      "FilledTorus[{0, 0, 0}, {1, 2}]"
    );
  }

  // Torus[{x,y,z}, {r1, r2}] is a surface: area = Pi^2 (r2^2 - r1^2).
  // The default is Torus[{0, 0, 0}, {1/2, 1}].
  #[test]
  fn torus_region_functions() {
    assert_eq!(interpret("RegionMeasure[Torus[]]").unwrap(), "(3*Pi^2)/4");
    assert_eq!(
      interpret("RegionMeasure[Torus[{0, 0, 0}, {1, 3}]]").unwrap(),
      "8*Pi^2"
    );
    assert_eq!(interpret("RegionDimension[Torus[]]").unwrap(), "2");
    assert_eq!(interpret("RegionEmbeddingDimension[Torus[]]").unwrap(), "3");
    assert_eq!(
      interpret("RegionCentroid[Torus[{1, 2, 3}, {1, 2}]]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(interpret("BoundedRegionQ[Torus[]]").unwrap(), "True");
    assert_eq!(
      interpret("RegionBounds[Torus[]]").unwrap(),
      "{{-1, 1}, {-1, 1}, {-1/4, 1/4}}"
    );
  }

  // FilledTorus is the solid: volume = Pi^2 (r2 - r1)^2 (r1 + r2) / 4.
  #[test]
  fn filled_torus_region_functions() {
    assert_eq!(
      interpret("RegionMeasure[FilledTorus[]]").unwrap(),
      "(3*Pi^2)/32"
    );
    assert_eq!(
      interpret("RegionMeasure[FilledTorus[{0, 0, 0}, {1, 3}]]").unwrap(),
      "4*Pi^2"
    );
    assert_eq!(interpret("RegionDimension[FilledTorus[]]").unwrap(), "3");
    assert_eq!(
      interpret("RegionCentroid[FilledTorus[]]").unwrap(),
      "{0, 0, 0}"
    );
    assert_eq!(interpret("BoundedRegionQ[FilledTorus[]]").unwrap(), "True");
  }

  #[test]
  fn parallelogram_region_functions() {
    // Area is |Det[{v1, v2}]| in the plane…
    assert_eq!(
      interpret("RegionMeasure[Parallelogram[{0, 0}, {{2, 0}, {1, 3}}]]")
        .unwrap(),
      "6"
    );
    // …and Sqrt of the Gram determinant in higher-dimensional space.
    assert_eq!(
      interpret(
        "RegionMeasure[Parallelogram[{0, 0, 0}, {{1, 0, 0}, {0, 2, 0}}]]"
      )
      .unwrap(),
      "2"
    );
    // Parallelogram[] is the unit square.
    assert_eq!(interpret("RegionMeasure[Parallelogram[]]").unwrap(), "1");
    assert_eq!(interpret("RegionDimension[Parallelogram[]]").unwrap(), "2");
    assert_eq!(
      interpret("RegionEmbeddingDimension[Parallelogram[]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("RegionCentroid[Parallelogram[{0, 0}, {{2, 0}, {1, 3}}]]")
        .unwrap(),
      "{3/2, 3/2}"
    );
    assert_eq!(
      interpret("RegionBounds[Parallelogram[{0, 0}, {{2, 0}, {1, 3}}]]")
        .unwrap(),
      "{{0, 3}, {0, 3}}"
    );
    assert_eq!(
      interpret("BoundedRegionQ[Parallelogram[]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn half_plane_region_functions() {
    assert_eq!(
      interpret("RegionMeasure[HalfPlane[{{0, 0}, {1, 0}}, {0, 1}]]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("RegionDimension[HalfPlane[{{0, 0}, {1, 0}}, {0, 1}]]")
        .unwrap(),
      "2"
    );
    assert_eq!(
      interpret(
        "RegionEmbeddingDimension[HalfPlane[{{0, 0}, {1, 0}}, {0, 1}]]"
      )
      .unwrap(),
      "2"
    );
    assert_eq!(
      interpret("BoundedRegionQ[HalfPlane[{{0, 0}, {1, 0}}, {0, 1}]]").unwrap(),
      "False"
    );
    // HalfPlane[p, v, w] form stays symbolic.
    assert_eq!(
      interpret("HalfPlane[{0, 0}, {1, 0}, {0, 1}]").unwrap(),
      "HalfPlane[{0, 0}, {1, 0}, {0, 1}]"
    );
  }

  #[test]
  fn infinite_plane_region_functions() {
    assert_eq!(
      interpret(
        "RegionMeasure[InfinitePlane[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]]"
      )
      .unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret(
        "RegionDimension[InfinitePlane[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]]"
      )
      .unwrap(),
      "2"
    );
    assert_eq!(
      interpret(
        "RegionEmbeddingDimension[InfinitePlane[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret(
        "RegionEmbeddingDimension[InfinitePlane[{0, 0}, {{1, 0}, {0, 1}}]]"
      )
      .unwrap(),
      "2"
    );
    assert_eq!(
      interpret(
        "BoundedRegionQ[InfinitePlane[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]]"
      )
      .unwrap(),
      "False"
    );
  }

  // Unbounded curves also have infinite measure.
  #[test]
  fn unbounded_curve_measures() {
    assert_eq!(
      interpret("RegionMeasure[InfiniteLine[{{0, 0}, {1, 1}}]]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("RegionMeasure[HalfLine[{{0, 0}, {1, 1}}]]").unwrap(),
      "Infinity"
    );
  }

  // JoinedCurve / BSplineSurface / Raster3D / AxisObject stay symbolic
  // (they are consumed by the Graphics/Graphics3D renderers).
  #[test]
  fn graphics_primitives_stay_symbolic() {
    assert_eq!(
      interpret("JoinedCurve[{Line[{{0, 0}, {1, 1}}]}]").unwrap(),
      "JoinedCurve[{Line[{{0, 0}, {1, 1}}]}]"
    );
    assert_eq!(
      interpret(
        "BSplineSurface[{{{0, 0, 0}, {1, 0, 1}}, {{0, 1, 1}, {1, 1, 0}}}]"
      )
      .unwrap(),
      "BSplineSurface[{{{0, 0, 0}, {1, 0, 1}}, {{0, 1, 1}, {1, 1, 0}}}]"
    );
    assert_eq!(
      interpret("Raster3D[{{{0, 1}, {1, 0}}, {{1, 0}, {0, 1}}}]").unwrap(),
      "Raster3D[{{{0, 1}, {1, 0}}, {{1, 0}, {0, 1}}}]"
    );
    assert_eq!(
      interpret("AxisObject[Line[{{0, 0}, {1, 0}}]]").unwrap(),
      "AxisObject[Line[{{0, 0}, {1, 0}}]]"
    );
  }
}

mod triangle_constructor_tests {
  use woxi::interpret;

  // Vertex A sits at the origin with side c along the x axis, exactly as
  // wolframscript places these.
  #[test]
  fn sas_triangle() {
    assert_eq!(
      interpret("SASTriangle[3, Pi/2, 4]").unwrap(),
      "Triangle[{{0, 0}, {5, 0}, {16/5, 12/5}}]"
    );
    assert_eq!(
      interpret("SASTriangle[1, Pi/3, 1]").unwrap(),
      "Triangle[{{0, 0}, {1, 0}, {1/2, Sqrt[3]/2}}]"
    );
    assert_eq!(
      interpret("SASTriangle[2, Pi/3, 3]").unwrap(),
      "Triangle[{{0, 0}, {Sqrt[7], 0}, {6/Sqrt[7], 3*Sqrt[3/7]}}]"
    );
    // A plain-integer (radian) angle stays symbolic in Cos/Sin form.
    assert_eq!(
      interpret("SASTriangle[3, 3, 4]").unwrap(),
      "Triangle[{{0, 0}, {Sqrt[25 - 24*Cos[3]], 0}, {(16 - 12*Cos[3])/Sqrt[25 - 24*Cos[3]], (12*Sin[3])/Sqrt[25 - 24*Cos[3]]}}]"
    );
    // Symbolic side lengths compute through.
    assert_eq!(
      interpret("SASTriangle[x, Pi/2, 4]").unwrap(),
      "Triangle[{{0, 0}, {Sqrt[16 + x^2], 0}, {16/Sqrt[16 + x^2], (4*x)/Sqrt[16 + x^2]}}]"
    );
  }

  #[test]
  fn asa_triangle() {
    assert_eq!(
      interpret("ASATriangle[Pi/4, 5, Pi/4]").unwrap(),
      "Triangle[{{0, 0}, {5, 0}, {5/2, 5/2}}]"
    );
    assert_eq!(
      interpret("ASATriangle[Pi/3, 2, Pi/6]").unwrap(),
      "Triangle[{{0, 0}, {2, 0}, {1/2, Sqrt[3]/2}}]"
    );
    assert_eq!(
      interpret("ASATriangle[Pi/4, s, Pi/4]").unwrap(),
      "Triangle[{{0, 0}, {s, 0}, {s/2, s/2}}]"
    );
  }

  #[test]
  fn aas_triangle() {
    assert_eq!(
      interpret("AASTriangle[Pi/6, Pi/3, 1]").unwrap(),
      "Triangle[{{0, 0}, {2, 0}, {3/2, Sqrt[3]/2}}]"
    );
    assert_eq!(
      interpret("AASTriangle[Pi/4, Pi/4, 2]").unwrap(),
      "Triangle[{{0, 0}, {2*Sqrt[2], 0}, {Sqrt[2], Sqrt[2]}}]"
    );
    assert_eq!(
      interpret("AASTriangle[Pi/6, Pi/3, x]").unwrap(),
      "Triangle[{{0, 0}, {2*x, 0}, {(3*x)/2, (Sqrt[3]*x)/2}}]"
    );
  }

  // Angle sums of Pi or more emit ::asm and echo; non-positive or invalid
  // sides/angles echo silently, matching wolframscript.
  #[test]
  fn invalid_inputs() {
    assert_eq!(
      interpret("AASTriangle[Pi/2, Pi/2, 1]").unwrap(),
      "AASTriangle[Pi/2, Pi/2, 1]"
    );
    assert_eq!(
      interpret("ASATriangle[Pi/2, 1, Pi/2]").unwrap(),
      "ASATriangle[Pi/2, 1, Pi/2]"
    );
    assert_eq!(
      interpret("SASTriangle[-1, Pi/3, 2]").unwrap(),
      "SASTriangle[-1, Pi/3, 2]"
    );
    assert_eq!(
      interpret("SASTriangle[0, Pi/3, 2]").unwrap(),
      "SASTriangle[0, Pi/3, 2]"
    );
    assert_eq!(
      interpret("SASTriangle[1, 4, 2]").unwrap(),
      "SASTriangle[1, 4, 2]"
    );
    // Symbolic angles evaluate through the Csc/Cot closed form with
    // wolframscript's trig phase canonicalization (Sin[a + Pi/3] ->
    // Cos[a - Pi/6]).
    assert_eq!(
      interpret("AASTriangle[a, Pi/3, 1]").unwrap(),
      "Triangle[{{0, 0}, {Cos[a - Pi/6]*Csc[a], 0}, {(Sqrt[3]*Cot[a])/2, Sqrt[3]/2}}]"
    );
  }

  // Fully symbolic constructors emit the general Csc/Cot closed forms.
  #[test]
  fn symbolic_angles() {
    assert_eq!(
      interpret("AASTriangle[a, b, c]").unwrap(),
      "Triangle[{{0, 0}, {c*Csc[a]*Sin[a + b], 0}, {c*Cot[a]*Sin[b], c*Sin[b]}}]"
    );
    assert_eq!(
      interpret("AASTriangle[a, Pi/4, 1]").unwrap(),
      "Triangle[{{0, 0}, {Csc[a]*Sin[a + Pi/4], 0}, {Cot[a]/Sqrt[2], 1/Sqrt[2]}}]"
    );
    assert_eq!(
      interpret("AASTriangle[Pi/3, a, 1]").unwrap(),
      "Triangle[{{0, 0}, {(2*Cos[a - Pi/6])/Sqrt[3], 0}, {Sin[a]/Sqrt[3], Sin[a]}}]"
    );
    assert_eq!(
      interpret("ASATriangle[a, 1, b]").unwrap(),
      "Triangle[{{0, 0}, {1, 0}, {Cos[a]*Csc[a + b]*Sin[b], Csc[a + b]*Sin[a]*Sin[b]}}]"
    );
    assert_eq!(
      interpret("SASTriangle[1, a, 2]").unwrap(),
      "Triangle[{{0, 0}, {Sqrt[5 - 4*Cos[a]], 0}, {(4 - 2*Cos[a])/Sqrt[5 - 4*Cos[a]], (2*Sin[a])/Sqrt[5 - 4*Cos[a]]}}]"
    );
  }
}

// The exact trig values now canonicalize on return, so quotients by them
// simplify instead of nesting divisions (regression for 2/Sin[Pi/4]).
mod exact_trig_canonicalization_tests {
  use woxi::interpret;

  #[test]
  fn quotients_by_exact_trig_values_simplify() {
    assert_eq!(interpret("2*Sin[Pi/2]/Sin[Pi/4]").unwrap(), "2*Sqrt[2]");
    assert_eq!(interpret("Sin[Pi/4]").unwrap(), "1/Sqrt[2]");
    assert_eq!(
      interpret("Sin[Pi/12]").unwrap(),
      "(-1 + Sqrt[3])/(2*Sqrt[2])"
    );
    assert_eq!(interpret("1/Cos[Pi/4]").unwrap(), "Sqrt[2]");
  }
}

mod triangle_measurement_tests {
  use woxi::interpret;

  // The 3-4-5 right triangle: every measurement is rational.
  #[test]
  fn pythagorean_triangle() {
    let t = r#"Triangle[{{0, 0}, {5, 0}, {16/5, 12/5}}]"#;
    assert_eq!(
      interpret(&format!(r#"TriangleMeasurement[{t}, "Area"]"#)).unwrap(),
      "6"
    );
    assert_eq!(
      interpret(&format!(r#"TriangleMeasurement[{t}, "Perimeter"]"#)).unwrap(),
      "12"
    );
    assert_eq!(
      interpret(&format!(r#"TriangleMeasurement[{t}, "Semiperimeter"]"#))
        .unwrap(),
      "6"
    );
    assert_eq!(
      interpret(&format!(r#"TriangleMeasurement[{t}, "Inradius"]"#)).unwrap(),
      "1"
    );
    assert_eq!(
      interpret(&format!(r#"TriangleMeasurement[{t}, "Circumradius"]"#))
        .unwrap(),
      "5/2"
    );
    // The one-argument form defaults to the area.
    assert_eq!(
      interpret(&format!("TriangleMeasurement[{t}]")).unwrap(),
      "6"
    );
  }

  #[test]
  fn exact_irrational_forms() {
    let u = "Triangle[{{0, 0}, {1, 0}, {0, 1}}]";
    assert_eq!(
      interpret(&format!(r#"TriangleMeasurement[{u}, "Area"]"#)).unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret(&format!(r#"TriangleMeasurement[{u}, "Perimeter"]"#)).unwrap(),
      "2 + Sqrt[2]"
    );
    // Term-wise halves fold the rational part, like wolframscript.
    assert_eq!(
      interpret(&format!(r#"TriangleMeasurement[{u}, "Semiperimeter"]"#))
        .unwrap(),
      "1 + 1/Sqrt[2]"
    );
    assert_eq!(
      interpret(&format!(r#"TriangleMeasurement[{u}, "Circumradius"]"#))
        .unwrap(),
      "1/Sqrt[2]"
    );
    assert_eq!(
      interpret(
        r#"TriangleMeasurement[Triangle[{{0, 0}, {3, 1}, {1, 4}}], "Perimeter"]"#
      )
      .unwrap(),
      "Sqrt[10] + Sqrt[13] + Sqrt[17]"
    );
    assert_eq!(
      interpret(
        r#"TriangleMeasurement[Triangle[{{0, 0}, {3, 1}, {1, 4}}], "Area"]"#
      )
      .unwrap(),
      "11/2"
    );
  }

  // Bare point lists and real coordinates work too.
  #[test]
  fn input_forms() {
    assert_eq!(
      interpret(r#"TriangleMeasurement[{{0, 0}, {1, 0}, {0, 1}}, "Area"]"#)
        .unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret(
        r#"TriangleMeasurement[Triangle[{{0., 0.}, {1., 0.}, {0., 1.}}], "Circumradius"]"#
      )
      .unwrap(),
      "0.7071067811865476"
    );
  }

  // Collinear points emit ::invtri; unknown properties echo silently.
  #[test]
  fn invalid_inputs() {
    assert_eq!(
      interpret(
        r#"TriangleMeasurement[Triangle[{{0, 0}, {1, 0}, {2, 0}}], "Area"]"#
      )
      .unwrap(),
      "TriangleMeasurement[Triangle[{{0, 0}, {1, 0}, {2, 0}}], Area]"
    );
    assert_eq!(
      interpret(
        r#"TriangleMeasurement[Triangle[{{0, 0}, {1, 1}, {2, 2}}], "Inradius"]"#
      )
      .unwrap(),
      "TriangleMeasurement[Triangle[{{0, 0}, {1, 1}, {2, 2}}], Inradius]"
    );
    assert_eq!(
      interpret(
        r#"TriangleMeasurement[Triangle[{{0, 0}, {5, 0}, {16/5, 12/5}}], "Foo"]"#
      )
      .unwrap(),
      "TriangleMeasurement[Triangle[{{0, 0}, {5, 0}, {16/5, 12/5}}], Foo]"
    );
  }
}

mod collinear_points {
  use super::*;

  #[test]
  fn planar_true_false() {
    assert_eq!(
      interpret("CollinearPoints[{{0, 0}, {1, 1}, {2, 2}}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("CollinearPoints[{{0, 0}, {1, 1}, {2, 3}}]").unwrap(),
      "False"
    );
    // More than three points, all on the line.
    assert_eq!(
      interpret("CollinearPoints[{{0, 0}, {1, 1}, {2, 2}, {3, 3}}]").unwrap(),
      "True"
    );
    // One point off the line breaks collinearity.
    assert_eq!(
      interpret("CollinearPoints[{{0, 0}, {1, 1}, {2, 2}, {3, 4}}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn three_dimensional() {
    assert_eq!(
      interpret(
        "CollinearPoints[{{0, 0, 0}, {1, 2, 3}, {2, 4, 6}, {-1, -2, -3}}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("CollinearPoints[{{1, 2, 3}, {2, 4, 6}, {3, 6, 10}}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn rationals_are_exact() {
    assert_eq!(
      interpret("CollinearPoints[{{0, 0}, {1/2, 1}, {1, 2}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn reals_are_strict() {
    // A 1e-10 deviation is not collinear — wolframscript uses no tolerance.
    assert_eq!(
      interpret("CollinearPoints[{{0., 0.}, {1., 1.}, {2., 2.0000000001}}]")
        .unwrap(),
      "False"
    );
  }

  #[test]
  fn degenerate_cases() {
    // Fewer than three points are trivially collinear.
    assert_eq!(interpret("CollinearPoints[{{5, 5}}]").unwrap(), "True");
    assert_eq!(
      interpret("CollinearPoints[{{0, 0}, {1, 1}}]").unwrap(),
      "True"
    );
    // Coincident points are collinear.
    assert_eq!(
      interpret("CollinearPoints[{{1, 1}, {1, 1}, {1, 1}}]").unwrap(),
      "True"
    );
  }
}

mod coplanar_points {
  use super::*;

  #[test]
  fn spatial_true_false() {
    // Four points in the z = 0 plane are coplanar.
    assert_eq!(
      interpret("CoplanarPoints[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}]")
        .unwrap(),
      "True"
    );
    // A tetrahedron's four vertices are not coplanar.
    assert_eq!(
      interpret("CoplanarPoints[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]")
        .unwrap(),
      "False"
    );
    assert_eq!(
      interpret("CoplanarPoints[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}}]")
        .unwrap(),
      "False"
    );
  }

  #[test]
  fn more_than_four_points() {
    assert_eq!(
      interpret(
        "CoplanarPoints[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {2, 3, 0}, {-1, 5, 0}}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "CoplanarPoints[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {2, 3, 0}, {-1, 5, 1}}]"
      )
      .unwrap(),
      "False"
    );
  }

  #[test]
  fn rationals_are_exact() {
    assert_eq!(
      interpret(
        "CoplanarPoints[{{0, 0, 0}, {1/2, 0, 0}, {0, 1/3, 0}, {1, 1, 0}}]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn reals_are_strict() {
    // A 1e-10 deviation out of plane is not coplanar.
    assert_eq!(
      interpret(
        "CoplanarPoints[{{0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.}, {1., 1., 0.0000000001}}]"
      )
      .unwrap(),
      "False"
    );
  }

  #[test]
  fn degenerate_cases() {
    // Three or fewer points are always coplanar.
    assert_eq!(
      interpret("CoplanarPoints[{{0, 0, 0}, {1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "True"
    );
    // Points with only two coordinates cannot leave a plane.
    assert_eq!(
      interpret("CoplanarPoints[{{1, 2}, {3, 4}, {5, 6}, {7, 8}}]").unwrap(),
      "True"
    );
  }
}

mod convex_polygon_q {
  use super::*;

  #[test]
  fn convex_shapes() {
    assert_eq!(
      interpret("ConvexPolygonQ[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]")
        .unwrap(),
      "True"
    );
    // A triangle is always convex.
    assert_eq!(
      interpret("ConvexPolygonQ[Polygon[{{0, 0}, {1, 0}, {1, 1}}]]").unwrap(),
      "True"
    );
    // Orientation (clockwise) does not matter.
    assert_eq!(
      interpret("ConvexPolygonQ[Polygon[{{0, 0}, {0, 1}, {1, 1}, {1, 0}}]]")
        .unwrap(),
      "True"
    );
    // A collinear vertex on an edge keeps the polygon convex.
    assert_eq!(
      interpret("ConvexPolygonQ[Polygon[{{0, 0}, {1, 0}, {2, 0}, {1, 1}}]]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn non_convex_shapes() {
    // A reflex vertex makes the polygon concave.
    assert_eq!(
      interpret(
        "ConvexPolygonQ[Polygon[{{0, 0}, {2, 0}, {2, 2}, {1, 1}, {0, 2}}]]"
      )
      .unwrap(),
      "False"
    );
    // A self-intersecting bowtie is not convex.
    assert_eq!(
      interpret("ConvexPolygonQ[Polygon[{{0, 0}, {1, 1}, {1, 0}, {0, 1}}]]")
        .unwrap(),
      "False"
    );
    // A pentagram turns consistently but winds around twice.
    assert_eq!(
      interpret(
        "ConvexPolygonQ[Polygon[{{0, 1}, {0.588, -0.809}, {-0.951, 0.309}, \
         {0.951, 0.309}, {-0.588, -0.809}}]]"
      )
      .unwrap(),
      "False"
    );
    // Fewer than three vertices cannot bound a polygon.
    assert_eq!(
      interpret("ConvexPolygonQ[Polygon[{{0, 0}, {1, 0}}]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn shape_constructors() {
    assert_eq!(
      interpret("ConvexPolygonQ[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("ConvexPolygonQ[Rectangle[{0, 0}, {2, 1}]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("ConvexPolygonQ[RegularPolygon[5]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn non_polygons_are_false() {
    // Bare point lists, plain values, symbols and non-polygon regions all
    // yield False rather than staying unevaluated.
    assert_eq!(
      interpret("ConvexPolygonQ[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]").unwrap(),
      "False"
    );
    assert_eq!(interpret("ConvexPolygonQ[5]").unwrap(), "False");
    assert_eq!(interpret("ConvexPolygonQ[x]").unwrap(), "False");
    assert_eq!(interpret("ConvexPolygonQ[Disk[]]").unwrap(), "False");
    // Symbolic coordinates cannot be verified convex.
    assert_eq!(
      interpret("ConvexPolygonQ[Polygon[{{0, 0}, {1, 0}, {a, 1}, {0, 1}}]]")
        .unwrap(),
      "False"
    );
  }
}

mod simple_polygon_q {
  use super::*;

  #[test]
  fn simple_shapes() {
    assert_eq!(
      interpret("SimplePolygonQ[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]")
        .unwrap(),
      "True"
    );
    // A triangle is always simple.
    assert_eq!(
      interpret("SimplePolygonQ[Polygon[{{0, 0}, {1, 0}, {1, 1}}]]").unwrap(),
      "True"
    );
    // Concave but non-self-intersecting is still simple.
    assert_eq!(
      interpret(
        "SimplePolygonQ[Polygon[{{0, 0}, {2, 0}, {2, 2}, {1, 1}, {0, 2}}]]"
      )
      .unwrap(),
      "True"
    );
    // A vertex touching a non-adjacent edge (no transversal crossing) is
    // still considered simple.
    assert_eq!(
      interpret("SimplePolygonQ[Polygon[{{0, 0}, {4, 0}, {2, 0}, {2, 2}}]]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn self_intersecting_shapes() {
    // A bowtie's non-adjacent edges cross.
    assert_eq!(
      interpret("SimplePolygonQ[Polygon[{{0, 0}, {1, 1}, {1, 0}, {0, 1}}]]")
        .unwrap(),
      "False"
    );
    // Two diagonals crossing transversally.
    assert_eq!(
      interpret("SimplePolygonQ[Polygon[{{0, 0}, {2, 2}, {2, 0}, {0, 2}}]]")
        .unwrap(),
      "False"
    );
    // A pentagram is self-intersecting.
    assert_eq!(
      interpret(
        "SimplePolygonQ[Polygon[{{0, 1}, {0.588, -0.809}, {-0.951, 0.309}, \
         {0.951, 0.309}, {-0.588, -0.809}}]]"
      )
      .unwrap(),
      "False"
    );
    // Fewer than three vertices cannot bound a polygon.
    assert_eq!(
      interpret("SimplePolygonQ[Polygon[{{0, 0}, {1, 0}}]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn shape_constructors() {
    assert_eq!(
      interpret("SimplePolygonQ[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "True"
    );
    assert_eq!(interpret("SimplePolygonQ[Rectangle[]]").unwrap(), "True");
    assert_eq!(
      interpret("SimplePolygonQ[RegularPolygon[6]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn non_polygons_are_false() {
    assert_eq!(
      interpret("SimplePolygonQ[{{0, 0}, {1, 0}, {1, 1}}]").unwrap(),
      "False"
    );
    assert_eq!(interpret("SimplePolygonQ[x]").unwrap(), "False");
    assert_eq!(interpret("SimplePolygonQ[Disk[]]").unwrap(), "False");
    assert_eq!(
      interpret("SimplePolygonQ[Polygon[{{0, 0}, {1, 0}, {a, 1}, {0, 1}}]]")
        .unwrap(),
      "False"
    );
  }
}

// Platonic-solid primitives (Cube, Octahedron, Dodecahedron, Icosahedron,
// and the regular-Tetrahedron forms) consumed by Volume, SurfaceArea,
// RegionMeasure, RegionCentroid, and RegionDimension. All outputs verified
// against wolframscript.
mod platonic_solid_primitives {
  use super::*;

  #[test]
  fn heads_stay_unevaluated() {
    assert_eq!(interpret("Dodecahedron[]").unwrap(), "Dodecahedron[]");
    assert_eq!(interpret("Icosahedron[2]").unwrap(), "Icosahedron[2]");
    assert_eq!(
      interpret("Cube[{1, 2, 3}, 2]").unwrap(),
      "Cube[{1, 2, 3}, 2]"
    );
    assert_eq!(interpret("Octahedron[a]").unwrap(), "Octahedron[a]");
  }

  #[test]
  fn unit_volumes() {
    assert_eq!(
      interpret("Volume[Dodecahedron[]]").unwrap(),
      "(15 + 7*Sqrt[5])/4"
    );
    assert_eq!(
      interpret("Volume[Icosahedron[]]").unwrap(),
      "(5*(3 + Sqrt[5]))/12"
    );
    assert_eq!(interpret("Volume[Cube[]]").unwrap(), "1");
    assert_eq!(
      interpret("Volume[Tetrahedron[2]]").unwrap(),
      "(2*Sqrt[2])/3"
    );
    assert_eq!(interpret("Volume[Octahedron[2]]").unwrap(), "(8*Sqrt[2])/3");
  }

  #[test]
  fn scaled_volumes() {
    assert_eq!(
      interpret("Volume[Dodecahedron[2]]").unwrap(),
      "2*(15 + 7*Sqrt[5])"
    );
    assert_eq!(
      interpret("Volume[Icosahedron[3]]").unwrap(),
      "(45*(3 + Sqrt[5]))/4"
    );
    assert_eq!(interpret("Volume[Cube[2]]").unwrap(), "8");
    assert_eq!(
      interpret("Volume[Dodecahedron[a]]").unwrap(),
      "((15 + 7*Sqrt[5])*a^3)/4"
    );
  }

  #[test]
  fn center_and_rotation_forms() {
    // Volume ignores the center and a {θ, ϕ} rotation spec.
    assert_eq!(
      interpret("Volume[Dodecahedron[{1, 2, 3}, 2]]").unwrap(),
      "2*(15 + 7*Sqrt[5])"
    );
    assert_eq!(
      interpret("Volume[Dodecahedron[{1, 2}]]").unwrap(),
      "(15 + 7*Sqrt[5])/4"
    );
    assert_eq!(interpret("Volume[Cube[{1, 2}, 3]]").unwrap(), "27");
    // A 3-element scalar list is a center, also for Tetrahedron.
    assert_eq!(
      interpret("Volume[Tetrahedron[{1, 2, 3}]]").unwrap(),
      "1/(6*Sqrt[2])"
    );
  }

  #[test]
  fn invalid_forms_stay_unevaluated() {
    // A concrete non-positive edge is invalid.
    assert_eq!(
      interpret("Volume[Dodecahedron[-2]]").unwrap(),
      "Volume[Dodecahedron[-2]]"
    );
    // The rotated-and-centered 3-argument form stays unevaluated.
    assert_eq!(
      interpret("Volume[Dodecahedron[{Pi/3, Pi/4}, {1, 2, 3}, 2]]").unwrap(),
      "Volume[Dodecahedron[{Pi/3, Pi/4}, {1, 2, 3}, 2]]"
    );
  }

  #[test]
  fn surface_areas() {
    assert_eq!(
      interpret("SurfaceArea[Dodecahedron[]]").unwrap(),
      "3*Sqrt[5*(5 + 2*Sqrt[5])]"
    );
    assert_eq!(
      interpret("SurfaceArea[Icosahedron[]]").unwrap(),
      "5*Sqrt[3]"
    );
    assert_eq!(interpret("SurfaceArea[Tetrahedron[]]").unwrap(), "Sqrt[3]");
    assert_eq!(interpret("SurfaceArea[Cube[2]]").unwrap(), "24");
    assert_eq!(
      interpret("SurfaceArea[Octahedron[3]]").unwrap(),
      "18*Sqrt[3]"
    );
    assert_eq!(
      interpret("SurfaceArea[Dodecahedron[a]]").unwrap(),
      "3*Sqrt[5*(5 + 2*Sqrt[5])]*a^2"
    );
    assert_eq!(
      interpret("SurfaceArea[Dodecahedron[{1, 2, 3}, 2]]").unwrap(),
      "12*Sqrt[5*(5 + 2*Sqrt[5])]"
    );
    assert_eq!(
      interpret("SurfaceArea[Icosahedron[{1, 2}, a]]").unwrap(),
      "5*Sqrt[3]*a^2"
    );
  }

  #[test]
  fn region_measure_is_volume() {
    assert_eq!(
      interpret("RegionMeasure[Dodecahedron[]]").unwrap(),
      "(15 + 7*Sqrt[5])/4"
    );
    assert_eq!(
      interpret("RegionMeasure[Icosahedron[2]]").unwrap(),
      "(10*(3 + Sqrt[5]))/3"
    );
    assert_eq!(interpret("RegionMeasure[Cube[2]]").unwrap(), "8");
    assert_eq!(
      interpret("RegionMeasure[Tetrahedron[]]").unwrap(),
      "1/(6*Sqrt[2])"
    );
  }

  #[test]
  fn region_centroid() {
    assert_eq!(
      interpret("RegionCentroid[Dodecahedron[]]").unwrap(),
      "{0, 0, 0}"
    );
    assert_eq!(
      interpret("RegionCentroid[Dodecahedron[{1, 2, 3}, 2]]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("RegionCentroid[Icosahedron[{1/2, 0, -1}, 3]]").unwrap(),
      "{1/2, 0, -1}"
    );
    assert_eq!(
      interpret("RegionCentroid[Icosahedron[2]]").unwrap(),
      "{0, 0, 0}"
    );
    assert_eq!(
      interpret("RegionCentroid[Tetrahedron[{1, 2, 3}]]").unwrap(),
      "{1, 2, 3}"
    );
    // The explicit-vertex form still averages the vertices.
    assert_eq!(
      interpret(
        "RegionCentroid[Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]]"
      )
      .unwrap(),
      "{1/4, 1/4, 1/4}"
    );
  }

  #[test]
  fn region_dimension() {
    assert_eq!(interpret("RegionDimension[Dodecahedron[]]").unwrap(), "3");
    assert_eq!(interpret("RegionDimension[Icosahedron[]]").unwrap(), "3");
  }
}

// SurfaceArea for the non-Platonic solids, plus Undefined for regions of
// intrinsic dimension < 3. All wolframscript-verified.
mod surface_area {
  use super::*;

  #[test]
  fn ball() {
    assert_eq!(interpret("SurfaceArea[Ball[]]").unwrap(), "4*Pi");
    assert_eq!(
      interpret("SurfaceArea[Ball[{1, 2, 3}, r]]").unwrap(),
      "4*Pi*r^2"
    );
    // A 2-D ball is a disk: no surface area.
    assert_eq!(
      interpret("SurfaceArea[Ball[{0, 0}, 1]]").unwrap(),
      "Undefined"
    );
  }

  #[test]
  fn cuboid() {
    assert_eq!(interpret("SurfaceArea[Cuboid[]]").unwrap(), "6");
    assert_eq!(
      interpret("SurfaceArea[Cuboid[{0, 0, 0}, {1, 2, 3}]]").unwrap(),
      "22"
    );
    assert_eq!(
      interpret("SurfaceArea[Cuboid[{0, 0}, {1, 2}]]").unwrap(),
      "Undefined"
    );
  }

  #[test]
  fn cylinder_and_cone() {
    assert_eq!(interpret("SurfaceArea[Cylinder[]]").unwrap(), "6*Pi");
    assert_eq!(
      interpret("SurfaceArea[Cylinder[{{0, 0, 0}, {0, 0, 3}}, 2]]").unwrap(),
      "20*Pi"
    );
    assert_eq!(
      interpret("SurfaceArea[Cone[]]").unwrap(),
      "(1 + Sqrt[5])*Pi"
    );
  }

  #[test]
  fn explicit_tetrahedron() {
    assert_eq!(
      interpret(
        "SurfaceArea[Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]]"
      )
      .unwrap(),
      "(3 + Sqrt[3])/2"
    );
    assert_eq!(
      interpret(
        "SurfaceArea[Tetrahedron[{{0, 0, 0}, {1, 2, 0}, {0, 1, 3}, {2, 0, 1}}]]"
      )
      .unwrap(),
      "(5*Sqrt[2] + Sqrt[21] + Sqrt[41] + Sqrt[46])/2"
    );
  }

  #[test]
  fn lower_dimensional_regions_are_undefined() {
    assert_eq!(interpret("SurfaceArea[Sphere[]]").unwrap(), "Undefined");
    assert_eq!(interpret("SurfaceArea[Disk[]]").unwrap(), "Undefined");
    assert_eq!(
      interpret("SurfaceArea[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "Undefined"
    );
  }
}

// Area of a triangle embedded in 3-space (half the cross-product norm),
// in wolframscript's canonical radical forms.
mod triangle_area_3d {
  use super::*;

  #[test]
  fn triangle_3d() {
    assert_eq!(
      interpret("Area[Triangle[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Area[Triangle[{{0, 0, 0}, {1, 0, 0}, {0, 1, 1}}]]").unwrap(),
      "1/Sqrt[2]"
    );
    assert_eq!(
      interpret("Area[Triangle[{{0, 0, 0}, {1, 2, 0}, {0, 1, 3}}]]").unwrap(),
      "Sqrt[23/2]"
    );
  }
}

// Sqrt of a fully numeric product containing a sum keeps the radical
// merged: wolframscript only extracts the perfect-square part.
mod numeric_radicand_no_split {
  use super::*;

  #[test]
  fn stays_merged() {
    assert_eq!(
      interpret("Sqrt[5*(5 + 2*Sqrt[5])]*a^2").unwrap(),
      "Sqrt[5*(5 + 2*Sqrt[5])]*a^2"
    );
    assert_eq!(
      interpret("Sqrt[2*(1 + Sqrt[2])]*a").unwrap(),
      "Sqrt[2*(1 + Sqrt[2])]*a"
    );
  }

  #[test]
  fn extracts_square_part() {
    assert_eq!(
      interpret("Sqrt[12*(1 + Sqrt[2])]").unwrap(),
      "2*Sqrt[3*(1 + Sqrt[2])]"
    );
    assert_eq!(
      interpret("Sqrt[12*(1 + Sqrt[2])]*a").unwrap(),
      "2*Sqrt[3*(1 + Sqrt[2])]*a"
    );
    assert_eq!(
      interpret("Sqrt[4*(1 + Sqrt[2])]").unwrap(),
      "2*Sqrt[1 + Sqrt[2]]"
    );
  }

  #[test]
  fn three_halves_exponent_still_distributes() {
    assert_eq!(
      interpret("(2*(1 + Sqrt[2]))^(3/2)").unwrap(),
      "2*Sqrt[2]*(1 + Sqrt[2])^(3/2)"
    );
  }
}
