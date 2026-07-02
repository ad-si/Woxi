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
