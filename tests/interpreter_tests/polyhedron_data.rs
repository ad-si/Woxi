use super::*;

mod polyhedron_data_tests {
  use super::*;

  // Counts for the five Platonic solids.
  #[test]
  fn polyhedron_data_counts() {
    assert_eq!(
      interpret(r#"PolyhedronData["Tetrahedron", "FaceCount"]"#).unwrap(),
      "4"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Cube", "EdgeCount"]"#).unwrap(),
      "12"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Octahedron", "VertexCount"]"#).unwrap(),
      "6"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Dodecahedron", "FaceCount"]"#).unwrap(),
      "12"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Icosahedron", "FaceCount"]"#).unwrap(),
      "20"
    );
  }

  // Exact metric properties for unit edge length.
  #[test]
  fn polyhedron_data_volumes() {
    assert_eq!(
      interpret(r#"PolyhedronData["Tetrahedron", "Volume"]"#).unwrap(),
      "1/(6*Sqrt[2])"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Cube", "Volume"]"#).unwrap(),
      "1"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Octahedron", "Volume"]"#).unwrap(),
      "Sqrt[2]/3"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Dodecahedron", "Volume"]"#).unwrap(),
      "(15 + 7*Sqrt[5])/4"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Icosahedron", "Volume"]"#).unwrap(),
      "(5*(3 + Sqrt[5]))/12"
    );
  }

  #[test]
  fn polyhedron_data_surface_areas() {
    assert_eq!(
      interpret(r#"PolyhedronData["Tetrahedron", "SurfaceArea"]"#).unwrap(),
      "Sqrt[3]"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Cube", "SurfaceArea"]"#).unwrap(),
      "6"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Icosahedron", "SurfaceArea"]"#).unwrap(),
      "5*Sqrt[3]"
    );
  }

  #[test]
  fn polyhedron_data_radii() {
    assert_eq!(
      interpret(r#"PolyhedronData["Cube", "Circumradius"]"#).unwrap(),
      "Sqrt[3]/2"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Cube", "Inradius"]"#).unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Octahedron", "Circumradius"]"#).unwrap(),
      "1/Sqrt[2]"
    );
  }

  // "Hexahedron" is an alternative name for the cube.
  #[test]
  fn polyhedron_data_hexahedron_alias() {
    assert_eq!(
      interpret(r#"PolyhedronData["Hexahedron", "Volume"]"#).unwrap(),
      "1"
    );
  }

  // PolyhedronData[name] renders the solid as a Graphics3D object.
  #[test]
  fn polyhedron_data_renders_graphics3d() {
    assert_eq!(
      interpret(r#"PolyhedronData["Cube"]"#).unwrap(),
      "-Graphics3D-"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Dodecahedron"]"#).unwrap(),
      "-Graphics3D-"
    );
  }

  // Unknown polyhedra yield $Failed (with a notent message).
  #[test]
  fn polyhedron_data_unknown_name() {
    assert_eq!(
      interpret(r#"PolyhedronData["NoSuchSolid", "Volume"]"#).unwrap(),
      "$Failed"
    );
  }

  // Unknown properties stay unevaluated.
  #[test]
  fn polyhedron_data_unknown_property() {
    assert_eq!(
      interpret(r#"PolyhedronData["Cube", "NoSuchProperty"]"#).unwrap(),
      "PolyhedronData[Cube, NoSuchProperty]"
    );
  }
}
