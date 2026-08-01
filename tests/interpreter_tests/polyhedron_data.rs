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

  // Exact vertex coordinates (unit edge length) in Wolfram's canonical
  // orientation: z is the polar symmetry axis where there is one.
  #[test]
  fn polyhedron_data_vertex_coordinates_icosahedron() {
    assert_eq!(
      interpret(r#"PolyhedronData["Icosahedron", "VertexCoordinates"]"#)
        .unwrap(),
      "{{0, 0, -Sqrt[5/8 + Sqrt[5]/8]}, {0, 0, Sqrt[5/8 + Sqrt[5]/8]}, \
       {-Sqrt[1/2 + 1/(2*Sqrt[5])], 0, -Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
       {Sqrt[1/2 + 1/(2*Sqrt[5])], 0, Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
       {Sqrt[1/4 + 1/(2*Sqrt[5])], -1/2, -Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
       {Sqrt[1/4 + 1/(2*Sqrt[5])], 1/2, -Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
       {-Sqrt[1/4 + 1/(2*Sqrt[5])], -1/2, Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
       {-Sqrt[1/4 + 1/(2*Sqrt[5])], 1/2, Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
       {-Sqrt[1/8 - 1/(8*Sqrt[5])], (-1 - Sqrt[5])/4, -Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
       {-Sqrt[1/8 - 1/(8*Sqrt[5])], (1 + Sqrt[5])/4, -Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
       {Sqrt[1/8 - 1/(8*Sqrt[5])], (-1 - Sqrt[5])/4, Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
       {Sqrt[1/8 - 1/(8*Sqrt[5])], (1 + Sqrt[5])/4, Sqrt[1/8 + 1/(8*Sqrt[5])]}}"
    );
  }

  #[test]
  fn polyhedron_data_vertex_coordinates_cube_and_octahedron() {
    assert_eq!(
      interpret(r#"PolyhedronData["Cube", "VertexCoordinates"]"#).unwrap(),
      "{{-1/2, -1/2, -1/2}, {-1/2, -1/2, 1/2}, {-1/2, 1/2, -1/2}, \
       {-1/2, 1/2, 1/2}, {1/2, -1/2, -1/2}, {1/2, -1/2, 1/2}, \
       {1/2, 1/2, -1/2}, {1/2, 1/2, 1/2}}"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Octahedron", "VertexCoordinates"]"#)
        .unwrap(),
      "{{-(1/Sqrt[2]), 0, 0}, {0, 1/Sqrt[2], 0}, {0, 0, -(1/Sqrt[2])}, \
       {0, 0, 1/Sqrt[2]}, {0, -(1/Sqrt[2]), 0}, {1/Sqrt[2], 0, 0}}"
    );
  }

  // The stored coordinates must describe a unit-edge solid: every solid's
  // shortest vertex-to-vertex distance is exactly 1, and the number of
  // pairs at that distance is the edge count.
  #[test]
  fn polyhedron_data_vertex_coordinates_have_unit_edges() {
    for name in [
      "Tetrahedron",
      "Cube",
      "Octahedron",
      "Dodecahedron",
      "Icosahedron",
    ] {
      let result = interpret(&format!(
        r#"With[{{v = N[PolyhedronData["{name}", "VertexCoordinates"]]}},
             {{Length[v],
               Round[1000000 * Min @@ Flatten[Table[
                 Norm[v[[i]] - v[[j]]],
                 {{i, Length[v] - 1}}, {{j, i + 1, Length[v]}}]]]}}]"#
      ))
      .unwrap();
      let expected_counts = match name {
        "Tetrahedron" => "{4, 1000000}",
        "Cube" => "{8, 1000000}",
        "Octahedron" => "{6, 1000000}",
        "Dodecahedron" => "{20, 1000000}",
        "Icosahedron" => "{12, 1000000}",
        _ => unreachable!(),
      };
      assert_eq!(result, expected_counts, "solid: {name}");
    }
  }

  // Edge lists as 1-based vertex index pairs in canonical order.
  #[test]
  fn polyhedron_data_edge_indices() {
    assert_eq!(
      interpret(r#"PolyhedronData["Tetrahedron", "EdgeIndices"]"#).unwrap(),
      "{{1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}}"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Cube", "EdgeIndices"]"#).unwrap(),
      "{{1, 2}, {1, 3}, {1, 5}, {2, 4}, {2, 6}, {3, 4}, {3, 7}, {4, 8}, \
       {5, 6}, {5, 7}, {6, 8}, {7, 8}}"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Icosahedron", "EdgeIndices"]"#).unwrap(),
      "{{1, 3}, {1, 5}, {1, 6}, {1, 9}, {1, 10}, {2, 4}, {2, 7}, {2, 8}, \
       {2, 11}, {2, 12}, {3, 7}, {3, 8}, {3, 9}, {3, 10}, {4, 5}, {4, 6}, \
       {4, 11}, {4, 12}, {5, 6}, {5, 9}, {5, 11}, {6, 10}, {6, 12}, {7, 8}, \
       {7, 9}, {7, 11}, {8, 10}, {8, 12}, {9, 11}, {10, 12}}"
    );
  }

  // Every solid's edge list has exactly EdgeCount entries.
  #[test]
  fn polyhedron_data_edge_indices_match_edge_count() {
    for name in [
      "Tetrahedron",
      "Cube",
      "Octahedron",
      "Dodecahedron",
      "Icosahedron",
    ] {
      let result = interpret(&format!(
        r#"Length[PolyhedronData["{name}", "EdgeIndices"]] ==
           PolyhedronData["{name}", "EdgeCount"]"#
      ))
      .unwrap();
      assert_eq!(result, "True", "solid: {name}");
    }
  }

  // The insphere is a Sphere at the origin with the exact inradius.
  #[test]
  fn polyhedron_data_insphere() {
    assert_eq!(
      interpret(r#"PolyhedronData["Cube", "Insphere"]"#).unwrap(),
      "Sphere[{0, 0, 0}, 1/2]"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Icosahedron", "Insphere"]"#).unwrap(),
      "Sphere[{0, 0, 0}, (3*Sqrt[3] + Sqrt[15])/12]"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Dodecahedron", "Insphere"]"#).unwrap(),
      "Sphere[{0, 0, 0}, Sqrt[250 + 110*Sqrt[5]]/20]"
    );
  }

  // The new data properties appear in the sorted "Properties" list.
  #[test]
  fn polyhedron_data_properties_include_data_properties() {
    assert_eq!(
      interpret(r#"PolyhedronData["Properties"]"#).unwrap(),
      "{Circumradius, EdgeCount, EdgeIndices, FaceCount, FaceIndices, \
       Inradius, Insphere, SurfaceArea, VertexCoordinates, VertexCount, \
       Volume}"
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

  // Unknown polyhedra stay unevaluated (with a notent message).
  #[test]
  fn polyhedron_data_unknown_name() {
    assert_eq!(
      interpret(r#"PolyhedronData["NoSuchSolid", "Volume"]"#).unwrap(),
      r#"PolyhedronData[NoSuchSolid, Volume]"#
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
  // Faces as 1-based vertex index lists, in Wolfram's order and winding.
  #[test]
  fn polyhedron_data_face_indices() {
    assert_eq!(
      interpret(r#"PolyhedronData["Tetrahedron", "FaceIndices"]"#).unwrap(),
      "{{2, 3, 4}, {3, 2, 1}, {4, 1, 2}, {1, 4, 3}}"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Cube", "FaceIndices"]"#).unwrap(),
      "{{8, 4, 2, 6}, {8, 6, 5, 7}, {8, 7, 3, 4}, {4, 3, 1, 2}, \
       {1, 3, 7, 5}, {2, 1, 5, 6}}"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Octahedron", "FaceIndices"]"#).unwrap(),
      "{{4, 5, 6}, {4, 6, 2}, {4, 2, 1}, {4, 1, 5}, {5, 1, 3}, {5, 3, 6}, \
       {3, 1, 2}, {6, 3, 2}}"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["RhombicDodecahedron", "FaceIndices"]"#)
        .unwrap(),
      "{{2, 1, 3, 4}, {1, 2, 7, 5}, {6, 8, 3, 1}, {2, 4, 9, 7}, \
       {8, 10, 4, 3}, {11, 6, 1, 5}, {9, 4, 10, 14}, {5, 7, 12, 11}, \
       {11, 13, 8, 6}, {7, 9, 14, 12}, {13, 14, 10, 8}, {14, 13, 11, 12}}"
    );
  }

  // Every solid's face list has exactly FaceCount entries, and every index
  // in it addresses one of the solid's vertices.
  #[test]
  fn polyhedron_data_face_indices_are_consistent() {
    for name in [
      "Tetrahedron",
      "Cube",
      "Octahedron",
      "Dodecahedron",
      "Icosahedron",
      "RhombicDodecahedron",
    ] {
      assert_eq!(
        interpret(&format!(
          r#"Length[PolyhedronData["{name}", "FaceIndices"]] ==
             PolyhedronData["{name}", "FaceCount"]"#
        ))
        .unwrap(),
        "True",
        "solid: {name}"
      );
      assert_eq!(
        interpret(&format!(
          r#"Max[Flatten[PolyhedronData["{name}", "FaceIndices"]]] ==
             PolyhedronData["{name}", "VertexCount"] &&
           Min[Flatten[PolyhedronData["{name}", "FaceIndices"]]] == 1"#
        ))
        .unwrap(),
        "True",
        "solid: {name}"
      );
    }
  }

  // The rhombic dodecahedron is a Catalan solid: twelve rhombic faces, and
  // no circumradius, because its two kinds of vertex sit at different
  // distances from the center.
  #[test]
  fn polyhedron_data_rhombic_dodecahedron() {
    assert_eq!(
      interpret(
        r#"{PolyhedronData["RhombicDodecahedron", "VertexCount"],
            PolyhedronData["RhombicDodecahedron", "EdgeCount"],
            PolyhedronData["RhombicDodecahedron", "FaceCount"]}"#
      )
      .unwrap(),
      "{14, 24, 12}"
    );
    assert_eq!(
      interpret(
        r#"{PolyhedronData["RhombicDodecahedron", "Volume"],
            PolyhedronData["RhombicDodecahedron", "SurfaceArea"],
            PolyhedronData["RhombicDodecahedron", "Inradius"]}"#
      )
      .unwrap(),
      "{16/(3*Sqrt[3]), 8*Sqrt[2], Sqrt[2/3]}"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["RhombicDodecahedron", "Circumradius"]"#)
        .unwrap(),
      "Missing[NotApplicable]"
    );
    // Its faces are rhombi, so every face has four corners.
    assert_eq!(
      interpret(
        r#"Union[Length /@ PolyhedronData["RhombicDodecahedron", "FaceIndices"]]"#
      )
      .unwrap(),
      "{4}"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["RhombicDodecahedron"]"#).unwrap(),
      "-Graphics3D-"
    );
  }

  // It joins the list of known entities.
  #[test]
  fn polyhedron_data_all_lists_every_entity() {
    assert_eq!(
      interpret("PolyhedronData[All]").unwrap(),
      "{Cube, Dodecahedron, Icosahedron, Octahedron, RhombicDodecahedron, \
       Tetrahedron}"
    );
  }
}
