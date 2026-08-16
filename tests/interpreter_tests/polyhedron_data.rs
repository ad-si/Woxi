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
      "TruncatedTetrahedron",
      "TruncatedOctahedron",
      "SmallRhombicuboctahedron",
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
      "{Circumradius, Classes, EdgeCount, EdgeIndices, FaceCount, \
       FaceIndices, Faces, Inradius, Insphere, SurfaceArea, \
       VertexCoordinates, VertexCount, Volume}"
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
      "TruncatedTetrahedron",
      "TruncatedOctahedron",
      "SmallRhombicuboctahedron",
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
      "{Cube, DeltoidalHexecontahedron, DisdyakisTriacontahedron, \
       Dodecahedron, GreatRhombicosidodecahedron, Icosahedron, \
       Icosidodecahedron, Octahedron, PentakisDodecahedron, \
       RhombicDodecahedron, RhombicTriacontahedron, \
       SmallRhombicosidodecahedron, SmallRhombicuboctahedron, \
       Tetrahedron, TruncatedDodecahedron, TruncatedIcosahedron, \
       TruncatedOctahedron, TruncatedTetrahedron}"
    );
  }

  // `"Faces"` is the vertex coordinates and the face index lists in one
  // `GraphicsComplex`, which is how a scene picks a solid apart: `[[1]]`
  // are the corners, `[[2, 1]]` the faces that index into them.
  #[test]
  fn polyhedron_data_faces_is_a_graphics_complex() {
    assert_eq!(
      interpret(r#"PolyhedronData["Cube", "Faces"]"#).unwrap(),
      "GraphicsComplex[{{-1/2, -1/2, -1/2}, {-1/2, -1/2, 1/2}, \
       {-1/2, 1/2, -1/2}, {-1/2, 1/2, 1/2}, {1/2, -1/2, -1/2}, \
       {1/2, -1/2, 1/2}, {1/2, 1/2, -1/2}, {1/2, 1/2, 1/2}}, \
       Polygon[{{8, 4, 2, 6}, {8, 6, 5, 7}, {8, 7, 3, 4}, {4, 3, 1, 2}, \
       {1, 3, 7, 5}, {2, 1, 5, 6}}]]"
    );
    // The two parts agree with the properties they are built from, for
    // every solid: the same corners (compared numerically, since the two
    // radical forms need not print alike) and the same face indices.
    assert_eq!(
      interpret(
        r#"Union @ Table[
             {Max @ Abs @ Flatten[N[PolyhedronData[s, "Faces"][[1]]] -
                N[PolyhedronData[s, "VertexCoordinates"]]] < 10^-10,
              PolyhedronData[s, "Faces"][[2, 1]] ===
                PolyhedronData[s, "FaceIndices"]},
             {s, PolyhedronData[All]}]"#
      )
      .unwrap(),
      "{{True, True}}"
    );
  }

  // The Archimedean solids with icosahedral symmetry, and their Catalan
  // duals — the shapes a polyhedral kaleidoscope is cut from.
  #[test]
  fn polyhedron_data_icosahedral_solids() {
    assert_eq!(
      interpret(
        r#"Table[{PolyhedronData[s, "VertexCount"],
                  PolyhedronData[s, "EdgeCount"],
                  PolyhedronData[s, "FaceCount"]},
             {s, {"Icosidodecahedron", "TruncatedIcosahedron",
                  "GreatRhombicosidodecahedron", "RhombicTriacontahedron",
                  "DisdyakisTriacontahedron"}}]"#
      )
      .unwrap(),
      "{{30, 60, 32}, {60, 90, 32}, {120, 180, 62}, {32, 60, 30}, \
       {62, 180, 120}}"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["Icosidodecahedron", "Volume"]"#).unwrap(),
      "(45 + 17*Sqrt[5])/6"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["TruncatedIcosahedron", "Circumradius"]"#)
        .unwrap(),
      "Sqrt[58 + 18*Sqrt[5]]/4"
    );
    // An Archimedean solid has a circumsphere but no insphere; its dual
    // has it the other way round.
    assert_eq!(
      interpret(r#"PolyhedronData["TruncatedIcosahedron", "Inradius"]"#)
        .unwrap(),
      "Missing[NotApplicable]"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["RhombicTriacontahedron", "Circumradius"]"#)
        .unwrap(),
      "Missing[NotApplicable]"
    );
    assert_eq!(
      interpret(r#"PolyhedronData["RhombicTriacontahedron", "Inradius"]"#)
        .unwrap(),
      "Sqrt[1 + 2/Sqrt[5]]"
    );
    // Face shapes: the truncated icosahedron is the football, twelve
    // pentagons and twenty hexagons.
    assert_eq!(
      interpret(
        r#"Tally[Length /@ PolyhedronData["TruncatedIcosahedron",
             "FaceIndices"]]"#
      )
      .unwrap(),
      "{{5, 12}, {6, 20}}"
    );
    // Every solid still draws.
    assert_eq!(
      interpret(r#"PolyhedronData["GreatRhombicosidodecahedron"]"#).unwrap(),
      "-Graphics3D-"
    );
  }

  // The Archimedean solids with cubic (octahedral) symmetry: each is a
  // Platonic solid with its corners truncated.
  #[test]
  fn polyhedron_data_cubic_archimedean_solids() {
    assert_eq!(
      interpret(
        r#"Table[{PolyhedronData[s, "VertexCount"],
                  PolyhedronData[s, "EdgeCount"],
                  PolyhedronData[s, "FaceCount"]},
             {s, {"TruncatedTetrahedron", "TruncatedOctahedron",
                  "SmallRhombicuboctahedron"}}]"#
      )
      .unwrap(),
      "{{12, 18, 8}, {24, 36, 14}, {24, 48, 26}}"
    );
    assert_eq!(
      interpret(
        r#"{PolyhedronData["TruncatedTetrahedron", "Volume"],
            PolyhedronData["TruncatedOctahedron", "Volume"],
            PolyhedronData["SmallRhombicuboctahedron", "Volume"]}"#
      )
      .unwrap(),
      "{(23*Sqrt[2])/12, 8*Sqrt[2], (12 + 10*Sqrt[2])/3}"
    );
    // None has a true insphere: their two face types sit at different
    // distances from the center.
    assert_eq!(
      interpret(
        r#"{PolyhedronData["TruncatedTetrahedron", "Inradius"],
            PolyhedronData["TruncatedOctahedron", "Inradius"],
            PolyhedronData["SmallRhombicuboctahedron", "Inradius"]}"#
      )
      .unwrap(),
      "{Missing[NotApplicable], Missing[NotApplicable], \
       Missing[NotApplicable]}"
    );
    // Face shapes: a truncated tetrahedron is 4 triangles + 4 hexagons, a
    // truncated octahedron is 6 squares + 8 hexagons, and a (small)
    // rhombicuboctahedron is 8 triangles + 18 squares.
    assert_eq!(
      interpret(
        r#"Tally[Length /@ PolyhedronData["TruncatedTetrahedron",
             "FaceIndices"]]"#
      )
      .unwrap(),
      "{{3, 4}, {6, 4}}"
    );
    assert_eq!(
      interpret(
        r#"Tally[Length /@ PolyhedronData["TruncatedOctahedron",
             "FaceIndices"]]"#
      )
      .unwrap(),
      "{{4, 6}, {6, 8}}"
    );
    assert_eq!(
      interpret(
        r#"Tally[Length /@ PolyhedronData["SmallRhombicuboctahedron",
             "FaceIndices"]]"#
      )
      .unwrap(),
      "{{3, 8}, {4, 18}}"
    );
    // All vertices sit at the same distance from the center (they are
    // vertex-transitive), matching the exact "Circumradius".
    assert_eq!(
      interpret(
        r#"Union @ Table[
             Max[Abs[
               N[Norm /@ PolyhedronData[s, "VertexCoordinates"]] -
                 N[PolyhedronData[s, "Circumradius"]]]] < 10^-10,
             {s, {"TruncatedTetrahedron", "TruncatedOctahedron",
                  "SmallRhombicuboctahedron"}}]"#
      )
      .unwrap(),
      "{True}"
    );
    // Every solid still draws.
    assert_eq!(
      interpret(r#"PolyhedronData["SmallRhombicuboctahedron"]"#).unwrap(),
      "-Graphics3D-"
    );
  }

  // Edges come off the faces, not off the shortest vertex distances: the
  // Catalan solids have two edge lengths, and taking only the shortest
  // ones lost half of a deltoidal hexecontahedron's 120 edges.
  #[test]
  fn polyhedron_data_edges_cover_solids_with_two_edge_lengths() {
    assert_eq!(
      interpret(
        r#"Union @ Table[
             Length[PolyhedronData[s, "EdgeIndices"]] ==
               PolyhedronData[s, "EdgeCount"],
             {s, PolyhedronData[All]}]"#
      )
      .unwrap(),
      "{True}"
    );
    // Euler's formula holds for all of them, so no face or edge is lost.
    assert_eq!(
      interpret(
        r#"Union @ Table[
             PolyhedronData[s, "VertexCount"] -
               PolyhedronData[s, "EdgeCount"] +
               PolyhedronData[s, "FaceCount"],
             {s, PolyhedronData[All]}]"#
      )
      .unwrap(),
      "{2}"
    );
    assert_eq!(
      interpret(
        r#"Take[PolyhedronData["DeltoidalHexecontahedron", "EdgeIndices"], 4]"#
      )
      .unwrap(),
      "{{1, 7}, {1, 9}, {1, 33}, {1, 47}}"
    );
  }

  // `"Classes"` reports what a solid is; the classless `PolyhedronData[
  // "Classes"]` call is the union over the entities.
  #[test]
  fn polyhedron_data_classes() {
    assert_eq!(
      interpret(r#"PolyhedronData["Icosidodecahedron", "Classes"]"#).unwrap(),
      "{Amphichiral, Archimedean, Canonical, Convex, Equilateral, \
       Quasiregular, Rigid, Rupert, Simple, Uniform}"
    );
    assert_eq!(
      interpret(
        r#"{MemberQ[PolyhedronData["Cube", "Classes"], "Platonic"],
            MemberQ[PolyhedronData["RhombicTriacontahedron", "Classes"],
              "Platonic"]}"#
      )
      .unwrap(),
      "{True, False}"
    );
    // Every class a solid claims is in the overall list, and the list has
    // no class no solid claims.
    assert_eq!(
      interpret(
        r#"Sort[Union @@ Table[PolyhedronData[s, "Classes"],
             {s, PolyhedronData[All]}]] === PolyhedronData["Classes"]"#
      )
      .unwrap(),
      "True"
    );
  }
}
