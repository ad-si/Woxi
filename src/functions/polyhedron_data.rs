//! PolyhedronData[name] and PolyhedronData[name, property] for the Platonic
//! solids. All metric properties refer to unit edge length and are stored as
//! exact Wolfram Language expressions so results stay symbolic.

#[allow(unused_imports)]
use super::*;

struct PolyhedronInfo {
  name: &'static str,
  vertex_count: i128,
  edge_count: i128,
  face_count: i128,
  /// Exact metric properties (unit edge length) as WL source.
  volume: &'static str,
  surface_area: &'static str,
  circumradius: &'static str,
  inradius: &'static str,
  /// Exact vertex coordinates for unit edge length, as WL source, in
  /// Wolfram's canonical orientation and vertex order (the z axis is the
  /// polar symmetry axis where there is one). Used both for the
  /// "VertexCoordinates" property and (numerically) for rendering.
  vertices_src: &'static str,
  /// Faces as 1-based indices into `vertices_src`, in Wolfram's order and
  /// winding, as WL source. Used both for the "FaceIndices" property and
  /// for rendering the solid.
  faces_src: &'static str,
  /// The classes this solid belongs to, as WL source, exactly as
  /// `PolyhedronData[name, "Classes"]` reports them.
  classes_src: &'static str,
}

static POLYHEDRA: &[PolyhedronInfo] = &[
  PolyhedronInfo {
    name: "Tetrahedron",
    vertex_count: 4,
    edge_count: 6,
    face_count: 4,
    volume: "1/(6*Sqrt[2])",
    surface_area: "Sqrt[3]",
    circumradius: "Sqrt[3/8]",
    inradius: "1/(2*Sqrt[6])",
    // Apex on the +z axis, then the base triangle in the
    // z = -Inradius plane.
    vertices_src: "{\
      {0, 0, Sqrt[2/3] - 1/(2*Sqrt[6])}, \
      {-1/(2*Sqrt[3]), -1/2, -1/(2*Sqrt[6])}, \
      {-1/(2*Sqrt[3]), 1/2, -1/(2*Sqrt[6])}, \
      {1/Sqrt[3], 0, -1/(2*Sqrt[6])}}",
    faces_src: "{{2, 3, 4}, {3, 2, 1}, {4, 1, 2}, {1, 4, 3}}",
    classes_src: "\
      {\"Amphichiral\", \"Canonical\", \"Convex\", \"Deltahedron\", \
      \"Equilateral\", \"Isohedron\", \"Platonic\", \
      \"PlatonicDual\", \"Pyramid\", \"Rigid\", \"Rupert\", \
      \"SelfDual\", \"Simple\", \"Uniform\", \"UniformDual\", \
      \"Zalgaller\"}",
  },
  PolyhedronInfo {
    name: "Cube",
    vertex_count: 8,
    edge_count: 12,
    face_count: 6,
    volume: "1",
    surface_area: "6",
    circumradius: "Sqrt[3]/2",
    inradius: "1/2",
    vertices_src: "{\
      {-1/2, -1/2, -1/2}, {-1/2, -1/2, 1/2}, \
      {-1/2, 1/2, -1/2}, {-1/2, 1/2, 1/2}, \
      {1/2, -1/2, -1/2}, {1/2, -1/2, 1/2}, \
      {1/2, 1/2, -1/2}, {1/2, 1/2, 1/2}}",
    faces_src: "{{8, 4, 2, 6}, {8, 6, 5, 7}, {8, 7, 3, 4}, \
      {4, 3, 1, 2}, {1, 3, 7, 5}, {2, 1, 5, 6}}",
    classes_src: "\
      {\"Amphichiral\", \"Canonical\", \"Convex\", \"Equilateral\", \
      \"Isohedron\", \"Parallelohedron\", \"Platonic\", \
      \"PlatonicDual\", \"Plesiohedron\", \"Prism\", \
      \"Rhombohedron\", \"Rigid\", \"Rupert\", \"Simple\", \
      \"SpaceFilling\", \"Stereohedron\", \"Trapezohedron\", \
      \"Uniform\", \"UniformDual\", \"Zonohedron\"}",
  },
  PolyhedronInfo {
    name: "Octahedron",
    vertex_count: 6,
    edge_count: 12,
    face_count: 8,
    volume: "Sqrt[2]/3",
    surface_area: "2*Sqrt[3]",
    circumradius: "1/Sqrt[2]",
    inradius: "1/Sqrt[6]",
    vertices_src: "{\
      {-1/Sqrt[2], 0, 0}, {0, 1/Sqrt[2], 0}, \
      {0, 0, -1/Sqrt[2]}, {0, 0, 1/Sqrt[2]}, \
      {0, -1/Sqrt[2], 0}, {1/Sqrt[2], 0, 0}}",
    faces_src: "{{4, 5, 6}, {4, 6, 2}, {4, 2, 1}, {4, 1, 5}, \
      {5, 1, 3}, {5, 3, 6}, {3, 1, 2}, {6, 3, 2}}",
    classes_src: "\
      {\"Amphichiral\", \"Antiprism\", \"Canonical\", \"Convex\", \
      \"Deltahedron\", \"Dipyramid\", \"Equilateral\", \
      \"Isohedron\", \"Platonic\", \"PlatonicDual\", \"Rigid\", \
      \"Rupert\", \"Simple\", \"Uniform\", \"UniformDual\"}",
  },
  PolyhedronInfo {
    name: "Dodecahedron",
    vertex_count: 20,
    edge_count: 30,
    face_count: 12,
    volume: "(15 + 7*Sqrt[5])/4",
    surface_area: "3*Sqrt[5*(5 + 2*Sqrt[5])]",
    circumradius: "(Sqrt[15] + Sqrt[3])/4",
    inradius: "Sqrt[250 + 110*Sqrt[5]]/20",
    // Two wide vertex rings around the equator (antipodal pairs first),
    // then the two rings of the top and bottom faces; z is the C5 axis.
    vertices_src: "{\
      {-Sqrt[1 + 2/Sqrt[5]], 0, Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[1 + 2/Sqrt[5]], 0, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[1/8 + Sqrt[5]/40], -(3 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[1/8 + Sqrt[5]/40], (3 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[5/8 + 11*Sqrt[5]/40], -(1 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[5/8 + 11*Sqrt[5]/40], (1 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[1/8 - Sqrt[5]/40], -(1 + Sqrt[5])/4, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[1/8 - Sqrt[5]/40], (1 + Sqrt[5])/4, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[1/4 + Sqrt[5]/10], -1/2, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[1/4 + Sqrt[5]/10], 1/2, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/4 + Sqrt[5]/10], -1/2, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/4 + Sqrt[5]/10], 1/2, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/2 + Sqrt[5]/10], 0, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[5/8 + 11*Sqrt[5]/40], -(1 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[5/8 + 11*Sqrt[5]/40], (1 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[1/2 + Sqrt[5]/10], 0, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/8 - Sqrt[5]/40], -(1 + Sqrt[5])/4, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/8 - Sqrt[5]/40], (1 + Sqrt[5])/4, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/8 + Sqrt[5]/40], -(3 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[1/8 + Sqrt[5]/40], (3 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}}",
    faces_src: "{{15, 10, 9, 14, 1}, {2, 6, 12, 11, 5}, {5, 11, 7, 3, 19}, \
      {11, 12, 8, 16, 7}, {12, 6, 20, 4, 8}, {6, 2, 13, 18, 20}, \
      {2, 5, 19, 17, 13}, {4, 20, 18, 10, 15}, {18, 13, 17, 9, 10}, \
      {17, 19, 3, 14, 9}, {3, 7, 16, 1, 14}, {16, 8, 4, 15, 1}}",
    classes_src: "\
      {\"Amphichiral\", \"Canonical\", \"Convex\", \"Equilateral\", \
      \"Goldberg\", \"Isohedron\", \"Platonic\", \"PlatonicDual\", \
      \"Rigid\", \"Rupert\", \"Simple\", \"Uniform\", \
      \"UniformDual\", \"Zalgaller\"}",
  },
  PolyhedronInfo {
    name: "Icosahedron",
    vertex_count: 12,
    edge_count: 30,
    face_count: 20,
    volume: "(5*(3 + Sqrt[5]))/12",
    surface_area: "5*Sqrt[3]",
    circumradius: "Sqrt[10 + 2*Sqrt[5]]/4",
    inradius: "(3*Sqrt[3] + Sqrt[15])/12",
    // The two poles, then the two staggered vertex rings (antipodal
    // pairs adjacent); z is the C5 axis through the poles.
    vertices_src: "{\
      {0, 0, -Sqrt[5/8 + Sqrt[5]/8]}, \
      {0, 0, Sqrt[5/8 + Sqrt[5]/8]}, \
      {-Sqrt[1/2 + Sqrt[5]/10], 0, -Sqrt[1/8 + Sqrt[5]/40]}, \
      {Sqrt[1/2 + Sqrt[5]/10], 0, Sqrt[1/8 + Sqrt[5]/40]}, \
      {Sqrt[1/4 + Sqrt[5]/10], -1/2, -Sqrt[1/8 + Sqrt[5]/40]}, \
      {Sqrt[1/4 + Sqrt[5]/10], 1/2, -Sqrt[1/8 + Sqrt[5]/40]}, \
      {-Sqrt[1/4 + Sqrt[5]/10], -1/2, Sqrt[1/8 + Sqrt[5]/40]}, \
      {-Sqrt[1/4 + Sqrt[5]/10], 1/2, Sqrt[1/8 + Sqrt[5]/40]}, \
      {-Sqrt[1/8 - Sqrt[5]/40], -(1 + Sqrt[5])/4, -Sqrt[1/8 + Sqrt[5]/40]}, \
      {-Sqrt[1/8 - Sqrt[5]/40], (1 + Sqrt[5])/4, -Sqrt[1/8 + Sqrt[5]/40]}, \
      {Sqrt[1/8 - Sqrt[5]/40], -(1 + Sqrt[5])/4, Sqrt[1/8 + Sqrt[5]/40]}, \
      {Sqrt[1/8 - Sqrt[5]/40], (1 + Sqrt[5])/4, Sqrt[1/8 + Sqrt[5]/40]}}",
    faces_src: "{{2, 12, 8}, {2, 8, 7}, {2, 7, 11}, {2, 11, 4}, {2, 4, 12}, \
      {5, 9, 1}, {6, 5, 1}, {10, 6, 1}, {3, 10, 1}, {9, 3, 1}, \
      {12, 10, 8}, {8, 3, 7}, {7, 9, 11}, {11, 5, 4}, {4, 6, 12}, \
      {5, 11, 9}, {6, 4, 5}, {10, 12, 6}, {3, 8, 10}, {9, 7, 3}}",
    classes_src: "\
      {\"Amphichiral\", \"Canonical\", \"Convex\", \"Deltahedron\", \
      \"Equilateral\", \"Isohedron\", \"Platonic\", \
      \"PlatonicDual\", \"Rigid\", \"Rupert\", \"Simple\", \
      \"Uniform\", \"UniformDual\"}",
  },
  // The rhombic dodecahedron is the one Catalan solid here: its faces are
  // rhombi, not regular polygons, so it has no circumradius (its vertices
  // are not all the same distance from the center).
  PolyhedronInfo {
    name: "RhombicDodecahedron",
    vertex_count: 14,
    edge_count: 24,
    face_count: 12,
    volume: "16/(3*Sqrt[3])",
    surface_area: "8*Sqrt[2]",
    circumradius: "Missing[\"NotApplicable\"]",
    inradius: "Sqrt[2/3]",
    vertices_src: "{\
      {-Sqrt[2/3], -Sqrt[2/3], 0}, {-Sqrt[2/3], 0, -1/Sqrt[3]}, \
      {-Sqrt[2/3], 0, 1/Sqrt[3]}, {-Sqrt[2/3], Sqrt[2/3], 0}, \
      {0, -Sqrt[2/3], -1/Sqrt[3]}, {0, -Sqrt[2/3], 1/Sqrt[3]}, \
      {0, 0, -2/Sqrt[3]}, {0, 0, 2/Sqrt[3]}, \
      {0, Sqrt[2/3], -1/Sqrt[3]}, {0, Sqrt[2/3], 1/Sqrt[3]}, \
      {Sqrt[2/3], -Sqrt[2/3], 0}, {Sqrt[2/3], 0, -1/Sqrt[3]}, \
      {Sqrt[2/3], 0, 1/Sqrt[3]}, {Sqrt[2/3], Sqrt[2/3], 0}}",
    faces_src: "{{2, 1, 3, 4}, {1, 2, 7, 5}, {6, 8, 3, 1}, {2, 4, 9, 7}, \
      {8, 10, 4, 3}, {11, 6, 1, 5}, {9, 4, 10, 14}, {5, 7, 12, 11}, \
      {11, 13, 8, 6}, {7, 9, 14, 12}, {13, 14, 10, 8}, {14, 13, 11, 12}}",
    classes_src: "\
      {\"Amphichiral\", \"ArchimedeanDual\", \"Canonical\", \
      \"Convex\", \"Equilateral\", \"Isohedron\", \
      \"Parallelohedron\", \"Plesiohedron\", \"Rigid\", \"Rupert\", \
      \"Simple\", \"SpaceFilling\", \"Stereohedron\", \
      \"UniformDual\", \"Zonohedron\"}",
  },
  PolyhedronInfo {
    name: "DeltoidalHexecontahedron",
    vertex_count: 62,
    edge_count: 120,
    face_count: 60,
    volume: "Sqrt[29530/9 + (13204*Sqrt[5])/9]",
    surface_area: "Sqrt[4370 + 1850*Sqrt[5]]",
    circumradius: "Missing[\"NotApplicable\"]",
    inradius: "Root[121 - 5710*#1^2 + 820*#1^4 & , 4, 0]",
    vertices_src: "\
      {{0, 0, -11/Sqrt[85 - 31*Sqrt[5]]}, {0, 0, \
      11/Sqrt[85 - 31*Sqrt[5]]}, {0, -11/Sqrt[85 - 31*Sqrt[5]], 0}, \
      {0, 11/Sqrt[85 - 31*Sqrt[5]], 0}, {0, \
      -1/3*Sqrt[53/2 + 59/Sqrt[5]], -1/6*Sqrt[41 + 89/Sqrt[5]]}, \
      {0, -1/3*Sqrt[53/2 + 59/Sqrt[5]], Sqrt[41 + 89/Sqrt[5]]/6}, \
      {0, -Sqrt[1/2 + 1/Sqrt[5]], -1/2*Sqrt[13 + 29/Sqrt[5]]}, {0, \
      -Sqrt[1/2 + 1/Sqrt[5]], Sqrt[13 + 29/Sqrt[5]]/2}, {0, \
      Sqrt[1/2 + 1/Sqrt[5]], -1/2*Sqrt[13 + 29/Sqrt[5]]}, {0, \
      Sqrt[1/2 + 1/Sqrt[5]], Sqrt[13 + 29/Sqrt[5]]/2}, {0, \
      Sqrt[53/18 + 59/(9*Sqrt[5])], -1/6*Sqrt[41 + 89/Sqrt[5]]}, \
      {0, Sqrt[53/18 + 59/(9*Sqrt[5])], Sqrt[41 + 89/Sqrt[5]]/6}, \
      {-11/Sqrt[85 - 31*Sqrt[5]], 0, 0}, \
      {-1/4*Sqrt[17 + 31/Sqrt[5]], -1/4*Sqrt[41 + 89/Sqrt[5]], \
      Sqrt[5/2 + 1/Sqrt[5]]/2}, {-1/4*Sqrt[17 + 31/Sqrt[5]], \
      -1/4*Sqrt[41 + 89/Sqrt[5]], -1/2*Sqrt[5/2 + 1/Sqrt[5]]}, \
      {-1/4*Sqrt[17 + 31/Sqrt[5]], Sqrt[41 + 89/Sqrt[5]]/4, \
      Sqrt[5/2 + 1/Sqrt[5]]/2}, {-1/4*Sqrt[17 + 31/Sqrt[5]], \
      Sqrt[41 + 89/Sqrt[5]]/4, -1/2*Sqrt[5/2 + 1/Sqrt[5]]}, \
      {Sqrt[17 + 31/Sqrt[5]]/4, -1/4*Sqrt[41 + 89/Sqrt[5]], \
      Sqrt[5/2 + 1/Sqrt[5]]/2}, {Sqrt[17 + 31/Sqrt[5]]/4, \
      -1/4*Sqrt[41 + 89/Sqrt[5]], -1/2*Sqrt[5/2 + 1/Sqrt[5]]}, \
      {Sqrt[17 + 31/Sqrt[5]]/4, Sqrt[41 + 89/Sqrt[5]]/4, \
      Sqrt[5/2 + 1/Sqrt[5]]/2}, {Sqrt[17 + 31/Sqrt[5]]/4, \
      Sqrt[41 + 89/Sqrt[5]]/4, -1/2*Sqrt[5/2 + 1/Sqrt[5]]}, \
      {11/Sqrt[85 - 31*Sqrt[5]], 0, 0}, \
      {-1/2*Sqrt[13 + 29/Sqrt[5]], 0, -Sqrt[1/2 + 1/Sqrt[5]]}, \
      {-1/2*Sqrt[13 + 29/Sqrt[5]], 0, Sqrt[1/2 + 1/Sqrt[5]]}, \
      {-1/2*Sqrt[5 + 11/Sqrt[5]], -1/2*Sqrt[5 + 11/Sqrt[5]], \
      -1/2*Sqrt[5 + 11/Sqrt[5]]}, {-1/2*Sqrt[5 + 11/Sqrt[5]], \
      -1/2*Sqrt[5 + 11/Sqrt[5]], Sqrt[5 + 11/Sqrt[5]]/2}, \
      {-1/2*Sqrt[5 + 11/Sqrt[5]], Sqrt[5 + 11/Sqrt[5]]/2, \
      -1/2*Sqrt[5 + 11/Sqrt[5]]}, {-1/2*Sqrt[5 + 11/Sqrt[5]], \
      Sqrt[5 + 11/Sqrt[5]]/2, Sqrt[5 + 11/Sqrt[5]]/2}, \
      {-1/3*Sqrt[53/2 + 59/Sqrt[5]], -1/6*Sqrt[41 + 89/Sqrt[5]], \
      0}, {-1/3*Sqrt[53/2 + 59/Sqrt[5]], Sqrt[41 + 89/Sqrt[5]]/6, \
      0}, {-Sqrt[1/2 + 1/Sqrt[5]], -1/2*Sqrt[13 + 29/Sqrt[5]], 0}, \
      {-Sqrt[1/2 + 1/Sqrt[5]], Sqrt[13 + 29/Sqrt[5]]/2, 0}, \
      {-1/6*Sqrt[41 + 89/Sqrt[5]], 0, \
      -1/3*Sqrt[53/2 + 59/Sqrt[5]]}, {-1/6*Sqrt[41 + 89/Sqrt[5]], \
      0, Sqrt[53/18 + 59/(9*Sqrt[5])]}, \
      {-1/4*Sqrt[41 + 89/Sqrt[5]], Sqrt[5/2 + 1/Sqrt[5]]/2, \
      -1/4*Sqrt[17 + 31/Sqrt[5]]}, {-1/4*Sqrt[41 + 89/Sqrt[5]], \
      Sqrt[5/2 + 1/Sqrt[5]]/2, Sqrt[17 + 31/Sqrt[5]]/4}, \
      {-1/4*Sqrt[41 + 89/Sqrt[5]], -1/2*Sqrt[5/2 + 1/Sqrt[5]], \
      -1/4*Sqrt[17 + 31/Sqrt[5]]}, {-1/4*Sqrt[41 + 89/Sqrt[5]], \
      -1/2*Sqrt[5/2 + 1/Sqrt[5]], Sqrt[17 + 31/Sqrt[5]]/4}, \
      {Sqrt[5/2 + 1/Sqrt[5]]/2, -1/4*Sqrt[17 + 31/Sqrt[5]], \
      -1/4*Sqrt[41 + 89/Sqrt[5]]}, {Sqrt[5/2 + 1/Sqrt[5]]/2, \
      -1/4*Sqrt[17 + 31/Sqrt[5]], Sqrt[41 + 89/Sqrt[5]]/4}, \
      {Sqrt[5/2 + 1/Sqrt[5]]/2, Sqrt[17 + 31/Sqrt[5]]/4, \
      -1/4*Sqrt[41 + 89/Sqrt[5]]}, {Sqrt[5/2 + 1/Sqrt[5]]/2, \
      Sqrt[17 + 31/Sqrt[5]]/4, Sqrt[41 + 89/Sqrt[5]]/4}, \
      {-1/2*Sqrt[5/2 + 1/Sqrt[5]], -1/4*Sqrt[17 + 31/Sqrt[5]], \
      -1/4*Sqrt[41 + 89/Sqrt[5]]}, {-1/2*Sqrt[5/2 + 1/Sqrt[5]], \
      -1/4*Sqrt[17 + 31/Sqrt[5]], Sqrt[41 + 89/Sqrt[5]]/4}, \
      {-1/2*Sqrt[5/2 + 1/Sqrt[5]], Sqrt[17 + 31/Sqrt[5]]/4, \
      -1/4*Sqrt[41 + 89/Sqrt[5]]}, {-1/2*Sqrt[5/2 + 1/Sqrt[5]], \
      Sqrt[17 + 31/Sqrt[5]]/4, Sqrt[41 + 89/Sqrt[5]]/4}, \
      {Sqrt[41 + 89/Sqrt[5]]/6, 0, -1/3*Sqrt[53/2 + 59/Sqrt[5]]}, \
      {Sqrt[41 + 89/Sqrt[5]]/6, 0, Sqrt[53/18 + 59/(9*Sqrt[5])]}, \
      {Sqrt[41 + 89/Sqrt[5]]/4, Sqrt[5/2 + 1/Sqrt[5]]/2, \
      -1/4*Sqrt[17 + 31/Sqrt[5]]}, {Sqrt[41 + 89/Sqrt[5]]/4, \
      Sqrt[5/2 + 1/Sqrt[5]]/2, Sqrt[17 + 31/Sqrt[5]]/4}, \
      {Sqrt[41 + 89/Sqrt[5]]/4, -1/2*Sqrt[5/2 + 1/Sqrt[5]], \
      -1/4*Sqrt[17 + 31/Sqrt[5]]}, {Sqrt[41 + 89/Sqrt[5]]/4, \
      -1/2*Sqrt[5/2 + 1/Sqrt[5]], Sqrt[17 + 31/Sqrt[5]]/4}, \
      {Sqrt[1/2 + 1/Sqrt[5]], -1/2*Sqrt[13 + 29/Sqrt[5]], 0}, \
      {Sqrt[1/2 + 1/Sqrt[5]], Sqrt[13 + 29/Sqrt[5]]/2, 0}, \
      {Sqrt[53/18 + 59/(9*Sqrt[5])], -1/6*Sqrt[41 + 89/Sqrt[5]], \
      0}, {Sqrt[53/18 + 59/(9*Sqrt[5])], Sqrt[41 + 89/Sqrt[5]]/6, \
      0}, {Sqrt[5 + 11/Sqrt[5]]/2, -1/2*Sqrt[5 + 11/Sqrt[5]], \
      -1/2*Sqrt[5 + 11/Sqrt[5]]}, {Sqrt[5 + 11/Sqrt[5]]/2, \
      -1/2*Sqrt[5 + 11/Sqrt[5]], Sqrt[5 + 11/Sqrt[5]]/2}, \
      {Sqrt[5 + 11/Sqrt[5]]/2, Sqrt[5 + 11/Sqrt[5]]/2, \
      -1/2*Sqrt[5 + 11/Sqrt[5]]}, {Sqrt[5 + 11/Sqrt[5]]/2, \
      Sqrt[5 + 11/Sqrt[5]]/2, Sqrt[5 + 11/Sqrt[5]]/2}, \
      {Sqrt[13 + 29/Sqrt[5]]/2, 0, -Sqrt[1/2 + 1/Sqrt[5]]}, \
      {Sqrt[13 + 29/Sqrt[5]]/2, 0, Sqrt[1/2 + 1/Sqrt[5]]}}",
    faces_src: "\
      {{43, 33, 1, 7}, {8, 2, 34, 44}, {33, 45, 9, 1}, {2, 10, 46, \
      34}, {31, 15, 5, 3}, {3, 6, 14, 31}, {17, 32, 4, 11}, {12, 4, \
      32, 16}, {39, 5, 43, 7}, {40, 8, 44, 6}, {9, 45, 11, 41}, \
      {12, 46, 10, 42}, {39, 7, 1, 47}, {48, 2, 8, 40}, {1, 9, 41, \
      47}, {48, 42, 10, 2}, {53, 3, 5, 19}, {18, 6, 3, 53}, {11, 4, \
      54, 21}, {20, 54, 4, 12}, {33, 37, 23, 35}, {24, 38, 34, 36}, \
      {15, 25, 43, 5}, {6, 44, 26, 14}, {45, 27, 17, 11}, {12, 16, \
      28, 46}, {29, 37, 25, 15}, {14, 26, 38, 29}, {27, 35, 30, \
      17}, {16, 30, 36, 28}, {29, 13, 23, 37}, {38, 24, 13, 29}, \
      {23, 13, 30, 35}, {36, 30, 13, 24}, {15, 31, 14, 29}, {25, \
      37, 33, 43}, {44, 34, 38, 26}, {33, 35, 27, 45}, {46, 28, 36, \
      34}, {17, 30, 16, 32}, {19, 5, 39, 57}, {58, 40, 6, 18}, {41, \
      11, 21, 59}, {60, 20, 12, 42}, {19, 57, 51, 55}, {55, 52, 58, \
      18}, {49, 59, 21, 56}, {56, 20, 60, 50}, {55, 51, 61, 22}, \
      {22, 62, 52, 55}, {61, 49, 56, 22}, {22, 56, 50, 62}, {55, \
      18, 53, 19}, {57, 39, 47, 51}, {52, 48, 40, 58}, {47, 41, 59, \
      49}, {50, 60, 42, 48}, {54, 20, 56, 21}, {49, 61, 51, 47}, \
      {50, 48, 52, 62}}",
    classes_src: "\
      {\"Amphichiral\", \"ArchimedeanDual\", \"Canonical\", \
      \"Convex\", \"Isohedron\", \"Rigid\", \"Simple\", \
      \"UniformDual\"}",
  },
  PolyhedronInfo {
    name: "Icosidodecahedron",
    vertex_count: 30,
    edge_count: 60,
    face_count: 32,
    volume: "(45 + 17*Sqrt[5])/6",
    surface_area: "Sqrt[30*(10 + 3*Sqrt[5] + Sqrt[75 + 30*Sqrt[5]])]",
    circumradius: "(1 + Sqrt[5])/2",
    inradius: "Missing[\"NotApplicable\"]",
    vertices_src: "\
      {{0, (-1 - Sqrt[5])/2, 0}, {0, (1 + Sqrt[5])/2, 0}, \
      {Sqrt[1/8 - 1/(8*Sqrt[5])], (-1 - Sqrt[5])/4, \
      -Sqrt[1 + 2/Sqrt[5]]}, {Sqrt[1/8 - 1/(8*Sqrt[5])], \
      (1 + Sqrt[5])/4, -Sqrt[1 + 2/Sqrt[5]]}, \
      {Sqrt[1/8 + 1/(8*Sqrt[5])], (-3 - Sqrt[5])/4, \
      Sqrt[(5 + Sqrt[5])/10]}, {Sqrt[1/8 + 1/(8*Sqrt[5])], \
      (3 + Sqrt[5])/4, Sqrt[(5 + Sqrt[5])/10]}, \
      {Sqrt[1/4 + 1/(2*Sqrt[5])], -1/2, Sqrt[1 + 2/Sqrt[5]]}, \
      {Sqrt[1/4 + 1/(2*Sqrt[5])], 1/2, Sqrt[1 + 2/Sqrt[5]]}, \
      {Sqrt[5/8 + 11/(8*Sqrt[5])], (-1 - Sqrt[5])/4, \
      Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0]}, \
      {Sqrt[5/8 + 11/(8*Sqrt[5])], (1 + Sqrt[5])/4, \
      Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0]}, {-Sqrt[1 + 2/Sqrt[5]], \
      0, Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0]}, \
      {-1/2*Sqrt[1 + 2/Sqrt[5]], -1/2, -Sqrt[1 + 2/Sqrt[5]]}, \
      {-1/2*Sqrt[1 + 2/Sqrt[5]], 1/2, -Sqrt[1 + 2/Sqrt[5]]}, \
      {Sqrt[1 + 2/Sqrt[5]], 0, Sqrt[(5 + Sqrt[5])/10]}, \
      {Sqrt[5/8 + Sqrt[5]/8], -1/8*(1 + Sqrt[5])^2, 0}, \
      {Sqrt[5/8 + Sqrt[5]/8], (3 + Sqrt[5])/4, 0}, \
      {Sqrt[(5 + Sqrt[5])/10], 0, -Sqrt[1 + 2/Sqrt[5]]}, \
      {-1/2*Sqrt[(5 + Sqrt[5])/2], -1/8*(1 + Sqrt[5])^2, 0}, \
      {-1/2*Sqrt[(5 + Sqrt[5])/2], (3 + Sqrt[5])/4, 0}, \
      {-1/2*Sqrt[5 + 2*Sqrt[5]], -1/2, 0}, \
      {-1/2*Sqrt[5 + 2*Sqrt[5]], 1/2, 0}, {Sqrt[5 + 2*Sqrt[5]]/2, \
      -1/2, 0}, {Sqrt[5 + 2*Sqrt[5]]/2, 1/2, 0}, \
      {Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0], 0, Sqrt[1 + 2/Sqrt[5]]}, \
      {Root[1 - 100*#1^2 + 80*#1^4 & , 1, 0], (-1 - Sqrt[5])/4, \
      Sqrt[(5 + Sqrt[5])/10]}, {Root[1 - 100*#1^2 + 80*#1^4 & , 1, \
      0], (1 + Sqrt[5])/4, Sqrt[(5 + Sqrt[5])/10]}, \
      {Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0], (-3 - Sqrt[5])/4, \
      Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0]}, \
      {Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0], (3 + Sqrt[5])/4, \
      Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0]}, \
      {Root[1 - 20*#1^2 + 80*#1^4 & , 2, 0], (-1 - Sqrt[5])/4, \
      Sqrt[1 + 2/Sqrt[5]]}, {Root[1 - 20*#1^2 + 80*#1^4 & , 2, 0], \
      (1 + Sqrt[5])/4, Sqrt[1 + 2/Sqrt[5]]}}",
    faces_src: "\
      {{30, 24, 29, 7, 8}, {26, 24, 30}, {25, 29, 24}, {5, 7, 29}, \
      {14, 8, 7}, {6, 30, 8}, {16, 2, 6}, {19, 21, 26}, {20, 18, \
      25}, {1, 15, 5}, {22, 23, 14}, {2, 19, 26, 30, 6}, {21, 20, \
      25, 24, 26}, {18, 1, 5, 29, 25}, {15, 22, 14, 7, 5}, {23, 16, \
      6, 8, 14}, {12, 13, 4, 17, 3}, {3, 17, 9}, {17, 4, 10}, {4, \
      13, 28}, {13, 12, 11}, {12, 3, 27}, {27, 1, 18}, {9, 22, 15}, \
      {10, 16, 23}, {28, 19, 2}, {11, 20, 21}, {27, 3, 9, 15, 1}, \
      {9, 17, 10, 23, 22}, {10, 4, 28, 2, 16}, {28, 13, 11, 21, \
      19}, {11, 12, 27, 18, 20}}",
    classes_src: "\
      {\"Amphichiral\", \"Archimedean\", \"Canonical\", \"Convex\", \
      \"Equilateral\", \"Quasiregular\", \"Rigid\", \"Rupert\", \
      \"Simple\", \"Uniform\"}",
  },
  PolyhedronInfo {
    name: "DisdyakisTriacontahedron",
    vertex_count: 62,
    edge_count: 180,
    face_count: 120,
    volume: "Sqrt[88590 + 39612*Sqrt[5]]/5",
    surface_area: "Sqrt[22626/5 + 9738/Sqrt[5]]",
    circumradius: "Missing[\"NotApplicable\"]",
    inradius: "Sqrt[3477/964 + 7707/(964*Sqrt[5])]",
    vertices_src: "\
      {{0, 0, -1/2*Sqrt[15 + 33/Sqrt[5]]}, {0, 0, \
      Sqrt[15 + 33/Sqrt[5]]/2}, {0, -1/2*Sqrt[15 + 33/Sqrt[5]], 0}, \
      {0, Sqrt[15 + 33/Sqrt[5]]/2, 0}, {0, \
      -Sqrt[5/6 + 1/(3*Sqrt[5])], \
      -1/2*Sqrt[41/3 + 89/(3*Sqrt[5])]}, {0, \
      Sqrt[5/6 + 1/(3*Sqrt[5])], -1/2*Sqrt[41/3 + 89/(3*Sqrt[5])]}, \
      {0, -Sqrt[5/6 + 1/(3*Sqrt[5])], \
      Sqrt[41/12 + 89/(12*Sqrt[5])]}, {0, \
      Sqrt[5/6 + 1/(3*Sqrt[5])], Sqrt[41/12 + 89/(12*Sqrt[5])]}, \
      {-1/2*Sqrt[15 + 33/Sqrt[5]], 0, 0}, {Sqrt[15 + 33/Sqrt[5]]/2, \
      0, 0}, {-11/Sqrt[255 - 93*Sqrt[5]], \
      -11/Sqrt[255 - 93*Sqrt[5]], -11/Sqrt[255 - 93*Sqrt[5]]}, \
      {11/Sqrt[255 - 93*Sqrt[5]], 11/Sqrt[255 - 93*Sqrt[5]], \
      11/Sqrt[255 - 93*Sqrt[5]]}, \
      {-1/2*Sqrt[41/3 + 89/(3*Sqrt[5])], 0, \
      -Sqrt[5/6 + 1/(3*Sqrt[5])]}, {-Sqrt[5/6 + 1/(3*Sqrt[5])], \
      -1/2*Sqrt[41/3 + 89/(3*Sqrt[5])], 0}, \
      {-1/2*Sqrt[3/2 + 3/Sqrt[5]], -1/4*Sqrt[15 + 33/Sqrt[5]], \
      -1/4*Sqrt[39 + 87/Sqrt[5]]}, {-1/4*Sqrt[39 + 87/Sqrt[5]], \
      -1/2*Sqrt[3/2 + 3/Sqrt[5]], -1/4*Sqrt[15 + 33/Sqrt[5]]}, \
      {-1/4*Sqrt[15 + 33/Sqrt[5]], -1/4*Sqrt[39 + 87/Sqrt[5]], \
      -1/2*Sqrt[3/2 + 3/Sqrt[5]]}, {-1/2*Sqrt[3/2 + 3/Sqrt[5]], \
      Sqrt[15 + 33/Sqrt[5]]/4, -1/4*Sqrt[39 + 87/Sqrt[5]]}, \
      {-1/4*Sqrt[39 + 87/Sqrt[5]], -1/2*Sqrt[3/2 + 3/Sqrt[5]], \
      Sqrt[15 + 33/Sqrt[5]]/4}, {Sqrt[15 + 33/Sqrt[5]]/4, \
      -1/4*Sqrt[39 + 87/Sqrt[5]], -1/2*Sqrt[3/2 + 3/Sqrt[5]]}, \
      {-1/2*Sqrt[41/3 + 89/(3*Sqrt[5])], 0, \
      Sqrt[5/6 + 1/(3*Sqrt[5])]}, {Sqrt[5/6 + 1/(3*Sqrt[5])], \
      -1/2*Sqrt[41/3 + 89/(3*Sqrt[5])], 0}, \
      {-1/10*Sqrt[123 + 267/Sqrt[5]], 0, \
      -1/5*Sqrt[159/2 + 177/Sqrt[5]]}, \
      {-1/5*Sqrt[159/2 + 177/Sqrt[5]], \
      -1/10*Sqrt[123 + 267/Sqrt[5]], 0}, {0, \
      -1/5*Sqrt[159/2 + 177/Sqrt[5]], \
      -1/10*Sqrt[123 + 267/Sqrt[5]]}, {-1/4*Sqrt[39 + 87/Sqrt[5]], \
      Sqrt[3/8 + 3/(4*Sqrt[5])], -1/4*Sqrt[15 + 33/Sqrt[5]]}, \
      {-1/4*Sqrt[15 + 33/Sqrt[5]], -1/4*Sqrt[39 + 87/Sqrt[5]], \
      Sqrt[3/8 + 3/(4*Sqrt[5])]}, {Sqrt[3/8 + 3/(4*Sqrt[5])], \
      -1/4*Sqrt[15 + 33/Sqrt[5]], -1/4*Sqrt[39 + 87/Sqrt[5]]}, \
      {-1/4*Sqrt[39 + 87/Sqrt[5]], Sqrt[3/8 + 3/(4*Sqrt[5])], \
      Sqrt[15 + 33/Sqrt[5]]/4}, {Sqrt[15 + 33/Sqrt[5]]/4, \
      -1/4*Sqrt[39 + 87/Sqrt[5]], Sqrt[3/8 + 3/(4*Sqrt[5])]}, \
      {Sqrt[3/8 + 3/(4*Sqrt[5])], Sqrt[15 + 33/Sqrt[5]]/4, \
      -1/4*Sqrt[39 + 87/Sqrt[5]]}, {-1/5*Sqrt[159/2 + 177/Sqrt[5]], \
      Sqrt[123 + 267/Sqrt[5]]/10, 0}, {0, \
      -1/5*Sqrt[159/2 + 177/Sqrt[5]], Sqrt[123 + 267/Sqrt[5]]/10}, \
      {Sqrt[123 + 267/Sqrt[5]]/10, 0, \
      -1/5*Sqrt[159/2 + 177/Sqrt[5]]}, {-11/Sqrt[255 - 93*Sqrt[5]], \
      -11/Sqrt[255 - 93*Sqrt[5]], 11/Sqrt[255 - 93*Sqrt[5]]}, \
      {-11/Sqrt[255 - 93*Sqrt[5]], 11/Sqrt[255 - 93*Sqrt[5]], \
      -11/Sqrt[255 - 93*Sqrt[5]]}, {11/Sqrt[255 - 93*Sqrt[5]], \
      -11/Sqrt[255 - 93*Sqrt[5]], -11/Sqrt[255 - 93*Sqrt[5]]}, \
      {-11/Sqrt[255 - 93*Sqrt[5]], 11/Sqrt[255 - 93*Sqrt[5]], \
      11/Sqrt[255 - 93*Sqrt[5]]}, {11/Sqrt[255 - 93*Sqrt[5]], \
      -11/Sqrt[255 - 93*Sqrt[5]], 11/Sqrt[255 - 93*Sqrt[5]]}, \
      {11/Sqrt[255 - 93*Sqrt[5]], 11/Sqrt[255 - 93*Sqrt[5]], \
      -11/Sqrt[255 - 93*Sqrt[5]]}, {-1/2*Sqrt[3/2 + 3/Sqrt[5]], \
      -1/4*Sqrt[15 + 33/Sqrt[5]], Sqrt[39 + 87/Sqrt[5]]/4}, \
      {-1/4*Sqrt[15 + 33/Sqrt[5]], Sqrt[39 + 87/Sqrt[5]]/4, \
      -1/2*Sqrt[3/2 + 3/Sqrt[5]]}, {Sqrt[39 + 87/Sqrt[5]]/4, \
      -1/2*Sqrt[3/2 + 3/Sqrt[5]], -1/4*Sqrt[15 + 33/Sqrt[5]]}, \
      {-1/2*Sqrt[3/2 + 3/Sqrt[5]], Sqrt[15 + 33/Sqrt[5]]/4, \
      Sqrt[39 + 87/Sqrt[5]]/4}, {Sqrt[15 + 33/Sqrt[5]]/4, \
      Sqrt[39 + 87/Sqrt[5]]/4, -1/2*Sqrt[3/2 + 3/Sqrt[5]]}, \
      {Sqrt[39 + 87/Sqrt[5]]/4, -1/2*Sqrt[3/2 + 3/Sqrt[5]], \
      Sqrt[15 + 33/Sqrt[5]]/4}, {-1/4*Sqrt[15 + 33/Sqrt[5]], \
      Sqrt[39 + 87/Sqrt[5]]/4, Sqrt[3/8 + 3/(4*Sqrt[5])]}, \
      {Sqrt[3/8 + 3/(4*Sqrt[5])], -1/4*Sqrt[15 + 33/Sqrt[5]], \
      Sqrt[39 + 87/Sqrt[5]]/4}, {Sqrt[39 + 87/Sqrt[5]]/4, \
      Sqrt[3/8 + 3/(4*Sqrt[5])], -1/4*Sqrt[15 + 33/Sqrt[5]]}, \
      {Sqrt[15 + 33/Sqrt[5]]/4, Sqrt[39 + 87/Sqrt[5]]/4, \
      Sqrt[3/8 + 3/(4*Sqrt[5])]}, {Sqrt[3/8 + 3/(4*Sqrt[5])], \
      Sqrt[15 + 33/Sqrt[5]]/4, Sqrt[39 + 87/Sqrt[5]]/4}, \
      {Sqrt[39 + 87/Sqrt[5]]/4, Sqrt[3/8 + 3/(4*Sqrt[5])], \
      Sqrt[15 + 33/Sqrt[5]]/4}, {-Sqrt[5/6 + 1/(3*Sqrt[5])], \
      Sqrt[41/12 + 89/(12*Sqrt[5])], 0}, \
      {Sqrt[41/12 + 89/(12*Sqrt[5])], 0, \
      -Sqrt[5/6 + 1/(3*Sqrt[5])]}, {Sqrt[5/6 + 1/(3*Sqrt[5])], \
      Sqrt[41/12 + 89/(12*Sqrt[5])], 0}, \
      {Sqrt[41/12 + 89/(12*Sqrt[5])], 0, \
      Sqrt[5/6 + 1/(3*Sqrt[5])]}, {-1/10*Sqrt[123 + 267/Sqrt[5]], \
      0, Sqrt[159/2 + 177/Sqrt[5]]/5}, {0, \
      Sqrt[159/2 + 177/Sqrt[5]]/5, -1/10*Sqrt[123 + 267/Sqrt[5]]}, \
      {Sqrt[159/2 + 177/Sqrt[5]]/5, -1/10*Sqrt[123 + 267/Sqrt[5]], \
      0}, {0, Sqrt[159/2 + 177/Sqrt[5]]/5, \
      Sqrt[123 + 267/Sqrt[5]]/10}, {Sqrt[123 + 267/Sqrt[5]]/10, 0, \
      Sqrt[159/2 + 177/Sqrt[5]]/5}, {Sqrt[159/2 + 177/Sqrt[5]]/5, \
      Sqrt[123 + 267/Sqrt[5]]/10, 0}}",
    faces_src: "\
      {{15, 23, 5}, {41, 7, 57}, {6, 23, 18}, {57, 8, 44}, {5, 23, \
      1}, {57, 7, 2}, {1, 23, 6}, {2, 8, 57}, {14, 25, 3}, {3, 33, \
      14}, {25, 15, 5}, {7, 41, 33}, {58, 53, 4}, {4, 53, 60}, {6, \
      18, 58}, {60, 44, 8}, {34, 5, 1}, {7, 61, 2}, {1, 6, 34}, {2, \
      61, 8}, {25, 22, 3}, {3, 22, 33}, {5, 28, 25}, {33, 48, 7}, \
      {55, 58, 4}, {4, 60, 55}, {58, 31, 6}, {8, 51, 60}, {28, 5, \
      34}, {48, 61, 7}, {34, 6, 31}, {8, 61, 51}, {13, 16, 24}, \
      {24, 19, 21}, {32, 26, 13}, {21, 29, 32}, {16, 11, 24}, {24, \
      35, 19}, {32, 36, 26}, {29, 38, 32}, {15, 11, 23}, {57, 35, \
      41}, {36, 18, 23}, {57, 44, 38}, {17, 25, 14}, {14, 33, 27}, \
      {11, 15, 25}, {33, 41, 35}, {18, 36, 58}, {44, 60, 38}, {42, \
      53, 58}, {60, 53, 47}, {23, 11, 16}, {19, 35, 57}, {26, 36, \
      23}, {57, 38, 29}, {9, 13, 24}, {24, 21, 9}, {32, 13, 9}, {9, \
      21, 32}, {11, 25, 17}, {27, 33, 35}, {42, 58, 36}, {47, 38, \
      60}, {23, 16, 13}, {21, 19, 57}, {13, 26, 23}, {57, 29, 21}, \
      {17, 14, 24}, {24, 14, 27}, {32, 53, 42}, {47, 53, 32}, {24, \
      11, 17}, {27, 35, 24}, {42, 36, 32}, {32, 38, 47}, {25, 37, \
      20}, {39, 33, 30}, {40, 58, 45}, {50, 60, 12}, {20, 37, 59}, \
      {59, 39, 30}, {62, 40, 45}, {50, 12, 62}, {59, 54, 10}, {10, \
      56, 59}, {10, 54, 62}, {62, 56, 10}, {59, 22, 20}, {30, 22, \
      59}, {45, 55, 62}, {62, 55, 50}, {20, 22, 25}, {33, 22, 30}, \
      {37, 25, 28}, {33, 39, 48}, {40, 31, 58}, {60, 51, 12}, {45, \
      58, 55}, {55, 60, 50}, {43, 37, 34}, {61, 39, 46}, {34, 40, \
      49}, {52, 12, 61}, {54, 43, 34}, {61, 46, 56}, {34, 49, 54}, \
      {56, 52, 61}, {37, 28, 34}, {61, 48, 39}, {31, 40, 34}, {61, \
      12, 51}, {59, 37, 43}, {46, 39, 59}, {49, 40, 62}, {62, 12, \
      52}, {59, 43, 54}, {56, 46, 59}, {54, 49, 62}, {62, 52, 56}}",
    classes_src: "\
      {\"Amphichiral\", \"ArchimedeanDual\", \"Canonical\", \
      \"Convex\", \"Isohedron\", \"Rigid\", \"Rupert\", \"Simple\", \
      \"UniformDual\"}",
  },
  PolyhedronInfo {
    name: "GreatRhombicosidodecahedron",
    vertex_count: 120,
    edge_count: 180,
    face_count: 62,
    volume: "95 + 50*Sqrt[5]",
    surface_area: "30*(1 + Sqrt[2*(4 + Sqrt[5] + Sqrt[15 + 6*Sqrt[5]])])",
    circumradius: "Sqrt[31 + 12*Sqrt[5]]/2",
    inradius: "Missing[\"NotApplicable\"]",
    vertices_src: "\
      {{-1, (-3 - Sqrt[5])/4, (-7 - 3*Sqrt[5])/4}, {-1, \
      (-3 - Sqrt[5])/4, (7 + 3*Sqrt[5])/4}, {-1, (3 + Sqrt[5])/4, \
      (-7 - 3*Sqrt[5])/4}, {-1, (3 + Sqrt[5])/4, \
      (7 + 3*Sqrt[5])/4}, {-1/2, -1/2, -3/2 - Sqrt[5]}, {-1/2, \
      -1/2, 3/2 + Sqrt[5]}, {-1/2, 1/2, -3/2 - Sqrt[5]}, {-1/2, \
      1/2, 3/2 + Sqrt[5]}, {-1/2, -3/2 - Sqrt[5], -1/2}, {-1/2, \
      -3/2 - Sqrt[5], 1/2}, {-1/2, -1 - Sqrt[5]/2, -2 - Sqrt[5]/2}, \
      {-1/2, -1 - Sqrt[5]/2, (4 + Sqrt[5])/2}, {-1/2, \
      3/2 + Sqrt[5], -1/2}, {-1/2, 3/2 + Sqrt[5], 1/2}, {-1/2, \
      (2 + Sqrt[5])/2, -2 - Sqrt[5]/2}, {-1/2, (2 + Sqrt[5])/2, \
      (4 + Sqrt[5])/2}, {1/2, -1/2, -3/2 - Sqrt[5]}, {1/2, -1/2, \
      3/2 + Sqrt[5]}, {1/2, 1/2, -3/2 - Sqrt[5]}, {1/2, 1/2, \
      3/2 + Sqrt[5]}, {1/2, -3/2 - Sqrt[5], -1/2}, {1/2, \
      -3/2 - Sqrt[5], 1/2}, {1/2, -1 - Sqrt[5]/2, -2 - Sqrt[5]/2}, \
      {1/2, -1 - Sqrt[5]/2, (4 + Sqrt[5])/2}, {1/2, 3/2 + Sqrt[5], \
      -1/2}, {1/2, 3/2 + Sqrt[5], 1/2}, {1/2, (2 + Sqrt[5])/2, \
      -2 - Sqrt[5]/2}, {1/2, (2 + Sqrt[5])/2, (4 + Sqrt[5])/2}, {1, \
      (-3 - Sqrt[5])/4, (-7 - 3*Sqrt[5])/4}, {1, (-3 - Sqrt[5])/4, \
      (7 + 3*Sqrt[5])/4}, {1, (3 + Sqrt[5])/4, (-7 - 3*Sqrt[5])/4}, \
      {1, (3 + Sqrt[5])/4, (7 + 3*Sqrt[5])/4}, {(-7 - 3*Sqrt[5])/4, \
      -1, (-3 - Sqrt[5])/4}, {(-7 - 3*Sqrt[5])/4, -1, \
      (3 + Sqrt[5])/4}, {(-7 - 3*Sqrt[5])/4, 1, (-3 - Sqrt[5])/4}, \
      {(-7 - 3*Sqrt[5])/4, 1, (3 + Sqrt[5])/4}, \
      {(-5 - 3*Sqrt[5])/4, (-5 - Sqrt[5])/4, (-1 - Sqrt[5])/2}, \
      {(-5 - 3*Sqrt[5])/4, (-5 - Sqrt[5])/4, (1 + Sqrt[5])/2}, \
      {(-5 - 3*Sqrt[5])/4, (5 + Sqrt[5])/4, (-1 - Sqrt[5])/2}, \
      {(-5 - 3*Sqrt[5])/4, (5 + Sqrt[5])/4, (1 + Sqrt[5])/2}, \
      {(-5 - Sqrt[5])/4, (-1 - Sqrt[5])/2, (-5 - 3*Sqrt[5])/4}, \
      {(-5 - Sqrt[5])/4, (-1 - Sqrt[5])/2, (5 + 3*Sqrt[5])/4}, \
      {(-5 - Sqrt[5])/4, (1 + Sqrt[5])/2, (-5 - 3*Sqrt[5])/4}, \
      {(-5 - Sqrt[5])/4, (1 + Sqrt[5])/2, (5 + 3*Sqrt[5])/4}, \
      {(-3 - Sqrt[5])/4, (-7 - 3*Sqrt[5])/4, -1}, \
      {(-3 - Sqrt[5])/4, (-7 - 3*Sqrt[5])/4, 1}, {(-3 - Sqrt[5])/4, \
      (-3*(1 + Sqrt[5]))/4, (-3 - Sqrt[5])/2}, {(-3 - Sqrt[5])/4, \
      (-3*(1 + Sqrt[5]))/4, (3 + Sqrt[5])/2}, {(-3 - Sqrt[5])/4, \
      (3*(1 + Sqrt[5]))/4, (-3 - Sqrt[5])/2}, {(-3 - Sqrt[5])/4, \
      (3*(1 + Sqrt[5]))/4, (3 + Sqrt[5])/2}, {(-3 - Sqrt[5])/4, \
      (7 + 3*Sqrt[5])/4, -1}, {(-3 - Sqrt[5])/4, (7 + 3*Sqrt[5])/4, \
      1}, {(-3 - Sqrt[5])/2, (-3 - Sqrt[5])/4, \
      (-3*(1 + Sqrt[5]))/4}, {(-3 - Sqrt[5])/2, (-3 - Sqrt[5])/4, \
      (3*(1 + Sqrt[5]))/4}, {(-3 - Sqrt[5])/2, (3 + Sqrt[5])/4, \
      (-3*(1 + Sqrt[5]))/4}, {(-3 - Sqrt[5])/2, (3 + Sqrt[5])/4, \
      (3*(1 + Sqrt[5]))/4}, {-3/2 - Sqrt[5], -1/2, -1/2}, \
      {-3/2 - Sqrt[5], -1/2, 1/2}, {-3/2 - Sqrt[5], 1/2, -1/2}, \
      {-3/2 - Sqrt[5], 1/2, 1/2}, {(-1 - Sqrt[5])/2, \
      (-5 - 3*Sqrt[5])/4, (-5 - Sqrt[5])/4}, {(-1 - Sqrt[5])/2, \
      (-5 - 3*Sqrt[5])/4, (5 + Sqrt[5])/4}, {(-1 - Sqrt[5])/2, \
      (5 + 3*Sqrt[5])/4, (-5 - Sqrt[5])/4}, {(-1 - Sqrt[5])/2, \
      (5 + 3*Sqrt[5])/4, (5 + Sqrt[5])/4}, {-2 - Sqrt[5]/2, -1/2, \
      -1 - Sqrt[5]/2}, {-2 - Sqrt[5]/2, -1/2, (2 + Sqrt[5])/2}, \
      {-2 - Sqrt[5]/2, 1/2, -1 - Sqrt[5]/2}, {-2 - Sqrt[5]/2, 1/2, \
      (2 + Sqrt[5])/2}, {-1 - Sqrt[5]/2, -2 - Sqrt[5]/2, -1/2}, \
      {-1 - Sqrt[5]/2, -2 - Sqrt[5]/2, 1/2}, {-1 - Sqrt[5]/2, \
      (4 + Sqrt[5])/2, -1/2}, {-1 - Sqrt[5]/2, (4 + Sqrt[5])/2, \
      1/2}, {(-3*(1 + Sqrt[5]))/4, (-3 - Sqrt[5])/2, \
      (-3 - Sqrt[5])/4}, {(-3*(1 + Sqrt[5]))/4, (-3 - Sqrt[5])/2, \
      (3 + Sqrt[5])/4}, {(-3*(1 + Sqrt[5]))/4, (3 + Sqrt[5])/2, \
      (-3 - Sqrt[5])/4}, {(-3*(1 + Sqrt[5]))/4, (3 + Sqrt[5])/2, \
      (3 + Sqrt[5])/4}, {(1 + Sqrt[5])/2, (-5 - 3*Sqrt[5])/4, \
      (-5 - Sqrt[5])/4}, {(1 + Sqrt[5])/2, (-5 - 3*Sqrt[5])/4, \
      (5 + Sqrt[5])/4}, {(1 + Sqrt[5])/2, (5 + 3*Sqrt[5])/4, \
      (-5 - Sqrt[5])/4}, {(1 + Sqrt[5])/2, (5 + 3*Sqrt[5])/4, \
      (5 + Sqrt[5])/4}, {(3*(1 + Sqrt[5]))/4, (-3 - Sqrt[5])/2, \
      (-3 - Sqrt[5])/4}, {(3*(1 + Sqrt[5]))/4, (-3 - Sqrt[5])/2, \
      (3 + Sqrt[5])/4}, {(3*(1 + Sqrt[5]))/4, (3 + Sqrt[5])/2, \
      (-3 - Sqrt[5])/4}, {(3*(1 + Sqrt[5]))/4, (3 + Sqrt[5])/2, \
      (3 + Sqrt[5])/4}, {3/2 + Sqrt[5], -1/2, -1/2}, \
      {3/2 + Sqrt[5], -1/2, 1/2}, {3/2 + Sqrt[5], 1/2, -1/2}, \
      {3/2 + Sqrt[5], 1/2, 1/2}, {(2 + Sqrt[5])/2, -2 - Sqrt[5]/2, \
      -1/2}, {(2 + Sqrt[5])/2, -2 - Sqrt[5]/2, 1/2}, \
      {(2 + Sqrt[5])/2, (4 + Sqrt[5])/2, -1/2}, {(2 + Sqrt[5])/2, \
      (4 + Sqrt[5])/2, 1/2}, {(3 + Sqrt[5])/4, (-7 - 3*Sqrt[5])/4, \
      -1}, {(3 + Sqrt[5])/4, (-7 - 3*Sqrt[5])/4, 1}, \
      {(3 + Sqrt[5])/4, (-3*(1 + Sqrt[5]))/4, (-3 - Sqrt[5])/2}, \
      {(3 + Sqrt[5])/4, (-3*(1 + Sqrt[5]))/4, (3 + Sqrt[5])/2}, \
      {(3 + Sqrt[5])/4, (3*(1 + Sqrt[5]))/4, (-3 - Sqrt[5])/2}, \
      {(3 + Sqrt[5])/4, (3*(1 + Sqrt[5]))/4, (3 + Sqrt[5])/2}, \
      {(3 + Sqrt[5])/4, (7 + 3*Sqrt[5])/4, -1}, {(3 + Sqrt[5])/4, \
      (7 + 3*Sqrt[5])/4, 1}, {(3 + Sqrt[5])/2, (-3 - Sqrt[5])/4, \
      (-3*(1 + Sqrt[5]))/4}, {(3 + Sqrt[5])/2, (-3 - Sqrt[5])/4, \
      (3*(1 + Sqrt[5]))/4}, {(3 + Sqrt[5])/2, (3 + Sqrt[5])/4, \
      (-3*(1 + Sqrt[5]))/4}, {(3 + Sqrt[5])/2, (3 + Sqrt[5])/4, \
      (3*(1 + Sqrt[5]))/4}, {(4 + Sqrt[5])/2, -1/2, \
      -1 - Sqrt[5]/2}, {(4 + Sqrt[5])/2, -1/2, (2 + Sqrt[5])/2}, \
      {(4 + Sqrt[5])/2, 1/2, -1 - Sqrt[5]/2}, {(4 + Sqrt[5])/2, \
      1/2, (2 + Sqrt[5])/2}, {(5 + Sqrt[5])/4, (-1 - Sqrt[5])/2, \
      (-5 - 3*Sqrt[5])/4}, {(5 + Sqrt[5])/4, (-1 - Sqrt[5])/2, \
      (5 + 3*Sqrt[5])/4}, {(5 + Sqrt[5])/4, (1 + Sqrt[5])/2, \
      (-5 - 3*Sqrt[5])/4}, {(5 + Sqrt[5])/4, (1 + Sqrt[5])/2, \
      (5 + 3*Sqrt[5])/4}, {(5 + 3*Sqrt[5])/4, (-5 - Sqrt[5])/4, \
      (-1 - Sqrt[5])/2}, {(5 + 3*Sqrt[5])/4, (-5 - Sqrt[5])/4, \
      (1 + Sqrt[5])/2}, {(5 + 3*Sqrt[5])/4, (5 + Sqrt[5])/4, \
      (-1 - Sqrt[5])/2}, {(5 + 3*Sqrt[5])/4, (5 + Sqrt[5])/4, \
      (1 + Sqrt[5])/2}, {(7 + 3*Sqrt[5])/4, -1, (-3 - Sqrt[5])/4}, \
      {(7 + 3*Sqrt[5])/4, -1, (3 + Sqrt[5])/4}, {(7 + 3*Sqrt[5])/4, \
      1, (-3 - Sqrt[5])/4}, {(7 + 3*Sqrt[5])/4, 1, \
      (3 + Sqrt[5])/4}}",
    faces_src: "\
      {{2, 6, 8, 4, 44, 56, 68, 66, 54, 42}, {109, 29, 17, 19, 31, \
      111, 103, 107, 105, 101}, {24, 30, 18, 6, 2, 12}, {7, 3, 15, \
      27, 31, 19}, {58, 57, 33, 37, 73, 69, 70, 74, 38, 34}, {84, \
      116, 120, 88, 87, 119, 115, 83, 91, 92}, {90, 89, 81, 113, \
      117, 85, 86, 118, 114, 82}, {36, 40, 76, 72, 71, 75, 39, 35, \
      59, 60}, {5, 17, 29, 23, 11, 1}, {4, 8, 20, 32, 28, 16}, {67, \
      55, 43, 3, 7, 5, 1, 41, 53, 65}, {18, 30, 110, 102, 106, 108, \
      104, 112, 32, 20}, {79, 83, 115, 103, 111, 97}, {38, 74, 62, \
      48, 42, 54}, {4, 16, 50, 44}, {23, 29, 109, 95}, {96, 110, \
      30, 24}, {43, 49, 15, 3}, {53, 41, 47, 61, 73, 37}, {98, 112, \
      104, 116, 84, 80}, {69, 45, 9, 10, 46, 70}, {26, 100, 92, 91, \
      99, 25}, {82, 114, 102, 110, 96, 78}, {55, 39, 75, 63, 49, \
      43}, {1, 11, 47, 41}, {28, 32, 112, 98}, {61, 47, 11, 23, 95, \
      77, 93, 21, 9, 45}, {50, 16, 28, 98, 80, 100, 26, 14, 52, \
      64}, {97, 111, 31, 27}, {42, 48, 12, 2}, {44, 50, 64, 76, 40, \
      56}, {77, 95, 109, 101, 113, 81}, {63, 51, 13, 25, 99, 79, \
      97, 27, 15, 49}, {46, 10, 22, 94, 78, 96, 24, 12, 48, 62}, \
      {52, 14, 13, 51, 71, 72}, {22, 21, 93, 89, 90, 94}, {115, \
      119, 107, 103}, {34, 38, 54, 66}, {71, 51, 63, 75}, {94, 90, \
      82, 78}, {114, 118, 106, 102}, {35, 39, 55, 67}, {70, 46, 62, \
      74}, {99, 91, 83, 79}, {65, 53, 37, 33}, {104, 108, 120, \
      116}, {77, 81, 89, 93}, {76, 64, 52, 72}, {59, 35, 67, 65, \
      33, 57}, {106, 118, 86, 88, 120, 108}, {68, 56, 40, 36}, \
      {101, 105, 117, 113}, {80, 84, 92, 100}, {73, 61, 45, 69}, \
      {34, 66, 68, 36, 60, 58}, {105, 107, 119, 87, 85, 117}, {7, \
      19, 17, 5}, {6, 18, 20, 8}, {14, 26, 25, 13}, {9, 21, 22, \
      10}, {58, 60, 59, 57}, {85, 87, 88, 86}}",
    classes_src: "\
      {\"Amphichiral\", \"Archimedean\", \"Canonical\", \"Convex\", \
      \"Equilateral\", \"Rigid\", \"Rupert\", \"Simple\", \
      \"Uniform\", \"Zalgaller\", \"Zonohedron\"}",
  },
  PolyhedronInfo {
    name: "PentakisDodecahedron",
    vertex_count: 32,
    edge_count: 90,
    face_count: 60,
    volume: "(5*(41 + 25*Sqrt[5]))/36",
    surface_area: "(5*Sqrt[(421 + 63*Sqrt[5])/2])/3",
    circumradius: "Missing[\"NotApplicable\"]",
    inradius: "Root[361 - 3816*#1^2 + 1744*#1^4 & , 4, 0]",
    vertices_src: "\
      {{Root[361 - 765*#1^2 + 405*#1^4 & , 2, 0], 0, \
      Sqrt[29 + 62/Sqrt[5]]/6}, {Sqrt[1 + 2/Sqrt[5]], 0, \
      Sqrt[1 + 2/Sqrt[5]]/2}, {0, 0, Sqrt[5 + 2*Sqrt[5]]/2}, \
      {Root[1 - 100*#1^2 + 80*#1^4 & , 1, 0], (-1 - Sqrt[5])/4, \
      Sqrt[1 + 2/Sqrt[5]]/2}, {Root[1 - 100*#1^2 + 80*#1^4 & , 1, \
      0], (1 + Sqrt[5])/4, Sqrt[1 + 2/Sqrt[5]]/2}, {0, 0, \
      -1/2*Sqrt[5 + 2*Sqrt[5]]}, {Sqrt[1/8 + 1/(8*Sqrt[5])], \
      (-3 - Sqrt[5])/4, Sqrt[1 + 2/Sqrt[5]]/2}, \
      {Sqrt[1/8 + 1/(8*Sqrt[5])], (3 + Sqrt[5])/4, \
      Sqrt[1 + 2/Sqrt[5]]/2}, {-1/6*Sqrt[13 - 22/Sqrt[5]], \
      (-1 - 2*Sqrt[5])/6, Sqrt[29 + 62/Sqrt[5]]/6}, \
      {-1/6*Sqrt[13 - 22/Sqrt[5]], (1 + 2*Sqrt[5])/6, \
      Sqrt[29 + 62/Sqrt[5]]/6}, {Sqrt[25/72 + 41/(72*Sqrt[5])], \
      (9 - Sqrt[5])/12, Sqrt[29 + 62/Sqrt[5]]/6}, \
      {Sqrt[25/72 + 41/(72*Sqrt[5])], (-9 + Sqrt[5])/12, \
      Sqrt[29 + 62/Sqrt[5]]/6}, {Sqrt[(85 - Sqrt[5])/10]/3, 0, \
      -1/6*Sqrt[29 + 62/Sqrt[5]]}, {Sqrt[13 - 22/Sqrt[5]]/6, \
      (-1 - 2*Sqrt[5])/6, -1/6*Sqrt[29 + 62/Sqrt[5]]}, \
      {Sqrt[13 - 22/Sqrt[5]]/6, (1 + 2*Sqrt[5])/6, \
      -1/6*Sqrt[29 + 62/Sqrt[5]]}, \
      {-1/6*Sqrt[25/2 + 41/(2*Sqrt[5])], (9 - Sqrt[5])/12, \
      -1/6*Sqrt[29 + 62/Sqrt[5]]}, \
      {-1/6*Sqrt[25/2 + 41/(2*Sqrt[5])], (-9 + Sqrt[5])/12, \
      -1/6*Sqrt[29 + 62/Sqrt[5]]}, {Sqrt[29 + 62/Sqrt[5]]/6, \
      (-1 - 2*Sqrt[5])/6, Sqrt[13 - 22/Sqrt[5]]/6}, \
      {-1/6*Sqrt[(85 - Sqrt[5])/10], (-11 - 3*Sqrt[5])/12, \
      Sqrt[13 - 22/Sqrt[5]]/6}, {Sqrt[29 + 62/Sqrt[5]]/6, \
      (1 + 2*Sqrt[5])/6, Sqrt[13 - 22/Sqrt[5]]/6}, \
      {-1/6*Sqrt[(85 - Sqrt[5])/10], (11 + 3*Sqrt[5])/12, \
      Sqrt[13 - 22/Sqrt[5]]/6}, {-1/3*Sqrt[25/2 + 41/(2*Sqrt[5])], \
      0, Sqrt[13 - 22/Sqrt[5]]/6}, {Sqrt[25/18 + 41/(18*Sqrt[5])], \
      0, -1/6*Sqrt[13 - 22/Sqrt[5]]}, {Sqrt[(85 - Sqrt[5])/10]/6, \
      (-11 - 3*Sqrt[5])/12, -1/6*Sqrt[13 - 22/Sqrt[5]]}, \
      {Sqrt[(85 - Sqrt[5])/10]/6, (11 + 3*Sqrt[5])/12, \
      -1/6*Sqrt[13 - 22/Sqrt[5]]}, {-1/6*Sqrt[29 + 62/Sqrt[5]], \
      (-1 - 2*Sqrt[5])/6, -1/6*Sqrt[13 - 22/Sqrt[5]]}, \
      {-1/6*Sqrt[29 + 62/Sqrt[5]], (1 + 2*Sqrt[5])/6, \
      -1/6*Sqrt[13 - 22/Sqrt[5]]}, {Sqrt[5/8 + 11/(8*Sqrt[5])], \
      (-1 - Sqrt[5])/4, -1/2*Sqrt[1 + 2/Sqrt[5]]}, \
      {-Sqrt[1 + 2/Sqrt[5]], 0, -1/2*Sqrt[1 + 2/Sqrt[5]]}, \
      {Sqrt[5/8 + 11/(8*Sqrt[5])], (1 + Sqrt[5])/4, \
      -1/2*Sqrt[1 + 2/Sqrt[5]]}, {Root[1 - 20*#1^2 + 80*#1^4 & , 1, \
      0], (-3 - Sqrt[5])/4, -1/2*Sqrt[1 + 2/Sqrt[5]]}, \
      {Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0], (3 + Sqrt[5])/4, \
      -1/2*Sqrt[1 + 2/Sqrt[5]]}}",
    faces_src: "\
      {{7, 9, 19}, {21, 10, 8}, {31, 14, 24}, {25, 15, 32}, {6, 14, \
      17}, {24, 19, 31}, {6, 16, 15}, {21, 25, 32}, {9, 12, 3}, {7, \
      19, 24}, {3, 11, 10}, {8, 25, 21}, {13, 14, 6}, {6, 15, 13}, \
      {7, 24, 18}, {8, 20, 25}, {4, 1, 22}, {22, 1, 5}, {9, 1, 4}, \
      {1, 10, 5}, {26, 17, 31}, {32, 16, 27}, {3, 1, 9}, {3, 10, \
      1}, {28, 14, 13}, {13, 15, 30}, {18, 12, 7}, {8, 11, 20}, \
      {16, 17, 29}, {11, 12, 2}, {19, 26, 31}, {27, 21, 32}, {29, \
      17, 26}, {27, 16, 29}, {28, 13, 23}, {23, 13, 30}, {23, 18, \
      28}, {30, 20, 23}, {2, 18, 23}, {23, 20, 2}, {19, 9, 4}, {5, \
      10, 21}, {24, 14, 28}, {30, 15, 25}, {26, 22, 29}, {22, 27, \
      29}, {4, 26, 19}, {5, 21, 27}, {18, 24, 28}, {25, 20, 30}, \
      {2, 12, 18}, {20, 11, 2}, {3, 12, 11}, {4, 22, 26}, {5, 27, \
      22}, {6, 17, 16}, {17, 14, 31}, {15, 16, 32}, {7, 12, 9}, \
      {10, 11, 8}}",
    classes_src: "\
      {\"Amphichiral\", \"ArchimedeanDual\", \"Canonical\", \
      \"Convex\", \"Isohedron\", \"Rigid\", \"Rupert\", \"Simple\", \
      \"UniformDual\"}",
  },
  PolyhedronInfo {
    name: "TruncatedDodecahedron",
    vertex_count: 60,
    edge_count: 90,
    face_count: 32,
    volume: "(5*(99 + 47*Sqrt[5]))/12",
    surface_area: "5*Sqrt[3*(61 + 24*Sqrt[5] + 4*Sqrt[15 + 6*Sqrt[5]])]",
    circumradius: "Sqrt[74 + 30*Sqrt[5]]/4",
    inradius: "Missing[\"NotApplicable\"]",
    vertices_src: "\
      {{0, (-1 - Sqrt[5])/2, Sqrt[25/8 + (11*Sqrt[5])/8]}, {0, \
      (-1 - Sqrt[5])/2, Root[5 - 100*#1^2 + 16*#1^4 & , 1, 0]}, {0, \
      (1 + Sqrt[5])/2, Sqrt[25/8 + (11*Sqrt[5])/8]}, {0, \
      (1 + Sqrt[5])/2, Root[5 - 100*#1^2 + 16*#1^4 & , 1, 0]}, \
      {Sqrt[1/8 + 1/(8*Sqrt[5])], (-5 - 3*Sqrt[5])/4, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 3, 0]}, \
      {Sqrt[1/8 + 1/(8*Sqrt[5])], (5 + 3*Sqrt[5])/4, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 3, 0]}, \
      {Sqrt[1/4 + 1/(2*Sqrt[5])], -1 - Sqrt[5]/2, \
      Sqrt[17/8 + 31/(8*Sqrt[5])]}, {Sqrt[1/4 + 1/(2*Sqrt[5])], \
      (2 + Sqrt[5])/2, Sqrt[17/8 + 31/(8*Sqrt[5])]}, \
      {-2*Sqrt[1 + 2/Sqrt[5]], 0, Root[1 - 100*#1^2 + 80*#1^4 & , \
      1, 0]}, {(-3*Sqrt[1 + 2/Sqrt[5]])/2, -1 - Sqrt[5]/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 3, 0]}, \
      {(-3*Sqrt[1 + 2/Sqrt[5]])/2, (2 + Sqrt[5])/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 3, 0]}, {-Sqrt[1 + 2/Sqrt[5]], \
      (-3 - Sqrt[5])/2, Root[1 - 20*#1^2 + 80*#1^4 & , 2, 0]}, \
      {-Sqrt[1 + 2/Sqrt[5]], (3 + Sqrt[5])/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 2, 0]}, \
      {-1/2*Sqrt[1 + 2/Sqrt[5]], -1 - Sqrt[5]/2, \
      -Sqrt[17/8 + 31/(8*Sqrt[5])]}, {-1/2*Sqrt[1 + 2/Sqrt[5]], \
      (2 + Sqrt[5])/2, -Sqrt[17/8 + 31/(8*Sqrt[5])]}, \
      {Sqrt[1 + 2/Sqrt[5]], (-3 - Sqrt[5])/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 3, 0]}, {Sqrt[1 + 2/Sqrt[5]], \
      (3 + Sqrt[5])/2, Root[1 - 20*#1^2 + 80*#1^4 & , 3, 0]}, \
      {2*Sqrt[1 + 2/Sqrt[5]], 0, Sqrt[5/8 + 11/(8*Sqrt[5])]}, \
      {Sqrt[13/8 + 29/(8*Sqrt[5])], (-3 - Sqrt[5])/4, \
      -Sqrt[17/8 + 31/(8*Sqrt[5])]}, {Sqrt[13/8 + 29/(8*Sqrt[5])], \
      (3 + Sqrt[5])/4, -Sqrt[17/8 + 31/(8*Sqrt[5])]}, \
      {Sqrt[9/4 + 9/(2*Sqrt[5])], -1 - Sqrt[5]/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 2, 0]}, \
      {Sqrt[9/4 + 9/(2*Sqrt[5])], (2 + Sqrt[5])/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 2, 0]}, \
      {Sqrt[5/2 + 11/(2*Sqrt[5])], 0, Sqrt[17/8 + 31/(8*Sqrt[5])]}, \
      {Sqrt[5/2 + 11/(2*Sqrt[5])], (-1 - Sqrt[5])/2, \
      Root[1 - 100*#1^2 + 80*#1^4 & , 1, 0]}, \
      {Sqrt[5/2 + 11/(2*Sqrt[5])], (1 + Sqrt[5])/2, \
      Root[1 - 100*#1^2 + 80*#1^4 & , 1, 0]}, \
      {-Sqrt[29/8 + 61/(8*Sqrt[5])], (-3 - Sqrt[5])/4, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 3, 0]}, \
      {-Sqrt[29/8 + 61/(8*Sqrt[5])], (3 + Sqrt[5])/4, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 3, 0]}, \
      {Sqrt[29/8 + 61/(8*Sqrt[5])], (-3 - Sqrt[5])/4, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 2, 0]}, \
      {Sqrt[29/8 + 61/(8*Sqrt[5])], (3 + Sqrt[5])/4, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 2, 0]}, \
      {Sqrt[17/4 + 19/(2*Sqrt[5])], -1/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 3, 0]}, \
      {Sqrt[17/4 + 19/(2*Sqrt[5])], 1/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 3, 0]}, \
      {-1/2*Sqrt[17 + 38/Sqrt[5]], -1/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 2, 0]}, \
      {-1/2*Sqrt[17 + 38/Sqrt[5]], 1/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 2, 0]}, \
      {Sqrt[5/8 + Sqrt[5]/8], (-3 - Sqrt[5])/4, \
      Sqrt[25/8 + (11*Sqrt[5])/8]}, {Sqrt[5/8 + Sqrt[5]/8], \
      (-3 - Sqrt[5])/4, Root[5 - 100*#1^2 + 16*#1^4 & , 1, 0]}, \
      {Sqrt[5/8 + Sqrt[5]/8], (3 + Sqrt[5])/4, \
      Sqrt[25/8 + (11*Sqrt[5])/8]}, {Sqrt[5/8 + Sqrt[5]/8], \
      (3 + Sqrt[5])/4, Root[5 - 100*#1^2 + 16*#1^4 & , 1, 0]}, \
      {Sqrt[(5 + Sqrt[5])/10], (-3 - Sqrt[5])/2, \
      Sqrt[5/8 + 11/(8*Sqrt[5])]}, {Sqrt[(5 + Sqrt[5])/10], \
      (3 + Sqrt[5])/2, Sqrt[5/8 + 11/(8*Sqrt[5])]}, \
      {-1/2*Sqrt[(5 + Sqrt[5])/2], (-3 - Sqrt[5])/4, \
      Sqrt[25/8 + (11*Sqrt[5])/8]}, {-1/2*Sqrt[(5 + Sqrt[5])/2], \
      (-3 - Sqrt[5])/4, Root[5 - 100*#1^2 + 16*#1^4 & , 1, 0]}, \
      {-1/2*Sqrt[(5 + Sqrt[5])/2], (3 + Sqrt[5])/4, \
      Sqrt[25/8 + (11*Sqrt[5])/8]}, {-1/2*Sqrt[(5 + Sqrt[5])/2], \
      (3 + Sqrt[5])/4, Root[5 - 100*#1^2 + 16*#1^4 & , 1, 0]}, \
      {-1/2*Sqrt[5 + 2*Sqrt[5]], -1/2, \
      Sqrt[25/8 + (11*Sqrt[5])/8]}, {-1/2*Sqrt[5 + 2*Sqrt[5]], \
      -1/2, Root[5 - 100*#1^2 + 16*#1^4 & , 1, 0]}, \
      {-1/2*Sqrt[5 + 2*Sqrt[5]], 1/2, Sqrt[25/8 + (11*Sqrt[5])/8]}, \
      {-1/2*Sqrt[5 + 2*Sqrt[5]], 1/2, \
      Root[5 - 100*#1^2 + 16*#1^4 & , 1, 0]}, \
      {Sqrt[5 + 2*Sqrt[5]]/2, -1/2, Sqrt[25/8 + (11*Sqrt[5])/8]}, \
      {Sqrt[5 + 2*Sqrt[5]]/2, -1/2, Root[5 - 100*#1^2 + 16*#1^4 & , \
      1, 0]}, {Sqrt[5 + 2*Sqrt[5]]/2, 1/2, \
      Sqrt[25/8 + (11*Sqrt[5])/8]}, {Sqrt[5 + 2*Sqrt[5]]/2, 1/2, \
      Root[5 - 100*#1^2 + 16*#1^4 & , 1, 0]}, \
      {Root[1 - 25*#1^2 + 5*#1^4 & , 1, 0], 0, \
      -Sqrt[17/8 + 31/(8*Sqrt[5])]}, {Root[1 - 25*#1^2 + 5*#1^4 & , \
      1, 0], (-1 - Sqrt[5])/2, Sqrt[5/8 + 11/(8*Sqrt[5])]}, \
      {Root[1 - 25*#1^2 + 5*#1^4 & , 1, 0], (1 + Sqrt[5])/2, \
      Sqrt[5/8 + 11/(8*Sqrt[5])]}, {Root[1 - 5*#1^2 + 5*#1^4 & , 1, \
      0], (-3 - Sqrt[5])/2, Root[1 - 100*#1^2 + 80*#1^4 & , 1, 0]}, \
      {Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0], (3 + Sqrt[5])/2, \
      Root[1 - 100*#1^2 + 80*#1^4 & , 1, 0]}, \
      {Root[1 - 260*#1^2 + 80*#1^4 & , 1, 0], (-3 - Sqrt[5])/4, \
      Sqrt[17/8 + 31/(8*Sqrt[5])]}, \
      {Root[1 - 260*#1^2 + 80*#1^4 & , 1, 0], (3 + Sqrt[5])/4, \
      Sqrt[17/8 + 31/(8*Sqrt[5])]}, {Root[1 - 20*#1^2 + 80*#1^4 & , \
      1, 0], (-5 - 3*Sqrt[5])/4, Root[1 - 20*#1^2 + 80*#1^4 & , 2, \
      0]}, {Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0], \
      (5 + 3*Sqrt[5])/4, Root[1 - 20*#1^2 + 80*#1^4 & , 2, 0]}}",
    faces_src: "\
      {{3, 42, 46, 44, 40, 1, 34, 48, 50, 36}, {47, 43, 4, 37, 51, \
      49, 35, 2, 41, 45}, {2, 35, 19, 24, 21, 16, 5, 59, 55, 14}, \
      {49, 51, 20, 25, 29, 31, 30, 28, 24, 19}, {37, 4, 15, 56, 60, \
      6, 17, 22, 25, 20}, {43, 47, 52, 9, 33, 27, 11, 13, 56, 15}, \
      {45, 41, 14, 55, 12, 10, 26, 32, 9, 52}, {6, 60, 13, 11, 54, \
      58, 42, 3, 8, 39}, {27, 33, 32, 26, 53, 57, 44, 46, 58, 54}, \
      {10, 12, 59, 5, 38, 7, 1, 40, 57, 53}, {16, 21, 28, 30, 18, \
      23, 48, 34, 7, 38}, {31, 29, 22, 17, 39, 8, 36, 50, 23, 18}, \
      {9, 32, 33}, {18, 30, 31}, {47, 45, 52}, {50, 48, 23}, {10, \
      53, 26}, {27, 54, 11}, {21, 24, 28}, {29, 25, 22}, {40, 44, \
      57}, {58, 46, 42}, {35, 49, 19}, {20, 51, 37}, {12, 55, 59}, \
      {60, 56, 13}, {41, 2, 14}, {15, 4, 43}, {34, 1, 7}, {8, 3, \
      36}, {38, 5, 16}, {17, 6, 39}}",
    classes_src: "\
      {\"Amphichiral\", \"Archimedean\", \"Canonical\", \"Convex\", \
      \"Equilateral\", \"Rigid\", \"Rupert\", \"Simple\", \
      \"Uniform\", \"Zalgaller\"}",
  },
  PolyhedronInfo {
    name: "TruncatedIcosahedron",
    vertex_count: 60,
    edge_count: 90,
    face_count: 32,
    volume: "(125 + 43*Sqrt[5])/4",
    surface_area: "3*Sqrt[5*(65 + 2*Sqrt[5] + 4*Sqrt[75 + 30*Sqrt[5]])]",
    circumradius: "Sqrt[58 + 18*Sqrt[5]]/4",
    inradius: "Missing[\"NotApplicable\"]",
    vertices_src: "\
      {{-1/2*Sqrt[1 - 2/Sqrt[5]], -1 - Sqrt[5]/2, \
      Sqrt[9/8 + 9/(8*Sqrt[5])]}, {-1/2*Sqrt[1 - 2/Sqrt[5]], \
      (2 + Sqrt[5])/2, Sqrt[9/8 + 9/(8*Sqrt[5])]}, \
      {Sqrt[1 - 2/Sqrt[5]]/2, -1 - Sqrt[5]/2, \
      (-3*Sqrt[(5 + Sqrt[5])/10])/2}, {Sqrt[1 - 2/Sqrt[5]]/2, \
      (2 + Sqrt[5])/2, (-3*Sqrt[(5 + Sqrt[5])/10])/2}, \
      {-1/4*Sqrt[2 - 2/Sqrt[5]], (1 - Sqrt[5])^(-1), \
      -Sqrt[25/8 + 41/(8*Sqrt[5])]}, {-1/4*Sqrt[2 - 2/Sqrt[5]], \
      (-3*(1 + Sqrt[5]))/4, Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0]}, \
      {-1/4*Sqrt[2 - 2/Sqrt[5]], (1 + Sqrt[5])/4, \
      -Sqrt[25/8 + 41/(8*Sqrt[5])]}, {-1/4*Sqrt[2 - 2/Sqrt[5]], \
      (3*(1 + Sqrt[5]))/4, Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0]}, \
      {Sqrt[2 - 2/Sqrt[5]]/4, (1 - Sqrt[5])^(-1), \
      Sqrt[25/8 + 41/(8*Sqrt[5])]}, {Sqrt[2 - 2/Sqrt[5]]/4, \
      (-3*(1 + Sqrt[5]))/4, Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
      {Sqrt[2 - 2/Sqrt[5]]/4, (1 + Sqrt[5])/4, \
      Sqrt[25/8 + 41/(8*Sqrt[5])]}, {Sqrt[2 - 2/Sqrt[5]]/4, \
      (3*(1 + Sqrt[5]))/4, Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
      {Sqrt[1/4 + 1/(2*Sqrt[5])], -1/2, \
      -Sqrt[25/8 + 41/(8*Sqrt[5])]}, {Sqrt[1/4 + 1/(2*Sqrt[5])], \
      1/2, -Sqrt[25/8 + 41/(8*Sqrt[5])]}, \
      {Sqrt[5/4 + 1/(2*Sqrt[5])], -1 - Sqrt[5]/2, \
      Sqrt[1/8 + 1/(8*Sqrt[5])]}, {Sqrt[5/4 + 1/(2*Sqrt[5])], \
      (2 + Sqrt[5])/2, Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
      {(-3*Sqrt[1 + 2/Sqrt[5]])/2, -1/2, \
      Sqrt[9/8 + 9/(8*Sqrt[5])]}, {(-3*Sqrt[1 + 2/Sqrt[5]])/2, 1/2, \
      Sqrt[9/8 + 9/(8*Sqrt[5])]}, {-Sqrt[1 + 2/Sqrt[5]], -1, \
      Sqrt[26 + 58/Sqrt[5]]/4}, {-Sqrt[1 + 2/Sqrt[5]], 1, \
      Sqrt[26 + 58/Sqrt[5]]/4}, {-Sqrt[1 + 2/Sqrt[5]], \
      -2/(-1 + Sqrt[5]), (-3*Sqrt[(5 + Sqrt[5])/10])/2}, \
      {-Sqrt[1 + 2/Sqrt[5]], (1 + Sqrt[5])/2, \
      (-3*Sqrt[(5 + Sqrt[5])/10])/2}, {-1/2*Sqrt[1 + 2/Sqrt[5]], \
      -1/2, Sqrt[25/8 + 41/(8*Sqrt[5])]}, \
      {-1/2*Sqrt[1 + 2/Sqrt[5]], 1/2, Sqrt[25/8 + 41/(8*Sqrt[5])]}, \
      {Sqrt[1 + 2/Sqrt[5]], -1, Root[1 - 260*#1^2 + 80*#1^4 & , 1, \
      0]}, {Sqrt[1 + 2/Sqrt[5]], 1, Root[1 - 260*#1^2 + 80*#1^4 & , \
      1, 0]}, {Sqrt[1 + 2/Sqrt[5]], -2/(-1 + Sqrt[5]), \
      Sqrt[9/8 + 9/(8*Sqrt[5])]}, {Sqrt[1 + 2/Sqrt[5]], \
      (1 + Sqrt[5])/2, Sqrt[9/8 + 9/(8*Sqrt[5])]}, \
      {-Sqrt[2 + 2/Sqrt[5]], 0, Root[1 - 260*#1^2 + 80*#1^4 & , 1, \
      0]}, {Sqrt[2 + 2/Sqrt[5]], 0, Sqrt[26 + 58/Sqrt[5]]/4}, \
      {-1/2*Sqrt[5 + 2/Sqrt[5]], -1 - Sqrt[5]/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0]}, \
      {-1/2*Sqrt[5 + 2/Sqrt[5]], (2 + Sqrt[5])/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0]}, \
      {-Sqrt[17/8 + 31/(8*Sqrt[5])], (1 - Sqrt[5])^(-1), \
      (-3*Sqrt[(5 + Sqrt[5])/10])/2}, \
      {-Sqrt[17/8 + 31/(8*Sqrt[5])], (1 + Sqrt[5])/4, \
      (-3*Sqrt[(5 + Sqrt[5])/10])/2}, {Sqrt[9/4 + 9/(2*Sqrt[5])], \
      -1/2, (-3*Sqrt[(5 + Sqrt[5])/10])/2}, \
      {Sqrt[9/4 + 9/(2*Sqrt[5])], 1/2, \
      (-3*Sqrt[(5 + Sqrt[5])/10])/2}, {Sqrt[5/2 + 11/(2*Sqrt[5])], \
      -1, Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0]}, \
      {Sqrt[5/2 + 11/(2*Sqrt[5])], 1, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0]}, \
      {Sqrt[13/4 + 11/(2*Sqrt[5])], -1/2, \
      Sqrt[1/8 + 1/(8*Sqrt[5])]}, {Sqrt[13/4 + 11/(2*Sqrt[5])], \
      1/2, Sqrt[1/8 + 1/(8*Sqrt[5])]}, {-1/4*Sqrt[10 + 22/Sqrt[5]], \
      (-5 - Sqrt[5])/4, Sqrt[9/8 + 9/(8*Sqrt[5])]}, \
      {-1/4*Sqrt[10 + 22/Sqrt[5]], (5 + Sqrt[5])/4, \
      Sqrt[9/8 + 9/(8*Sqrt[5])]}, {Sqrt[10 + 22/Sqrt[5]]/4, \
      (-5 - Sqrt[5])/4, (-3*Sqrt[(5 + Sqrt[5])/10])/2}, \
      {Sqrt[10 + 22/Sqrt[5]]/4, (5 + Sqrt[5])/4, \
      (-3*Sqrt[(5 + Sqrt[5])/10])/2}, {-1/2*Sqrt[13 + 22/Sqrt[5]], \
      -1/2, Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0]}, \
      {-1/2*Sqrt[13 + 22/Sqrt[5]], 1/2, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0]}, \
      {-1/4*Sqrt[26 + 38/Sqrt[5]], (-5 - Sqrt[5])/4, \
      Sqrt[1/8 + 1/(8*Sqrt[5])]}, {-1/4*Sqrt[26 + 38/Sqrt[5]], \
      (5 + Sqrt[5])/4, Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
      {Sqrt[26 + 38/Sqrt[5]]/4, (-5 - Sqrt[5])/4, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0]}, \
      {Sqrt[26 + 38/Sqrt[5]]/4, (5 + Sqrt[5])/4, \
      Root[1 - 20*#1^2 + 80*#1^4 & , 1, 0]}, \
      {Sqrt[34 + 62/Sqrt[5]]/4, (1 - Sqrt[5])^(-1), \
      Sqrt[9/8 + 9/(8*Sqrt[5])]}, {Sqrt[34 + 62/Sqrt[5]]/4, \
      (1 + Sqrt[5])/4, Sqrt[9/8 + 9/(8*Sqrt[5])]}, \
      {Sqrt[(5 + Sqrt[5])/10], 0, Sqrt[25/8 + 41/(8*Sqrt[5])]}, \
      {Root[1 - 25*#1^2 + 5*#1^4 & , 1, 0], -1, \
      Sqrt[1/8 + 1/(8*Sqrt[5])]}, {Root[1 - 25*#1^2 + 5*#1^4 & , 1, \
      0], 1, Sqrt[1/8 + 1/(8*Sqrt[5])]}, \
      {Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0], 0, \
      -Sqrt[25/8 + 41/(8*Sqrt[5])]}, {Root[1 - 5*#1^2 + 5*#1^4 & , \
      2, 0], -2/(-1 + Sqrt[5]), Root[1 - 260*#1^2 + 80*#1^4 & , 1, \
      0]}, {Root[1 - 5*#1^2 + 5*#1^4 & , 2, 0], (1 + Sqrt[5])/2, \
      Root[1 - 260*#1^2 + 80*#1^4 & , 1, 0]}, \
      {Root[1 - 5*#1^2 + 5*#1^4 & , 3, 0], -2/(-1 + Sqrt[5]), \
      Sqrt[26 + 58/Sqrt[5]]/4}, {Root[1 - 5*#1^2 + 5*#1^4 & , 3, \
      0], (1 + Sqrt[5])/2, Sqrt[26 + 58/Sqrt[5]]/4}}",
    faces_src: "\
      {{53, 11, 24, 23, 9}, {51, 39, 40, 52, 30}, {60, 28, 16, 12, \
      2}, {20, 42, 48, 55, 18}, {19, 17, 54, 47, 41}, {1, 10, 15, \
      27, 59}, {36, 26, 44, 50, 38}, {4, 58, 22, 32, 8}, {34, 29, \
      33, 45, 46}, {21, 57, 3, 6, 31}, {37, 49, 43, 25, 35}, {13, \
      5, 56, 7, 14}, {9, 59, 27, 51, 30, 53}, {53, 30, 52, 28, 60, \
      11}, {11, 60, 2, 42, 20, 24}, {24, 20, 18, 17, 19, 23}, {23, \
      19, 41, 1, 59, 9}, {13, 25, 43, 3, 57, 5}, {5, 57, 21, 33, \
      29, 56}, {56, 29, 34, 22, 58, 7}, {7, 58, 4, 44, 26, 14}, \
      {14, 26, 36, 35, 25, 13}, {40, 38, 50, 16, 28, 52}, {16, 50, \
      44, 4, 8, 12}, {12, 8, 32, 48, 42, 2}, {48, 32, 22, 34, 46, \
      55}, {55, 46, 45, 54, 17, 18}, {54, 45, 33, 21, 31, 47}, {47, \
      31, 6, 10, 1, 41}, {10, 6, 3, 43, 49, 15}, {15, 49, 37, 39, \
      51, 27}, {39, 37, 35, 36, 38, 40}}",
    classes_src: "\
      {\"Amphichiral\", \"Archimedean\", \"Canonical\", \"Convex\", \
      \"Equilateral\", \"Goldberg\", \"Rigid\", \"Rupert\", \
      \"Simple\", \"Uniform\", \"Zalgaller\"}",
  },
  PolyhedronInfo {
    name: "SmallRhombicosidodecahedron",
    vertex_count: 60,
    edge_count: 120,
    face_count: 62,
    volume: "20 + (29*Sqrt[5])/3",
    surface_area: "5*(6 + Sqrt[3]) + 3*5^(3/4)*Sqrt[2 + Sqrt[5]]",
    circumradius: "Sqrt[11 + 4*Sqrt[5]]/2",
    inradius: "Missing[\"NotApplicable\"]",
    vertices_src: "\
      {{-1/2, -1/2, -1 - Sqrt[5]/2}, {-1/2, -1/2, (2 + Sqrt[5])/2}, \
      {-1/2, 1/2, -1 - Sqrt[5]/2}, {-1/2, 1/2, (2 + Sqrt[5])/2}, \
      {-1/2, -1 - Sqrt[5]/2, -1/2}, {-1/2, -1 - Sqrt[5]/2, 1/2}, \
      {-1/2, (2 + Sqrt[5])/2, -1/2}, {-1/2, (2 + Sqrt[5])/2, 1/2}, \
      {0, (-3 + Sqrt[5])^(-1), (-5 - Sqrt[5])/4}, {0, \
      (-3 + Sqrt[5])^(-1), (5 + Sqrt[5])/4}, {0, (3 + Sqrt[5])/4, \
      (-5 - Sqrt[5])/4}, {0, (3 + Sqrt[5])/4, (5 + Sqrt[5])/4}, \
      {1/2, -1/2, -1 - Sqrt[5]/2}, {1/2, -1/2, (2 + Sqrt[5])/2}, \
      {1/2, 1/2, -1 - Sqrt[5]/2}, {1/2, 1/2, (2 + Sqrt[5])/2}, \
      {1/2, -1 - Sqrt[5]/2, -1/2}, {1/2, -1 - Sqrt[5]/2, 1/2}, \
      {1/2, (2 + Sqrt[5])/2, -1/2}, {1/2, (2 + Sqrt[5])/2, 1/2}, \
      {(-5 - Sqrt[5])/4, 0, (-3 + Sqrt[5])^(-1)}, \
      {(-5 - Sqrt[5])/4, 0, (3 + Sqrt[5])/4}, {(-1 - Sqrt[5])/4, \
      (-1 - Sqrt[5])/2, (-3 + Sqrt[5])^(-1)}, {(-1 - Sqrt[5])/4, \
      (-1 - Sqrt[5])/2, (3 + Sqrt[5])/4}, {(-1 - Sqrt[5])/4, \
      (1 + Sqrt[5])/2, (-3 + Sqrt[5])^(-1)}, {(-1 - Sqrt[5])/4, \
      (1 + Sqrt[5])/2, (3 + Sqrt[5])/4}, {(-1 - Sqrt[5])/2, \
      (-3 + Sqrt[5])^(-1), (-1 - Sqrt[5])/4}, {(-1 - Sqrt[5])/2, \
      (-3 + Sqrt[5])^(-1), (1 + Sqrt[5])/4}, {(-1 - Sqrt[5])/2, \
      (3 + Sqrt[5])/4, (-1 - Sqrt[5])/4}, {(-1 - Sqrt[5])/2, \
      (3 + Sqrt[5])/4, (1 + Sqrt[5])/4}, {-1 - Sqrt[5]/2, -1/2, \
      -1/2}, {-1 - Sqrt[5]/2, -1/2, 1/2}, {-1 - Sqrt[5]/2, 1/2, \
      -1/2}, {-1 - Sqrt[5]/2, 1/2, 1/2}, {(-3 + Sqrt[5])^(-1), \
      (-5 - Sqrt[5])/4, 0}, {(-3 + Sqrt[5])^(-1), (-1 - Sqrt[5])/4, \
      (-1 - Sqrt[5])/2}, {(-3 + Sqrt[5])^(-1), (-1 - Sqrt[5])/4, \
      (1 + Sqrt[5])/2}, {(-3 + Sqrt[5])^(-1), (1 + Sqrt[5])/4, \
      (-1 - Sqrt[5])/2}, {(-3 + Sqrt[5])^(-1), (1 + Sqrt[5])/4, \
      (1 + Sqrt[5])/2}, {(-3 + Sqrt[5])^(-1), (5 + Sqrt[5])/4, 0}, \
      {(1 + Sqrt[5])/4, (-1 - Sqrt[5])/2, (-3 + Sqrt[5])^(-1)}, \
      {(1 + Sqrt[5])/4, (-1 - Sqrt[5])/2, (3 + Sqrt[5])/4}, \
      {(1 + Sqrt[5])/4, (1 + Sqrt[5])/2, (-3 + Sqrt[5])^(-1)}, \
      {(1 + Sqrt[5])/4, (1 + Sqrt[5])/2, (3 + Sqrt[5])/4}, \
      {(1 + Sqrt[5])/2, (-3 + Sqrt[5])^(-1), (-1 - Sqrt[5])/4}, \
      {(1 + Sqrt[5])/2, (-3 + Sqrt[5])^(-1), (1 + Sqrt[5])/4}, \
      {(1 + Sqrt[5])/2, (3 + Sqrt[5])/4, (-1 - Sqrt[5])/4}, \
      {(1 + Sqrt[5])/2, (3 + Sqrt[5])/4, (1 + Sqrt[5])/4}, \
      {(2 + Sqrt[5])/2, -1/2, -1/2}, {(2 + Sqrt[5])/2, -1/2, 1/2}, \
      {(2 + Sqrt[5])/2, 1/2, -1/2}, {(2 + Sqrt[5])/2, 1/2, 1/2}, \
      {(3 + Sqrt[5])/4, (-5 - Sqrt[5])/4, 0}, {(3 + Sqrt[5])/4, \
      (-1 - Sqrt[5])/4, (-1 - Sqrt[5])/2}, {(3 + Sqrt[5])/4, \
      (-1 - Sqrt[5])/4, (1 + Sqrt[5])/2}, {(3 + Sqrt[5])/4, \
      (1 + Sqrt[5])/4, (-1 - Sqrt[5])/2}, {(3 + Sqrt[5])/4, \
      (1 + Sqrt[5])/4, (1 + Sqrt[5])/2}, {(3 + Sqrt[5])/4, \
      (5 + Sqrt[5])/4, 0}, {(5 + Sqrt[5])/4, 0, \
      (-3 + Sqrt[5])^(-1)}, {(5 + Sqrt[5])/4, 0, (3 + Sqrt[5])/4}}",
    faces_src: "\
      {{36, 23, 27}, {37, 28, 24}, {40, 8, 7}, {35, 5, 6}, {38, 29, \
      25}, {39, 26, 30}, {10, 14, 2}, {9, 1, 13}, {12, 4, 16}, {11, \
      15, 3}, {54, 45, 41}, {55, 42, 46}, {58, 19, 20}, {53, 18, \
      17}, {56, 43, 47}, {57, 48, 44}, {34, 32, 22}, {33, 21, 31}, \
      {59, 51, 49}, {60, 50, 52}, {27, 31, 21, 36}, {23, 36, 1, 9}, \
      {10, 2, 37, 24}, {37, 22, 32, 28}, {8, 40, 30, 26}, {25, 29, \
      40, 7}, {35, 27, 23, 5}, {6, 24, 28, 35}, {3, 38, 25, 11}, \
      {21, 33, 29, 38}, {39, 30, 34, 22}, {12, 26, 39, 4}, {55, 14, \
      10, 42}, {41, 9, 13, 54}, {57, 44, 12, 16}, {15, 11, 43, 56}, \
      {45, 54, 59, 49}, {50, 60, 55, 46}, {48, 58, 20, 44}, {43, \
      19, 58, 47}, {53, 17, 41, 45}, {46, 42, 18, 53}, {59, 56, 47, \
      51}, {52, 48, 57, 60}, {31, 32, 34, 33}, {17, 18, 6, 5}, {1, \
      3, 15, 13}, {14, 16, 4, 2}, {7, 8, 20, 19}, {51, 52, 50, 49}, \
      {3, 1, 36, 21, 38}, {22, 37, 2, 4, 39}, {29, 33, 34, 30, 40}, \
      {27, 35, 28, 32, 31}, {42, 10, 24, 6, 18}, {41, 17, 5, 23, \
      9}, {20, 8, 26, 12, 44}, {11, 25, 7, 19, 43}, {56, 59, 54, \
      13, 15}, {57, 16, 14, 55, 60}, {58, 48, 52, 51, 47}, {49, 50, \
      46, 53, 45}}",
    classes_src: "\
      {\"Amphichiral\", \"Archimedean\", \"Canonical\", \"Convex\", \
      \"Equilateral\", \"Rigid\", \"Simple\", \"Uniform\"}",
  },
  PolyhedronInfo {
    name: "RhombicTriacontahedron",
    vertex_count: 32,
    edge_count: 60,
    face_count: 30,
    volume: "4*Sqrt[5 + 2*Sqrt[5]]",
    surface_area: "12*Sqrt[5]",
    circumradius: "Missing[\"NotApplicable\"]",
    inradius: "Sqrt[1 + 2/Sqrt[5]]",
    vertices_src: "\
      {{0, 0, (-1 - Sqrt[5])/2}, {0, 0, (1 + Sqrt[5])/2}, \
      {(5 - Sqrt[5])/10, Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0], \
      (5 + 3*Sqrt[5])/10}, {(5 - Sqrt[5])/10, \
      Sqrt[(5 + Sqrt[5])/10], (5 + 3*Sqrt[5])/10}, {2/Sqrt[5], 0, \
      (5 + 3*Sqrt[5])/10}, {(5 + 3*Sqrt[5])/10, \
      Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0], (5 + Sqrt[5])/10}, \
      {(5 + 3*Sqrt[5])/10, Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0], \
      (-5 + Sqrt[5])/10}, {(5 + 3*Sqrt[5])/10, \
      Sqrt[(5 + Sqrt[5])/10], (5 + Sqrt[5])/10}, \
      {(5 + 3*Sqrt[5])/10, Sqrt[(5 + Sqrt[5])/10], \
      (-5 + Sqrt[5])/10}, {-2/Sqrt[5], 0, (-5 - 3*Sqrt[5])/10}, \
      {-(1/Sqrt[5]), -Sqrt[1 + 2/Sqrt[5]], (5 + Sqrt[5])/10}, \
      {-(1/Sqrt[5]), -Sqrt[1 + 2/Sqrt[5]], (-5 + Sqrt[5])/10}, \
      {-(1/Sqrt[5]), Sqrt[1 + 2/Sqrt[5]], (5 + Sqrt[5])/10}, \
      {-(1/Sqrt[5]), Sqrt[1 + 2/Sqrt[5]], (-5 + Sqrt[5])/10}, \
      {1/Sqrt[5], -Sqrt[1 + 2/Sqrt[5]], (5 - Sqrt[5])/10}, \
      {1/Sqrt[5], -Sqrt[1 + 2/Sqrt[5]], (-5 - Sqrt[5])/10}, \
      {1/Sqrt[5], Sqrt[1 + 2/Sqrt[5]], (5 - Sqrt[5])/10}, \
      {1/Sqrt[5], Sqrt[1 + 2/Sqrt[5]], (-5 - Sqrt[5])/10}, \
      {-1 - 1/Sqrt[5], 0, (5 + Sqrt[5])/10}, {-1 - 1/Sqrt[5], 0, \
      (-5 + Sqrt[5])/10}, {(-5 - Sqrt[5])/10, \
      Root[1 - 5*#1^2 + 5*#1^4 & , 2, 0], (5 + 3*Sqrt[5])/10}, \
      {(-5 - Sqrt[5])/10, Sqrt[2/(5 + Sqrt[5])], \
      (5 + 3*Sqrt[5])/10}, {(5 + Sqrt[5])/10, \
      Root[1 - 5*#1^2 + 5*#1^4 & , 2, 0], (-5 - 3*Sqrt[5])/10}, \
      {(5 + Sqrt[5])/10, Sqrt[2/(5 + Sqrt[5])], \
      (-5 - 3*Sqrt[5])/10}, {1 + 1/Sqrt[5], 0, (5 - Sqrt[5])/10}, \
      {1 + 1/Sqrt[5], 0, (-5 - Sqrt[5])/10}, {(-5 - 3*Sqrt[5])/10, \
      Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0], (5 - Sqrt[5])/10}, \
      {(-5 - 3*Sqrt[5])/10, Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0], \
      (-5 - Sqrt[5])/10}, {(-5 - 3*Sqrt[5])/10, \
      Sqrt[(5 + Sqrt[5])/10], (5 - Sqrt[5])/10}, \
      {(-5 - 3*Sqrt[5])/10, Sqrt[(5 + Sqrt[5])/10], \
      (-5 - Sqrt[5])/10}, {(-5 + Sqrt[5])/10, \
      Root[1 - 5*#1^2 + 5*#1^4 & , 1, 0], (-5 - 3*Sqrt[5])/10}, \
      {(-5 + Sqrt[5])/10, Sqrt[(5 + Sqrt[5])/10], \
      (-5 - 3*Sqrt[5])/10}}",
    faces_src: "\
      {{16, 15, 11, 12}, {14, 13, 17, 18}, {10, 28, 20, 30}, {8, 5, \
      6, 25}, {12, 28, 31, 16}, {32, 30, 14, 18}, {6, 3, 11, 15}, \
      {8, 17, 13, 4}, {11, 21, 19, 27}, {13, 29, 19, 22}, {7, 16, \
      23, 26}, {24, 18, 9, 26}, {12, 11, 27, 28}, {30, 29, 13, 14}, \
      {7, 6, 15, 16}, {18, 17, 8, 9}, {2, 22, 19, 21}, {23, 1, 24, \
      26}, {3, 2, 21, 11}, {4, 13, 22, 2}, {16, 31, 1, 23}, {1, 32, \
      18, 24}, {31, 28, 10, 1}, {10, 30, 32, 1}, {6, 5, 2, 3}, {8, \
      4, 2, 5}, {28, 27, 19, 20}, {20, 19, 29, 30}, {26, 25, 6, 7}, \
      {9, 8, 25, 26}}",
    classes_src: "\
      {\"Amphichiral\", \"ArchimedeanDual\", \"Canonical\", \
      \"Convex\", \"Equilateral\", \"Isohedron\", \"Rigid\", \
      \"Rupert\", \"Simple\", \"UniformDual\", \"Zonohedron\"}",
  },
];

fn find_polyhedron(name: &str) -> Option<&'static PolyhedronInfo> {
  // "Hexahedron" is the standard alternative name for the cube.
  let name = if name == "Hexahedron" { "Cube" } else { name };
  POLYHEDRA.iter().find(|p| p.name == name)
}

/// The exact unit-edge volume of a Platonic solid, as WL source.
pub fn unit_volume_src(name: &str) -> Option<&'static str> {
  find_polyhedron(name).map(|p| p.volume)
}

/// The exact unit-edge surface area of a Platonic solid, as WL source.
pub fn unit_surface_area_src(name: &str) -> Option<&'static str> {
  find_polyhedron(name).map(|p| p.surface_area)
}

/// Evaluate a polyhedron's exact vertex list to numeric `[x, y, z]` rows
/// (for rendering and for deriving the edge list).
fn numeric_vertices(
  info: &PolyhedronInfo,
) -> Result<Vec<[f64; 3]>, InterpreterError> {
  let evaluated = eval_wl(&format!("N[{}]", info.vertices_src))?;
  let Expr::List(rows) = &evaluated else {
    return Err(InterpreterError::EvaluationError(format!(
      "PolyhedronData: vertex data for {} did not evaluate to a list",
      info.name
    )));
  };
  let mut vertices = Vec::with_capacity(rows.len());
  for row in rows.iter() {
    let Expr::List(coords) = row else {
      return Err(InterpreterError::EvaluationError(format!(
        "PolyhedronData: vertex row for {} is not a coordinate triple",
        info.name
      )));
    };
    let mut point = [0.0; 3];
    for (slot, coord) in point.iter_mut().zip(coords.iter()) {
      *slot = match coord {
        Expr::Real(r) => *r,
        Expr::Integer(i) => *i as f64,
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "PolyhedronData: vertex coordinate for {} is not numeric",
            info.name
          )));
        }
      };
    }
    vertices.push(point);
  }
  Ok(vertices)
}

/// The edges of a polyhedron as 1-based vertex index pairs, in canonical
/// (lexicographic) order: every vertex pair at minimal (= edge) distance.
fn edge_indices(info: &PolyhedronInfo) -> Result<Expr, InterpreterError> {
  // Every edge borders two faces, so walking each face's rim and keeping
  // the distinct unordered pairs gives them all. Reading them off the
  // faces (rather than off the shortest vertex distances) is what makes
  // this right for solids whose edges are not all the same length — the
  // Catalan solids have two edge lengths, and only the short ones would
  // show up otherwise.
  let faces = numeric_faces(info)?;
  let mut pairs: Vec<(usize, usize)> = Vec::new();
  for face in &faces {
    for (k, &a) in face.iter().enumerate() {
      let b = face[(k + 1) % face.len()];
      let pair = if a <= b { (a, b) } else { (b, a) };
      if !pairs.contains(&pair) {
        pairs.push(pair);
      }
    }
  }
  // Wolfram lists them sorted by the lower then the higher index.
  pairs.sort_unstable();
  Ok(Expr::List(
    pairs
      .into_iter()
      .map(|(a, b)| {
        Expr::List(
          vec![Expr::Integer(a as i128 + 1), Expr::Integer(b as i128 + 1)]
            .into(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  ))
}

/// `Sphere[{0, 0, 0}, r]` with the polyhedron's exact inradius: the sphere
/// inscribed in the (origin-centered) solid.
fn insphere(info: &PolyhedronInfo) -> Result<Expr, InterpreterError> {
  let center = Expr::List(
    vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)].into(),
  );
  let radius = eval_wl(info.inradius)?;
  Ok(Expr::FunctionCall {
    name: "Sphere".to_string(),
    args: vec![center, radius].into(),
  })
}

/// The faces of a polyhedron as 1-based vertex index lists, in Wolfram's
/// order and winding.
fn face_indices(info: &PolyhedronInfo) -> Result<Expr, InterpreterError> {
  eval_wl(info.faces_src)
}

/// `"Faces"`: the exact vertex coordinates with the faces as index lists,
/// as `GraphicsComplex[coords, Polygon[indices]]` — the form Wolfram
/// returns, so `data[[1]]` are the vertices and `data[[2, 1]]` the faces.
fn faces_complex(info: &PolyhedronInfo) -> Result<Expr, InterpreterError> {
  Ok(Expr::FunctionCall {
    name: "GraphicsComplex".to_string(),
    args: vec![
      eval_wl(info.vertices_src)?,
      Expr::FunctionCall {
        name: "Polygon".to_string(),
        args: vec![face_indices(info)?].into(),
      },
    ]
    .into(),
  })
}

/// The face index lists as plain `usize` rows, for rendering.
fn numeric_faces(
  info: &PolyhedronInfo,
) -> Result<Vec<Vec<usize>>, InterpreterError> {
  let evaluated = face_indices(info)?;
  let Expr::List(rows) = &evaluated else {
    return Err(InterpreterError::EvaluationError(format!(
      "PolyhedronData: face data for {} did not evaluate to a list",
      info.name
    )));
  };
  let mut faces = Vec::with_capacity(rows.len());
  for row in rows.iter() {
    let Expr::List(items) = row else {
      return Err(InterpreterError::EvaluationError(format!(
        "PolyhedronData: face row for {} is not a list",
        info.name
      )));
    };
    let mut face = Vec::with_capacity(items.len());
    for item in items.iter() {
      let Expr::Integer(idx) = item else {
        return Err(InterpreterError::EvaluationError(format!(
          "PolyhedronData: face index for {} is not an integer",
          info.name
        )));
      };
      face.push(*idx as usize - 1);
    }
    faces.push(face);
  }
  Ok(faces)
}

/// Build the Graphics3D expression for a polyhedron and evaluate it into
/// the rendered graphics object.
fn polyhedron_graphics(
  info: &PolyhedronInfo,
) -> Result<Expr, InterpreterError> {
  let vertices = numeric_vertices(info)?;
  let faces = numeric_faces(info)?;
  let polygons: Vec<Expr> = faces
    .iter()
    .map(|face| {
      let pts: Vec<Expr> = face
        .iter()
        .map(|&idx| {
          Expr::List(
            vertices[idx]
              .iter()
              .map(|&c| Expr::Real(c))
              .collect::<Vec<_>>()
              .into(),
          )
        })
        .collect();
      Expr::FunctionCall {
        name: "Polygon".to_string(),
        args: vec![Expr::List(pts.into())].into(),
      }
    })
    .collect();
  let graphics = Expr::FunctionCall {
    name: "Graphics3D".to_string(),
    args: vec![Expr::List(polygons.into())].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&graphics)
}

/// Evaluate a stored exact WL value.
fn eval_wl(src: &str) -> Result<Expr, InterpreterError> {
  let parsed = crate::functions::string_ast::parse_program_to_expr(src)?;
  crate::evaluator::evaluate_expr_to_expr(&parsed)
}

/// Metric/count properties exposed by `PolyhedronData[name, property]`,
/// returned (sorted) by `PolyhedronData["Properties"]`.
static PROPERTIES: &[&str] = &[
  "Circumradius",
  "Classes",
  "EdgeCount",
  "EdgeIndices",
  "FaceCount",
  "FaceIndices",
  "Faces",
  "Inradius",
  "Insphere",
  "SurfaceArea",
  "VertexCoordinates",
  "VertexCount",
  "Volume",
];

/// Classes the built-in solids belong to, returned (sorted, without
/// duplicates) by `PolyhedronData["Classes"]`. Derived from the entities
/// rather than listed separately, so the two can never drift apart.
/// Disjoint from `PROPERTIES`.
fn classes() -> Result<Expr, InterpreterError> {
  let mut names: Vec<String> = Vec::new();
  for info in POLYHEDRA {
    let evaluated = eval_wl(info.classes_src)?;
    let Expr::List(items) = &evaluated else {
      continue;
    };
    for item in items.iter() {
      if let Expr::String(class) = item
        && !names.contains(class)
      {
        names.push(class.clone());
      }
    }
  }
  names.sort_unstable();
  Ok(Expr::List(
    names
      .into_iter()
      .map(Expr::String)
      .collect::<Vec<_>>()
      .into(),
  ))
}

/// Build a `List` of string entries.
fn string_list(items: &[&str]) -> Expr {
  Expr::List(
    items
      .iter()
      .map(|s| Expr::String(s.to_string()))
      .collect::<Vec<_>>()
      .into(),
  )
}

pub fn polyhedron_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("PolyhedronData", args));

  // `PolyhedronData[All]` — the list of known entities (by name).
  if let Some(Expr::Identifier(sym)) = args.first()
    && sym == "All"
    && args.len() == 1
  {
    let mut names: Vec<&str> = POLYHEDRA.iter().map(|p| p.name).collect();
    names.sort_unstable();
    return Ok(string_list(&names));
  }

  // `PolyhedronData["Properties"]` / `PolyhedronData["Classes"]` — the
  // available property and class names. Handled before `find_polyhedron`
  // so these reserved strings aren't reported as unknown entities.
  if let Some(Expr::String(kind)) = args.first()
    && args.len() == 1
  {
    match kind.as_str() {
      "Properties" => return Ok(string_list(PROPERTIES)),
      "Classes" => return classes(),
      _ => {}
    }
  }

  let Some(Expr::String(name)) = args.first() else {
    return unevaluated();
  };
  let Some(info) = find_polyhedron(name) else {
    crate::emit_message(&format!(
      "PolyhedronData::notent: {name} is not a known entity, class, or tag for PolyhedronData. Use PolyhedronData[] for a list of entities."
    ));
    // Wolfram emits the message but leaves the call unevaluated.
    return unevaluated();
  };
  match args.len() {
    1 => polyhedron_graphics(info),
    2 => {
      let Expr::String(property) = &args[1] else {
        return unevaluated();
      };
      match property.as_str() {
        "Classes" => eval_wl(info.classes_src),
        "VertexCount" => Ok(Expr::Integer(info.vertex_count)),
        "EdgeCount" => Ok(Expr::Integer(info.edge_count)),
        "FaceCount" => Ok(Expr::Integer(info.face_count)),
        "Volume" => eval_wl(info.volume),
        "SurfaceArea" => eval_wl(info.surface_area),
        "Circumradius" => eval_wl(info.circumradius),
        "Inradius" => eval_wl(info.inradius),
        "VertexCoordinates" => eval_wl(info.vertices_src),
        "EdgeIndices" => edge_indices(info),
        "FaceIndices" => face_indices(info),
        "Faces" => faces_complex(info),
        "Insphere" => insphere(info),
        _ => unevaluated(),
      }
    }
    _ => unevaluated(),
  }
}
