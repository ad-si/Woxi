//! `FiniteGroupData` — the 32 three-dimensional crystallographic point
//! groups, `{"CrystallographicPointGroup", n}`, numbered as in the
//! International Tables (1 = C1 … 27 = D6h … 32 = Oh).
//!
//! Each group is described by its abstract structure, from which the
//! multiplication table and the Cayley graph follow:
//!
//! - a cyclic group lists the powers of its generator `r`;
//! - a dihedral group lists the rotations `r^a`, then the coset `s r^a`;
//! - the tetrahedral and octahedral rotation groups are the even and all
//!   permutations of four points in lexicographic order;
//! - a centrosymmetric group repeats its rotation part multiplied by the
//!   inversion.
//!
//! Element 1 is always the identity, and entry `(i, j)` of the table is
//! the product "element i, then element j".

use super::*;

const FAMILY: &str = "CrystallographicPointGroup";

/// The abstract structure of the rotation part of a point group.
#[derive(Clone, Copy)]
enum Core {
  /// Cyclic group of the given order.
  Cyclic(usize),
  /// Dihedral group with the given number of rotations (order `2n`).
  Dihedral(usize),
  /// Even permutations of four points (order 12).
  Alternating4,
  /// All permutations of four points (order 24).
  Symmetric4,
}

struct PointGroup {
  /// Schoenflies symbol in plain text.
  schoenflies: &'static str,
  crystal_system: &'static str,
  crystal_form: &'static str,
  symmetry_types: &'static [&'static str],
  core: Core,
  /// Whether the group is the direct product of `core` with a group of
  /// order two.
  doubled: bool,
}

const fn pg(
  schoenflies: &'static str,
  crystal_system: &'static str,
  crystal_form: &'static str,
  symmetry_types: &'static [&'static str],
  core: Core,
  doubled: bool,
) -> PointGroup {
  PointGroup {
    schoenflies,
    crystal_system,
    crystal_form,
    symmetry_types,
    core,
    doubled,
  }
}

const ENANTIOMORPHIC_POLAR: &[&str] = &["Enantiomorphic", "Polar"];
const ENANTIOMORPHIC: &[&str] = &["Enantiomorphic"];
const CENTROSYMMETRIC: &[&str] = &["Centrosymmetric"];
const POLAR: &[&str] = &["Polar"];
const NO_TYPE: &[&str] = &[];

use Core::{Alternating4, Cyclic, Dihedral, Symmetric4};

#[rustfmt::skip]
const POINT_GROUPS: [PointGroup; 32] = [
  pg("C1", "Triclinic", "triclinic pedial", ENANTIOMORPHIC_POLAR, Cyclic(1), false),
  pg("Ci", "Triclinic", "triclinic pinacoidal", CENTROSYMMETRIC, Cyclic(1), true),
  pg("C2", "Monoclinic", "monoclinic sphenoidal", ENANTIOMORPHIC_POLAR, Cyclic(2), false),
  pg("Cs", "Monoclinic", "monoclinic domatic", POLAR, Cyclic(2), false),
  pg("C2h", "Monoclinic", "monoclinic prismatic", CENTROSYMMETRIC, Cyclic(2), true),
  pg("D2", "Orthorhombic", "orthorhombic sphenoidal", ENANTIOMORPHIC, Dihedral(2), false),
  pg("C2v", "Orthorhombic", "orthorhombic pyramidal", POLAR, Dihedral(2), false),
  pg("D2h", "Orthorhombic", "orthorhombic bipyramidal", CENTROSYMMETRIC, Dihedral(2), true),
  pg("C4", "Tetragonal", "tetragonal pyramidal", ENANTIOMORPHIC_POLAR, Cyclic(4), false),
  pg("S4", "Tetragonal", "tetragonal disphenoidal", NO_TYPE, Cyclic(4), false),
  pg("C4h", "Tetragonal", "tetragonal dipyramidal", CENTROSYMMETRIC, Cyclic(4), true),
  pg("D4", "Tetragonal", "tetragonal trapezoidal", ENANTIOMORPHIC, Dihedral(4), false),
  pg("C4v", "Tetragonal", "ditetragonal pyramidal", POLAR, Dihedral(4), false),
  pg("D2d", "Tetragonal", "tetragonal scalenoidal", NO_TYPE, Dihedral(4), false),
  pg("D4h", "Tetragonal", "ditetragonal dipyramidal", CENTROSYMMETRIC, Dihedral(4), true),
  pg("C3", "Trigonal", "trigonal pyramidal", ENANTIOMORPHIC_POLAR, Cyclic(3), false),
  pg("S6", "Trigonal", "rhombohedral", CENTROSYMMETRIC, Cyclic(3), true),
  pg("D3", "Trigonal", "trigonal trapezoidal", ENANTIOMORPHIC, Dihedral(3), false),
  pg("C3v", "Trigonal", "ditrigonal pyramidal", POLAR, Dihedral(3), false),
  pg("D3d", "Trigonal", "ditrigonal scalahedral", CENTROSYMMETRIC, Dihedral(3), true),
  pg("C6", "Hexagonal", "hexagonal pyramidal", ENANTIOMORPHIC_POLAR, Cyclic(6), false),
  pg("C3h", "Hexagonal", "trigonal dipyramidal", NO_TYPE, Cyclic(6), false),
  pg("C6h", "Hexagonal", "hexagonal dipyramidal", CENTROSYMMETRIC, Cyclic(6), true),
  pg("D6", "Hexagonal", "hexagonal trapezoidal", ENANTIOMORPHIC, Dihedral(6), false),
  pg("C6v", "Hexagonal", "dihexagonal pyramidal", POLAR, Dihedral(6), false),
  pg("D3h", "Hexagonal", "ditrigonal dipyramidal", NO_TYPE, Dihedral(6), false),
  pg("D6h", "Hexagonal", "dihexagonal dipyramidal", CENTROSYMMETRIC, Dihedral(6), true),
  pg("T", "Cubic", "tetartoidal", ENANTIOMORPHIC, Alternating4, false),
  pg("Th", "Cubic", "diploidal", CENTROSYMMETRIC, Alternating4, true),
  pg("O", "Cubic", "gyroidal", ENANTIOMORPHIC, Symmetric4, false),
  pg("Td", "Cubic", "tetrahedral", NO_TYPE, Symmetric4, false),
  pg("Oh", "Cubic", "hexoctahedral", CENTROSYMMETRIC, Symmetric4, true),
];

/// The permutations of four points, in lexicographic order, optionally
/// only the even ones.
fn permutations_of_four(even_only: bool) -> Vec<[u8; 4]> {
  let mut out = Vec::new();
  for a in 0..4u8 {
    for b in 0..4u8 {
      for c in 0..4u8 {
        for d in 0..4u8 {
          let p = [a, b, c, d];
          let distinct = (0..4).all(|i| (0..i).all(|j| p[i] != p[j]));
          let inversions = (0..4)
            .flat_map(|i| (0..i).map(move |j| (j, i)))
            .filter(|&(j, i)| p[j] > p[i])
            .count();
          if distinct && (!even_only || inversions % 2 == 0) {
            out.push(p);
          }
        }
      }
    }
  }
  out
}

/// Multiplication table (0-based) and generators (0-based element
/// indices) of the rotation part.
fn core_table(core: Core) -> (Vec<Vec<usize>>, Vec<usize>) {
  match core {
    Cyclic(n) => {
      let table = (0..n)
        .map(|a| (0..n).map(|b| (a + b) % n).collect())
        .collect();
      (table, if n > 1 { vec![1] } else { vec![] })
    }
    Dihedral(n) => {
      // Element `f n + a` is `s^f r^a`, and `r^a s = s r^(-a)`.
      let order = 2 * n;
      let table = (0..order)
        .map(|x| {
          let (f1, a1) = (x / n, x % n);
          (0..order)
            .map(|y| {
              let (f2, a2) = (y / n, y % n);
              let a = if f2 == 1 { (n - a1) % n } else { a1 };
              ((f1 + f2) % 2) * n + (a + a2) % n
            })
            .collect()
        })
        .collect();
      (table, vec![1, n])
    }
    Alternating4 | Symmetric4 => {
      let even_only = matches!(core, Alternating4);
      let elements = permutations_of_four(even_only);
      let position = |p: &[u8; 4]| {
        elements
          .iter()
          .position(|q| q == p)
          .expect("permutations are closed under composition")
      };
      let table = elements
        .iter()
        .map(|p| {
          elements
            .iter()
            .map(|q| position(&[0, 1, 2, 3].map(|x: usize| q[p[x] as usize])))
            .collect()
        })
        .collect();
      // A three-cycle and a double transposition generate the even
      // permutations; the adjacent transpositions generate all of them.
      let generators: &[[u8; 4]] = if even_only {
        &[[1, 2, 0, 3], [1, 0, 3, 2]]
      } else {
        &[[1, 0, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2]]
      };
      (table, generators.iter().map(position).collect())
    }
  }
}

/// Multiplication table (0-based) and generators of the whole group.
fn group_table(group: &PointGroup) -> (Vec<Vec<usize>>, Vec<usize>) {
  let (core, mut generators) = core_table(group.core);
  if !group.doubled {
    return (core, generators);
  }
  let n = core.len();
  let table = (0..2 * n)
    .map(|x| {
      (0..2 * n)
        .map(|y| ((x / n + y / n) % 2) * n + core[x % n][y % n])
        .collect()
    })
    .collect();
  generators.push(n);
  (table, generators)
}

fn standard_name(n: usize) -> Expr {
  Expr::List(
    vec![Expr::String(FAMILY.to_string()), Expr::Integer(n as i128)].into(),
  )
}

fn integer_list(values: impl Iterator<Item = usize>) -> Expr {
  Expr::List(
    values
      .map(|v| Expr::Integer(v as i128))
      .collect::<Vec<_>>()
      .into(),
  )
}

/// The Cayley graph: one directed edge per element and generator, the
/// edges of each generator in a color of their own.
fn cayley_graph(group: &PointGroup) -> Result<Expr, InterpreterError> {
  let (table, generators) = group_table(group);
  let mut edges = Vec::new();
  let mut styles = Vec::new();
  for (k, &g) in generators.iter().enumerate() {
    let color = crate::evaluator::evaluate_expr_to_expr(
      &crate::functions::string_ast::parse_program_to_expr(&format!(
        "ColorData[97][{}]",
        k + 1
      ))?,
    )?;
    for (i, row) in table.iter().enumerate() {
      let edge = Expr::FunctionCall {
        name: "DirectedEdge".to_string(),
        args: vec![
          Expr::Integer(i as i128 + 1),
          Expr::Integer(row[g] as i128 + 1),
        ]
        .into(),
      };
      styles.push(Expr::Rule {
        pattern: Box::new(edge.clone()),
        replacement: Box::new(color.clone()),
      });
      edges.push(edge);
    }
  }
  let mut args = vec![integer_list(1..=table.len()), Expr::List(edges.into())];
  if !styles.is_empty() {
    args.push(Expr::Rule {
      pattern: Box::new(Expr::Identifier("EdgeStyle".to_string())),
      replacement: Box::new(Expr::List(styles.into())),
    });
  }
  crate::evaluator::evaluate_function_call_ast("Graph", &args)
}

fn property(
  n: usize,
  group: &PointGroup,
  prop: &str,
) -> Result<Option<Expr>, InterpreterError> {
  let order = || group_table(group).0.len();
  Ok(Some(match prop {
    "StandardName" => standard_name(n),
    "Name" => Expr::String(format!(
      "crystallographic point group {}",
      group.schoenflies
    )),
    "Order" => Expr::Integer(order() as i128),
    "Elements" => integer_list(1..=order()),
    "CrystalSystem" => Expr::String(group.crystal_system.to_string()),
    "CrystalForm" => Expr::String(group.crystal_form.to_string()),
    "PointGroupType" => Expr::List(
      group
        .symmetry_types
        .iter()
        .map(|t| Expr::String(t.to_string()))
        .collect::<Vec<_>>()
        .into(),
    ),
    "MultiplicationTable" => Expr::List(
      group_table(group)
        .0
        .into_iter()
        .map(|row| integer_list(row.into_iter().map(|v| v + 1)))
        .collect::<Vec<_>>()
        .into(),
    ),
    "CayleyGraph" => cayley_graph(group)?,
    // Woxi carries no names for the symmetry operations.
    "ElementNames" => Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![Expr::String("NotAvailable".to_string())].into(),
    },
    _ => return Ok(None),
  }))
}

/// The index of a `{"CrystallographicPointGroup", n}` specification.
fn point_group_index(spec: &Expr) -> Option<usize> {
  let Expr::List(items) = spec else {
    return None;
  };
  match &items[..] {
    [Expr::String(family), Expr::Integer(n)]
      if family == FAMILY && (1..=32).contains(n) =>
    {
      Some(*n as usize)
    }
    _ => None,
  }
}

pub fn finite_group_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("FiniteGroupData", args));
  // FiniteGroupData["CrystallographicPointGroup"] lists the family.
  if let [Expr::String(class)] = args
    && class == FAMILY
  {
    return Ok(Expr::List(
      (1..=32).map(standard_name).collect::<Vec<_>>().into(),
    ));
  }
  let Some(n) = args.first().and_then(point_group_index) else {
    return unevaluated();
  };
  let group = &POINT_GROUPS[n - 1];
  match &args[1..] {
    [] => Ok(standard_name(n)),
    [Expr::String(prop)] => match property(n, group, prop)? {
      Some(value) => Ok(value),
      None => unevaluated(),
    },
    // A list of properties gives the list of their values.
    [Expr::List(props)] => {
      let mut values = Vec::with_capacity(props.len());
      for prop in props {
        let Expr::String(prop) = prop else {
          return unevaluated();
        };
        match property(n, group, prop)? {
          Some(value) => values.push(value),
          None => return unevaluated(),
        }
      }
      Ok(Expr::List(values.into()))
    }
    _ => unevaluated(),
  }
}
