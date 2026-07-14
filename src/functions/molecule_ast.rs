//! AST-native chemistry functions.
//!
//! Implements Molecule, MoleculeQ, AtomList, BondList, MoleculeValue, and
//! molecule property access (`mol["AtomCount"]`).
//!
//! A molecule's canonical form is
//! `Molecule[{Atom["sym", opts…], …}, {Bond[{i, j}, "type"], …}]`,
//! matching the evaluated form wolframscript produces. Strings are
//! interpreted first as chemical names (via a built-in name → SMILES table)
//! and then as SMILES structure specifications. Hydrogen atoms are added
//! explicitly to fill the normal valences, as wolframscript does.

use crate::InterpreterError;
use crate::functions::element_data::{
  abbreviation_for_atomic_number, is_element_abbreviation,
};
use crate::syntax::{Expr, unevaluated};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BondKind {
  Single,
  Double,
  Triple,
  Aromatic,
}

impl BondKind {
  fn as_str(self) -> &'static str {
    match self {
      BondKind::Single => "Single",
      BondKind::Double => "Double",
      BondKind::Triple => "Triple",
      BondKind::Aromatic => "Aromatic",
    }
  }

  fn from_str(s: &str) -> Option<Self> {
    match s {
      "Single" => Some(BondKind::Single),
      "Double" => Some(BondKind::Double),
      "Triple" => Some(BondKind::Triple),
      "Aromatic" => Some(BondKind::Aromatic),
      _ => None,
    }
  }

  /// Integer bond order used for valence bookkeeping. Aromatic bonds count
  /// as one σ-bond; the delocalized π-system is accounted for separately.
  fn order(self) -> u32 {
    match self {
      BondKind::Single | BondKind::Aromatic => 1,
      BondKind::Double => 2,
      BondKind::Triple => 3,
    }
  }
}

#[derive(Clone, Debug)]
struct AtomData {
  symbol: String,
  formal_charge: i64,
  mass_number: Option<i64>,
  /// Lowercase (aromatic) atom in SMILES input.
  aromatic: bool,
  /// Hydrogen count given explicitly in a SMILES bracket atom. `None` means
  /// "fill the normal valence"; bracket atoms always specify it (default 0).
  explicit_h: Option<u32>,
}

impl AtomData {
  fn plain(symbol: &str) -> Self {
    AtomData {
      symbol: symbol.to_string(),
      formal_charge: 0,
      mass_number: None,
      aromatic: false,
      explicit_h: None,
    }
  }
}

/// A molecular graph with 0-based atom indices.
struct MolGraph {
  atoms: Vec<AtomData>,
  bonds: Vec<(usize, usize, BondKind)>,
}

// ---------------------------------------------------------------------------
// Chemical name database
// ---------------------------------------------------------------------------

/// Common chemical names resolvable without the Wolfram curated data
/// service. Keys are lowercase; lookup normalizes case and whitespace.
static NAME_TO_SMILES: &[(&str, &str)] = &[
  ("acetaldehyde", "CC=O"),
  ("acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
  ("acetic acid", "CC(=O)O"),
  ("acetone", "CC(=O)C"),
  ("acetylene", "C#C"),
  ("alanine", "CC(N)C(=O)O"),
  ("ammonia", "N"),
  ("aniline", "Nc1ccccc1"),
  ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
  ("benzene", "c1ccccc1"),
  ("benzoic acid", "OC(=O)c1ccccc1"),
  ("butane", "CCCC"),
  ("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
  ("carbon dioxide", "O=C=O"),
  ("carbon monoxide", "[C-]#[O+]"),
  ("carbon tetrachloride", "ClC(Cl)(Cl)Cl"),
  ("chloroform", "ClC(Cl)Cl"),
  ("chloromethane", "CCl"),
  ("cyclohexane", "C1CCCCC1"),
  ("cyclopropane", "C1CC1"),
  ("dichloromethane", "ClCCl"),
  ("diethyl ether", "CCOCC"),
  ("dimethyl ether", "COC"),
  ("ethane", "CC"),
  ("ethanol", "CCO"),
  ("ethylene", "C=C"),
  ("ethylene glycol", "OCCO"),
  ("formaldehyde", "C=O"),
  ("formic acid", "OC=O"),
  ("furan", "c1ccoc1"),
  ("glucose", "OCC1OC(O)C(O)C(O)C1O"),
  ("glycerol", "OCC(O)CO"),
  ("glycine", "NCC(=O)O"),
  ("hexane", "CCCCCC"),
  ("hydrochloric acid", "Cl"),
  ("hydrogen", "[H][H]"),
  ("hydrogen chloride", "Cl"),
  ("hydrogen peroxide", "OO"),
  ("hydrogen sulfide", "S"),
  ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
  ("imidazole", "c1c[nH]cn1"),
  ("isopropanol", "CC(C)O"),
  ("methane", "C"),
  ("methanol", "CO"),
  ("methylamine", "CN"),
  ("naphthalene", "c1ccc2ccccc2c1"),
  ("nicotine", "CN1CCCC1c1cccnc1"),
  ("nitric acid", "O[N+](=O)[O-]"),
  ("nitrogen", "N#N"),
  ("nitrous oxide", "[N-]=[N+]=O"),
  ("oxygen", "O=O"),
  ("ozone", "[O-][O+]=O"),
  ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
  ("pentane", "CCCCC"),
  ("phenol", "Oc1ccccc1"),
  ("phosphoric acid", "OP(=O)(O)O"),
  ("propane", "CCC"),
  ("pyridine", "c1ccncc1"),
  ("pyrrole", "c1cc[nH]c1"),
  ("sodium chloride", "[Na+].[Cl-]"),
  ("styrene", "C=Cc1ccccc1"),
  ("sulfur dioxide", "O=S=O"),
  ("sulfuric acid", "OS(=O)(=O)O"),
  ("tetrahydrofuran", "C1CCOC1"),
  ("toluene", "Cc1ccccc1"),
  ("urea", "NC(N)=O"),
  ("water", "O"),
];

fn smiles_for_name(name: &str) -> Option<&'static str> {
  let normalized = name
    .to_lowercase()
    .split_whitespace()
    .collect::<Vec<_>>()
    .join(" ");
  NAME_TO_SMILES
    .iter()
    .find(|(n, _)| *n == normalized)
    .map(|(_, s)| *s)
}

// ---------------------------------------------------------------------------
// SMILES parsing
// ---------------------------------------------------------------------------

/// Elements of the SMILES "organic subset", writable without brackets and
/// eligible for implicit hydrogen filling.
fn organic_subset_valences(symbol: &str) -> Option<&'static [u32]> {
  match symbol {
    "B" => Some(&[3]),
    "C" => Some(&[4]),
    "N" => Some(&[3, 5]),
    "O" => Some(&[2]),
    "P" => Some(&[3, 5]),
    "S" => Some(&[2, 4, 6]),
    "F" | "Cl" | "Br" | "I" => Some(&[1]),
    _ => None,
  }
}

/// Charge-adjusted valence: `B` gains a bond per negative charge, `C` loses
/// a bond per unit of charge of either sign, and the more electronegative
/// elements (N, O, P, S, halogens) gain a bond per positive charge.
fn adjust_valence(symbol: &str, valence: u32, charge: i64) -> u32 {
  let v = valence as i64;
  let adjusted = match symbol {
    "B" => v - charge,
    "C" => v - charge.abs(),
    _ => v + charge,
  };
  adjusted.max(0) as u32
}

struct SmilesParser<'a> {
  bytes: &'a [u8],
  pos: usize,
  graph: MolGraph,
  /// Atom the next bond attaches to.
  prev: Option<usize>,
  /// Open branch atoms.
  stack: Vec<usize>,
  /// Explicit bond symbol waiting for the next atom or ring closure.
  pending: Option<BondKind>,
  /// Open ring closures: number → (atom index, explicit bond at opening).
  rings: Vec<(u32, usize, Option<BondKind>)>,
}

impl<'a> SmilesParser<'a> {
  fn new(input: &'a str) -> Self {
    SmilesParser {
      bytes: input.as_bytes(),
      pos: 0,
      graph: MolGraph {
        atoms: Vec::new(),
        bonds: Vec::new(),
      },
      prev: None,
      stack: Vec::new(),
      pending: None,
      rings: Vec::new(),
    }
  }

  fn peek(&self) -> Option<u8> {
    self.bytes.get(self.pos).copied()
  }

  /// Default bond kind between two atoms without an explicit bond symbol.
  fn implied_bond(&self, a: usize, b: usize) -> BondKind {
    if self.graph.atoms[a].aromatic && self.graph.atoms[b].aromatic {
      BondKind::Aromatic
    } else {
      BondKind::Single
    }
  }

  fn add_atom(&mut self, atom: AtomData) {
    self.graph.atoms.push(atom);
    let idx = self.graph.atoms.len() - 1;
    if let Some(p) = self.prev {
      let kind = self
        .pending
        .take()
        .unwrap_or_else(|| self.implied_bond(p, idx));
      self.graph.bonds.push((p, idx, kind));
    } else {
      self.pending = None;
    }
    self.prev = Some(idx);
  }

  fn ring_closure(&mut self, number: u32) -> Option<()> {
    let cur = self.prev?;
    let here = self.pending.take();
    if let Some(open_pos) = self.rings.iter().position(|(n, _, _)| *n == number)
    {
      let (_, open_atom, open_bond) = self.rings.remove(open_pos);
      if open_atom == cur {
        return None; // ring bond to itself
      }
      // The bond symbol may be written at either end; they must agree.
      let kind = match (open_bond, here) {
        (Some(a), Some(b)) if a != b => return None,
        (Some(k), _) | (_, Some(k)) => k,
        (None, None) => self.implied_bond(open_atom, cur),
      };
      self.graph.bonds.push((open_atom, cur, kind));
    } else {
      self.rings.push((number, cur, here));
    }
    Some(())
  }

  /// Parse a bracket atom starting after `[`. Grammar:
  /// `[` isotope? symbol chiral? hcount? charge? class? `]`
  fn parse_bracket_atom(&mut self) -> Option<AtomData> {
    // isotope
    let mut mass: Option<i64> = None;
    while let Some(c) = self.peek() {
      if c.is_ascii_digit() {
        mass = Some(mass.unwrap_or(0) * 10 + (c - b'0') as i64);
        self.pos += 1;
      } else {
        break;
      }
    }
    // element symbol: uppercase (+ optional lowercase), or a lowercase
    // aromatic symbol
    let first = self.peek()?;
    let (symbol, aromatic) = if first.is_ascii_uppercase() {
      self.pos += 1;
      let mut sym = (first as char).to_string();
      if let Some(c) = self.peek()
        && c.is_ascii_lowercase()
        && is_element_abbreviation(&format!("{}{}", sym, c as char))
      {
        sym.push(c as char);
        self.pos += 1;
      }
      (sym, false)
    } else if first.is_ascii_lowercase() {
      // aromatic symbols allowed in brackets: b, c, n, o, p, s, se, as, te, si
      let two = self
        .bytes
        .get(self.pos..self.pos + 2)
        .map(|w| std::str::from_utf8(w).unwrap_or_default().to_string());
      match two.as_deref() {
        Some("se") | Some("as") | Some("te") | Some("si") => {
          self.pos += 2;
          let s = two.unwrap();
          let mut chars = s.chars();
          let cap: String = chars
            .next()
            .map(|c| c.to_ascii_uppercase())
            .into_iter()
            .chain(chars)
            .collect();
          (cap, true)
        }
        _ => {
          if !matches!(first, b'b' | b'c' | b'n' | b'o' | b'p' | b's') {
            return None;
          }
          self.pos += 1;
          ((first as char).to_ascii_uppercase().to_string(), true)
        }
      }
    } else {
      return None;
    };
    if !is_element_abbreviation(&symbol) {
      return None;
    }
    // chirality (ignored: stereochemistry is not represented)
    while self.peek() == Some(b'@') {
      self.pos += 1;
    }
    // hydrogen count
    let mut hcount: u32 = 0;
    if self.peek() == Some(b'H') {
      self.pos += 1;
      hcount = 1;
      let mut digits = 0u32;
      while let Some(c) = self.peek() {
        if c.is_ascii_digit() {
          if digits == 0 {
            hcount = 0;
          }
          hcount = hcount * 10 + (c - b'0') as u32;
          digits += 1;
          self.pos += 1;
        } else {
          break;
        }
      }
    }
    // charge: +, -, ++, --, +2, -3, …
    let mut charge: i64 = 0;
    if let Some(sign @ (b'+' | b'-')) = self.peek() {
      let unit: i64 = if sign == b'+' { 1 } else { -1 };
      self.pos += 1;
      charge = unit;
      if let Some(c) = self.peek()
        && c.is_ascii_digit()
      {
        let mut magnitude: i64 = 0;
        while let Some(c) = self.peek() {
          if c.is_ascii_digit() {
            magnitude = magnitude * 10 + (c - b'0') as i64;
            self.pos += 1;
          } else {
            break;
          }
        }
        charge = unit * magnitude;
      } else {
        while self.peek() == Some(sign) {
          charge += unit;
          self.pos += 1;
        }
      }
    }
    // atom class (ignored)
    if self.peek() == Some(b':') {
      self.pos += 1;
      let mut saw_digit = false;
      while let Some(c) = self.peek() {
        if c.is_ascii_digit() {
          saw_digit = true;
          self.pos += 1;
        } else {
          break;
        }
      }
      if !saw_digit {
        return None;
      }
    }
    if self.peek() != Some(b']') {
      return None;
    }
    self.pos += 1;
    Some(AtomData {
      symbol,
      formal_charge: charge,
      mass_number: mass,
      aromatic,
      explicit_h: Some(hcount),
    })
  }

  fn parse(mut self) -> Option<MolGraph> {
    while let Some(c) = self.peek() {
      match c {
        b'(' => {
          self.stack.push(self.prev?);
          self.pos += 1;
        }
        b')' => {
          if self.pending.is_some() {
            return None;
          }
          self.prev = Some(self.stack.pop()?);
          self.pos += 1;
        }
        b'-' => {
          self.pending = Some(BondKind::Single);
          self.pos += 1;
        }
        b'=' => {
          self.pending = Some(BondKind::Double);
          self.pos += 1;
        }
        b'#' => {
          self.pending = Some(BondKind::Triple);
          self.pos += 1;
        }
        b':' => {
          self.pending = Some(BondKind::Aromatic);
          self.pos += 1;
        }
        // Directional (stereo) bonds are treated as plain single bonds.
        b'/' | b'\\' => {
          self.pending = Some(BondKind::Single);
          self.pos += 1;
        }
        b'.' => {
          if self.pending.is_some() {
            return None;
          }
          self.prev = None;
          self.pos += 1;
        }
        b'%' => {
          self.pos += 1;
          let d1 = self.peek()?;
          self.pos += 1;
          let d2 = self.peek()?;
          self.pos += 1;
          if !d1.is_ascii_digit() || !d2.is_ascii_digit() {
            return None;
          }
          let number = ((d1 - b'0') as u32) * 10 + (d2 - b'0') as u32;
          self.ring_closure(number)?;
        }
        b'0'..=b'9' => {
          self.pos += 1;
          self.ring_closure((c - b'0') as u32)?;
        }
        b'[' => {
          self.pos += 1;
          let atom = self.parse_bracket_atom()?;
          self.add_atom(atom);
        }
        b'B' | b'C' | b'N' | b'O' | b'P' | b'S' | b'F' | b'I' => {
          self.pos += 1;
          let mut sym = (c as char).to_string();
          // two-letter organic-subset symbols: Cl, Br
          if let Some(next) = self.peek()
            && ((c == b'C' && next == b'l') || (c == b'B' && next == b'r'))
          {
            sym.push(next as char);
            self.pos += 1;
          }
          self.add_atom(AtomData::plain(&sym));
        }
        b'b' | b'c' | b'n' | b'o' | b'p' | b's' => {
          self.pos += 1;
          let mut atom =
            AtomData::plain(&(c as char).to_ascii_uppercase().to_string());
          atom.aromatic = true;
          self.add_atom(atom);
        }
        _ => return None,
      }
    }
    if !self.rings.is_empty()
      || !self.stack.is_empty()
      || self.pending.is_some()
      || self.graph.atoms.is_empty()
    {
      return None;
    }
    Some(self.graph)
  }
}

fn parse_smiles(input: &str) -> Option<MolGraph> {
  SmilesParser::new(input).parse()
}

// ---------------------------------------------------------------------------
// Hydrogen filling
// ---------------------------------------------------------------------------

/// Append explicit hydrogen atoms filling each atom's normal valence, the
/// way wolframscript canonicalizes molecules. Bracket SMILES atoms carry
/// their hydrogen count explicitly; other atoms of the organic subset are
/// filled up to the smallest charge-adjusted valence that fits the existing
/// bonds. Atoms that already take part in an aromatic bond but were not
/// parsed from SMILES (no aromatic flag) are left untouched: their hydrogen
/// count is not deducible without kekulization.
fn add_explicit_hydrogens(graph: &mut MolGraph) {
  let heavy_count = graph.atoms.len();
  let mut h_counts: Vec<u32> = vec![0; heavy_count];
  for (i, atom) in graph.atoms.iter().enumerate() {
    if let Some(h) = atom.explicit_h {
      h_counts[i] = h;
      continue;
    }
    if atom.symbol == "H" {
      continue;
    }
    let Some(valences) = organic_subset_valences(&atom.symbol) else {
      continue;
    };
    let mut bond_sum: u32 = 0;
    let mut has_aromatic_bond = false;
    for (a, b, kind) in &graph.bonds {
      if *a == i || *b == i {
        bond_sum += kind.order();
        if *kind == BondKind::Aromatic {
          has_aromatic_bond = true;
        }
      }
    }
    if atom.aromatic || has_aromatic_bond {
      if !atom.aromatic {
        continue;
      }
      // One valence slot is consumed by the delocalized π-system.
      bond_sum += 1;
    }
    let h = valences
      .iter()
      .map(|v| adjust_valence(&atom.symbol, *v, atom.formal_charge))
      .filter(|v| *v >= bond_sum)
      .map(|v| v - bond_sum)
      .next()
      .unwrap_or(0);
    h_counts[i] = h;
  }
  for (i, count) in h_counts.iter().enumerate() {
    for _ in 0..*count {
      graph.atoms.push(AtomData::plain("H"));
      let h_idx = graph.atoms.len() - 1;
      graph.bonds.push((i, h_idx, BondKind::Single));
    }
  }
}

// ---------------------------------------------------------------------------
// Conversion between MolGraph and Expr
// ---------------------------------------------------------------------------

fn rule(key: &str, value: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(Expr::String(key.to_string())),
    replacement: Box::new(value),
  }
}

fn atom_to_expr(atom: &AtomData) -> Expr {
  let mut args = vec![Expr::String(atom.symbol.clone())];
  if atom.formal_charge != 0 {
    args.push(rule(
      "FormalCharge",
      Expr::Integer(atom.formal_charge as i128),
    ));
  }
  if let Some(mass) = atom.mass_number {
    args.push(rule("MassNumber", Expr::Integer(mass as i128)));
  }
  Expr::FunctionCall {
    name: "Atom".to_string(),
    args: args.into(),
  }
}

fn bond_to_expr(bond: &(usize, usize, BondKind)) -> Expr {
  Expr::FunctionCall {
    name: "Bond".to_string(),
    args: vec![
      Expr::List(
        vec![
          Expr::Integer((bond.0 + 1) as i128),
          Expr::Integer((bond.1 + 1) as i128),
        ]
        .into(),
      ),
      Expr::String(bond.2.as_str().to_string()),
    ]
    .into(),
  }
}

fn graph_to_expr(graph: &MolGraph) -> Expr {
  Expr::FunctionCall {
    name: "Molecule".to_string(),
    args: vec![
      Expr::List(
        graph
          .atoms
          .iter()
          .map(atom_to_expr)
          .collect::<Vec<_>>()
          .into(),
      ),
      Expr::List(
        graph
          .bonds
          .iter()
          .map(bond_to_expr)
          .collect::<Vec<_>>()
          .into(),
      ),
    ]
    .into(),
  }
}

/// Parse one atom specification: `"C"`, an atomic number, or
/// `Atom["C", "FormalCharge" -> -1, "MassNumber" -> 13]`.
fn atom_from_expr(expr: &Expr) -> Option<AtomData> {
  match expr {
    Expr::String(sym) => {
      if is_element_abbreviation(sym) {
        Some(AtomData::plain(sym))
      } else {
        None
      }
    }
    Expr::Integer(z) => abbreviation_for_atomic_number(*z).map(AtomData::plain),
    Expr::FunctionCall { name, args } if name == "Atom" && !args.is_empty() => {
      let Expr::String(sym) = &args[0] else {
        return None;
      };
      if !is_element_abbreviation(sym) {
        return None;
      }
      let mut atom = AtomData::plain(sym);
      for opt in &args[1..] {
        let Expr::Rule {
          pattern,
          replacement,
        } = opt
        else {
          return None;
        };
        let Expr::String(key) = pattern.as_ref() else {
          return None;
        };
        let Expr::Integer(value) = replacement.as_ref() else {
          return None;
        };
        match key.as_str() {
          "FormalCharge" => atom.formal_charge = *value as i64,
          "MassNumber" => atom.mass_number = Some(*value as i64),
          _ => return None,
        }
      }
      Some(atom)
    }
    _ => None,
  }
}

/// Parse one bond specification: `Bond[{i, j}]` or `Bond[{i, j}, "type"]`,
/// with 1-based indices into an atom list of length `atom_count`.
fn bond_from_expr(
  expr: &Expr,
  atom_count: usize,
) -> Option<(usize, usize, BondKind)> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Bond" || args.is_empty() || args.len() > 2 {
    return None;
  }
  let Expr::List(pair) = &args[0] else {
    return None;
  };
  if pair.len() != 2 {
    return None;
  }
  let (Expr::Integer(i), Expr::Integer(j)) = (&pair[0], &pair[1]) else {
    return None;
  };
  let kind = match args.get(1) {
    None => BondKind::Single,
    Some(Expr::String(s)) => BondKind::from_str(s)?,
    Some(_) => return None,
  };
  let (i, j) = (*i, *j);
  let n = atom_count as i128;
  if i < 1 || j < 1 || i > n || j > n || i == j {
    return None;
  }
  Some(((i - 1) as usize, (j - 1) as usize, kind))
}

/// Build a molecular graph from explicit atom and bond lists. Returns None
/// if any specification is invalid.
fn graph_from_lists(atoms: &[Expr], bonds: &[Expr]) -> Option<MolGraph> {
  let atom_data: Vec<AtomData> = atoms
    .iter()
    .map(atom_from_expr)
    .collect::<Option<Vec<_>>>()?;
  let bond_data: Vec<(usize, usize, BondKind)> = bonds
    .iter()
    .map(|b| bond_from_expr(b, atom_data.len()))
    .collect::<Option<Vec<_>>>()?;
  Some(MolGraph {
    atoms: atom_data,
    bonds: bond_data,
  })
}

/// A molecule reduced to the atoms and bonds that a 2-D structure diagram
/// actually draws. Hydrogens bonded to a carbon that has at least one heavy
/// (non-hydrogen) neighbor are suppressed — the standard skeletal-formula
/// convention — while hydrogens on heteroatoms (and on an otherwise bare
/// carbon, e.g. methane) are kept and drawn explicitly.
pub struct DrawMolecule {
  /// `(symbol, formal_charge, mass_number)` for each drawn atom.
  pub atoms: Vec<(String, i64, Option<i64>)>,
  /// `(atom_i, atom_j, bond_kind)` with 0-based indices into `atoms`.
  pub bonds: Vec<(usize, usize, &'static str)>,
}

/// Summary facts shown in a molecule's information tile:
/// `(molecular_formula, atom_count, bond_count)`. `None` if `expr` is not a
/// valid molecule.
pub fn molecule_info(expr: &Expr) -> Option<(String, i128, i128)> {
  let formula = match molecule_property_from_expr(expr, "MolecularFormula")? {
    Expr::String(ref s) => s.clone(),
    _ => return None,
  };
  let atoms = match molecule_property_from_expr(expr, "AtomCount")? {
    Expr::Integer(n) => n,
    _ => return None,
  };
  let bonds = match molecule_property_from_expr(expr, "BondCount")? {
    Expr::Integer(n) => n,
    _ => return None,
  };
  Some((formula, atoms, bonds))
}

/// Build the drawable (skeletal) view of a canonical molecule expression.
/// Returns `None` when `expr` is not a valid `Molecule[{atoms…}, {bonds…}]`.
pub fn drawable_molecule(expr: &Expr) -> Option<DrawMolecule> {
  let graph = graph_from_molecule_expr(expr)?;
  let n = graph.atoms.len();
  let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
  for (a, b, _) in &graph.bonds {
    neighbors[*a].push(*b);
    neighbors[*b].push(*a);
  }
  let is_h = |i: usize| graph.atoms[i].symbol == "H";
  let heavy_degree =
    |i: usize| neighbors[i].iter().filter(|&&j| !is_h(j)).count();

  // Decide which atoms survive into the skeletal drawing.
  let mut keep = vec![true; n];
  for i in 0..n {
    if is_h(i)
      && let Some(&j) = neighbors[i].first()
      && graph.atoms[j].symbol == "C"
      && heavy_degree(j) >= 1
    {
      keep[i] = false;
    }
  }

  let mut new_index = vec![usize::MAX; n];
  let mut atoms = Vec::new();
  for (i, atom) in graph.atoms.iter().enumerate() {
    if keep[i] {
      new_index[i] = atoms.len();
      atoms.push((atom.symbol.clone(), atom.formal_charge, atom.mass_number));
    }
  }
  let mut bonds = Vec::new();
  for (a, b, kind) in &graph.bonds {
    if keep[*a] && keep[*b] {
      bonds.push((new_index[*a], new_index[*b], kind.as_str()));
    }
  }
  Some(DrawMolecule { atoms, bonds })
}

/// Extract the validated graph from an already-evaluated
/// `Molecule[{atoms…}, {bonds…}]` expression.
fn graph_from_molecule_expr(expr: &Expr) -> Option<MolGraph> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Molecule" || args.len() != 2 {
    return None;
  }
  let (Expr::List(atoms), Expr::List(bonds)) = (&args[0], &args[1]) else {
    return None;
  };
  graph_from_lists(atoms, bonds)
}

// ---------------------------------------------------------------------------
// Molecular formula (Hill order)
// ---------------------------------------------------------------------------

fn molecular_formula(graph: &MolGraph) -> String {
  let mut counts: Vec<(String, usize)> = Vec::new();
  for atom in &graph.atoms {
    match counts.iter_mut().find(|(s, _)| *s == atom.symbol) {
      Some((_, c)) => *c += 1,
      None => counts.push((atom.symbol.clone(), 1)),
    }
  }
  let has_carbon = counts.iter().any(|(s, _)| s == "C");
  counts.sort_by(|(a, _), (b, _)| {
    let rank = |s: &str| -> (u8, String) {
      if has_carbon {
        match s {
          "C" => (0, String::new()),
          "H" => (1, String::new()),
          _ => (2, s.to_string()),
        }
      } else {
        (2, s.to_string())
      }
    };
    rank(a).cmp(&rank(b))
  });
  let mut formula = String::new();
  for (symbol, count) in &counts {
    formula.push_str(symbol);
    if *count > 1 {
      formula.push_str(&count.to_string());
    }
  }
  let net_charge: i64 = graph.atoms.iter().map(|a| a.formal_charge).sum();
  match net_charge {
    0 => {}
    1 => formula.push('+'),
    -1 => formula.push('-'),
    c if c > 1 => formula.push_str(&format!("+{}", c)),
    c => formula.push_str(&format!("-{}", -c)),
  }
  formula
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Molecule[spec] / Molecule[{atoms…}, {bonds…}] — build the canonical
/// molecule expression.
pub fn molecule_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args {
    [Expr::String(spec)] => {
      let graph = match smiles_for_name(spec) {
        Some(smiles) => parse_smiles(smiles),
        None => parse_smiles(spec),
      };
      match graph {
        Some(mut graph) => {
          add_explicit_hydrogens(&mut graph);
          Ok(graph_to_expr(&graph))
        }
        None => {
          crate::emit_message(&format!(
            "Molecule::nointerp: Unable to interpret the input \"{}\" as a molecule.",
            spec
          ));
          Ok(unevaluated("Molecule", args))
        }
      }
    }
    [Expr::List(atoms), Expr::List(bonds)] => {
      match graph_from_lists(atoms, bonds) {
        Some(mut graph) => {
          add_explicit_hydrogens(&mut graph);
          Ok(graph_to_expr(&graph))
        }
        None => Ok(unevaluated("Molecule", args)),
      }
    }
    _ => Ok(unevaluated("Molecule", args)),
  }
}

/// MoleculeQ[expr] — True for a valid canonical molecule expression.
pub fn molecule_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(unevaluated("MoleculeQ", args));
  }
  let valid = graph_from_molecule_expr(&args[0]).is_some();
  Ok(Expr::Identifier(
    if valid { "True" } else { "False" }.to_string(),
  ))
}

/// AtomList[mol] — the list of atoms.
pub fn atom_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1
    && let Some(result) = molecule_property_from_expr(&args[0], "AtomList")
  {
    return Ok(result);
  }
  Ok(unevaluated("AtomList", args))
}

/// BondList[mol] — the list of bonds.
pub fn bond_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1
    && let Some(result) = molecule_property_from_expr(&args[0], "BondList")
  {
    return Ok(result);
  }
  Ok(unevaluated("BondList", args))
}

/// MoleculeValue[mol, "prop"] / MoleculeValue[mol, {props…}].
pub fn molecule_value_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 2 {
    match &args[1] {
      Expr::String(prop) => {
        if let Some(result) = molecule_property_from_expr(&args[0], prop) {
          return Ok(result);
        }
      }
      Expr::List(props) => {
        let results: Option<Vec<Expr>> = props
          .iter()
          .map(|p| match p {
            Expr::String(prop) => molecule_property_from_expr(&args[0], prop),
            _ => None,
          })
          .collect();
        if let Some(results) = results {
          return Ok(Expr::List(results.into()));
        }
      }
      _ => {}
    }
  }
  Ok(unevaluated("MoleculeValue", args))
}

/// Property access on a molecule expression. Returns None when the
/// expression is not a valid molecule or the property is unsupported,
/// leaving the call unevaluated.
fn molecule_property_from_expr(mol: &Expr, prop: &str) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = mol else {
    return None;
  };
  if name != "Molecule" || args.len() != 2 {
    return None;
  }
  molecule_property(args, prop)
}

/// Property access on the argument list of a valid canonical molecule.
/// Used both by MoleculeValue and by subvalue application
/// (`Molecule[…]["AtomCount"]`).
pub fn molecule_property(mol_args: &[Expr], prop: &str) -> Option<Expr> {
  if mol_args.len() != 2 {
    return None;
  }
  let graph = graph_from_lists(
    match &mol_args[0] {
      Expr::List(atoms) => atoms,
      _ => return None,
    },
    match &mol_args[1] {
      Expr::List(bonds) => bonds,
      _ => return None,
    },
  )?;
  match prop {
    "AtomCount" => Some(Expr::Integer(graph.atoms.len() as i128)),
    "BondCount" => Some(Expr::Integer(graph.bonds.len() as i128)),
    "AtomList" => Some(Expr::List(
      graph
        .atoms
        .iter()
        .map(atom_to_expr)
        .collect::<Vec<_>>()
        .into(),
    )),
    "BondList" => Some(Expr::List(
      graph
        .bonds
        .iter()
        .map(bond_to_expr)
        .collect::<Vec<_>>()
        .into(),
    )),
    "MolecularFormula" => Some(Expr::String(molecular_formula(&graph))),
    "NetCharge" => Some(Expr::Integer(
      graph.atoms.iter().map(|a| a.formal_charge as i128).sum(),
    )),
    _ => None,
  }
}
