//! 2-D structure-diagram rendering for `Molecule[…]` expressions.
//!
//! A canonical molecule (`Molecule[{Atom[…], …}, {Bond[…], …}]`) carries only
//! connectivity, not geometry. To draw it we:
//!
//!   1. reduce it to a skeletal graph (implicit hydrogens on carbon dropped;
//!      see [`crate::functions::molecule_ast::drawable_molecule`]),
//!   2. generate 2-D atom coordinates with a spring-embedder (Fruchterman–
//!      Reingold) seeded by a bond-angle-aware breadth-first placement, and
//!   3. emit an SVG with skeletal bonds (single / double / triple / aromatic),
//!      heteroatom labels, formal charges, and isotope mass numbers.
//!
//! The layout is fully deterministic (no randomness), so the same molecule
//! always renders identically.

#[allow(unused_imports)]
use super::*;
use crate::functions::molecule_ast::{
  DrawMolecule, drawable_molecule_with, molecule_ast,
};

/// `MoleculePlot[mol, opts…]` — the 2-D structure diagram as a `Graphics`
/// object. The argument may be a `Molecule[…]` object or any specification a
/// plain `Molecule[…]` call accepts (a chemical name or SMILES string).
/// Returns the call unevaluated if it does not describe a molecule or a
/// trailing argument is not an option.
pub fn molecule_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("MoleculePlot", args));
  let Some((mol_expr, opts)) = molecule_and_options(args)? else {
    return unevaluated();
  };
  match molecule_to_svg_with(&mol_expr, &opts) {
    Some(svg) => Ok(crate::graphics_result(svg)),
    None => unevaluated(),
  }
}

/// `MoleculePlot3D[mol, opts…]` — a ball-and-stick model of the molecule
/// with every hydrogen shown, as a `Graphics3D` object. Atom coordinates come
/// from a deterministic 3-D embedding that keeps bond lengths at the sum of
/// the covalent radii and spreads each atom's bonds at the angle its
/// hybridization implies.
pub fn molecule_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("MoleculePlot3D", args));
  let Some((mol_expr, opts)) = molecule_and_options(args)? else {
    return unevaluated();
  };
  let Some(mol) = drawable_molecule_with(&mol_expr, true) else {
    return unevaluated();
  };
  if mol.atoms.is_empty() {
    return unevaluated();
  }
  let coords = layout_3d(&mol);
  let graphics = ball_and_stick(&mol, &coords, &opts);
  crate::evaluator::evaluate_expr_to_expr(&graphics)
}

/// Options of `MoleculePlot` / `MoleculePlot3D` that change the drawing.
/// Options that only concern the surrounding frame (`ImagePadding`, …) are
/// accepted and need no handling.
#[derive(Default)]
pub struct MoleculePlotOptions {
  /// `PlotTheme -> "AllAtom"`: draw every hydrogen explicitly.
  pub all_atom: bool,
  /// `PlotTheme -> "Monochrome"`: draw atom labels in the bond color.
  pub monochrome: bool,
  /// The `ImageSize` value, if one was given.
  pub image_size: Option<Expr>,
  /// The `AtomLabels` value, if one was given.
  pub atom_labels: Option<Expr>,
}

/// Split `MoleculePlot`-style arguments into the canonical `Molecule[…]` and
/// its options. `None` when the first argument does not describe a molecule
/// or a later argument is not a rule.
fn molecule_and_options(
  args: &[Expr],
) -> Result<Option<(Expr, MoleculePlotOptions)>, InterpreterError> {
  let Some(first) = args.first() else {
    return Ok(None);
  };
  let mut opts = MoleculePlotOptions::default();
  for opt in &args[1..] {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = opt
    else {
      return Ok(None);
    };
    let Expr::Identifier(name) = pattern.as_ref() else {
      return Ok(None);
    };
    match name.as_str() {
      "PlotTheme" => apply_plot_theme(replacement, &mut opts),
      "ImageSize" => opts.image_size = Some(replacement.as_ref().clone()),
      "AtomLabels" => opts.atom_labels = Some(replacement.as_ref().clone()),
      _ => {}
    }
  }
  let mol_expr = match first {
    Expr::FunctionCall { name, .. } if name == "Molecule" => first.clone(),
    other => match molecule_ast(std::slice::from_ref(other))? {
      e @ Expr::FunctionCall { .. } if matches!(&e, Expr::FunctionCall { name, .. } if name == "Molecule") => {
        e
      }
      _ => return Ok(None),
    },
  };
  Ok(Some((mol_expr, opts)))
}

/// Read a `PlotTheme` value: a theme name or a list of them. Entries that are
/// not names (such as `None` from an unset control) are skipped.
fn apply_plot_theme(theme: &Expr, opts: &mut MoleculePlotOptions) {
  match theme {
    Expr::List(items) => {
      for item in items {
        apply_plot_theme(item, opts);
      }
    }
    Expr::String(name) => match name.as_str() {
      "AllAtom" => opts.all_atom = true,
      "Monochrome" => opts.monochrome = true,
      _ => {}
    },
    _ => {}
  }
}

/// Render a molecule's 2-D structure diagram (the `MoleculePlot[…]` graphic)
/// to a self-contained SVG string, or `None` if `expr` is not a valid
/// molecule.
pub fn molecule_to_svg(expr: &Expr) -> Option<String> {
  molecule_to_svg_with(expr, &MoleculePlotOptions::default())
}

/// [`molecule_to_svg`] honoring `MoleculePlot` options.
pub fn molecule_to_svg_with(
  expr: &Expr,
  opts: &MoleculePlotOptions,
) -> Option<String> {
  let mol = drawable_molecule_with(expr, opts.all_atom)?;
  if mol.atoms.is_empty() {
    return None;
  }
  let coords = layout(&mol);
  let (body, ox, oy, w, h) = render_parts(&mol, &coords, opts);
  let (out_w, out_h) = opts
    .image_size
    .as_ref()
    .and_then(|size| {
      crate::functions::plot::parse_image_size(
        size,
        w.round().max(1.0) as u32,
        h.round().max(1.0) as u32,
      )
    })
    .map_or((w, h), |(iw, ih, _)| (iw as f64, ih as f64));
  Some(format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{out_w:.0}\" \
     height=\"{out_h:.0}\" viewBox=\"{ox:.2} {oy:.2} {w:.2} {h:.2}\" \
     stroke-linecap=\"round\">{body}</svg>"
  ))
}

/// Whether `AtomLabels` asks for atom `i`'s index to be shown. Accepts
/// `"AtomIndex"` for every atom, or rules (single or in a list) whose
/// left-hand side is `All`, an element symbol string, or `Atom["sym"]`.
fn shows_atom_index(labels: &Expr, symbol: &str) -> bool {
  let matches_atom = |lhs: &Expr| match lhs {
    Expr::Identifier(all) => all == "All",
    Expr::String(sym) => sym == symbol,
    Expr::FunctionCall { name, args } if name == "Atom" && args.len() == 1 => {
      matches!(&args[0], Expr::String(sym) if sym == symbol)
    }
    _ => false,
  };
  match labels {
    Expr::String(s) => s == "AtomIndex",
    Expr::List(rules) => rules.iter().any(|r| shows_atom_index(r, symbol)),
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      matches!(replacement.as_ref(), Expr::String(s) if s == "AtomIndex")
        && matches_atom(pattern)
    }
    _ => false,
  }
}

/// Target on-screen bond length, in pixels.
const BOND_PX: f64 = 42.0;
/// Half the separation between the two strokes of a double bond, in pixels.
const DOUBLE_GAP: f64 = 5.0;
/// Margin around the drawing, in pixels.
const MARGIN: f64 = 22.0;
/// Font size for atom labels, in pixels.
const FONT: f64 = 17.0;
/// Color of `AtomLabels -> "AtomIndex"` numbers.
const INDEX_COLOR: &str = "#3c64b4";

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

type Pt = (f64, f64);

/// Compute 2-D coordinates (in bond-length units) for every atom. Connected
/// components are laid out independently, then packed left to right so that
/// salts and other multi-fragment species do not overlap.
fn layout(mol: &DrawMolecule) -> Vec<Pt> {
  let n = mol.atoms.len();
  let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
  for (a, b, _) in &mol.bonds {
    adj[*a].push(*b);
    adj[*b].push(*a);
  }

  let mut coords = vec![(0.0, 0.0); n];
  let mut seen = vec![false; n];
  let mut x_cursor = 0.0;
  for start in 0..n {
    if seen[start] {
      continue;
    }
    // Gather this component's atoms.
    let mut component = Vec::new();
    let mut stack = vec![start];
    seen[start] = true;
    while let Some(u) = stack.pop() {
      component.push(u);
      for &v in &adj[u] {
        if !seen[v] {
          seen[v] = true;
          stack.push(v);
        }
      }
    }
    let local = layout_component(&component, &adj);
    // Shift so the component sits to the right of everything drawn so far.
    let (min_x, min_y, max_x) = local.iter().fold(
      (f64::MAX, f64::MAX, f64::MIN),
      |(mnx, mny, mxx), &(x, y)| (mnx.min(x), mny.min(y), mxx.max(x)),
    );
    let shift_x = x_cursor - min_x + if x_cursor > 0.0 { 1.5 } else { 0.0 };
    for (k, &atom) in component.iter().enumerate() {
      coords[atom] = (local[k].0 + shift_x, local[k].1 - min_y);
    }
    x_cursor += (max_x - min_x) + if x_cursor > 0.0 { 1.5 } else { 0.0 };
  }
  coords
}

/// Lay out a single connected component: a bond-angle-aware BFS seed followed
/// by spring-embedder relaxation. Returns coordinates indexed to match
/// `component` order.
fn layout_component(component: &[usize], adj: &[Vec<usize>]) -> Vec<Pt> {
  let m = component.len();
  if m == 1 {
    return vec![(0.0, 0.0)];
  }
  // Map global atom index -> local (0..m) index.
  let mut local_of = std::collections::HashMap::new();
  for (k, &a) in component.iter().enumerate() {
    local_of.insert(a, k);
  }
  let local_adj: Vec<Vec<usize>> = component
    .iter()
    .map(|&a| adj[a].iter().map(|v| local_of[v]).collect())
    .collect();

  let mut pos = seed_positions(&local_adj);
  fruchterman_reingold(&mut pos, &local_adj);
  pos
}

/// Seed atom positions with a breadth-first placement that spreads a vertex's
/// bonds at ~120° increments — the geometry a spring embedder converges toward
/// but reaches faster and more reliably from a good start.
fn seed_positions(adj: &[Vec<usize>]) -> Vec<Pt> {
  let m = adj.len();
  let mut pos = vec![(0.0, 0.0); m];
  let mut placed = vec![false; m];
  // Direction (radians) of the bond that arrived at each atom.
  let mut in_dir = vec![0.0f64; m];
  let mut depth = vec![0usize; m];

  // Start from the highest-degree atom for a compact core.
  let start = (0..m).max_by_key(|&i| adj[i].len()).unwrap_or(0);
  placed[start] = true;
  let mut queue = std::collections::VecDeque::new();
  queue.push_back(start);

  while let Some(u) = queue.pop_front() {
    let children: Vec<usize> =
      adj[u].iter().copied().filter(|&v| !placed[v]).collect();
    if children.is_empty() {
      continue;
    }
    let base = in_dir[u];
    let is_root = u == start;
    let sign = if depth[u].is_multiple_of(2) {
      1.0
    } else {
      -1.0
    };
    let angles = child_angles(base, children.len(), is_root, sign);
    for (child, ang) in children.into_iter().zip(angles) {
      pos[child] = (pos[u].0 + ang.cos(), pos[u].1 + ang.sin());
      in_dir[child] = ang;
      depth[child] = depth[u] + 1;
      placed[child] = true;
      queue.push_back(child);
    }
  }
  pos
}

/// Bond directions (radians) for the `count` new neighbors of an atom whose
/// incoming bond points along `base`. A root spreads its bonds evenly around
/// the full circle; other atoms fan their bonds forward, keeping ~120° from
/// the incoming bond and alternating the turn direction (`sign`) to trace the
/// zig-zag of a skeletal chain.
fn child_angles(base: f64, count: usize, is_root: bool, sign: f64) -> Vec<f64> {
  use std::f64::consts::PI;
  let deg = PI / 180.0;
  if is_root {
    return (0..count)
      .map(|i| 2.0 * PI * i as f64 / count as f64)
      .collect();
  }
  match count {
    1 => vec![base + sign * 60.0 * deg],
    2 => vec![base + 60.0 * deg, base - 60.0 * deg],
    3 => vec![base, base + 120.0 * deg, base - 120.0 * deg],
    k => (0..k)
      .map(|i| base + 120.0 * deg - i as f64 * (240.0 * deg / (k - 1) as f64))
      .collect(),
  }
}

/// Fruchterman–Reingold spring embedder with unit ideal edge length. Repulsion
/// acts between every atom pair, attraction along every bond; a cooling
/// schedule caps per-step motion so the layout settles instead of oscillating.
fn fruchterman_reingold(pos: &mut [Pt], adj: &[Vec<usize>]) {
  let m = pos.len();
  let k = 1.0; // ideal edge length
  let iterations = (400 + 20 * m).min(1200);
  let mut temp = 0.35;
  let cool = 0.985;

  for _ in 0..iterations {
    let mut disp = vec![(0.0, 0.0); m];
    // Repulsion between all pairs.
    for i in 0..m {
      for j in (i + 1)..m {
        let mut dx = pos[i].0 - pos[j].0;
        let mut dy = pos[i].1 - pos[j].1;
        let mut d = (dx * dx + dy * dy).sqrt();
        if d < 1e-4 {
          // Deterministic nudge for coincident atoms.
          dx = ((i * 7 + 3) % 5) as f64 * 1e-3 - 2e-3;
          dy = ((j * 5 + 1) % 5) as f64 * 1e-3 - 2e-3;
          d = (dx * dx + dy * dy).sqrt().max(1e-4);
        }
        let force = k * k / d;
        let (ux, uy) = (dx / d, dy / d);
        disp[i].0 += ux * force;
        disp[i].1 += uy * force;
        disp[j].0 -= ux * force;
        disp[j].1 -= uy * force;
      }
    }
    // Attraction along bonds.
    for (i, neigh) in adj.iter().enumerate() {
      for &j in neigh {
        if j <= i {
          continue; // each undirected edge once
        }
        let dx = pos[i].0 - pos[j].0;
        let dy = pos[i].1 - pos[j].1;
        let d = (dx * dx + dy * dy).sqrt().max(1e-4);
        let force = d * d / k;
        let (ux, uy) = (dx / d, dy / d);
        disp[i].0 -= ux * force;
        disp[i].1 -= uy * force;
        disp[j].0 += ux * force;
        disp[j].1 += uy * force;
      }
    }
    // Apply, capped by the current temperature.
    for i in 0..m {
      let (dx, dy) = disp[i];
      let d = (dx * dx + dy * dy).sqrt();
      if d > 1e-9 {
        let step = d.min(temp);
        pos[i].0 += dx / d * step;
        pos[i].1 += dy / d * step;
      }
    }
    temp = (temp * cool).max(0.005);
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Standard CPK-inspired label colors for the heteroatoms Woxi draws. Carbon
/// and hydrogen fall back to the theme's primary text color so they stay
/// legible in both light and dark mode.
fn atom_color(symbol: &str) -> Option<&'static str> {
  match symbol {
    "O" => Some("#e00000"),
    "N" => Some("#3050f8"),
    "S" => Some("#c8a000"),
    "P" => Some("#ff8000"),
    "F" | "Cl" => Some("#20a020"),
    "Br" => Some("#a02020"),
    "I" => Some("#8000d0"),
    "B" => Some("#c08050"),
    _ => None,
  }
}

/// Whether an atom is drawn as a text label (vs. an implicit vertex). Carbon
/// with heavy neighbors is a bare vertex; everything else gets a label.
fn has_label(mol: &DrawMolecule, i: usize) -> bool {
  let (sym, ..) = &mol.atoms[i];
  if sym != "C" {
    return true;
  }
  // A carbon is labeled only when it has no heavy neighbor (e.g. methane),
  // which shows up as having only hydrogens among its drawn neighbors. With
  // every hydrogen drawn (`"AllAtom"`), a carbon in a chain stays a vertex.
  let neighbors = mol.bonds.iter().filter_map(|&(a, b, _)| {
    if a == i {
      Some(b)
    } else if b == i {
      Some(a)
    } else {
      None
    }
  });
  let mut has_h = false;
  for j in neighbors {
    if mol.atoms[j].0 == "H" {
      has_h = true;
    } else {
      return false;
    }
  }
  has_h
}

/// Render the structure diagram's inner markup plus its viewBox bounds
/// (`body`, `ox`, `oy`, `w`, `h`).
fn render_parts(
  mol: &DrawMolecule,
  coords: &[Pt],
  opts: &MoleculePlotOptions,
) -> (String, f64, f64, f64, f64) {
  let stroke = crate::functions::graphics::theme().text_primary;
  // Scale to pixels; flip Y so the diagram is not upside down.
  let px: Vec<Pt> = coords
    .iter()
    .map(|&(x, y)| (x * BOND_PX, -y * BOND_PX))
    .collect();

  let labeled: Vec<bool> =
    (0..mol.atoms.len()).map(|i| has_label(mol, i)).collect();
  // How far a bond stops short of a labeled atom's center.
  let trim = |i: usize| if labeled[i] { FONT * 0.62 } else { 0.0 };

  let mut body = String::new();

  // --- Bonds ---
  for &(a, b, kind) in &mol.bonds {
    let (pa, pb) = (px[a], px[b]);
    let (ta, tb) = trim_segment(pa, pb, trim(a), trim(b));
    match kind {
      "Double" => {
        let internal = other_neighbor_dir(mol, &px, a, b).is_some()
          && other_neighbor_dir(mol, &px, b, a).is_some();
        if internal {
          // Ring/interior double bond: main line plus a shortened inner line.
          draw_line(&mut body, ta, tb, stroke, 1.5);
          let side = double_bond_side(mol, &px, a, b);
          let (ia, ib) = offset_segment(ta, tb, side, DOUBLE_GAP);
          let (ia, ib) = shrink(ia, ib, 0.16);
          draw_line(&mut body, ia, ib, stroke, 1.5);
        } else {
          // Terminal double bond (e.g. C=O): two symmetric lines.
          let (a1, b1) = offset_segment(ta, tb, 1.0, DOUBLE_GAP * 0.5);
          let (a2, b2) = offset_segment(ta, tb, -1.0, DOUBLE_GAP * 0.5);
          draw_line(&mut body, a1, b1, stroke, 1.5);
          draw_line(&mut body, a2, b2, stroke, 1.5);
        }
      }
      "Triple" => {
        draw_line(&mut body, ta, tb, stroke, 1.5);
        let (a1, b1) = offset_segment(ta, tb, 1.0, DOUBLE_GAP);
        let (a2, b2) = offset_segment(ta, tb, -1.0, DOUBLE_GAP);
        draw_line(&mut body, a1, b1, stroke, 1.5);
        draw_line(&mut body, a2, b2, stroke, 1.5);
      }
      "Aromatic" => {
        // Draw the sigma bond plus an inner line toward the ring center.
        draw_line(&mut body, ta, tb, stroke, 1.5);
        let side = double_bond_side(mol, &px, a, b);
        let (ia, ib) = offset_segment(ta, tb, side, DOUBLE_GAP);
        let (ia, ib) = shrink(ia, ib, 0.16);
        draw_line(&mut body, ia, ib, stroke, 1.3);
      }
      _ => draw_line(&mut body, ta, tb, stroke, 1.5),
    }
  }

  // --- Atom labels ---
  for i in 0..mol.atoms.len() {
    if !labeled[i] {
      continue;
    }
    let (sym, charge, mass) = &mol.atoms[i];
    let (x, y) = px[i];
    let color = if opts.monochrome {
      stroke
    } else {
      atom_color(sym).unwrap_or(stroke)
    };
    body.push_str(&format!(
      "<text x=\"{x:.2}\" y=\"{y:.2}\" font-family=\"Helvetica, Arial, \
       sans-serif\" font-size=\"{FONT:.1}\" fill=\"{color}\" \
       text-anchor=\"middle\" dominant-baseline=\"central\">{sym}</text>"
    ));
    // Formal charge as a superscript to the upper-right of the symbol.
    if *charge != 0 {
      let label = charge_label(*charge);
      body.push_str(&format!(
        "<text x=\"{sx:.2}\" y=\"{sy:.2}\" font-family=\"Helvetica, Arial, \
         sans-serif\" font-size=\"{fs:.1}\" fill=\"{color}\" \
         text-anchor=\"start\" dominant-baseline=\"central\">{label}</text>",
        sx = x + FONT * 0.42,
        sy = y - FONT * 0.42,
        fs = FONT * 0.68,
      ));
    }
    // Isotope mass number as a superscript to the upper-left.
    if let Some(mass) = mass {
      body.push_str(&format!(
        "<text x=\"{sx:.2}\" y=\"{sy:.2}\" font-family=\"Helvetica, Arial, \
         sans-serif\" font-size=\"{fs:.1}\" fill=\"{color}\" \
         text-anchor=\"end\" dominant-baseline=\"central\">{mass}</text>",
        sx = x - FONT * 0.42,
        sy = y - FONT * 0.42,
        fs = FONT * 0.68,
      ));
    }
  }

  // --- Atom indices (`AtomLabels -> "AtomIndex"`) ---
  if let Some(labels) = &opts.atom_labels {
    for i in 0..mol.atoms.len() {
      if !shows_atom_index(labels, &mol.atoms[i].0) {
        continue;
      }
      // Place the index on the side facing away from the atom's bonds.
      let (x, y) = px[i];
      let (dx, dy) = other_neighbor_dir(mol, &px, i, usize::MAX)
        .map_or((1.0, -1.0), |(ox, oy)| (-ox, -oy));
      let len = (dx * dx + dy * dy).sqrt();
      let (ux, uy) = if len > 1e-6 {
        (dx / len, dy / len)
      } else {
        (
          std::f64::consts::FRAC_1_SQRT_2,
          -std::f64::consts::FRAC_1_SQRT_2,
        )
      };
      let dist = if labeled[i] { FONT * 0.95 } else { FONT * 0.6 };
      body.push_str(&format!(
        "<text x=\"{ix:.2}\" y=\"{iy:.2}\" font-family=\"Helvetica, Arial, \
         sans-serif\" font-size=\"{fs:.1}\" fill=\"{INDEX_COLOR}\" \
         text-anchor=\"middle\" dominant-baseline=\"central\">{n}</text>",
        ix = x + ux * dist,
        iy = y + uy * dist,
        fs = FONT * 0.6,
        n = mol.index[i],
      ));
    }
  }

  // --- Frame ---
  let (mut min_x, mut min_y, mut max_x, mut max_y) =
    (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
  for &(x, y) in &px {
    min_x = min_x.min(x);
    min_y = min_y.min(y);
    max_x = max_x.max(x);
    max_y = max_y.max(y);
  }
  let (w, h) = (
    (max_x - min_x) + 2.0 * MARGIN,
    (max_y - min_y) + 2.0 * MARGIN,
  );
  let (ox, oy) = (min_x - MARGIN, min_y - MARGIN);
  (body, ox, oy, w, h)
}

fn draw_line(out: &mut String, a: Pt, b: Pt, stroke: &str, width: f64) {
  // Bonds are drawn as two-point polylines (not <line>), matching the vector
  // markup wolframscript emits for a molecule structure diagram.
  out.push_str(&format!(
    "<polyline points=\"{:.2},{:.2} {:.2},{:.2}\" fill=\"none\" \
     stroke=\"{stroke}\" stroke-width=\"{width}\"/>",
    a.0, a.1, b.0, b.1
  ));
}

/// Move both endpoints of a segment inward along its own direction by `da`
/// (from `a`) and `db` (from `b`).
fn trim_segment(a: Pt, b: Pt, da: f64, db: f64) -> (Pt, Pt) {
  let (dx, dy) = (b.0 - a.0, b.1 - a.1);
  let len = (dx * dx + dy * dy).sqrt().max(1e-6);
  let (ux, uy) = (dx / len, dy / len);
  (
    (a.0 + ux * da, a.1 + uy * da),
    (b.0 - ux * db, b.1 - uy * db),
  )
}

/// Offset a segment sideways by `dist * side` along its left-hand normal.
fn offset_segment(a: Pt, b: Pt, side: f64, dist: f64) -> (Pt, Pt) {
  let (dx, dy) = (b.0 - a.0, b.1 - a.1);
  let len = (dx * dx + dy * dy).sqrt().max(1e-6);
  let (nx, ny) = (-dy / len * dist * side, dx / len * dist * side);
  ((a.0 + nx, a.1 + ny), (b.0 + nx, b.1 + ny))
}

/// Pull both endpoints of a segment toward its midpoint by fraction `frac`.
fn shrink(a: Pt, b: Pt, frac: f64) -> (Pt, Pt) {
  let mid = (f64::midpoint(a.0, b.0), f64::midpoint(a.1, b.1));
  (
    (a.0 + (mid.0 - a.0) * frac, a.1 + (mid.1 - a.1) * frac),
    (b.0 + (mid.0 - b.0) * frac, b.1 + (mid.1 - b.1) * frac),
  )
}

/// Direction (as a point offset) from atom `a` toward its neighbors other than
/// `exclude`, averaged. `None` if `a` has no such neighbor (it is terminal).
fn other_neighbor_dir(
  mol: &DrawMolecule,
  px: &[Pt],
  a: usize,
  exclude: usize,
) -> Option<Pt> {
  let mut acc = (0.0, 0.0);
  let mut count = 0;
  for &(i, j, _) in &mol.bonds {
    let other = if i == a {
      j
    } else if j == a {
      i
    } else {
      continue;
    };
    if other == exclude {
      continue;
    }
    acc.0 += px[other].0 - px[a].0;
    acc.1 += px[other].1 - px[a].1;
    count += 1;
  }
  if count == 0 { None } else { Some(acc) }
}

/// Choose which side of bond `a`-`b` the inner (double/aromatic) line goes on:
/// toward the substituents of the two atoms (the ring interior for ring bonds).
fn double_bond_side(mol: &DrawMolecule, px: &[Pt], a: usize, b: usize) -> f64 {
  let (ax, ay) = px[a];
  let (bx, by) = px[b];
  // Left-hand normal of a->b.
  let (dx, dy) = (bx - ax, by - ay);
  let len = (dx * dx + dy * dy).sqrt().max(1e-6);
  let (nx, ny) = (-dy / len, dx / len);
  let mut vote = 0.0;
  for (atom, exclude) in [(a, b), (b, a)] {
    if let Some((ox, oy)) = other_neighbor_dir(mol, px, atom, exclude) {
      vote += ox * nx + oy * ny;
    }
  }
  if vote >= 0.0 { 1.0 } else { -1.0 }
}

/// Superscript text for a formal charge (`+`, `-`, `2+`, `3-`, …).
fn charge_label(charge: i64) -> String {
  match charge {
    1 => "+".to_string(),
    -1 => "\u{2013}".to_string(), // en dash reads better than hyphen
    c if c > 0 => format!("{c}+"),
    c => format!("{}\u{2013}", -c),
  }
}

// ---------------------------------------------------------------------------
// 3-D ball-and-stick model (`MoleculePlot3D`)
// ---------------------------------------------------------------------------

type P3 = [f64; 3];

/// Single-bond covalent radius of an element, in ångström.
fn covalent_radius(symbol: &str) -> f64 {
  match symbol {
    "H" => 0.31,
    "B" => 0.84,
    "C" => 0.76,
    "N" => 0.71,
    "O" => 0.66,
    "F" => 0.57,
    "Si" => 1.11,
    "P" => 1.07,
    "S" => 1.05,
    "Cl" => 1.02,
    "Br" => 1.20,
    "I" => 1.39,
    _ => 1.2,
  }
}

/// Ball color of an element in the 3-D model (CPK-style).
fn atom_rgb(symbol: &str) -> [f64; 3] {
  match symbol {
    "H" => [0.65, 0.7, 0.7],
    "C" => [0.4, 0.4, 0.4],
    "N" => [0.19, 0.31, 0.97],
    "O" => [0.8, 0.05, 0.11],
    "F" => [0.56, 0.88, 0.31],
    "Cl" => [0.12, 0.94, 0.12],
    "Br" => [0.65, 0.16, 0.16],
    "I" => [0.58, 0.0, 0.58],
    "S" => [0.9, 0.78, 0.2],
    "P" => [1.0, 0.5, 0.0],
    "B" => [1.0, 0.71, 0.71],
    _ => [0.75, 0.5, 0.75],
  }
}

/// Bond-order factor that shortens multiple bonds relative to single ones.
fn bond_length_factor(kind: &str) -> f64 {
  match kind {
    "Double" => 0.87,
    "Triple" => 0.78,
    "Aromatic" => 0.92,
    _ => 1.0,
  }
}

fn sub3(a: P3, b: P3) -> P3 {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm3(a: P3) -> f64 {
  (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn cross3(a: P3, b: P3) -> P3 {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

fn scale3(a: P3, s: f64) -> P3 {
  [a[0] * s, a[1] * s, a[2] * s]
}

/// Deterministic 3-D atom coordinates (in ångström) for an all-atom
/// molecule. Starts from the 2-D layout lifted slightly out of plane, then
/// relaxes a simple force field: bond stretches toward covalent-radius bond
/// lengths, 1-3 distances toward the bond angle of each center's
/// hybridization (180°, 120° or 109.5°), and a soft repulsion between atoms
/// that are farther apart in the bond graph.
fn layout_3d(mol: &DrawMolecule) -> Vec<P3> {
  let n = mol.atoms.len();
  let mut adj: Vec<Vec<(usize, &str)>> = vec![Vec::new(); n];
  for &(a, b, kind) in &mol.bonds {
    adj[a].push((b, kind));
    adj[b].push((a, kind));
  }
  let bond_len = |a: usize, b: usize, kind: &str| {
    (covalent_radius(&mol.atoms[a].0) + covalent_radius(&mol.atoms[b].0))
      * bond_length_factor(kind)
  };
  // Distance restraints: (i, j, target, stiffness).
  let mut restraints: Vec<(usize, usize, f64, f64)> = Vec::new();
  let mut related = vec![vec![false; n]; n];
  for &(a, b, kind) in &mol.bonds {
    restraints.push((a, b, bond_len(a, b, kind), 1.0));
    related[a][b] = true;
    related[b][a] = true;
  }
  for center in 0..n {
    let neigh = &adj[center];
    let doubles = neigh.iter().filter(|(_, k)| *k == "Double").count();
    let triple = neigh.iter().any(|(_, k)| *k == "Triple");
    let pi = neigh
      .iter()
      .any(|(_, k)| *k == "Double" || *k == "Aromatic");
    let angle = if triple || doubles >= 2 {
      std::f64::consts::PI
    } else if pi {
      120f64.to_radians()
    } else {
      109.47f64.to_radians()
    };
    for x in 0..neigh.len() {
      for y in (x + 1)..neigh.len() {
        let (i, ki) = neigh[x];
        let (j, kj) = neigh[y];
        let (li, lj) = (bond_len(center, i, ki), bond_len(center, j, kj));
        let target = (li * li + lj * lj - 2.0 * li * lj * angle.cos()).sqrt();
        restraints.push((i, j, target, 1.0));
        related[i][j] = true;
        related[j][i] = true;
      }
    }
  }

  // Seed: the 2-D layout scaled to bond lengths, with a small deterministic
  // out-of-plane offset so the relaxation can leave the plane.
  let flat = layout(mol);
  let mut pos: Vec<P3> = flat
    .iter()
    .enumerate()
    .map(|(i, &(x, y))| [x * 1.4, y * 1.4, 0.25 * ((i as f64) * 1.7).sin()])
    .collect();

  let iterations = 3000;
  for step in 0..iterations {
    let mut grad = vec![[0.0f64; 3]; n];
    for &(i, j, target, k) in &restraints {
      let d = sub3(pos[i], pos[j]);
      let len = norm3(d).max(1e-6);
      let f = k * (len - target) / len;
      for c in 0..3 {
        grad[i][c] += f * d[c];
        grad[j][c] -= f * d[c];
      }
    }
    for i in 0..n {
      for j in (i + 1)..n {
        if related[i][j] {
          continue;
        }
        let d = sub3(pos[i], pos[j]);
        let len = norm3(d).max(1e-6);
        let reach = 2.4;
        if len < reach {
          let f = -0.15 * (reach - len) / len;
          for c in 0..3 {
            grad[i][c] += f * d[c];
            grad[j][c] -= f * d[c];
          }
        }
      }
    }
    let rate = if step < iterations / 2 { 0.1 } else { 0.05 };
    for i in 0..n {
      for c in 0..3 {
        pos[i][c] -= rate * grad[i][c].clamp(-2.0, 2.0);
      }
    }
  }

  // Center the model on its centroid.
  let mut centroid = [0.0; 3];
  for p in &pos {
    for c in 0..3 {
      centroid[c] += p[c] / n as f64;
    }
  }
  pos.iter().map(|&p| sub3(p, centroid)).collect()
}

fn real(x: f64) -> Expr {
  // Round away float noise so the coordinates print compactly.
  Expr::Real((x * 1e4).round() / 1e4)
}

fn point(p: P3) -> Expr {
  Expr::List(vec![real(p[0]), real(p[1]), real(p[2])].into())
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn rgb(c: [f64; 3]) -> Expr {
  call("RGBColor", vec![real(c[0]), real(c[1]), real(c[2])])
}

/// The `Graphics3D[…]` ball-and-stick model: a sphere per atom and, per bond,
/// one cylinder per bond order, each half colored like the atom it touches.
fn ball_and_stick(
  mol: &DrawMolecule,
  pos: &[P3],
  opts: &MoleculePlotOptions,
) -> Expr {
  let mut prims = Vec::new();
  for (i, (sym, ..)) in mol.atoms.iter().enumerate() {
    let radius = 0.1 + 0.4 * covalent_radius(sym);
    prims.push(Expr::List(
      vec![
        rgb(atom_rgb(sym)),
        call("Sphere", vec![point(pos[i]), real(radius)]),
      ]
      .into(),
    ));
  }
  for &(a, b, kind) in &mol.bonds {
    let axis = sub3(pos[b], pos[a]);
    // A sideways direction for multiple bonds, in the plane of a neighbor.
    let side = {
      let neighbor = mol
        .bonds
        .iter()
        .find_map(|&(i, j, _)| {
          if i == a && j != b {
            Some(j)
          } else if j == a && i != b {
            Some(i)
          } else if i == b && j != a {
            Some(j)
          } else if j == b && i != a {
            Some(i)
          } else {
            None
          }
        })
        .map(|k| sub3(pos[k], pos[a]));
      let normal = neighbor
        .map(|v| cross3(axis, v))
        .filter(|v| norm3(*v) > 1e-6)
        .unwrap_or_else(|| {
          let helper = if axis[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
          } else {
            [0.0, 1.0, 0.0]
          };
          cross3(axis, helper)
        });
      let s = cross3(normal, axis);
      scale3(s, 1.0 / norm3(s).max(1e-9))
    };
    let (offsets, radius): (Vec<f64>, f64) = match kind {
      "Double" => (vec![-0.11, 0.11], 0.07),
      "Triple" => (vec![-0.16, 0.0, 0.16], 0.06),
      _ => (vec![0.0], 0.11),
    };
    let mid = [
      f64::midpoint(pos[a][0], pos[b][0]),
      f64::midpoint(pos[a][1], pos[b][1]),
      f64::midpoint(pos[a][2], pos[b][2]),
    ];
    for off in offsets {
      let shift = scale3(side, off);
      let at = |p: P3| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
      for (end, atom) in [(pos[a], a), (pos[b], b)] {
        prims.push(Expr::List(
          vec![
            rgb(atom_rgb(&mol.atoms[atom].0)),
            call(
              "Cylinder",
              vec![
                Expr::List(vec![point(at(end)), point(at(mid))].into()),
                real(radius),
              ],
            ),
          ]
          .into(),
        ));
      }
    }
  }
  let rule = |name: &str, value: Expr| Expr::Rule {
    pattern: Box::new(Expr::Identifier(name.to_string())),
    replacement: Box::new(value),
  };
  let mut args = vec![
    Expr::List(prims.into()),
    rule("Boxed", Expr::Identifier("False".to_string())),
    rule("Lighting", Expr::String("Neutral".to_string())),
  ];
  if let Some(size) = &opts.image_size {
    args.push(rule("ImageSize", size.clone()));
  }
  call("Graphics3D", args)
}
