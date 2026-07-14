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

use crate::InterpreterError;
use crate::functions::molecule_ast::{
  DrawMolecule, drawable_molecule, molecule_ast,
};
use crate::syntax::Expr;

/// `MoleculePlot[mol]` — the 2-D structure diagram as a `Graphics` object.
/// The argument may be a `Molecule[…]` object or any specification a plain
/// `Molecule[…]` call accepts (a chemical name or SMILES string). Returns the
/// call unevaluated if it does not describe a molecule.
pub fn molecule_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "MoleculePlot".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 1 {
    return unevaluated();
  }
  // Resolve to a canonical Molecule[…] expression.
  let mol_expr = match &args[0] {
    Expr::FunctionCall { name, .. } if name == "Molecule" => args[0].clone(),
    other => match molecule_ast(std::slice::from_ref(other))? {
      e @ Expr::FunctionCall { .. } if matches!(&e, Expr::FunctionCall { name, .. } if name == "Molecule") => {
        e
      }
      _ => return unevaluated(),
    },
  };
  match molecule_to_svg(&mol_expr) {
    Some(svg) => Ok(crate::graphics_result(svg)),
    None => unevaluated(),
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

/// Render a molecule's 2-D structure diagram (the `MoleculePlot[…]` graphic)
/// to a self-contained SVG string, or `None` if `expr` is not a valid
/// molecule.
pub fn molecule_to_svg(expr: &Expr) -> Option<String> {
  let mol = drawable_molecule(expr)?;
  if mol.atoms.is_empty() {
    return None;
  }
  let coords = layout(&mol);
  let (body, ox, oy, w, h) = render_parts(&mol, &coords);
  Some(format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.0}\" \
     height=\"{h:.0}\" viewBox=\"{ox:.2} {oy:.2} {w:.2} {h:.2}\" \
     stroke-linecap=\"round\">{body}</svg>"
  ))
}

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
  // which shows up as having a hydrogen among its drawn neighbors.
  mol.bonds.iter().any(|&(a, b, _)| {
    (a == i && mol.atoms[b].0 == "H") || (b == i && mol.atoms[a].0 == "H")
  })
}

/// Render the structure diagram's inner markup plus its viewBox bounds
/// (`body`, `ox`, `oy`, `w`, `h`).
fn render_parts(
  mol: &DrawMolecule,
  coords: &[Pt],
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
    let color = atom_color(sym).unwrap_or(stroke);
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
  let mid = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
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
    c if c > 0 => format!("{}+", c),
    c => format!("{}\u{2013}", -c),
  }
}
