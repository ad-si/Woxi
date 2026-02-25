use crate::InterpreterError;
use crate::functions::expr_form::{ExprForm, decompose_expr};
use crate::functions::graphics::graphics_ast;
use crate::syntax::{Expr, expr_to_string};

/// A node in the expression tree
struct TreeNode {
  label: String,
  children: Vec<Self>,
}

/// Recursively build a tree from an expression, respecting an optional depth limit.
fn build_tree(
  expr: &Expr,
  max_depth: Option<usize>,
  current_depth: usize,
) -> TreeNode {
  // If we've reached the max depth, show the expression as a leaf
  if let Some(max) = max_depth
    && current_depth >= max
  {
    return TreeNode {
      label: expr_to_string(expr),
      children: vec![],
    };
  }

  match decompose_expr(expr) {
    ExprForm::Atom(label) => TreeNode {
      label,
      children: vec![],
    },
    ExprForm::Composite { head, children } => TreeNode {
      label: head,
      children: children
        .iter()
        .map(|c| build_tree(c, max_depth, current_depth + 1))
        .collect(),
    },
  }
}

/// Positioned node after layout
struct LayoutNode {
  x: f64,
  y: f64,
  label: String,
  children_indices: Vec<usize>,
}

/// Maximum full box width across all nodes in the tree
fn max_node_box_width(
  node: &TreeNode,
  char_width: f64,
  box_padding: f64,
) -> f64 {
  let self_width = node.label.len() as f64 * char_width + 2.0 * box_padding;
  let child_max = node
    .children
    .iter()
    .map(|c| max_node_box_width(c, char_width, box_padding))
    .fold(0.0_f64, f64::max);
  self_width.max(child_max)
}

/// Perform a simple tree layout using Reingold-Tilford-style centering.
/// Returns a flat list of positioned nodes.
fn layout_tree(root: &TreeNode, leaf_step: f64) -> Vec<LayoutNode> {
  let mut nodes = Vec::new();
  let y_spacing = 1.0;
  layout_recursive(root, 0, y_spacing, leaf_step, &mut nodes, &mut 0.0);
  nodes
}

/// Recursive layout: returns the index of the placed node.
/// `next_leaf_x` is advanced for each leaf to ensure no overlap.
fn layout_recursive(
  node: &TreeNode,
  depth: usize,
  y_spacing: f64,
  leaf_step: f64,
  nodes: &mut Vec<LayoutNode>,
  next_leaf_x: &mut f64,
) -> usize {
  let idx = nodes.len();
  // Reserve a spot
  nodes.push(LayoutNode {
    x: 0.0,
    y: -(depth as f64) * y_spacing,
    label: node.label.clone(),
    children_indices: vec![],
  });

  if node.children.is_empty() {
    // Leaf node: place at next available x position
    nodes[idx].x = *next_leaf_x;
    *next_leaf_x += leaf_step;
  } else {
    // Internal node: layout children, then center over them
    let mut child_indices = Vec::new();
    for child in &node.children {
      let child_idx = layout_recursive(
        child,
        depth + 1,
        y_spacing,
        leaf_step,
        nodes,
        next_leaf_x,
      );
      child_indices.push(child_idx);
    }

    // Center parent over children
    let first_child_x = nodes[child_indices[0]].x;
    let last_child_x = nodes[child_indices[child_indices.len() - 1]].x;
    nodes[idx].x = (first_child_x + last_child_x) / 2.0;
    nodes[idx].children_indices = child_indices;
  }

  idx
}

/// Implementation of TreeForm[expr] and TreeForm[expr, maxDepth].
/// Generates a tree diagram rendered as Graphics primitives.
pub fn tree_form_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "TreeForm".to_string(),
      args: vec![],
    });
  }

  let expr = &args[0];

  // Parse optional max depth
  let max_depth = if args.len() >= 2 {
    match &args[1] {
      Expr::Integer(n) if *n > 0 => Some(*n as usize),
      _ => None,
    }
  } else {
    None
  };

  // Build the tree from the expression
  let tree = build_tree(expr, max_depth, 0);

  // Box sizing constants (in coordinate units)
  let box_half_height = 0.18;
  let char_width = 0.09; // approximate width per character
  let box_padding = 0.08; // horizontal padding around text

  let box_half_width = |label: &str| -> f64 {
    let text_width = label.len() as f64 * char_width;
    text_width / 2.0 + box_padding
  };

  // Compute adaptive leaf spacing: ensure no box overlap at any depth level
  let max_bw = max_node_box_width(&tree, char_width, box_padding);
  let leaf_step = (max_bw + 0.1).max(1.0);

  // If it's just a single atom (no children), still show it as a node
  let layout = layout_tree(&tree, leaf_step);

  if layout.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "TreeForm".to_string(),
      args: args.to_vec(),
    });
  }

  // Compute coordinate-space bounding box (including box extents)
  let coord_x_min = layout
    .iter()
    .map(|n| n.x - box_half_width(&n.label))
    .fold(f64::INFINITY, f64::min);
  let coord_x_max = layout
    .iter()
    .map(|n| n.x + box_half_width(&n.label))
    .fold(f64::NEG_INFINITY, f64::max);
  let coord_w = (coord_x_max - coord_x_min).max(0.01);

  // graphics_ast adds 4% padding on each side, so effective range is ~1.08x
  let padded_w = coord_w * 1.08;

  // Compute the font-size that fits within the smallest box at a given image width.
  // For monospace text: char_pixel_width ≈ 0.6 * font_size
  // For a label of length L in a box of coordinate width (L*cw + 2*bp):
  //   text_px = L * 0.6 * fs
  //   box_px  = (L*cw + 2*bp) * pixels_per_unit
  //   Need: text_px <= box_px
  //   => fs <= (cw + 2*bp/L) * ppu / 0.6
  // The worst case is the longest label (largest L), approaching fs <= cw * ppu / 0.6.
  let char_px_ratio = 0.6_f64;
  let max_label_len = layout
    .iter()
    .map(|n| n.label.len())
    .max()
    .unwrap_or(1)
    .max(1) as f64;
  let effective_cw = char_width + 2.0 * box_padding / max_label_len;

  // Determine image width so text at font-size 10-14 fits in boxes
  let base_width = 360.0_f64;
  let ppu_at_base = base_width / padded_w;
  let max_fs_at_base = effective_cw * ppu_at_base / char_px_ratio;

  let min_fs = 10.0_f64;
  let max_fs = 14.0_f64;

  let (image_width, font_size) = if max_fs_at_base >= max_fs {
    // At 360px, font-size 14 fits comfortably
    (360_i64, max_fs)
  } else if max_fs_at_base >= min_fs {
    // At 360px, use the largest font-size that fits
    (360, max_fs_at_base)
  } else {
    // Need wider image; scale up so font-size reaches min_fs
    let scale = min_fs / max_fs_at_base;
    let needed_w = (base_width * scale).ceil() as i64;
    let clamped_w = needed_w.min(800);
    let actual_fs = max_fs_at_base * (clamped_w as f64 / base_width);
    (clamped_w, actual_fs.min(max_fs))
  };

  let font_size_int = (font_size.round() as i128).max(8);

  // Generate Graphics primitives: Lines for edges, then boxes and text
  let mut primitives: Vec<Expr> = Vec::new();

  // Draw edges first (so they appear behind boxes and text)
  // Edge color: gray
  primitives.push(Expr::FunctionCall {
    name: "RGBColor".to_string(),
    args: vec![Expr::Real(0.6), Expr::Real(0.6), Expr::Real(0.6)],
  });

  for node in &layout {
    for &child_idx in &node.children_indices {
      let child = &layout[child_idx];
      primitives.push(Expr::FunctionCall {
        name: "Line".to_string(),
        args: vec![Expr::List(vec![
          Expr::List(vec![Expr::Real(node.x), Expr::Real(node.y)]),
          Expr::List(vec![Expr::Real(child.x), Expr::Real(child.y)]),
        ])],
      });
    }
  }

  // First pass: rectangular boxes for every node (drawn on top of edges)
  for node in &layout {
    let is_leaf = node.children_indices.is_empty();
    let hw = box_half_width(&node.label);
    let hh = box_half_height;

    if is_leaf {
      // Leaf nodes: white fill, black border
      primitives.push(Expr::FunctionCall {
        name: "EdgeForm".to_string(),
        args: vec![Expr::FunctionCall {
          name: "RGBColor".to_string(),
          args: vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(0.0)],
        }],
      });
      primitives.push(Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: vec![Expr::Real(1.0), Expr::Real(1.0), Expr::Real(1.0)],
      });
    } else {
      // Internal nodes: light orange fill, orange border
      primitives.push(Expr::FunctionCall {
        name: "EdgeForm".to_string(),
        args: vec![Expr::FunctionCall {
          name: "RGBColor".to_string(),
          args: vec![Expr::Real(0.84), Expr::Real(0.48), Expr::Real(0.0)],
        }],
      });
      primitives.push(Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: vec![Expr::Real(1.0), Expr::Real(0.95), Expr::Real(0.85)],
      });
    }

    primitives.push(Expr::FunctionCall {
      name: "Rectangle".to_string(),
      args: vec![
        Expr::List(vec![Expr::Real(node.x - hw), Expr::Real(node.y - hh)]),
        Expr::List(vec![Expr::Real(node.x + hw), Expr::Real(node.y + hh)]),
      ],
    });
  }

  // Second pass: text labels for all nodes (on top of boxes)
  primitives.push(Expr::FunctionCall {
    name: "EdgeForm".to_string(),
    args: vec![],
  });

  for node in &layout {
    let is_leaf = node.children_indices.is_empty();

    if is_leaf {
      // Leaf nodes: black text
      primitives.push(Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(0.0)],
      });
    } else {
      // Internal nodes: dark orange text
      primitives.push(Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: vec![Expr::Real(0.84), Expr::Real(0.48), Expr::Real(0.0)],
      });
    }

    primitives.push(Expr::FunctionCall {
      name: "Text".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Style".to_string(),
          args: vec![
            Expr::String(node.label.clone()),
            Expr::Integer(font_size_int),
          ],
        },
        Expr::List(vec![Expr::Real(node.x), Expr::Real(node.y)]),
      ],
    });
  }

  // Build Graphics[{primitives...}, ImageSize -> width]
  let content = Expr::List(primitives);
  let image_size_opt = Expr::Rule {
    pattern: Box::new(Expr::Identifier("ImageSize".to_string())),
    replacement: Box::new(Expr::Integer(image_width as i128)),
  };

  graphics_ast(&[content, image_size_opt])
}
