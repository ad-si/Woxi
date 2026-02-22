use crate::syntax::{BinaryOperator, ComparisonOp, Expr, UnaryOperator};

/// Result of decomposing an expression into its canonical form.
pub enum ExprForm {
  /// An atomic expression (leaf node in the tree).
  Atom(String),
  /// A composite expression with a head and children.
  Composite { head: String, children: Vec<Expr> },
}

/// Format a floating-point number for FullForm display.
pub fn format_real_helper(f: f64) -> String {
  if f.fract() == 0.0 && f.abs() < 1e15 {
    format!("{:.1}", f)
  } else {
    format!("{}", f)
  }
}

/// Collect all operands for an associative binary operator (Plus, Times),
/// flattening nested applications of the same operator.
fn collect_children(expr: &Expr, target_op: &BinaryOperator) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp { op, left, right } if op == target_op => {
      let mut parts = collect_children(left, target_op);
      parts.extend(collect_children(right, target_op));
      parts
    }
    _ => vec![expr.clone()],
  }
}

/// Flatten nested Part expressions into (base_expr, [index1, index2, ...]).
fn flatten_part(expr: &Expr) -> (&Expr, Vec<Expr>) {
  let mut indices = Vec::new();
  let mut base = expr;
  while let Expr::Part {
    expr: inner_expr,
    index: inner_index,
  } = base
  {
    indices.push(inner_index.as_ref().clone());
    base = inner_expr.as_ref();
  }
  indices.reverse();
  (base, indices)
}

/// Decompose an expression into its canonical FullForm representation.
///
/// Returns either an `Atom` (leaf label) or a `Composite` (head name + children).
/// This handles all canonical transformations (Minus→Plus, Divide→Times, etc.)
/// and associative flattening (Plus, Times).
pub fn decompose_expr(expr: &Expr) -> ExprForm {
  match expr {
    // --- Atoms ---
    Expr::Integer(n) => ExprForm::Atom(n.to_string()),
    Expr::BigInteger(n) => ExprForm::Atom(n.to_string()),
    Expr::Real(f) => ExprForm::Atom(format_real_helper(*f)),
    Expr::BigFloat(digits, prec) => {
      ExprForm::Atom(format!("{}`{}.", digits, prec))
    }
    Expr::String(s) => ExprForm::Atom(format!("\"{}\"", s)),
    Expr::Identifier(s) => ExprForm::Atom(s.clone()),
    Expr::Constant(c) => ExprForm::Atom(c.clone()),
    Expr::Raw(s) => ExprForm::Atom(s.clone()),
    Expr::Image { .. } => ExprForm::Atom("-Image-".to_string()),

    // --- Slot/SlotSequence ---
    Expr::Slot(n) => ExprForm::Composite {
      head: "Slot".to_string(),
      children: vec![Expr::Integer(*n as i128)],
    },
    Expr::SlotSequence(n) => ExprForm::Composite {
      head: "SlotSequence".to_string(),
      children: vec![Expr::Integer(*n as i128)],
    },

    // --- Simple composites ---
    Expr::List(items) => ExprForm::Composite {
      head: "List".to_string(),
      children: items.clone(),
    },
    Expr::FunctionCall { name, args } => {
      // Sqrt[x] → Power[x, Rational[1, 2]] in FullForm
      if name == "Sqrt" && args.len() == 1 {
        return ExprForm::Composite {
          head: "Power".to_string(),
          children: vec![
            args[0].clone(),
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(1), Expr::Integer(2)],
            },
          ],
        };
      }
      ExprForm::Composite {
        head: name.clone(),
        children: args.clone(),
      }
    }
    Expr::Rule {
      pattern,
      replacement,
    } => ExprForm::Composite {
      head: "Rule".to_string(),
      children: vec![pattern.as_ref().clone(), replacement.as_ref().clone()],
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => ExprForm::Composite {
      head: "RuleDelayed".to_string(),
      children: vec![pattern.as_ref().clone(), replacement.as_ref().clone()],
    },
    Expr::ReplaceAll { expr, rules } => ExprForm::Composite {
      head: "ReplaceAll".to_string(),
      children: vec![expr.as_ref().clone(), rules.as_ref().clone()],
    },
    Expr::ReplaceRepeated { expr, rules } => ExprForm::Composite {
      head: "ReplaceRepeated".to_string(),
      children: vec![expr.as_ref().clone(), rules.as_ref().clone()],
    },
    Expr::Map { func, list } => ExprForm::Composite {
      head: "Map".to_string(),
      children: vec![func.as_ref().clone(), list.as_ref().clone()],
    },
    Expr::Apply { func, list } => ExprForm::Composite {
      head: "Apply".to_string(),
      children: vec![func.as_ref().clone(), list.as_ref().clone()],
    },
    Expr::MapApply { func, list } => ExprForm::Composite {
      head: "MapApply".to_string(),
      children: vec![func.as_ref().clone(), list.as_ref().clone()],
    },
    Expr::CompoundExpr(exprs) => ExprForm::Composite {
      head: "CompoundExpression".to_string(),
      children: exprs.clone(),
    },
    Expr::Function { body } => ExprForm::Composite {
      head: "Function".to_string(),
      children: vec![body.as_ref().clone()],
    },

    // --- Binary operators ---
    Expr::BinaryOp { op, left, right } => match op {
      // Associative: flatten nested Plus/Times
      BinaryOperator::Plus => ExprForm::Composite {
        head: "Plus".to_string(),
        children: collect_children(expr, &BinaryOperator::Plus),
      },
      BinaryOperator::Times => ExprForm::Composite {
        head: "Times".to_string(),
        children: collect_children(expr, &BinaryOperator::Times),
      },
      // Canonical transformations
      BinaryOperator::Minus => ExprForm::Composite {
        head: "Plus".to_string(),
        children: vec![
          left.as_ref().clone(),
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: right.clone(),
          },
        ],
      },
      BinaryOperator::Divide => {
        // Canonicalize: a/b → simplified form using power and times
        // This handles cases like 1/z → Power[z, -1],
        // x/Sqrt[5] → Times[Power[5, Rational[-1, 2]], x]
        if let Ok(b_inv) =
          crate::functions::power_two(right, &Expr::Integer(-1))
        {
          if matches!(left.as_ref(), Expr::Integer(1)) {
            return decompose_expr(&b_inv);
          }
          if let Ok(product) = crate::functions::times_ast(&[
            left.as_ref().clone(),
            b_inv,
          ]) {
            return decompose_expr(&product);
          }
        }
        // Fallback to structural decomposition
        ExprForm::Composite {
          head: "Times".to_string(),
          children: vec![
            left.as_ref().clone(),
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: right.clone(),
              right: Box::new(Expr::Integer(-1)),
            },
          ],
        }
      }
      // Simple binary → named head
      BinaryOperator::Power => ExprForm::Composite {
        head: "Power".to_string(),
        children: vec![left.as_ref().clone(), right.as_ref().clone()],
      },
      BinaryOperator::And => ExprForm::Composite {
        head: "And".to_string(),
        children: vec![left.as_ref().clone(), right.as_ref().clone()],
      },
      BinaryOperator::Or => ExprForm::Composite {
        head: "Or".to_string(),
        children: vec![left.as_ref().clone(), right.as_ref().clone()],
      },
      BinaryOperator::StringJoin => ExprForm::Composite {
        head: "StringJoin".to_string(),
        children: vec![left.as_ref().clone(), right.as_ref().clone()],
      },
      BinaryOperator::Alternatives => ExprForm::Composite {
        head: "Alternatives".to_string(),
        children: vec![left.as_ref().clone(), right.as_ref().clone()],
      },
    },

    // --- Unary operators ---
    Expr::UnaryOp { op, operand } => match op {
      UnaryOperator::Minus => ExprForm::Composite {
        head: "Times".to_string(),
        children: vec![Expr::Integer(-1), operand.as_ref().clone()],
      },
      UnaryOperator::Not => ExprForm::Composite {
        head: "Not".to_string(),
        children: vec![operand.as_ref().clone()],
      },
    },

    // --- Comparison ---
    Expr::Comparison {
      operands,
      operators,
    } => {
      if operators.len() == 1 {
        let head = match &operators[0] {
          ComparisonOp::Equal => "Equal",
          ComparisonOp::NotEqual => "Unequal",
          ComparisonOp::Less => "Less",
          ComparisonOp::LessEqual => "LessEqual",
          ComparisonOp::Greater => "Greater",
          ComparisonOp::GreaterEqual => "GreaterEqual",
          ComparisonOp::SameQ => "SameQ",
          ComparisonOp::UnsameQ => "UnsameQ",
        };
        ExprForm::Composite {
          head: head.to_string(),
          children: operands.clone(),
        }
      } else {
        ExprForm::Composite {
          head: "Inequality".to_string(),
          children: operands.clone(),
        }
      }
    }

    // --- Association ---
    Expr::Association(items) => ExprForm::Composite {
      head: "Association".to_string(),
      children: items
        .iter()
        .map(|(k, v)| Expr::Rule {
          pattern: Box::new(k.clone()),
          replacement: Box::new(v.clone()),
        })
        .collect(),
    },

    // --- Part (flatten nested) ---
    Expr::Part { .. } => {
      let (base, indices) = flatten_part(expr);
      let mut children = vec![base.clone()];
      children.extend(indices);
      ExprForm::Composite {
        head: "Part".to_string(),
        children,
      }
    }

    // --- PrefixApply, Postfix, CurriedCall: head is the rendered function ---
    Expr::PrefixApply { func, arg } => ExprForm::Composite {
      head: render_full_form(func),
      children: vec![arg.as_ref().clone()],
    },
    Expr::Postfix { expr, func } => ExprForm::Composite {
      head: render_full_form(func),
      children: vec![expr.as_ref().clone()],
    },
    Expr::CurriedCall { func, args } => ExprForm::Composite {
      head: render_full_form(func),
      children: args.clone(),
    },

    // --- NamedFunction ---
    Expr::NamedFunction { params, body } => {
      let params_expr = if params.len() == 1 {
        Expr::Identifier(params[0].clone())
      } else {
        Expr::List(params.iter().map(|p| Expr::Identifier(p.clone())).collect())
      };
      ExprForm::Composite {
        head: "Function".to_string(),
        children: vec![params_expr, body.as_ref().clone()],
      }
    }

    // --- Pattern matching ---
    Expr::Pattern { name, head } => {
      let blank = if let Some(h) = head {
        Expr::FunctionCall {
          name: "Blank".to_string(),
          args: vec![Expr::Identifier(h.clone())],
        }
      } else {
        Expr::FunctionCall {
          name: "Blank".to_string(),
          args: vec![],
        }
      };
      ExprForm::Composite {
        head: "Pattern".to_string(),
        children: vec![Expr::Identifier(name.clone()), blank],
      }
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => {
      let blank = if let Some(h) = head {
        Expr::FunctionCall {
          name: "Blank".to_string(),
          args: vec![Expr::Identifier(h.clone())],
        }
      } else {
        Expr::FunctionCall {
          name: "Blank".to_string(),
          args: vec![],
        }
      };
      let pattern = Expr::FunctionCall {
        name: "Pattern".to_string(),
        args: vec![Expr::Identifier(name.clone()), blank],
      };
      ExprForm::Composite {
        head: "Optional".to_string(),
        children: vec![pattern, default.as_ref().clone()],
      }
    }
    Expr::PatternTest { name, test } => {
      let blank_part = if name.is_empty() {
        Expr::FunctionCall {
          name: "Blank".to_string(),
          args: vec![],
        }
      } else {
        Expr::FunctionCall {
          name: "Pattern".to_string(),
          args: vec![
            Expr::Identifier(name.clone()),
            Expr::FunctionCall {
              name: "Blank".to_string(),
              args: vec![],
            },
          ],
        }
      };
      ExprForm::Composite {
        head: "PatternTest".to_string(),
        children: vec![blank_part, test.as_ref().clone()],
      }
    }
  }
}

/// Recursively render an expression in FullForm notation.
pub fn render_full_form(expr: &Expr) -> String {
  match decompose_expr(expr) {
    ExprForm::Atom(label) => label,
    ExprForm::Composite { head, children } => {
      let parts: Vec<String> = children.iter().map(render_full_form).collect();
      format!("{}[{}]", head, parts.join(", "))
    }
  }
}
