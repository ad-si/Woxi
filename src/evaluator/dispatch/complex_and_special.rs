#[allow(unused_imports)]
use super::*;

pub fn dispatch_complex_and_special(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Unitize" if args.len() == 1 => {
      return Some(crate::functions::math_ast::unitize_ast(args));
    }
    "Ramp" if args.len() == 1 => {
      return Some(crate::functions::math_ast::ramp_ast(args));
    }
    "KroneckerDelta" => {
      return Some(crate::functions::math_ast::kronecker_delta_ast(args));
    }
    "UnitStep" if !args.is_empty() => {
      return Some(crate::functions::math_ast::unit_step_ast(args));
    }
    "Complex" if args.len() == 2 => {
      // Complex[a, b] -> a + b*I, evaluated to simplify iterated Complex
      let real = &args[0];
      let imag = &args[1];
      // If imaginary part is 0 (integer), return just the real part
      if matches!(imag, Expr::Integer(0)) {
        return Some(Ok(real.clone()));
      }
      // If real part is 0 and imaginary is 1, return I
      if matches!(real, Expr::Integer(0)) && matches!(imag, Expr::Integer(1)) {
        return Some(Ok(Expr::Identifier("I".to_string())));
      }
      // Check if both parts are purely real numbers (no I involved) for non-evaluated path
      let imag_has_i = contains_i(imag);
      if !imag_has_i {
        // If real part is 0, return b*I
        if matches!(real, Expr::Integer(0)) {
          return Some(Ok(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(imag.clone()),
            right: Box::new(Expr::Identifier("I".to_string())),
          }));
        }
        // If imaginary is 1, return a + I
        if matches!(imag, Expr::Integer(1)) {
          return Some(Ok(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(real.clone()),
            right: Box::new(Expr::Identifier("I".to_string())),
          }));
        }
        // General case without I in imag: a + b*I (or a - |b|*I if b < 0)
        // Check if imag is negative to format as subtraction
        let (is_neg, abs_imag) = match imag {
          Expr::Real(f) if *f < 0.0 => (true, Expr::Real(-f)),
          Expr::Integer(n) if *n < 0 => (true, Expr::Integer(-n)),
          _ => (false, imag.clone()),
        };
        if is_neg {
          return Some(Ok(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Minus,
            left: Box::new(real.clone()),
            right: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(abs_imag),
              right: Box::new(Expr::Identifier("I".to_string())),
            }),
          }));
        }
        return Some(Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(real.clone()),
          right: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(imag.clone()),
            right: Box::new(Expr::Identifier("I".to_string())),
          }),
        }));
      }
      // Imaginary part contains I (iterated Complex), evaluate algebraically
      // Complex[a, b] where b has form (re_b + im_b*I):
      // a + (re_b + im_b*I)*I = a + re_b*I + im_b*I^2 = (a - im_b) + re_b*I
      // Try to extract complex components from imag
      if let Some((re_b, im_b)) =
        crate::functions::math_ast::try_extract_complex_float(imag)
      {
        // Both a and components are numeric
        if let Some(a) = crate::functions::math_ast::try_eval_to_f64(real) {
          let new_re = a - im_b;
          let new_im = re_b;
          // Reconstruct as Complex[new_re, new_im]
          let re_expr = if new_re == (new_re as i128 as f64) {
            Expr::Integer(new_re as i128)
          } else {
            Expr::Real(new_re)
          };
          let im_expr = if new_im == (new_im as i128 as f64) {
            Expr::Integer(new_im as i128)
          } else {
            Expr::Real(new_im)
          };
          return Some(evaluate_function_call_ast(
            "Complex",
            &[re_expr, im_expr],
          ));
        }
      }
      // Fallback: build a + b*I expression and evaluate
      let bi = match evaluate_function_call_ast(
        "Times",
        &[imag.clone(), Expr::Identifier("I".to_string())],
      ) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      return Some(evaluate_function_call_ast("Plus", &[real.clone(), bi]));
    }
    "ConditionalExpression" if args.len() == 2 => match &args[1] {
      Expr::Identifier(name) if name == "True" => {
        return Some(Ok(args[0].clone()));
      }
      Expr::Identifier(name) if name == "False" => {
        return Some(Ok(Expr::Identifier("Undefined".to_string())));
      }
      _ => {
        return Some(Ok(Expr::FunctionCall {
          name: "ConditionalExpression".to_string(),
          args: args.to_vec(),
        }));
      }
    },
    "DirectedInfinity" if args.len() <= 1 => {
      if args.is_empty() {
        // DirectedInfinity[] = ComplexInfinity
        return Some(Ok(Expr::Identifier("ComplexInfinity".to_string())));
      }
      match &args[0] {
        Expr::Integer(1) => {
          return Some(Ok(Expr::Identifier("Infinity".to_string())));
        }
        Expr::Integer(-1) => {
          return Some(Ok(Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand: Box::new(Expr::Identifier("Infinity".to_string())),
          }));
        }
        Expr::Integer(0) => {
          return Some(Ok(Expr::Identifier("ComplexInfinity".to_string())));
        }
        _ => {
          // Try to normalize: DirectedInfinity[z] -> DirectedInfinity[z/Abs[z]]
          if let Some(((re_n, re_d), (im_n, im_d))) =
            crate::functions::math_ast::try_extract_complex_exact(&args[0])
          {
            if im_n == 0 {
              // Pure real: just check sign
              if re_n > 0 {
                return Some(Ok(Expr::Identifier("Infinity".to_string())));
              } else if re_n < 0 {
                return Some(Ok(Expr::UnaryOp {
                  op: crate::syntax::UnaryOperator::Minus,
                  operand: Box::new(Expr::Identifier("Infinity".to_string())),
                }));
              } else {
                return Some(Ok(Expr::Identifier(
                  "ComplexInfinity".to_string(),
                )));
              }
            }
            // Compute magnitude squared: (re_n/re_d)^2 + (im_n/im_d)^2
            let mag_sq_num = re_n
              .checked_mul(re_n)
              .and_then(|a| {
                im_d.checked_mul(im_d).and_then(|b| a.checked_mul(b))
              })
              .and_then(|a| {
                im_n
                  .checked_mul(im_n)
                  .and_then(|c| {
                    re_d.checked_mul(re_d).and_then(|d| c.checked_mul(d))
                  })
                  .and_then(|b| a.checked_add(b))
              });
            let mag_sq_den = re_d.checked_mul(re_d).and_then(|a| {
              im_d.checked_mul(im_d).and_then(|b| a.checked_mul(b))
            });

            if let (Some(msn), Some(msd)) = (mag_sq_num, mag_sq_den) {
              // Build z/Abs[z] = z / Sqrt[msn/msd]
              let sqrt_arg = if msd == 1 {
                Expr::Integer(msn)
              } else {
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(msn), Expr::Integer(msd)],
                }
              };
              let normalized = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Divide,
                left: Box::new(args[0].clone()),
                right: Box::new(Expr::FunctionCall {
                  name: "Sqrt".to_string(),
                  args: vec![sqrt_arg],
                }),
              };
              let normalized = match evaluate_expr_to_expr(&normalized) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
              };
              // Check if normalized reduced to 1 or -1
              if matches!(&normalized, Expr::Integer(1)) {
                return Some(Ok(Expr::Identifier("Infinity".to_string())));
              }
              if matches!(&normalized, Expr::Integer(-1)) {
                return Some(Ok(Expr::UnaryOp {
                  op: crate::syntax::UnaryOperator::Minus,
                  operand: Box::new(Expr::Identifier("Infinity".to_string())),
                }));
              }
              return Some(Ok(Expr::FunctionCall {
                name: "DirectedInfinity".to_string(),
                args: vec![normalized],
              }));
            }
          }
          return Some(Ok(Expr::FunctionCall {
            name: "DirectedInfinity".to_string(),
            args: args.to_vec(),
          }));
        }
      }
    }

    // Echo[expr] - prints ">> expr" and returns expr
    // Echo[expr, label] - prints ">> label expr" and returns expr
    // Echo[expr, label, f] - prints ">> label f[expr]" and returns expr
    "Echo" if !args.is_empty() && args.len() <= 3 => {
      let label = if args.len() >= 2 {
        crate::syntax::expr_to_output(&args[1])
      } else {
        ">> ".to_string()
      };
      let display_expr = if args.len() == 3 {
        let f_applied = match &args[2] {
          Expr::Identifier(f_name) => Expr::FunctionCall {
            name: f_name.clone(),
            args: vec![args[0].clone()],
          },
          other => Expr::FunctionCall {
            name: "Apply".to_string(),
            args: vec![other.clone(), args[0].clone()],
          },
        };
        let result = match evaluate_expr_to_expr(&f_applied) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        crate::syntax::expr_to_output(&result)
      } else {
        crate::syntax::expr_to_output(&args[0])
      };
      let line = if args.len() >= 2 {
        format!(">> {} {}", label, display_expr)
      } else {
        format!(">> {}", display_expr)
      };
      println!("{}", line);
      crate::capture_stdout(&line);
      return Some(Ok(args[0].clone()));
    }

    // Sow[expr] or Sow[expr, tag] - adds expr to the current Reap collection
    "Sow" if args.len() == 1 || args.len() == 2 => {
      let tag = if args.len() == 2 {
        args[1].clone()
      } else {
        Expr::Identifier("None".to_string())
      };
      crate::SOW_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if let Some(last) = stack.last_mut() {
          last.push((args[0].clone(), tag));
        }
      });
      return Some(Ok(args[0].clone()));
    }

    // Reap[expr] or Reap[expr, pattern] - evaluates expr, collecting all Sow'd values
    "Reap" if args.len() == 1 || args.len() == 2 => {
      // Push a new collection
      crate::SOW_STACK.with(|stack| {
        stack.borrow_mut().push(Vec::new());
      });
      // Evaluate the expression
      let result = match evaluate_expr_to_expr(&args[0]) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      // Pop the collection
      let sowed = crate::SOW_STACK
        .with(|stack| stack.borrow_mut().pop().unwrap_or_default());

      if args.len() == 1 {
        // Reap[expr] - group by unique tags, preserving order of first appearance
        if sowed.is_empty() {
          return Some(Ok(Expr::List(vec![result, Expr::List(vec![])])));
        }
        let mut tag_order: Vec<Expr> = Vec::new();
        let mut tag_groups: Vec<Vec<Expr>> = Vec::new();
        for (val, tag) in &sowed {
          if let Some(idx) = tag_order
            .iter()
            .position(|t| expr_to_string(t) == expr_to_string(tag))
          {
            tag_groups[idx].push(val.clone());
          } else {
            tag_order.push(tag.clone());
            tag_groups.push(vec![val.clone()]);
          }
        }
        let groups: Vec<Expr> =
          tag_groups.into_iter().map(Expr::List).collect();
        return Some(Ok(Expr::List(vec![result, Expr::List(groups)])));
      } else {
        // Reap[expr, patt] or Reap[expr, {patt1, patt2, ...}]
        let patt_arg = match evaluate_expr_to_expr(&args[1]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        let patterns = match &patt_arg {
          Expr::List(pats) => pats.clone(),
          _ => vec![patt_arg.clone()],
        };
        let is_list_form = matches!(&patt_arg, Expr::List(_));

        let mut result_groups: Vec<Expr> = Vec::new();
        for patt in &patterns {
          // Collect all sowed values whose tag matches the pattern
          let mut matched: Vec<Expr> = Vec::new();
          let is_blank = matches!(patt, Expr::Pattern { .. });
          for (val, tag) in &sowed {
            if is_blank || expr_to_string(tag) == expr_to_string(patt) {
              matched.push(val.clone());
            }
          }
          if is_list_form {
            // {patt1, patt2, ...} form: each pattern gets a list wrapping
            if matched.is_empty() {
              result_groups.push(Expr::List(vec![]));
            } else {
              result_groups.push(Expr::List(vec![Expr::List(matched)]));
            }
          } else {
            // single pattern form: just the matched list
            if !matched.is_empty() {
              result_groups.push(Expr::List(matched));
            }
          }
        }
        return Some(Ok(Expr::List(vec![result, Expr::List(result_groups)])));
      }
    }

    // ReplaceAll and ReplaceRepeated function call forms
    "ReplaceAll" if args.len() == 2 => {
      return Some(apply_replace_all_ast(&args[0], &args[1]));
    }
    "ReplaceRepeated" if args.len() == 2 => {
      return Some(apply_replace_repeated_ast(&args[0], &args[1]));
    }
    "Replace" if args.len() == 2 => {
      return Some(apply_replace_ast(&args[0], &args[1]));
    }

    // Form wrappers -- transparent, just return the inner expression
    "MathMLForm" | "StandardForm" | "InputForm" | "OutputForm"
      if !args.is_empty() =>
    {
      return Some(Ok(args[0].clone()));
    }

    // Symbolic operators with no built-in meaning -- just return as-is with evaluated args
    "Therefore" | "Because" | "TableForm" | "Row" | "In" | "Grid" => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }

    _ => {}
  }
  None
}
