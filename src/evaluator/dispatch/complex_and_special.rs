#[allow(unused_imports)]
use super::*;
use crate::functions::math_ast::{is_sqrt, make_sqrt};

pub fn dispatch_complex_and_special(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Unitize" if args.len() == 1 || args.len() == 2 => {
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
    "HeavisideTheta" if !args.is_empty() => {
      return Some(crate::functions::math_ast::heaviside_theta_ast(args));
    }
    "DiracDelta" if !args.is_empty() => {
      return Some(crate::functions::math_ast::dirac_delta_ast(args));
    }
    "DiscreteDelta" => {
      return Some(crate::functions::math_ast::discrete_delta_ast(args));
    }
    "UnitBox" if args.len() == 1 => {
      return Some(crate::functions::math_ast::unit_box_ast(args));
    }
    "HeavisidePi" if args.len() == 1 => {
      return Some(crate::functions::math_ast::heaviside_pi_ast(args));
    }
    "UnitTriangle" if args.len() == 1 => {
      return Some(crate::functions::math_ast::unit_triangle_ast(args));
    }
    "HeavisideLambda" if args.len() == 1 => {
      return Some(crate::functions::math_ast::heaviside_lambda_ast(args));
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
      // If either component is inexact (Real/BigFloat) and the other is an
      // exact Integer/Rational, coerce the exact one to Real so the formed
      // a + b*I is fully inexact (matches Wolfram).
      fn is_inexact(e: &Expr) -> bool {
        matches!(e, Expr::Real(_) | Expr::BigFloat(_, _))
      }
      let need_coerce = is_inexact(real) ^ is_inexact(imag);
      let (real_owned, imag_owned);
      let (real, imag) = if need_coerce {
        let to_real = |e: &Expr| -> Expr {
          match e {
            Expr::Integer(n) => Expr::Real(*n as f64),
            Expr::FunctionCall { name, args }
              if name == "Rational" && args.len() == 2 =>
            {
              if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1])
              {
                Expr::Real(*n as f64 / *d as f64)
              } else {
                e.clone()
              }
            }
            _ => e.clone(),
          }
        };
        real_owned = if is_inexact(real) {
          real.clone()
        } else {
          to_real(real)
        };
        imag_owned = if is_inexact(imag) {
          imag.clone()
        } else {
          to_real(imag)
        };
        (&real_owned, &imag_owned)
      } else {
        (real, imag)
      };
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
          // Real numeric arguments collapse to ±Infinity by sign.
          // Sqrt[3], Pi, 2*Sqrt[3], etc. evaluate to a real f64, so we can
          // decide directly without going through the rational complex path.
          if let Some(v) = crate::functions::math_ast::try_eval_to_f64(&args[0])
            && v.is_finite()
          {
            if v > 0.0 {
              return Some(Ok(Expr::Identifier("Infinity".to_string())));
            }
            if v < 0.0 {
              return Some(Ok(Expr::UnaryOp {
                op: crate::syntax::UnaryOperator::Minus,
                operand: Box::new(Expr::Identifier("Infinity".to_string())),
              }));
            }
            return Some(Ok(Expr::Identifier("ComplexInfinity".to_string())));
          }
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
                right: Box::new(make_sqrt(sqrt_arg)),
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

    // Information[symbol] or Information[symbol, LongForm -> True]
    // ?symbol parses as Information[symbol]
    // ??symbol parses as Information[symbol, LongForm -> True]
    // Legacy "Full" string form is still accepted for backward compatibility.
    "Information" if args.len() == 1 || args.len() == 2 => {
      let is_full = args.len() == 2
        && (matches!(&args[1], Expr::String(s) if s == "Full")
          || matches!(&args[1],
            Expr::Rule { pattern, replacement }
              if matches!(pattern.as_ref(), Expr::Identifier(p) if p == "LongForm")
                && matches!(replacement.as_ref(),
                  Expr::Identifier(v) if v == "True")));

      if let Expr::Identifier(sym) = &args[0] {
        // Check if this is a built-in function (in functions.csv)
        let builtin_info = crate::evaluator::get_builtin_function_info(sym);
        let builtin_attrs =
          crate::evaluator::attributes::get_builtin_attributes(sym);

        if builtin_info.is_some() || !builtin_attrs.is_empty() {
          // Built-in symbol
          return Some(Ok(format_builtin_information(
            sym,
            builtin_info,
            &builtin_attrs,
            is_full,
          )));
        }

        // User-defined symbol: check OwnValues, DownValues, Attributes
        let own_value = crate::ENV.with(|e| {
          let env = e.borrow();
          env.get(sym).cloned()
        });
        let down_values = crate::FUNC_DEFS.with(|m| {
          let defs = m.borrow();
          defs.get(sym).cloned()
        });
        let user_attrs =
          crate::FUNC_ATTRS.with(|m| m.borrow().get(sym).cloned());

        let has_own = own_value.is_some();
        let has_down = down_values.is_some();
        let has_attrs = user_attrs.as_ref().is_some_and(|a| !a.is_empty());

        if !has_own && !has_down && !has_attrs {
          return Some(Ok(Expr::FunctionCall {
            name: "Missing".to_string(),
            args: vec![
              Expr::String("UnknownSymbol".to_string()),
              Expr::String(sym.clone()),
            ],
          }));
        }

        return Some(Ok(format_user_information(
          sym,
          own_value,
          down_values,
          user_attrs,
          is_full,
        )));
      }

      // Pattern query: ?Plot* or ?*Plot* — first arg is a String with wildcards
      if let Expr::String(pattern) = &args[0]
        && pattern.contains('*')
      {
        let regex_pattern =
          format!("^{}$", pattern.replace('.', "\\.").replace('*', ".*"));
        if let Ok(re) = regex::Regex::new(&regex_pattern) {
          // Collect all known names: built-in + user-defined
          let builtin_names = crate::evaluator::get_builtin_function_names();
          let user_names = crate::get_defined_names();
          let mut all_names: Vec<String> =
            builtin_names.into_iter().map(|s| s.to_string()).collect();
          for name in user_names {
            if !all_names.contains(&name) {
              all_names.push(name);
            }
          }
          all_names.sort();

          let matching: Vec<Expr> = all_names
            .into_iter()
            .filter(|n| re.is_match(n))
            .map(Expr::Identifier)
            .collect();

          return Some(Ok(Expr::FunctionCall {
            name: "InformationDataGrid".to_string(),
            args: vec![
              Expr::List(vec![Expr::FunctionCall {
                name: "Rule".to_string(),
                args: vec![
                  Expr::Identifier("System`".to_string()),
                  Expr::List(matching),
                ],
              }]),
              Expr::Identifier(
                if is_full { "True" } else { "False" }.to_string(),
              ),
            ],
          }));
        }
      }

      // Non-identifier argument — return unevaluated
      return Some(Ok(Expr::FunctionCall {
        name: "Information".to_string(),
        args: args.to_vec(),
      }));
    }

    // Definition[symbol] - show definition of a symbol
    "Definition" if args.len() == 1 => {
      if let Expr::Identifier(sym) = &args[0] {
        let mut lines: Vec<String> = Vec::new();

        // 1. Show user-set attributes (if any)
        let user_attrs =
          crate::FUNC_ATTRS.with(|m| m.borrow().get(sym).cloned());
        if let Some(attrs) = &user_attrs
          && !attrs.is_empty()
        {
          lines.push(format!("Attributes[{}] = {{{}}}", sym, attrs.join(", ")));
        }

        // 2. Show OwnValues (variable assignments)
        let own_value = crate::ENV.with(|e| {
          let env = e.borrow();
          env.get(sym).cloned()
        });
        if let Some(stored) = own_value {
          let val_str = match stored {
            crate::StoredValue::ExprVal(e) => expr_to_string(&e),
            crate::StoredValue::Raw(val) => val,
            crate::StoredValue::Association(items) => {
              let items_expr: Vec<(crate::syntax::Expr, crate::syntax::Expr)> =
                items
                  .iter()
                  .map(|(k, v)| {
                    let key_expr = crate::syntax::string_to_expr(k)
                      .unwrap_or(crate::syntax::Expr::Identifier(k.clone()));
                    let val_expr = crate::syntax::string_to_expr(v)
                      .unwrap_or(crate::syntax::Expr::Raw(v.clone()));
                    (key_expr, val_expr)
                  })
                  .collect();
              expr_to_string(&crate::syntax::Expr::Association(items_expr))
            }
          };
          lines.push(format!("{} = {}", sym, val_str));
        }

        // 3. Show UpValues first (rules attached via Real /: F[x_Real] := x
        // etc.), matching wolframscript's ordering. UpValues precede
        // DownValues in Definition output.
        let up_values = crate::UPVALUES.with(|m| {
          let defs = m.borrow();
          defs.get(sym).cloned()
        });
        if let Some(entries) = up_values {
          for (
            _outer,
            _params,
            _conds,
            _defaults,
            _heads,
            _body,
            orig_lhs,
            orig_body,
          ) in &entries
          {
            lines.push(format!(
              "{} ^:= {}",
              expr_to_string(orig_lhs),
              expr_to_string(orig_body)
            ));
          }
        }

        // 4. Show DownValues (function definitions). Filter out entries
        // that were stored via TagSet/TagSetDelayed — those belong in
        // UpValues, not DownValues, even though Woxi stores them in
        // FUNC_DEFS too so ordinary dispatch can find them.
        let upvalue_keys: std::collections::HashSet<(
          Vec<String>,
          Vec<Option<String>>,
        )> = crate::UPVALUES.with(|m| {
          let defs = m.borrow();
          let mut keys = std::collections::HashSet::new();
          for entries in defs.values() {
            for (outer, params, _, _, heads, _, _, _) in entries {
              if outer == sym {
                keys.insert((params.clone(), heads.clone()));
              }
            }
          }
          keys
        });
        let down_values = crate::FUNC_DEFS.with(|m| {
          let defs = m.borrow();
          defs.get(sym).cloned()
        });
        if let Some(overloads) = down_values {
          for (params, conds, _defaults, heads, _blank_types, body) in
            overloads.iter().filter(|(params, _, _, heads, _, _)| {
              !upvalue_keys.contains(&(params.clone(), heads.clone()))
            })
          {
            // Check if this is a specific-value definition (SameQ conditions)
            let has_sameq_conds = conds.iter().any(|c| {
              if let Some(Expr::Comparison { operators, .. }) = c {
                operators
                  .iter()
                  .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
              } else {
                false
              }
            });

            if has_sameq_conds {
              // Reconstruct f[val1, val2] = body
              let args_strs: Vec<String> = params
                .iter()
                .zip(conds.iter())
                .map(|(p, c)| {
                  if let Some(Expr::Comparison {
                    operands,
                    operators,
                    ..
                  }) = c
                    && operators.iter().any(|op| {
                      matches!(op, crate::syntax::ComparisonOp::SameQ)
                    })
                    && operands.len() == 2
                    && matches!(&operands[0], Expr::Identifier(n) if n == p)
                  {
                    return expr_to_string(&operands[1]);
                  }
                  format!("{}_", p)
                })
                .collect();
              lines.push(format!(
                "{}[{}] = {}",
                sym,
                args_strs.join(", "),
                expr_to_string(body)
              ));
            } else {
              // Reconstruct f[x_, y_Integer] := body
              let params_str: Vec<String> = params
                .iter()
                .enumerate()
                .map(|(i, p)| {
                  if let Some(head) = heads.get(i).and_then(|h| h.as_ref()) {
                    format!("{}_{}", p, head)
                  } else {
                    format!("{}_", p)
                  }
                })
                .collect();
              lines.push(format!(
                "{}[{}] := {}",
                sym,
                params_str.join(", "),
                expr_to_string(body)
              ));
            }
          }
        }

        // For built-in symbols: show attributes
        if lines.is_empty() {
          let builtin_attrs =
            crate::evaluator::attributes::get_builtin_attributes(sym);
          if !builtin_attrs.is_empty() {
            lines.push(format!(
              "Attributes[{}] = {{{}}}",
              sym,
              builtin_attrs.join(", ")
            ));
          }
        }

        // Show built-in DefaultValues (e.g. Default[Plus] := 0)
        if let Some(def_str) = builtin_default_value_str(sym) {
          lines.push(format!("Default[{}] := {}", sym, def_str));
        }

        // Show Options if the symbol has any (user-stored or built-in).
        let stored_opts =
          crate::FUNC_OPTIONS.with(|m| m.borrow().get(sym).cloned());
        let opts = stored_opts.unwrap_or_else(|| {
          crate::evaluator::dispatch::predicate_functions::builtin_default_options(sym)
        });
        if !opts.is_empty() {
          let opts_str: Vec<String> = opts.iter().map(expr_to_string).collect();
          lines.push(format!("Options[{}] = {{{}}}", sym, opts_str.join(", ")));
        }

        if lines.is_empty() {
          // Undefined symbol - return Null
          return Some(Ok(Expr::Identifier("Null".to_string())));
        }

        return Some(Ok(Expr::Raw(lines.join("\n \n"))));
      }

      return Some(Ok(Expr::FunctionCall {
        name: "Definition".to_string(),
        args: args.to_vec(),
      }));
    }

    // FullDefinition[symbol] - show definition of a symbol and all its dependencies
    "FullDefinition" if args.len() == 1 => {
      if let Expr::Identifier(sym) = &args[0] {
        // Helper: collect all Identifier names referenced in an expression
        fn collect_identifiers(expr: &Expr, out: &mut Vec<String>) {
          match expr {
            Expr::Identifier(name) => out.push(name.clone()),
            Expr::FunctionCall { name, args } => {
              out.push(name.clone());
              for a in args {
                collect_identifiers(a, out);
              }
            }
            Expr::List(elems) | Expr::CompoundExpr(elems) => {
              for e in elems {
                collect_identifiers(e, out);
              }
            }
            Expr::BinaryOp { left, right, .. } => {
              collect_identifiers(left, out);
              collect_identifiers(right, out);
            }
            Expr::UnaryOp { operand, .. } => collect_identifiers(operand, out),
            Expr::Comparison { operands, .. } => {
              for o in operands {
                collect_identifiers(o, out);
              }
            }
            Expr::Rule {
              pattern,
              replacement,
            }
            | Expr::RuleDelayed {
              pattern,
              replacement,
            } => {
              collect_identifiers(pattern, out);
              collect_identifiers(replacement, out);
            }
            Expr::ReplaceAll { expr, rules }
            | Expr::ReplaceRepeated { expr, rules } => {
              collect_identifiers(expr, out);
              collect_identifiers(rules, out);
            }
            Expr::Map { func, list }
            | Expr::Apply { func, list }
            | Expr::MapApply { func, list } => {
              collect_identifiers(func, out);
              collect_identifiers(list, out);
            }
            Expr::PrefixApply { func, arg } => {
              collect_identifiers(func, out);
              collect_identifiers(arg, out);
            }
            Expr::Postfix { expr, func } => {
              collect_identifiers(expr, out);
              collect_identifiers(func, out);
            }
            Expr::Part { expr, index } => {
              collect_identifiers(expr, out);
              collect_identifiers(index, out);
            }
            Expr::CurriedCall { func, args } => {
              collect_identifiers(func, out);
              for a in args {
                collect_identifiers(a, out);
              }
            }
            Expr::Function { body } | Expr::NamedFunction { body, .. } => {
              collect_identifiers(body, out);
            }
            Expr::Association(pairs) => {
              for (k, v) in pairs {
                collect_identifiers(k, out);
                collect_identifiers(v, out);
              }
            }
            Expr::PatternOptional { default, .. } => {
              if let Some(d) = default {
                collect_identifiers(d, out)
              }
            }
            Expr::PatternTest { test, .. } => collect_identifiers(test, out),
            _ => {}
          }
        }

        // Helper: get definition lines for a symbol (same logic as Definition)
        fn get_definition_lines(sym: &str) -> Vec<String> {
          let mut lines = Vec::new();

          let user_attrs =
            crate::FUNC_ATTRS.with(|m| m.borrow().get(sym).cloned());
          if let Some(attrs) = &user_attrs
            && !attrs.is_empty()
          {
            lines.push(format!(
              "Attributes[{}] = {{{}}}",
              sym,
              attrs.join(", ")
            ));
          }

          let own_value = crate::ENV.with(|e| {
            let env = e.borrow();
            env.get(sym).cloned()
          });
          if let Some(stored) = own_value {
            let val_str = match stored {
              crate::StoredValue::ExprVal(e) => expr_to_string(&e),
              crate::StoredValue::Raw(val) => val,
              crate::StoredValue::Association(items) => {
                let items_expr: Vec<(
                  crate::syntax::Expr,
                  crate::syntax::Expr,
                )> = items
                  .iter()
                  .map(|(k, v)| {
                    let key_expr = crate::syntax::string_to_expr(k)
                      .unwrap_or(crate::syntax::Expr::Identifier(k.clone()));
                    let val_expr = crate::syntax::string_to_expr(v)
                      .unwrap_or(crate::syntax::Expr::Raw(v.clone()));
                    (key_expr, val_expr)
                  })
                  .collect();
                expr_to_string(&crate::syntax::Expr::Association(items_expr))
              }
            };
            lines.push(format!("{} = {}", sym, val_str));
          }

          let down_values = crate::FUNC_DEFS.with(|m| {
            let defs = m.borrow();
            defs.get(sym).cloned()
          });
          if let Some(overloads) = down_values {
            for (params, conds, _defaults, heads, _blank_types, body) in
              &overloads
            {
              let has_sameq_conds = conds.iter().any(|c| {
                if let Some(Expr::Comparison { operators, .. }) = c {
                  operators
                    .iter()
                    .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
                } else {
                  false
                }
              });

              if has_sameq_conds {
                let args_strs: Vec<String> = params
                  .iter()
                  .zip(conds.iter())
                  .map(|(p, c)| {
                    if let Some(Expr::Comparison {
                      operands,
                      operators,
                      ..
                    }) = c
                      && operators.iter().any(|op| {
                        matches!(op, crate::syntax::ComparisonOp::SameQ)
                      })
                      && operands.len() == 2
                      && matches!(&operands[0], Expr::Identifier(n) if n == p)
                    {
                      return expr_to_string(&operands[1]);
                    }
                    format!("{}_", p)
                  })
                  .collect();
                lines.push(format!(
                  "{}[{}] = {}",
                  sym,
                  args_strs.join(", "),
                  expr_to_string(body)
                ));
              } else {
                let params_str: Vec<String> = params
                  .iter()
                  .enumerate()
                  .map(|(i, p)| {
                    if let Some(head) = heads.get(i).and_then(|h| h.as_ref()) {
                      format!("{}_{}", p, head)
                    } else {
                      format!("{}_", p)
                    }
                  })
                  .collect();
                lines.push(format!(
                  "{}[{}] := {}",
                  sym,
                  params_str.join(", "),
                  expr_to_string(body)
                ));
              }
            }
          }

          if lines.is_empty() {
            let builtin_attrs =
              crate::evaluator::attributes::get_builtin_attributes(sym);
            if !builtin_attrs.is_empty() {
              lines.push(format!(
                "Attributes[{}] = {{{}}}",
                sym,
                builtin_attrs.join(", ")
              ));
            }
          }

          // Show built-in DefaultValues (e.g. Default[Plus] := 0)
          if let Some(def_str) = builtin_default_value_str(sym) {
            lines.push(format!("Default[{}] := {}", sym, def_str));
          }

          lines
        }

        // Helper: check if a symbol has user-defined values
        fn has_user_definition(sym: &str) -> bool {
          let has_own = crate::ENV.with(|e| e.borrow().contains_key(sym));
          let has_down =
            crate::FUNC_DEFS.with(|m| m.borrow().contains_key(sym));
          let has_attrs =
            crate::FUNC_ATTRS.with(|m| m.borrow().contains_key(sym));
          has_own || has_down || has_attrs
        }

        // Helper: collect body expressions from a symbol's definitions
        fn get_body_exprs(sym: &str) -> Vec<Expr> {
          let mut bodies = Vec::new();

          // OwnValues body
          let own_value = crate::ENV.with(|e| {
            let env = e.borrow();
            env.get(sym).cloned()
          });
          if let Some(crate::StoredValue::ExprVal(e)) = own_value {
            bodies.push(e);
          }

          // DownValues bodies
          let down_values = crate::FUNC_DEFS.with(|m| {
            let defs = m.borrow();
            defs.get(sym).cloned()
          });
          if let Some(overloads) = down_values {
            for (_params, _conds, _defaults, _heads, _blank_types, body) in
              overloads
            {
              bodies.push(body);
            }
          }

          bodies
        }

        // Get main definition
        let main_lines = get_definition_lines(sym);
        if main_lines.is_empty() {
          return Some(Ok(Expr::Raw("".to_string())));
        }

        let mut all_sections: Vec<String> = vec![main_lines.join("\n \n")];
        let mut seen: std::collections::HashSet<String> =
          std::collections::HashSet::new();
        seen.insert(sym.clone());

        // BFS for dependent symbols
        let mut queue: std::collections::VecDeque<String> =
          std::collections::VecDeque::new();

        // Collect symbols from main symbol's bodies
        let bodies = get_body_exprs(sym);
        let mut referenced = Vec::new();
        for body in &bodies {
          collect_identifiers(body, &mut referenced);
        }
        // Deduplicate while preserving first-occurrence order
        let mut seen_in_queue: std::collections::HashSet<String> =
          std::collections::HashSet::new();
        for name in &referenced {
          if !seen.contains(name) && seen_in_queue.insert(name.clone()) {
            queue.push_back(name.clone());
          }
        }

        while let Some(dep_sym) = queue.pop_front() {
          if !seen.insert(dep_sym.clone()) {
            continue;
          }
          if !has_user_definition(&dep_sym) {
            continue;
          }

          let dep_lines = get_definition_lines(&dep_sym);
          if !dep_lines.is_empty() {
            all_sections.push(dep_lines.join("\n \n"));
          }

          // Collect further dependencies
          let dep_bodies = get_body_exprs(&dep_sym);
          let mut dep_referenced = Vec::new();
          for body in &dep_bodies {
            collect_identifiers(body, &mut dep_referenced);
          }
          for name in &dep_referenced {
            if !seen.contains(name) {
              queue.push_back(name.clone());
            }
          }
        }

        return Some(Ok(Expr::Raw(all_sections.join("\n \n"))));
      }

      return Some(Ok(Expr::FunctionCall {
        name: "FullDefinition".to_string(),
        args: args.to_vec(),
      }));
    }

    // Sow[expr] or Sow[expr, tag] - adds expr to the current Reap collection
    "Sow" if args.len() == 1 || args.len() == 2 => {
      // Sow[val, {tag1, tag2, ...}] emits one (val, tag_i) pair per tag;
      // with a single tag or 1-arg form, emits one pair.
      let tags: Vec<Expr> = match args.get(1) {
        Some(Expr::List(items)) => items.clone(),
        Some(t) => vec![t.clone()],
        None => vec![Expr::Identifier("None".to_string())],
      };
      crate::SOW_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if let Some(last) = stack.last_mut() {
          for tag in &tags {
            last.push((args[0].clone(), tag.clone()));
          }
        }
      });
      return Some(Ok(args[0].clone()));
    }

    // Reap[expr] or Reap[expr, pattern] - evaluates expr, collecting all Sow'd values
    "Reap" if (1..=3).contains(&args.len()) => {
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
        // Reap[expr, patt] / Reap[expr, {patt1, ...}] / Reap[expr, patt, f]
        let patt_arg = match evaluate_expr_to_expr(&args[1]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        let patterns = match &patt_arg {
          Expr::List(pats) => pats.clone(),
          _ => vec![patt_arg.clone()],
        };
        let is_list_form = matches!(&patt_arg, Expr::List(_));
        let wrap_fn: Option<Expr> = if args.len() == 3 {
          match evaluate_expr_to_expr(&args[2]) {
            Ok(f) => Some(f),
            Err(e) => return Some(Err(e)),
          }
        } else {
          None
        };

        // Build per-pattern groups: for each pattern, find all matching
        // tags (in order of first sow), and for each unique tag collect
        // its sowed values. Sows whose tags do not match any of the given
        // patterns are propagated to the enclosing Reap scope (if any),
        // matching Wolfram's behaviour.
        let mut matched_indices: std::collections::HashSet<usize> =
          std::collections::HashSet::new();
        let mut result_groups: Vec<Expr> = Vec::new();
        for patt in &patterns {
          let mut tag_order: Vec<Expr> = Vec::new();
          let mut tag_groups: Vec<Vec<Expr>> = Vec::new();
          for (i, (val, tag)) in sowed.iter().enumerate() {
            if !tag_matches_pattern(tag, patt) {
              continue;
            }
            matched_indices.insert(i);
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
          // Build the group list for this pattern.
          let per_pattern: Vec<Expr> = tag_order
            .into_iter()
            .zip(tag_groups)
            .map(|(tag, vals)| {
              let vals_list = Expr::List(vals);
              if let Some(f) = &wrap_fn {
                // Apply f[tag, {values}] — handles named heads, anonymous
                // functions (Function[body]), and NamedFunction forms.
                crate::evaluator::function_application::apply_curried_call(
                  f,
                  &[tag, vals_list],
                )
                .unwrap_or(Expr::FunctionCall {
                  name: "List".to_string(),
                  args: vec![],
                })
              } else {
                vals_list
              }
            })
            .collect();

          if is_list_form {
            // {patt1, patt2, ...} form: each pattern contributes a list,
            // even if empty.
            result_groups.push(Expr::List(per_pattern));
          } else {
            // Single-pattern form: the result groups are the per-tag
            // entries directly (flattened — no extra wrapping).
            result_groups.extend(per_pattern);
          }
        }
        // Forward unmatched sows to the enclosing Reap scope (if any) so
        // nested Reap[..., patt] expressions propagate sows that did not
        // match the inner pattern upward.
        let unmatched: Vec<(Expr, Expr)> = sowed
          .iter()
          .enumerate()
          .filter(|(i, _)| !matched_indices.contains(i))
          .map(|(_, pair)| pair.clone())
          .collect();
        if !unmatched.is_empty() {
          crate::SOW_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some(parent) = stack.last_mut() {
              parent.extend(unmatched);
            }
          });
        }
        return Some(Ok(Expr::List(vec![result, Expr::List(result_groups)])));
      }
    }

    // ReplaceAll and ReplaceRepeated function call forms
    "ReplaceAll" if args.len() == 2 => {
      // Unwrap Dispatch[rules] → just use rules
      let rules = if let Expr::FunctionCall { name: dn, args: da } = &args[1] {
        if dn == "Dispatch" && da.len() == 1 {
          &da[0]
        } else {
          &args[1]
        }
      } else {
        &args[1]
      };
      // Re-evaluate the result so e.g. {1 + 2} becomes {3} after substitution.
      // This matches the /. operator form, which re-evaluates via TailCall.
      return Some(
        apply_replace_all_ast(&args[0], rules)
          .and_then(|r| evaluate_expr_to_expr(&r)),
      );
    }
    "ReplaceRepeated" if args.len() == 2 => {
      let rules = if let Expr::FunctionCall { name: dn, args: da } = &args[1] {
        if dn == "Dispatch" && da.len() == 1 {
          &da[0]
        } else {
          &args[1]
        }
      } else {
        &args[1]
      };
      return Some(apply_replace_repeated_ast(&args[0], rules));
    }
    "Replace" if args.len() == 2 => {
      return Some(
        apply_replace_ast(&args[0], &args[1])
          .and_then(|r| evaluate_expr_to_expr(&r)),
      );
    }
    // ReplaceList[expr, rules] / ReplaceList[expr, rules, n]
    // Enumerates all distinct bindings that match a list-shaped sequence rule
    // like `{___, x__, ___} -> {x}`. Falls back to the single-match path for
    // other rule shapes.
    "ReplaceList" if args.len() == 2 || args.len() == 3 => {
      let n_arg = args.get(2).cloned();
      let max_matches: Option<i128> = match &n_arg {
        Some(Expr::Integer(n)) => Some(*n),
        Some(Expr::Identifier(s)) if s == "Infinity" => None,
        None => None,
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "ReplaceList".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      if max_matches == Some(0) {
        return Some(Ok(Expr::List(vec![])));
      }

      // ReplaceList[expr, {{rules1}, {rules2}, ...}, n] form: when every
      // top-level element of the rules argument is itself a list (with the
      // first element being a Rule/RuleDelayed), apply each rules-list
      // separately and collect the per-rules-list results into an outer list.
      if let Expr::List(rule_groups) = &args[1]
        && !rule_groups.is_empty()
        && rule_groups.iter().all(|g| {
          matches!(g, Expr::List(items) if items.iter().all(|it| matches!(it, Expr::Rule { .. } | Expr::RuleDelayed { .. })))
        })
      {
        let mut outer: Vec<Expr> = Vec::new();
        for group in rule_groups {
          let group_args: Vec<Expr> = if args.len() == 3 {
            vec![args[0].clone(), group.clone(), args[2].clone()]
          } else {
            vec![args[0].clone(), group.clone()]
          };
          let inner_call = Expr::FunctionCall {
            name: "ReplaceList".to_string(),
            args: group_args,
          };
          match evaluate_expr_to_expr(&inner_call) {
            Ok(r) => outer.push(r),
            Err(e) => return Some(Err(e)),
          }
        }
        return Some(Ok(Expr::List(outer)));
      }

      // ReplaceList[expr, {rule1, rule2, ...}, n]: a flat list of rules.
      // Apply each rule via the single-rule path and concatenate the
      // results, capping the total at `n`.
      if let Expr::List(rules) = &args[1]
        && !rules.is_empty()
        && rules
          .iter()
          .all(|r| matches!(r, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
      {
        let mut combined: Vec<Expr> = Vec::new();
        'outer: for rule in rules {
          let inner_args: Vec<Expr> = if args.len() == 3 {
            vec![args[0].clone(), rule.clone(), args[2].clone()]
          } else {
            vec![args[0].clone(), rule.clone()]
          };
          let inner_call = Expr::FunctionCall {
            name: "ReplaceList".to_string(),
            args: inner_args,
          };
          let result = match evaluate_expr_to_expr(&inner_call) {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
          };
          if let Expr::List(items) = &result {
            for it in items {
              combined.push(it.clone());
              if let Some(n) = max_matches
                && combined.len() as i128 >= n
              {
                break 'outer;
              }
            }
          } else {
            combined.push(result);
            if let Some(n) = max_matches
              && combined.len() as i128 >= n
            {
              break 'outer;
            }
          }
        }
        return Some(Ok(Expr::List(combined)));
      }
      // Try the list-sequence enumerator first.
      if let Expr::List(expr_elems) = &args[0]
        && let Expr::Rule {
          pattern,
          replacement,
        } = &args[1]
        && let Expr::List(pat_elems) = pattern.as_ref()
        && let Some(all_bindings) =
          enumerate_list_sequence_matches(expr_elems, pat_elems)
      {
        let mut results: Vec<Expr> = Vec::new();
        for bindings in all_bindings {
          let mut rhs = replacement.as_ref().clone();
          for (name, value) in bindings {
            rhs = crate::syntax::substitute_variable(&rhs, &name, &value);
          }
          let evaluated = match evaluate_expr_to_expr(&rhs) {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
          };
          results.push(evaluated);
          if let Some(n) = max_matches
            && results.len() as i128 >= n
          {
            break;
          }
        }
        return Some(Ok(Expr::List(results)));
      }
      // Flat partition enumerator: e.g. ReplaceList[a+b+c, x_+y_ -> {x,y}]
      // enumerates all Flat partitions of the expression args into
      // pat_args.len() non-empty groups.
      if let Expr::FunctionCall {
        name: expr_head,
        args: expr_fargs,
      } = &args[0]
        && let Expr::Rule {
          pattern,
          replacement,
        } = &args[1]
        && let Expr::FunctionCall {
          name: pat_head,
          args: pat_fargs,
        } = pattern.as_ref()
        && expr_head == pat_head
        && !pat_fargs.is_empty()
        && pat_fargs.len() <= expr_fargs.len()
      {
        let has_flat = crate::evaluator::listable::is_builtin_flat(expr_head)
          || crate::FUNC_ATTRS.with(|m| {
            m.borrow()
              .get(expr_head.as_str())
              .is_some_and(|attrs| attrs.contains(&"Flat".to_string()))
          });
        if has_flat {
          let all_bindings =
            crate::evaluator::pattern_matching::enumerate_flat_partition_matches(
              expr_head, pat_fargs, expr_fargs,
            );
          if !all_bindings.is_empty() {
            let mut results: Vec<Expr> = Vec::new();
            for bindings in all_bindings {
              let mut rhs = replacement.as_ref().clone();
              for (name, value) in bindings {
                rhs = crate::syntax::substitute_variable(&rhs, &name, &value);
              }
              let evaluated = match evaluate_expr_to_expr(&rhs) {
                Ok(r) => r,
                Err(e) => return Some(Err(e)),
              };
              results.push(evaluated);
              if let Some(n) = max_matches
                && results.len() as i128 >= n
              {
                break;
              }
            }
            return Some(Ok(Expr::List(results)));
          }
        }
      }
      let result = match apply_replace_ast(&args[0], &args[1])
        .and_then(|r| evaluate_expr_to_expr(&r))
      {
        Ok(r) => r,
        Err(e) => return Some(Err(e)),
      };
      // `apply_replace_ast` returns the original expr unchanged when no
      // top-level match fires, so compare strings to detect a match.
      let before = crate::syntax::expr_to_string(&args[0]);
      let after = crate::syntax::expr_to_string(&result);
      if before == after {
        return Some(Ok(Expr::List(vec![])));
      }
      return Some(Ok(Expr::List(vec![result])));
    }
    "Replace" if args.len() == 3 || args.len() == 4 => {
      // Replace[expr, rules, levelspec] or
      // Replace[expr, rules, levelspec, Heads -> True]
      let heads_on = args
        .get(3)
        .map(|opt| {
          matches!(opt,
            Expr::Rule { pattern, replacement }
              if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Heads")
                 && matches!(replacement.as_ref(), Expr::Identifier(s) if s == "True"))
        })
        .unwrap_or(false);
      return Some(
        apply_replace_with_level_ast(&args[0], &args[1], &args[2], heads_on)
          .and_then(|r| evaluate_expr_to_expr(&r)),
      );
    }

    // Format[expr, OutputForm] and OutputForm[expr] → 2D rendering
    "Format"
      if args.len() == 2
        && matches!(&args[1], Expr::Identifier(f) if f == "OutputForm") =>
    {
      let rendered = crate::syntax::expr_to_output_form_2d(&args[0]);
      return Some(Ok(Expr::Raw(rendered)));
    }
    "OutputForm" if args.len() == 1 => {
      // Graphics/Graphics3D bodies haven't been converted to Expr::Graphics
      // yet at dispatch time (that happens post-evaluation in lib.rs), so
      // render them to their "-Graphics-" abbreviation here explicitly.
      if let Expr::FunctionCall { name: inner, .. } = &args[0] {
        if inner == "Graphics" {
          return Some(Ok(Expr::Raw("-Graphics-".to_string())));
        }
        if inner == "Graphics3D" {
          return Some(Ok(Expr::Raw("-Graphics3D-".to_string())));
        }
      }
      let rendered = crate::syntax::expr_to_output_form_2d(&args[0]);
      return Some(Ok(Expr::Raw(rendered)));
    }
    // ToBoxes[expr] / ToBoxes[expr, form] — convert expression to box form
    "ToBoxes" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(expr_to_box_form(&args[0])));
    }
    // MakeBoxes[expr] / MakeBoxes[expr, form] — convert unevaluated expression to box form
    // MakeBoxes has HoldAllComplete so its argument is not evaluated first.
    "MakeBoxes" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(expr_to_box_form(&args[0])));
    }
    // RawBoxes[boxes] — wrapper that tells the display system to render boxes directly
    // It simply returns itself; the rendering pipeline in generate_output_svg() handles it.
    "RawBoxes" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "RawBoxes".to_string(),
        args: args.to_vec(),
      }));
    }
    // DisplayForm[boxes] — wrapper that causes box expressions to be rendered visually.
    // It stays unevaluated; the rendering pipeline handles it like RawBoxes.
    "DisplayForm" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "DisplayForm".to_string(),
        args: args.to_vec(),
      }));
    }
    // Low-level typesetting box constructors — these are inert and return themselves.
    "FractionBox" if args.len() == 2 || args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "SqrtBox" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "SuperscriptBox" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "SubscriptBox" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "SubsuperscriptBox" if args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "OverscriptBox" if args.len() == 2 || args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "UnderscriptBox" if args.len() == 2 || args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "UnderoverscriptBox" if args.len() == 3 || args.len() == 4 => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "RadicalBox" if args.len() == 2 || args.len() == 3 => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "FrameBox" if !args.is_empty() => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "StyleBox" if !args.is_empty() => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "GridBox" if !args.is_empty() => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "TagBox" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    "InterpretationBox" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    // Area[region] — compute the area of a geometric region
    "Area" if args.len() == 1 => {
      return Some(compute_area(&args[0]));
    }
    "RegionCentroid" if args.len() == 1 => {
      return Some(compute_region_centroid(&args[0]));
    }
    "ArcLength" if args.len() == 1 => {
      return Some(compute_arc_length(&args[0]));
    }
    "Perimeter" if args.len() == 1 => {
      return Some(compute_perimeter(&args[0]));
    }
    "Insphere" if args.len() == 1 => {
      return Some(compute_insphere(&args[0]));
    }
    "RegionWithin" if args.len() == 2 => {
      return Some(region_within(&args[0], &args[1], args));
    }
    // PlanarAngle[{p1, vertex, p2}] — angle at vertex between rays to p1 and p2
    "PlanarAngle" if args.len() == 1 => {
      if let Expr::List(pts) = &args[0]
        && pts.len() == 3
      {
        return Some(compute_planar_angle(&pts[0], &pts[1], &pts[2]));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "PlanarAngle".to_string(),
        args: args.to_vec(),
      }));
    }
    // QBinomial[n, k, q] — Gaussian binomial coefficient (q-binomial)
    "QBinomial" if args.len() == 3 => {
      if let (Expr::Integer(n), Expr::Integer(k)) = (&args[0], &args[1]) {
        let n = *n;
        let k = *k;
        if k < 0 || k > n || n < 0 {
          return Some(Ok(Expr::Integer(0)));
        }
        if k == 0 || k == n {
          return Some(Ok(Expr::Integer(1)));
        }
        let q = &args[2];
        // Only evaluate for numeric q
        let is_numeric = matches!(q, Expr::Integer(_) | Expr::Real(_))
          || matches!(q, Expr::FunctionCall { name, .. } if name == "Rational");
        if !is_numeric {
          return Some(Ok(Expr::FunctionCall {
            name: "QBinomial".to_string(),
            args: args.to_vec(),
          }));
        }
        // Product[(1 - q^(n-i)) / (1 - q^(i+1)), {i, 0, k-1}]
        let mut result = Expr::Integer(1);
        for i in 0..k {
          let num = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              Expr::Integer(1),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  Expr::Integer(-1),
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![q.clone(), Expr::Integer(n - i)],
                  },
                ],
              },
            ],
          };
          let den = Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              Expr::Integer(1),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  Expr::Integer(-1),
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![q.clone(), Expr::Integer(i + 1)],
                  },
                ],
              },
            ],
          };
          result = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              result,
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  num,
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![den, Expr::Integer(-1)],
                  },
                ],
              },
            ],
          };
        }
        return Some(crate::evaluator::evaluate_expr_to_expr(&result));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "QBinomial".to_string(),
        args: args.to_vec(),
      }));
    }
    // RegionEqual[r1, r2, ...] — test whether regions are equal
    "RegionEqual" => {
      return Some(compute_region_equal(args));
    }
    // FindSequenceFunction[list, var] — find a formula for an integer sequence
    "FindSequenceFunction" if args.len() == 2 => {
      return Some(find_sequence_function(&args[0], &args[1]));
    }
    // Activate[expr] — replace Inactive[f][args...] with f[args...] and evaluate
    "Activate" if args.len() == 1 || args.len() == 2 => {
      let filter: Option<Vec<String>> = if args.len() == 2 {
        // Activate[expr, f] or Activate[expr, f1 | f2 | ...]
        Some(extract_activate_filter(&args[1]))
      } else {
        None
      };
      let activated = activate_expr(&args[0], &filter);
      return Some(crate::evaluator::evaluate_expr_to_expr(&activated));
    }
    // Form wrappers -- keep as wrappers (matching wolframscript OutputForm behavior)
    "MathMLForm" | "StandardForm" | "InputForm" if !args.is_empty() => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }
    // TraditionalForm[expr] — keep as-is (formatting wrapper, not eagerly evaluated)
    "TraditionalForm" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "TraditionalForm".to_string(),
        args: args.to_vec(),
      }));
    }
    // Format is transparent -- returns the inner expression
    "Format" if !args.is_empty() => {
      return Some(Ok(args[0].clone()));
    }

    // Around[value, uncertainty] — convert integer value to real when uncertainty is real
    "Around" if args.len() >= 2 => {
      let mut new_args = args.to_vec();
      if let Expr::Integer(n) = &new_args[0] {
        // If any other argument is Real, convert integer to Real
        let has_real = new_args[1..].iter().any(|a| matches!(a, Expr::Real(_)));
        if has_real {
          new_args[0] = Expr::Real(*n as f64);
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Around".to_string(),
        args: new_args,
      }));
    }

    // Symbolic operators with no built-in meaning -- just return as-is with evaluated args
    "Therefore" | "Because" | "TableForm" | "MatrixForm" | "Row" | "In"
    | "Grid" | "TextGrid" | "Column" | "Framed" => {
      return Some(Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      }));
    }

    // SequenceForm[a, b, c, ...] prints the concatenated string forms of
    // its arguments (strings are printed without surrounding quotes).
    "SequenceForm" => {
      let rendered = args
        .iter()
        .map(|a| match a {
          Expr::String(s) => s.clone(),
          _ => crate::syntax::expr_to_string(a),
        })
        .collect::<String>();
      return Some(Ok(Expr::Raw(rendered)));
    }

    // Default[symbol] - return the default value for a built-in symbol
    "Default" if args.len() == 1 => {
      if let Expr::Identifier(sym) = &args[0]
        && let Some(val) = builtin_default_value(sym)
      {
        return Some(Ok(val));
      }
      // Return unevaluated for symbols without defaults
      return Some(Ok(Expr::FunctionCall {
        name: "Default".to_string(),
        args: args.to_vec(),
      }));
    }

    // Default[f, n] / Default[f, n, m] — when the user hasn't installed a
    // specific multi-argument Default[f, n] DownValue, fall through to the
    // single-argument Default[f] (matching Wolfram's lookup chain).
    "Default" if (args.len() == 2 || args.len() == 3) => {
      if let Expr::Identifier(sym) = &args[0] {
        // First try the position-less form.
        let one_arg_call = Expr::FunctionCall {
          name: "Default".to_string(),
          args: vec![args[0].clone()],
        };
        if let Ok(result) = evaluate_expr_to_expr(&one_arg_call) {
          let original = crate::syntax::expr_to_string(&one_arg_call);
          let after = crate::syntax::expr_to_string(&result);
          if after != original {
            return Some(Ok(result));
          }
        }
        // Try the builtin per-position default.
        if let Expr::Integer(pos) = &args[1]
          && *pos > 0
          && let Some(val) = builtin_default_value_at_position(sym, *pos as usize)
        {
          return Some(Ok(val));
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Default".to_string(),
        args: args.to_vec(),
      }));
    }

    _ => {}
  }
  None
}

/// Enumerate all ways a list of sequence-pattern slots can match a list of
/// expression elements. Each pattern slot is a `Pattern[name, BlankSequence]`
/// or `Pattern[name, BlankNullSequence]` (possibly with an empty name).
/// Returns `None` if the pattern structure isn't recognised.
/// Reap tag matching: a bare `_`-style pattern matches any tag via the
/// pattern matcher; any other expression compares structurally against the
/// tag. Keeps the behaviour consistent with Wolfram's `Reap[expr, form]`,
/// where `form` is a symbol or a pattern like `_Symbol`, `_Integer`, etc.
fn tag_matches_pattern(tag: &Expr, patt: &Expr) -> bool {
  if crate::evaluator::pattern_matching::contains_pattern(patt) {
    crate::evaluator::pattern_matching::match_pattern(tag, patt).is_some()
  } else {
    crate::syntax::expr_to_string(tag) == crate::syntax::expr_to_string(patt)
  }
}

fn enumerate_list_sequence_matches(
  expr_elems: &[Expr],
  pattern_elems: &[Expr],
) -> Option<Vec<Vec<(String, Expr)>>> {
  // Parse each pattern slot into (name, min_count, max_count_unbounded).
  struct Slot<'a> {
    name: &'a str,
    min: usize,
    // `None` means unbounded — consume the rest.
  }
  let mut slots: Vec<Slot> = Vec::with_capacity(pattern_elems.len());
  for p in pattern_elems {
    let (name, min): (&str, usize) = match p {
      Expr::Pattern {
        name, blank_type, ..
      } => {
        let min = match blank_type {
          2 => 1usize,
          3 => 0,
          _ => return None,
        };
        (name.as_str(), min)
      }
      Expr::FunctionCall {
        name: fname,
        args: fargs,
      } if fname == "Pattern" && fargs.len() == 2 => {
        let name = if let Expr::Identifier(s) = &fargs[0] {
          s.as_str()
        } else {
          ""
        };
        let min = match &fargs[1] {
          Expr::FunctionCall {
            name: bname,
            args: bargs,
          } if bargs.is_empty() && bname == "BlankSequence" => 1usize,
          Expr::FunctionCall {
            name: bname,
            args: bargs,
          } if bargs.is_empty() && bname == "BlankNullSequence" => 0,
          _ => return None,
        };
        (name, min)
      }
      Expr::FunctionCall {
        name: bname,
        args: bargs,
      } if bargs.is_empty() && bname == "BlankSequence" => ("", 1),
      Expr::FunctionCall {
        name: bname,
        args: bargs,
      } if bargs.is_empty() && bname == "BlankNullSequence" => ("", 0),
      _ => return None,
    };
    slots.push(Slot { name, min });
  }

  // Brute-force enumeration: each slot with min=m takes >= m elements.
  let n = expr_elems.len();
  let k = slots.len();
  if k == 0 {
    return if n == 0 {
      Some(vec![vec![]])
    } else {
      Some(vec![])
    };
  }
  // Total minimum; verify feasibility.
  let total_min: usize = slots.iter().map(|s| s.min).sum();
  if total_min > n {
    return Some(vec![]);
  }
  // Enumerate all compositions of (n − total_min) into k non-negative parts,
  // where slot i gets `extras[i] + slots[i].min` elements.
  let slack = n - total_min;
  let mut results: Vec<Vec<(String, Expr)>> = Vec::new();
  let mut extras = vec![0usize; k];
  fn recurse(
    idx: usize,
    remaining: usize,
    extras: &mut Vec<usize>,
    slots: &[Slot<'_>],
    expr_elems: &[Expr],
    results: &mut Vec<Vec<(String, Expr)>>,
  ) {
    if idx + 1 == slots.len() {
      extras[idx] = remaining;
      // Build this binding set.
      let mut pos = 0;
      let mut bindings: Vec<(String, Expr)> = Vec::new();
      for (i, slot) in slots.iter().enumerate() {
        let len = slot.min + extras[i];
        let slice = &expr_elems[pos..pos + len];
        pos += len;
        if !slot.name.is_empty() {
          // Bind to a Sequence so substitute_variable can splice/wrap
          // appropriately. Single elements stay atomic.
          let value = if len == 1 {
            slice[0].clone()
          } else {
            Expr::FunctionCall {
              name: "Sequence".to_string(),
              args: slice.to_vec(),
            }
          };
          bindings.push((slot.name.to_string(), value));
        }
      }
      results.push(bindings);
      extras[idx] = 0;
      return;
    }
    for e in 0..=remaining {
      extras[idx] = e;
      recurse(idx + 1, remaining - e, extras, slots, expr_elems, results);
    }
    extras[idx] = 0;
  }
  recurse(0, slack, &mut extras, &slots, expr_elems, &mut results);
  Some(results)
}

/// Format Information output for a built-in symbol.
fn format_builtin_information(
  sym: &str,
  info: Option<&crate::evaluator::BuiltinFunctionInfo>,
  builtin_attrs: &[&str],
  is_full: bool,
) -> Expr {
  // Collect user-set attributes (merged with built-in)
  let user_attrs = crate::FUNC_ATTRS.with(|m| m.borrow().get(sym).cloned());
  let mut all_attrs: Vec<String> =
    builtin_attrs.iter().map(|a| a.to_string()).collect();
  if let Some(ua) = user_attrs {
    for a in ua {
      if !all_attrs.contains(&a) {
        all_attrs.push(a);
      }
    }
  }
  all_attrs.sort();

  let description = info.map(|i| i.description).unwrap_or("");

  // Build association fields
  let mut fields: Vec<String> = Vec::new();
  fields.push(format!("Name -> {}", sym));
  if !description.is_empty() {
    fields.push(format!("Usage -> {}", description));
  }

  if is_full {
    // Full output: include all fields
    fields.push("ObjectType -> Symbol".to_string());

    // Options
    let stored_opts =
      crate::FUNC_OPTIONS.with(|m| m.borrow().get(sym).cloned());
    if let Some(opts) = stored_opts
      && !opts.is_empty()
    {
      let opts_str = opts
        .iter()
        .map(expr_to_string)
        .collect::<Vec<_>>()
        .join(", ");
      fields.push(format!("Options -> {{{}}}", opts_str));
    } else {
      fields.push("Options -> {}".to_string());
    }

    // Attributes
    if all_attrs.is_empty() {
      fields.push("Attributes -> {}".to_string());
    } else {
      fields.push(format!("Attributes -> {{{}}}", all_attrs.join(", ")));
    }

    fields.push(format!("FullName -> System`{}", sym));
  }

  let result_str = format!(
    "InformationData[<|{}|>, {}]",
    fields.join(", "),
    if is_full { "True" } else { "False" }
  );
  Expr::Raw(result_str)
}

/// Format Information output for a user-defined symbol.
fn format_user_information(
  sym: &str,
  own_value: Option<crate::StoredValue>,
  down_values: Option<
    Vec<(
      Vec<String>,
      Vec<Option<Expr>>,
      Vec<Option<Expr>>,
      Vec<Option<String>>,
      Vec<u8>,
      Expr,
    )>,
  >,
  user_attrs: Option<Vec<String>>,
  is_full: bool,
) -> Expr {
  // Build OwnValues field
  let own_str = if let Some(stored) = own_value {
    let val_str = match stored {
      crate::StoredValue::ExprVal(e) => expr_to_string(&e),
      crate::StoredValue::Raw(val) => val,
      crate::StoredValue::Association(items) => {
        let items_expr: Vec<(crate::syntax::Expr, crate::syntax::Expr)> = items
          .iter()
          .map(|(k, v)| {
            let key_expr = crate::syntax::string_to_expr(k)
              .unwrap_or(crate::syntax::Expr::Identifier(k.clone()));
            let val_expr = crate::syntax::string_to_expr(v)
              .unwrap_or(crate::syntax::Expr::Raw(v.clone()));
            (key_expr, val_expr)
          })
          .collect();
        expr_to_string(&crate::syntax::Expr::Association(items_expr))
      }
    };
    format!(
      "Information`InformationValueForm[OwnValues, {}, {{{} -> {}}}]",
      sym, sym, val_str
    )
  } else {
    "None".to_string()
  };

  // Build DownValues field
  let down_str = if let Some(overloads) = down_values {
    let rules: Vec<String> = overloads
      .iter()
      .map(|(params, conds, _defaults, heads, _blank_types, body)| {
        let params_str = params
          .iter()
          .enumerate()
          .map(|(i, p)| {
            if let Some(Some(Expr::Comparison {
              operands,
              operators,
            })) = conds.get(i)
              && operators
                .iter()
                .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
              && let Some(literal_val) = operands.get(1)
            {
              return expr_to_string(literal_val);
            }
            if let Some(head) = heads.get(i).and_then(|h| h.as_ref()) {
              format!("{}_{}", p, head)
            } else {
              format!("{}_", p)
            }
          })
          .collect::<Vec<_>>()
          .join(", ");
        let body_str = expr_to_string(body);
        format!("{}[{}] :> {}", sym, params_str, body_str)
      })
      .collect();
    format!(
      "Information`InformationValueForm[DownValues, {}, {{{}}}]",
      sym,
      rules.join(", ")
    )
  } else {
    "None".to_string()
  };

  // Build Attributes field
  let attrs_str = if let Some(attrs) = user_attrs {
    if attrs.is_empty() {
      "{}".to_string()
    } else {
      format!("{{{}}}", attrs.join(", "))
    }
  } else {
    "{}".to_string()
  };

  if is_full {
    // Full output: all fields
    // Options
    let stored_opts =
      crate::FUNC_OPTIONS.with(|m| m.borrow().get(sym).cloned());
    let opts_str = if let Some(opts) = stored_opts
      && !opts.is_empty()
    {
      let s = opts
        .iter()
        .map(expr_to_string)
        .collect::<Vec<_>>()
        .join(", ");
      format!("{{{}}}", s)
    } else {
      "None".to_string()
    };

    let result_str = format!(
      "InformationData[<|ObjectType -> Symbol, \
       Usage -> Global`{sym}, \
       Documentation -> None, \
       OwnValues -> {own_str}, \
       UpValues -> None, \
       DownValues -> {down_str}, \
       SubValues -> None, \
       DefaultValues -> None, \
       NValues -> None, \
       FormatValues -> None, \
       Options -> {opts_str}, \
       Attributes -> {attrs_str}, \
       FullName -> Global`{sym}|>, True]"
    );
    Expr::Raw(result_str)
  } else {
    let result_str = format!(
      "InformationData[<|ObjectType -> Symbol, \
       Usage -> Global`{sym}, \
       Documentation -> None, \
       OwnValues -> {own_str}, \
       UpValues -> None, \
       DownValues -> {down_str}, \
       SubValues -> None, \
       DefaultValues -> None, \
       NValues -> None, \
       FormatValues -> None, \
       Options -> None, \
       Attributes -> {attrs_str}, \
       FullName -> Global`{sym}|>, False]"
    );
    Expr::Raw(result_str)
  }
}

/// Return the built-in Default value for a symbol as an Expr, if one exists.
/// This is Default[f] — the position-independent default.
pub fn builtin_default_value(sym: &str) -> Option<Expr> {
  match sym {
    "Plus" => Some(Expr::Integer(0)),
    "Times" => Some(Expr::Integer(1)),
    _ => None,
  }
}

/// Return the built-in Default value for a symbol at a specific position.
/// This is Default[f, position] — position is 1-indexed.
pub fn builtin_default_value_at_position(
  sym: &str,
  position: usize,
) -> Option<Expr> {
  match (sym, position) {
    ("Plus", _) => Some(Expr::Integer(0)),
    ("Times", _) => Some(Expr::Integer(1)),
    ("Power", 2) => Some(Expr::Integer(1)),
    _ => None,
  }
}

/// Return the string representation of the built-in Default value for a symbol, if one exists.
fn builtin_default_value_str(sym: &str) -> Option<&'static str> {
  match sym {
    "Plus" => Some("0"),
    "Times" => Some("1"),
    _ => None,
  }
}

/// Wrap a box form in parentheses: RowBox[{"(", inner, ")"}].
fn paren_box(inner: Expr) -> Expr {
  Expr::FunctionCall {
    name: "RowBox".to_string(),
    args: vec![Expr::List(vec![
      Expr::String("(".to_string()),
      inner,
      Expr::String(")".to_string()),
    ])],
  }
}

/// Returns true if `expr` is an additive expression (Plus or BinaryOp::Plus/Minus)
/// that needs to be parenthesized when used as a sub-expression in a Power base
/// or Times factor (so the rendered output is unambiguous).
fn needs_paren_in_product(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      (name == "Plus" && args.len() >= 2)
        // Negative term Times[-n, ...] renders with a leading minus sign
        // and also needs parens when used as a multiplicand/Power base.
        || (name == "Times"
          && args.len() >= 2
          && matches!(&args[0], Expr::Integer(n) if *n < 0))
    }
    Expr::BinaryOp { op, .. } => matches!(
      op,
      crate::syntax::BinaryOperator::Plus
        | crate::syntax::BinaryOperator::Minus
    ),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      ..
    } => true,
    _ => false,
  }
}

/// Box form of `expr`, parenthesized when the expression is additive so it
/// renders unambiguously as the base of a Power or a factor in a Times product.
fn box_with_paren_if_needed(expr: &Expr) -> Expr {
  let boxed = expr_to_box_form(expr);
  if needs_paren_in_product(expr) {
    paren_box(boxed)
  } else {
    boxed
  }
}

/// Convert graphics primitives to their box equivalents, recursing into
/// Lists and inner Graphics/Graphics3D expressions. Leaves non-primitive
/// expressions (numbers, symbols, options, colors) untouched. Used only
/// when building a GraphicsBox/Graphics3DBox from a Graphics expression.
fn to_graphics_boxes(expr: &Expr) -> Expr {
  match expr {
    Expr::List(items) => {
      Expr::List(items.iter().map(to_graphics_boxes).collect())
    }
    Expr::FunctionCall { name, args } => {
      const PRIMITIVES: &[&str] = &[
        "Arrow",
        "Arrowheads",
        "BezierCurve",
        "Circle",
        "Cone",
        "Cuboid",
        "Cylinder",
        "Disk",
        "FilledCurve",
        "Line",
        "Parallelepiped",
        "Point",
        "Polygon",
        "Polyhedron",
        "Rectangle",
        "RegularPolygon",
        "Sphere",
        "Tetrahedron",
        "Text",
        "Tube",
      ];
      if PRIMITIVES.contains(&name.as_str()) {
        return Expr::FunctionCall {
          name: format!("{}Box", name),
          args: args.to_vec(),
        };
      }
      Expr::FunctionCall {
        name: name.clone(),
        args: args.iter().map(to_graphics_boxes).collect(),
      }
    }
    _ => expr.clone(),
  }
}

/// Convert an expression to its box form representation for TraditionalForm/StandardForm.
pub fn expr_to_box_form(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(n) => Expr::String(n.to_string()),
    Expr::BigInteger(n) => Expr::String(n.to_string()),
    Expr::Real(f) => Expr::String(crate::syntax::format_real(*f)),
    Expr::BigFloat(digits, prec) => {
      Expr::String(crate::syntax::format_bigfloat(digits, *prec))
    }
    Expr::Identifier(s) | Expr::Constant(s) => Expr::String(s.clone()),
    Expr::String(s) => Expr::String(format!("\"{}\"", s)),
    Expr::Slot(n) => Expr::String(if *n == 1 {
      "#".to_string()
    } else {
      format!("#{}", n)
    }),
    Expr::SlotSequence(n) => Expr::String(if *n == 1 {
      "##".to_string()
    } else {
      format!("##{}", n)
    }),
    // UnaryOp: -x → RowBox[{"-", box(x)}], !x → RowBox[{"!", box(x)}]
    Expr::UnaryOp { op, operand } => {
      let op_str = match op {
        crate::syntax::UnaryOperator::Minus => "-",
        crate::syntax::UnaryOperator::Not => "!",
      };
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(vec![
          Expr::String(op_str.to_string()),
          expr_to_box_form(operand),
        ])],
      }
    }
    // BinaryOp::Plus/Minus/Times/And/Or/StringJoin/Alternatives
    Expr::BinaryOp { op, left, right }
      if !matches!(
        op,
        crate::syntax::BinaryOperator::Power
          | crate::syntax::BinaryOperator::Divide
      ) =>
    {
      let (op_str, spaced) = match op {
        crate::syntax::BinaryOperator::Plus => ("+", true),
        crate::syntax::BinaryOperator::Minus => ("-", true),
        crate::syntax::BinaryOperator::Times => (" ", false),
        crate::syntax::BinaryOperator::And => ("&&", true),
        crate::syntax::BinaryOperator::Or => ("||", true),
        crate::syntax::BinaryOperator::StringJoin => ("<>", false),
        crate::syntax::BinaryOperator::Alternatives => ("|", true),
        crate::syntax::BinaryOperator::Power
        | crate::syntax::BinaryOperator::Divide => unreachable!(),
      };
      let sep = if spaced {
        op_str.to_string()
      } else {
        op_str.to_string()
      };
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(vec![
          expr_to_box_form(left),
          Expr::String(sep),
          expr_to_box_form(right),
        ])],
      }
    }
    // Comparison: a == b < c → RowBox[{box(a), " == ", box(b), " < ", box(c)}]
    Expr::Comparison {
      operands,
      operators,
    } => {
      let mut parts = Vec::new();
      parts.push(expr_to_box_form(&operands[0]));
      for (i, op) in operators.iter().enumerate() {
        let op_str = match op {
          crate::syntax::ComparisonOp::Equal => "==",
          crate::syntax::ComparisonOp::NotEqual => "!=",
          crate::syntax::ComparisonOp::Less => "<",
          crate::syntax::ComparisonOp::LessEqual => "<=",
          crate::syntax::ComparisonOp::Greater => ">",
          crate::syntax::ComparisonOp::GreaterEqual => ">=",
          crate::syntax::ComparisonOp::SameQ => "===",
          crate::syntax::ComparisonOp::UnsameQ => "=!=",
        };
        parts.push(Expr::String(op_str.to_string()));
        parts.push(expr_to_box_form(&operands[i + 1]));
      }
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts)],
      }
    }
    // Rule: pattern -> replacement
    Expr::Rule {
      pattern,
      replacement,
    } => {
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(vec![
          expr_to_box_form(pattern),
          Expr::String("\u{f522}".to_string()), // Mathematica's Rule arrow
          expr_to_box_form(replacement),
        ])],
      }
    }
    // RuleDelayed: pattern :> replacement
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(vec![
          expr_to_box_form(pattern),
          Expr::String("\u{f51f}".to_string()), // Mathematica's RuleDelayed arrow
          expr_to_box_form(replacement),
        ])],
      }
    }
    // Association: <|k1 -> v1, ...|>
    Expr::Association(items) => {
      let mut parts = Vec::new();
      parts.push(Expr::String("<|".to_string()));
      for (i, (k, v)) in items.iter().enumerate() {
        if i > 0 {
          parts.push(Expr::String(",".to_string()));
        }
        parts.push(Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(vec![
            expr_to_box_form(k),
            Expr::String("\u{f522}".to_string()),
            expr_to_box_form(v),
          ])],
        });
      }
      parts.push(Expr::String("|>".to_string()));
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts)],
      }
    }
    // CompoundExpr: e1; e2; e3
    Expr::CompoundExpr(exprs) => {
      let mut parts = Vec::new();
      for (i, e) in exprs.iter().enumerate() {
        if i > 0 {
          parts.push(Expr::String(";".to_string()));
        }
        parts.push(expr_to_box_form(e));
      }
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts)],
      }
    }
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2 => {
      // Plus[a, b, c] → RowBox[{box(a), "+", box(b), "+", box(c)}]
      // Handle negative terms: Times[-1, x] → "x", "-", box(x)
      let mut parts = Vec::new();
      for (i, arg) in args.iter().enumerate() {
        if i > 0 {
          // Check for negative term: Times[-1, x]
          if let Expr::FunctionCall { name: tn, args: ta } = arg
            && tn == "Times"
            && ta.len() == 2
            && matches!(&ta[0], Expr::Integer(-1))
          {
            parts.push(Expr::String("-".to_string()));
            parts.push(expr_to_box_form(&ta[1]));
            continue;
          }
          // Check for negative coefficient: Times[-n, x]
          if let Expr::FunctionCall { name: tn, args: ta } = arg
            && tn == "Times"
            && ta.len() >= 2
            && matches!(&ta[0], Expr::Integer(n) if *n < 0)
          {
            parts.push(Expr::String("-".to_string()));
            // Negate the coefficient and box the positive term
            let mut pos_args = ta.clone();
            if let Expr::Integer(n) = &ta[0] {
              pos_args[0] = Expr::Integer(-n);
            }
            let pos_term = Expr::FunctionCall {
              name: "Times".to_string(),
              args: pos_args,
            };
            parts.push(expr_to_box_form(&pos_term));
            continue;
          }
          parts.push(Expr::String("+".to_string()));
        }
        parts.push(expr_to_box_form(arg));
      }
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts)],
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() >= 2
        && matches!(&args[0], Expr::Integer(-1)) =>
    {
      // Times[-1, x] → RowBox[{"-", box(x)}]
      // Times[-1, a, b, c, ...] → RowBox[{"-", RowBox[{"(", a " " b " " c, ")"}]}]
      // so the leading minus applies to the whole product without showing
      // a literal "1" coefficient.
      if args.len() == 2 {
        return Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(vec![
            Expr::String("-".to_string()),
            box_with_paren_if_needed(&args[1]),
          ])],
        };
      }
      let rest = Expr::FunctionCall {
        name: "Times".to_string(),
        args: args[1..].to_vec(),
      };
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(vec![
          Expr::String("-".to_string()),
          paren_box(expr_to_box_form(&rest)),
        ])],
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(n) if *n < 0) =>
    {
      // Times[-n, x] → RowBox[{"-", RowBox[{n, " ", box(x)}]}]
      if let Expr::Integer(n) = &args[0] {
        let pos_n = Expr::Integer(-n);
        Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(vec![
            Expr::String("-".to_string()),
            Expr::FunctionCall {
              name: "RowBox".to_string(),
              args: vec![Expr::List(vec![
                expr_to_box_form(&pos_n),
                Expr::String(" ".to_string()),
                expr_to_box_form(&args[1]),
              ])],
            },
          ])],
        }
      } else {
        box_as_output_string(expr)
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      // Check for fraction form: Times[..., Power[den, -1]]
      let (num, den) =
        crate::functions::polynomial_ast::together::extract_num_den(expr);
      if !matches!(&den, Expr::Integer(1)) {
        return Expr::FunctionCall {
          name: "FractionBox".to_string(),
          args: vec![expr_to_box_form(&num), expr_to_box_form(&den)],
        };
      }
      // General Times: RowBox[{a, " ", b, " ", ...}]
      // Additive sub-expressions (e.g. Plus[1, x]) are wrapped in parens so
      // that `2*(1+x)` renders as `2 (1+x)` rather than `2 1+x`.
      let mut parts = Vec::new();
      for (i, arg) in args.iter().enumerate() {
        if i > 0 {
          parts.push(Expr::String(" ".to_string()));
        }
        parts.push(box_with_paren_if_needed(arg));
      }
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts)],
      }
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Expr::FunctionCall { name: rn, args: ra } = &args[1]
        && rn == "Rational"
        && ra.len() == 2
      {
        // Power[base, 1/2] → SqrtBox[box(base)]
        if matches!(&ra[0], Expr::Integer(1))
          && matches!(&ra[1], Expr::Integer(2))
        {
          return Expr::FunctionCall {
            name: "SqrtBox".to_string(),
            args: vec![expr_to_box_form(&args[0])],
          };
        }
        // Power[base, -1/2] → FractionBox["1", SqrtBox[box(base)]]
        if matches!(&ra[0], Expr::Integer(-1))
          && matches!(&ra[1], Expr::Integer(2))
        {
          return Expr::FunctionCall {
            name: "FractionBox".to_string(),
            args: vec![
              Expr::String("1".to_string()),
              Expr::FunctionCall {
                name: "SqrtBox".to_string(),
                args: vec![expr_to_box_form(&args[0])],
              },
            ],
          };
        }
      }
      // Power[Subscript[x, sub], exp] → SubsuperscriptBox[box(x), box(sub), box(exp)]
      if let Expr::FunctionCall { name: bn, args: ba } = &args[0]
        && bn == "Subscript"
        && ba.len() == 2
      {
        return Expr::FunctionCall {
          name: "SubsuperscriptBox".to_string(),
          args: vec![
            expr_to_box_form(&ba[0]),
            expr_to_box_form(&ba[1]),
            expr_to_box_form(&args[1]),
          ],
        };
      }
      // General power: SuperscriptBox[box(base), box(exp)]
      // The base is parenthesized when it's additive (e.g. (1+x)^2) so the
      // result reads unambiguously instead of 1+x² looking like 1 + x².
      Expr::FunctionCall {
        name: "SuperscriptBox".to_string(),
        args: vec![
          box_with_paren_if_needed(&args[0]),
          expr_to_box_form(&args[1]),
        ],
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Subscript" && args.len() == 2 =>
    {
      Expr::FunctionCall {
        name: "SubscriptBox".to_string(),
        args: vec![expr_to_box_form(&args[0]), expr_to_box_form(&args[1])],
      }
    }
    expr if is_sqrt(expr).is_some() => {
      let sqrt_arg = is_sqrt(expr).unwrap();
      Expr::FunctionCall {
        name: "SqrtBox".to_string(),
        args: vec![expr_to_box_form(sqrt_arg)],
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      // Rational[n, d] → FractionBox[n, d]
      Expr::FunctionCall {
        name: "FractionBox".to_string(),
        args: vec![expr_to_box_form(&args[0]), expr_to_box_form(&args[1])],
      }
    }
    // BinaryOp::Divide → FractionBox
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => Expr::FunctionCall {
      name: "FractionBox".to_string(),
      args: vec![expr_to_box_form(left), expr_to_box_form(right)],
    },
    // BinaryOp::Power with rational exponents
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::FunctionCall { name: rn, args: ra } = right.as_ref()
        && rn == "Rational"
        && ra.len() == 2
      {
        if matches!(&ra[0], Expr::Integer(1))
          && matches!(&ra[1], Expr::Integer(2))
        {
          return Expr::FunctionCall {
            name: "SqrtBox".to_string(),
            args: vec![expr_to_box_form(left)],
          };
        }
        if matches!(&ra[0], Expr::Integer(-1))
          && matches!(&ra[1], Expr::Integer(2))
        {
          return Expr::FunctionCall {
            name: "FractionBox".to_string(),
            args: vec![
              Expr::String("1".to_string()),
              Expr::FunctionCall {
                name: "SqrtBox".to_string(),
                args: vec![expr_to_box_form(left)],
              },
            ],
          };
        }
      }
      // BinaryOp::Power with Subscript base → SubsuperscriptBox
      if let Expr::FunctionCall { name: bn, args: ba } = left.as_ref()
        && bn == "Subscript"
        && ba.len() == 2
      {
        return Expr::FunctionCall {
          name: "SubsuperscriptBox".to_string(),
          args: vec![
            expr_to_box_form(&ba[0]),
            expr_to_box_form(&ba[1]),
            expr_to_box_form(right),
          ],
        };
      }
      Expr::FunctionCall {
        name: "SuperscriptBox".to_string(),
        args: vec![box_with_paren_if_needed(left), expr_to_box_form(right)],
      }
    }
    // List → RowBox[{"{", RowBox[{elem, ",", elem, ...}], "}"}]
    Expr::List(items) => {
      let mut parts = Vec::new();
      parts.push(Expr::String("{".to_string()));
      if !items.is_empty() {
        if items.len() == 1 {
          parts.push(expr_to_box_form(&items[0]));
        } else {
          let mut inner = Vec::new();
          for (i, item) in items.iter().enumerate() {
            if i > 0 {
              inner.push(Expr::String(",".to_string()));
            }
            inner.push(expr_to_box_form(item));
          }
          parts.push(Expr::FunctionCall {
            name: "RowBox".to_string(),
            args: vec![Expr::List(inner)],
          });
        }
      }
      parts.push(Expr::String("}".to_string()));
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts)],
      }
    }
    // Quantity[magnitude, unit] → RowBox[{box(magnitude), " ", unit-boxes}]
    Expr::FunctionCall { name, args }
      if name == "Quantity" && args.len() == 2 =>
    {
      let unit_boxes = unit_to_box_form(&args[1], &args[0]);
      let mut parts =
        vec![expr_to_box_form(&args[0]), Expr::String(" ".to_string())];
      parts.push(unit_boxes);
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts)],
      }
    }
    // FullForm[expr] → render in canonical notation as a single string
    Expr::FunctionCall { name, args }
      if name == "FullForm" && args.len() == 1 =>
    {
      let full = crate::functions::predicate_ast::expr_to_full_form(&args[0]);
      Expr::String(full)
    }
    // Style[content, ...] → just the content
    Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
    {
      expr_to_box_form(&args[0])
    }
    // HoldForm[expr] → just the content
    Expr::FunctionCall { name, args }
      if name == "HoldForm" && args.len() == 1 =>
    {
      expr_to_box_form(&args[0])
    }
    // CForm/TeXForm/FortranForm → converted text as a string
    Expr::FunctionCall { name, args }
      if (name == "CForm" || name == "TeXForm" || name == "FortranForm")
        && args.len() == 1 =>
    {
      let text = match name.as_str() {
        "CForm" => crate::functions::string_ast::expr_to_c(&args[0]),
        "TeXForm" => crate::functions::string_ast::expr_to_tex(&args[0]),
        "FortranForm" => {
          crate::functions::string_ast::expr_to_fortran(&args[0])
        }
        _ => unreachable!(),
      };
      Expr::String(text)
    }
    // Graphics[...] / Graphics3D[...] get dedicated box wrappers matching
    // Wolfram: Head[ToBoxes[Graphics[...]]] → GraphicsBox. Handles both the
    // unevaluated FunctionCall and the evaluated Expr::Graphics variants.
    // Graphics primitives like Disk/Sphere get converted to box heads too.
    Expr::FunctionCall { name, args }
      if name == "Graphics" || name == "Graphics3D" =>
    {
      let box_head = if name == "Graphics" {
        "GraphicsBox"
      } else {
        "Graphics3DBox"
      };
      Expr::FunctionCall {
        name: box_head.to_string(),
        args: args.iter().map(to_graphics_boxes).collect(),
      }
    }
    Expr::Graphics { is_3d, .. } => {
      let box_head = if *is_3d {
        "Graphics3DBox"
      } else {
        "GraphicsBox"
      };
      Expr::FunctionCall {
        name: box_head.to_string(),
        args: vec![],
      }
    }
    // General function call f[x, y] → RowBox[{f, "[", RowBox[{x, ",", y}], "]"}]
    Expr::FunctionCall { name, args } => {
      let mut parts = Vec::new();
      parts.push(Expr::String(name.clone()));
      parts.push(Expr::String("[".to_string()));
      if !args.is_empty() {
        if args.len() == 1 {
          parts.push(expr_to_box_form(&args[0]));
        } else {
          let mut inner = Vec::new();
          for (i, arg) in args.iter().enumerate() {
            if i > 0 {
              inner.push(Expr::String(",".to_string()));
            }
            inner.push(expr_to_box_form(arg));
          }
          parts.push(Expr::FunctionCall {
            name: "RowBox".to_string(),
            args: vec![Expr::List(inner)],
          });
        }
      }
      parts.push(Expr::String("]".to_string()));
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts)],
      }
    }
    // Default: use the string representation
    _ => box_as_output_string(expr),
  }
}

/// Convert a Quantity unit expression to box form with proper abbreviations.
/// Uses `unit_to_abbreviation` for known units and handles compound units
/// (division, multiplication, powers).
fn unit_to_box_form(unit: &Expr, magnitude: &Expr) -> Expr {
  use crate::functions::quantity_ast::unit_to_abbreviation;
  use crate::syntax::BinaryOperator;

  // Helper: abbreviate a single unit identifier
  fn abbrev(s: &str, mag: &Expr) -> Expr {
    let abbr = unit_to_abbreviation(s).unwrap_or(s);
    let abbr = crate::syntax::singularize_unit_if_one(mag, abbr);
    Expr::String(abbr)
  }

  // Handle Power in both BinaryOp and FunctionCall form
  if let Some((base, exp)) = crate::functions::graphics::as_power_pub(unit) {
    let base_box = unit_to_box_form_inner(base);
    let exp_box = expr_to_box_form(exp);
    return Expr::FunctionCall {
      name: "SuperscriptBox".to_string(),
      args: vec![base_box, exp_box],
    };
  }

  match unit {
    Expr::Identifier(s) | Expr::String(s) => abbrev(s, magnitude),
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => Expr::FunctionCall {
      name: "RowBox".to_string(),
      args: vec![Expr::List(vec![
        unit_to_box_form_inner(left),
        Expr::String("/".to_string()),
        unit_to_box_form_inner(right),
      ])],
    },
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => Expr::FunctionCall {
      name: "RowBox".to_string(),
      args: vec![Expr::List(vec![
        unit_to_box_form_inner(left),
        Expr::String("\u{22c5}".to_string()),
        unit_to_box_form_inner(right),
      ])],
    },
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Check for fraction form: Times[..., Power[den, -n]]
      let mut numer_parts: Vec<Expr> = Vec::new();
      let mut denom_parts: Vec<Expr> = Vec::new();
      for a in args {
        if let Some((base, neg_exp)) = crate::syntax::extract_neg_power_info(a)
        {
          let base_box = unit_to_box_form_inner(base);
          if neg_exp == -1 {
            denom_parts.push(base_box);
          } else {
            denom_parts.push(Expr::FunctionCall {
              name: "SuperscriptBox".to_string(),
              args: vec![base_box, Expr::String((-neg_exp).to_string())],
            });
          }
        } else {
          numer_parts.push(unit_to_box_form_inner(a));
        }
      }
      if denom_parts.is_empty() {
        join_with_dot(numer_parts)
      } else {
        let numer = if numer_parts.is_empty() {
          Expr::String("1".to_string())
        } else {
          join_with_dot(numer_parts)
        };
        let denom = join_with_dot(denom_parts);
        Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(vec![
            numer,
            Expr::String("/".to_string()),
            denom,
          ])],
        }
      }
    }
    _ => expr_to_box_form(unit),
  }
}

/// Like `unit_to_box_form` but without singularization (for compound sub-units).
fn unit_to_box_form_inner(unit: &Expr) -> Expr {
  use crate::functions::quantity_ast::unit_to_abbreviation;
  use crate::syntax::BinaryOperator;

  if let Some((base, exp)) = crate::functions::graphics::as_power_pub(unit) {
    let base_box = unit_to_box_form_inner(base);
    let exp_box = expr_to_box_form(exp);
    return Expr::FunctionCall {
      name: "SuperscriptBox".to_string(),
      args: vec![base_box, exp_box],
    };
  }

  match unit {
    Expr::Identifier(s) | Expr::String(s) => {
      let abbr = unit_to_abbreviation(s).unwrap_or(s);
      Expr::String(abbr.to_string())
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => Expr::FunctionCall {
      name: "RowBox".to_string(),
      args: vec![Expr::List(vec![
        unit_to_box_form_inner(left),
        Expr::String("/".to_string()),
        unit_to_box_form_inner(right),
      ])],
    },
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => Expr::FunctionCall {
      name: "RowBox".to_string(),
      args: vec![Expr::List(vec![
        unit_to_box_form_inner(left),
        Expr::String("\u{22c5}".to_string()),
        unit_to_box_form_inner(right),
      ])],
    },
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut numer_parts: Vec<Expr> = Vec::new();
      let mut denom_parts: Vec<Expr> = Vec::new();
      for a in args {
        if let Some((base, neg_exp)) = crate::syntax::extract_neg_power_info(a)
        {
          let base_box = unit_to_box_form_inner(base);
          if neg_exp == -1 {
            denom_parts.push(base_box);
          } else {
            denom_parts.push(Expr::FunctionCall {
              name: "SuperscriptBox".to_string(),
              args: vec![base_box, Expr::String((-neg_exp).to_string())],
            });
          }
        } else {
          numer_parts.push(unit_to_box_form_inner(a));
        }
      }
      if denom_parts.is_empty() {
        join_with_dot(numer_parts)
      } else {
        let numer = if numer_parts.is_empty() {
          Expr::String("1".to_string())
        } else {
          join_with_dot(numer_parts)
        };
        let denom = join_with_dot(denom_parts);
        Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(vec![
            numer,
            Expr::String("/".to_string()),
            denom,
          ])],
        }
      }
    }
    _ => expr_to_box_form(unit),
  }
}

/// Join box expressions with middle-dot separator.
fn join_with_dot(parts: Vec<Expr>) -> Expr {
  if parts.len() == 1 {
    return parts.into_iter().next().unwrap();
  }
  let mut result = Vec::new();
  for (i, p) in parts.into_iter().enumerate() {
    if i > 0 {
      result.push(Expr::String("\u{22c5}".to_string()));
    }
    result.push(p);
  }
  Expr::FunctionCall {
    name: "RowBox".to_string(),
    args: vec![Expr::List(result)],
  }
}

/// Convert an expression to a string box (for expressions we don't have explicit box forms for)
fn box_as_output_string(expr: &Expr) -> Expr {
  let s = crate::syntax::expr_to_output(expr);
  // If it's a simple identifier-like string, return as Identifier
  if s
    .chars()
    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
    && !s.is_empty()
  {
    Expr::Identifier(s)
  } else {
    Expr::String(s)
  }
}

/// Extract function names from an Activate filter (second argument).
/// Handles: single identifier, or Alternatives (f1 | f2 | ...).
fn extract_activate_filter(filter: &Expr) -> Vec<String> {
  match filter {
    Expr::Identifier(name) => vec![name.clone()],
    Expr::FunctionCall { name, args } if name == "Alternatives" => args
      .iter()
      .filter_map(|a| {
        if let Expr::Identifier(n) = a {
          Some(n.clone())
        } else {
          None
        }
      })
      .collect(),
    _ => vec![],
  }
}

/// Recursively replace Inactive[f][args...] with f[args...] in an expression.
/// If `filter` is Some, only activate the specified functions.
fn activate_expr(expr: &Expr, filter: &Option<Vec<String>>) -> Expr {
  match expr {
    // CurriedCall where func is Inactive[f] → f[args...]
    Expr::CurriedCall { func, args } => {
      if let Expr::FunctionCall {
        name,
        args: inactive_args,
      } = func.as_ref()
        && name == "Inactive"
        && inactive_args.len() == 1
        && let Expr::Identifier(func_name) = &inactive_args[0]
      {
        // Check filter
        let should_activate = match filter {
          Some(allowed) => allowed.contains(func_name),
          None => true,
        };
        if should_activate {
          // Activate: replace with f[args...], recursing into args
          let activated_args: Vec<Expr> =
            args.iter().map(|a| activate_expr(a, filter)).collect();
          return Expr::FunctionCall {
            name: func_name.clone(),
            args: activated_args,
          };
        }
      }
      // Recurse into both func and args
      Expr::CurriedCall {
        func: Box::new(activate_expr(func, filter)),
        args: args.iter().map(|a| activate_expr(a, filter)).collect(),
      }
    }
    // Recurse into function call args
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(|a| activate_expr(a, filter)).collect(),
    },
    // Recurse into lists
    Expr::List(items) => {
      Expr::List(items.iter().map(|a| activate_expr(a, filter)).collect())
    }
    // Recurse into binary ops
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(activate_expr(left, filter)),
      right: Box::new(activate_expr(right, filter)),
    },
    // Recurse into unary ops
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(activate_expr(operand, filter)),
    },
    // Atoms: return as-is
    _ => expr.clone(),
  }
}

// ─── Area ──────────────────────────────────────────────────────────────

/// Compute the area of a geometric region.
fn compute_area(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      // Disk[] = Pi, Disk[center, r] = Pi*r^2, Disk[center, {a, b}] = Pi*a*b
      "Disk" => {
        if args.is_empty() || (args.len() == 1) {
          // Unit disk
          Ok(Expr::Constant("Pi".to_string()))
        } else if args.len() == 2 {
          match &args[1] {
            Expr::List(radii) if radii.len() == 2 => {
              // Elliptical disk: Pi * a * b
              let area = Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  Expr::Constant("Pi".to_string()),
                  radii[0].clone(),
                  radii[1].clone(),
                ],
              };
              crate::evaluator::evaluate_expr_to_expr(&area)
            }
            r => {
              // Circular disk: Pi * r^2
              let area = Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  Expr::Constant("Pi".to_string()),
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![r.clone(), Expr::Integer(2)],
                  },
                ],
              };
              crate::evaluator::evaluate_expr_to_expr(&area)
            }
          }
        } else {
          Ok(Expr::FunctionCall {
            name: "Area".to_string(),
            args: vec![expr.clone()],
          })
        }
      }
      // Rectangle[] = 1, Rectangle[{x1,y1}, {x2,y2}] = |x2-x1| * |y2-y1|
      "Rectangle" => {
        if args.is_empty() {
          Ok(Expr::Integer(1))
        } else if args.len() == 2 {
          if let (Expr::List(p1), Expr::List(p2)) = (&args[0], &args[1])
            && p1.len() == 2
            && p2.len() == 2
          {
            let width = Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  p2[0].clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), p1[0].clone()],
                  },
                ],
              }],
            };
            let height = Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  p2[1].clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), p1[1].clone()],
                  },
                ],
              }],
            };
            let area = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![width, height],
            };
            return crate::evaluator::evaluate_expr_to_expr(&area);
          }
          Ok(Expr::FunctionCall {
            name: "Area".to_string(),
            args: vec![expr.clone()],
          })
        } else {
          Ok(Expr::FunctionCall {
            name: "Area".to_string(),
            args: vec![expr.clone()],
          })
        }
      }
      // Triangle[{{x1,y1},{x2,y2},{x3,y3}}] = |det| / 2
      "Triangle" => {
        if args.len() == 1
          && let Expr::List(pts) = &args[0]
          && pts.len() == 3
          && let (Expr::List(p1), Expr::List(p2), Expr::List(p3)) =
            (&pts[0], &pts[1], &pts[2])
          && p1.len() == 2
          && p2.len() == 2
          && p3.len() == 2
        {
          // Area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)| / 2
          let area_expr = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(1), Expr::Integer(2)],
              },
              Expr::FunctionCall {
                name: "Abs".to_string(),
                args: vec![Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![
                    Expr::FunctionCall {
                      name: "Times".to_string(),
                      args: vec![
                        p1[0].clone(),
                        Expr::FunctionCall {
                          name: "Plus".to_string(),
                          args: vec![
                            p2[1].clone(),
                            Expr::FunctionCall {
                              name: "Times".to_string(),
                              args: vec![Expr::Integer(-1), p3[1].clone()],
                            },
                          ],
                        },
                      ],
                    },
                    Expr::FunctionCall {
                      name: "Times".to_string(),
                      args: vec![
                        p2[0].clone(),
                        Expr::FunctionCall {
                          name: "Plus".to_string(),
                          args: vec![
                            p3[1].clone(),
                            Expr::FunctionCall {
                              name: "Times".to_string(),
                              args: vec![Expr::Integer(-1), p1[1].clone()],
                            },
                          ],
                        },
                      ],
                    },
                    Expr::FunctionCall {
                      name: "Times".to_string(),
                      args: vec![
                        p3[0].clone(),
                        Expr::FunctionCall {
                          name: "Plus".to_string(),
                          args: vec![
                            p1[1].clone(),
                            Expr::FunctionCall {
                              name: "Times".to_string(),
                              args: vec![Expr::Integer(-1), p2[1].clone()],
                            },
                          ],
                        },
                      ],
                    },
                  ],
                }],
              },
            ],
          };
          return crate::evaluator::evaluate_expr_to_expr(&area_expr);
        }
        Ok(Expr::FunctionCall {
          name: "Area".to_string(),
          args: vec![expr.clone()],
        })
      }
      // Polygon[{{x1,y1},{x2,y2},...}] — Shoelace formula
      "Polygon" => {
        if args.len() == 1
          && let Expr::List(pts) = &args[0]
          && pts.len() >= 3
        {
          return compute_polygon_area(pts);
        }
        Ok(Expr::FunctionCall {
          name: "Area".to_string(),
          args: vec![expr.clone()],
        })
      }
      // Circle has no area (it's 1D)
      "Circle" => Ok(Expr::Identifier("Undefined".to_string())),
      _ => Ok(Expr::FunctionCall {
        name: "Area".to_string(),
        args: vec![expr.clone()],
      }),
    },
    _ => Ok(Expr::FunctionCall {
      name: "Area".to_string(),
      args: vec![expr.clone()],
    }),
  }
}

/// Compute the area of a polygon using the Shoelace formula.
fn compute_polygon_area(pts: &[Expr]) -> Result<Expr, InterpreterError> {
  // Extract 2D coordinates
  let coords: Vec<(&Expr, &Expr)> = pts
    .iter()
    .filter_map(|p| {
      if let Expr::List(xy) = p
        && xy.len() == 2
      {
        return Some((&xy[0], &xy[1]));
      }
      None
    })
    .collect();

  if coords.len() != pts.len() {
    return Ok(Expr::FunctionCall {
      name: "Area".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Polygon".to_string(),
        args: vec![Expr::List(pts.to_vec())],
      }],
    });
  }

  let n = coords.len();
  // Shoelace: 2*A = sum_{i=0}^{n-1} (x_i * y_{i+1} - x_{i+1} * y_i)
  let mut sum_terms = Vec::new();
  for i in 0..n {
    let j = (i + 1) % n;
    // x_i * y_j
    sum_terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![coords[i].0.clone(), coords[j].1.clone()],
    });
    // -x_j * y_i
    sum_terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), coords[j].0.clone(), coords[i].1.clone()],
    });
  }

  let area_expr = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)],
      },
      Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![Expr::FunctionCall {
          name: "Plus".to_string(),
          args: sum_terms,
        }],
      },
    ],
  };

  crate::evaluator::evaluate_expr_to_expr(&area_expr)
}

/// Compute the centroid of a geometric region.
fn compute_region_centroid(expr: &Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionCentroid".to_string(),
      args: vec![expr.clone()],
    })
  };
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      // Point[{x, y, ...}] — centroid is the point itself
      "Point" if args.len() == 1 => {
        if let Expr::List(_) = &args[0] {
          Ok(args[0].clone())
        } else {
          unevaluated()
        }
      }
      // Disk[{x, y}, r] or Disk[] — centroid is the center
      "Disk" => {
        if args.is_empty() {
          // Unit disk centered at origin
          Ok(Expr::List(vec![Expr::Integer(0), Expr::Integer(0)]))
        } else {
          // Center is the first argument
          Ok(args[0].clone())
        }
      }
      // Ball[{x, y, z}, r] or Ball[] — centroid is the center
      "Ball" => {
        if args.is_empty() {
          Ok(Expr::List(vec![
            Expr::Integer(0),
            Expr::Integer(0),
            Expr::Integer(0),
          ]))
        } else {
          Ok(args[0].clone())
        }
      }
      // Circle[{x, y}, r] — centroid is the center
      "Circle" => {
        if args.is_empty() {
          Ok(Expr::List(vec![Expr::Integer(0), Expr::Integer(0)]))
        } else {
          Ok(args[0].clone())
        }
      }
      // Rectangle[{x1, y1}, {x2, y2}] — centroid is midpoint
      "Rectangle" => {
        if args.is_empty() {
          // Rectangle[] = Rectangle[{0,0},{1,1}]
          Ok(Expr::List(vec![
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(1), Expr::Integer(2)],
            },
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(1), Expr::Integer(2)],
            },
          ]))
        } else if args.len() == 1 {
          // Rectangle[{x1,y1}] = Rectangle[{x1,y1},{x1+1,y1+1}]
          if let Expr::List(p1) = &args[0]
            && p1.len() == 2
          {
            let cx = Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                p1[0].clone(),
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(2)],
                },
              ],
            };
            let cy = Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                p1[1].clone(),
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(2)],
                },
              ],
            };
            let result = Expr::List(vec![cx, cy]);
            return crate::evaluator::evaluate_expr_to_expr(&result);
          }
          unevaluated()
        } else if args.len() == 2 {
          if let (Expr::List(p1), Expr::List(p2)) = (&args[0], &args[1])
            && p1.len() == 2
            && p2.len() == 2
          {
            let cx = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(2)],
                },
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![p1[0].clone(), p2[0].clone()],
                },
              ],
            };
            let cy = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(2)],
                },
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![p1[1].clone(), p2[1].clone()],
                },
              ],
            };
            let result = Expr::List(vec![cx, cy]);
            return crate::evaluator::evaluate_expr_to_expr(&result);
          }
          unevaluated()
        } else {
          unevaluated()
        }
      }
      // Triangle[{{x1,y1},{x2,y2},{x3,y3}}] — centroid is average of vertices
      "Triangle" => {
        if args.len() == 1
          && let Expr::List(pts) = &args[0]
          && pts.len() == 3
        {
          return compute_triangle_centroid(pts);
        }
        unevaluated()
      }
      // Polygon[{{x1,y1},{x2,y2},...}] — centroid via shoelace-derived formula
      "Polygon" => {
        if args.len() == 1
          && let Expr::List(pts) = &args[0]
          && pts.len() >= 3
        {
          return compute_polygon_centroid(pts);
        }
        unevaluated()
      }
      // Line[{{x1,y1},{x2,y2},...}] — centroid is weighted average by segment length
      "Line" => {
        if args.len() == 1
          && let Expr::List(pts) = &args[0]
          && pts.len() >= 2
        {
          return compute_line_centroid(pts);
        }
        unevaluated()
      }
      _ => unevaluated(),
    },
    _ => unevaluated(),
  }
}

/// Compute centroid of a triangle: average of the 3 vertices.
fn compute_triangle_centroid(pts: &[Expr]) -> Result<Expr, InterpreterError> {
  // Get the dimension from the first point
  let coords: Vec<&Vec<Expr>> = pts
    .iter()
    .filter_map(|p| {
      if let Expr::List(xy) = p {
        Some(xy)
      } else {
        None
      }
    })
    .collect();
  if coords.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "RegionCentroid".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Triangle".to_string(),
        args: vec![Expr::List(pts.to_vec())],
      }],
    });
  }
  let dim = coords[0].len();
  if !coords.iter().all(|c| c.len() == dim) {
    return Ok(Expr::FunctionCall {
      name: "RegionCentroid".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Triangle".to_string(),
        args: vec![Expr::List(pts.to_vec())],
      }],
    });
  }
  let mut result = Vec::new();
  for d in 0..dim {
    let avg = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(3)],
        },
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            coords[0][d].clone(),
            coords[1][d].clone(),
            coords[2][d].clone(),
          ],
        },
      ],
    };
    result.push(avg);
  }
  crate::evaluator::evaluate_expr_to_expr(&Expr::List(result))
}

/// Compute centroid of a polygon using the standard formula:
/// Cx = 1/(6A) * sum((xi + xi+1)(xi*yi+1 - xi+1*yi))
/// Cy = 1/(6A) * sum((yi + yi+1)(xi*yi+1 - xi+1*yi))
/// where A is the signed area from the shoelace formula.
fn compute_polygon_centroid(pts: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionCentroid".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Polygon".to_string(),
        args: vec![Expr::List(pts.to_vec())],
      }],
    })
  };

  let coords: Vec<(&Expr, &Expr)> = pts
    .iter()
    .filter_map(|p| {
      if let Expr::List(xy) = p
        && xy.len() == 2
      {
        return Some((&xy[0], &xy[1]));
      }
      None
    })
    .collect();

  if coords.len() != pts.len() {
    return unevaluated();
  }

  let n = coords.len();

  // Build signed area: 2A = sum(xi*yi+1 - xi+1*yi)
  let mut area_terms = Vec::new();
  let mut cx_terms = Vec::new();
  let mut cy_terms = Vec::new();

  for i in 0..n {
    let j = (i + 1) % n;
    // cross = xi*yj - xj*yi
    let cross = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![coords[i].0.clone(), coords[j].1.clone()],
        },
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(-1),
            coords[j].0.clone(),
            coords[i].1.clone(),
          ],
        },
      ],
    };

    area_terms.push(cross.clone());

    // (xi + xj) * cross
    cx_terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![coords[i].0.clone(), coords[j].0.clone()],
        },
        cross.clone(),
      ],
    });

    // (yi + yj) * cross
    cy_terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![coords[i].1.clone(), coords[j].1.clone()],
        },
        cross,
      ],
    });
  }

  // signed_area_2 = sum of area_terms
  let signed_area_2 = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: area_terms,
  };

  // 1/(6A) = 1/(3 * signed_area_2)
  let inv_6a = Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(3), signed_area_2],
      },
      Expr::Integer(-1),
    ],
  };

  let cx = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      inv_6a.clone(),
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: cx_terms,
      },
    ],
  };

  let cy = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      inv_6a,
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: cy_terms,
      },
    ],
  };

  crate::evaluator::evaluate_expr_to_expr(&Expr::List(vec![cx, cy]))
}

/// Compute centroid of a line (polyline): weighted average of segment midpoints.
fn compute_line_centroid(pts: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionCentroid".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Line".to_string(),
        args: vec![Expr::List(pts.to_vec())],
      }],
    })
  };

  // Extract coordinates
  let coords: Vec<&Vec<Expr>> = pts
    .iter()
    .filter_map(|p| {
      if let Expr::List(xy) = p {
        Some(xy)
      } else {
        None
      }
    })
    .collect();

  if coords.len() != pts.len() || coords.is_empty() {
    return unevaluated();
  }

  let dim = coords[0].len();
  if !coords.iter().all(|c| c.len() == dim) {
    return unevaluated();
  }

  // For a 2-point line, centroid is simply the midpoint
  if coords.len() == 2 {
    let mut result = Vec::new();
    for d in 0..dim {
      result.push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)],
          },
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![coords[0][d].clone(), coords[1][d].clone()],
          },
        ],
      });
    }
    return crate::evaluator::evaluate_expr_to_expr(&Expr::List(result));
  }

  // For multi-segment lines, weight midpoints by segment length
  // Build symbolic expressions for: sum(length_i * midpoint_i) / sum(length_i)
  let mut length_terms = Vec::new();
  let mut weighted_midpoints: Vec<Vec<Expr>> = vec![Vec::new(); dim];

  for i in 0..coords.len() - 1 {
    let j = i + 1;
    // length = Sqrt[sum of (xj_d - xi_d)^2]
    let mut sq_terms = Vec::new();
    for d in 0..dim {
      sq_terms.push(Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              coords[j][d].clone(),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), coords[i][d].clone()],
              },
            ],
          },
          Expr::Integer(2),
        ],
      });
    }
    let seg_length = make_sqrt(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: sq_terms,
    });

    length_terms.push(seg_length.clone());

    for d in 0..dim {
      // midpoint_d * length
      let mid = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)],
          },
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![coords[i][d].clone(), coords[j][d].clone()],
          },
          seg_length.clone(),
        ],
      };
      weighted_midpoints[d].push(mid);
    }
  }

  let total_length = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: length_terms,
  };

  let mut result = Vec::new();
  for d in 0..dim {
    result.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![total_length.clone(), Expr::Integer(-1)],
        },
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: weighted_midpoints[d].clone(),
        },
      ],
    });
  }

  crate::evaluator::evaluate_expr_to_expr(&Expr::List(result))
}

// ─── FindSequenceFunction ──────────────────────────────────────────────

/// Find a formula for an integer sequence.
/// Tries: polynomial fit, exponential fit, factorial detection.
fn find_sequence_function(
  data_expr: &Expr,
  var_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match data_expr {
    Expr::List(items) if items.len() >= 2 => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "FindSequenceFunction: first argument must be a list with at least 2 elements".into(),
      ));
    }
  };

  let var_name = match var_expr {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "FindSequenceFunction: second argument must be a variable".into(),
      ));
    }
  };

  // Convert items to rational numbers (as i128 numerator/denominator pairs)
  let vals: Vec<(i128, i128)> = items
    .iter()
    .map(expr_to_rational)
    .collect::<Option<Vec<_>>>()
    .ok_or_else(|| {
      InterpreterError::EvaluationError(
        "FindSequenceFunction: could not convert all elements to numbers"
          .into(),
      )
    })?;

  let _n = vals.len();
  let var = Expr::Identifier(var_name.clone());

  // Try factorial: a(n) = n!
  if try_factorial(&vals) {
    return Ok(Expr::FunctionCall {
      name: "Factorial".to_string(),
      args: vec![var],
    });
  }

  // Try exponential: a(n) = base^n
  if let Some(base) = try_exponential(&vals) {
    return Ok(Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![rational_to_expr(base.0, base.1), var],
    });
  }

  // Try polynomial via finite differences
  if let Some(expr) = try_polynomial(&vals, &var_name) {
    return Ok(expr);
  }

  // If nothing works, return unevaluated
  Ok(Expr::FunctionCall {
    name: "FindSequenceFunction".to_string(),
    args: vec![data_expr.clone(), var_expr.clone()],
  })
}

/// Convert an Expr to a rational number (numerator, denominator).
fn expr_to_rational(expr: &Expr) -> Option<(i128, i128)> {
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n, *d))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Convert a rational (n, d) back to an Expr.
fn rational_to_expr(n: i128, d: i128) -> Expr {
  if d == 1 {
    Expr::Integer(n)
  } else {
    let g = gcd_i128(n.abs(), d.abs());
    let (n, d) = (n / g, d / g);
    if d < 0 {
      rational_to_expr(-n, -d)
    } else if d == 1 {
      Expr::Integer(n)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(n), Expr::Integer(d)],
      }
    }
  }
}

fn gcd_i128(a: i128, b: i128) -> i128 {
  if b == 0 { a } else { gcd_i128(b, a % b) }
}

/// Rational arithmetic helpers
fn rat_sub(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  let n = a.0 * b.1 - b.0 * a.1;
  let d = a.1 * b.1;
  let g = gcd_i128(n.abs(), d.abs());
  (n / g, d / g)
}

fn rat_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  let n = a.0 * b.0;
  let d = a.1 * b.1;
  let g = gcd_i128(n.abs(), d.abs());
  (n / g, d / g)
}

fn rat_div(a: (i128, i128), b: (i128, i128)) -> Option<(i128, i128)> {
  if b.0 == 0 {
    return None;
  }
  Some(rat_mul(a, (b.1, b.0)))
}

fn rat_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  let n = a.0 * b.1 + b.0 * a.1;
  let d = a.1 * b.1;
  let g = gcd_i128(n.abs(), d.abs());
  (n / g, d / g)
}

/// Check if the sequence is n! (starting from n=1)
fn try_factorial(vals: &[(i128, i128)]) -> bool {
  let mut fact: i128 = 1;
  for (i, val) in vals.iter().enumerate() {
    fact *= (i + 1) as i128;
    if val.1 != 1 || val.0 != fact {
      return false;
    }
  }
  true
}

/// Check if the sequence is base^n (starting from n=1).
/// Returns the base if successful.
fn try_exponential(vals: &[(i128, i128)]) -> Option<(i128, i128)> {
  if vals.len() < 2 {
    return None;
  }
  // Check constant ratio between consecutive terms
  let ratio = rat_div(vals[1], vals[0])?;
  // ratio must not be 1 (that would be constant, handled by polynomial)
  if ratio == (1, 1) {
    return None;
  }
  // Verify all consecutive ratios are the same
  for i in 2..vals.len() {
    let r = rat_div(vals[i], vals[i - 1])?;
    if r != ratio {
      return None;
    }
  }
  // vals[0] = base^1 = base, vals[1] = base^2, so base = vals[0]
  // But we need to verify vals[0] == ratio (since a(1) = base^1 = base)
  if vals[0] == ratio { Some(ratio) } else { None }
}

/// Try to fit a polynomial using finite differences method.
/// The sequence is assumed to be indexed starting at n=1.
fn try_polynomial(vals: &[(i128, i128)], var_name: &str) -> Option<Expr> {
  let n = vals.len();

  // Build the difference table
  // diffs[k] = k-th order forward differences at position 0
  let mut current: Vec<(i128, i128)> = vals.to_vec();
  let mut diffs: Vec<(i128, i128)> = vec![current[0]]; // 0th diff = first value

  for _order in 1..n {
    let mut next = Vec::with_capacity(current.len() - 1);
    for j in 0..current.len() - 1 {
      next.push(rat_sub(current[j + 1], current[j]));
    }
    diffs.push(next[0]);
    current = next;
    if current.is_empty() {
      break;
    }
  }

  // Find the degree: the last non-zero difference
  let mut degree = diffs.len() - 1;
  while degree > 0 && diffs[degree] == (0, 1) {
    degree -= 1;
  }

  // Verify: all differences of order > degree should be zero
  // (this is already guaranteed by the finite difference method for polynomial sequences)

  // Construct the Newton forward difference formula:
  // p(n) = sum_{k=0}^{degree} diffs[k] * C(n-1, k)
  // where C(n-1, k) = (n-1)(n-2)...(n-k) / k!
  // Note: our sequence is 1-indexed, so we use (n-1) in place of x in the formula

  // Build symbolic expression
  let var = Expr::Identifier(var_name.to_string());
  let mut terms: Vec<Expr> = Vec::new();

  for k in 0..=degree {
    if diffs[k] == (0, 1) {
      continue;
    }

    // Build C(n-1, k) = product_{j=0}^{k-1} (n - 1 - j) / k!
    if k == 0 {
      terms.push(rational_to_expr(diffs[k].0, diffs[k].1));
    } else {
      // Compute coefficient: diffs[k] / k!
      let mut factorial: i128 = 1;
      for j in 1..=k as i128 {
        factorial *= j;
      }
      let coeff = rat_div(diffs[k], (factorial, 1))?;

      // Build product (n-1)(n-2)...(n-k) as a polynomial
      // Expand symbolically: multiply (n - j) for j = 1..k
      // Start with [1] (constant polynomial), multiply by (n - j) each time
      // Polynomial coefficients: poly[i] = coefficient of n^i
      let mut poly: Vec<(i128, i128)> = vec![(1, 1)]; // start with 1
      for j in 1..=k {
        // Multiply poly by (n - j)
        let j_val = j as i128;
        let mut new_poly = vec![(0, 1); poly.len() + 1];
        for (i, &p) in poly.iter().enumerate() {
          // p * n term -> goes to i+1
          new_poly[i + 1] = rat_add(new_poly[i + 1], p);
          // p * (-j) term -> goes to i
          new_poly[i] = rat_sub(new_poly[i], rat_mul(p, (j_val, 1)));
        }
        poly = new_poly;
      }

      // Multiply polynomial coefficients by coeff
      for p in &mut poly {
        *p = rat_mul(*p, coeff);
      }

      // Add each term: poly[i] * n^i
      for (i, &c) in poly.iter().enumerate() {
        if c == (0, 1) {
          continue;
        }
        let c_expr = rational_to_expr(c.0, c.1);
        if i == 0 {
          terms.push(c_expr);
        } else {
          let n_power = if i == 1 {
            var.clone()
          } else {
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![var.clone(), Expr::Integer(i as i128)],
            }
          };
          if c == (1, 1) {
            terms.push(n_power);
          } else {
            terms.push(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![c_expr, n_power],
            });
          }
        }
      }
    }
  }

  if terms.is_empty() {
    return Some(Expr::Integer(0));
  }

  // Combine terms with Plus, then simplify via evaluation
  let expr = if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms,
    }
  };

  // Evaluate to simplify
  let simplified = crate::evaluator::evaluate_expr_to_expr(&expr).ok()?;

  // Verify: check that the formula produces the correct values
  for (i, val) in vals.iter().enumerate() {
    let n_val = Expr::Integer((i + 1) as i128);
    let substituted =
      crate::syntax::substitute_variable(&simplified, var_name, &n_val);
    let result = crate::evaluator::evaluate_expr_to_expr(&substituted).ok()?;
    let result_rat = expr_to_rational(&result)?;
    if result_rat != *val {
      return None;
    }
  }

  Some(simplified)
}

/// Compute the arc length of a 1D curve.
/// ArcLength works for Circle and Line. Other regions return Undefined.
fn compute_arc_length(expr: &Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "ArcLength".to_string(),
      args: vec![expr.clone()],
    })
  };
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      // Circle[{x, y}, r] -> 2*Pi*r, Circle[] -> 2*Pi
      "Circle" => {
        if args.is_empty() || args.len() == 1 {
          // Unit circle: 2*Pi
          let result = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(2), Expr::Constant("Pi".to_string())],
          };
          crate::evaluator::evaluate_expr_to_expr(&result)
        } else if args.len() == 2 {
          // Circle[center, r] -> 2*Pi*r
          let result = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(2),
              Expr::Constant("Pi".to_string()),
              args[1].clone(),
            ],
          };
          crate::evaluator::evaluate_expr_to_expr(&result)
        } else {
          unevaluated()
        }
      }
      // Line[{{x1,y1},{x2,y2},...}] -> sum of segment lengths
      "Line" => {
        if args.len() == 1
          && let Expr::List(pts) = &args[0]
          && pts.len() >= 2
        {
          return compute_polyline_length(pts, "ArcLength");
        }
        unevaluated()
      }
      // Other regions (Disk, Polygon, Triangle, Rectangle) -> Undefined
      "Disk" | "Polygon" | "Triangle" | "Rectangle" | "Ball" => {
        Ok(Expr::Identifier("Undefined".to_string()))
      }
      _ => unevaluated(),
    },
    _ => unevaluated(),
  }
}

/// Compute the perimeter of a region.
fn compute_perimeter(expr: &Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "Perimeter".to_string(),
      args: vec![expr.clone()],
    })
  };
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      // Disk[{x, y}, r] -> 2*Pi*r, Disk[] -> 2*Pi
      "Disk" => {
        if args.is_empty() || args.len() == 1 {
          let result = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(2), Expr::Constant("Pi".to_string())],
          };
          crate::evaluator::evaluate_expr_to_expr(&result)
        } else if args.len() == 2 {
          let result = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(2),
              Expr::Constant("Pi".to_string()),
              args[1].clone(),
            ],
          };
          crate::evaluator::evaluate_expr_to_expr(&result)
        } else {
          unevaluated()
        }
      }
      // Circle is a 1D curve, not a 2D region – Perimeter is Undefined
      "Circle" => Ok(Expr::Identifier("Undefined".to_string())),
      // Rectangle[{x1,y1},{x2,y2}] -> 2*(|x2-x1| + |y2-y1|)
      "Rectangle" => {
        if args.is_empty() {
          // Rectangle[] = Rectangle[{0,0},{1,1}], perimeter = 4
          Ok(Expr::Integer(4))
        } else if args.len() == 1 {
          // Rectangle[{x1,y1}] = Rectangle[{x1,y1},{x1+1,y1+1}], perimeter = 4
          Ok(Expr::Integer(4))
        } else if args.len() == 2 {
          if let (Expr::List(p1), Expr::List(p2)) = (&args[0], &args[1])
            && p1.len() == 2
            && p2.len() == 2
          {
            let width = Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  p2[0].clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), p1[0].clone()],
                  },
                ],
              }],
            };
            let height = Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  p2[1].clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), p1[1].clone()],
                  },
                ],
              }],
            };
            let perimeter = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::Integer(2),
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![width, height],
                },
              ],
            };
            return crate::evaluator::evaluate_expr_to_expr(&perimeter);
          }
          unevaluated()
        } else {
          unevaluated()
        }
      }
      // Triangle[{{x1,y1},{x2,y2},{x3,y3}}] -> sum of side lengths
      "Triangle" => {
        if args.len() == 1
          && let Expr::List(pts) = &args[0]
          && pts.len() == 3
        {
          // Close the polygon: add first point at end
          let mut closed = pts.to_vec();
          closed.push(pts[0].clone());
          return compute_polyline_length(&closed, "Perimeter");
        }
        unevaluated()
      }
      // Polygon[{{x1,y1},...}] -> sum of side lengths (closed)
      "Polygon" => {
        if args.len() == 1
          && let Expr::List(pts) = &args[0]
          && pts.len() >= 3
        {
          let mut closed = pts.to_vec();
          closed.push(pts[0].clone());
          return compute_polyline_length(&closed, "Perimeter");
        }
        unevaluated()
      }
      // Line is a 1D curve, Perimeter is Undefined
      "Line" => Ok(Expr::Identifier("Undefined".to_string())),
      _ => unevaluated(),
    },
    _ => unevaluated(),
  }
}

/// Compute the total length of a polyline (list of points).
fn compute_polyline_length(
  pts: &[Expr],
  func_name: &str,
) -> Result<Expr, InterpreterError> {
  let coords: Vec<&Vec<Expr>> = pts
    .iter()
    .filter_map(|p| {
      if let Expr::List(xy) = p {
        Some(xy)
      } else {
        None
      }
    })
    .collect();

  if coords.len() != pts.len() {
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![Expr::FunctionCall {
        name: "Line".to_string(),
        args: vec![Expr::List(pts.to_vec())],
      }],
    });
  }

  let dim = coords[0].len();
  if !coords.iter().all(|c| c.len() == dim) {
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![Expr::FunctionCall {
        name: "Line".to_string(),
        args: vec![Expr::List(pts.to_vec())],
      }],
    });
  }

  let mut segment_lengths = Vec::new();
  for i in 0..coords.len() - 1 {
    let j = i + 1;
    let mut sq_terms = Vec::new();
    for d in 0..dim {
      sq_terms.push(Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              coords[j][d].clone(),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), coords[i][d].clone()],
              },
            ],
          },
          Expr::Integer(2),
        ],
      });
    }
    segment_lengths.push(make_sqrt(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: sq_terms,
    }));
  }

  if segment_lengths.len() == 1 {
    crate::evaluator::evaluate_expr_to_expr(&segment_lengths[0])
  } else {
    let total = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: segment_lengths,
    };
    crate::evaluator::evaluate_expr_to_expr(&total)
  }
}

/// Normalize a geometric region to its canonical form for comparison.
/// Returns None if the expression is not a recognized region.
fn normalize_region(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      // Point[coords] — already canonical
      "Point" if !args.is_empty() => Some(expr.clone()),

      // Disk[] → Disk[{0,0}, 1]; Disk[c] → Disk[c, 1]
      // Ball[c, r] in 2D → Disk[c, r]
      "Disk" | "Ball" => {
        let (center, radius) = match args.len() {
          0 => {
            let dim = if name == "Ball" { 3 } else { 2 };
            let center = Expr::List(vec![Expr::Integer(0); dim]);
            (center, Expr::Integer(1))
          }
          1 => (args[0].clone(), Expr::Integer(1)),
          _ => (args[0].clone(), args[1].clone()),
        };
        // Determine dimensionality from center
        let dim = match &center {
          Expr::List(coords) => coords.len(),
          _ => {
            if name == "Ball" {
              3
            } else {
              2
            }
          }
        };
        let norm_name = if dim <= 2 { "Disk" } else { "Ball" };
        Some(Expr::FunctionCall {
          name: norm_name.to_string(),
          args: vec![center, radius],
        })
      }

      // Circle[] → Circle[{0,0}, 1]; Circle[c] → Circle[c, 1]
      // Sphere[c, r] in 2D → Circle[c, r]
      "Circle" | "Sphere" => {
        let (center, radius) = match args.len() {
          0 => {
            let dim = if name == "Sphere" { 3 } else { 2 };
            let center = Expr::List(vec![Expr::Integer(0); dim]);
            (center, Expr::Integer(1))
          }
          1 => (args[0].clone(), Expr::Integer(1)),
          _ => (args[0].clone(), args[1].clone()),
        };
        let dim = match &center {
          Expr::List(coords) => coords.len(),
          _ => {
            if name == "Sphere" {
              3
            } else {
              2
            }
          }
        };
        let norm_name = if dim <= 2 { "Circle" } else { "Sphere" };
        Some(Expr::FunctionCall {
          name: norm_name.to_string(),
          args: vec![center, radius],
        })
      }

      // Rectangle[] → Polygon[{{0,0},{1,0},{1,1},{0,1}}]
      // Rectangle[{x1,y1},{x2,y2}] → Polygon with sorted vertices
      "Rectangle" => {
        let (p1, p2) = match args.len() {
          0 => (
            vec![Expr::Integer(0), Expr::Integer(0)],
            vec![Expr::Integer(1), Expr::Integer(1)],
          ),
          1 => {
            if let Expr::List(coords) = &args[0] {
              let p2 = coords
                .iter()
                .map(|c| Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![c.clone(), Expr::Integer(1)],
                })
                .map(|e| {
                  crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e)
                })
                .collect();
              (coords.clone(), p2)
            } else {
              return Some(expr.clone());
            }
          }
          _ => {
            if let (Expr::List(c1), Expr::List(c2)) = (&args[0], &args[1]) {
              (c1.clone(), c2.clone())
            } else {
              return Some(expr.clone());
            }
          }
        };
        if p1.len() == 2 && p2.len() == 2 {
          let vertices = vec![
            Expr::List(vec![p1[0].clone(), p1[1].clone()]),
            Expr::List(vec![p2[0].clone(), p1[1].clone()]),
            Expr::List(vec![p2[0].clone(), p2[1].clone()]),
            Expr::List(vec![p1[0].clone(), p2[1].clone()]),
          ];
          normalize_polygon_vertices(vertices)
        } else {
          Some(expr.clone())
        }
      }

      // Triangle[] → Polygon[{{0,0},{1,0},{0,1}}] (sorted)
      // Triangle[{p1,p2,p3}] → Polygon with sorted vertices
      "Triangle" => {
        let vertices = if args.is_empty() {
          vec![
            Expr::List(vec![Expr::Integer(0), Expr::Integer(0)]),
            Expr::List(vec![Expr::Integer(1), Expr::Integer(0)]),
            Expr::List(vec![Expr::Integer(0), Expr::Integer(1)]),
          ]
        } else if let Some(Expr::List(pts)) = args.first() {
          pts.clone()
        } else {
          return Some(expr.clone());
        };
        normalize_polygon_vertices(vertices)
      }

      // Polygon[{v1, v2, ...}] → sorted canonical form
      "Polygon" => {
        if let Some(Expr::List(vertices)) = args.first() {
          normalize_polygon_vertices(vertices.clone())
        } else {
          Some(expr.clone())
        }
      }

      // Line[{p1, p2, ...}] — sort endpoints for 2-point lines
      "Line" => {
        if let Some(Expr::List(pts)) = args.first() {
          if pts.len() == 2 {
            let mut sorted = pts.clone();
            sorted.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
            Some(Expr::FunctionCall {
              name: "Line".to_string(),
              args: vec![Expr::List(sorted)],
            })
          } else {
            Some(expr.clone())
          }
        } else {
          Some(expr.clone())
        }
      }

      // Interval[{a, b}] — already canonical
      "Interval" => Some(expr.clone()),

      _ => None,
    },
    _ => None,
  }
}

/// Normalize polygon vertices to a canonical sorted form for comparison.
/// We sort vertices lexicographically by their string representation to get
/// a rotation/order-independent comparison.
fn normalize_polygon_vertices(mut vertices: Vec<Expr>) -> Option<Expr> {
  vertices.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
  Some(Expr::FunctionCall {
    name: "Polygon".to_string(),
    args: vec![Expr::List(vertices)],
  })
}

/// Compute RegionEqual[r1, r2, ...].
fn compute_region_equal(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // RegionEqual[] and RegionEqual[r] → True
  if args.len() <= 1 {
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Try to normalize all regions
  let normalized: Vec<Option<Expr>> =
    args.iter().map(normalize_region).collect();

  // If any region is not recognized, return unevaluated
  if normalized.iter().any(|n| n.is_none()) {
    return Ok(Expr::FunctionCall {
      name: "RegionEqual".to_string(),
      args: args.to_vec(),
    });
  }

  let normalized: Vec<Expr> =
    normalized.into_iter().map(|n| n.unwrap()).collect();

  // Compare all pairs: all must be equal to the first
  let first = &normalized[0];
  let all_equal = normalized[1..]
    .iter()
    .all(|n| format!("{n:?}") == format!("{first:?}"));

  if all_equal {
    Ok(Expr::Identifier("True".to_string()))
  } else {
    Ok(Expr::Identifier("False".to_string()))
  }
}

/// Compute PlanarAngle[{p1, vertex, p2}] — the angle at vertex between rays to p1 and p2.
/// Uses ArcCos of the dot product divided by the product of magnitudes.
fn compute_planar_angle(
  p1: &Expr,
  vertex: &Expr,
  p2: &Expr,
) -> Result<Expr, InterpreterError> {
  // Build symbolic vectors v1 = p1 - vertex, v2 = p2 - vertex
  let v1 = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      p1.clone(),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), vertex.clone()],
      },
    ],
  };
  let v2 = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      p2.clone(),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), vertex.clone()],
      },
    ],
  };

  // Evaluate vectors
  let v1_eval = crate::evaluator::evaluate_expr_to_expr(&v1)?;
  let v2_eval = crate::evaluator::evaluate_expr_to_expr(&v2)?;

  // Extract components
  let (v1_comps, v2_comps) = match (&v1_eval, &v2_eval) {
    (Expr::List(a), Expr::List(b)) if a.len() == b.len() => (a, b),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PlanarAngle".to_string(),
        args: vec![Expr::List(vec![p1.clone(), vertex.clone(), p2.clone()])],
      });
    }
  };

  // Check for zero vectors (degenerate case)
  let is_zero = |v: &[Expr]| {
    v.iter().all(|c| match c {
      Expr::Integer(0) => true,
      Expr::Real(f) => *f == 0.0,
      _ => false,
    })
  };
  if is_zero(v1_comps) || is_zero(v2_comps) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  // Build dot product: v1.v2
  let dot_terms: Vec<Expr> = v1_comps
    .iter()
    .zip(v2_comps.iter())
    .map(|(a, b)| Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![a.clone(), b.clone()],
    })
    .collect();
  let dot = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: dot_terms,
  };

  // Build magnitudes: |v1|, |v2|
  let mag1_terms: Vec<Expr> = v1_comps
    .iter()
    .map(|a| Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![a.clone(), Expr::Integer(2)],
    })
    .collect();
  let mag1 = make_sqrt(Expr::FunctionCall {
    name: "Plus".to_string(),
    args: mag1_terms,
  });

  let mag2_terms: Vec<Expr> = v2_comps
    .iter()
    .map(|a| Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![a.clone(), Expr::Integer(2)],
    })
    .collect();
  let mag2 = make_sqrt(Expr::FunctionCall {
    name: "Plus".to_string(),
    args: mag2_terms,
  });

  // ArcCos[dot / (mag1 * mag2)]
  let cos_angle = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      dot,
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![mag1, mag2],
          },
          Expr::Integer(-1),
        ],
      },
    ],
  };
  let angle = Expr::FunctionCall {
    name: "ArcCos".to_string(),
    args: vec![cos_angle],
  };

  crate::evaluator::evaluate_expr_to_expr(&angle)
}

// ─── Insphere ──────────────────────────────────────────────────────────────

/// Compute the insphere (incircle) of a geometric region.
/// For a 2D Triangle: returns Sphere[{cx, cy}, r]
/// For a 3D Tetrahedron: returns Sphere[{cx, cy, cz}, r]
fn compute_insphere(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Triangle" if args.len() == 1 => {
        if let Expr::List(vertices) = &args[0]
          && vertices.len() == 3
        {
          // Extract 2D vertices
          let pts: Vec<&[Expr]> = vertices
            .iter()
            .filter_map(|v| {
              if let Expr::List(coords) = v {
                Some(coords.as_slice())
              } else {
                None
              }
            })
            .collect();
          if pts.len() == 3
            && pts[0].len() == 2
            && pts[1].len() == 2
            && pts[2].len() == 2
          {
            return insphere_triangle_2d(pts[0], pts[1], pts[2]);
          }
        }
        Ok(Expr::FunctionCall {
          name: "Insphere".to_string(),
          args: vec![expr.clone()],
        })
      }
      "Tetrahedron" if args.len() == 1 => {
        if let Expr::List(vertices) = &args[0]
          && vertices.len() == 4
        {
          let pts: Vec<&[Expr]> = vertices
            .iter()
            .filter_map(|v| {
              if let Expr::List(coords) = v {
                Some(coords.as_slice())
              } else {
                None
              }
            })
            .collect();
          if pts.len() == 4 && pts.iter().all(|p| p.len() == 3) {
            return insphere_tetrahedron(pts[0], pts[1], pts[2], pts[3]);
          }
        }
        Ok(Expr::FunctionCall {
          name: "Insphere".to_string(),
          args: vec![expr.clone()],
        })
      }
      _ => {
        // Normalize no-arg primitives like Disk[] → Disk[{0,0}]
        let normalized =
          if args.is_empty() && (name == "Disk" || name == "Ball") {
            let center = match name.as_str() {
              "Disk" => Expr::List(vec![Expr::Integer(0), Expr::Integer(0)]),
              "Ball" => Expr::List(vec![
                Expr::Integer(0),
                Expr::Integer(0),
                Expr::Integer(0),
              ]),
              _ => unreachable!(),
            };
            Expr::FunctionCall {
              name: name.clone(),
              args: vec![center],
            }
          } else {
            expr.clone()
          };
        Ok(Expr::FunctionCall {
          name: "Insphere".to_string(),
          args: vec![normalized],
        })
      }
    },
    _ => Ok(Expr::FunctionCall {
      name: "Insphere".to_string(),
      args: vec![expr.clone()],
    }),
  }
}

/// Helper: build Sqrt[expr]
fn insphere_sqrt(e: Expr) -> Expr {
  make_sqrt(e)
}

/// Helper: build a + b
fn insphere_plus(a: Expr, b: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![a, b],
  }
}

/// Helper: build a * b
fn insphere_times(a: Expr, b: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![a, b],
  }
}

/// Helper: build a - b
fn insphere_minus(a: Expr, b: Expr) -> Expr {
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Minus,
    left: Box::new(a),
    right: Box::new(b),
  }
}

/// Helper: build a^n
fn insphere_power(base: Expr, exp: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![base, exp],
  }
}

/// Compute distance between two 2D points as symbolic expression
fn dist_2d(p1: &[Expr], p2: &[Expr]) -> Expr {
  let dx = insphere_minus(p2[0].clone(), p1[0].clone());
  let dy = insphere_minus(p2[1].clone(), p1[1].clone());
  insphere_sqrt(insphere_plus(
    insphere_power(dx, Expr::Integer(2)),
    insphere_power(dy, Expr::Integer(2)),
  ))
}

/// Compute incircle of a 2D triangle.
/// Vertices: A (p1), B (p2), C (p3)
/// Side a = |BC| (opposite A), b = |AC| (opposite B), c = |AB| (opposite C)
/// Center = (a*A + b*B + c*C) / (a + b + c)
/// Radius = Area / semiperimeter
fn insphere_triangle_2d(
  p1: &[Expr],
  p2: &[Expr],
  p3: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Side lengths
  let a = dist_2d(p2, p3); // opposite vertex A (p1)
  let b = dist_2d(p1, p3); // opposite vertex B (p2)
  let c = dist_2d(p1, p2); // opposite vertex C (p3)

  // Perimeter
  let perimeter = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![a.clone(), b.clone(), c.clone()],
  };

  // Center coordinates: (a*x1 + b*x2 + c*x3) / (a+b+c)
  let cx_num = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      insphere_times(a.clone(), p1[0].clone()),
      insphere_times(b.clone(), p2[0].clone()),
      insphere_times(c.clone(), p3[0].clone()),
    ],
  };
  let cy_num = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      insphere_times(a, p1[1].clone()),
      insphere_times(b, p2[1].clone()),
      insphere_times(c, p3[1].clone()),
    ],
  };

  let cx = insphere_times(
    cx_num,
    insphere_power(perimeter.clone(), Expr::Integer(-1)),
  );
  let cy = insphere_times(
    cy_num,
    insphere_power(perimeter.clone(), Expr::Integer(-1)),
  );

  // Area via shoelace formula: |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)| / 2
  let area_2 = Expr::FunctionCall {
    name: "Abs".to_string(),
    args: vec![Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        insphere_times(
          p1[0].clone(),
          insphere_minus(p2[1].clone(), p3[1].clone()),
        ),
        insphere_times(
          p2[0].clone(),
          insphere_minus(p3[1].clone(), p1[1].clone()),
        ),
        insphere_times(
          p3[0].clone(),
          insphere_minus(p1[1].clone(), p2[1].clone()),
        ),
      ],
    }],
  };

  // radius = area / semiperimeter = (area_2/2) / (perimeter/2) = area_2 / perimeter
  let radius =
    insphere_times(area_2, insphere_power(perimeter, Expr::Integer(-1)));

  // Build Sphere[{cx, cy}, r]
  let center = Expr::List(vec![cx, cy]);
  let sphere = Expr::FunctionCall {
    name: "Sphere".to_string(),
    args: vec![center, radius],
  };

  crate::evaluator::evaluate_expr_to_expr(&sphere)
}

/// Compute |AB x AC| for a 3D triangle (= 2 * area).
/// We return twice the area to avoid fractions; the caller must account for this.
fn triangle_cross_mag_3d(p1: &[Expr], p2: &[Expr], p3: &[Expr]) -> Expr {
  let ab = [
    insphere_minus(p2[0].clone(), p1[0].clone()),
    insphere_minus(p2[1].clone(), p1[1].clone()),
    insphere_minus(p2[2].clone(), p1[2].clone()),
  ];
  let ac = [
    insphere_minus(p3[0].clone(), p1[0].clone()),
    insphere_minus(p3[1].clone(), p1[1].clone()),
    insphere_minus(p3[2].clone(), p1[2].clone()),
  ];

  let cx = insphere_minus(
    insphere_times(ab[1].clone(), ac[2].clone()),
    insphere_times(ab[2].clone(), ac[1].clone()),
  );
  let cy = insphere_minus(
    insphere_times(ab[2].clone(), ac[0].clone()),
    insphere_times(ab[0].clone(), ac[2].clone()),
  );
  let cz = insphere_minus(
    insphere_times(ab[0].clone(), ac[1].clone()),
    insphere_times(ab[1].clone(), ac[0].clone()),
  );

  insphere_sqrt(Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      insphere_power(cx, Expr::Integer(2)),
      insphere_power(cy, Expr::Integer(2)),
      insphere_power(cz, Expr::Integer(2)),
    ],
  })
}

/// Compute insphere of a tetrahedron.
/// Center = weighted average of vertices by opposite face areas
/// Radius = 3 * Volume / surface_area
fn insphere_tetrahedron(
  p1: &[Expr],
  p2: &[Expr],
  p3: &[Expr],
  p4: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Face cross magnitudes (= 2 * face area) for each face opposite a vertex
  let cm1 = triangle_cross_mag_3d(p2, p3, p4); // opposite p1
  let cm2 = triangle_cross_mag_3d(p1, p3, p4); // opposite p2
  let cm3 = triangle_cross_mag_3d(p1, p2, p4); // opposite p3
  let cm4 = triangle_cross_mag_3d(p1, p2, p3); // opposite p4

  // total_cross = sum of cross mags = 2 * total_surface_area
  let total_cross = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![cm1.clone(), cm2.clone(), cm3.clone(), cm4.clone()],
  };

  // Center = (cm1*p1 + cm2*p2 + cm3*p3 + cm4*p4) / total_cross
  // (Same as using actual areas since the 2x factor cancels)
  let mut center_coords = Vec::new();
  for dim in 0..3 {
    let num = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        insphere_times(cm1.clone(), p1[dim].clone()),
        insphere_times(cm2.clone(), p2[dim].clone()),
        insphere_times(cm3.clone(), p3[dim].clone()),
        insphere_times(cm4.clone(), p4[dim].clone()),
      ],
    };
    center_coords.push(insphere_times(
      num,
      insphere_power(total_cross.clone(), Expr::Integer(-1)),
    ));
  }

  // Volume = |det(AB, AC, AD)| / 6
  // Radius = 3V / S = 3 * (|det|/6) / (total_cross/2) = |det| / total_cross
  let ab = [
    insphere_minus(p2[0].clone(), p1[0].clone()),
    insphere_minus(p2[1].clone(), p1[1].clone()),
    insphere_minus(p2[2].clone(), p1[2].clone()),
  ];
  let ac = [
    insphere_minus(p3[0].clone(), p1[0].clone()),
    insphere_minus(p3[1].clone(), p1[1].clone()),
    insphere_minus(p3[2].clone(), p1[2].clone()),
  ];
  let ad = [
    insphere_minus(p4[0].clone(), p1[0].clone()),
    insphere_minus(p4[1].clone(), p1[1].clone()),
    insphere_minus(p4[2].clone(), p1[2].clone()),
  ];

  let cross_x = insphere_minus(
    insphere_times(ac[1].clone(), ad[2].clone()),
    insphere_times(ac[2].clone(), ad[1].clone()),
  );
  let cross_y = insphere_minus(
    insphere_times(ac[2].clone(), ad[0].clone()),
    insphere_times(ac[0].clone(), ad[2].clone()),
  );
  let cross_z = insphere_minus(
    insphere_times(ac[0].clone(), ad[1].clone()),
    insphere_times(ac[1].clone(), ad[0].clone()),
  );

  let det = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      insphere_times(ab[0].clone(), cross_x),
      insphere_times(ab[1].clone(), cross_y),
      insphere_times(ab[2].clone(), cross_z),
    ],
  };

  // radius = |det| / total_cross
  let radius = insphere_times(
    Expr::FunctionCall {
      name: "Abs".to_string(),
      args: vec![det],
    },
    insphere_power(total_cross, Expr::Integer(-1)),
  );

  let center = Expr::List(center_coords);
  let sphere = Expr::FunctionCall {
    name: "Sphere".to_string(),
    args: vec![center, radius],
  };

  crate::evaluator::evaluate_expr_to_expr(&sphere)
}

/// RegionWithin[reg1, reg2] - True if reg2 is entirely within reg1
fn region_within(
  reg1: &Expr,
  reg2: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::evaluator::type_helpers::expr_to_number;

  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionWithin".to_string(),
      args: args.to_vec(),
    })
  };

  let true_expr = || Ok(Expr::Identifier("True".to_string()));
  let false_expr = || Ok(Expr::Identifier("False".to_string()));

  // Extract region info: (name, center_coords, radius_or_half_sizes)
  let parse_disk_ball = |r: &Expr| -> Option<(String, Vec<f64>, f64)> {
    if let Expr::FunctionCall { name, args: rargs } = r {
      match name.as_str() {
        "Disk" | "Ball" => {
          let (center, radius) = match rargs.len() {
            0 => {
              let dim = if name == "Disk" { 2 } else { 3 };
              (vec![0.0; dim], 1.0)
            }
            1 => {
              if let Expr::List(coords) = &rargs[0] {
                let c: Vec<f64> =
                  coords.iter().filter_map(expr_to_number).collect();
                if c.len() == coords.len() {
                  (c, 1.0)
                } else {
                  return None;
                }
              } else {
                return None;
              }
            }
            2 => {
              let c = if let Expr::List(coords) = &rargs[0] {
                let c: Vec<f64> =
                  coords.iter().filter_map(expr_to_number).collect();
                if c.len() == coords.len() {
                  c
                } else {
                  return None;
                }
              } else {
                return None;
              };
              let r = expr_to_number(&rargs[1])?;
              (c, r)
            }
            _ => return None,
          };
          Some((name.clone(), center, radius))
        }
        _ => None,
      }
    } else {
      None
    }
  };

  let parse_point = |r: &Expr| -> Option<Vec<f64>> {
    if let Expr::FunctionCall { name, args: rargs } = r
      && name == "Point"
      && rargs.len() == 1
      && let Expr::List(coords) = &rargs[0]
    {
      let c: Vec<f64> = coords.iter().filter_map(expr_to_number).collect();
      if c.len() == coords.len() {
        Some(c)
      } else {
        None
      }
    } else {
      None
    }
  };

  // Point within Disk/Ball
  if let Some(pt) = parse_point(reg2)
    && let Some((_, center, radius)) = parse_disk_ball(reg1)
    && pt.len() == center.len()
  {
    let dist_sq: f64 = pt
      .iter()
      .zip(center.iter())
      .map(|(a, b)| (a - b).powi(2))
      .sum();
    return if dist_sq <= radius * radius + 1e-10 {
      true_expr()
    } else {
      false_expr()
    };
  }

  // Disk/Ball within Disk/Ball
  if let (Some((_, c1, r1)), Some((_, c2, r2))) =
    (parse_disk_ball(reg1), parse_disk_ball(reg2))
    && c1.len() == c2.len()
  {
    let dist: f64 = c1
      .iter()
      .zip(c2.iter())
      .map(|(a, b)| (a - b).powi(2))
      .sum::<f64>()
      .sqrt();
    // reg2 is within reg1 if dist(centers) + r2 <= r1
    return if dist + r2 <= r1 + 1e-10 {
      true_expr()
    } else {
      false_expr()
    };
  }

  unevaluated()
}
