#[allow(unused_imports)]
use super::*;
use crate::functions::math_ast::{
  expr_to_rational, gcd_i128, is_sqrt, make_sqrt,
};
use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, bool_expr, unevaluated,
};

fn binop(op: BinaryOperator, left: Expr, right: Expr) -> Expr {
  Expr::BinaryOp {
    op,
    left: Box::new(left),
    right: Box::new(right),
  }
}

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
    // These step/impulse functions thread element-wise over a single list
    // argument in wolframscript (HeavisideTheta and DiracDelta are Listable;
    // the others thread the same way). Map the scalar form over the elements.
    "HeavisideTheta" | "DiracDelta" | "UnitBox" | "UnitTriangle"
    | "HeavisidePi" | "HeavisideLambda"
      if args.len() == 1 && matches!(&args[0], Expr::List(_)) =>
    {
      let Expr::List(items) = &args[0] else {
        unreachable!();
      };
      let results: Result<Vec<Expr>, InterpreterError> = items
        .iter()
        .map(|x| {
          crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
            name: name.to_string(),
            args: vec![x.clone()].into(),
          })
        })
        .collect();
      return Some(results.map(|v| Expr::List(v.into())));
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
          return Some(Ok(binop(
            BinaryOperator::Times,
            imag.clone(),
            Expr::Identifier("I".to_string()),
          )));
        }
        // If imaginary is 1, return a + I
        if matches!(imag, Expr::Integer(1)) {
          return Some(Ok(binop(
            BinaryOperator::Plus,
            real.clone(),
            Expr::Identifier("I".to_string()),
          )));
        }
        // General case without I in imag. For concrete numeric components,
        // build a + b*I and EVALUATE it so the result lands in the canonical
        // Plus form — a raw Minus/Plus BinaryOp is opaque to plus_ast's
        // flattening, so `(1.5 + 2.5 I) + Complex[2., -3.75]` never combined.
        let numeric_part = |x: &Expr| {
          matches!(
            x,
            Expr::Integer(_)
              | Expr::BigInteger(_)
              | Expr::Real(_)
              | Expr::BigFloat(_, _)
          ) || matches!(x, Expr::FunctionCall { name, args }
               if name == "Rational" && args.len() == 2)
        };
        if numeric_part(real) && numeric_part(imag) {
          let bi = match evaluate_function_call_ast(
            "Times",
            &[imag.clone(), Expr::Identifier("I".to_string())],
          ) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
          };
          return Some(evaluate_function_call_ast("Plus", &[real.clone(), bi]));
        }
        // Symbolic components (e.g. the pattern Complex[a_, b_]) keep the
        // raw a + b*I BinaryOp shape so structural pattern matching against
        // Plus/Times complex trees still works.
        return Some(Ok(binop(
          BinaryOperator::Plus,
          real.clone(),
          binop(
            BinaryOperator::Times,
            imag.clone(),
            Expr::Identifier("I".to_string()),
          ),
        )));
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
        return Some(Ok(unevaluated("ConditionalExpression", args)));
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
            op: UnaryOperator::Minus,
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
                op: UnaryOperator::Minus,
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
                  op: UnaryOperator::Minus,
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
                  args: vec![Expr::Integer(msn), Expr::Integer(msd)].into(),
                }
              };
              let normalized = binop(
                BinaryOperator::Divide,
                args[0].clone(),
                make_sqrt(sqrt_arg),
              );
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
                  op: UnaryOperator::Minus,
                  operand: Box::new(Expr::Identifier("Infinity".to_string())),
                }));
              }
              return Some(Ok(Expr::FunctionCall {
                name: "DirectedInfinity".to_string(),
                args: vec![normalized].into(),
              }));
            }
          }
          // Symbolic complex with real-valued numeric components (e.g.
          // `-1 + 2*Pi*I`). When neither part is identically zero and both
          // are real-valued, normalize to `DirectedInfinity[z/Sqrt[Re^2+Im^2]]`
          // — matching wolframscript's `(-1 + 2*Pi*I)*Infinity` →
          // `DirectedInfinity[(-1 + 2*I*Pi)/Sqrt[1 + 4*Pi^2]]`.
          if !expr_contains_real(&args[0])
            && let Some((re, im)) = split_real_imag_symbolic(&args[0])
            && !is_zero_expr(&im)
          {
            let re_sq =
              binop(BinaryOperator::Power, re.clone(), Expr::Integer(2));
            let im_sq =
              binop(BinaryOperator::Power, im.clone(), Expr::Integer(2));
            let mag_sq = binop(BinaryOperator::Plus, re_sq, im_sq);
            let mag_sq = match evaluate_expr_to_expr(&mag_sq) {
              Ok(v) => v,
              Err(e) => return Some(Err(e)),
            };
            let direction =
              binop(BinaryOperator::Divide, args[0].clone(), make_sqrt(mag_sq));
            let direction = match evaluate_expr_to_expr(&direction) {
              Ok(v) => v,
              Err(e) => return Some(Err(e)),
            };
            return Some(Ok(Expr::FunctionCall {
              name: "DirectedInfinity".to_string(),
              args: vec![direction].into(),
            }));
          }
          // Floating-point complex direction: normalize numerically so
          // `DirectedInfinity[1. + 2. I]` displays as
          // `DirectedInfinity[0.4472… + 0.8944…*I]`, matching Wolfram.
          // Only apply when the input is genuinely inexact — an exact
          // symbolic expression like `(1 + 2 I)/Sqrt[5]` should keep
          // its closed form, even though `try_extract_complex_f64`
          // could compute a float for it.
          if expr_contains_real(&args[0])
            && let Some((re, im)) =
              crate::functions::math_ast::try_extract_complex_f64(&args[0])
            && (re != 0.0 || im != 0.0)
            && (re.is_finite() && im.is_finite())
          {
            let mag = (re * re + im * im).sqrt();
            if mag.is_finite() && mag > 0.0 {
              let nre = re / mag;
              let nim = im / mag;
              if nim == 0.0 {
                if nre > 0.0 {
                  return Some(Ok(Expr::Identifier("Infinity".to_string())));
                }
                if nre < 0.0 {
                  return Some(Ok(Expr::UnaryOp {
                    op: UnaryOperator::Minus,
                    operand: Box::new(Expr::Identifier("Infinity".to_string())),
                  }));
                }
              }
              // Build `re + im*I` so the regular Times printer handles
              // sign placement and `0. + r*I` Re/Im split.
              let im_term = binop(
                BinaryOperator::Times,
                Expr::Real(nim),
                Expr::Identifier("I".to_string()),
              );
              let direction =
                binop(BinaryOperator::Plus, Expr::Real(nre), im_term);
              let direction = match evaluate_expr_to_expr(&direction) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
              };
              return Some(Ok(Expr::FunctionCall {
                name: "DirectedInfinity".to_string(),
                args: vec![direction].into(),
              }));
            }
          }
          return Some(Ok(unevaluated("DirectedInfinity", args)));
        }
      }
    }

    // Echo[expr] - prints ">> expr" and returns expr
    // Echo[expr, label] - prints ">> label expr" and returns expr
    // Echo[expr, label, f] - prints ">> label f[expr]" and returns expr
    // EchoFunction[…] and EchoLabel[…] are operator forms: they stay
    // symbolic until applied to an argument (handled in apply_curried_call).
    "EchoFunction" if args.len() <= 2 => {
      return Some(Ok(unevaluated("EchoFunction", args)));
    }
    "EchoLabel" if args.len() <= 1 => {
      return Some(Ok(unevaluated("EchoLabel", args)));
    }
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
            args: vec![args[0].clone()].into(),
          },
          other => Expr::FunctionCall {
            name: "Apply".to_string(),
            args: vec![other.clone(), args[0].clone()].into(),
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

    // TimeConstrained[body, t] / TimeConstrained[body, t, fallback]
    // Woxi cannot interrupt a running evaluation, but we can still
    // detect an after-the-fact overshoot: evaluate the body, then
    // compare the wall-clock duration to the requested limit. If the
    // body overshot, return the supplied fallback (3-arg form) or the
    // sentinel symbol `$Aborted`. Returning the sentinel rather than
    // raising the global Abort signal keeps subsequent CompoundExpression
    // elements alive — wolframscript continues evaluating after a
    // single TimeConstrained timeout.
    //
    // Additional heuristic for `Integrate[…];` bodies that finish under
    // the limit only because Woxi already gave up (the result expression
    // contains an unevaluated `Integrate[…]` call): treat the give-up
    // as a timeout so a chained `TimeConstrained[Integrate[…], t,
    // fallback]` returns the supplied fallback. wolframscript reaches
    // the fallback by actually exhausting the budget; we'd otherwise
    // return whatever the `;`-discarded body left behind.
    // MemoryConstrained[body, b] / MemoryConstrained[body, b, fallback]
    // Woxi cannot intercept allocations as they happen, so we evaluate
    // the body (HoldFirst) and then compare the result's ByteCount to
    // the requested budget. On overshoot, return $Aborted (2-arg form)
    // or the supplied fallback.
    "MemoryConstrained" if args.len() == 2 || args.len() == 3 => {
      let limit = match &args[1] {
        Expr::Real(f) => Some(*f as i128),
        Expr::Integer(n) => Some(*n),
        _ => None,
      };
      let result = crate::evaluator::evaluate_expr_to_expr(&args[0]);
      let exceeded = match (&result, limit) {
        (Ok(r), Some(limit)) => {
          let bc = crate::functions::predicate_ast::byte_count_ast(
            std::slice::from_ref(r),
          );
          matches!(bc, Ok(Expr::Integer(n)) if n > limit)
        }
        _ => false,
      };
      if exceeded {
        if args.len() == 3 {
          return Some(crate::evaluator::evaluate_expr_to_expr(&args[2]));
        }
        return Some(Ok(Expr::Identifier("$Aborted".to_string())));
      }
      return Some(result);
    }
    "TimeConstrained" if args.len() == 2 || args.len() == 3 => {
      let limit_secs = match &args[1] {
        Expr::Real(f) => Some(*f),
        Expr::Integer(n) => Some(*n as f64),
        _ => None,
      };
      let start = web_time::Instant::now();
      let result = crate::evaluator::evaluate_expr_to_expr(&args[0]);
      let elapsed = start.elapsed().as_secs_f64();
      let mut overshot = matches!(limit_secs, Some(limit) if elapsed > limit);
      if !overshot && contains_unevaluated_integrate(&args[0]) {
        // Body had an `Integrate[…]` that Woxi couldn't simplify.
        // Treat as a timeout so case 4528's chained TimeConstrained
        // hits its fallback instead of returning the body's discarded
        // (`;`-stripped) value.
        overshot = true;
      }
      if overshot {
        if args.len() == 3 {
          return Some(crate::evaluator::evaluate_expr_to_expr(&args[2]));
        }
        return Some(Ok(Expr::Identifier("$Aborted".to_string())));
      }
      return Some(result);
    }
    // Information[symbol] or Information[symbol, LongForm -> True]
    // `?symbol` parses as Information[Unevaluated[symbol], LongForm -> False]
    // `??symbol` parses as Information[Unevaluated[symbol], LongForm -> True]
    // The Unevaluated wrapper is what makes the REPL shortcuts inspect the
    // symbol even when its OwnValue would otherwise replace it. Strip it
    // here before classifying the argument.
    // Legacy "Full" string form is still accepted for backward compatibility.
    "Information" if args.len() == 1 || args.len() == 2 => {
      let is_full = args.len() == 2
        && (matches!(&args[1], Expr::String(s) if s == "Full")
          || matches!(&args[1],
            Expr::Rule { pattern, replacement }
              if matches!(pattern.as_ref(), Expr::Identifier(p) if p == "LongForm")
                && matches!(replacement.as_ref(),
                  Expr::Identifier(v) if v == "True")));

      // The `?sym` REPL shortcut wraps its argument in `Unevaluated[…]`;
      // wolframscript returns `Missing[UnknownSymbol, …]` for the shortcut
      // on a never-defined user symbol, but the bare `Information[sym]`
      // form returns a default `InformationData` record. Distinguish the
      // two paths by whether the original argument was wrapped.
      let was_unevaluated_wrap = matches!(&args[0],
        Expr::FunctionCall { name: n, args: ua }
          if n == "Unevaluated" && ua.len() == 1);
      let first_arg = match &args[0] {
        Expr::FunctionCall { name: n, args: ua }
          if n == "Unevaluated" && ua.len() == 1 =>
        {
          &ua[0]
        }
        other => other,
      };

      if let Expr::Identifier(sym) = first_arg {
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
        let down_values = crate::down_values_with_memo(sym);
        let user_attrs =
          crate::FUNC_ATTRS.with(|m| m.borrow().get(sym).cloned());

        let has_own = own_value.is_some();
        let has_down = down_values.is_some();
        let has_attrs = user_attrs.as_ref().is_some_and(|a| !a.is_empty());

        // The `?sym` REPL form returns `Missing[UnknownSymbol, sym]` when
        // the symbol has no definition, attributes, or values — matching
        // wolframscript. The bare `Information[sym]` call falls through
        // to the default-record path below.
        if was_unevaluated_wrap && !has_own && !has_down && !has_attrs {
          return Some(Ok(Expr::FunctionCall {
            name: "Missing".to_string(),
            args: vec![
              Expr::String("UnknownSymbol".to_string()),
              Expr::String(sym.clone()),
            ]
            .into(),
          }));
        }

        // Wolfram returns a default InformationData record (with all
        // value-set fields = None and Attributes = {}) for unknown
        // user-context symbols, rather than `Missing[…]`.
        return Some(Ok(format_user_information(
          sym,
          own_value,
          down_values,
          user_attrs,
          is_full,
        )));
      }
      // String form `Information["sym"]` (without wildcards): treat as a
      // Global` symbol lookup and return the default InformationData record
      // (matching wolframscript when the symbol has been mentioned in the
      // current session). Wildcard strings like "Plot*" are handled below.
      if let Expr::String(sym) = first_arg
        && !sym.contains('*')
      {
        let own_value = crate::ENV.with(|e| e.borrow().get(sym).cloned());
        let down_values = crate::down_values_with_memo(sym);
        let user_attrs =
          crate::FUNC_ATTRS.with(|m| m.borrow().get(sym).cloned());
        return Some(Ok(format_user_information(
          sym,
          own_value,
          down_values,
          user_attrs,
          is_full,
        )));
      }

      // Pattern query: ?Plot* or ?*Plot* — first arg is a String with wildcards
      if let Expr::String(pattern) = first_arg
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

          let matching_names: Vec<String> =
            all_names.into_iter().filter(|n| re.is_match(n)).collect();
          let matching: Vec<Expr> = matching_names
            .iter()
            .cloned()
            .map(Expr::Identifier)
            .collect();

          // Capture a graphical SVG card alongside the textual result.
          let groups = vec![("System`".to_string(), matching_names.clone())];
          if let Some(svg) =
            crate::functions::information_render::render_information_grid_svg(
              &groups,
            )
          {
            crate::capture_graphics(&svg);
          }

          return Some(Ok(Expr::FunctionCall {
            name: "InformationDataGrid".to_string(),
            args: vec![
              Expr::List(
                vec![Expr::FunctionCall {
                  name: "Rule".to_string(),
                  args: vec![
                    Expr::Identifier("System`".to_string()),
                    Expr::List(matching.into()),
                  ]
                  .into(),
                }]
                .into(),
              ),
              bool_expr(is_full),
            ]
            .into(),
          }));
        }
      }

      // Non-identifier argument — return unevaluated
      return Some(Ok(unevaluated("Information", args)));
    }

    // Definition[symbol] - show definition of a symbol
    "Definition" if args.len() == 1 => {
      if let Expr::Identifier(sym) = &args[0] {
        let mut lines: Vec<String> = Vec::new();

        // 1. Show user-set attributes if present, otherwise fall back to
        // built-in attributes — `Definition[In]` should print
        // `Attributes[In] = {Listable, NHoldFirst, Protected}` even with no
        // user-installed attrs, matching wolframscript.
        let user_attrs =
          crate::FUNC_ATTRS.with(|m| m.borrow().get(sym).cloned());
        let attrs_to_show: Vec<String> = match &user_attrs {
          Some(a) if !a.is_empty() => a.clone(),
          _ => crate::evaluator::attributes::get_builtin_attributes(sym)
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        };
        if !attrs_to_show.is_empty() {
          lines.push(format!(
            "Attributes[{}] = {{{}}}",
            sym,
            attrs_to_show.join(", ")
          ));
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
              let items_expr: Vec<(Expr, Expr)> = items
                .iter()
                .map(|(k, v)| {
                  let key_expr = crate::syntax::string_to_expr(k)
                    .unwrap_or(Expr::Identifier(k.clone()));
                  (key_expr, v.clone())
                })
                .collect();
              expr_to_string(&Expr::Association(items_expr))
            }
          };
          lines.push(format!("{} = {}", sym, val_str));
        }

        // ReadProtected hides the symbol's implementation details
        // (DownValues, UpValues, Format/MakeBoxes, NValues, SubValues),
        // surfacing only Attributes / Default / Options. Skip those
        // sections when the symbol is read-protected.
        let read_protected = attrs_to_show.iter().any(|a| a == "ReadProtected");
        // 3. Show UpValues first (rules attached via Real /: F[x_Real] := x
        // etc.), matching wolframscript's ordering. UpValues precede
        // DownValues in Definition output.
        let up_values = if read_protected {
          None
        } else {
          crate::UPVALUES.with(|m| {
            let defs = m.borrow();
            defs.get(sym).cloned()
          })
        };
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
        let down_values = if read_protected {
          None
        } else {
          crate::down_values_with_memo(sym)
        };
        if let Some(overloads) = down_values {
          for (params, conds, _defaults, heads, blank_types, body) in
            overloads.iter().filter(|(params, _, _, heads, _, _)| {
              !upvalue_keys.contains(&(params.clone(), heads.clone()))
            })
          {
            // List-pattern params reconstruct to a surface `{…}` pattern with
            // the original element names, body, and `/;` guard.
            if let Some((pattern_args, display_body)) =
              crate::evaluator::assignment::reconstruct_list_downvalue(
                params,
                conds,
                heads,
                blank_types,
                body,
              )
            {
              let params_str = pattern_args
                .iter()
                .map(expr_to_string)
                .collect::<Vec<_>>()
                .join(", ");
              lines.push(format!(
                "{}[{}] := {}",
                sym,
                params_str,
                expr_to_string(&display_body)
              ));
              continue;
            }
            // Check if this is a specific-value definition (SameQ conditions)
            let has_sameq_conds = conds.iter().any(|c| {
              if let Some(Expr::Comparison { operators, .. }) = c {
                operators.iter().any(|op| matches!(op, ComparisonOp::SameQ))
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
                    && operators
                      .iter()
                      .any(|op| matches!(op, ComparisonOp::SameQ))
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

        // 4b. Show SubValues (rules like `f[1][x_] := x` keyed under f).
        let sub_value_entries = if read_protected {
          None
        } else {
          crate::evaluator::assignment::SUB_VALUES
            .with(|m| m.borrow().get(sym).cloned())
        };
        if let Some(entries) = sub_value_entries {
          for (lhs, rhs) in &entries {
            lines.push(format!(
              "{} := {}",
              expr_to_string(lhs),
              expr_to_string(rhs)
            ));
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
        // Show user-set DefaultValues. They live in `FUNC_DEFS["Default"]`
        // because that's how Default[r, n] := v is dispatched, but
        // wolframscript surfaces them as a TagSet line: `r /: Default[r, n] := v`.
        let default_defs = crate::FUNC_DEFS
          .with(|m| m.borrow().get("Default").cloned().unwrap_or_default());
        for (params, _conds, _defaults, _heads, _blanks, body) in &default_defs
        {
          // Our SetDelayed stores `Default[r, 1]` with the first param named
          // `"r"` (a pattern variable named after the symbol). Match by name.
          if params.first().is_none_or(|p| p != sym) {
            continue;
          }
          // Reconstruct the inner Default[sym, …] arguments. The first
          // arg is the literal symbol; trailing slot-literal conditions
          // give the additional args (e.g. `1` for `Default[r, 1]`).
          let mut default_args = vec![sym.to_string()];
          for p in params.iter().skip(1) {
            // Skip the param name; the actual literal lives in the
            // SameQ condition on this slot. Reconstruct via the same
            // helper used by DefaultValues.
            let lit = _conds.iter().find_map(|c| {
              if let Some(Expr::Comparison {
                operands,
                operators,
              }) = c
                && operators.len() == 1
                && matches!(operators[0], ComparisonOp::SameQ)
                && operands.len() == 2
                && let Expr::Identifier(name) = &operands[0]
                && name == p
              {
                Some(expr_to_string(&operands[1]))
              } else {
                None
              }
            });
            if let Some(s) = lit {
              default_args.push(s);
            }
          }
          lines.push(format!(
            "{} /: Default[{}] := {}",
            sym,
            default_args.join(", "),
            expr_to_string(body)
          ));
        }

        // Show Options if the symbol has any (user-stored or built-in).
        let stored_opts =
          crate::FUNC_OPTIONS.with(|m| m.borrow().get(sym).cloned());
        let is_user_stored = stored_opts.is_some();
        let opts = stored_opts.unwrap_or_else(|| {
          crate::evaluator::dispatch::predicate_functions::builtin_default_options(sym)
        });
        if !opts.is_empty() {
          let opts_str: Vec<String> = opts.iter().map(expr_to_string).collect();
          // For user-stored options, use the operator the user wrote
          // (tracked in `FUNC_OPTIONS_DELAYED`). For built-in options,
          // wolframscript prints `:=` when the symbol carries
          // `ReadProtected` and `=` otherwise — matches the Definition
          // outputs of `D`/`Integrate` (`:=`) vs. `Solve`/`Reduce` (`=`).
          let user_delayed =
            crate::FUNC_OPTIONS_DELAYED.with(|m| m.borrow().contains(sym));
          let op = if is_user_stored {
            if user_delayed { ":=" } else { "=" }
          } else if get_builtin_attributes(sym).contains(&"ReadProtected") {
            ":="
          } else {
            "="
          };
          lines.push(format!(
            "Options[{}] {} {{{}}}",
            sym,
            op,
            opts_str.join(", ")
          ));
        }

        if lines.is_empty() {
          // Undefined symbol — wolframscript displays a blank
          // `Definition[…]` panel (printable form is empty), so emit
          // an empty raw string here to match.
          return Some(Ok(Expr::Raw(String::new())));
        }

        return Some(Ok(Expr::Raw(lines.join("\n \n"))));
      }

      return Some(Ok(unevaluated("Definition", args)));
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
            Expr::List(elems) => {
              for e in elems {
                collect_identifiers(e, out);
              }
            }
            Expr::CompoundExpr(elems) => {
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
                let items_expr: Vec<(Expr, Expr)> = items
                  .iter()
                  .map(|(k, v)| {
                    let key_expr = crate::syntax::string_to_expr(k)
                      .unwrap_or(Expr::Identifier(k.clone()));
                    (key_expr, v.clone())
                  })
                  .collect();
                expr_to_string(&Expr::Association(items_expr))
              }
            };
            lines.push(format!("{} = {}", sym, val_str));
          }

          let down_values = crate::down_values_with_memo(sym);
          if let Some(overloads) = down_values {
            for (params, conds, _defaults, heads, _blank_types, body) in
              &overloads
            {
              let has_sameq_conds = conds.iter().any(|c| {
                if let Some(Expr::Comparison { operators, .. }) = c {
                  operators.iter().any(|op| matches!(op, ComparisonOp::SameQ))
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
                      && operators
                        .iter()
                        .any(|op| matches!(op, ComparisonOp::SameQ))
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
          let down_values = crate::down_values_with_memo(sym);
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

      return Some(Ok(unevaluated("FullDefinition", args)));
    }

    // Sow[expr] or Sow[expr, tag] - adds expr to the current Reap collection
    "Sow" if args.len() == 1 || args.len() == 2 => {
      // Sow[val, {tag1, tag2, ...}] emits one (val, tag_i) pair per tag;
      // with a single tag or 1-arg form, emits one pair.
      let tags: Vec<Expr> = match args.get(1) {
        Some(Expr::List(items)) => items.to_vec(),
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
          return Some(Ok(Expr::List(
            vec![result, Expr::List(vec![].into())].into(),
          )));
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
        let groups: Vec<Expr> = tag_groups
          .into_iter()
          .map(|v| Expr::List(v.into()))
          .collect();
        return Some(Ok(Expr::List(
          vec![result, Expr::List(groups.into())].into(),
        )));
      } else {
        // Reap[expr, patt] / Reap[expr, {patt1, ...}] / Reap[expr, patt, f]
        let patt_arg = match evaluate_expr_to_expr(&args[1]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        let patterns = match &patt_arg {
          Expr::List(pats) => pats.clone(),
          _ => vec![patt_arg.clone()].into(),
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
              let vals_list = Expr::List(vals.into());
              if let Some(f) = &wrap_fn {
                // Apply f[tag, {values}] — handles named heads, anonymous
                // functions (Function[body]), and NamedFunction forms.
                crate::evaluator::function_application::apply_curried_call(
                  f,
                  &[tag, vals_list],
                )
                .unwrap_or(Expr::FunctionCall {
                  name: "List".to_string(),
                  args: vec![].into(),
                })
              } else {
                vals_list
              }
            })
            .collect();

          if is_list_form {
            // {patt1, patt2, ...} form: each pattern contributes a list,
            // even if empty.
            result_groups.push(Expr::List(per_pattern.into()));
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
        return Some(Ok(Expr::List(
          vec![result, Expr::List(result_groups.into())].into(),
        )));
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
    "ReplaceRepeated" if args.len() == 2 || args.len() == 3 => {
      let rules = if let Expr::FunctionCall { name: dn, args: da } = &args[1] {
        if dn == "Dispatch" && da.len() == 1 {
          &da[0]
        } else {
          &args[1]
        }
      } else {
        &args[1]
      };
      // Optional third argument: MaxIterations -> n (default 65536).
      let max_iterations = if args.len() == 3 {
        extract_max_iterations(&args[2])?
      } else {
        crate::evaluator::pattern_matching::REPLACE_REPEATED_DEFAULT_MAX
      };
      return Some(
        crate::evaluator::pattern_matching::apply_replace_repeated_ast_with_max(
          &args[0],
          rules,
          max_iterations,
        ),
      );
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
          return Some(Ok(unevaluated("ReplaceList", args)));
        }
      };
      if max_matches == Some(0) {
        return Some(Ok(Expr::List(vec![].into())));
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
            args: group_args.into(),
          };
          match evaluate_expr_to_expr(&inner_call) {
            Ok(r) => outer.push(r),
            Err(e) => return Some(Err(e)),
          }
        }
        return Some(Ok(Expr::List(outer.into())));
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
            args: inner_args.into(),
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
        return Some(Ok(Expr::List(combined.into())));
      }
      // Try the list-sequence enumerator first. Both `->` (Rule) and `:>`
      // (RuleDelayed) carry the same pattern/replacement shape here; the
      // replacement is substituted then evaluated either way.
      let rule_parts = match &args[1] {
        Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        } => Some((pattern, replacement)),
        _ => None,
      };
      if let Expr::List(expr_elems) = &args[0]
        && let Some((pattern, replacement)) = rule_parts
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
        return Some(Ok(Expr::List(results.into())));
      }
      // Flat partition enumerator: e.g. ReplaceList[a+b+c, x_+y_ -> {x,y}]
      // enumerates all Flat partitions of the expression args into
      // pat_args.len() non-empty groups. Handles both Rule and RuleDelayed,
      // and both the FunctionCall and the parsed BinaryOp forms of the
      // expression / pattern (e.g. `a b c` and `x_ y_` parse as BinaryOp Times,
      // not FunctionCall Times).
      let flat_rule = match &args[1] {
        Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        } => Some((pattern, replacement)),
        _ => None,
      };
      if let Some((pattern, replacement)) = flat_rule {
        // Normalize FunctionCall and operator-BinaryOp forms to (head, args).
        let head_args = |e: &Expr| -> Option<(String, Vec<Expr>)> {
          match e {
            Expr::FunctionCall { name, args } => {
              Some((name.clone(), args.to_vec()))
            }
            _ => crate::functions::expr_to_head_args(e),
          }
        };
        if let (Some((expr_head, expr_fargs)), Some((pat_head, pat_fargs))) =
          (head_args(&args[0]), head_args(pattern))
          && expr_head == pat_head
          && !pat_fargs.is_empty()
          && pat_fargs.len() <= expr_fargs.len()
        {
          let has_flat =
            crate::evaluator::listable::is_builtin_flat(&expr_head)
              || crate::FUNC_ATTRS.with(|m| {
                m.borrow()
                  .get(expr_head.as_str())
                  .is_some_and(|attrs| attrs.contains(&"Flat".to_string()))
              });
          if has_flat {
            let all_bindings =
              crate::evaluator::pattern_matching::enumerate_flat_partition_matches(
                &expr_head, &pat_fargs, &expr_fargs,
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
              return Some(Ok(Expr::List(results.into())));
            }
          }
        }
      }
      // Fallback: ReplaceList matches the rule against the WHOLE expression
      // (level 0) only — it does not descend into subparts. Test the single
      // top-level match directly with the pattern matcher so that a rule
      // whose result equals the input (e.g. `x_ -> x`) is still reported,
      // instead of inferring "no match" from string equality.
      let rule_top = match &args[1] {
        Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        } => Some((pattern.clone(), replacement.clone())),
        _ => None,
      };
      if let Some((pattern, replacement)) = rule_top {
        match crate::evaluator::pattern_matching::match_pattern(
          &args[0], &pattern,
        ) {
          Some(bindings) => {
            let mut rhs = replacement.as_ref().clone();
            for (name, value) in bindings {
              rhs = crate::syntax::substitute_variable(&rhs, &name, &value);
            }
            let evaluated = match evaluate_expr_to_expr(&rhs) {
              Ok(r) => r,
              Err(e) => return Some(Err(e)),
            };
            return Some(Ok(Expr::List(vec![evaluated].into())));
          }
          None => return Some(Ok(Expr::List(vec![].into()))),
        }
      }
      // Second argument is not a rule — leave unevaluated.
      return Some(Ok(unevaluated("ReplaceList", args)));
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

    "OutputForm" if args.len() == 1 => {
      // `wolframscript -code 'OutputForm[expr]'` returns the unevaluated
      // wrapper `OutputForm[<expr>]`; preserve it. Graphics/Graphics3D
      // arguments still collapse to the abbreviated `-Graphics-` /
      // `-Graphics3D-` form inside the wrapper, matching wolframscript.
      let inner = match &args[0] {
        Expr::FunctionCall { name: head, .. } if head == "Graphics" => {
          Expr::Raw("-Graphics-".to_string())
        }
        Expr::FunctionCall { name: head, .. } if head == "Graphics3D" => {
          Expr::Raw("-Graphics3D-".to_string())
        }
        other => other.clone(),
      };
      return Some(Ok(Expr::FunctionCall {
        name: "OutputForm".to_string(),
        args: vec![inner].into(),
      }));
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
      return Some(Ok(unevaluated("RawBoxes", args)));
    }
    // DisplayForm[boxes] — wrapper that causes box expressions to be rendered visually.
    // It stays unevaluated; the rendering pipeline handles it like RawBoxes.
    "DisplayForm" if args.len() == 1 => {
      return Some(Ok(unevaluated("DisplayForm", args)));
    }
    // Low-level typesetting box constructors — these are inert and return themselves.
    "FractionBox" if args.len() == 2 || args.len() == 3 => {
      return Some(Ok(unevaluated(name, args)));
    }
    "SqrtBox" if args.len() == 1 || args.len() == 2 => {
      return Some(Ok(unevaluated(name, args)));
    }
    "SuperscriptBox" if args.len() == 2 => {
      return Some(Ok(unevaluated(name, args)));
    }
    "SubscriptBox" if args.len() == 2 => {
      return Some(Ok(unevaluated(name, args)));
    }
    "SubsuperscriptBox" if args.len() == 3 => {
      return Some(Ok(unevaluated(name, args)));
    }
    "OverscriptBox" if args.len() == 2 || args.len() == 3 => {
      return Some(Ok(unevaluated(name, args)));
    }
    "UnderscriptBox" if args.len() == 2 || args.len() == 3 => {
      return Some(Ok(unevaluated(name, args)));
    }
    "UnderoverscriptBox" if args.len() == 3 || args.len() == 4 => {
      return Some(Ok(unevaluated(name, args)));
    }
    "RadicalBox" if args.len() == 2 || args.len() == 3 => {
      return Some(Ok(unevaluated(name, args)));
    }
    "FrameBox" if !args.is_empty() => {
      return Some(Ok(unevaluated(name, args)));
    }
    "StyleBox" if !args.is_empty() => {
      return Some(Ok(unevaluated(name, args)));
    }
    "GridBox" if !args.is_empty() => {
      return Some(Ok(unevaluated(name, args)));
    }
    "TagBox" if args.len() == 2 => {
      return Some(Ok(unevaluated(name, args)));
    }
    "InterpretationBox" if args.len() == 2 => {
      return Some(Ok(unevaluated(name, args)));
    }
    // FormBox[content, form] — inert box wrapper tagging the
    // content with a form (e.g. TraditionalForm). Stays unevaluated.
    "FormBox" if args.len() == 2 => {
      return Some(Ok(unevaluated(name, args)));
    }
    // Area[region] — compute the area of a geometric region
    "Area" if args.len() == 1 => {
      return Some(compute_area(strip_region_wrapper(&args[0])));
    }
    // Volume[region] — n-dimensional measure of a geometric region
    "Volume" if args.len() == 1 => {
      return Some(compute_volume(strip_region_wrapper(&args[0])));
    }
    // SurfaceArea[region] — total boundary area of a 3-D solid
    "SurfaceArea" if args.len() == 1 => {
      return Some(compute_surface_area(strip_region_wrapper(&args[0])));
    }
    // RegionMeasure[region] — n-dim measure for explicit regions
    "RegionMeasure" if args.len() == 1 => {
      return Some(compute_region_measure(strip_region_wrapper(&args[0])));
    }
    // RegionMoment[region, {i1, …, in}] — the polynomial moment
    // Integrate[x1^i1 ⋯ xn^in, region]
    "RegionMoment" if args.len() == 2 => {
      return Some(compute_region_moment(
        strip_region_wrapper(&args[0]),
        &args[1],
      ));
    }
    // MomentOfInertia[reg] / [reg, pt] / [reg, pt, v]
    "MomentOfInertia" if (1..=3).contains(&args.len()) => {
      return Some(compute_moment_of_inertia(
        strip_region_wrapper(&args[0]),
        args,
      ));
    }
    "RegionMember" if args.len() == 2 => {
      return Some(compute_region_member(
        strip_region_wrapper(&args[0]),
        &args[1],
      ));
    }
    "RegionDisjoint" => {
      return Some(compute_region_disjoint(args));
    }
    "RegionDistance" if args.len() == 2 => {
      return Some(compute_region_distance(
        strip_region_wrapper(&args[0]),
        &args[1],
      ));
    }
    "RegionNearest" if args.len() == 2 => {
      return Some(compute_region_nearest(
        strip_region_wrapper(&args[0]),
        &args[1],
      ));
    }
    "SignedRegionDistance" if args.len() == 2 => {
      return Some(compute_signed_region_distance(
        strip_region_wrapper(&args[0]),
        &args[1],
      ));
    }
    "FindShortestCurve" if args.len() == 3 => {
      return Some(compute_find_shortest_curve(&args[0], &args[1], &args[2]));
    }
    "ShortestCurveDistance" if args.len() == 3 => {
      return Some(compute_shortest_curve_distance(
        &args[0], &args[1], &args[2],
      ));
    }
    "RegionBounds" if args.len() == 1 => {
      return Some(compute_region_bounds(strip_region_wrapper(&args[0])));
    }
    "RegionDimension" if args.len() == 1 => {
      return Some(compute_region_dimension(strip_region_wrapper(&args[0])));
    }
    "RegionEmbeddingDimension" if args.len() == 1 => {
      return Some(compute_region_embedding_dimension(strip_region_wrapper(
        &args[0],
      )));
    }
    "RegionCentroid" if args.len() == 1 => {
      return Some(compute_region_centroid(strip_region_wrapper(&args[0])));
    }
    "ArcLength" if args.len() == 1 => {
      return Some(compute_arc_length(strip_region_wrapper(&args[0])));
    }
    "ArcLength" if args.len() == 2 => {
      return Some(compute_arc_length_curve(&args[0], &args[1], args));
    }
    "Perimeter" if args.len() == 1 => {
      return Some(compute_perimeter(strip_region_wrapper(&args[0])));
    }
    "Insphere" if args.len() == 1 => {
      return Some(compute_insphere(&args[0]));
    }
    "Circumsphere" if args.len() == 1 => {
      return Some(compute_circumsphere(&args[0]));
    }
    // AngleBisector[{q1, p, q2}] — the interior-angle bisector at p, as an
    // InfiniteLine through p (2-D points only).
    "AngleBisector" if args.len() == 1 => {
      return Some(compute_angle_bisector(&args[0]));
    }
    // PerpendicularBisector[{p1, p2}] — the perpendicular bisector of the
    // segment, as an InfiniteLine through its midpoint (2-D points only).
    "PerpendicularBisector" if args.len() == 1 => {
      return Some(compute_perpendicular_bisector(&args[0]));
    }
    // TriangleCenter[tri] / TriangleCenter[tri, ctype] — named triangle center
    "TriangleCenter" if args.len() == 1 || args.len() == 2 => {
      return Some(compute_triangle_center(args));
    }
    // TriangleMeasurement[tri] / TriangleMeasurement[tri, prop] — scalar
    // triangle measurements (Area default, Perimeter, radii, …)
    "TriangleMeasurement" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::math_ast::triangle_measurement_ast(args));
    }
    // BoundingRegion[pts] — smallest axis-aligned bounding box of a point list:
    // Rectangle for 2D points, Cuboid for 1D or >=3D. Named-method and region
    // forms are left unevaluated.
    "BoundingRegion" if args.len() == 1 => {
      return Some(compute_bounding_region(&args[0]));
    }
    "RegionWithin" if args.len() == 2 => {
      return Some(region_within(
        strip_region_wrapper(&args[0]),
        strip_region_wrapper(&args[1]),
        args,
      ));
    }
    // PlanarAngle is a planar (2-D) construct: every point must be a pair, and
    // higher-dimensional or malformed input stays unevaluated (matching
    // wolframscript, which leaves e.g. 3-D points unevaluated).
    "PlanarAngle" if args.len() == 1 => {
      let is_2d = |e: &Expr| matches!(e, Expr::List(c) if c.len() == 2);
      // PlanarAngle[{q1, p, q2}] — angle at the middle vertex p.
      if let Expr::List(pts) = &args[0]
        && pts.len() == 3
        && pts.iter().all(&is_2d)
      {
        return Some(compute_planar_angle(&pts[0], &pts[1], &pts[2]));
      }
      // PlanarAngle[p -> {q1, q2}] — angle at p between the half-lines through
      // q1 and q2.
      if let Expr::Rule {
        pattern,
        replacement,
      } = &args[0]
        && is_2d(pattern)
        && let Expr::List(rays) = replacement.as_ref()
        && rays.len() == 2
        && rays.iter().all(is_2d)
      {
        return Some(compute_planar_angle(&rays[0], pattern, &rays[1]));
      }
      return Some(Ok(unevaluated("PlanarAngle", args)));
    }
    // PolygonCoordinates[poly] — the vertices of a 2-D polygon in canonical
    // (sorted) order. Degenerate (zero-area / collinear) polygons and non-2-D
    // vertex lists are left unevaluated, matching wolframscript.
    "PolygonCoordinates" if args.len() == 1 => {
      return Some(polygon_coordinates(strip_region_wrapper(&args[0]), args));
    }
    // PolygonAngle[poly] — interior angles at each vertex;
    // PolygonAngle[poly, vertex] — interior angle at one vertex.
    "PolygonAngle" if args.len() == 1 || args.len() == 2 => {
      return Some(compute_polygon_angle(args));
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
          return Some(Ok(unevaluated("QBinomial", args)));
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
                    args: vec![q.clone(), Expr::Integer(n - i)].into(),
                  },
                ]
                .into(),
              },
            ]
            .into(),
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
                    args: vec![q.clone(), Expr::Integer(i + 1)].into(),
                  },
                ]
                .into(),
              },
            ]
            .into(),
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
                    args: vec![den, Expr::Integer(-1)].into(),
                  },
                ]
                .into(),
              },
            ]
            .into(),
          };
        }
        return Some(crate::evaluator::evaluate_expr_to_expr(&result));
      }
      return Some(Ok(unevaluated("QBinomial", args)));
    }
    // RegionEqual[r1, r2, ...] — test whether regions are equal
    "RegionEqual" => {
      return Some(compute_region_equal(args));
    }
    // FindSequenceFunction[list, var] — find a formula for an integer sequence
    "FindSequenceFunction" if args.len() == 2 => {
      return Some(find_sequence_function(&args[0], &args[1]));
    }
    // Operator form: `FindSequenceFunction[seq]` returns a pure function such
    // that `FindSequenceFunction[seq][k]` is the k-th term. We build the
    // formula over a temporary variable and rewrite it to a `#1`-slot
    // function so it composes with Map etc.
    "FindSequenceFunction" if args.len() == 1 => {
      let formula = match find_sequence_function(
        &args[0],
        &Expr::Identifier("\u{f3a7}fsfvar".to_string()),
      ) {
        Ok(f) => f,
        Err(e) => return Some(Err(e)),
      };
      // If the formula couldn't be found it comes back as the unevaluated
      // 2-arg call; keep the operator form unevaluated in that case.
      if let Expr::FunctionCall {
        name: fname,
        args: _,
      } = &formula
        && fname == "FindSequenceFunction"
      {
        return Some(Ok(unevaluated("FindSequenceFunction", args)));
      }
      let body = crate::syntax::substitute_variable(
        &formula,
        "\u{f3a7}fsfvar",
        &Expr::Slot(1),
      );
      return Some(Ok(Expr::Function {
        body: Box::new(body),
      }));
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
    // Inactivate[expr] wraps every head H in `expr` with Inactive[H];
    // Inactivate[expr, h] only wraps heads named `h`. The first argument is
    // held (HoldFirst), so e.g. Inactivate[1 + 1] = Inactive[Plus][1, 1].
    "Inactivate" if args.len() == 1 || args.len() == 2 => {
      let filter: Option<String> = if args.len() == 2 {
        match &args[1] {
          Expr::Identifier(h) => Some(h.clone()),
          // A list of heads is rejected by Wolfram (Inactivate::sympatt);
          // keep the call unevaluated to mirror that non-result.
          _ => {
            return Some(Ok(unevaluated("Inactivate", args)));
          }
        }
      } else {
        None
      };
      return Some(Ok(inactivate_expr(&args[0], filter.as_deref())));
    }
    // Form wrappers -- keep as wrappers (matching wolframscript OutputForm behavior)
    "MathMLForm" | "StandardForm" | "InputForm" if !args.is_empty() => {
      return Some(Ok(unevaluated(name, args)));
    }
    // TraditionalForm[expr] — keep as-is (formatting wrapper, not eagerly evaluated)
    "TraditionalForm" if args.len() == 1 => {
      return Some(Ok(unevaluated("TraditionalForm", args)));
    }
    // Format[expr] / Format[expr, fmt] — look up user-defined rules in
    // FORMAT_VALUES (registered by `Format[pat] := body` and
    // `Format[pat, fmt] := body`). 1-arg rules (form-name = "") apply
    // to every form. With no matching rule the call is transparent and
    // returns the inner expression unchanged.
    "Format" if !args.is_empty() => {
      if let Expr::FunctionCall { name: head, .. } = &args[0] {
        let target_form = if args.len() >= 2 {
          if let Expr::Identifier(form) = &args[1] {
            Some(form.clone())
          } else {
            None
          }
        } else {
          None
        };
        let rules = crate::evaluator::assignment::FORMAT_VALUES
          .with(|m| m.borrow().get(head).cloned().unwrap_or_default());
        // Two-phase lookup: when a target form is given, prefer rules
        // tagged with that form, then fall back to 1-arg rules (empty
        // form name). Without a target form, only consider 1-arg rules.
        let phases: Vec<&str> = match &target_form {
          Some(t) => vec![t.as_str(), ""],
          None => vec![""],
        };
        for phase in phases {
          for (rule_form, lhs, rhs) in &rules {
            if rule_form != phase {
              continue;
            }
            if let Some(bindings) =
              crate::evaluator::pattern_matching::match_pattern(&args[0], lhs)
            {
              let substituted =
                bindings.iter().fold(rhs.clone(), |acc, (k, v)| {
                  crate::syntax::substitute_variable(&acc, k, v)
                });
              return Some(crate::evaluator::evaluate_expr_to_expr(
                &substituted,
              ));
            }
          }
        }
      }
      // No user rule matched. With `Format[expr, OutputForm]`, wolframscript
      // returns the 2D ASCII rendering of the expression.
      if args.len() == 2
        && matches!(&args[1], Expr::Identifier(f) if f == "OutputForm")
      {
        let rendered = crate::syntax::expr_to_output_form_2d(&args[0]);
        return Some(Ok(Expr::Raw(rendered)));
      }
      return Some(Ok(args[0].clone()));
    }

    // MeanAround[{x1, ..., xn}] = Around[N[Mean], N[StandardDeviation/Sqrt[n]]]
    "MeanAround" if args.len() == 1 => {
      let Expr::List(items) = &args[0] else {
        return Some(Ok(unevaluated("MeanAround", args)));
      };
      let n = items.len();
      if n == 0 {
        return Some(Ok(unevaluated("MeanAround", args)));
      }
      // Mean
      let mean_expr = Expr::FunctionCall {
        name: "Mean".to_string(),
        args: vec![args[0].clone()].into(),
      };
      let mean_n = Expr::FunctionCall {
        name: "N".to_string(),
        args: vec![mean_expr].into(),
      };
      // StandardError = StandardDeviation / Sqrt[n]
      let std_dev = Expr::FunctionCall {
        name: "StandardDeviation".to_string(),
        args: vec![args[0].clone()].into(),
      };
      let sqrt_n = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(n as i128)].into(),
      };
      let std_err = binop(BinaryOperator::Divide, std_dev, sqrt_n);
      let std_err_n = Expr::FunctionCall {
        name: "N".to_string(),
        args: vec![std_err].into(),
      };
      let mean_val = crate::evaluator::evaluate_expr_to_expr(&mean_n).ok();
      let err_val = crate::evaluator::evaluate_expr_to_expr(&std_err_n).ok();
      match (mean_val, err_val) {
        (Some(m), Some(e)) => {
          return Some(Ok(Expr::FunctionCall {
            name: "Around".to_string(),
            args: vec![m, e].into(),
          }));
        }
        _ => {
          return Some(Ok(unevaluated("MeanAround", args)));
        }
      }
    }

    // Around[dist] — an approximate number from a distribution:
    // Around[N[Mean[dist]], N[StandardDeviation[dist]]].
    // Around[Interval[{a, b}]] treats the interval as a uniform distribution:
    // Around[(a+b)/2, (b-a)/(2 Sqrt[3])].
    "Around" if args.len() == 1 => {
      let unevaluated = || Some(Ok(unevaluated("Around", args)));
      return match &args[0] {
        Expr::FunctionCall {
          name: iname,
          args: iargs,
        } if iname == "Interval" && iargs.len() == 1 => {
          if let Expr::List(bounds) = &iargs[0]
            && bounds.len() == 2
            && let (Some(a), Some(b)) = (
              crate::functions::math_ast::try_eval_to_f64(&bounds[0]),
              crate::functions::math_ast::try_eval_to_f64(&bounds[1]),
            )
          {
            let value = (a + b) / 2.0;
            // Around[Interval[{a, b}]] uses the interval half-width as the
            // uncertainty (wolframscript), not the uniform-distribution std.
            let uncertainty = (b - a).abs() / 2.0;
            if uncertainty == 0.0 {
              // Degenerate interval collapses to its exact bound value.
              return Some(Ok(bounds[0].clone()));
            }
            return Some(Ok(Expr::FunctionCall {
              name: "Around".to_string(),
              args: vec![Expr::Real(value), Expr::Real(uncertainty)].into(),
            }));
          }
          unevaluated()
        }
        Expr::FunctionCall { name: dname, .. }
          if dname.ends_with("Distribution") =>
        {
          let numeric_of = |stat: &str| -> Option<f64> {
            let expr = Expr::FunctionCall {
              name: "N".to_string(),
              args: vec![Expr::FunctionCall {
                name: stat.to_string(),
                args: vec![args[0].clone()].into(),
              }]
              .into(),
            };
            let value = crate::evaluator::evaluate_expr_to_expr(&expr).ok()?;
            crate::functions::math_ast::try_eval_to_f64(&value)
          };
          if let (Some(mean), Some(sd)) =
            (numeric_of("Mean"), numeric_of("StandardDeviation"))
          {
            if sd == 0.0 {
              return Some(Ok(Expr::Real(mean)));
            }
            return Some(Ok(Expr::FunctionCall {
              name: "Around".to_string(),
              args: vec![Expr::Real(mean), Expr::Real(sd)].into(),
            }));
          }
          unevaluated()
        }
        _ => unevaluated(),
      };
    }

    // Around[value, uncertainty] — convert integer value to real when uncertainty is real
    // Around[{v1, v2, …}, u] threads over the list of central values, giving
    // each the same uncertainty: {Around[v1, u], Around[v2, u], …}.
    "Around" if args.len() == 2 && matches!(&args[0], Expr::List(_)) => {
      let Expr::List(values) = &args[0] else {
        unreachable!();
      };
      let results: Result<Vec<Expr>, InterpreterError> = values
        .iter()
        .map(|v| {
          crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "Around".to_string(),
            args: vec![v.clone(), args[1].clone()].into(),
          })
        })
        .collect();
      return Some(results.map(|r| Expr::List(r.into())));
    }
    "Around" if args.len() >= 2 => {
      let mut new_args = args.to_vec();
      // Around[x, Scaled[s]] → Around[x, x*s] when both numeric.
      if let Expr::FunctionCall {
        name: sname,
        args: sargs,
      } = &new_args[1]
        && sname == "Scaled"
        && sargs.len() == 1
      {
        let x_num = crate::functions::math_ast::try_eval_to_f64(&new_args[0]);
        let s_num = crate::functions::math_ast::try_eval_to_f64(&sargs[0]);
        if let (Some(x), Some(s)) = (x_num, s_num) {
          new_args[1] = Expr::Real(x.abs() * s);
          if matches!(&new_args[0], Expr::Integer(_)) {
            new_args[0] = Expr::Real(x);
          }
          return Some(Ok(Expr::FunctionCall {
            name: "Around".to_string(),
            args: new_args.into(),
          }));
        }
      }
      // A zero uncertainty collapses to the bare value (Around[5, 0] -> 5,
      // Around[5., 0.] -> 5.), matching wolframscript. The asymmetric
      // Around[x, {0, 0}] collapses the same way.
      fn is_zero_num(e: &Expr) -> bool {
        matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(z) if *z == 0.0)
      }
      let zero_uncertainty = new_args.len() == 2
        && match &new_args[1] {
          Expr::List(items) => {
            items.len() == 2 && items.iter().all(is_zero_num)
          }
          other => is_zero_num(other),
        };
      if zero_uncertainty {
        return Some(Ok(new_args[0].clone()));
      }
      // Around stores its central value and uncertainty as machine reals, so
      // exact arguments are promoted to Real: Around[5, 1] -> Around[5., 1.],
      // Around[3/4, 1/8] -> Around[0.75, 0.125] (matching wolframscript). The
      // components of an asymmetric {δ₋, δ₊} uncertainty are promoted the same
      // way. Symbolic arguments are left untouched.
      fn promote_to_real(a: &mut Expr) {
        match a {
          Expr::Integer(n) => *a = Expr::Real(*n as f64),
          Expr::FunctionCall { name, args }
            if name == "Rational" && args.len() == 2 =>
          {
            if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1])
              && *d != 0
            {
              *a = Expr::Real(*n as f64 / *d as f64);
            }
          }
          _ => {}
        }
      }
      for a in new_args.iter_mut() {
        if let Expr::List(items) = a {
          if items.len() == 2 {
            let mut promoted = items.to_vec();
            for item in promoted.iter_mut() {
              promote_to_real(item);
            }
            *a = Expr::List(promoted.into());
          }
        } else {
          promote_to_real(a);
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Around".to_string(),
        args: new_args.into(),
      }));
    }

    // Symbolic operators with no built-in meaning -- just return as-is with evaluated args
    "Therefore" | "Because" | "TableForm" | "MatrixForm" | "Row" | "Grid"
    | "TextGrid" | "Column" | "Framed" | "Highlighted" => {
      return Some(Ok(unevaluated(name, args)));
    }

    // `In[k]` references prior input. With no input history (script mode),
    // `In[-N]` resolves to `In[$Line - N]`, which clamps to `In[0]` for any
    // non-positive index. Positive indices stay unevaluated as `In[k]`.
    "In" if args.len() == 1 => {
      if let Expr::Integer(k) = &args[0]
        && *k <= 0
      {
        return Some(Ok(Expr::FunctionCall {
          name: "In".to_string(),
          args: vec![Expr::Integer(0)].into(),
        }));
      }
      return Some(Ok(unevaluated("In", args)));
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
      return Some(Ok(unevaluated("Default", args)));
    }

    // Default[f, n] / Default[f, n, m] — when the user hasn't installed a
    // specific multi-argument Default[f, n] DownValue, fall through to the
    // single-argument Default[f] (matching Wolfram's lookup chain).
    "Default" if (args.len() == 2 || args.len() == 3) => {
      if let Expr::Identifier(sym) = &args[0] {
        // First try the position-less form.
        let one_arg_call = Expr::FunctionCall {
          name: "Default".to_string(),
          args: vec![args[0].clone()].into(),
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
          && let Some(val) =
            builtin_default_value_at_position(sym, *pos as usize)
        {
          return Some(Ok(val));
        }
      }
      return Some(Ok(unevaluated("Default", args)));
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
/// Parse a `MaxIterations -> n` option argument into an iteration count.
/// Accepts a non-negative integer (clamped to >= 0) or `Infinity`, which maps
/// to the default safety cap. Returns `None` for any other shape.
fn extract_max_iterations(opt: &Expr) -> Option<usize> {
  let (lhs, rhs) = match opt {
    Expr::Rule {
      pattern,
      replacement,
    } => (pattern.as_ref(), replacement.as_ref()),
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => (pattern.as_ref(), replacement.as_ref()),
    Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2 =>
    {
      (&args[0], &args[1])
    }
    _ => return None,
  };
  match lhs {
    Expr::Identifier(s) if s == "MaxIterations" => {}
    _ => return None,
  }
  match rhs {
    Expr::Integer(n) if *n >= 0 => Some(*n as usize),
    Expr::Constant(c) if c == "Infinity" => {
      Some(crate::evaluator::pattern_matching::REPLACE_REPEATED_DEFAULT_MAX)
    }
    Expr::Identifier(s) if s == "Infinity" => {
      Some(crate::evaluator::pattern_matching::REPLACE_REPEATED_DEFAULT_MAX)
    }
    _ => None,
  }
}

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
  use crate::evaluator::pattern_matching::get_expr_head;
  // Parse each pattern slot into (name, min, max, head-constraint).
  // `max == None` means unbounded — consume as many as available.
  struct Slot<'a> {
    name: &'a str,
    min: usize,
    max: Option<usize>,
    head: Option<&'a str>,
  }
  // Map a blank_type (1=Blank, 2=BlankSequence, 3=BlankNullSequence) to
  // (min, max).
  let bounds = |blank_type: u8| -> Option<(usize, Option<usize>)> {
    match blank_type {
      1 => Some((1, Some(1))),
      2 => Some((1, None)),
      3 => Some((0, None)),
      _ => None,
    }
  };
  let mut slots: Vec<Slot> = Vec::with_capacity(pattern_elems.len());
  for p in pattern_elems {
    let slot: Slot = match p {
      Expr::Pattern {
        name,
        blank_type,
        head,
      } => {
        let (min, max) = bounds(*blank_type)?;
        Slot {
          name: name.as_str(),
          min,
          max,
          head: head.as_deref(),
        }
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
        let (blank_kind, head) = parse_blank_fc(&fargs[1])?;
        let (min, max) = bounds(blank_kind)?;
        Slot {
          name,
          min,
          max,
          head,
        }
      }
      _ => {
        let (blank_kind, head) = parse_blank_fc(p)?;
        let (min, max) = bounds(blank_kind)?;
        Slot {
          name: "",
          min,
          max,
          head,
        }
      }
    };
    slots.push(slot);
  }

  // Brute-force enumeration: each slot takes between `min` and `max` elements.
  let n = expr_elems.len();
  let k = slots.len();
  if k == 0 {
    return if n == 0 {
      Some(vec![vec![]])
    } else {
      Some(vec![])
    };
  }
  let total_min: usize = slots.iter().map(|s| s.min).sum();
  if total_min > n {
    return Some(vec![]);
  }

  // Recurse over slot counts, respecting per-slot max bounds and head
  // constraints on the consumed elements.
  let mut results: Vec<Vec<(String, Expr)>> = Vec::new();
  let mut counts = vec![0usize; k];
  #[allow(clippy::too_many_arguments)]
  fn recurse(
    idx: usize,
    pos: usize,
    counts: &mut Vec<usize>,
    slots: &[Slot<'_>],
    expr_elems: &[Expr],
    results: &mut Vec<Vec<(String, Expr)>>,
  ) {
    let n = expr_elems.len();
    if idx == slots.len() {
      if pos != n {
        return;
      }
      // Build the binding set from the chosen counts.
      let mut p = 0;
      let mut bindings: Vec<(String, Expr)> = Vec::new();
      for (i, slot) in slots.iter().enumerate() {
        let len = counts[i];
        let slice = &expr_elems[p..p + len];
        p += len;
        if !slot.name.is_empty() {
          let value = if len == 1 {
            slice[0].clone()
          } else {
            unevaluated("Sequence", slice)
          };
          bindings.push((slot.name.to_string(), value));
        }
      }
      results.push(bindings);
      return;
    }
    let slot = &slots[idx];
    // Remaining minimum required by later slots.
    let later_min: usize = slots[idx + 1..].iter().map(|s| s.min).sum();
    let avail = n - pos;
    let max_take = avail.saturating_sub(later_min);
    let upper = match slot.max {
      Some(m) => m.min(max_take),
      None => max_take,
    };
    for take in slot.min..=upper {
      // Verify the head constraint on every element this slot consumes.
      if let Some(h) = slot.head
        && !expr_elems[pos..pos + take]
          .iter()
          .all(|e| get_expr_head(e) == h)
      {
        continue;
      }
      counts[idx] = take;
      recurse(idx + 1, pos + take, counts, slots, expr_elems, results);
    }
    counts[idx] = 0;
  }
  recurse(0, 0, &mut counts, &slots, expr_elems, &mut results);
  Some(results)
}

/// Parse a blank-like expression into (blank_type, head-constraint).
/// Returns `None` for anything that is not a Blank/BlankSequence/
/// BlankNullSequence (with an optional single head argument).
fn parse_blank_fc(expr: &Expr) -> Option<(u8, Option<&str>)> {
  match expr {
    Expr::Pattern {
      blank_type, head, ..
    } => Some((*blank_type, head.as_deref())),
    Expr::FunctionCall { name, args } => {
      let kind = match name.as_str() {
        "Blank" => 1u8,
        "BlankSequence" => 2,
        "BlankNullSequence" => 3,
        _ => return None,
      };
      let head = match args.first() {
        None => None,
        Some(Expr::Identifier(s)) => Some(s.as_str()),
        Some(_) => return None,
      };
      if args.len() > 1 {
        return None;
      }
      Some((kind, head))
    }
    _ => None,
  }
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

  // Build association fields as parallel lists: textual key -> value lines
  // (for the InputForm echo) and structured field entries (for the Grid SVG
  // card, which can render hyperlinks for fields with an associated URL).
  use crate::functions::information_render::InfoField;
  let mut fields: Vec<String> = Vec::new();
  let mut display_fields: Vec<InfoField> = Vec::new();
  fields.push(format!("Name -> {}", sym));
  display_fields.push(InfoField::text("Name", sym));
  if !description.is_empty() {
    fields.push(format!("Usage -> {}", description));
    display_fields.push(InfoField::text("Usage", description));
  }

  // Attributes are always shown — they affect evaluation semantics and are
  // cheap to display, so users shouldn't need `??sym` to see them.
  let attrs_repr = if all_attrs.is_empty() {
    "{}".to_string()
  } else {
    format!("{{{}}}", all_attrs.join(", "))
  };
  fields.push(format!("Attributes -> {}", attrs_repr));
  display_fields.push(InfoField::text("Attributes", &attrs_repr));

  // Documentation URL — only when the symbol has a public docs page.
  // Display strips the `https://` prefix; the SVG card keeps the full URL
  // as the anchor target so the link remains clickable.
  if let Some(url) = crate::evaluator::get_doc_url(sym) {
    let display_url = url.trim_start_matches("https://").to_string();
    fields.push(format!("Documentation -> {}", display_url));
    display_fields.push(InfoField::link("Documentation", &display_url, url));
  }

  if is_full {
    // Full output: include all fields
    fields.push("ObjectType -> Symbol".to_string());
    display_fields.push(InfoField::text("ObjectType", "Symbol"));

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
      display_fields
        .push(InfoField::text("Options", format!("{{{}}}", opts_str)));
    } else {
      fields.push("Options -> {}".to_string());
      display_fields.push(InfoField::text("Options", "{}"));
    }

    fields.push(format!("FullName -> System`{}", sym));
    display_fields.push(InfoField::text("FullName", format!("System`{}", sym)));
  }

  // Capture a graphical SVG card alongside the textual InputForm result.
  if let Some(svg) =
    crate::functions::information_render::render_information_card_svg(
      sym,
      &display_fields,
    )
  {
    crate::capture_graphics(&svg);
  }

  let _ = is_full; // is_full controls which fields appear above; the
  // wrapper is single-arg in wolframscript regardless.
  let result_str = format!("InformationData[<|{}|>]", fields.join(", "));
  Expr::Raw(result_str)
}

/// Look up `sym::usage` in the global MessageName DownValues, returning the
/// stored string body if any. Used by `Information[sym]` to populate the
/// Usage field.
fn lookup_usage_message(sym: &str) -> Option<String> {
  let func_defs = crate::FUNC_DEFS
    .with(|m| m.borrow().get("MessageName").cloned().unwrap_or_default());
  for (params, conds, _defaults, _heads, _blank_types, body) in &func_defs {
    if params.len() != 2 {
      continue;
    }
    let mut slot0_match = false;
    let mut slot1_match = false;
    for c in conds {
      if let Some(Expr::Comparison {
        operands,
        operators,
      }) = c
        && operators.len() == 1
        && matches!(operators[0], ComparisonOp::SameQ)
        && operands.len() == 2
        && let Expr::Identifier(pname) = &operands[0]
      {
        if pname == &params[0]
          && matches!(&operands[1], Expr::Identifier(s) if s == sym)
        {
          slot0_match = true;
        } else if pname == &params[1]
          && matches!(&operands[1], Expr::String(s) if s == "usage")
        {
          slot1_match = true;
        }
      }
    }
    if slot0_match
      && slot1_match
      && let Expr::String(text) = body
    {
      return Some(text.clone());
    }
  }
  None
}

/// Format the stored UpValues for `sym` as the `Information`InformationValueForm`
/// wrapper Wolfram emits in `Information[sym]`. Returns `None` when no
/// UpValues are stored.
fn format_upvalues_field(sym: &str) -> Option<String> {
  let entries = crate::UPVALUES.with(|m| m.borrow().get(sym).cloned())?;
  if entries.is_empty() {
    return None;
  }
  let rules: Vec<String> = entries
    .iter()
    .map(
      |(_outer, _params, _conds, _defs, _heads, _body, lhs, body)| {
        format!("{} :> {}", expr_to_string(lhs), expr_to_string(body))
      },
    )
    .collect();
  Some(format!(
    "Information`InformationValueForm[UpValues, {}, {{{}}}]",
    sym,
    rules.join(", ")
  ))
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
        let items_expr: Vec<(Expr, Expr)> = items
          .iter()
          .map(|(k, v)| {
            let key_expr = crate::syntax::string_to_expr(k)
              .unwrap_or(Expr::Identifier(k.clone()));
            (key_expr, v.clone())
          })
          .collect();
        expr_to_string(&Expr::Association(items_expr))
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
              && operators.iter().any(|op| matches!(op, ComparisonOp::SameQ))
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

  // Usage: prefer the user-installed `sym::usage` message, falling back
  // to the bare context-qualified name `Global` `sym` that wolframscript
  // shows when no usage has been set.
  let usage_str = match lookup_usage_message(sym) {
    Some(text) => text,
    None => format!("Global`{}", sym),
  };

  // UpValues from `g[…sym…] ^:= …` style assignments.
  let up_str = format_upvalues_field(sym).unwrap_or_else(|| "None".to_string());

  // Capture a graphical SVG card alongside the textual InputForm result.
  // The card surfaces only the fields that carry information (skipping
  // the long list of `None` placeholders that bloat the textual form).
  use crate::functions::information_render::InfoField;
  let mut display_fields: Vec<InfoField> = Vec::new();
  display_fields.push(InfoField::text("ObjectType", "Symbol"));
  display_fields.push(InfoField::text("Usage", &usage_str));
  if own_str != "None" {
    display_fields.push(InfoField::text("OwnValues", &own_str));
  }
  if up_str != "None" {
    display_fields.push(InfoField::text("UpValues", &up_str));
  }
  if down_str != "None" {
    display_fields.push(InfoField::text("DownValues", &down_str));
  }
  display_fields.push(InfoField::text("Attributes", &attrs_str));
  display_fields.push(InfoField::text("FullName", format!("Global`{}", sym)));
  if let Some(svg) =
    crate::functions::information_render::render_information_card_svg(
      sym,
      &display_fields,
    )
  {
    crate::capture_graphics(&svg);
  }

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
       Usage -> {usage_str}, \
       Documentation -> None, \
       OwnValues -> {own_str}, \
       UpValues -> {up_str}, \
       DownValues -> {down_str}, \
       SubValues -> None, \
       DefaultValues -> None, \
       NValues -> None, \
       FormatValues -> None, \
       Options -> {opts_str}, \
       Attributes -> {attrs_str}, \
       FullName -> Global`{sym}|>]"
    );
    Expr::Raw(result_str)
  } else {
    let result_str = format!(
      "InformationData[<|ObjectType -> Symbol, \
       Usage -> {usage_str}, \
       Documentation -> None, \
       OwnValues -> {own_str}, \
       UpValues -> {up_str}, \
       DownValues -> {down_str}, \
       SubValues -> None, \
       DefaultValues -> None, \
       NValues -> None, \
       FormatValues -> None, \
       Options -> None, \
       Attributes -> {attrs_str}, \
       FullName -> Global`{sym}|>]"
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
    args: vec![Expr::List(
      vec![
        Expr::String("(".to_string()),
        inner,
        Expr::String(")".to_string()),
      ]
      .into(),
    )]
    .into(),
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
    Expr::BinaryOp { op, .. } => {
      matches!(op, BinaryOperator::Plus | BinaryOperator::Minus)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
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
          args: args.to_vec().into(),
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

/// Convert an expression to its FullForm box-tree representation.
/// Atoms become bare strings (negative integers `-n` become
/// `RowBox[{"-", n}]`); compound expressions render as
/// `RowBox[{"<Head>", "[", RowBox[{<arg1>, ",", <arg2>, …}], "]"}]`,
/// with operator BinaryOp/UnaryOp first rewritten to their canonical
/// FunctionCall head (Plus / Times / Power; Subtract → Plus[a, Times[-1,
/// b]]; Divide → Times[a, Power[b, -1]]; UnaryMinus → Times[-1, x]).
fn expr_to_full_box_form(expr: &Expr) -> Expr {
  // Atoms.
  match expr {
    Expr::Integer(n) => {
      if *n < 0 {
        return Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(
            vec![
              Expr::String("-".to_string()),
              // `unsigned_abs` avoids overflow when `n == i128::MIN`
              // (e.g. `-2^127`), where `-n` has no representable value.
              Expr::String(n.unsigned_abs().to_string()),
            ]
            .into(),
          )]
          .into(),
        };
      }
      return Expr::String(n.to_string());
    }
    Expr::BigInteger(n) => {
      let s = n.to_string();
      if let Some(rest) = s.strip_prefix('-') {
        return Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(
            vec![
              Expr::String("-".to_string()),
              Expr::String(rest.to_string()),
            ]
            .into(),
          )]
          .into(),
        };
      }
      return Expr::String(s);
    }
    Expr::Real(f) => {
      let text = format!("{}`", crate::syntax::format_real(f.abs()));
      if *f < 0.0 {
        return Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(
            vec![Expr::String("-".to_string()), Expr::String(text)].into(),
          )]
          .into(),
        };
      }
      return Expr::String(text);
    }
    Expr::BigFloat(digits, prec) => {
      let text = crate::syntax::format_bigfloat(digits, *prec);
      if let Some(without_minus) = text.strip_prefix('-') {
        return Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(
            vec![
              Expr::String("-".to_string()),
              Expr::String(without_minus.to_string()),
            ]
            .into(),
          )]
          .into(),
        };
      }
      return Expr::String(text);
    }
    Expr::Identifier(s) | Expr::Constant(s) => {
      return Expr::String(s.clone());
    }
    Expr::String(s) => {
      // Strings render with explicit quotes inside FullForm.
      return Expr::String(format!("\"{}\"", s));
    }
    _ => {}
  }
  // Reduce operator forms to canonical FunctionCall (head, args).
  let (head, args): (String, Vec<Expr>) = match expr {
    Expr::FunctionCall { name, args } => (name.clone(), args.to_vec()),
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => ("Plus".to_string(), vec![*left.clone(), *right.clone()]),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => ("Times".to_string(), vec![*left.clone(), *right.clone()]),
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => ("Power".to_string(), vec![*left.clone(), *right.clone()]),
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => (
      "Times".to_string(),
      vec![
        *left.clone(),
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![*right.clone(), Expr::Integer(-1)].into(),
        },
      ],
    ),
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => (
      "Plus".to_string(),
      vec![
        *left.clone(),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), *right.clone()].into(),
        },
      ],
    ),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => (
      "Times".to_string(),
      vec![Expr::Integer(-1), *operand.clone()],
    ),
    _ => return Expr::String(crate::syntax::expr_to_string(expr)),
  };
  let mut parts: Vec<Expr> =
    vec![Expr::String(head), Expr::String("[".to_string())];
  if args.len() == 1 {
    parts.push(expr_to_full_box_form(&args[0]));
  } else if args.len() >= 2 {
    let mut inner = Vec::with_capacity(args.len() * 2 - 1);
    for (i, arg) in args.iter().enumerate() {
      if i > 0 {
        inner.push(Expr::String(",".to_string()));
      }
      inner.push(expr_to_full_box_form(arg));
    }
    parts.push(Expr::FunctionCall {
      name: "RowBox".to_string(),
      args: vec![Expr::List(inner.into())].into(),
    });
  }
  parts.push(Expr::String("]".to_string()));
  Expr::FunctionCall {
    name: "RowBox".to_string(),
    args: vec![Expr::List(parts.into())].into(),
  }
}

/// Convert an expression to its box form representation for TraditionalForm/StandardForm.
/// Walk `expr` bottom-up and apply any user-defined `Format[head[…]]`
/// rules so subexpressions surface in their formatted shape. Used by
/// `MakeBoxes[OutputForm[expr], …]` so the printed text matches what
/// the OutputForm renderer would show after Format substitution.
pub fn apply_format_recursively(expr: &Expr, target_form: &str) -> Expr {
  let recursed = match expr {
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| apply_format_recursively(a, target_form))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|i| apply_format_recursively(i, target_form))
        .collect(),
    ),
    _ => expr.clone(),
  };
  if let Expr::FunctionCall { name: head, .. } = &recursed {
    let has_format = crate::evaluator::assignment::FORMAT_VALUES
      .with(|m| m.borrow().contains_key(head));
    if has_format {
      let format_call = Expr::FunctionCall {
        name: "Format".to_string(),
        args: vec![recursed.clone(), Expr::Identifier(target_form.to_string())]
          .into(),
      };
      if let Ok(formatted) =
        crate::evaluator::evaluate_expr_to_expr(&format_call)
      {
        let unchanged = matches!(
          &formatted,
          Expr::FunctionCall { name, args }
            if name == "Format"
            && (args.len() == 1 || args.len() == 2)
            && crate::evaluator::pattern_matching::expr_equal(&args[0], &recursed)
        );
        if !unchanged {
          return formatted;
        }
      }
    }
  }
  recursed
}

/// Box a sub-expression while honoring any user-defined `Format[…]`
/// or `MakeBoxes[…]` DownValues. `Format` is checked first — when it
/// is defined, the formatted result is what gets boxed (this matches
/// wolframscript: `Format` definitions take precedence over user
/// `MakeBoxes` definitions). With neither rule defined, the
/// evaluator's default branch falls back to `expr_to_box_form`, so
/// untouched sub-expressions render the same as before.
fn box_subexpr_via_user_rules(expr: &Expr) -> Expr {
  if let Expr::FunctionCall { name: head, .. } = expr {
    let has_format_rule = crate::evaluator::assignment::FORMAT_VALUES
      .with(|m| m.borrow().contains_key(head));
    if has_format_rule {
      let format_call = Expr::FunctionCall {
        name: "Format".to_string(),
        args: vec![expr.clone(), Expr::Identifier("StandardForm".to_string())]
          .into(),
      };
      if let Ok(formatted) =
        crate::evaluator::evaluate_expr_to_expr(&format_call)
      {
        // The Format dispatcher returns `args[0].clone()` when no user
        // rule matches; only treat the result as a real substitution
        // when it actually differs from the input.
        let is_format_wrapper = matches!(
          &formatted,
          Expr::FunctionCall { name, args }
            if name == "Format"
              && (args.len() == 1 || args.len() == 2)
              && crate::evaluator::pattern_matching::expr_equal(&args[0], expr)
        );
        let unchanged = is_format_wrapper
          || crate::evaluator::pattern_matching::expr_equal(&formatted, expr);
        if !unchanged {
          return expr_to_box_form(&formatted);
        }
      }
    }
  }
  let has_user_rule =
    crate::FUNC_DEFS.with(|m| m.borrow().contains_key("MakeBoxes"));
  if !has_user_rule {
    return expr_to_box_form(expr);
  }
  let call = Expr::FunctionCall {
    name: "MakeBoxes".to_string(),
    args: vec![expr.clone(), Expr::Identifier("StandardForm".to_string())]
      .into(),
  };
  match crate::evaluator::evaluate_expr_to_expr(&call) {
    Ok(result) => result,
    Err(_) => expr_to_box_form(expr),
  }
}

/// Box form of a `ScientificForm`/`EngineeringForm`/`NumberForm` call, matching
/// wolframscript's
/// `TagBox[InterpretationBox[StyleBox[<inner>, ShowStringCharacters -> False],
/// <value>, AutoDelete -> True], <Head>]`. Returns `None` for heads/arguments
/// these forms leave symbolic, so the caller renders the literal function call.
fn number_display_form_box(name: &str, args: &[Expr]) -> Option<Expr> {
  use crate::functions::string_ast;
  // Default precision (n) for a missing second argument is 6 significant
  // figures; an explicit integer second argument overrides it.
  let precision = |arg: Option<&Expr>| -> Option<i64> {
    match arg {
      None => Some(6),
      Some(Expr::Integer(n)) => Some(*n as i64),
      _ => None,
    }
  };
  // Decompose a single numeric value into its (mantissa, exponent) parts for
  // the requested form. Shared by the scalar and list (threaded) paths.
  let parts_for = |val: &Expr| -> Option<(String, Option<i64>)> {
    Some(match name {
      "ScientificForm" if args.len() == 1 || args.len() == 2 => {
        string_ast::scientific_form_parts(val, precision(args.get(1))?)?
      }
      "EngineeringForm" if args.len() == 1 || args.len() == 2 => {
        string_ast::engineering_form_parts(val, precision(args.get(1))?)?
      }
      "NumberForm" if args.len() == 1 => string_ast::number_form_parts(val, 6)?,
      "NumberForm" if args.len() == 2 => match &args[1] {
        Expr::Integer(n) => string_ast::number_form_parts(val, *n as i64)?,
        Expr::List(spec) if spec.len() == 2 => match &spec[1] {
          Expr::Integer(f) => {
            string_ast::number_form_fixed_parts(val, *f as i64)?
          }
          _ => return None,
        },
        _ => return None,
      },
      _ => return None,
    })
  };
  // These forms are Listable in wolframscript: a list argument is rendered as a
  // braced, comma-separated row of per-element display boxes, all wrapped in a
  // single `TagBox[..., Head]`.
  let first = args.first()?;
  if let Expr::List(items) = first {
    let inner_boxes: Option<Vec<Expr>> = items
      .iter()
      .map(|v| {
        let (mantissa, exp) = parts_for(v)?;
        Some(number_display_inner_box(
          scientific_value_box(&mantissa, exp),
          v,
        ))
      })
      .collect();
    return Some(wrap_list_number_display_box(inner_boxes?, name));
  }
  let (mantissa, exp) = parts_for(first)?;
  Some(wrap_number_display_box(
    scientific_value_box(&mantissa, exp),
    first,
    name,
  ))
}

/// Inner box for a decomposed number-display form: `String(mantissa)` when there
/// is no `× 10^e` factor, otherwise
/// `RowBox[{mantissa, " × ", SuperscriptBox["10", exp]}]`.
fn scientific_value_box(mantissa: &str, exp: Option<i64>) -> Expr {
  match exp {
    None => Expr::String(mantissa.to_string()),
    Some(e) => Expr::FunctionCall {
      name: "RowBox".to_string(),
      args: vec![Expr::List(
        vec![
          Expr::String(mantissa.to_string()),
          Expr::String(" \u{00d7} ".to_string()),
          Expr::FunctionCall {
            name: "SuperscriptBox".to_string(),
            args: vec![
              Expr::String("10".to_string()),
              Expr::String(e.to_string()),
            ]
            .into(),
          },
        ]
        .into(),
      )]
      .into(),
    },
  }
}

/// Per-value display box (no `TagBox` head wrapper):
/// `InterpretationBox[StyleBox[inner, ShowStringCharacters -> False], value,
/// AutoDelete -> True]`. The interpretation/style layers are rendering
/// pass-throughs for the SVG layout engine; they preserve the link to the
/// original numeric `value`.
fn number_display_inner_box(inner: Expr, value: &Expr) -> Expr {
  let rule = |k: &str, v: &str| Expr::Rule {
    pattern: Box::new(Expr::Identifier(k.to_string())),
    replacement: Box::new(Expr::Identifier(v.to_string())),
  };
  let style = Expr::FunctionCall {
    name: "StyleBox".to_string(),
    args: vec![inner, rule("ShowStringCharacters", "False")].into(),
  };
  Expr::FunctionCall {
    name: "InterpretationBox".to_string(),
    args: vec![style, value.clone(), rule("AutoDelete", "True")].into(),
  }
}

/// Wrap a scalar inner display box with wolframscript's
/// `TagBox[InterpretationBox[StyleBox[inner, ShowStringCharacters -> False],
/// value, AutoDelete -> True], head]`, preserving the form head for `MakeBoxes`.
fn wrap_number_display_box(inner: Expr, value: &Expr, head: &str) -> Expr {
  Expr::FunctionCall {
    name: "TagBox".to_string(),
    args: vec![
      number_display_inner_box(inner, value),
      Expr::Identifier(head.to_string()),
    ]
    .into(),
  }
}

/// Wrap the per-element display boxes of a list argument with wolframscript's
/// `TagBox[RowBox[{"{", RowBox[{elem, ",", elem, ...}], "}"}], head]`.
fn wrap_list_number_display_box(elems: Vec<Expr>, head: &str) -> Expr {
  let mut row = Vec::with_capacity(elems.len().saturating_mul(2));
  for (i, elem) in elems.into_iter().enumerate() {
    if i > 0 {
      row.push(Expr::String(",".to_string()));
    }
    row.push(elem);
  }
  let inner_row = Expr::FunctionCall {
    name: "RowBox".to_string(),
    args: vec![Expr::List(row.into())].into(),
  };
  let braced = Expr::FunctionCall {
    name: "RowBox".to_string(),
    args: vec![Expr::List(
      vec![
        Expr::String("{".to_string()),
        inner_row,
        Expr::String("}".to_string()),
      ]
      .into(),
    )]
    .into(),
  };
  Expr::FunctionCall {
    name: "TagBox".to_string(),
    args: vec![braced, Expr::Identifier(head.to_string())].into(),
  }
}

pub fn expr_to_box_form(expr: &Expr) -> Expr {
  // MakeBoxes has HoldAllComplete, so a postfix `expr // f` arg is
  // delivered as `Expr::Postfix { expr, func }` rather than being
  // normalised to `FunctionCall { name: f, args: [expr] }`. The
  // form-specific branches below match on FunctionCall heads (e.g.
  // `FullForm[…]`), so unwrap any Postfix chain first so
  // `MakeBoxes[a-b // FullForm]` and `MakeBoxes[FullForm[a-b]]`
  // produce identical TagBox/StyleBox output.
  if let Expr::Postfix { expr: inner, func } = expr
    && let Expr::Identifier(fname) = func.as_ref()
  {
    let normalised = Expr::FunctionCall {
      name: fname.clone(),
      args: vec![(**inner).clone()].into(),
    };
    return expr_to_box_form(&normalised);
  }
  // Number-display forms (ScientificForm/EngineeringForm/NumberForm) render as
  // 2D `mantissa × 10^exp` (or plain) notation rather than as a literal
  // function call. This is what the Playground/Studio SVG output and
  // `MakeBoxes` both consume. Falls through to the generic rendering for
  // arguments these forms leave symbolic.
  if let Expr::FunctionCall { name, args } = expr
    && let Some(boxed) = number_display_form_box(name, args)
  {
    return boxed;
  }
  match expr {
    // wolframscript: positive integers render as a single String
    // (`"14"`); negative integers decompose into
    // `RowBox[{"-", "14"}]` (the sign is its own token).
    Expr::Integer(n) => {
      if *n < 0 {
        Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(
            vec![
              Expr::String("-".to_string()),
              // `unsigned_abs` avoids overflow when `n == i128::MIN`
              // (e.g. `-2^127`), where `-n` has no representable value.
              Expr::String(n.unsigned_abs().to_string()),
            ]
            .into(),
          )]
          .into(),
        }
      } else {
        Expr::String(n.to_string())
      }
    }
    Expr::BigInteger(n) => {
      if *n < 0i128.into() {
        Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(
            vec![
              Expr::String("-".to_string()),
              Expr::String((-n.clone()).to_string()),
            ]
            .into(),
          )]
          .into(),
        }
      } else {
        Expr::String(n.to_string())
      }
    }
    Expr::Real(f) => {
      // Machine-precision floats render with a backtick precision
      // marker in MakeBoxes output: plain magnitude gets a
      // trailing `` ` `` (e.g. `2.``), but scientific notation
      // places the `` ` `` between the mantissa and the `*^`
      // exponent (e.g. `3.4`*^10`, not `3.4*^10``). Negative
      // values decompose into `RowBox[{"-", "abs_text"}]`.
      let abs_text = crate::syntax::format_real(f.abs());
      let text = if let Some(idx) = abs_text.find("*^") {
        format!("{}`{}", &abs_text[..idx], &abs_text[idx..])
      } else {
        format!("{}`", abs_text)
      };
      if *f < 0.0 {
        Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(
            vec![Expr::String("-".to_string()), Expr::String(text)].into(),
          )]
          .into(),
        }
      } else {
        Expr::String(text)
      }
    }
    Expr::BigFloat(digits, prec) => {
      // Same sign-decomposition rule as Expr::Real for
      // precision-tagged big-float literals.
      let text = crate::syntax::format_bigfloat(digits, *prec);
      if let Some(without_minus) = text.strip_prefix('-') {
        Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(
            vec![
              Expr::String("-".to_string()),
              Expr::String(without_minus.to_string()),
            ]
            .into(),
          )]
          .into(),
        }
      } else {
        Expr::String(text)
      }
    }
    Expr::Identifier(s) | Expr::Constant(s) => Expr::String(s.clone()),
    Expr::String(s) => Expr::String(format!("\"{}\"", s)),
    // Part[expr, indices…] (Expr::Part / chained Part) →
    //   RowBox[{<head>, 〚, <i1>, ",", <i2>, …, 〛}]
    // wolframscript uses the Unicode double-bracket glyphs
    // (U+301A `〚` / U+301B `〛`).
    Expr::Part { expr, index } => {
      // Walk a chain of nested `Part`s so a[[1, 2, 3]] flattens
      // back into a single bracket pair. wolframscript wraps a
      // multi-index list in a nested `RowBox[{1, ",", 2, ",", 3}]`
      // and uses a single-token inside the outer RowBox when only
      // one index is present.
      let mut indices: Vec<&Expr> = vec![index.as_ref()];
      let mut head_expr: &Expr = expr.as_ref();
      while let Expr::Part {
        expr: inner,
        index: inner_idx,
      } = head_expr
      {
        indices.insert(0, inner_idx.as_ref());
        head_expr = inner.as_ref();
      }
      let inner_box = if indices.len() == 1 {
        expr_to_box_form(indices[0])
      } else {
        let mut inner_args: Vec<Expr> = Vec::with_capacity(indices.len() * 2);
        for (i, idx) in indices.iter().enumerate() {
          if i > 0 {
            inner_args.push(Expr::String(",".to_string()));
          }
          inner_args.push(expr_to_box_form(idx));
        }
        Expr::FunctionCall {
          name: "RowBox".to_string(),
          args: vec![Expr::List(inner_args.into())].into(),
        }
      };
      let row_args: Vec<Expr> = vec![
        expr_to_box_form(head_expr),
        Expr::String("\u{301A}".to_string()),
        inner_box,
        Expr::String("\u{301B}".to_string()),
      ];
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(row_args.into())].into(),
      }
    }
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
        UnaryOperator::Minus => "-",
        UnaryOperator::Not => "!",
      };
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(
          vec![Expr::String(op_str.to_string()), expr_to_box_form(operand)]
            .into(),
        )]
        .into(),
      }
    }
    // BinaryOp::Plus/Minus/Times/And/Or/StringJoin/Alternatives
    Expr::BinaryOp { op, left, right }
      if !matches!(op, BinaryOperator::Power | BinaryOperator::Divide) =>
    {
      let (op_str, spaced) = match op {
        BinaryOperator::Plus => ("+", true),
        BinaryOperator::Minus => ("-", true),
        BinaryOperator::Times => (" ", false),
        BinaryOperator::And => ("&&", true),
        BinaryOperator::Or => ("||", true),
        BinaryOperator::StringJoin => ("<>", false),
        BinaryOperator::Alternatives => ("|", true),
        BinaryOperator::Power | BinaryOperator::Divide => unreachable!(),
      };
      let sep = if spaced {
        op_str.to_string()
      } else {
        op_str.to_string()
      };
      // Additive operands of a product need parens so `(-5+n) (-4+n)`
      // doesn't render as `-5+n -4+n` (issue #135).
      let box_operand = |e: &Expr| {
        if matches!(op, BinaryOperator::Times) {
          box_with_paren_if_needed(e)
        } else {
          expr_to_box_form(e)
        }
      };
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(
          vec![box_operand(left), Expr::String(sep), box_operand(right)].into(),
        )]
        .into(),
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
          ComparisonOp::Equal => "==",
          ComparisonOp::NotEqual => "!=",
          ComparisonOp::Less => "<",
          ComparisonOp::LessEqual => "<=",
          ComparisonOp::Greater => ">",
          ComparisonOp::GreaterEqual => ">=",
          ComparisonOp::SameQ => "===",
          ComparisonOp::UnsameQ => "=!=",
        };
        parts.push(Expr::String(op_str.to_string()));
        parts.push(expr_to_box_form(&operands[i + 1]));
      }
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts.into())].into(),
      }
    }
    // Rule: pattern -> replacement
    Expr::Rule {
      pattern,
      replacement,
    } => {
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(
          vec![
            expr_to_box_form(pattern),
            Expr::String("\u{f522}".to_string()), // Mathematica's Rule arrow
            expr_to_box_form(replacement),
          ]
          .into(),
        )]
        .into(),
      }
    }
    // RuleDelayed: pattern :> replacement
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(
          vec![
            expr_to_box_form(pattern),
            Expr::String("\u{f51f}".to_string()), // Mathematica's RuleDelayed arrow
            expr_to_box_form(replacement),
          ]
          .into(),
        )]
        .into(),
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
          args: vec![Expr::List(
            vec![
              expr_to_box_form(k),
              Expr::String("\u{f522}".to_string()),
              expr_to_box_form(v),
            ]
            .into(),
          )]
          .into(),
        });
      }
      parts.push(Expr::String("|>".to_string()));
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts.into())].into(),
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
        args: vec![Expr::List(parts.into())].into(),
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
          // Check for negative BigInteger coefficient: Times[-n, x] where the
          // coefficient overflows i128 (e.g. 2^129). Mirrors the Integer arm
          // above so `1 + -2^129 x` folds to `1 - 2^129 x`.
          if let Expr::FunctionCall { name: tn, args: ta } = arg
            && tn == "Times"
            && ta.len() >= 2
            && matches!(&ta[0], Expr::BigInteger(n) if n.sign() == num_bigint::Sign::Minus)
          {
            parts.push(Expr::String("-".to_string()));
            let mut pos_args = ta.clone();
            if let Expr::BigInteger(n) = &ta[0] {
              pos_args[0] = Expr::BigInteger(-n);
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
        args: vec![Expr::List(parts.into())].into(),
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
          args: vec![Expr::List(
            vec![
              Expr::String("-".to_string()),
              box_with_paren_if_needed(&args[1]),
            ]
            .into(),
          )]
          .into(),
        };
      }
      let rest = unevaluated("Times", &args[1..]);
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(
          vec![
            Expr::String("-".to_string()),
            paren_box(expr_to_box_form(&rest)),
          ]
          .into(),
        )]
        .into(),
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
          args: vec![Expr::List(
            vec![
              Expr::String("-".to_string()),
              Expr::FunctionCall {
                name: "RowBox".to_string(),
                args: vec![Expr::List(
                  vec![
                    expr_to_box_form(&pos_n),
                    Expr::String(" ".to_string()),
                    expr_to_box_form(&args[1]),
                  ]
                  .into(),
                )]
                .into(),
              },
            ]
            .into(),
          )]
          .into(),
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
          args: vec![expr_to_box_form(&num), expr_to_box_form(&den)].into(),
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
        args: vec![Expr::List(parts.into())].into(),
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
            args: vec![expr_to_box_form(&args[0])].into(),
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
                args: vec![expr_to_box_form(&args[0])].into(),
              },
            ]
            .into(),
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
          ]
          .into(),
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
        ]
        .into(),
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Subscript" && args.len() == 2 =>
    {
      Expr::FunctionCall {
        name: "SubscriptBox".to_string(),
        args: vec![expr_to_box_form(&args[0]), expr_to_box_form(&args[1])]
          .into(),
      }
    }
    expr if is_sqrt(expr).is_some() => {
      let sqrt_arg = is_sqrt(expr).unwrap();
      Expr::FunctionCall {
        name: "SqrtBox".to_string(),
        args: vec![expr_to_box_form(sqrt_arg)].into(),
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      // Rational[n, d] → FractionBox[n, d]
      Expr::FunctionCall {
        name: "FractionBox".to_string(),
        args: vec![expr_to_box_form(&args[0]), expr_to_box_form(&args[1])]
          .into(),
      }
    }
    // BinaryOp::Divide → FractionBox
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => Expr::FunctionCall {
      name: "FractionBox".to_string(),
      args: vec![expr_to_box_form(left), expr_to_box_form(right)].into(),
    },
    // BinaryOp::Power with rational exponents
    Expr::BinaryOp {
      op: BinaryOperator::Power,
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
            args: vec![expr_to_box_form(left)].into(),
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
                args: vec![expr_to_box_form(left)].into(),
              },
            ]
            .into(),
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
          ]
          .into(),
        };
      }
      Expr::FunctionCall {
        name: "SuperscriptBox".to_string(),
        args: vec![box_with_paren_if_needed(left), expr_to_box_form(right)]
          .into(),
      }
    }
    // List → RowBox[{"{", RowBox[{elem, ",", elem, ...}], "}"}]
    Expr::List(items) => {
      let mut parts = Vec::new();
      parts.push(Expr::String("{".to_string()));
      if !items.is_empty() {
        if items.len() == 1 {
          parts.push(box_subexpr_via_user_rules(&items[0]));
        } else {
          let mut inner = Vec::new();
          for (i, item) in items.iter().enumerate() {
            if i > 0 {
              inner.push(Expr::String(",".to_string()));
            }
            inner.push(box_subexpr_via_user_rules(item));
          }
          parts.push(Expr::FunctionCall {
            name: "RowBox".to_string(),
            args: vec![Expr::List(inner.into())].into(),
          });
        }
      }
      parts.push(Expr::String("}".to_string()));
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts.into())].into(),
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
        args: vec![Expr::List(parts.into())].into(),
      }
    }
    // FullForm[expr] inside MakeBoxes: TagBox[StyleBox[<full-form-boxes>,
    // ShowSpecialCharacters -> False, ShowStringCharacters -> True,
    // NumberMarks -> True], FullForm]. The inner content uses head[args]
    // notation (RowBox of "Head", "[", arguments separated by ",", "]")
    // even for built-in operators like Plus/Times/Power/Divide.
    Expr::FunctionCall { name, args }
      if name == "FullForm" && args.len() == 1 =>
    {
      let full_box = expr_to_full_box_form(&args[0]);
      Expr::FunctionCall {
        name: "TagBox".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "StyleBox".to_string(),
            args: vec![
              full_box,
              Expr::Rule {
                pattern: Box::new(Expr::Identifier(
                  "ShowSpecialCharacters".to_string(),
                )),
                replacement: Box::new(bool_expr(false)),
              },
              Expr::Rule {
                pattern: Box::new(Expr::Identifier(
                  "ShowStringCharacters".to_string(),
                )),
                replacement: Box::new(bool_expr(true)),
              },
              Expr::Rule {
                pattern: Box::new(Expr::Identifier("NumberMarks".to_string())),
                replacement: Box::new(bool_expr(true)),
              },
            ]
            .into(),
          },
          Expr::Identifier("FullForm".to_string()),
        ]
        .into(),
      }
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
    // StandardForm[expr] / TraditionalForm[expr]:
    //   TagBox[FormBox[<inner-box>, <form>], <form>, Editable -> True]
    // The inner content uses [/] brackets in StandardForm and
    // (/) parentheses in TraditionalForm for function calls.
    Expr::FunctionCall { name, args }
      if (name == "StandardForm" || name == "TraditionalForm")
        && args.len() == 1 =>
    {
      let inner_box = if name == "TraditionalForm" {
        expr_to_box_form_traditional(&args[0])
      } else {
        expr_to_box_form(&args[0])
      };
      Expr::FunctionCall {
        name: "TagBox".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "FormBox".to_string(),
            args: vec![inner_box, Expr::Identifier(name.clone())].into(),
          },
          Expr::Identifier(name.clone()),
          Expr::Rule {
            pattern: Box::new(Expr::Identifier("Editable".to_string())),
            replacement: Box::new(bool_expr(true)),
          },
        ]
        .into(),
      }
    }
    // Format[expr] / Format[expr, form] → wolframscript boxes
    // these as
    //   TagBox[FormBox[<inner-box>, <form>], <tag>]
    // where <form> defaults to StandardForm and <tag> is the
    // bare symbol `Format` for the 1-arg form, or `#1 &` for the
    // 2-arg form. Only StandardForm and TraditionalForm route
    // through this branch; OutputForm/InputForm/FullForm in the
    // form-slot fall through to the surrounding handlers.
    Expr::FunctionCall { name, args }
      if name == "Format"
        && (args.len() == 1
          || (args.len() == 2
            && matches!(
              &args[1],
              Expr::Identifier(s)
                if s == "StandardForm" || s == "TraditionalForm"
            ))) =>
    {
      let form_name = if args.len() == 2 {
        if let Expr::Identifier(s) = &args[1] {
          s.clone()
        } else {
          "StandardForm".to_string()
        }
      } else {
        "StandardForm".to_string()
      };
      let inner_box = expr_to_box_form(&args[0]);
      let form_box = Expr::FunctionCall {
        name: "FormBox".to_string(),
        args: vec![inner_box, Expr::Identifier(form_name)].into(),
      };
      let tag = if args.len() == 1 {
        Expr::Identifier("Format".to_string())
      } else {
        // `#1 &` — an anonymous Function with body Slot(1).
        Expr::Function {
          body: Box::new(Expr::Slot(1)),
        }
      };
      Expr::FunctionCall {
        name: "TagBox".to_string(),
        args: vec![form_box, tag].into(),
      }
    }
    // CForm/TeXForm/FortranForm: wolframscript wraps the
    // converted text in
    //   InterpretationBox["<text>", <Form>[<orig>],
    //     Editable -> True, AutoDelete -> True]
    // so the rendered string keeps a reference back to the
    // original expression. Match that shape.
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
      // wolframscript bakes one set of quotes into the String
      // content so the rendered text reads `"a-b"` (with quotes)
      // when Woxi's top-level output strips outer String quotes.
      let quoted_text = format!("\"{}\"", text);
      Expr::FunctionCall {
        name: "InterpretationBox".to_string(),
        args: vec![
          Expr::String(quoted_text),
          Expr::FunctionCall {
            name: name.clone(),
            args: vec![args[0].clone()].into(),
          },
          Expr::Rule {
            pattern: Box::new(Expr::Identifier("Editable".to_string())),
            replacement: Box::new(bool_expr(true)),
          },
          Expr::Rule {
            pattern: Box::new(Expr::Identifier("AutoDelete".to_string())),
            replacement: Box::new(bool_expr(true)),
          },
        ]
        .into(),
      }
    }
    // Color specs render via a TemplateBox swatch wrapper:
    //   RGBColor[r, g, b]      → TemplateBox[<|color -> RGBColor[…]|>, RGBColorSwatchTemplate]
    //   GrayLevel[g], Hue[…], CMYKColor[…], XYZColor[…] follow the same
    //   shape but with `<Name>ColorSwatchTemplate` when the head name does
    //   not already end in `Color`.
    Expr::FunctionCall { name, args }
      if matches!(
        name.as_str(),
        "RGBColor" | "Hue" | "GrayLevel" | "CMYKColor" | "XYZColor"
      ) && !args.is_empty() =>
    {
      let template = if name.ends_with("Color") {
        format!("{}SwatchTemplate", name)
      } else {
        format!("{}ColorSwatchTemplate", name)
      };
      let assoc = Expr::Association(vec![(
        Expr::String("color".to_string()),
        expr.clone(),
      )]);
      Expr::FunctionCall {
        name: "TemplateBox".to_string(),
        args: vec![assoc, Expr::String(template)].into(),
      }
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
        args: vec![].into(),
      }
    }
    // MakeBoxes[OutputForm[expr], _]: InterpretationBox[PaneBox[<2D
    // string with literal quotes>, BaselinePosition -> Baseline], <2D
    // text>, Editable -> False]. The 2D rendering uses the existing
    // `expr_to_output_form_2d` helper (spaces for `*`, ASCII fractions,
    // etc.) so e.g. `a + b * c // OutputForm // MakeBoxes` produces the
    // same string the test asserts.
    Expr::FunctionCall { name, args }
      if name == "OutputForm" && args.len() == 1 =>
    {
      // Apply any user-defined `Format[head[…]]` rules to the inner
      // expression bottom-up so e.g. `Format[F[x_]] := {...}` causes
      // `OutputForm[G[F[3.002]]]` to render as the formatted form
      // `G[{Formatted f, {3.002}, Standard}]`.
      let formatted_inner = apply_format_recursively(&args[0], "OutputForm");
      // OutputForm strips the precision-tag backtick from
      // precision-tagged BigFloats and trims the decimal
      // representation to its precision digits — e.g.
      // `OutputForm[3.142`3]` renders as `3.14`, not `3.142`3.`.
      // Walk the expression and rewrite BigFloat leaves to plain
      // Reals truncated to their stored precision before
      // rendering.
      let formatted_inner =
        trim_bigfloat_to_precision_for_output(&formatted_inner);
      // Replace held `Graphics[…]` / `Graphics3D[…]` FunctionCalls
      // with their `-Graphics-` / `-Graphics3D-` placeholder text
      // so wolframscript's text rendering of held graphics inside
      // OutputForm matches.
      let formatted_inner = replace_graphics_with_placeholder(&formatted_inner);
      let output_text = crate::syntax::expr_to_output_form_2d(&formatted_inner);
      // wolframscript's MakeBoxes[OutputForm[expr]] stores the
      // rendered text as a String whose content has literal `"`
      // characters at the start and end. When displayed in
      // script-mode (which strips outer String quotes) this
      // reproduces the visible `"a - b"` quoting.
      let quoted_text = format!("\"{}\"", output_text);
      Expr::FunctionCall {
        name: "InterpretationBox".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "PaneBox".to_string(),
            args: vec![
              Expr::String(quoted_text),
              Expr::Rule {
                pattern: Box::new(Expr::Identifier(
                  "BaselinePosition".to_string(),
                )),
                replacement: Box::new(Expr::Identifier("Baseline".to_string())),
              },
            ]
            .into(),
          },
          // wolframscript's second arg is the wrapped OutputForm
          // expression itself, not the plain text. Graphics/
          // Graphics3D inside should print as their placeholder
          // (`-Graphics-` / `-Graphics3D-`); apply the same
          // substitution we used for the PaneBox text.
          Expr::FunctionCall {
            name: "OutputForm".to_string(),
            args: vec![replace_graphics_with_placeholder(&args[0])].into(),
          },
          Expr::Rule {
            pattern: Box::new(Expr::Identifier("Editable".to_string())),
            replacement: Box::new(bool_expr(false)),
          },
        ]
        .into(),
      }
    }
    // MakeBoxes[InputForm[expr], _]: wolframscript wraps the InputForm
    // string in `InterpretationBox[StyleBox[<input>, ShowStringCharacters
    // -> True, NumberMarks -> True], InputForm[<expr>], Editable -> True,
    // AutoDelete -> True]`. Mirroring that here lets `expr // InputForm //
    // MakeBoxes` round-trip rather than degenerating into the generic
    // `RowBox[{InputForm, [, …, ]}]` form.
    Expr::FunctionCall { name, args }
      if name == "InputForm" && args.len() == 1 =>
    {
      // Apply user-defined `Format[head[…], InputForm]` rules to the
      // inner expression bottom-up so the rendered text reflects the
      // formatted shape (e.g. `Format[F[x_, y_], InputForm] := {…}`
      // turns `InputForm[F[1., "l"]]` into `{"In", GG[…]}` text).
      let formatted_inner = apply_format_recursively(&args[0], "InputForm");
      let inner_str = crate::syntax::expr_to_string(&formatted_inner);
      Expr::FunctionCall {
        name: "InterpretationBox".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "StyleBox".to_string(),
            args: vec![
              Expr::String(inner_str),
              Expr::Rule {
                pattern: Box::new(Expr::Identifier(
                  "ShowStringCharacters".to_string(),
                )),
                replacement: Box::new(bool_expr(true)),
              },
              Expr::Rule {
                pattern: Box::new(Expr::Identifier("NumberMarks".to_string())),
                replacement: Box::new(bool_expr(true)),
              },
            ]
            .into(),
          },
          Expr::FunctionCall {
            name: "InputForm".to_string(),
            args: vec![args[0].clone()].into(),
          },
          Expr::Rule {
            pattern: Box::new(Expr::Identifier("Editable".to_string())),
            replacement: Box::new(bool_expr(true)),
          },
          Expr::Rule {
            pattern: Box::new(Expr::Identifier("AutoDelete".to_string())),
            replacement: Box::new(bool_expr(true)),
          },
        ]
        .into(),
      }
    }
    // Hyperlink[uri] / Hyperlink[label, uri] →
    //   TemplateBox[{<box(label)>, "<uri>"}, "HyperlinkURL"]
    // matching wolframscript's MakeBoxes output. The SVG layout
    // recognizes the HyperlinkURL template and turns the label into a
    // clickable `<a href>` element; consumers that don't (e.g. plain
    // text rendering) fall back gracefully.
    Expr::FunctionCall { name, args }
      if name == "Hyperlink"
        && (args.len() == 1 || args.len() == 2)
        && matches!(
          if args.len() == 1 { &args[0] } else { &args[1] },
          Expr::String(_)
        ) =>
    {
      let (label, uri_str) = if args.len() == 1 {
        let s = match &args[0] {
          Expr::String(s) => s.clone(),
          _ => unreachable!(),
        };
        (Expr::String(s.clone()), s)
      } else {
        let s = match &args[1] {
          Expr::String(s) => s.clone(),
          _ => unreachable!(),
        };
        (args[0].clone(), s)
      };
      Expr::FunctionCall {
        name: "TemplateBox".to_string(),
        args: vec![
          Expr::List(
            vec![expr_to_box_form(&label), Expr::String(uri_str)].into(),
          ),
          Expr::String("HyperlinkURL".to_string()),
        ]
        .into(),
      }
    }
    // General function call f[x, y] → RowBox[{f, "[", RowBox[{x, ",", y}], "]"}]
    Expr::FunctionCall { name, args } => {
      let mut parts = Vec::new();
      parts.push(Expr::String(name.clone()));
      parts.push(Expr::String("[".to_string()));
      if !args.is_empty() {
        if args.len() == 1 {
          parts.push(box_subexpr_via_user_rules(&args[0]));
        } else {
          let mut inner = Vec::new();
          for (i, arg) in args.iter().enumerate() {
            if i > 0 {
              inner.push(Expr::String(",".to_string()));
            }
            inner.push(box_subexpr_via_user_rules(arg));
          }
          parts.push(Expr::FunctionCall {
            name: "RowBox".to_string(),
            args: vec![Expr::List(inner.into())].into(),
          });
        }
      }
      parts.push(Expr::String("]".to_string()));
      Expr::FunctionCall {
        name: "RowBox".to_string(),
        args: vec![Expr::List(parts.into())].into(),
      }
    }
    // Default: use the string representation
    _ => box_as_output_string(expr),
  }
}

/// Trim every BigFloat leaf in `expr` to its stored precision so
/// the OutputForm rendering drops the trailing backtick precision
/// tag and reports only the significant digits. For example
/// `BigFloat("3.142", 3)` → `Real(3.14)`. Used by
/// `MakeBoxes[OutputForm[…]]` so wolframscript's
/// "OutputForm trims digits up to precision" behavior is honored.
/// Walk the expression replacing `Graphics[…]` and `Graphics3D[…]`
/// FunctionCalls with their `-Graphics-` / `-Graphics3D-` text
/// placeholder. wolframscript's `MakeBoxes[OutputForm[Graphics[…]]]`
/// renders the inner graphics as the placeholder string in both
/// the PaneBox text *and* the second InterpretationBox arg (since
/// Graphics-headed FunctionCalls print as the placeholder).
fn replace_graphics_with_placeholder(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } if name == "Graphics" => {
      let _ = args;
      Expr::Raw("-Graphics-".to_string())
    }
    Expr::FunctionCall { name, args } if name == "Graphics3D" => {
      let _ = args;
      Expr::Raw("-Graphics3D-".to_string())
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(replace_graphics_with_placeholder)
        .collect::<Vec<_>>()
        .into(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(replace_graphics_with_placeholder)
        .collect::<Vec<_>>()
        .into(),
    ),
    other => other.clone(),
  }
}

fn trim_bigfloat_to_precision_for_output(expr: &Expr) -> Expr {
  match expr {
    Expr::BigFloat(digits, prec) => {
      // OutputForm shows exactly `prec` significant digits:
      // truncate when the stored digits exceed it, pad with
      // trailing zeros when they fall short. Examples:
      //   BigFloat("3.142", 3) → "3.14"     (truncate)
      //   BigFloat("3.14",  5) → "3.1400"   (pad)
      let prec_target = prec.round().max(1.0) as usize;
      let (sign, rest) = if let Some(r) = digits.strip_prefix('-') {
        ("-", r)
      } else {
        ("", digits.as_str())
      };
      let (int_part, frac_part) = match rest.find('.') {
        Some(dp) => (&rest[..dp], &rest[dp + 1..]),
        None => (rest, ""),
      };
      let int_sig = if int_part == "0" {
        0
      } else {
        int_part.trim_start_matches('0').len()
      };
      let frac_needed = prec_target.saturating_sub(int_sig);
      let formatted = if frac_needed == 0 {
        format!("{}{}.", sign, int_part)
      } else if frac_part.len() >= frac_needed {
        format!("{}{}.{}", sign, int_part, &frac_part[..frac_needed])
      } else {
        format!(
          "{}{}.{}{}",
          sign,
          int_part,
          frac_part,
          "0".repeat(frac_needed - frac_part.len())
        )
      };
      // OutputForm uses `Raw` so the rendered text is the exact
      // padded/truncated decimal — bypasses `format_real`'s
      // default trimming.
      Expr::Raw(formatted)
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(trim_bigfloat_to_precision_for_output)
        .collect::<Vec<_>>()
        .into(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(trim_bigfloat_to_precision_for_output)
        .collect::<Vec<_>>()
        .into(),
    ),
    other => other.clone(),
  }
}

// ── TraditionalForm typesetting helpers ───────────────────────────────
//
// These build the 2D box tree that the Playground/Studio SVG renderer
// consumes so held mathematical expressions display in conventional
// notation: `∑`/`∫`/`∏` operators with limits, invisible HoldForm, `π`/`∞`
// glyphs, `sin(x)` instead of `Sin[x]`, stacked fractions, radicals, and
// bracketed matrices. Only the *display* shape is produced here — no
// evaluation happens.

fn tf_string(s: &str) -> Expr {
  Expr::String(s.to_string())
}

fn tf_row(items: Vec<Expr>) -> Expr {
  if items.len() == 1 {
    return items.into_iter().next().unwrap();
  }
  Expr::FunctionCall {
    name: "RowBox".to_string(),
    args: vec![Expr::List(items.into())].into(),
  }
}

fn tf_box(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

/// Thin space used between implicitly-multiplied factors (`2 a`, `n x`).
/// The SVG text renderer draws a standalone U+2009 as a narrow gap.
fn tf_thin_space() -> Expr {
  Expr::String("\u{2009}".to_string())
}

fn tf_parens(inner: Expr) -> Expr {
  tf_row(vec![tf_string("("), inner, tf_string(")")])
}

/// Map a symbol/constant name to its TraditionalForm glyph. Greek letters
/// entered as `\[Mu]` etc. already arrive as their Unicode character, so
/// only the named mathematical constants need translating here.
fn tf_symbol(name: &str) -> String {
  match name {
    "Pi" => "\u{03C0}".to_string(),       // π
    "Infinity" => "\u{221E}".to_string(), // ∞
    "E" | "ExponentialE" => "\u{2147}".to_string(), // ⅇ
    "I" | "ImaginaryI" => "\u{2148}".to_string(), // ⅈ
    "Degree" => "\u{00B0}".to_string(),   // °
    "EulerGamma" => "\u{03B3}".to_string(), // γ
    "GoldenRatio" => "\u{03C6}".to_string(), // φ
    "ComplexInfinity" => "\u{221E}".to_string(), // ∞
    _ => name.to_string(),
  }
}

/// TraditionalForm display name for a well-known function (lowercase roman),
/// or `None` if the head should be shown verbatim.
fn tf_known_func(name: &str) -> Option<&'static str> {
  Some(match name {
    "Sin" => "sin",
    "Cos" => "cos",
    "Tan" => "tan",
    "Cot" => "cot",
    "Sec" => "sec",
    "Csc" => "csc",
    "Sinh" => "sinh",
    "Cosh" => "cosh",
    "Tanh" => "tanh",
    "ArcSin" => "arcsin",
    "ArcCos" => "arccos",
    "ArcTan" => "arctan",
    "Log" => "log",
    "Log10" => "log",
    "Ln" => "ln",
    "Max" => "max",
    "Min" => "min",
    "Sign" => "sgn",
    "Mod" => "mod",
    "Gcd" => "gcd",
    "Lcm" => "lcm",
    "Gamma" => "\u{0393}", // Γ
    "Zeta" => "\u{03B6}",  // ζ
    _ => return None,
  })
}

/// True for a trig/hyperbolic function whose power is written on the name
/// (`sin²(x)`) rather than around the whole call.
fn tf_is_trig(name: &str) -> bool {
  matches!(
    name,
    "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc" | "Sinh" | "Cosh" | "Tanh"
  )
}

/// Whether a List is a matrix (a non-empty list whose every element is a
/// list of equal length ≥ 1).
fn tf_is_matrix(items: &[Expr]) -> bool {
  if items.is_empty() {
    return false;
  }
  let mut cols = None;
  for it in items {
    match it {
      Expr::List(row) if !row.is_empty() => {
        if *cols.get_or_insert(row.len()) != row.len() {
          return false;
        }
      }
      _ => return false,
    }
  }
  true
}

/// True if `expr` is the literal integer 0 (used to strip the spurious
/// `0 -` the parser emits for a leading unary minus like `-x`).
fn tf_is_zero(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(0))
}

/// Collect the signed terms of an additive expression, flattening nested
/// Plus/Minus and unary minus and dropping literal `0` terms. Each entry is
/// `(is_negative, term)`.
fn tf_flatten_additive(expr: &Expr, neg: bool, out: &mut Vec<(bool, Expr)>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      tf_flatten_additive(left, neg, out);
      tf_flatten_additive(right, neg, out);
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      tf_flatten_additive(left, neg, out);
      tf_flatten_additive(right, !neg, out);
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => tf_flatten_additive(operand, !neg, out),
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for a in args.iter() {
        tf_flatten_additive(a, neg, out);
      }
    }
    _ if tf_is_zero(expr) => {}
    _ => out.push((neg, expr.clone())),
  }
}

/// Flatten a multiplicative expression into its factors.
fn tf_flatten_times(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      tf_flatten_times(left, out);
      tf_flatten_times(right, out);
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args.iter() {
        tf_flatten_times(a, out);
      }
    }
    _ => out.push(expr.clone()),
  }
}

/// Does this term need parentheses when placed as a factor in a product or
/// as the base of a power?
fn tf_needs_paren_factor(expr: &Expr) -> bool {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus,
      ..
    } => true,
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      ..
    } => true,
    Expr::Comparison { .. } => true,
    Expr::FunctionCall { name, args } => {
      (name == "Plus" && args.len() >= 2)
        || (name == "Times"
          && matches!(args.first(), Some(Expr::Integer(n)) if *n < 0))
    }
    _ => false,
  }
}

fn tf_factor_boxed(expr: &Expr) -> Expr {
  let boxed = tf(expr);
  if tf_needs_paren_factor(expr) {
    tf_parens(boxed)
  } else {
    boxed
  }
}

/// TraditionalForm box of `base^exp`, with trig special-casing (`sin²(x)`)
/// and half-integer exponents rendered as radicals.
fn tf_power(base: &Expr, exp: &Expr) -> Expr {
  // sin[x]^2 → sin²(x)
  if let Expr::FunctionCall { name, args } = base
    && tf_is_trig(name)
    && args.len() == 1
    && matches!(exp, Expr::Integer(n) if *n > 0)
  {
    let disp = tf_known_func(name).unwrap_or(name.as_str());
    return tf_row(vec![
      tf_box("SuperscriptBox", vec![tf_string(disp), tf(exp)]),
      tf_string("("),
      tf(&args[0]),
      tf_string(")"),
    ]);
  }
  // x^(1/2) → √x
  let is_half = matches!(exp, Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2
      && matches!(&args[0], Expr::Integer(1))
      && matches!(&args[1], Expr::Integer(2)))
    || matches!(exp, Expr::Real(f) if (*f - 0.5).abs() < f64::EPSILON);
  if is_half {
    return tf_box("SqrtBox", vec![tf(base)]);
  }
  let negative_number = matches!(base, Expr::Integer(n) if *n < 0)
    || matches!(base, Expr::Real(f) if *f < 0.0);
  let base_box = if tf_needs_paren_factor(base)
    || negative_number
    || matches!(
      base,
      Expr::BinaryOp {
        op: BinaryOperator::Power | BinaryOperator::Divide,
        ..
      }
    )
    || matches!(base, Expr::FunctionCall { name, .. } if name == "Power" || name == "Rational")
  {
    tf_parens(tf(base))
  } else {
    tf(base)
  };
  tf_box("SuperscriptBox", vec![base_box, tf(exp)])
}

fn tf_fraction(num: &Expr, den: &Expr) -> Expr {
  tf_box("FractionBox", vec![tf(num), tf(den)])
}

/// TraditionalForm box of a product (already flattened factor list).
fn tf_times(expr: &Expr) -> Expr {
  let mut factors = Vec::new();
  tf_flatten_times(expr, &mut factors);
  // Pull a leading -1 / negative numeric coefficient into a sign.
  let mut negative = false;
  if let Some(Expr::Integer(n)) = factors.first()
    && *n < 0
  {
    negative = true;
    if *n == -1 {
      factors.remove(0);
    } else {
      factors[0] = Expr::Integer(-*n);
    }
  }
  let mut items: Vec<Expr> = Vec::new();
  for (i, f) in factors.iter().enumerate() {
    if i > 0 {
      items.push(tf_thin_space());
    }
    items.push(tf_factor_boxed(f));
  }
  let body = tf_row(items);
  if negative {
    tf_row(vec![tf_string("-"), body])
  } else {
    body
  }
}

/// TraditionalForm box of an additive expression.
fn tf_plus(expr: &Expr) -> Expr {
  let mut terms = Vec::new();
  tf_flatten_additive(expr, false, &mut terms);
  if terms.is_empty() {
    return tf_string("0");
  }
  let mut items: Vec<Expr> = Vec::new();
  for (i, (neg, term)) in terms.iter().enumerate() {
    if i == 0 {
      if *neg {
        items.push(tf_string("-"));
      }
    } else {
      items.push(tf_string(if *neg { "-" } else { "+" }));
    }
    items.push(tf(term));
  }
  tf_row(items)
}

/// Render the iterator body of Sum/Product with the given large operator.
fn tf_big_operator(glyph: &str, args: &[Expr]) -> Expr {
  // Build one Underoverscript operator per iterator spec.
  let mut ops: Vec<Expr> = Vec::new();
  for spec in &args[1..] {
    if let Expr::List(parts) = spec {
      match parts.as_slice() {
        [var, lo, hi] => {
          let under = tf_row(vec![tf(var), tf_string("="), tf(lo)]);
          ops.push(tf_box(
            "UnderoverscriptBox",
            vec![tf_string(glyph), under, tf(hi)],
          ));
        }
        [var, hi] => {
          ops.push(tf_box(
            "UnderoverscriptBox",
            vec![tf_string(glyph), tf(var), tf(hi)],
          ));
        }
        _ => ops.push(tf_string(glyph)),
      }
    } else {
      ops.push(tf_string(glyph));
    }
  }
  if ops.is_empty() {
    ops.push(tf_string(glyph));
  }
  let mut items = ops;
  items.push(tf_thin_space());
  items.push(tf(&args[0]));
  tf_row(items)
}

/// Render `Integrate[body, {x, a, b}, …]` with ∫ operators and ⅆx factors.
fn tf_integrate(args: &[Expr]) -> Expr {
  let mut ops: Vec<Expr> = Vec::new();
  let mut diffs: Vec<Expr> = Vec::new();
  for spec in &args[1..] {
    match spec {
      Expr::List(parts) if parts.len() == 3 => {
        ops.push(tf_box(
          "SubsuperscriptBox",
          vec![tf_string("\u{222B}"), tf(&parts[1]), tf(&parts[2])],
        ));
        diffs.push(tf_string("\u{2146}")); // ⅆ
        diffs.push(tf(&parts[0]));
      }
      Expr::List(parts) if parts.len() == 1 => {
        ops.push(tf_string("\u{222B}"));
        diffs.push(tf_string("\u{2146}"));
        diffs.push(tf(&parts[0]));
      }
      other => {
        ops.push(tf_string("\u{222B}"));
        diffs.push(tf_string("\u{2146}"));
        diffs.push(tf(other));
      }
    }
  }
  if ops.is_empty() {
    return tf_generic_call("Integrate", args);
  }
  let mut items = ops;
  items.push(tf_thin_space());
  let body_needs_paren = matches!(
    &args[0],
    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus,
      ..
    }
  ) || matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Plus");
  if body_needs_paren {
    items.push(tf_parens(tf(&args[0])));
  } else {
    items.push(tf(&args[0]));
  }
  items.push(tf_thin_space());
  items.extend(diffs);
  tf_row(items)
}

/// Render `D[body, x]`, `D[body, {x, n}]`, `D[body, s1, s2, …]` as a
/// Leibniz-style derivative `∂ⁿ/(∂x^a ∂y^b) body`.
fn tf_derivative(args: &[Expr]) -> Expr {
  // Parse each spec into (var, order).
  let mut specs: Vec<(Expr, i128)> = Vec::new();
  for spec in &args[1..] {
    match spec {
      Expr::List(parts) if parts.len() == 2 => {
        let order = match &parts[1] {
          Expr::Integer(n) => *n,
          _ => 1,
        };
        specs.push((parts[0].clone(), order));
      }
      other => specs.push((other.clone(), 1)),
    }
  }
  if specs.is_empty() {
    return tf_generic_call("D", args);
  }
  let total: i128 = specs.iter().map(|(_, n)| *n).sum();
  let partial = "\u{2202}"; // ∂
  let num = if total > 1 {
    tf_box(
      "SuperscriptBox",
      vec![tf_string(partial), tf(&Expr::Integer(total))],
    )
  } else {
    tf_string(partial)
  };
  let mut den_items: Vec<Expr> = Vec::new();
  for (var, order) in &specs {
    den_items.push(tf_string(partial));
    if *order > 1 {
      den_items.push(tf_box(
        "SuperscriptBox",
        vec![tf(var), tf(&Expr::Integer(*order))],
      ));
    } else {
      den_items.push(tf(var));
    }
  }
  let frac = tf_box("FractionBox", vec![num, tf_row(den_items)]);
  let body = &args[0];
  let body_box = if tf_needs_paren_factor(body)
    || matches!(
      body,
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        ..
      }
    )
    || matches!(body, Expr::FunctionCall { name, .. } if name == "Times")
  {
    tf_parens(tf(body))
  } else {
    tf(body)
  };
  tf_row(vec![frac, tf_thin_space(), body_box])
}

/// Build a GridBox from a matrix (list of row-lists), boxing each cell.
fn tf_matrix_grid(rows: &[Expr]) -> Expr {
  let grid_rows: Vec<Expr> = rows
    .iter()
    .map(|row| {
      if let Expr::List(cells) = row {
        Expr::List(cells.iter().map(tf).collect::<Vec<_>>().into())
      } else {
        Expr::List(vec![tf(row)].into())
      }
    })
    .collect();
  tf_box("GridBox", vec![Expr::List(grid_rows.into())])
}

/// Generic `head(arg, …)` rendering (unknown functions). The head is shown
/// verbatim — the constant glyph map (E→ⅇ, I→ⅈ, …) is deliberately *not*
/// applied here, so a symbol used as a function head (e.g. the field `E` in
/// `E[x, y, z]`) stays `E(x, y, z)` rather than becoming `ⅇ(x, y, z)`.
fn tf_generic_call(name: &str, args: &[Expr]) -> Expr {
  let mut items: Vec<Expr> = Vec::with_capacity(args.len() * 2 + 3);
  items.push(tf_string(name));
  items.push(tf_string("("));
  for (i, arg) in args.iter().enumerate() {
    if i > 0 {
      items.push(tf_string(","));
    }
    items.push(tf(arg));
  }
  items.push(tf_string(")"));
  tf_row(items)
}

/// Dispatch a function call to its TraditionalForm rendering.
fn tf_call(name: &str, args: &[Expr]) -> Expr {
  match name {
    "HoldForm" | "HoldComplete" | "Defer" | "Identity" if args.len() == 1 => {
      tf(&args[0])
    }
    "Sqrt" if args.len() == 1 => tf_box("SqrtBox", vec![tf(&args[0])]),
    "Power" if args.len() == 2 => tf_power(&args[0], &args[1]),
    "Rational" if args.len() == 2 => tf_fraction(&args[0], &args[1]),
    "Times" if args.len() >= 2 => tf_times(&unevaluated("Times", args)),
    "Plus" if args.len() >= 2 => tf_plus(&unevaluated("Plus", args)),
    "Divide" if args.len() == 2 => tf_fraction(&args[0], &args[1]),
    "Exp" if args.len() == 1 => {
      tf_box("SuperscriptBox", vec![tf_string("\u{2147}"), tf(&args[0])])
    }
    "Sum" if args.len() >= 2 => tf_big_operator("\u{2211}", args), // ∑
    "Product" if args.len() >= 2 => tf_big_operator("\u{220F}", args), // ∏
    "Integrate" if args.len() >= 2 => tf_integrate(args),
    "D" if args.len() >= 2 => tf_derivative(args),
    "Abs" if args.len() == 1 => {
      tf_row(vec![tf_string("|"), tf(&args[0]), tf_string("|")])
    }
    "Det" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0]
        && tf_is_matrix(rows)
      {
        tf_row(vec![tf_string("|"), tf_matrix_grid(rows), tf_string("|")])
      } else {
        tf_row(vec![tf_string("det"), tf_parens(tf(&args[0]))])
      }
    }
    "MatrixForm" | "TableForm" if args.len() == 1 => {
      if let Expr::List(rows) = &args[0]
        && tf_is_matrix(rows)
      {
        tf_row(vec![tf_string("("), tf_matrix_grid(rows), tf_string(")")])
      } else {
        tf(&args[0])
      }
    }
    // Vector-calculus operators drop their coordinate-list argument and use
    // ∇ notation: Grad → ∇f, Div → ∇·f, Curl → ∇×f, Laplacian → ∇²f.
    "Grad" if args.len() == 2 && matches!(&args[1], Expr::List(_)) => {
      tf_row(vec![tf_string("\u{2207}"), tf(&args[0])])
    }
    "Div" if args.len() == 2 && matches!(&args[1], Expr::List(_)) => {
      tf_row(vec![
        tf_string("\u{2207}"),
        tf_string("\u{22C5}"),
        tf(&args[0]),
      ])
    }
    "Curl" if args.len() == 2 && matches!(&args[1], Expr::List(_)) => {
      tf_row(vec![
        tf_string("\u{2207}"),
        tf_string("\u{00D7}"),
        tf(&args[0]),
      ])
    }
    "Laplacian" if args.len() == 2 && matches!(&args[1], Expr::List(_)) => {
      tf_row(vec![
        tf_box(
          "SuperscriptBox",
          vec![tf_string("\u{2207}"), tf_string("2")],
        ),
        tf(&args[0]),
      ])
    }
    // Limit[body, x -> x0] → lim_{x→x0} body
    "Limit" if args.len() >= 2 => {
      let sub = match &args[1] {
        Expr::Rule {
          pattern,
          replacement,
        } => tf_row(vec![tf(pattern), tf_string("\u{2192}"), tf(replacement)]),
        Expr::FunctionCall { name, args: ra }
          if name == "Rule" && ra.len() == 2 =>
        {
          tf_row(vec![tf(&ra[0]), tf_string("\u{2192}"), tf(&ra[1])])
        }
        other => tf(other),
      };
      let op = tf_box("UnderscriptBox", vec![tf_string("lim"), sub]);
      let body_box = if tf_needs_paren_factor(&args[0]) {
        tf_parens(tf(&args[0]))
      } else {
        tf(&args[0])
      };
      tf_row(vec![op, tf_thin_space(), body_box])
    }
    // Bessel functions render with a subscript order: BesselJ[n, x] → Jₙ(x).
    "BesselJ" | "BesselY" | "BesselI" | "BesselK" if args.len() == 2 => {
      let letter = match name {
        "BesselJ" => "J",
        "BesselY" => "Y",
        "BesselI" => "I",
        _ => "K",
      };
      tf_row(vec![
        tf_box("SubscriptBox", vec![tf_string(letter), tf(&args[0])]),
        tf_string("("),
        tf(&args[1]),
        tf_string(")"),
      ])
    }
    "Subscript" if args.len() >= 2 => {
      let sub = if args.len() == 2 {
        tf(&args[1])
      } else {
        let mut parts: Vec<Expr> = Vec::new();
        for (i, a) in args[1..].iter().enumerate() {
          if i > 0 {
            parts.push(tf_string(","));
          }
          parts.push(tf(a));
        }
        tf_row(parts)
      };
      tf_box("SubscriptBox", vec![tf(&args[0]), sub])
    }
    "Superscript" if args.len() == 2 => {
      tf_box("SuperscriptBox", vec![tf(&args[0]), tf(&args[1])])
    }
    _ => {
      if let Some(disp) = tf_known_func(name) {
        let mut items: Vec<Expr> = vec![tf_string(disp), tf_string("(")];
        for (i, arg) in args.iter().enumerate() {
          if i > 0 {
            items.push(tf_string(","));
          }
          items.push(tf(arg));
        }
        items.push(tf_string(")"));
        tf_row(items)
      } else {
        tf_generic_call(name, args)
      }
    }
  }
}

/// Core TraditionalForm typesetter: expression → 2D box tree.
fn tf(expr: &Expr) -> Expr {
  match expr {
    Expr::Identifier(s) | Expr::Constant(s) => tf_string(&tf_symbol(s)),
    Expr::FunctionCall { name, args } => tf_call(name, args),
    Expr::List(items) => {
      if tf_is_matrix(items) {
        tf_row(vec![tf_string("("), tf_matrix_grid(items), tf_string(")")])
      } else {
        let mut parts: Vec<Expr> = vec![tf_string("{")];
        for (i, it) in items.iter().enumerate() {
          if i > 0 {
            parts.push(tf_string(","));
          }
          parts.push(tf(it));
        }
        parts.push(tf_string("}"));
        tf_row(parts)
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => tf_power(left, right),
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => tf_fraction(left, right),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    } => tf_times(expr),
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } if tf_is_zero(left) => {
      // Parser artifact: leading unary minus `-x` becomes `0 - x`.
      tf_row(vec![tf_string("-"), tf_factor_boxed(right)])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus,
      ..
    } => tf_plus(expr),
    Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } => tf_row(vec![tf(left), tf_string("\u{2227}"), tf(right)]),
    Expr::BinaryOp {
      op: BinaryOperator::Or,
      left,
      right,
    } => tf_row(vec![tf(left), tf_string("\u{2228}"), tf(right)]),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => tf_row(vec![tf_string("-"), tf_factor_boxed(operand)]),
    Expr::UnaryOp {
      op: UnaryOperator::Not,
      operand,
    } => tf_row(vec![tf_string("\u{00AC}"), tf(operand)]),
    Expr::Comparison {
      operands,
      operators,
    } => {
      let mut parts = vec![tf(&operands[0])];
      for (i, op) in operators.iter().enumerate() {
        let sym = match op {
          ComparisonOp::Equal => "=",
          ComparisonOp::NotEqual => "\u{2260}", // ≠
          ComparisonOp::Less => "<",
          ComparisonOp::LessEqual => "\u{2264}", // ≤
          ComparisonOp::Greater => ">",
          ComparisonOp::GreaterEqual => "\u{2265}", // ≥
          ComparisonOp::SameQ => "\u{2261}",        // ≡
          ComparisonOp::UnsameQ => "\u{2262}",      // ≢
        };
        parts.push(tf_string(sym));
        parts.push(tf(&operands[i + 1]));
      }
      tf_row(parts)
    }
    Expr::Rule {
      pattern,
      replacement,
    } => tf_row(vec![tf(pattern), tf_string("\u{2192}"), tf(replacement)]),
    _ => expr_to_box_form(expr),
  }
}

/// Convert an expression to its TraditionalForm box tree for typeset SVG
/// display (Playground / Studio) and for `MakeBoxes[…, TraditionalForm]`.
/// Produces conventional mathematical notation: `∑`/`∫`/`∏` operators with
/// limits, invisible HoldForm, `π`/`∞`/`ⅇ` glyphs, `sin(x)` for `Sin[x]`,
/// stacked fractions, radicals, and bracketed matrices. Unknown functions
/// render as `head(args)`.
pub fn expr_to_box_form_traditional(expr: &Expr) -> Expr {
  tf(expr)
}

/// Convert a Quantity unit expression to box form with proper abbreviations.
/// Uses `unit_to_abbreviation` for known units and handles compound units
/// (division, multiplication, powers).
fn unit_to_box_form(unit: &Expr, magnitude: &Expr) -> Expr {
  use crate::functions::quantity_ast::unit_to_abbreviation;

  // Helper: abbreviate a single unit identifier
  fn abbrev(s: &str, mag: &Expr) -> Expr {
    let abbr = unit_to_abbreviation(s).unwrap_or(s);
    let abbr = crate::syntax::singularize_unit_if_one(mag, abbr);
    Expr::String(abbr)
  }

  // Handle Power in both BinaryOp and FunctionCall form
  if let Some((base, exp)) = crate::functions::graphics::as_power(unit) {
    let base_box = unit_to_box_form_inner(base);
    let exp_box = expr_to_box_form(exp);
    return Expr::FunctionCall {
      name: "SuperscriptBox".to_string(),
      args: vec![base_box, exp_box].into(),
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
      args: vec![Expr::List(
        vec![
          unit_to_box_form_inner(left),
          Expr::String("/".to_string()),
          unit_to_box_form_inner(right),
        ]
        .into(),
      )]
      .into(),
    },
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => Expr::FunctionCall {
      name: "RowBox".to_string(),
      args: vec![Expr::List(
        vec![
          unit_to_box_form_inner(left),
          Expr::String("\u{22c5}".to_string()),
          unit_to_box_form_inner(right),
        ]
        .into(),
      )]
      .into(),
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
              args: vec![base_box, Expr::String((-neg_exp).to_string())].into(),
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
          args: vec![Expr::List(
            vec![numer, Expr::String("/".to_string()), denom].into(),
          )]
          .into(),
        }
      }
    }
    _ => expr_to_box_form(unit),
  }
}

/// Like `unit_to_box_form` but without singularization (for compound sub-units).
fn unit_to_box_form_inner(unit: &Expr) -> Expr {
  use crate::functions::quantity_ast::unit_to_abbreviation;

  if let Some((base, exp)) = crate::functions::graphics::as_power(unit) {
    let base_box = unit_to_box_form_inner(base);
    let exp_box = expr_to_box_form(exp);
    return Expr::FunctionCall {
      name: "SuperscriptBox".to_string(),
      args: vec![base_box, exp_box].into(),
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
      args: vec![Expr::List(
        vec![
          unit_to_box_form_inner(left),
          Expr::String("/".to_string()),
          unit_to_box_form_inner(right),
        ]
        .into(),
      )]
      .into(),
    },
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => Expr::FunctionCall {
      name: "RowBox".to_string(),
      args: vec![Expr::List(
        vec![
          unit_to_box_form_inner(left),
          Expr::String("\u{22c5}".to_string()),
          unit_to_box_form_inner(right),
        ]
        .into(),
      )]
      .into(),
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
              args: vec![base_box, Expr::String((-neg_exp).to_string())].into(),
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
          args: vec![Expr::List(
            vec![numer, Expr::String("/".to_string()), denom].into(),
          )]
          .into(),
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
    args: vec![Expr::List(result.into())].into(),
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
            args: activated_args.into(),
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
    Expr::BinaryOp { op, left, right } => binop(
      *op,
      activate_expr(left, filter),
      activate_expr(right, filter),
    ),
    // Recurse into unary ops
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(activate_expr(operand, filter)),
    },
    // Atoms: return as-is
    _ => expr.clone(),
  }
}

/// Build `Inactive[head][args...]` (a CurriedCall whose func is
/// `Inactive[head]`), or the plain `head[args...]` when `head` is filtered
/// out. `wrap` decides whether this head should be inactivated.
fn make_inactive_head(head: &str, args: Vec<Expr>, wrap: bool) -> Expr {
  if wrap {
    Expr::CurriedCall {
      func: Box::new(Expr::FunctionCall {
        name: "Inactive".to_string(),
        args: vec![Expr::Identifier(head.to_string())].into(),
      }),
      args,
    }
  } else {
    Expr::FunctionCall {
      name: head.to_string(),
      args: args.into(),
    }
  }
}

/// Flatten an associative binary chain (Plus/Times/And/Or/...) of `op` rooted
/// at `expr` into its operand list, in left-to-right order.
fn flatten_binop(expr: &Expr, op: BinaryOperator) -> Vec<Expr> {
  if let Expr::BinaryOp { op: o, left, right } = expr
    && *o == op
  {
    let mut out = flatten_binop(left, op);
    out.extend(flatten_binop(right, op));
    out
  } else {
    vec![expr.clone()]
  }
}

/// Negate a Plus operand: numeric literals flip sign in place (`1` → `-1`),
/// everything else becomes `Times[-1, inactivate(e)]`.
fn negate_plus_term(e: &Expr, filter: Option<&str>) -> Expr {
  let wants_times = filter.is_none_or(|f| f == "Times");
  match e {
    Expr::Integer(n) => Expr::Integer(-n),
    Expr::Real(r) => Expr::Real(-r),
    Expr::BigInteger(n) => Expr::BigInteger(-n.clone()),
    _ => make_inactive_head(
      "Times",
      vec![Expr::Integer(-1), inactivate_expr(e, filter)],
      wants_times,
    ),
  }
}

/// Flatten a mixed Plus/Minus chain into the list of (inactivated) Plus
/// operands, in left-to-right order. A `0` left operand of a subtraction
/// (Woxi's encoding of unary minus) contributes no term.
fn collect_plus_terms(expr: &Expr, out: &mut Vec<Expr>, filter: Option<&str>) {
  use BinaryOperator as B;
  match expr {
    Expr::BinaryOp {
      op: B::Plus,
      left,
      right,
    } => {
      collect_plus_terms(left, out, filter);
      collect_plus_terms(right, out, filter);
    }
    Expr::BinaryOp {
      op: B::Minus,
      left,
      right,
    } => {
      if !matches!(left.as_ref(), Expr::Integer(0)) {
        collect_plus_terms(left, out, filter);
      }
      out.push(negate_plus_term(right, filter));
    }
    _ => out.push(inactivate_expr(expr, filter)),
  }
}

/// Recursively replace each head `H` in `expr` with `Inactive[H]`, the inverse
/// of `activate_expr`. Operators are mapped to their full-form heads (`+`→Plus,
/// `-a`→Times[-1,a], `a/b`→Times[a,Power[b,-1]], …). `List` is left structural.
/// When `filter` is Some, only that head name is inactivated.
fn inactivate_expr(expr: &Expr, filter: Option<&str>) -> Expr {
  let wants = |head: &str| filter.is_none_or(|f| f == head);
  let inact = |e: &Expr| inactivate_expr(e, filter);
  match expr {
    Expr::BinaryOp { op, left, right } => {
      use BinaryOperator as B;
      match op {
        B::Times | B::And | B::Or | B::StringJoin | B::Alternatives => {
          let head = match op {
            B::Times => "Times",
            B::And => "And",
            B::Or => "Or",
            B::StringJoin => "StringJoin",
            _ => "Alternatives",
          };
          let operands = flatten_binop(expr, *op).iter().map(&inact).collect();
          make_inactive_head(head, operands, wants(head))
        }
        // Plus/Minus form one flat Plus chain. Subtraction `a - b` becomes
        // `Plus[a, -b]`, negating numeric literals directly and wrapping
        // other terms as `Times[-1, b]`. A unary minus (`-b`, which Woxi
        // parses as `0 - b`) collapses to that single negated term.
        B::Plus | B::Minus => {
          let mut terms: Vec<Expr> = Vec::new();
          let unary =
            matches!(op, B::Minus) && matches!(left.as_ref(), Expr::Integer(0));
          collect_plus_terms(expr, &mut terms, filter);
          if unary && terms.len() == 1 {
            terms.into_iter().next().unwrap()
          } else {
            make_inactive_head("Plus", terms, wants("Plus"))
          }
        }
        // a / b  →  Times[a, Power[b, -1]]
        B::Divide => {
          let inv_b = make_inactive_head(
            "Power",
            vec![inact(right), Expr::Integer(-1)],
            wants("Power"),
          );
          make_inactive_head("Times", vec![inact(left), inv_b], wants("Times"))
        }
        B::Power => make_inactive_head(
          "Power",
          vec![inact(left), inact(right)],
          wants("Power"),
        ),
      }
    }
    Expr::UnaryOp { op, operand } => match op {
      // -a  →  Times[-1, a]
      UnaryOperator::Minus => make_inactive_head(
        "Times",
        vec![Expr::Integer(-1), inact(operand)],
        wants("Times"),
      ),
      UnaryOperator::Not => {
        make_inactive_head("Not", vec![inact(operand)], wants("Not"))
      }
    },
    // Single-operator comparison `a op b` → Inactive[<Head>][a, b].
    Expr::Comparison {
      operands,
      operators,
    } if operators.len() == 1 && operands.len() == 2 => {
      use ComparisonOp as C;
      let head = match operators[0] {
        C::Equal => "Equal",
        C::NotEqual => "Unequal",
        C::Less => "Less",
        C::LessEqual => "LessEqual",
        C::Greater => "Greater",
        C::GreaterEqual => "GreaterEqual",
        C::SameQ => "SameQ",
        C::UnsameQ => "UnsameQ",
      };
      make_inactive_head(
        head,
        operands.iter().map(&inact).collect(),
        wants(head),
      )
    }
    // List is structural: never inactivated, but recurse into elements.
    Expr::List(items) => Expr::List(items.iter().map(&inact).collect()),
    Expr::FunctionCall { name, args } => {
      make_inactive_head(name, args.iter().map(&inact).collect(), wants(name))
    }
    Expr::CurriedCall { func, args } => Expr::CurriedCall {
      func: Box::new(inact(func)),
      args: args.iter().map(&inact).collect(),
    },
    // Atoms (numbers, symbols, strings) are returned unchanged.
    _ => expr.clone(),
  }
}

// ─── RegionMember ────────────────────────────────────────────────────────

/// RegionMember[region, point] — test whether `point` lies in the (closed)
/// `region`. Handles Disk/Ball (solid) and Circle/Sphere (their boundary) and
/// Rectangle/Cuboid (axis-aligned boxes) with numeric coordinates; returns the
/// call unevaluated for other regions or non-numeric input.
fn compute_region_member(
  region: &Expr,
  point: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionMember".to_string(),
      args: vec![region.clone(), point.clone()].into(),
    })
  };
  let to_vec = |e: &Expr| -> Option<Vec<f64>> {
    if let Expr::List(items) = e {
      items
        .iter()
        .map(crate::functions::math_ast::try_eval_to_f64)
        .collect()
    } else {
      None
    }
  };
  let boolean = |b: bool| Ok(bool_expr(b));
  const EPS: f64 = 1e-10;

  let Some(pt) = to_vec(point) else {
    return unevaluated();
  };
  let Expr::FunctionCall { name, args } = region else {
    return unevaluated();
  };

  match name.as_str() {
    // Point-in-stadium: distance from the point to the segment is at
    // most r (closed region). Numeric data only.
    "StadiumShape" if stadium_parts(args).is_some() => {
      let (p1, p2, r) = stadium_parts(args).unwrap();
      let num = crate::functions::math_ast::try_eval_to_f64;
      let pt = match point {
        Expr::List(items) if items.len() == 2 => items,
        _ => return unevaluated(),
      };
      let vals: Option<Vec<f64>> = p1
        .iter()
        .chain(p2.iter())
        .chain(pt.iter())
        .map(num)
        .collect();
      let (Some(v), Some(rv)) = (vals, num(&r)) else {
        return unevaluated();
      };
      let (ax, ay, bx, by, px, py) = (v[0], v[1], v[2], v[3], v[4], v[5]);
      let (dx, dy) = (bx - ax, by - ay);
      let len2 = dx * dx + dy * dy;
      let t = if len2 == 0.0 {
        0.0
      } else {
        (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0)
      };
      let (cx, cy) = (ax + t * dx, ay + t * dy);
      let d2 = (px - cx) * (px - cx) + (py - cy) * (py - cy);
      Ok(bool_expr(d2 <= rv * rv))
    }
    "Disk" | "Ball" | "Circle" | "Sphere" => {
      let default_dim = if name == "Ball" || name == "Sphere" {
        3
      } else {
        2
      };
      let center = match args.first() {
        None => vec![0.0; default_dim],
        Some(c) => match to_vec(c) {
          Some(v) => v,
          None => return unevaluated(),
        },
      };
      let radius = match args.get(1) {
        None => 1.0,
        Some(r) => match crate::functions::math_ast::try_eval_to_f64(r) {
          Some(v) => v,
          None => return unevaluated(),
        },
      };
      if pt.len() != center.len() {
        return unevaluated();
      }
      let dist2: f64 =
        pt.iter().zip(&center).map(|(a, b)| (a - b).powi(2)).sum();
      let r2 = radius * radius;
      let tol = EPS * r2.max(1.0);
      // Disk/Ball are solid (distance <= r); Circle/Sphere are the boundary.
      let inside = match name.as_str() {
        "Circle" | "Sphere" => (dist2 - r2).abs() <= tol,
        _ => dist2 <= r2 + tol,
      };
      boolean(inside)
    }
    "Rectangle" | "Cuboid" => {
      // Defaults: Rectangle[] = {0,…}..{1,…}; Rectangle[c] = c..c+1.
      let (lo, hi) = match (args.first(), args.get(1)) {
        (None, _) => (vec![0.0; pt.len()], vec![1.0; pt.len()]),
        (Some(c1), None) => {
          let Some(c1) = to_vec(c1) else {
            return unevaluated();
          };
          let c2: Vec<f64> = c1.iter().map(|x| x + 1.0).collect();
          (c1, c2)
        }
        (Some(c1), Some(c2)) => {
          let (Some(c1), Some(c2)) = (to_vec(c1), to_vec(c2)) else {
            return unevaluated();
          };
          (c1, c2)
        }
      };
      if pt.len() != lo.len() || pt.len() != hi.len() {
        return unevaluated();
      }
      let inside =
        pt.iter().zip(lo.iter().zip(hi.iter())).all(|(p, (a, b))| {
          let (mn, mx) = if a <= b { (a, b) } else { (b, a) };
          *p >= mn - EPS && *p <= mx + EPS
        });
      boolean(inside)
    }
    // HalfSpace[n, c] / HalfSpace[n, p] — closed half-space n.x <= c.
    "HalfSpace" => {
      let Some((normal, bound)) = half_space_parts(args) else {
        return unevaluated();
      };
      if pt.len() != normal.len() {
        return unevaluated();
      }
      let n_f: Option<Vec<f64>> = normal
        .iter()
        .map(crate::functions::math_ast::try_eval_to_f64)
        .collect();
      let (Some(n_f), Some(c_f)) =
        (n_f, crate::functions::math_ast::try_eval_to_f64(&bound))
      else {
        return unevaluated();
      };
      let dot: f64 = n_f.iter().zip(&pt).map(|(a, b)| a * b).sum();
      let scale = n_f.iter().map(|x| x.abs()).fold(1.0f64, f64::max);
      boolean(dot <= c_f + EPS * scale.max(c_f.abs()))
    }
    // Triangle/Polygon: a closed 2D region. A point is a member when it is
    // inside or on the boundary.
    "Triangle" | "Polygon" if args.len() == 1 => {
      let Expr::List(vs) = &args[0] else {
        return unevaluated();
      };
      let (Some(verts), 2) = (polygon_verts_f64(vs), pt.len()) else {
        return unevaluated();
      };
      if verts.len() < 3 {
        return unevaluated();
      }
      boolean(point_in_polygon_2d(&verts, [pt[0], pt[1]], EPS))
    }
    _ => unevaluated(),
  }
}

/// A region shape supported by RegionDisjoint, reduced to floats.
enum DisjointShape {
  /// Disk/Ball (solid) or Circle/Sphere (shell).
  Round {
    center: Vec<f64>,
    r: f64,
    shell: bool,
  },
  /// Rectangle/Cuboid with lo ≤ hi per axis.
  AxisBox { lo: Vec<f64>, hi: Vec<f64> },
  /// Triangle/Polygon in the plane (solid).
  Poly { verts: Vec<[f64; 2]> },
  /// A single point.
  Dot { p: Vec<f64> },
}

impl DisjointShape {
  fn dim(&self) -> usize {
    match self {
      DisjointShape::Round { center, .. } => center.len(),
      DisjointShape::AxisBox { lo, .. } => lo.len(),
      DisjointShape::Poly { .. } => 2,
      DisjointShape::Dot { p } => p.len(),
    }
  }
}

/// Parse a region expression into a DisjointShape (floats), or None if the
/// region kind / argument shape is unsupported.
fn disjoint_shape(expr: &Expr) -> Option<DisjointShape> {
  let to_vec = |e: &Expr| -> Option<Vec<f64>> {
    if let Expr::List(items) = e {
      items
        .iter()
        .map(crate::functions::math_ast::try_eval_to_f64)
        .collect()
    } else {
      None
    }
  };
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  match name.as_str() {
    "Disk" | "Ball" | "Circle" | "Sphere" => {
      let default_dim = if name == "Ball" || name == "Sphere" {
        3
      } else {
        2
      };
      let center = match args.first() {
        None => vec![0.0; default_dim],
        Some(c) => to_vec(c)?,
      };
      // A radius list is an ellipse/ellipsoid — unsupported.
      let r = match args.get(1) {
        None => 1.0,
        Some(Expr::List(_)) => return None,
        Some(r) => crate::functions::math_ast::try_eval_to_f64(r)?,
      };
      if args.len() > 2 || r <= 0.0 {
        return None;
      }
      Some(DisjointShape::Round {
        center,
        r,
        shell: matches!(name.as_str(), "Circle" | "Sphere"),
      })
    }
    "Rectangle" | "Cuboid" => {
      let default_dim = if name == "Cuboid" { 3 } else { 2 };
      let (c1, c2) = match (args.first(), args.get(1)) {
        (None, _) => (vec![0.0; default_dim], vec![1.0; default_dim]),
        (Some(c1), None) => {
          let c1 = to_vec(c1)?;
          let c2: Vec<f64> = c1.iter().map(|x| x + 1.0).collect();
          (c1, c2)
        }
        (Some(c1), Some(c2)) => (to_vec(c1)?, to_vec(c2)?),
      };
      if args.len() > 2 || c1.len() != c2.len() || c1.is_empty() {
        return None;
      }
      let lo: Vec<f64> = c1.iter().zip(&c2).map(|(a, b)| a.min(*b)).collect();
      let hi: Vec<f64> = c1.iter().zip(&c2).map(|(a, b)| a.max(*b)).collect();
      Some(DisjointShape::AxisBox { lo, hi })
    }
    "Triangle" | "Polygon" => {
      let verts: Vec<[f64; 2]> = if name == "Triangle" && args.is_empty() {
        vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
      } else if args.len() == 1
        && let Expr::List(vs) = &args[0]
      {
        polygon_verts_f64(vs)?
      } else {
        return None;
      };
      if verts.len() < 3 {
        return None;
      }
      Some(DisjointShape::Poly { verts })
    }
    "Point" if args.len() == 1 => Some(DisjointShape::Dot {
      p: to_vec(&args[0])?,
    }),
    _ => None,
  }
}

/// Distance from point `p` to the closed segment `a`–`b` (2-D).
fn seg_point_dist(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
  let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
  let len2 = dx * dx + dy * dy;
  let t = if len2 == 0.0 {
    0.0
  } else {
    (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
  };
  let (cx, cy) = (a[0] + t * dx, a[1] + t * dy);
  ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt()
}

/// Minimum distance between the closed segments `p1`–`p2` and `q1`–`q2`
/// (0 when they cross or touch).
fn seg_seg_dist(p1: [f64; 2], p2: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> f64 {
  let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  };
  let d1 = cross(p1, p2, q1);
  let d2 = cross(p1, p2, q2);
  let d3 = cross(q1, q2, p1);
  let d4 = cross(q1, q2, p2);
  if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
    return 0.0; // proper crossing
  }
  seg_point_dist(p1, p2, q1)
    .min(seg_point_dist(p1, p2, q2))
    .min(seg_point_dist(q1, q2, p1))
    .min(seg_point_dist(q1, q2, p2))
}

/// Distance from point `p` to the SOLID polygon (0 when inside or on the
/// boundary).
fn poly_point_dist(verts: &[[f64; 2]], p: [f64; 2], eps: f64) -> f64 {
  if point_in_polygon_2d(verts, p, eps) {
    return 0.0;
  }
  let mut best = f64::INFINITY;
  for i in 0..verts.len() {
    let j = (i + 1) % verts.len();
    best = best.min(seg_point_dist(verts[i], verts[j], p));
  }
  best
}

/// Whether two solid polygons intersect (share any point, boundaries
/// included).
fn polys_intersect(a: &[[f64; 2]], b: &[[f64; 2]], eps: f64) -> bool {
  if a.iter().any(|v| point_in_polygon_2d(b, *v, eps))
    || b.iter().any(|v| point_in_polygon_2d(a, *v, eps))
  {
    return true;
  }
  for i in 0..a.len() {
    let i2 = (i + 1) % a.len();
    for j in 0..b.len() {
      let j2 = (j + 1) % b.len();
      if seg_seg_dist(a[i], a[i2], b[j], b[j2]) <= eps {
        return true;
      }
    }
  }
  false
}

/// Whether two supported shapes share at least one point (regions are
/// closed, so touching counts as intersecting). Shapes living in different
/// ambient dimensions never intersect (wolframscript-verified:
/// RegionDisjoint[Disk[], Ball[]] is True).
fn disjoint_shapes_intersect(a: &DisjointShape, b: &DisjointShape) -> bool {
  use DisjointShape::{AxisBox, Dot, Poly, Round};
  const EPS: f64 = 1e-9;
  if a.dim() != b.dim() {
    return false;
  }
  let dist = |p: &[f64], q: &[f64]| -> f64 {
    p.iter()
      .zip(q)
      .map(|(x, y)| (x - y) * (x - y))
      .sum::<f64>()
      .sqrt()
  };
  // Distance from a point to the closed box, and to its farthest point.
  let box_min_dist = |lo: &[f64], hi: &[f64], p: &[f64]| -> f64 {
    p.iter()
      .zip(lo.iter().zip(hi))
      .map(|(x, (l, h))| {
        let d = if x < l {
          l - x
        } else if x > h {
          x - h
        } else {
          0.0
        };
        d * d
      })
      .sum::<f64>()
      .sqrt()
  };
  let box_max_dist = |lo: &[f64], hi: &[f64], p: &[f64]| -> f64 {
    p.iter()
      .zip(lo.iter().zip(hi))
      .map(|(x, (l, h))| (x - l).abs().max((x - h).abs()).powi(2))
      .sum::<f64>()
      .sqrt()
  };
  match (a, b) {
    (Dot { p }, Dot { p: q }) => dist(p, q) <= EPS,
    (Dot { p }, Round { center, r, shell })
    | (Round { center, r, shell }, Dot { p }) => {
      let d = dist(p, center);
      if *shell {
        (d - r).abs() <= EPS
      } else {
        d <= r + EPS
      }
    }
    (Dot { p }, AxisBox { lo, hi }) | (AxisBox { lo, hi }, Dot { p }) => {
      box_min_dist(lo, hi, p) <= EPS
    }
    (Dot { p }, Poly { verts }) | (Poly { verts }, Dot { p }) => {
      point_in_polygon_2d(verts, [p[0], p[1]], EPS)
    }
    (
      Round {
        center: c1,
        r: r1,
        shell: s1,
      },
      Round {
        center: c2,
        r: r2,
        shell: s2,
      },
    ) => {
      let d = dist(c1, c2);
      match (s1, s2) {
        (false, false) => d <= r1 + r2 + EPS,
        (true, true) => (r1 - r2).abs() - EPS <= d && d <= r1 + r2 + EPS,
        // Shell of radius rs vs solid of radius rd: the nearest shell
        // point to the solid's center is at |d - rs|.
        (true, false) => (d - r1).abs() <= r2 + EPS,
        (false, true) => (d - r2).abs() <= r1 + EPS,
      }
    }
    (Round { center, r, shell }, AxisBox { lo, hi })
    | (AxisBox { lo, hi }, Round { center, r, shell }) => {
      let mind = box_min_dist(lo, hi, center);
      if *shell {
        mind <= r + EPS && *r <= box_max_dist(lo, hi, center) + EPS
      } else {
        mind <= r + EPS
      }
    }
    (Round { center, r, shell }, Poly { verts })
    | (Poly { verts }, Round { center, r, shell }) => {
      let c = [center[0], center[1]];
      let mind = poly_point_dist(verts, c, EPS);
      if *shell {
        let maxd = verts
          .iter()
          .map(|v| ((v[0] - c[0]).powi(2) + (v[1] - c[1]).powi(2)).sqrt())
          .fold(0.0f64, f64::max);
        mind <= r + EPS && *r <= maxd + EPS
      } else {
        mind <= r + EPS
      }
    }
    (AxisBox { lo: l1, hi: h1 }, AxisBox { lo: l2, hi: h2 }) => l1
      .iter()
      .zip(h1)
      .zip(l2.iter().zip(h2))
      .all(|((l1, h1), (l2, h2))| l1 <= &(h2 + EPS) && l2 <= &(h1 + EPS)),
    (AxisBox { lo, hi }, Poly { verts })
    | (Poly { verts }, AxisBox { lo, hi }) => {
      let corners = vec![
        [lo[0], lo[1]],
        [hi[0], lo[1]],
        [hi[0], hi[1]],
        [lo[0], hi[1]],
      ];
      polys_intersect(&corners, verts, EPS)
    }
    (Poly { verts: v1 }, Poly { verts: v2 }) => polys_intersect(v1, v2, EPS),
  }
}

/// RegionDisjoint[reg1, reg2, …] — True when the regions are pairwise
/// disjoint. Zero regions are vacuously disjoint; a single supported
/// region is too. Unsupported arguments leave the call unevaluated.
fn compute_region_disjoint(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let shapes: Option<Vec<DisjointShape>> = args
    .iter()
    .map(|a| disjoint_shape(strip_region_wrapper(a)))
    .collect();
  let Some(shapes) = shapes else {
    if args.is_empty() {
      return Ok(bool_expr(true));
    }
    return Ok(unevaluated("RegionDisjoint", args));
  };
  for i in 0..shapes.len() {
    for j in i + 1..shapes.len() {
      if disjoint_shapes_intersect(&shapes[i], &shapes[j]) {
        return Ok(bool_expr(false));
      }
    }
  }
  Ok(bool_expr(true))
}

/// Nearest point of a `Line[{v1, v2, …}]` (a segment or polyline) to `point`,
/// built symbolically so exact inputs give exact coordinates. Each segment's
/// clamped projection is a candidate; the closest by Euclidean distance wins.
/// Returns `None` if the vertices are not coordinate vectors of length `n`.
fn line_nearest_point(
  verts: &[Expr],
  point: &Expr,
  pt: &[Expr],
  n: usize,
) -> Result<Option<Expr>, InterpreterError> {
  use BinaryOperator::{Divide, Minus, Plus, Times};
  if verts.len() < 2 {
    return Ok(None);
  }
  let mut vlists: Vec<Vec<Expr>> = Vec::with_capacity(verts.len());
  for v in verts {
    match v {
      Expr::List(c) if c.len() == n => vlists.push(c.iter().cloned().collect()),
      _ => return Ok(None),
    }
  }
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let to_f64 = crate::functions::math_ast::try_eval_to_f64;
  let plus = |terms: Vec<Expr>| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  };
  let dist_to = |c: &Expr| -> Result<Option<f64>, InterpreterError> {
    Ok(to_f64(&eval(&Expr::FunctionCall {
      name: "EuclideanDistance".to_string(),
      args: vec![point.clone(), c.clone()].into(),
    })?))
  };

  let mut best: Option<(f64, Expr)> = None;
  for seg in vlists.windows(2) {
    let (a, b) = (&seg[0], &seg[1]);
    // t = dot(p − a, b − a) / dot(b − a, b − a), clamped to [0, 1].
    let num = plus(
      (0..n)
        .map(|i| {
          binop(
            Times,
            binop(Minus, pt[i].clone(), a[i].clone()),
            binop(Minus, b[i].clone(), a[i].clone()),
          )
        })
        .collect(),
    );
    let den = plus(
      (0..n)
        .map(|i| {
          let d = binop(Minus, b[i].clone(), a[i].clone());
          binop(Times, d.clone(), d)
        })
        .collect(),
    );
    let den_val = to_f64(&eval(&den)?);
    let cand: Vec<Expr> = if den_val.is_none_or(|d| d.abs() < 1e-15) {
      a.clone() // degenerate (zero-length) segment
    } else {
      let t = binop(Divide, num, den);
      let t_val = to_f64(&eval(&t)?).unwrap_or(0.0);
      if t_val <= 0.0 {
        a.clone()
      } else if t_val >= 1.0 {
        b.clone()
      } else {
        (0..n)
          .map(|i| {
            binop(
              Plus,
              a[i].clone(),
              binop(Times, t.clone(), binop(Minus, b[i].clone(), a[i].clone())),
            )
          })
          .collect()
      }
    };
    let cand_list = eval(&Expr::List(cand.into()))?;
    let d = dist_to(&cand_list)?.unwrap_or(f64::INFINITY);
    if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
      best = Some((d, cand_list));
    }
  }
  Ok(best.map(|(_, c)| c))
}

/// Convert a list of vertex expressions into 2D floating-point coordinates.
/// Returns `None` unless every vertex is a length-2 numeric list.
fn polygon_verts_f64(vs: &[Expr]) -> Option<Vec<[f64; 2]>> {
  vs.iter()
    .map(|v| match v {
      Expr::List(c) if c.len() == 2 => {
        let x = crate::functions::math_ast::try_eval_to_f64(&c[0])?;
        let y = crate::functions::math_ast::try_eval_to_f64(&c[1])?;
        Some([x, y])
      }
      _ => None,
    })
    .collect()
}

/// Is `p` on the segment `a`–`b` (within `eps`)?
fn point_on_segment_2d(
  a: [f64; 2],
  b: [f64; 2],
  p: [f64; 2],
  eps: f64,
) -> bool {
  let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
  let scale = 1.0 + (b[0] - a[0]).abs() + (b[1] - a[1]).abs();
  if cross.abs() > eps * scale {
    return false;
  }
  p[0] >= a[0].min(b[0]) - eps
    && p[0] <= a[0].max(b[0]) + eps
    && p[1] >= a[1].min(b[1]) - eps
    && p[1] <= a[1].max(b[1]) + eps
}

/// Is `p` inside or on the boundary of the (closed) polygon `verts`? Uses an
/// exact on-boundary test plus an even–odd ray cast for the strict interior.
fn point_in_polygon_2d(verts: &[[f64; 2]], p: [f64; 2], eps: f64) -> bool {
  let n = verts.len();
  for i in 0..n {
    if point_on_segment_2d(verts[i], verts[(i + 1) % n], p, eps) {
      return true;
    }
  }
  let mut inside = false;
  let mut j = n - 1;
  for i in 0..n {
    let (xi, yi) = (verts[i][0], verts[i][1]);
    let (xj, yj) = (verts[j][0], verts[j][1]);
    if (yi > p[1]) != (yj > p[1]) {
      let x_int = xi + (p[1] - yi) / (yj - yi) * (xj - xi);
      if p[0] < x_int {
        inside = !inside;
      }
    }
    j = i;
  }
  inside
}

/// RegionDistance[region, point] — the shortest distance from `point` to the
/// (closed) `region`, zero when inside a solid region. Handles Point, Disk/Ball
/// (solid), Circle/Sphere (boundary) and Rectangle/Cuboid (axis-aligned boxes)
/// by building a symbolic distance expression and evaluating it, so exact
/// inputs give exact results and machine numbers give machine results.
fn compute_region_distance(
  region: &Expr,
  point: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionDistance".to_string(),
      args: vec![region.clone(), point.clone()].into(),
    })
  };
  if !matches!(point, Expr::List(_)) {
    return unevaluated();
  }
  let Expr::FunctionCall { name, args } = region else {
    return unevaluated();
  };

  let call = |n: &str, a: Vec<Expr>| Expr::FunctionCall {
    name: n.to_string(),
    args: a.into(),
  };
  let euclid = |a: Expr, b: Expr| call("EuclideanDistance", vec![a, b]);
  let sub = |a: Expr, b: Expr| binop(BinaryOperator::Minus, a, b);
  let zeros = |n: usize| Expr::List(vec![Expr::Integer(0); n].into());

  let expr = match name.as_str() {
    "Point" if args.len() == 1 => euclid(point.clone(), args[0].clone()),
    // HalfSpace — 0 inside, (n.x - c)/Norm[n] outside.
    "HalfSpace" => {
      let Some((normal, bound)) = half_space_parts(args) else {
        return unevaluated();
      };
      let Expr::List(pt) = point else {
        return unevaluated();
      };
      if pt.len() != normal.len() {
        return unevaluated();
      }
      let pt: Vec<Expr> = pt.iter().cloned().collect();
      let dot = half_space_dot(&normal, &pt)?;
      // Signed excess n.x - c decides the side.
      let excess =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            dot,
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), bound].into(),
            },
          ]
          .into(),
        })?;
      let Some(excess_f) = crate::functions::math_ast::try_eval_to_f64(&excess)
      else {
        return unevaluated();
      };
      if excess_f <= 0.0 {
        return Ok(Expr::Integer(0));
      }
      let norm_sq: Vec<Expr> = normal
        .iter()
        .map(|a| Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![a.clone(), Expr::Integer(2)].into(),
        })
        .collect();
      let norm = call(
        "Sqrt",
        vec![Expr::FunctionCall {
          name: "Plus".to_string(),
          args: norm_sq.into(),
        }],
      );
      return crate::evaluator::evaluate_expr_to_expr(&binop(
        BinaryOperator::Divide,
        excess,
        norm,
      ));
    }
    "Disk" | "Ball" => {
      let dim = if name == "Ball" { 3 } else { 2 };
      let center = args.first().cloned().unwrap_or_else(|| zeros(dim));
      let radius = args.get(1).cloned().unwrap_or(Expr::Integer(1));
      // Solid: Max[0, dist - r]. The zero's type controls the result type for
      // points inside the region (0 for exact input, 0. for machine input).
      let zero = if expr_contains_real(point) || expr_contains_real(region) {
        Expr::Real(0.0)
      } else {
        Expr::Integer(0)
      };
      call(
        "Max",
        vec![zero, sub(euclid(point.clone(), center), radius)],
      )
    }
    "Circle" | "Sphere" => {
      let dim = if name == "Sphere" { 3 } else { 2 };
      let center = args.first().cloned().unwrap_or_else(|| zeros(dim));
      let radius = args.get(1).cloned().unwrap_or(Expr::Integer(1));
      // Boundary: Abs[dist - r].
      call("Abs", vec![sub(euclid(point.clone(), center), radius)])
    }
    "Rectangle" | "Cuboid" => {
      let Expr::List(pt) = point else {
        return unevaluated();
      };
      let n = pt.len();
      // Corners with the same defaults as RegionMember.
      let (c1, c2): (Vec<Expr>, Vec<Expr>) = match (args.first(), args.get(1)) {
        (None, _) => (vec![Expr::Integer(0); n], vec![Expr::Integer(1); n]),
        (Some(Expr::List(a)), None) => {
          let a: Vec<Expr> = a.iter().cloned().collect();
          let b = a
            .iter()
            .map(|x| call("Plus", vec![x.clone(), Expr::Integer(1)]))
            .collect();
          (a, b)
        }
        (Some(Expr::List(a)), Some(Expr::List(b))) => {
          (a.iter().cloned().collect(), b.iter().cloned().collect())
        }
        _ => return unevaluated(),
      };
      if c1.len() != n || c2.len() != n {
        return unevaluated();
      }
      // Clamp each coordinate to [min(c1,c2), max(c1,c2)], then take the
      // distance to the clamped point.
      let clamped: Vec<Expr> = (0..n)
        .map(|i| {
          let lo = call("Min", vec![c1[i].clone(), c2[i].clone()]);
          let hi = call("Max", vec![c1[i].clone(), c2[i].clone()]);
          let inner = call("Min", vec![hi, pt[i].clone()]);
          call("Max", vec![lo, inner])
        })
        .collect();
      euclid(point.clone(), Expr::List(clamped.into()))
    }
    "Line" if args.len() == 1 => {
      let Expr::List(pt) = point else {
        return unevaluated();
      };
      let pt_vec: Vec<Expr> = pt.iter().cloned().collect();
      let Expr::List(verts) = &args[0] else {
        return unevaluated();
      };
      match line_nearest_point(verts, point, &pt_vec, pt_vec.len())? {
        Some(nearest) => euclid(point.clone(), nearest),
        None => return unevaluated(),
      }
    }
    "Triangle" | "Polygon" if args.len() == 1 => {
      let Expr::List(pt) = point else {
        return unevaluated();
      };
      let pt_vec: Vec<Expr> = pt.iter().cloned().collect();
      if pt_vec.len() != 2 {
        return unevaluated();
      }
      let Expr::List(vs) = &args[0] else {
        return unevaluated();
      };
      let Some(verts) = polygon_verts_f64(vs) else {
        return unevaluated();
      };
      if verts.len() < 3 {
        return unevaluated();
      }
      let pf = match (
        crate::functions::math_ast::try_eval_to_f64(&pt_vec[0]),
        crate::functions::math_ast::try_eval_to_f64(&pt_vec[1]),
      ) {
        (Some(x), Some(y)) => [x, y],
        _ => return unevaluated(),
      };
      // Inside the closed region → distance 0 (typed by input exactness).
      if point_in_polygon_2d(&verts, pf, 1e-10) {
        return Ok(
          if expr_contains_real(point) || expr_contains_real(region) {
            Expr::Real(0.0)
          } else {
            Expr::Integer(0)
          },
        );
      }
      let mut closed: Vec<Expr> = vs.iter().cloned().collect();
      closed.push(vs[0].clone());
      match line_nearest_point(&closed, point, &pt_vec, 2)? {
        Some(nearest) => euclid(point.clone(), nearest),
        None => return unevaluated(),
      }
    }
    _ => return unevaluated(),
  };

  crate::evaluator::evaluate_expr_to_expr(&expr)
}

/// RegionNearest[region, point] — the point of `region` closest to `point`.
/// For a solid Disk/Ball an interior point maps to itself; otherwise the point
/// is projected onto the boundary. Circle/Sphere always project; Rectangle/
/// Cuboid clamp each coordinate into the box. Built symbolically so exact
/// inputs give exact coordinates (e.g. {6/5, 8/5}).
fn compute_region_nearest(
  region: &Expr,
  point: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionNearest".to_string(),
      args: vec![region.clone(), point.clone()].into(),
    })
  };
  let Expr::List(pt) = point else {
    return unevaluated();
  };
  let pt: Vec<Expr> = pt.iter().cloned().collect();
  let n = pt.len();
  let Expr::FunctionCall { name, args } = region else {
    return unevaluated();
  };

  let call = |nm: &str, a: Vec<Expr>| Expr::FunctionCall {
    name: nm.to_string(),
    args: a.into(),
  };
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let to_f64 = crate::functions::math_ast::try_eval_to_f64;

  match name.as_str() {
    // The nearest point of a single Point is the point itself.
    "Point" if args.len() == 1 => eval(&args[0]),
    "Disk" | "Ball" | "Circle" | "Sphere" => {
      let dim = if name == "Ball" || name == "Sphere" {
        3
      } else {
        2
      };
      let center: Vec<Expr> = match args.first() {
        None => vec![Expr::Integer(0); dim],
        Some(Expr::List(c)) => c.iter().cloned().collect(),
        Some(_) => return unevaluated(),
      };
      let radius = args.get(1).cloned().unwrap_or(Expr::Integer(1));
      if center.len() != n {
        return unevaluated();
      }
      let center_list = Expr::List(center.clone().into());
      let dist_sym =
        call("EuclideanDistance", vec![point.clone(), center_list]);
      let dist = match to_f64(&eval(&dist_sym)?) {
        Some(d) => d,
        None => return unevaluated(),
      };
      let r = match to_f64(&radius) {
        Some(r) => r,
        None => return unevaluated(),
      };
      // A point at the center has no unique projection onto the boundary.
      if dist <= 1e-12 {
        return unevaluated();
      }
      let solid = matches!(name.as_str(), "Disk" | "Ball");
      if solid && dist <= r + 1e-10 * r.max(1.0) {
        // Inside the solid region → the point itself.
        return eval(point);
      }
      // Project onto the boundary: center + r * (point - center) / dist.
      let proj: Vec<Expr> = (0..n)
        .map(|i| {
          let diff =
            binop(BinaryOperator::Minus, pt[i].clone(), center[i].clone());
          let scaled = binop(
            BinaryOperator::Divide,
            binop(BinaryOperator::Times, radius.clone(), diff),
            dist_sym.clone(),
          );
          binop(BinaryOperator::Plus, center[i].clone(), scaled)
        })
        .collect();
      eval(&Expr::List(proj.into()))
    }
    "Rectangle" | "Cuboid" => {
      let (c1, c2): (Vec<Expr>, Vec<Expr>) = match (args.first(), args.get(1)) {
        (None, _) => (vec![Expr::Integer(0); n], vec![Expr::Integer(1); n]),
        (Some(Expr::List(a)), None) => {
          let a: Vec<Expr> = a.iter().cloned().collect();
          let b = a
            .iter()
            .map(|x| call("Plus", vec![x.clone(), Expr::Integer(1)]))
            .collect();
          (a, b)
        }
        (Some(Expr::List(a)), Some(Expr::List(b))) => {
          (a.iter().cloned().collect(), b.iter().cloned().collect())
        }
        _ => return unevaluated(),
      };
      if c1.len() != n || c2.len() != n {
        return unevaluated();
      }
      let clamped: Vec<Expr> = (0..n)
        .map(|i| {
          let lo = call("Min", vec![c1[i].clone(), c2[i].clone()]);
          let hi = call("Max", vec![c1[i].clone(), c2[i].clone()]);
          let inner = call("Min", vec![hi, pt[i].clone()]);
          call("Max", vec![lo, inner])
        })
        .collect();
      eval(&Expr::List(clamped.into()))
    }
    "Line" if args.len() == 1 => {
      if let Expr::List(verts) = &args[0]
        && let Some(nearest) = line_nearest_point(verts, point, &pt, n)?
      {
        return Ok(nearest);
      }
      unevaluated()
    }
    // Triangle/Polygon: an interior point maps to itself; otherwise project
    // onto the closest edge of the closed boundary.
    "Triangle" | "Polygon" if args.len() == 1 && n == 2 => {
      let Expr::List(vs) = &args[0] else {
        return unevaluated();
      };
      let Some(verts) = polygon_verts_f64(vs) else {
        return unevaluated();
      };
      if verts.len() < 3 {
        return unevaluated();
      }
      let pf = match (
        crate::functions::math_ast::try_eval_to_f64(&pt[0]),
        crate::functions::math_ast::try_eval_to_f64(&pt[1]),
      ) {
        (Some(x), Some(y)) => [x, y],
        _ => return unevaluated(),
      };
      if point_in_polygon_2d(&verts, pf, 1e-10) {
        return eval(point);
      }
      // Closed boundary = vertices with the first vertex appended.
      let mut closed: Vec<Expr> = vs.iter().cloned().collect();
      closed.push(vs[0].clone());
      match line_nearest_point(&closed, point, &pt, n)? {
        Some(nearest) => Ok(nearest),
        None => unevaluated(),
      }
    }
    _ => unevaluated(),
  }
}

/// SignedRegionDistance[region, point] — like RegionDistance but negative inside
/// a solid region. For Point/Circle/Sphere (measure-zero) it equals the
/// (non-negative) RegionDistance; for Disk/Ball it is `dist - r`, and for
/// Rectangle/Cuboid it is the axis-aligned-box signed distance field.
fn compute_signed_region_distance(
  region: &Expr,
  point: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "SignedRegionDistance".to_string(),
      args: vec![region.clone(), point.clone()].into(),
    })
  };
  let Expr::List(pt) = point else {
    return unevaluated();
  };
  let pt: Vec<Expr> = pt.iter().cloned().collect();
  let n = pt.len();
  let Expr::FunctionCall { name, args } = region else {
    return unevaluated();
  };

  let call = |nm: &str, a: Vec<Expr>| Expr::FunctionCall {
    name: nm.to_string(),
    args: a.into(),
  };
  let sub = |a: Expr, b: Expr| binop(BinaryOperator::Minus, a, b);
  let euclid = |a: Expr, b: Expr| call("EuclideanDistance", vec![a, b]);
  let zeros = |k: usize| Expr::List(vec![Expr::Integer(0); k].into());

  let expr = match name.as_str() {
    // Measure-zero regions: signed distance equals the ordinary distance.
    "Point" if args.len() == 1 => euclid(point.clone(), args[0].clone()),
    "Circle" | "Sphere" => {
      let dim = if name == "Sphere" { 3 } else { 2 };
      let center = args.first().cloned().unwrap_or_else(|| zeros(dim));
      let radius = args.get(1).cloned().unwrap_or(Expr::Integer(1));
      call("Abs", vec![sub(euclid(point.clone(), center), radius)])
    }
    // Solid balls: dist - r (negative inside).
    "Disk" | "Ball" => {
      let dim = if name == "Ball" { 3 } else { 2 };
      let center = args.first().cloned().unwrap_or_else(|| zeros(dim));
      let radius = args.get(1).cloned().unwrap_or(Expr::Integer(1));
      sub(euclid(point.clone(), center), radius)
    }
    // Axis-aligned box signed distance:
    //   d_i = max(lo_i - p_i, p_i - hi_i)
    //   sdf = Norm[Max[d_i, 0]] + Min[Max_i d_i, 0]
    "Rectangle" | "Cuboid" => {
      let (c1, c2): (Vec<Expr>, Vec<Expr>) = match (args.first(), args.get(1)) {
        (None, _) => (vec![Expr::Integer(0); n], vec![Expr::Integer(1); n]),
        (Some(Expr::List(a)), None) => {
          let a: Vec<Expr> = a.iter().cloned().collect();
          let b = a
            .iter()
            .map(|x| call("Plus", vec![x.clone(), Expr::Integer(1)]))
            .collect();
          (a, b)
        }
        (Some(Expr::List(a)), Some(Expr::List(b))) => {
          (a.iter().cloned().collect(), b.iter().cloned().collect())
        }
        _ => return unevaluated(),
      };
      if c1.len() != n || c2.len() != n {
        return unevaluated();
      }
      let dx: Vec<Expr> = (0..n)
        .map(|i| {
          let lo = call("Min", vec![c1[i].clone(), c2[i].clone()]);
          let hi = call("Max", vec![c1[i].clone(), c2[i].clone()]);
          call("Max", vec![sub(lo, pt[i].clone()), sub(pt[i].clone(), hi)])
        })
        .collect();
      let clamped: Vec<Expr> = dx
        .iter()
        .map(|d| call("Max", vec![d.clone(), Expr::Integer(0)]))
        .collect();
      let outside = call("Norm", vec![Expr::List(clamped.into())]);
      let inside = call("Min", vec![call("Max", dx), Expr::Integer(0)]);
      binop(BinaryOperator::Plus, outside, inside)
    }
    // A Line is measure-zero: its signed distance is the ordinary distance.
    "Line" if args.len() == 1 => {
      let Expr::List(verts) = &args[0] else {
        return unevaluated();
      };
      match line_nearest_point(verts, point, &pt, n)? {
        Some(nearest) => euclid(point.clone(), nearest),
        None => return unevaluated(),
      }
    }
    // Solid 2D Triangle/Polygon: distance to the boundary, negated inside.
    "Triangle" | "Polygon" if args.len() == 1 && n == 2 => {
      let Expr::List(vs) = &args[0] else {
        return unevaluated();
      };
      let Some(verts_f) = polygon_verts_f64(vs) else {
        return unevaluated();
      };
      if verts_f.len() < 3 {
        return unevaluated();
      }
      let pf = match (
        crate::functions::math_ast::try_eval_to_f64(&pt[0]),
        crate::functions::math_ast::try_eval_to_f64(&pt[1]),
      ) {
        (Some(x), Some(y)) => [x, y],
        _ => return unevaluated(),
      };
      let mut closed: Vec<Expr> = vs.iter().cloned().collect();
      closed.push(vs[0].clone());
      let Some(nearest) = line_nearest_point(&closed, point, &pt, n)? else {
        return unevaluated();
      };
      let dist = euclid(point.clone(), nearest);
      if point_in_polygon_2d(&verts_f, pf, 1e-10) {
        binop(BinaryOperator::Times, Expr::Integer(-1), dist)
      } else {
        dist
      }
    }
    _ => return unevaluated(),
  };

  crate::evaluator::evaluate_expr_to_expr(&expr)
}

// ─── FindShortestCurve / ShortestCurveDistance ─────────────────────────

/// Numeric coordinates of a point expression: a list whose components all
/// evaluate to machine floats (exact rationals and constants included).
fn shortest_curve_point(e: &Expr) -> Option<Vec<f64>> {
  if let Expr::List(items) = e {
    items
      .iter()
      .map(crate::functions::math_ast::try_eval_to_f64)
      .collect()
  } else {
    None
  }
}

/// Center and radius of a `Circle`/`Sphere`/`Disk`/`Ball` primitive with the
/// Wolfram Language defaults filled in (center at the origin, radius 1).
/// Fails for the ellipse forms (radius given as a list) and for the arc /
/// sector forms (a third argument), which are not handled by the geodesic
/// code below.
fn circle_center_radius(args: &[Expr], dim: usize) -> Option<(Expr, Expr)> {
  if args.len() > 2 {
    return None;
  }
  let center = match args.first() {
    None => Expr::List(vec![Expr::Integer(0); dim].into()),
    Some(c) => c.clone(),
  };
  let radius = match args.get(1) {
    None => Expr::Integer(1),
    Some(Expr::List(_)) => return None,
    Some(r) => r.clone(),
  };
  Some((center, radius))
}

/// Convex solid primitives where the shortest curve between two member
/// points is the straight segment. Sector/arc argument forms (which break
/// convexity) are excluded by the argument-count guards.
fn is_convex_region(name: &str, args: &[Expr]) -> bool {
  match name {
    "Disk" | "Ball" => {
      // Disk[c, {rx, ry}] (an ellipse) is still convex; Disk[c, r, {θ1, θ2}]
      // (a sector) is not always geodesically convex, so require ≤ 2 args.
      args.len() <= 2
    }
    "Rectangle" | "Cuboid" => args.len() <= 2,
    "Triangle" => args.len() == 1,
    _ => false,
  }
}

/// True when `point` provably belongs to `region` (numeric check via
/// RegionMember). `None` means the membership could not be decided (e.g.
/// symbolic coordinates).
fn region_member_check(region: &Expr, point: &Expr) -> Option<bool> {
  match compute_region_member(region, point) {
    Ok(Expr::Identifier(ref b)) if b == "True" => Some(true),
    Ok(Expr::Identifier(ref b)) if b == "False" => Some(false),
    _ => None,
  }
}

/// Locate `p` on the polyline `verts`: the segment index and parameter in
/// [0, 1] of the closest chain point, provided `p` actually lies on the
/// chain (within a small relative tolerance).
fn polyline_locate(verts: &[Vec<f64>], p: &[f64]) -> Option<(usize, f64)> {
  let dist = |a: &[f64], b: &[f64]| -> f64 {
    a.iter()
      .zip(b)
      .map(|(x, y)| (x - y) * (x - y))
      .sum::<f64>()
      .sqrt()
  };
  let mut best: Option<(usize, f64, f64)> = None;
  for i in 0..verts.len() - 1 {
    let a = &verts[i];
    let b = &verts[i + 1];
    let len2: f64 = a.iter().zip(b).map(|(x, y)| (y - x) * (y - x)).sum();
    let u = if len2 == 0.0 {
      0.0
    } else {
      let dot: f64 = p
        .iter()
        .zip(a.iter().zip(b))
        .map(|(pi, (ai, bi))| (pi - ai) * (bi - ai))
        .sum();
      (dot / len2).clamp(0.0, 1.0)
    };
    let proj: Vec<f64> =
      a.iter().zip(b).map(|(ai, bi)| ai + u * (bi - ai)).collect();
    let err = dist(p, &proj);
    if best.is_none_or(|(_, _, e)| err < e) {
      best = Some((i, u, err));
    }
  }
  let (i, u, err) = best?;
  let scale = p.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
  (err <= 1e-8 * scale).then_some((i, u))
}

/// The shortest path along the polyline `verts` between two points lying on
/// it. Walks the chain between the two positions; for a closed chain (first
/// vertex equals last) both ways around are compared. Returns the path
/// vertices (starting at `s`, ending at `t`) and its length, or `None` when
/// either point is not on the polyline.
fn polyline_shortest_path(
  verts: &[Vec<f64>],
  s: &[f64],
  t: &[f64],
) -> Option<(Vec<Vec<f64>>, f64)> {
  let dim = verts.first()?.len();
  if verts.len() < 2
    || s.len() != dim
    || t.len() != dim
    || verts.iter().any(|v| v.len() != dim)
  {
    return None;
  }
  let dist = |a: &[f64], b: &[f64]| -> f64 {
    a.iter()
      .zip(b)
      .map(|(x, y)| (x - y) * (x - y))
      .sum::<f64>()
      .sqrt()
  };
  let (si, su) = polyline_locate(verts, s)?;
  let (ti, tu) = polyline_locate(verts, t)?;

  // Drop consecutive duplicates (a located endpoint often coincides with a
  // chain vertex) and measure the result.
  let finish = |raw: Vec<Vec<f64>>| -> (Vec<Vec<f64>>, f64) {
    let mut path: Vec<Vec<f64>> = Vec::with_capacity(raw.len());
    for p in raw {
      if path.last().map(|q| dist(q, &p) > 1e-12) != Some(false) {
        path.push(p);
      }
    }
    if path.len() == 1 {
      path.push(path[0].clone());
    }
    let len = path.windows(2).map(|w| dist(&w[0], &w[1])).sum();
    (path, len)
  };

  // Forward walk from the earlier chain position to the later one.
  let (from, to, reversed) = if (si, su) <= (ti, tu) {
    ((si, su, s), (ti, tu, t), false)
  } else {
    ((ti, tu, t), (si, su, s), true)
  };
  let mut raw = vec![from.2.to_vec()];
  for v in verts.iter().take(to.0 + 1).skip(from.0 + 1) {
    raw.push(v.clone());
  }
  raw.push(to.2.to_vec());
  let (mut path, mut len) = finish(raw);

  // A closed chain also offers the way around through the seam.
  if dist(&verts[0], verts.last().unwrap()) <= 1e-12 {
    let mut raw = vec![from.2.to_vec()];
    for v in verts.iter().take(from.0 + 1).rev() {
      raw.push(v.clone());
    }
    // verts[len-1] == verts[0] was just visited; continue before the seam.
    for v in verts.iter().take(verts.len() - 1).skip(to.0 + 1).rev() {
      raw.push(v.clone());
    }
    raw.push(to.2.to_vec());
    let (alt_path, alt_len) = finish(raw);
    if alt_len < len {
      path = alt_path;
      len = alt_len;
    }
  }

  if reversed {
    path.reverse();
  }
  Some((path, len))
}

/// A polyline path as a machine-precision Line[…] expression, matching the
/// mesh-based (numeric) results wolframscript produces for curve regions.
/// The two query points `s` and `t` are the path's first and last vertices;
/// wolframscript returns them verbatim (exactly as supplied), so they are
/// kept as their original expressions while the interior mesh vertices are
/// emitted at machine precision.
fn polyline_path_to_line(path: &[Vec<f64>], s: &Expr, t: &Expr) -> Expr {
  let last = path.len().saturating_sub(1);
  Expr::FunctionCall {
    name: "Line".to_string(),
    args: vec![Expr::List(
      path
        .iter()
        .enumerate()
        .map(|(i, p)| {
          if i == 0 {
            s.clone()
          } else if i == last {
            t.clone()
          } else {
            Expr::List(
              p.iter().map(|&x| Expr::Real(x)).collect::<Vec<_>>().into(),
            )
          }
        })
        .collect::<Vec<_>>()
        .into(),
    )]
    .into(),
  }
}

/// FindShortestCurve[reg, s, t] — the shortest curve (minimizing geodesic)
/// between two points s and t on the region reg. Handled analytically:
///   • Circle[c, r]  → the shorter arc, as Circle[c, r, {θ1, θ2}]
///   • convex solids (Disk, Ball, Rectangle, Cuboid, Triangle) → Line[{s, t}]
///   • Line[{p1, …, pn}] chains → the sub-path along the polyline, at machine
///     precision (wolframscript treats curve regions as meshes)
/// Anything else stays unevaluated.
fn compute_find_shortest_curve(
  region: &Expr,
  s: &Expr,
  t: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "FindShortestCurve".to_string(),
      args: vec![region.clone(), s.clone(), t.clone()].into(),
    })
  };
  let Expr::FunctionCall { name, args } = region else {
    return unevaluated();
  };
  let call = |n: &str, a: Vec<Expr>| Expr::FunctionCall {
    name: n.to_string(),
    args: a.into(),
  };

  match name.as_str() {
    "Circle" => {
      let Some((center, radius)) = circle_center_radius(args, 2) else {
        return unevaluated();
      };
      // The arc endpoints must be concrete: the short way around depends on
      // the numeric angles.
      let (Some(cf), Some(rf), Some(sf), Some(tf)) = (
        shortest_curve_point(&center),
        crate::functions::math_ast::try_eval_to_f64(&radius),
        shortest_curve_point(s),
        shortest_curve_point(t),
      ) else {
        return unevaluated();
      };
      if cf.len() != 2 || sf.len() != 2 || tf.len() != 2 {
        return unevaluated();
      }
      // Both points must lie on the circle.
      for p in [&sf, &tf] {
        let d = ((p[0] - cf[0]).powi(2) + (p[1] - cf[1]).powi(2)).sqrt();
        if (d - rf).abs() > 1e-8 * rf.abs().max(1.0) {
          return unevaluated();
        }
      }
      // Exact angular position of a point: ArcTan[x - cx, y - cy].
      let (Expr::List(cc), Expr::List(sc), Expr::List(tc)) = (&center, s, t)
      else {
        return unevaluated();
      };
      let angle_of = |p: &crate::ExprList| -> Result<Expr, InterpreterError> {
        let comp = |i: usize| {
          call(
            "Plus",
            vec![
              p[i].clone(),
              call("Times", vec![Expr::Integer(-1), cc[i].clone()]),
            ],
          )
        };
        evaluate_expr_to_expr(&call("ArcTan", vec![comp(0), comp(1)]))
      };
      let theta_s = angle_of(sc)?;
      let theta_t = angle_of(tc)?;
      let ns = (sf[1] - cf[1]).atan2(sf[0] - cf[0]);
      let nt = (tf[1] - cf[1]).atan2(tf[0] - cf[0]);
      let ((lo, nlo), (hi, nhi)) = if ns <= nt {
        ((theta_s, ns), (theta_t, nt))
      } else {
        ((theta_t, nt), (theta_s, ns))
      };
      let spec = if nhi - nlo <= std::f64::consts::PI + 1e-12 {
        vec![lo, hi]
      } else {
        // The short way crosses the ±π branch cut of ArcTan: start at the
        // larger angle and end at the smaller one shifted by a full turn.
        let shifted = evaluate_expr_to_expr(&call(
          "Plus",
          vec![
            lo,
            call(
              "Times",
              vec![Expr::Integer(2), Expr::Constant("Pi".to_string())],
            ),
          ],
        ))?;
        vec![hi, shifted]
      };
      Ok(call(
        "Circle",
        vec![center, radius, Expr::List(spec.into())],
      ))
    }
    _ if is_convex_region(name, args) => {
      // In a convex solid the geodesic is the straight segment. Both
      // endpoints must provably belong to the region.
      for p in [s, t] {
        if region_member_check(region, p) != Some(true) {
          return unevaluated();
        }
      }
      Ok(call(
        "Line",
        vec![Expr::List(vec![s.clone(), t.clone()].into())],
      ))
    }
    "Line" if args.len() == 1 => {
      let Expr::List(pts) = &args[0] else {
        return unevaluated();
      };
      let Some(verts) = pts
        .iter()
        .map(shortest_curve_point)
        .collect::<Option<Vec<_>>>()
      else {
        return unevaluated();
      };
      let (Some(sf), Some(tf)) =
        (shortest_curve_point(s), shortest_curve_point(t))
      else {
        return unevaluated();
      };
      match polyline_shortest_path(&verts, &sf, &tf) {
        Some((path, _)) => Ok(polyline_path_to_line(&path, s, t)),
        None => unevaluated(),
      }
    }
    _ => unevaluated(),
  }
}

/// ShortestCurveDistance[reg, s, t] — the geodesic distance between two
/// points on reg, i.e. ArcLength[FindShortestCurve[reg, s, t]]:
///   • Circle/Sphere → r ArcCos[(s-c).(t-c)/r²] (kept symbolic when the
///     coordinates are, e.g. ShortestCurveDistance[Sphere[], {1,0,0},
///     {x,y,z}] gives ArcCos[x])
///   • convex solids → the Euclidean distance Sqrt[Σ (tᵢ-sᵢ)²]
///   • Line[…] chains → the machine-precision length along the polyline
fn compute_shortest_curve_distance(
  region: &Expr,
  s: &Expr,
  t: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "ShortestCurveDistance".to_string(),
      args: vec![region.clone(), s.clone(), t.clone()].into(),
    })
  };
  let Expr::FunctionCall { name, args } = region else {
    return unevaluated();
  };
  let call = |n: &str, a: Vec<Expr>| Expr::FunctionCall {
    name: n.to_string(),
    args: a.into(),
  };
  let sub = |a: &Expr, b: &Expr| {
    call(
      "Plus",
      vec![a.clone(), call("Times", vec![Expr::Integer(-1), b.clone()])],
    )
  };

  match name.as_str() {
    "Circle" | "Sphere" => {
      let dim = if name.as_str() == "Sphere" { 3 } else { 2 };
      let Some((center, radius)) = circle_center_radius(args, dim) else {
        return unevaluated();
      };
      let (Expr::List(cc), Expr::List(sc), Expr::List(tc)) = (&center, s, t)
      else {
        return unevaluated();
      };
      if sc.len() != cc.len() || tc.len() != cc.len() {
        return unevaluated();
      }
      // Concrete points must lie on the circle/sphere; symbolic coordinates
      // are taken at face value (the doc formula case).
      if let (Some(cf), Some(rf)) = (
        shortest_curve_point(&center),
        crate::functions::math_ast::try_eval_to_f64(&radius),
      ) {
        for p in [s, t] {
          if let Some(pf) = shortest_curve_point(p) {
            let d2: f64 =
              pf.iter().zip(&cf).map(|(a, b)| (a - b) * (a - b)).sum();
            if (d2.sqrt() - rf).abs() > 1e-8 * rf.abs().max(1.0) {
              return unevaluated();
            }
          }
        }
      }
      // r ArcCos[(s-c).(t-c)/r²] — the great-circle distance.
      let dot = call(
        "Plus",
        (0..cc.len())
          .map(|i| {
            call("Times", vec![sub(&sc[i], &cc[i]), sub(&tc[i], &cc[i])])
          })
          .collect(),
      );
      let unit_radius = matches!(radius, Expr::Integer(1));
      let inner = if unit_radius {
        dot
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(dot),
          right: Box::new(call(
            "Power",
            vec![radius.clone(), Expr::Integer(2)],
          )),
        }
      };
      let arc = call("ArcCos", vec![inner]);
      let expr = if unit_radius {
        arc
      } else {
        call("Times", vec![radius, arc])
      };
      evaluate_expr_to_expr(&expr)
    }
    _ if is_convex_region(name, args) => {
      for p in [s, t] {
        // Reject only points that provably lie outside; symbolic points
        // fall through to the distance formula.
        if region_member_check(region, p) == Some(false) {
          return unevaluated();
        }
      }
      let (Expr::List(sc), Expr::List(tc)) = (s, t) else {
        return unevaluated();
      };
      if sc.len() != tc.len() {
        return unevaluated();
      }
      // Sqrt[Σ (tᵢ-sᵢ)²] — spelled out (rather than EuclideanDistance) so
      // symbolic coordinates give Sqrt[x² + (-1 + y)²] without Abs, as
      // wolframscript does for this function.
      let sum = call(
        "Plus",
        (0..sc.len())
          .map(|i| call("Power", vec![sub(&tc[i], &sc[i]), Expr::Integer(2)]))
          .collect(),
      );
      evaluate_expr_to_expr(&make_sqrt(sum))
    }
    "Line" if args.len() == 1 => {
      let Expr::List(pts) = &args[0] else {
        return unevaluated();
      };
      let Some(verts) = pts
        .iter()
        .map(shortest_curve_point)
        .collect::<Option<Vec<_>>>()
      else {
        return unevaluated();
      };
      let (Some(sf), Some(tf)) =
        (shortest_curve_point(s), shortest_curve_point(t))
      else {
        return unevaluated();
      };
      match polyline_shortest_path(&verts, &sf, &tf) {
        Some((_, len)) => Ok(Expr::Real(len)),
        None => unevaluated(),
      }
    }
    _ => unevaluated(),
  }
}

// ─── Area ──────────────────────────────────────────────────────────────

/// Compute the area of a geometric region.
/// `Region[reg, opts…]` is a display wrapper around a geometric region;
/// region functions operate on the wrapped region itself.
fn strip_region_wrapper(expr: &Expr) -> &Expr {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Region"
    && !args.is_empty()
  {
    &args[0]
  } else {
    expr
  }
}

/// PolygonCoordinates[Polygon[pts]] / [Triangle[pts]] — the polygon vertices in
/// canonical (Sort) order. Only non-degenerate 2-D polygons are handled; the
/// call is left unevaluated for 3-D vertices, fewer than three points, or a
/// zero-area (collinear/duplicate) polygon, matching wolframscript.
fn polygon_coordinates(
  region: &Expr,
  orig: &[Expr],
) -> Result<Expr, InterpreterError> {
  let uneval = || Ok(unevaluated("PolygonCoordinates", orig));
  let pts = match region {
    Expr::FunctionCall { name, args }
      if (name == "Polygon" || name == "Triangle") && !args.is_empty() =>
    {
      match &args[0] {
        Expr::List(pts) => pts,
        _ => return uneval(),
      }
    }
    _ => return uneval(),
  };
  // Every vertex must be a 2-D point.
  if pts.len() < 3
    || !pts
      .iter()
      .all(|p| matches!(p, Expr::List(c) if c.len() == 2))
  {
    return uneval();
  }
  // Shoelace area: Sum_i (x_i*y_{i+1} - x_{i+1}*y_i). Build it symbolically so
  // the zero test is exact for integer/rational vertices.
  let coord = |p: &Expr, i: usize| -> Expr {
    if let Expr::List(c) = p {
      c[i].clone()
    } else {
      Expr::Integer(0)
    }
  };
  let mut terms = Vec::with_capacity(pts.len());
  for i in 0..pts.len() {
    let j = (i + 1) % pts.len();
    let cross = Expr::FunctionCall {
      name: "Subtract".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![coord(&pts[i], 0), coord(&pts[j], 1)].into(),
        },
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![coord(&pts[j], 0), coord(&pts[i], 1)].into(),
        },
      ]
      .into(),
    };
    terms.push(cross);
  }
  let area2 = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  })?;
  // A zero (twice-)area means the polygon is degenerate (collinear/duplicate).
  let degenerate = match &area2 {
    Expr::Integer(0) => true,
    Expr::Real(r) => r.abs() < 1e-12,
    _ => false,
  };
  if degenerate {
    return uneval();
  }
  // Canonical vertex order.
  crate::evaluator::evaluate_function_call_ast(
    "Sort",
    &[Expr::List(pts.clone())],
  )
}

fn compute_region_measure(expr: &Expr) -> Result<Expr, InterpreterError> {
  // Ellipsoid[c, {r1, ..., rn}]:
  //   2D -> Pi * r1 * r2
  //   3D -> (4 Pi r1 r2 r3) / 3
  // Other shapes: not yet routed through RegionMeasure (Area/Volume
  // already cover Disk, Cuboid, Sphere, Cylinder, Cone elsewhere).
  if let Expr::FunctionCall { name, args } = expr
    && name == "Ellipsoid"
    && args.len() == 2
    && let (Expr::List(center), Expr::List(radii)) = (&args[0], &args[1])
    && center.len() == radii.len()
  {
    match radii.len() {
      2 => {
        let area = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Constant("Pi".to_string()),
            radii[0].clone(),
            radii[1].clone(),
          ]
          .into(),
        };
        return crate::evaluator::evaluate_expr_to_expr(&area);
      }
      3 => {
        let product = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(4),
            Expr::Constant("Pi".to_string()),
            radii[0].clone(),
            radii[1].clone(),
            radii[2].clone(),
          ]
          .into(),
        };
        let volume = binop(BinaryOperator::Divide, product, Expr::Integer(3));
        return crate::evaluator::evaluate_expr_to_expr(&volume);
      }
      _ => {}
    }
  }

  // For the remaining primitives, RegionMeasure is the measure of the
  // region's *intrinsic* dimension, so it delegates to the matching helper:
  //   2D regions / surfaces → Area, 3D solids → Volume, curves → ArcLength.
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionMeasure".to_string(),
      args: vec![expr.clone()].into(),
    })
  };
  // A delegated helper may return its own unevaluated wrapper (Area[…],
  // Volume[…], ArcLength[…]) when it cannot handle the shape; in that case
  // present it back as an unevaluated RegionMeasure instead.
  let or_unevaluated = |result: Result<Expr, InterpreterError>| match &result {
    Ok(Expr::FunctionCall { name, .. })
      if matches!(name.as_str(), "Area" | "Volume" | "ArcLength") =>
    {
      unevaluated()
    }
    _ => result,
  };

  if let Expr::FunctionCall { name, args } = expr {
    match name.as_str() {
      "StadiumShape" if stadium_parts(args).is_some() => {
        let (p1, p2, r) = stadium_parts(args).unwrap();
        return stadium_area(&p1, &p2, &r, false);
      }
      // 2-dimensional regions: area.
      "Disk" | "Rectangle" | "Triangle" | "Polygon" | "Annulus" => {
        return or_unevaluated(compute_area(expr));
      }
      // Unbounded regions have infinite measure in their intrinsic dimension.
      "HalfPlane" | "InfinitePlane" | "HalfLine" | "InfiniteLine"
      | "HalfSpace" | "ConicHullRegion" => {
        return Ok(Expr::Identifier("Infinity".to_string()));
      }
      // Parallelogram[p, {v1, v2}] — the area spanned by v1 and v2 is
      // Sqrt[Det of the Gram matrix] = Sqrt[(v1.v1)(v2.v2) - (v1.v2)^2],
      // which reduces to |Det[{v1, v2}]| in the plane but also covers
      // parallelograms embedded in higher-dimensional space.
      // Parallelogram[] is the unit square {0,0} + {{0,1},{1,0}}.
      // A two-vector Parallelepiped is the same planar region; the
      // three-vector form falls through to the volume arm below.
      "Parallelogram" | "Parallelepiped"
        if parallelogram_parts(args).is_some() =>
      {
        let Some((_, v1, v2)) = parallelogram_parts(args) else {
          return unevaluated();
        };
        if v1.len() == 2 {
          return det_measure(
            vec![Expr::List(v1.into()), Expr::List(v2.into())],
            1,
          );
        }
        let dot = |a: &[Expr], b: &[Expr]| Expr::FunctionCall {
          name: "Dot".to_string(),
          args: vec![
            Expr::List(a.to_vec().into()),
            Expr::List(b.to_vec().into()),
          ]
          .into(),
        };
        let gram_det = Expr::FunctionCall {
          name: "Subtract".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![dot(&v1, &v1), dot(&v2, &v2)].into(),
            },
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![dot(&v1, &v2), Expr::Integer(2)].into(),
            },
          ]
          .into(),
        };
        let area = Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![gram_det].into(),
        };
        return crate::evaluator::evaluate_expr_to_expr(&area);
      }
      // Torus[{x,y,z}, {r1, r2}] is the surface with inner radius r1 and
      // outer radius r2: tube radius a = (r2-r1)/2 around a circle of
      // radius c = (r1+r2)/2, so the area is 4 Pi^2 a c = Pi^2 (r2^2 - r1^2).
      // FilledTorus is the enclosed solid: 2 Pi^2 a^2 c.
      "Torus" | "FilledTorus" => {
        let Some((_, r1, r2)) = torus_parts(args) else {
          return unevaluated();
        };
        let half = |e: Expr| binop(BinaryOperator::Divide, e, Expr::Integer(2));
        let tube = half(Expr::FunctionCall {
          name: "Subtract".to_string(),
          args: vec![r2.clone(), r1.clone()].into(),
        });
        let ring = half(Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![r1, r2].into(),
        });
        let pi_sq = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![Expr::Constant("Pi".to_string()), Expr::Integer(2)].into(),
        };
        let measure = if name == "Torus" {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(4), pi_sq, tube, ring].into(),
          }
        } else {
          let tube_sq = Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![tube, Expr::Integer(2)].into(),
          };
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(2), pi_sq, tube_sq, ring].into(),
          }
        };
        return crate::evaluator::evaluate_expr_to_expr(&measure);
      }
      // A Sphere is a 2-dimensional surface embedded in 3-space, so its
      // measure is the surface area (Area already returns 4 Pi r^2).
      "Sphere" => {
        return or_unevaluated(compute_area(expr));
      }
      // 3-dimensional solids: volume.
      "Cuboid" | "Cylinder" | "Cone" | "Parallelepiped" | "Prism"
      | "Pyramid" => {
        return or_unevaluated(compute_volume(expr));
      }
      // Platonic-solid primitives are 3-D solids: volume.
      "Cube" | "Hexahedron" | "Octahedron" | "Dodecahedron" | "Icosahedron"
        if platonic_center_edge(args).is_some() =>
      {
        return or_unevaluated(compute_volume(expr));
      }
      // wolframscript's RegionMeasure for a SphericalShell is (quirkily)
      // NOT its volume but 4 Pi (r2^2 - r1^2) — replicated for
      // conformance.
      "SphericalShell" if spherical_shell_radii(args).is_some() => {
        let (r1, r2) = spherical_shell_radii(args).unwrap();
        return spherical_shell_measure(&r1, &r2, 2, 1);
      }
      // A CapsuleShape measures as its volume.
      "CapsuleShape" if capsule_height_sq_radius(args).is_some() => {
        return or_unevaluated(compute_volume(expr));
      }
      // wolframscript's RegionMeasure for any valid DiskSegment is
      // (quirkily) the constant 2 — its dimension, not its area.
      // Replicated for conformance.
      "DiskSegment"
        if disk_segment_parts(args).is_some_and(|(.., d)| d >= 0.0) =>
      {
        return Ok(Expr::Integer(2));
      }
      // A simplex's measure is in its intrinsic dimension: for n vertices in
      // (n-1)-space it is |Det[edges]| / (n-1)! (triangle area, tetrahedron
      // volume, ...).
      "Tetrahedron" | "Simplex" => {
        // The regular-tetrahedron primitive forms ([], [l], [center, l], …)
        // measure as their volume.
        if name == "Tetrahedron" && platonic_center_edge(args).is_some() {
          return or_unevaluated(compute_volume(expr));
        }
        // Simplex[n] is the standard n-simplex (origin plus the n unit
        // vectors); its intrinsic measure is 1/n!. The degenerate 0-simplex
        // (a point) is left unevaluated, matching wolframscript.
        if name == "Simplex"
          && args.len() == 1
          && let Expr::Integer(n) = &args[0]
          && *n >= 1
        {
          return Ok(crate::functions::math_ast::make_rational(
            1,
            factorial_small(*n as usize),
          ));
        }
        if args.len() == 1
          && let Expr::List(pts) = &args[0]
          && let Some(edges) = simplex_edges(pts)
        {
          return det_measure(edges, factorial_small(pts.len() - 1));
        }
        return unevaluated();
      }
      // 1-dimensional curves: arc length.
      "Circle" | "Line" => {
        return or_unevaluated(compute_arc_length(expr));
      }
      // Ball[c, r] is an n-ball whose dimension is the length of its center
      // vector (default {0, 0, 0}, radius 1). Its measure is the closed-form
      // n-ball volume  Pi^(n/2) r^n / Gamma[n/2 + 1].
      "Ball" => {
        let (n, radius) = match args.len() {
          0 => (3usize, Expr::Integer(1)),
          1 | 2 => {
            let Expr::List(center) = &args[0] else {
              return unevaluated();
            };
            if center.is_empty() {
              return unevaluated();
            }
            let radius = if args.len() == 2 {
              args[1].clone()
            } else {
              Expr::Integer(1)
            };
            (center.len(), radius)
          }
          _ => return unevaluated(),
        };
        let half_n = binop(
          BinaryOperator::Divide,
          Expr::Integer(n as i128),
          Expr::Integer(2),
        );
        let pi_pow = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![Expr::Constant("Pi".to_string()), half_n.clone()].into(),
        };
        let r_pow = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![radius, Expr::Integer(n as i128)].into(),
        };
        let gamma = Expr::FunctionCall {
          name: "Gamma".to_string(),
          args: vec![Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![half_n, Expr::Integer(1)].into(),
          }]
          .into(),
        };
        let measure = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![pi_pow, r_pow].into(),
          }),
          right: Box::new(gamma),
        };
        return crate::evaluator::evaluate_expr_to_expr(&measure);
      }
      // Point — 0-dimensional, so the measure is the counting measure:
      // a single point ({x, y, …}) has measure 1; a list of points
      // ({{…}, {…}, …}) has measure equal to the number of points.
      "Point" if args.len() == 1 => {
        if let Expr::List(items) = &args[0]
          && !items.is_empty()
        {
          let all_points = items.iter().all(|e| matches!(e, Expr::List(_)));
          if all_points {
            return Ok(Expr::Integer(items.len() as i128));
          }
          let any_points = items.iter().any(|e| matches!(e, Expr::List(_)));
          if !any_points {
            // A flat coordinate list is a single point.
            return Ok(Expr::Integer(1));
          }
        }
        return unevaluated();
      }
      _ => {}
    }
  }
  unevaluated()
}

/// Compute the intrinsic (manifold) dimension of a geometric region: the
/// number of independent directions you can move within it. This is distinct
/// from the embedding dimension — a Circle in the plane has dimension 1, a
/// Sphere in 3-space has dimension 2.
fn compute_region_dimension(expr: &Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionDimension".to_string(),
      args: vec![expr.clone()].into(),
    })
  };
  // Length of a coordinate-list argument (a single point {x, y, …}).
  let coord_len = |e: &Expr| -> Option<usize> {
    match e {
      Expr::List(items) if !items.is_empty() => Some(items.len()),
      _ => None,
    }
  };
  if let Expr::FunctionCall { name, args } = expr {
    match name.as_str() {
      "StadiumShape" if stadium_parts(args).is_some() => {
        return Ok(Expr::Integer(2));
      }
      // Regions of fixed intrinsic dimension.
      "Disk" | "Rectangle" | "Triangle" | "Polygon" | "RegularPolygon"
      | "Annulus" | "Parallelogram" | "HalfPlane" | "InfinitePlane"
      | "Torus" => return Ok(Expr::Integer(2)),
      // DiskSegment needs a valid argument shape (DiskSegment[] stays
      // unevaluated in wolframscript).
      "DiskSegment" if disk_segment_parts(args).is_some() => {
        return Ok(Expr::Integer(2));
      }
      // A half-space has the dimension of its ambient space.
      "HalfSpace" if half_space_parts(args).is_some() => {
        let (normal, _) = half_space_parts(args).unwrap();
        return Ok(Expr::Integer(normal.len() as i128));
      }
      "Circle" | "Line" | "HalfLine" | "InfiniteLine" => {
        return Ok(Expr::Integer(1));
      }
      "Cylinder" | "Cone" | "Tetrahedron" | "FilledTorus" | "Cube"
      | "Hexahedron" | "Octahedron" | "Dodecahedron" | "Icosahedron"
      | "SphericalShell" | "CapsuleShape" | "Prism" | "Pyramid" => {
        return Ok(Expr::Integer(3));
      }
      "Point" => return Ok(Expr::Integer(0)),
      // Ball / Ellipsoid are solid: their dimension is the length of the
      // center vector (default unit ball / sphere lives in 3-space).
      "Ball" | "Ellipsoid" => {
        if args.is_empty() {
          return Ok(Expr::Integer(3));
        }
        if let Some(n) = coord_len(&args[0]) {
          return Ok(Expr::Integer(n as i128));
        }
        return unevaluated();
      }
      // Cuboid[p1, …] is a solid hyper-rectangle whose dimension is the
      // length of its defining corner (default {0,0,0} → 3).
      "Cuboid" => {
        if args.is_empty() {
          return Ok(Expr::Integer(3));
        }
        if let Some(n) = coord_len(&args[0]) {
          return Ok(Expr::Integer(n as i128));
        }
        return unevaluated();
      }
      // A Sphere is the (n-1)-dimensional surface of an n-ball, so its
      // dimension is (length of center) - 1; the default sphere is 3-D.
      "Sphere" => {
        if args.is_empty() {
          return Ok(Expr::Integer(2));
        }
        if let Some(n) = coord_len(&args[0])
          && n >= 1
        {
          return Ok(Expr::Integer(n as i128 - 1));
        }
        return unevaluated();
      }
      // Simplex[n] is an n-simplex; Simplex[{p0, …, pk}] is a k-simplex.
      "Simplex" if args.len() == 1 => match &args[0] {
        Expr::Integer(n) if *n >= 0 => return Ok(Expr::Integer(*n)),
        Expr::List(pts) if !pts.is_empty() => {
          return Ok(Expr::Integer(pts.len() as i128 - 1));
        }
        _ => return unevaluated(),
      },
      // Parallelepiped[p, {v1, …, vk}] is spanned by k vectors.
      "Parallelepiped" if args.len() == 2 => {
        if let Expr::List(vecs) = &args[1]
          && !vecs.is_empty()
        {
          return Ok(Expr::Integer(vecs.len() as i128));
        }
        return unevaluated();
      }
      _ => {}
    }
  }
  unevaluated()
}

/// RegionEmbeddingDimension[region] — the dimension of the ambient space the
/// region lives in, i.e. the number of coordinates. This equals the number of
/// `{min, max}` pairs in the region's bounding box.
fn compute_region_embedding_dimension(
  expr: &Expr,
) -> Result<Expr, InterpreterError> {
  if let Expr::FunctionCall { name, args } = expr {
    match name.as_str() {
      "StadiumShape" if stadium_parts(args).is_some() => {
        return Ok(Expr::Integer(2));
      }
      "Torus" | "FilledTorus" => {
        if torus_parts(args).is_some() {
          return Ok(Expr::Integer(3));
        }
      }
      "Parallelogram" => {
        if let Some((p, _, _)) = parallelogram_parts(args) {
          return Ok(Expr::Integer(p.len() as i128));
        }
      }
      // HalfPlane[{p1, p2}, w] / HalfPlane[p, v, w] — ambient space of the
      // defining points. InfinitePlane[{p1, p2, p3}] / InfinitePlane[p,
      // {v1, v2}] likewise.
      "HalfPlane" | "InfinitePlane" => {
        let first_coord = match args.first() {
          Some(Expr::List(items)) if !items.is_empty() => {
            if let Expr::List(p) = &items[0] {
              // A list of defining points: use one point's length.
              Some(p.len())
            } else {
              // A single coordinate vector.
              Some(items.len())
            }
          }
          _ => None,
        };
        if let Some(n) = first_coord {
          return Ok(Expr::Integer(n as i128));
        }
      }
      _ => {}
    }
  }
  match compute_region_bounds(expr)? {
    Expr::List(ref bounds) => Ok(Expr::Integer(bounds.len() as i128)),
    _ => Ok(Expr::FunctionCall {
      name: "RegionEmbeddingDimension".to_string(),
      args: vec![expr.clone()].into(),
    }),
  }
}

/// Compute the axis-aligned bounding box of a geometric region as a list of
/// `{min, max}` pairs, one per dimension.
fn compute_region_bounds(expr: &Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "RegionBounds".to_string(),
    args: vec![expr.clone()].into(),
  };
  // HalfLine[{p1, p2}] — half-line from p1 toward p2. Each dimension's
  // bounds depend on the sign of (p2 - p1)[d]:
  //   positive  → [p1[d], Infinity]
  //   negative  → [-Infinity, p1[d]]
  //   zero      → [p1[d], p1[d]]
  if let Expr::FunctionCall { name, args } = expr
    && name == "HalfLine"
    && args.len() == 1
    && let Expr::List(points) = &args[0]
    && points.len() == 2
    && let (Expr::List(p1), Expr::List(p2)) = (&points[0], &points[1])
    && p1.len() == p2.len()
  {
    let inf = || Expr::Identifier("Infinity".to_string());
    let neg_inf = || Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    };
    let mut bounds = Vec::with_capacity(p1.len());
    for (a, b) in p1.iter().zip(p2.iter()) {
      let diff = Expr::FunctionCall {
        name: "Subtract".to_string(),
        args: vec![b.clone(), a.clone()].into(),
      };
      let diff_eval = crate::evaluator::evaluate_expr_to_expr(&diff)
        .unwrap_or_else(|_| diff.clone());
      let sign = match &diff_eval {
        Expr::Integer(n) => Some(n.cmp(&0)),
        Expr::Real(f) if *f > 0.0 => Some(std::cmp::Ordering::Greater),
        Expr::Real(f) if *f < 0.0 => Some(std::cmp::Ordering::Less),
        Expr::Real(f) if *f == 0.0 => Some(std::cmp::Ordering::Equal),
        _ => None,
      };
      let pair = match sign {
        Some(std::cmp::Ordering::Greater) => {
          Expr::List(vec![a.clone(), inf()].into())
        }
        Some(std::cmp::Ordering::Less) => {
          Expr::List(vec![neg_inf(), a.clone()].into())
        }
        Some(std::cmp::Ordering::Equal) => {
          Expr::List(vec![a.clone(), a.clone()].into())
        }
        None => return Ok(unevaluated()),
      };
      bounds.push(pair);
    }
    return Ok(Expr::List(bounds.into()));
  }
  if let Expr::FunctionCall { name, args } = expr {
    // Line/Triangle/Polygon[{p1, ..., pk}] — min/max over all vertices.
    if matches!(name.as_str(), "Line" | "Triangle" | "Polygon")
      && args.len() == 1
      && let Expr::List(points) = &args[0]
    {
      return Ok(points_bounds(points).unwrap_or_else(unevaluated));
    }
    // Simplex[n]: the standard n-simplex spans [0, 1] in each of its n
    // coordinates. Simplex[{p0, …, pk}]: the box over its vertices.
    if name == "Simplex" && args.len() == 1 {
      if let Expr::Integer(n) = &args[0]
        && *n >= 1
      {
        let pair = Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into());
        return Ok(Expr::List(vec![pair; *n as usize].into()));
      }
      if let Expr::List(points) = &args[0] {
        return Ok(points_bounds(points).unwrap_or_else(unevaluated));
      }
    }
    // Point[{x, y, ...}] — a degenerate box {{x, x}, {y, y}, ...}.
    if name == "Point"
      && args.len() == 1
      && let Expr::List(coords) = &args[0]
      && coords.iter().all(|c| !matches!(c, Expr::List(_)))
    {
      let bounds: Vec<Expr> = coords
        .iter()
        .map(|c| Expr::List(vec![c.clone(), c.clone()].into()))
        .collect();
      return Ok(Expr::List(bounds.into()));
    }
    // Disk/Circle (2D) and Ball/Sphere (3D): center ± radius per dimension.
    // The radius may be a scalar or, for Disk, a {rx, ry} semi-axis list.
    if let Some(dim) = match name.as_str() {
      "Disk" | "Circle" => Some(2),
      "Ball" | "Sphere" => Some(3),
      _ => None,
    } {
      let center: Vec<Expr> = match args.first() {
        Some(Expr::List(c)) => c.to_vec(),
        None => vec![Expr::Integer(0); dim],
        _ => return Ok(unevaluated()),
      };
      if center.len() != dim {
        return Ok(unevaluated());
      }
      let radius_at = |d: usize| -> Expr {
        match args.get(1) {
          Some(Expr::List(rs)) if rs.len() == dim => rs[d].clone(),
          Some(r) if !matches!(r, Expr::List(_)) => r.clone(),
          _ => Expr::Integer(1),
        }
      };
      // Reject a malformed radius spec.
      if let Some(Expr::List(rs)) = args.get(1)
        && rs.len() != dim
      {
        return Ok(unevaluated());
      }
      let mut bounds = Vec::with_capacity(dim);
      for (d, c) in center.iter().enumerate() {
        let r = radius_at(d);
        let lo = eval_binop("Subtract", c.clone(), r.clone());
        let hi = eval_binop("Plus", c.clone(), r);
        bounds.push(Expr::List(vec![lo, hi].into()));
      }
      return Ok(Expr::List(bounds.into()));
    }
    // Torus/FilledTorus — x and y span center ± outer radius r2, z spans
    // center ± tube radius (r2 - r1)/2.
    if matches!(name.as_str(), "Torus" | "FilledTorus")
      && let Some((center, r1, r2)) = torus_parts(args)
    {
      let tube = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::FunctionCall {
          name: "Subtract".to_string(),
          args: vec![r2.clone(), r1].into(),
        }),
        right: Box::new(Expr::Integer(2)),
      };
      let extents = [r2.clone(), r2, tube];
      let bounds: Vec<Expr> = center
        .iter()
        .zip(extents.iter())
        .map(|(c, e)| {
          let lo = eval_binop("Subtract", c.clone(), e.clone());
          let hi = eval_binop("Plus", c.clone(), e.clone());
          Expr::List(vec![lo, hi].into())
        })
        .collect();
      return Ok(Expr::List(bounds.into()));
    }
    // Parallelogram — min/max over the four corners p, p+v1, p+v2, p+v1+v2.
    if name == "Parallelogram"
      && let Some((p, v1, v2)) = parallelogram_parts(args)
    {
      let add = |a: &[Expr], b: &[Expr]| -> Vec<Expr> {
        a.iter()
          .zip(b.iter())
          .map(|(x, y)| eval_binop("Plus", x.clone(), y.clone()))
          .collect()
      };
      let corners: Vec<Expr> = vec![
        Expr::List(p.clone().into()),
        Expr::List(add(&p, &v1).into()),
        Expr::List(add(&p, &v2).into()),
        Expr::List(add(&add(&p, &v1), &v2).into()),
      ];
      return Ok(points_bounds(&corners).unwrap_or_else(unevaluated));
    }
    // Rectangle/Cuboid[c1, c2] — per-dimension sorted corners.
    if matches!(name.as_str(), "Rectangle" | "Cuboid") {
      let default_dim = if name == "Cuboid" { 3 } else { 2 };
      let c1: Vec<Expr> = match args.first() {
        Some(Expr::List(c)) => c.to_vec(),
        None => vec![Expr::Integer(0); default_dim],
        _ => return Ok(unevaluated()),
      };
      let c2: Vec<Expr> = match args.get(1) {
        Some(Expr::List(c)) => c.to_vec(),
        None => vec![Expr::Integer(1); c1.len()],
        _ => return Ok(unevaluated()),
      };
      if c1.len() != c2.len() {
        return Ok(unevaluated());
      }
      let bounds: Vec<Expr> = c1
        .into_iter()
        .zip(c2)
        .map(|(a, b)| {
          let lo = eval_binop("Min", a.clone(), b.clone());
          let hi = eval_binop("Max", a, b);
          Expr::List(vec![lo, hi].into())
        })
        .collect();
      return Ok(Expr::List(bounds.into()));
    }
  }
  Ok(unevaluated())
}

/// Evaluate `head[a, b]` (e.g. Plus/Subtract/Min/Max), falling back to the
/// unevaluated call on error.
fn eval_binop(head: &str, a: Expr, b: Expr) -> Expr {
  let call = Expr::FunctionCall {
    name: head.to_string(),
    args: vec![a, b].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&call).unwrap_or(call)
}

/// Axis-aligned bounding box over a list of coordinate vectors, as
/// `{{min, max}, ...}` per dimension. Returns None for malformed input.
fn points_bounds(points: &[Expr]) -> Option<Expr> {
  let Expr::List(first) = points.first()? else {
    return None;
  };
  let dim = first.len();
  let mut mins: Vec<Expr> = first.to_vec();
  let mut maxs: Vec<Expr> = first.to_vec();
  for p in points.iter().skip(1) {
    let Expr::List(coords) = p else {
      return None;
    };
    if coords.len() != dim {
      return None;
    }
    for (d, c) in coords.iter().enumerate() {
      let cmp_lt = Expr::FunctionCall {
        name: "Less".to_string(),
        args: vec![c.clone(), mins[d].clone()].into(),
      };
      if matches!(
        crate::evaluator::evaluate_expr_to_expr(&cmp_lt),
        Ok(Expr::Identifier(ref s)) if s == "True"
      ) {
        mins[d] = c.clone();
      }
      let cmp_gt = Expr::FunctionCall {
        name: "Greater".to_string(),
        args: vec![c.clone(), maxs[d].clone()].into(),
      };
      if matches!(
        crate::evaluator::evaluate_expr_to_expr(&cmp_gt),
        Ok(Expr::Identifier(ref s)) if s == "True"
      ) {
        maxs[d] = c.clone();
      }
    }
  }
  let bounds: Vec<Expr> = mins
    .into_iter()
    .zip(maxs)
    .map(|(lo, hi)| Expr::List(vec![lo, hi].into()))
    .collect();
  Some(Expr::List(bounds.into()))
}

/// Elementwise `v - base` as a coordinate-vector `List`.
fn coord_vec_sub(v: &[Expr], base: &[Expr]) -> Expr {
  Expr::List(
    v.iter()
      .zip(base.iter())
      .map(|(a, b)| Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          a.clone(),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), b.clone()].into(),
          },
        ]
        .into(),
      })
      .collect::<Vec<_>>()
      .into(),
  )
}

/// `Abs[Det[rows]] / divisor`, evaluated.
fn det_measure(
  rows: Vec<Expr>,
  divisor: i128,
) -> Result<Expr, InterpreterError> {
  let det = Expr::FunctionCall {
    name: "Det".to_string(),
    args: vec![Expr::List(rows.into())].into(),
  };
  let abs = Expr::FunctionCall {
    name: "Abs".to_string(),
    args: vec![det].into(),
  };
  let vol = if divisor == 1 {
    abs
  } else {
    binop(BinaryOperator::Divide, abs, Expr::Integer(divisor))
  };
  crate::evaluator::evaluate_expr_to_expr(&vol)
}

/// A 3-D integer coordinate point `{a, b, c}`.
fn pt3i(a: i128, b: i128, c: i128) -> Expr {
  Expr::List(vec![Expr::Integer(a), Expr::Integer(b), Expr::Integer(c)].into())
}

/// The three coordinates of a 3-D point, or None if `p` is not a length-3 list.
fn as_pt3(p: &Expr) -> Option<Vec<Expr>> {
  match p {
    Expr::List(c) if c.len() == 3 => Some(c.iter().cloned().collect()),
    _ => None,
  }
}

/// The vertex list for a Prism, applying the default triangular prism
/// Prism[] == Prism[{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,0,1},{0,1,1}}].
/// Returns None for malformed arguments.
fn prism_pts(args: &[Expr]) -> Option<Vec<Expr>> {
  match args {
    [] => Some(vec![
      pt3i(0, 0, 0),
      pt3i(1, 0, 0),
      pt3i(0, 1, 0),
      pt3i(0, 0, 1),
      pt3i(1, 0, 1),
      pt3i(0, 1, 1),
    ]),
    [Expr::List(pts)] => Some(pts.iter().cloned().collect()),
    _ => None,
  }
}

/// The vertex list for a (square) Pyramid, applying the default
/// Pyramid[] == Pyramid[{{-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0},{0,0,1}}].
/// Returns None for malformed arguments.
fn pyramid_pts(args: &[Expr]) -> Option<Vec<Expr>> {
  match args {
    [] => Some(vec![
      pt3i(-1, -1, 0),
      pt3i(1, -1, 0),
      pt3i(1, 1, 0),
      pt3i(-1, 1, 0),
      pt3i(0, 0, 1),
    ]),
    [Expr::List(pts)] => Some(pts.iter().cloned().collect()),
    _ => None,
  }
}

/// Volume of a translational triangular Prism[{p1, …, p6}]:
/// (1/2)|Det[p2-p1, p3-p1, p4-p1]|. Defined only for six 3-D points.
fn prism_volume(pts: &[Expr]) -> Option<Result<Expr, InterpreterError>> {
  if pts.len() != 6 {
    return None;
  }
  let c: Vec<Vec<Expr>> = pts.iter().map(as_pt3).collect::<Option<_>>()?;
  let rows = vec![
    coord_vec_sub(&c[1], &c[0]),
    coord_vec_sub(&c[2], &c[0]),
    coord_vec_sub(&c[3], &c[0]),
  ];
  Some(det_measure(rows, 2))
}

/// Volume of a square Pyramid[{p1, p2, p3, p4, apex}] with a quadrilateral base
/// (p1..p4) and apex p5, via the fan triangulation of the base:
/// (1/6)|Det[p2-p1, p3-p1, apex-p1] + Det[p3-p1, p4-p1, apex-p1]|.
/// Defined only for five 3-D points (matching wolframscript, which leaves
/// other vertex counts unevaluated).
fn pyramid_volume(pts: &[Expr]) -> Option<Result<Expr, InterpreterError>> {
  if pts.len() != 5 {
    return None;
  }
  let c: Vec<Vec<Expr>> = pts.iter().map(as_pt3).collect::<Option<_>>()?;
  let apex = &c[4];
  let det_term = |i: usize, j: usize| Expr::FunctionCall {
    name: "Det".to_string(),
    args: vec![Expr::List(
      vec![
        coord_vec_sub(&c[i], &c[0]),
        coord_vec_sub(&c[j], &c[0]),
        coord_vec_sub(apex, &c[0]),
      ]
      .into(),
    )]
    .into(),
  };
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![det_term(1, 2), det_term(2, 3)].into(),
  };
  let abs = Expr::FunctionCall {
    name: "Abs".to_string(),
    args: vec![sum].into(),
  };
  let vol = binop(BinaryOperator::Divide, abs, Expr::Integer(6));
  Some(crate::evaluator::evaluate_expr_to_expr(&vol))
}

/// If `pts` are `n` coordinate vectors each of length `n-1` (a full-dimensional
/// simplex in its own space), return the `n-1` edge vectors from the first
/// vertex; otherwise None.
fn simplex_edges(pts: &[Expr]) -> Option<Vec<Expr>> {
  let n = pts.len();
  if n < 2 {
    return None;
  }
  let coords: Vec<Vec<Expr>> = pts
    .iter()
    .map(|p| match p {
      Expr::List(c) if c.len() == n - 1 => {
        Some(c.iter().cloned().collect::<Vec<_>>())
      }
      _ => None,
    })
    .collect::<Option<_>>()?;
  Some(
    coords[1..]
      .iter()
      .map(|v| coord_vec_sub(v, &coords[0]))
      .collect(),
  )
}

/// `n!` as an `i128` for small `n`.
fn factorial_small(n: usize) -> i128 {
  (1..=n as i128).product::<i128>().max(1)
}

/// Decompose Parallelogram arguments into (base point, v1, v2), applying the
/// default Parallelogram[] == Parallelogram[{0, 0}, {{0, 1}, {1, 0}}].
/// Returns None for malformed argument lists.
fn parallelogram_parts(
  args: &[Expr],
) -> Option<(Vec<Expr>, Vec<Expr>, Vec<Expr>)> {
  match args.len() {
    0 => Some((
      vec![Expr::Integer(0), Expr::Integer(0)],
      vec![Expr::Integer(0), Expr::Integer(1)],
      vec![Expr::Integer(1), Expr::Integer(0)],
    )),
    2 => {
      let Expr::List(p) = &args[0] else {
        return None;
      };
      let Expr::List(vecs) = &args[1] else {
        return None;
      };
      if vecs.len() != 2 {
        return None;
      }
      let (Expr::List(v1), Expr::List(v2)) = (&vecs[0], &vecs[1]) else {
        return None;
      };
      if v1.len() != p.len() || v2.len() != p.len() || p.is_empty() {
        return None;
      }
      Some((p.to_vec(), v1.to_vec(), v2.to_vec()))
    }
    _ => None,
  }
}

/// Decompose Torus/FilledTorus arguments into (center, r1, r2), applying the
/// defaults Torus[] == Torus[{0, 0, 0}, {1/2, 1}]. Returns None for
/// malformed argument lists.
fn torus_parts(args: &[Expr]) -> Option<(Vec<Expr>, Expr, Expr)> {
  let default_center = || vec![Expr::Integer(0); 3];
  let default_radii = || {
    (
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      },
      Expr::Integer(1),
    )
  };
  match args.len() {
    0 => {
      let (r1, r2) = default_radii();
      Some((default_center(), r1, r2))
    }
    1 | 2 => {
      let Expr::List(center) = &args[0] else {
        return None;
      };
      if center.len() != 3 {
        return None;
      }
      let (r1, r2) = match args.get(1) {
        None => default_radii(),
        Some(Expr::List(radii)) if radii.len() == 2 => {
          (radii[0].clone(), radii[1].clone())
        }
        Some(_) => return None,
      };
      Some((center.to_vec(), r1, r2))
    }
    _ => None,
  }
}

/// RegionMoment[reg, {i1, …, in}] — the polynomial moment
/// Integrate[x1^i1 ⋯ xn^in, reg], exactly, for Disk/Ball (any center and
/// radius, both possibly symbolic), Rectangle/Cuboid (symbolic corners
/// allowed), and Triangle (default or explicit vertices).
fn compute_region_moment(
  region: &Expr,
  spec: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionMoment".to_string(),
      args: vec![region.clone(), spec.clone()].into(),
    })
  };
  let ev = |e: &Expr| crate::evaluator::evaluate_expr_to_expr(e);
  let int_e = |n: i128| Expr::Integer(n);
  let times = |fs: Vec<Expr>| {
    let fs: Vec<Expr> = fs
      .into_iter()
      .filter(|f| !matches!(f, Expr::Integer(1)))
      .collect();
    match fs.len() {
      0 => Expr::Integer(1),
      1 => fs.into_iter().next().unwrap(),
      _ => Expr::FunctionCall {
        name: "Times".to_string(),
        args: fs.into(),
      },
    }
  };
  let plus = |ts: Vec<Expr>| match ts.len() {
    0 => Expr::Integer(0),
    1 => ts.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Plus".to_string(),
      args: ts.into(),
    },
  };
  let pow = |b: &Expr, e: i128| -> Expr {
    match e {
      0 => Expr::Integer(1),
      1 => b.clone(),
      _ => Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![b.clone(), Expr::Integer(e)].into(),
      },
    }
  };
  let ratio = |num: Expr, den: Expr| binop(BinaryOperator::Divide, num, den);
  let is_zero = |e: &Expr| matches!(e, Expr::Integer(0));

  // The supported region kinds with their embedding dimension.
  enum MomentShape {
    Round { center: Vec<Expr>, radius: Expr },
    AxisBox { lo: Vec<Expr>, hi: Vec<Expr> },
    Tri { v: Vec<[Expr; 2]> },
  }
  let Expr::FunctionCall { name, args } = region else {
    return unevaluated();
  };
  let coords = |e: &Expr| -> Option<Vec<Expr>> {
    match e {
      Expr::List(items) if !items.is_empty() => {
        Some(items.iter().cloned().collect())
      }
      _ => None,
    }
  };
  let shape = match name.as_str() {
    "Disk" | "Ball" => {
      let default_dim = if name == "Ball" { 3 } else { 2 };
      let center = match args.first() {
        None => vec![Expr::Integer(0); default_dim],
        Some(c) => match coords(c) {
          Some(c) => c,
          None => return unevaluated(),
        },
      };
      let radius = match args.get(1) {
        None => Expr::Integer(1),
        Some(Expr::List(_)) => return unevaluated(),
        Some(r) => r.clone(),
      };
      if args.len() > 2 {
        return unevaluated();
      }
      MomentShape::Round { center, radius }
    }
    "Rectangle" | "Cuboid" => {
      let default_dim = if name == "Cuboid" { 3 } else { 2 };
      let (lo, hi) = match (args.first(), args.get(1)) {
        (None, _) => (
          vec![Expr::Integer(0); default_dim],
          vec![Expr::Integer(1); default_dim],
        ),
        (Some(c1), None) => {
          let lo = match coords(c1) {
            Some(c) => c,
            None => return unevaluated(),
          };
          let hi: Result<Vec<Expr>, InterpreterError> = lo
            .iter()
            .map(|x| {
              ev(&Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![x.clone(), Expr::Integer(1)].into(),
              })
            })
            .collect();
          (lo, hi?)
        }
        (Some(c1), Some(c2)) => match (coords(c1), coords(c2)) {
          (Some(lo), Some(hi)) if lo.len() == hi.len() => (lo, hi),
          _ => return unevaluated(),
        },
      };
      if args.len() > 2 {
        return unevaluated();
      }
      MomentShape::AxisBox { lo, hi }
    }
    "Triangle" => {
      let verts: Vec<[Expr; 2]> = if args.is_empty() {
        vec![
          [Expr::Integer(0), Expr::Integer(0)],
          [Expr::Integer(1), Expr::Integer(0)],
          [Expr::Integer(0), Expr::Integer(1)],
        ]
      } else if args.len() == 1
        && let Expr::List(pts) = &args[0]
        && pts.len() == 3
        && pts
          .iter()
          .all(|p| matches!(p, Expr::List(c) if c.len() == 2))
      {
        pts
          .iter()
          .map(|p| {
            let Expr::List(c) = p else { unreachable!() };
            [c[0].clone(), c[1].clone()]
          })
          .collect()
      } else {
        return unevaluated();
      };
      MomentShape::Tri { v: verts }
    }
    _ => return unevaluated(),
  };
  let dim = match &shape {
    MomentShape::Round { center, .. } => center.len(),
    MomentShape::AxisBox { lo, .. } => lo.len(),
    MomentShape::Tri { .. } => 2,
  };

  // The moment spec: a list of non-negative integers of length dim.
  let powers: Option<Vec<i128>> = match spec {
    Expr::List(items) if items.len() == dim => items
      .iter()
      .map(|e| match e {
        Expr::Integer(n) if *n >= 0 => Some(*n),
        _ => None,
      })
      .collect(),
    _ => None,
  };
  let Some(powers) = powers else {
    crate::emit_message(
      "RegionMoment::mexp: Invalid moment index specification at position 2 in 2. A list of non-negative integers matching the embedding dimension is expected.",
    );
    return unevaluated();
  };
  if powers.iter().sum::<i128>() > 24 {
    return unevaluated();
  }

  match shape {
    // Π (hi^(p+1) - lo^(p+1))/(p+1), per axis.
    MomentShape::AxisBox { lo, hi } => {
      let factors: Vec<Expr> = powers
        .iter()
        .zip(lo.iter().zip(hi.iter()))
        .map(|(&p, (l, h))| {
          ratio(
            plus(vec![pow(h, p + 1), times(vec![int_e(-1), pow(l, p + 1)])]),
            int_e(p + 1),
          )
        })
        .collect();
      ev(&times(factors))
    }
    // Binomial expansion about the center; the centered unit-ball moment
    // with all-even exponents k is Π Gamma[(k_i+1)/2] / Gamma[Σ(k_i+1)/2
    // + 1], scaled by r^(Σk + n).
    MomentShape::Round { center, radius } => {
      let n = center.len();
      let mut terms: Vec<Expr> = Vec::new();
      let mut k = vec![0i128; n];
      'outer: loop {
        // Skip terms with an odd centered exponent (they integrate to 0)
        // and terms multiplied by a literal-zero center power.
        let all_even = k.iter().all(|ki| ki % 2 == 0);
        let zero_factor = k
          .iter()
          .zip(powers.iter().zip(center.iter()))
          .any(|(ki, (pi, ci))| ki < pi && is_zero(ci));
        if all_even && !zero_factor {
          let mut factors: Vec<Expr> = Vec::new();
          let mut gamma_num: Vec<Expr> = Vec::new();
          let mut half_sum_num: i128 = 0;
          fn binom_i128(n: i128, k: i128) -> i128 {
            let k = k.min(n - k);
            let mut acc: i128 = 1;
            for t in 0..k {
              acc = acc * (n - t) / (t + 1);
            }
            acc
          }
          for i in 0..n {
            let (pi, ki, ci) = (powers[i], k[i], &center[i]);
            let binom = binom_i128(pi, ki);
            if binom != 1 {
              factors.push(int_e(binom));
            }
            factors.push(pow(ci, pi - ki));
            gamma_num.push(Expr::FunctionCall {
              name: "Gamma".to_string(),
              args: vec![ratio(int_e(ki + 1), int_e(2))].into(),
            });
            half_sum_num += ki + 1;
          }
          let k_sum: i128 = k.iter().sum();
          factors.push(pow(&radius, k_sum + n as i128));
          let gamma_den = Expr::FunctionCall {
            name: "Gamma".to_string(),
            args: vec![plus(vec![
              ratio(int_e(half_sum_num), int_e(2)),
              int_e(1),
            ])]
            .into(),
          };
          factors.push(ratio(times(gamma_num), gamma_den));
          terms.push(times(factors));
        }
        // Odometer over 0..=powers.
        for i in 0..n {
          if k[i] < powers[i] {
            k[i] += 1;
            continue 'outer;
          }
          k[i] = 0;
        }
        break;
      }
      ev(&plus(terms))
    }
    // Affine map to the unit simplex: x = v0 + u d1 + v d2 with
    // Integrate[u^a v^b] = a! b!/(a + b + 2)!, times |det|.
    MomentShape::Tri { v } => {
      let (det, sum) = triangle_moment_parts(&v, powers[0], powers[1])?;
      let abs_det = ev(&Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![det].into(),
      })?;
      ev(&times(vec![abs_det, sum]))
    }
  }
}

/// The moment of the triangle (v0, v1, v2) split as (det, sum): the true
/// moment is Abs[det]·sum and the orientation-SIGNED moment (used for
/// polygon fan decomposition) is det·sum. Uses the affine map to the unit
/// simplex with Integrate[u^a v^b] = a! b!/(a + b + 2)! and exact
/// multinomial expansion.
fn triangle_moment_parts(
  v: &[[Expr; 2]],
  i_pow: i128,
  j_pow: i128,
) -> Result<(Expr, Expr), InterpreterError> {
  let ev = |e: &Expr| crate::evaluator::evaluate_expr_to_expr(e);
  let int_e = Expr::Integer;
  let times = |fs: Vec<Expr>| {
    let fs: Vec<Expr> = fs
      .into_iter()
      .filter(|f| !matches!(f, Expr::Integer(1)))
      .collect();
    match fs.len() {
      0 => Expr::Integer(1),
      1 => fs.into_iter().next().unwrap(),
      _ => Expr::FunctionCall {
        name: "Times".to_string(),
        args: fs.into(),
      },
    }
  };
  let plus = |ts: Vec<Expr>| match ts.len() {
    0 => Expr::Integer(0),
    1 => ts.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Plus".to_string(),
      args: ts.into(),
    },
  };
  let pow = |b: &Expr, e: i128| -> Expr {
    match e {
      0 => Expr::Integer(1),
      1 => b.clone(),
      _ => Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![b.clone(), Expr::Integer(e)].into(),
      },
    }
  };
  let ratio = |num: Expr, den: Expr| binop(BinaryOperator::Divide, num, den);
  let is_zero = |e: &Expr| matches!(e, Expr::Integer(0));
  let sub = |a: &Expr, b: &Expr| -> Result<Expr, InterpreterError> {
    ev(&plus(vec![a.clone(), times(vec![int_e(-1), b.clone()])]))
  };
  let d1 = [sub(&v[1][0], &v[0][0])?, sub(&v[1][1], &v[0][1])?];
  let d2 = [sub(&v[2][0], &v[0][0])?, sub(&v[2][1], &v[0][1])?];
  let det = ev(&plus(vec![
    times(vec![d1[0].clone(), d2[1].clone()]),
    times(vec![int_e(-1), d1[1].clone(), d2[0].clone()]),
  ]))?;
  let factorial = |m: i128| -> i128 { (1..=m).product::<i128>().max(1) };
  let multinom = |m: i128, a: i128, b: i128| -> i128 {
    factorial(m) / (factorial(a) * factorial(b) * factorial(m - a - b))
  };
  let mut terms: Vec<Expr> = Vec::new();
  for k2 in 0..=i_pow {
    for k3 in 0..=(i_pow - k2) {
      let k1 = i_pow - k2 - k3;
      if k1 > 0 && is_zero(&v[0][0]) {
        continue;
      }
      for l2 in 0..=j_pow {
        for l3 in 0..=(j_pow - l2) {
          let l1 = j_pow - l2 - l3;
          if l1 > 0 && is_zero(&v[0][1]) {
            continue;
          }
          let a = k2 + l2;
          let b = k3 + l3;
          // multinom_i * multinom_j * a! b!/(a+b+2)! as one exact
          // rational.
          let num = multinom(i_pow, k2, k3)
            * multinom(j_pow, l2, l3)
            * factorial(a)
            * factorial(b);
          let den = factorial(a + b + 2);
          let g = {
            let mut x = num;
            let mut y = den;
            while y != 0 {
              let t = x % y;
              x = y;
              y = t;
            }
            x.max(1)
          };
          terms.push(times(vec![
            ratio(int_e(num / g), int_e(den / g)),
            pow(&v[0][0], k1),
            pow(&d1[0], k2),
            pow(&d2[0], k3),
            pow(&v[0][1], l1),
            pow(&d1[1], l2),
            pow(&d2[1], l3),
          ]));
        }
      }
    }
  }
  Ok((det, plus(terms)))
}

/// The exact raw moment of a simple polygon about the origin, via the
/// orientation-signed fan decomposition from the first vertex. Requires
/// numerically evaluable vertices (the overall orientation sign must be
/// decidable); returns None otherwise. Used by MomentOfInertia — NOT by
/// RegionMoment, where wolframscript returns machine-precision quadrature
/// values instead of exact ones.
fn polygon_raw_moment(
  verts: &[[Expr; 2]],
  i_pow: i128,
  j_pow: i128,
) -> Result<Option<Expr>, InterpreterError> {
  let ev = |e: &Expr| crate::evaluator::evaluate_expr_to_expr(e);
  let mut signed_terms: Vec<Expr> = Vec::new();
  let mut signed_area_2 = 0.0f64;
  for t in 1..verts.len() - 1 {
    let tri = [verts[0].clone(), verts[t].clone(), verts[t + 1].clone()];
    let (det, sum) = triangle_moment_parts(&tri, i_pow, j_pow)?;
    let Some(det_f) = crate::functions::math_ast::try_eval_to_f64(&det) else {
      return Ok(None);
    };
    signed_area_2 += det_f;
    signed_terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![det, sum].into(),
    });
  }
  if signed_area_2 == 0.0 {
    return Ok(None);
  }
  let mut factors: Vec<Expr> = Vec::new();
  if signed_area_2 < 0.0 {
    factors.push(Expr::Integer(-1));
  }
  factors.push(Expr::FunctionCall {
    name: "Plus".to_string(),
    args: signed_terms.into(),
  });
  Ok(Some(ev(&Expr::FunctionCall {
    name: "Times".to_string(),
    args: factors.into(),
  })?))
}

/// The embedding dimension of a region supported by MomentOfInertia, and
/// the same region translated by -pt (so moments about pt become raw
/// moments about the origin of the translated region).
fn moment_region_dim(region: &Expr) -> Option<usize> {
  let Expr::FunctionCall { name, args } = region else {
    return None;
  };
  match name.as_str() {
    "Disk" | "Rectangle" | "Triangle" => Some(2),
    "Polygon"
      if args.len() == 1
        && matches!(&args[0], Expr::List(pts) if pts.len() >= 3
          && pts.iter().all(
            |p| matches!(p, Expr::List(c) if c.len() == 2))) =>
    {
      Some(2)
    }
    "Ball" | "Cuboid" => Some(3),
    _ => None,
  }
  .map(|default| match name.as_str() {
    "Disk" | "Ball" => match args.first() {
      Some(Expr::List(c)) => c.len(),
      _ => default,
    },
    "Rectangle" | "Cuboid" => match args.first() {
      Some(Expr::List(c)) => c.len(),
      _ => default,
    },
    _ => default,
  })
}

fn translate_region(region: &Expr, pt: &[Expr]) -> Option<Expr> {
  let shift = |coord: &Expr, delta: &Expr| -> Option<Expr> {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        coord.clone(),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), delta.clone()].into(),
        },
      ]
      .into(),
    })
    .ok()
  };
  let shift_point = |p: &[Expr]| -> Option<Expr> {
    let coords: Option<Vec<Expr>> =
      p.iter().zip(pt.iter()).map(|(c, d)| shift(c, d)).collect();
    Some(Expr::List(coords?.into()))
  };
  let Expr::FunctionCall { name, args } = region else {
    return None;
  };
  let n = pt.len();
  match name.as_str() {
    "Disk" | "Ball" => {
      let center = match args.first() {
        None => vec![Expr::Integer(0); n],
        Some(Expr::List(c)) if c.len() == n => c.iter().cloned().collect(),
        _ => return None,
      };
      let radius = match args.get(1) {
        None => Expr::Integer(1),
        Some(Expr::List(_)) => return None,
        Some(r) => r.clone(),
      };
      if args.len() > 2 {
        return None;
      }
      Some(Expr::FunctionCall {
        name: name.clone(),
        args: vec![shift_point(&center)?, radius].into(),
      })
    }
    "Rectangle" | "Cuboid" => {
      let (lo, hi): (Vec<Expr>, Vec<Expr>) = match (args.first(), args.get(1)) {
        (None, _) if args.is_empty() => {
          (vec![Expr::Integer(0); n], vec![Expr::Integer(1); n])
        }
        (Some(Expr::List(p)), None) if p.len() == n => {
          let lo: Vec<Expr> = p.iter().cloned().collect();
          let hi: Option<Vec<Expr>> = lo
            .iter()
            .map(|x| {
              crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![x.clone(), Expr::Integer(1)].into(),
              })
              .ok()
            })
            .collect();
          (lo, hi?)
        }
        (Some(Expr::List(p1)), Some(Expr::List(p2)))
          if p1.len() == n && p2.len() == n =>
        {
          (p1.iter().cloned().collect(), p2.iter().cloned().collect())
        }
        _ => return None,
      };
      Some(Expr::FunctionCall {
        name: name.clone(),
        args: vec![shift_point(&lo)?, shift_point(&hi)?].into(),
      })
    }
    "Triangle" | "Polygon" => {
      let verts: Vec<Vec<Expr>> = if name == "Triangle" && args.is_empty() {
        vec![
          vec![Expr::Integer(0), Expr::Integer(0)],
          vec![Expr::Integer(1), Expr::Integer(0)],
          vec![Expr::Integer(0), Expr::Integer(1)],
        ]
      } else if args.len() == 1
        && let Expr::List(pts) = &args[0]
        && (if name == "Triangle" {
          pts.len() == 3
        } else {
          pts.len() >= 3
        })
        && pts
          .iter()
          .all(|p| matches!(p, Expr::List(c) if c.len() == n))
      {
        pts
          .iter()
          .map(|p| {
            let Expr::List(c) = p else { unreachable!() };
            c.iter().cloned().collect()
          })
          .collect()
      } else {
        return None;
      };
      let shifted: Option<Vec<Expr>> =
        verts.iter().map(|v| shift_point(v)).collect();
      Some(Expr::FunctionCall {
        name: name.clone(),
        args: vec![Expr::List(shifted?.into())].into(),
      })
    }
    _ => None,
  }
}

/// MomentOfInertia[reg] (about the centroid), MomentOfInertia[reg, pt],
/// and MomentOfInertia[reg, pt, v]: the inertia matrix
/// I_ab = Integrate[delta_ab |x - pt|^2 - (x - pt)_a (x - pt)_b, reg],
/// or the scalar v.I.v / v.v about the axis through pt in direction v.
fn compute_moment_of_inertia(
  region: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("MomentOfInertia", args));
  let Some(dim) = moment_region_dim(region) else {
    return unevaluated();
  };
  let pt: Vec<Expr> = if args.len() >= 2 {
    match &args[1] {
      Expr::List(p) if p.len() == dim => p.iter().cloned().collect(),
      _ => return unevaluated(),
    }
  } else {
    match compute_region_centroid(region)? {
      Expr::List(ref c) if c.len() == dim => c.iter().cloned().collect(),
      _ => return unevaluated(),
    }
  };
  let Some(translated) = translate_region(region, &pt) else {
    return unevaluated();
  };
  // Raw moment of the translated region; None when it could not be
  // evaluated. Polygons use the exact signed fan decomposition (which
  // RegionMoment deliberately does NOT expose — wolframscript's
  // RegionMoment returns machine-precision quadrature values for
  // polygons, but its MomentOfInertia is exact).
  let polygon_verts: Option<Vec<[Expr; 2]>> = match &translated {
    Expr::FunctionCall { name, args }
      if name == "Polygon" && args.len() == 1 =>
    {
      match &args[0] {
        Expr::List(pts) => Some(
          pts
            .iter()
            .map(|p| {
              let Expr::List(c) = p else { unreachable!() };
              [c[0].clone(), c[1].clone()]
            })
            .collect(),
        ),
        _ => None,
      }
    }
    _ => None,
  };
  let raw = |spec: Vec<i128>| -> Result<Option<Expr>, InterpreterError> {
    if let Some(verts) = &polygon_verts {
      return polygon_raw_moment(verts, spec[0], spec[1]);
    }
    let spec_expr = Expr::List(
      spec
        .into_iter()
        .map(Expr::Integer)
        .collect::<Vec<_>>()
        .into(),
    );
    let m = compute_region_moment(&translated, &spec_expr)?;
    if matches!(&m, Expr::FunctionCall { name, .. } if name == "RegionMoment") {
      Ok(None)
    } else {
      Ok(Some(m))
    }
  };
  // Second moments per axis and product moments.
  let mut m2: Vec<Expr> = Vec::with_capacity(dim);
  for k in 0..dim {
    let mut spec = vec![0i128; dim];
    spec[k] = 2;
    match raw(spec)? {
      Some(m) => m2.push(m),
      None => return unevaluated(),
    }
  }
  let mut entries: Vec<Vec<Expr>> = vec![vec![Expr::Integer(0); dim]; dim];
  for a in 0..dim {
    for b in 0..dim {
      let entry = if a == b {
        let others: Vec<Expr> = (0..dim)
          .filter(|k| *k != a)
          .map(|k| m2[k].clone())
          .collect();
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: others.into(),
        }
      } else {
        let mut spec = vec![0i128; dim];
        spec[a] = 1;
        spec[b] = 1;
        match raw(spec)? {
          Some(m) => Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), m].into(),
          },
          None => return unevaluated(),
        }
      };
      entries[a][b] = crate::evaluator::evaluate_expr_to_expr(&entry)?;
    }
  }
  if args.len() < 3 {
    return Ok(Expr::List(
      entries
        .into_iter()
        .map(|row| Expr::List(row.into()))
        .collect::<Vec<_>>()
        .into(),
    ));
  }
  // Axis form: v.I.v / v.v.
  let Expr::List(v) = &args[2] else {
    return unevaluated();
  };
  if v.len() != dim {
    return unevaluated();
  }
  let mut num_terms: Vec<Expr> = Vec::new();
  for a in 0..dim {
    for b in 0..dim {
      num_terms.push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![v[a].clone(), v[b].clone(), entries[a][b].clone()].into(),
      });
    }
  }
  let den_terms: Vec<Expr> = v
    .iter()
    .map(|c| Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![c.clone(), Expr::Integer(2)].into(),
    })
    .collect();
  crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: num_terms.into(),
    }),
    right: Box::new(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: den_terms.into(),
    }),
  })
}

/// Parses HalfSpace[n, c] / HalfSpace[n, p] into (normal, bound) with the
/// point form converted to the scalar bound c = n.p, so membership is
/// n.x <= c in both cases.
fn half_space_parts(args: &[Expr]) -> Option<(Vec<Expr>, Expr)> {
  match args {
    [Expr::List(n), second] if !n.is_empty() => {
      let normal: Vec<Expr> = n.iter().cloned().collect();
      let bound = match second {
        Expr::List(p) if p.len() == normal.len() => {
          half_space_dot(&normal, &p.iter().cloned().collect::<Vec<_>>())
            .ok()?
        }
        Expr::List(_) => return None,
        c => c.clone(),
      };
      Some((normal, bound))
    }
    _ => None,
  }
}

/// The exact dot product n.x of a half-space normal with a point.
fn half_space_dot(
  normal: &[Expr],
  point: &[Expr],
) -> Result<Expr, InterpreterError> {
  let terms: Vec<Expr> = normal
    .iter()
    .zip(point.iter())
    .map(|(a, b)| Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![a.clone(), b.clone()].into(),
    })
    .collect();
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.into(),
  })
}

/// Parses DiskSegment[{x, y}, r | {rx, ry}, {θ1, θ2}] into
/// (center, rx, ry, θ1, θ2, Δθ as f64). rx == ry for the circular case.
/// Angles must be concrete; None otherwise.
fn disk_segment_parts(
  args: &[Expr],
) -> Option<(Vec<Expr>, Expr, Expr, Expr, Expr, f64)> {
  match args {
    [Expr::List(c), r, Expr::List(th)] if c.len() == 2 && th.len() == 2 => {
      let (rx, ry) = match r {
        Expr::List(rr) if rr.len() == 2 => (rr[0].clone(), rr[1].clone()),
        Expr::List(_) => return None,
        _ => (r.clone(), r.clone()),
      };
      let t1 = crate::functions::math_ast::try_eval_to_f64(&th[0])?;
      let t2 = crate::functions::math_ast::try_eval_to_f64(&th[1])?;
      Some((
        c.iter().cloned().collect(),
        rx,
        ry,
        th[0].clone(),
        th[1].clone(),
        t2 - t1,
      ))
    }
    _ => None,
  }
}

/// The exact angular width θ2 - θ1 of a disk segment.
fn disk_segment_dtheta(
  th1: &Expr,
  th2: &Expr,
) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      th2.clone(),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), th1.clone()].into(),
      },
    ]
    .into(),
  })
}

/// Parses a normalized SphericalShell[{x, y, z}, {r1, r2}] into its radii
/// (sorted ascending when both are concrete, since wolframscript treats
/// the pair as unordered numerically).
fn spherical_shell_radii(args: &[Expr]) -> Option<(Expr, Expr)> {
  match args {
    [Expr::List(c), Expr::List(rr)] if c.len() == 3 && rr.len() == 2 => {
      let (r1, r2) = (rr[0].clone(), rr[1].clone());
      if let (Some(a), Some(b)) = (
        crate::functions::math_ast::expr_to_num(&r1),
        crate::functions::math_ast::expr_to_num(&r2),
      ) && a > b
      {
        return Some((r2, r1));
      }
      Some((r1, r2))
    }
    _ => None,
  }
}

/// Parses a normalized CapsuleShape[{p1, p2}, r] with 3-D endpoints into
/// (squared axis length, radius).
/// The two 2-D endpoints and the radius of StadiumShape[{{...},{...}}, r].
fn stadium_parts(args: &[Expr]) -> Option<(Vec<Expr>, Vec<Expr>, Expr)> {
  match args {
    [Expr::List(points), r] if points.len() == 2 => {
      let (Expr::List(p1), Expr::List(p2)) = (&points[0], &points[1]) else {
        return None;
      };
      if p1.len() != 2 || p2.len() != 2 {
        return None;
      }
      Some((p1.to_vec(), p2.to_vec(), r.clone()))
    }
    _ => None,
  }
}

/// Segment length Sqrt[(x1-x2)^2 + (y1-y2)^2] as an expression.
fn stadium_length(p1: &[Expr], p2: &[Expr]) -> Expr {
  let squares: Vec<Expr> = p1
    .iter()
    .zip(p2.iter())
    .map(|(a, b)| Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            a.clone(),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), b.clone()].into(),
            },
          ]
          .into(),
        },
        Expr::Integer(2),
      ]
      .into(),
    })
    .collect();
  Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![Expr::FunctionCall {
      name: "Plus".to_string(),
      args: squares.into(),
    }]
    .into(),
  }
}

/// Stadium area 2 L r + Pi r^2. `hoist` factors the numeric content out
/// like wolframscript's Area (16 + 4 Pi shows as 4 (4 + Pi)); Area uses
/// it, RegionMeasure keeps the plain sum.
fn stadium_area(
  p1: &[Expr],
  p2: &[Expr],
  r: &Expr,
  hoist: bool,
) -> Result<Expr, InterpreterError> {
  let ev = crate::evaluator::evaluate_expr_to_expr;
  let length = stadium_length(p1, p2);
  let rect = ev(&Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(2), length, r.clone()].into(),
  })?;
  let cap_coeff = ev(&Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![r.clone(), Expr::Integer(2)].into(),
  })?;
  // wolframscript factors out the whole Pi coefficient — but only when
  // it divides the rectangle term (16 + 4 Pi → 4 (4 + Pi); 6 + 9 Pi
  // stays as is).
  if hoist
    && let (Expr::Integer(a), Expr::Integer(b)) = (&rect, &cap_coeff)
    && *a > 0
    && *b > 1
    && *a % *b == 0
  {
    return Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Integer(*b),
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![Expr::Integer(*a / *b), Expr::Constant("Pi".to_string())]
            .into(),
        },
      ]
      .into(),
    });
  }
  ev(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      rect,
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Constant("Pi".to_string()), cap_coeff].into(),
      },
    ]
    .into(),
  })
}

fn capsule_height_sq_radius(args: &[Expr]) -> Option<(Expr, Expr)> {
  match args {
    [Expr::List(points), r] if points.len() == 2 => {
      let (Expr::List(p1), Expr::List(p2)) = (&points[0], &points[1]) else {
        return None;
      };
      if p1.len() != 3 || p2.len() != 3 {
        return None;
      }
      let squares: Vec<Expr> = p1
        .iter()
        .zip(p2.iter())
        .map(|(a, b)| Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                a.clone(),
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![Expr::Integer(-1), b.clone()].into(),
                },
              ]
              .into(),
            },
            Expr::Integer(2),
          ]
          .into(),
        })
        .collect();
      Some((
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: squares.into(),
        },
        r.clone(),
      ))
    }
    _ => None,
  }
}

/// The volume of a SphericalShell with the given radii, in wolframscript's
/// displayed form (4*Pi*(-r1^p + r2^p))/den — p = 3, den = 3 for Volume;
/// the p = 2, den = 1 variant is wolframscript's (quirky) RegionMeasure.
fn spherical_shell_measure(
  r1: &Expr,
  r2: &Expr,
  p: i128,
  den: i128,
) -> Result<Expr, InterpreterError> {
  let pow = |b: &Expr, e: i128| Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![b.clone(), Expr::Integer(e)].into(),
  };
  let diff = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), pow(r1, p)].into(),
      },
      pow(r2, p),
    ]
    .into(),
  };
  let product = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(4), Expr::Constant("Pi".to_string()), diff].into(),
  };
  let measure = if den == 1 {
    product
  } else {
    binop(BinaryOperator::Divide, product, Expr::Integer(den))
  };
  crate::evaluator::evaluate_expr_to_expr(&measure)
}

/// The volume of a CapsuleShape: Pi r^2 h + (4 Pi r^3)/3.
fn capsule_volume(
  height_sq: &Expr,
  radius: &Expr,
) -> Result<Expr, InterpreterError> {
  let pow = |b: &Expr, e: i128| Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![b.clone(), Expr::Integer(e)].into(),
  };
  let height = Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![height_sq.clone()].into(),
  };
  let cylinder = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Constant("Pi".to_string()), pow(radius, 2), height].into(),
  };
  let sphere = Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Integer(4),
        Expr::Constant("Pi".to_string()),
        pow(radius, 3),
      ]
      .into(),
    }),
    right: Box::new(Expr::Integer(3)),
  };
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![cylinder, sphere].into(),
  })
}

/// Parses the arguments of a Platonic-solid primitive (Cube, Octahedron,
/// Dodecahedron, Icosahedron, Tetrahedron) into (center, edge length):
///   Head[]                     → origin, 1
///   Head[l]                    → origin, l
///   Head[{θ, ϕ}]               → origin, 1  (rotation — metrics invariant)
///   Head[{x, y, z}]            → center, 1
///   Head[{θ, ϕ} | {x, y, z}, l] → center-or-origin, l
/// Returns None for any other shape — including Tetrahedron's explicit
/// four-vertex form (handled separately by the callers), three or more
/// arguments (wolframscript leaves the rotated-and-centered form
/// unevaluated), and a concrete non-positive edge (Volume[Dodecahedron[-2]]
/// stays unevaluated in wolframscript).
fn platonic_center_edge(args: &[Expr]) -> Option<(Vec<Expr>, Expr)> {
  let origin = || vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)];
  let is_scalar = |e: &Expr| {
    !matches!(e, Expr::List(_) | Expr::String(_) | Expr::Rule { .. })
  };
  // Some(Some(c)) → explicit center; Some(None) → rotation spec (origin).
  let center_of = |e: &Expr| -> Option<Option<Vec<Expr>>> {
    match e {
      Expr::List(items) if items.len() == 2 && items.iter().all(is_scalar) => {
        Some(None)
      }
      Expr::List(items) if items.len() == 3 && items.iter().all(is_scalar) => {
        Some(Some(items.iter().cloned().collect()))
      }
      _ => None,
    }
  };
  let (center, edge) = match args {
    [] => (origin(), Expr::Integer(1)),
    [e] if is_scalar(e) => (origin(), e.clone()),
    [c] => match center_of(c)? {
      Some(cv) => (cv, Expr::Integer(1)),
      None => (origin(), Expr::Integer(1)),
    },
    [c, l] if is_scalar(l) => match center_of(c)? {
      Some(cv) => (cv, l.clone()),
      None => (origin(), l.clone()),
    },
    _ => return None,
  };
  if let Some(v) = crate::functions::math_ast::expr_to_num(&edge)
    && v <= 0.0
  {
    return None;
  }
  Some((center, edge))
}

/// A unit-edge Platonic metric (volume or surface area, given as WL source)
/// scaled by edge^power.
fn platonic_scaled_metric(
  unit_src: &str,
  edge: &Expr,
  power: i128,
) -> Result<Expr, InterpreterError> {
  let unit = crate::functions::string_ast::parse_program_to_expr(unit_src)?;
  let scaled = if matches!(edge, Expr::Integer(1)) {
    unit
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        unit,
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![edge.clone(), Expr::Integer(power)].into(),
        },
      ]
      .into(),
    }
  };
  crate::evaluator::evaluate_expr_to_expr(&scaled)
}

/// SurfaceArea[reg] — the total boundary area of a 3-D solid. Regions of
/// intrinsic dimension < 3 are Undefined (wolframscript-verified for
/// Sphere, Disk, Triangle, and 2-D Cuboid/Ball).
fn compute_surface_area(expr: &Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "SurfaceArea".to_string(),
      args: vec![expr.clone()].into(),
    })
  };
  let undefined = || Ok(Expr::Identifier("Undefined".to_string()));
  let Expr::FunctionCall { name, args } = expr else {
    return unevaluated();
  };
  match name.as_str() {
    "Cube" | "Hexahedron" | "Octahedron" | "Dodecahedron" | "Icosahedron"
    | "Tetrahedron" => {
      if let Some((_, edge)) = platonic_center_edge(args)
        && let Some(unit) =
          crate::functions::polyhedron_data::unit_surface_area_src(name)
      {
        return platonic_scaled_metric(unit, &edge, 2);
      }
      // Tetrahedron[{p1, p2, p3, p4}] — the sum of the four triangular
      // face areas, assembled as (Σ 2·area_i)/2 so the radicals combine
      // over the common denominator like wolframscript's
      // (5*Sqrt[2] + Sqrt[21] + Sqrt[41] + Sqrt[46])/2.
      if name == "Tetrahedron"
        && args.len() == 1
        && let Expr::List(pts) = &args[0]
        && pts.len() == 4
        && pts
          .iter()
          .all(|p| matches!(p, Expr::List(c) if c.len() == 3))
      {
        let faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        // 2·area of a face = Sqrt[m1^2 + m2^2 + m3^2] (cross-product norm).
        let terms: Vec<Expr> = faces
          .iter()
          .map(|f| {
            let (Expr::List(p1), Expr::List(p2), Expr::List(p3)) =
              (&pts[f[0]], &pts[f[1]], &pts[f[2]])
            else {
              unreachable!("verified 3-coordinate lists above");
            };
            let diff = |a: &Expr, b: &Expr| Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                a.clone(),
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![Expr::Integer(-1), b.clone()].into(),
                },
              ]
              .into(),
            };
            let u: Vec<Expr> = (0..3).map(|d| diff(&p2[d], &p1[d])).collect();
            let v: Vec<Expr> = (0..3).map(|d| diff(&p3[d], &p1[d])).collect();
            let minor = |i: usize, j: usize| Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![u[i].clone(), v[j].clone()].into(),
                },
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![Expr::Integer(-1), u[j].clone(), v[i].clone()]
                    .into(),
                },
              ]
              .into(),
            };
            let sq = |e: Expr| Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![e, Expr::Integer(2)].into(),
            };
            Expr::FunctionCall {
              name: "Sqrt".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![sq(minor(1, 2)), sq(minor(2, 0)), sq(minor(0, 1))]
                  .into(),
              }]
              .into(),
            }
          })
          .collect();
        return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::FunctionCall {
            name: "Plus".to_string(),
            args: terms.into(),
          }),
          right: Box::new(Expr::Integer(2)),
        });
      }
      unevaluated()
    }
    // Ball[c, r] — 4 Pi r^2, defined only for the 3-D ball.
    "Ball" => {
      let (n, radius) = match args.as_slice() {
        [] => (3usize, Expr::Integer(1)),
        [Expr::List(center)] => (center.len(), Expr::Integer(1)),
        [Expr::List(center), r] => (center.len(), r.clone()),
        _ => return unevaluated(),
      };
      if n != 3 {
        return undefined();
      }
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(4),
          Expr::Constant("Pi".to_string()),
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![radius, Expr::Integer(2)].into(),
          },
        ]
        .into(),
      })
    }
    // Cuboid — 2 (|d1 d2| + |d1 d3| + |d2 d3|); only the 3-D box has a
    // surface area.
    "Cuboid" => {
      let diffs: Vec<Expr> = match args.as_slice() {
        [] => vec![Expr::Integer(1); 3],
        [Expr::List(p)] => {
          if p.len() != 3 {
            return undefined();
          }
          vec![Expr::Integer(1); 3]
        }
        [Expr::List(p1), Expr::List(p2)] if p1.len() == p2.len() => {
          if p1.len() != 3 {
            return undefined();
          }
          p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                b.clone(),
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![Expr::Integer(-1), a.clone()].into(),
                },
              ]
              .into(),
            })
            .collect()
        }
        _ => return unevaluated(),
      };
      let pair = |i: usize, j: usize| Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![diffs[i].clone(), diffs[j].clone()].into(),
        }]
        .into(),
      };
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(2),
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![pair(0, 1), pair(0, 2), pair(1, 2)].into(),
          },
        ]
        .into(),
      })
    }
    // Cylinder — 2 Pi r (r + h); Cone — Pi r (r + Sqrt[r^2 + h^2]).
    "Cylinder" | "Cone" => {
      let Some((height_sq, radius)) = cylinder_height_sq_radius(args) else {
        return unevaluated();
      };
      let height = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![height_sq.clone()].into(),
      };
      let r_plus = if name == "Cylinder" {
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![radius.clone(), height].into(),
        }
      } else {
        // slant height Sqrt[r^2 + h^2]
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            radius.clone(),
            Expr::FunctionCall {
              name: "Sqrt".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![radius.clone(), Expr::Integer(2)].into(),
                  },
                  height_sq,
                ]
                .into(),
              }]
              .into(),
            },
          ]
          .into(),
        }
      };
      let mut factors = vec![];
      if name == "Cylinder" {
        factors.push(Expr::Integer(2));
      }
      factors.push(Expr::Constant("Pi".to_string()));
      factors.push(radius);
      factors.push(r_plus);
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Times".to_string(),
        args: factors.into(),
      })
    }
    // SphericalShell — the boundary is BOTH spheres: 4 Pi (r1^2 + r2^2).
    // Only the numeric case: wolframscript hangs on symbolic radii.
    "SphericalShell" => {
      let Some((r1, r2)) = spherical_shell_radii(args) else {
        return unevaluated();
      };
      if crate::functions::math_ast::expr_to_num(&r1).is_none()
        || crate::functions::math_ast::expr_to_num(&r2).is_none()
      {
        return unevaluated();
      }
      let sq = |b: &Expr| Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![b.clone(), Expr::Integer(2)].into(),
      };
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(4),
          Expr::Constant("Pi".to_string()),
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![sq(&r1), sq(&r2)].into(),
          },
        ]
        .into(),
      })
    }
    // CapsuleShape — 2 Pi r h + 4 Pi r^2, numeric parameters only.
    "CapsuleShape" => {
      let Some((h_sq, r)) = capsule_height_sq_radius(args) else {
        return unevaluated();
      };
      if crate::functions::math_ast::try_eval_to_f64(&h_sq).is_none()
        || crate::functions::math_ast::try_eval_to_f64(&r).is_none()
      {
        return unevaluated();
      }
      let side = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(2),
          Expr::Constant("Pi".to_string()),
          r.clone(),
          Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![h_sq].into(),
          },
        ]
        .into(),
      };
      let caps = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(4),
          Expr::Constant("Pi".to_string()),
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![r, Expr::Integer(2)].into(),
          },
        ]
        .into(),
      };
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![side, caps].into(),
      })
    }
    // Regions of intrinsic dimension < 3 have no surface area.
    "Sphere" | "Disk" | "Rectangle" | "Triangle" | "Polygon"
    | "RegularPolygon" | "Circle" | "Line" | "Point" | "Annulus"
    | "Parallelogram" => undefined(),
    _ => unevaluated(),
  }
}

/// The squared axis length and radius of a Cylinder/Cone spec:
/// Head[] → endpoints {0,0,±1}, r = 1; Head[{p1, p2}] → r = 1;
/// Head[{p1, p2}, r].
fn cylinder_height_sq_radius(args: &[Expr]) -> Option<(Expr, Expr)> {
  let (p1, p2, radius): (Vec<Expr>, Vec<Expr>, Expr) = match args {
    [] => (
      vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(-1)],
      vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(1)],
      Expr::Integer(1),
    ),
    [Expr::List(points)] | [Expr::List(points), _] if points.len() == 2 => {
      let (Expr::List(p1), Expr::List(p2)) = (&points[0], &points[1]) else {
        return None;
      };
      if p1.len() != p2.len() || p1.is_empty() {
        return None;
      }
      let radius = if args.len() == 2 {
        args[1].clone()
      } else {
        Expr::Integer(1)
      };
      (
        p1.iter().cloned().collect(),
        p2.iter().cloned().collect(),
        radius,
      )
    }
    _ => return None,
  };
  let squares: Vec<Expr> = p1
    .iter()
    .zip(p2.iter())
    .map(|(a, b)| Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            a.clone(),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), b.clone()].into(),
            },
          ]
          .into(),
        },
        Expr::Integer(2),
      ]
      .into(),
    })
    .collect();
  let sum_sq = if squares.len() == 1 {
    squares.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: squares.into(),
    }
  };
  Some((sum_sq, radius))
}

fn compute_volume(expr: &Expr) -> Result<Expr, InterpreterError> {
  // Cylinder[]/Cylinder[{p1, p2}]/Cylinder[{p1, p2}, r]:
  //   Volume = Pi * r^2 * Sqrt[Sum_i (p2_i - p1_i)^2].
  // Cone[]/Cone[{p1, p2}]/Cone[{p1, p2}, r]:
  //   Volume = (Pi * r^2 * Sqrt[Sum_i (p2_i - p1_i)^2]) / 3.
  // Default endpoints are {0, 0, -1} and {0, 0, 1}, default radius is 1.
  if let Expr::FunctionCall { name, args } = expr
    && matches!(name.as_str(), "Cylinder" | "Cone")
  {
    let (p1_vec, p2_vec, radius) = match args.len() {
      0 => (
        vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(-1)],
        vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(1)],
        Expr::Integer(1),
      ),
      1 | 2 => {
        let Expr::List(points) = &args[0] else {
          return Ok(Expr::FunctionCall {
            name: "Volume".to_string(),
            args: vec![expr.clone()].into(),
          });
        };
        if points.len() != 2 {
          return Ok(Expr::FunctionCall {
            name: "Volume".to_string(),
            args: vec![expr.clone()].into(),
          });
        }
        let (Expr::List(p1), Expr::List(p2)) = (&points[0], &points[1]) else {
          return Ok(Expr::FunctionCall {
            name: "Volume".to_string(),
            args: vec![expr.clone()].into(),
          });
        };
        if p1.len() != p2.len() || p1.is_empty() {
          return Ok(Expr::FunctionCall {
            name: "Volume".to_string(),
            args: vec![expr.clone()].into(),
          });
        }
        let radius = if args.len() == 2 {
          args[1].clone()
        } else {
          Expr::Integer(1)
        };
        (
          p1.iter().cloned().collect::<Vec<_>>(),
          p2.iter().cloned().collect::<Vec<_>>(),
          radius,
        )
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Volume".to_string(),
          args: vec![expr.clone()].into(),
        });
      }
    };
    // Build Sum[(p1_i - p2_i)^2]
    let squares: Vec<Expr> = p1_vec
      .iter()
      .zip(p2_vec.iter())
      .map(|(a, b)| Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              a.clone(),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), b.clone()].into(),
              },
            ]
            .into(),
          },
          Expr::Integer(2),
        ]
        .into(),
      })
      .collect();
    let sum_sq = if squares.len() == 1 {
      squares.into_iter().next().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: squares.into(),
      }
    };
    let length = Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![sum_sq].into(),
    };
    let r_squared = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![radius, Expr::Integer(2)].into(),
    };
    let mut volume = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Constant("Pi".to_string()), r_squared, length].into(),
    };
    if name == "Cone" {
      volume = binop(BinaryOperator::Divide, volume, Expr::Integer(3));
    }
    return crate::evaluator::evaluate_expr_to_expr(&volume);
  }

  // Cuboid[] / Cuboid[p] are unit hypercubes (Volume = 1 when 3D).
  // Cuboid[p1, p2] has Volume = Abs[(p2_1 - p1_1) * ... * (p2_n - p1_n)]
  // *only when n == 3*; for any other dimensionality Volume is Undefined
  // (use Area for 2D, RegionMeasure for the n-dim measure).
  if let Expr::FunctionCall { name, args } = expr
    && name == "Cuboid"
  {
    return match args.len() {
      0 => Ok(Expr::Integer(1)),
      1 => {
        // Cuboid[{x,y,z}] is a unit hypercube at that corner; Volume only
        // defined when the point is 3D.
        let Expr::List(p) = &args[0] else {
          return Ok(Expr::FunctionCall {
            name: "Volume".to_string(),
            args: vec![expr.clone()].into(),
          });
        };
        if p.len() == 3 {
          Ok(Expr::Integer(1))
        } else {
          Ok(Expr::Identifier("Undefined".to_string()))
        }
      }
      2 => {
        let (Expr::List(p1), Expr::List(p2)) = (&args[0], &args[1]) else {
          return Ok(Expr::FunctionCall {
            name: "Volume".to_string(),
            args: vec![expr.clone()].into(),
          });
        };
        if p1.len() != p2.len() || p1.is_empty() {
          return Ok(Expr::FunctionCall {
            name: "Volume".to_string(),
            args: vec![expr.clone()].into(),
          });
        }
        if p1.len() != 3 {
          return Ok(Expr::Identifier("Undefined".to_string()));
        }
        // Build (p2_i - p1_i) for each dimension and take Abs of the product.
        let diffs: Vec<Expr> = p1
          .iter()
          .zip(p2.iter())
          .map(|(a, b)| Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              b.clone(),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), a.clone()].into(),
              },
            ]
            .into(),
          })
          .collect();
        let product = if diffs.len() == 1 {
          diffs.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: diffs.into(),
          }
        };
        let abs_expr = Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![product].into(),
        };
        crate::evaluator::evaluate_expr_to_expr(&abs_expr)
      }
      _ => Ok(Expr::FunctionCall {
        name: "Volume".to_string(),
        args: vec![expr.clone()].into(),
      }),
    };
  }

  // Volume is the 3-dimensional measure, so it is only defined for solids of
  // intrinsic dimension 3. Lower-dimensional regions (and surfaces) return
  // Undefined; 3-D balls and ellipsoids get their closed-form volume.
  let undefined = || Ok(Expr::Identifier("Undefined".to_string()));
  if let Expr::FunctionCall { name, args } = expr {
    match name.as_str() {
      // Ball[c, r] — the solid n-ball. Volume is defined only in 3-D, where
      // it equals (4/3) Pi r^3; any other dimension is Undefined.
      "Ball" => {
        let (n, radius) = match args.len() {
          0 => (3usize, Expr::Integer(1)),
          1 | 2 => {
            let Expr::List(center) = &args[0] else {
              return Ok(Expr::FunctionCall {
                name: "Volume".to_string(),
                args: vec![expr.clone()].into(),
              });
            };
            if center.is_empty() {
              return undefined();
            }
            let radius = if args.len() == 2 {
              args[1].clone()
            } else {
              Expr::Integer(1)
            };
            (center.len(), radius)
          }
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Volume".to_string(),
              args: vec![expr.clone()].into(),
            });
          }
        };
        if n != 3 {
          return undefined();
        }
        // (4 Pi r^3) / 3
        let vol = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(4),
              Expr::Constant("Pi".to_string()),
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![radius, Expr::Integer(3)].into(),
              },
            ]
            .into(),
          }),
          right: Box::new(Expr::Integer(3)),
        };
        return crate::evaluator::evaluate_expr_to_expr(&vol);
      }
      // Ellipsoid[c, {r1, r2, r3}] — solid ellipsoid, 3-D volume
      // (4/3) Pi r1 r2 r3. Any other dimension is Undefined.
      "Ellipsoid" if args.len() == 2 => {
        let (Expr::List(center), Expr::List(radii)) = (&args[0], &args[1])
        else {
          return Ok(Expr::FunctionCall {
            name: "Volume".to_string(),
            args: vec![expr.clone()].into(),
          });
        };
        if center.len() != radii.len() {
          return Ok(Expr::FunctionCall {
            name: "Volume".to_string(),
            args: vec![expr.clone()].into(),
          });
        }
        if radii.len() != 3 {
          return undefined();
        }
        let vol = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(4),
              Expr::Constant("Pi".to_string()),
              radii[0].clone(),
              radii[1].clone(),
              radii[2].clone(),
            ]
            .into(),
          }),
          right: Box::new(Expr::Integer(3)),
        };
        return crate::evaluator::evaluate_expr_to_expr(&vol);
      }
      // A 3-D half-space has infinite volume; other dimensions have no
      // 3-volume.
      "HalfSpace" if half_space_parts(args).is_some() => {
        let (normal, _) = half_space_parts(args).unwrap();
        return Ok(Expr::Identifier(
          if normal.len() == 3 {
            "Infinity"
          } else {
            "Undefined"
          }
          .to_string(),
        ));
      }
      // SphericalShell[c, {r1, r2}] — (4 Pi (r2^3 - r1^3))/3.
      "SphericalShell" if spherical_shell_radii(args).is_some() => {
        let (r1, r2) = spherical_shell_radii(args).unwrap();
        return spherical_shell_measure(&r1, &r2, 3, 3);
      }
      // CapsuleShape[{p1, p2}, r] — Pi r^2 h + (4 Pi r^3)/3.
      "CapsuleShape" if capsule_height_sq_radius(args).is_some() => {
        let (h_sq, r) = capsule_height_sq_radius(args).unwrap();
        return capsule_volume(&h_sq, &r);
      }
      // Platonic-solid primitives: Volume = (unit-edge volume) * l^3.
      "Cube" | "Hexahedron" | "Octahedron" | "Dodecahedron" | "Icosahedron"
      | "Tetrahedron"
        if platonic_center_edge(args).is_some()
          && crate::functions::polyhedron_data::unit_volume_src(name)
            .is_some() =>
      {
        let (_, edge) = platonic_center_edge(args).unwrap();
        let unit =
          crate::functions::polyhedron_data::unit_volume_src(name).unwrap();
        return platonic_scaled_metric(unit, &edge, 3);
      }
      // Tetrahedron[{p1, p2, p3, p4}] / Simplex of 4 points in 3-space:
      //   Volume = |Det[{p2-p1, p3-p1, p4-p1}]| / 6.
      "Tetrahedron" | "Simplex" if args.len() == 1 => {
        // Simplex[n]: Volume (a 3-measure) is defined only for the 3-simplex,
        // where it is 1/3!; every other dimension is Undefined.
        if name == "Simplex"
          && let Expr::Integer(n) = &args[0]
          && *n >= 0
        {
          return if *n == 3 {
            Ok(crate::functions::math_ast::make_rational(
              1,
              factorial_small(3),
            ))
          } else {
            Ok(Expr::Identifier("Undefined".to_string()))
          };
        }
        if let Expr::List(pts) = &args[0]
          && pts.len() == 4
          && let Some(edges) = simplex_edges(pts)
        {
          return det_measure(edges, factorial_small(3));
        }
      }
      // Parallelepiped[p, {v1, v2, v3}] — Volume = |Det[{v1, v2, v3}]|.
      "Parallelepiped" if args.len() == 2 => {
        if let Expr::List(vs) = &args[1] {
          if vs.len() == 3
            && vs
              .iter()
              .all(|v| matches!(v, Expr::List(c) if c.len() == 3))
          {
            let rows = vs.iter().cloned().collect();
            return det_measure(rows, 1);
          }
          // A parallelepiped spanned by fewer than three vectors is not a
          // 3-D solid, so its 3-volume is Undefined.
          if vs.len() < 3 {
            return undefined();
          }
        }
      }
      // Prism[{p1, …, p6}] — translational triangular prism.
      "Prism" => {
        if let Some(pts) = prism_pts(args)
          && let Some(result) = prism_volume(&pts)
        {
          return result;
        }
      }
      // Pyramid[{p1, …, p4, apex}] — square pyramid.
      "Pyramid" => {
        if let Some(pts) = pyramid_pts(args)
          && let Some(result) = pyramid_volume(&pts)
        {
          return result;
        }
      }
      // A Sphere is a 2-D surface and the following are regions of dimension
      // < 3, so their 3-volume is Undefined.
      "Sphere" | "Disk" | "Rectangle" | "Triangle" | "Polygon"
      | "RegularPolygon" | "Circle" | "Line" | "Point" => {
        return undefined();
      }
      _ => {}
    }
  }
  Ok(Expr::FunctionCall {
    name: "Volume".to_string(),
    args: vec![expr.clone()].into(),
  })
}

fn compute_area(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      "StadiumShape" if stadium_parts(args).is_some() => {
        let (p1, p2, r) = stadium_parts(args).unwrap();
        stadium_area(&p1, &p2, &r, true)
      }
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
                ]
                .into(),
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
                    args: vec![r.clone(), Expr::Integer(2)].into(),
                  },
                ]
                .into(),
              };
              crate::evaluator::evaluate_expr_to_expr(&area)
            }
          }
        } else {
          Ok(Expr::FunctionCall {
            name: "Area".to_string(),
            args: vec![expr.clone()].into(),
          })
        }
      }
      // Annulus[] = region between radii 1/2 and 1; Annulus[c, {r1, r2}] is
      // the ring with inner radius r1 and outer radius r2, area
      // Pi*(r2^2 - r1^2). The sector form Annulus[c, {r1, r2}, {t1, t2}]
      // subtends angle (t2 - t1), so its area is (t2 - t1)/2 * (r2^2 - r1^2).
      "Annulus" => {
        let annulus_unevaluated = || {
          Ok(Expr::FunctionCall {
            name: "Area".to_string(),
            args: vec![expr.clone()].into(),
          })
        };
        let (r1, r2, angles) = match args.len() {
          0 => (
            crate::functions::math_ast::make_rational(1, 2),
            Expr::Integer(1),
            None,
          ),
          2 | 3 => {
            let Expr::List(radii) = &args[1] else {
              return annulus_unevaluated();
            };
            if radii.len() != 2 {
              return annulus_unevaluated();
            }
            let angles = if args.len() == 3 {
              match &args[2] {
                Expr::List(a) if a.len() == 2 => {
                  Some((a[0].clone(), a[1].clone()))
                }
                _ => return annulus_unevaluated(),
              }
            } else {
              None
            };
            (radii[0].clone(), radii[1].clone(), angles)
          }
          _ => return annulus_unevaluated(),
        };
        let sq = |r: Expr| Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![r, Expr::Integer(2)].into(),
        };
        let radial = Expr::FunctionCall {
          name: "Subtract".to_string(),
          args: vec![sq(r2), sq(r1)].into(),
        };
        // Angular factor: Pi for the full ring, (t2 - t1)/2 for a sector.
        let angular = match angles {
          None => Expr::Constant("Pi".to_string()),
          Some((t1, t2)) => Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              crate::functions::math_ast::make_rational(1, 2),
              Expr::FunctionCall {
                name: "Subtract".to_string(),
                args: vec![t2, t1].into(),
              },
            ]
            .into(),
          },
        };
        let area = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![angular, radial].into(),
        };
        crate::evaluator::evaluate_expr_to_expr(&area)
      }
      // Ellipsoid[center, {a, b}] (2D) is a filled ellipse: Area = Pi*a*b.
      // Area is the 2-dimensional measure, so an Ellipsoid of any other
      // dimension (e.g. a 3D solid) has Undefined area, matching WL.
      "Ellipsoid"
        if args.len() == 2
          && matches!(&args[0], Expr::List(c) if c.len() == 2)
          && matches!(&args[1], Expr::List(r) if r.len() == 2) =>
      {
        let Expr::List(radii) = &args[1] else {
          unreachable!()
        };
        let area = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Constant("Pi".to_string()),
            radii[0].clone(),
            radii[1].clone(),
          ]
          .into(),
        };
        crate::evaluator::evaluate_expr_to_expr(&area)
      }
      "Ellipsoid"
        if args.len() == 2
          && matches!((&args[0], &args[1]),
            (Expr::List(c), Expr::List(r)) if c.len() == r.len() && c.len() != 2) =>
      {
        Ok(Expr::Identifier("Undefined".to_string()))
      }
      // Rectangle[] = 1, Rectangle[{x1,y1}] = 1, Rectangle[{x1,y1}, {x2,y2}] = |x2-x1| * |y2-y1|
      "Rectangle" => {
        if args.is_empty() || args.len() == 1 {
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
                    args: vec![Expr::Integer(-1), p1[0].clone()].into(),
                  },
                ]
                .into(),
              }]
              .into(),
            };
            let height = Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  p2[1].clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), p1[1].clone()].into(),
                  },
                ]
                .into(),
              }]
              .into(),
            };
            let area = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![width, height].into(),
            };
            return crate::evaluator::evaluate_expr_to_expr(&area);
          }
          Ok(Expr::FunctionCall {
            name: "Area".to_string(),
            args: vec![expr.clone()].into(),
          })
        } else {
          Ok(Expr::FunctionCall {
            name: "Area".to_string(),
            args: vec![expr.clone()].into(),
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
                args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
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
                              args: vec![Expr::Integer(-1), p3[1].clone()]
                                .into(),
                            },
                          ]
                          .into(),
                        },
                      ]
                      .into(),
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
                              args: vec![Expr::Integer(-1), p1[1].clone()]
                                .into(),
                            },
                          ]
                          .into(),
                        },
                      ]
                      .into(),
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
                              args: vec![Expr::Integer(-1), p2[1].clone()]
                                .into(),
                            },
                          ]
                          .into(),
                        },
                      ]
                      .into(),
                    },
                  ]
                  .into(),
                }]
                .into(),
              },
            ]
            .into(),
          };
          return crate::evaluator::evaluate_expr_to_expr(&area_expr);
        }
        // Triangle embedded in 3-space: half the norm of the cross product
        // of two edge vectors.
        if args.len() == 1
          && let Expr::List(pts) = &args[0]
          && pts.len() == 3
          && let (Expr::List(p1), Expr::List(p2), Expr::List(p3)) =
            (&pts[0], &pts[1], &pts[2])
          && p1.len() == 3
          && p2.len() == 3
          && p3.len() == 3
        {
          let diff = |a: &Expr, b: &Expr| Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              a.clone(),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), b.clone()].into(),
              },
            ]
            .into(),
          };
          let u: Vec<Expr> = (0..3).map(|d| diff(&p2[d], &p1[d])).collect();
          let v: Vec<Expr> = (0..3).map(|d| diff(&p3[d], &p1[d])).collect();
          let minor = |i: usize, j: usize| Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![u[i].clone(), v[j].clone()].into(),
              },
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), u[j].clone(), v[i].clone()]
                  .into(),
              },
            ]
            .into(),
          };
          let sq = |e: Expr| Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![e, Expr::Integer(2)].into(),
          };
          // Sqrt[(m1^2 + m2^2 + m3^2)/4] — the 1/4 inside the radical
          // reproduces wolframscript's canonical forms (Sqrt[23/2],
          // 1/Sqrt[2], Sqrt[3]/2, …).
          let area_expr = Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(4)].into(),
                },
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![sq(minor(1, 2)), sq(minor(2, 0)), sq(minor(0, 1))]
                    .into(),
                },
              ]
              .into(),
            }]
            .into(),
          };
          return crate::evaluator::evaluate_expr_to_expr(&area_expr);
        }
        Ok(Expr::FunctionCall {
          name: "Area".to_string(),
          args: vec![expr.clone()].into(),
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
          args: vec![expr.clone()].into(),
        })
      }
      // A 2-D half-space has infinite area; other dimensions have no
      // 2-area.
      "HalfSpace" if half_space_parts(args).is_some() => {
        let (normal, _) = half_space_parts(args).unwrap();
        Ok(Expr::Identifier(
          if normal.len() == 2 {
            "Infinity"
          } else {
            "Undefined"
          }
          .to_string(),
        ))
      }
      // DiskSegment[{x, y}, r | {rx, ry}, {θ1, θ2}] — the circular
      // (elliptical) segment area rx ry (Δθ - Sin[Δθ])/2. wolframscript's
      // closed form uses two UnitSteps that BOTH fire at Δθ == 2 Pi
      // (giving 2 Pi rx ry) and clamp to Pi rx ry above — replicated.
      "DiskSegment" => {
        let Some((_, rx, ry, th1, th2, d)) = disk_segment_parts(args) else {
          return Ok(Expr::FunctionCall {
            name: "Area".to_string(),
            args: vec![expr.clone()].into(),
          });
        };
        if d < 0.0 {
          return Ok(Expr::Identifier("Undefined".to_string()));
        }
        let factor = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![rx, ry].into(),
        };
        const TWO_PI: f64 = std::f64::consts::TAU;
        let area = if (d - TWO_PI).abs() < 1e-12 {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(2),
              Expr::Constant("Pi".to_string()),
              factor,
            ]
            .into(),
          }
        } else if d > TWO_PI {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Constant("Pi".to_string()), factor].into(),
          }
        } else {
          let dt = disk_segment_dtheta(&th1, &th2)?;
          // Together hoists the rational content of Δθ - Sin[Δθ] so the
          // display matches wolframscript ((-2 + Pi)/4, not
          // (-1 + Pi/2)/2).
          Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                factor,
                Expr::FunctionCall {
                  name: "Together".to_string(),
                  args: vec![Expr::FunctionCall {
                    name: "Plus".to_string(),
                    args: vec![
                      dt.clone(),
                      Expr::FunctionCall {
                        name: "Times".to_string(),
                        args: vec![
                          Expr::Integer(-1),
                          Expr::FunctionCall {
                            name: "Sin".to_string(),
                            args: vec![dt].into(),
                          },
                        ]
                        .into(),
                      },
                    ]
                    .into(),
                  }]
                  .into(),
                },
              ]
              .into(),
            }),
            right: Box::new(Expr::Integer(2)),
          }
        };
        crate::evaluator::evaluate_expr_to_expr(&area)
      }
      // Circle has no area (it's 1D)
      "Circle" => Ok(Expr::Identifier("Undefined".to_string())),
      // A Tetrahedron is a 3-D solid, so its 2-area is Undefined.
      "Tetrahedron" => Ok(Expr::Identifier("Undefined".to_string())),
      // A Parallelogram is always a planar (2-D) region, so its Area equals
      // its RegionMeasure. Delegate to keep the two in sync. A two-vector
      // Parallelepiped is likewise planar; any other Parallelepiped is a
      // higher-dimensional solid with Undefined 2-area.
      "Parallelogram" => {
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "RegionMeasure".to_string(),
          args: vec![expr.clone()].into(),
        })
      }
      "Parallelepiped" if parallelogram_parts(args).is_some() => {
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "RegionMeasure".to_string(),
          args: vec![expr.clone()].into(),
        })
      }
      "Parallelepiped" => Ok(Expr::Identifier("Undefined".to_string())),
      // Simplex[{p0, p1, p2}] in the plane — the triangle area |Det[edges]|/2.
      // A higher-dimensional simplex has Undefined 2-area.
      "Simplex" if args.len() == 1 => {
        // Simplex[n]: Area (a 2-measure) is defined only for the 2-simplex,
        // where it is 1/2!; every other dimension is Undefined.
        if let Expr::Integer(n) = &args[0]
          && *n >= 0
        {
          return if *n == 2 {
            Ok(crate::functions::math_ast::make_rational(1, 2))
          } else {
            Ok(Expr::Identifier("Undefined".to_string()))
          };
        }
        if let Expr::List(pts) = &args[0]
          && pts.len() == 3
          && let Some(edges) = simplex_edges(pts)
        {
          det_measure(edges, 2)
        } else if matches!(&args[0], Expr::List(pts) if pts.len() != 3) {
          Ok(Expr::Identifier("Undefined".to_string()))
        } else {
          Ok(Expr::FunctionCall {
            name: "Area".to_string(),
            args: vec![expr.clone()].into(),
          })
        }
      }
      // Sphere[] / Sphere[p] / Sphere[p, r] — 4 Pi r^2 surface area for
      // a 3-D sphere. p (a 3-element list) is discarded since the
      // surface area is translation-invariant.
      "Sphere" => {
        let radius = match args.len() {
          0 => Expr::Integer(1),
          1 => {
            if let Expr::List(p) = &args[0]
              && p.len() == 3
            {
              Expr::Integer(1)
            } else {
              return Ok(Expr::FunctionCall {
                name: "Area".to_string(),
                args: vec![expr.clone()].into(),
              });
            }
          }
          2 => {
            if let Expr::List(p) = &args[0]
              && p.len() == 3
            {
              args[1].clone()
            } else {
              return Ok(Expr::FunctionCall {
                name: "Area".to_string(),
                args: vec![expr.clone()].into(),
              });
            }
          }
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Area".to_string(),
              args: vec![expr.clone()].into(),
            });
          }
        };
        // Area = 4 * Pi * r^2
        let area = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(4),
            Expr::Constant("Pi".to_string()),
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![radius, Expr::Integer(2)].into(),
            },
          ]
          .into(),
        };
        crate::evaluator::evaluate_expr_to_expr(&area)
      }
      // RegularPolygon[n] / [r, n] / [{r, theta}, n] / [{x, y}, rspec, n]
      // Area = (n/2) * r^2 * Sin[2 Pi / n]. Rotation and centre offsets
      // don't change the area, so they're discarded.
      "RegularPolygon" => {
        let (radius, n_expr) = match args.len() {
          1 => (Expr::Integer(1), args[0].clone()),
          2 => {
            // RegularPolygon[r, n] or RegularPolygon[{r, theta}, n]
            let r = match &args[0] {
              Expr::List(items) if items.len() == 2 => items[0].clone(),
              other => other.clone(),
            };
            (r, args[1].clone())
          }
          3 => {
            // RegularPolygon[{x, y}, rspec, n]; rspec is either a scalar
            // radius or a {radius, theta} list.
            let r = match &args[1] {
              Expr::List(items) if items.len() == 2 => items[0].clone(),
              other => other.clone(),
            };
            (r, args[2].clone())
          }
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Area".to_string(),
              args: vec![expr.clone()].into(),
            });
          }
        };
        // area = n/2 * r^2 * Sin[2 Pi / n]
        let half_n = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
            },
            n_expr.clone(),
          ]
          .into(),
        };
        let r_squared = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![radius, Expr::Integer(2)].into(),
        };
        let two_pi_over_n = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(2),
            Expr::Constant("Pi".to_string()),
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![n_expr, Expr::Integer(-1)].into(),
            },
          ]
          .into(),
        };
        let sin_term = Expr::FunctionCall {
          name: "Sin".to_string(),
          args: vec![two_pi_over_n].into(),
        };
        let area = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![half_n, r_squared, sin_term].into(),
        };
        crate::evaluator::evaluate_expr_to_expr(&area)
      }
      _ => Ok(Expr::FunctionCall {
        name: "Area".to_string(),
        args: vec![expr.clone()].into(),
      }),
    },
    _ => Ok(Expr::FunctionCall {
      name: "Area".to_string(),
      args: vec![expr.clone()].into(),
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
        args: vec![Expr::List(pts.to_vec().into())].into(),
      }]
      .into(),
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
      args: vec![coords[i].0.clone(), coords[j].1.clone()].into(),
    });
    // -x_j * y_i
    sum_terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), coords[j].0.clone(), coords[i].1.clone()]
        .into(),
    });
  }

  let area_expr = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      },
      Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![Expr::FunctionCall {
          name: "Plus".to_string(),
          args: sum_terms.into(),
        }]
        .into(),
      },
    ]
    .into(),
  };

  crate::evaluator::evaluate_expr_to_expr(&area_expr)
}

/// Compute the centroid of a geometric region.
fn compute_region_centroid(expr: &Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionCentroid".to_string(),
      args: vec![expr.clone()].into(),
    })
  };
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      "StadiumShape" if stadium_parts(args).is_some() => {
        let (p1, p2, _) = stadium_parts(args).unwrap();
        let mid: Vec<Expr> = p1
          .iter()
          .zip(p2.iter())
          .map(|(a, b)| Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![a.clone(), b.clone()].into(),
            }),
            right: Box::new(Expr::Integer(2)),
          })
          .collect();
        crate::evaluator::evaluate_expr_to_expr(&Expr::List(mid.into()))
      }
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
          Ok(Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()))
        } else {
          // Center is the first argument
          Ok(args[0].clone())
        }
      }
      // Ball[{x, y, z}, r] or Ball[] — centroid is the center
      "Ball" => {
        if args.is_empty() {
          Ok(Expr::List(
            vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)].into(),
          ))
        } else {
          Ok(args[0].clone())
        }
      }
      // Circle[{x, y}, r] — centroid is the center
      "Circle" => {
        if args.is_empty() {
          Ok(Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()))
        } else {
          Ok(args[0].clone())
        }
      }
      // Annulus[{x, y}, {r1, r2}] — the full ring is symmetric about its
      // center; Annulus[] is centered at the origin. The sector form
      // (three arguments) is not centrally symmetric, so leave it alone.
      "Annulus" => match args.len() {
        0 => Ok(Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into())),
        2 if matches!(&args[0], Expr::List(c) if c.len() == 2) => {
          Ok(args[0].clone())
        }
        _ => unevaluated(),
      },
      // Ellipsoid[center, {r1, ...}] — centroid is the center (works for any
      // dimension since an ellipsoid is symmetric about its center).
      "Ellipsoid" if args.len() == 2 && matches!(&args[0], Expr::List(_)) => {
        Ok(args[0].clone())
      }
      // Rectangle[{x1, y1}, {x2, y2}] — centroid is midpoint
      // Cuboid — the centroid is the midpoint of its corners.
      "Cuboid" => {
        let half = || Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
        };
        let (lo, hi): (Vec<Expr>, Option<Vec<Expr>>) =
          match (args.first(), args.get(1)) {
            (None, _) => (vec![Expr::Integer(0); 3], None),
            (Some(Expr::List(p)), None) if !p.is_empty() => {
              (p.iter().cloned().collect(), None)
            }
            (Some(Expr::List(p1)), Some(Expr::List(p2)))
              if p1.len() == p2.len() && !p1.is_empty() =>
            {
              (
                p1.iter().cloned().collect(),
                Some(p2.iter().cloned().collect()),
              )
            }
            _ => return unevaluated(),
          };
        let coords: Vec<Expr> = match hi {
          // Cuboid[] / Cuboid[p]: unit cube, centroid p + 1/2 per axis.
          None => lo
            .iter()
            .map(|l| Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![l.clone(), half()].into(),
            })
            .collect(),
          Some(hi) => lo
            .iter()
            .zip(hi.iter())
            .map(|(l, h)| Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                half(),
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![l.clone(), h.clone()].into(),
                },
              ]
              .into(),
            })
            .collect(),
        };
        crate::evaluator::evaluate_expr_to_expr(&Expr::List(coords.into()))
      }
      "Rectangle" => {
        if args.is_empty() {
          // Rectangle[] = Rectangle[{0,0},{1,1}]
          Ok(Expr::List(
            vec![
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
              },
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
              },
            ]
            .into(),
          ))
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
                  args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
                },
              ]
              .into(),
            };
            let cy = Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                p1[1].clone(),
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
                },
              ]
              .into(),
            };
            let result = Expr::List(vec![cx, cy].into());
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
                  args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
                },
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![p1[0].clone(), p2[0].clone()].into(),
                },
              ]
              .into(),
            };
            let cy = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
                },
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![p1[1].clone(), p2[1].clone()].into(),
                },
              ]
              .into(),
            };
            let result = Expr::List(vec![cx, cy].into());
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
      // Torus / FilledTorus — symmetric about the center.
      "Torus" | "FilledTorus" => {
        if let Some((center, _, _)) = torus_parts(args) {
          Ok(Expr::List(center.into()))
        } else {
          unevaluated()
        }
      }
      // Parallelogram[p, {v1, v2}] — centroid is p + (v1 + v2)/2.
      "Parallelogram" => {
        let (p, v1, v2) = match parallelogram_parts(args) {
          Some(parts) => parts,
          None => return unevaluated(),
        };
        let coords: Vec<Expr> = (0..p.len())
          .map(|d| Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              p[d].clone(),
              Expr::BinaryOp {
                op: BinaryOperator::Divide,
                left: Box::new(Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![v1[d].clone(), v2[d].clone()].into(),
                }),
                right: Box::new(Expr::Integer(2)),
              },
            ]
            .into(),
          })
          .collect();
        crate::evaluator::evaluate_expr_to_expr(&Expr::List(coords.into()))
      }
      // Parallelepiped[p, {v1, ..., vk}] — centroid is p + (v1 + ... + vk)/2.
      "Parallelepiped" if args.len() == 2 => {
        let (Expr::List(p), Expr::List(vecs)) = (&args[0], &args[1]) else {
          return unevaluated();
        };
        let dim = p.len();
        if dim == 0 || vecs.is_empty() {
          return unevaluated();
        }
        let mut cols: Vec<&[Expr]> = Vec::with_capacity(vecs.len());
        for v in vecs.iter() {
          let Expr::List(vc) = v else {
            return unevaluated();
          };
          if vc.len() != dim {
            return unevaluated();
          }
          cols.push(vc);
        }
        let coords: Vec<Expr> = (0..dim)
          .map(|d| {
            let sum: Vec<Expr> = cols.iter().map(|vc| vc[d].clone()).collect();
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                p[d].clone(),
                Expr::BinaryOp {
                  op: BinaryOperator::Divide,
                  left: Box::new(Expr::FunctionCall {
                    name: "Plus".to_string(),
                    args: sum.into(),
                  }),
                  right: Box::new(Expr::Integer(2)),
                },
              ]
              .into(),
            }
          })
          .collect();
        crate::evaluator::evaluate_expr_to_expr(&Expr::List(coords.into()))
      }
      // Platonic-solid primitives are centered at their center argument
      // (origin by default; a rotation spec keeps the origin).
      "Cube" | "Hexahedron" | "Octahedron" | "Dodecahedron" | "Icosahedron"
      | "Tetrahedron"
        if platonic_center_edge(args).is_some() =>
      {
        let (center, _) = platonic_center_edge(args).unwrap();
        crate::evaluator::evaluate_expr_to_expr(&Expr::List(center.into()))
      }
      // A SphericalShell is centered at its center argument.
      "SphericalShell" if spherical_shell_radii(args).is_some() => {
        crate::evaluator::evaluate_expr_to_expr(&args[0].clone())
      }
      // An unbounded half-space has no centroid: a list of Indeterminate.
      "HalfSpace" if half_space_parts(args).is_some() => {
        let (normal, _) = half_space_parts(args).unwrap();
        Ok(Expr::List(
          vec![Expr::Identifier("Indeterminate".to_string()); normal.len()]
            .into(),
        ))
      }
      // DiskSegment — the segment centroid lies on the angular bisector
      // at distance 4 r Sin[Δθ/2]^3 / (3 (Δθ - Sin[Δθ])) from the center
      // (each axis scaled by its semiaxis for the elliptical case).
      "DiskSegment"
        if disk_segment_parts(args).is_some_and(|(.., d)| d > 1e-12) =>
      {
        let (c, rx, ry, th1, th2, _) = disk_segment_parts(args).unwrap();
        let dt = disk_segment_dtheta(&th1, &th2)?;
        let mid = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![th1, th2].into(),
          }),
          right: Box::new(Expr::Integer(2)),
        };
        let half_dt =
          binop(BinaryOperator::Divide, dt.clone(), Expr::Integer(2));
        // Unit-circle offset 4 Sin[Δθ/2]^3 / (3 (Δθ - Sin[Δθ])).
        let offset = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(4),
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![
                  Expr::FunctionCall {
                    name: "Sin".to_string(),
                    args: vec![half_dt].into(),
                  },
                  Expr::Integer(3),
                ]
                .into(),
              },
            ]
            .into(),
          }),
          right: Box::new(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(3),
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  dt.clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![
                      Expr::Integer(-1),
                      Expr::FunctionCall {
                        name: "Sin".to_string(),
                        args: vec![dt].into(),
                      },
                    ]
                    .into(),
                  },
                ]
                .into(),
              },
            ]
            .into(),
          }),
        };
        let component =
          |center: &Expr, semi: &Expr, trig: &str| Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              center.clone(),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  semi.clone(),
                  offset.clone(),
                  Expr::FunctionCall {
                    name: trig.to_string(),
                    args: vec![mid.clone()].into(),
                  },
                ]
                .into(),
              },
            ]
            .into(),
          };
        crate::evaluator::evaluate_expr_to_expr(&Expr::List(
          vec![component(&c[0], &rx, "Cos"), component(&c[1], &ry, "Sin")]
            .into(),
        ))
      }
      // A CapsuleShape is centered at the midpoint of its axis.
      "CapsuleShape" if capsule_height_sq_radius(args).is_some() => {
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Mean".to_string(),
          args: vec![args[0].clone()].into(),
        })
      }
      // Tetrahedron / Simplex — the centroid is the mean of the vertices.
      "Tetrahedron" | "Simplex" => {
        // Simplex[n]: the standard n-simplex has vertices {0, e1, …, en}, so
        // its centroid is the mean {1/(n+1), …, 1/(n+1)} (n coordinates).
        if name == "Simplex"
          && args.len() == 1
          && let Expr::Integer(n) = &args[0]
          && *n >= 1
        {
          let coord = crate::functions::math_ast::make_rational(1, *n + 1);
          return Ok(Expr::List(vec![coord; *n as usize].into()));
        }
        if args.len() == 1
          && let Expr::List(pts) = &args[0]
          && pts.len() >= 2
          && pts.iter().all(|p| matches!(p, Expr::List(_)))
        {
          let mean = Expr::FunctionCall {
            name: "Mean".to_string(),
            args: vec![args[0].clone()].into(),
          };
          return crate::evaluator::evaluate_expr_to_expr(&mean);
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
  let coords: Vec<&crate::ExprList> = pts
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
        args: vec![Expr::List(pts.to_vec().into())].into(),
      }]
      .into(),
    });
  }
  let dim = coords[0].len();
  if !coords.iter().all(|c| c.len() == dim) {
    return Ok(Expr::FunctionCall {
      name: "RegionCentroid".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Triangle".to_string(),
        args: vec![Expr::List(pts.to_vec().into())].into(),
      }]
      .into(),
    });
  }
  let mut result = Vec::new();
  for d in 0..dim {
    let avg = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(3)].into(),
        },
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            coords[0][d].clone(),
            coords[1][d].clone(),
            coords[2][d].clone(),
          ]
          .into(),
        },
      ]
      .into(),
    };
    result.push(avg);
  }
  crate::evaluator::evaluate_expr_to_expr(&Expr::List(result.into()))
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
        args: vec![Expr::List(pts.to_vec().into())].into(),
      }]
      .into(),
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
          args: vec![coords[i].0.clone(), coords[j].1.clone()].into(),
        },
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(-1),
            coords[j].0.clone(),
            coords[i].1.clone(),
          ]
          .into(),
        },
      ]
      .into(),
    };

    area_terms.push(cross.clone());

    // (xi + xj) * cross
    cx_terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![coords[i].0.clone(), coords[j].0.clone()].into(),
        },
        cross.clone(),
      ]
      .into(),
    });

    // (yi + yj) * cross
    cy_terms.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![coords[i].1.clone(), coords[j].1.clone()].into(),
        },
        cross,
      ]
      .into(),
    });
  }

  // signed_area_2 = sum of area_terms
  let signed_area_2 = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: area_terms.into(),
  };

  // 1/(6A) = 1/(3 * signed_area_2)
  let inv_6a = Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(3), signed_area_2].into(),
      },
      Expr::Integer(-1),
    ]
    .into(),
  };

  let cx = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      inv_6a.clone(),
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: cx_terms.into(),
      },
    ]
    .into(),
  };

  let cy = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      inv_6a,
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: cy_terms.into(),
      },
    ]
    .into(),
  };

  crate::evaluator::evaluate_expr_to_expr(&Expr::List(vec![cx, cy].into()))
}

/// Compute centroid of a line (polyline): weighted average of segment midpoints.
fn compute_line_centroid(pts: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "RegionCentroid".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Line".to_string(),
        args: vec![Expr::List(pts.to_vec().into())].into(),
      }]
      .into(),
    })
  };

  // Extract coordinates
  let coords: Vec<&crate::ExprList> = pts
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
            args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
          },
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![coords[0][d].clone(), coords[1][d].clone()].into(),
          },
        ]
        .into(),
      });
    }
    return crate::evaluator::evaluate_expr_to_expr(&Expr::List(result.into()));
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
                args: vec![Expr::Integer(-1), coords[i][d].clone()].into(),
              },
            ]
            .into(),
          },
          Expr::Integer(2),
        ]
        .into(),
      });
    }
    let seg_length = make_sqrt(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: sq_terms.into(),
    });

    length_terms.push(seg_length.clone());

    for d in 0..dim {
      // midpoint_d * length
      let mid = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
          },
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![coords[i][d].clone(), coords[j][d].clone()].into(),
          },
          seg_length.clone(),
        ]
        .into(),
      };
      weighted_midpoints[d].push(mid);
    }
  }

  let total_length = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: length_terms.into(),
  };

  let mut result = Vec::new();
  for d in 0..dim {
    result.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![total_length.clone(), Expr::Integer(-1)].into(),
        },
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: weighted_midpoints[d].clone().into(),
        },
      ]
      .into(),
    });
  }

  crate::evaluator::evaluate_expr_to_expr(&Expr::List(result.into()))
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
      args: vec![var].into(),
    });
  }

  // Try exponential: a(n) = base^n
  if let Some(base) = try_exponential(&vals) {
    return Ok(Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![rational_to_expr(base.0, base.1), var].into(),
    });
  }

  // Try polynomial via finite differences
  if let Some(expr) = try_polynomial(&vals, &var_name) {
    return Ok(expr);
  }

  // If nothing works, return unevaluated
  Ok(Expr::FunctionCall {
    name: "FindSequenceFunction".to_string(),
    args: vec![data_expr.clone(), var_expr.clone()].into(),
  })
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
        args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
      }
    }
  }
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
              args: vec![var.clone(), Expr::Integer(i as i128)].into(),
            }
          };
          if c == (1, 1) {
            terms.push(n_power);
          } else {
            terms.push(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![c_expr, n_power].into(),
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
      args: terms.into(),
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
/// ArcLength[curve, {t, a, b}] — arc length of a curve parameterized by `t`.
///   • parametric: curve = {f1(t), …, fk(t)} → ∫ Sqrt[Σ fi'(t)²] dt
///   • scalar: curve = f(t) (the graph {t, f(t)}) → ∫ Sqrt[1 + f'(t)²] dt
/// over the range [a, b]. Leaves the call unevaluated if Woxi's Integrate
/// cannot close the resulting integral.
fn compute_arc_length_curve(
  curve: &Expr,
  iter: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("ArcLength", args));
  // The iterator must be {t, a, b} with `t` a symbol.
  let Expr::List(spec) = iter else {
    return unevaluated();
  };
  if spec.len() != 3 || !matches!(&spec[0], Expr::Identifier(_)) {
    return unevaluated();
  }
  let t = &spec[0];
  // Evaluate each derivative to a concrete expression BEFORE squaring. Squaring
  // a still-unevaluated `D[…]` whose value is `-g(t)` mis-signs the result
  // (Power over a freshly-reduced Times[-1, …] base), whereas squaring the
  // already-evaluated derivative is correct.
  let deriv = |f: &Expr| -> Result<Expr, InterpreterError> {
    evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "D".to_string(),
      args: vec![f.clone(), t.clone()].into(),
    })
  };
  let square = |e: Expr| Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![e, Expr::Integer(2)].into(),
  };
  let sum_of_squares = match curve {
    Expr::List(comps) if !comps.is_empty() => {
      let mut terms = Vec::with_capacity(comps.len());
      for c in comps.iter() {
        terms.push(square(deriv(c)?));
      }
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      }
    }
    // Scalar function f(t): the graph {t, f(t)} has speed Sqrt[1 + f'(t)^2].
    scalar => Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![Expr::Integer(1), square(deriv(scalar)?)].into(),
    },
  };
  // Simplify the speed first: Woxi's Integrate does not apply the Pythagorean
  // identity on its own, so e.g. Sqrt[Cos[t]^2 + Sin[t]^2] must collapse to 1
  // before integration.
  let integrand = Expr::FunctionCall {
    name: "Simplify".to_string(),
    args: vec![make_sqrt(sum_of_squares)].into(),
  };
  let integral = Expr::FunctionCall {
    name: "Integrate".to_string(),
    args: vec![integrand, iter.clone()].into(),
  };
  let result = evaluate_expr_to_expr(&integral)?;
  // If Integrate could not resolve it, keep ArcLength unevaluated rather than
  // leaking an Integrate[...] expression.
  if matches!(&result, Expr::FunctionCall { name, .. } if name == "Integrate") {
    return unevaluated();
  }
  Ok(result)
}

fn compute_arc_length(expr: &Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "ArcLength".to_string(),
      args: vec![expr.clone()].into(),
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
            args: vec![Expr::Integer(2), Expr::Constant("Pi".to_string())]
              .into(),
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
            ]
            .into(),
          };
          crate::evaluator::evaluate_expr_to_expr(&result)
        } else if args.len() == 3
          && !matches!(&args[1], Expr::List(_))
          && let Expr::List(spec) = &args[2]
          && spec.len() == 2
        {
          // Circle[center, r, {θ1, θ2}] (a circular arc) -> r*(θ2 - θ1)
          let result = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              args[1].clone(),
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  spec[1].clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), spec[0].clone()].into(),
                  },
                ]
                .into(),
              },
            ]
            .into(),
          };
          crate::evaluator::evaluate_expr_to_expr(&result)
        } else {
          unevaluated()
        }
      }
      // Unbounded 1-D regions have infinite arc length.
      "HalfLine" | "InfiniteLine" if !args.is_empty() => {
        Ok(Expr::Identifier("Infinity".to_string()))
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
      // Other filled regions (Disk, Polygon, Triangle, Rectangle, Ball,
      // Ellipsoid) are not curves, so their arc length is Undefined.
      "Disk" | "Polygon" | "Triangle" | "Rectangle" | "Ball" | "Ellipsoid" => {
        Ok(Expr::Identifier("Undefined".to_string()))
      }
      // Simplex[n]: only the 1-simplex is a curve, with arc length 1; every
      // other standard simplex is a filled region, so its arc length is
      // Undefined.
      "Simplex"
        if args.len() == 1
          && matches!(&args[0], Expr::Integer(n) if *n >= 0) =>
      {
        let Expr::Integer(n) = &args[0] else {
          unreachable!()
        };
        if *n == 1 {
          Ok(Expr::Integer(1))
        } else {
          Ok(Expr::Identifier("Undefined".to_string()))
        }
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
      args: vec![expr.clone()].into(),
    })
  };
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      // Stadium boundary: two straight sides plus the full circle,
      // 2 L + 2 Pi r.
      "StadiumShape" if stadium_parts(args).is_some() => {
        let (p1, p2, r) = stadium_parts(args).unwrap();
        let length = stadium_length(&p1, &p2);
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(2), length].into(),
            },
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(2), Expr::Constant("Pi".to_string()), r]
                .into(),
            },
          ]
          .into(),
        })
      }
      // DiskSegment (circular only) — chord + arc:
      // 2 r Sin[Δθ/2] + r Δθ. Elliptical segments need elliptic
      // integrals and stay unevaluated.
      "DiskSegment" => {
        let Some((_, rx, _ry, th1, th2, d)) = disk_segment_parts(args) else {
          return unevaluated();
        };
        // Elliptical case: the radius argument is a list.
        if matches!(&args[1], Expr::List(_)) || d < 0.0 {
          return unevaluated();
        }
        let dt = disk_segment_dtheta(&th1, &th2)?;
        let half_dt =
          binop(BinaryOperator::Divide, dt.clone(), Expr::Integer(2));
        let chord = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(2),
            rx.clone(),
            Expr::FunctionCall {
              name: "Sin".to_string(),
              args: vec![half_dt].into(),
            },
          ]
          .into(),
        };
        let arc = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![rx, dt].into(),
        };
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![chord, arc].into(),
        })
      }
      // Disk[{x, y}, r] -> 2*Pi*r, Disk[] -> 2*Pi
      "Disk" => {
        if args.is_empty() || args.len() == 1 {
          let result = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(2), Expr::Constant("Pi".to_string())]
              .into(),
          };
          crate::evaluator::evaluate_expr_to_expr(&result)
        } else if args.len() == 2 {
          let result = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(2),
              Expr::Constant("Pi".to_string()),
              args[1].clone(),
            ]
            .into(),
          };
          crate::evaluator::evaluate_expr_to_expr(&result)
        } else {
          unevaluated()
        }
      }
      // Ellipsoid[center, {r1, r2}] (2D) is a filled ellipse; its perimeter is
      // the ellipse circumference 4*r2*EllipticE[1 - (r1/r2)^2], matching WL's
      // convention of using the second semi-axis as the reference. A circular
      // ellipse (r1 == r2) simplifies to 2*Pi*r since EllipticE[0] = Pi/2.
      "Ellipsoid"
        if args.len() == 2
          && matches!(&args[0], Expr::List(c) if c.len() == 2)
          && matches!(&args[1], Expr::List(r) if r.len() == 2) =>
      {
        let Expr::List(radii) = &args[1] else {
          unreachable!()
        };
        let (r1, r2) = (radii[0].clone(), radii[1].clone());
        // m = 1 - (r1/r2)^2
        let ratio_sq = Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            binop(BinaryOperator::Divide, r1, r2.clone()),
            Expr::Integer(2),
          ]
          .into(),
        };
        let m = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Integer(1),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), ratio_sq].into(),
            },
          ]
          .into(),
        };
        let perimeter = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(4),
            r2,
            Expr::FunctionCall {
              name: "EllipticE".to_string(),
              args: vec![m].into(),
            },
          ]
          .into(),
        };
        crate::evaluator::evaluate_expr_to_expr(&perimeter)
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
                    args: vec![Expr::Integer(-1), p1[0].clone()].into(),
                  },
                ]
                .into(),
              }]
              .into(),
            };
            let height = Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  p2[1].clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), p1[1].clone()].into(),
                  },
                ]
                .into(),
              }]
              .into(),
            };
            let perimeter = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::Integer(2),
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![width, height].into(),
                },
              ]
              .into(),
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
  let coords: Vec<&crate::ExprList> = pts
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
        args: vec![Expr::List(pts.to_vec().into())].into(),
      }]
      .into(),
    });
  }

  let dim = coords[0].len();
  if !coords.iter().all(|c| c.len() == dim) {
    return Ok(Expr::FunctionCall {
      name: func_name.to_string(),
      args: vec![Expr::FunctionCall {
        name: "Line".to_string(),
        args: vec![Expr::List(pts.to_vec().into())].into(),
      }]
      .into(),
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
                args: vec![Expr::Integer(-1), coords[i][d].clone()].into(),
              },
            ]
            .into(),
          },
          Expr::Integer(2),
        ]
        .into(),
      });
    }
    segment_lengths.push(make_sqrt(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: sq_terms.into(),
    }));
  }

  if segment_lengths.len() == 1 {
    crate::evaluator::evaluate_expr_to_expr(&segment_lengths[0])
  } else {
    let total = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: segment_lengths.into(),
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
            let center = Expr::List(vec![Expr::Integer(0); dim].into());
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
          args: vec![center, radius].into(),
        })
      }

      // Circle[] → Circle[{0,0}, 1]; Circle[c] → Circle[c, 1]
      // Sphere[c, r] in 2D → Circle[c, r]
      "Circle" | "Sphere" => {
        let (center, radius) = match args.len() {
          0 => {
            let dim = if name == "Sphere" { 3 } else { 2 };
            let center = Expr::List(vec![Expr::Integer(0); dim].into());
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
          args: vec![center, radius].into(),
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
              let p2: Vec<Expr> = coords
                .iter()
                .map(|c| Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![c.clone(), Expr::Integer(1)].into(),
                })
                .map(|e| {
                  crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(e)
                })
                .collect();
              (coords.to_vec(), p2)
            } else {
              return Some(expr.clone());
            }
          }
          _ => {
            if let (Expr::List(c1), Expr::List(c2)) = (&args[0], &args[1]) {
              (c1.to_vec(), c2.to_vec())
            } else {
              return Some(expr.clone());
            }
          }
        };
        if p1.len() == 2 && p2.len() == 2 {
          let vertices = vec![
            Expr::List(vec![p1[0].clone(), p1[1].clone()].into()),
            Expr::List(vec![p2[0].clone(), p1[1].clone()].into()),
            Expr::List(vec![p2[0].clone(), p2[1].clone()].into()),
            Expr::List(vec![p1[0].clone(), p2[1].clone()].into()),
          ];
          normalize_polygon_vertices(vertices)
        } else {
          Some(expr.clone())
        }
      }

      // Triangle[] → Polygon[{{0,0},{1,0},{0,1}}] (sorted)
      // Triangle[{p1,p2,p3}] → Polygon with sorted vertices
      "Triangle" => {
        let vertices: Vec<Expr> = if args.is_empty() {
          vec![
            Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
            Expr::List(vec![Expr::Integer(1), Expr::Integer(0)].into()),
            Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into()),
          ]
        } else if let Some(Expr::List(pts)) = args.first() {
          pts.to_vec()
        } else {
          return Some(expr.clone());
        };
        normalize_polygon_vertices(vertices)
      }

      // Polygon[{v1, v2, ...}] → sorted canonical form
      "Polygon" => {
        if let Some(Expr::List(vertices)) = args.first() {
          normalize_polygon_vertices(vertices.to_vec())
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
              args: vec![Expr::List(sorted)].into(),
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
    args: vec![Expr::List(vertices.into())].into(),
  })
}

/// Compute RegionEqual[r1, r2, ...].
fn compute_region_equal(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // RegionEqual[] and RegionEqual[r] → True
  if args.len() <= 1 {
    return Ok(bool_expr(true));
  }

  // Try to normalize all regions
  let normalized: Vec<Option<Expr>> =
    args.iter().map(normalize_region).collect();

  // If any region is not recognized, return unevaluated
  if normalized.iter().any(|n| n.is_none()) {
    return Ok(unevaluated("RegionEqual", args));
  }

  let normalized: Vec<Expr> =
    normalized.into_iter().map(|n| n.unwrap()).collect();

  // Compare all pairs: all must be equal to the first
  let first = &normalized[0];
  let all_equal = normalized[1..]
    .iter()
    .all(|n| format!("{n:?}") == format!("{first:?}"));

  if all_equal {
    Ok(bool_expr(true))
  } else {
    Ok(bool_expr(false))
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
        args: vec![Expr::Integer(-1), vertex.clone()].into(),
      },
    ]
    .into(),
  };
  let v2 = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      p2.clone(),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), vertex.clone()].into(),
      },
    ]
    .into(),
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
        args: vec![Expr::List(
          vec![p1.clone(), vertex.clone(), p2.clone()].into(),
        )]
        .into(),
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
      args: vec![a.clone(), b.clone()].into(),
    })
    .collect();
  let dot = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: dot_terms.into(),
  };

  // Build magnitudes: |v1|, |v2|
  let mag1_terms: Vec<Expr> = v1_comps
    .iter()
    .map(|a| Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![a.clone(), Expr::Integer(2)].into(),
    })
    .collect();
  let mag1 = make_sqrt(Expr::FunctionCall {
    name: "Plus".to_string(),
    args: mag1_terms.into(),
  });

  let mag2_terms: Vec<Expr> = v2_comps
    .iter()
    .map(|a| Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![a.clone(), Expr::Integer(2)].into(),
    })
    .collect();
  let mag2 = make_sqrt(Expr::FunctionCall {
    name: "Plus".to_string(),
    args: mag2_terms.into(),
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
            args: vec![mag1, mag2].into(),
          },
          Expr::Integer(-1),
        ]
        .into(),
      },
    ]
    .into(),
  };
  let angle = Expr::FunctionCall {
    name: "ArcCos".to_string(),
    args: vec![cos_angle].into(),
  };

  crate::evaluator::evaluate_expr_to_expr(&angle)
}

/// Numerically evaluate an expression to f64 (via `N`), or `None` if it does
/// not reduce to a real number. Used only to decide convex vs. reflex turns.
fn polygon_numeric(e: &Expr) -> Option<f64> {
  let n = Expr::FunctionCall {
    name: "N".to_string(),
    args: vec![e.clone()].into(),
  };
  match crate::evaluator::evaluate_expr_to_expr(&n).ok()? {
    Expr::Real(f) => Some(f),
    Expr::Integer(i) => Some(i as f64),
    _ => None,
  }
}

/// Extract a 2-D point's coordinates as f64, or `None` if non-numeric.
fn polygon_point_2d(v: &Expr) -> Option<(f64, f64)> {
  match v {
    Expr::List(c) if c.len() == 2 => {
      Some((polygon_numeric(&c[0])?, polygon_numeric(&c[1])?))
    }
    _ => None,
  }
}

/// PolygonAngle[poly] / PolygonAngle[poly, vertex] — the interior angle(s) of
/// a polygon. Each angle is the (unsigned) angle between the two edges meeting
/// at a vertex; at a reflex (concave) vertex the interior angle is 2 Pi minus
/// that, detected from the polygon's orientation and the local turn direction.
fn compute_polygon_angle(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("PolygonAngle", args));
  // Accept Polygon[pts] or Triangle[pts].
  let verts: Vec<Expr> = match &args[0] {
    Expr::FunctionCall { name, args: ra }
      if (name == "Polygon" || name == "Triangle") && !ra.is_empty() =>
    {
      match &ra[0] {
        Expr::List(v) if v.len() >= 3 => v.iter().cloned().collect(),
        _ => return unevaluated(),
      }
    }
    _ => return unevaluated(),
  };
  let n = verts.len();

  // Polygon orientation via the shoelace sum (numeric when possible): the
  // sign tells which turn direction corresponds to a convex vertex.
  let orientation: Option<f64> = (|| {
    let mut sum = 0.0;
    for i in 0..n {
      let (x1, y1) = polygon_point_2d(&verts[i])?;
      let (x2, y2) = polygon_point_2d(&verts[(i + 1) % n])?;
      sum += x1 * y2 - x2 * y1;
    }
    Some(sum)
  })();

  let interior_angle = |i: usize| -> Result<Expr, InterpreterError> {
    let prev = &verts[(i + n - 1) % n];
    let cur = &verts[i];
    let next = &verts[(i + 1) % n];
    let base = compute_planar_angle(prev, cur, next)?;
    // Reflex correction only when every coordinate is numeric.
    if let (Some(orient), Some((px, py)), Some((cx, cy)), Some((nx, ny))) = (
      orientation,
      polygon_point_2d(prev),
      polygon_point_2d(cur),
      polygon_point_2d(next),
    ) {
      // Cross product of the incoming and outgoing edge vectors.
      let (e1x, e1y) = (cx - px, cy - py);
      let (e2x, e2y) = (nx - cx, ny - cy);
      let cross = e1x * e2y - e1y * e2x;
      // A turn opposite to the polygon's orientation is a reflex vertex.
      if cross * orient < 0.0 {
        let reflex = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(2), Expr::Constant("Pi".to_string())]
                .into(),
            },
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), base].into(),
            },
          ]
          .into(),
        };
        return crate::evaluator::evaluate_expr_to_expr(&reflex);
      }
    }
    Ok(base)
  };

  // Two-argument form: the angle at the requested vertex.
  if args.len() == 2 {
    let target = polygon_point_2d(&args[1]);
    for (i, v) in verts.iter().enumerate() {
      let same = match (polygon_point_2d(v), target) {
        (Some((vx, vy)), Some((tx, ty))) => {
          (vx - tx).abs() < 1e-12 && (vy - ty).abs() < 1e-12
        }
        _ => expr_to_string(v) == expr_to_string(&args[1]),
      };
      if same {
        return interior_angle(i);
      }
    }
    return unevaluated();
  }

  // One-argument form: angles at every vertex. wolframscript lists them
  // starting from the vertex with minimum x (ties broken by minimum y) and
  // then follows the polygon's cyclic order — its canonical ordering for the
  // list form. When the coordinates are not all numeric we cannot pick that
  // start, so we fall back to input order.
  let mut start = 0usize;
  let mut best: Option<(f64, f64)> = None;
  let mut all_numeric = true;
  for (i, v) in verts.iter().enumerate() {
    match polygon_point_2d(v) {
      Some((x, y)) => {
        if best.is_none_or(|(bx, by)| x < bx || (x == bx && y < by)) {
          best = Some((x, y));
          start = i;
        }
      }
      None => {
        all_numeric = false;
        break;
      }
    }
  }
  if !all_numeric {
    start = 0;
  }
  let mut angles = Vec::with_capacity(n);
  for j in 0..n {
    angles.push(interior_angle((start + j) % n)?);
  }
  Ok(Expr::List(angles.into()))
}

// ─── Insphere ──────────────────────────────────────────────────────────────

/// Compute the insphere (incircle) of a geometric region.
/// For a 2D Triangle: returns Sphere[{cx, cy}, r]
/// For a 3D Tetrahedron: returns Sphere[{cx, cy, cz}, r]
/// Solve the square rational linear system `A x = b` by Gauss-Jordan
/// elimination. Returns None if the matrix is singular (degenerate points).
fn solve_rational_system(
  mut a: Vec<Vec<(i128, i128)>>,
  b: &[(i128, i128)],
) -> Option<Vec<(i128, i128)>> {
  let n = a.len();
  for (i, row) in a.iter_mut().enumerate() {
    row.push(b[i]);
  }
  for col in 0..n {
    let pivot_row = (col..n).find(|&r| a[r][col].0 != 0)?;
    a.swap(col, pivot_row);
    let pivot = a[col][col];
    for j in col..=n {
      a[col][j] = rat_div(a[col][j], pivot)?;
    }
    for r in 0..n {
      if r == col {
        continue;
      }
      let factor = a[r][col];
      if factor.0 == 0 {
        continue;
      }
      for j in col..=n {
        a[r][j] = rat_sub(a[r][j], rat_mul(factor, a[col][j]));
      }
    }
  }
  Some((0..n).map(|i| a[i][n]).collect())
}

/// Circumsphere[{p0, …, pd}] — the unique sphere through `d + 1` points in
/// `d` dimensions, returned as `Sphere[center, radius]`. The circumcenter is
/// equidistant from every point, giving the linear system
/// `2 (pi - p0) . c = |pi|^2 - |p0|^2`. Exact rational inputs yield an exact
/// center and a (possibly radical) radius; otherwise a machine-float result.
/// Wrong point counts or degenerate (collinear/coplanar) inputs stay
/// unevaluated, matching wolframscript.
/// BoundingRegion[{pt, ...}] — the smallest axis-aligned box containing the
/// points. wolframscript returns Rectangle[{mins}, {maxs}] for 2D points and
/// Cuboid[{mins}, {maxs}] for 1D or >=3D points. The min/max are exact (Min/Max
/// preserve integers and rationals). Malformed or non-numeric input, and the
/// two-argument (named-method) form, are left unevaluated.
fn compute_bounding_region(expr: &Expr) -> Result<Expr, InterpreterError> {
  let uneval = || {
    Ok(Expr::FunctionCall {
      name: "BoundingRegion".to_string(),
      args: vec![expr.clone()].into(),
    })
  };
  // wolframscript emits this when the argument is neither a region nor a
  // structurally-valid list of equal-length coordinate vectors.
  let regl = || {
    crate::emit_message(&format!(
      "BoundingRegion::regl: The argument {} should be a region or a list of points.",
      crate::syntax::expr_to_string(expr)
    ));
    uneval()
  };

  let Expr::List(points) = expr else {
    return regl();
  };
  if points.is_empty() {
    return regl();
  }
  // Every point must be a coordinate list of the same dimension d.
  let coords: Vec<&[Expr]> = points
    .iter()
    .filter_map(|p| match p {
      Expr::List(c) => Some(c.as_slice()),
      _ => None,
    })
    .collect();
  if coords.len() != points.len() {
    return regl();
  }
  let d = coords[0].len();
  if d == 0 || coords.iter().any(|c| c.len() != d) {
    return regl();
  }

  // Min/Max per coordinate; symbolic coordinates stay as Min[…]/Max[…],
  // matching wolframscript.
  let mut mins = Vec::with_capacity(d);
  let mut maxs = Vec::with_capacity(d);
  for j in 0..d {
    let col: Vec<Expr> = coords.iter().map(|c| c[j].clone()).collect();
    mins.push(crate::functions::math_ast::min_ast(&col)?);
    maxs.push(crate::functions::math_ast::max_ast(&col)?);
  }

  // 2D points give a Rectangle; 1D and >=3D give a Cuboid.
  let head = if d == 2 { "Rectangle" } else { "Cuboid" };
  Ok(Expr::FunctionCall {
    name: head.to_string(),
    args: vec![Expr::List(mins.into()), Expr::List(maxs.into())].into(),
  })
}

/// PerpendicularBisector[{p1, p2}] / PerpendicularBisector[Line[{p1, p2}]] —
/// the perpendicular bisector of the segment p1–p2, returned as
/// `InfiniteLine[midpoint, {dy, -dx}]` where `{dx, dy} = p2 - p1` and
/// `midpoint = (p1 + p2)/2`. Only 2-D points are handled (wolframscript leaves
/// higher-dimensional or malformed input unevaluated).
fn compute_perpendicular_bisector(
  expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let uneval = || {
    Ok(Expr::FunctionCall {
      name: "PerpendicularBisector".to_string(),
      args: vec![expr.clone()].into(),
    })
  };
  // Accept either a bare {p1, p2} list or a Line[{p1, p2}] wrapper.
  let pts = match expr {
    Expr::List(pts) => pts,
    Expr::FunctionCall { name, args } if name == "Line" && args.len() == 1 => {
      match &args[0] {
        Expr::List(pts) => pts,
        _ => return uneval(),
      }
    }
    _ => return uneval(),
  };
  if pts.len() != 2 {
    return uneval();
  }
  let mut coords: Vec<&crate::ExprList> = Vec::with_capacity(2);
  for pt in pts.iter() {
    let Expr::List(c) = pt else {
      return uneval();
    };
    if c.len() != 2 {
      return uneval();
    }
    coords.push(c);
  }
  let (p1, p2) = (coords[0], coords[1]);

  let eval = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e);
  let sub = |a: &Expr, b: &Expr| Expr::FunctionCall {
    name: "Subtract".to_string(),
    args: vec![a.clone(), b.clone()].into(),
  };
  let midpoint_coord = |a: &Expr, b: &Expr| Expr::FunctionCall {
    name: "Divide".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![a.clone(), b.clone()].into(),
      },
      Expr::Integer(2),
    ]
    .into(),
  };

  // Midpoint = ((p1 + p2)/2).
  let mid = Expr::List(
    vec![
      eval(midpoint_coord(&p1[0], &p2[0]))?,
      eval(midpoint_coord(&p1[1], &p2[1]))?,
    ]
    .into(),
  );
  // Direction perpendicular to p2 - p1 = {dx, dy}: {dy, -dx}.
  let dir = Expr::List(
    vec![eval(sub(&p2[1], &p1[1]))?, eval(sub(&p1[0], &p2[0]))?].into(),
  );

  Ok(Expr::FunctionCall {
    name: "InfiniteLine".to_string(),
    args: vec![mid, dir].into(),
  })
}

/// AngleBisector[{q1, p, q2}] — the bisector of the interior angle at `p`
/// formed by the rays p→q1 and p→q2, returned as `InfiniteLine[p, dir]` where
/// `dir = Normalize[q1 - p] + Normalize[q2 - p]`. Only 2-D points are handled
/// (wolframscript leaves higher-dimensional or malformed input unevaluated).
fn compute_angle_bisector(expr: &Expr) -> Result<Expr, InterpreterError> {
  let uneval = || {
    Ok(Expr::FunctionCall {
      name: "AngleBisector".to_string(),
      args: vec![expr.clone()].into(),
    })
  };
  let Expr::List(pts) = expr else {
    return uneval();
  };
  if pts.len() != 3 {
    return uneval();
  }
  // Each of q1, p, q2 must be a 2-element coordinate list.
  let mut coords: Vec<&crate::ExprList> = Vec::with_capacity(3);
  for pt in pts.iter() {
    let Expr::List(c) = pt else {
      return uneval();
    };
    if c.len() != 2 {
      return uneval();
    }
    coords.push(c);
  }
  let (q1, p, q2) = (coords[0], coords[1], coords[2]);

  // Normalize[qi - p] for i in {1, 2}, as a 2-vector expression.
  let normalized_leg = |q: &crate::ExprList| -> Expr {
    let diff = Expr::List(
      vec![
        Expr::FunctionCall {
          name: "Subtract".to_string(),
          args: vec![q[0].clone(), p[0].clone()].into(),
        },
        Expr::FunctionCall {
          name: "Subtract".to_string(),
          args: vec![q[1].clone(), p[1].clone()].into(),
        },
      ]
      .into(),
    );
    Expr::FunctionCall {
      name: "Normalize".to_string(),
      args: vec![diff].into(),
    }
  };

  // dir = Simplify[Normalize[q1 - p] + Normalize[q2 - p]]. Simplify reaches
  // wolframscript's canonical radical form (e.g. 1/Sqrt[2] + 1/Sqrt[2] ->
  // Sqrt[2]), which plain Plus evaluation leaves as 2/Sqrt[2].
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![normalized_leg(q1), normalized_leg(q2)].into(),
  };
  let dir = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Simplify".to_string(),
    args: vec![sum].into(),
  })?;

  Ok(Expr::FunctionCall {
    name: "InfiniteLine".to_string(),
    args: vec![Expr::List(p.clone()), dir].into(),
  })
}

fn compute_circumsphere(expr: &Expr) -> Result<Expr, InterpreterError> {
  let uneval = || {
    Ok(Expr::FunctionCall {
      name: "Circumsphere".to_string(),
      args: vec![expr.clone()].into(),
    })
  };
  let Expr::List(points) = expr else {
    return uneval();
  };
  // Every point must be a coordinate list of the same dimension d, and there
  // must be exactly d + 1 of them.
  let coords: Vec<&[Expr]> = points
    .iter()
    .filter_map(|p| match p {
      Expr::List(c) => Some(c.as_slice()),
      _ => None,
    })
    .collect();
  if coords.len() != points.len() {
    return uneval();
  }
  let n = coords.len();
  if n < 2 {
    return uneval();
  }
  let d = coords[0].len();
  if d == 0 || n != d + 1 || coords.iter().any(|c| c.len() != d) {
    return uneval();
  }

  // Exact path when every coordinate is an integer or rational.
  let rpts: Option<Vec<Vec<(i128, i128)>>> = coords
    .iter()
    .map(|pt| pt.iter().map(expr_to_rational).collect::<Option<Vec<_>>>())
    .collect();
  if let Some(rpts) = rpts {
    let p0 = &rpts[0];
    let mut a: Vec<Vec<(i128, i128)>> = Vec::with_capacity(d);
    let mut b: Vec<(i128, i128)> = Vec::with_capacity(d);
    for pi in rpts.iter().skip(1) {
      let mut row = Vec::with_capacity(d);
      let mut rhs = (0i128, 1i128);
      for j in 0..d {
        row.push(rat_mul((2, 1), rat_sub(pi[j], p0[j])));
        rhs =
          rat_add(rhs, rat_sub(rat_mul(pi[j], pi[j]), rat_mul(p0[j], p0[j])));
      }
      a.push(row);
      b.push(rhs);
    }
    let Some(center) = solve_rational_system(a, &b) else {
      return uneval();
    };
    // radius^2 = sum (center_j - p0_j)^2
    let mut r2 = (0i128, 1i128);
    for j in 0..d {
      let diff = rat_sub(center[j], p0[j]);
      r2 = rat_add(r2, rat_mul(diff, diff));
    }
    let center_expr = Expr::List(
      center
        .iter()
        .map(|&(cn, cd)| rational_to_expr(cn, cd))
        .collect::<Vec<_>>()
        .into(),
    );
    let radius = crate::evaluator::evaluate_function_call_ast(
      "Sqrt",
      &[rational_to_expr(r2.0, r2.1)],
    )?;
    return Ok(Expr::FunctionCall {
      name: "Sphere".to_string(),
      args: vec![center_expr, radius].into(),
    });
  }

  // Float path.
  let fpts: Option<Vec<Vec<f64>>> = coords
    .iter()
    .map(|pt| {
      pt.iter()
        .map(crate::functions::math_ast::try_eval_to_f64)
        .collect::<Option<Vec<_>>>()
    })
    .collect();
  let Some(fpts) = fpts else {
    return uneval();
  };
  let p0 = &fpts[0];
  let mut a = vec![vec![0.0f64; d]; d];
  let mut b = vec![0.0f64; d];
  for (i, pi) in fpts.iter().skip(1).enumerate() {
    for j in 0..d {
      a[i][j] = 2.0 * (pi[j] - p0[j]);
      b[i] += pi[j] * pi[j] - p0[j] * p0[j];
    }
  }
  let Some(center) = solve_float_system(a, b) else {
    return uneval();
  };
  let r2: f64 = (0..d).map(|j| (center[j] - p0[j]).powi(2)).sum();
  let center_expr = Expr::List(
    center
      .iter()
      .map(|&c| Expr::Real(c))
      .collect::<Vec<_>>()
      .into(),
  );
  Ok(Expr::FunctionCall {
    name: "Sphere".to_string(),
    args: vec![center_expr, Expr::Real(r2.sqrt())].into(),
  })
}

/// Gauss-Jordan solve for a small f64 system. None if singular.
fn solve_float_system(
  mut a: Vec<Vec<f64>>,
  mut b: Vec<f64>,
) -> Option<Vec<f64>> {
  let n = a.len();
  for (i, row) in a.iter_mut().enumerate() {
    row.push(b[i]);
  }
  for col in 0..n {
    let pivot_row = (col..n).max_by(|&r1, &r2| {
      a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap()
    })?;
    if a[pivot_row][col].abs() < 1e-12 {
      return None;
    }
    a.swap(col, pivot_row);
    let pivot = a[col][col];
    for j in col..=n {
      a[col][j] /= pivot;
    }
    for r in 0..n {
      if r == col {
        continue;
      }
      let factor = a[r][col];
      for j in col..=n {
        a[r][j] -= factor * a[col][j];
      }
    }
  }
  b = (0..n).map(|i| a[i][n]).collect();
  Some(b)
}

fn compute_insphere(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    // Raw point-list form: Insphere[{p1, …, p_{n+1}}] — the sphere inscribed
    // in the simplex spanned by the points. Handles a triangle (3 points in
    // 2D) and a tetrahedron (4 points in 3D), reusing the same exact helpers
    // as the Triangle[…] / Tetrahedron[…] wrapper forms.
    Expr::List(vertices) => {
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
      if pts.len() == vertices.len() {
        if pts.len() == 3 && pts.iter().all(|p| p.len() == 2) {
          return insphere_triangle_2d(pts[0], pts[1], pts[2]);
        }
        if pts.len() == 4 && pts.iter().all(|p| p.len() == 3) {
          return insphere_tetrahedron(pts[0], pts[1], pts[2], pts[3]);
        }
      }
      Ok(Expr::FunctionCall {
        name: "Insphere".to_string(),
        args: vec![expr.clone()].into(),
      })
    }
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
          args: vec![expr.clone()].into(),
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
          args: vec![expr.clone()].into(),
        })
      }
      // Simplex[{p0, …, pk}] — the k-simplex spanned by its vertices. A 2-simplex
      // (3 points in 2D) is a triangle and a 3-simplex (4 points in 3D) is a
      // tetrahedron, so both reuse the same exact helpers as the corresponding
      // Triangle[…]/Tetrahedron[…] wrapper forms.
      "Simplex" if args.len() == 1 => {
        if let Expr::List(vertices) = &args[0] {
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
          if pts.len() == vertices.len() {
            if pts.len() == 3 && pts.iter().all(|p| p.len() == 2) {
              return insphere_triangle_2d(pts[0], pts[1], pts[2]);
            }
            if pts.len() == 4 && pts.iter().all(|p| p.len() == 3) {
              return insphere_tetrahedron(pts[0], pts[1], pts[2], pts[3]);
            }
          }
        }
        Ok(Expr::FunctionCall {
          name: "Insphere".to_string(),
          args: vec![expr.clone()].into(),
        })
      }
      _ => {
        // Normalize no-arg primitives like Disk[] → Disk[{0,0}]
        let normalized = if args.is_empty()
          && (name == "Disk" || name == "Ball")
        {
          let center = match name.as_str() {
            "Disk" => {
              Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into())
            }
            "Ball" => Expr::List(
              vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)].into(),
            ),
            _ => unreachable!(),
          };
          Expr::FunctionCall {
            name: name.clone(),
            args: vec![center].into(),
          }
        } else {
          expr.clone()
        };
        crate::emit_message(&format!(
          "Insphere::indep: Insphere does not exist for {}.",
          crate::syntax::expr_to_string(&normalized)
        ));
        Ok(Expr::FunctionCall {
          name: "Insphere".to_string(),
          args: vec![normalized].into(),
        })
      }
    },
    _ => Ok(Expr::FunctionCall {
      name: "Insphere".to_string(),
      args: vec![expr.clone()].into(),
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
    args: vec![a, b].into(),
  }
}

/// Helper: build a * b
fn insphere_times(a: Expr, b: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![a, b].into(),
  }
}

/// Helper: build a - b
fn insphere_minus(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Minus, a, b)
}

/// Helper: build a^n
fn insphere_power(base: Expr, exp: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![base, exp].into(),
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
    args: vec![a.clone(), b.clone(), c.clone()].into(),
  };

  // Center coordinates: (a*x1 + b*x2 + c*x3) / (a+b+c)
  let cx_num = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      insphere_times(a.clone(), p1[0].clone()),
      insphere_times(b.clone(), p2[0].clone()),
      insphere_times(c.clone(), p3[0].clone()),
    ]
    .into(),
  };
  let cy_num = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      insphere_times(a, p1[1].clone()),
      insphere_times(b, p2[1].clone()),
      insphere_times(c, p3[1].clone()),
    ]
    .into(),
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
      ]
      .into(),
    }]
    .into(),
  };

  // radius = area / semiperimeter = (area_2/2) / (perimeter/2) = area_2 / perimeter
  let radius =
    insphere_times(area_2, insphere_power(perimeter, Expr::Integer(-1)));

  // Build Sphere[{cx, cy}, r]
  let center = Expr::List(vec![cx, cy].into());
  let sphere = Expr::FunctionCall {
    name: "Sphere".to_string(),
    args: vec![center, radius].into(),
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
    ]
    .into(),
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
    args: vec![cm1.clone(), cm2.clone(), cm3.clone(), cm4.clone()].into(),
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
      ]
      .into(),
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
    ]
    .into(),
  };

  // radius = |det| / total_cross
  let radius = insphere_times(
    Expr::FunctionCall {
      name: "Abs".to_string(),
      args: vec![det].into(),
    },
    insphere_power(total_cross, Expr::Integer(-1)),
  );

  let center = Expr::List(center_coords.into());
  let sphere = Expr::FunctionCall {
    name: "Sphere".to_string(),
    args: vec![center, radius].into(),
  };

  crate::evaluator::evaluate_expr_to_expr(&sphere)
}

/// Squared distance between two points of equal dimension, as a symbolic
/// Plus of squared coordinate differences.
fn dist2_nd(p1: &[Expr], p2: &[Expr]) -> Expr {
  let terms: Vec<Expr> = p1
    .iter()
    .zip(p2.iter())
    .map(|(c1, c2)| {
      insphere_power(insphere_minus(c2.clone(), c1.clone()), Expr::Integer(2))
    })
    .collect();
  if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    }
  }
}

/// TriangleCenter[tri] / TriangleCenter[tri, ctype] — a named center of a
/// triangle. The triangle can be Triangle[{p1, p2, p3}] or a bare vertex
/// list; ctype is one of "Centroid" (the default), "Circumcenter",
/// "Incenter", "NinePointCenter", "Orthocenter" or "SymmedianPoint".
///
/// Every supported center is the barycentric combination
/// (w1*p1 + w2*p2 + w3*p3) / (w1 + w2 + w3) whose weights are polynomial in
/// the squared side lengths (the incenter uses the side lengths themselves),
/// so exact input stays exact and the formulas work in any embedding
/// dimension.
fn compute_triangle_center(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let uneval = || Ok(unevaluated("TriangleCenter", args));
  let vertices = match &args[0] {
    Expr::FunctionCall { name, args: targs }
      if name == "Triangle" && targs.len() == 1 =>
    {
      &targs[0]
    }
    v @ Expr::List(_) => v,
    _ => return uneval(),
  };
  let Expr::List(vs) = vertices else {
    return uneval();
  };
  if vs.len() != 3 {
    return uneval();
  }
  let pts: Vec<&[Expr]> = vs
    .iter()
    .filter_map(|v| match v {
      Expr::List(c) => Some(c.as_slice()),
      _ => None,
    })
    .collect();
  if pts.len() != 3 {
    return uneval();
  }
  let d = pts[0].len();
  if d < 2 || pts.iter().any(|p| p.len() != d) {
    return uneval();
  }
  let ctype = match args.get(1) {
    None => "Centroid",
    Some(Expr::String(s)) => s.as_str(),
    Some(_) => return uneval(),
  };

  // Squared side lengths; side a is opposite vertex p1, etc.
  let a2 = dist2_nd(pts[1], pts[2]);
  let b2 = dist2_nd(pts[0], pts[2]);
  let c2 = dist2_nd(pts[0], pts[1]);

  let plus3 = |x: Expr, y: Expr, z: Expr| Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![x, y, z].into(),
  };
  let neg = |x: Expr| insphere_times(Expr::Integer(-1), x);
  // Law-of-cosines terms: b² + c² − a² = 2*b*c*Cos[A], and cyclic.
  let ca = plus3(b2.clone(), c2.clone(), neg(a2.clone()));
  let cb = plus3(c2.clone(), a2.clone(), neg(b2.clone()));
  let cc = plus3(a2.clone(), b2.clone(), neg(c2.clone()));

  let weights: [Expr; 3] = match ctype {
    // (p1 + p2 + p3) / 3
    "Centroid" => [Expr::Integer(1), Expr::Integer(1), Expr::Integer(1)],
    // a : b : c
    "Incenter" => [
      make_sqrt(a2.clone()),
      make_sqrt(b2.clone()),
      make_sqrt(c2.clone()),
    ],
    // a²(b² + c² − a²) : b²(c² + a² − b²) : c²(a² + b² − c²)
    "Circumcenter" => [
      insphere_times(a2.clone(), ca.clone()),
      insphere_times(b2.clone(), cb.clone()),
      insphere_times(c2.clone(), cc.clone()),
    ],
    // tan A : tan B : tan C, cleared of denominators
    "Orthocenter" => [
      insphere_times(cb.clone(), cc.clone()),
      insphere_times(ca.clone(), cc.clone()),
      insphere_times(ca.clone(), cb.clone()),
    ],
    // a²(b² + c²) − (b² − c²)², and cyclic
    "NinePointCenter" => [
      insphere_minus(
        insphere_times(a2.clone(), insphere_plus(b2.clone(), c2.clone())),
        insphere_power(
          insphere_minus(b2.clone(), c2.clone()),
          Expr::Integer(2),
        ),
      ),
      insphere_minus(
        insphere_times(b2.clone(), insphere_plus(c2.clone(), a2.clone())),
        insphere_power(
          insphere_minus(c2.clone(), a2.clone()),
          Expr::Integer(2),
        ),
      ),
      insphere_minus(
        insphere_times(c2.clone(), insphere_plus(a2.clone(), b2.clone())),
        insphere_power(
          insphere_minus(a2.clone(), b2.clone()),
          Expr::Integer(2),
        ),
      ),
    ],
    // a² : b² : c²
    "SymmedianPoint" => [a2.clone(), b2.clone(), c2.clone()],
    _ => return uneval(),
  };

  let total = plus3(weights[0].clone(), weights[1].clone(), weights[2].clone());
  let coords: Vec<Expr> = (0..d)
    .map(|j| {
      let num = plus3(
        insphere_times(weights[0].clone(), pts[0][j].clone()),
        insphere_times(weights[1].clone(), pts[1][j].clone()),
        insphere_times(weights[2].clone(), pts[2][j].clone()),
      );
      insphere_times(num, insphere_power(total.clone(), Expr::Integer(-1)))
    })
    .collect();
  crate::evaluator::evaluate_expr_to_expr(&Expr::List(coords.into()))
}

/// RegionWithin[reg1, reg2] - True if reg2 is entirely within reg1
fn region_within(
  reg1: &Expr,
  reg2: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::evaluator::type_helpers::expr_to_number;

  let unevaluated = || Ok(unevaluated("RegionWithin", args));

  let true_expr = || Ok(bool_expr(true));
  let false_expr = || Ok(bool_expr(false));

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

/// `true` if `expr` is structurally zero (e.g. `Integer(0)`). Used to
/// short-circuit the symbolic-complex DirectedInfinity normalisation when
/// the imaginary part collapses to zero — that's the pure-real case
/// already handled above.
fn is_zero_expr(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(0))
}

/// Split a complex-shaped expression into its real and imaginary parts as
/// separate Expr values. Each Plus term either contains a single `I`
/// factor (imaginary contribution) or none (real contribution); both kinds
/// of contribution must themselves be real-valued (no `I` anywhere). When
/// any term doesn't fit, returns None and the caller leaves the input
/// unchanged. Used by `DirectedInfinity` to normalise symbolic directions
/// like `-1 + 2*Pi*I` into `(-1 + 2*Pi*I)/Sqrt[1 + 4*Pi^2]`.
pub fn split_real_imag_symbolic(expr: &Expr) -> Option<(Expr, Expr)> {
  fn pull_i_factor(e: &Expr) -> Option<Expr> {
    // If e contains `I` exactly once as a simple Times factor (or IS `I`),
    // return e/I (the real coefficient). Returns None otherwise.
    match e {
      Expr::Identifier(name) if name == "I" => Some(Expr::Integer(1)),
      Expr::FunctionCall { name, args } if name == "Times" => {
        let mut found = false;
        let mut rest: Vec<Expr> = Vec::new();
        for a in args {
          if matches!(a, Expr::Identifier(n) if n == "I") {
            if found {
              return None;
            }
            found = true;
          } else if expr_contains_imag(a) {
            return None;
          } else {
            rest.push(a.clone());
          }
        }
        if !found {
          return None;
        }
        Some(match rest.len() {
          0 => Expr::Integer(1),
          1 => rest.into_iter().next().unwrap(),
          _ => Expr::FunctionCall {
            name: "Times".to_string(),
            args: rest.into(),
          },
        })
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        // Recurse with a flat list of factors.
        let mut factors: Vec<Expr> = Vec::new();
        flatten_times(left, &mut factors);
        flatten_times(right, &mut factors);
        let times_expr = Expr::FunctionCall {
          name: "Times".to_string(),
          args: factors.into(),
        };
        pull_i_factor(&times_expr)
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => pull_i_factor(operand).map(|r| Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(r),
      }),
      _ => None,
    }
  }
  fn flatten_times(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args {
          flatten_times(a, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        flatten_times(left, out);
        flatten_times(right, out);
      }
      _ => out.push(e.clone()),
    }
  }
  fn collect_plus(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::FunctionCall { name, args } if name == "Plus" => {
        for a in args {
          collect_plus(a, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left,
        right,
      } => {
        collect_plus(left, out);
        collect_plus(right, out);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left,
        right,
      } => {
        collect_plus(left, out);
        out.push(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new((**right).clone()),
        });
      }
      _ => out.push(e.clone()),
    }
  }
  let mut terms = Vec::new();
  collect_plus(expr, &mut terms);
  let mut re_terms: Vec<Expr> = Vec::new();
  let mut im_terms: Vec<Expr> = Vec::new();
  for t in terms {
    match pull_i_factor(&t) {
      Some(r) => im_terms.push(r),
      None => {
        if expr_contains_imag(&t) {
          return None;
        }
        re_terms.push(t);
      }
    }
  }
  fn build_plus(terms: Vec<Expr>) -> Expr {
    match terms.len() {
      0 => Expr::Integer(0),
      1 => terms.into_iter().next().unwrap(),
      _ => Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      },
    }
  }
  Some((build_plus(re_terms), build_plus(im_terms)))
}

/// `true` if `expr` contains the symbol `I` anywhere. Used as a guard in
/// [`split_real_imag_symbolic`] so a "real" term doesn't accidentally
/// hide an imaginary factor.
fn expr_contains_imag(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(name) if name == "I" => true,
    Expr::UnaryOp { operand, .. } => expr_contains_imag(operand),
    Expr::BinaryOp { left, right, .. } => {
      expr_contains_imag(left) || expr_contains_imag(right)
    }
    Expr::FunctionCall { args, .. } => args.iter().any(expr_contains_imag),
    Expr::List(items) => items.iter().any(expr_contains_imag),
    _ => false,
  }
}

/// `true` if `expr` contains an `Integrate[…]` call anywhere in its
/// structure. Used by `TimeConstrained` to detect the case where an
/// integration in the body short-circuits because Woxi gave up rather
/// than because it succeeded — wolframscript would actually exhaust the
/// time budget in that scenario, so we treat it as a timeout.
fn contains_unevaluated_integrate(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      name == "Integrate" || args.iter().any(contains_unevaluated_integrate)
    }
    Expr::CompoundExpr(items) => {
      items.iter().any(contains_unevaluated_integrate)
    }
    Expr::List(items) => items.iter().any(contains_unevaluated_integrate),
    Expr::BinaryOp { left, right, .. } => {
      contains_unevaluated_integrate(left)
        || contains_unevaluated_integrate(right)
    }
    Expr::UnaryOp { operand, .. } => contains_unevaluated_integrate(operand),
    _ => false,
  }
}

/// `true` if `expr` contains a machine-precision Real anywhere in its
/// structure. Used by `DirectedInfinity` to distinguish exact closed-form
/// directions like `(1 + 2 I)/Sqrt[5]` from inexact ones like
/// `1. + 2. I` — only the latter should be normalised numerically.
use crate::functions::math_ast::expr_contains_real;
