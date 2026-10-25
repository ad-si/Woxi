//! Complex-valued ODE systems for `NDSolve`.
//!
//! The integrator works on real states, so a system whose equations carry
//! `I` — the amplitude equations of a driven quantum system,
//!
//! ```text
//! {c1'[t] == -I (ϵ c2[t] Cos[ω t]), c2'[t] == …, c1[0] == 1, c2[0] == 0}
//! ```
//!
//! — is solved as the real system of its parts: every unknown `f` becomes
//! `fRe + I fIm`, each equation splits into its real and imaginary parts
//! (`ComplexExpand`, every unknown being real), and each initial condition
//! `f[x0] == v` into `fRe[x0] == Re[v]` and `fIm[x0] == Im[v]`. The
//! solution recombines the two interpolations as `fRe[x] + I fIm[x]`.

use super::*;

pub(super) fn ndsolve_complex_split(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  if !mentions_imaginary(&args[0]) {
    return Ok(None);
  }
  let Expr::List(domain) = &args[2] else {
    return Ok(None);
  };
  let Some(Expr::Identifier(x_name)) = domain.first() else {
    return Ok(None);
  };

  // The unknowns, as `f` or `f[x]`.
  let dep_items: Vec<&Expr> = match &args[1] {
    Expr::List(items) => items.iter().collect(),
    other => vec![other],
  };
  let mut unknowns: Vec<(String, bool)> = Vec::new();
  for item in dep_items {
    match item {
      Expr::Identifier(name) => unknowns.push((name.clone(), true)),
      Expr::FunctionCall { name, args: fargs } if matches!(fargs.as_slice(), [Expr::Identifier(a)] if a == x_name) =>
      {
        unknowns.push((name.clone(), false));
      }
      _ => return Ok(None),
    }
  }
  let part_name = |f: &str, part: &str| format!("{f}${part}");

  let mut eq_items = Vec::new();
  flatten_eq_list(&args[0], &mut eq_items);

  // `f -> Function[{x}, fRe[x] + I fIm[x]]`, which also turns a
  // derivative of `f` into the same combination of the parts' derivatives.
  let slot = Expr::Identifier("NDSolve$z".to_string());
  let split_rules = Expr::List(
    unknowns
      .iter()
      .map(|(f, _)| {
        rule_expr(
          Expr::Identifier(f.clone()),
          Expr::NamedFunction {
            params: vec!["NDSolve$z".to_string()],
            body: Box::new(plus2(
              call(&part_name(f, "re"), vec![slot.clone()]),
              times2(
                Expr::Identifier("I".to_string()),
                call(&part_name(f, "im"), vec![slot.clone()]),
              ),
            )),
            bracketed: true,
          },
        )
      })
      .collect(),
  );
  let part_of = |part: &str, e: &Expr| -> Result<Expr, InterpreterError> {
    crate::evaluator::evaluate_expr_to_expr(&call(
      "ComplexExpand",
      vec![call(part, vec![e.clone()])],
    ))
  };

  let mut split_eqs: Vec<Expr> = Vec::new();
  for eq in &eq_items {
    let Some((lhs, rhs)) = as_equal_pair(eq) else {
      // Events, options and the like are not split.
      return Ok(None);
    };
    // An initial condition `f[x0] == v` / `f'[x0] == v` keeps its shape.
    if let Some((f, _)) = unknowns
      .iter()
      .find(|(f, _)| is_initial_condition(eq, f, x_name))
    {
      let value = crate::evaluator::evaluate_expr_to_expr(rhs)?;
      for part in ["re", "im"] {
        let renamed = crate::syntax::substitute_variable(
          lhs,
          f,
          &Expr::Identifier(part_name(f, part)),
        );
        let value = crate::evaluator::evaluate_expr_to_expr(&call(
          if part == "re" { "Re" } else { "Im" },
          vec![value.clone()],
        ))?;
        split_eqs.push(Expr::Comparison {
          operands: vec![renamed, value],
          operators: vec![ComparisonOp::Equal],
        });
      }
      continue;
    }
    let difference = crate::evaluator::evaluate_expr_to_expr(&call(
      "ReplaceAll",
      vec![minus2(lhs.clone(), rhs.clone()), split_rules.clone()],
    ))?;
    for part in ["Re", "Im"] {
      let p = part_of(part, &difference)?;
      if matches!(p, Expr::Integer(0)) {
        continue;
      }
      split_eqs.push(Expr::Comparison {
        operands: vec![p, Expr::Integer(0)],
        operators: vec![ComparisonOp::Equal],
      });
    }
  }

  let part_spec = |f: &str, function_form: bool| -> Expr {
    if function_form {
      Expr::Identifier(f.to_string())
    } else {
      call(f, vec![Expr::Identifier(x_name.clone())])
    }
  };
  let new_unknowns: Vec<Expr> = unknowns
    .iter()
    .flat_map(|(f, function_form)| {
      [
        part_spec(&part_name(f, "re"), *function_form),
        part_spec(&part_name(f, "im"), *function_form),
      ]
    })
    .collect();
  let split_args = vec![
    Expr::List(split_eqs.into()),
    Expr::List(new_unknowns.into()),
    args[2].clone(),
  ];
  let Some(Expr::List(ref solutions)) = ndsolve_system(&split_args, None)?
  else {
    return Ok(None);
  };

  // Recombine every unknown from its parts.
  let mut combined = Vec::with_capacity(solutions.len());
  for solution in solutions {
    let Expr::List(rules) = solution else {
      return Ok(None);
    };
    let value_of = |name: &str| -> Option<Expr> {
      rules.iter().find_map(|r| match r {
        Expr::Rule {
          pattern,
          replacement,
        } => match pattern.as_ref() {
          Expr::Identifier(n) if n == name => {
            Some(replacement.as_ref().clone())
          }
          Expr::FunctionCall { name: n, .. } if n == name => {
            Some(replacement.as_ref().clone())
          }
          _ => None,
        },
        _ => None,
      })
    };
    let mut new_rules = Vec::with_capacity(unknowns.len());
    for (f, function_form) in &unknowns {
      let (Some(re), Some(im)) =
        (value_of(&part_name(f, "re")), value_of(&part_name(f, "im")))
      else {
        return Ok(None);
      };
      if *function_form {
        // `re` and `im` are interpolating functions of one argument.
        let z = Expr::Identifier("NDSolve$z".to_string());
        let apply = |g: Expr| Expr::CurriedCall {
          func: Box::new(g),
          args: vec![z.clone()],
        };
        new_rules.push(rule_expr(
          Expr::Identifier(f.clone()),
          Expr::NamedFunction {
            params: vec!["NDSolve$z".to_string()],
            body: Box::new(plus2(
              apply(re),
              times2(Expr::Identifier("I".to_string()), apply(im)),
            )),
            bracketed: true,
          },
        ));
      } else {
        // `re` and `im` are already applied to the variable.
        new_rules.push(rule_expr(
          part_spec(f, false),
          plus2(re, times2(Expr::Identifier("I".to_string()), im)),
        ));
      }
    }
    combined.push(Expr::List(new_rules.into()));
  }
  Ok(Some(Expr::List(combined.into())))
}

/// Whether `expr` mentions `I` or a complex number anywhere.
fn mentions_imaginary(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(s) | Expr::Constant(s) => s == "I",
    Expr::FunctionCall { name, .. } if name == "Complex" => true,
    _ => crate::syntax::expr_children(expr)
      .into_iter()
      .any(mentions_imaginary),
  }
}
