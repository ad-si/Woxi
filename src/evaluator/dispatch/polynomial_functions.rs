#[allow(unused_imports)]
use super::*;

pub fn dispatch_polynomial_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "HornerForm" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::horner_form_ast(args));
    }
    "Distribute" if !args.is_empty() && args.len() <= 3 => {
      return Some(distribute_ast(args));
    }
    "PolynomialRemainder" if args.len() == 3 => {
      return Some(crate::functions::polynomial_ast::polynomial_remainder_ast(
        args,
      ));
    }
    "PolynomialQuotient" if args.len() == 3 => {
      return Some(crate::functions::polynomial_ast::polynomial_quotient_ast(
        args,
      ));
    }
    "PolynomialQuotientRemainder" if args.len() == 3 => {
      return Some(
        crate::functions::polynomial_ast::polynomial_quotient_remainder_ast(
          args,
        ),
      );
    }
    "PolynomialGCD" if args.len() >= 2 => {
      return Some(crate::functions::polynomial_ast::polynomial_gcd_ast(args));
    }
    "Expand" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::expand_ast(args));
    }
    "Decompose" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::decompose_ast(args));
    }
    "Factor" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::factor_ast(args));
    }
    "FactorSquareFree" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::factor_square_free_ast(
        args,
      ));
    }
    "FactorSquareFreeList" if args.len() == 1 => {
      return Some(
        crate::functions::polynomial_ast::factor_square_free_list_ast(args),
      );
    }
    "FactorTermsList" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::polynomial_ast::factor_terms_list_ast(
        args,
      ));
    }
    "FactorTerms" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::polynomial_ast::factor_terms_ast(args));
    }
    "FactorList" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::factor_list_ast(args));
    }
    "Simplify" if !args.is_empty() => {
      return Some(crate::functions::polynomial_ast::simplify_ast(args));
    }
    "FullSimplify" if args.len() <= 2 => {
      return Some(crate::functions::polynomial_ast::full_simplify_ast(args));
    }
    "Refine" if args.len() <= 2 => {
      return Some(crate::functions::polynomial_ast::refine_ast(args));
    }
    "Coefficient" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::polynomial_ast::coefficient_ast(args));
    }
    "MonomialList" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::monomial_list_ast(args));
    }
    "CoefficientRules" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::coefficient_rules_ast(
        args,
      ));
    }
    "CoefficientArrays" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::coefficient_arrays_ast(
        args,
      ));
    }
    "Exponent" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::polynomial_ast::exponent_ast(args));
    }
    "PolynomialQ" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::polynomial_ast::polynomial_q_ast(args));
    }
    "Solve" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::polynomial_ast::solve_ast(args));
    }
    "SolveValues" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::polynomial_ast::solve_values_ast(args));
    }
    "NSolve" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::nsolve_ast(args));
    }
    "SolveAlways" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::solve_always_ast(args));
    }
    "Roots" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::roots_ast(args));
    }
    "NRoots" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::nroots_ast(args));
    }
    "Root" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::root_ast(args));
    }
    "FunctionExpand" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::function_expand_ast(args));
    }
    "ToRadicals" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::to_radicals_ast(args));
    }
    "ToRules" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::to_rules_ast(args));
    }
    "Eliminate" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::eliminate_ast(args));
    }
    "Resultant" if args.len() == 3 => {
      return Some(crate::functions::polynomial_ast::resultant_ast(args));
    }
    "Discriminant" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::discriminant_ast(args));
    }
    "Cyclotomic" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::cyclotomic_ast(args));
    }
    "Reduce" if args.len() == 3 => {
      // Reduce[expr, vars, Modulus -> n]: brute-force enumerate
      // solutions in Z/nZ. Falls through to the generic reduce_ast
      // when the third arg isn't a Modulus rule.
      if let Some(result) = try_reduce_modulus(&args[0], &args[1], &args[2]) {
        return Some(Ok(result));
      }
      return Some(crate::functions::polynomial_ast::reduce_ast(args));
    }
    "Reduce" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::reduce_ast(args));
    }
    "FindInstance" if args.len() >= 2 && args.len() <= 4 => {
      return Some(crate::functions::polynomial_ast::find_instance_ast(args));
    }
    "FindRoot" if args.len() >= 2 => {
      // Catch "cannot evaluate numerically" errors and return unevaluated
      // with a FindRoot::nlnum warning (matches wolframscript).
      match crate::functions::polynomial_ast::find_root_ast(args) {
        Err(crate::InterpreterError::EvaluationError(msg))
          if msg.contains("cannot evaluate expression numerically") =>
        {
          crate::emit_message(
            "FindRoot::nlnum: The function value is not a number at the starting point.",
          );
          return Some(Ok(Expr::FunctionCall {
            name: "FindRoot".to_string(),
            args: args.to_vec().into(),
          }));
        }
        other => return Some(other),
      }
    }
    "FindMinimum" if args.len() >= 2 => {
      return Some(crate::functions::polynomial_ast::find_minimum_ast(
        args, false,
      ));
    }
    "FindMaximum" if args.len() >= 2 => {
      return Some(crate::functions::polynomial_ast::find_minimum_ast(
        args, true,
      ));
    }
    "Minimize" | "Maximize"
      if args.len() == 2
        && matches!(
          &args[0],
          Expr::List(items) if items.len() == 2
        )
        && matches!(
          &args[1],
          Expr::List(items) if items.len() == 2
            && items.iter().all(|v| matches!(v, Expr::Identifier(_)))
        ) =>
    {
      if let Some(result) = try_constrained_linear_disk_symbolic(name, args) {
        return Some(Ok(result));
      }
      return Some(crate::functions::polynomial_ast::minimize_ast(
        args,
        name == "Maximize",
      ));
    }
    "Minimize" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::polynomial_ast::minimize_ast(args, false));
    }
    "Maximize" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::polynomial_ast::minimize_ast(args, true));
    }
    "NMinimize" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::nminimize_ast(
        args, false,
      ));
    }
    "NMaximize" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::nminimize_ast(args, true));
    }
    // NMaxValue/NMinValue/NArgMax/NArgMin/FindArgMax/FindArgMin with a
    // constrained form: `{f, cons}, {v1, v2}`. Currently only the
    // closed-form case "linear objective on a unit-disk-like quadratic
    // constraint" is recognised; other inputs fall through.
    "NMaxValue" | "NMinValue" | "NArgMax" | "NArgMin" | "FindArgMax"
    | "FindArgMin"
      if args.len() == 2
        && matches!(
          &args[0],
          Expr::List(items) if items.len() == 2
        )
        && matches!(
          &args[1],
          Expr::List(items) if items.len() == 2
            && items.iter().all(|v| matches!(v, Expr::Identifier(_)))
        ) =>
    {
      if let Some(result) = try_constrained_linear_disk(name, args) {
        return Some(Ok(result));
      }
    }
    "FindArgMin" if args.len() == 2 => {
      // FindArgMin[f, x] => calls FindMinimum[f, {x, 0}] and extracts the arg
      if let Expr::Identifier(var) = &args[1] {
        let find_args = vec![
          args[0].clone(),
          Expr::List(
            vec![Expr::Identifier(var.clone()), Expr::Integer(0)].into(),
          ),
        ];
        if let Ok(Expr::List(ref result)) =
          crate::functions::polynomial_ast::find_minimum_ast(&find_args, false)
        {
          // Result is {value, {x -> arg}}
          if result.len() == 2
            && let Expr::List(rules) = &result[1]
          {
            let args_list: Vec<Expr> = rules
              .iter()
              .filter_map(|r| {
                if let Expr::Rule { replacement, .. } = r {
                  Some(replacement.as_ref().clone())
                } else {
                  None
                }
              })
              .collect();
            return Some(Ok(Expr::List(args_list.into())));
          }
        }
      }
    }
    "FindArgMax" if args.len() == 2 => {
      if let Expr::Identifier(var) = &args[1] {
        let find_args = vec![
          args[0].clone(),
          Expr::List(
            vec![Expr::Identifier(var.clone()), Expr::Integer(0)].into(),
          ),
        ];
        if let Ok(Expr::List(ref result)) =
          crate::functions::polynomial_ast::find_minimum_ast(&find_args, true)
          && result.len() == 2
          && let Expr::List(rules) = &result[1]
        {
          let args_list: Vec<Expr> = rules
            .iter()
            .filter_map(|r| {
              if let Expr::Rule { replacement, .. } = r {
                Some(replacement.as_ref().clone())
              } else {
                None
              }
            })
            .collect();
          return Some(Ok(Expr::List(args_list.into())));
        }
      }
    }
    "NMinValue" if args.len() == 2 => {
      // NMinValue[f, x] => calls FindMinimum[f, {x, 0}] and extracts the value
      if let Expr::Identifier(var) = &args[1] {
        let find_args = vec![
          args[0].clone(),
          Expr::List(
            vec![Expr::Identifier(var.clone()), Expr::Integer(0)].into(),
          ),
        ];
        if let Ok(Expr::List(ref result)) =
          crate::functions::polynomial_ast::find_minimum_ast(&find_args, false)
          && !result.is_empty()
        {
          return Some(Ok(result[0].clone()));
        }
      }
    }
    "NMaxValue" if args.len() == 2 => {
      if let Expr::Identifier(var) = &args[1] {
        let find_args = vec![
          args[0].clone(),
          Expr::List(
            vec![Expr::Identifier(var.clone()), Expr::Integer(0)].into(),
          ),
        ];
        if let Ok(Expr::List(ref result)) =
          crate::functions::polynomial_ast::find_minimum_ast(&find_args, true)
          && !result.is_empty()
        {
          return Some(Ok(result[0].clone()));
        }
      }
    }
    "Tuples" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::list_helpers_ast::tuples_ast(args));
    }
    "ExpandAll" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::expand_all_ast(args));
    }
    "ExpandNumerator" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::expand_numerator_ast(
        args,
      ));
    }
    "ExpandDenominator" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::expand_denominator_ast(
        args,
      ));
    }
    "Cancel" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::cancel_ast(args));
    }
    "Collect" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::polynomial_ast::collect_ast(args));
    }
    "Together" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::together_ast(args));
    }
    "Apart" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::polynomial_ast::apart_ast(args));
    }
    "PolynomialMod" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::polynomial_mod_ast(args));
    }
    "InterpolatingPolynomial" if args.len() == 2 => {
      return Some(
        crate::functions::polynomial_ast::interpolating_polynomial_ast(args),
      );
    }
    "MinimalPolynomial" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::minimal_polynomial_ast(
        args,
      ));
    }
    "SymmetricPolynomial" if args.len() == 2 => {
      // SymmetricPolynomial[k, {x1, x2, ..., xn}] = e_k(x1,...,xn)
      // The k-th elementary symmetric polynomial
      if let (Some(k), Expr::List(vars)) = (expr_to_i128(&args[0]), &args[1]) {
        let k = k as usize;
        let n = vars.len();
        if k == 0 {
          return Some(Ok(Expr::Integer(1)));
        }
        if k > n {
          return Some(Ok(Expr::Integer(0)));
        }
        // Generate all k-element subsets of vars and multiply elements in each
        let mut terms: Vec<Expr> = Vec::new();
        let mut indices: Vec<usize> = (0..k).collect();
        loop {
          // Build product of vars[indices[0]] * vars[indices[1]] * ...
          let mut product = vars[indices[0]].clone();
          for &idx in &indices[1..] {
            product = Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(product),
              right: Box::new(vars[idx].clone()),
            };
          }
          terms.push(product);
          // Next k-combination
          let mut i = k;
          loop {
            if i == 0 {
              // All combinations exhausted
              // Build sum of all terms
              let mut result = terms[0].clone();
              for term in &terms[1..] {
                result = Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Plus,
                  left: Box::new(result),
                  right: Box::new(term.clone()),
                };
              }
              return Some(evaluate_expr_to_expr(&result));
            }
            i -= 1;
            indices[i] += 1;
            if indices[i] <= n - k + i {
              // Fill remaining indices
              for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
              }
              break;
            }
          }
        }
      }
    }
    // FromCoefficientRules[{{exp} -> coeff, ...}, var]
    "FromCoefficientRules" if args.len() == 2 => {
      if let (Expr::List(rules), Expr::Identifier(var)) = (&args[0], &args[1]) {
        let mut terms: Vec<Expr> = Vec::new();
        for rule in rules {
          if let Expr::Rule {
            pattern,
            replacement,
          } = rule
          {
            {
              if let Expr::List(exps) = pattern.as_ref()
                && exps.len() == 1
              {
                let coeff = replacement.as_ref();
                let exp = &exps[0];
                let term = match exp {
                  Expr::Integer(0) => coeff.clone(),
                  Expr::Integer(1) => Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Times,
                    left: Box::new(coeff.clone()),
                    right: Box::new(Expr::Identifier(var.clone())),
                  },
                  _ => Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Times,
                    left: Box::new(coeff.clone()),
                    right: Box::new(Expr::BinaryOp {
                      op: crate::syntax::BinaryOperator::Power,
                      left: Box::new(Expr::Identifier(var.clone())),
                      right: Box::new(exp.clone()),
                    }),
                  },
                };
                terms.push(term);
              }
            }
          }
        }
        if !terms.is_empty() {
          let mut result = terms[0].clone();
          for term in &terms[1..] {
            result = Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(result),
              right: Box::new(term.clone()),
            };
          }
          return Some(crate::evaluator::evaluate_expr_to_expr(&result));
        }
      }
    }
    _ => {}
  }
  None
}

/// Closed-form solver for `Op[{f, cons}, {v1, v2}]` where:
/// `Reduce[equation, vars, Modulus -> n]` — enumerate all integer
/// solutions in `[0, n)^k`. The third argument must be a `Modulus -> n`
/// Rule with a positive integer modulus; otherwise returns None and the
/// caller falls through to the generic Reduce.
fn try_reduce_modulus(expr: &Expr, vars: &Expr, opt: &Expr) -> Option<Expr> {
  // Parse Modulus -> n option.
  let modulus = match opt {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      if !matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Modulus") {
        return None;
      }
      match replacement.as_ref() {
        Expr::Integer(n) if *n >= 2 => *n,
        _ => return None,
      }
    }
    _ => return None,
  };

  // Collect variable names: either a single Identifier or a list of them.
  let var_names: Vec<String> = match vars {
    Expr::Identifier(s) => vec![s.clone()],
    Expr::List(items) => {
      let mut v = Vec::with_capacity(items.len());
      for it in items.iter() {
        match it {
          Expr::Identifier(s) => v.push(s.clone()),
          _ => return None,
        }
      }
      v
    }
    _ => return None,
  };
  if var_names.is_empty() || var_names.len() > 4 {
    // Cap at 4 variables to avoid runaway 256^4 explosions.
    return None;
  }

  // The equation must be `lhs == rhs` (Equal). Anything else (And, Or,
  // inequality) is left for the generic reducer.
  let (lhs, rhs) = match expr {
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      (args[0].clone(), args[1].clone())
    }
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && matches!(operators[0], crate::syntax::ComparisonOp::Equal) =>
    {
      (operands[0].clone(), operands[1].clone())
    }
    _ => return None,
  };

  // Iterate over all assignments in [0, n)^k.
  let n = modulus as usize;
  let k = var_names.len();
  let mut indices = vec![0usize; k];
  let mut solutions: Vec<Vec<i128>> = Vec::new();
  loop {
    // Substitute each variable with its current integer value.
    let mut sub_lhs = lhs.clone();
    let mut sub_rhs = rhs.clone();
    for (i, name) in var_names.iter().enumerate() {
      sub_lhs = crate::syntax::substitute_variable(
        &sub_lhs,
        name,
        &Expr::Integer(indices[i] as i128),
      );
      sub_rhs = crate::syntax::substitute_variable(
        &sub_rhs,
        name,
        &Expr::Integer(indices[i] as i128),
      );
    }
    // Evaluate both sides; require integer results so we can take them mod n.
    let l_eval = crate::evaluator::evaluate_expr_to_expr(&sub_lhs).ok()?;
    let r_eval = crate::evaluator::evaluate_expr_to_expr(&sub_rhs).ok()?;
    let (Expr::Integer(li), Expr::Integer(ri)) = (&l_eval, &r_eval) else {
      return None;
    };
    let l_mod = li.rem_euclid(modulus);
    let r_mod = ri.rem_euclid(modulus);
    if l_mod == r_mod {
      solutions.push(indices.iter().map(|&x| x as i128).collect());
    }
    // Increment counters like an odometer: the LAST variable changes
    // fastest, matching wolframscript's lexicographic enumeration.
    let mut carry = true;
    for i in (0..k).rev() {
      if !carry {
        break;
      }
      indices[i] += 1;
      if indices[i] < n {
        carry = false;
      } else {
        indices[i] = 0;
      }
    }
    if carry {
      break;
    }
  }

  if solutions.is_empty() {
    return Some(Expr::Identifier("False".to_string()));
  }

  // Build (x == v1 && y == w1) || (x == v2 && y == w2) || ...
  let mut clauses: Vec<Expr> = Vec::with_capacity(solutions.len());
  for sol in &solutions {
    let eqs: Vec<Expr> = sol
      .iter()
      .zip(var_names.iter())
      .map(|(v, name)| {
        if k == 1 {
          Expr::Comparison {
            operands: vec![
              Expr::Identifier(name.clone()),
              Expr::Integer(*v),
            ],
            operators: vec![crate::syntax::ComparisonOp::Equal],
          }
        } else {
          Expr::Comparison {
            operands: vec![
              Expr::Identifier(name.clone()),
              Expr::Integer(*v),
            ],
            operators: vec![crate::syntax::ComparisonOp::Equal],
          }
        }
      })
      .collect();
    let clause = if eqs.len() == 1 {
      eqs.into_iter().next().unwrap()
    } else {
      Expr::FunctionCall {
        name: "And".to_string(),
        args: eqs.into(),
      }
    };
    clauses.push(clause);
  }
  let result = if clauses.len() == 1 {
    clauses.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Or".to_string(),
      args: clauses.into(),
    }
  };
  Some(result)
}

///   * f is linear in v1, v2 with numeric coefficients,
///   * cons is `v1^2 + v2^2 op R^2` (with op being `LessEqual`, `Less`,
///     `GreaterEqual`, `Greater`, or `Equal`),
/// returning the optimum value or argument depending on `name`. Linear
/// objective on a disk has the closed form `R * sqrt(a^2 + b^2)`.
fn try_constrained_linear_disk(name: &str, args: &[Expr]) -> Option<Expr> {
  let (f, cons) = match &args[0] {
    Expr::List(items) if items.len() == 2 => (&items[0], &items[1]),
    _ => return None,
  };
  let (v1, v2) = match &args[1] {
    Expr::List(items) if items.len() == 2 => match (&items[0], &items[1]) {
      (Expr::Identifier(a), Expr::Identifier(b)) => (a.clone(), b.clone()),
      _ => return None,
    },
    _ => return None,
  };

  // Linearity check: f must have zero second partial derivatives in v1, v2.
  // Coefficients: a = ∂f/∂v1|origin, b = ∂f/∂v2|origin.
  let zero_at = |expr: &Expr| -> Option<f64> {
    let with_v1 =
      crate::syntax::substitute_variable(expr, &v1, &Expr::Integer(0));
    let with_v2 =
      crate::syntax::substitute_variable(&with_v1, &v2, &Expr::Integer(0));
    let evaluated = crate::evaluator::evaluate_expr_to_expr(&with_v2).ok()?;
    crate::functions::math_ast::expr_to_f64(&evaluated)
  };
  let diff = |expr: &Expr, var: &str| -> Option<Expr> {
    crate::functions::calculus_ast::differentiate_expr(expr, var).ok()
  };
  let df_dv1 = diff(f, &v1)?;
  let df_dv2 = diff(f, &v2)?;
  let d2f_dv1 = diff(&df_dv1, &v1)?;
  let d2f_dv2 = diff(&df_dv2, &v2)?;
  let d2f_mixed = diff(&df_dv1, &v2)?;
  if zero_at(&d2f_dv1)? != 0.0
    || zero_at(&d2f_dv2)? != 0.0
    || zero_at(&d2f_mixed)? != 0.0
  {
    return None;
  }
  let a = zero_at(&df_dv1)?;
  let b = zero_at(&df_dv2)?;
  let c0 = zero_at(f)?;

  // Constraint must be `g op c` where g = v1^2 + v2^2 (plus possible
  // linear/constant term we leave out for now) and c is a real number.
  let (g_expr, c_val) = match cons {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && matches!(
        operators[0],
        crate::syntax::ComparisonOp::LessEqual
          | crate::syntax::ComparisonOp::Less
          | crate::syntax::ComparisonOp::GreaterEqual
          | crate::syntax::ComparisonOp::Greater
          | crate::syntax::ComparisonOp::Equal
      ) =>
    {
      let c = crate::functions::math_ast::expr_to_f64(&operands[1])?;
      (operands[0].clone(), c)
    }
    Expr::FunctionCall {
      name: cn,
      args: cargs,
    } if matches!(
      cn.as_str(),
      "LessEqual" | "Less" | "GreaterEqual" | "Greater" | "Equal"
    ) && cargs.len() == 2 =>
    {
      let c = crate::functions::math_ast::expr_to_f64(&cargs[1])?;
      (cargs[0].clone(), c)
    }
    _ => return None,
  };
  // Quadratic form: ∂²g/∂v1² = 2, ∂²g/∂v2² = 2, mixed = 0,
  // and g(0,0) = 0, ∂g/∂v1(0,0) = 0, ∂g/∂v2(0,0) = 0.
  let dg_dv1 = diff(&g_expr, &v1)?;
  let dg_dv2 = diff(&g_expr, &v2)?;
  let d2g_dv1 = diff(&dg_dv1, &v1)?;
  let d2g_dv2 = diff(&dg_dv2, &v2)?;
  let d2g_mixed = diff(&dg_dv1, &v2)?;
  if zero_at(&d2g_dv1)? != 2.0
    || zero_at(&d2g_dv2)? != 2.0
    || zero_at(&d2g_mixed)? != 0.0
    || zero_at(&dg_dv1)? != 0.0
    || zero_at(&dg_dv2)? != 0.0
    || zero_at(&g_expr)? != 0.0
  {
    return None;
  }

  // c_val is the upper (or lower) bound on g, equal to R^2.
  if c_val < 0.0 {
    return None;
  }
  let radius = c_val.sqrt();
  let norm = (a * a + b * b).sqrt();
  if norm == 0.0 {
    // Constant objective: every feasible point is optimal; return c0 or {0, 0}.
    return Some(match name {
      "NMaxValue" | "NMinValue" => Expr::Real(c0),
      _ => Expr::List(vec![Expr::Real(0.0), Expr::Real(0.0)].into()),
    });
  }
  let max_val = c0 + radius * norm;
  let min_val = c0 - radius * norm;
  let max_pt = (radius * a / norm, radius * b / norm);
  let min_pt = (-max_pt.0, -max_pt.1);

  let result = match name {
    "NMaxValue" => Expr::Real(max_val),
    "NMinValue" => Expr::Real(min_val),
    "NArgMax" | "FindArgMax" => {
      Expr::List(vec![Expr::Real(max_pt.0), Expr::Real(max_pt.1)].into())
    }
    "NArgMin" | "FindArgMin" => {
      Expr::List(vec![Expr::Real(min_pt.0), Expr::Real(min_pt.1)].into())
    }
    _ => return None,
  };
  Some(result)
}

/// Symbolic closed form for Minimize/Maximize on a linear objective with a
/// disk constraint: `{a*v1 + b*v2 + c0, v1^2 + v2^2 <= R^2}`. Returns
/// `{value, {v1 -> px, v2 -> py}}` evaluated through the AST evaluator so
/// perfect-square radicals collapse to rationals.
fn try_constrained_linear_disk_symbolic(
  name: &str,
  args: &[Expr],
) -> Option<Expr> {
  let (f, cons) = match &args[0] {
    Expr::List(items) if items.len() == 2 => (&items[0], &items[1]),
    _ => return None,
  };
  let (v1, v2) = match &args[1] {
    Expr::List(items) if items.len() == 2 => match (&items[0], &items[1]) {
      (Expr::Identifier(a), Expr::Identifier(b)) => (a.clone(), b.clone()),
      _ => return None,
    },
    _ => return None,
  };

  let diff = |expr: &Expr, var: &str| -> Option<Expr> {
    crate::functions::calculus_ast::differentiate_expr(expr, var).ok()
  };
  let zero_at = |expr: &Expr| -> Option<Expr> {
    let with_v1 =
      crate::syntax::substitute_variable(expr, &v1, &Expr::Integer(0));
    let with_v2 =
      crate::syntax::substitute_variable(&with_v1, &v2, &Expr::Integer(0));
    crate::evaluator::evaluate_expr_to_expr(&with_v2).ok()
  };

  // Linearity check on f.
  let df_dv1 = diff(f, &v1)?;
  let df_dv2 = diff(f, &v2)?;
  let d2_11 = diff(&df_dv1, &v1)?;
  let d2_22 = diff(&df_dv2, &v2)?;
  let d2_12 = diff(&df_dv1, &v2)?;
  let is_zero = |e: &Expr| matches!(e, Expr::Integer(0));
  if !is_zero(&zero_at(&d2_11)?)
    || !is_zero(&zero_at(&d2_22)?)
    || !is_zero(&zero_at(&d2_12)?)
  {
    return None;
  }
  let a_expr = zero_at(&df_dv1)?;
  let b_expr = zero_at(&df_dv2)?;
  let c0_expr = zero_at(f)?;

  // Constraint: relational form with rhs equal to R^2.
  let (g_expr, rhs) = match cons {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && operators.len() == 1
      && matches!(
        operators[0],
        crate::syntax::ComparisonOp::LessEqual
          | crate::syntax::ComparisonOp::Less
          | crate::syntax::ComparisonOp::GreaterEqual
          | crate::syntax::ComparisonOp::Greater
          | crate::syntax::ComparisonOp::Equal
      ) =>
    {
      (operands[0].clone(), operands[1].clone())
    }
    Expr::FunctionCall {
      name: cn,
      args: cargs,
    } if matches!(
      cn.as_str(),
      "LessEqual" | "Less" | "GreaterEqual" | "Greater" | "Equal"
    ) && cargs.len() == 2 =>
    {
      (cargs[0].clone(), cargs[1].clone())
    }
    _ => return None,
  };

  // Standard x^2+y^2 form check.
  let dg_dv1 = diff(&g_expr, &v1)?;
  let dg_dv2 = diff(&g_expr, &v2)?;
  let d2g_11 = diff(&dg_dv1, &v1)?;
  let d2g_22 = diff(&dg_dv2, &v2)?;
  let d2g_12 = diff(&dg_dv1, &v2)?;
  let is_two = |e: &Expr| matches!(e, Expr::Integer(2));
  if !is_two(&zero_at(&d2g_11)?)
    || !is_two(&zero_at(&d2g_22)?)
    || !is_zero(&zero_at(&d2g_12)?)
    || !is_zero(&zero_at(&dg_dv1)?)
    || !is_zero(&zero_at(&dg_dv2)?)
    || !is_zero(&zero_at(&g_expr)?)
  {
    return None;
  }
  let r2_expr = crate::evaluator::evaluate_expr_to_expr(&rhs).ok()?;

  let eval = |e: Expr| -> Expr {
    crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or(Expr::Integer(0))
  };
  let plus = |x: Expr, y: Expr| -> Expr {
    eval(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(x),
      right: Box::new(y),
    })
  };
  let minus = |x: Expr, y: Expr| -> Expr {
    eval(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(x),
      right: Box::new(y),
    })
  };
  let times = |x: Expr, y: Expr| -> Expr {
    eval(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(x),
      right: Box::new(y),
    })
  };
  let div = |x: Expr, y: Expr| -> Expr {
    eval(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(x),
      right: Box::new(y),
    })
  };
  let neg = |x: Expr| -> Expr {
    eval(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(Expr::Integer(0)),
      right: Box::new(x),
    })
  };
  let square = |x: Expr| -> Expr { times(x.clone(), x) };
  let sqrt_e = |e: Expr| -> Expr {
    eval(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![e].into(),
    })
  };

  let radius = sqrt_e(r2_expr);
  let norm_sq = plus(square(a_expr.clone()), square(b_expr.clone()));
  let norm = sqrt_e(norm_sq);
  if matches!(&norm, Expr::Integer(0)) {
    return None;
  }

  let want_max = name == "Maximize";
  let r_times_norm = times(radius.clone(), norm.clone());
  let value = if want_max {
    plus(c0_expr.clone(), r_times_norm)
  } else {
    minus(c0_expr.clone(), r_times_norm)
  };
  let r_over_norm = div(radius, norm);
  let x_coord = times(r_over_norm.clone(), a_expr);
  let y_coord = times(r_over_norm, b_expr);
  let (px, py) = if want_max {
    (x_coord, y_coord)
  } else {
    (neg(x_coord), neg(y_coord))
  };

  let rule = |var: &str, val: Expr| -> Expr {
    Expr::FunctionCall {
      name: "Rule".to_string(),
      args: vec![Expr::Identifier(var.to_string()), val].into(),
    }
  };
  let argmap = Expr::List(vec![rule(&v1, px), rule(&v2, py)].into());
  Some(Expr::List(vec![value, argmap].into()))
}
