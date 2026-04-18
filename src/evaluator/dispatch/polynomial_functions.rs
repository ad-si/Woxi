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
    "Simplify" if args.len() <= 2 => {
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
    "Exponent" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::polynomial_ast::exponent_ast(args));
    }
    "PolynomialQ" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::polynomial_ast::polynomial_q_ast(args));
    }
    "Solve" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::polynomial_ast::solve_ast(args));
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
    "Reduce" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::polynomial_ast::reduce_ast(args));
    }
    "FindInstance" if args.len() >= 2 && args.len() <= 4 => {
      return Some(crate::functions::polynomial_ast::find_instance_ast(args));
    }
    "FindRoot" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::find_root_ast(args));
    }
    "FindMinimum" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::find_minimum_ast(
        args, false,
      ));
    }
    "FindMaximum" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::find_minimum_ast(
        args, true,
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
    "FindArgMin" if args.len() == 2 => {
      // FindArgMin[f, x] => calls FindMinimum[f, {x, 0}] and extracts the arg
      if let Expr::Identifier(var) = &args[1] {
        let find_args = vec![
          args[0].clone(),
          Expr::List(vec![Expr::Identifier(var.clone()), Expr::Integer(0)]),
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
            return Some(Ok(Expr::List(args_list)));
          }
        }
      }
    }
    "FindArgMax" if args.len() == 2 => {
      if let Expr::Identifier(var) = &args[1] {
        let find_args = vec![
          args[0].clone(),
          Expr::List(vec![Expr::Identifier(var.clone()), Expr::Integer(0)]),
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
          return Some(Ok(Expr::List(args_list)));
        }
      }
    }
    "NMinValue" if args.len() == 2 => {
      // NMinValue[f, x] => calls FindMinimum[f, {x, 0}] and extracts the value
      if let Expr::Identifier(var) = &args[1] {
        let find_args = vec![
          args[0].clone(),
          Expr::List(vec![Expr::Identifier(var.clone()), Expr::Integer(0)]),
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
          Expr::List(vec![Expr::Identifier(var.clone()), Expr::Integer(0)]),
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
    "ExpandAll" if args.len() == 1 => {
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
