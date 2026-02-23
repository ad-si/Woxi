#[allow(unused_imports)]
use super::*;

pub fn dispatch_polynomial_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
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
    "Expand" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::expand_ast(args));
    }
    "Factor" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::factor_ast(args));
    }
    "FactorTerms" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::polynomial_ast::factor_terms_ast(args));
    }
    "FactorList" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::factor_list_ast(args));
    }
    "Simplify" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::simplify_ast(args));
    }
    "Coefficient" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::polynomial_ast::coefficient_ast(args));
    }
    "Exponent" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::polynomial_ast::exponent_ast(args));
    }
    "PolynomialQ" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::polynomial_ast::polynomial_q_ast(args));
    }
    "Solve" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::solve_ast(args));
    }
    "SolveAlways" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::solve_always_ast(args));
    }
    "Roots" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::roots_ast(args));
    }
    "ToRules" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::to_rules_ast(args));
    }
    "Eliminate" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::eliminate_ast(args));
    }
    "Reduce" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::polynomial_ast::reduce_ast(args));
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
    "Tuples" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::list_helpers_ast::tuples_ast(args));
    }
    "ExpandAll" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::expand_all_ast(args));
    }
    "Cancel" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::cancel_ast(args));
    }
    "Collect" if args.len() == 2 => {
      return Some(crate::functions::polynomial_ast::collect_ast(args));
    }
    "Together" if args.len() == 1 => {
      return Some(crate::functions::polynomial_ast::together_ast(args));
    }
    "Apart" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::polynomial_ast::apart_ast(args));
    }
    _ => {}
  }
  None
}
