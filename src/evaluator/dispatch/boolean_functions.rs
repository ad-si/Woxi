#[allow(unused_imports)]
use super::*;

pub fn dispatch_boolean_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "And" if args.len() >= 2 => {
      return Some(crate::functions::boolean_ast::and_ast(args));
    }
    "Or" if args.len() >= 2 => {
      return Some(crate::functions::boolean_ast::or_ast(args));
    }
    "Not" => {
      if args.len() == 1 {
        return Some(crate::functions::boolean_ast::not_ast(args));
      } else {
        println!(
          "\nNot::argx: Not called with {} arguments; 1 argument is expected.",
          args.len()
        );
        use std::io::{self, Write};
        io::stdout().flush().ok();
      }
    }
    "Xor" if !args.is_empty() => {
      return Some(crate::functions::boolean_ast::xor_ast(args));
    }
    "Equivalent" if args.len() >= 2 => {
      return Some(crate::functions::boolean_ast::equivalent_ast(args));
    }
    "Return" => {
      let val = if args.is_empty() {
        Expr::Identifier("Null".to_string())
      } else {
        args[0].clone()
      };
      return Some(Err(InterpreterError::ReturnValue(Box::new(val))));
    }
    "SameQ" => {
      return Some(crate::functions::boolean_ast::same_q_ast(args));
    }
    "UnsameQ" => {
      return Some(crate::functions::boolean_ast::unsame_q_ast(args));
    }
    "Which" if args.len() >= 2 && args.len().is_multiple_of(2) => {
      return Some(crate::functions::boolean_ast::which_ast(args));
    }
    "While" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::boolean_ast::while_ast(args));
    }
    "Equal" => {
      return Some(crate::functions::boolean_ast::equal_ast(args));
    }
    "Unequal" if args.len() >= 2 => {
      return Some(crate::functions::boolean_ast::unequal_ast(args));
    }
    "Less" if args.len() >= 2 => {
      return Some(crate::functions::boolean_ast::less_ast(args));
    }
    "Greater" if args.len() >= 2 => {
      return Some(crate::functions::boolean_ast::greater_ast(args));
    }
    "LessEqual" if args.len() >= 2 => {
      return Some(crate::functions::boolean_ast::less_equal_ast(args));
    }
    "GreaterEqual" if args.len() >= 2 => {
      return Some(crate::functions::boolean_ast::greater_equal_ast(args));
    }
    "Boole" if args.len() == 1 => {
      return Some(crate::functions::boolean_ast::boole_ast(args));
    }
    "TrueQ" if args.len() == 1 => {
      return Some(crate::functions::boolean_ast::true_q_ast(args));
    }
    "Implies" if args.len() == 2 => {
      return Some(crate::functions::boolean_ast::implies_ast(args));
    }
    "Nand" if args.len() >= 2 => {
      return Some(crate::functions::boolean_ast::nand_ast(args));
    }
    "Nor" if args.len() >= 2 => {
      return Some(crate::functions::boolean_ast::nor_ast(args));
    }
    "LogicalExpand" if args.len() == 1 => {
      return Some(crate::functions::boolean_ast::logical_expand_ast(args));
    }
    _ => {}
  }
  None
}
