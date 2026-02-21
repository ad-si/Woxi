#[allow(unused_imports)]
use super::*;

pub fn dispatch_quantity_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Quantity" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::quantity_ast::quantity_ast(args));
    }
    "QuantityMagnitude" if args.len() == 1 || args.len() == 2 => {
      return Some(crate::functions::quantity_ast::quantity_magnitude_ast(
        args,
      ));
    }
    "QuantityUnit" if args.len() == 1 => {
      return Some(crate::functions::quantity_ast::quantity_unit_ast(args));
    }
    "QuantityQ" if args.len() == 1 => {
      return Some(crate::functions::quantity_ast::quantity_q_ast(args));
    }
    "CompatibleUnitQ" => {
      return Some(crate::functions::quantity_ast::compatible_unit_q_ast(args));
    }
    "UnitConvert" => {
      return Some(crate::functions::quantity_ast::unit_convert_ast(args));
    }
    _ => {}
  }
  None
}
