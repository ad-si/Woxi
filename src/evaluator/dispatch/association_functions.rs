#[allow(unused_imports)]
use super::*;

pub fn dispatch_association_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Keys" if args.len() == 1 => {
      return Some(crate::functions::association_ast::keys_ast(args));
    }
    "Values" if args.len() == 1 => {
      return Some(crate::functions::association_ast::values_ast(args));
    }
    "KeyDropFrom" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_drop_from_ast(args));
    }
    "KeyExistsQ" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_exists_q_ast(args));
    }
    "Lookup" if args.len() >= 2 => {
      return Some(crate::functions::association_ast::lookup_ast(args));
    }
    "KeySort" if args.len() == 1 => {
      return Some(crate::functions::association_ast::key_sort_ast(args));
    }
    "KeyValueMap" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_value_map_ast(args));
    }
    "FilterRules" if args.len() == 2 => {
      return Some(filter_rules_ast(&args[0], &args[1]));
    }
    "Dataset" if !args.is_empty() => {
      return Some(Ok(crate::functions::dataset_ast::dataset_ast(args)));
    }
    "AssociationMap" if args.len() == 2 => {
      return Some(crate::functions::association_ast::association_map_ast(
        args,
      ));
    }
    "AssociationThread" if args.len() == 2 => {
      return Some(crate::functions::association_ast::association_thread_ast(
        args,
      ));
    }
    "Merge" if args.len() == 2 => {
      return Some(crate::functions::association_ast::merge_ast(args));
    }
    "KeyMap" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_map_ast(args));
    }
    "KeySelect" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_select_ast(args));
    }
    "KeyTake" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_take_ast(args));
    }
    "KeyDrop" if args.len() == 2 => {
      return Some(crate::functions::association_ast::key_drop_ast(args));
    }
    _ => {}
  }
  None
}
