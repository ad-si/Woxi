#[allow(unused_imports)]
use super::*;

pub fn dispatch_predicate_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "NumberQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::number_q_ast(args));
    }
    "RealValuedNumberQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::real_valued_number_q_ast(
        args,
      ));
    }
    "Element" if args.len() == 2 => {
      return Some(element_ast(&args[0], &args[1]));
    }
    "IntegerQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::integer_q_ast(args));
    }
    "BooleanQ" if args.len() == 1 => {
      return Some(Ok(match &args[0] {
        Expr::Identifier(name) if name == "True" || name == "False" => {
          Expr::Identifier("True".to_string())
        }
        _ => Expr::Identifier("False".to_string()),
      }));
    }
    // SymbolQ is not a standard Wolfram built-in (it's from GeneralUtilities package),
    // so return unevaluated to match Wolfram behavior
    // "SymbolQ" if args.len() == 1 => { ... }
    "Boole" if args.len() == 1 => {
      return Some(Ok(match &args[0] {
        Expr::Identifier(name) if name == "True" => Expr::Integer(1),
        Expr::Identifier(name) if name == "False" => Expr::Integer(0),
        _ => Expr::FunctionCall {
          name: "Boole".to_string(),
          args: args.to_vec(),
        },
      }));
    }
    "DigitQ" if args.len() == 1 => {
      return Some(Ok(match &args[0] {
        Expr::String(s) => {
          if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) {
            Expr::Identifier("True".to_string())
          } else {
            Expr::Identifier("False".to_string())
          }
        }
        _ => Expr::Identifier("False".to_string()),
      }));
    }
    "LetterQ" if args.len() == 1 => {
      return Some(Ok(match &args[0] {
        Expr::String(s) => {
          if !s.is_empty() && s.chars().all(|c| c.is_alphabetic()) {
            Expr::Identifier("True".to_string())
          } else {
            Expr::Identifier("False".to_string())
          }
        }
        _ => Expr::Identifier("False".to_string()),
      }));
    }
    "Precision" if args.len() == 1 => {
      return Some(crate::functions::math_ast::precision_ast(args));
    }
    "Accuracy" if args.len() == 1 => {
      return Some(crate::functions::math_ast::accuracy_ast(args));
    }
    "O" if args.len() == 1 || args.len() == 2 => {
      // O[x] -> SeriesData[x, 0, {}, 1, 1, 1]
      // O[x, x0] -> SeriesData[x, x0, {}, 1, 1, 1]
      let var = args[0].clone();
      let center = if args.len() == 2 {
        args[1].clone()
      } else {
        Expr::Integer(0)
      };
      return Some(Ok(Expr::FunctionCall {
        name: "SeriesData".to_string(),
        args: vec![
          var,
          center,
          Expr::List(vec![]),
          Expr::Integer(1),
          Expr::Integer(1),
          Expr::Integer(1),
        ],
      }));
    }
    "EvenQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::even_q_ast(args));
    }
    "LeapYearQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::leap_year_q_ast(args));
    }
    "OddQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::odd_q_ast(args));
    }
    "PalindromeQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::palindrome_q_ast(args));
    }
    "SquareFreeQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::square_free_q_ast(args));
    }
    "ListQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::list_q_ast(args));
    }
    "StringQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::string_q_ast(args));
    }
    // Symbol["name"] - Convert string to symbol identifier
    "Symbol" if args.len() == 1 => {
      if let Expr::String(name) = &args[0] {
        return Some(Ok(Expr::Identifier(name.clone())));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Symbol".to_string(),
        args: args.to_vec(),
      }));
    }
    // SymbolName[sym] - Get the name of a symbol as a string
    "SymbolName" if args.len() == 1 => {
      if let Expr::Identifier(name) = &args[0] {
        return Some(Ok(Expr::String(name.clone())));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "SymbolName".to_string(),
        args: args.to_vec(),
      }));
    }
    // Unique[] - generate a unique symbol $nnn
    // Unique[x] - generate a unique symbol x$nnn
    // Unique["xxx"] - generate a unique symbol xxxnnn
    // Unique[{x, y, ...}] - generate list of unique symbols
    "Unique" if args.is_empty() => {
      let sym_name = crate::functions::scoping::unique_symbol("");
      // For Unique[], format is $nnn (just $counter)
      return Some(Ok(Expr::Identifier(sym_name)));
    }
    "Unique" if args.len() == 1 => {
      match &args[0] {
        Expr::Identifier(name) => {
          let sym_name = crate::functions::scoping::unique_symbol(name);
          return Some(Ok(Expr::Identifier(sym_name)));
        }
        Expr::String(name) => {
          // For strings, use sequential numbering without $
          let sym_name =
            crate::functions::scoping::unique_symbol_from_string(name);
          return Some(Ok(Expr::Identifier(sym_name)));
        }
        Expr::List(items) => {
          let mut result = Vec::new();
          for item in items {
            match item {
              Expr::Identifier(name) => {
                let sym_name = crate::functions::scoping::unique_symbol(name);
                result.push(Expr::Identifier(sym_name));
              }
              Expr::String(name) => {
                let sym_name =
                  crate::functions::scoping::unique_symbol_from_string(name);
                result.push(Expr::Identifier(sym_name));
              }
              _ => {
                return Some(Ok(Expr::FunctionCall {
                  name: "Unique".to_string(),
                  args: args.to_vec(),
                }));
              }
            }
          }
          return Some(Ok(Expr::List(result)));
        }
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Unique".to_string(),
            args: args.to_vec(),
          }));
        }
      }
    }
    "AtomQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::atom_q_ast(args));
    }
    "NumericQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::numeric_q_ast(args));
    }
    "ExactNumberQ" if args.len() == 1 => {
      let is_exact = match &args[0] {
        Expr::Integer(_) | Expr::BigInteger(_) => true,
        Expr::FunctionCall {
          name: fname,
          args: rargs,
        } if fname == "Rational"
          && rargs.len() == 2
          && matches!(rargs[0], Expr::Integer(_))
          && matches!(rargs[1], Expr::Integer(_)) =>
        {
          true
        }
        _ => false,
      };
      return Some(Ok(Expr::Identifier(
        if is_exact { "True" } else { "False" }.to_string(),
      )));
    }
    "InexactNumberQ" if args.len() == 1 => {
      let is_inexact = matches!(&args[0], Expr::Real(_) | Expr::BigFloat(_, _));
      return Some(Ok(Expr::Identifier(
        if is_inexact { "True" } else { "False" }.to_string(),
      )));
    }
    "LevelQ" if args.len() == 1 => {
      let is_valid = match &args[0] {
        Expr::Integer(_) => true,
        Expr::Identifier(name) if name == "Infinity" => true,
        Expr::List(items) => items.iter().all(|item| {
          matches!(item, Expr::Integer(_))
            || matches!(item, Expr::Identifier(n) if n == "Infinity")
        }),
        _ => false,
      };
      return Some(Ok(Expr::Identifier(
        if is_valid { "True" } else { "False" }.to_string(),
      )));
    }
    "Positive" | "PositiveQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::positive_q_ast(args));
    }
    "Negative" | "NegativeQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::negative_q_ast(args));
    }
    "NonPositive" | "NonPositiveQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::non_positive_q_ast(args));
    }
    "NonNegative" | "NonNegativeQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::non_negative_q_ast(args));
    }
    "PrimeQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::prime_q_ast(args));
    }
    "CompositeQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::composite_q_ast(args));
    }
    "PrimePowerQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::prime_power_q_ast(args));
    }
    "AssociationQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::association_q_ast(args));
    }
    "Between" if args.len() == 2 => {
      use crate::functions::math_ast::try_eval_to_f64;
      // Normalize arg order: Between[x, range] or Between[range, x] (from operator form)
      let (x, range_expr) = if matches!(&args[0], Expr::List(_))
        && !matches!(&args[1], Expr::List(_))
      {
        (&args[1], &args[0])
      } else {
        (&args[0], &args[1])
      };
      // Between[x, {min, max}] or Between[x, {{min1, max1}, {min2, max2}, ...}]
      if let Expr::List(range) = range_expr {
        // Check if it's a list of ranges (all elements are lists)
        let is_list_of_ranges =
          !range.is_empty() && range.iter().all(|r| matches!(r, Expr::List(_)));
        if range.len() == 2 && !is_list_of_ranges {
          // Single range: Between[x, {min, max}]
          if let (Some(xv), Some(lo), Some(hi)) = (
            try_eval_to_f64(x),
            try_eval_to_f64(&range[0]),
            try_eval_to_f64(&range[1]),
          ) {
            return Some(Ok(Expr::Identifier(
              if lo <= xv && xv <= hi {
                "True"
              } else {
                "False"
              }
              .to_string(),
            )));
          }
        } else if is_list_of_ranges {
          // Multiple ranges: Between[x, {{min1, max1}, ...}]
          if let Some(xv) = try_eval_to_f64(x) {
            for r in range {
              if let Expr::List(pair) = r
                && pair.len() == 2
                && let (Some(lo), Some(hi)) =
                  (try_eval_to_f64(&pair[0]), try_eval_to_f64(&pair[1]))
                && lo <= xv
                && xv <= hi
              {
                return Some(Ok(Expr::Identifier("True".to_string())));
              }
            }
            return Some(Ok(Expr::Identifier("False".to_string())));
          }
        }
      }
    }
    "Between" if args.len() == 1 => {
      // Operator form: Between[{min, max}] returns itself (handled by curried call)
      if let Expr::List(range) = &args[0]
        && range.len() == 2
      {
        return Some(Ok(Expr::FunctionCall {
          name: "Between".to_string(),
          args: args.to_vec(),
        }));
      }
    }
    "MemberQ" if args.len() == 2 => {
      return Some(crate::functions::predicate_ast::member_q_ast(args));
    }
    "FreeQ" if args.len() == 2 => {
      return Some(crate::functions::predicate_ast::free_q_ast(args));
    }
    "MatchQ" if args.len() == 2 => {
      return Some(crate::functions::predicate_ast::match_q_ast(args));
    }
    "Divisible" if args.len() == 2 => {
      return Some(crate::functions::predicate_ast::divisible_ast(args));
    }
    "SubsetQ" if args.len() == 2 => {
      return Some(crate::functions::predicate_ast::subset_q_ast(args));
    }
    "OptionQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::option_q_ast(args));
    }
    "Head" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::head_ast(args));
    }
    "Length" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::length_ast(args));
    }
    "Depth" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::depth_ast(args));
    }
    "LeafCount" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::leaf_count_ast(args));
    }
    "ByteCount" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::byte_count_ast(args));
    }
    // MaxMemoryUsed[] - peak memory usage of the process
    "MaxMemoryUsed" if args.is_empty() => {
      let peak_bytes = crate::functions::memory::max_memory_used();
      return Some(Ok(Expr::Integer(peak_bytes)));
    }
    // MemoryInUse[] - current memory usage of the process
    "MemoryInUse" if args.is_empty() => {
      let rss_bytes = crate::functions::memory::memory_in_use();
      return Some(Ok(Expr::Integer(rss_bytes)));
    }
    // Introspection functions - return {} for symbols without stored definitions
    "Messages" | "DownValues" | "OwnValues" | "SubValues" | "NValues"
    | "FormatValues" | "DefaultValues"
      if args.len() == 1 =>
    {
      return Some(Ok(Expr::List(vec![])));
    }
    "UpValues" if args.len() == 1 => {
      if let Expr::Identifier(sym) = &args[0] {
        let up_defs = crate::UPVALUES
          .with(|m| m.borrow().get(sym).cloned().unwrap_or_default());
        if up_defs.is_empty() {
          return Some(Ok(Expr::List(vec![])));
        }
        // Return a list of RuleDelayed expressions
        let rules: Vec<Expr> = up_defs
          .iter()
          .map(|(outer_func, _params, _conds, _defaults, _heads, body)| {
            Expr::RuleDelayed {
              pattern: Box::new(Expr::FunctionCall {
                name: "HoldPattern".to_string(),
                args: vec![Expr::FunctionCall {
                  name: outer_func.clone(),
                  args: vec![Expr::Identifier("__".to_string())],
                }],
              }),
              replacement: Box::new(body.clone()),
            }
          })
          .collect();
        return Some(Ok(Expr::List(rules)));
      }
      return Some(Ok(Expr::List(vec![])));
    }
    // FullForm - returns full form representation (unevaluated)
    "FullForm" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::full_form_ast(&args[0]));
    }
    "CForm" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "CForm".to_string(),
        args: args.to_vec(),
      }));
    }
    // Attributes[symbol] - returns the attributes of a built-in symbol
    "Attributes" if args.len() == 1 => {
      let sym_name = match &args[0] {
        Expr::Identifier(name) => name.as_str(),
        Expr::Constant(name) => name.as_str(),
        Expr::String(name) => name.as_str(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Attributes".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      // Check user-defined attributes first, then combine with built-in
      let user_attrs =
        crate::FUNC_ATTRS.with(|m| m.borrow().get(sym_name).cloned());
      let builtin = get_builtin_attributes(sym_name);
      let mut all_attr_strs: Vec<String> =
        builtin.iter().map(|a| a.to_string()).collect();
      if let Some(user) = user_attrs {
        for a in user {
          if !all_attr_strs.contains(&a) {
            all_attr_strs.push(a);
          }
        }
        all_attr_strs.sort();
      }
      return Some(Ok(Expr::List(
        all_attr_strs
          .iter()
          .map(|a| Expr::Identifier(a.to_string()))
          .collect(),
      )));
    }
    // Context[] - return current context
    // Context[symbol] - return context of a symbol
    "Context" if args.is_empty() => {
      return Some(Ok(Expr::String("Global`".to_string())));
    }
    "Context" if args.len() == 1 => {
      let (sym_name, from_string) = match &args[0] {
        Expr::Identifier(name) => (name.clone(), false),
        Expr::String(name) => (name.clone(), true),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Context".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      // Built-in symbols are in System` context
      let builtin = get_builtin_attributes(&sym_name);
      if !builtin.is_empty() {
        return Some(Ok(Expr::String("System`".to_string())));
      }
      // For string arguments, check if symbol exists; if not, return unevaluated
      if from_string {
        let exists = crate::ENV.with(|e| e.borrow().contains_key(&sym_name))
          || crate::FUNC_DEFS.with(|m| m.borrow().contains_key(&sym_name))
          || crate::FUNC_ATTRS.with(|m| m.borrow().contains_key(&sym_name));
        if !exists {
          return Some(Ok(Expr::FunctionCall {
            name: "Context".to_string(),
            args: args.to_vec(),
          }));
        }
      }
      // User-defined symbols are in Global` context
      return Some(Ok(Expr::String("Global`".to_string())));
    }
    // Options[f] - return stored options for function f
    // Options[f, opt] - return specific option for function f
    "Options" if args.len() == 1 || args.len() == 2 => {
      let func_arg = match evaluate_expr_to_expr(&args[0]) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      let func_name = match &func_arg {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Options".to_string(),
            args: vec![func_arg],
          }));
        }
      };
      let stored =
        crate::FUNC_OPTIONS.with(|m| m.borrow().get(&func_name).cloned());
      let opts = stored.unwrap_or_default();
      if args.len() == 1 {
        return Some(Ok(Expr::List(opts)));
      } else {
        // Options[f, opt] - find the matching option
        let opt_arg = match evaluate_expr_to_expr(&args[1]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        let opt_name = match &opt_arg {
          Expr::Identifier(name) => name.clone(),
          _ => {
            return Some(Ok(Expr::List(vec![])));
          }
        };
        let matching: Vec<Expr> = opts
          .into_iter()
          .filter(|rule| match rule {
            Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
              matches!(pattern.as_ref(), Expr::Identifier(n) if *n == opt_name)
            }
            _ => false,
          })
          .collect();
        return Some(Ok(Expr::List(matching)));
      }
    }
    // Construct - creates function call f[a][b] etc.
    "Construct" if !args.is_empty() => {
      return Some(crate::functions::predicate_ast::construct_ast(args));
    }
    // NameQ["name"] - check if a symbol with that name exists
    "NameQ" if args.len() == 1 => {
      if let Expr::String(name) = &args[0] {
        // Check if the symbol has been defined (OwnValues, DownValues, or is built-in)
        let has_own = crate::ENV.with(|e| e.borrow().contains_key(name));
        let has_down = crate::FUNC_DEFS.with(|m| m.borrow().contains_key(name));
        let has_builtin_attrs =
          !crate::evaluator::attributes::get_builtin_attributes(name)
            .is_empty();
        if has_own || has_down || has_builtin_attrs {
          return Some(Ok(Expr::Identifier("True".to_string())));
        }
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "NameQ".to_string(),
        args: args.to_vec(),
      }));
    }
    // Share[expr] - memory optimization, returns 0 (no-op in Woxi)
    "Share" => {
      return Some(Ok(Expr::Integer(0)));
    }
    _ => {}
  }
  None
}
