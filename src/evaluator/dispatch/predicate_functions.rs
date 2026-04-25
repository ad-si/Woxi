#[allow(unused_imports)]
use super::*;

/// True if `expr` contains any Real or BigFloat node — used to decide
/// between exact and inexact number predicates.
fn contains_real(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(contains_real)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_real(left) || contains_real(right)
    }
    Expr::UnaryOp { operand, .. } => contains_real(operand),
    _ => false,
  }
}

/// ExactNumberQ[x]: True for Integer/Rational/exact Complex (and their
/// expanded forms). `I`, `1 + I`, `4 I + 5/6` are all exact.
fn is_exact_number(expr: &Expr) -> bool {
  use crate::functions::math_ast::try_extract_complex_exact;
  if contains_real(expr) {
    return false;
  }
  if matches!(expr, Expr::Integer(_) | Expr::BigInteger(_)) {
    return true;
  }
  if let Expr::FunctionCall { name, args } = expr
    && name == "Rational"
    && args.len() == 2
    && matches!(args[0], Expr::Integer(_) | Expr::BigInteger(_))
    && matches!(args[1], Expr::Integer(_) | Expr::BigInteger(_))
  {
    return true;
  }
  // Pure imaginary unit and exact complex combinations — try the exact
  // (integer/rational) extractor.
  matches!(expr, Expr::Identifier(s) if s == "I")
    || try_extract_complex_exact(expr).is_some_and(|((_, rd), (_, id))| {
      // Both denominators must be non-zero for a meaningful exact rational.
      rd != 0 && id != 0
    })
}

/// InexactNumberQ[x]: True when `x` is a numeric value that contains a
/// Real or BigFloat (and otherwise extracts to a complex form).
fn is_inexact_number(expr: &Expr) -> bool {
  use crate::functions::math_ast::try_extract_complex_float;
  if matches!(expr, Expr::Real(_) | Expr::BigFloat(_, _)) {
    return true;
  }
  contains_real(expr) && try_extract_complex_float(expr).is_some()
}

/// Mirror of the same-family rule in core_eval's Comparison handler, but
/// for Inequality's string operator names. Returns true iff the chain
/// should be split into pairwise `&&`.
fn should_split_inequality(names: &[&str]) -> bool {
  // Homogeneous → never split.
  if names.iter().skip(1).all(|n| *n == names[0]) {
    return false;
  }
  let mut has_unequal = false;
  let mut has_less = false;
  let mut has_greater = false;
  for n in names {
    match *n {
      "Unequal" | "NotEqual" | "UnsameQ" => has_unequal = true,
      "Less" | "LessEqual" => has_less = true,
      "Greater" | "GreaterEqual" => has_greater = true,
      _ => {}
    }
  }
  has_unequal || (has_less && has_greater)
}

pub fn dispatch_predicate_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    // Inequality[a, Less, b, Less, c, ...] - evaluate chained comparisons
    "Inequality" if args.len() >= 3 && args.len() % 2 == 1 => {
      let comparison_op = |op_name: &str| -> Option<fn(f64, f64) -> bool> {
        match op_name {
          "Less" => Some(|a: f64, b: f64| a < b),
          "LessEqual" => Some(|a: f64, b: f64| a <= b),
          "Greater" => Some(|a: f64, b: f64| a > b),
          "GreaterEqual" => Some(|a: f64, b: f64| a >= b),
          "Equal" => Some(|a: f64, b: f64| (a - b).abs() < f64::EPSILON),
          _ => None,
        }
      };
      // Try to evaluate all numeric values
      let mut all_numeric = true;
      let mut values: Vec<f64> = Vec::new();
      let mut ops: Vec<fn(f64, f64) -> bool> = Vec::new();
      for (idx, arg) in args.iter().enumerate() {
        if idx % 2 == 0 {
          // Value position
          if let Some(f) = crate::functions::math_ast::try_eval_to_f64(arg) {
            values.push(f);
          } else {
            all_numeric = false;
            break;
          }
        } else {
          // Operator position
          if let Expr::Identifier(op_name) = arg {
            if let Some(op) = comparison_op(op_name) {
              ops.push(op);
            } else {
              all_numeric = false;
              break;
            }
          } else {
            all_numeric = false;
            break;
          }
        }
      }
      if all_numeric && ops.len() + 1 == values.len() {
        let result = ops
          .iter()
          .zip(values.windows(2))
          .all(|(op, pair)| op(pair[0], pair[1]));
        return Some(Ok(Expr::Identifier(
          if result { "True" } else { "False" }.to_string(),
        )));
      }
      // For mixed-direction chains (e.g. `Greater` with `LessEqual`, or
      // any NotEqual mixed in), split into pairwise `&&` — wolframscript's
      // behavior. Preserves the `Inequality[…]` head in all other cases so
      // `ToString[Inequality[0, LessEqual, x, LessEqual, 1], InputForm]`
      // still renders the head form.
      if args.len() >= 5 && args.len() % 2 == 1 {
        use crate::syntax::ComparisonOp;
        let op_names: Vec<Option<&str>> = args
          .iter()
          .enumerate()
          .filter(|(i, _)| i % 2 == 1)
          .map(|(_, a)| match a {
            Expr::Identifier(s) => Some(s.as_str()),
            _ => None,
          })
          .collect();
        if op_names.iter().all(|o| o.is_some()) {
          let names: Vec<&str> = op_names.into_iter().flatten().collect();
          if should_split_inequality(&names) {
            let operands: Vec<Expr> = args
              .iter()
              .enumerate()
              .filter(|(i, _)| i % 2 == 0)
              .map(|(_, a)| a.clone())
              .collect();
            let op_for = |s: &str| -> Option<ComparisonOp> {
              match s {
                "Equal" => Some(ComparisonOp::Equal),
                "Unequal" | "NotEqual" => Some(ComparisonOp::NotEqual),
                "Less" => Some(ComparisonOp::Less),
                "LessEqual" => Some(ComparisonOp::LessEqual),
                "Greater" => Some(ComparisonOp::Greater),
                "GreaterEqual" => Some(ComparisonOp::GreaterEqual),
                "SameQ" => Some(ComparisonOp::SameQ),
                "UnsameQ" => Some(ComparisonOp::UnsameQ),
                _ => None,
              }
            };
            if names.iter().all(|n| op_for(n).is_some()) {
              let mut terms: Vec<Expr> = Vec::new();
              for (i, n) in names.iter().enumerate() {
                terms.push(Expr::Comparison {
                  operands: vec![operands[i].clone(), operands[i + 1].clone()],
                  operators: vec![op_for(n).unwrap()],
                });
              }
              return Some(crate::evaluator::evaluate_function_call_ast(
                "And", &terms,
              ));
            }
          }
        }
      }
    }
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
    "NotElement" if args.len() == 2 => {
      return Some(not_element_ast(&args[0], &args[1]));
    }
    "IntegerQ" if args.len() == 1 => {
      return Some(crate::functions::predicate_ast::integer_q_ast(args));
    }
    "MachineNumberQ" if args.len() == 1 => {
      fn contains_real(e: &Expr) -> bool {
        match e {
          Expr::Real(_) => true,
          Expr::FunctionCall { args, .. } => args.iter().any(contains_real),
          Expr::BinaryOp { left, right, .. } => {
            contains_real(left) || contains_real(right)
          }
          Expr::UnaryOp { operand, .. } => contains_real(operand),
          _ => false,
        }
      }
      let is_machine = match &args[0] {
        Expr::Real(_) => true,
        Expr::FunctionCall { name, args: ca }
          if name == "Complex" && ca.len() == 2 =>
        {
          matches!(&ca[0], Expr::Real(_)) || matches!(&ca[1], Expr::Real(_))
        }
        _ => {
          // A Plus/Times expression like `1.5 + 2.3 I` that forms a complex number
          // with at least one machine Real component.
          crate::functions::predicate_ast::is_complex_number(&args[0])
            && contains_real(&args[0])
        }
      };
      return Some(Ok(Expr::Identifier(
        if is_machine { "True" } else { "False" }.to_string(),
      )));
    }
    "MissingQ" if args.len() == 1 => {
      let is_missing = matches!(
        &args[0],
        Expr::FunctionCall { name, .. } if name == "Missing"
      );
      return Some(Ok(Expr::Identifier(
        if is_missing { "True" } else { "False" }.to_string(),
      )));
    }
    "ColorQ" if args.len() == 1 => {
      let is_color = match &args[0] {
        Expr::FunctionCall { name, .. } => matches!(
          name.as_str(),
          "RGBColor"
            | "Hue"
            | "GrayLevel"
            | "CMYKColor"
            | "XYZColor"
            | "LABColor"
            | "LCHColor"
            | "LUVColor"
            | "Opacity"
        ),
        _ => false,
      };
      return Some(Ok(Expr::Identifier(
        if is_color { "True" } else { "False" }.to_string(),
      )));
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
    "Precedence" if args.len() == 1 => {
      return Some(Ok(Expr::Real(precedence_value(&args[0]))));
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
    "PerfectNumberQ" if args.len() == 1 => {
      if let Expr::Integer(n) = &args[0] {
        if *n <= 0 {
          return Some(Ok(Expr::Identifier("False".to_string())));
        }
        let n_val = *n;
        // Sum of proper divisors
        let mut sum: i128 = 1;
        let mut i: i128 = 2;
        while i * i <= n_val {
          if n_val % i == 0 {
            sum += i;
            if i != n_val / i {
              sum += n_val / i;
            }
          }
          i += 1;
        }
        if n_val == 1 {
          sum = 0;
        }
        return Some(Ok(Expr::Identifier(
          if sum == n_val { "True" } else { "False" }.to_string(),
        )));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "PerfectNumberQ".to_string(),
        args: args.to_vec(),
      }));
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
      let is_exact = is_exact_number(&args[0]);
      return Some(Ok(Expr::Identifier(
        if is_exact { "True" } else { "False" }.to_string(),
      )));
    }
    "InexactNumberQ" if args.len() == 1 => {
      let is_inexact = is_inexact_number(&args[0]);
      return Some(Ok(Expr::Identifier(
        if is_inexact { "True" } else { "False" }.to_string(),
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
    "MemberQ" if args.len() == 2 || args.len() == 3 => {
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
    "PossibleZeroQ" => {
      return Some(crate::functions::predicate_ast::possible_zero_q_ast(args));
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
    // MemoryAvailable[] - estimate of free system memory in bytes
    "MemoryAvailable" if args.is_empty() => {
      let bytes = crate::functions::memory::memory_available();
      return Some(Ok(Expr::Integer(bytes)));
    }
    // OwnValues[sym]: if `sym` has a stored value (from `sym = value`),
    // return `{HoldPattern[sym] :> value}`, matching Wolfram. Otherwise {}.
    "OwnValues" if args.len() == 1 => {
      if let Expr::Identifier(name) = &args[0] {
        let value_expr = crate::lookup_env_as_expr(name);
        if let Some(v) = value_expr {
          return Some(Ok(Expr::List(vec![Expr::RuleDelayed {
            pattern: Box::new(Expr::FunctionCall {
              name: "HoldPattern".to_string(),
              args: vec![Expr::Identifier(name.clone())],
            }),
            replacement: Box::new(v),
          }])));
        }
      }
      return Some(Ok(Expr::List(vec![])));
    }
    // Messages[sym] — return all stored MessageName DownValues that
    // target this symbol. `sym::tag = "text"` is stored as a DownValue
    // on MessageName itself (`MessageName[sym, "tag"] :> "text"`); pull
    // out the ones whose first argument is `sym`.
    "Messages" if args.len() == 1 => {
      let sym = match &args[0] {
        Expr::Identifier(s) => s.clone(),
        _ => return Some(Ok(Expr::List(vec![]))),
      };
      let func_defs = crate::FUNC_DEFS.with(|m| {
        m.borrow().get("MessageName").cloned().unwrap_or_default()
      });
      let rules: Vec<Expr> = func_defs
        .iter()
        .filter_map(|(params, _conds, _defaults, _heads, _blank_types, body)| {
          // MessageName has exactly two slots: first is the symbol literal
          // (matched as a SameQ condition with param `_dv0`), second is
          // the tag string (`_dv1`). Re-derive the literal MessageName
          // pattern from `body`'s associated structural-pattern conditions.
          // The simpler path: use the conditions to read the literal sym
          // value matched in slot 0. If the rule's slot-0 SameQ value
          // doesn't equal `sym`, skip it.
          let slot0_literal = _conds.iter().find_map(|c| {
            if let Some(Expr::Comparison { operands, operators }) = c
              && operators.len() == 1
              && matches!(
                operators[0],
                crate::syntax::ComparisonOp::SameQ
              )
              && operands.len() == 2
              && let Expr::Identifier(name) = &operands[0]
              && name == &params[0]
            {
              Some(operands[1].clone())
            } else {
              None
            }
          })?;
          if !matches!(&slot0_literal, Expr::Identifier(s) if s == &sym) {
            return None;
          }
          // Slot 1 is the tag string (or symbol). It's matched the same
          // way; pull its literal value out of the conditions.
          let slot1_literal = _conds.iter().find_map(|c| {
            if let Some(Expr::Comparison { operands, operators }) = c
              && operators.len() == 1
              && matches!(
                operators[0],
                crate::syntax::ComparisonOp::SameQ
              )
              && operands.len() == 2
              && let Expr::Identifier(name) = &operands[0]
              && name == &params[1]
            {
              Some(operands[1].clone())
            } else {
              None
            }
          })?;
          Some(Expr::RuleDelayed {
            pattern: Box::new(Expr::FunctionCall {
              name: "HoldPattern".to_string(),
              args: vec![Expr::FunctionCall {
                name: "MessageName".to_string(),
                args: vec![slot0_literal, slot1_literal],
              }],
            }),
            replacement: Box::new(body.clone()),
          })
        })
        .collect();
      return Some(Ok(Expr::List(rules)));
    }
    // Other introspection functions - return {} for symbols without stored definitions
    "SubValues" | "NValues" | "FormatValues" if args.len() == 1 => {
      return Some(Ok(Expr::List(vec![])));
    }
    // DefaultValues exposes the built-in identity elements used by
    // Optional/OneIdentity pattern matching: Plus → 0, Times → 1, and
    // Power's second slot → 1. Anything else with no stored definition
    // returns {}.
    "DefaultValues" if args.len() == 1 => {
      let rule = |pat: Expr, val: Expr| Expr::RuleDelayed {
        pattern: Box::new(Expr::FunctionCall {
          name: "HoldPattern".to_string(),
          args: vec![pat],
        }),
        replacement: Box::new(val),
      };
      let default_of = |sym: &str, extra: Vec<Expr>| {
        let mut default_args = vec![Expr::Identifier(sym.to_string())];
        default_args.extend(extra);
        Expr::FunctionCall {
          name: "Default".to_string(),
          args: default_args,
        }
      };
      if let Expr::Identifier(sym) = &args[0] {
        let values: Vec<Expr> = match sym.as_str() {
          "Plus" => vec![rule(default_of("Plus", vec![]), Expr::Integer(0))],
          "Times" => vec![rule(default_of("Times", vec![]), Expr::Integer(1))],
          "Power" => vec![rule(
            default_of("Power", vec![Expr::Integer(2)]),
            Expr::Integer(1),
          )],
          _ => vec![],
        };
        return Some(Ok(Expr::List(values)));
      }
      return Some(Ok(Expr::List(vec![])));
    }
    "DownValues" if args.len() == 1 => {
      if let Expr::Identifier(sym) = &args[0] {
        let func_defs = crate::FUNC_DEFS
          .with(|m| m.borrow().get(sym).cloned().unwrap_or_default());
        if func_defs.is_empty() {
          return Some(Ok(Expr::List(vec![])));
        }
        // TagSet/TagSetDelayed definitions are also stored in FUNC_DEFS so
        // the usual dispatch picks them up, but they belong to UpValues,
        // not DownValues of the outer head. Collect (params, heads) pairs
        // for every UPVALUES entry whose outer_func matches this symbol,
        // then filter those entries out below.
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
        let rules: Vec<Expr> = func_defs
          .iter()
          .filter(|(params, _, _, heads, _, _)| {
            !upvalue_keys.contains(&(params.clone(), heads.clone()))
          })
          .map(|(params, conds, _defaults, heads, blank_types, body)| {
            let pattern_args: Vec<Expr> = params
              .iter()
              .enumerate()
              .map(|(i, p)| {
                // Check if this param has a literal-match condition (SameQ)
                if let Some(Some(Expr::Comparison {
                  operands,
                  operators,
                })) = conds.get(i)
                  && operators
                    .iter()
                    .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
                  && let Some(literal_val) = operands.get(1)
                {
                  return literal_val.clone();
                }
                Expr::Pattern {
                  name: p.clone(),
                  head: heads.get(i).and_then(|h| h.clone()),
                  blank_type: blank_types.get(i).copied().unwrap_or(1),
                }
              })
              .collect();
            Expr::RuleDelayed {
              pattern: Box::new(Expr::FunctionCall {
                name: "HoldPattern".to_string(),
                args: vec![Expr::FunctionCall {
                  name: sym.clone(),
                  args: pattern_args,
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
    "UpValues" if args.len() == 1 => {
      if let Expr::Identifier(sym) = &args[0] {
        let up_defs = crate::UPVALUES
          .with(|m| m.borrow().get(sym).cloned().unwrap_or_default());
        if up_defs.is_empty() {
          return Some(Ok(Expr::List(vec![])));
        }
        // Return a list of RuleDelayed expressions using the original LHS and body
        let rules: Vec<Expr> = up_defs
          .iter()
          .map(
            |(
              _outer_func,
              _params,
              _conds,
              _defaults,
              _heads,
              _body,
              orig_lhs,
              orig_body,
            )| {
              Expr::RuleDelayed {
                pattern: Box::new(Expr::FunctionCall {
                  name: "HoldPattern".to_string(),
                  args: vec![orig_lhs.clone()],
                }),
                replacement: Box::new(orig_body.clone()),
              }
            },
          )
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
    "TeXForm" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "TeXForm".to_string(),
        args: args.to_vec(),
      }));
    }
    "FortranForm" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "FortranForm".to_string(),
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
    // Contexts[] — list known contexts; Contexts["pattern*"] — filter by
    // glob pattern. Woxi only tracks System` and Global` contexts; any other
    // pattern matches none.
    "Contexts" if args.is_empty() => {
      return Some(Ok(Expr::List(vec![
        Expr::String("System`".to_string()),
        Expr::String("Global`".to_string()),
      ])));
    }
    "Contexts" if args.len() == 1 => {
      let pattern = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "Contexts".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      // Convert glob (* wildcards) to a simple matcher.
      let glob_match = |name: &str| -> bool {
        // Split the pattern on '*' and check each literal part appears in
        // order, anchored to start/end unless the pattern begins/ends with '*'.
        let parts: Vec<&str> = pattern.split('*').collect();
        let mut pos = 0usize;
        for (i, part) in parts.iter().enumerate() {
          if part.is_empty() {
            continue;
          }
          if i == 0 {
            // Must start with this prefix.
            if !name[pos..].starts_with(part) {
              return false;
            }
            pos += part.len();
          } else if i == parts.len() - 1 {
            // Last part must match the remaining end unless pattern ends in '*'.
            if pattern.ends_with('*') {
              if !name[pos..].contains(part) {
                return false;
              }
            } else if !name[pos..].ends_with(part) {
              return false;
            }
          } else {
            match name[pos..].find(part) {
              Some(idx) => pos += idx + part.len(),
              None => return false,
            }
          }
        }
        if !pattern.contains('*') {
          name == pattern
        } else {
          true
        }
      };
      let all = ["System`", "Global`"];
      let matches: Vec<Expr> = all
        .iter()
        .filter(|n| glob_match(n))
        .map(|n| Expr::String(n.to_string()))
        .collect();
      return Some(Ok(Expr::List(matches)));
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
      let opts = stored.unwrap_or_else(|| builtin_default_options(&func_name));
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
    // OptionValue[name] - look up option value from current OptionsPattern context
    "OptionValue" if args.len() == 1 => {
      let opt_arg = match evaluate_expr_to_expr(&args[0]) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      let opt_name = match &opt_arg {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "OptionValue".to_string(),
            args: vec![opt_arg],
          }));
        }
      };
      // Look up in the current option value context stack (innermost first)
      let result = crate::OPTION_VALUE_CONTEXT.with(|ctx| {
        let stack = ctx.borrow();
        for (_func_name, bindings) in stack.iter().rev() {
          for (key, val) in bindings {
            if *key == opt_name {
              return Some(val.clone());
            }
          }
        }
        None
      });
      match result {
        Some(val) => return Some(Ok(val)),
        None => {
          return Some(Ok(Expr::FunctionCall {
            name: "OptionValue".to_string(),
            args: vec![opt_arg],
          }));
        }
      }
    }
    // OptionValue[f, name] - look up option value from Options[f]
    // OptionValue[f, opts, name] - look up from explicit opts list, falling back to Options[f]
    "OptionValue" if args.len() == 2 || args.len() == 3 => {
      let func_arg = match evaluate_expr_to_expr(&args[0]) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      let func_name = match &func_arg {
        Expr::Identifier(name) => Some(name.clone()),
        _ => None,
      };
      // Collect extra opts if 3-arg form
      let extra_opts: Vec<Expr> = if args.len() == 3 {
        let ev = match evaluate_expr_to_expr(&args[1]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        if let Expr::List(items) = &ev {
          items.clone()
        } else {
          vec![ev]
        }
      } else {
        Vec::new()
      };
      // The name argument
      let name_idx = args.len() - 1;
      let name_arg = match evaluate_expr_to_expr(&args[name_idx]) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
      };
      // Resolve lookup key as string
      let (lookup_key, _is_string) = match &name_arg {
        Expr::Identifier(n) => (n.clone(), false),
        Expr::String(s) => (s.clone(), true),
        _ => {
          return Some(Ok(Expr::FunctionCall {
            name: "OptionValue".to_string(),
            args: args.to_vec(),
          }));
        }
      };
      // Helper: match rule key against lookup_key
      let matches_key = |pattern: &Expr| -> bool {
        match pattern {
          Expr::Identifier(n) => *n == lookup_key,
          Expr::String(s) => *s == lookup_key,
          _ => false,
        }
      };
      let find_value = |rules: &[Expr]| -> Option<Expr> {
        for rule in rules {
          match rule {
            Expr::Rule {
              pattern,
              replacement,
            }
            | Expr::RuleDelayed {
              pattern,
              replacement,
            } if matches_key(pattern.as_ref()) => {
              return Some((**replacement).clone());
            }
            _ => {}
          }
        }
        None
      };
      // Search order: extra_opts first, then Options[f]
      if let Some(v) = find_value(&extra_opts) {
        return Some(Ok(v));
      }
      if let Some(fname) = &func_name {
        let stored =
          crate::FUNC_OPTIONS.with(|m| m.borrow().get(fname).cloned());
        if let Some(rules) = stored
          && let Some(v) = find_value(&rules)
        {
          return Some(Ok(v));
        }
      }
      // Not found: emit optnf warning (to stderr via eprintln), return name as-is
      let fname_display = match &func_name {
        Some(n) => n.clone(),
        None => crate::syntax::expr_to_string(&func_arg),
      };
      eprintln!(
        "OptionValue::optnf: Option name {} not found in defaults for {}.",
        lookup_key, fname_display
      );
      return Some(Ok(name_arg));
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
    // Operator forms: EqualTo[y][x] → x == y, etc.
    "EqualTo" if args.len() == 1 => {
      return Some(Ok(Expr::Function {
        body: Box::new(Expr::Comparison {
          operands: vec![Expr::Slot(1), args[0].clone()],
          operators: vec![crate::syntax::ComparisonOp::Equal],
        }),
      }));
    }
    "UnequalTo" if args.len() == 1 => {
      return Some(Ok(Expr::Function {
        body: Box::new(Expr::Comparison {
          operands: vec![Expr::Slot(1), args[0].clone()],
          operators: vec![crate::syntax::ComparisonOp::NotEqual],
        }),
      }));
    }
    "GreaterThan" if args.len() == 1 => {
      return Some(Ok(Expr::Function {
        body: Box::new(Expr::Comparison {
          operands: vec![Expr::Slot(1), args[0].clone()],
          operators: vec![crate::syntax::ComparisonOp::Greater],
        }),
      }));
    }
    "LessThan" if args.len() == 1 => {
      return Some(Ok(Expr::Function {
        body: Box::new(Expr::Comparison {
          operands: vec![Expr::Slot(1), args[0].clone()],
          operators: vec![crate::syntax::ComparisonOp::Less],
        }),
      }));
    }
    "GreaterEqualThan" if args.len() == 1 => {
      return Some(Ok(Expr::Function {
        body: Box::new(Expr::Comparison {
          operands: vec![Expr::Slot(1), args[0].clone()],
          operators: vec![crate::syntax::ComparisonOp::GreaterEqual],
        }),
      }));
    }
    "LessEqualThan" if args.len() == 1 => {
      return Some(Ok(Expr::Function {
        body: Box::new(Expr::Comparison {
          operands: vec![Expr::Slot(1), args[0].clone()],
          operators: vec![crate::syntax::ComparisonOp::LessEqual],
        }),
      }));
    }
    _ => {}
  }
  None
}

/// Helper to create a Rule expression: name -> value
fn make_rule(name: &str, value: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(Expr::Identifier(name.to_string())),
    replacement: Box::new(value),
  }
}

/// Precedence of a symbol or expression — matches Wolfram's `Precedence`.
/// Returns the parsing precedence: built-in operators get specific values,
/// unknown symbols default to 670., non-symbol expressions get 1000.
fn precedence_value(expr: &Expr) -> f64 {
  let name = match expr {
    Expr::Identifier(s) | Expr::Constant(s) => s.as_str(),
    Expr::FunctionCall { name, args } if args.is_empty() => name.as_str(),
    _ => return 1000.0,
  };
  match name {
    "CompoundExpression" => 10.0,
    "Put" | "PutAppend" => 30.0,
    "Set" | "SetDelayed" | "UpSet" | "UpSetDelayed" | "TagSet"
    | "TagSetDelayed" | "TagUnset" | "Unset" => 40.0,
    "Because" => 50.0,
    "Therefore" => 60.0,
    "Postfix" => 70.0,
    "Colon" => 80.0,
    "Function" => 90.0,
    "AddTo" | "SubtractFrom" | "TimesBy" | "DivideBy" => 100.0,
    "ReplaceAll" | "ReplaceRepeated" => 110.0,
    "Rule" | "RuleDelayed" => 120.0,
    "StringExpression" => 135.0,
    "Condition" => 140.0,
    "Pattern" => 150.0,
    "Alternatives" => 160.0,
    "Implies" => 200.0,
    "Equivalent" => 205.0,
    "Or" | "Nor" | "Xor" | "Nand" | "And" => 215.0,
    "Not" => 230.0,
    "Equal" | "Unequal" | "SameQ" | "UnsameQ" | "Less" | "Greater"
    | "LessEqual" | "GreaterEqual" | "Inequality" => 290.0,
    "Span" => 305.0,
    "Plus" | "Subtract" => 310.0,
    "PlusMinus" => 315.0,
    "Sum" | "Limit" => 320.0,
    "Integrate" => 325.0,
    "Product" => 380.0,
    "Union" => 390.0,
    "Intersection" => 395.0,
    "Times" | "Divide" => 400.0,
    "Cross" => 410.0,
    "CenterDot" => 420.0,
    "CircleTimes" => 430.0,
    "SmallCircle" => 440.0,
    "Backslash" => 460.0,
    "Diamond" => 470.0,
    "Wedge" => 480.0,
    "Dot" => 490.0,
    "NonCommutativeMultiply" => 510.0,
    "Minus" => 480.0,
    "D" => 550.0,
    "Power" => 590.0,
    "Apply" => 620.0,
    "Map" | "MapAll" => 620.0,
    "Factorial" | "Factorial2" => 610.0,
    "Conjugate" | "Transpose" => 625.0,
    "Repeated" | "RepeatedNull" => 635.0,
    "RightComposition" => 648.0,
    "Composition" => 650.0,
    "Prefix" => 640.0,
    "StringJoin" => 600.0,
    "MessageName" => 740.0,
    "Blank" | "BlankSequence" | "BlankNullSequence" => 730.0,
    "Increment" | "Decrement" | "PreIncrement" | "PreDecrement" => 660.0,
    "Optional" => 680.0,
    "PatternTest" => 680.0,
    // Unknown symbols default to 670
    _ => 670.0,
  }
}

/// Helper to create a RuleDelayed expression: name :> value
fn make_rule_delayed(name: &str, value: Expr) -> Expr {
  Expr::RuleDelayed {
    pattern: Box::new(Expr::Identifier(name.to_string())),
    replacement: Box::new(value),
  }
}

/// Return built-in default options for known functions.
pub fn builtin_default_options(func_name: &str) -> Vec<Expr> {
  let id = |s: &str| Expr::Identifier(s.to_string());
  let real = |f: f64| Expr::Real(f);
  let list = |v: Vec<Expr>| Expr::List(v);
  let func_slot1 = |body: Expr| Expr::Function {
    body: Box::new(body),
  };
  match func_name {
    "Plot" => vec![
      make_rule("AlignmentPoint", id("Center")),
      make_rule(
        "AspectRatio",
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![id("GoldenRatio"), Expr::Integer(-1)],
        },
      ),
      make_rule("Axes", id("True")),
      make_rule("AxesLabel", id("None")),
      make_rule("AxesOrigin", id("Automatic")),
      make_rule("AxesStyle", list(vec![])),
      make_rule("Background", id("None")),
      make_rule("BaselinePosition", id("Automatic")),
      make_rule("BaseStyle", list(vec![])),
      make_rule("ClippingStyle", id("None")),
      make_rule("ColorFunction", id("Automatic")),
      make_rule("ColorFunctionScaling", id("True")),
      make_rule("ColorOutput", id("Automatic")),
      make_rule("ContentSelectable", id("Automatic")),
      make_rule("CoordinatesToolOptions", id("Automatic")),
      make_rule_delayed("DisplayFunction", id("$DisplayFunction")),
      make_rule("Epilog", list(vec![])),
      make_rule("Evaluated", id("Automatic")),
      make_rule("EvaluationMonitor", id("None")),
      make_rule("Exclusions", id("Automatic")),
      make_rule("ExclusionsStyle", id("None")),
      make_rule("Filling", id("None")),
      make_rule("FillingStyle", id("Automatic")),
      make_rule_delayed("FormatType", id("TraditionalForm")),
      make_rule("Frame", id("False")),
      make_rule("FrameLabel", id("None")),
      make_rule("FrameStyle", list(vec![])),
      make_rule("FrameTicks", id("Automatic")),
      make_rule("FrameTicksStyle", list(vec![])),
      make_rule("GridLines", id("None")),
      make_rule("GridLinesStyle", list(vec![])),
      make_rule("ImageMargins", real(0.)),
      make_rule("ImagePadding", id("All")),
      make_rule("ImageSize", id("Automatic")),
      make_rule("ImageSizeRaw", id("Automatic")),
      make_rule("IntervalMarkers", id("Automatic")),
      make_rule("IntervalMarkersStyle", id("Automatic")),
      make_rule("LabelingSize", id("Automatic")),
      make_rule("LabelStyle", list(vec![])),
      make_rule("MaxRecursion", id("Automatic")),
      make_rule("Mesh", id("None")),
      make_rule("MeshFunctions", list(vec![func_slot1(Expr::Slot(1))])),
      make_rule("MeshShading", id("None")),
      make_rule("MeshStyle", id("Automatic")),
      make_rule("Method", id("Automatic")),
      make_rule_delayed("PerformanceGoal", id("$PerformanceGoal")),
      make_rule("PlotHighlighting", id("Automatic")),
      make_rule_delayed("PlotInteractivity", id("$PlotInteractivity")),
      make_rule("PlotLabel", id("None")),
      make_rule("PlotLabels", id("None")),
      make_rule("PlotLayout", id("Automatic")),
      make_rule("PlotLegends", id("None")),
      make_rule("PlotPoints", id("Automatic")),
      make_rule("PlotRange", list(vec![id("Full"), id("Automatic")])),
      make_rule("PlotRangeClipping", id("True")),
      make_rule("PlotRangePadding", id("Automatic")),
      make_rule("PlotRegion", id("Automatic")),
      make_rule("PlotStyle", id("Automatic")),
      make_rule_delayed("PlotTheme", id("$PlotTheme")),
      make_rule("PreserveImageOptions", id("Automatic")),
      make_rule("Prolog", list(vec![])),
      make_rule("RegionFunction", func_slot1(id("True"))),
      make_rule("RotateLabel", id("True")),
      make_rule("ScalingFunctions", id("None")),
      make_rule("TargetUnits", id("Automatic")),
      make_rule("Ticks", id("Automatic")),
      make_rule("TicksStyle", list(vec![])),
      make_rule("WorkingPrecision", id("MachinePrecision")),
    ],
    // Traversal functions that default Heads -> False — matches
    // wolframscript's built-in defaults so `Options[Level]`,
    // `Definition[Level]`, etc. report `{Heads -> False}` instead of `{}`.
    "Level" | "Map" | "Cases" | "Count" | "MapIndexed" | "Scan" | "FreeQ"
    | "MemberQ" | "DeleteCases" | "Replace" => {
      vec![make_rule("Heads", id("False"))]
    }
    // Same shape, but Position defaults Heads -> True.
    "Position" => vec![make_rule("Heads", id("True"))],
    _ => vec![],
  }
}
