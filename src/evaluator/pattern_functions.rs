#[allow(unused_imports)]
use super::*;

/// Pattern[name, Blank[]] → name_ or Pattern[name, Blank[h]] → name_h
#[inline(never)]
pub fn evaluate_pattern_function_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "Pattern".to_string(),
      args: args.to_vec(),
    });
  }
  let pattern_name = match &args[0] {
    Expr::Identifier(n) => n.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Pattern".to_string(),
        args: args.to_vec(),
      });
    }
  };
  // Evaluate the second argument (the blank part)
  let blank = evaluate_expr_to_expr(&args[1])?;
  match &blank {
    Expr::Pattern {
      name: blank_name,
      head,
    } if blank_name.is_empty() => Ok(Expr::Pattern {
      name: pattern_name,
      head: head.clone(),
    }),
    _ => Ok(Expr::FunctionCall {
      name: "Pattern".to_string(),
      args: vec![args[0].clone(), blank],
    }),
  }
}

/// RuleDelayed[lhs, rhs]: evaluate lhs, hold rhs unevaluated
#[inline(never)]
pub fn evaluate_rule_delayed_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let lhs = evaluate_expr_to_expr(&args[0])?;
  Ok(Expr::RuleDelayed {
    pattern: Box::new(lhs),
    replacement: Box::new(args[1].clone()),
  })
}

/// PatternTest[pattern, test]: return as symbolic FunctionCall
#[inline(never)]
pub fn evaluate_pattern_test_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  Ok(Expr::FunctionCall {
    name: "PatternTest".to_string(),
    args: args.to_vec(),
  })
}

/// BlankSequence/BlankNullSequence: return as symbolic FunctionCall
#[inline(never)]
pub fn evaluate_blank_sequence_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  Ok(Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec(),
  })
}

/// Blank[] → _ or Blank[h] → _h
#[inline(never)]
pub fn evaluate_blank_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    0 => Ok(Expr::Pattern {
      name: String::new(),
      head: None,
    }),
    1 => {
      if let Expr::Identifier(h) = &args[0] {
        Ok(Expr::Pattern {
          name: String::new(),
          head: Some(h.clone()),
        })
      } else {
        Ok(Expr::FunctionCall {
          name: "Blank".to_string(),
          args: args.to_vec(),
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "Blank".to_string(),
      args: args.to_vec(),
    }),
  }
}
