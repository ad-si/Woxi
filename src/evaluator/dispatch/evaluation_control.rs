#[allow(unused_imports)]
use super::*;

pub fn dispatch_evaluation_control(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "HoldForm" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "HoldForm".to_string(),
        args: args.to_vec(),
      }));
    }
    "Hold" if !args.is_empty() => {
      return Some(Ok(Expr::FunctionCall {
        name: "Hold".to_string(),
        args: args.to_vec(),
      }));
    }
    "HoldComplete" if !args.is_empty() => {
      return Some(Ok(Expr::FunctionCall {
        name: "HoldComplete".to_string(),
        args: args.to_vec(),
      }));
    }
    "Unevaluated" if !args.is_empty() => {
      return Some(Ok(Expr::FunctionCall {
        name: "Unevaluated".to_string(),
        args: args.to_vec(),
      }));
    }
    "ReleaseHold" if args.len() == 1 => match &args[0] {
      Expr::FunctionCall {
        name: hold_name,
        args: hold_args,
      } if (hold_name == "Hold"
        || hold_name == "HoldForm"
        || hold_name == "HoldPattern")
        && hold_args.len() == 1 =>
      {
        return Some(evaluate_expr_to_expr(&hold_args[0]));
      }
      Expr::FunctionCall {
        name: hold_name,
        args: hold_args,
      } if (hold_name == "Hold"
        || hold_name == "HoldForm"
        || hold_name == "HoldPattern")
        && hold_args.len() > 1 =>
      {
        let evaluated: Result<Vec<Expr>, _> =
          hold_args.iter().map(evaluate_expr_to_expr).collect();
        match evaluated {
          Ok(evaled) => {
            return Some(Ok(Expr::FunctionCall {
              name: "Sequence".to_string(),
              args: evaled,
            }));
          }
          Err(e) => return Some(Err(e)),
        }
      }
      other => {
        return Some(evaluate_expr_to_expr(other));
      }
    },
    "TimeRemaining" if args.is_empty() => {
      return Some(Ok(Expr::Identifier("Infinity".to_string())));
    }
    "Evaluate" if args.len() == 1 => {
      return Some(Ok(args[0].clone()));
    }
    "RegularExpression" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "RegularExpression".to_string(),
        args: args.to_vec(),
      }));
    }
    "UniformDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "UniformDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "NormalDistribution" => {
      let norm_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else {
        args.to_vec()
      };
      return Some(Ok(Expr::FunctionCall {
        name: "NormalDistribution".to_string(),
        args: norm_args,
      }));
    }
    "Names" if args.len() <= 1 => {
      let all_names = crate::get_defined_names();
      if args.is_empty() {
        let items: Vec<Expr> =
          all_names.into_iter().map(Expr::String).collect();
        return Some(Ok(Expr::List(items)));
      }
      if let Expr::String(pattern) = &args[0] {
        let regex_pattern = format!(
          "^{}$",
          pattern
            .replace('.', "\\.")
            .replace('*', ".*")
            .replace('@', "[a-z0-9]*")
        );
        let re = regex::Regex::new(&regex_pattern);
        if let Ok(re) = re {
          let items: Vec<Expr> = all_names
            .into_iter()
            .filter(|n| re.is_match(n))
            .map(Expr::String)
            .collect();
          return Some(Ok(Expr::List(items)));
        }
      }
      return Some(Ok(Expr::List(vec![])));
    }
    "ValueQ" if args.len() == 1 => {
      if let Expr::Identifier(sym) = &args[0] {
        let has_value = ENV.with(|e| e.borrow().contains_key(sym));
        let has_func = crate::FUNC_DEFS.with(|m| m.borrow().contains_key(sym));
        return Some(Ok(Expr::Identifier(
          if has_value || has_func {
            "True"
          } else {
            "False"
          }
          .to_string(),
        )));
      }
      return Some(Ok(Expr::Identifier("False".to_string())));
    }
    "If" => {
      if args.len() >= 2 && args.len() <= 4 {
        let cond = match evaluate_expr_to_expr(&args[0]) {
          Ok(c) => c,
          Err(e) => return Some(Err(e)),
        };
        if matches!(&cond, Expr::Identifier(s) if s == "True") {
          return Some(evaluate_expr_to_expr(&args[1]));
        } else if matches!(&cond, Expr::Identifier(s) if s == "False") {
          if args.len() >= 3 {
            return Some(evaluate_expr_to_expr(&args[2]));
          } else {
            return Some(Ok(Expr::Identifier("Null".to_string())));
          }
        } else if args.len() == 4 {
          return Some(evaluate_expr_to_expr(&args[3]));
        }
      } else if args.len() < 2 || args.len() > 4 {
        println!(
          "\nIf::argb: If called with {} arguments; between 2 and 4 arguments are expected.",
          args.len()
        );
        use std::io::{self, Write};
        io::stdout().flush().ok();
      }
    }
    _ => {}
  }
  None
}
