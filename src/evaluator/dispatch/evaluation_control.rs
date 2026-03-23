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
    "UniformDistribution" if args.len() <= 1 => {
      let uni_args = if args.is_empty() {
        vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(1)])]
      } else {
        args.to_vec()
      };
      return Some(Ok(Expr::FunctionCall {
        name: "UniformDistribution".to_string(),
        args: uni_args,
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
    "ExponentialDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ExponentialDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "PoissonDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "PoissonDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "BernoulliDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "BernoulliDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "InverseGammaDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "InverseGammaDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "GammaDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "GammaDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "BetaDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "BetaDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "StudentTDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "StudentTDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "LogNormalDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "LogNormalDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "LogisticDistribution" => {
      let logistic_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else {
        args.to_vec()
      };
      return Some(Ok(Expr::FunctionCall {
        name: "LogisticDistribution".to_string(),
        args: logistic_args,
      }));
    }
    "GompertzMakehamDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "GompertzMakehamDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "InverseGaussianDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "InverseGaussianDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "FrechetDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "FrechetDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "ExtremeValueDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ExtremeValueDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "InverseChiSquareDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "InverseChiSquareDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "ChiSquareDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ChiSquareDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "ParetoDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ParetoDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "WeibullDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "WeibullDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "GeometricDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "GeometricDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "CauchyDistribution" => {
      let cauchy_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else {
        args.to_vec()
      };
      return Some(Ok(Expr::FunctionCall {
        name: "CauchyDistribution".to_string(),
        args: cauchy_args,
      }));
    }
    "DiscreteUniformDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "DiscreteUniformDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "LaplaceDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "LaplaceDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "RayleighDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "RayleighDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "NegativeBinomialDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "NegativeBinomialDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "MultinomialDistribution" if args.len() == 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "MultinomialDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "HalfNormalDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "HalfNormalDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "ChiDistribution" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "ChiDistribution".to_string(),
        args: args.to_vec(),
      }));
    }
    "StableDistribution"
      if args.len() == 2 || args.len() == 4 || args.len() == 5 =>
    {
      // Normalize to canonical 5-parameter form: StableDistribution[1, alpha, beta, mu, sigma]
      // 2-param: StableDistribution[alpha, beta] -> StableDistribution[1, alpha, beta, 0, 1]
      // 4-param: StableDistribution[alpha, beta, mu, sigma] -> StableDistribution[1, alpha, beta, mu, sigma]
      // 5-param: already canonical
      let canonical_args = match args.len() {
        2 => vec![
          Expr::Integer(1),
          args[0].clone(),
          args[1].clone(),
          Expr::Integer(0),
          Expr::Integer(1),
        ],
        4 => vec![
          Expr::Integer(1),
          args[0].clone(),
          args[1].clone(),
          args[2].clone(),
          args[3].clone(),
        ],
        _ => args.to_vec(), // 5-param, already canonical
      };
      return Some(Ok(Expr::FunctionCall {
        name: "StableDistribution".to_string(),
        args: canonical_args,
      }));
    }
    // ByteArray[{b1, b2, ...}] — create a byte array from a list of unsigned bytes
    // ByteArray["base64string"] — create a byte array from base64
    "ByteArray" if args.len() == 1 => {
      match &args[0] {
        Expr::List(items) => {
          // Validate all items are integers in 0..255
          let mut bytes = Vec::new();
          for item in items {
            match item {
              Expr::Integer(n) if (0..=255).contains(n) => {
                bytes.push(Expr::Integer(*n));
              }
              _ => {
                crate::emit_message(
                  "ByteArray::lend: The argument at position 1 in ByteArray[...] should be a vector of unsigned byte values or a Base64-encoded string.",
                );
                return Some(Ok(Expr::FunctionCall {
                  name: "ByteArray".to_string(),
                  args: args.to_vec(),
                }));
              }
            }
          }
          return Some(Ok(Expr::FunctionCall {
            name: "ByteArray".to_string(),
            args: vec![Expr::List(bytes)],
          }));
        }
        Expr::String(s) => {
          // Decode base64 string to byte list
          use base64::Engine;
          let engine = base64::engine::general_purpose::STANDARD;
          match engine.decode(s) {
            Ok(decoded) => {
              let bytes: Vec<Expr> =
                decoded.iter().map(|b| Expr::Integer(*b as i128)).collect();
              return Some(Ok(Expr::FunctionCall {
                name: "ByteArray".to_string(),
                args: vec![Expr::List(bytes)],
              }));
            }
            Err(_) => {
              crate::emit_message(
                "ByteArray::lend: The argument at position 1 in ByteArray[...] should be a vector of unsigned byte values or a Base64-encoded string.",
              );
              return Some(Ok(Expr::FunctionCall {
                name: "ByteArray".to_string(),
                args: args.to_vec(),
              }));
            }
          }
        }
        _ => {
          crate::emit_message(&format!(
            "ByteArray::lend: The argument at position 1 in ByteArray[{}] should be a vector of unsigned byte values or a Base64-encoded string.",
            crate::syntax::expr_to_string(&args[0])
          ));
          return Some(Ok(Expr::FunctionCall {
            name: "ByteArray".to_string(),
            args: args.to_vec(),
          }));
        }
      }
    }
    // CensoredDistribution[{min, max}, dist] — censored distribution
    "CensoredDistribution" if args.len() == 2 => {
      // Evaluate the underlying distribution to normalize it
      let dist = crate::evaluator::evaluate_expr_to_expr(&args[1])
        .unwrap_or_else(|_| args[1].clone());
      let bounds = crate::evaluator::evaluate_expr_to_expr(&args[0])
        .unwrap_or_else(|_| args[0].clone());
      return Some(Ok(Expr::FunctionCall {
        name: "CensoredDistribution".to_string(),
        args: vec![bounds, dist],
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
    "Piecewise" if !args.is_empty() && args.len() <= 2 => {
      return Some(crate::functions::control_flow_ast::piecewise_ast(args));
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
        crate::emit_message(&format!(
          "If::argb: If called with {} arguments; between 2 and 4 arguments are expected.",
          args.len()
        ));
      }
    }
    _ => {}
  }
  None
}
