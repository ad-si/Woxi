use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::syntax::{BinaryOperator, Expr};

/// InterpolatingPolynomial[data, x] — find the polynomial that interpolates the data.
///
/// data can be:
///   {{x1,y1},{x2,y2},...} — explicit x,y pairs
///   {y1,y2,...}           — y values at x = 1,2,3,...
///
/// Uses Newton's divided differences form.
pub fn interpolating_polynomial_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "InterpolatingPolynomial".to_string(),
      args: args.to_vec(),
    });
  }

  let data = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "InterpolatingPolynomial".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if data.is_empty() {
    return Ok(Expr::Integer(0));
  }

  let var = &args[1];

  // Determine if data is {{x,y},...} or {y,...}
  let (x_vals, y_vals): (Vec<Expr>, Vec<Expr>) =
    if let Some(Expr::List(pair)) = data.first() {
      if pair.len() == 2 {
        // {{x1,y1},{x2,y2},...}
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for item in &data {
          if let Expr::List(pair) = item {
            if pair.len() == 2 {
              xs.push(pair[0].clone());
              ys.push(pair[1].clone());
            } else {
              return Ok(Expr::FunctionCall {
                name: "InterpolatingPolynomial".to_string(),
                args: args.to_vec(),
              });
            }
          } else {
            return Ok(Expr::FunctionCall {
              name: "InterpolatingPolynomial".to_string(),
              args: args.to_vec(),
            });
          }
        }
        (xs, ys)
      } else {
        return Ok(Expr::FunctionCall {
          name: "InterpolatingPolynomial".to_string(),
          args: args.to_vec(),
        });
      }
    } else {
      // {y1,y2,...} — x values are 1,2,3,...
      let xs: Vec<Expr> = (1..=data.len() as i128).map(Expr::Integer).collect();
      (xs, data)
    };

  let n = x_vals.len();

  // Compute Newton's divided differences table
  // dd[i] = f[x0,...,xi]
  let mut dd: Vec<Expr> = y_vals;

  for j in 1..n {
    for i in (j..n).rev() {
      // dd[i] = (dd[i] - dd[i-1]) / (x_vals[i] - x_vals[i-j])
      let numer = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          dd[i].clone(),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), dd[i - 1].clone()],
          },
        ],
      };
      let denom = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          x_vals[i].clone(),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), x_vals[i - j].clone()],
          },
        ],
      };
      let divided = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(numer),
        right: Box::new(denom),
      };
      dd[i] = evaluate_expr_to_expr(&divided)?;
    }
  }

  // Build Newton form: dd[0] + dd[1]*(x-x0) + dd[2]*(x-x0)*(x-x1) + ...
  // We build it in nested form: dd[0] + (x-x0)*(dd[1] + (x-x1)*(dd[2] + ...))
  let mut result = dd[n - 1].clone();

  for i in (0..n - 1).rev() {
    // result = dd[i] + (x - x_vals[i]) * result
    let x_minus_xi = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        var.clone(),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), x_vals[i].clone()],
        },
      ],
    };
    let product = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![x_minus_xi, result],
    };
    result = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![dd[i].clone(), product],
    };
  }

  evaluate_expr_to_expr(&result)
}
