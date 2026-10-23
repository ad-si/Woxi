#[allow(unused_imports)]
use super::*;
use crate::evaluator::evaluate_expr_to_expr;

/// InterpolatingPolynomial[data, x] — find the polynomial that interpolates the data.
///
/// data can be:
///   {{x1,y1},{x2,y2},...} — explicit x,y pairs
///   {y1,y2,...}           — y values at x = 1,2,3,...
///   {{{x1},f1,df1,...},{{x2},f2,...},...} — values with derivatives (Hermite)
///
/// Uses Newton's divided differences form.
pub fn interpolating_polynomial_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("InterpolatingPolynomial", args));
  }

  let data = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(unevaluated("InterpolatingPolynomial", args));
    }
  };

  if data.is_empty() {
    return Ok(Expr::Integer(0));
  }

  // A variable given as `{x}` is the same as `x` for one-dimensional data.
  let var = match &args[1] {
    Expr::List(v) if v.len() == 1 => &v[0],
    other => other,
  };

  // Each node is (x, [f, f', f'', ...]). Supported layouts:
  //   {{x1,y1},{x2,y2},...}              — values only
  //   {{{x1},f1,df1,...},{{x2},f2},...}  — values with derivatives (Hermite)
  //   {y1,y2,...}                        — y values at x = 1,2,3,...
  let mut nodes: Vec<(Expr, Vec<Expr>)> = Vec::new();
  for (idx, item) in data.iter().enumerate() {
    match item {
      Expr::List(entry)
        if entry.len() == 2 && !matches!(entry[0], Expr::List(_)) =>
      {
        nodes.push((entry[0].clone(), vec![entry[1].clone()]));
      }
      Expr::List(entry)
        if entry.len() >= 2
          && matches!(&entry[0], Expr::List(xs) if xs.len() == 1) =>
      {
        let Expr::List(xs) = &entry[0] else {
          unreachable!()
        };
        nodes.push((xs[0].clone(), entry[1..].to_vec()));
      }
      Expr::List(_) => {
        return Ok(unevaluated("InterpolatingPolynomial", args));
      }
      _ if idx == 0 || nodes.iter().all(|(_, d)| d.len() == 1) => {
        nodes.push((Expr::Integer(idx as i128 + 1), vec![item.clone()]));
      }
      _ => return Ok(unevaluated("InterpolatingPolynomial", args)),
    }
  }

  // Repeat every node once per supplied derivative order, so the Newton
  // divided-difference table also covers confluent (Hermite) data.
  let mut x_vals: Vec<Expr> = Vec::new();
  let mut node_of: Vec<usize> = Vec::new();
  for (k, (x, derivs)) in nodes.iter().enumerate() {
    for _ in 0..derivs.len() {
      x_vals.push(x.clone());
      node_of.push(k);
    }
  }
  let n = x_vals.len();

  // coef[i] = f[z0,...,zi]; column j of the table is built from column j-1.
  let mut column: Vec<Expr> =
    node_of.iter().map(|&k| nodes[k].1[0].clone()).collect();
  let mut dd: Vec<Expr> = vec![column[0].clone()];
  for j in 1..n {
    let mut next = vec![Expr::Integer(0); n];
    for i in j..n {
      next[i] = if node_of[i] == node_of[i - j] {
        // Confluent nodes: the j-th derivative divided by j!.
        let k = node_of[i];
        let Some(d) = nodes[k].1.get(j) else {
          return Ok(unevaluated("InterpolatingPolynomial", args));
        };
        let fact = (1..=j as i128).product::<i128>();
        evaluate_expr_to_expr(&div2(d.clone(), Expr::Integer(fact)))?
      } else {
        let numer = plus(vec![
          column[i].clone(),
          times(vec![Expr::Integer(-1), column[i - 1].clone()]),
        ]);
        let denom = plus(vec![
          x_vals[i].clone(),
          times(vec![Expr::Integer(-1), x_vals[i - j].clone()]),
        ]);
        evaluate_expr_to_expr(&div2(numer, denom))?
      };
    }
    dd.push(next[j].clone());
    column = next;
  }

  // Build Newton form: dd[0] + dd[1]*(x-x0) + dd[2]*(x-x0)*(x-x1) + ...
  // We build it in nested form: dd[0] + (x-x0)*(dd[1] + (x-x1)*(dd[2] + ...))
  let mut result = dd[n - 1].clone();

  for i in (0..n - 1).rev() {
    // result = dd[i] + (x - x_vals[i]) * result
    let x_minus_xi = plus(vec![var.clone(), neg(x_vals[i].clone())]);
    let product = times(vec![x_minus_xi, result]);
    result = plus(vec![dd[i].clone(), product]);
  }

  evaluate_expr_to_expr(&result)
}
