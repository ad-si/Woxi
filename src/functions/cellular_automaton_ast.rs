use crate::InterpreterError;
use crate::syntax::Expr;

/// CellularAutomaton[rule, init, steps]
/// Two forms for init:
/// 1. {{center_cells...}, background} - grows by `steps` cells on each side
/// 2. {cell1, cell2, ...} - fixed-width with periodic boundaries
pub fn cellular_automaton_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "CellularAutomaton".to_string(),
      args: args.to_vec(),
    });
  }

  let rule_num = match &args[0] {
    Expr::Integer(n) => *n as u8,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CellularAutomaton".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let steps = match &args[2] {
    Expr::Integer(n) => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CellularAutomaton".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Determine init form
  let (init_cells, mode) = parse_init(&args[1])?;

  let rows = match mode {
    InitMode::Expanding { background } => {
      evolve_expanding(rule_num, &init_cells, background, steps)
    }
    InitMode::Periodic => evolve_periodic(rule_num, &init_cells, steps),
  };

  let result: Vec<Expr> = rows
    .into_iter()
    .map(|row| {
      Expr::List(row.into_iter().map(|c| Expr::Integer(c as i128)).collect())
    })
    .collect();

  Ok(Expr::List(result))
}

enum InitMode {
  Expanding { background: u8 },
  Periodic,
}

fn parse_init(expr: &Expr) -> Result<(Vec<u8>, InitMode), InterpreterError> {
  if let Expr::List(items) = expr {
    if items.len() == 2
      && let Expr::List(center) = &items[0]
    {
      // Form: {{center_cells...}, background}
      let bg = match &items[1] {
        Expr::Integer(n) => *n as u8,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "CellularAutomaton: background must be an integer".to_string(),
          ));
        }
      };
      let cells: Result<Vec<u8>, _> = center
        .iter()
        .map(|e| match e {
          Expr::Integer(n) => Ok(*n as u8),
          _ => Err(InterpreterError::EvaluationError(
            "CellularAutomaton: cells must be integers".to_string(),
          )),
        })
        .collect();
      return Ok((cells?, InitMode::Expanding { background: bg }));
    }
    // Form: {cell1, cell2, ...} - periodic
    let cells: Result<Vec<u8>, _> = items
      .iter()
      .map(|e| match e {
        Expr::Integer(n) => Ok(*n as u8),
        _ => Err(InterpreterError::EvaluationError(
          "CellularAutomaton: cells must be integers".to_string(),
        )),
      })
      .collect();
    Ok((cells?, InitMode::Periodic))
  } else {
    Err(InterpreterError::EvaluationError(
      "CellularAutomaton: init must be a list".to_string(),
    ))
  }
}

fn apply_rule(rule: u8, left: u8, center: u8, right: u8) -> u8 {
  let index = (left << 2) | (center << 1) | right;
  (rule >> index) & 1
}

fn evolve_expanding(
  rule: u8,
  center_cells: &[u8],
  background: u8,
  steps: usize,
) -> Vec<Vec<u8>> {
  // Total width = len(center_cells) + 2*steps
  let width = center_cells.len() + 2 * steps;
  let offset = steps;

  // Build initial row
  let mut current = vec![background; width];
  for (i, &c) in center_cells.iter().enumerate() {
    current[offset + i] = c;
  }

  let mut rows = vec![current.clone()];

  for _ in 0..steps {
    let mut next = vec![background; width];
    for j in 0..width {
      let left = if j == 0 { background } else { current[j - 1] };
      let center = current[j];
      let right = if j == width - 1 {
        background
      } else {
        current[j + 1]
      };
      next[j] = apply_rule(rule, left, center, right);
    }
    current = next;
    rows.push(current.clone());
  }

  // Trim all-background columns from left and right
  trim_background_columns(&mut rows, background)
}

fn trim_background_columns(
  rows: &mut Vec<Vec<u8>>,
  background: u8,
) -> Vec<Vec<u8>> {
  if rows.is_empty() || rows[0].is_empty() {
    return rows.clone();
  }
  let width = rows[0].len();

  // Find leftmost column that has a non-background cell in any row
  let left = (0..width)
    .find(|&col| rows.iter().any(|row| row[col] != background))
    .unwrap_or(0);

  // Find rightmost column that has a non-background cell in any row
  let right = (0..width)
    .rfind(|&col| rows.iter().any(|row| row[col] != background))
    .unwrap_or(0);

  rows.iter().map(|row| row[left..=right].to_vec()).collect()
}

fn evolve_periodic(rule: u8, init: &[u8], steps: usize) -> Vec<Vec<u8>> {
  let width = init.len();
  let mut current = init.to_vec();
  let mut rows = vec![current.clone()];

  for _ in 0..steps {
    let mut next = vec![0u8; width];
    for j in 0..width {
      let left = current[(j + width - 1) % width];
      let center = current[j];
      let right = current[(j + 1) % width];
      next[j] = apply_rule(rule, left, center, right);
    }
    current = next;
    rows.push(current.clone());
  }

  rows
}
