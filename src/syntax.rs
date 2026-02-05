#[derive(Debug, Clone, Copy)]
pub enum WoNum {
  Int(i128),
  Float(f64),
}

impl std::ops::Add for WoNum {
  type Output = Self;

  fn add(self, rhs: Self) -> Self {
    match (self, rhs) {
      (Self::Int(a), Self::Int(b)) => Self::Int(a + b),
      (Self::Float(a), Self::Float(b)) => Self::Float(a + b),
      (Self::Float(a), Self::Int(b)) => Self::Float(a + b as f64),
      (Self::Int(a), Self::Float(b)) => Self::Float(a as f64 + b),
    }
  }
}

impl std::ops::Mul for WoNum {
  type Output = Self;

  fn mul(self, rhs: Self) -> Self {
    match (self, rhs) {
      (Self::Int(a), Self::Int(b)) => Self::Int(a * b),
      (Self::Float(a), Self::Float(b)) => Self::Float(a * b),
      (Self::Float(a), Self::Int(b)) => Self::Float(a * b as f64),
      (Self::Int(a), Self::Float(b)) => Self::Float(a as f64 * b),
    }
  }
}

impl std::ops::Sub for WoNum {
  type Output = Self;

  fn sub(self, rhs: Self) -> Self {
    match (self, rhs) {
      (Self::Int(a), Self::Int(b)) => Self::Int(a - b),
      (Self::Float(a), Self::Float(b)) => Self::Float(a - b),
      (Self::Float(a), Self::Int(b)) => Self::Float(a - b as f64),
      (Self::Int(a), Self::Float(b)) => Self::Float(a as f64 - b),
    }
  }
}

impl std::ops::Div for WoNum {
  type Output = Self;

  fn div(self, rhs: Self) -> Self {
    // Division always produces a float to match Wolfram Language behavior
    let lhs_f64 = match self {
      Self::Int(i) => i as f64,
      Self::Float(f) => f,
    };
    let rhs_f64 = match rhs {
      Self::Int(i) => i as f64,
      Self::Float(f) => f,
    };
    Self::Float(lhs_f64 / rhs_f64)
  }
}

impl std::ops::Neg for WoNum {
  type Output = Self;

  fn neg(self) -> Self {
    match self {
      Self::Int(i) => Self::Int(-i),
      Self::Float(f) => Self::Float(-f),
    }
  }
}

impl std::iter::Sum for WoNum {
  fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
    iter.fold(Self::Int(0), |a, b| a + b)
  }
}

impl std::iter::FromIterator<WoNum> for WoNum {
  fn from_iter<I: IntoIterator<Item = Self>>(iter: I) -> Self {
    let mut sum_i128 = 0i128;
    let mut sum_f64 = 0.0;

    for num in iter {
      match num {
        WoNum::Int(i) => {
          sum_i128 += i;
        }
        WoNum::Float(f) => {
          sum_f64 += f;
        }
      }
    }

    if sum_f64 != 0.0
      || (sum_i128 != 0i128
        && std::mem::size_of::<f64>() < std::mem::size_of::<i128>())
    {
      WoNum::Float(sum_f64)
    } else {
      WoNum::Int(sum_i128)
    }
  }
}

pub fn wonum_to_number_str(wo_num: WoNum) -> String {
  match wo_num {
    WoNum::Int(x) => x.to_string(),
    WoNum::Float(x) => x.to_string(),
  }
}

pub fn str_to_wonum(num_str: &str) -> WoNum {
  num_str
    .parse::<i128>()
    .map(WoNum::Int)
    .or(num_str.parse::<f64>().map(WoNum::Float))
    .unwrap()
}

impl WoNum {
  pub fn abs(self) -> Self {
    match self {
      WoNum::Int(i) => WoNum::Int(i.abs()),
      WoNum::Float(f) => WoNum::Float(f.abs()),
    }
  }

  pub fn sign(&self) -> i8 {
    match self {
      WoNum::Int(i) => match i.cmp(&0) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
      },
      WoNum::Float(f) => {
        if *f > 0.0 {
          1
        } else if *f < 0.0 {
          -1
        } else {
          0
        }
      }
    }
  }

  pub fn sqrt(self) -> Result<Self, String> {
    let val = match self {
      WoNum::Int(i) => {
        if i < 0 {
          return Err("Sqrt function argument must be non-negative".into());
        }
        i as f64
      }
      WoNum::Float(f) => {
        if f < 0.0 {
          return Err("Sqrt function argument must be non-negative".into());
        }
        f
      }
    };
    Ok(WoNum::Float(val.sqrt()))
  }

  pub fn floor(self) -> Self {
    match self {
      WoNum::Int(i) => WoNum::Int(i),
      WoNum::Float(f) => {
        let result = f.floor();
        // Handle -0.0
        if result == -0.0 {
          WoNum::Float(0.0)
        } else {
          WoNum::Float(result)
        }
      }
    }
  }

  pub fn ceiling(self) -> Self {
    match self {
      WoNum::Int(i) => WoNum::Int(i),
      WoNum::Float(f) => {
        let result = f.ceil();
        // Handle -0.0
        if result == -0.0 {
          WoNum::Float(0.0)
        } else {
          WoNum::Float(result)
        }
      }
    }
  }

  pub fn round(self) -> Self {
    match self {
      WoNum::Int(i) => WoNum::Int(i),
      WoNum::Float(f) => {
        // Banker's rounding (half-to-even)
        let base = f.trunc();
        let frac = f - base;
        let result = if frac.abs() == 0.5 {
          if (base as i64) % 2 == 0 {
            base
          } else if f.is_sign_positive() {
            base + 1.0
          } else {
            base - 1.0
          }
        } else {
          f.round()
        };
        // Handle -0.0
        if result == -0.0 {
          WoNum::Float(0.0)
        } else {
          WoNum::Float(result)
        }
      }
    }
  }
}

#[derive(Debug)]
pub enum AST {
  Plus(Vec<WoNum>),
  Times(Vec<WoNum>),
  Minus(Vec<WoNum>),
  Divide(Vec<WoNum>),
  Abs(WoNum),
  Sign(WoNum),
  Sqrt(WoNum),
  Floor(WoNum),
  Ceiling(WoNum),
  Round(WoNum),
  CreateFile(Option<String>),
}

/// Owned expression tree for storing parsed function bodies.
/// This avoids re-parsing function bodies on every call.
#[derive(Debug, Clone)]
pub enum Expr {
  /// Integer literal
  Integer(i128),
  /// Real/float literal
  Real(f64),
  /// String literal (without quotes)
  String(String),
  /// Identifier/symbol
  Identifier(String),
  /// Slot (#, #1, #2, etc.)
  Slot(usize),
  /// List: {e1, e2, ...}
  List(Vec<Expr>),
  /// Function call: f[e1, e2, ...]
  FunctionCall { name: String, args: Vec<Expr> },
  /// Binary operator: e1 op e2
  BinaryOp {
    op: BinaryOperator,
    left: Box<Expr>,
    right: Box<Expr>,
  },
  /// Unary operator: op e
  UnaryOp {
    op: UnaryOperator,
    operand: Box<Expr>,
  },
  /// Comparison chain: e1 op1 e2 op2 e3 ...
  Comparison {
    operands: Vec<Expr>,
    operators: Vec<ComparisonOp>,
  },
  /// Compound expression: e1; e2; e3
  CompoundExpr(Vec<Expr>),
  /// Association: <| key1 -> val1, key2 -> val2, ... |>
  Association(Vec<(Expr, Expr)>),
  /// Rule: pattern -> replacement
  Rule {
    pattern: Box<Expr>,
    replacement: Box<Expr>,
  },
  /// Delayed rule: pattern :> replacement
  RuleDelayed {
    pattern: Box<Expr>,
    replacement: Box<Expr>,
  },
  /// ReplaceAll: expr /. rules
  ReplaceAll { expr: Box<Expr>, rules: Box<Expr> },
  /// ReplaceRepeated: expr //. rules
  ReplaceRepeated { expr: Box<Expr>, rules: Box<Expr> },
  /// Map: f /@ list
  Map { func: Box<Expr>, list: Box<Expr> },
  /// Apply: f @@ list
  Apply { func: Box<Expr>, list: Box<Expr> },
  /// Postfix application: expr // f
  Postfix { expr: Box<Expr>, func: Box<Expr> },
  /// Part extraction: expr[[index]]
  Part { expr: Box<Expr>, index: Box<Expr> },
  /// Anonymous function: body &
  Function { body: Box<Expr> },
  /// Pattern: name_
  Pattern { name: String, head: Option<String> },
  /// Constant like Pi, E, etc.
  Constant(String),
  /// Raw unparsed text (fallback)
  Raw(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
  Plus,
  Minus,
  Times,
  Divide,
  Power,
  And,
  Or,
  StringJoin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
  Minus,
  Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
  Equal,        // ==
  NotEqual,     // !=
  Less,         // <
  LessEqual,    // <=
  Greater,      // >
  GreaterEqual, // >=
  SameQ,        // ===
  UnsameQ,      // =!=
}

use crate::Rule;
use pest::iterators::Pair;

/// Convert a pest Pair to an owned Expr AST.
/// This is used to store function bodies without re-parsing.
pub fn pair_to_expr(pair: Pair<Rule>) -> Expr {
  match pair.as_rule() {
    Rule::Integer => {
      let s = pair.as_str();
      Expr::Integer(s.parse().unwrap_or(0))
    }
    Rule::Real => {
      let s = pair.as_str();
      Expr::Real(s.parse().unwrap_or(0.0))
    }
    Rule::String => {
      let s = pair.as_str();
      // Remove surrounding quotes
      Expr::String(s[1..s.len() - 1].to_string())
    }
    Rule::Identifier => Expr::Identifier(pair.as_str().to_string()),
    Rule::Slot => {
      let s = pair.as_str();
      // # is slot 1, #1 is slot 1, #2 is slot 2, etc.
      let num = if s.len() > 1 {
        s[1..].parse().unwrap_or(1)
      } else {
        1
      };
      Expr::Slot(num)
    }
    Rule::Constant => Expr::Constant(pair.as_str().to_string()),
    Rule::NumericValue => {
      let inner = pair.into_inner().next().unwrap();
      pair_to_expr(inner)
    }
    Rule::List => {
      let items: Vec<Expr> = pair
        .into_inner()
        .filter(|p| p.as_str() != ",")
        .map(pair_to_expr)
        .collect();
      Expr::List(items)
    }
    Rule::FunctionCall => {
      let mut inner = pair.into_inner();
      let name_pair = inner.next().unwrap();
      let name = name_pair.as_str().to_string();
      let args: Vec<Expr> = inner
        .filter(|p| p.as_str() != ",")
        .map(pair_to_expr)
        .collect();
      Expr::FunctionCall { name, args }
    }
    Rule::BaseFunctionCall => {
      let mut inner = pair.into_inner();
      let name_pair = inner.next().unwrap();
      let name = name_pair.as_str().to_string();
      let args: Vec<Expr> = inner
        .filter(|p| p.as_str() != ",")
        .map(pair_to_expr)
        .collect();
      Expr::FunctionCall { name, args }
    }
    Rule::Expression | Rule::ExpressionNoImplicit => parse_expression(pair),
    Rule::CompoundExpression => {
      let exprs: Vec<Expr> = pair
        .into_inner()
        .filter(|p| p.as_str() != ";")
        .map(pair_to_expr)
        .collect();
      if exprs.len() == 1 {
        exprs.into_iter().next().unwrap()
      } else {
        Expr::CompoundExpr(exprs)
      }
    }
    Rule::Association => {
      let items: Vec<(Expr, Expr)> = pair
        .into_inner()
        .filter(|p| p.as_rule() == Rule::AssociationItem)
        .map(|item| {
          let mut inner = item.into_inner();
          let key = pair_to_expr(inner.next().unwrap());
          let val = pair_to_expr(inner.next().unwrap());
          (key, val)
        })
        .collect();
      Expr::Association(items)
    }
    Rule::AssociationItem => {
      let mut inner = pair.into_inner();
      let key = pair_to_expr(inner.next().unwrap());
      let val = pair_to_expr(inner.next().unwrap());
      Expr::Rule {
        pattern: Box::new(key),
        replacement: Box::new(val),
      }
    }
    Rule::ReplacementRule => {
      let full_str = pair.as_str();
      let is_delayed = full_str.contains(":>");
      let mut inner = pair.into_inner();
      let pattern = pair_to_expr(inner.next().unwrap());
      let replacement = pair_to_expr(inner.next().unwrap());
      if is_delayed {
        Expr::RuleDelayed {
          pattern: Box::new(pattern),
          replacement: Box::new(replacement),
        }
      } else {
        Expr::Rule {
          pattern: Box::new(pattern),
          replacement: Box::new(replacement),
        }
      }
    }
    Rule::PatternSimple => {
      let s = pair.as_str();
      let name = s.trim_end_matches('_').to_string();
      Expr::Pattern { name, head: None }
    }
    Rule::PatternWithHead => {
      let mut inner = pair.into_inner();
      let name = inner.next().unwrap().as_str().to_string();
      let head = inner.next().map(|p| p.as_str().to_string());
      Expr::Pattern { name, head }
    }
    Rule::PatternTest | Rule::PatternCondition => {
      // For now, just store as pattern with name
      let s = pair.as_str();
      let name = s.split('_').next().unwrap_or(s).to_string();
      Expr::Pattern { name, head: None }
    }
    Rule::SimpleAnonymousFunction
    | Rule::FunctionAnonymousFunction
    | Rule::ParenAnonymousFunction
    | Rule::ListAnonymousFunction => {
      // Anonymous function like #^2& or If[#>0,#,0]&
      let s = pair.as_str().trim().trim_end_matches('&');
      // Parse the body
      let body = parse_anonymous_body(s);
      Expr::Function {
        body: Box::new(body),
      }
    }
    Rule::PartExtract => {
      let mut inner = pair.into_inner();
      let expr = pair_to_expr(inner.next().unwrap());
      let index = pair_to_expr(inner.next().unwrap());
      Expr::Part {
        expr: Box::new(expr),
        index: Box::new(index),
      }
    }
    Rule::ImplicitTimes => {
      // Implicit multiplication: x y z -> Times[x, y, z]
      let factors: Vec<Expr> = pair.into_inner().map(pair_to_expr).collect();
      if factors.len() == 1 {
        factors.into_iter().next().unwrap()
      } else {
        // Build nested Times
        let mut iter = factors.into_iter();
        let first = iter.next().unwrap();
        iter.fold(first, |acc, f| Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(acc),
          right: Box::new(f),
        })
      }
    }
    Rule::PostfixApplication => {
      let mut inner: Vec<_> = pair.into_inner().collect();
      if inner.is_empty() {
        return Expr::Raw(String::new());
      }
      let mut result = pair_to_expr(inner.remove(0));
      for func_pair in inner {
        let func = pair_to_expr(func_pair);
        result = Expr::Postfix {
          expr: Box::new(result),
          func: Box::new(func),
        };
      }
      result
    }
    Rule::PostfixBase => {
      let inner = pair.into_inner().next().unwrap();
      pair_to_expr(inner)
    }
    Rule::PostfixFunction => {
      let inner = pair.into_inner().next().unwrap();
      pair_to_expr(inner)
    }
    Rule::ReplaceAllExpr => {
      let mut inner = pair.into_inner();
      let expr = pair_to_expr(inner.next().unwrap());
      let rules = pair_to_expr(inner.next().unwrap());
      Expr::ReplaceAll {
        expr: Box::new(expr),
        rules: Box::new(rules),
      }
    }
    Rule::ReplaceRepeatedExpr => {
      let mut inner = pair.into_inner();
      let expr = pair_to_expr(inner.next().unwrap());
      let rules = pair_to_expr(inner.next().unwrap());
      Expr::ReplaceRepeated {
        expr: Box::new(expr),
        rules: Box::new(rules),
      }
    }
    _ => {
      // Fallback: store as raw text
      Expr::Raw(pair.as_str().to_string())
    }
  }
}

/// Parse an expression with operators into an Expr
fn parse_expression(pair: Pair<Rule>) -> Expr {
  let mut inner: Vec<Pair<Rule>> = pair.into_inner().collect();

  if inner.is_empty() {
    return Expr::Raw(String::new());
  }

  // Single term case
  if inner.len() == 1 {
    return pair_to_expr(inner.remove(0));
  }

  // Check for postfix application pattern: Term (// PostfixFunction)+
  // The grammar puts PostfixFunction after Term when // is used
  let mut i = 1;
  while i < inner.len() {
    if inner[i].as_rule() == Rule::PostfixFunction {
      // This is a postfix chain
      let mut result = pair_to_expr(inner.remove(0));
      while !inner.is_empty() && inner[0].as_rule() == Rule::PostfixFunction {
        let func = pair_to_expr(inner.remove(0));
        result = Expr::Postfix {
          expr: Box::new(result),
          func: Box::new(func),
        };
      }
      return result;
    }
    i += 1;
  }

  // Check for ReplaceAll/ReplaceRepeated patterns
  // These show up as: Term (List|ReplacementRule) (PostfixFunction)*
  if inner.len() >= 2 {
    let second_rule = inner[1].as_rule();
    if second_rule == Rule::List || second_rule == Rule::ReplacementRule {
      // Check if this is ReplaceAll or ReplaceRepeated by looking at remaining structure
      let expr = pair_to_expr(inner.remove(0));
      let rules = pair_to_expr(inner.remove(0));
      let mut result = Expr::ReplaceAll {
        expr: Box::new(expr),
        rules: Box::new(rules),
      };
      // Apply any remaining postfix functions
      while !inner.is_empty() {
        let func = pair_to_expr(inner.remove(0));
        result = Expr::Postfix {
          expr: Box::new(result),
          func: Box::new(func),
        };
      }
      return result;
    }
  }

  // Parse operators: Term (Operator Term)*
  // Build expression with proper precedence
  let mut terms: Vec<Expr> = Vec::new();
  let mut operators: Vec<String> = Vec::new();

  for item in inner {
    match item.as_rule() {
      Rule::Operator => {
        operators.push(item.as_str().to_string());
      }
      _ => {
        terms.push(pair_to_expr(item));
      }
    }
  }

  if terms.len() == 1 {
    return terms.remove(0);
  }

  // Check if all operators are comparison operators
  let all_comparisons = operators.iter().all(|op| {
    matches!(
      op.as_str(),
      "==" | "!=" | "<" | "<=" | ">" | ">=" | "===" | "=!="
    )
  });

  if all_comparisons && !operators.is_empty() {
    // Build comparison chain
    let comp_ops: Vec<ComparisonOp> = operators
      .iter()
      .map(|op| match op.as_str() {
        "==" => ComparisonOp::Equal,
        "!=" => ComparisonOp::NotEqual,
        "<" => ComparisonOp::Less,
        "<=" => ComparisonOp::LessEqual,
        ">" => ComparisonOp::Greater,
        ">=" => ComparisonOp::GreaterEqual,
        "===" => ComparisonOp::SameQ,
        "=!=" => ComparisonOp::UnsameQ,
        _ => ComparisonOp::Equal,
      })
      .collect();
    return Expr::Comparison {
      operands: terms,
      operators: comp_ops,
    };
  }

  // Build binary operation tree (left-to-right for same precedence)
  // For simplicity, we don't implement full precedence here since
  // the pest parser already handles grouping
  build_binary_tree(terms, operators)
}

/// Build a binary operation tree from terms and operators
fn build_binary_tree(mut terms: Vec<Expr>, operators: Vec<String>) -> Expr {
  if terms.len() == 1 {
    return terms.remove(0);
  }

  let mut result = terms.remove(0);
  for op_str in operators.into_iter() {
    if terms.is_empty() {
      break;
    }
    let right = terms.remove(0);

    let op = match op_str.as_str() {
      "+" => Some(BinaryOperator::Plus),
      "-" => Some(BinaryOperator::Minus),
      "*" => Some(BinaryOperator::Times),
      "/" => Some(BinaryOperator::Divide),
      "^" => Some(BinaryOperator::Power),
      "&&" => Some(BinaryOperator::And),
      "||" => Some(BinaryOperator::Or),
      "<>" => Some(BinaryOperator::StringJoin),
      "/@" => {
        result = Expr::Map {
          func: Box::new(result),
          list: Box::new(right),
        };
        continue;
      }
      "@@" => {
        result = Expr::Apply {
          func: Box::new(result),
          list: Box::new(right),
        };
        continue;
      }
      "->" => {
        result = Expr::Rule {
          pattern: Box::new(result),
          replacement: Box::new(right),
        };
        continue;
      }
      ":>" => {
        result = Expr::RuleDelayed {
          pattern: Box::new(result),
          replacement: Box::new(right),
        };
        continue;
      }
      _ => None,
    };

    if let Some(binary_op) = op {
      result = Expr::BinaryOp {
        op: binary_op,
        left: Box::new(result),
        right: Box::new(right),
      };
    } else {
      // Unknown operator, wrap as raw
      result = Expr::Raw(format!(
        "{} {} {}",
        expr_to_string(&result),
        op_str,
        expr_to_string(&right)
      ));
    }
  }

  result
}

/// Parse the body of an anonymous function
fn parse_anonymous_body(s: &str) -> Expr {
  // This is a simplified parser for anonymous function bodies
  // It handles common cases like #, #^2, #+1, etc.
  let s = s.trim();

  if s.is_empty() {
    return Expr::Slot(1);
  }

  // Check for slot
  if s == "#" {
    return Expr::Slot(1);
  }
  if s.starts_with('#')
    && s.len() > 1
    && s[1..].chars().all(|c| c.is_ascii_digit())
  {
    let num: usize = s[1..].parse().unwrap_or(1);
    return Expr::Slot(num);
  }

  // For more complex bodies, store as raw
  Expr::Raw(s.to_string())
}

/// Convert an Expr back to a string representation
pub fn expr_to_string(expr: &Expr) -> String {
  match expr {
    Expr::Integer(n) => n.to_string(),
    Expr::Real(f) => f.to_string(),
    Expr::String(s) => format!("\"{}\"", s),
    Expr::Identifier(s) => s.clone(),
    Expr::Slot(n) => {
      if *n == 1 {
        "#".to_string()
      } else {
        format!("#{}", n)
      }
    }
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_string).collect();
      format!("{{{}}}", parts.join(", "))
    }
    Expr::FunctionCall { name, args } => {
      let parts: Vec<String> = args.iter().map(expr_to_string).collect();
      format!("{}[{}]", name, parts.join(", "))
    }
    Expr::BinaryOp { op, left, right } => {
      let op_str = match op {
        BinaryOperator::Plus => "+",
        BinaryOperator::Minus => "-",
        BinaryOperator::Times => "*",
        BinaryOperator::Divide => "/",
        BinaryOperator::Power => "^",
        BinaryOperator::And => "&&",
        BinaryOperator::Or => "||",
        BinaryOperator::StringJoin => "<>",
      };
      format!(
        "{} {} {}",
        expr_to_string(left),
        op_str,
        expr_to_string(right)
      )
    }
    Expr::UnaryOp { op, operand } => {
      let op_str = match op {
        UnaryOperator::Minus => "-",
        UnaryOperator::Not => "!",
      };
      format!("{}{}", op_str, expr_to_string(operand))
    }
    Expr::Comparison {
      operands,
      operators,
    } => {
      let mut result = expr_to_string(&operands[0]);
      for (i, op) in operators.iter().enumerate() {
        let op_str = match op {
          ComparisonOp::Equal => "==",
          ComparisonOp::NotEqual => "!=",
          ComparisonOp::Less => "<",
          ComparisonOp::LessEqual => "<=",
          ComparisonOp::Greater => ">",
          ComparisonOp::GreaterEqual => ">=",
          ComparisonOp::SameQ => "===",
          ComparisonOp::UnsameQ => "=!=",
        };
        if i + 1 < operands.len() {
          result = format!(
            "{} {} {}",
            result,
            op_str,
            expr_to_string(&operands[i + 1])
          );
        }
      }
      result
    }
    Expr::CompoundExpr(exprs) => {
      let parts: Vec<String> = exprs.iter().map(expr_to_string).collect();
      parts.join("; ")
    }
    Expr::Association(items) => {
      let parts: Vec<String> = items
        .iter()
        .map(|(k, v)| format!("{} -> {}", expr_to_string(k), expr_to_string(v)))
        .collect();
      format!("<|{}|>", parts.join(", "))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => {
      format!(
        "{} -> {}",
        expr_to_string(pattern),
        expr_to_string(replacement)
      )
    }
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      format!(
        "{} :> {}",
        expr_to_string(pattern),
        expr_to_string(replacement)
      )
    }
    Expr::ReplaceAll { expr, rules } => {
      format!("{} /. {}", expr_to_string(expr), expr_to_string(rules))
    }
    Expr::ReplaceRepeated { expr, rules } => {
      format!("{} //. {}", expr_to_string(expr), expr_to_string(rules))
    }
    Expr::Map { func, list } => {
      format!("{} /@ {}", expr_to_string(func), expr_to_string(list))
    }
    Expr::Apply { func, list } => {
      format!("{} @@ {}", expr_to_string(func), expr_to_string(list))
    }
    Expr::Postfix { expr, func } => {
      format!("{} // {}", expr_to_string(expr), expr_to_string(func))
    }
    Expr::Part { expr, index } => {
      format!("{}[[{}]]", expr_to_string(expr), expr_to_string(index))
    }
    Expr::Function { body } => {
      format!("{}&", expr_to_string(body))
    }
    Expr::Pattern { name, head } => {
      if let Some(h) = head {
        format!("{}_{}", name, h)
      } else {
        format!("{}_", name)
      }
    }
    Expr::Constant(s) => s.clone(),
    Expr::Raw(s) => s.clone(),
  }
}
