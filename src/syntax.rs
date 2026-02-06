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
  /// MapApply: f @@@ list (applies f to each sublist)
  MapApply { func: Box<Expr>, list: Box<Expr> },
  /// Prefix application: f @ x (equivalent to f[x])
  PrefixApply { func: Box<Expr>, arg: Box<Expr> },
  /// Postfix application: expr // f
  Postfix { expr: Box<Expr>, func: Box<Expr> },
  /// Part extraction: expr[[index]]
  Part { expr: Box<Expr>, index: Box<Expr> },
  /// Curried/chained function call: f[a][b] - func is f[a], args is {b}
  CurriedCall { func: Box<Expr>, args: Vec<Expr> },
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
    Rule::Integer | Rule::UnsignedInteger => {
      let s = pair.as_str();
      Expr::Integer(s.parse().unwrap_or(0))
    }
    Rule::Real | Rule::UnsignedReal => {
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
    Rule::NumericValue | Rule::UnsignedNumericValue => {
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
      // Collect bracket sequences separately for proper chained call handling
      let bracket_sequences: Vec<Vec<Expr>> = inner
        .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
        .map(|bracket| {
          bracket
            .into_inner()
            .filter(|p| {
              p.as_str() != "[" && p.as_str() != "]" && p.as_str() != ","
            })
            .map(pair_to_expr)
            .collect()
        })
        .collect();
      // Check if the function head is an anonymous function
      if matches!(name_pair.as_rule(), Rule::SimpleAnonymousFunction) {
        let anon_expr = pair_to_expr(name_pair);
        // Build curried calls: (#&)[1] or (#^2&)[{1,2,3}]
        let mut result = Expr::CurriedCall {
          func: Box::new(anon_expr),
          args: bracket_sequences[0].clone(),
        };
        for args in bracket_sequences.into_iter().skip(1) {
          result = Expr::CurriedCall {
            func: Box::new(result),
            args,
          };
        }
        result
      } else {
        let name = name_pair.as_str().to_string();
        // Build chained calls: f[a][b] becomes Apply(f[a], b)
        if bracket_sequences.len() == 1 {
          Expr::FunctionCall {
            name,
            args: bracket_sequences.into_iter().next().unwrap(),
          }
        } else {
          // Multiple bracket sequences: build nested Apply calls
          // f[a][b][c] becomes: first build f[a], then apply [b], then apply [c]
          let mut result = Expr::FunctionCall {
            name,
            args: bracket_sequences[0].clone(),
          };
          for args in bracket_sequences.into_iter().skip(1) {
            // Wrap as a curried call: FunctionCall applied to new args
            result = Expr::CurriedCall {
              func: Box::new(result),
              args,
            };
          }
          result
        }
      }
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
    Rule::Expression | Rule::ExpressionNoImplicit | Rule::ConditionExpr => {
      parse_expression(pair)
    }
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
      // Store the full pattern string as Raw to preserve test/condition info
      // The string-based pattern matching in apply_replace_all_direct handles these
      Expr::Raw(pair.as_str().to_string())
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
  let pair_str = pair.as_str().to_string();
  let mut inner: Vec<Pair<Rule>> = pair.into_inner().collect();

  if inner.is_empty() {
    return Expr::Raw(String::new());
  }

  // Collect trailing PostfixFunction pairs (lowest precedence, always at end)
  let mut postfix_funcs: Vec<Pair<Rule>> = Vec::new();
  while inner
    .last()
    .is_some_and(|p| p.as_rule() == Rule::PostfixFunction)
  {
    postfix_funcs.push(inner.pop().unwrap());
  }
  postfix_funcs.reverse(); // restore left-to-right order

  // Single term case
  if inner.len() == 1 {
    let mut result = pair_to_expr(inner.remove(0));
    for func_pair in postfix_funcs {
      let func = pair_to_expr(func_pair);
      result = Expr::Postfix {
        expr: Box::new(result),
        func: Box::new(func),
      };
    }
    return result;
  }

  // Check for ReplaceAll/ReplaceRepeated patterns
  // These show up as: Term (List|ReplacementRule)
  if inner.len() >= 2 {
    let second_rule = inner[1].as_rule();
    if second_rule == Rule::List || second_rule == Rule::ReplacementRule {
      // Determine if this is ReplaceRepeated by checking if original expression had "//.
      let is_replace_repeated =
        pair_str.contains("//.") && !pair_str.contains("///.");

      let expr = pair_to_expr(inner.remove(0));
      let rules = pair_to_expr(inner.remove(0));
      let mut result = if is_replace_repeated {
        Expr::ReplaceRepeated {
          expr: Box::new(expr),
          rules: Box::new(rules),
        }
      } else {
        Expr::ReplaceAll {
          expr: Box::new(expr),
          rules: Box::new(rules),
        }
      };
      // Apply postfix functions
      for func_pair in postfix_funcs {
        let func = pair_to_expr(func_pair);
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
      Rule::Operator | Rule::ConditionOp => {
        operators.push(item.as_str().to_string());
      }
      _ => {
        terms.push(pair_to_expr(item));
      }
    }
  }

  let mut result = if terms.len() == 1 {
    terms.remove(0)
  } else {
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
      Expr::Comparison {
        operands: terms,
        operators: comp_ops,
      }
    } else {
      // Build binary operation tree (left-to-right for same precedence)
      build_binary_tree(terms, operators)
    }
  };

  // Apply postfix functions (lowest precedence)
  for func_pair in postfix_funcs {
    let func = pair_to_expr(func_pair);
    result = Expr::Postfix {
      expr: Box::new(result),
      func: Box::new(func),
    };
  }

  result
}

/// Get precedence of an operator (higher = binds tighter)
fn operator_precedence(op: &str) -> u8 {
  match op {
    "||" => 1,
    "&&" => 2,
    "->" | ":>" => 3,
    "/@" | "@@" | "@" => 4,
    "+" | "-" => 5,
    "*" | "/" => 6,
    "^" => 7,
    "<>" => 5, // Same as + for string concatenation
    _ => 0,
  }
}

/// Build a binary operation tree from terms and operators with correct precedence
fn build_binary_tree(terms: Vec<Expr>, operators: Vec<String>) -> Expr {
  if terms.len() == 1 {
    return terms.into_iter().next().unwrap();
  }
  if terms.is_empty() {
    return Expr::Raw(String::new());
  }

  // Use a precedence climbing algorithm
  build_expr_with_precedence(&terms, &operators, 0, 0)
}

/// Build expression with precedence climbing
fn build_expr_with_precedence(
  terms: &[Expr],
  operators: &[String],
  term_start: usize,
  min_prec: u8,
) -> Expr {
  if term_start >= terms.len() {
    return Expr::Raw(String::new());
  }

  let mut result = terms[term_start].clone();
  let mut op_idx = term_start; // operators[i] is between terms[i] and terms[i+1]

  while op_idx < operators.len() {
    let op_str = &operators[op_idx];
    let prec = operator_precedence(op_str);

    if prec < min_prec {
      break;
    }

    // For right-associative operators (like ^ and @), use prec, otherwise use prec + 1
    let next_min_prec = if op_str == "^" || op_str == "@" {
      prec
    } else {
      prec + 1
    };

    // Build the right side with higher precedence
    let right =
      build_expr_with_precedence(terms, operators, op_idx + 1, next_min_prec);

    // Create the binary operation
    result = make_binary_op(&result, op_str, &right);

    // Count how many terms were consumed on the right
    let mut right_terms = 1;
    let mut i = op_idx + 1;
    while i < operators.len()
      && operator_precedence(&operators[i]) >= next_min_prec
    {
      right_terms += 1;
      i += 1;
    }
    op_idx += right_terms;
  }

  result
}

/// Create a binary operation from two expressions and an operator string
fn make_binary_op(left: &Expr, op_str: &str, right: &Expr) -> Expr {
  match op_str {
    "+" => Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "-" => Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "*" => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "/" => Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "^" => Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "&&" => Expr::BinaryOp {
      op: BinaryOperator::And,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "||" => Expr::BinaryOp {
      op: BinaryOperator::Or,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "<>" => Expr::BinaryOp {
      op: BinaryOperator::StringJoin,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "/@" => Expr::Map {
      func: Box::new(left.clone()),
      list: Box::new(right.clone()),
    },
    "@@@" => Expr::MapApply {
      func: Box::new(left.clone()),
      list: Box::new(right.clone()),
    },
    "@@" => Expr::Apply {
      func: Box::new(left.clone()),
      list: Box::new(right.clone()),
    },
    "@" => Expr::PrefixApply {
      func: Box::new(left.clone()),
      arg: Box::new(right.clone()),
    },
    "->" => Expr::Rule {
      pattern: Box::new(left.clone()),
      replacement: Box::new(right.clone()),
    },
    ":>" => Expr::RuleDelayed {
      pattern: Box::new(left.clone()),
      replacement: Box::new(right.clone()),
    },
    "=" => Expr::FunctionCall {
      name: "Set".to_string(),
      args: vec![left.clone(), right.clone()],
    },
    ":=" => Expr::FunctionCall {
      name: "SetDelayed".to_string(),
      args: vec![left.clone(), right.clone()],
    },
    _ => Expr::Raw(format!(
      "{} {} {}",
      expr_to_string(left),
      op_str,
      expr_to_string(right)
    )),
  }
}

/// Parse the body of an anonymous function using the pest parser
fn parse_anonymous_body(s: &str) -> Expr {
  let s = s.trim();

  if s.is_empty() {
    return Expr::Slot(1);
  }

  // Check for simple slot (fast path)
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

  // Use the pest parser to properly parse complex expressions
  match crate::parse(s) {
    Ok(pairs) => {
      for pair in pairs {
        for node in pair.into_inner() {
          if matches!(node.as_rule(), Rule::Expression) {
            return pair_to_expr(node);
          }
        }
      }
      // Fallback if parsing succeeded but no expression found
      Expr::Raw(s.to_string())
    }
    Err(_) => {
      // If parsing fails, fall back to Raw
      Expr::Raw(s.to_string())
    }
  }
}

/// Format a real number for output (matches Wolfram Language format)
fn format_real(f: f64) -> String {
  if f.fract() == 0.0 {
    // Whole number - format with decimal point to indicate it's a Real
    format!("{}.", f as i64)
  } else {
    // Check if a shorter representation (10 digits) rounds to the same value
    let short = format!("{:.10}", f).trim_end_matches('0').to_string();
    if let Ok(parsed) = short.parse::<f64>()
      && (parsed - f).abs() < 1e-14
    {
      return short;
    }
    // Otherwise use full precision (16 digits), trim trailing zeros
    format!("{:.16}", f).trim_end_matches('0').to_string()
  }
}

/// Convert an Expr back to a string representation
pub fn expr_to_string(expr: &Expr) -> String {
  match expr {
    Expr::Integer(n) => n.to_string(),
    Expr::Real(f) => format_real(*f),
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
      // Special case: Rational[num, denom] displays as num/denom
      if name == "Rational" && args.len() == 2 {
        return format!(
          "{}/{}",
          expr_to_string(&args[0]),
          expr_to_string(&args[1])
        );
      }
      // Special case: Factorial[n] displays as n!
      if name == "Factorial" && args.len() == 1 {
        return format!("{}!", expr_to_string(&args[0]));
      }
      let parts: Vec<String> = args.iter().map(expr_to_string).collect();
      format!("{}[{}]", name, parts.join(", "))
    }
    Expr::BinaryOp { op, left, right } => {
      // Special case: a + (-b) should display as a - b
      if matches!(op, BinaryOperator::Plus)
        && let Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } = right.as_ref()
      {
        let left_str = expr_to_string(left);
        let operand_str = expr_to_string(operand);
        return format!("{} - {}", left_str, operand_str);
      }

      // Mathematica uses no spaces for *, /, ^ but spaces for +, -, &&, ||
      let (op_str, needs_space) = match op {
        BinaryOperator::Plus => ("+", true),
        BinaryOperator::Minus => ("−", true),
        BinaryOperator::Times => ("*", false),
        BinaryOperator::Divide => ("/", false),
        BinaryOperator::Power => ("^", false),
        BinaryOperator::And => ("&&", true),
        BinaryOperator::Or => ("||", true),
        BinaryOperator::StringJoin => ("<>", false),
      };

      // Helper to check if expr needs parentheses based on precedence
      let needs_parens = |e: &Expr, for_power_exp: bool| -> bool {
        match e {
          // For Power exponent, Plus needs parens
          Expr::BinaryOp {
            op: BinaryOperator::Plus | BinaryOperator::Minus,
            ..
          } if for_power_exp => true,
          Expr::FunctionCall { name, .. }
            if name == "Plus" && for_power_exp =>
          {
            true
          }
          _ => false,
        }
      };

      let left_str = expr_to_string(left);
      let right_str = expr_to_string(right);

      // Wrap right operand in parens if needed (for Power with Plus/Minus exponent)
      let is_power = matches!(op, BinaryOperator::Power);
      let right_formatted = if is_power && needs_parens(right, true) {
        format!("({})", right_str)
      } else {
        right_str
      };

      if needs_space {
        format!("{} {} {}", left_str, op_str, right_formatted)
      } else {
        format!("{}{}{}", left_str, op_str, right_formatted)
      }
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
    Expr::MapApply { func, list } => {
      format!("{} @@@ {}", expr_to_string(func), expr_to_string(list))
    }
    Expr::PrefixApply { func, arg } => {
      format!("{} @ {}", expr_to_string(func), expr_to_string(arg))
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
    Expr::CurriedCall { func, args } => {
      // Display as nested calls: f[a][b, c]
      let args_str: Vec<String> = args.iter().map(expr_to_string).collect();
      format!("{}[{}]", expr_to_string(func), args_str.join(", "))
    }
  }
}

/// Render Expr for display output - strings are shown without quotes.
/// This is used for the final output in interpret(), not for round-tripping.
pub fn expr_to_output(expr: &Expr) -> String {
  match expr {
    Expr::String(s) => s.clone(), // No quotes for display
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_output).collect();
      format!("{{{}}}", parts.join(", "))
    }
    Expr::FunctionCall { name, args } => {
      // Special case: Rational[num, denom] displays as num/denom
      if name == "Rational" && args.len() == 2 {
        return format!(
          "{}/{}",
          expr_to_output(&args[0]),
          expr_to_output(&args[1])
        );
      }
      // Special case: Factorial[n] displays as n!
      if name == "Factorial" && args.len() == 1 {
        return format!("{}!", expr_to_output(&args[0]));
      }
      // Special case: Plus displays as infix with + (handling - for negative terms)
      if name == "Plus" && args.len() >= 2 {
        let mut result = expr_to_output(&args[0]);
        for arg in args.iter().skip(1) {
          // Check if this term is a UnaryOp minus - if so, use " - " instead of " + "
          if let Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } = arg
          {
            result.push_str(" - ");
            result.push_str(&expr_to_output(operand));
          } else {
            result.push_str(" + ");
            result.push_str(&expr_to_output(arg));
          }
        }
        return result;
      }
      // Special case: Times displays as infix with * (no spaces)
      if name == "Times" && args.len() >= 2 {
        return args
          .iter()
          .map(|a| {
            // Wrap lower-precedence operations in parens
            let s = expr_to_output(a);
            if matches!(a, Expr::FunctionCall { name, .. } if name == "Plus")
              || matches!(
                a,
                Expr::BinaryOp {
                  op: BinaryOperator::Plus | BinaryOperator::Minus,
                  ..
                }
              )
            {
              format!("({})", s)
            } else {
              s
            }
          })
          .collect::<Vec<_>>()
          .join("*");
      }
      // Special case: Power displays as infix with ^ (no spaces)
      if name == "Power" && args.len() == 2 {
        let base = expr_to_output(&args[0]);
        let exp_str = expr_to_output(&args[1]);
        // Wrap exponent in parens if it's a Plus (lower precedence)
        let exp = if matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Plus")
          || matches!(
            &args[1],
            Expr::BinaryOp {
              op: BinaryOperator::Plus | BinaryOperator::Minus,
              ..
            }
          ) {
          format!("({})", exp_str)
        } else {
          exp_str
        };
        return format!("{}^{}", base, exp);
      }
      let parts: Vec<String> = args.iter().map(expr_to_output).collect();
      format!("{}[{}]", name, parts.join(", "))
    }
    Expr::Association(items) => {
      let parts: Vec<String> = items
        .iter()
        .map(|(k, v)| format!("{} -> {}", expr_to_output(k), expr_to_output(v)))
        .collect();
      format!("<|{}|>", parts.join(", "))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => {
      format!(
        "{} -> {}",
        expr_to_output(pattern),
        expr_to_output(replacement)
      )
    }
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      format!(
        "{} :> {}",
        expr_to_output(pattern),
        expr_to_output(replacement)
      )
    }
    // For all other cases, delegate to expr_to_string
    _ => expr_to_string(expr),
  }
}

/// Parse a string into an Expr AST.
/// This is used when we need to convert external string input to AST form.
pub fn string_to_expr(s: &str) -> Result<Expr, crate::InterpreterError> {
  let trimmed = s.trim();

  // Handle empty string - return as empty quoted string literal
  if trimmed.is_empty() {
    return Ok(Expr::Raw(String::new()));
  }

  // Fast path for simple literals
  if let Ok(n) = trimmed.parse::<i128>() {
    return Ok(Expr::Integer(n));
  }
  if let Ok(f) = trimmed.parse::<f64>() {
    return Ok(Expr::Real(f));
  }

  // Check for quoted string
  if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
    let inner = &trimmed[1..trimmed.len() - 1];
    return Ok(Expr::String(inner.to_string()));
  }

  // Check for simple identifier
  if !trimmed.is_empty()
    && trimmed
      .chars()
      .next()
      .map(|c| c.is_ascii_alphabetic() || c == '$')
      .unwrap_or(false)
    && trimmed
      .chars()
      .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
  {
    return Ok(Expr::Identifier(trimmed.to_string()));
  }

  // Check for slot
  if trimmed == "#" {
    return Ok(Expr::Slot(1));
  }
  if trimmed.starts_with('#')
    && trimmed.len() > 1
    && let Ok(n) = trimmed[1..].parse::<usize>()
  {
    return Ok(Expr::Slot(n));
  }

  // Parse using pest
  let pairs = crate::parse(trimmed)?;
  let mut pairs_iter = pairs.into_iter();
  let program = pairs_iter
    .next()
    .ok_or(crate::InterpreterError::EmptyInput)?;

  for node in program.into_inner() {
    match node.as_rule() {
      Rule::Expression => {
        return Ok(pair_to_expr(node));
      }
      _ => continue,
    }
  }

  // Fallback to Raw
  Ok(Expr::Raw(trimmed.to_string()))
}

/// Substitute slots (#, #1, #2, etc.) in an expression with values.
/// values[0] replaces #1 (or #), values[1] replaces #2, etc.
pub fn substitute_slots(expr: &Expr, values: &[Expr]) -> Expr {
  match expr {
    Expr::Slot(n) => {
      let index = if *n == 0 { 0 } else { n - 1 };
      if index < values.len() {
        values[index].clone()
      } else {
        expr.clone()
      }
    }
    Expr::List(items) => {
      Expr::List(items.iter().map(|e| substitute_slots(e, values)).collect())
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(|e| substitute_slots(e, values)).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_slots(left, values)),
      right: Box::new(substitute_slots(right, values)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_slots(operand, values)),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands
        .iter()
        .map(|e| substitute_slots(e, values))
        .collect(),
      operators: operators.clone(),
    },
    Expr::CompoundExpr(exprs) => Expr::CompoundExpr(
      exprs.iter().map(|e| substitute_slots(e, values)).collect(),
    ),
    Expr::Association(items) => Expr::Association(
      items
        .iter()
        .map(|(k, v)| {
          (substitute_slots(k, values), substitute_slots(v, values))
        })
        .collect(),
    ),
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(substitute_slots(pattern, values)),
      replacement: Box::new(substitute_slots(replacement, values)),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(substitute_slots(pattern, values)),
      replacement: Box::new(substitute_slots(replacement, values)),
    },
    Expr::ReplaceAll { expr: e, rules } => Expr::ReplaceAll {
      expr: Box::new(substitute_slots(e, values)),
      rules: Box::new(substitute_slots(rules, values)),
    },
    Expr::ReplaceRepeated { expr: e, rules } => Expr::ReplaceRepeated {
      expr: Box::new(substitute_slots(e, values)),
      rules: Box::new(substitute_slots(rules, values)),
    },
    Expr::Map { func, list } => Expr::Map {
      func: Box::new(substitute_slots(func, values)),
      list: Box::new(substitute_slots(list, values)),
    },
    Expr::Apply { func, list } => Expr::Apply {
      func: Box::new(substitute_slots(func, values)),
      list: Box::new(substitute_slots(list, values)),
    },
    Expr::MapApply { func, list } => Expr::MapApply {
      func: Box::new(substitute_slots(func, values)),
      list: Box::new(substitute_slots(list, values)),
    },
    Expr::PrefixApply { func, arg } => Expr::PrefixApply {
      func: Box::new(substitute_slots(func, values)),
      arg: Box::new(substitute_slots(arg, values)),
    },
    Expr::Postfix { expr: e, func } => Expr::Postfix {
      expr: Box::new(substitute_slots(e, values)),
      func: Box::new(substitute_slots(func, values)),
    },
    Expr::Part { expr: e, index } => Expr::Part {
      expr: Box::new(substitute_slots(e, values)),
      index: Box::new(substitute_slots(index, values)),
    },
    Expr::Function { body } => Expr::Function {
      body: Box::new(substitute_slots(body, values)),
    },
    // Atoms that don't contain slots
    _ => expr.clone(),
  }
}

/// Substitute a variable name with a value in an expression.
pub fn substitute_variable(expr: &Expr, var_name: &str, value: &Expr) -> Expr {
  match expr {
    Expr::Identifier(name) if name == var_name => value.clone(),
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| substitute_variable(e, var_name, value))
        .collect(),
    ),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|e| substitute_variable(e, var_name, value))
        .collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_variable(left, var_name, value)),
      right: Box::new(substitute_variable(right, var_name, value)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_variable(operand, var_name, value)),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands
        .iter()
        .map(|e| substitute_variable(e, var_name, value))
        .collect(),
      operators: operators.clone(),
    },
    Expr::CompoundExpr(exprs) => Expr::CompoundExpr(
      exprs
        .iter()
        .map(|e| substitute_variable(e, var_name, value))
        .collect(),
    ),
    Expr::Association(items) => Expr::Association(
      items
        .iter()
        .map(|(k, v)| {
          (
            substitute_variable(k, var_name, value),
            substitute_variable(v, var_name, value),
          )
        })
        .collect(),
    ),
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(substitute_variable(pattern, var_name, value)),
      replacement: Box::new(substitute_variable(replacement, var_name, value)),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(substitute_variable(pattern, var_name, value)),
      replacement: Box::new(substitute_variable(replacement, var_name, value)),
    },
    Expr::ReplaceAll { expr: e, rules } => Expr::ReplaceAll {
      expr: Box::new(substitute_variable(e, var_name, value)),
      rules: Box::new(substitute_variable(rules, var_name, value)),
    },
    Expr::ReplaceRepeated { expr: e, rules } => Expr::ReplaceRepeated {
      expr: Box::new(substitute_variable(e, var_name, value)),
      rules: Box::new(substitute_variable(rules, var_name, value)),
    },
    Expr::Map { func, list } => Expr::Map {
      func: Box::new(substitute_variable(func, var_name, value)),
      list: Box::new(substitute_variable(list, var_name, value)),
    },
    Expr::Apply { func, list } => Expr::Apply {
      func: Box::new(substitute_variable(func, var_name, value)),
      list: Box::new(substitute_variable(list, var_name, value)),
    },
    Expr::MapApply { func, list } => Expr::MapApply {
      func: Box::new(substitute_variable(func, var_name, value)),
      list: Box::new(substitute_variable(list, var_name, value)),
    },
    Expr::PrefixApply { func, arg } => Expr::PrefixApply {
      func: Box::new(substitute_variable(func, var_name, value)),
      arg: Box::new(substitute_variable(arg, var_name, value)),
    },
    Expr::Postfix { expr: e, func } => Expr::Postfix {
      expr: Box::new(substitute_variable(e, var_name, value)),
      func: Box::new(substitute_variable(func, var_name, value)),
    },
    Expr::Part { expr: e, index } => Expr::Part {
      expr: Box::new(substitute_variable(e, var_name, value)),
      index: Box::new(substitute_variable(index, var_name, value)),
    },
    Expr::Function { body } => Expr::Function {
      body: Box::new(substitute_variable(body, var_name, value)),
    },
    // Atoms that don't contain the variable
    _ => expr.clone(),
  }
}
