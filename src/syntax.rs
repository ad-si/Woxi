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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageType {
  Byte,
  Bit16,
  Real32,
  Real64,
}

/// Owned expression tree for storing parsed function bodies.
/// This avoids re-parsing function bodies on every call.
#[derive(Debug)]
pub enum Expr {
  /// Integer literal
  Integer(i128),
  /// Big integer (exceeds i128 range)
  BigInteger(num_bigint::BigInt),
  /// Real/float literal
  Real(f64),
  /// Arbitrary-precision real: (formatted_digits, precision_in_decimal_digits)
  BigFloat(String, usize),
  /// String literal (without quotes)
  String(String),
  /// Identifier/symbol
  Identifier(String),
  /// Slot (#, #1, #2, etc.)
  Slot(usize),
  /// SlotSequence (##, ##1, ##2, etc.) — represents a sequence of arguments
  SlotSequence(usize),
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
  /// Named-parameter function: Function[x, body] or Function[{x,y,...}, body]
  NamedFunction {
    params: Vec<String>,
    body: Box<Expr>,
  },
  /// Pattern: name_ or name_Head
  Pattern { name: String, head: Option<String> },
  /// Optional pattern: name_ : default or name_Head : default
  PatternOptional {
    name: String,
    head: Option<String>,
    default: Box<Expr>,
  },
  /// PatternTest: _?test or x_?test — matches if test[x] is True
  PatternTest { name: String, test: Box<Expr> },
  /// Constant like Pi, E, etc.
  Constant(String),
  /// Raw unparsed text (fallback)
  Raw(String),
  /// Image: raster image data
  Image {
    width: u32,
    height: u32,
    channels: u8,
    data: std::sync::Arc<Vec<f64>>,
    image_type: ImageType,
  },
  /// Graphics output: holds SVG string, displays as -Graphics- (or -Graphics3D- if is_3d)
  Graphics { svg: String, is_3d: bool },
}

/// Extract all child `Expr` nodes from a variant, leaving it childless.
/// Used by the iterative `Drop` implementation.
fn take_expr_children(expr: &mut Expr, stack: &mut Vec<Expr>) {
  match expr {
    // Leaf variants — no children
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _)
    | Expr::String(_)
    | Expr::Identifier(_)
    | Expr::Slot(_)
    | Expr::SlotSequence(_)
    | Expr::Pattern { .. }
    | Expr::Constant(_)
    | Expr::Raw(_)
    | Expr::Image { .. }
    | Expr::Graphics { .. } => {}

    // Vec<Expr> children
    Expr::List(children) | Expr::CompoundExpr(children) => {
      stack.append(children);
    }
    Expr::FunctionCall { args, .. } => {
      stack.append(args);
    }
    Expr::Comparison { operands, .. } => {
      stack.append(operands);
    }

    // Box<Expr> children (two boxes)
    Expr::BinaryOp { left, right, .. } => {
      stack.push(*std::mem::replace(left, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(right, Box::new(Expr::Integer(0))));
    }
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      stack.push(*std::mem::replace(pattern, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(replacement, Box::new(Expr::Integer(0))));
    }
    Expr::ReplaceAll { expr, rules }
    | Expr::ReplaceRepeated { expr, rules } => {
      stack.push(*std::mem::replace(expr, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(rules, Box::new(Expr::Integer(0))));
    }
    Expr::Map { func, list }
    | Expr::Apply { func, list }
    | Expr::MapApply { func, list } => {
      stack.push(*std::mem::replace(func, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(list, Box::new(Expr::Integer(0))));
    }
    Expr::PrefixApply { func, arg } => {
      stack.push(*std::mem::replace(func, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(arg, Box::new(Expr::Integer(0))));
    }
    Expr::Postfix { expr, func } => {
      stack.push(*std::mem::replace(expr, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(func, Box::new(Expr::Integer(0))));
    }
    Expr::Part { expr, index } => {
      stack.push(*std::mem::replace(expr, Box::new(Expr::Integer(0))));
      stack.push(*std::mem::replace(index, Box::new(Expr::Integer(0))));
    }

    // Single Box<Expr>
    Expr::UnaryOp { operand, .. } => {
      stack.push(*std::mem::replace(operand, Box::new(Expr::Integer(0))));
    }
    Expr::Function { body } => {
      stack.push(*std::mem::replace(body, Box::new(Expr::Integer(0))));
    }
    Expr::NamedFunction { body, .. } => {
      stack.push(*std::mem::replace(body, Box::new(Expr::Integer(0))));
    }
    Expr::PatternOptional { default, .. } => {
      stack.push(*std::mem::replace(default, Box::new(Expr::Integer(0))));
    }
    Expr::PatternTest { test, .. } => {
      stack.push(*std::mem::replace(test, Box::new(Expr::Integer(0))));
    }

    // Box<Expr> + Vec<Expr>
    Expr::CurriedCall { func, args } => {
      stack.push(*std::mem::replace(func, Box::new(Expr::Integer(0))));
      stack.append(args);
    }

    // Vec<(Expr, Expr)>
    Expr::Association(pairs) => {
      for (k, v) in pairs.drain(..) {
        stack.push(k);
        stack.push(v);
      }
    }
  }
}

impl Drop for Expr {
  fn drop(&mut self) {
    let mut work = Vec::new();
    take_expr_children(self, &mut work);
    while let Some(mut child) = work.pop() {
      take_expr_children(&mut child, &mut work);
      // child now has no Expr children, so its recursive drop is trivial
    }
  }
}

impl Clone for Expr {
  fn clone(&self) -> Self {
    // Fast path for leaf variants — avoid heap allocation
    match self {
      Expr::Integer(n) => return Expr::Integer(*n),
      Expr::BigInteger(n) => return Expr::BigInteger(n.clone()),
      Expr::Real(f) => return Expr::Real(*f),
      Expr::BigFloat(s, p) => return Expr::BigFloat(s.clone(), *p),
      Expr::String(s) => return Expr::String(s.clone()),
      Expr::Identifier(s) => return Expr::Identifier(s.clone()),
      Expr::Slot(n) => return Expr::Slot(*n),
      Expr::SlotSequence(n) => return Expr::SlotSequence(*n),
      Expr::Pattern { name, head } => {
        return Expr::Pattern {
          name: name.clone(),
          head: head.clone(),
        };
      }
      Expr::Constant(s) => return Expr::Constant(s.clone()),
      Expr::Raw(s) => return Expr::Raw(s.clone()),
      Expr::Image {
        width,
        height,
        channels,
        data,
        image_type,
      } => {
        return Expr::Image {
          width: *width,
          height: *height,
          channels: *channels,
          data: data.clone(),
          image_type: *image_type,
        };
      }
      Expr::Graphics { svg, is_3d } => {
        return Expr::Graphics {
          svg: svg.clone(),
          is_3d: *is_3d,
        };
      }
      _ => {} // fall through to iterative clone
    }

    // Iterative clone for non-leaf variants
    enum CloneTask<'a> {
      Visit(&'a Expr),
      Build(CloneFrame),
    }

    enum CloneFrame {
      List(usize),
      FunctionCall(String, usize),
      BinaryOp(BinaryOperator),
      UnaryOp(UnaryOperator),
      Comparison(Vec<ComparisonOp>, usize),
      CompoundExpr(usize),
      Association(usize),
      Rule,
      RuleDelayed,
      ReplaceAll,
      ReplaceRepeated,
      Map,
      Apply,
      MapApply,
      PrefixApply,
      Postfix,
      Part,
      CurriedCall(usize),
      Function,
      NamedFunction(Vec<String>),
      PatternOptional(String, Option<String>),
      PatternTest(String),
    }

    let mut tasks: Vec<CloneTask> = vec![CloneTask::Visit(self)];
    let mut results: Vec<Expr> = Vec::new();

    while let Some(task) = tasks.pop() {
      match task {
        CloneTask::Visit(expr) => match expr {
          // Leaf variants
          Expr::Integer(n) => results.push(Expr::Integer(*n)),
          Expr::BigInteger(n) => results.push(Expr::BigInteger(n.clone())),
          Expr::Real(f) => results.push(Expr::Real(*f)),
          Expr::BigFloat(s, p) => results.push(Expr::BigFloat(s.clone(), *p)),
          Expr::String(s) => results.push(Expr::String(s.clone())),
          Expr::Identifier(s) => results.push(Expr::Identifier(s.clone())),
          Expr::Slot(n) => results.push(Expr::Slot(*n)),
          Expr::SlotSequence(n) => results.push(Expr::SlotSequence(*n)),
          Expr::Pattern { name, head } => results.push(Expr::Pattern {
            name: name.clone(),
            head: head.clone(),
          }),
          Expr::Constant(s) => results.push(Expr::Constant(s.clone())),
          Expr::Raw(s) => results.push(Expr::Raw(s.clone())),
          Expr::Image {
            width,
            height,
            channels,
            data,
            image_type,
          } => results.push(Expr::Image {
            width: *width,
            height: *height,
            channels: *channels,
            data: data.clone(),
            image_type: *image_type,
          }),
          Expr::Graphics { svg, is_3d } => results.push(Expr::Graphics {
            svg: svg.clone(),
            is_3d: *is_3d,
          }),

          // Vec<Expr> children
          Expr::List(children) => {
            let count = children.len();
            tasks.push(CloneTask::Build(CloneFrame::List(count)));
            for child in children.iter().rev() {
              tasks.push(CloneTask::Visit(child));
            }
          }
          Expr::CompoundExpr(children) => {
            let count = children.len();
            tasks.push(CloneTask::Build(CloneFrame::CompoundExpr(count)));
            for child in children.iter().rev() {
              tasks.push(CloneTask::Visit(child));
            }
          }
          Expr::FunctionCall { name, args } => {
            let count = args.len();
            tasks.push(CloneTask::Build(CloneFrame::FunctionCall(
              name.clone(),
              count,
            )));
            for child in args.iter().rev() {
              tasks.push(CloneTask::Visit(child));
            }
          }
          Expr::Comparison {
            operands,
            operators,
          } => {
            let count = operands.len();
            tasks.push(CloneTask::Build(CloneFrame::Comparison(
              operators.clone(),
              count,
            )));
            for child in operands.iter().rev() {
              tasks.push(CloneTask::Visit(child));
            }
          }

          // Two Box<Expr> children
          Expr::BinaryOp { op, left, right } => {
            tasks.push(CloneTask::Build(CloneFrame::BinaryOp(*op)));
            tasks.push(CloneTask::Visit(right));
            tasks.push(CloneTask::Visit(left));
          }
          Expr::Rule {
            pattern,
            replacement,
          } => {
            tasks.push(CloneTask::Build(CloneFrame::Rule));
            tasks.push(CloneTask::Visit(replacement));
            tasks.push(CloneTask::Visit(pattern));
          }
          Expr::RuleDelayed {
            pattern,
            replacement,
          } => {
            tasks.push(CloneTask::Build(CloneFrame::RuleDelayed));
            tasks.push(CloneTask::Visit(replacement));
            tasks.push(CloneTask::Visit(pattern));
          }
          Expr::ReplaceAll { expr, rules } => {
            tasks.push(CloneTask::Build(CloneFrame::ReplaceAll));
            tasks.push(CloneTask::Visit(rules));
            tasks.push(CloneTask::Visit(expr));
          }
          Expr::ReplaceRepeated { expr, rules } => {
            tasks.push(CloneTask::Build(CloneFrame::ReplaceRepeated));
            tasks.push(CloneTask::Visit(rules));
            tasks.push(CloneTask::Visit(expr));
          }
          Expr::Map { func, list } => {
            tasks.push(CloneTask::Build(CloneFrame::Map));
            tasks.push(CloneTask::Visit(list));
            tasks.push(CloneTask::Visit(func));
          }
          Expr::Apply { func, list } => {
            tasks.push(CloneTask::Build(CloneFrame::Apply));
            tasks.push(CloneTask::Visit(list));
            tasks.push(CloneTask::Visit(func));
          }
          Expr::MapApply { func, list } => {
            tasks.push(CloneTask::Build(CloneFrame::MapApply));
            tasks.push(CloneTask::Visit(list));
            tasks.push(CloneTask::Visit(func));
          }
          Expr::PrefixApply { func, arg } => {
            tasks.push(CloneTask::Build(CloneFrame::PrefixApply));
            tasks.push(CloneTask::Visit(arg));
            tasks.push(CloneTask::Visit(func));
          }
          Expr::Postfix { expr, func } => {
            tasks.push(CloneTask::Build(CloneFrame::Postfix));
            tasks.push(CloneTask::Visit(func));
            tasks.push(CloneTask::Visit(expr));
          }
          Expr::Part { expr, index } => {
            tasks.push(CloneTask::Build(CloneFrame::Part));
            tasks.push(CloneTask::Visit(index));
            tasks.push(CloneTask::Visit(expr));
          }

          // Single Box<Expr>
          Expr::UnaryOp { op, operand } => {
            tasks.push(CloneTask::Build(CloneFrame::UnaryOp(*op)));
            tasks.push(CloneTask::Visit(operand));
          }
          Expr::Function { body } => {
            tasks.push(CloneTask::Build(CloneFrame::Function));
            tasks.push(CloneTask::Visit(body));
          }
          Expr::NamedFunction { params, body } => {
            tasks.push(CloneTask::Build(CloneFrame::NamedFunction(
              params.clone(),
            )));
            tasks.push(CloneTask::Visit(body));
          }
          Expr::PatternOptional {
            name,
            head,
            default,
          } => {
            tasks.push(CloneTask::Build(CloneFrame::PatternOptional(
              name.clone(),
              head.clone(),
            )));
            tasks.push(CloneTask::Visit(default));
          }
          Expr::PatternTest { name, test } => {
            tasks.push(CloneTask::Build(CloneFrame::PatternTest(name.clone())));
            tasks.push(CloneTask::Visit(test));
          }

          // Box<Expr> + Vec<Expr>
          Expr::CurriedCall { func, args } => {
            let count = args.len();
            // Build needs: func first, then count args
            tasks.push(CloneTask::Build(CloneFrame::CurriedCall(count)));
            for child in args.iter().rev() {
              tasks.push(CloneTask::Visit(child));
            }
            tasks.push(CloneTask::Visit(func));
          }

          // Vec<(Expr, Expr)>
          Expr::Association(pairs) => {
            let count = pairs.len();
            tasks.push(CloneTask::Build(CloneFrame::Association(count)));
            // Push pairs in reverse; each pair = (key, value)
            for (k, v) in pairs.iter().rev() {
              tasks.push(CloneTask::Visit(v));
              tasks.push(CloneTask::Visit(k));
            }
          }
        },

        CloneTask::Build(frame) => {
          let expr = match frame {
            CloneFrame::List(count) => {
              let children: Vec<Expr> =
                results.drain(results.len() - count..).collect();
              Expr::List(children)
            }
            CloneFrame::CompoundExpr(count) => {
              let children: Vec<Expr> =
                results.drain(results.len() - count..).collect();
              Expr::CompoundExpr(children)
            }
            CloneFrame::FunctionCall(name, count) => {
              let args: Vec<Expr> =
                results.drain(results.len() - count..).collect();
              Expr::FunctionCall { name, args }
            }
            CloneFrame::Comparison(operators, count) => {
              let operands: Vec<Expr> =
                results.drain(results.len() - count..).collect();
              Expr::Comparison {
                operands,
                operators,
              }
            }
            CloneFrame::BinaryOp(op) => {
              let right = Box::new(results.pop().unwrap());
              let left = Box::new(results.pop().unwrap());
              Expr::BinaryOp { op, left, right }
            }
            CloneFrame::UnaryOp(op) => {
              let operand = Box::new(results.pop().unwrap());
              Expr::UnaryOp { op, operand }
            }
            CloneFrame::Rule => {
              let replacement = Box::new(results.pop().unwrap());
              let pattern = Box::new(results.pop().unwrap());
              Expr::Rule {
                pattern,
                replacement,
              }
            }
            CloneFrame::RuleDelayed => {
              let replacement = Box::new(results.pop().unwrap());
              let pattern = Box::new(results.pop().unwrap());
              Expr::RuleDelayed {
                pattern,
                replacement,
              }
            }
            CloneFrame::ReplaceAll => {
              let rules = Box::new(results.pop().unwrap());
              let expr = Box::new(results.pop().unwrap());
              Expr::ReplaceAll { expr, rules }
            }
            CloneFrame::ReplaceRepeated => {
              let rules = Box::new(results.pop().unwrap());
              let expr = Box::new(results.pop().unwrap());
              Expr::ReplaceRepeated { expr, rules }
            }
            CloneFrame::Map => {
              let list = Box::new(results.pop().unwrap());
              let func = Box::new(results.pop().unwrap());
              Expr::Map { func, list }
            }
            CloneFrame::Apply => {
              let list = Box::new(results.pop().unwrap());
              let func = Box::new(results.pop().unwrap());
              Expr::Apply { func, list }
            }
            CloneFrame::MapApply => {
              let list = Box::new(results.pop().unwrap());
              let func = Box::new(results.pop().unwrap());
              Expr::MapApply { func, list }
            }
            CloneFrame::PrefixApply => {
              let arg = Box::new(results.pop().unwrap());
              let func = Box::new(results.pop().unwrap());
              Expr::PrefixApply { func, arg }
            }
            CloneFrame::Postfix => {
              let func = Box::new(results.pop().unwrap());
              let expr = Box::new(results.pop().unwrap());
              Expr::Postfix { expr, func }
            }
            CloneFrame::Part => {
              let index = Box::new(results.pop().unwrap());
              let expr = Box::new(results.pop().unwrap());
              Expr::Part { expr, index }
            }
            CloneFrame::Function => {
              let body = Box::new(results.pop().unwrap());
              Expr::Function { body }
            }
            CloneFrame::NamedFunction(params) => {
              let body = Box::new(results.pop().unwrap());
              Expr::NamedFunction { params, body }
            }
            CloneFrame::PatternOptional(name, head) => {
              let default = Box::new(results.pop().unwrap());
              Expr::PatternOptional {
                name,
                head,
                default,
              }
            }
            CloneFrame::PatternTest(name) => {
              let test = Box::new(results.pop().unwrap());
              Expr::PatternTest { name, test }
            }
            CloneFrame::CurriedCall(count) => {
              let args: Vec<Expr> =
                results.drain(results.len() - count..).collect();
              let func = Box::new(results.pop().unwrap());
              Expr::CurriedCall { func, args }
            }
            CloneFrame::Association(count) => {
              let mut pairs = Vec::with_capacity(count);
              let start = results.len() - count * 2;
              let flat: Vec<Expr> = results.drain(start..).collect();
              let mut iter = flat.into_iter();
              for _ in 0..count {
                let k = iter.next().unwrap();
                let v = iter.next().unwrap();
                pairs.push((k, v));
              }
              Expr::Association(pairs)
            }
          };
          results.push(expr);
        }
      }
    }

    results.pop().unwrap()
  }
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
  Alternatives,
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
      match s.parse::<i128>() {
        Ok(n) => Expr::Integer(n),
        Err(_) => {
          // Overflows i128 — use BigInteger
          match s.parse::<num_bigint::BigInt>() {
            Ok(n) => Expr::BigInteger(n),
            Err(_) => Expr::Integer(0),
          }
        }
      }
    }
    Rule::Real | Rule::UnsignedReal => {
      let s = pair.as_str();
      // Handle Wolfram's *^ scientific notation (e.g. 2.7*^7 = 2.7e7)
      if let Some(idx) = s.find("*^") {
        let mantissa: f64 = s[..idx].parse().unwrap_or(0.0);
        let exponent: i32 = s[idx + 2..].parse().unwrap_or(0);
        Expr::Real(mantissa * 10_f64.powi(exponent))
      } else {
        Expr::Real(s.parse().unwrap_or(0.0))
      }
    }
    Rule::BasePrefix => {
      let s = pair.as_str();
      // Parse base^^digits format (e.g. 16^^FF = 255, 2^^1010 = 10)
      let parts: Vec<&str> = s.splitn(2, "^^").collect();
      let base: u32 = parts[0].parse().unwrap_or(10);
      let digits = parts[1];
      // i128::from_str_radix only supports lowercase, so normalize
      match i128::from_str_radix(&digits.to_lowercase(), base) {
        Ok(val) => Expr::Integer(val),
        Err(_) => {
          // Overflows i128 — try BigInteger
          use num_bigint::BigInt;
          use num_traits::Num;
          match BigInt::from_str_radix(&digits.to_lowercase(), base) {
            Ok(n) => Expr::BigInteger(n),
            Err(_) => Expr::Integer(0),
          }
        }
      }
    }
    Rule::String => {
      let s = pair.as_str();
      // Remove surrounding quotes and process escape sequences
      let raw = &s[1..s.len() - 1];
      let mut result = String::with_capacity(raw.len());
      let mut chars = raw.chars();
      while let Some(c) = chars.next() {
        if c == '\\' {
          match chars.next() {
            Some('n') => result.push('\n'),
            Some('t') => result.push('\t'),
            Some('r') => result.push('\r'),
            Some('\\') => result.push('\\'),
            Some('"') => result.push('"'),
            Some('\n') => {} // line continuation: skip the newline
            Some(other) => {
              result.push('\\');
              result.push(other);
            }
            None => result.push('\\'),
          }
        } else {
          result.push(c);
        }
      }
      Expr::String(result)
    }
    Rule::InformationQuery => {
      let symbol_name = pair.into_inner().next().unwrap().as_str().to_string();
      Expr::FunctionCall {
        name: "Information".to_string(),
        args: vec![Expr::Identifier(symbol_name)],
      }
    }
    Rule::Identifier => Expr::Identifier(pair.as_str().to_string()),
    Rule::DerivativeIdentifier => {
      // Standalone f' → Derivative[1][f], f'' → Derivative[2][f], etc.
      let inner_pairs: Vec<_> = pair.into_inner().collect();
      let name = inner_pairs[0].as_str().to_string();
      let order = inner_pairs[1].as_str().len();
      Expr::CurriedCall {
        func: Box::new(Expr::FunctionCall {
          name: "Derivative".to_string(),
          args: vec![Expr::Integer(order as i128)],
        }),
        args: vec![Expr::Identifier(name)],
      }
    }
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
    Rule::SlotSequence => {
      let s = pair.as_str();
      // ## is slot sequence starting at 1, ##2 starts at 2, etc.
      let num = if s.len() > 2 {
        s[2..].parse().unwrap_or(1)
      } else {
        1
      };
      Expr::SlotSequence(num)
    }
    Rule::Constant | Rule::UnsignedConstant => {
      Expr::Constant(pair.as_str().trim().to_string())
    }
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
    Rule::ListExtended => {
      // Merged rule: List + optional suffix (PartIndexSuffix, ListAnonSuffix, ListCallSuffix)
      // Eliminates exponential backtracking for deeply nested lists.
      let inner_pairs: Vec<_> = pair.clone().into_inner().collect();
      let full_str = pair.as_str();

      // First inner pair is always List
      let list_expr = pair_to_expr(inner_pairs[0].clone());

      // Check for suffix types
      let has_part_index = inner_pairs
        .iter()
        .any(|p| matches!(p.as_rule(), Rule::PartIndexSuffix));
      let has_list_anon = inner_pairs
        .iter()
        .any(|p| matches!(p.as_rule(), Rule::ListAnonSuffix));
      let has_list_call = inner_pairs
        .iter()
        .any(|p| matches!(p.as_rule(), Rule::ListCallSuffix));
      let has_part_anon = inner_pairs
        .iter()
        .any(|p| matches!(p.as_rule(), Rule::ListPartAnonSuffix));

      if has_part_index && has_part_anon {
        // List[[...]] with anonymous function suffix: {a,b}[[1]] op ... &
        let part_indices: Vec<Expr> = inner_pairs
          .iter()
          .filter(|p| matches!(p.as_rule(), Rule::PartIndexSuffix))
          .flat_map(|p| p.clone().into_inner().map(pair_to_expr))
          .collect();
        let mut part_result = list_expr;
        for idx in &part_indices {
          part_result = Expr::Part {
            expr: Box::new(part_result),
            index: Box::new(idx.clone()),
          };
        }
        // Parse the full body (everything before &) as anonymous function body
        let body_str = {
          let s = full_str.trim();
          if let Some(amp_pos) = s.rfind('&') {
            if amp_pos == 0 || s.as_bytes()[amp_pos - 1] != b'&' {
              s[..amp_pos].trim()
            } else {
              s
            }
          } else {
            s
          }
        };
        let body = parse_anonymous_body(body_str);
        let suffix_pair = inner_pairs
          .iter()
          .find(|p| matches!(p.as_rule(), Rule::ListPartAnonSuffix))
          .unwrap();
        let anon_brackets: Vec<Vec<Expr>> = suffix_pair
          .clone()
          .into_inner()
          .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
          .map(|bracket| bracket.into_inner().map(pair_to_expr).collect())
          .collect();
        let anon_func = Expr::Function {
          body: Box::new(body),
        };
        if anon_brackets.is_empty() {
          anon_func
        } else {
          let mut result = Expr::CurriedCall {
            func: Box::new(anon_func),
            args: anon_brackets[0].clone(),
          };
          for args in anon_brackets.into_iter().skip(1) {
            result = Expr::CurriedCall {
              func: Box::new(result),
              args,
            };
          }
          result
        }
      } else if has_part_index {
        // List[[...]]: PartExtract with List base
        let part_indices: Vec<Expr> = inner_pairs
          .iter()
          .filter(|p| matches!(p.as_rule(), Rule::PartIndexSuffix))
          .flat_map(|p| p.clone().into_inner().map(pair_to_expr))
          .collect();
        let mut result = list_expr;
        for idx in &part_indices {
          result = Expr::Part {
            expr: Box::new(result),
            index: Box::new(idx.clone()),
          };
        }
        result
      } else if has_list_anon {
        // List& : ListAnonymousFunction
        let body_str = {
          let s = full_str.trim();
          if let Some(amp_pos) = s.rfind('&') {
            if amp_pos == 0 || s.as_bytes()[amp_pos - 1] != b'&' {
              s[..amp_pos].trim()
            } else {
              s
            }
          } else {
            s
          }
        };
        let body = parse_anonymous_body(body_str);
        let suffix_pair = inner_pairs
          .iter()
          .find(|p| matches!(p.as_rule(), Rule::ListAnonSuffix))
          .unwrap();
        let anon_brackets: Vec<Vec<Expr>> = suffix_pair
          .clone()
          .into_inner()
          .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
          .map(|bracket| bracket.into_inner().map(pair_to_expr).collect())
          .collect();
        let anon_func = Expr::Function {
          body: Box::new(body),
        };
        if anon_brackets.is_empty() {
          anon_func
        } else {
          let mut result = Expr::CurriedCall {
            func: Box::new(anon_func),
            args: anon_brackets[0].clone(),
          };
          for args in anon_brackets.into_iter().skip(1) {
            result = Expr::CurriedCall {
              func: Box::new(result),
              args,
            };
          }
          result
        }
      } else if has_list_call {
        // List[...]: ListCall
        let suffix_pair = inner_pairs
          .iter()
          .find(|p| matches!(p.as_rule(), Rule::ListCallSuffix))
          .unwrap();
        let bracket_sequences: Vec<Vec<Expr>> = suffix_pair
          .clone()
          .into_inner()
          .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
          .map(|bracket| bracket.into_inner().map(pair_to_expr).collect())
          .collect();
        let mut result = Expr::CurriedCall {
          func: Box::new(list_expr),
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
        // Plain List
        list_expr
      }
    }
    Rule::FunctionCallExtended => {
      // Merged rule: FunctionCall + optional Part extraction + optional anonymous function suffix
      // Inner pairs: (Identifier|SimpleAnonymousFunction) DerivativePrime? BracketArgs+ [PartIndexSuffix [FunctionCallPartAnonSuffix] | FunctionCallAnonSuffix]
      let inner_pairs: Vec<_> = pair.clone().into_inner().collect();
      let full_str = pair.as_str();

      let name_pair = &inner_pairs[0];

      // Check for derivative prime notation (f', f'', f''')
      let derivative_order = inner_pairs
        .iter()
        .find(|p| matches!(p.as_rule(), Rule::DerivativePrime))
        .map(|p| p.as_str().len());

      // Collect the function call's BracketArgs (consecutive BracketArgs after name, before any suffix)
      let fc_bracket_args: Vec<Vec<Expr>> = inner_pairs
        .iter()
        .skip(1) // skip Identifier/SimpleAnonymousFunction
        .skip_while(|p| matches!(p.as_rule(), Rule::DerivativePrime)) // skip optional DerivativePrime
        .take_while(|p| matches!(p.as_rule(), Rule::BracketArgs))
        .map(|bracket| bracket.clone().into_inner().map(pair_to_expr).collect())
        .collect();

      // Detect suffix types via named rules
      let has_part_index = inner_pairs
        .iter()
        .any(|p| matches!(p.as_rule(), Rule::PartIndexSuffix));
      let has_anon_suffix = inner_pairs
        .iter()
        .any(|p| matches!(p.as_rule(), Rule::FunctionCallAnonSuffix));
      let has_part_anon_suffix = inner_pairs
        .iter()
        .any(|p| matches!(p.as_rule(), Rule::FunctionCallPartAnonSuffix));

      // Extract part indices if present
      let part_indices: Vec<Expr> = inner_pairs
        .iter()
        .filter(|p| matches!(p.as_rule(), Rule::PartIndexSuffix))
        .flat_map(|p| p.clone().into_inner().map(pair_to_expr))
        .collect();

      // Build the base function call expression
      let base_func =
        if matches!(name_pair.as_rule(), Rule::SimpleAnonymousFunction) {
          let anon_expr = pair_to_expr(name_pair.clone());
          let mut result = Expr::CurriedCall {
            func: Box::new(anon_expr),
            args: fc_bracket_args[0].clone(),
          };
          for args in fc_bracket_args.iter().skip(1) {
            result = Expr::CurriedCall {
              func: Box::new(result),
              args: args.clone(),
            };
          }
          result
        } else if let Some(order) = derivative_order {
          // f'[x] → Derivative[1][f][x], f''[x] → Derivative[2][f][x], etc.
          let name = name_pair.as_str().to_string();
          // Derivative[n][f]
          let mut result = Expr::CurriedCall {
            func: Box::new(Expr::FunctionCall {
              name: "Derivative".to_string(),
              args: vec![Expr::Integer(order as i128)],
            }),
            args: vec![Expr::Identifier(name)],
          };
          // Apply bracket args: Derivative[n][f][x], then any further chained calls
          for args in fc_bracket_args.iter() {
            result = Expr::CurriedCall {
              func: Box::new(result),
              args: args.clone(),
            };
          }
          result
        } else {
          let name = name_pair.as_str().to_string();
          if fc_bracket_args.len() == 1 {
            Expr::FunctionCall {
              name,
              args: fc_bracket_args[0].clone(),
            }
          } else {
            let mut result = Expr::FunctionCall {
              name,
              args: fc_bracket_args[0].clone(),
            };
            for args in fc_bracket_args.iter().skip(1) {
              result = Expr::CurriedCall {
                func: Box::new(result),
                args: args.clone(),
              };
            }
            result
          }
        };

      // Helper: extract BracketArgs from a suffix pair
      let extract_suffix_brackets =
        |suffix_pair: &pest::iterators::Pair<Rule>| -> Vec<Vec<Expr>> {
          suffix_pair
            .clone()
            .into_inner()
            .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
            .map(|bracket| bracket.into_inner().map(pair_to_expr).collect())
            .collect::<Vec<Vec<Expr>>>()
        };

      // Helper: wrap body in Function and optionally apply to bracket args
      let make_anon_func = |body: Expr, bracket_args: Vec<Vec<Expr>>| -> Expr {
        let anon_func = Expr::Function {
          body: Box::new(body),
        };
        if bracket_args.is_empty() {
          anon_func
        } else {
          let mut result = Expr::CurriedCall {
            func: Box::new(anon_func),
            args: bracket_args[0].clone(),
          };
          for args in bracket_args.into_iter().skip(1) {
            result = Expr::CurriedCall {
              func: Box::new(result),
              args,
            };
          }
          result
        }
      };

      if has_part_index && has_part_anon_suffix {
        // PartAnonymousFunction: f[x][[i]] op ... &[args]
        let mut part_result = base_func;
        for idx in &part_indices {
          part_result = Expr::Part {
            expr: Box::new(part_result),
            index: Box::new(idx.clone()),
          };
        }
        // Re-parse the full body (everything before &) as anonymous function body
        let body_str = {
          let s = full_str.trim();
          if let Some(amp_pos) = s.rfind('&') {
            if amp_pos == 0 || s.as_bytes()[amp_pos - 1] != b'&' {
              s[..amp_pos].trim()
            } else {
              s
            }
          } else {
            s
          }
        };
        let body = parse_anonymous_body(body_str);
        let suffix_pair = inner_pairs
          .iter()
          .find(|p| matches!(p.as_rule(), Rule::FunctionCallPartAnonSuffix))
          .unwrap();
        let anon_brackets = extract_suffix_brackets(suffix_pair);
        make_anon_func(body, anon_brackets)
      } else if has_part_index {
        // Plain PartExtract: f[x][[i]]
        let mut result = base_func;
        for idx in &part_indices {
          result = Expr::Part {
            expr: Box::new(result),
            index: Box::new(idx.clone()),
          };
        }
        result
      } else if has_anon_suffix {
        // FunctionAnonymousFunction: f[x]&[args]
        let body_str = {
          let s = full_str.trim();
          if let Some(amp_pos) = s.rfind('&') {
            s[..amp_pos].trim()
          } else {
            s
          }
        };
        let body = parse_anonymous_body(body_str);
        let suffix_pair = inner_pairs
          .iter()
          .find(|p| matches!(p.as_rule(), Rule::FunctionCallAnonSuffix))
          .unwrap();
        let anon_brackets = extract_suffix_brackets(suffix_pair);
        make_anon_func(body, anon_brackets)
      } else {
        // Plain FunctionCall
        base_func
      }
    }
    Rule::FunctionCall => {
      let inner_pairs: Vec<_> = pair.into_inner().collect();
      let name_pair = &inner_pairs[0];
      // Check for derivative prime notation
      let derivative_order = inner_pairs
        .iter()
        .find(|p| matches!(p.as_rule(), Rule::DerivativePrime))
        .map(|p| p.as_str().len());
      // Collect bracket sequences separately for proper chained call handling
      let bracket_sequences: Vec<Vec<Expr>> = inner_pairs
        .iter()
        .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
        .map(|bracket| {
          bracket
            .clone()
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
        let anon_expr = pair_to_expr(name_pair.clone());
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
      } else if let Some(order) = derivative_order {
        // f'[x] → Derivative[1][f][x]
        let name = name_pair.as_str().to_string();
        let mut result = Expr::CurriedCall {
          func: Box::new(Expr::FunctionCall {
            name: "Derivative".to_string(),
            args: vec![Expr::Integer(order as i128)],
          }),
          args: vec![Expr::Identifier(name)],
        };
        for args in bracket_sequences.iter() {
          result = Expr::CurriedCall {
            func: Box::new(result),
            args: args.clone(),
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
      let inner_pairs: Vec<_> = pair.into_inner().collect();
      let name_pair = &inner_pairs[0];
      let name = name_pair.as_str().to_string();
      let derivative_order = inner_pairs
        .iter()
        .find(|p| matches!(p.as_rule(), Rule::DerivativePrime))
        .map(|p| p.as_str().len());
      let args: Vec<Expr> = inner_pairs
        .iter()
        .skip(1)
        .filter(|p| {
          !matches!(p.as_rule(), Rule::DerivativePrime) && p.as_str() != ","
        })
        .map(|p| pair_to_expr(p.clone()))
        .collect();
      if let Some(order) = derivative_order {
        // f'[x] → Derivative[n][f][x]
        Expr::CurriedCall {
          func: Box::new(Expr::CurriedCall {
            func: Box::new(Expr::FunctionCall {
              name: "Derivative".to_string(),
              args: vec![Expr::Integer(order as i128)],
            }),
            args: vec![Expr::Identifier(name)],
          }),
          args,
        }
      } else {
        Expr::FunctionCall { name, args }
      }
    }
    Rule::LeadingMinus => {
      // LeadingMinus is handled in parse_expression, not here directly
      // But if encountered standalone, treat as a sentinel
      Expr::Integer(0) // Should not be reached
    }
    Rule::Expression | Rule::ExpressionNoImplicit | Rule::ConditionExpr => {
      parse_expression(pair)
    }
    Rule::CompoundExpression => {
      let trailing_semi = pair.as_str().trim_end().ends_with(';');
      let mut exprs: Vec<Expr> = pair
        .into_inner()
        .filter(|p| p.as_str() != ";")
        .map(pair_to_expr)
        .collect();
      // A trailing ; means the result is Null (CompoundExpression[..., Null])
      if trailing_semi {
        exprs.push(Expr::Identifier("Null".to_string()));
      }
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
      let children: Vec<_> = pair.into_inner().collect();
      // Grammar: ConditionExpr ~ ("/;" ~ ConditionExpr)? ~ ("->" | ":>") ~ ConditionExpr
      // 2 children: pattern -> replacement
      // 3 children: pattern /; condition -> replacement
      let (pattern, replacement) = if children.len() == 3 {
        // pattern /; condition :> replacement
        // Store as Raw so the string-based pattern matcher can handle the /; condition
        let pattern_expr = pair_to_expr(children[0].clone());
        let condition_expr = pair_to_expr(children[1].clone());
        let pattern_str = format!(
          "{} /; {}",
          expr_to_string(&pattern_expr),
          expr_to_string(&condition_expr)
        );
        (Expr::Raw(pattern_str), pair_to_expr(children[2].clone()))
      } else {
        (
          pair_to_expr(children[0].clone()),
          pair_to_expr(children[1].clone()),
        )
      };
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
    Rule::PatternOptionalSimple => {
      // PatternOptionalSimple = { PatternName ~ "_" ~ ":" ~ Term }
      let mut inner = pair.into_inner();
      let name = inner.next().unwrap().as_str().to_string();
      let default_pair = inner.next().unwrap();
      let default = pair_to_expr(default_pair);
      Expr::PatternOptional {
        name,
        head: None,
        default: Box::new(default),
      }
    }
    Rule::PatternOptionalWithHead => {
      // PatternOptionalWithHead = { PatternName ~ "_" ~ Identifier ~ ":" ~ Term }
      let mut inner = pair.into_inner();
      let name = inner.next().unwrap().as_str().to_string();
      let head = Some(inner.next().unwrap().as_str().to_string());
      let default_pair = inner.next().unwrap();
      let default = pair_to_expr(default_pair);
      Expr::PatternOptional {
        name,
        head,
        default: Box::new(default),
      }
    }
    Rule::PatternTest => {
      // PatternTest: x_?test or _?test or _?(expr) or x_?(expr)
      let mut inner = pair.into_inner();
      // PatternName is optional; if present it's the first child
      let first = inner.next().unwrap();
      let (name, test_pair) = if first.as_rule() == Rule::PatternName {
        (first.as_str().to_string(), inner.next().unwrap())
      } else {
        // No PatternName — anonymous blank; first child is the test expression
        (String::new(), first)
      };
      let test = pair_to_expr(test_pair);
      Expr::PatternTest {
        name,
        test: Box::new(test),
      }
    }
    Rule::PatternCondition => {
      // Store the full pattern string as Raw to preserve test/condition info
      // The string-based pattern matching in apply_replace_all_direct handles these
      Expr::Raw(pair.as_str().to_string())
    }
    Rule::SimpleAnonymousFunction => {
      // Simple anonymous function like #^2& — never has BracketArgs
      let s = pair.as_str().trim().trim_end_matches('&');
      let body = parse_anonymous_body(s);
      Expr::Function {
        body: Box::new(body),
      }
    }
    Rule::ListCall => {
      // {f, g}[x] → CurriedCall with List as the func
      let inner_pairs: Vec<_> = pair.into_inner().collect();
      // First inner pair is the List, rest are BracketArgs
      let list_expr = pair_to_expr(inner_pairs[0].clone());
      let bracket_sequences: Vec<Vec<Expr>> = inner_pairs[1..]
        .iter()
        .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
        .map(|bracket| {
          bracket
            .clone()
            .into_inner()
            .filter(|p| {
              p.as_str() != "[" && p.as_str() != "]" && p.as_str() != ","
            })
            .map(pair_to_expr)
            .collect()
        })
        .collect();
      let mut result = Expr::CurriedCall {
        func: Box::new(list_expr),
        args: bracket_sequences[0].clone(),
      };
      for args in bracket_sequences.into_iter().skip(1) {
        result = Expr::CurriedCall {
          func: Box::new(result),
          args,
        };
      }
      result
    }
    Rule::FunctionAnonymousFunction
    | Rule::ParenAnonymousFunction
    | Rule::ListAnonymousFunction
    | Rule::PartAnonymousFunction => {
      // Anonymous function like If[#>0,#,0]& or (#===0)& or {#,#^2}&
      // May optionally have BracketArgs for direct calls: (#+1)&[5]
      let inner_pairs: Vec<_> = pair.clone().into_inner().collect();
      let bracket_args: Vec<Vec<Expr>> = inner_pairs
        .iter()
        .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
        .map(|bracket| {
          bracket
            .clone()
            .into_inner()
            .filter(|p| {
              p.as_str() != "[" && p.as_str() != "]" && p.as_str() != ","
            })
            .map(pair_to_expr)
            .collect()
        })
        .collect();
      // Extract body string: everything before the first BracketArgs, trimmed of &
      let body_str = if bracket_args.is_empty() {
        pair.as_str().trim().trim_end_matches('&').to_string()
      } else {
        // Find where the first BracketArgs starts in the string
        let first_bracket = inner_pairs
          .iter()
          .find(|p| matches!(p.as_rule(), Rule::BracketArgs))
          .unwrap();
        let bracket_start = first_bracket.as_span().start();
        let pair_start = pair.as_span().start();
        let body_end = bracket_start - pair_start;
        pair.as_str()[..body_end]
          .trim()
          .trim_end_matches('&')
          .to_string()
      };
      let body = parse_anonymous_body(&body_str);
      let anon_func = Expr::Function {
        body: Box::new(body),
      };
      if bracket_args.is_empty() {
        anon_func
      } else {
        // Build curried calls for direct application: (#+1)&[5] or (#+1)&[5][6]
        let mut result = Expr::CurriedCall {
          func: Box::new(anon_func),
          args: bracket_args[0].clone(),
        };
        for args in bracket_args.into_iter().skip(1) {
          result = Expr::CurriedCall {
            func: Box::new(result),
            args,
          };
        }
        result
      }
    }
    Rule::RuleAnonymousFunction => {
      // Anonymous function with Rule body: {#, First@#2} -> "Q" &
      let inner = pair.into_inner().next().unwrap(); // The ReplacementRule
      let body = pair_to_expr(inner);
      Expr::Function {
        body: Box::new(body),
      }
    }
    Rule::ListItemRule | Rule::FunctionArgRule => {
      // Combined ReplacementRule + optional anonymous function suffix (&)
      let inner_pairs: Vec<_> = pair.into_inner().collect();
      let rule_expr = pair_to_expr(inner_pairs[0].clone());
      let has_anon = inner_pairs
        .iter()
        .any(|p| matches!(p.as_rule(), Rule::RuleAnonSuffix));
      if has_anon {
        Expr::Function {
          body: Box::new(rule_expr),
        }
      } else {
        rule_expr
      }
    }
    Rule::PartExtract => {
      let mut inner = pair.into_inner();
      let base_expr = pair_to_expr(inner.next().unwrap());
      // Chain multiple indices as nested Part: a[[1,2,3]] -> Part[Part[Part[a,1],2],3]
      let mut result = base_expr;
      for idx_pair in inner {
        let index = pair_to_expr(idx_pair);
        result = Expr::Part {
          expr: Box::new(result),
          index: Box::new(index),
        };
      }
      result
    }
    Rule::Increment => {
      // x++ -> Increment[x]
      let inner = pair.into_inner().next().unwrap();
      let var = pair_to_expr(inner);
      Expr::FunctionCall {
        name: "Increment".to_string(),
        args: vec![var],
      }
    }
    Rule::Decrement => {
      // x-- -> Decrement[x]
      let inner = pair.into_inner().next().unwrap();
      let var = pair_to_expr(inner);
      Expr::FunctionCall {
        name: "Decrement".to_string(),
        args: vec![var],
      }
    }
    Rule::PreIncrement => {
      // ++x -> PreIncrement[x]
      let inner = pair.into_inner().next().unwrap();
      let var = pair_to_expr(inner);
      Expr::FunctionCall {
        name: "PreIncrement".to_string(),
        args: vec![var],
      }
    }
    Rule::PreDecrement => {
      // --x -> PreDecrement[x]
      let inner = pair.into_inner().next().unwrap();
      let var = pair_to_expr(inner);
      Expr::FunctionCall {
        name: "PreDecrement".to_string(),
        args: vec![var],
      }
    }
    Rule::Unset => {
      // x =. -> Unset[x]
      let inner = pair.into_inner().next().unwrap();
      let var = pair_to_expr(inner);
      Expr::FunctionCall {
        name: "Unset".to_string(),
        args: vec![var],
      }
    }
    Rule::AddTo => {
      // x += y -> AddTo[x, y]
      let mut inner = pair.into_inner();
      let var = pair_to_expr(inner.next().unwrap());
      let val = pair_to_expr(inner.next().unwrap());
      Expr::FunctionCall {
        name: "AddTo".to_string(),
        args: vec![var, val],
      }
    }
    Rule::SubtractFrom => {
      // x -= y -> SubtractFrom[x, y]
      let mut inner = pair.into_inner();
      let var = pair_to_expr(inner.next().unwrap());
      let val = pair_to_expr(inner.next().unwrap());
      Expr::FunctionCall {
        name: "SubtractFrom".to_string(),
        args: vec![var, val],
      }
    }
    Rule::TimesBy => {
      // x *= y -> TimesBy[x, y]
      let mut inner = pair.into_inner();
      let var = pair_to_expr(inner.next().unwrap());
      let val = pair_to_expr(inner.next().unwrap());
      Expr::FunctionCall {
        name: "TimesBy".to_string(),
        args: vec![var, val],
      }
    }
    Rule::DivideBy => {
      // x /= y -> DivideBy[x, y]
      let mut inner = pair.into_inner();
      let var = pair_to_expr(inner.next().unwrap());
      let val = pair_to_expr(inner.next().unwrap());
      Expr::FunctionCall {
        name: "DivideBy".to_string(),
        args: vec![var, val],
      }
    }
    Rule::ImplicitTimes => {
      // Implicit multiplication: x y z -> Times[x, y, z]
      // Each factor can optionally have a power suffix (ImplicitPowerSuffix)
      let inners: Vec<_> = pair.into_inner().collect();
      let mut factors: Vec<Expr> = Vec::new();
      let mut i = 0;
      while i < inners.len() {
        if inners[i].as_rule() == Rule::ImplicitPowerSuffix {
          // Power suffix follows the previous factor
          if let Some(base) = factors.pop() {
            let exponent =
              pair_to_expr(inners[i].clone().into_inner().next().unwrap());
            factors.push(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(base),
              right: Box::new(exponent),
            });
          }
        } else {
          factors.push(pair_to_expr(inners[i].clone()));
        }
        i += 1;
      }
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

/// Parse a PostfixFunction pair into an Expr, wrapping in Function if & is present.
/// In Wolfram, `x // f &` parses as `(f &)[x]`, i.e. & binds tighter than //.
fn parse_postfix_function(pair: Pair<Rule>) -> Expr {
  let has_ampersand = pair.as_str().trim_end().ends_with('&');
  // The inner children are the same (BaseFunctionCall or Identifier);
  // the "&" is an anonymous literal in the grammar and doesn't appear as a child.
  let func = pair_to_expr(pair.into_inner().next().unwrap());
  if has_ampersand {
    Expr::Function {
      body: Box::new(func),
    }
  } else {
    func
  }
}

/// Parse an expression with operators into an Expr
fn parse_expression(pair: Pair<Rule>) -> Expr {
  let mut inner: Vec<Pair<Rule>> = pair.into_inner().collect();

  if inner.is_empty() {
    return Expr::Raw(String::new());
  }

  // Find AnonymousFunctionSuffix position (if any) to split pre-& and post-& parts
  let anon_idx = inner
    .iter()
    .position(|p| p.as_rule() == Rule::AnonymousFunctionSuffix);

  // Split off continuation operators/postfix AFTER & (if any)
  let (anon_func_suffix, post_anon_pairs) = if let Some(idx) = anon_idx {
    let suffix = inner.remove(idx);
    let post_pairs: Vec<Pair<Rule>> = inner.drain(idx..).collect();
    let bracket_args: Vec<Vec<Expr>> = suffix
      .into_inner()
      .filter(|p| matches!(p.as_rule(), Rule::BracketArgs))
      .map(|bracket| bracket.into_inner().map(pair_to_expr).collect())
      .collect();
    (Some(bracket_args), post_pairs)
  } else {
    (None, Vec::new())
  };

  // Collect trailing PostfixFunction pairs from pre-& part
  let mut postfix_funcs: Vec<Pair<Rule>> = Vec::new();
  while inner
    .last()
    .is_some_and(|p| p.as_rule() == Rule::PostfixFunction)
  {
    postfix_funcs.push(inner.pop().unwrap());
  }
  postfix_funcs.reverse(); // restore left-to-right order

  // Check for Repeated/RepeatedNull suffix (.., ...)
  let repeated_suffix = if inner.last().is_some_and(|p| {
    p.as_rule() == Rule::RepeatedSuffix
      || p.as_rule() == Rule::RepeatedNullSuffix
  }) {
    let suffix = inner.pop().unwrap();
    Some(suffix.as_rule() == Rule::RepeatedNullSuffix)
  } else {
    None
  };

  // Check for trailing ReplaceAll/ReplaceRepeated suffix:
  // Term (Operator Term)* (ReplaceAllSuffix | ReplaceRepeatedSuffix)?
  let replace_rules = if inner.last().is_some_and(|p| {
    p.as_rule() == Rule::ReplaceAllSuffix
      || p.as_rule() == Rule::ReplaceRepeatedSuffix
  }) {
    let suffix = inner.pop().unwrap();
    let is_replace_repeated = suffix.as_rule() == Rule::ReplaceRepeatedSuffix;
    // The suffix contains the List or ReplacementRule as its child
    let rules_pair = suffix.into_inner().next().unwrap();
    Some((rules_pair, is_replace_repeated))
  } else {
    None
  };

  // Single term case (no operators, no replace, no repeated, no post-& continuation)
  if inner.len() == 1
    && replace_rules.is_none()
    && repeated_suffix.is_none()
    && post_anon_pairs.is_empty()
  {
    let mut result = pair_to_expr(inner.remove(0));
    for func_pair in postfix_funcs {
      let func = parse_postfix_function(func_pair);
      result = Expr::Postfix {
        expr: Box::new(result),
        func: Box::new(func),
      };
    }
    if let Some(bracket_args) = anon_func_suffix {
      result = Expr::Function {
        body: Box::new(result),
      };
      for args in bracket_args {
        result = Expr::CurriedCall {
          func: Box::new(result),
          args,
        };
      }
    }
    return result;
  }

  // Parse operators: Term (Operator Term)*
  // Build expression with proper precedence
  let mut terms: Vec<Expr> = Vec::new();
  let mut operators: Vec<String> = Vec::new();
  let mut leading_minus = false;
  let mut leading_not = false;

  for item in inner {
    match item.as_rule() {
      Rule::LeadingMinus => {
        // Insert synthetic 0 and "-" operator so that -x^2 becomes 0 - x^2
        // This ensures ^ binds tighter than unary minus
        leading_minus = true;
      }
      Rule::LeadingNot => {
        // !expr becomes Not[expr]
        leading_not = true;
      }
      Rule::Operator | Rule::ConditionOp => {
        operators.push(item.as_str().trim().to_string());
      }
      Rule::FactorialSuffix => {
        // n! → Factorial[n], n!! → Factorial2[n]
        if let Some(last) = terms.pop() {
          let func_name = if item.as_str() == "!!" {
            "Factorial2"
          } else {
            "Factorial"
          };
          terms.push(Expr::FunctionCall {
            name: func_name.to_string(),
            args: vec![last],
          });
        }
      }
      _ => {
        if leading_minus {
          terms.push(Expr::Integer(0));
          operators.push("-".to_string());
          leading_minus = false;
        }
        let expr = pair_to_expr(item);
        if leading_not {
          terms.push(Expr::FunctionCall {
            name: "Not".to_string(),
            args: vec![expr],
          });
          leading_not = false;
        } else {
          terms.push(expr);
        }
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

  // Apply Repeated/RepeatedNull suffix if present
  if let Some(is_repeated_null) = repeated_suffix {
    let name = if is_repeated_null {
      "RepeatedNull"
    } else {
      "Repeated"
    };
    result = Expr::FunctionCall {
      name: name.to_string(),
      args: vec![result],
    };
  }

  // Apply ReplaceAll/ReplaceRepeated if present.
  // In Wolfram Language, /. has higher precedence than = and :=, so:
  //   intA = expr /. rules  →  Set[intA, ReplaceAll[expr, rules]]
  // not: ReplaceAll[Set[intA, expr], rules]
  if let Some((rules_pair, is_replace_repeated)) = replace_rules {
    let rules = pair_to_expr(rules_pair);
    let make_replace = |e: Expr, r: Expr| -> Expr {
      if is_replace_repeated {
        Expr::ReplaceRepeated {
          expr: Box::new(e),
          rules: Box::new(r),
        }
      } else {
        Expr::ReplaceAll {
          expr: Box::new(e),
          rules: Box::new(r),
        }
      }
    };
    // If result is an assignment, push /. inside to the right-hand side
    result = match result {
      Expr::FunctionCall { ref name, ref args }
        if (name == "Set" || name == "SetDelayed") && args.len() == 2 =>
      {
        let lhs = args[0].clone();
        let rhs = args[1].clone();
        Expr::FunctionCall {
          name: name.clone(),
          args: vec![lhs, make_replace(rhs, rules)],
        }
      }
      _ => make_replace(result, rules),
    };
  }

  // Apply postfix functions
  for func_pair in postfix_funcs {
    let func = parse_postfix_function(func_pair);
    result = Expr::Postfix {
      expr: Box::new(result),
      func: Box::new(func),
    };
  }

  // Apply AnonymousFunctionSuffix (lowest precedence): expr &
  if let Some(bracket_args) = anon_func_suffix {
    result = Expr::Function {
      body: Box::new(result),
    };
    for args in bracket_args {
      result = Expr::CurriedCall {
        func: Box::new(result),
        args,
      };
    }

    // Apply continuation operators after & (e.g., #^2& @ 3)
    if !post_anon_pairs.is_empty() {
      let mut post_terms: Vec<Expr> = Vec::new();
      let mut post_ops: Vec<String> = Vec::new();
      let mut post_postfix: Vec<Pair<Rule>> = Vec::new();

      // Collect postfix functions from the end of post-& pairs
      let mut post_pairs = post_anon_pairs;
      while post_pairs
        .last()
        .is_some_and(|p| p.as_rule() == Rule::PostfixFunction)
      {
        post_postfix.push(post_pairs.pop().unwrap());
      }
      post_postfix.reverse();

      // Parse continuation as operator-term pairs
      post_terms.push(result);
      let mut iter = post_pairs.into_iter();
      while let Some(op_pair) = iter.next() {
        if op_pair.as_rule() == Rule::Operator {
          post_ops.push(op_pair.as_str().to_string());
          if let Some(term_pair) = iter.next() {
            post_terms.push(pair_to_expr(term_pair));
          }
        }
      }

      // Build expression tree with precedence
      result = build_binary_tree(post_terms, post_ops);

      // Apply post-& postfix functions
      for func_pair in post_postfix {
        let func = parse_postfix_function(func_pair);
        result = Expr::Postfix {
          expr: Box::new(result),
          func: Box::new(func),
        };
      }
    }
  }

  result
}

/// Get precedence of an operator (higher = binds tighter)
fn operator_precedence(op: &str) -> u8 {
  match op {
    ">>" | ">>>" => 0, // Put/PutAppend (lowest precedence)
    "=" | ":=" => 1,   // Assignment
    "~~" => 2,         // StringExpression (lower than Alternatives)
    "|" => 3,          // Alternatives
    "||" => 3,
    "&&" => 4,
    "==" | "!=" | "<" | "<=" | ">" | ">=" | "===" | "=!=" => 5, // Comparisons
    "->" | ":>" => 6,
    "+" | "-" => 7,
    "*" | "/" => 8,
    "<>" => 7,          // Same as + for string concatenation
    "." => 9,           // Dot (higher than arithmetic)
    "@@@" | "@@" => 10, // Apply/MapApply
    "/@" => 11,         // Map (higher than Apply)
    "@" => 12,          // Prefix application (higher than Map)
    "^" => 13,          // Power (highest)
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

    // For right-associative operators, use prec, otherwise use prec + 1
    let next_min_prec =
      if op_str == "^" || op_str == "@" || op_str == "=" || op_str == ":=" {
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
    "|" => Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      left: Box::new(left.clone()),
      right: Box::new(right.clone()),
    },
    "~~" => {
      // Flatten nested StringExpression (it's Flat/associative)
      let mut parts = Vec::new();
      match left {
        Expr::FunctionCall { name, args } if name == "StringExpression" => {
          parts.extend(args.clone());
        }
        _ => parts.push(left.clone()),
      }
      match right {
        Expr::FunctionCall { name, args } if name == "StringExpression" => {
          parts.extend(args.clone());
        }
        _ => parts.push(right.clone()),
      }
      Expr::FunctionCall {
        name: "StringExpression".to_string(),
        args: parts,
      }
    }
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
    "." => Expr::FunctionCall {
      name: "Dot".to_string(),
      args: vec![left.clone(), right.clone()],
    },
    "->" => Expr::Rule {
      pattern: Box::new(left.clone()),
      replacement: Box::new(right.clone()),
    },
    ":>" => Expr::RuleDelayed {
      pattern: Box::new(left.clone()),
      replacement: Box::new(right.clone()),
    },
    ">>" => Expr::FunctionCall {
      name: "Put".to_string(),
      args: vec![left.clone(), right.clone()],
    },
    ">>>" => Expr::FunctionCall {
      name: "PutAppend".to_string(),
      args: vec![left.clone(), right.clone()],
    },
    "=" => Expr::FunctionCall {
      name: "Set".to_string(),
      args: vec![left.clone(), right.clone()],
    },
    "^=" => Expr::FunctionCall {
      name: "UpSet".to_string(),
      args: vec![left.clone(), right.clone()],
    },
    "^:=" => Expr::FunctionCall {
      name: "UpSetDelayed".to_string(),
      args: vec![left.clone(), right.clone()],
    },
    ":=" => Expr::FunctionCall {
      name: "SetDelayed".to_string(),
      args: vec![left.clone(), right.clone()],
    },
    "==" | "!=" | "<" | "<=" | ">" | ">=" | "===" | "=!=" => {
      let comp_op = match op_str {
        "==" => ComparisonOp::Equal,
        "!=" => ComparisonOp::NotEqual,
        "<" => ComparisonOp::Less,
        "<=" => ComparisonOp::LessEqual,
        ">" => ComparisonOp::Greater,
        ">=" => ComparisonOp::GreaterEqual,
        "===" => ComparisonOp::SameQ,
        "=!=" => ComparisonOp::UnsameQ,
        _ => ComparisonOp::Equal,
      };
      // If the left side is already a Comparison, extend the chain
      {
        let mut cloned = left.clone();
        if let Expr::Comparison {
          operands: ref mut ops,
          operators: ref mut comp_ops,
        } = cloned
        {
          ops.push(right.clone());
          comp_ops.push(comp_op);
          cloned
        } else {
          Expr::Comparison {
            operands: vec![left.clone(), right.clone()],
            operators: vec![comp_op],
          }
        }
      }
    }
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
pub fn format_real(f: f64) -> String {
  if f.is_infinite() {
    if f > 0.0 {
      return "Infinity".to_string();
    } else {
      return "-Infinity".to_string();
    }
  }
  if f.is_nan() {
    return "Indeterminate".to_string();
  }
  let abs = f.abs();
  if abs == 0.0 {
    return "0.".to_string();
  }
  // Wolfram uses *^ scientific notation for |f| >= 1e6 or |f| < 1e-5
  if !(1e-5..1e6).contains(&abs) {
    format_real_scientific(f)
  } else if f.fract() == 0.0 {
    // Whole number in normal range - format with trailing dot
    format!("{}.", f as i64)
  } else {
    // Use Rust's default formatter which produces the shortest
    // representation that round-trips to the same f64 value
    format!("{}", f)
  }
}

/// Format a real number using Wolfram's *^ scientific notation.
/// E.g. 2.733467611516948*^33 or -1.5*^-6
///
/// Uses string manipulation on Rust's shortest round-trip representation
/// to avoid precision loss from dividing by 10^exp.
fn format_real_scientific(f: f64) -> String {
  let negative = f < 0.0;
  let abs = f.abs();
  // Use Rust's shortest round-trip representation (like Wolfram's approach)
  let s = format!("{}", abs);
  // Find the decimal point position
  let (digits, dot_pos) = if let Some(dot) = s.find('.') {
    // Remove the dot to get all digits, remember where it was
    let mut d = String::with_capacity(s.len());
    d.push_str(&s[..dot]);
    d.push_str(&s[dot + 1..]);
    (d, dot as i32)
  } else {
    (s.clone(), s.len() as i32)
  };
  // Remove leading zeros to find first significant digit
  let leading_zeros = digits.chars().take_while(|&c| c == '0').count();
  let sig_digits = &digits[leading_zeros..];
  if sig_digits.is_empty() {
    return "0.*^0".to_string();
  }
  // Exponent: position of first significant digit relative to decimal point
  let exp = dot_pos - leading_zeros as i32 - 1;
  // Build mantissa: first digit, dot, remaining digits
  let mut mantissa = String::new();
  if negative {
    mantissa.push('-');
  }
  mantissa.push_str(&sig_digits[..1]);
  mantissa.push('.');
  if sig_digits.len() > 1 {
    mantissa.push_str(&sig_digits[1..]);
  }
  // Trim trailing zeros after the decimal point, keeping the dot
  // Wolfram uses "1.*^6" not "1.0*^6"
  while mantissa.ends_with('0') {
    mantissa.pop();
  }
  format!("{}*^{}", mantissa, exp)
}

/// If expr is Times[negative_coeff, rest...], return Some(Times[abs(coeff), rest...]).
/// Works for both BinaryOp{Times} and FunctionCall{Times} forms.
fn negate_leading_negative_in_times(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => match left.as_ref() {
      Expr::Integer(n) if *n < 0 => {
        if *n == -1 {
          Some(right.as_ref().clone())
        } else {
          Some(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-n)),
            right: right.clone(),
          })
        }
      }
      Expr::FunctionCall { name, args }
        if name == "Rational"
          && args.len() == 2
          && matches!(&args[0], Expr::Integer(n) if *n < 0) =>
      {
        let n = if let Expr::Integer(n) = &args[0] {
          *n
        } else {
          return None;
        };
        let pos_rat = Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-n), args[1].clone()],
        };
        if -n == 1 {
          // Rational[-1, d] * x → x/d
          Some(Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: right.clone(),
            right: Box::new(args[1].clone()),
          })
        } else {
          Some(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(pos_rat),
            right: right.clone(),
          })
        }
      }
      _ => None,
    },
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      match &args[0] {
        Expr::Integer(n) if *n < 0 => {
          if *n == -1 {
            let rest = args[1..].to_vec();
            Some(if rest.len() == 1 {
              rest[0].clone()
            } else {
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: rest,
              }
            })
          } else {
            let mut new_args = vec![Expr::Integer(-n)];
            new_args.extend_from_slice(&args[1..]);
            Some(if new_args.len() == 1 {
              new_args[0].clone()
            } else {
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: new_args,
              }
            })
          }
        }
        Expr::FunctionCall { name: rn, args: ra }
          if rn == "Rational"
            && ra.len() == 2
            && matches!(&ra[0], Expr::Integer(n) if *n < 0) =>
        {
          let n = if let Expr::Integer(n) = &ra[0] {
            *n
          } else {
            return None;
          };
          let pos_rat = Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(-n), ra[1].clone()],
          };
          let mut new_args = vec![pos_rat];
          new_args.extend_from_slice(&args[1..]);
          Some(if new_args.len() == 1 {
            new_args[0].clone()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: new_args,
            }
          })
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// Format a Quantity unit expression without quoting.
/// Wolfram displays units unquoted: Meters, Miles/Hours, Meters/Seconds^2
fn quantity_unit_to_string(unit: &Expr) -> String {
  match unit {
    Expr::Identifier(s) | Expr::String(s) => s.clone(),
    // Power must come before the general BinaryOp arm to avoid being shadowed
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let exp_str = expr_to_string(right);
      let exp_fmt = if matches!(right.as_ref(), Expr::Integer(_)) {
        exp_str
      } else {
        format!("({})", exp_str)
      };
      format!("{}^{}", quantity_unit_to_string(left), exp_fmt)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      format!(
        "{}/{}",
        quantity_unit_to_string(left),
        quantity_unit_to_string(right)
      )
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      format!(
        "{}*{}",
        quantity_unit_to_string(left),
        quantity_unit_to_string(right)
      )
    }
    Expr::BinaryOp { .. } => expr_to_string(unit),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      let exp_str = expr_to_string(&args[1]);
      let exp_fmt = if matches!(args[1], Expr::Integer(_)) {
        exp_str
      } else {
        format!("({})", exp_str)
      };
      format!("{}^{}", quantity_unit_to_string(&args[0]), exp_fmt)
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let parts: Vec<String> =
        args.iter().map(quantity_unit_to_string).collect();
      parts.join("*")
    }
    _ => expr_to_string(unit),
  }
}

/// Convert an Expr back to a string representation
pub fn expr_to_string(expr: &Expr) -> String {
  match expr {
    Expr::Integer(n) => n.to_string(),
    Expr::BigInteger(n) => n.to_string(),
    Expr::Real(f) => format_real(*f),
    Expr::BigFloat(digits, prec) => format!("{}`{}.", digits, prec),
    Expr::String(s) => {
      let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r");
      format!("\"{}\"", escaped)
    }
    Expr::Identifier(s) => s.clone(),
    Expr::Slot(n) => {
      format!("#{}", n)
    }
    Expr::SlotSequence(n) => {
      format!("##{}", n)
    }
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_string).collect();
      format!("{{{}}}", parts.join(", "))
    }
    Expr::FunctionCall { name, args } => {
      // Special case: Quantity[n, unit] — unit shown as quoted string(s)
      if name == "Quantity" && args.len() == 2 {
        let mag_str = expr_to_string(&args[0]);
        let unit_str = quantity_unit_to_string(&args[1]);
        return format!("Quantity[{}, {}]", mag_str, unit_str);
      }
      // Special case: Skeleton[n] displays as <<n>>
      if name == "Skeleton" && args.len() == 1 {
        return format!("<<{}>>", expr_to_string(&args[0]));
      }
      // Special case: StringSkeleton[n] displays as <<n>>
      if name == "StringSkeleton" && args.len() == 1 {
        return format!("<<{}>>", expr_to_string(&args[0]));
      }
      // Special case: Repeated[x] displays as x..
      if name == "Repeated" && args.len() == 1 {
        return format!("{}..", expr_to_string(&args[0]));
      }
      // Special case: RepeatedNull[x] displays as x...
      if name == "RepeatedNull" && args.len() == 1 {
        return format!("{}...", expr_to_string(&args[0]));
      }
      // BlankSequence[] → __, BlankSequence[h] → __h
      if name == "BlankSequence" {
        if args.is_empty() {
          return "__".to_string();
        }
        if args.len() == 1
          && let Expr::Identifier(h) = &args[0]
        {
          return format!("__{}", h);
        }
      }
      // BlankNullSequence[] → ___, BlankNullSequence[h] → ___h
      if name == "BlankNullSequence" {
        if args.is_empty() {
          return "___".to_string();
        }
        if args.len() == 1
          && let Expr::Identifier(h) = &args[0]
        {
          return format!("___{}", h);
        }
      }
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
      if name == "Rule" && args.len() == 2 {
        return format!(
          "{} -> {}",
          expr_to_string(&args[0]),
          expr_to_string(&args[1])
        );
      }
      if name == "RuleDelayed" && args.len() == 2 {
        return format!(
          "{} :> {}",
          expr_to_string(&args[0]),
          expr_to_string(&args[1])
        );
      }
      if name == "Condition" && args.len() == 2 {
        return format!(
          "{} /; {}",
          expr_to_string(&args[0]),
          expr_to_string(&args[1])
        );
      }
      if name == "PatternTest" && args.len() == 2 {
        let pat = expr_to_string(&args[0]);
        let test = expr_to_string(&args[1]);
        // Simple _ doesn't need parens, everything else does
        if pat == "_" || pat == "__" || pat == "___" {
          return format!("{}?{}", pat, test);
        }
        return format!("({})?{}", pat, test);
      }
      if name == "Increment" && args.len() == 1 {
        return format!("{}++", expr_to_string(&args[0]));
      }
      if name == "Decrement" && args.len() == 1 {
        return format!("{}--", expr_to_string(&args[0]));
      }
      if name == "PreIncrement" && args.len() == 1 {
        return format!("++{}", expr_to_string(&args[0]));
      }
      if name == "PreDecrement" && args.len() == 1 {
        return format!("--{}", expr_to_string(&args[0]));
      }
      if name == "Optional" && args.len() == 1 {
        return format!("{}.", expr_to_string(&args[0]));
      }
      if name == "Optional" && args.len() == 2 {
        return format!(
          "{}:{}",
          expr_to_string(&args[0]),
          expr_to_string(&args[1])
        );
      }
      if name == "NonCommutativeMultiply" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_string).collect();
        return parts.join("**");
      }
      // Special case: Minus[a, b, ...] with wrong arity displays with Unicode minus
      if name == "Minus" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_string).collect();
        return parts.join(" \u{2212} ");
      }
      // Special case: Dot[a, b] displays as a . b (infix notation)
      if name == "Dot" && args.len() == 2 {
        return format!(
          "{} . {}",
          expr_to_string(&args[0]),
          expr_to_string(&args[1])
        );
      }
      if name == "Composition" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_string).collect();
        return parts.join(" @* ");
      }
      // Special case: Therefore[a, b, ...] displays as a ∴ b ∴ ...
      if name == "Therefore" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_string).collect();
        return parts.join(" \u{2234} ");
      }
      // Special case: Because[a, b, ...] displays as a ∵ b ∵ ...
      if name == "Because" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_string).collect();
        return parts.join(" \u{2235} ");
      }
      // Special case: Or[a, b, ...] displays as a || b || ...
      // Wolfram wraps And subterms in parens: (a && b) || (c && d)
      if name == "Or" && args.len() >= 2 {
        let parts: Vec<String> = args
          .iter()
          .map(|arg| {
            let s = expr_to_string(arg);
            let is_and = matches!(
              arg,
              Expr::BinaryOp {
                op: BinaryOperator::And,
                ..
              }
            ) || matches!(arg, Expr::FunctionCall { name, .. } if name == "And");
            if is_and {
              format!("({})", s)
            } else {
              s
            }
          })
          .collect();
        return parts.join(" || ");
      }
      // Special case: And[a, b, ...] displays as a && b && ...
      if name == "And" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_string).collect();
        return parts.join(" && ");
      }
      // Special case: Times displays as infix with *
      if name == "Times" && args.len() >= 2 {
        // Handle Times[Rational[1, d], expr] as "expr/d"
        // Handle Times[Rational[-1, d], expr] as "-expr/d"
        // Handle Times[Rational[1, d], expr] as "expr/d" (Wolfram convention)
        if args.len() == 2
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && matches!((&rargs[0], &rargs[1]), (Expr::Integer(1), Expr::Integer(d)) if *d > 0)
          && let Expr::Integer(d) = &rargs[1]
        {
          let inner = expr_to_string(&args[1]);
          return format!("{}/{}", inner, d);
        }
        // Handle Times[Rational[n, d], expr] as "(n*expr)/d" (Wolfram convention)
        if args.len() == 2
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && let Expr::Integer(n) = &rargs[0]
          && let Expr::Integer(d) = &rargs[1]
          && *n != 1
          && *n != -1
          && *d > 0
        {
          let inner = expr_to_string(&args[1]);
          let inner_str = if matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Plus")
            || matches!(
              &args[1],
              Expr::BinaryOp {
                op: BinaryOperator::Plus | BinaryOperator::Minus,
                ..
              }
            ) {
            format!("({})", inner)
          } else {
            inner
          };
          return format!("({}*{})/{}", n, inner_str, d);
        }
        // Handle Times[-1, x, ...] as "-x*..."
        if matches!(&args[0], Expr::Integer(-1)) {
          let rest = args[1..]
            .iter()
            .map(|a| {
              let s = expr_to_string(a);
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
          // Wolfram wraps negated products in parens when the factors are
          // all non-constant symbols: -(a*b), but -I*a, -2*a*b without parens.
          // The rule: parens needed when coefficient is -1 and ALL remaining
          // factors are regular variables (not I or numeric constants).
          let rest_factors = &args[1..];
          let all_regular_symbols = rest_factors
            .iter()
            .all(|a| matches!(a, Expr::Identifier(n) if n != "I"));
          let needs_neg_parens = (rest_factors.len() >= 2
            && all_regular_symbols)
            || (args.len() == 2
              && (matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Times")
                || matches!(
                  &args[1],
                  Expr::BinaryOp {
                    op: BinaryOperator::Times | BinaryOperator::Divide,
                    ..
                  }
                )));
          if needs_neg_parens {
            return format!("-({})", rest);
          }
          return format!("-{}", rest);
        }
        // Complex number grouping: Times containing I with non-numeric remaining factors
        // e.g. Times[2, I, Sqrt[3]] → (2*I)*Sqrt[3], Times[Rational[1,2], I, Pi] → (I/2)*Pi
        let has_imaginary =
          args.iter().any(|a| matches!(a, Expr::Identifier(n) if n == "I"));
        if has_imaginary {
          let mut numeric_factors: Vec<&Expr> = Vec::new();
          let mut symbolic_factors: Vec<&Expr> = Vec::new();
          for arg in args.iter() {
            match arg {
              Expr::Integer(_) | Expr::Real(_) => numeric_factors.push(arg),
              Expr::Identifier(n) if n == "I" => {}
              Expr::FunctionCall { name: rn, .. } if rn == "Rational" => {
                numeric_factors.push(arg);
              }
              _ => symbolic_factors.push(arg),
            }
          }
          if !symbolic_factors.is_empty() {
            let i_part_opt: Option<String> = if numeric_factors.is_empty() {
              Some("I".to_string())
            } else if numeric_factors.len() == 1 {
              match numeric_factors[0] {
                Expr::Integer(1) => Some("I".to_string()),
                Expr::Integer(-1) => Some("-I".to_string()),
                Expr::Integer(n) => Some(format!("({}*I)", n)),
                Expr::FunctionCall { name: rn, args: ra }
                  if rn == "Rational" && ra.len() == 2 =>
                {
                  if let (Expr::Integer(num), Expr::Integer(den)) = (&ra[0], &ra[1]) {
                    if *num == 1 {
                      Some(format!("(I/{})", den))
                    } else if *num == -1 {
                      Some(format!("(-(I/{}))", den))
                    } else {
                      Some(format!("(({num}*I)/{den})"))
                    }
                  } else {
                    None
                  }
                }
                _ => None,
              }
            } else {
              None
            };
            if let Some(i_part) = i_part_opt {
              let rest: Vec<String> = symbolic_factors
                .iter()
                .map(|a| {
                  let s = expr_to_string(a);
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
                .collect();
              return format!("{}*{}", i_part, rest.join("*"));
            }
          }
        }
        return args
          .iter()
          .map(|a| {
            let s = expr_to_string(a);
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
      // Special case: Plus displays as infix with + (with spaces)
      if name == "Plus" && args.len() >= 2 {
        let mut result = expr_to_string(&args[0]);
        for arg in args.iter().skip(1) {
          if let Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } = arg
          {
            result.push_str(" - ");
            result.push_str(&expr_to_string(operand));
          } else if let Expr::BinaryOp {
            op: BinaryOperator::Times,
            left,
            right,
          } = arg
          {
            if matches!(left.as_ref(), Expr::Integer(-1)) {
              result.push_str(" - ");
              result.push_str(&expr_to_string(right));
            } else if let Expr::Integer(n) = left.as_ref() {
              if *n < 0 {
                result.push_str(" - ");
                let pos = Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(Expr::Integer(-n)),
                  right: right.clone(),
                };
                result.push_str(&expr_to_string(&pos));
              } else {
                result.push_str(" + ");
                result.push_str(&expr_to_string(arg));
              }
            } else {
              result.push_str(" + ");
              result.push_str(&expr_to_string(arg));
            }
          } else if let Expr::FunctionCall {
            name: fn_name,
            args: fn_args,
          } = arg
          {
            if fn_name == "Times" && fn_args.len() >= 2 {
              // Check if leading factor is negative
              let neg_coeff = match &fn_args[0] {
                Expr::Integer(n) if *n < 0 => Some(if *n == -1 {
                  None // coefficient of -1 means just negate
                } else {
                  Some(Expr::Integer(-n))
                }),
                Expr::FunctionCall { name: rn, args: ra }
                  if rn == "Rational"
                    && ra.len() == 2
                    && matches!(&ra[0], Expr::Integer(n) if *n < 0) =>
                {
                  if let Expr::Integer(n) = &ra[0] {
                    if *n == -1 {
                      // Rational[-1, d] → just Rational[1, d] = 1/d
                      Some(Some(Expr::FunctionCall {
                        name: "Rational".to_string(),
                        args: vec![Expr::Integer(1), ra[1].clone()],
                      }))
                    } else {
                      Some(Some(Expr::FunctionCall {
                        name: "Rational".to_string(),
                        args: vec![Expr::Integer(-n), ra[1].clone()],
                      }))
                    }
                  } else {
                    None
                  }
                }
                _ => None,
              };
              if let Some(pos_coeff) = neg_coeff {
                result.push_str(" - ");
                let pos_term = match pos_coeff {
                  None => {
                    // Times[-1, rest...] → rest
                    let pos_args = fn_args[1..].to_vec();
                    if pos_args.len() == 1 {
                      pos_args[0].clone()
                    } else {
                      Expr::FunctionCall {
                        name: "Times".to_string(),
                        args: pos_args,
                      }
                    }
                  }
                  Some(new_coeff) => {
                    let mut new_args = vec![new_coeff];
                    new_args.extend_from_slice(&fn_args[1..]);
                    if new_args.len() == 1 {
                      new_args[0].clone()
                    } else {
                      Expr::FunctionCall {
                        name: "Times".to_string(),
                        args: new_args,
                      }
                    }
                  }
                };
                result.push_str(&expr_to_string(&pos_term));
              } else {
                result.push_str(" + ");
                result.push_str(&expr_to_string(arg));
              }
            } else {
              result.push_str(" + ");
              result.push_str(&expr_to_string(arg));
            }
          } else if let Expr::Integer(n) = arg {
            if *n < 0 {
              result.push_str(" - ");
              result.push_str(&expr_to_string(&Expr::Integer(-n)));
            } else {
              result.push_str(" + ");
              result.push_str(&expr_to_string(arg));
            }
          } else {
            // Fallback: check if the rendered form starts with "-"
            let s = expr_to_string(arg);
            if s.starts_with('-') {
              result.push_str(" - ");
              result.push_str(&s[1..]);
            } else {
              result.push_str(" + ");
              result.push_str(&s);
            }
          }
        }
        return result;
      }
      // Special case: Power displays as infix with ^ (no spaces)
      if name == "Power" && args.len() == 2 {
        let base_str = expr_to_string(&args[0]);
        let exp_str = expr_to_string(&args[1]);
        // Wrap base in parens if it's a Plus (lower precedence than Power)
        let base = if matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Plus")
          || matches!(
            &args[0],
            Expr::BinaryOp {
              op: BinaryOperator::Plus | BinaryOperator::Minus,
              ..
            }
          ) {
          format!("({})", base_str)
        } else {
          base_str
        };
        // Wrap exponent in parens if it's a Plus, negative, or Times with negative coefficient
        let exp = if matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Plus")
          || matches!(
            &args[1],
            Expr::BinaryOp {
              op: BinaryOperator::Plus | BinaryOperator::Minus,
              ..
            }
          )
          || matches!(&args[1], Expr::Integer(n) if *n < 0)
          || matches!(
            &args[1],
            Expr::UnaryOp {
              op: UnaryOperator::Minus,
              ..
            }
          )
          || matches!(&args[1], Expr::FunctionCall { name: tname, args: targs } if tname == "Times" && !targs.is_empty() && matches!(&targs[0], Expr::Integer(n) if *n < 0))
        {
          format!("({})", exp_str)
        } else {
          exp_str
        };
        return format!("{}^{}", base, exp);
      }
      // Special case: Derivative[n, f, x] displays as Derivative[n][f][x]
      // and Derivative[n, f] displays as Derivative[n][f]
      if name == "Derivative" && args.len() >= 2 {
        let n_str = expr_to_string(&args[0]);
        let f_str = expr_to_string(&args[1]);
        if args.len() == 3 {
          let x_str = expr_to_string(&args[2]);
          return format!("Derivative[{}][{}][{}]", n_str, f_str, x_str);
        }
        return format!("Derivative[{}][{}]", n_str, f_str);
      }
      let parts: Vec<String> = args.iter().map(expr_to_string).collect();
      format!("{}[{}]", name, parts.join(", "))
    }
    Expr::BinaryOp { op, left, right } => {
      // Special case: (-x)/y should display as -(x/y) (Wolfram convention)
      // Only when the numerator is exactly Times[-1, x], not a negative integer coefficient
      if matches!(op, BinaryOperator::Divide) {
        let negated_inner = match left.as_ref() {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: t_left,
            right: t_right,
          } if matches!(t_left.as_ref(), Expr::Integer(-1)) => {
            Some(t_right.as_ref())
          }
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            operand,
          } => Some(operand.as_ref()),
          Expr::FunctionCall { name, args }
            if name == "Times"
              && args.len() >= 2
              && matches!(&args[0], Expr::Integer(-1)) =>
          {
            None // handled below for FunctionCall variant
          }
          _ => None,
        };
        if let Some(inner) = negated_inner {
          let inner_div = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(inner.clone()),
            right: right.clone(),
          };
          let inner_str = expr_to_string(&inner_div);
          return format!("-({})", inner_str);
        }
        // Handle FunctionCall Times[-1, ...] as numerator
        if let Expr::FunctionCall { name, args } = left.as_ref()
          && name == "Times"
          && args.len() >= 2
          && matches!(&args[0], Expr::Integer(-1))
        {
          let pos_args = args[1..].to_vec();
          let pos_numerator = if pos_args.len() == 1 {
            pos_args[0].clone()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: pos_args,
            }
          };
          let inner_div = Expr::BinaryOp {
            op: BinaryOperator::Divide,
            left: Box::new(pos_numerator),
            right: right.clone(),
          };
          let inner_str = expr_to_string(&inner_div);
          return format!("-({})", inner_str);
        }
        // Special case: 1/identifier → identifier^(-1) (Wolfram InputForm convention)
        // Only for simple identifiers; products, functions, and sums are handled differently
        if matches!(left.as_ref(), Expr::Integer(1)) {
          if let Expr::Identifier(s) = right.as_ref() {
            return format!("{}^(-1)", s);
          }
        }
      }

      // Special case: BinaryOp Times with Rational[1, d] or Rational[-1, d]
      // Rational[1, d] * expr → "expr/d", Rational[-1, d] * expr → "-expr/d"
      if matches!(op, BinaryOperator::Times)
        && let Expr::FunctionCall {
          name: rname,
          args: rargs,
        } = left.as_ref()
        && rname == "Rational"
        && rargs.len() == 2
        && let (Expr::Integer(num), Expr::Integer(den)) = (&rargs[0], &rargs[1])
        && *num == 1
        && *den > 0
      {
        let inner = expr_to_string(right);
        return format!("{}/{}", inner, den);
      }

      // Special case: BinaryOp Times with Rational[n, d] * expr → "(n*expr)/d"
      if matches!(op, BinaryOperator::Times)
        && let Expr::FunctionCall {
          name: rname,
          args: rargs,
        } = left.as_ref()
        && rname == "Rational"
        && rargs.len() == 2
        && let Expr::Integer(num) = &rargs[0]
        && let Expr::Integer(den) = &rargs[1]
        && *num != 1
        && *num != -1
        && *den > 0
      {
        let inner = expr_to_string(right);
        let inner_str = if matches!(right.as_ref(), Expr::FunctionCall { name, .. } if name == "Plus")
          || matches!(
            right.as_ref(),
            Expr::BinaryOp {
              op: BinaryOperator::Plus | BinaryOperator::Minus,
              ..
            }
          ) {
          format!("({})", inner)
        } else {
          inner
        };
        return format!("({}*{})/{}", num, inner_str, den);
      }

      // Special case: Times[-1, expr] should display as -expr
      if matches!(op, BinaryOperator::Times)
        && matches!(left.as_ref(), Expr::Integer(-1))
      {
        let right_str = expr_to_string(right);
        // Add parens for lower-precedence ops (Plus/Minus), products (Times), and divisions (Divide)
        // Wolfram displays -(a*b) not -a*b, and -(a/b) not -a/b
        return if matches!(
          right.as_ref(),
          Expr::BinaryOp {
            op: BinaryOperator::Plus
              | BinaryOperator::Minus
              | BinaryOperator::Times
              | BinaryOperator::Divide,
            ..
          }
        ) || matches!(right.as_ref(), Expr::FunctionCall { name, .. } if name == "Plus" || name == "Times")
        {
          format!("-({})", right_str)
        } else {
          format!("-{}", right_str)
        };
      }

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
      // Special case: a + Times[neg, ...] should display as a - abs(neg)*...
      // Handles both BinaryOp{Times} and FunctionCall{Times} forms
      if matches!(op, BinaryOperator::Plus) {
        let negated_term = negate_leading_negative_in_times(right);
        if let Some(abs_term) = negated_term {
          let left_str = expr_to_string(left);
          let right_str = expr_to_string(&abs_term);
          return format!("{} - {}", left_str, right_str);
        }
      }

      // Mathematica uses no spaces for *, /, ^ but spaces for +, -, &&, ||
      let (op_str, needs_space) = match op {
        BinaryOperator::Plus => ("+", true),
        BinaryOperator::Minus => ("-", true),
        BinaryOperator::Times => ("*", false),
        BinaryOperator::Divide => ("/", false),
        BinaryOperator::Power => ("^", false),
        BinaryOperator::And => ("&&", true),
        BinaryOperator::Or => ("||", true),
        BinaryOperator::StringJoin => ("<>", false),
        BinaryOperator::Alternatives => ("|", true),
      };

      // Helper to check if expr is a lower-precedence additive expression
      let is_additive = |e: &Expr| -> bool {
        matches!(
          e,
          Expr::BinaryOp {
            op: BinaryOperator::Plus | BinaryOperator::Minus,
            ..
          }
        ) || matches!(e, Expr::FunctionCall { name, .. } if name == "Plus")
      };

      let is_multiplicative = matches!(
        op,
        BinaryOperator::Times | BinaryOperator::Divide | BinaryOperator::Power
      );

      let left_str = expr_to_string(left);
      let right_str = expr_to_string(right);

      // Special case: Or wraps And children in parens (Wolfram convention)
      if matches!(op, BinaryOperator::Or) {
        let is_and_expr = |e: &Expr| -> bool {
          matches!(
            e,
            Expr::BinaryOp {
              op: BinaryOperator::And,
              ..
            }
          ) || matches!(e, Expr::FunctionCall { name, .. } if name == "And")
        };
        let lf = if is_and_expr(left) {
          format!("({})", left_str)
        } else {
          left_str
        };
        let rf = if is_and_expr(right) {
          format!("({})", right_str)
        } else {
          right_str
        };
        return format!("{} || {}", lf, rf);
      }

      // Add parens when a lower-precedence expr is inside a higher-precedence one,
      // or when the numerator of a division is a product (Wolfram convention)
      let left_needs_parens = (is_multiplicative && is_additive(left))
        || (matches!(op, BinaryOperator::Divide)
          && (matches!(
            left.as_ref(),
            Expr::BinaryOp {
              op: BinaryOperator::Times,
              ..
            }
          ) || matches!(
            left.as_ref(),
            Expr::FunctionCall { name, .. } if name == "Times"
          )));
      let left_formatted = if left_needs_parens {
        format!("({})", left_str)
      } else {
        left_str
      };
      let is_right_multiplicative = |e: &Expr| -> bool {
        matches!(
          e,
          Expr::BinaryOp {
            op: BinaryOperator::Times | BinaryOperator::Divide,
            ..
          }
        ) || matches!(e, Expr::FunctionCall { name, .. } if name == "Times")
      };
      // Check if right side of Power is a negative (Times[-1, ...]) or UnaryOp::Minus
      let is_negative_expr = |e: &Expr| -> bool {
        matches!(
          e,
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left,
            ..
          } if matches!(left.as_ref(), Expr::Integer(-1))
        ) || matches!(
          e,
          Expr::UnaryOp {
            op: UnaryOperator::Minus,
            ..
          }
        ) || matches!(e, Expr::Integer(n) if *n < 0)
          || matches!(e, Expr::FunctionCall { name, args } if name == "Times" && !args.is_empty() && matches!(&args[0], Expr::Integer(n) if *n < 0))
      };
      let needs_right_parens = (is_multiplicative && is_additive(right))
        || (matches!(op, BinaryOperator::Divide)
          && is_right_multiplicative(right))
        || (matches!(op, BinaryOperator::Power)
          && (matches!(
            right.as_ref(),
            Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2
          ) || matches!(
            right.as_ref(),
            Expr::BinaryOp {
              op: BinaryOperator::Divide,
              ..
            }
          ) || is_right_multiplicative(right)
            || is_negative_expr(right)));
      let right_formatted = if needs_right_parens {
        format!("({})", right_str)
      } else {
        right_str
      };

      if needs_space {
        format!("{} {} {}", left_formatted, op_str, right_formatted)
      } else {
        format!("{}{}{}", left_formatted, op_str, right_formatted)
      }
    }
    Expr::UnaryOp { op, operand } => {
      let inner = expr_to_string(operand);
      if matches!(op, UnaryOperator::Not) {
        // Not: Wolfram formats as " !expr" (leading space) or " !(expr)"
        let needs_parens = matches!(
          operand.as_ref(),
          Expr::BinaryOp {
            op: BinaryOperator::And | BinaryOperator::Or,
            ..
          }
        ) || matches!(
          operand.as_ref(),
          Expr::FunctionCall { name, .. } if name == "And" || name == "Or"
        );
        if needs_parens {
          format!(" !({})", inner)
        } else {
          format!(" !{}", inner)
        }
      } else {
        // Minus needs parens around Plus, Minus, Times, Divide
        let needs_parens = matches!(
          operand.as_ref(),
          Expr::BinaryOp {
            op: BinaryOperator::Plus
              | BinaryOperator::Minus
              | BinaryOperator::Times
              | BinaryOperator::Divide,
            ..
          }
        ) || matches!(
          operand.as_ref(),
          Expr::FunctionCall { name, args } if (name == "Times" || name == "Plus") && args.len() >= 2
        );
        if needs_parens {
          format!("-({})", inner)
        } else {
          format!("-{}", inner)
        }
      }
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
      let func_str = expr_to_string(func);
      // Parenthesize func if it's a Function or NamedFunction (lower precedence than /@ )
      let func_display = match func.as_ref() {
        Expr::Function { .. } | Expr::NamedFunction { .. } => format!("({})", func_str),
        _ => func_str,
      };
      format!("{} /@ {}", func_display, expr_to_string(list))
    }
    Expr::Apply { func, list } => {
      format!("{} @@ {}", expr_to_string(func), expr_to_string(list))
    }
    Expr::MapApply { func, list } => {
      format!("{} @@@ {}", expr_to_string(func), expr_to_string(list))
    }
    Expr::PrefixApply { func, arg } => {
      // f @ g is displayed as f[g] (Wolfram converts @ to function call notation)
      let func_str = expr_to_string(func);
      let arg_str = expr_to_string(arg);
      // Parenthesize func if it's complex (not a simple identifier or function call)
      let func_display = match func.as_ref() {
        Expr::Identifier(_) | Expr::FunctionCall { .. } | Expr::CurriedCall { .. } => func_str,
        _ => format!("({})", func_str),
      };
      format!("{}[{}]", func_display, arg_str)
    }
    Expr::Postfix { expr, func } => {
      format!("{} // {}", expr_to_string(expr), expr_to_string(func))
    }
    Expr::Part { expr, index } => {
      // Flatten nested Part into a single [[i, j, k]] notation
      let mut indices = vec![expr_to_string(index)];
      let mut base = expr.as_ref();
      while let Expr::Part {
        expr: inner_expr,
        index: inner_index,
      } = base
      {
        indices.push(expr_to_string(inner_index));
        base = inner_expr.as_ref();
      }
      indices.reverse();
      format!("{}[[{}]]", expr_to_string(base), indices.join(","))
    }
    Expr::Function { body } => {
      // Wolfram shows anonymous functions with trailing space: "f & " (not "f &")
      format!("{} & ", expr_to_string(body))
    }
    Expr::NamedFunction { params, body } => {
      if params.len() == 1 {
        format!("Function[{}, {}]", params[0], expr_to_string(body))
      } else {
        format!(
          "Function[{{{}}}, {}]",
          params.join(", "),
          expr_to_string(body)
        )
      }
    }
    Expr::Pattern { name, head } => {
      if let Some(h) = head {
        format!("{}_{}", name, h)
      } else {
        format!("{}_", name)
      }
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => {
      if let Some(h) = head {
        format!("{}_{}:{}", name, h, expr_to_string(default))
      } else {
        format!("{}_:{}", name, expr_to_string(default))
      }
    }
    Expr::PatternTest { name, test } => {
      let test_str = expr_to_string(test);
      // If test is a simple identifier, use x_?Test form; otherwise wrap in parens
      let needs_parens = !matches!(test.as_ref(), Expr::Identifier(_));
      if needs_parens {
        format!("{}_?({})", name, test_str)
      } else {
        format!("{}_?{}", name, test_str)
      }
    }
    Expr::Constant(s) => s.clone(),
    Expr::Raw(s) => s.clone(),
    Expr::Image { .. } => "-Image-".to_string(),
    Expr::Graphics { is_3d, .. } => {
      if *is_3d {
        "-Graphics3D-".to_string()
      } else {
        "-Graphics-".to_string()
      }
    }
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
      // Sequence[] (empty sequence) displays as nothing in Wolfram output
      if name == "Sequence" && args.is_empty() {
        return String::new();
      }
      // Special case: Quantity[n, unit] — unit shown as quoted string(s)
      if name == "Quantity" && args.len() == 2 {
        let mag_str = expr_to_output(&args[0]);
        let unit_str = quantity_unit_to_string(&args[1]);
        return format!("Quantity[{}, {}]", mag_str, unit_str);
      }
      // Special case: FullForm[expr] displays as FullForm[<output form of inner>]
      // This matches wolframscript behavior: FullForm[1/z] → FullForm[z^(-1)]
      if name == "FullForm" && args.len() == 1 {
        return format!("FullForm[{}]", expr_to_output(&args[0]));
      }
      // CForm[expr] displays as CForm[evaluated_expr] in OutputForm
      if name == "CForm" && args.len() == 1 {
        return format!("CForm[{}]", expr_to_output(&args[0]));
      }
      // Special case: Skeleton[n] displays as <<n>>
      if name == "Skeleton" && args.len() == 1 {
        return format!("<<{}>>", expr_to_output(&args[0]));
      }
      // Special case: StringSkeleton[n] displays as <<n>>
      if name == "StringSkeleton" && args.len() == 1 {
        return format!("<<{}>>", expr_to_output(&args[0]));
      }
      // Special case: Repeated[x] displays as x..
      if name == "Repeated" && args.len() == 1 {
        return format!("{}..", expr_to_output(&args[0]));
      }
      // Special case: RepeatedNull[x] displays as x...
      if name == "RepeatedNull" && args.len() == 1 {
        return format!("{}...", expr_to_output(&args[0]));
      }
      // BlankSequence[] → __, BlankSequence[h] → __h
      if name == "BlankSequence" {
        if args.is_empty() {
          return "__".to_string();
        }
        if args.len() == 1
          && let Expr::Identifier(h) = &args[0]
        {
          return format!("__{}", h);
        }
      }
      // BlankNullSequence[] → ___, BlankNullSequence[h] → ___h
      if name == "BlankNullSequence" {
        if args.is_empty() {
          return "___".to_string();
        }
        if args.len() == 1
          && let Expr::Identifier(h) = &args[0]
        {
          return format!("___{}", h);
        }
      }
      // Special case: BaseForm[expr, base] displays as BaseForm[expr, base] in OutputForm
      // (matching wolframscript; the subscript rendering only appears in notebook cells)
      if name == "BaseForm" && args.len() == 2 {
        return format!(
          "BaseForm[{}, {}]",
          expr_to_output(&args[0]),
          expr_to_output(&args[1])
        );
      }
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
      // Special case: Minus[a, b, ...] with wrong arity displays with Unicode minus
      if name == "Minus" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_output).collect();
        return parts.join(" \u{2212} ");
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
          } else if let Expr::BinaryOp {
            op: BinaryOperator::Times,
            left,
            right,
          } = arg
          {
            if matches!(left.as_ref(), Expr::Integer(-1)) {
              result.push_str(" - ");
              result.push_str(&expr_to_output(right));
            } else if let Expr::Integer(n) = left.as_ref() {
              if *n < 0 {
                result.push_str(" - ");
                // Display as (-n)*right
                let pos = Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(Expr::Integer(-n)),
                  right: right.clone(),
                };
                result.push_str(&expr_to_output(&pos));
              } else {
                result.push_str(" + ");
                result.push_str(&expr_to_output(arg));
              }
            } else {
              result.push_str(" + ");
              result.push_str(&expr_to_output(arg));
            }
          } else if let Expr::FunctionCall {
            name: fn_name,
            args: fn_args,
          } = arg
          {
            if fn_name == "Times" && fn_args.len() >= 2 {
              // Check if leading factor is negative
              let neg_coeff = match &fn_args[0] {
                Expr::Integer(n) if *n < 0 => Some(if *n == -1 {
                  None // coefficient of -1 means just negate
                } else {
                  Some(Expr::Integer(-n))
                }),
                Expr::FunctionCall { name: rn, args: ra }
                  if rn == "Rational"
                    && ra.len() == 2
                    && matches!(&ra[0], Expr::Integer(n) if *n < 0) =>
                {
                  if let Expr::Integer(n) = &ra[0] {
                    if *n == -1 {
                      Some(Some(Expr::FunctionCall {
                        name: "Rational".to_string(),
                        args: vec![Expr::Integer(1), ra[1].clone()],
                      }))
                    } else {
                      Some(Some(Expr::FunctionCall {
                        name: "Rational".to_string(),
                        args: vec![Expr::Integer(-n), ra[1].clone()],
                      }))
                    }
                  } else {
                    None
                  }
                }
                _ => None,
              };
              if let Some(pos_coeff) = neg_coeff {
                result.push_str(" - ");
                let pos_term = match pos_coeff {
                  None => {
                    // Times[-1, rest...] → rest
                    let pos_args = fn_args[1..].to_vec();
                    if pos_args.len() == 1 {
                      pos_args[0].clone()
                    } else {
                      Expr::FunctionCall {
                        name: "Times".to_string(),
                        args: pos_args,
                      }
                    }
                  }
                  Some(new_coeff) => {
                    let mut new_args = vec![new_coeff];
                    new_args.extend_from_slice(&fn_args[1..]);
                    if new_args.len() == 1 {
                      new_args[0].clone()
                    } else {
                      Expr::FunctionCall {
                        name: "Times".to_string(),
                        args: new_args,
                      }
                    }
                  }
                };
                result.push_str(&expr_to_output(&pos_term));
              } else {
                result.push_str(" + ");
                result.push_str(&expr_to_output(arg));
              }
            } else {
              result.push_str(" + ");
              result.push_str(&expr_to_output(arg));
            }
          } else if let Expr::Integer(n) = arg {
            if *n < 0 {
              result.push_str(" - ");
              result.push_str(&expr_to_output(&Expr::Integer(-n)));
            } else {
              result.push_str(" + ");
              result.push_str(&expr_to_output(arg));
            }
          } else {
            result.push_str(" + ");
            result.push_str(&expr_to_output(arg));
          }
        }
        return result;
      }
      // Special case: Times displays as infix with * (no spaces)
      if name == "Times" && args.len() >= 2 {
        // Handle Times[Rational[1, d], expr] as "expr/d"
        if args.len() == 2
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && matches!((&rargs[0], &rargs[1]), (Expr::Integer(1), Expr::Integer(d)) if *d > 0)
          && let Expr::Integer(d) = &rargs[1]
        {
          let inner = expr_to_output(&args[1]);
          return format!("{}/{}", inner, d);
        }
        // Handle Times[Rational[n, d], expr] as "(n*expr)/d" (Wolfram convention)
        if args.len() == 2
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && let Expr::Integer(n) = &rargs[0]
          && let Expr::Integer(d) = &rargs[1]
          && *n != 1
          && *n != -1
          && *d > 0
        {
          let inner = expr_to_output(&args[1]);
          let inner_str = if matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Plus")
            || matches!(
              &args[1],
              Expr::BinaryOp {
                op: BinaryOperator::Plus | BinaryOperator::Minus,
                ..
              }
            ) {
            format!("({})", inner)
          } else {
            inner
          };
          return format!("({}*{})/{}", n, inner_str, d);
        }
        // Handle Times[-1, x] as "-x" and Times[-1, x, y, ...] as "-x*y*..."
        if args.len() >= 2 && matches!(&args[0], Expr::Integer(-1)) {
          let rest = args[1..]
            .iter()
            .map(|a| {
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
          // Wolfram wraps negated products in parens when all remaining factors
          // are regular variables: -(a*b), but not when I or function calls are
          // involved: -I*Conjugate[a], -2*Conjugate[a]*I.
          let rest_factors = &args[1..];
          let all_regular_symbols = rest_factors
            .iter()
            .all(|a| matches!(a, Expr::Identifier(n) if n != "I"));
          let needs_neg_parens = (rest_factors.len() >= 2
            && all_regular_symbols)
            || (args.len() == 2
              && (matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Times")
                || matches!(
                  &args[1],
                  Expr::BinaryOp {
                    op: BinaryOperator::Times | BinaryOperator::Divide,
                    ..
                  }
                )));
          if needs_neg_parens {
            return format!("-({})", rest);
          }
          return format!("-{}", rest);
        }
        // Complex number grouping: Times containing I with non-numeric remaining factors
        // e.g. Times[2, I, Sqrt[3]] → (2*I)*Sqrt[3], Times[Rational[1,2], I, Pi] → (I/2)*Pi
        let has_imaginary =
          args.iter().any(|a| matches!(a, Expr::Identifier(n) if n == "I"));
        if has_imaginary {
          let mut numeric_factors: Vec<&Expr> = Vec::new();
          let mut symbolic_factors: Vec<&Expr> = Vec::new();
          for arg in args.iter() {
            match arg {
              Expr::Integer(_) | Expr::Real(_) => numeric_factors.push(arg),
              Expr::Identifier(n) if n == "I" => {}
              Expr::FunctionCall { name: rn, .. } if rn == "Rational" => {
                numeric_factors.push(arg);
              }
              _ => symbolic_factors.push(arg),
            }
          }
          if !symbolic_factors.is_empty() {
            let i_part_opt: Option<String> = if numeric_factors.is_empty() {
              Some("I".to_string())
            } else if numeric_factors.len() == 1 {
              match numeric_factors[0] {
                Expr::Integer(1) => Some("I".to_string()),
                Expr::Integer(-1) => Some("-I".to_string()),
                Expr::Integer(n) => Some(format!("({}*I)", n)),
                Expr::FunctionCall { name: rn, args: ra }
                  if rn == "Rational" && ra.len() == 2 =>
                {
                  if let (Expr::Integer(num), Expr::Integer(den)) = (&ra[0], &ra[1]) {
                    if *num == 1 {
                      Some(format!("(I/{})", den))
                    } else if *num == -1 {
                      Some(format!("(-(I/{}))", den))
                    } else {
                      Some(format!("(({num}*I)/{den})"))
                    }
                  } else {
                    None
                  }
                }
                _ => None,
              }
            } else {
              None
            };
            if let Some(i_part) = i_part_opt {
              let rest: Vec<String> = symbolic_factors
                .iter()
                .map(|a| {
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
                .collect();
              return format!("{}*{}", i_part, rest.join("*"));
            }
          }
        }
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
        let base_str = expr_to_output(&args[0]);
        let exp_str = expr_to_output(&args[1]);
        // Wrap base in parens if it's a Plus (lower precedence than Power)
        let base = if matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Plus")
          || matches!(
            &args[0],
            Expr::BinaryOp {
              op: BinaryOperator::Plus | BinaryOperator::Minus,
              ..
            }
          ) {
          format!("({})", base_str)
        } else {
          base_str
        };
        // Wrap exponent in parens if it's a Plus, negative, or Times with negative coefficient
        let exp = if matches!(&args[1], Expr::FunctionCall { name, .. } if name == "Plus")
          || matches!(
            &args[1],
            Expr::BinaryOp {
              op: BinaryOperator::Plus | BinaryOperator::Minus,
              ..
            }
          )
          || matches!(&args[1], Expr::Integer(n) if *n < 0)
          || matches!(
            &args[1],
            Expr::UnaryOp {
              op: UnaryOperator::Minus,
              ..
            }
          )
          || matches!(&args[1], Expr::FunctionCall { name: tname, args: targs } if tname == "Times" && !targs.is_empty() && matches!(&targs[0], Expr::Integer(n) if *n < 0))
        {
          format!("({})", exp_str)
        } else {
          exp_str
        };
        return format!("{}^{}", base, exp);
      }
      if name == "Rule" && args.len() == 2 {
        return format!(
          "{} -> {}",
          expr_to_output(&args[0]),
          expr_to_output(&args[1])
        );
      }
      if name == "RuleDelayed" && args.len() == 2 {
        return format!(
          "{} :> {}",
          expr_to_output(&args[0]),
          expr_to_output(&args[1])
        );
      }
      if name == "Condition" && args.len() == 2 {
        return format!(
          "{} /; {}",
          expr_to_output(&args[0]),
          expr_to_output(&args[1])
        );
      }
      if name == "PatternTest" && args.len() == 2 {
        let pat = expr_to_output(&args[0]);
        let test = expr_to_output(&args[1]);
        if pat == "_" || pat == "__" || pat == "___" {
          return format!("{}?{}", pat, test);
        }
        return format!("({})?{}", pat, test);
      }
      if name == "Increment" && args.len() == 1 {
        return format!("{}++", expr_to_output(&args[0]));
      }
      if name == "Decrement" && args.len() == 1 {
        return format!("{}--", expr_to_output(&args[0]));
      }
      if name == "PreIncrement" && args.len() == 1 {
        return format!("++{}", expr_to_output(&args[0]));
      }
      if name == "PreDecrement" && args.len() == 1 {
        return format!("--{}", expr_to_output(&args[0]));
      }
      if name == "Optional" && args.len() == 1 {
        return format!("{}.", expr_to_output(&args[0]));
      }
      if name == "Optional" && args.len() == 2 {
        return format!(
          "{}:{}",
          expr_to_output(&args[0]),
          expr_to_output(&args[1])
        );
      }
      if name == "NonCommutativeMultiply" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_output).collect();
        return parts.join("**");
      }
      // Special case: Dot[a, b] displays as a . b (infix notation)
      if name == "Dot" && args.len() == 2 {
        return format!(
          "{} . {}",
          expr_to_output(&args[0]),
          expr_to_output(&args[1])
        );
      }
      if name == "Composition" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_output).collect();
        return parts.join(" @* ");
      }
      // Special case: Therefore[a, b, ...] displays as a ∴ b ∴ ...
      if name == "Therefore" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_output).collect();
        return parts.join(" \u{2234} ");
      }
      // Special case: Because[a, b, ...] displays as a ∵ b ∵ ...
      if name == "Because" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_output).collect();
        return parts.join(" \u{2235} ");
      }
      // Special case: Or[a, b, ...] displays as a || b || ...
      // Wolfram wraps And subterms in parens: (a && b) || (c && d)
      if name == "Or" && args.len() >= 2 {
        let parts: Vec<String> = args
          .iter()
          .map(|arg| {
            let s = expr_to_output(arg);
            let is_and = matches!(
              arg,
              Expr::BinaryOp {
                op: BinaryOperator::And,
                ..
              }
            ) || matches!(arg, Expr::FunctionCall { name, .. } if name == "And");
            if is_and {
              format!("({})", s)
            } else {
              s
            }
          })
          .collect();
        return parts.join(" || ");
      }
      // Special case: And[a, b, ...] displays as a && b && ...
      if name == "And" && args.len() >= 2 {
        let parts: Vec<String> = args.iter().map(expr_to_output).collect();
        return parts.join(" && ");
      }
      // Special case: Row[{exprs...}] concatenates; Row[{exprs...}, sep] joins with separator
      if name == "Row"
        && (args.len() == 1 || args.len() == 2)
        && let Some(Expr::List(items)) = args.first()
      {
        let parts: Vec<String> = items.iter().map(expr_to_output).collect();
        if args.len() == 2 {
          let sep = expr_to_output(&args[1]);
          return parts.join(&sep);
        }
        return parts.concat();
      }
      // Special case: Derivative[n, f, x] displays as Derivative[n][f][x]
      if name == "Derivative" && args.len() >= 2 {
        let n_str = expr_to_output(&args[0]);
        let f_str = expr_to_output(&args[1]);
        if args.len() == 3 {
          let x_str = expr_to_output(&args[2]);
          return format!("Derivative[{}][{}][{}]", n_str, f_str, x_str);
        }
        return format!("Derivative[{}][{}]", n_str, f_str);
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

/// Render Expr in InputForm - like expr_to_output but strings are quoted.
pub fn expr_to_input_form(expr: &Expr) -> String {
  match expr {
    Expr::String(s) => {
      let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r");
      format!("\"{}\"", escaped)
    }
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_input_form).collect();
      format!("{{{}}}", parts.join(", "))
    }
    Expr::Association(items) => {
      let parts: Vec<String> = items
        .iter()
        .map(|(k, v)| {
          format!("{} -> {}", expr_to_input_form(k), expr_to_input_form(v))
        })
        .collect();
      format!("<|{}|>", parts.join(", "))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => {
      format!(
        "{} -> {}",
        expr_to_input_form(pattern),
        expr_to_input_form(replacement)
      )
    }
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      format!(
        "{} :> {}",
        expr_to_input_form(pattern),
        expr_to_input_form(replacement)
      )
    }
    // FunctionCall: handle FullForm specially in InputForm (keep the wrapper)
    Expr::FunctionCall { name, args }
      if name == "FullForm" && args.len() == 1 =>
    {
      format!("FullForm[{}]", expr_to_input_form(&args[0]))
    }
    // BaseForm: InputForm shows BaseForm[n, base] structure (not subscript notation)
    Expr::FunctionCall { name, args } if name == "BaseForm" && args.len() == 2 => {
      format!(
        "BaseForm[{}, {}]",
        expr_to_input_form(&args[0]),
        expr_to_input_form(&args[1])
      )
    }
    // CForm: InputForm shows CForm[expr] structure (not C code string)
    Expr::FunctionCall { name, args } if name == "CForm" && args.len() == 1 => {
      format!("CForm[{}]", expr_to_input_form(&args[0]))
    }
    // Unevaluated: InputForm strips the wrapper, showing just the inner expression
    Expr::FunctionCall { name, args } if name == "Unevaluated" && args.len() == 1 => {
      expr_to_input_form(&args[0])
    }
    // StringSkeleton[n]: InputForm shows <<n>> with InputForm content
    Expr::FunctionCall { name, args } if name == "StringSkeleton" && args.len() == 1 => {
      format!("<<{}>>", expr_to_input_form(&args[0]))
    }
    // StringExpression[a, b, c]: InputForm shows a~~b~~c with quoted strings
    Expr::FunctionCall { name, args } if name == "StringExpression" && !args.is_empty() => {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      parts.join("~~")
    }
    // StringForm: InputForm shows StringForm["template", args...] with quoted string
    Expr::FunctionCall { name, args } if name == "StringForm" && !args.is_empty() => {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      format!("StringForm[{}]", parts.join(", "))
    }
    // Row, TableForm, MatrixForm: display directive wrappers, show structure in InputForm
    Expr::FunctionCall { name, args }
      if (name == "Row" || name == "TableForm" || name == "MatrixForm")
        && !args.is_empty() =>
    {
      let parts: Vec<String> = args.iter().map(expr_to_input_form).collect();
      format!("{}[{}]", name, parts.join(", "))
    }
    // For all other cases, delegate to expr_to_output (which handles
    // infix Plus/Times/Power, Rational, etc.)
    _ => expr_to_output(expr),
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
/// Substitute slots in a list of expressions, expanding SlotSequence into multiple args.
fn substitute_slots_expand(exprs: &[Expr], values: &[Expr]) -> Vec<Expr> {
  let mut result = Vec::new();
  for e in exprs {
    let substituted = substitute_slots(e, values);
    // Flatten Sequence[...] into the argument list
    if let Expr::FunctionCall { name, args } = &substituted
      && name == "Sequence"
    {
      result.extend(args.iter().cloned());
      continue;
    }
    result.push(substituted);
  }
  result
}

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
    Expr::SlotSequence(n) => {
      let start = if *n == 0 { 0 } else { n - 1 };
      if start < values.len() {
        let seq: Vec<Expr> = values[start..].to_vec();
        Expr::FunctionCall {
          name: "Sequence".to_string(),
          args: seq,
        }
      } else {
        Expr::FunctionCall {
          name: "Sequence".to_string(),
          args: vec![],
        }
      }
    }
    Expr::List(items) => Expr::List(substitute_slots_expand(items, values)),
    Expr::FunctionCall { name, args } if name == "Slot" && args.len() == 1 => {
      if let Expr::Integer(n) = &args[0] {
        let index = if *n <= 0 { 0 } else { (*n as usize) - 1 };
        if index < values.len() {
          values[index].clone()
        } else {
          expr.clone()
        }
      } else {
        expr.clone()
      }
    }
    Expr::FunctionCall { name, args }
      if name == "SlotSequence" && args.len() == 1 =>
    {
      if let Expr::Integer(n) = &args[0] {
        let start = if *n <= 0 { 0 } else { (*n as usize) - 1 };
        if start < values.len() {
          let seq: Vec<Expr> = values[start..].to_vec();
          Expr::FunctionCall {
            name: "Sequence".to_string(),
            args: seq,
          }
        } else {
          Expr::FunctionCall {
            name: "Sequence".to_string(),
            args: vec![],
          }
        }
      } else {
        expr.clone()
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: substitute_slots_expand(args, values),
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
    Expr::Function { body } => {
      // Do NOT substitute slots inside nested Function bodies.
      // Slots (#, #2, etc.) bind to the innermost & (Function).
      Expr::Function { body: body.clone() }
    }
    Expr::NamedFunction { params, body } => {
      // Named functions don't use slots, so no substitution needed
      Expr::NamedFunction {
        params: params.clone(),
        body: body.clone(),
      }
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => Expr::PatternOptional {
      name: name.clone(),
      head: head.clone(),
      default: Box::new(substitute_slots(default, values)),
    },
    Expr::PatternTest { name, test } => Expr::PatternTest {
      name: name.clone(),
      test: Box::new(substitute_slots(test, values)),
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
    Expr::FunctionCall { name, args } => {
      let new_args: Vec<Expr> = args
        .iter()
        .map(|e| substitute_variable(e, var_name, value))
        .collect();
      if name == var_name {
        // The function name matches the variable being substituted.
        // Transform into a CurriedCall so the value is applied to the args.
        Expr::CurriedCall {
          func: Box::new(value.clone()),
          args: new_args,
        }
      } else {
        Expr::FunctionCall {
          name: name.clone(),
          args: new_args,
        }
      }
    }
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
    Expr::NamedFunction { params, body } => {
      // Don't substitute if var_name is one of the function's own parameters
      // (they are locally scoped)
      if params.contains(&var_name.to_string()) {
        Expr::NamedFunction {
          params: params.clone(),
          body: body.clone(),
        }
      } else {
        Expr::NamedFunction {
          params: params.clone(),
          body: Box::new(substitute_variable(body, var_name, value)),
        }
      }
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => Expr::PatternOptional {
      name: name.clone(),
      head: head.clone(),
      default: Box::new(substitute_variable(default, var_name, value)),
    },
    Expr::PatternTest { name, test } => Expr::PatternTest {
      name: name.clone(),
      test: Box::new(substitute_variable(test, var_name, value)),
    },
    // Atoms that don't contain the variable
    _ => expr.clone(),
  }
}

/// Convert an Expr to i64 if possible (for BaseForm base argument etc.)
fn expr_to_i64(expr: &Expr) -> Option<i64> {
  match expr {
    Expr::Integer(n) => Some(*n as i64),
    _ => None,
  }
}

/// Convert an integer to a string in the given base (2-36).
fn integer_to_base_string(mut n: i128, base: u32) -> String {
  if n == 0 {
    return "0".to_string();
  }
  let negative = n < 0;
  if negative {
    n = -n;
  }
  let mut digits = Vec::new();
  let mut val = n as u128;
  while val > 0 {
    let digit = (val % base as u128) as u32;
    digits.push(char::from_digit(digit, base).unwrap());
    val /= base as u128;
  }
  digits.reverse();
  let s: String = digits.into_iter().collect();
  if negative { format!("-{}", s) } else { s }
}

/// Convert a BigInteger to a string in the given base.
fn bigint_to_base_string(n: &num_bigint::BigInt, base: u32) -> String {
  use num_bigint::Sign;
  use num_traits::Zero;

  if n.is_zero() {
    return "0".to_string();
  }

  let (sign, mut val) = (n.sign(), n.magnitude().clone());
  let base_big = num_bigint::BigUint::from(base);
  let mut digits = Vec::new();

  while !val.is_zero() {
    let rem = &val % &base_big;
    use num_traits::ToPrimitive;
    let digit = rem.to_u32().unwrap();
    digits.push(char::from_digit(digit, base).unwrap());
    val /= &base_big;
  }

  digits.reverse();
  let s: String = digits.into_iter().collect();
  if sign == Sign::Minus {
    format!("-{}", s)
  } else {
    s
  }
}

/// Convert a real number to a string in the given base.
fn real_to_base_string(f: f64, base: u32) -> String {
  if f == 0.0 {
    return "0.".to_string();
  }
  let negative = f < 0.0;
  let f = f.abs();

  // Integer part
  let int_part = f.floor() as u128;
  let frac_part = f - int_part as f64;

  let int_str = if int_part == 0 {
    "0".to_string()
  } else {
    let mut digits = Vec::new();
    let mut val = int_part;
    while val > 0 {
      let digit = (val % base as u128) as u32;
      digits.push(char::from_digit(digit, base).unwrap());
      val /= base as u128;
    }
    digits.reverse();
    digits.into_iter().collect()
  };

  if frac_part == 0.0 {
    return if negative {
      format!("-{}.", int_str)
    } else {
      format!("{}.", int_str)
    };
  }

  // Fractional part
  let mut frac_digits = Vec::new();
  let mut frac = frac_part;
  let max_digits = 16; // enough precision for f64
  for _ in 0..max_digits {
    frac *= base as f64;
    let digit = frac.floor() as u32;
    frac_digits.push(char::from_digit(digit.min(base - 1), base).unwrap());
    frac -= digit as f64;
    if frac.abs() < 1e-15 {
      break;
    }
  }

  // Remove trailing zeros
  while frac_digits.last() == Some(&'0') {
    frac_digits.pop();
  }

  let frac_str: String = frac_digits.into_iter().collect();
  if negative {
    format!("-{}.{}", int_str, frac_str)
  } else {
    format!("{}.{}", int_str, frac_str)
  }
}

/// Format an expression in the given base for BaseForm display.
fn format_in_base(expr: &Expr, base: u32) -> String {
  match expr {
    Expr::Integer(n) => integer_to_base_string(*n, base),
    Expr::BigInteger(n) => bigint_to_base_string(n, base),
    Expr::Real(f) => real_to_base_string(*f, base),
    Expr::List(items) => {
      let parts: Vec<String> =
        items.iter().map(|e| format_in_base(e, base)).collect();
      format!("{{{}}}", parts.join(", "))
    }
    _ => expr_to_output(expr),
  }
}

/// Convert a number to Unicode subscript digit characters.
fn to_subscript_digits(mut n: u64) -> String {
  const SUBSCRIPTS: [char; 10] =
    ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
  if n == 0 {
    return "₀".to_string();
  }
  let mut digits = Vec::new();
  while n > 0 {
    digits.push(SUBSCRIPTS[(n % 10) as usize]);
    n /= 10;
  }
  digits.reverse();
  digits.into_iter().collect()
}

// ─── 2D OutputForm rendering ───────────────────────────────────────────

/// A rectangular block of text used for 2D layout.
#[derive(Debug, Clone)]
struct TextBox {
  lines: Vec<String>,
  baseline: usize, // 0-indexed row that is the "main" line
}

impl TextBox {
  /// Create a single-line box.
  fn atom(s: &str) -> Self {
    TextBox {
      lines: vec![s.to_string()],
      baseline: 0,
    }
  }

  /// Width of the widest line.
  fn width(&self) -> usize {
    self.lines.iter().map(|l| l.len()).max().unwrap_or(0)
  }

  /// Height (number of lines).
  fn height(&self) -> usize {
    self.lines.len()
  }

  /// Pad all lines to the same width with trailing spaces.
  fn normalize(&mut self) {
    let w = self.width();
    for line in &mut self.lines {
      if line.len() < w {
        line.push_str(&" ".repeat(w - line.len()));
      }
    }
  }

  /// Horizontal concatenation of multiple boxes, aligned on baselines.
  fn hconcat(parts: &[TextBox]) -> TextBox {
    if parts.is_empty() {
      return TextBox::atom("");
    }
    if parts.len() == 1 {
      return parts[0].clone();
    }

    // Find max baseline and max height-above-baseline
    let max_above: usize = parts.iter().map(|b| b.baseline).max().unwrap_or(0);
    let max_below: usize = parts
      .iter()
      .map(|b| b.height().saturating_sub(b.baseline + 1))
      .max()
      .unwrap_or(0);
    let total_height = max_above + 1 + max_below;

    let mut result_lines: Vec<String> = vec![String::new(); total_height];

    for part in parts {
      let mut p = part.clone();
      p.normalize();
      let w = p.width();
      let offset = max_above - p.baseline; // lines of padding above this part

      for row in 0..total_height {
        if row >= offset && row < offset + p.height() {
          result_lines[row].push_str(&p.lines[row - offset]);
        } else {
          result_lines[row].push_str(&" ".repeat(w));
        }
      }
    }

    let baseline = max_above;
    TextBox {
      lines: result_lines,
      baseline,
    }
  }

  /// Place exponent as superscript to the right and above the base.
  fn superscript(base: &TextBox, exp: &TextBox) -> TextBox {
    let mut base = base.clone();
    let mut exp = exp.clone();
    base.normalize();
    exp.normalize();

    let bw = base.width();
    let ew = exp.width();

    // The exponent sits above the top of the base, shifted right by base width.
    // Total lines = exp.height() + base.height()
    // Baseline = exp.height() + base.baseline

    let total = exp.height() + base.height();
    let mut lines = Vec::with_capacity(total);

    // Exponent lines (shifted right by base width)
    for i in 0..exp.height() {
      lines.push(format!("{}{}", " ".repeat(bw), &exp.lines[i]));
    }
    // Base lines (padded right by exponent width)
    for i in 0..base.height() {
      lines.push(format!("{}{}", &base.lines[i], " ".repeat(ew)));
    }

    TextBox {
      baseline: exp.height() + base.baseline,
      lines,
    }
  }

  /// Render a fraction:  numerator / bar / denominator
  fn fraction(num: &TextBox, denom: &TextBox) -> TextBox {
    let mut num = num.clone();
    let mut denom = denom.clone();
    num.normalize();
    denom.normalize();

    let bar_width = num.width().max(denom.width());

    // Center numerator and denominator
    let num_pad = (bar_width.saturating_sub(num.width())) / 2;
    let denom_pad = (bar_width.saturating_sub(denom.width())) / 2;

    let mut lines = Vec::new();

    // Numerator lines (centered)
    for l in &num.lines {
      lines.push(format!("{}{}", " ".repeat(num_pad), l));
    }
    // Bar
    lines.push("-".repeat(bar_width));
    // Denominator lines (centered)
    for l in &denom.lines {
      lines.push(format!("{}{}", " ".repeat(denom_pad), l));
    }

    // Baseline is the bar line
    let baseline = num.height();
    TextBox { lines, baseline }
  }

  /// Convert to final string (trim trailing whitespace per line).
  fn to_string(&self) -> String {
    self
      .lines
      .iter()
      .map(|l| l.trim_end().to_string())
      .collect::<Vec<_>>()
      .join("\n")
  }
}

/// Convert an expression to a 2D TextBox for OutputForm rendering.
fn expr_to_textbox(expr: &Expr) -> TextBox {
  match expr {
    Expr::Integer(n) => TextBox::atom(&n.to_string()),
    Expr::BigInteger(n) => TextBox::atom(&n.to_string()),
    Expr::Real(f) => TextBox::atom(&format_real(*f)),
    Expr::String(s) => TextBox::atom(s),
    Expr::Identifier(s) | Expr::Constant(s) => TextBox::atom(s),
    Expr::Raw(s) => TextBox::atom(s),

    // Power[base, exp]
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      let base = expr_to_textbox_base(left);
      let exp = expr_to_textbox(right);
      TextBox::superscript(&base, &exp)
    }

    // FunctionCall Power
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      // Check for fraction: Power[denom, -1] inside Times will be handled by Times
      // Here handle standalone Power
      let base = expr_to_textbox_base(&args[0]);
      let exp = expr_to_textbox(&args[1]);
      TextBox::superscript(&base, &exp)
    }

    // Plus[args...]
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2 => {
      let mut parts: Vec<TextBox> = Vec::new();
      parts.push(expr_to_textbox(&args[0]));
      for arg in args.iter().skip(1) {
        // Check for negative terms
        let (sign, term) = extract_sign_for_plus(arg);
        parts.push(TextBox::atom(sign));
        parts.push(expr_to_textbox(&term));
      }
      TextBox::hconcat(&parts)
    }

    // BinaryOp Plus
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let l = expr_to_textbox(left);
      let (sign, term) = extract_sign_for_plus(right);
      let r = expr_to_textbox(&term);
      TextBox::hconcat(&[l, TextBox::atom(sign), r])
    }

    // Times[args...] - handle fractions and products
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      render_times_textbox(args)
    }

    // BinaryOp Times
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => render_times_textbox(&[*left.clone(), *right.clone()]),

    // BinaryOp Divide
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let num = expr_to_textbox(left);
      let denom = expr_to_textbox(right);
      TextBox::fraction(&num, &denom)
    }

    // Rational[num, denom]
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      let num = expr_to_textbox(&args[0]);
      let denom = expr_to_textbox(&args[1]);
      TextBox::fraction(&num, &denom)
    }

    // UnaryOp Minus
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let inner = expr_to_textbox(operand);
      TextBox::hconcat(&[TextBox::atom("-"), inner])
    }

    // List
    Expr::List(items) => {
      let mut parts: Vec<TextBox> = Vec::new();
      parts.push(TextBox::atom("{"));
      for (i, item) in items.iter().enumerate() {
        if i > 0 {
          parts.push(TextBox::atom(", "));
        }
        parts.push(expr_to_textbox(item));
      }
      parts.push(TextBox::atom("}"));
      TextBox::hconcat(&parts)
    }

    // For everything else, fall back to 1D rendering
    _ => TextBox::atom(&expr_to_output(expr)),
  }
}

/// Render a base expression for Power, adding parens if needed for precedence.
fn expr_to_textbox_base(expr: &Expr) -> TextBox {
  let needs_parens = matches!(
    expr,
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      ..
    }
  ) || matches!(expr, Expr::FunctionCall { name, .. } if name == "Plus");

  if needs_parens {
    let inner = expr_to_textbox(expr);
    TextBox::hconcat(&[TextBox::atom("("), inner, TextBox::atom(")")])
  } else {
    expr_to_textbox(expr)
  }
}

/// Extract sign and unsigned term for Plus rendering.
fn extract_sign_for_plus(expr: &Expr) -> (&'static str, Expr) {
  match expr {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => (" - ", *operand.clone()),
    Expr::Integer(n) if *n < 0 => (" - ", Expr::Integer(-n)),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(-1)) => (" - ", *right.clone()),
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(n) if *n < 0) => {
      if let Expr::Integer(n) = left.as_ref() {
        (
          " - ",
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-n)),
            right: right.clone(),
          },
        )
      } else {
        (" + ", expr.clone())
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Times"
        && !args.is_empty()
        && matches!(&args[0], Expr::Integer(n) if *n < 0) =>
    {
      if let Expr::Integer(n) = &args[0] {
        if *n == -1 {
          let new_args = args[1..].to_vec();
          if new_args.len() == 1 {
            (" - ", new_args[0].clone())
          } else {
            (
              " - ",
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: new_args,
              },
            )
          }
        } else {
          let mut new_args = vec![Expr::Integer(-n)];
          new_args.extend_from_slice(&args[1..]);
          (
            " - ",
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: new_args,
            },
          )
        }
      } else {
        (" + ", expr.clone())
      }
    }
    _ => (" + ", expr.clone()),
  }
}

/// Render Times arguments as 2D, handling fractions (Power[x, -1]).
fn render_times_textbox(args: &[Expr]) -> TextBox {
  // Separate numerator and denominator factors
  let mut num_factors: Vec<&Expr> = Vec::new();
  let mut denom_factors: Vec<&Expr> = Vec::new();

  for arg in args {
    match arg {
      Expr::FunctionCall { name, args: pargs }
        if name == "Power"
          && pargs.len() == 2
          && matches!(&pargs[1], Expr::Integer(-1)) =>
      {
        denom_factors.push(&pargs[0]);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        right,
        left,
      } if matches!(right.as_ref(), Expr::Integer(-1)) => {
        denom_factors.push(left);
      }
      _ => {
        num_factors.push(arg);
      }
    }
  }

  if denom_factors.is_empty() {
    // No fractions - just render as product
    let mut parts: Vec<TextBox> = Vec::new();
    for (i, f) in num_factors.iter().enumerate() {
      if i > 0 {
        parts.push(TextBox::atom(" "));
      }
      parts.push(expr_to_textbox(f));
    }
    TextBox::hconcat(&parts)
  } else {
    // Has fractions - render as num/denom
    let num_box = if num_factors.is_empty() {
      TextBox::atom("1")
    } else if num_factors.len() == 1 {
      expr_to_textbox(num_factors[0])
    } else {
      let mut parts: Vec<TextBox> = Vec::new();
      for (i, f) in num_factors.iter().enumerate() {
        if i > 0 {
          parts.push(TextBox::atom(" "));
        }
        parts.push(expr_to_textbox(f));
      }
      TextBox::hconcat(&parts)
    };
    let denom_box = if denom_factors.len() == 1 {
      expr_to_textbox(denom_factors[0])
    } else {
      let mut parts: Vec<TextBox> = Vec::new();
      for (i, f) in denom_factors.iter().enumerate() {
        if i > 0 {
          parts.push(TextBox::atom(" "));
        }
        parts.push(expr_to_textbox(f));
      }
      TextBox::hconcat(&parts)
    };
    TextBox::fraction(&num_box, &denom_box)
  }
}

/// Render an expression in 2D OutputForm.
pub fn expr_to_output_form_2d(expr: &Expr) -> String {
  let tb = expr_to_textbox(expr);
  tb.to_string()
}

/// Top-level output: like expr_to_output but Sequence[a, b, ...] displays as
/// concatenated elements (matching Wolfram REPL behavior where Sequence splices
/// into the output context). Only applies at the outermost level.
pub fn top_level_output(expr: &Expr) -> String {
  match expr {
    Expr::FunctionCall { name, args } if name == "Sequence" => {
      args.iter().map(expr_to_output).collect::<Vec<_>>().join("")
    }
    _ => expr_to_output(expr),
  }
}
