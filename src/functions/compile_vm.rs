//! A machine-number evaluator for the bodies of `Compile`d functions.
//!
//! `CompiledFunction[specs, body][args…]` otherwise substitutes its
//! arguments into `body` and evaluates it symbolically, which is the
//! semantics but far too slow for the inner loops Demonstrations compile —
//! an escape-time fractal iterates a `FixedPoint` a hundred times for each
//! of a quarter million pixels. Here the body is translated once into a
//! tree of machine operations over integers, reals, complex numbers,
//! booleans and lists of them, with local variables in slots, and run
//! directly. Anything outside that subset (a symbol without a value, a
//! function the translator does not know, an integer overflow, a value of
//! the wrong kind at run time) makes the run give up, and the caller falls
//! back to the symbolic evaluation — the way Wolfram's compiled code falls
//! back to uncompiled evaluation.

use crate::functions::polynomial_ast::complex_numeric::C;
use crate::syntax::{BinaryOperator, ComparisonOp, Expr, UnaryOperator};
use std::collections::HashMap;
use std::rc::Rc;

/// A run-time value.
#[derive(Clone, Debug)]
pub(crate) enum V {
  I(i64),
  R(f64),
  Z(C),
  B(bool),
  L(Rc<Vec<V>>),
  /// A built-in function's name, passed on to an external call
  /// (`Outer[Plus, a, b]`).
  Sym(Rc<str>),
  Null,
}

/// The declared type of a compiled argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Ty {
  Integer,
  Real,
  Complex,
  Boolean,
}

/// Why a run stopped early.
enum Flow {
  Break,
  Continue,
  Return(V),
  /// Outside what the evaluator can do: fall back.
  Fail,
}

type R = Result<V, Flow>;

fn fail<T>() -> Result<T, Flow> {
  Err(Flow::Fail)
}

#[derive(Clone, Copy, Debug)]
enum F1 {
  Sin,
  Cos,
  Tan,
  Cot,
  Sec,
  Csc,
  Exp,
  Log,
  Sqrt,
  Abs,
  ArcTan,
  ArcSin,
  ArcCos,
  Sinh,
  Cosh,
  Tanh,
  Floor,
  Ceiling,
  Round,
  IntegerPart,
  FractionalPart,
  Re,
  Im,
  Conjugate,
  Arg,
  Sign,
  N,
  EvenQ,
  OddQ,
  Boole,
  Length,
  Total,
  Reverse,
  First,
  Last,
  Rest,
  Most,
  Norm,
  Range,
}

#[derive(Clone, Copy, Debug)]
enum F2 {
  Mod,
  Quotient,
  ArcTan2,
  Log2,
  BitAnd,
  BitOr,
  BitXor,
  Append,
  Prepend,
  Join,
  ConstantArray,
  Dot,
  Range2,
}

struct Iter {
  slot: Option<usize>,
  start: N,
  end: N,
  step: Option<N>,
  /// `{i, list}`: iterate over the values of a list.
  over_list: bool,
}

struct Lambda {
  params: Vec<usize>,
  body: Box<N>,
}

enum N {
  Const(V),
  Var(usize),
  List(Vec<N>),
  Plus(Vec<N>),
  Times(Vec<N>),
  Sub(Box<N>, Box<N>),
  Div(Box<N>, Box<N>),
  Pow(Box<N>, Box<N>),
  Neg(Box<N>),
  F1(F1, Box<N>),
  F2(F2, Box<N>, Box<N>),
  MinMax(bool, Vec<N>),
  Cmp(Vec<N>, Vec<ComparisonOp>),
  And(Vec<N>),
  Or(Vec<N>),
  Not(Box<N>),
  If(Box<N>, Box<N>, Option<Box<N>>),
  Which(Vec<(N, N)>),
  /// `Switch[x, v1, r1, …]` over literal values; `None` stands for `_`.
  Switch(Box<N>, Vec<(Option<V>, N)>),
  Seq(Vec<N>),
  Set(usize, Box<N>),
  SetPart(usize, Vec<N>, Box<N>),
  /// `x += d` etc.: the slot, the operation, the operand, whether the old
  /// value is returned (`x++`).
  Update(usize, char, Box<N>, bool),
  While(Box<N>, Option<Box<N>>),
  For(Box<N>, Box<N>, Box<N>, Box<N>),
  Do(Box<N>, Vec<Iter>),
  Table(Box<N>, Vec<Iter>),
  Sum(Box<N>, Vec<Iter>, bool),
  Part(Box<N>, Vec<N>),
  Nest(Rc<Lambda>, Box<N>, Box<N>, bool),
  FixedPoint(Rc<Lambda>, Box<N>, Option<Box<N>>, Option<Rc<Lambda>>, bool),
  NestWhile(Rc<Lambda>, Box<N>, Rc<Lambda>, Option<Box<N>>, bool),
  Map(Rc<Lambda>, Box<N>),
  Apply(Rc<Lambda>, Vec<N>),
  /// `x = Append[x, v]` (`AppendTo`) or `Prepend` (`PrependTo`).
  AppendTo(usize, Box<N>, bool),
  /// A function outside the evaluator, called on the arguments' values
  /// through the symbolic evaluator (Wolfram's `MainEvaluate`).
  Extern(String, Vec<N>),
  Break,
  Continue,
  Return(Option<Box<N>>),
}

/// The translator's view of names: scopes of local slots, innermost last.
struct Comp {
  scopes: Vec<HashMap<String, usize>>,
  /// Pure-function slot frames (`#`, `#2`, …), innermost last.
  slot_frames: Vec<Vec<usize>>,
  nslots: usize,
}

impl Comp {
  fn new_slot(&mut self) -> usize {
    self.nslots += 1;
    self.nslots - 1
  }
  fn lookup(&self, name: &str) -> Option<usize> {
    self.scopes.iter().rev().find_map(|s| s.get(name).copied())
  }
  fn push_scope(&mut self, names: &[String]) -> Vec<usize> {
    let mut scope = HashMap::new();
    let slots: Vec<usize> = names
      .iter()
      .map(|n| {
        let s = self.new_slot();
        scope.insert(n.clone(), s);
        s
      })
      .collect();
    self.scopes.push(scope);
    slots
  }

  fn lambda(&mut self, f: &Expr, arity: usize) -> Option<Rc<Lambda>> {
    match f {
      Expr::Function { body } => {
        let params: Vec<usize> =
          (0..arity.max(1)).map(|_| self.new_slot()).collect();
        self.slot_frames.push(params.clone());
        let body = self.c(body);
        self.slot_frames.pop();
        Some(Rc::new(Lambda {
          params,
          body: Box::new(body?),
        }))
      }
      Expr::NamedFunction { params, body, .. } => {
        let slots = self.push_scope(params);
        let body = self.c(body);
        self.scopes.pop();
        Some(Rc::new(Lambda {
          params: slots,
          body: Box::new(body?),
        }))
      }
      Expr::FunctionCall { name, args } if name == "Function" => {
        match &args[..] {
          [body] => self.lambda(
            &Expr::Function {
              body: Box::new(body.clone()),
            },
            arity,
          ),
          [Expr::Identifier(p), body] => self.lambda(
            &Expr::NamedFunction {
              params: vec![p.clone()],
              body: Box::new(body.clone()),
              bracketed: false,
            },
            arity,
          ),
          [Expr::List(ps), body] => {
            let params = ps
              .iter()
              .map(|p| match p {
                Expr::Identifier(n) => Some(n.clone()),
                _ => None,
              })
              .collect::<Option<Vec<_>>>()?;
            self.lambda(
              &Expr::NamedFunction {
                params,
                body: Box::new(body.clone()),
                bracketed: true,
              },
              arity,
            )
          }
          _ => None,
        }
      }
      _ => None,
    }
  }

  /// An iterator spec `{i, a, b, d}`, `{i, a, b}`, `{i, b}`, `{i, list}`
  /// or `{n}`. The variable is bound in a new scope the caller pops.
  fn iter(&mut self, spec: &Expr) -> Option<Iter> {
    let Expr::List(items) = spec else {
      let end = self.c(spec)?;
      return Some(Iter {
        slot: None,
        start: N::Const(V::I(1)),
        end,
        step: None,
        over_list: false,
      });
    };
    match &items[..] {
      [n] => Some(Iter {
        slot: None,
        start: N::Const(V::I(1)),
        end: self.c(n)?,
        step: None,
        over_list: false,
      }),
      [Expr::Identifier(v), rest @ ..]
        if !rest.is_empty() && rest.len() <= 3 =>
      {
        // The bounds are read outside the iterator's own scope.
        let parts: Vec<N> =
          rest.iter().map(|e| self.c(e)).collect::<Option<_>>()?;
        let slot = self.push_scope(std::slice::from_ref(v))[0];
        let mut parts = parts.into_iter();
        Some(match rest.len() {
          1 => {
            let end = parts.next()?;
            let over_list = matches!(&rest[0], Expr::List(_))
              || !matches!(
                &rest[0],
                Expr::Integer(_) | Expr::Real(_) | Expr::Identifier(_)
              );
            Iter {
              slot: Some(slot),
              start: N::Const(V::I(1)),
              end,
              step: None,
              over_list,
            }
          }
          2 => Iter {
            slot: Some(slot),
            start: parts.next()?,
            end: parts.next()?,
            step: None,
            over_list: false,
          },
          _ => Iter {
            slot: Some(slot),
            start: parts.next()?,
            end: parts.next()?,
            step: parts.next(),
            over_list: false,
          },
        })
      }
      _ => None,
    }
  }

  /// Iterators bound in nested scopes; returns how many scopes to pop.
  fn iters(&mut self, specs: &[Expr]) -> Option<(Vec<Iter>, usize)> {
    let mut out = Vec::new();
    let mut scopes = 0;
    for s in specs {
      let before = self.scopes.len();
      out.push(self.iter(s)?);
      scopes += self.scopes.len() - before;
    }
    Some((out, scopes))
  }

  fn scoped(&mut self, vars: &Expr, body: &Expr) -> Option<N> {
    let Expr::List(items) = vars else { return None };
    let mut names = Vec::new();
    let mut inits = Vec::new();
    for item in items {
      match item {
        Expr::Identifier(n) => names.push((n.clone(), None)),
        Expr::FunctionCall { name, args }
          if name == "Set" && args.len() == 2 =>
        {
          let Expr::Identifier(n) = &args[0] else {
            return None;
          };
          // An initial value is read outside the new scope.
          inits.push(self.c(&args[1])?);
          names.push((n.clone(), Some(inits.len() - 1)));
        }
        _ => return None,
      }
    }
    let plain: Vec<String> = names.iter().map(|(n, _)| n.clone()).collect();
    let slots = self.push_scope(&plain);
    let mut inits: Vec<Option<N>> = inits.into_iter().map(Some).collect();
    let mut seq = Vec::new();
    for ((_, init), slot) in names.iter().zip(&slots) {
      let value = match init {
        Some(k) => inits[*k].take()?,
        None => N::Const(V::Null),
      };
      seq.push(N::Set(*slot, Box::new(value)));
    }
    let body = self.c(body);
    self.scopes.pop();
    seq.push(body?);
    Some(N::Seq(seq))
  }

  fn assign_target<'e>(&self, e: &'e Expr) -> Option<(usize, Vec<&'e Expr>)> {
    match e {
      Expr::Identifier(n) => Some((self.lookup(n)?, Vec::new())),
      Expr::Part { .. } => {
        let (base, idx) = flatten_part(e);
        let Expr::Identifier(n) = base else {
          return None;
        };
        Some((self.lookup(n)?, idx))
      }
      Expr::FunctionCall { name, args }
        if name == "Part" && args.len() >= 2 =>
      {
        let Expr::Identifier(n) = &args[0] else {
          return None;
        };
        Some((self.lookup(n)?, args[1..].iter().collect()))
      }
      _ => None,
    }
  }

  fn update(
    &mut self,
    target: &Expr,
    op: char,
    operand: N,
    post: bool,
  ) -> Option<N> {
    let (slot, idx) = self.assign_target(target)?;
    if !idx.is_empty() {
      // `x[[i]] += d` as `x[[i]] = x[[i]] + d`.
      let idx_nodes: Vec<N> =
        idx.iter().map(|e| self.c(e)).collect::<Option<_>>()?;
      let idx_nodes2: Vec<N> =
        idx.iter().map(|e| self.c(e)).collect::<Option<_>>()?;
      let current = N::Part(Box::new(N::Var(slot)), idx_nodes2);
      let new = arith(op, current, operand);
      return Some(N::SetPart(slot, idx_nodes, Box::new(new)));
    }
    Some(N::Update(slot, op, Box::new(operand), post))
  }

  fn c(&mut self, e: &Expr) -> Option<N> {
    Some(match e {
      Expr::Integer(n) => N::Const(V::I(i64::try_from(*n).ok()?)),
      Expr::Real(r) => N::Const(V::R(*r)),
      Expr::BigFloat(..) => {
        N::Const(V::R(crate::functions::math_ast::try_eval_to_f64(e)?))
      }
      Expr::Identifier(n) => match n.as_str() {
        "True" => N::Const(V::B(true)),
        "False" => N::Const(V::B(false)),
        "Null" => N::Const(V::Null),
        "I" => N::Const(V::Z(C::new(0.0, 1.0))),
        "Pi" => N::Const(V::R(std::f64::consts::PI)),
        "E" => N::Const(V::R(std::f64::consts::E)),
        "Degree" => N::Const(V::R(std::f64::consts::PI / 180.0)),
        "GoldenRatio" => N::Const(V::R(f64::midpoint(1.0, 5.0_f64.sqrt()))),
        "EulerGamma" => N::Const(V::R(std::f64::consts::EULER_GAMMA)),
        "Infinity" => N::Const(V::R(f64::INFINITY)),
        _ => match self.lookup(n) {
          Some(s) => N::Var(s),
          None => {
            if let Some(v) = global_value(n) {
              N::Const(v)
            } else if crate::evaluator::is_builtin_symbol(n) {
              N::Const(V::Sym(Rc::from(n.as_str())))
            } else {
              if trace() {
                eprintln!("compile_vm: free symbol {n}");
              }
              return None;
            }
          }
        },
      },
      Expr::Constant(n) => {
        return self.c(&Expr::Identifier(n.clone()));
      }
      Expr::Slot(k) => {
        let frame = self.slot_frames.last()?;
        N::Var(*frame.get(k.checked_sub(1)?)?)
      }
      Expr::List(items) => {
        N::List(items.iter().map(|i| self.c(i)).collect::<Option<_>>()?)
      }
      Expr::CompoundExpr(items) => {
        N::Seq(items.iter().map(|i| self.c(i)).collect::<Option<_>>()?)
      }
      Expr::UnaryOp { op, operand } => match op {
        UnaryOperator::Minus => N::Neg(Box::new(self.c(operand)?)),
        UnaryOperator::Not => N::Not(Box::new(self.c(operand)?)),
      },
      Expr::BinaryOp { op, left, right } => {
        let (l, r) = (self.c(left)?, self.c(right)?);
        match op {
          BinaryOperator::Plus => N::Plus(vec![l, r]),
          BinaryOperator::Minus => N::Sub(Box::new(l), Box::new(r)),
          BinaryOperator::Times => N::Times(vec![l, r]),
          BinaryOperator::Divide => N::Div(Box::new(l), Box::new(r)),
          BinaryOperator::Power => N::Pow(Box::new(l), Box::new(r)),
          BinaryOperator::And => N::And(vec![l, r]),
          BinaryOperator::Or => N::Or(vec![l, r]),
          _ => return None,
        }
      }
      Expr::Comparison {
        operands,
        operators,
      } => N::Cmp(
        operands.iter().map(|o| self.c(o)).collect::<Option<_>>()?,
        operators.clone(),
      ),
      Expr::Part { .. } => {
        let (base, idx) = flatten_part(e);
        N::Part(
          Box::new(self.c(base)?),
          idx.iter().map(|i| self.c(i)).collect::<Option<_>>()?,
        )
      }
      Expr::CurriedCall { func, args } => {
        let f = self.lambda(func, args.len())?;
        N::Apply(f, args.iter().map(|a| self.c(a)).collect::<Option<_>>()?)
      }
      Expr::FunctionCall { name, args } => {
        let r = self.call(name, args);
        if r.is_none() && trace() {
          eprintln!(
            "compile_vm: cannot compile {name}[…] ({} args)",
            args.len()
          );
        }
        return r;
      }
      _ => {
        if trace() {
          eprintln!(
            "compile_vm: cannot compile {}",
            crate::syntax::expr_to_string(e)
          );
        }
        return None;
      }
    })
  }

  fn call(&mut self, name: &str, args: &[Expr]) -> Option<N> {
    let one = |s: &mut Self, f: F1| -> Option<N> {
      match args {
        [a] => Some(N::F1(f, Box::new(s.c(a)?))),
        _ => None,
      }
    };
    let two = |s: &mut Self, f: F2| -> Option<N> {
      match args {
        [a, b] => Some(N::F2(f, Box::new(s.c(a)?), Box::new(s.c(b)?))),
        _ => None,
      }
    };
    let all = |s: &mut Self| -> Option<Vec<N>> {
      args.iter().map(|a| s.c(a)).collect::<Option<Vec<_>>>()
    };
    Some(match name {
      "Plus" => N::Plus(all(self)?),
      "Times" => N::Times(all(self)?),
      "Subtract" if args.len() == 2 => {
        let v = all(self)?;
        let mut v = v.into_iter();
        N::Sub(Box::new(v.next()?), Box::new(v.next()?))
      }
      "Divide" | "Rational" if args.len() == 2 => {
        let v = all(self)?;
        let mut v = v.into_iter();
        N::Div(Box::new(v.next()?), Box::new(v.next()?))
      }
      "Power" if args.len() == 2 => {
        let v = all(self)?;
        let mut v = v.into_iter();
        N::Pow(Box::new(v.next()?), Box::new(v.next()?))
      }
      "Minus" if args.len() == 1 => N::Neg(Box::new(self.c(&args[0])?)),
      "Complex" if args.len() == 2 => N::Const(V::Z(C::new(
        crate::functions::math_ast::try_eval_to_f64(&args[0])?,
        crate::functions::math_ast::try_eval_to_f64(&args[1])?,
      ))),
      "List" => N::List(all(self)?),
      "Sin" => one(self, F1::Sin)?,
      "Cos" => one(self, F1::Cos)?,
      "Tan" => one(self, F1::Tan)?,
      "Cot" => one(self, F1::Cot)?,
      "Sec" => one(self, F1::Sec)?,
      "Csc" => one(self, F1::Csc)?,
      "Exp" => one(self, F1::Exp)?,
      "Log" if args.len() == 1 => one(self, F1::Log)?,
      "Log" => two(self, F2::Log2)?,
      "Sqrt" => one(self, F1::Sqrt)?,
      "Abs" => one(self, F1::Abs)?,
      "ArcTan" if args.len() == 1 => one(self, F1::ArcTan)?,
      "ArcTan" => two(self, F2::ArcTan2)?,
      "ArcSin" => one(self, F1::ArcSin)?,
      "ArcCos" => one(self, F1::ArcCos)?,
      "Sinh" => one(self, F1::Sinh)?,
      "Cosh" => one(self, F1::Cosh)?,
      "Tanh" => one(self, F1::Tanh)?,
      "Floor" => one(self, F1::Floor)?,
      "Ceiling" => one(self, F1::Ceiling)?,
      "Round" if args.len() == 1 => one(self, F1::Round)?,
      "IntegerPart" => one(self, F1::IntegerPart)?,
      "FractionalPart" => one(self, F1::FractionalPart)?,
      "Re" => one(self, F1::Re)?,
      "Im" => one(self, F1::Im)?,
      "Conjugate" => one(self, F1::Conjugate)?,
      "Arg" => one(self, F1::Arg)?,
      "Sign" => one(self, F1::Sign)?,
      "N" if args.len() == 1 => one(self, F1::N)?,
      "EvenQ" => one(self, F1::EvenQ)?,
      "OddQ" => one(self, F1::OddQ)?,
      "Boole" => one(self, F1::Boole)?,
      // Wolfram's compiler does not compile `NestWhileList`; the call is
      // evaluated outside the compiled code, whose results come back typed
      // real — so its `Length` is `21.`, not `21`.
      "Length" if matches!(args, [Expr::FunctionCall { name, .. }] if name == "NestWhileList") => {
        N::F1(F1::N, Box::new(one(self, F1::Length)?))
      }
      "Length" => one(self, F1::Length)?,
      "Total" if args.len() == 1 => one(self, F1::Total)?,
      "Reverse" if args.len() == 1 => one(self, F1::Reverse)?,
      "First" if args.len() == 1 => one(self, F1::First)?,
      "Last" if args.len() == 1 => one(self, F1::Last)?,
      "Rest" => one(self, F1::Rest)?,
      "Most" => one(self, F1::Most)?,
      "Norm" if args.len() == 1 => one(self, F1::Norm)?,
      "Range" if args.len() == 1 => one(self, F1::Range)?,
      "Range" if args.len() == 2 => two(self, F2::Range2)?,
      "Mod" => two(self, F2::Mod)?,
      "Quotient" => two(self, F2::Quotient)?,
      "BitAnd" => two(self, F2::BitAnd)?,
      "BitOr" => two(self, F2::BitOr)?,
      "BitXor" => two(self, F2::BitXor)?,
      "Append" => two(self, F2::Append)?,
      "Prepend" => two(self, F2::Prepend)?,
      "Join" => two(self, F2::Join)?,
      "ConstantArray" => two(self, F2::ConstantArray)?,
      "Dot" => two(self, F2::Dot)?,
      "Max" => N::MinMax(true, all(self)?),
      "Min" => N::MinMax(false, all(self)?),
      "And" => N::And(all(self)?),
      "Or" => N::Or(all(self)?),
      "Not" if args.len() == 1 => N::Not(Box::new(self.c(&args[0])?)),
      "Equal" | "Unequal" | "Less" | "LessEqual" | "Greater"
      | "GreaterEqual"
        if args.len() >= 2 =>
      {
        let op = match name {
          "Equal" => ComparisonOp::Equal,
          "Unequal" => ComparisonOp::NotEqual,
          "Less" => ComparisonOp::Less,
          "LessEqual" => ComparisonOp::LessEqual,
          "Greater" => ComparisonOp::Greater,
          _ => ComparisonOp::GreaterEqual,
        };
        N::Cmp(all(self)?, vec![op; args.len() - 1])
      }
      "If" if args.len() == 2 || args.len() == 3 => N::If(
        Box::new(self.c(&args[0])?),
        Box::new(self.c(&args[1])?),
        match args.get(2) {
          Some(e) => Some(Box::new(self.c(e)?)),
          None => None,
        },
      ),
      "Which" if args.len().is_multiple_of(2) => {
        let mut cases = Vec::new();
        for pair in args.chunks(2) {
          cases.push((self.c(&pair[0])?, self.c(&pair[1])?));
        }
        N::Which(cases)
      }
      "Switch" if args.len() >= 3 && args.len() % 2 == 1 => {
        let subject = Box::new(self.c(&args[0])?);
        let mut cases = Vec::new();
        for pair in args[1..].chunks(2) {
          let pat = match &pair[0] {
            Expr::Pattern {
              name,
              head: None,
              blank_type: 1,
            } if name.is_empty() => None,
            Expr::FunctionCall { name, args }
              if name == "Blank" && args.is_empty() =>
            {
              None
            }
            literal => Some(from_expr(literal)?),
          };
          cases.push((pat, self.c(&pair[1])?));
        }
        N::Switch(subject, cases)
      }
      "CompoundExpression" => N::Seq(all(self)?),
      "Module" | "Block" | "With" if args.len() == 2 => {
        self.scoped(&args[0], &args[1])?
      }
      "Set" if args.len() == 2 => {
        let value = self.c(&args[1])?;
        let (slot, idx) = self.assign_target(&args[0])?;
        if idx.is_empty() {
          N::Set(slot, Box::new(value))
        } else {
          let idx: Vec<N> =
            idx.iter().map(|e| self.c(e)).collect::<Option<_>>()?;
          N::SetPart(slot, idx, Box::new(value))
        }
      }
      "Increment" | "Decrement" | "PreIncrement" | "PreDecrement"
        if args.len() == 1 =>
      {
        let op = if name.ends_with("Increment") {
          '+'
        } else {
          '-'
        };
        let post = !name.starts_with("Pre");
        self.update(&args[0], op, N::Const(V::I(1)), post)?
      }
      "AddTo" | "SubtractFrom" | "TimesBy" | "DivideBy" if args.len() == 2 => {
        let op = match name {
          "AddTo" => '+',
          "SubtractFrom" => '-',
          "TimesBy" => '*',
          _ => '/',
        };
        let operand = self.c(&args[1])?;
        self.update(&args[0], op, operand, false)?
      }
      "While" if args.len() == 1 || args.len() == 2 => N::While(
        Box::new(self.c(&args[0])?),
        match args.get(1) {
          Some(b) => Some(Box::new(self.c(b)?)),
          None => None,
        },
      ),
      "For" if args.len() == 4 => N::For(
        Box::new(self.c(&args[0])?),
        Box::new(self.c(&args[1])?),
        Box::new(self.c(&args[2])?),
        Box::new(self.c(&args[3])?),
      ),
      "Do" | "Table" | "Sum" | "Product" if args.len() >= 2 => {
        let (iters, scopes) = self.iters(&args[1..])?;
        let body = self.c(&args[0]);
        for _ in 0..scopes {
          self.scopes.pop();
        }
        let body = Box::new(body?);
        match name {
          "Do" => N::Do(body, iters),
          "Table" => N::Table(body, iters),
          "Sum" => N::Sum(body, iters, true),
          _ => N::Sum(body, iters, false),
        }
      }
      "Part" if args.len() >= 2 => N::Part(
        Box::new(self.c(&args[0])?),
        args[1..].iter().map(|a| self.c(a)).collect::<Option<_>>()?,
      ),
      "Nest" | "NestList" if args.len() == 3 => N::Nest(
        self.lambda(&args[0], 1)?,
        Box::new(self.c(&args[1])?),
        Box::new(self.c(&args[2])?),
        name == "NestList",
      ),
      "FixedPoint" | "FixedPointList" if (2..=4).contains(&args.len()) => {
        let f = self.lambda(&args[0], 1)?;
        let x = Box::new(self.c(&args[1])?);
        let mut max = None;
        let mut same = None;
        for extra in &args[2..] {
          match extra {
            Expr::Rule {
              pattern,
              replacement,
            } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "SameTest") =>
            {
              same = Some(self.lambda(replacement, 2)?);
            }
            Expr::Rule { .. } | Expr::RuleDelayed { .. } => return None,
            other => max = Some(Box::new(self.c(other)?)),
          }
        }
        N::FixedPoint(f, x, max, same, name == "FixedPointList")
      }
      "NestWhile" | "NestWhileList" if args.len() == 3 || args.len() == 5 => {
        // Only the one-argument test form (`m` = 1), with an optional
        // iteration cap.
        if args.len() == 5 && !matches!(&args[3], Expr::Integer(1)) {
          return None;
        }
        N::NestWhile(
          self.lambda(&args[0], 1)?,
          Box::new(self.c(&args[1])?),
          self.lambda(&args[2], 1)?,
          match args.get(4) {
            Some(m) => Some(Box::new(self.c(m)?)),
            None => None,
          },
          name == "NestWhileList",
        )
      }
      "Map" if args.len() == 2 => {
        N::Map(self.lambda(&args[0], 1)?, Box::new(self.c(&args[1])?))
      }
      "AppendTo" | "PrependTo" if args.len() == 2 => {
        let Expr::Identifier(n) = &args[0] else {
          return None;
        };
        let slot = self.lookup(n)?;
        N::AppendTo(slot, Box::new(self.c(&args[1])?), name == "AppendTo")
      }
      "Break" if args.is_empty() => N::Break,
      "Continue" if args.is_empty() => N::Continue,
      "Return" if args.len() <= 1 => N::Return(match args.first() {
        Some(a) => Some(Box::new(self.c(a)?)),
        None => None,
      }),
      // Any other function whose arguments are evaluated: call it on their
      // values. One that holds its arguments (a plotting or scoping
      // construct) would see them differently, so it is not.
      _ if extern_callable(name) => N::Extern(name.to_string(), all(self)?),
      _ => return None,
    })
  }
}

fn extern_callable(name: &str) -> bool {
  use crate::evaluator::Attributes as A;
  if name.is_empty() {
    return false;
  }
  let holds = A::HoldAll | A::HoldFirst | A::HoldRest | A::HoldAllComplete;
  let builtin = crate::evaluator::get_builtin_attributes(name);
  !builtin.contains(holds)
    && !crate::func_attrs_contains(name, holds)
    && !matches!(
      name,
      // Output would be repeated should the run fall back and start over.
      "Print"
        | "Echo"
        | "Sow"
        | "Reap"
        | "Message"
        | "PrintTemporary"
        | "Set"
        | "SetDelayed"
        | "Module"
        | "Block"
        | "With"
        | "Function"
        | "Compile"
        | "Pattern"
        | "Rule"
        | "RuleDelayed"
    )
}

/// `WOXI_COMPILE_TRACE=1` reports what kept a body from compiling.
fn trace() -> bool {
  std::env::var_os("WOXI_COMPILE_TRACE").is_some()
}

/// The current value of a global symbol the body reads, if it is a
/// machine number (or a list of them).
fn global_value(name: &str) -> Option<V> {
  let value = crate::ENV.with(|e| match e.borrow().get(name) {
    Some(crate::StoredValue::ExprVal(v)) => Some(v.clone()),
    _ => None,
  })?;
  from_expr(&value)
}

fn flatten_part(e: &Expr) -> (&Expr, Vec<&Expr>) {
  let mut idx = Vec::new();
  let mut cur = e;
  while let Expr::Part { expr, index } = cur {
    idx.push(index.as_ref());
    cur = expr;
  }
  idx.reverse();
  (cur, idx)
}

fn arith(op: char, a: N, b: N) -> N {
  match op {
    '+' => N::Plus(vec![a, b]),
    '-' => N::Sub(Box::new(a), Box::new(b)),
    '*' => N::Times(vec![a, b]),
    _ => N::Div(Box::new(a), Box::new(b)),
  }
}

/// A machine value from an expression, if it is one.
pub(crate) fn from_expr(e: &Expr) -> Option<V> {
  match e {
    Expr::Integer(n) => i64::try_from(*n).ok().map(V::I),
    Expr::Real(r) => Some(V::R(*r)),
    Expr::Identifier(n) if n == "True" => Some(V::B(true)),
    Expr::Identifier(n) if n == "False" => Some(V::B(false)),
    Expr::List(items) => Some(V::L(Rc::new(
      items.iter().map(from_expr).collect::<Option<Vec<_>>>()?,
    ))),
    _ if crate::functions::predicate_ast::is_complex_number(e) => {
      let re = crate::functions::math_ast::try_eval_to_f64(
        &crate::evaluator::evaluate_expr_to_expr(&crate::helpers::call1(
          "Re",
          e.clone(),
        ))
        .ok()?,
      )?;
      let im = crate::functions::math_ast::try_eval_to_f64(
        &crate::evaluator::evaluate_expr_to_expr(&crate::helpers::call1(
          "Im",
          e.clone(),
        ))
        .ok()?,
      )?;
      Some(V::Z(C::new(re, im)))
    }
    _ => None,
  }
}

/// The value bound to an argument of the declared type and rank.
pub(crate) fn bind_value(e: &Expr, ty: Ty, rank: usize) -> Option<V> {
  bind(e, ty, rank)
}

fn bind(e: &Expr, ty: Ty, rank: usize) -> Option<V> {
  if rank > 0 {
    let Expr::List(items) = e else { return None };
    return Some(V::L(Rc::new(
      items
        .iter()
        .map(|i| bind(i, ty, rank - 1))
        .collect::<Option<Vec<_>>>()?,
    )));
  }
  let v = from_expr(e).or_else(|| {
    // An exact number (a rational, `Pi`) enters as a machine number.
    let x = crate::functions::math_ast::try_eval_to_f64(e)?;
    Some(V::R(x))
  })?;
  Some(match (ty, v) {
    (Ty::Integer, V::I(n)) => V::I(n),
    (Ty::Integer, _) => return None,
    (Ty::Real, V::I(n)) => V::R(n as f64),
    (Ty::Real, V::R(x)) => V::R(x),
    (Ty::Real, _) => return None,
    (Ty::Complex, V::I(n)) => V::Z(C::new(n as f64, 0.0)),
    (Ty::Complex, V::R(x)) => V::Z(C::new(x, 0.0)),
    (Ty::Complex, V::Z(z)) => V::Z(z),
    (Ty::Boolean, V::B(b)) => V::B(b),
    _ => return None,
  })
}

/// The expression form of a result.
pub(crate) fn to_expr(v: &V) -> Expr {
  match v {
    V::I(n) => Expr::Integer(*n as i128),
    V::R(x) => Expr::Real(*x),
    V::Z(z) => crate::evaluator::evaluate_function_call_ast(
      "Complex",
      &[Expr::Real(z.re), Expr::Real(z.im)],
    )
    .unwrap_or(Expr::Real(f64::NAN)),
    V::B(b) => Expr::Identifier(if *b { "True" } else { "False" }.to_string()),
    // A tensor holds one kind of number: integers alongside reals are
    // reals, anything alongside a complex is complex.
    V::L(items) => {
      let kind = items.iter().map(leaf_kind).max().unwrap_or(0);
      Expr::List(
        items
          .iter()
          .map(|v| to_expr(&promote(v, kind)))
          .collect::<Vec<_>>()
          .into(),
      )
    }
    V::Sym(n) => Expr::Identifier(n.to_string()),
    V::Null => Expr::Identifier("Null".to_string()),
  }
}

/// 0 integer, 1 real, 2 complex — the widest number kind in a value.
fn leaf_kind(v: &V) -> u8 {
  match v {
    V::R(_) => 1,
    V::Z(_) => 2,
    V::L(items) => items.iter().map(leaf_kind).max().unwrap_or(0),
    _ => 0,
  }
}

fn promote(v: &V, kind: u8) -> V {
  match (v, kind) {
    (V::I(n), 1) => V::R(*n as f64),
    (V::I(n), 2) => V::Z(C::new(*n as f64, 0.0)),
    (V::R(x), 2) => V::Z(C::new(*x, 0.0)),
    (V::L(items), _) => {
      V::L(Rc::new(items.iter().map(|i| promote(i, kind)).collect()))
    }
    _ => v.clone(),
  }
}

/// A compiled body, ready to run on argument values.
pub(crate) struct Program {
  node: N,
  nslots: usize,
  args: Vec<(usize, Ty, usize)>,
}

impl Program {
  pub(crate) fn compile(
    params: &[(String, Ty, usize)],
    body: &Expr,
  ) -> Option<Self> {
    let mut comp = Comp {
      scopes: Vec::new(),
      slot_frames: Vec::new(),
      nslots: 0,
    };
    let names: Vec<String> = params.iter().map(|(n, _, _)| n.clone()).collect();
    let slots = comp.push_scope(&names);
    let node = comp.c(body)?;
    Some(Program {
      node,
      nslots: comp.nslots,
      args: slots
        .into_iter()
        .zip(params)
        .map(|(s, (_, ty, rank))| (s, *ty, *rank))
        .collect(),
    })
  }

  /// The declared (type, rank) of each argument.
  pub(crate) fn arg_types(&self) -> Vec<(Ty, usize)> {
    self.args.iter().map(|(_, t, r)| (*t, *r)).collect()
  }

  /// Run on bound argument values.
  pub(crate) fn call(&self, values: Vec<V>) -> Option<V> {
    if values.len() != self.args.len() {
      return None;
    }
    let mut env = vec![V::Null; self.nslots];
    for ((slot, _, _), v) in self.args.iter().zip(values) {
      env[*slot] = v;
    }
    let mut m = Machine { env, steps: 0 };
    match m.eval(&self.node) {
      Ok(v) | Err(Flow::Return(v)) => Some(v),
      Err(_) => None,
    }
  }

  /// Run on machine reals (each argument scalar and real-typed, or given as
  /// a value), giving a real result.
  pub(crate) fn call_real(&self, values: Vec<V>) -> f64 {
    let bound: Option<Vec<V>> = values
      .into_iter()
      .zip(&self.args)
      .map(|(v, (_, ty, _))| match (v, ty) {
        (V::R(x), Ty::Integer) if x.fract() == 0.0 => Some(V::I(x as i64)),
        (V::R(x), Ty::Complex) => Some(V::Z(C::new(x, 0.0))),
        (v, _) => Some(v),
      })
      .collect();
    match bound.and_then(|b| self.call(b)) {
      Some(V::R(x)) => x,
      Some(V::I(n)) => n as f64,
      _ => f64::NAN,
    }
  }
}

/// Run `body` with the arguments bound per `params` (name, type, rank);
/// `None` when the body or a value at run time is outside the evaluator.
pub(crate) fn run(
  params: &[(String, Ty, usize)],
  body: &Expr,
  args: &[Expr],
) -> Option<Expr> {
  if params.len() != args.len() {
    return None;
  }
  let program = Program::compile(params, body)?;
  let values: Vec<V> = args
    .iter()
    .zip(params)
    .map(|(a, (_, ty, rank))| bind(a, *ty, *rank))
    .collect::<Option<_>>()?;
  program.call(values).map(|v| to_expr(&v))
}

struct Machine {
  env: Vec<V>,
  steps: u64,
}

fn num(v: &V) -> Option<C> {
  match v {
    V::I(n) => Some(C::new(*n as f64, 0.0)),
    V::R(x) => Some(C::new(*x, 0.0)),
    V::Z(z) => Some(*z),
    _ => None,
  }
}

fn real(v: &V) -> Option<f64> {
  match v {
    V::I(n) => Some(*n as f64),
    V::R(x) => Some(*x),
    _ => None,
  }
}

fn int(v: &V) -> Option<i64> {
  match v {
    V::I(n) => Some(*n),
    V::R(x) if x.fract() == 0.0 && x.abs() < 9e15 => Some(*x as i64),
    _ => None,
  }
}

/// The value of a machine complex number, narrowed to a real when its
/// imaginary part is exactly zero only where Wolfram's compiled arithmetic
/// keeps it complex — i.e. never: a complex stays complex.
fn zval(z: C) -> V {
  V::Z(z)
}

/// Elementwise application over lists, broadcasting scalars.
fn listable2(a: &V, b: &V, f: &dyn Fn(&V, &V) -> R) -> R {
  match (a, b) {
    (V::L(x), V::L(y)) => {
      if x.len() != y.len() {
        return fail();
      }
      Ok(V::L(Rc::new(
        x.iter()
          .zip(y.iter())
          .map(|(p, q)| listable2(p, q, f))
          .collect::<Result<_, _>>()?,
      )))
    }
    (V::L(x), s) => Ok(V::L(Rc::new(
      x.iter()
        .map(|p| listable2(p, s, f))
        .collect::<Result<_, _>>()?,
    ))),
    (s, V::L(y)) => Ok(V::L(Rc::new(
      y.iter()
        .map(|q| listable2(s, q, f))
        .collect::<Result<_, _>>()?,
    ))),
    _ => f(a, b),
  }
}

fn listable1(a: &V, f: &dyn Fn(&V) -> R) -> R {
  match a {
    V::L(x) => Ok(V::L(Rc::new(
      x.iter()
        .map(|p| listable1(p, f))
        .collect::<Result<_, _>>()?,
    ))),
    _ => f(a),
  }
}

fn add(a: &V, b: &V) -> R {
  listable2(a, b, &|a, b| match (a, b) {
    (V::I(x), V::I(y)) => x.checked_add(*y).map(V::I).ok_or(Flow::Fail),
    (V::Z(_), _) | (_, V::Z(_)) => {
      let (x, y) = (num(a).ok_or(Flow::Fail)?, num(b).ok_or(Flow::Fail)?);
      Ok(zval(C::new(x.re + y.re, x.im + y.im)))
    }
    _ => Ok(V::R(
      real(a).ok_or(Flow::Fail)? + real(b).ok_or(Flow::Fail)?,
    )),
  })
}

fn mul(a: &V, b: &V) -> R {
  listable2(a, b, &|a, b| match (a, b) {
    (V::I(x), V::I(y)) => x.checked_mul(*y).map(V::I).ok_or(Flow::Fail),
    (V::Z(_), _) | (_, V::Z(_)) => {
      let (x, y) = (num(a).ok_or(Flow::Fail)?, num(b).ok_or(Flow::Fail)?);
      Ok(zval(C::new(
        x.re * y.re - x.im * y.im,
        x.re * y.im + x.im * y.re,
      )))
    }
    _ => Ok(V::R(
      real(a).ok_or(Flow::Fail)? * real(b).ok_or(Flow::Fail)?,
    )),
  })
}

fn neg(a: &V) -> R {
  listable1(a, &|a| match a {
    V::I(x) => x.checked_neg().map(V::I).ok_or(Flow::Fail),
    V::R(x) => Ok(V::R(-x)),
    V::Z(z) => Ok(zval(C::new(-z.re, -z.im))),
    _ => fail(),
  })
}

fn div(a: &V, b: &V) -> R {
  listable2(a, b, &|a, b| match (a, b) {
    (V::Z(_), _) | (_, V::Z(_)) => {
      let (x, y) = (num(a).ok_or(Flow::Fail)?, num(b).ok_or(Flow::Fail)?);
      Ok(zval(x.div(y)))
    }
    _ => {
      let d = real(b).ok_or(Flow::Fail)?;
      if d == 0.0 {
        return fail();
      }
      Ok(V::R(real(a).ok_or(Flow::Fail)? / d))
    }
  })
}

fn pow(a: &V, b: &V) -> R {
  listable2(a, b, &|a, b| match (a, b) {
    (V::I(x), V::I(y)) if *y >= 0 => {
      let e = u32::try_from(*y).map_err(|_| Flow::Fail)?;
      x.checked_pow(e).map(V::I).ok_or(Flow::Fail)
    }
    (V::Z(_), _) | (_, V::Z(_)) => {
      let (x, y) = (num(a).ok_or(Flow::Fail)?, num(b).ok_or(Flow::Fail)?);
      Ok(zval(x.pow(y)))
    }
    _ => {
      let (x, y) = (real(a).ok_or(Flow::Fail)?, real(b).ok_or(Flow::Fail)?);
      let v = if y.fract() == 0.0 && y.abs() < 1e9 {
        x.powi(y as i32)
      } else {
        x.powf(y)
      };
      if v.is_nan() {
        return fail();
      }
      Ok(V::R(v))
    }
  })
}

fn cmp_vals(a: &V, b: &V, op: ComparisonOp) -> Result<bool, Flow> {
  use ComparisonOp as O;
  if let (V::Z(_), _) | (_, V::Z(_)) = (a, b) {
    let (x, y) = (num(a).ok_or(Flow::Fail)?, num(b).ok_or(Flow::Fail)?);
    return match op {
      O::Equal => Ok(x.re == y.re && x.im == y.im),
      O::NotEqual => Ok(!(x.re == y.re && x.im == y.im)),
      _ => fail(),
    };
  }
  if let (V::B(x), V::B(y)) = (a, b) {
    return match op {
      O::Equal | O::SameQ => Ok(x == y),
      O::NotEqual | O::UnsameQ => Ok(x != y),
      _ => fail(),
    };
  }
  let (x, y) = (real(a).ok_or(Flow::Fail)?, real(b).ok_or(Flow::Fail)?);
  Ok(match op {
    O::Equal | O::SameQ => x == y,
    O::NotEqual | O::UnsameQ => x != y,
    O::Less => x < y,
    O::LessEqual => x <= y,
    O::Greater => x > y,
    O::GreaterEqual => x >= y,
  })
}

fn truth(v: &V) -> Result<bool, Flow> {
  match v {
    V::B(b) => Ok(*b),
    _ => fail(),
  }
}

fn f1(f: F1, a: &V) -> R {
  use F1::{
    Abs, ArcCos, ArcSin, ArcTan, Arg, Boole, Ceiling, Conjugate, Cos, Cosh,
    Cot, Csc, EvenQ, Exp, First, Floor, FractionalPart, Im, IntegerPart, Last,
    Length, Log, Most, N, Norm, OddQ, Range, Re, Rest, Reverse, Round, Sec,
    Sign, Sin, Sinh, Sqrt, Tan, Tanh, Total,
  };
  let realf = |g: fn(f64) -> f64, cg: fn(C) -> C| {
    move |a: &V| -> R {
      if let V::Z(z) = a {
        Ok(zval(cg(*z)))
      } else {
        let v = g(real(a).ok_or(Flow::Fail)?);
        if v.is_nan() { fail() } else { Ok(V::R(v)) }
      }
    }
  };
  match f {
    Length => match a {
      V::L(x) => Ok(V::I(x.len() as i64)),
      _ => Ok(V::I(0)),
    },
    Total => match a {
      V::L(x) => x.iter().try_fold(V::I(0), |acc, v| add(&acc, v)),
      _ => fail(),
    },
    Reverse => match a {
      V::L(x) => Ok(V::L(Rc::new(x.iter().rev().cloned().collect()))),
      _ => fail(),
    },
    First | Last => match a {
      V::L(x) if !x.is_empty() => Ok(if matches!(f, First) {
        x[0].clone()
      } else {
        x[x.len() - 1].clone()
      }),
      _ => fail(),
    },
    Rest | Most => match a {
      V::L(x) if !x.is_empty() => Ok(V::L(Rc::new(if matches!(f, Rest) {
        x[1..].to_vec()
      } else {
        x[..x.len() - 1].to_vec()
      }))),
      _ => fail(),
    },
    Norm => match a {
      V::L(x) => {
        let mut s = 0.0;
        for v in x.iter() {
          let z = num(v).ok_or(Flow::Fail)?;
          s += z.re * z.re + z.im * z.im;
        }
        Ok(V::R(s.sqrt()))
      }
      _ => Ok(V::R(
        num(a)
          .ok_or(Flow::Fail)?
          .re
          .hypot(num(a).ok_or(Flow::Fail)?.im),
      )),
    },
    Range => {
      let n = int(a).ok_or(Flow::Fail)?;
      if !(0..=10_000_000).contains(&n) {
        return fail();
      }
      Ok(V::L(Rc::new((1..=n).map(V::I).collect())))
    }
    _ => listable1(a, &|a| match f {
      Sin => realf(f64::sin, super::polynomial_ast::complex_numeric::C::sin)(a),
      Cos => realf(f64::cos, super::polynomial_ast::complex_numeric::C::cos)(a),
      Tan => realf(f64::tan, |z| z.sin().div(z.cos()))(a),
      Cot => realf(|x| 1.0 / x.tan(), |z| z.cos().div(z.sin()))(a),
      Sec => realf(|x| 1.0 / x.cos(), |z| C::new(1.0, 0.0).div(z.cos()))(a),
      Csc => realf(|x| 1.0 / x.sin(), |z| C::new(1.0, 0.0).div(z.sin()))(a),
      Exp => realf(f64::exp, super::polynomial_ast::complex_numeric::C::exp)(a),
      Sinh => {
        realf(f64::sinh, super::polynomial_ast::complex_numeric::C::sinh)(a)
      }
      Cosh => {
        realf(f64::cosh, super::polynomial_ast::complex_numeric::C::cosh)(a)
      }
      Tanh => realf(f64::tanh, |z| z.sinh().div(z.cosh()))(a),
      Log => {
        if let V::Z(z) = a {
          Ok(zval(z.ln()))
        } else {
          let x = real(a).ok_or(Flow::Fail)?;
          if x < 0.0 {
            Ok(zval(C::new(x, 0.0).ln()))
          } else {
            Ok(V::R(x.ln()))
          }
        }
      }
      Sqrt => {
        if let V::Z(z) = a {
          Ok(zval(z.sqrt()))
        } else {
          let x = real(a).ok_or(Flow::Fail)?;
          if x < 0.0 {
            Ok(zval(C::new(x, 0.0).sqrt()))
          } else {
            Ok(V::R(x.sqrt()))
          }
        }
      }
      ArcTan => realf(f64::atan, |_| C::new(f64::NAN, f64::NAN))(a),
      ArcSin => realf(f64::asin, |_| C::new(f64::NAN, f64::NAN))(a),
      ArcCos => realf(f64::acos, |_| C::new(f64::NAN, f64::NAN))(a),
      Abs => match a {
        V::I(n) => n.checked_abs().map(V::I).ok_or(Flow::Fail),
        V::R(x) => Ok(V::R(x.abs())),
        V::Z(z) => Ok(V::R(z.re.hypot(z.im))),
        _ => fail(),
      },
      Re => match a {
        V::Z(z) => Ok(V::R(z.re)),
        V::I(_) | V::R(_) => Ok(a.clone()),
        _ => fail(),
      },
      Im => match a {
        V::Z(z) => Ok(V::R(z.im)),
        V::I(_) => Ok(V::I(0)),
        V::R(_) => Ok(V::R(0.0)),
        _ => fail(),
      },
      Conjugate => match a {
        V::Z(z) => Ok(zval(C::new(z.re, -z.im))),
        V::I(_) | V::R(_) => Ok(a.clone()),
        _ => fail(),
      },
      Arg => {
        if let V::Z(z) = a {
          Ok(V::R(z.im.atan2(z.re)))
        } else {
          let x = real(a).ok_or(Flow::Fail)?;
          Ok(if x < 0.0 {
            V::R(std::f64::consts::PI)
          } else {
            V::I(0)
          })
        }
      }
      Sign => match a {
        V::I(n) => Ok(V::I(n.signum())),
        V::R(x) => Ok(V::I(if *x > 0.0 {
          1
        } else if *x < 0.0 {
          -1
        } else {
          0
        })),
        V::Z(z) => {
          let m = z.re.hypot(z.im);
          if m == 0.0 {
            Ok(V::I(0))
          } else {
            Ok(zval(C::new(z.re / m, z.im / m)))
          }
        }
        _ => fail(),
      },
      Floor | Ceiling | Round | IntegerPart => match a {
        V::I(n) => Ok(V::I(*n)),
        V::R(x) => {
          let r = match f {
            Floor => x.floor(),
            Ceiling => x.ceil(),
            Round => x.round_ties_even(),
            _ => x.trunc(),
          };
          if r.abs() < 9e15 {
            Ok(V::I(r as i64))
          } else {
            fail()
          }
        }
        _ => fail(),
      },
      FractionalPart => match a {
        V::I(_) => Ok(V::I(0)),
        V::R(x) => Ok(V::R(x.fract())),
        _ => fail(),
      },
      N => match a {
        V::I(n) => Ok(V::R(*n as f64)),
        other => Ok(other.clone()),
      },
      EvenQ | OddQ => match a {
        V::I(n) => Ok(V::B((n % 2 == 0) == matches!(f, EvenQ))),
        _ => Ok(V::B(false)),
      },
      Boole => match a {
        V::B(b) => Ok(V::I(i64::from(*b))),
        _ => fail(),
      },
      _ => fail(),
    }),
  }
}

fn f2(f: F2, a: &V, b: &V) -> R {
  use F2::{
    Append, ArcTan2, BitAnd, BitOr, BitXor, ConstantArray, Dot, Join, Log2,
    Mod, Prepend, Quotient, Range2,
  };
  match f {
    Append | Prepend => match a {
      V::L(x) => {
        let mut v = x.as_ref().clone();
        if matches!(f, Append) {
          v.push(b.clone());
        } else {
          v.insert(0, b.clone());
        }
        Ok(V::L(Rc::new(v)))
      }
      _ => fail(),
    },
    Join => match (a, b) {
      (V::L(x), V::L(y)) => {
        let mut v = x.as_ref().clone();
        v.extend(y.iter().cloned());
        Ok(V::L(Rc::new(v)))
      }
      _ => fail(),
    },
    ConstantArray => {
      let dims: Vec<i64> = match b {
        V::L(d) => {
          d.iter().map(int).collect::<Option<_>>().ok_or(Flow::Fail)?
        }
        other => vec![int(other).ok_or(Flow::Fail)?],
      };
      fn build(v: &V, dims: &[i64]) -> R {
        match dims.split_first() {
          None => Ok(v.clone()),
          Some((&n, rest)) => {
            if !(0..=10_000_000).contains(&n) {
              return fail();
            }
            let inner = build(v, rest)?;
            Ok(V::L(Rc::new(vec![inner; n as usize])))
          }
        }
      }
      build(a, &dims)
    }
    Range2 => {
      let (lo, hi) = (int(a).ok_or(Flow::Fail)?, int(b).ok_or(Flow::Fail)?);
      if hi - lo > 10_000_000 {
        return fail();
      }
      Ok(V::L(Rc::new((lo..=hi).map(V::I).collect())))
    }
    Dot => match (a, b) {
      (V::L(x), V::L(y)) if x.len() == y.len() => {
        let matrix =
          |l: &Rc<Vec<V>>| l.first().is_some_and(|r| matches!(r, V::L(_)));
        match (matrix(x), matrix(y)) {
          (false, false) => {
            let mut acc = V::I(0);
            for (p, q) in x.iter().zip(y.iter()) {
              acc = add(&acc, &mul(p, q)?)?;
            }
            Ok(acc)
          }
          (true, false) => Ok(V::L(Rc::new(
            x.iter()
              .map(|row| f2(Dot, row, b))
              .collect::<Result<_, _>>()?,
          ))),
          _ => fail(),
        }
      }
      _ => fail(),
    },
    _ => listable2(a, b, &|a, b| match f {
      Mod | Quotient => {
        if let (V::I(x), V::I(y)) = (a, b) {
          if *y == 0 {
            return fail();
          }
          // Wolfram's Mod takes the sign of the divisor and Quotient
          // rounds down: x == y Quotient[x, y] + Mod[x, y].
          let r = x.rem_euclid(*y);
          let adjust = *y < 0 && r != 0;
          Ok(V::I(if matches!(f, Mod) {
            if adjust { r + y } else { r }
          } else {
            x.div_euclid(*y) - i64::from(adjust)
          }))
        } else {
          let (x, y) = (real(a).ok_or(Flow::Fail)?, real(b).ok_or(Flow::Fail)?);
          if y == 0.0 {
            return fail();
          }
          let q = (x / y).floor();
          if matches!(f, Mod) {
            Ok(V::R(x - y * q))
          } else if q.abs() < 9e15 {
            Ok(V::I(q as i64))
          } else {
            fail()
          }
        }
      }
      ArcTan2 => {
        let (x, y) = (real(a).ok_or(Flow::Fail)?, real(b).ok_or(Flow::Fail)?);
        Ok(V::R(y.atan2(x)))
      }
      Log2 => {
        let (base, x) =
          (real(a).ok_or(Flow::Fail)?, real(b).ok_or(Flow::Fail)?);
        let v = x.ln() / base.ln();
        if v.is_nan() { fail() } else { Ok(V::R(v)) }
      }
      BitAnd | BitOr | BitXor => {
        let (x, y) = (int(a).ok_or(Flow::Fail)?, int(b).ok_or(Flow::Fail)?);
        Ok(V::I(match f {
          BitAnd => x & y,
          BitOr => x | y,
          _ => x ^ y,
        }))
      }
      _ => fail(),
    }),
  }
}

/// Wolfram's `Part` index into a list of `len`: 1-based, negative from the
/// end.
fn index(i: &V, len: usize) -> Result<usize, Flow> {
  let k = int(i).ok_or(Flow::Fail)?;
  let len = len as i64;
  let pos = if k > 0 { k - 1 } else { len + k };
  if k == 0 || pos < 0 || pos >= len {
    return fail();
  }
  Ok(pos as usize)
}

fn part(v: &V, idx: &[V]) -> R {
  let Some((first, rest)) = idx.split_first() else {
    return Ok(v.clone());
  };
  let V::L(items) = v else { return fail() };
  match first {
    V::L(sel) => Ok(V::L(Rc::new(
      sel
        .iter()
        .map(|i| part(&items[index(i, items.len())?], rest))
        .collect::<Result<_, _>>()?,
    ))),
    _ => part(&items[index(first, items.len())?], rest),
  }
}

fn set_part(v: &mut V, idx: &[V], value: V) -> Result<(), Flow> {
  let Some((first, rest)) = idx.split_first() else {
    *v = value;
    return Ok(());
  };
  let V::L(items) = v else { return fail() };
  let items = Rc::make_mut(items);
  let k = index(first, items.len())?;
  set_part(&mut items[k], rest, value)
}

impl Machine {
  fn tick(&mut self) -> Result<(), Flow> {
    self.steps += 1;
    // Give up on a run that does not finish in reasonable time; the
    // symbolic evaluation then reports what Wolfram would.
    if self.steps > 2_000_000_000 {
      return fail();
    }
    Ok(())
  }

  fn apply(&mut self, f: &Lambda, args: &[V]) -> R {
    for (slot, a) in f.params.iter().zip(args) {
      self.env[*slot] = a.clone();
    }
    match self.eval(&f.body) {
      Err(Flow::Return(v)) => Ok(v),
      other => other,
    }
  }

  /// The values an iterator takes.
  fn iter_values(&mut self, it: &Iter) -> Result<Vec<V>, Flow> {
    let end = self.eval(&it.end)?;
    if it.over_list {
      return match end {
        V::L(items) => Ok(items.as_ref().clone()),
        _ => fail(),
      };
    }
    let start = self.eval(&it.start)?;
    let step = match &it.step {
      Some(s) => self.eval(s)?,
      None => V::I(1),
    };
    if let (V::I(a), V::I(b), V::I(d)) = (&start, &end, &step) {
      if *d == 0 {
        return fail();
      }
      let count = (b - a).div_euclid(*d) + 1;
      if count <= 0 {
        return Ok(Vec::new());
      }
      if count > 50_000_000 {
        return fail();
      }
      Ok((0..count).map(|k| V::I(a + k * d)).collect())
    } else {
      let (a, b, d) = (
        real(&start).ok_or(Flow::Fail)?,
        real(&end).ok_or(Flow::Fail)?,
        real(&step).ok_or(Flow::Fail)?,
      );
      if d == 0.0 || !(a.is_finite() && b.is_finite()) {
        return fail();
      }
      let n = ((b - a) / d + 1e-10).floor() as i64 + 1;
      if n <= 0 {
        return Ok(Vec::new());
      }
      if n > 50_000_000 {
        return fail();
      }
      // An integer start keeps integer values when the step is one.
      let int_start = matches!(start, V::I(_)) && matches!(step, V::I(_));
      Ok(
        (0..n)
          .map(|k| {
            if int_start {
              V::I(int(&start).unwrap() + k * int(&step).unwrap())
            } else {
              V::R(a + k as f64 * d)
            }
          })
          .collect(),
      )
    }
  }

  /// Runs `body` for every combination of the iterators, calling `each`
  /// with the body's value.
  fn nested(
    &mut self,
    iters: &[Iter],
    body: &N,
    each: &mut dyn FnMut(&mut Self, V) -> Result<(), Flow>,
  ) -> Result<(), Flow> {
    let Some((it, rest)) = iters.split_first() else {
      let v = self.eval(body)?;
      return each(self, v);
    };
    for v in self.iter_values(it)? {
      if let Some(s) = it.slot {
        self.env[s] = v;
      }
      self.nested(rest, body, each)?;
    }
    Ok(())
  }

  fn table(&mut self, iters: &[Iter], body: &N) -> R {
    let Some((it, rest)) = iters.split_first() else {
      return self.eval(body);
    };
    let mut out = Vec::new();
    for v in self.iter_values(it)? {
      if let Some(s) = it.slot {
        self.env[s] = v;
      }
      out.push(self.table(rest, body)?);
    }
    Ok(V::L(Rc::new(out)))
  }

  fn eval(&mut self, n: &N) -> R {
    match n {
      N::Const(v) => Ok(v.clone()),
      N::Var(s) => match &self.env[*s] {
        // An unassigned local is a symbol: outside machine arithmetic.
        V::Null => fail(),
        v => Ok(v.clone()),
      },
      N::List(items) => Ok(V::L(Rc::new(
        items
          .iter()
          .map(|i| self.eval(i))
          .collect::<Result<_, _>>()?,
      ))),
      N::Plus(xs) => {
        let mut acc = self.eval(&xs[0])?;
        for x in &xs[1..] {
          let v = self.eval(x)?;
          acc = add(&acc, &v)?;
        }
        Ok(acc)
      }
      N::Times(xs) => {
        let mut acc = self.eval(&xs[0])?;
        for x in &xs[1..] {
          let v = self.eval(x)?;
          acc = mul(&acc, &v)?;
        }
        Ok(acc)
      }
      N::Sub(a, b) => {
        let (x, y) = (self.eval(a)?, self.eval(b)?);
        add(&x, &neg(&y)?)
      }
      N::Div(a, b) => {
        let (x, y) = (self.eval(a)?, self.eval(b)?);
        div(&x, &y)
      }
      N::Pow(a, b) => {
        let (x, y) = (self.eval(a)?, self.eval(b)?);
        pow(&x, &y)
      }
      N::Neg(a) => {
        let x = self.eval(a)?;
        neg(&x)
      }
      N::F1(f, a) => {
        let x = self.eval(a)?;
        f1(*f, &x)
      }
      N::F2(f, a, b) => {
        let (x, y) = (self.eval(a)?, self.eval(b)?);
        f2(*f, &x, &y)
      }
      N::MinMax(max, xs) => {
        let mut best: Option<V> = None;
        let push = |v: V, best: &mut Option<V>| -> Result<(), Flow> {
          let replace = match best {
            None => true,
            Some(b) => {
              let (x, y) =
                (real(&v).ok_or(Flow::Fail)?, real(b).ok_or(Flow::Fail)?);
              if *max { x > y } else { x < y }
            }
          };
          if replace {
            *best = Some(v);
          }
          Ok(())
        };
        for x in xs {
          match self.eval(x)? {
            V::L(items) => {
              for v in items.iter() {
                push(v.clone(), &mut best)?;
              }
            }
            v => push(v, &mut best)?,
          }
        }
        best.ok_or(Flow::Fail)
      }
      N::Cmp(xs, ops) => {
        let vals: Vec<V> =
          xs.iter().map(|x| self.eval(x)).collect::<Result<_, _>>()?;
        for (k, op) in ops.iter().enumerate() {
          if !cmp_vals(&vals[k], &vals[k + 1], *op)? {
            return Ok(V::B(false));
          }
        }
        Ok(V::B(true))
      }
      N::And(xs) => {
        for x in xs {
          let v = self.eval(x)?;
          if !truth(&v)? {
            return Ok(V::B(false));
          }
        }
        Ok(V::B(true))
      }
      N::Or(xs) => {
        for x in xs {
          let v = self.eval(x)?;
          if truth(&v)? {
            return Ok(V::B(true));
          }
        }
        Ok(V::B(false))
      }
      N::Not(a) => {
        let v = self.eval(a)?;
        Ok(V::B(!truth(&v)?))
      }
      N::If(c, a, b) => {
        let v = self.eval(c)?;
        if truth(&v)? {
          self.eval(a)
        } else {
          match b {
            Some(b) => self.eval(b),
            None => Ok(V::Null),
          }
        }
      }
      N::Which(cases) => {
        for (c, v) in cases {
          let t = self.eval(c)?;
          if truth(&t)? {
            return self.eval(v);
          }
        }
        Ok(V::Null)
      }
      N::Switch(subject, cases) => {
        let v = self.eval(subject)?;
        for (pat, r) in cases {
          let hit = match pat {
            None => true,
            Some(p) => match (p, &v) {
              (V::I(a), V::I(b)) => a == b,
              (V::R(a), V::R(b)) => a == b,
              (V::B(a), V::B(b)) => a == b,
              _ => false,
            },
          };
          if hit {
            return self.eval(r);
          }
        }
        Ok(V::Null)
      }
      N::Seq(items) => {
        let mut last = V::Null;
        for i in items {
          last = self.eval(i)?;
        }
        Ok(last)
      }
      N::Set(s, v) => {
        let value = self.eval(v)?;
        self.env[*s] = value.clone();
        Ok(value)
      }
      N::SetPart(s, idx, v) => {
        let value = self.eval(v)?;
        let idx: Vec<V> =
          idx.iter().map(|i| self.eval(i)).collect::<Result<_, _>>()?;
        let mut target = std::mem::replace(&mut self.env[*s], V::Null);
        let r = set_part(&mut target, &idx, value.clone());
        self.env[*s] = target;
        r?;
        Ok(value)
      }
      N::Update(s, op, operand, post) => {
        let old = match &self.env[*s] {
          V::Null => return fail(),
          v => v.clone(),
        };
        let d = self.eval(operand)?;
        let new = match op {
          '+' => add(&old, &d)?,
          '-' => add(&old, &neg(&d)?)?,
          '*' => mul(&old, &d)?,
          _ => div(&old, &d)?,
        };
        self.env[*s] = new.clone();
        Ok(if *post { old } else { new })
      }
      N::While(c, body) => {
        loop {
          self.tick()?;
          let t = self.eval(c)?;
          if !truth(&t)? {
            break;
          }
          if let Some(b) = body {
            match self.eval(b) {
              Ok(_) | Err(Flow::Continue) => {}
              Err(Flow::Break) => break,
              Err(e) => return Err(e),
            }
          }
        }
        Ok(V::Null)
      }
      N::For(init, cond, incr, body) => {
        self.eval(init)?;
        loop {
          self.tick()?;
          let t = self.eval(cond)?;
          if !truth(&t)? {
            break;
          }
          match self.eval(body) {
            Ok(_) | Err(Flow::Continue) => {}
            Err(Flow::Break) => break,
            Err(e) => return Err(e),
          }
          self.eval(incr)?;
        }
        Ok(V::Null)
      }
      N::Do(body, iters) => {
        let r = self.nested(iters, body, &mut |m, _| m.tick());
        match r {
          Ok(()) | Err(Flow::Break) => Ok(V::Null),
          Err(e) => Err(e),
        }
      }
      N::Table(body, iters) => self.table(iters, body),
      N::Sum(body, iters, sum) => {
        let mut acc = V::I(i64::from(!*sum));
        self.nested(iters, body, &mut |_, v| {
          acc = if *sum { add(&acc, &v)? } else { mul(&acc, &v)? };
          Ok(())
        })?;
        Ok(acc)
      }
      N::Part(base, idx) => {
        let b = self.eval(base)?;
        let idx: Vec<V> =
          idx.iter().map(|i| self.eval(i)).collect::<Result<_, _>>()?;
        part(&b, &idx)
      }
      N::Nest(f, x, count, list) => {
        let mut v = self.eval(x)?;
        let n = int(&self.eval(count)?).ok_or(Flow::Fail)?;
        if n < 0 {
          return fail();
        }
        let mut out = vec![v.clone()];
        for _ in 0..n {
          self.tick()?;
          v = self.apply(f, &[v])?;
          if *list {
            out.push(v.clone());
          }
        }
        Ok(if *list { V::L(Rc::new(out)) } else { v })
      }
      N::FixedPoint(f, x, max, same, list) => {
        let mut v = self.eval(x)?;
        let n = match max {
          Some(m) => match self.eval(m)? {
            V::R(x) if x.is_infinite() => i64::MAX,
            other => int(&other).ok_or(Flow::Fail)?,
          },
          None => 65536,
        };
        let mut out = vec![v.clone()];
        let mut k = 0;
        while k < n {
          self.tick()?;
          let next = self.apply(f, &[v.clone()])?;
          if *list {
            out.push(next.clone());
          }
          let done = match same {
            Some(t) => truth(&self.apply(t, &[v.clone(), next.clone()])?)?,
            None => same_value(&v, &next),
          };
          v = next;
          k += 1;
          if done {
            break;
          }
        }
        Ok(if *list { V::L(Rc::new(out)) } else { v })
      }
      N::NestWhile(f, x, test, max, list) => {
        let mut v = self.eval(x)?;
        let n = match max {
          Some(m) => match self.eval(m)? {
            V::R(x) if x.is_infinite() => i64::MAX,
            other => int(&other).ok_or(Flow::Fail)?,
          },
          None => i64::MAX,
        };
        let mut out = vec![v.clone()];
        let mut k = 0;
        while k < n && truth(&self.apply(test, &[v.clone()])?)? {
          self.tick()?;
          v = self.apply(f, &[v])?;
          if *list {
            out.push(v.clone());
          }
          k += 1;
        }
        Ok(if *list { V::L(Rc::new(out)) } else { v })
      }
      N::Map(f, l) => match self.eval(l)? {
        V::L(items) => {
          let mut out = Vec::with_capacity(items.len());
          for v in items.iter() {
            out.push(self.apply(f, std::slice::from_ref(v))?);
          }
          Ok(V::L(Rc::new(out)))
        }
        _ => fail(),
      },
      N::Apply(f, args) => {
        let vals: Vec<V> = args
          .iter()
          .map(|a| self.eval(a))
          .collect::<Result<_, _>>()?;
        self.apply(f, &vals)
      }
      N::AppendTo(s, v, append) => {
        let value = self.eval(v)?;
        let V::L(items) = &mut self.env[*s] else {
          return fail();
        };
        let items = Rc::make_mut(items);
        if *append {
          items.push(value);
        } else {
          items.insert(0, value);
        }
        Ok(V::L(Rc::new(items.clone())))
      }
      N::Extern(name, args) => {
        let vals: Vec<Expr> = args
          .iter()
          .map(|a| self.eval(a).map(|v| to_expr(&v)))
          .collect::<Result<_, _>>()?;
        let r = crate::evaluator::evaluate_function_call_ast(name, &vals)
          .map_err(|_| Flow::Fail)?;
        from_expr(&r).ok_or(Flow::Fail)
      }
      N::Break => Err(Flow::Break),
      N::Continue => Err(Flow::Continue),
      N::Return(v) => {
        let value = match v {
          Some(v) => self.eval(v)?,
          None => V::Null,
        };
        Err(Flow::Return(value))
      }
    }
  }
}

/// `FixedPoint`'s default test: the values are equal (machine numbers to
/// within a few ulps, as `SameQ` compares them).
fn same_value(a: &V, b: &V) -> bool {
  match (a, b) {
    (V::I(x), V::I(y)) => x == y,
    (V::L(x), V::L(y)) => {
      x.len() == y.len()
        && x.iter().zip(y.iter()).all(|(p, q)| same_value(p, q))
    }
    _ => match (num(a), num(b)) {
      (Some(x), Some(y)) => {
        let close = |p: f64, q: f64| {
          p == q || (p - q).abs() <= 4.0 * f64::EPSILON * p.abs().max(q.abs())
        };
        close(x.re, y.re) && close(x.im, y.im)
      }
      _ => false,
    },
  }
}
