//! Turn a recorded skeleton back into a value of the same shape.
//!
//! The metadata table describes data it does not hold by a skeleton:
//! `Arr[dims, heads]` for numeric arrays, `Rep[n, {variants}]` for long
//! lists, `Assoc[n, keys, values]` for large associations, `Sk[head, …]` for
//! expressions with other heads, and bare `Integer`, `Real`, `String` for
//! atoms. [`instantiate`] builds a value with exactly that structure, drawing
//! numbers from a [`NumSpec`] and words from a [`Lexicon`]. Collections use
//! it for irregular properties their main generator does not cover.

use super::rng::Rng;
use super::tabular::{NumSpec, quantize};
use super::words::Lexicon;
use crate::functions::*;

pub struct Filler<'a> {
  pub rng: Rng,
  pub spec: NumSpec,
  pub lexicon: &'a Lexicon,
}

impl Filler<'_> {
  fn number(&mut self, integer: bool) -> Expr {
    let s = &self.spec;
    let v = (s.mean + s.sd * self.rng.normal()).clamp(s.min, s.max);
    if integer {
      Expr::Integer(v.round() as i128)
    } else {
      quantize(
        v,
        &NumSpec {
          integer: false,
          ..s.clone()
        },
      )
    }
  }

  fn word(&mut self) -> Expr {
    Expr::String(super::words::capitalize(&self.lexicon.word(&mut self.rng)))
  }
}

fn heads_say_integer(heads: &Expr) -> bool {
  match heads {
    Expr::List(h) => {
      !h.is_empty()
        && h
          .iter()
          .all(|x| matches!(x, Expr::Identifier(n) if n == "Integer"))
    }
    _ => false,
  }
}

fn dims_of(e: &Expr) -> Vec<usize> {
  match e {
    Expr::List(d) => d
      .iter()
      .filter_map(|x| match x {
        Expr::Integer(n) => Some(*n as usize),
        _ => None,
      })
      .collect(),
    _ => Vec::new(),
  }
}

fn array(f: &mut Filler, dims: &[usize], integer: bool) -> Expr {
  match dims.split_first() {
    None => f.number(integer),
    Some((&n, rest)) => Expr::List(
      (0..n)
        .map(|_| array(f, rest, integer))
        .collect::<Vec<_>>()
        .into(),
    ),
  }
}

/// A value with the structure the skeleton `sk` describes.
pub fn instantiate(sk: &Expr, f: &mut Filler) -> Expr {
  match sk {
    Expr::Identifier(h) => match h.as_str() {
      "Integer" => f.number(true),
      "Real" | "Rational" => f.number(false),
      "String" => f.word(),
      _ => sk.clone(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|x| instantiate(x, f))
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::Association(pairs) => Expr::Association(
      pairs
        .iter()
        .map(|(k, v)| {
          let key = match k {
            Expr::String(_) => k.clone(),
            other => instantiate(other, f),
          };
          (key, instantiate(v, f))
        })
        .collect(),
    ),
    Expr::Rule {
      pattern,
      replacement,
    } => rule_expr(
      match pattern.as_ref() {
        Expr::Identifier(_) | Expr::String(_) => pattern.as_ref().clone(),
        other => instantiate(other, f),
      },
      instantiate(replacement, f),
    ),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Arr" => {
        let dims = dims_of(args.first().unwrap_or(&Expr::Integer(0)));
        let integer = args.get(1).is_some_and(heads_say_integer);
        array(f, &dims, integer)
      }
      "Rep" => {
        let n = match args.first() {
          Some(Expr::Integer(n)) => *n as usize,
          _ => 0,
        };
        let variants: Vec<Expr> = match args.get(1) {
          Some(Expr::List(v)) => v
            .iter()
            .filter(|x| !matches!(x, Expr::String(s) if s == "..."))
            .cloned()
            .collect(),
          _ => Vec::new(),
        };
        if variants.is_empty() {
          return Expr::List(crate::ExprList::default());
        }
        // The first variant is the common one; others appear now and then.
        Expr::List(
          (0..n)
            .map(|i| {
              let v = if i == 0 || f.rng.chance(0.8) {
                &variants[0]
              } else {
                f.rng.pick(&variants)
              };
              instantiate(v, f)
            })
            .collect::<Vec<_>>()
            .into(),
        )
      }
      "Assoc" => {
        let n = match args.first() {
          Some(Expr::Integer(n)) => *n as usize,
          _ => 0,
        };
        let key_sk = match args.get(1) {
          Some(Expr::List(k)) => {
            k.first().cloned().unwrap_or(id_expr("String"))
          }
          _ => id_expr("String"),
        };
        let val_sk = match args.get(2) {
          Some(Expr::List(v)) => {
            v.first().cloned().unwrap_or(id_expr("Integer"))
          }
          _ => id_expr("Integer"),
        };
        let mut seen = std::collections::HashSet::new();
        let mut pairs = Vec::with_capacity(n);
        let mut guard = 0;
        while pairs.len() < n && guard < n * 20 + 100 {
          guard += 1;
          let k = instantiate(&key_sk, f);
          if seen.insert(crate::syntax::expr_to_string(&k)) {
            pairs.push((k, instantiate(&val_sk, f)));
          }
        }
        Expr::Association(pairs)
      }
      "Sk" => {
        // Sk[head, args…]: head is a symbol (or a string for atomic kinds
        // the caller handles itself).
        let mut it = args.iter();
        match it.next() {
          Some(Expr::Identifier(head)) => {
            call(head, it.map(|x| instantiate(x, f)).collect())
          }
          Some(Expr::String(head)) if head == "Quantity" => {
            call("Quantity", it.map(|x| instantiate(x, f)).collect())
          }
          _ => sk.clone(),
        }
      }
      _ => call(name, args.iter().map(|x| instantiate(x, f)).collect()),
    },
    other => other.clone(),
  }
}
