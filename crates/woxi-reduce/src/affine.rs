//! Canonical sparse affine terms for linear quantifier elimination.

use std::collections::{BTreeMap, BTreeSet};

use super::exact::Rational;

/// Binder-safe identity for a mathematical variable.
///
/// Free variables have no binder id. Every bound occurrence receives the id
/// of its lexical binder, so shadowing cannot capture a free parameter.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Variable {
  pub name: String,
  pub binder: Option<u32>,
}

impl Variable {
  pub fn free(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      binder: None,
    }
  }

  pub fn bound(_source_name: impl Into<String>, binder: u32) -> Self {
    Self {
      name: format!("$bound{binder}"),
      binder: Some(binder),
    }
  }
}

/// `constant + sum(coefficients[variable] * variable)`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AffineTerm {
  pub constant: Rational,
  pub coefficients: BTreeMap<Variable, Rational>,
}

impl AffineTerm {
  pub fn zero() -> Self {
    Self {
      constant: Rational::zero(),
      coefficients: BTreeMap::new(),
    }
  }

  pub fn constant(value: Rational) -> Self {
    Self {
      constant: value,
      coefficients: BTreeMap::new(),
    }
  }

  pub fn variable(variable: Variable) -> Self {
    Self {
      constant: Rational::zero(),
      coefficients: BTreeMap::from([(variable, Rational::one())]),
    }
  }

  pub fn is_constant(&self) -> bool {
    self.coefficients.is_empty()
  }

  pub fn variables(&self) -> BTreeSet<Variable> {
    self.coefficients.keys().cloned().collect()
  }

  pub fn coefficient(&self, variable: &Variable) -> Rational {
    self
      .coefficients
      .get(variable)
      .cloned()
      .unwrap_or_else(Rational::zero)
  }

  pub fn add(&self, other: &Self) -> Self {
    let mut result = self.clone();
    result.constant = result.constant.add(&other.constant);
    for (variable, coefficient) in &other.coefficients {
      result.add_coefficient(variable.clone(), coefficient.clone());
    }
    result
  }

  pub fn subtract(&self, other: &Self) -> Self {
    self.add(&other.scaled(&Rational::integer((-1).into())))
  }

  pub fn scaled(&self, factor: &Rational) -> Self {
    if factor.is_zero() {
      return Self::zero();
    }
    Self {
      constant: self.constant.multiply(factor),
      coefficients: self
        .coefficients
        .iter()
        .map(|(variable, coefficient)| {
          (variable.clone(), coefficient.multiply(factor))
        })
        .collect(),
    }
  }

  /// Multiplies two terms exactly when at least one is constant.
  pub fn checked_multiply(&self, other: &Self) -> Option<Self> {
    if self.is_constant() {
      Some(other.scaled(&self.constant))
    } else if other.is_constant() {
      Some(self.scaled(&other.constant))
    } else {
      None
    }
  }

  pub fn checked_divide(&self, divisor: &Self) -> Option<Self> {
    if !divisor.is_constant() {
      return None;
    }
    Some(self.scaled(&Rational::one().checked_divide(&divisor.constant)?))
  }

  /// Replaces `variable` by another affine term.
  pub fn substitute(&self, variable: &Variable, replacement: &Self) -> Self {
    let Some(coefficient) = self.coefficients.get(variable) else {
      return self.clone();
    };
    let mut result = self.clone();
    result.coefficients.remove(variable);
    result.add(&replacement.scaled(coefficient))
  }

  fn add_coefficient(&mut self, variable: Variable, coefficient: Rational) {
    let sum = self
      .coefficients
      .get(&variable)
      .map_or_else(|| coefficient.clone(), |old| old.add(&coefficient));
    if sum.is_zero() {
      self.coefficients.remove(&variable);
    } else {
      self.coefficients.insert(variable, sum);
    }
  }
}

#[cfg(test)]
mod tests {
  use num_bigint::BigInt;

  use super::*;

  fn integer(value: i64) -> Rational {
    Rational::integer(BigInt::from(value))
  }

  #[test]
  fn like_terms_combine_and_zero_coefficients_disappear() {
    let x = Variable::free("x");
    let term = AffineTerm::variable(x.clone())
      .scaled(&integer(7))
      .add(&AffineTerm::variable(x).scaled(&integer(-7)))
      .add(&AffineTerm::constant(integer(3)));
    assert!(term.is_constant());
    assert_eq!(term.constant, integer(3));
  }

  #[test]
  fn nonlinear_product_and_variable_divisor_are_rejected() {
    let x = AffineTerm::variable(Variable::free("x"));
    let y = AffineTerm::variable(Variable::free("y"));
    assert!(x.checked_multiply(&y).is_none());
    assert!(x.checked_divide(&y).is_none());
    assert_eq!(
      x.checked_multiply(&AffineTerm::constant(integer(2))),
      Some(x.scaled(&integer(2)))
    );
  }

  #[test]
  fn substitution_is_exact_and_removes_the_variable() {
    let x = Variable::free("x");
    let y = Variable::free("y");
    let source = AffineTerm::variable(x.clone())
      .scaled(&integer(2))
      .add(&AffineTerm::variable(y.clone()))
      .add(&AffineTerm::constant(integer(1)));
    let replacement = AffineTerm::variable(y.clone())
      .scaled(&integer(3))
      .add(&AffineTerm::constant(integer(-2)));
    let result = source.substitute(&x, &replacement);
    assert_eq!(result.coefficient(&x), Rational::zero());
    assert_eq!(result.coefficient(&y), integer(7));
    assert_eq!(result.constant, integer(-3));
  }

  #[test]
  fn binder_identity_prevents_capture() {
    let free = Variable::free("x");
    let outer = Variable::bound("x", 0);
    let inner = Variable::bound("x", 1);
    let term = AffineTerm::variable(free)
      .add(&AffineTerm::variable(outer))
      .add(&AffineTerm::variable(inner));
    assert_eq!(term.variables().len(), 3);
  }
}
