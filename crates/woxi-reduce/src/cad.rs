//! Cylindrical decomposition of quantifier-free dense linear formulas.
//!
//! The variables are ordered `v1, …, vn`. Every cell bounds `vk` by affine
//! terms in `v1, …, v(k-1)` only, so the cells describe the solution set the
//! way a cylindrical algebraic decomposition does: the first variable ranges
//! over constant bounds, and each later one over bounds that depend only on
//! the variables before it.
//!
//! Each disjunct of the formula's DNF is decomposed by eliminating the last
//! variable first. When several lower (or upper) bounds compete, the cell is
//! split by which of them is the binding one, and the comparison that makes it
//! binding is projected onto the earlier variables together with the
//! condition that the interval is nonempty. This is Fourier-Motzkin
//! elimination with explicit case splits, so the result is exact.

use std::cmp::Ordering;

use num_bigint::Sign;

use super::affine::{AffineTerm, Variable};
use super::exact::Rational;
use super::formula::{Atom, Formula, Relation};

/// One side of an interval: the bounding term and whether it is excluded.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Endpoint {
  pub term: AffineTerm,
  pub strict: bool,
}

/// The constraint a cell puts on one variable, in terms of earlier ones.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Bound {
  /// The variable is unconstrained.
  Free,
  /// The variable equals the term.
  Equal(AffineTerm),
  /// The variable lies strictly or weakly between the two endpoints; a
  /// missing endpoint is unbounded. At least one endpoint is present.
  Interval {
    lower: Option<Endpoint>,
    upper: Option<Endpoint>,
  },
}

/// A cell holds one bound per variable, in variable order.
pub type Cell = Vec<Bound>;

/// The largest number of DNF disjuncts the decomposition expands.
const MAX_DISJUNCTS: usize = 1 << 14;
/// The largest number of recursive projection steps for one formula.
const MAX_STEPS: usize = 1 << 20;

type Conjunction = Vec<(Relation, AffineTerm)>;

/// Decomposes `formula` cylindrically in `variables` order. Returns `None`
/// when the formula has quantifiers, divisibility atoms, or free variables
/// outside `variables`, or when the expansion exceeds the size limits.
pub fn cylindrical_cells(
  formula: &Formula,
  variables: &[Variable],
) -> Option<Vec<Cell>> {
  if formula.contains_quantifier() || formula.contains_divisibility() {
    return None;
  }
  if !formula
    .free_variables()
    .iter()
    .all(|variable| variables.contains(variable))
  {
    return None;
  }
  let disjuncts = dnf(formula)?;
  let mut steps = 0;
  let mut cells = Vec::new();
  for conjunction in disjuncts {
    for cell in decompose(conjunction, variables, variables.len(), &mut steps)?
    {
      let cell = substitute_equalities(cell, variables);
      if !cells.contains(&cell) {
        cells.push(cell);
      }
    }
  }
  merge_cells(&mut cells);
  Some(cells)
}

fn sign(value: &Rational) -> Sign {
  value.numerator.sign()
}

fn constant_holds(relation: Relation, value: &Rational) -> bool {
  let sign = sign(value);
  match relation {
    Relation::Equal => sign == Sign::NoSign,
    Relation::NotEqual => sign != Sign::NoSign,
    Relation::Less => sign == Sign::Minus,
    Relation::LessEqual => sign != Sign::Plus,
    Relation::Greater => sign == Sign::Plus,
    Relation::GreaterEqual => sign != Sign::Minus,
  }
}

/// Expands an NNF formula into a list of conjunctions of `term rel 0`
/// atoms. A `!=` atom splits into `<` and `>`.
fn dnf(formula: &Formula) -> Option<Vec<Conjunction>> {
  match formula {
    Formula::True => Some(vec![Vec::new()]),
    Formula::False => Some(Vec::new()),
    Formula::Atom(Atom::Relation(Relation::NotEqual, term)) => Some(vec![
      vec![(Relation::Less, term.clone())],
      vec![(Relation::Greater, term.clone())],
    ]),
    Formula::Atom(Atom::Relation(relation, term)) => {
      if term.is_constant() {
        return Some(if constant_holds(*relation, &term.constant) {
          vec![Vec::new()]
        } else {
          Vec::new()
        });
      }
      Some(vec![vec![(*relation, term.clone())]])
    }
    Formula::Atom(Atom::Divides { .. }) | Formula::Quantified(..) => None,
    Formula::Not(inner) => dnf(&inner.as_ref().clone().into_nnf()),
    Formula::Or(children) => {
      let mut result = Vec::new();
      for child in children {
        result.extend(dnf(child)?);
        if result.len() > MAX_DISJUNCTS {
          return None;
        }
      }
      Some(result)
    }
    Formula::And(children) => {
      let mut result: Vec<Conjunction> = vec![Vec::new()];
      for child in children {
        let parts = dnf(child)?;
        let mut next = Vec::new();
        for left in &result {
          for right in &parts {
            let mut conjunction = left.clone();
            for atom in right {
              if !conjunction.contains(atom) {
                conjunction.push(atom.clone());
              }
            }
            next.push(conjunction);
            if next.len() > MAX_DISJUNCTS {
              return None;
            }
          }
        }
        result = next;
        if result.is_empty() {
          break;
        }
      }
      Some(result)
    }
  }
}

/// Decomposes one conjunction in the first `count` variables. Every atom
/// mentions only those variables.
fn decompose(
  atoms: Conjunction,
  variables: &[Variable],
  count: usize,
  steps: &mut usize,
) -> Option<Vec<Cell>> {
  *steps += 1;
  if *steps > MAX_STEPS {
    return None;
  }
  let mut pending = Vec::new();
  for (relation, term) in atoms {
    if term.is_constant() {
      if !constant_holds(relation, &term.constant) {
        return Some(Vec::new());
      }
    } else if !pending.contains(&(relation, term.clone())) {
      pending.push((relation, term));
    }
  }
  if count == 0 {
    return Some(if pending.is_empty() {
      vec![Vec::new()]
    } else {
      Vec::new()
    });
  }
  let variable = &variables[count - 1];

  // Split off the atoms on the last variable, each solved as
  // `variable relation bound`.
  let mut rest = Vec::new();
  let mut solved: Vec<(Relation, AffineTerm, AffineTerm)> = Vec::new();
  for (relation, term) in pending {
    let coefficient = term.coefficient(variable);
    if coefficient.is_zero() {
      rest.push((relation, term));
      continue;
    }
    let mut remainder = term.clone();
    remainder.coefficients.remove(variable);
    // c·x + r rel 0  ⇔  x rel' −r/c, flipping the relation when c < 0.
    let bound = remainder
      .scaled(&Rational::one().checked_divide(&coefficient)?.negated());
    let relation = if sign(&coefficient) == Sign::Minus {
      flipped(relation)
    } else {
      relation
    };
    solved.push((relation, bound, term));
  }

  // An equation fixes the variable; every other atom on it becomes a
  // condition on the earlier variables.
  if let Some(position) = solved
    .iter()
    .position(|(relation, ..)| *relation == Relation::Equal)
  {
    let value = solved[position].1.clone();
    for (index, (_, _, term)) in solved.iter().enumerate() {
      if index != position {
        let relation = solved[index].0;
        let original = original_relation(relation, term, variable);
        rest.push((original, term.substitute(variable, &value)));
      }
    }
    return Some(
      decompose(rest, variables, count - 1, steps)?
        .into_iter()
        .map(|mut cell| {
          cell.push(Bound::Equal(value.clone()));
          cell
        })
        .collect(),
    );
  }

  let mut lowers: Vec<Endpoint> = Vec::new();
  let mut uppers: Vec<Endpoint> = Vec::new();
  for (relation, bound, _) in solved {
    let (list, strict) = match relation {
      Relation::Greater => (&mut lowers, true),
      Relation::GreaterEqual => (&mut lowers, false),
      Relation::Less => (&mut uppers, true),
      Relation::LessEqual => (&mut uppers, false),
      Relation::Equal | Relation::NotEqual => {
        unreachable!("equations are handled above and != is split in DNF")
      }
    };
    add_endpoint(list, bound, strict);
  }
  if lowers.is_empty() && uppers.is_empty() {
    return Some(
      decompose(rest, variables, count - 1, steps)?
        .into_iter()
        .map(|mut cell| {
          cell.push(Bound::Free);
          cell
        })
        .collect(),
    );
  }
  // A strict endpoint wins a tie against a weak one, so it comes first.
  lowers.sort_by_key(|endpoint| !endpoint.strict);
  uppers.sort_by_key(|endpoint| !endpoint.strict);

  let lower_choices: Vec<Option<usize>> = if lowers.is_empty() {
    vec![None]
  } else {
    (0..lowers.len()).map(Some).collect()
  };
  let upper_choices: Vec<Option<usize>> = if uppers.is_empty() {
    vec![None]
  } else {
    (0..uppers.len()).map(Some).collect()
  };
  let mut cells = Vec::new();
  for lower in &lower_choices {
    for upper in &upper_choices {
      let mut conditions = rest.clone();
      // The chosen lower bound is the largest: it beats the earlier ones
      // strictly and the later ones weakly, so ties have one winner.
      if let Some(chosen) = lower {
        for (index, other) in lowers.iter().enumerate() {
          if index != *chosen {
            let difference = lowers[*chosen].term.subtract(&other.term);
            let relation = if index < *chosen {
              Relation::Greater
            } else {
              Relation::GreaterEqual
            };
            conditions.push((relation, difference));
          }
        }
      }
      if let Some(chosen) = upper {
        for (index, other) in uppers.iter().enumerate() {
          if index != *chosen {
            let difference = uppers[*chosen].term.subtract(&other.term);
            let relation = if index < *chosen {
              Relation::Less
            } else {
              Relation::LessEqual
            };
            conditions.push((relation, difference));
          }
        }
      }
      let lower_endpoint = lower.map(|index| lowers[index].clone());
      let upper_endpoint = upper.map(|index| uppers[index].clone());
      if let (Some(low), Some(high)) = (&lower_endpoint, &upper_endpoint) {
        let gap = high.term.subtract(&low.term);
        if !low.strict && !high.strict {
          // A closed interval degenerates to a point when its ends meet.
          let mut point = conditions.clone();
          point.push((Relation::Equal, gap.clone()));
          for mut cell in decompose(point, variables, count - 1, steps)? {
            cell.push(Bound::Equal(low.term.clone()));
            cells.push(cell);
          }
        }
        conditions.push((Relation::Greater, gap));
      }
      for mut cell in decompose(conditions, variables, count - 1, steps)? {
        cell.push(Bound::Interval {
          lower: lower_endpoint.clone(),
          upper: upper_endpoint.clone(),
        });
        cells.push(cell);
      }
    }
  }
  Some(cells)
}

/// Writes each bound in terms of the free earlier variables only: a
/// variable a cell fixes by an equation is replaced by its value in the
/// bounds after it.
fn substitute_equalities(mut cell: Cell, variables: &[Variable]) -> Cell {
  let mut known: Vec<(Variable, AffineTerm)> = Vec::new();
  let apply = |term: &AffineTerm, known: &[(Variable, AffineTerm)]| {
    known
      .iter()
      .rev()
      .fold(term.clone(), |term, (variable, value)| {
        term.substitute(variable, value)
      })
  };
  for (index, bound) in cell.iter_mut().enumerate() {
    match bound {
      Bound::Free => {}
      Bound::Equal(term) => {
        *term = apply(term, &known);
        known.push((variables[index].clone(), term.clone()));
      }
      Bound::Interval { lower, upper } => {
        for endpoint in [lower, upper].into_iter().flatten() {
          endpoint.term = apply(&endpoint.term, &known);
        }
      }
    }
  }
  cell
}

/// Records an endpoint, keeping the stricter of two equal terms.
fn add_endpoint(list: &mut Vec<Endpoint>, term: AffineTerm, strict: bool) {
  if let Some(existing) = list.iter_mut().find(|endpoint| endpoint.term == term)
  {
    existing.strict |= strict;
  } else {
    list.push(Endpoint { term, strict });
  }
}

fn flipped(relation: Relation) -> Relation {
  match relation {
    Relation::Less => Relation::Greater,
    Relation::LessEqual => Relation::GreaterEqual,
    Relation::Greater => Relation::Less,
    Relation::GreaterEqual => Relation::LessEqual,
    other => other,
  }
}

/// The relation of the original `term rel 0` atom, recovered from the
/// relation solved for `variable` (which was flipped for a negative
/// coefficient).
fn original_relation(
  solved: Relation,
  term: &AffineTerm,
  variable: &Variable,
) -> Relation {
  if sign(&term.coefficient(variable)) == Sign::Minus {
    flipped(solved)
  } else {
    solved
  }
}

/// Joins cells that agree everywhere except on one variable whose pieces
/// meet: an open interval and the point or interval that closes it.
fn merge_cells(cells: &mut Vec<Cell>) {
  loop {
    let mut merged = None;
    'search: for first in 0..cells.len() {
      for second in 0..cells.len() {
        if first == second {
          continue;
        }
        if let Some(cell) = merge_pair(&cells[first], &cells[second]) {
          merged = Some((first, second, cell));
          break 'search;
        }
      }
    }
    let Some((first, second, cell)) = merged else {
      return;
    };
    let (low, high) = if first < second {
      (first, second)
    } else {
      (second, first)
    };
    cells.remove(high);
    cells[low] = cell;
    // A merge can make the cell equal to another one.
    let mut index = 0;
    while index < cells.len() {
      if index != low && cells[index] == cells[low] {
        cells.remove(index);
      } else {
        index += 1;
      }
    }
  }
}

/// Merges `left` (the lower piece) with `right` when they differ in exactly
/// one bound and those bounds join into one.
fn merge_pair(left: &Cell, right: &Cell) -> Option<Cell> {
  if left.len() != right.len() {
    return None;
  }
  let mut differing =
    (0..left.len()).filter(|&index| left[index] != right[index]);
  let position = differing.next()?;
  if differing.next().is_some() {
    return None;
  }
  let joined = join_bounds(&left[position], &right[position])?;
  let mut cell = left.clone();
  cell[position] = joined;
  Some(cell)
}

fn interval_parts(
  bound: &Bound,
) -> Option<(Option<Endpoint>, Option<Endpoint>)> {
  match bound {
    Bound::Free => Some((None, None)),
    Bound::Equal(term) => Some((
      Some(Endpoint {
        term: term.clone(),
        strict: false,
      }),
      Some(Endpoint {
        term: term.clone(),
        strict: false,
      }),
    )),
    Bound::Interval { lower, upper } => Some((lower.clone(), upper.clone())),
  }
}

/// Joins `left` and `right` when `left` ends exactly where `right` begins,
/// with the shared end point covered once.
fn join_bounds(left: &Bound, right: &Bound) -> Option<Bound> {
  let (left_lower, left_upper) = interval_parts(left)?;
  let (right_lower, right_upper) = interval_parts(right)?;
  let left_end = left_upper?;
  let right_start = right_lower?;
  if left_end.term != right_start.term || left_end.strict == right_start.strict
  {
    return None;
  }
  Some(match (left_lower, right_upper) {
    (None, None) => Bound::Free,
    (lower, upper) => Bound::Interval { lower, upper },
  })
}

/// Orders bounds of the same variable along the real line when all their
/// endpoints are constants; otherwise leaves them in place.
pub fn bound_cmp(left: &Bound, right: &Bound) -> Ordering {
  fn start(bound: &Bound) -> Option<(Option<Rational>, bool)> {
    match bound {
      Bound::Free => Some((None, false)),
      Bound::Equal(term) => term
        .is_constant()
        .then(|| (Some(term.constant.clone()), false)),
      Bound::Interval { lower: None, .. } => Some((None, false)),
      Bound::Interval {
        lower: Some(endpoint),
        ..
      } => endpoint
        .term
        .is_constant()
        .then(|| (Some(endpoint.term.constant.clone()), endpoint.strict)),
    }
  }
  match (start(left), start(right)) {
    (Some((left_value, left_strict)), Some((right_value, right_strict))) => {
      match (left_value, right_value) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(a), Some(b)) => {
          a.numeric_cmp(&b).then(left_strict.cmp(&right_strict))
        }
      }
    }
    _ => Ordering::Equal,
  }
}

#[cfg(test)]
mod tests {
  use num_bigint::BigInt;

  use super::*;

  fn integer(value: i64) -> Rational {
    Rational::integer(BigInt::from(value))
  }

  fn var(name: &str) -> AffineTerm {
    AffineTerm::variable(Variable::free(name))
  }

  fn atom(relation: Relation, term: AffineTerm) -> Formula {
    Formula::Atom(Atom::Relation(relation, term))
  }

  /// `y > x && y > -x` in `{x, y}` splits `x` where the binding lower
  /// bound changes, and every cell bounds `y` by `x` alone.
  #[test]
  fn competing_lower_bounds_split_the_first_variable() {
    let x = Variable::free("x");
    let y = Variable::free("y");
    let formula = Formula::And(vec![
      atom(Relation::Greater, var("y").subtract(&var("x"))),
      atom(Relation::Greater, var("y").add(&var("x"))),
    ]);
    let cells = cylindrical_cells(&formula, &[x.clone(), y]).unwrap();
    assert_eq!(cells.len(), 2);
    for cell in &cells {
      let Bound::Interval {
        lower: Some(lower),
        upper: None,
      } = &cell[1]
      else {
        panic!("unexpected bound {:?}", cell[1]);
      };
      assert!(lower.strict);
      assert_eq!(
        lower.term.variables().into_iter().collect::<Vec<_>>(),
        std::slice::from_ref(&x)
      );
    }
  }

  /// `x != 1` has the two open half-lines as its cells, and `x > 0 ||
  /// x <= 0` merges into the whole line.
  #[test]
  fn unequal_splits_and_adjacent_pieces_merge() {
    let x = Variable::free("x");
    let shifted = var("x").subtract(&AffineTerm::constant(integer(1)));
    let cells = cylindrical_cells(
      &atom(Relation::NotEqual, shifted),
      std::slice::from_ref(&x),
    )
    .unwrap();
    assert_eq!(cells.len(), 2);
    let whole = Formula::Or(vec![
      atom(Relation::Greater, var("x")),
      atom(Relation::LessEqual, var("x")),
    ]);
    assert_eq!(
      cylindrical_cells(&whole, &[x]).unwrap(),
      vec![vec![Bound::Free]]
    );
  }

  /// A variable outside the order is rejected rather than decomposed.
  #[test]
  fn foreign_variables_are_rejected() {
    let formula = atom(Relation::Greater, var("x").subtract(&var("a")));
    assert!(cylindrical_cells(&formula, &[Variable::free("x")]).is_none());
  }
}
