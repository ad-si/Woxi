//! Deterministic conversion from the linear IR back to Woxi expressions.

use std::collections::BTreeSet;

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use woxi_reduce::{
  AffineTerm, Atom, Formula, Quantifier, Rational, Relation, Variable,
  crt_pair, euclidean_mod, lcm, solve_linear_congruence,
};

use crate::helpers::call;
use crate::syntax::{BinaryOperator, ComparisonOp, Expr, UnaryOperator};

use super::{bigint_expr, rational_expr};

pub(super) fn formula_expr(formula: &Formula) -> Expr {
  formula_expr_for_targets(formula, &[])
}

pub(super) fn formula_expr_for_targets(
  formula: &Formula,
  targets: &[Variable],
) -> Expr {
  match formula {
    Formula::True => Expr::Identifier("True".to_string()),
    Formula::False => Expr::Identifier("False".to_string()),
    Formula::Atom(atom) => atom_expr(atom, targets),
    Formula::And(children) => interval_expr(children, targets)
      .unwrap_or_else(|| fold_binary(children, BinaryOperator::And, targets)),
    Formula::Or(children) => fold_binary(children, BinaryOperator::Or, targets),
    Formula::Not(inner) => Expr::UnaryOp {
      op: UnaryOperator::Not,
      operand: Box::new(formula_expr_for_targets(inner, targets)),
    },
    Formula::Quantified(quantifier, variables, body) => call(
      match quantifier {
        Quantifier::Exists => "Exists",
        Quantifier::ForAll => "ForAll",
      },
      vec![
        variables_expr(variables),
        formula_expr_for_targets(body, targets),
      ],
    ),
  }
}

/// Emits a finite unary integer result by deriving its interval and residue
/// class exactly. Iteration here materializes the required finite output; it
/// is not used to decide satisfiability or to search for a witness.
pub(super) fn finite_integer_target_expr(
  formula: &Formula,
  target: &Variable,
) -> Option<Expr> {
  let branches = integer_dnf(formula)?;
  let mut values = BTreeSet::new();
  for branch in branches {
    values.extend(finite_integer_branch_values(&branch, target)?);
    if values.len() > 1_000_000 {
      return None;
    }
  }
  Some(integer_values_expr(values, target))
}

/// `x == 1 || x == 3 || …` — how a finite integer solution set is reported.
fn integer_values_expr(
  values: impl IntoIterator<Item = BigInt>,
  target: &Variable,
) -> Expr {
  fold_owned_binary(
    values
      .into_iter()
      .map(|value| Expr::Comparison {
        operands: vec![
          Expr::Identifier(target.name.clone()),
          bigint_expr(&value),
        ],
        operators: vec![ComparisonOp::Equal],
      })
      .collect(),
    BinaryOperator::Or,
  )
}

fn integer_dnf(formula: &Formula) -> Option<Vec<Vec<Atom>>> {
  match formula {
    Formula::True => Some(vec![Vec::new()]),
    Formula::False => Some(Vec::new()),
    Formula::Atom(atom) => Some(vec![vec![atom.clone()]]),
    Formula::Or(children) => {
      let mut output = Vec::new();
      for child in children {
        output.extend(integer_dnf(child)?);
        if output.len() > 100_000 {
          return None;
        }
      }
      Some(output)
    }
    Formula::And(children) => {
      let mut product = vec![Vec::new()];
      for child in children {
        let alternatives = integer_dnf(child)?;
        let mut next = Vec::new();
        for prefix in &product {
          for alternative in &alternatives {
            let mut conjunction = prefix.clone();
            conjunction.extend(alternative.iter().cloned());
            next.push(conjunction);
            if next.len() > 100_000 {
              return None;
            }
          }
        }
        product = next;
      }
      Some(product)
    }
    Formula::Not(_) | Formula::Quantified(_, _, _) => None,
  }
}

fn finite_integer_branch_values(
  atoms: &[Atom],
  target: &Variable,
) -> Option<Vec<BigInt>> {
  let mut lower: Option<BigInt> = None;
  let mut upper: Option<BigInt> = None;
  let mut congruence = (BigInt::zero(), BigInt::one());
  let mut forbidden = Vec::new();

  for atom in atoms {
    match atom {
      Atom::Relation(Relation::LessEqual, term) => {
        if term.coefficients.len() != 1
          || !term.constant.is_integer()
          || !term.coefficient(target).is_integer()
        {
          return None;
        }
        let coefficient = term.coefficient(target).numerator;
        if coefficient.is_positive() {
          let bound =
            floor_div(&(-term.constant.numerator.clone()), &coefficient);
          upper = Some(match upper {
            Some(old) => std::cmp::min(old, bound),
            None => bound,
          });
        } else if coefficient.is_negative() {
          let bound =
            ceil_div(&(-term.constant.numerator.clone()), &coefficient);
          lower = Some(match lower {
            Some(old) => std::cmp::max(old, bound),
            None => bound,
          });
        } else {
          return None;
        }
      }
      Atom::Divides {
        modulus,
        term,
        negated,
      } => {
        if term.coefficients.len() != 1
          || !term.constant.is_integer()
          || !term.coefficient(target).is_integer()
        {
          return None;
        }
        let solved = solve_linear_congruence(
          &term.coefficient(target).numerator,
          &(-term.constant.numerator.clone()),
          modulus,
        );
        if *negated {
          if let Some(solved) = solved {
            forbidden.push(solved);
          }
        } else {
          let Some(solved) = solved else {
            return Some(Vec::new());
          };
          let Some(combined) =
            crt_pair(&congruence.0, &congruence.1, &solved.0, &solved.1)
          else {
            return Some(Vec::new());
          };
          congruence = combined;
        }
      }
      Atom::Relation(..) => return None,
    }
  }

  let (lower, upper) = (lower?, upper?);
  if lower > upper {
    return Some(Vec::new());
  }
  let combined_period = forbidden
    .iter()
    .fold(congruence.1.clone(), |period, (_, forbidden_period)| {
      lcm(&period, forbidden_period)
    });
  let residue_count = &combined_period / &congruence.1;
  if residue_count > BigInt::from(1_000_000_u32) {
    return None;
  }

  let mut output = Vec::new();
  let mut residue_index = BigInt::zero();
  while residue_index < residue_count {
    let residue = euclidean_mod(
      &congruence.0 + &residue_index * &congruence.1,
      &combined_period,
    );
    residue_index += 1;
    if forbidden.iter().any(|(forbidden_residue, period)| {
      euclidean_mod(residue.clone(), period) == *forbidden_residue
    }) {
      continue;
    }
    let mut value = &residue
      + ceil_div(&(&lower - &residue), &combined_period) * &combined_period;
    let output_size = if value > upper {
      BigInt::zero()
    } else {
      floor_div(&(&upper - &value), &combined_period) + 1
    };
    if BigInt::from(output.len()) + output_size > BigInt::from(1_000_000_u32) {
      return None;
    }
    while value <= upper {
      output.push(value.clone());
      value += &combined_period;
    }
  }
  Some(output)
}

/// Rounds concrete unary integer bounds inward and reduces concrete unary
/// congruences to a unit-coefficient canonical residue.
pub(super) fn canonical_integer_formula(
  formula: &Formula,
  targets: &[Variable],
) -> Formula {
  match formula {
    Formula::Atom(Atom::Relation(Relation::LessEqual, term))
      if term.coefficients.len() == 1
        && term.constant.is_integer()
        && targets.contains(term.coefficients.first_key_value().unwrap().0) =>
    {
      let (variable, coefficient) =
        term.coefficients.first_key_value().unwrap();
      if !coefficient.is_integer() || coefficient.is_zero() {
        return formula.clone();
      }
      let positive = coefficient.numerator.is_positive();
      let (coefficient, bound) = if positive {
        (
          Rational::one(),
          floor_div(
            &(-term.constant.numerator.clone()),
            &coefficient.numerator,
          ),
        )
      } else {
        (
          Rational::integer(BigInt::from(-1)),
          ceil_div(&(-term.constant.numerator.clone()), &coefficient.numerator),
        )
      };
      let term = AffineTerm::variable(variable.clone())
        .scaled(&coefficient)
        .add(&AffineTerm::constant(if positive {
          Rational::integer(-bound)
        } else {
          Rational::integer(bound)
        }));
      Formula::Atom(Atom::Relation(Relation::LessEqual, term))
    }
    Formula::Atom(Atom::Divides {
      modulus,
      term,
      negated,
    }) if term.coefficients.len() == 1
      && term.constant.is_integer()
      && targets.contains(term.coefficients.first_key_value().unwrap().0) =>
    {
      let (variable, coefficient) =
        term.coefficients.first_key_value().unwrap();
      if !coefficient.is_integer() {
        return formula.clone();
      }
      let Some((residue, period)) = solve_linear_congruence(
        &coefficient.numerator,
        &(-term.constant.numerator.clone()),
        modulus,
      ) else {
        return if *negated {
          Formula::True
        } else {
          Formula::False
        };
      };
      Formula::Atom(
        Atom::divides(
          period,
          AffineTerm::variable(variable.clone())
            .subtract(&AffineTerm::constant(Rational::integer(residue))),
          *negated,
        )
        .expect("a solved congruence is integral with positive modulus"),
      )
    }
    Formula::And(children) => Formula::And(
      children
        .iter()
        .map(|child| canonical_integer_formula(child, targets))
        .collect(),
    )
    .normalized(),
    Formula::Or(children) => Formula::Or(
      children
        .iter()
        .map(|child| canonical_integer_formula(child, targets))
        .collect(),
    )
    .normalized(),
    Formula::Not(inner) => {
      Formula::Not(Box::new(canonical_integer_formula(inner, targets)))
        .normalized()
    }
    Formula::Quantified(quantifier, variables, body) => Formula::Quantified(
      *quantifier,
      variables.clone(),
      Box::new(canonical_integer_formula(body, targets)),
    )
    .normalized(),
    Formula::True
    | Formula::False
    | Formula::Atom(Atom::Relation(_, _) | Atom::Divides { .. }) => {
      formula.clone()
    }
  }
}

fn floor_div(numerator: &BigInt, denominator: &BigInt) -> BigInt {
  debug_assert!(!denominator.is_zero());
  let quotient = numerator / denominator;
  let remainder = numerator % denominator;
  if !remainder.is_zero() && remainder.sign() != denominator.sign() {
    quotient - 1
  } else {
    quotient
  }
}

fn ceil_div(numerator: &BigInt, denominator: &BigInt) -> BigInt {
  -floor_div(&(-numerator), denominator)
}

fn atom_expr(atom: &Atom, targets: &[Variable]) -> Expr {
  match atom {
    Atom::Relation(relation, term) => relation_expr(*relation, term, targets),
    Atom::Divides {
      modulus,
      term,
      negated,
    } => {
      if let Some(expression) =
        target_congruence_expr(modulus, term, *negated, targets)
      {
        return expression;
      }
      let divides =
        call("Divisible", vec![term_expr(term), bigint_expr(modulus)]);
      if *negated {
        Expr::UnaryOp {
          op: UnaryOperator::Not,
          operand: Box::new(divides),
        }
      } else {
        divides
      }
    }
  }
}

fn target_congruence_expr(
  modulus: &BigInt,
  term: &AffineTerm,
  negated: bool,
  targets: &[Variable],
) -> Option<Expr> {
  for target in targets {
    let coefficient = term.coefficient(target);
    if !coefficient.is_integer()
      || coefficient.numerator.abs() != BigInt::one()
      || term.coefficients.len() != 1
      || !term.constant.is_integer()
    {
      continue;
    }
    let residue = if coefficient.numerator.sign() == num_bigint::Sign::Plus {
      -term.constant.numerator.clone()
    } else {
      term.constant.numerator.clone()
    };
    let mut residue = residue % modulus;
    if residue.sign() == num_bigint::Sign::Minus {
      residue += modulus;
    }
    return Some(Expr::Comparison {
      operands: vec![
        call(
          "Mod",
          vec![Expr::Identifier(target.name.clone()), bigint_expr(modulus)],
        ),
        bigint_expr(&residue),
      ],
      operators: vec![if negated {
        ComparisonOp::NotEqual
      } else {
        ComparisonOp::Equal
      }],
    });
  }
  None
}

fn relation_expr(
  relation: Relation,
  term: &AffineTerm,
  targets: &[Variable],
) -> Expr {
  let isolated = targets
    .iter()
    .find(|target| !term.coefficient(target).is_zero())
    .cloned()
    .or_else(|| {
      (term.coefficients.len() == 1)
        .then(|| term.coefficients.keys().next().unwrap().clone())
    });
  if let Some(variable) = isolated
    && let Some((relation, boundary)) =
      isolate_relation(relation, term, &variable)
  {
    return Expr::Comparison {
      operands: vec![Expr::Identifier(variable.name), term_expr(&boundary)],
      operators: vec![comparison_op(relation)],
    };
  }
  Expr::Comparison {
    operands: vec![term_expr(term), Expr::Integer(0)],
    operators: vec![comparison_op(relation)],
  }
}

fn isolate_relation(
  relation: Relation,
  term: &AffineTerm,
  variable: &Variable,
) -> Option<(Relation, AffineTerm)> {
  let coefficient = term.coefficient(variable);
  if coefficient.is_zero() {
    return None;
  }
  let mut rest = term.clone();
  rest.coefficients.remove(variable);
  let factor = Rational::integer((-1).into()).checked_divide(&coefficient)?;
  let boundary = rest.scaled(&factor);
  let relation = if coefficient.numerator.sign() == num_bigint::Sign::Minus {
    reverse_order(relation)
  } else {
    relation
  };
  Some((relation, boundary))
}

fn interval_expr(children: &[Formula], targets: &[Variable]) -> Option<Expr> {
  if children.len() != 2 {
    return None;
  }
  'targets: for target in targets {
    let mut lower = None;
    let mut upper = None;
    for child in children {
      let Formula::Atom(Atom::Relation(relation, term)) = child else {
        break;
      };
      let Some((relation, boundary)) =
        isolate_relation(*relation, term, target)
      else {
        continue 'targets;
      };
      match relation {
        Relation::Greater | Relation::GreaterEqual if lower.is_none() => {
          lower = Some((relation, boundary));
        }
        Relation::Less | Relation::LessEqual if upper.is_none() => {
          upper = Some((relation, boundary));
        }
        _ => break,
      }
    }
    if let (
      Some((lower_relation, lower_bound)),
      Some((upper_relation, upper_bound)),
    ) = (lower, upper)
    {
      return Some(call(
        "Inequality",
        vec![
          term_expr(&lower_bound),
          Expr::Identifier(
            match lower_relation {
              Relation::Greater => "Less",
              Relation::GreaterEqual => "LessEqual",
              _ => unreachable!(),
            }
            .to_string(),
          ),
          Expr::Identifier(target.name.clone()),
          Expr::Identifier(
            match upper_relation {
              Relation::Less => "Less",
              Relation::LessEqual => "LessEqual",
              _ => unreachable!(),
            }
            .to_string(),
          ),
          term_expr(&upper_bound),
        ],
      ));
    }
  }
  None
}

fn reverse_order(relation: Relation) -> Relation {
  match relation {
    Relation::Less => Relation::Greater,
    Relation::LessEqual => Relation::GreaterEqual,
    Relation::Greater => Relation::Less,
    Relation::GreaterEqual => Relation::LessEqual,
    Relation::Equal | Relation::NotEqual => relation,
  }
}

pub(super) fn term_expr(term: &AffineTerm) -> Expr {
  let mut summands = Vec::new();
  for (variable, coefficient) in &term.coefficients {
    summands.push(coefficient_variable_expr(coefficient, variable));
  }
  if !term.constant.is_zero() || summands.is_empty() {
    summands.push(rational_expr(&term.constant));
  }
  fold_owned_binary(summands, BinaryOperator::Plus)
}

fn coefficient_variable_expr(
  coefficient: &Rational,
  variable: &Variable,
) -> Expr {
  let variable = Expr::Identifier(variable.name.clone());
  if coefficient.numerator == BigInt::one() && coefficient.denominator.is_one()
  {
    variable
  } else if coefficient.numerator == -BigInt::one()
    && coefficient.denominator.is_one()
  {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(variable),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(rational_expr(coefficient)),
      right: Box::new(variable),
    }
  }
}

fn variables_expr(variables: &[Variable]) -> Expr {
  if variables.len() == 1 {
    Expr::Identifier(variables[0].name.clone())
  } else {
    Expr::List(
      variables
        .iter()
        .map(|variable| Expr::Identifier(variable.name.clone()))
        .collect::<Vec<_>>()
        .into(),
    )
  }
}

fn fold_binary(
  children: &[Formula],
  operator: BinaryOperator,
  targets: &[Variable],
) -> Expr {
  let mut ordered = children.iter().collect::<Vec<_>>();
  if operator == BinaryOperator::And && !targets.is_empty() {
    ordered.sort_by_key(|child| {
      targets.iter().any(|target| child.contains_variable(target))
    });
  }
  fold_owned_binary(
    ordered
      .into_iter()
      .map(|child| formula_expr_for_targets(child, targets))
      .collect::<Vec<_>>(),
    operator,
  )
}

fn fold_owned_binary(expressions: Vec<Expr>, operator: BinaryOperator) -> Expr {
  let mut expressions = expressions.into_iter();
  let Some(first) = expressions.next() else {
    return match operator {
      BinaryOperator::And => Expr::Identifier("True".to_string()),
      BinaryOperator::Or => Expr::Identifier("False".to_string()),
      BinaryOperator::Plus => Expr::Integer(0),
      _ => unreachable!("only associative n-ary operators are folded"),
    };
  };
  expressions.fold(first, |left, right| Expr::BinaryOp {
    op: operator,
    left: Box::new(left),
    right: Box::new(right),
  })
}

// ---------------------------------------------------------------------------
// Parametrized integer solution sets
// ---------------------------------------------------------------------------

/// wolframscript never reports an integer solution set as a congruence. It
/// parametrizes the residue class instead, with a fresh integer parameter per
/// constrained variable, and rewrites every bound on the variable as a bound
/// on that parameter:
///
/// ```text
/// Reduce[Mod[x, 2] == 1, x, Integers]            Element[C[1], Integers] &&
///                                                  x == 1 + 2*C[1]
/// Reduce[Mod[x, 6] == 4 && x > 10, x, Integers]  Element[C[1], Integers] &&
///                                                  C[1] >= 2 && x == 4 + 6*C[1]
/// ```
///
/// which is the same shape the Diophantine reducer in `reduce.rs` already
/// prints for a lone linear equation. This rewrites the Presburger engine's
/// congruence answers into it, one disjunct at a time, and gives back `None`
/// when there is no congruence to parametrize (a purely bounded answer keeps
/// its `Element[x, Integers] && x >= 1` shape) or when the branch is outside
/// what the rewrite can state exactly.
const MAX_PARAMETRIZED_CLASSES: usize = 64;

/// The residue classes one target is confined to within one branch:
/// `target ≡ residues[i] (mod modulus)`, with every residue in `[0, modulus)`.
struct TargetClasses {
  target: Variable,
  modulus: BigInt,
  residues: Vec<BigInt>,
}

/// What a single conjunctive branch says about its targets.
enum BranchClasses {
  /// No integer satisfies the branch.
  Unsatisfiable,
  Classes(Vec<TargetClasses>),
}

pub(super) fn parametrized_integer_expr(
  formula: &Formula,
  targets: &[Variable],
) -> Option<Expr> {
  let branches = integer_dnf(formula)?;
  if branches.is_empty() {
    return None;
  }

  let mut parametrized_any = false;
  let mut disjuncts = Vec::new();
  for branch in &branches {
    let BranchClasses::Classes(classes) = branch_classes(branch, targets)?
    else {
      continue;
    };
    if classes.is_empty() {
      disjuncts.extend(plain_branch_expr(branch, targets));
      continue;
    }
    parametrized_any = true;
    disjuncts.extend(parametrized_branch_exprs(branch, targets, &classes));
  }

  if !parametrized_any {
    return None;
  }
  Some(fold_owned_binary(disjuncts, BinaryOperator::Or))
}

/// The residue classes each target is confined to in one branch. `None` means
/// the branch is outside the rewrite: a congruence tying several variables
/// together, more classes than are worth writing out, or an equation pinning a
/// congruent variable (`Mod[x, 2] == 1 && x == 5`, which wolframscript answers
/// with the value `x == 5` rather than with a parametrization).
fn branch_classes(
  branch: &[Atom],
  targets: &[Variable],
) -> Option<BranchClasses> {
  let mut classes = Vec::new();
  for target in targets {
    let mut residue = BigInt::zero();
    let mut modulus = BigInt::one();
    let mut forbidden: Vec<(BigInt, BigInt)> = Vec::new();

    for atom in branch {
      let Atom::Divides {
        modulus: atom_modulus,
        term,
        negated,
      } = atom
      else {
        continue;
      };
      if term.coefficient(target).is_zero() {
        continue;
      }
      if term.coefficients.len() != 1
        || !term.constant.is_integer()
        || !term.coefficient(target).is_integer()
      {
        return None;
      }
      let solved = solve_linear_congruence(
        &term.coefficient(target).numerator,
        &(-term.constant.numerator.clone()),
        atom_modulus,
      );
      match (solved, negated) {
        (Some(solved), true) => forbidden.push(solved),
        (Some(solved), false) => {
          let Some(combined) =
            crt_pair(&residue, &modulus, &solved.0, &solved.1)
          else {
            return Some(BranchClasses::Unsatisfiable);
          };
          (residue, modulus) = combined;
        }
        // A congruence no integer solves: unsatisfiable, or vacuous when it
        // is the one being denied.
        (None, true) => {}
        (None, false) => return Some(BranchClasses::Unsatisfiable),
      }
    }

    let period = forbidden
      .iter()
      .fold(modulus.clone(), |period, (_, forbidden_period)| {
        lcm(&period, forbidden_period)
      });
    if period > BigInt::from(MAX_PARAMETRIZED_CLASSES) {
      return None;
    }
    let period_len = period.to_usize()?;
    let residues = (0..period_len)
      .map(BigInt::from)
      .filter(|candidate| {
        euclidean_mod(candidate.clone(), &modulus) == residue
          && !forbidden
            .iter()
            .any(|(forbidden_residue, forbidden_period)| {
              euclidean_mod(candidate.clone(), forbidden_period)
                == *forbidden_residue
            })
      })
      .collect::<Vec<_>>();

    if residues.is_empty() {
      return Some(BranchClasses::Unsatisfiable);
    }
    // Every residue allowed is no constraint at all, so the target keeps its
    // plain `Element[x, Integers]` membership and needs no parameter.
    if residues.len() == period_len {
      continue;
    }
    classes.push(TargetClasses {
      target: target.clone(),
      modulus: period,
      residues,
    });
  }

  let pins_a_class = branch.iter().any(|atom| {
    matches!(atom, Atom::Relation(Relation::Equal, term)
      if classes.iter().any(|class| !term.coefficient(&class.target).is_zero()))
  });
  if pins_a_class {
    return None;
  }

  // One disjunct per combination of classes, so a product of many classes is
  // left as the congruence it is rather than written out.
  let combinations = classes.iter().try_fold(1_usize, |total, class| {
    total.checked_mul(class.residues.len())
  })?;
  if combinations > MAX_PARAMETRIZED_CLASSES {
    return None;
  }

  Some(BranchClasses::Classes(classes))
}

/// A branch with no congruence keeps the shape it has always had: the
/// membership of every target it mentions, then the branch itself. A branch
/// that pins its target between two bounds is reported by its values instead
/// (`x == 8`, not `Element[x, Integers] && 8 <= x <= 8`), the same way a
/// wholly finite answer is. `None` means no integer satisfies the branch.
fn plain_branch_expr(branch: &[Atom], targets: &[Variable]) -> Option<Expr> {
  if let [target] = targets
    && let Some(values) = finite_integer_branch_values(branch, target)
  {
    if values.is_empty() {
      return None;
    }
    return Some(integer_values_expr(values, target));
  }
  let formula = Formula::And(
    branch
      .iter()
      .map(|atom| Formula::Atom(atom.clone()))
      .collect(),
  )
  .normalized();
  let mut conjuncts = targets
    .iter()
    .filter(|target| formula.contains_variable(target))
    .map(|target| integer_membership(Expr::Identifier(target.name.clone())))
    .collect::<Vec<_>>();
  conjuncts.push(formula_expr_for_targets(&formula, targets));
  Some(fold_owned_binary(conjuncts, BinaryOperator::And))
}

/// One disjunct per combination of residue classes, in the order the targets
/// were asked for. Combinations no integer satisfies are dropped.
fn parametrized_branch_exprs(
  branch: &[Atom],
  targets: &[Variable],
  classes: &[TargetClasses],
) -> Vec<Expr> {
  let mut disjuncts = Vec::new();
  for choice in residue_combinations(classes) {
    if let Some(expression) =
      parametrized_branch_expr(branch, targets, classes, &choice)
    {
      disjuncts.push(expression);
    }
  }
  if disjuncts.is_empty() {
    return Vec::new();
  }
  factor_shared_membership(disjuncts)
}

fn residue_combinations(classes: &[TargetClasses]) -> Vec<Vec<BigInt>> {
  let mut combinations = vec![Vec::new()];
  for class in classes {
    let mut next = Vec::new();
    for prefix in &combinations {
      for residue in &class.residues {
        let mut extended = prefix.clone();
        extended.push(residue.clone());
        next.push(extended);
      }
    }
    combinations = next;
  }
  combinations
}

/// `Element[C[1], Integers] && x == 3*C[1]` together with
/// `Element[C[1], Integers] && x == 2 + 3*C[1]` is written by wolframscript
/// with the membership pulled out front — but only while the classes carry
/// nothing else, since a per-class bound has to stay with its own class.
fn factor_shared_membership(disjuncts: Vec<Expr>) -> Vec<Expr> {
  if disjuncts.len() < 2 {
    return disjuncts;
  }
  let mut membership: Option<Expr> = None;
  let mut equations = Vec::new();
  for disjunct in &disjuncts {
    let Expr::BinaryOp {
      op: BinaryOperator::And,
      left,
      right,
    } = disjunct
    else {
      return disjuncts;
    };
    let same_membership = membership.as_ref().is_none_or(|first| {
      crate::syntax::expr_to_string(first)
        == crate::syntax::expr_to_string(left)
    });
    if !matches!(**right, Expr::Comparison { .. }) || !same_membership {
      return disjuncts;
    }
    membership = Some((**left).clone());
    equations.push((**right).clone());
  }
  let Some(membership) = membership else {
    return disjuncts;
  };
  vec![Expr::BinaryOp {
    op: BinaryOperator::And,
    left: Box::new(membership),
    right: Box::new(fold_owned_binary(equations, BinaryOperator::Or)),
  }]
}

fn parametrized_branch_expr(
  branch: &[Atom],
  targets: &[Variable],
  classes: &[TargetClasses],
  choice: &[BigInt],
) -> Option<Expr> {
  let parameters = classes
    .iter()
    .enumerate()
    .zip(choice)
    .map(|((slot, class), residue)| {
      let variable = parameter_variable(slot);
      let replacement = AffineTerm::variable(variable.clone())
        .scaled(&Rational::integer(class.modulus.clone()))
        .add(&AffineTerm::constant(Rational::integer(residue.clone())));
      (class, variable, residue, replacement)
    })
    .collect::<Vec<_>>();

  // The parameters lead the isolation order, so a relation that survives the
  // substitution is stated as a bound on the parameter it came from.
  let substituted_targets = parameters
    .iter()
    .map(|(_, variable, _, _)| variable.clone())
    .chain(
      targets
        .iter()
        .filter(|target| !classes.iter().any(|class| class.target == **target))
        .cloned(),
    )
    .collect::<Vec<_>>();

  let mut parameter_bounds = Vec::new();
  let mut remaining = Vec::new();
  for atom in branch {
    if matches!(atom, Atom::Divides { term, .. }
      if classes.iter().any(|class| !term.coefficient(&class.target).is_zero()))
    {
      continue;
    }
    let mut atom = atom.clone();
    for (class, _, _, replacement) in &parameters {
      atom = atom.substitute(&class.target, replacement);
    }
    let atom = match Formula::Atom(atom).normalized() {
      Formula::True => continue,
      // The bounds rule this residue class out entirely.
      Formula::False => return None,
      Formula::Atom(atom) => atom,
      _ => unreachable!("normalizing one atom cannot introduce structure"),
    };
    let only_parameters = atom.variables().iter().all(|variable| {
      parameters
        .iter()
        .any(|(_, parameter, _, _)| parameter == variable)
    });
    let expression = formula_expr_for_targets(
      &canonical_integer_formula(&Formula::Atom(atom), &substituted_targets),
      &substituted_targets,
    );
    if only_parameters {
      parameter_bounds.push(expression);
    } else {
      remaining.push(expression);
    }
  }

  // `Element[y | C[1], Integers]`: the targets that keep themselves first, in
  // the order they were asked for, then one parameter per congruent target.
  let mut members = targets
    .iter()
    .filter(|target| {
      !classes.iter().any(|class| class.target == **target)
        && branch.iter().any(|atom| atom.variables().contains(*target))
    })
    .map(|target| Expr::Identifier(target.name.clone()))
    .collect::<Vec<_>>();
  members.extend(
    parameters
      .iter()
      .map(|(_, variable, _, _)| Expr::Identifier(variable.name.clone())),
  );
  let membership =
    members.into_iter().reduce(|left, right| Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      left: Box::new(left),
      right: Box::new(right),
    })?;

  let mut conjuncts = vec![integer_membership(membership)];
  conjuncts.append(&mut parameter_bounds);
  for (class, variable, residue, _) in &parameters {
    conjuncts.push(Expr::Comparison {
      operands: vec![
        Expr::Identifier(class.target.name.clone()),
        parametrized_value(residue, &class.modulus, variable),
      ],
      operators: vec![ComparisonOp::Equal],
    });
  }
  conjuncts.append(&mut remaining);

  let expression = fold_owned_binary(conjuncts, BinaryOperator::And);
  Some(name_parameters(expression, parameters.len()))
}

/// `residue + modulus*C`, spelled the way wolframscript spells it.
fn parametrized_value(
  residue: &BigInt,
  modulus: &BigInt,
  parameter: &Variable,
) -> Expr {
  let parameter = Expr::Identifier(parameter.name.clone());
  let scaled = if modulus.is_one() {
    parameter
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(bigint_expr(modulus)),
      right: Box::new(parameter),
    }
  };
  if residue.is_zero() {
    scaled
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(bigint_expr(residue)),
      right: Box::new(scaled),
    }
  }
}

/// Parameters travel through the affine IR as plain variables, since it has
/// no room for `C[1]`'s indexed head. They take their real spelling once the
/// branch has been emitted.
fn parameter_variable(slot: usize) -> Variable {
  Variable::free(format!("$WoxiIntegerParameter{}", slot + 1))
}

fn name_parameters(expression: Expr, count: usize) -> Expr {
  (0..count).fold(expression, |expression, slot| {
    crate::syntax::substitute_variable(
      &expression,
      &parameter_variable(slot).name,
      &call("C", vec![Expr::Integer(slot as i128 + 1)]),
    )
  })
}

fn integer_membership(what: Expr) -> Expr {
  call(
    "Element",
    vec![what, Expr::Identifier("Integers".to_string())],
  )
}

fn comparison_op(relation: Relation) -> ComparisonOp {
  match relation {
    Relation::Equal => ComparisonOp::Equal,
    Relation::NotEqual => ComparisonOp::NotEqual,
    Relation::Less => ComparisonOp::Less,
    Relation::LessEqual => ComparisonOp::LessEqual,
    Relation::Greater => ComparisonOp::Greater,
    Relation::GreaterEqual => ComparisonOp::GreaterEqual,
  }
}

#[cfg(test)]
mod tests {
  use std::collections::BTreeMap;

  use super::*;
  use crate::syntax::expr_to_string;

  #[test]
  fn affine_emission_is_stable_and_exact() {
    let term = AffineTerm {
      constant: Rational::new(BigInt::from(2), BigInt::from(3)).unwrap(),
      coefficients: BTreeMap::from([
        (
          Variable::free("y"),
          Rational::new(BigInt::from(-1), BigInt::from(2)).unwrap(),
        ),
        (Variable::free("x"), Rational::one()),
      ]),
    };
    assert_eq!(expr_to_string(&term_expr(&term)), "x - y/2 + 2/3");
  }

  #[test]
  fn formula_emission_preserves_boolean_and_divisibility_structure() {
    let x = Variable::free("x");
    let less = Formula::Atom(Atom::Relation(
      Relation::Less,
      AffineTerm::variable(x.clone()),
    ));
    let odd = Formula::Atom(
      Atom::divides(
        BigInt::from(2),
        AffineTerm::variable(x)
          .subtract(&AffineTerm::constant(Rational::one())),
        false,
      )
      .unwrap(),
    );
    let formula = Formula::Or(vec![less, odd]).normalized();
    assert_eq!(
      expr_to_string(&formula_expr(&formula)),
      "x < 0 || Divisible[x + -1, 2]"
    );
  }

  #[test]
  fn target_relation_is_isolated_and_negative_coefficients_reverse_order() {
    let x = Variable::free("x");
    let term = AffineTerm::variable(x.clone())
      .scaled(&Rational::integer(BigInt::from(-2)))
      .add(&AffineTerm::constant(Rational::integer(BigInt::from(3))));
    let formula = Formula::Atom(Atom::Relation(Relation::LessEqual, term));
    assert_eq!(
      expr_to_string(&formula_expr_for_targets(&formula, &[x])),
      "x >= 3/2"
    );
  }

  #[test]
  fn two_target_bounds_emit_as_an_inequality_chain() {
    let x = Variable::free("x");
    let formula = Formula::And(vec![
      Formula::Atom(Atom::Relation(
        Relation::Greater,
        AffineTerm::variable(x.clone()),
      )),
      Formula::Atom(Atom::Relation(
        Relation::Less,
        AffineTerm::variable(x.clone())
          .subtract(&AffineTerm::constant(Rational::integer(BigInt::from(2)))),
      )),
    ])
    .normalized();
    assert_eq!(
      expr_to_string(&formula_expr_for_targets(&formula, &[x])),
      "Inequality[0, Less, x, Less, 2]"
    );
  }

  #[test]
  fn unit_target_congruence_emits_a_canonical_mod_residue() {
    let x = Variable::free("x");
    let formula = Formula::Atom(
      Atom::divides(
        BigInt::from(6),
        AffineTerm::variable(x.clone())
          .subtract(&AffineTerm::constant(Rational::integer(BigInt::from(7)))),
        false,
      )
      .unwrap(),
    );
    assert_eq!(
      expr_to_string(&formula_expr_for_targets(&formula, &[x])),
      "Mod[x, 6] == 1"
    );
  }

  #[test]
  fn finite_integer_interval_and_residue_emit_an_arithmetic_progression() {
    let x = Variable::free("x");
    let formula = Formula::And(vec![
      Formula::Atom(Atom::Relation(
        Relation::LessEqual,
        AffineTerm::variable(x.clone())
          .scaled(&Rational::integer(BigInt::from(-1))),
      )),
      Formula::Atom(Atom::Relation(
        Relation::LessEqual,
        AffineTerm::variable(x.clone())
          .subtract(&AffineTerm::constant(Rational::integer(BigInt::from(10)))),
      )),
      Formula::Atom(
        Atom::divides(
          BigInt::from(3),
          AffineTerm::variable(x.clone()).subtract(&AffineTerm::constant(
            Rational::integer(BigInt::from(1)),
          )),
          false,
        )
        .unwrap(),
      ),
    ])
    .normalized();
    assert_eq!(
      expr_to_string(&finite_integer_target_expr(&formula, &x).unwrap()),
      "x == 1 || x == 4 || x == 7 || x == 10"
    );
  }

  #[test]
  fn exact_signed_floor_and_ceiling_division_bracket_rationals() {
    assert_eq!(floor_div(&BigInt::from(-7), &BigInt::from(3)), (-3).into());
    assert_eq!(ceil_div(&BigInt::from(-7), &BigInt::from(3)), (-2).into());
    assert_eq!(floor_div(&BigInt::from(7), &BigInt::from(-3)), (-3).into());
    assert_eq!(ceil_div(&BigInt::from(7), &BigInt::from(-3)), (-2).into());
  }

  #[test]
  fn integer_output_rounds_bounds_and_reduces_linear_congruences() {
    let x = Variable::free("x");
    let lower = Formula::Atom(Atom::Relation(
      Relation::LessEqual,
      AffineTerm::variable(x.clone())
        .scaled(&Rational::integer(BigInt::from(-3)))
        .add(&AffineTerm::constant(Rational::integer(BigInt::from(8)))),
    ));
    assert_eq!(
      expr_to_string(&formula_expr_for_targets(
        &canonical_integer_formula(&lower, std::slice::from_ref(&x)),
        std::slice::from_ref(&x),
      )),
      "x >= 3"
    );

    let congruence = Formula::Atom(
      Atom::divides(
        BigInt::from(10),
        AffineTerm::variable(x.clone())
          .scaled(&Rational::integer(BigInt::from(6)))
          .add(&AffineTerm::constant(Rational::integer(BigInt::from(4)))),
        false,
      )
      .unwrap(),
    );
    assert_eq!(
      expr_to_string(&formula_expr_for_targets(
        &canonical_integer_formula(&congruence, std::slice::from_ref(&x)),
        std::slice::from_ref(&x),
      )),
      "Mod[x, 5] == 1"
    );
  }

  #[test]
  fn finite_integer_dnf_materializes_negated_residue_classes() {
    let x = Variable::free("x");
    let formula = Formula::And(vec![
      Formula::Atom(Atom::Relation(
        Relation::LessEqual,
        AffineTerm::variable(x.clone())
          .scaled(&Rational::integer(BigInt::from(-1))),
      )),
      Formula::Atom(Atom::Relation(
        Relation::LessEqual,
        AffineTerm::variable(x.clone())
          .subtract(&AffineTerm::constant(Rational::integer(BigInt::from(4)))),
      )),
      Formula::Atom(
        Atom::divides(BigInt::from(2), AffineTerm::variable(x.clone()), true)
          .unwrap(),
      ),
    ])
    .normalized();
    assert_eq!(
      expr_to_string(&finite_integer_target_expr(&formula, &x).unwrap()),
      "x == 1 || x == 3"
    );
  }
}
