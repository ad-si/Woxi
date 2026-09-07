use criterion::{Criterion, black_box, criterion_group, criterion_main};
use num_bigint::BigInt;
use woxi_reduce::presburger;
use woxi_reduce::rational_qe;
use woxi_reduce::{
  AffineTerm, Atom, Formula, Quantifier, Rational, Relation, Variable,
};

fn integer(value: i64) -> Rational {
  Rational::integer(BigInt::from(value))
}

fn term(variable: &Variable) -> AffineTerm {
  AffineTerm::variable(variable.clone())
}

fn relation(relation: Relation, term: AffineTerm) -> Formula {
  Formula::Atom(Atom::Relation(relation, term))
}

fn sparse_system() -> Formula {
  let x = Variable::bound("x", 0);
  let y = Variable::bound("y", 1);
  let z = Variable::bound("z", 2);
  let a = Variable::free("a");
  Formula::Quantified(
    Quantifier::Exists,
    vec![x.clone(), y.clone(), z.clone()],
    Box::new(Formula::And(vec![
      relation(Relation::Equal, term(&x).subtract(&term(&a))),
      relation(Relation::Equal, term(&y).subtract(&term(&x))),
      relation(Relation::Equal, term(&z).subtract(&term(&y))),
      relation(Relation::GreaterEqual, term(&z)),
    ])),
  )
}

fn dense_system() -> Formula {
  let x = Variable::bound("x", 0);
  let mut bounds = Vec::new();
  for index in 0..6 {
    let lower = Variable::free(format!("lower{index}"));
    let upper = Variable::free(format!("upper{index}"));
    bounds.push(relation(
      Relation::Greater,
      term(&x).subtract(&term(&lower)),
    ));
    bounds.push(relation(
      Relation::LessEqual,
      term(&x).subtract(&term(&upper)),
    ));
  }
  Formula::Quantified(
    Quantifier::Exists,
    vec![x],
    Box::new(Formula::And(bounds)),
  )
}

fn boolean_branching() -> Formula {
  let x = Variable::bound("x", 0);
  let branches = (-4_i64..4)
    .map(|value| {
      Formula::And(vec![
        relation(
          Relation::GreaterEqual,
          term(&x).subtract(&AffineTerm::constant(integer(value))),
        ),
        relation(
          Relation::Less,
          term(&x).subtract(&AffineTerm::constant(integer(value + 1))),
        ),
      ])
    })
    .collect();
  Formula::Quantified(
    Quantifier::Exists,
    vec![x],
    Box::new(Formula::Or(branches)),
  )
}

fn large_coefficients() -> Formula {
  let x = Variable::bound("x", 0);
  let y = Variable::free("y");
  let coefficient = (BigInt::from(1_u8) << 256) + BigInt::from(297_u16);
  Formula::Quantified(
    Quantifier::Exists,
    vec![x.clone()],
    Box::new(Formula::And(vec![
      relation(
        Relation::Equal,
        term(&x)
          .scaled(&Rational::integer(coefficient))
          .add(&term(&y)),
      ),
      relation(Relation::Greater, term(&x)),
    ])),
  )
}

fn congruence_lcm_growth() -> Formula {
  let x = Variable::bound("x", 0);
  Formula::Quantified(
    Quantifier::Exists,
    vec![x.clone()],
    Box::new(Formula::And(vec![
      Formula::Atom(Atom::divides(BigInt::from(6), term(&x), false).unwrap()),
      Formula::Atom(
        Atom::divides(
          BigInt::from(10),
          term(&x).subtract(&AffineTerm::constant(integer(1))),
          false,
        )
        .unwrap(),
      ),
      Formula::Atom(
        Atom::divides(
          BigInt::from(15),
          term(&x).subtract(&AffineTerm::constant(integer(4))),
          false,
        )
        .unwrap(),
      ),
    ])),
  )
}

fn alternating_quantifiers() -> Formula {
  let x = Variable::bound("x", 0);
  let y = Variable::bound("y", 1);
  Formula::Quantified(
    Quantifier::ForAll,
    vec![x.clone()],
    Box::new(Formula::Quantified(
      Quantifier::Exists,
      vec![y.clone()],
      Box::new(relation(Relation::Greater, term(&y).subtract(&term(&x)))),
    )),
  )
}

fn benchmarks(criterion: &mut Criterion) {
  let sparse = sparse_system();
  criterion.bench_function("reduce/sparse_dense_system", |bencher| {
    bencher.iter(|| {
      black_box(rational_qe::eliminate_quantifiers(black_box(
        sparse.clone(),
      )))
    });
  });

  let dense = dense_system();
  criterion.bench_function("reduce/dense_fourier_motzkin", |bencher| {
    bencher.iter(|| {
      black_box(rational_qe::eliminate_quantifiers(black_box(dense.clone())))
    });
  });

  let branching = boolean_branching();
  criterion.bench_function("reduce/boolean_branching", |bencher| {
    bencher.iter(|| {
      black_box(rational_qe::eliminate_quantifiers(black_box(
        branching.clone(),
      )))
    });
  });

  let large = large_coefficients();
  criterion.bench_function("reduce/large_coefficients", |bencher| {
    bencher.iter(|| {
      black_box(rational_qe::eliminate_quantifiers(black_box(large.clone())))
    });
  });

  let congruences = congruence_lcm_growth();
  criterion.bench_function("reduce/congruence_lcm_growth", |bencher| {
    bencher.iter(|| {
      black_box(presburger::eliminate_quantifiers(black_box(
        congruences.clone(),
      )))
    });
  });

  let alternating = alternating_quantifiers();
  criterion.bench_function("reduce/alternating_quantifiers", |bencher| {
    bencher.iter(|| {
      black_box(rational_qe::eliminate_quantifiers(black_box(
        alternating.clone(),
      )))
    });
  });
}

criterion_group!(reduce, benchmarks);
criterion_main!(reduce);
