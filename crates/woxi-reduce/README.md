# woxi-reduce

`woxi-reduce` is a self-contained Rust library for exact quantifier
elimination in first-order linear arithmetic:

- ordered linear arithmetic over exact rationals and reals, using exact
  equality substitution and Fourier–Motzkin projection;
- Presburger arithmetic over integers, including order, equality,
  disequality, divisibility, congruences, Boolean structure, and arbitrarily
  nested existential and universal quantifiers.

The library has no parser, evaluator, subprocess, network, Wolfram, SMT, or
bounded-search runtime dependency. Mathematical coefficients use canonical
`BigInt` rationals. Frontends construct the binder-safe `Formula` IR and call
`rational_qe::eliminate_quantifiers` or
`presburger::eliminate_quantifiers`.

```rust
use num_bigint::BigInt;
use woxi_reduce::rational_qe;
use woxi_reduce::{AffineTerm, Atom, Formula, Quantifier, Rational, Relation, Variable};

let x = Variable::bound("x", 0);
let formula = Formula::Quantified(
  Quantifier::Exists,
  vec![x.clone()],
  Box::new(Formula::Atom(Atom::Relation(
    Relation::Equal,
    AffineTerm::variable(x)
      .subtract(&AffineTerm::constant(Rational::integer(BigInt::from(3)))),
  ))),
);

assert_eq!(rational_qe::eliminate_quantifiers(formula), Some(Formula::True));
```

Run the isolated development loop with:

```console
cargo test -p woxi-reduce
```

Woxi supplies the Wolfram Language lowering and emission adapters separately.
