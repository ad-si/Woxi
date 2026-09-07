//! Exact, deterministic decision procedures for first-order linear arithmetic.
//!
//! The crate is independent of Woxi's parser, evaluator, and expression tree.
//! Frontends lower their own syntax into the binder-safe [`Formula`] IR, call
//! one of the theory engines, and translate the quantifier-free result back to
//! their surface representation.

pub mod affine;
pub mod equalities;
pub mod exact;
pub mod formula;
pub mod presburger;
pub mod rational_qe;

pub use affine::{AffineTerm, Variable};
pub use exact::{
  Rational, ceil_div, crt_pair, euclidean_mod, extended_gcd, floor_div, gcd,
  lcm, solve_linear_congruence,
};
pub use formula::{Atom, Formula, FormulaMemo, Quantifier, Relation};
