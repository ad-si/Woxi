# Self-contained linear `Reduce` roadmap

Status: active. The exact linear rational/real and Presburger decision
procedures are integrated and the former SMT-RAT subprocess bridge has been
removed. Robustness, performance, full-repository audit, and external Wolfram
surface-conformance gates remain open.

## Completion contract

The project is complete when Woxi has an exact, deterministic, pure-Rust
decision procedure for this grammar:

```text
term    := exact-rational | variable | exact-rational * variable | term + term
atom    := term == term | term != term | term < term | term <= term
         | term > term | term >= term
         | Divisible[term, positive-integer]
formula := True | False | atom | !formula | formula && formula
         | formula || formula | formula xor formula
         | Exists[variables, formula] | ForAll[variables, formula]
domain  := Reals | Rationals | Integers
```

`Mod[term, m] == r` and its negation are accepted as surface forms of integer
divisibility/congruence atoms. Chained comparisons and the Wolfram Language
function-call forms of the Boolean operators are part of the surface grammar.
Quantifier nesting and alternation are unrestricted; practical running time is
necessarily subject to the known worst-case complexity of the theories.

Nonlinear polynomial, algebraic-number, complex, transcendental, approximate,
and mixed integer/real formulas are explicitly outside this campaign. Existing
specialized Woxi behavior for them must not regress, but it does not count
toward completion of the linear decision procedures.

For every input inside the scoped grammar:

1. `Reduce` and `Resolve` return a logically equivalent, quantifier-free result.
2. No requested or bound variable that should be eliminated occurs in the
   result.
3. The result is computed with `BigInt` and normalized exact rationals; no
   conversion through `f32`, `f64`, `i64`, or `i128` is allowed.
4. No subprocess, network service, Wolfram installation, SMT solver, or bounded
   search is required at runtime.
5. Output is deterministic and matches `wolframscript` exactly for the checked
   conformance corpus. Generated tests additionally check semantic equivalence,
   because equivalent quantifier-free formulas need not have identical text.
6. The implementation works in native and WebAssembly builds.

An unsupported formula may remain unevaluated through the existing fallback.
An in-scope formula may not. The active goal is not complete until every gate
below is complete and its evidence is recorded in the evidence ledger.

## Gate 0: specification and executable harness

- [x] Write `docs/reduce/algorithms.md` with normalization rules, elimination
      theorems, pseudocode, invariants, and a theorem-to-module reference map.
- [x] Add a dedicated `tests/reduce/` corpus divided into rational semantics,
      Presburger semantics, Wolfram surface conformance, and regressions.
- [x] Add `make test-reduce`, `make test-reduce-oracle`, and
      `make test-reduce-fuzz` targets. Only the oracle target may need external
      development tools; `make test-reduce` must be completely self-contained.
- [x] Record deterministic generator seeds and retain every discovered failure
      as a named regression.
- [x] Capture the pre-change Woxi test baseline.

Exit test: the three targets exist, the self-contained target runs without an
SMT-RAT or Wolfram executable, and every row of the grammar above has at least
one harness case.

## Gate 1: exact formula intermediate representation

Implement the pure theory in the standalone `crates/woxi-reduce/` package,
leaving `src/functions/polynomial_ast/reduce_backend/` responsible for Woxi
syntax lowering/emission and `reduce.rs` responsible for evaluator integration
and existing out-of-scope fallbacks.

- [x] One canonical normalized rational type backed by `BigInt`.
- [x] Sparse affine terms with sorted variables and primitive coefficient
      normalization.
- [x] Relations, divisibility atoms, Boolean formulas, domains, and explicit
      quantifier binders.
- [x] Capture-avoiding alpha-renaming, free-variable analysis, NNF conversion,
      constant folding, and associative flattening.
- [x] Complete lowering from every in-scope Woxi `Expr`; reject nonlinear or
      inexact terms without partially interpreting them as linear.
- [x] A deterministic emitter back to Woxi `Expr`.

Exit tests:

- 100% of the grammar matrix lowers successfully.
- 100% of the rejection matrix is rejected as out of scope.
- Property tests cover normalization idempotence, alpha-renaming invariance,
  rational sign/GCD invariants, and input-order-independent output.
- Coefficients larger than 128 bits occur in the permanent test corpus.

## Gate 2: full linear rational and real arithmetic

- [x] Equality substitution with exact pivoting.
- [x] Fourier-Motzkin elimination for strict and non-strict bounds, including
      unbounded variables and contradiction/tautology detection.
- [x] Disequality splitting and Boolean quantifier elimination for arbitrary
      NNF formulas, not merely conjunctions.
- [x] Universal quantifiers through checked logical duality.
- [x] Formula simplification, subsumption, and deterministic branch ordering.
- [x] `Reduce[..., Reals]`, `Reduce[..., Rationals]`, and corresponding
      `Resolve` forms route through the new engine.

Exit tests:

- At least 200 curated tests span all atom kinds, empty/unbounded regions,
  multiple eliminated variables, free parameters, negation, and quantifier
  alternation through depth three.
- 10,000 fixed-seed generated formulas agree semantically before and after
  elimination under exact rational valuations.
- At least 500 generated closed formulas agree with an independent development
  oracle, and at least 100 curated cases match `wolframscript` output exactly.
- Every in-scope rational/real test result is quantifier-free, has the correct
  free-variable set, and contains no unevaluated `Reduce` or `Resolve`.

## Gate 3: full Presburger arithmetic

- [x] Normalize integer comparisons to primitive affine inequalities and
      normalize congruences to positive moduli and canonical residues.
- [x] Implement GCD consistency, extended GCD, LCM scaling, and generalized
      CRT helpers with exact `BigInt` arithmetic.
- [x] Implement Cooper elimination, including coefficient normalization,
      divisibility constraints, finite boundary/residue sets, and the
      no-lower-bound/no-upper-bound cases.
- [x] Support equality, disequality, strict order, `Mod`, negation, arbitrary
      Boolean structure, multiple variables, and arbitrary quantifier
      alternation.
- [x] Replace bounded integer enumeration for every in-scope request. Small
      bounded enumeration remains useful only inside tests as an oracle.
- [x] Route integer `Reduce` and `Resolve` through the same canonical engine.

Exit tests:

- At least 300 curated tests cover signs, zero coefficients, negative inputs,
  non-coprime moduli, inconsistent congruences, one-sided formulas, large
  constants, strict-bound shifts, and alternation through depth three.
- 25,000 fixed-seed bounded generated formulas agree with exact exhaustive
  evaluation; every failure becomes a permanent regression before proceeding.
- At least 1,000 generated closed formulas agree with an independent
  Presburger oracle, and at least 150 curated cases match `wolframscript`
  output exactly.
- In-scope unbounded examples terminate by decision procedure rather than a
  search cap, including satisfiable and unsatisfiable cases.

## Gate 4: Woxi integration and removal of the exoskeleton

- [x] `Reduce` and `Resolve` share one normalization and elimination engine;
      neither reimplements the theory.
- [x] Existing `Solve`/`ToRules` callers consume the new canonical result
      without theory-specific duplication.
- [x] Remove the SMT-RAT subprocess bridge and its environment variables from
      production code and user documentation.
- [x] Keep existing nonlinear, modular-polynomial, complex, and transcendental
      behavior on explicit fallback routes.
- [x] Add unit, CLI documentation, and script snapshot tests according to
      `AGENTS.md`; update `functions.csv` descriptions.
- [x] Verify a WebAssembly build and a native build with an environment that
      has no solver executables on `PATH`.

Exit test: repository search and runtime tracing show no external command can
be reached from the scoped `Reduce`/`Resolve` routes, while all old Reduce,
Resolve, and Solve regressions pass.

## Gate 5: robustness and performance engineering

- [x] Add formula hash-consing or equivalent sharing where measurements show
      repeated subformulas.
- [x] Add elimination-order heuristics with deterministic tie breaking.
- [x] Add simplification and subsumption between elimination steps, with
      equivalence property tests for every rewrite.
- [x] Add benchmarks for sparse/dense systems, Boolean branching, large
      coefficients, congruence LCM growth, and alternating quantifiers.
- [ ] Run at least 100,000 parser/normalizer/eliminator fuzz cases without a
      panic, nondeterministic result, invalid rational, or leaked binder.
- [ ] Set generous CI smoke budgets for a named industrial benchmark corpus;
      retain detailed timing and formula-growth measurements as non-flaky
      benchmark reports rather than pretending Presburger has a universal
      polynomial-time bound.

Exit test: the benchmark corpus completes within its documented smoke budgets,
all rewrite property tests pass, and no correctness gate is waived for speed.

## Gate 6: final conformance and audit

- [ ] `make format`
- [ ] `make lint`
- [ ] `make test-reduce`
- [ ] `make test`
- [ ] `make test-cli`
- [ ] `make test-conformance` in an oracle-enabled environment
- [ ] Native and `wasm32-unknown-unknown` release builds
- [ ] Reference/provenance audit and an independent review of Cooper's boundary
      construction, strict integer shifts, and divisibility normalization
- [ ] Public documentation states the exact supported grammar and does not
      imply nonlinear or mixed-theory completeness.

Only after all commands pass from a clean checkout, all checkboxes are closed,
and the evidence ledger is complete may the active implementation goal be
marked complete.

## Correctness references

- D. C. Cooper, *Theorem Proving in Arithmetic without Multiplication*:
  <https://www.cs.cmu.edu/~emc/spring06/home1_files/Cooper.pdf>
- J. Ferrante and C. Rackoff, *A Decision Procedure for the First Order Theory
  of Real Addition with Order*:
  <https://janos.cs.technion.ac.il/COURSES/238900-13/Papers/ferranterackoffrealaddition.pdf>
- A. Chaieb and T. Nipkow, *Verifying and Reflecting Quantifier Elimination for
  Presburger Arithmetic*:
  <https://www.proof.cit.tum.de/~nipkow/pubs/lpar05.pdf>
- Isabelle AFP, executable verified *Linear Quantifier Elimination* theories:
  <https://isa-afp.org/entries/LinearQuantifierElim.html>
- W. Pugh, *The Omega Test* (a later optimization reference, not the semantic
  foundation for arbitrary quantified formulas):
  <https://doi.org/10.1145/125826.125848>

The papers define the algorithms. The Isabelle development is a specification
and independent executable cross-check. Wolfram is a surface-behavior oracle.
No external implementation is to be translated line for line.

## Evidence ledger

| Gate | Status | Commit(s) | Verification evidence |
| --- | --- | --- | --- |
| 0 | Complete | Working tree | 99 existing Reduce/Resolve tests passed; `make test-reduce` passed 12 acceptance plus 14 backend/core tests; fixed-seed 4,096-case property run passed. Full baseline reached 12,505 passes before two unrelated URL-import tests were denied network access. Oracle target correctly reports the absent `wolframscript`. |
| 1 | Complete | Working tree | Canonical exact arithmetic, binder-safe sparse affine/formula IR, transactional grammar lowering/rejection matrix, NNF/normalization, and deterministic emission implemented. The backend-inclusive suite passes all arithmetic, alpha-renaming, lowering, rejection, property, and emission tests. |
| 2 | In progress | Working tree | Exact dense QE and broad explicit-domain evaluator routing implemented. The broad-route audit passed 163/164 cases, exposing only parameter/target clause order; the correction passed its regression. Exact proportional-bound simplification, contradiction/tautology detection, and compositional `And`/`Or` branch subsumption now pass fixed-point/permutation tests, the 10,000-formula exact boundary-sampling oracle, and 500 closed formulas against Z3. A contractual 240-case curated public-interpreter matrix passes, including multiple eliminated variables and depth-three alternation; XOR surface conformance is mandatory. The permanent target includes all 99 legacy Reduce/Resolve tests. The 100-case `wolframscript` comparison remains open because that executable is absent in this environment. |
| 3 | In progress | Working tree | Exact integer normalization, GCD/extended-GCD/LCM/generalized-CRT primitives, and Cooper boundary/residue elimination are implemented. Core tests cover strict shifts, equality, unbounded congruences, inconsistent non-coprime congruences, universal duality, alternation, and parity projection. The fixed-seed 25,000-formula exhaustive oracle passes in 89.44 seconds after budgeting dense-only subsumption; 1,000 generated closed formulas pass against Z3 in 1.09 seconds. A contractual 320-case curated public-interpreter matrix passes. Every fully lowered explicit-integer `Reduce` request and explicit-integer `Resolve` formula route through the same engine. Exact output shaping rounds unary bounds, GCD-reduces congruences, and symbolically decomposes finite Boolean/residue formulas before materializing required Wolfram-compatible finite output; it is not used as the decision procedure. The 150-case `wolframscript` comparison remains externally unavailable. |
| 4 | Complete | Working tree | `Reduce` and explicit-domain `Resolve` share the same lowerer and theory engines. Finite integer `Solve` is a client of the shared exact IR/Presburger normalizer; 25 `Solve` domain and `ToRules` integration tests pass. The production SMT-RAT bridge, environment switches, SMT encoder/parser, subprocess runner, and bridge tests were removed, shrinking `reduce_backend.rs` from 1,988 to 192 pure-Rust facade/helper lines; repository search finds no SMT-RAT references in production or user docs. The post-removal self-contained Reduce gate passes 177/177 and all test targets compile. The `Reduce` CLI document passes 3/3 executable examples against the release binary, and the Rule54 finite-integer `Solve` script snapshot passes. Solver-free release builds pass for native (15m13s), the CLI binary (6m50s), and `wasm32-unknown-unknown` with Woxi's `wasm` feature (9m44s). |
| 5 | In progress | Working tree | The pure 2,500-line exact/IR/QE implementation is now the standalone `woxi-reduce` workspace crate; Woxi retains only syntax lowering, emission, and evaluator clients. A cold package check completes in 4.02s, the first package test build in 14.31s, a warm check in 0.92s, and the warm `test-reduce-core` path in 0.14s plus 0.50s for 31 tests, versus 2–3 minute monolithic test links. The new `make test-reduce` boundary passes 31/31 package tests and 157/157 Woxi adapter/interpreter tests; a dependency-only Woxi rebuild took 42.26s, while an adapter edit still requires the legacy Woxi link (3m07s in the final run). `cargo package` produced and independently rebuilt a 22.2 KiB compressed crate with only `num-bigint` and `num-traits`, and all-target Clippy passes with warnings denied. Both engines share structural result memoization with measured cache-hit tests, use deterministic domain-specific elimination cost heuristics, and retain exact simplification/subsumption properties. The six-category Criterion corpus reports: sparse 130 µs, dense Fourier–Motzkin 30–33 ms, Boolean branching 3.59–3.75 ms, 256-bit coefficients 31.9–33.1 µs, congruence LCM 195–200 µs, and alternating quantifiers 11.2–11.7 µs. The 100,000-case robustness run and CI smoke budgets remain open. |
| 6 | Not started | — | — |
