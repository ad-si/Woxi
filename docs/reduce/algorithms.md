# Exact linear `Reduce`: algorithm specification

This document is the implementation contract for Woxi's self-contained linear
quantifier-elimination engine. It covers ordered linear arithmetic over the
reals and rationals and first-order Presburger arithmetic over the integers.
The precise supported surface grammar and completion gates are recorded in
[`REDUCE_ROADMAP.md`](../../REDUCE_ROADMAP.md).

The engine is a decision procedure, not a collection of expression templates.
Every accepted input must be lowered completely, decided with exact arithmetic,
and emitted without calling another solver. Failure to lower an expression is
an explicit out-of-scope result; it must never silently discard an atom or
coerce an exact coefficient through a machine number.

## Semantic boundary

`Reduce[formula, vars, domain]` keeps `vars` as the coordinates described by
the returned solution set. Explicitly quantified variables inside `formula`
are eliminated. Symbols that are neither target variables nor bound variables
remain parameters. `Resolve` eliminates its explicit binders and returns a
Boolean formula in the remaining free parameters, or `True`/`False` when the
input is closed.

For the scoped linear fragment, rational and real quantifier elimination share
the dense ordered-field algorithm. Their surface emission differs: a rational
solution may require domain-membership conditions that are unnecessary over
the reals. Integer arithmetic uses a separate Presburger procedure because
strict bounds, gaps, and congruence classes are semantically significant.

## Module ownership

The pure solver is the standalone `woxi-reduce` workspace crate. It depends
only on `std`, `num-bigint`, and `num-traits`; it has no dependency on Woxi's
parser, expression tree, evaluator, CLI, or runtime. `reduce_backend.rs` is the
small evaluator-integration facade. It selects a theory engine only after
transactional lowering accepts the whole request and contains no SMT encoding,
subprocess management, or external-backend configuration.

Theory logic in `crates/woxi-reduce/src/` is split as follows:

| Module | Sole responsibility |
| --- | --- |
| `exact.rs` | Normalized `BigInt` rational arithmetic and integer GCD/LCM/extended-GCD helpers |
| `affine.rs` | Sparse affine terms with deterministic variable ordering |
| `formula.rs` | Relations, divisibility atoms, Boolean formulas, binders, free-variable analysis, NNF, stable normalization, memoized sharing, and budgeted exact subsumption |
| `rational_qe.rs` | Dense linear quantifier elimination with deterministic growth-based variable ordering |
| `presburger.rs` | Cooper elimination, integer-specific normalization, congruence simplification, and deterministic finite-instantiation cost ordering |

Woxi-specific adapter logic remains in
`src/functions/polynomial_ast/reduce_backend/`:

| Module | Sole responsibility |
| --- | --- |
| `lower.rs` | All-or-nothing lowering from Woxi `Expr` and capture-avoiding binder renaming |
| `emit.rs` | Deterministic Woxi `Expr` construction, integer result shaping, and surface canonicalization |
| `integer_solve.rs` | Finite-model `Solve[..., Integers]` client using the shared lowerer, IR, and Presburger normalizer |

`reduce.rs` remains the evaluator entry point and owner of existing nonlinear,
complex, and transcendental fallbacks. `resolve_ast.rs` delegates accepted
linear formulas to the same facade rather than maintaining a second decision
procedure. The finite integer `Solve` client shares the exact lowerer, formula
IR, and integer normalizer; it enumerates only after proving the requested
model set finite and never substitutes a search cap for quantifier
elimination.

For solver development, `cargo test -p woxi-reduce` compiles and links only
the small theory crate. `make test-reduce` first runs that package suite and
then the explicitly scoped Woxi adapter/interpreter targets. Native release,
WebAssembly, CLI, and full-repository checks remain milestone gates rather than
part of the inner algorithm-edit loop.

## Exact representation

### Rationals

A rational is `(numerator: BigInt, denominator: BigInt)` with these invariants:

1. `denominator > 0`.
2. `gcd(abs(numerator), denominator) == 1`.
3. Zero has the unique representation `0/1`.
4. Construction with denominator zero fails before a formula is accepted.

All arithmetic returns another normalized value. Conversions to fixed-width
integers are permitted only after a checked proof that the value is an index or
small allocation size; never for a mathematical coefficient or bound.

### Affine terms

An affine term is:

```text
constant + coefficient_1 * variable_1 + ... + coefficient_n * variable_n
```

represented by a constant and a sorted map from variable identity to nonzero
coefficient. Variable identity must be binder-safe; textual names alone are
not sufficient internally. Addition, negation, and multiplication or division
by an exact constant preserve affinity. Multiplication of two nonconstant terms,
non-integer powers other than syntactic division by a constant, inexact
numbers, and variable denominators reject the entire lowering request.

### Formula IR

The shared formula tree contains:

```text
True | False
Relation(Equal | NotEqual | Less | LessEqual | Greater | GreaterEqual,
         affine, affine)
Divides(positive BigInt modulus, integer affine)
NotDivides(positive BigInt modulus, integer affine)
And(children) | Or(children)
Exists(bound variables, body) | ForAll(bound variables, body)
```

`Xor`, implication, equivalence, chained comparisons, `Mod[t,m] == r`, and
alternative function-call syntax are surface constructs lowered into this
core. The normalizer may temporarily retain equality or disequality, but each
theory pass must document the atomic forms it accepts.

## Lowering and normalization pipeline

Lowering is transactional: it returns either a complete formula and domain or
an out-of-scope classification. There is no partial formula result.

```text
lower_reduce_request(expr, targets, domain):
    validate target variables and domain
    allocate fresh internal identities for every binder
    lower every exact affine term
    lower every relation and Boolean connective
    desugar chained comparisons and congruence surface forms
    verify every ordered atom is meaningful in the selected domain
    return normalized binder-safe formula
```

The shared normalization sequence is:

1. Alpha-rename binders so no bound identity can capture a free parameter.
2. Desugar `Xor`, implication, equivalence, and chained comparisons.
3. Push negation to atoms (NNF), swapping quantifiers under negation.
4. Normalize rational signs/GCDs and remove zero affine coefficients.
5. Move relation right-hand sides to zero.
6. Evaluate constant atoms exactly.
7. Flatten nested `And`/`Or`, remove identities, short-circuit annihilators,
   deduplicate children, and sort them deterministically.

Every simplification is an equivalence rewrite. Rewrites that are valid only
over integers or only over dense orders belong to their theory module rather
than the shared normalizer.

Required normalization properties:

- Idempotence: `normalize(normalize(f)) == normalize(f)`.
- Alpha invariance: consistently renaming a binder cannot change emitted free
  variables or truth.
- Permutation invariance: reordering associative input children produces the
  same normalized tree.
- Free-variable preservation, except for variables deliberately eliminated.
- NNF contains no `Not` node above a compound formula.

## Dense linear arithmetic over rationals and reals

The baseline implementation uses exact equality substitution and
Fourier-Motzkin projection. The latter is sufficient because both domains are
dense, unbounded ordered fields for formulas with rational coefficients.

### Atomic preparation

For the variable `x`, normalize every comparison to one of:

```text
a*x + t <  0
a*x + t <= 0
a*x + t == 0
a*x + t != 0
```

where `t` does not contain `x`. `>` and `>=` are reversed during normalization.
If `a == 0`, the atom is independent of `x` and is retained unchanged.

For an equality with `a != 0`, solve `x == -t/a`, substitute that exact affine
term throughout the conjunction, and eliminate `x`. A disequality can be split
as `(a*x+t < 0) || (a*x+t > 0)` when the surrounding Boolean branch requires
projection.

### Projecting a conjunction

Each ordered atom involving `x` becomes a lower or upper bound with a strict
flag. For example, after division by the sign-aware coefficient:

```text
x >  lower       x >= lower
x <  upper       x <= upper
```

An existential witness exists exactly when every lower/upper pair is
compatible. For each pair `(lower, lower_strict)` and
`(upper, upper_strict)`, emit:

```text
lower <  upper   if lower_strict || upper_strict
lower <= upper   otherwise
```

If there are no lower bounds or no upper bounds, the ordered field's
unboundedness supplies a witness; only atoms independent of `x` remain. This
also covers a variable absent from the conjunction.

```text
project_conjunction(conjunction, x):
    if an equality in x exists:
        choose a deterministic pivot
        substitute its solution into all other atoms
        return normalize(remaining atoms)
    split disequalities involving x into ordered branches
    classify x-atoms as lower bounds or upper bounds
    result = atoms independent of x
    for lower in lower_bounds:
        for upper in upper_bounds:
            result += compatible(lower, upper)
    return normalize(result)
```

### Boolean formulas and quantifiers

The correctness-first path converts the NNF body of an existential quantifier
to disjunctive normal form, projects each conjunction, then disjoins and
simplifies the results. Implementations may later avoid materializing full DNF
by recursive projection or sharing, but must be property-tested against the
baseline semantics.

```text
eliminate_exists_dense(body, x):
    branches = to_dnf(normalize_nnf(body))
    return normalize(OR(project_conjunction(branch, x)
                        for branch in branches))

eliminate_forall_dense(body, x):
    return normalize(NOT(eliminate_exists_dense(NOT(body), x)))
```

Eliminate an innermost binder at a time. Binder lists use their semantic order
but adjacent binders of the same kind may share preparation work. After every
step, assert in debug/tests that the eliminated identity is absent.

### Dense-QE invariants

- Projection is logically equivalent to existentially quantifying the input.
- The eliminated variable is absent.
- Strictness is lost only when both paired bounds are non-strict.
- No floating-point comparison determines a pivot or bound.
- A target-independent parameter condition is retained.
- Universal elimination agrees with `!Exists[x, !body]`.

## Presburger arithmetic

Presburger arithmetic consists of integer affine terms, order, equality,
Boolean operations, divisibility/congruence, and quantifiers. Cooper's method
is the semantic baseline. Optimized conjunctive solvers such as Omega may be
added later only behind equivalence tests against that baseline.

### Integer atom normalization

1. Clear rational denominators using their positive LCM. This is valid only
   when the multiplication direction and relation are preserved.
2. Convert strict inequalities exactly:

   ```text
   t < 0  <=> t <= -1
   t > 0  <=> -t <= -1
   ```

3. Convert equality to two non-strict inequalities when Cooper preparation
   requires it; convert disequality to the corresponding disjunction.
4. Store divisibility as `d | t` with `d > 0`. Reduce common GCD factors only
   under the exact divisibility equivalence conditions.
5. Canonicalize residues with Euclidean modulo, including negative inputs.

The following helpers are independently unit tested before Cooper elimination:

- Euclidean GCD and positive LCM over `BigInt`.
- Extended GCD with a checked Bézout identity.
- Solvability of `a*x == b (mod m)` via `gcd(a,m) | b`.
- Generalized CRT for non-coprime moduli, including inconsistency detection.

### Coefficient normalization

For an eliminated variable `x`, let `L` be the positive LCM of the nonzero
absolute coefficients of `x`. Multiply each comparison by the positive factor
needed to make its `x` coefficient `-L`, `0`, or `L`. For divisibility atoms,
multiply both the modulus and divisible term by that factor, preserving
equivalence. Introduce `y = L*x`, substitute unit occurrences of `y`, and add
the constraint `L | y`. Existentially quantifying `x` is then equivalent to
existentially quantifying `y` under that divisibility constraint.

This transformation must be implemented as a separately tested equivalence
step. In particular, multiplying only the divisible term without multiplying
its modulus is unsound.

### Cooper elimination structure

After coefficient normalization, all occurrences of the eliminated variable
have unit coefficient. The truth of divisibility atoms is periodic in that
variable. Let `delta` be the positive LCM of their moduli (or one when none are
present). Ordered atoms define finitely many boundary terms; sufficiently far
in the negative direction, their truth stabilizes while divisibility retains
period `delta`.

Cooper's theorem reduces the existential formula to a finite disjunction of:

1. one instance for each residue class in the stabilized negative-infinity
   tail; and
2. instances immediately after every relevant lower boundary, for every
   residue class modulo `delta`.

```text
eliminate_exists_presburger(body, x):
    f = integer_nnf_and_atom_normalization(body)
    f = normalize_x_coefficients(f, x)  // includes L | replacement variable
    delta = lcm_of_x_divisibility_moduli(f)
    (tail, boundaries) = cooper_negative_infinity_and_boundaries(f, x)

    result = []
    for residue in canonical_residues(delta):
        result += instantiate_periodic_tail(tail, x, residue)
    for boundary in boundaries:
        for residue in canonical_residues(delta):
            result += instantiate_after_boundary(f, x, boundary, residue)

    return normalize(OR(result))
```

The exact boundary polarity, offset convention, and tail construction must be
implemented directly against Cooper's theorem and the verified Isabelle
formalization, with theorem examples retained beside the code. The pseudocode
above deliberately names those operations rather than hiding them in ad-hoc
substitution arithmetic. The required postconditions are:

- the disjunction is finite and contains no eliminated variable;
- every residue class modulo `delta` is covered exactly under the chosen
  canonical convention;
- formulas with no lower boundary are decided by the periodic tail rather than
  bounded search;
- negated divisibility atoms remain correct;
- negative coefficients, terms, moduli at the surface, and residues have one
  normalized meaning;
- universal quantification is checked through logical duality.

## Simplification and canonical emission

The decision procedures produce formulas, not presentation strings. Emission
first applies proven simplifications:

- Boolean identities, annihilators, complements, and deduplication.
- Constant relation/divisibility evaluation.
- Duplicate or dominated affine bounds in conjunctions.
- Tautological or contradictory interval pairs.
- Congruence merging by generalized CRT where logically valid.
- Deterministic sorting by structural keys.

The Woxi emitter then:

1. Orients simple solved relations with a target variable on the left.
2. Combines compatible one-variable lower/upper bounds into `Inequality`.
3. Emits exact rational syntax with a positive denominator.
4. Emits integer or rational `Element` conditions where Wolfram's surface
   semantics requires them.
5. Uses a stable order for parameters, target variables, conjunctions, and
   branches.

Semantic correctness takes priority over textual resemblance, but the checked
surface corpus must match `wolframscript` exactly under `AGENTS.md`. A surface
canonicalization change therefore needs both semantic property tests and a
Wolfram regression.

## Verification strategy

No single oracle is sufficient. Evidence is layered:

### Unit and regression tests

Every arithmetic helper, normalization rule, elimination boundary case, and
reported bug receives a named test. Values beyond 128 bits are mandatory.

### Metamorphic properties

- Normalization is idempotent.
- Reordering `And`/`Or` operands does not change canonical output.
- Alpha-renaming bound variables does not change semantics.
- Eliminated variables never occur in the result.
- `ForAll[x,p]` agrees with `!Exists[x,!p]`.
- Adding a fresh unused quantified variable does not change a formula.
- Substitution of exact valuations agrees before and after elimination.
- Every simplifier rewrite preserves truth.

### Exhaustive small-model tests

Generated formulas whose variables have explicit small integer bounds are
evaluated exhaustively with a simple test-only interpreter. This oracle must
not call the production eliminator. It is especially useful for strict shifts,
negated congruences, Boolean nesting, and alternating quantifiers.

For dense arithmetic, generated quantifier-free results are checked at exact
rational sample points including atom boundaries and points immediately
between adjacent rational boundaries. Closed generated formulas are also sent
to an independent development oracle.

### Differential development oracles

- The Isabelle AFP executable formalization is the primary independent
  semantic cross-check for generated linear QE cases.
- A separate SMT solver may be used in development to increase generated-case
  volume, but it is not authoritative and is never a runtime dependency.
- `wolframscript` is the oracle for Woxi surface form and public behavior.

Oracle cases are deterministic and versioned. Every mismatch is classified as
a Woxi semantic bug, Woxi presentation bug, unsupported case, or oracle issue;
unexplained mismatches block the relevant gate.

## Runtime and failure policy

Presburger quantifier elimination has unavoidable worst-case explosion. The
engine may expose progress metrics and documented resource limits, but a
silent bounded search, guessed result, or fallback to another executable is
never acceptable. Resource exhaustion must be a distinct evaluator error and
must not be reported as `True`, `False`, or a partial solution set.

Deterministic elimination ordering and structural sharing are performance
features. They may change formula size but cannot change truth or free-variable
sets. Each heuristic is introduced only after the correctness baseline passes.

## References and provenance

The implementation is derived from algorithm descriptions and independently
verified specifications, not transliterated from another solver:

- D. C. Cooper, *Theorem Proving in Arithmetic without Multiplication*:
  <https://www.cs.cmu.edu/~emc/spring06/home1_files/Cooper.pdf>
- J. Ferrante and C. Rackoff, *A Decision Procedure for the First Order Theory
  of Real Addition with Order*:
  <https://janos.cs.technion.ac.il/COURSES/238900-13/Papers/ferranterackoffrealaddition.pdf>
- A. Chaieb and T. Nipkow, *Verifying and Reflecting Quantifier Elimination for
  Presburger Arithmetic*:
  <https://www.proof.cit.tum.de/~nipkow/pubs/lpar05.pdf>
- Isabelle AFP, *Linear Quantifier Elimination*:
  <https://isa-afp.org/entries/LinearQuantifierElim.html>
- W. Pugh, *The Omega Test*:
  <https://doi.org/10.1145/125826.125848>

Reference-to-module mapping:

| Result | Implementation location | Required cross-check |
| --- | --- | --- |
| Dense-order projection | `rational_qe.rs` | Ferrante-Rackoff/AFP examples plus exact generated valuations |
| NNF and Boolean QE scaffolding | `normalize.rs`, `formula.rs` | AFP structure plus metamorphic duality tests |
| Cooper finite boundary/residue theorem | `presburger.rs` | Original paper and Chaieb-Nipkow/AFP executable cases |
| GCD/CRT normalization | `exact.rs`, `presburger.rs` | Bézout identities and exhaustive small moduli |
| Omega-style fast paths, if added | a later optimization module | Must agree with Cooper baseline on every accepted input |

Any nontrivial departure from these references is documented beside the code
with its equivalence argument and tests.
