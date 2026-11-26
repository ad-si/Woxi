# Improve Woxi evaluator performance

Handoff notes for the remaining work on
<https://github.com/ad-si/Woxi/issues/2142> ("Speed and memory
consumption"). Read `CLAUDE.md` first; its rules apply to everything here.

## Goal

The issue's script builds a 135135-term sum by a recursive Dirac trace.
Timings on one machine (release build, n = 14):

| | Before | Now (commit `9e49a38`) | wolframscript (issue, other machine) |
|---|---|---|---|
| Trace generation | 30.4 s | 15.7–17.2 s | 3.4 s |
| `Cases` on 135135 terms | 0.87 s | 0.70 s | 0.30 s |
| Peak memory | 517 MB | 318 MB | 100 MB |

Target: within 2× of wolframscript on time and memory, without special
cases for this script. Every item below is a general evaluator cost that
this script happens to expose.

## The benchmark

```wl
$absTime = First @ AbsoluteTiming[
  Print["$Version = ", $Version];
  diracTr[] := 4;
  diracTr[a_, b_] := 4 g[a, b];
  diracTr[a_, rest___] := Module[{list = {rest}, n},
    Sum[(-1)^(m + 1) * Expand[g[a, list[[m]]] * diracTr @@ Delete[list, m]],
      {m, 1, Length[list]}]];
  n = 14;
  Print["AbsoluteTiming trace ", n, " generation ",
    First @ AbsoluteTiming[res[n] = diracTr @@ Array[x, n]; Length @ res[n]]];
  Print["AbsoluteTiming Cases on ", Length @ res[n], " terms :",
    AbsoluteTiming[Cases[res[n], _g, Infinity] // Union // Length]];
];
Print["MaxMemoryUsed[] = ", MaxMemoryUsed[]/1024.^2, " MB"];
Print["AbsoluteTiming after start: ", $absTime];
```

Expected output: 135135 terms and 91 distinct `g` for n = 14 (10395 and 66
for n = 12, 945 and 45 for n = 10). `benches/interpreter.rs` has
`dirac_trace_10/12/14`, and `tests/interpreter_tests/algebra.rs` has
`dirac_trace_recursion*` regression tests.

## Where the time goes now

A trace of n arguments calls itself n − 1 times on n − 2 arguments. For
n = 14 that is 1 call at 14 arguments, 13 at 12, 143 at 10, 1287 at 8,
9009 at 6, 45045 at 4 and 135135 leaf calls at 2. Every level produces the
same 135135 terms in total, but the low levels do it in tens of thousands
of tiny calls.

Cost added by each level, measured on a non-LTO optimized build with plain
symbols as arguments, each row normalized to about 30000 leaf terms:

| Level (arguments) | Calls | Added time |
|---|---|---|
| 2 | 30000 | 0.13 s |
| 4 | 10000 | 1.20 s |
| 6 | 2000 | 0.90 s |
| 8 | 286 | 0.16 s |
| 10 | 32 | 0.20 s |
| 12 | 3 | 0.55 s |

So roughly two thirds of the time is fixed per-call overhead at levels 4
and 6, where each `Expand` handles one to three terms. The per-term
algebra that used to dominate the top levels is now the smaller part.
With `x[1], x[2], …` as arguments (what the issue uses) everything is about
1.6× slower than with plain symbols, because arguments are compound.

Callgrind profile of the 4-argument call in a loop (inclusive, non-LTO
build, percentages overlap):

| Share | Where |
|---|---|
| 77% | `sum_ast_unevaluated_body` (the `Sum` and everything in it) |
| 17% | `Drop for Expr` |
| 13.5% | `expand_ast` |
| 11% | `apply_apply_ast` (the leaf call) |
| 9% | `plus_ast` |
| 7.5% | `substitute_index_variable` (one body copy per `Sum` iteration) |
| 6.3% | `times_ast` |
| 4.7% | `ExprList::clone` |
| 4.2% + 3.8% + 2.4% | `map_children`, `substitute_variables_impl`, `substitute_pattern_bindings` |
| 3.4% + 2.7% | `shield_nested_scopes`, `rename_symbol` (`Module`) |
| 3.5% + 3.5% | `dispatch_plotting`, `get_builtin_attributes` (generic call overhead) |

About half of a small call is copying, substituting into and dropping
expression trees.

## What is already done (do not redo)

- `Expand` over calls of undefined heads ranks its distinct calls once and
  assembles products and the sum in canonical order
  (`plain_kernel_sum` in `src/functions/polynomial_ast/poly_expand.rs`,
  `plain_factor_orders` in
  `src/functions/math_ast/canonical_polynomial.rs`).
- `Plus` over products of calls sorts by per-call ranks
  (`plain_call_term_order`); `-1 * sum` flips signs in place
  (`negated_plain_call_sum`); `Sum` moves its summands
  (`plain_call_sum_owned`).
- `canonical_cmp` has a direct path for integers, plain symbols and calls
  of plain symbols (`plain_canonical_cmp` in
  `src/functions/list_helpers_ast/sorting.rs`).
- A function's rules are snapshotted once per definitions change
  (`user_overloads` in `src/evaluator/dispatch/evaluate_functions.rs`).
- `take_symbol_values` / `restore_symbol_values` leave stores without an
  entry untouched, so a `Sum` index or `Block` variable no longer advances
  the global definitions epoch.
- `f[k] = value` marks a long value as evaluated, like `x = value`.
- `is_builtin_symbol` answers for lowercase-initial names without a lookup.

## Work items, in order of expected payoff

Gains are estimates from the profiles above, not measurements.

### 1. Make symbol and head names cheap to copy

`Expr::Identifier(String)` and `Expr::FunctionCall { name: String, args }`
(`src/syntax.rs`) allocate on every clone. Cloning `g[x[1], x[2]]` with
shared arguments still allocates a `String` for `"g"`; a term of six
factors costs seven allocations to copy, and every substitution pass
copies every name in a function body. `malloc`/`free` are 15–20% of every
profile taken.

Proposal: replace the two `String`s with a cheaply clonable name type
(an interned `Arc<str>` handle, or a small-string type that stores short
names inline). Keep `Deref<Target = str>`, `PartialEq<str>`, `Display` and
`From<&str>`/`From<String>` so that `name == "Plus"`, `name.as_str()` and
`match name.as_str()` keep compiling. With an interned handle, equality
and hashing can use the pointer.

This touches thousands of sites. Do it mechanically in stages: introduce
the type as an alias of `String`, convert construction sites to
`.into()`, then switch the alias. Expected gain: 15–25% across the board,
more for symbolic workloads.

### 2. Stop copying function bodies on every call

One call of `diracTr` with four arguments copies the rule body about five
times and then once more per `Sum` iteration:

1. `substitute_pattern_bindings` instantiates the body
   (`apply_user_overloads`).
2. `module_ast` (`src/evaluator/scoping.rs`) runs `shield_nested_scopes`,
   then `rename_symbol` once per local variable, then
   `restore_shadowed_names`.
3. `sum_ast` runs `expand_index_dependent_symbols`, then
   `substitute_index_variable` for every iteration
   (`src/functions/list_helpers_ast/summation.rs`).

Steps, cheapest first:

- Rename all `Module` locals in one traversal instead of one per variable,
  and fold the shielding pass into it.
- In `expand_index_dependent_symbols`, decide whether any symbol of the
  summand has a value that mentions an index before taking and restoring
  the index's definitions.
- Cache per rule what does not depend on the arguments (which locals a
  `Module` body has, whether nested scopes need shielding).
- Larger: evaluate bodies against a binding environment instead of
  substituting. This changes how held expressions and nested scopes see
  bound names, so it needs its own design; do the items above first.

Expected gain: 10–20% on small recursive functions.

### 3. Dispatch built-ins and undefined heads directly

`evaluate_function_call_ast_inner`
(`src/evaluator/dispatch/evaluate_functions.rs`) is about 12000 lines of
sequential `if name == "…"` checks and `dispatch_*` probes. `Times` and
`Plus` reach `dispatch_math_functions` only after some 25 earlier probes,
and a call of an undefined head such as `g[a, b]` walks the whole function
before falling through. Measured in release: 0.66 µs for `g[x1, x2]`,
1.5 µs for `g[x[1], x[2]]`; one iteration at level 4 does about twelve
such walks. Non-LTO builds make this three to four times worse.

Two steps:

- **Undefined heads.** After the user-rule lookup, a head that is not a
  built-in, has no context mark, no rules, no attributes and no own value
  can return `name[args]` at once. The risk is the handful of checks in
  the chain that look at the arguments rather than the name
  (`try_around_unary`, `numeric_array_list_operation`,
  `structured_matrix_call`, `function_of_series`, dataset passthrough).
  Validate dynamically: behind a test-only switch, run the full chain
  anyway and assert it returns the same unevaluated call, then run the
  whole suite with the switch on.
- **Built-ins.** Route by name through a table (name → dispatcher) built
  once, instead of probing every dispatcher in order. Dispatchers return
  `None` to fall through, so keep the chain as the fallback and record
  for each name the first dispatcher that can answer.

Expected gain: about 10% on this script, more on code that calls many
undefined functions.

### 4. Evaluation stamps that survive unrelated assignments

`src/epoch.rs` keeps two epochs. A large shared list is stamped with the
epoch at which it was found to evaluate to itself, and
`evaluate_reusing_fixed_point` (`src/evaluator/core_eval.rs`) returns it
unchanged while the stamp is current. Any own-value assignment to an
unprotected symbol advances `EPOCH`, and only expressions that mention
nothing but protected symbols get the more durable `PROTECTED_EPOCH`
stamp. Every `Module` call assigns a fresh local, so a sum in `g` and `x`
loses its stamp at once. Consequences measured here: reading a stored
135135-term sum after any assignment costs about 1 s, and `Expand`
re-evaluates every kernel on every call (two to fifteen full evaluations
per small `Expand`).

Proposal: a third stamp for expressions whose unprotected symbols are all
inert (no own value, no rules, no attributes). Keep a set of the symbol
names such stamps depend on; `bump_epoch_for_own_value(name)` advances
this epoch only when `name` is in the set (then clear the set). Every
other kind of change already advances the global epoch. Skip subtrees
that carry a current stamp when collecting the names, so stamping stays
cheap. Then:

- stamp results the evaluator just produced, not only stored values;
- share large argument lists before a built-in receives them
  (`evaluate_function_call_owned`), so built-ins that keep an argument
  copy a pointer;
- let `Expand` skip kernels whose stamp is current.

Soundness to check: rules with conditions that read globals (`f[x_] /;
flag := …`) are why only symbols without any rules may count as inert.
`tests/interpreter_tests/evaluation_reuse.rs` covers the existing stamps;
extend it for each new case.

Expected gain: large for programs that keep big symbolic values in
variables; 5–10% on this script.

### 5. `Expand` internals

In `src/functions/polynomial_ast/poly_expand.rs` and `mfactor.rs`:

- `Mono` is a dense `Vec<u32>` with one slot per generator. At the top
  level there are up to 79 generators, so every term allocates and
  compares about 316 bytes. Use a sparse or bitset representation for the
  multilinear case.
- Coefficients are `Q` with two `BigInt`s; keep machine integers unboxed
  until they overflow.
- The product of one term and an expanded sum needs no polynomial at all:
  the terms stay distinct and in order, and only the position of the new
  factor in each product has to be found. Generalize this as "distribute
  without collecting when one factor is a monomial".
- `collect_generators` and `to_poly` each look up every factor; merge the
  two passes.

Expected gain: 5–10% overall, mostly at the top levels, plus lower peak
memory.

### 6. Memory

- `Module` never removes its `name$n` bindings from `ENV` ("needs no
  cleanup"), so every call leaks its initial values. Remove them when the
  result does not mention the symbol and nothing defined during the body
  can refer to it.
- Terms are copied when a sum is rebuilt (`plain_call_sum`, negation).
  Sharing each term's argument list, or building from owned terms as
  `plain_call_sum_owned` does, avoids holding two copies of a long sum.
- Items 1 and 5 reduce the size of every term.

## How to measure

There is no `perf` in the cloud container; use `valgrind --tool=callgrind`
(instruction counts are deterministic) and `--tool=massif` for the heap.

- A release build takes about 29 minutes and 7 GB (fat LTO, one codegen
  unit). Adding debug info to it runs out of memory. Use it only for final
  numbers.
- For iteration, build a separate profile from the command line:

  ```sh
  cargo build --bin woxi --profile fast \
    --config 'profile.fast.inherits="release"' \
    --config 'profile.fast.lto="off"' \
    --config 'profile.fast.codegen-units=256' \
    --config 'profile.fast.incremental=true' \
    --config 'profile.fast.debug="line-tables-only"'
  ```

  It rebuilds in one to two minutes. With `codegen-units=16` and no
  `incremental` it takes seven minutes and runs within 15–25% of release.
- Non-LTO builds exaggerate the dispatch chain (item 3) three to four
  times. Confirm any conclusion about it on a real release build.
- Never run two builds at once, and run tests with
  `CARGO_BUILD_JOBS=1 cargo nextest run --no-fail-fast …`: two compiles of
  the crate exceed the container's memory and one gets killed.
- A variable holding a big value is re-evaluated on its first read after
  any assignment (item 4). In micro-benchmarks read the variable once
  before timing, and time loops, or the numbers include that cost.

Level-by-level benchmark used for the table above:

```wl
Print["L2: ", First@AbsoluteTiming[Do[diracTr[x1, x2], {30000}]]];
Print["L4: ", First@AbsoluteTiming[Do[diracTr[x1, x2, x3, x4], {10000}]]];
Print["L6: ", First@AbsoluteTiming[Do[diracTr[x1, x2, x3, x4, x5, x6], {2000}]]];
Print["L8: ", First@AbsoluteTiming[Do[diracTr @@ {x1, x2, x3, x4, x5, x6, x7, x8}, {286}]]];
```

## How to stay correct

- The output of every step must stay identical. Fast paths in
  `canonical_polynomial.rs`, `poly_expand.rs` and `sorting.rs` each have a
  test-only switch (`GENERIC_ONLY`, `GENERAL_ONLY`) and randomized tests
  that compare them with the general code. Add the same for every new
  fast path, and never implement a construct twice: delegate to the
  shared helper.
- A fast path that bypasses `evaluate_function_call_ast` must check
  `head_has_user_rules` (`src/evaluator/core_eval.rs`) for the heads it
  skips; upvalues are stored under the outer head in `FUNC_DEFS`.
- `wolframscript` is not available in the container. Compare against a
  saved binary of the previous commit and against the existing snapshot
  tests.
- Run `make test`, `make format` and `make lint` before committing, and
  add a `changelog.md` entry.
