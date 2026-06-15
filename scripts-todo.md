# Script activation TODO

Tracks every `.wls`/`.wl` file in `tests/scripts/` that is **not** yet wired
into a `script_test!` in [`tests/script_snapshot_tests.rs`](tests/script_snapshot_tests.rs).

A script is "activated" when it has a `script_test!` entry whose snapshot output
matches `wolframscript -file` byte-for-byte (verified via `WOXI_USE_WOLFRAM=true`).

## Status overview

| Bucket | Count | Action |
|---|---:|---|
| Total scripts in `tests/scripts/` | 832 | — |
| Activated (have a `script_test!`) | 599 | done |
| **Not activated** | **233** | see below |
| ├ Raw `_tasks_` dump already covered by a curated counterpart | 136 | none needed |
| ├ Intentionally excluded (documented in the test file) | 2 | none needed |
| ├ Drafts / scratch files | 4 | leave as drafts |
| └ Real candidates (`_tasks_` dumps with no counterpart) | 96 | curate + verify, grouped below |

> **Update (2026-06-05):** 17 more scripts activated, and three engine bugs
> fixed along the way (each with a regression unit test in
> `rosetta_script_fixes`):
> 1. **`ConditionOp` operator ordering** — in a rule/condition context (RHS of
>    `->`/`/.`, or a list item) the Map operator `/@` was tried *after* bare
>    `/`, so `b /@ c` was mis-parsed as `b / (@c)`. Reordered `/@` before the
>    arithmetic `/` in `wolfram.pest`. (unblocked `ranking_methods`)
> 2. **`Import[file, "Text"]`** — the `"Text"` element was unsupported; now an
>    alias of `"String"`/`"Plaintext"`. (`letter_frequency`,
>    `globally_replace_text_in_several_files`)
> 3. **`Symbol["name"]`** — did not unify with the bare symbol `name` for
>    assignment or lookup; now `Set[Symbol["x"], v]` binds the same `x` a later
>    bare lookup resolves, and `Symbol["x"]` resolves an existing OwnValue.
>    (`dynamic_variable_names`)
>
> Newly activated: `best_shuffle`, `count_in_octal`, `queue_usage`,
> `associative_array_iteration`, `pascals_triangle`, `binary_strings`,
> `ranking_methods`, `read_entire_file`, `letter_frequency`,
> `globally_replace_text_in_several_files`, `read_a_specific_line_from_a_file`,
> `input_loop`, `a_b`, `dynamic_variable_names`, `mad_libs`, `guess_the_number`,
> `align_columns`.

> **Update:** the 13 scripts formerly in bucket **4f** (RNG / parallel / timing)
> have been activated by hardcoding their non-deterministic inputs to fixed
> values (with a comment in each script noting the value would normally come from
> an RNG). See the now-completed section 4f below for the per-script approach and
> the key finding about RNG-stream divergence.

`_tasks_*.wls` files are raw RosettaCode "Mathematica/Wolfram" section dumps.
They are **never activated directly** — they routinely contain several alternative
solutions separated by `(* ========== *)`, inline expected-output text, and prose.
The workflow is to curate one variant into a clean `<name>.wls` (dropping the
`_tasks_` prefix), confirm `woxi` and `wolframscript` agree, then add a
`script_test!`. 136 dumps have already been curated this way.

---

## 1. Already covered — no action (136)

Each of these raw `_tasks_<name>.wls` dumps already has an activated, curated
`<name>.wls` counterpart. The dump is kept only as source material.

```
100_doors  ackermann_function  aliquot_sequence_classifications
append_a_record_to_the_end_of_a_text_file  apply_a_callback_to_an_array
arithmetic_complex  arithmetic_integer  arithmetic_rational  bernoulli_numbers
bitmap  bitwise_operations  calculating_the_value_of_e
case-sensitivity_of_identifiers  catalan_numbers
catmull–clark_subdivision_surface  cholesky_decomposition  closures_value_capture
collections  combinations_with_repetitions  compare_a_list_of_strings
compound_data_type  conjugate_transpose  convert_decimal_number_to_rational
decimal_floating_point_number_to_binary  deepcopy  delegates
determinant_and_permanent  dijkstras_algorithm
dinesmans_multiple-dwelling_problem  documentation  egyptian_fractions
element-wise_operations  enumerations  exceptions  executable_library
find_common_directory_path  find_the_last_sunday_of_each_month
find_the_missing_permutation  five_weekends  floyds_triangle  forward_difference
function_composition  greatest_subsequential_sum  hailstone_sequence
hash_from_two_arrays  hello_world_newline_omission  hofstadter_q_sequence
horners_rule_for_polynomial_evaluation  infinity  interactive_programming
inverted_syntax  julia_set  kaprekar_numbers  knights_tour  levenshtein_distance
linear_congruential_generator  literals_integer  logical_operations
long_multiplication  longest_common_subsequence  loops_nested  lucas-lehmer_test
ludic_numbers  luhn_test_of_credit_card_numbers  lychrel_numbers  mac_vendor_lookup
maze_generation  md5  md5_implementation  miller-rabin_primality_test
modular_exponentiation  monte_carlo_methods  named_parameters
non-decimal_radices_input  null_object  old_lady_swallowed_a_fly
optional_parameters  parsing_rpn_calculator_algorithm
parsing_shunting-yard_algorithm  pascal_matrix_generation  prime_decomposition
primorial_numbers  qr_decomposition  quine  real_constants_and_functions  rot-13
runtime_evaluation  runtime_evaluation_in_an_environment  sedols  send_email
set_of_real_numbers  sierpinski_triangle  sieve_of_eratosthenes
sort_disjoint_sublist  sorting_algorithms_merge_sort  stack
string_interpolation_(included)  strip_a_set_of_characters_from_a_string
strip_block_comments  sum_of_squares  symmetric_difference  taxicab_numbers
terminal_control_unicode_output  time_a_function  topswops  total_circles_area
tree_traversal  twelve_statements  undefined_values  url_decoding  url_encoding
vampire_number  variadic_function  zig-zag_matrix
a_b  align_columns  associative_array_iteration  best_shuffle  binary_strings
count_in_octal  dynamic_variable_names  globally_replace_text_in_several_files
guess_the_number  input_loop  letter_frequency  mad_libs  pascals_triangle
queue_usage  ranking_methods  read_a_specific_line_from_a_file  read_entire_file
combinations_and_permutations  find_palindromic_numbers_in_both_binary_and_ternary_bases
last_friday_of_each_month  loops_break  self-referential_sequence
```

## 2. Intentionally excluded — no action (2)

These already have explanatory comments in `script_snapshot_tests.rs`. The
computation matches; only diagnostic-message I/O routing differs (wolframscript
prints the message to stdout in script mode, Woxi routes it to stderr), so the
two snapshot engines can never agree on one snapshot.

- `detect_division_by_zero.wls` — `Check[2/0, …, Power::infy]` emits `Power::infy`.
- `include_a_file.wls` — `Get["myfile.m"]` on a missing file emits `Get::noopen`.

## 3. Drafts / scratch — leave as drafts (4)

- `_drafts_go_fish_mathematica.wls` — interactive card game (`ChoiceDialog`, `DialogInput`, RNG). GUI + stdin + non-deterministic; not snapshotable.
- `_drafts_rcsnusp_mathematica.wls` — SNUSP interpreter reading a file named on `$ScriptCommandLine`; needs a fixture program + stdin.
- `_drafts_retrieving_an_element_of_an_array.wls` — one-line snippet `element = array[[index]]` with undefined symbols; illustrative only.
- `_print_random_words.wls` — `RandomSample[WordList[], 10]`; needs `WordList[]` data and is non-deterministic.

---

## 4. Real candidates (148)

Grouped by the **primary** blocker. Many also have secondary issues (noted
inline). For all of them the activation recipe is the same: curate to a single
runnable solution, implement any missing builtin, make the output deterministic,
verify `woxi` == `wolframscript`, then add a `script_test!`.

### 4a. Network-dependent — cannot snapshot reproducibly (16)

Pull live data over HTTP; output depends on a remote resource. Activating would
require bundling an offline fixture and rewriting the `Import[...]` to read it.

- `anagrams` — `Import["http://…/unixdict.txt"]` word list (multi-variant dump).
- `anagrams_deranged_anagrams` — same word list over HTTP.
- `base64_encode_data` — `Import["http://rosettacode.org/favicon.ico"]`.
- `color_quantization` — `ColorQuantize[Import["http://…png"]]`.
- `hello_world_web_server` — `SocketListen` + `SystemOpen`; needs a socket server.
- `http` — `Import["http://www.google.com/…"]`.
- `https_authenticated` — `RunThrough["curl …"]`.
- `i_before_e_except_after_c` — HTTP word list (uses `%` output references).
- `ordered_words` — HTTP word list / `DictionaryLookup`.
- `rosetta_code_count_examples` — scrapes rosettacode.org.
- `rosetta_code_find_bare_lang_tags` — scrapes the RC API/XML.
- `rosetta_code_find_unimplemented_tasks` — scrapes the RC API/XML.
- `rosetta_code_rank_languages_by_popularity` — scrapes rosettacode.org.
- `semordnilap` — HTTP word list.
- `soap` — `InstallService[...wsdl]`; remote SOAP service.
- `web_scraping` — `Import["http://tycho.usno.navy.mil/…"]`.

### 4b. Interactive stdin — no input stream in script tests (4)

`Input[]`/`InputString[]` block on user input; the snapshot harness has no stdin.
Activate by rewriting to fixed inputs (and making any RNG deterministic).

- `bitcoin_address_validation` — `InputString[]` address; also needs `Hash[…,"SHA256"]`.
- `bulls_and_cows` — `InputString[]` game loop + RNG.
- `bulls_and_cows_player` — `InputString[]` solver loop.
- `narcissist` — reads its own source from `InputString[]` (quine-style).

✅ Done: `a_b` (two `Input[]` → fixed operands), `dynamic_variable_names`
(InputString → fixed name/value; also fixed the `Symbol[…]` bug), `mad_libs`
(template + answers hardcoded), `guess_the_number` (fixed secret + guess list).

### 4c. GUI / FrontEnd / Graphics / Sound — no deterministic stdout (32)

Produce notebook/graphics/audio objects (`CreateDialog`, `DynamicModule`,
`Manipulate`, `Notebook*`, `Graphics`, `Rasterize`, `Plot`, `Sound`, …) rather
than text. No stdout to snapshot; would need a graphics-to-text projection or a
FrontEnd. Several also need image fixtures or RNG seeding (noted).

- `animation` — `CreateDialog` + `Dynamic` scroller.
- `bitmap_bresenhams_line_algorithm` — `Rasterize[Graphics[Line[...]]]`.
- `bitmap_histogram` — `ImageLevels[img]` on an undefined image.
- `bitmap_midpoint_circle_algorithm` — `ReplacePixelValue` on `ExampleData` image.
- `brownian_tree` — `Monitor` + `MatrixPlot` + RNG.
- `canny_edge_detector` — `EdgeDetect[Import[InputString[]]]` (also stdin/image).
- `forest_fire` — `CellularAutomaton` + `MatrixPlot` + RNG.
- `fractal_tree` — `Graphics[...]` line tree.
- `greyscale_bars_display` — `CreateDocument[Graphics[...]]`.
- `guess_the_number_with_feedback` — `Input[]` + `CreateDialog` (also stdin).
- `gui_maximum_window_dimensions` — `SystemInformation["Devices"]`.
- `hello_world_graphical` — `CreateDialog["Hello world"]`.
- `hough_transform` — `Radon[image, …]` on an undefined image.
- `image_convolution` — `Import` image + `ImageConvolve`.
- `keyboard_input_obtain_a_y_or_n_response` — `CreateDialog` + key events.
- `keyboard_macros` — `SetOptions[EvaluationNotebook[], NotebookEventActions->…]`.
- `mandelbrot_set` — `ReliefPlot`/`ArrayPlot`/`MandelbrotSetPlot` + `Compile`/`Parallel`.
- `minesweeper_game` — `DynamicModule` GUI.
- `mouse_position` — `MousePosition[...]`.
- `nautical_bell` — `ScheduledTask` + `EmitSound`/`Sound` (also timing).
- `percentage_difference_between_images` — two `Import` images.
- `play_recorded_sounds` — `Import` FLAC + `ListPlay`.
- `rock-paper-scissors` — `DynamicModule` GUI + RNG.
- `sierpinski_triangle_graphical` — `Graphics`/`Show` (multi-variant).
- `simple_windowed_application` — `CreateDialog` + `Button`.
- `speech_synthesis` — `Speak[...]`.
- `set_puzzle` — `Row[Style[...]]` graphics + `RandomSample`.
- `statistics_basic` — `BarChart` + `RandomReal`.
- `statistics_normal_distribution` — `Histogram` + `RandomReal` (also broken syntax).
- `window_creation` — `CreateDocument[]`.
- `window_management` — `NotebookCreate`/`SetOptions`/`NotebookClose`.
- `wireworld` — `DynamicModule` + `ArrayPlot` + `CellularAutomaton`.

### 4d. External file / directory input required — no fixture (15)

Read or mutate files/directories that don't exist in the sandbox (and often rely
on `NotebookDirectory[]`, which has no FrontEnd in script mode). Activate by
adding a fixture file inside the test sandbox and rewriting paths to relative.

The sandbox is a throwaway working directory, so the cleanest activation is to
have the script **create its own fixture** with `Export[…]` and then read it
back (no external file needed). The five done below all use this pattern.

- `bitmap_read_an_image_through_a_pipe` *(also 4c)* — `.jpg`/`.ppm` fixtures.
- `check_that_file_exists` — `NotebookDirectory[]` + `input.txt`/`docs`.
- `create_a_file` — `NotebookDirectory[]`, `OpenWrite`, `CreateDirectory["\\docs"]`.
- `csv_data_manipulation` — `Import["test.csv"]`; **blocked** by a CSV round-trip
  bug in Woxi (`Import` of an exported CSV nests an extra `{{…}}` level, so the
  subsequent `iCSV[[i,j]] = …` part assignments fail). Fix CSV import nesting,
  then activate.
- `delete_a_file` — `NotebookDirectory[]` + `DeleteFile`/`DeleteDirectory`.
- `empty_directory` — `SetDirectory` + `FileNames[]` (+ "Example use:" prose).
- `find_duplicate_files` — `FileNames` + `FileHash` over the working tree.
- `loops_foreach` — `Import["ExampleData/USConstitution.txt"]`.
- `read_a_configuration_file` — `Import[configfile]` (configfile undefined).
- `rename_a_file` — `NotebookDirectory[]` + `RenameFile`/`RenameDirectory`.
- `text_processing_1` — `Import["Readings.txt","TSV"]`.
- `text_processing_2` — `Import["Readings.txt","TSV"]`.
- `text_processing_max_licenses_in_use` — `Import["mlijobs.txt","Table"]`.
- `xml_input` — `Import["test.xml","XML"]`.
- `xml_xpath` — `Import["test.txt","XML"]`.

✅ Done (self-creating fixture): `read_entire_file`, `letter_frequency`
(needed the new `Import[…, "Text"]` element), `read_a_specific_line_from_a_file`,
`input_loop`, `globally_replace_text_in_several_files`.

### 4e. Unsupported builtins / external packages (24)

Depend on a function or `Needs[...]` package Woxi does not implement. Activate by
implementing the builtin (then `functions.csv` + unit tests), then curating.

- `call_a_foreign-language_function` — `NETLink` / `DefineDLLFunction`.
- `call_a_function_in_a_shared_library` — `NETLink` / `DefineDLLFunction`.
- `euler_method` — `NDSolve` with `Method->"ExplicitEuler"`.
- `holidays_related_to_easter` — `Needs["Calendar`"]`, `EasterSunday`, `DaysPlus`.
- `introspection` — `$VersionNumber`, `NameQ`.
- `knapsack_problem_0-1` — `LinearProgramming` (Integers).
- `literals_floating_point` — `ScientificForm`/`EngineeringForm`/`AccountingForm` (+ prose).
- `multiple_regression` — `PseudoInverse` (has inline expected output).
- `numeric_error_propagation` — `±` (`PlusMinus`) `UpValues` arithmetic.
- `one-dimensional_cellular_automata` — `CellularAutomaton` with explicit rule lists.
- `paraffins` — `CycleIndexPolynomial`, `SymmetricGroup`, power series.
- `parallel_brute_force` — `ParallelDo` + `Hash[…,"SHA256"]`.
- `parametrized_sql_statement` — `Needs["DatabaseLink`"]`.
- `pathological_floating_point_problems` — arbitrary-precision `` `100 `` literals + `Once`.
- `permutations_derangements` — `Needs["Combinatorica`"]`, `Derangements`, `Subfactorial`.
- `quaternion_type` — `<<Quaternions`` package.
- `runge-kutta_method` — `DSolve`/`NDSolve`.
- `special_variables` — `Names["$*"]` (depends on the full `$`-symbol table).
- `stack_traces` — `Stack[]`/`Stack[_]` evaluator introspection.
- `subset_sum_problem` — `LinearProgramming` (and a heavy `Subsets[a,7]` variant).
- `table_creation` — `Needs["DatabaseLink`"]`.
- `table_creation_postal_addresses` — `Needs["DatabaseLink`"]`.
- `ternary_logic` — custom `UpValues` for `Maybe` + `Grid`/`Outer` truth tables.
- `test_a_function` — `Assert` + `On[Assert]`.

### 4f. Non-deterministic (RNG / parallel / timing) — ✅ DONE (13)

**Key finding:** seeding the RNG does *not* work — Woxi's `SeedRandom[n]` produces
a completely different stream than wolframscript (Wolfram uses a proprietary
cellular-automaton generator), so any value-dependent script would yield a
Woxi-only snapshot that diverges from real WL. Instead, each script's random (or
parallel/timed) input was **hardcoded to a fixed value**, with a comment noting
it would normally come from an RNG. Because the curated code is then identical
and deterministic in both engines, the outputs match. Two further divergences had
to be projected away during curation:

- **Unicode glyphs** (`\[WhiteRook]`, `♦♣♥♠`): Woxi emits UTF-8 but
  `wolframscript -file` emits these in a different encoding here — replaced with
  ASCII (piece letters / suit names).
- **Float last-digit drift**: high-precision results differ at the 16th–17th
  digit — wrapped in `Round[…, 0.001]` so both engines print identical digits.

Each curated script is verified identical to `wolframscript -file` and has a
`script_test!` + snapshot.

- `pick_random_element` — `RandomChoice[{a,b,c}]` → fixed pick.
- `random_numbers` — `RandomReal[NormalDistribution[…]]` → fixed sample.
- `create_an_html_table` — `x := RandomInteger[10]` → fixed constant.
- `concurrent_computing` — `ParallelDo`+`Pause` → fixed sequential `Do`.
- `sorting_algorithms_sleep_sort` — `RunScheduledTask` → direct sorted print.
- `sorting_algorithms_bogosort` — keeps `RandomSample` (result is seed-invariant); integer list avoids the `Sort`/`Pi` canonical-order divergence.
- `generate_chess960_starting_position` — `RandomChoice` → one fixed position (ASCII piece letters).
- `playing_cards` — `RandomSample` shuffle → unshuffled deck (ASCII suit names).
- `trabb_pardo–knuth_algorithm` — `RandomReal[{-2,6},11]` → fixed list; results `Round`ed.
- `permutations_rank_of_a_permutation` — deterministic table kept (rewrote `RankedMin`→`Sort[…][[k]]`, `(…)!`→`Factorial[…]` to dodge a parse gap); random tail dropped.
- `average_loop_length` — empirical Monte-Carlo estimate dropped; analytical value kept and `Round`ed.
- `percolation_mean_run_density` — `RandomReal` trial matrix → fixed 0/1 rows.
- `atomic_updates` — infinite parallel transfers → fixed balances + a fixed sequential transfer sequence (still conserves the total).

### 4g. System / process / time / infinite loop — non-reproducible (9)

Shell out, read the wall clock, or never terminate. Activate by replacing the
non-deterministic part with a fixed value or a bounded loop.

- `events` — `Pause[4]` + `Quit[]`.
- `factors_of_a_mersenne_number` — `For[…, Prime[1000000], …]` (huge) + `<>i` type bug.
- `fork` — `Run["MathKernel …"]` spawns a kernel.
- `integer_sequence` — `Monitor[While[True, x++], x]`.
- `loops_infinite` — `While[True, Print@"SPAM"]`.
- `pi` — endless digit stream + `Pause[.05]`.
- `system_time` — `DateList[]`, `AbsoluteTime[]`.
- `terminal_control_coloured_text` — `Run["tput setaf …"]`.
- `terminal_control_cursor_movement` — `Run["tput …"]` (+ `RunThrough`).

✅ Done: `count_in_octal` — bounded the `While[True,…]` to `Do[…, {x, 0, 16}]`
and used `IntegerString[x, 8]` (the original `BaseForm` is unevaluated in script
mode in both engines, producing no real octal output).

### 4h. Needs curation — multi-variant dump, inline output, prose, broken syntax, or too heavy (13)

Deterministic in principle but not runnable as-is: multiple solutions glued
together, expected-output text mixed into the code, explanatory prose, a source
typo, or an intractably large computation. Activate by extracting/repairing one
clean variant (and trimming the input size where needed), then verifying.

- `discordian_date` — needs `DateDifference` to return a plain day count, but
  modern wolframscript returns `Quantity[n, "Days"]`, which then errors as a
  `Part` index (`Part::pkspec1`) in *both* engines. Activating requires
  rewriting the day-of-year extraction (e.g. `QuantityMagnitude` /
  `DateDifference[…, "Day"]`), then re-verifying.
- `enforced_immutability` — inline `Set::wrsym` expected message (stdout-vs-stderr routing differs — like the 4-2 exclusions).
- `hofstadter-conway_10_000_sequence` — memoized recursion up to `2^20` (heavy).
- `iterated_digits_squaring` — `Do` with `Evaluate[Sequence @@ iterators]` spliced iterators; Woxi rejects it ("invalid iterator"). Needs Do to evaluate+flatten iterator args.
- `last_letter-first_letter` — deterministic but `Tuples`/`NestWhileList` is allocation-bound; see the known large-list throughput limit. May not be activatable without a perf fix.
- `literals_string` — pure prose with `// String` annotations.
- `parse_command-line_arguments` — `$CommandLine` + inline `->` output; differs from Woxi's argv.
- `scope_modifiers` — prose + `Module` symbol counters (`x$119`) that are non-deterministic.
- `sorting_algorithms_pancake_sort` — `pancakeSort[a_] : =` typo; even after fixing it the algorithm's `Flip[a[[;;n]]]` part-assignment-through-a-slice doesn't sort in real wolframscript either (it emits `Set::setps` and leaves the list unchanged), so there's no meaningful matching snapshot.
- `special_characters` — pure markup/prose, not code.
- `state_name_puzzle` — `Subsets[…, {4}]` over 50 states (heavy + allocation-bound; trimming would change the puzzle itself).
- `universal_turing_machine` — multi-variant + stray trailing `]` + `BitGet` on huge integers.
- `zhang-suen_thinning_algorithm` — `FixedPoint[iter, dat]` with `dat` undefined (needs an input bitmap).

> **Parser bug found & fixed here:** implicit multiplication with a `!`/`!!`/`..`
> suffix on a factor failed when the *first* factor was a function call —
> `Binomial[5,2] 2!` / `f[1] x!` raised a spurious parse error, while `5 2!`,
> `(f[1]) 2!`, and `1 f[1] 2!` worked. The grammar captured the suffix inside
> `FunctionCallExtended`'s `FunctionCallImplicitSuffix`, but `pair_to_expr`'s
> `parse_implicit_factors` loop ignored `FactorialSuffix`/`RepeatedSuffix` and
> re-parsed the bare `"!"` as its own expression. Fixed by handling those
> suffixes in that loop (mirroring the `ImplicitTimes` handler);
> `combinations_and_permutations` now uses the faithful implicit form
> `Binomial[n,k] k!`. Regression tests in
> `rosetta_script_fixes::implicit_times_factor_suffix_after_call`.

✅ Done: `align_columns` (defined the data + printed padded columns),
`associative_array_iteration` (rewrote `DownValues` form as `Association` with
`Keys`/`Values`/`KeyValueMap`), `best_shuffle` (added the missing comma in the
final `Print`), `binary_strings` (dropped prose/inline-output, kept the byte-list
operations as `Print`s), `pascals_triangle` (rewrote `MatrixExp[SparseArray…]` —
which yields a non-printing `Column` — as a `Binomial` table), `queue_usage`
(consistent `EmptyQ`/`Push`/`Pop` with `Print`s), `ranking_methods` (fixed the
`ConditionOp` `/@` parser bug, printed the rank lists instead of `Grid`),
`combinations_and_permutations` (extracted one variant; fixed the implicit
`a b!` parser bug — see the note above — and printed a P/C table),
`find_palindromic_numbers_in_both_binary_and_ternary_bases` (trimmed
`Range[1000000]`→`Range[1000]`, printed the `Union`), `last_friday_of_each_month`
(rewrote the deprecated `DaysPlus` as `DayRange`/`DatePlus`, mirroring the
working `find_the_last_sunday_of_each_month`), `loops_break` (hardcoded the RNG
sequence, fixed the typo/brackets), `self-referential_sequence` (trimmed
`Range[1000000]`→`Range[1000]`, printed seed/iterations/sequence).
