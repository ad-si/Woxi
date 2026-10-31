use super::*;

// A large stored value that already evaluated to itself is returned as is on
// later reads, until a definition, attribute, option or context changes (see
// `src/epoch.rs`). These tests pin down both halves: reads are cheap, and
// every kind of change still shows through. The values hold at least 16
// elements, the size at which a stored value is shared and can be reused.
mod evaluation_reuse {
  use super::*;

  mod invalidation {
    use super::*;

    #[test]
    fn new_down_value_reaches_a_stored_value() {
      clear_state();
      assert_eq!(
        interpret("r = Table[f[i], {i, 20}]; r; f[3] = 5; r[[3]]").unwrap(),
        "5"
      );
    }

    #[test]
    fn new_own_value_reaches_a_stored_value() {
      clear_state();
      assert_eq!(
        interpret("r = Table[y^i, {i, 20}]; r; y = 2; Total[r]").unwrap(),
        "2097150"
      );
    }

    #[test]
    fn unset_reaches_a_stored_value() {
      clear_state();
      assert_eq!(
        interpret("r = Table[w^i, {i, 20}]; w = 2; r; w =.; r[[2]]").unwrap(),
        "w^2"
      );
    }

    #[test]
    fn clear_reaches_a_stored_value() {
      clear_state();
      assert_eq!(
        interpret(
          "r = Table[g[i], {i, 20}]; r; g[x_] := x^2; t = Total[r]; \
           Clear[g]; {t, r[[2]]}"
        )
        .unwrap(),
        "{2870, g[2]}"
      );
    }

    #[test]
    fn set_attributes_reaches_a_stored_value() {
      clear_state();
      assert_eq!(
        interpret(
          "r = Table[h[b, a], {20}]; r; SetAttributes[h, Orderless]; r[[1]]"
        )
        .unwrap(),
        "h[a, b]"
      );
    }

    #[test]
    fn clear_attributes_reaches_a_stored_value() {
      clear_state();
      assert_eq!(
        interpret(
          "SetAttributes[h, HoldAll]; r = Table[h[1 + 1], {20}]; r; \
           ClearAttributes[h, HoldAll]; r[[1]]"
        )
        .unwrap(),
        "h[2]"
      );
    }

    #[test]
    fn block_reaches_a_stored_value_and_is_undone() {
      clear_state();
      assert_eq!(
        interpret(
          "r = Table[q^i, {i, 20}]; r; {Block[{q = 2}, Total[r]], r[[3]]}"
        )
        .unwrap(),
        "{2097150, q^3}"
      );
    }

    #[test]
    fn assignment_inside_a_loop_reaches_a_stored_value() {
      clear_state();
      assert_eq!(
        interpret("r = Table[q^i, {i, 20}]; Table[q = k; Total[r], {k, 3}]")
          .unwrap(),
        "{20, 2097150, 5230176600}"
      );
    }

    #[test]
    fn up_value_reaches_a_stored_value() {
      clear_state();
      assert_eq!(
        interpret(
          "r = Table[u[t[i]], {i, 20}]; r; t /: u[t[x_]] := x; Total[r]"
        )
        .unwrap(),
        "210"
      );
    }

    #[test]
    fn memoized_value_sees_later_definitions() {
      clear_state();
      assert_eq!(
        interpret(
          "res[1] = Table[k[i], {i, 20}]; res[1]; k[3] = 0; res[1][[3]]"
        )
        .unwrap(),
        "0"
      );
    }

    #[test]
    fn a_read_that_issues_a_message_issues_it_every_time() {
      clear_state();
      assert_eq!(
        interpret(
          "r = Quiet[Table[{1, 2}[[5]], {20}]]; Quiet[r]; Check[r, \"msg\"]"
        )
        .unwrap(),
        "msg"
      );
    }
  }

  mod speed {
    use super::*;

    #[test]
    fn reading_a_large_expanded_sum_repeatedly_is_cheap() {
      clear_state();
      interpret("r = Expand[(a + b + c + d)^12];").unwrap();
      let start = std::time::Instant::now();
      assert_eq!(interpret("Do[r, {1000}]; Length[r]").unwrap(), "455");
      let elapsed = start.elapsed();
      assert!(
        elapsed.as_secs() < 2,
        "1000 reads of a 455-term sum took {elapsed:?}"
      );
    }

    #[test]
    fn reading_a_large_memoized_value_repeatedly_is_cheap() {
      clear_state();
      interpret("res[1] = Expand[(a + b + c + d)^12];").unwrap();
      let start = std::time::Instant::now();
      assert_eq!(
        interpret("Do[res[1], {1000}]; Length[res[1]]").unwrap(),
        "455"
      );
      let elapsed = start.elapsed();
      assert!(
        elapsed.as_secs() < 2,
        "1000 reads of a memoized 455-term sum took {elapsed:?}"
      );
    }

    #[test]
    fn reading_a_stored_ndsolve_result_is_cheap() {
      clear_state();
      // The tabulated points sit behind a rule and a curried call, so a
      // copy of the result used to copy every one of them.
      interpret(
        "s = NDSolve[{y'[t] == -y[t] + Sin[50 t], y[0] == 1}, y, {t, 0, 99}]; \
         s2 = s;",
      )
      .unwrap();
      let start = std::time::Instant::now();
      assert_eq!(
        interpret("Do[s2, {2000}]; (y[3.] /. s2) == (y[3.] /. s)").unwrap(),
        "True"
      );
      let elapsed = start.elapsed();
      assert!(
        elapsed.as_secs() < 2,
        "2000 reads of a stored NDSolve result took {elapsed:?}"
      );
    }

    #[test]
    fn replacing_a_symbol_that_does_not_occur_is_cheap() {
      clear_state();
      interpret("r = Expand[(a + b + c + d)^12];").unwrap();
      let start = std::time::Instant::now();
      assert_eq!(
        interpret("Do[r /. q -> 1, {100}]; (r /. q -> 1) === r").unwrap(),
        "True"
      );
      let elapsed = start.elapsed();
      assert!(
        elapsed.as_secs() < 2,
        "100 replacements of an absent symbol took {elapsed:?}"
      );
    }
  }

  mod machine_real_symbol {
    use super::*;

    #[test]
    fn small_real_keeps_its_value() {
      clear_state();
      assert_eq!(
        interpret("w = 632.8*^-9; {w == 6.328*^-7, Head[w], w*2}").unwrap(),
        "{True, Real, 1.2656*^-6}"
      );
    }

    #[test]
    fn small_real_is_not_reparsed_on_each_read() {
      clear_state();
      interpret("w = 632.8*^-9; h = 2.49*^-3;").unwrap();
      let start = std::time::Instant::now();
      interpret("Do[h/w, {20000}]").unwrap();
      let elapsed = start.elapsed();
      assert!(
        elapsed.as_secs() < 2,
        "20000 reads of a small real took {elapsed:?}"
      );
    }
  }
}
