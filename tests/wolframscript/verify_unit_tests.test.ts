import assert from "node:assert/strict";
import { test } from "node:test";
import {
  evaluationFailureReason,
  incompleteCaseCount,
  runBatchResilient,
} from "./verify_unit_tests.ts";

test("kernel evaluation limits are distinct from cold-start failures", () => {
  for (const reason of ["RecursionLimit", "IterationLimit"]) {
    assert.equal(
      evaluationFailureReason(`message\nTerminatedEvaluation[${reason}]\n`),
      reason
    );
  }
  for (const output of ["", "not activated", "DONE", "FAIL #12\n  Wolfram: 3"]) {
    assert.equal(evaluationFailureReason(output), null);
  }
});

const exampleCase = (idx: number) => ({
  idx, expr: `f[${idx}]`, key: `f[${idx}]`, woxiResult: "1",
});

test("evaluation errors isolate the failing case without startup retries", (t) => {
  t.mock.method(console, "error", () => {});
  const failures: string[] = [];
  const sizes: number[] = [];
  const output = runBatchResilient([exampleCase(0), exampleCase(1)], failures, "batch", (batch) => {
    sizes.push(batch.length);
    return batch.some((entry) => entry.idx === 0)
      ? "TerminatedEvaluation[RecursionLimit]"
      : "DONE\n";
  });
  assert.deepEqual(sizes, [2, 1, 1]);
  assert.deepEqual(output, ["DONE"]);
  assert.equal(failures.length, 1);
  assert.match(failures[0], /^ERROR case #1 terminated with RecursionLimit:/);
});

test("transient startup failures are still retried", (t) => {
  t.mock.method(console, "error", () => {});
  let calls = 0;
  const failures: string[] = [];
  assert.deepEqual(runBatchResilient([exampleCase(0)], failures, "batch", () => {
    calls++;
    return calls === 1 ? "" : "DONE";
  }), ["DONE"]);
  assert.equal(calls, 2);
  assert.deepEqual(failures, []);
});

test("kernel errors on stdout survive a nonzero launcher exit", (t) => {
  t.mock.method(console, "error", () => {});
  const failures: string[] = [];
  let calls = 0;
  runBatchResilient([exampleCase(0)], failures, "batch", () => {
    calls++;
    throw Object.assign(new Error("wolframscript exited with 1"), {
      output: "TerminatedEvaluation[IterationLimit]",
    });
  });
  assert.equal(calls, 1);
  assert.match(failures[0], /^ERROR case #1 terminated with IterationLimit:/);
});

test("uncompleted comparisons do not count as passes", () => {
  assert.equal(incompleteCaseCount([]), 0);
  assert.equal(
    incompleteCaseCount([
      "FAIL #1: 1 + 1",
      "  Woxi: 3",
      "  Wolfram: 2",
      "ERROR case #2 terminated with RecursionLimit: f[x]",
      "FLAKY case #3 never produced DONE: g[x]",
    ]),
    2
  );
});
