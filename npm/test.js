// Tests for the npm wrapper. Run with `node --test` (or `npm test`) after
// building the wasm bundle via `make npm-build` in the repo root.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const path = require("node:path");
const { execFileSync } = require("node:child_process");

const woxi = require("./index.js");

test("evaluate returns the result as a string", () => {
  assert.equal(woxi.evaluate("Plus[1, 2]"), "3");
});

test("evaluate solves symbolically", () => {
  assert.equal(woxi.evaluate("1/3 + 1/6"), "1/2");
  assert.equal(woxi.evaluate("Sqrt[8]"), "2*Sqrt[2]");
});

test("evaluate includes Print output", () => {
  assert.equal(woxi.evaluate('Print["hello"]; 1 + 1'), "hello\n2");
});

test("evaluate reports errors as text", () => {
  assert.match(woxi.evaluate("1 +"), /^Error: /);
});

test("state persists across calls until clear()", () => {
  woxi.evaluate("x = 42");
  assert.equal(woxi.evaluate("x + 1"), "43");
  woxi.clear();
  assert.equal(woxi.evaluate("x"), "x");
});

test("evaluateAll returns structured output items", () => {
  woxi.clear();
  const items = woxi.evaluateAll('Print["out"]\n1 + 1');
  assert.equal(items.length, 2);
  assert.deepEqual(items[0], { type: "print", text: "out" });
  assert.equal(items[1].type, "text");
  assert.equal(items[1].text, "2");
  // Text items also carry an SVG rendering of the result.
  assert.match(items[1].svg, /^<svg/);
});

test("evaluateAll returns graphics items with SVG", () => {
  woxi.clear();
  const items = woxi.evaluateAll("Graphics[{Disk[]}]");
  assert.equal(items.length, 1);
  assert.equal(items[0].type, "graphics");
  assert.match(items[0].svg, /^<svg/);
});

test("splitStatements splits top-level statements", () => {
  assert.deepEqual(woxi.splitStatements("a = 1\nb = 2"), ["a = 1", "b = 2"]);
});

test("evaluateStatement evaluates a single statement", () => {
  woxi.clear();
  const items = woxi.evaluateStatement("3 * 4");
  assert.equal(items.length, 1);
  assert.equal(items[0].type, "text");
  assert.equal(items[0].text, "12");
});

test("getGraphics returns the SVG of the last evaluation", () => {
  woxi.clear();
  woxi.evaluate("Graphics[{Circle[]}]");
  assert.match(woxi.getGraphics(), /^<svg/);
});

test("getWarnings returns warnings as an array", () => {
  woxi.clear();
  assert.deepEqual(woxi.getWarnings(), []);
});

test("virtual files back Import[]", () => {
  woxi.clear();
  woxi.setVirtualFile("data.csv", "a,b\n1,2\n");
  assert.equal(woxi.evaluate('Import["data.csv"]'), "{{a, b}, {1, 2}}");
  woxi.clearVirtualFiles();
  assert.match(woxi.evaluate('Import["data.csv"]'), /Error/);
});

test("setVirtualFile accepts binary data", () => {
  woxi.clear();
  woxi.setVirtualFile("bytes.txt", new TextEncoder().encode("hi"));
  assert.equal(woxi.evaluate('Import["bytes.txt"]'), "hi");
  woxi.clearVirtualFiles();
});

test("cli evaluates expressions", () => {
  const cli = path.join(__dirname, "cli.js");
  const out = execFileSync(process.execPath, [cli, "eval", "Plus[1, 2]"], {
    encoding: "utf8",
  });
  assert.equal(out, "3\n");
});

test("cli reads code from stdin with `eval -`", () => {
  const cli = path.join(__dirname, "cli.js");
  const out = execFileSync(process.execPath, [cli, "eval", "-"], {
    encoding: "utf8",
    input: "9 / 3",
  });
  assert.equal(out, "3\n");
});

test("cli exits non-zero on evaluation errors", () => {
  const cli = path.join(__dirname, "cli.js");
  assert.throws(
    () => execFileSync(process.execPath, [cli, "eval", "1 +"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    }),
    (err) => err.status === 1,
  );
});
