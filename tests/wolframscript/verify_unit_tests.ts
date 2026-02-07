#!/usr/bin/env node
/**
 * Extracts assert_eq!(interpret("EXPR").unwrap(), "EXPECTED") pairs
 * from Rust unit test files and verifies them against wolframscript.
 *
 * Usage: npx tsx tests/wolframscript/verify_unit_tests.ts
 */

import { readFileSync, writeFileSync, unlinkSync } from "fs";
import { execSync } from "child_process";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "../..");

interface TestCase {
  expr: string;
  expected: string;
  file: string;
  line: number;
  /** Setup expressions from prior interpret() calls in the same test function */
  setup?: string[];
}

/** Unescape Rust string escapes: \" → ", \\ → \, \n → newline */
function unescapeRust(s: string): string {
  return s
    .replace(/\\"/g, '"')
    .replace(/\\n/g, "\n")
    .replace(/\\\\/g, "\\");
}

/**
 * Extract a Rust string literal starting at position `pos` in `src`.
 * Handles both "..." (with escapes) and r#"..."# raw strings.
 * Returns [content, endPos] or null if no string found.
 */
function extractRustString(
  src: string,
  pos: number
): [string, number] | null {
  // Skip whitespace
  while (pos < src.length && /\s/.test(src[pos])) pos++;

  if (pos >= src.length) return null;

  // Raw string: r#"..."#
  if (src.startsWith('r#"', pos)) {
    const start = pos + 3;
    const end = src.indexOf('"#', start);
    if (end === -1) return null;
    return [src.substring(start, end), end + 2];
  }

  // Regular string: "..."
  if (src[pos] === '"') {
    let i = pos + 1;
    let content = "";
    while (i < src.length) {
      if (src[i] === "\\") {
        content += src[i] + src[i + 1];
        i += 2;
      } else if (src[i] === '"') {
        return [unescapeRust(content), i + 1];
      } else {
        content += src[i];
        i++;
      }
    }
    return null;
  }

  return null;
}

/**
 * Extract test cases from a Rust test file using a parser approach
 * instead of a single regex, to correctly handle raw strings.
 *
 * Tracks test function boundaries: if there's a `fn ` declaration
 * between two interpret() calls, the second one starts fresh.
 * Otherwise, the prior expression(s) become setup code.
 */
function extractTestCases(filePath: string): TestCase[] {
  const content = readFileSync(filePath, "utf-8");
  const cases: TestCase[] = [];
  const relPath = filePath.replace(ROOT + "/", "");

  // Track expressions within the current test function for stateful follow-ups.
  let priorExprsInFn: string[] = [];
  let lastInterpretEnd = 0;

  // Find all interpret( calls inside assert_eq! or let result =
  // We search for `interpret(` and classify by context.
  const interpretMarker = "interpret(";

  let searchPos = 0;
  while (searchPos < content.length) {
    const idx = content.indexOf(interpretMarker, searchPos);
    if (idx === -1) break;

    // Look backwards from `interpret(` to determine the form:
    // 1. `assert_eq!(interpret(` — possibly with whitespace/newlines between assert_eq!( and interpret(
    // 2. `let result = interpret(`
    const preceding = content.substring(Math.max(0, idx - 80), idx);
    const isAssertEqForm = /assert_eq!\(\s*$/.test(preceding);
    const isLetForm = /let\s+result\s*=\s*$/.test(preceding);

    if (!isAssertEqForm && !isLetForm) {
      searchPos = idx + 1;
      continue;
    }

    const line = content.substring(0, idx).split("\n").length;

    // Check if there's a new test function between last interpret() and this one.
    // If so, reset the accumulated expressions.
    const between = content.substring(lastInterpretEnd, idx);
    if (/\bfn\s+\w+\s*\(\s*\)/.test(between)) {
      priorExprsInFn = [];
    }

    let expr: string | null = null;
    let expected: string | null = null;
    let afterEnd: number = idx + 1;

    // Position right after `interpret(`
    const afterInterpret = idx + interpretMarker.length;

    if (isAssertEqForm) {
      // assert_eq!(interpret("EXPR").unwrap(), "EXPECTED")
      const exprResult = extractRustString(content, afterInterpret);
      if (!exprResult) {
        searchPos = idx + 1;
        continue;
      }
      const [e, afterExpr] = exprResult;

      // Skip .unwrap(), or ).unwrap(),
      const unwrapPattern = /\s*\)\s*\.unwrap\(\)\s*,\s*/;
      const afterExprStr = content.substring(afterExpr);
      const unwrapMatch = afterExprStr.match(unwrapPattern);
      if (!unwrapMatch) {
        searchPos = idx + 1;
        continue;
      }
      const afterUnwrap = afterExpr + unwrapMatch[0].length;

      const expectedResult = extractRustString(content, afterUnwrap);
      if (!expectedResult) {
        searchPos = idx + 1;
        continue;
      }

      expr = e;
      expected = expectedResult[0];
      afterEnd = expectedResult[1];
    } else {
      // let result = interpret("EXPR").unwrap(); assert_eq!(result, "EXPECTED")
      const exprResult = extractRustString(content, afterInterpret);
      if (!exprResult) {
        searchPos = idx + 1;
        continue;
      }
      const [e, afterExpr] = exprResult;

      // Skip ).unwrap(); and find assert_eq!(result,
      const restStr = content.substring(afterExpr);
      const assertPattern =
        /\s*\)\s*\.unwrap\(\);\s*assert_eq!\(\s*result\s*,\s*/;
      const assertMatch = restStr.match(assertPattern);
      if (!assertMatch) {
        searchPos = idx + 1;
        continue;
      }
      const afterAssert = afterExpr + assertMatch[0].length;

      const expectedResult = extractRustString(content, afterAssert);
      if (!expectedResult) {
        searchPos = idx + 1;
        continue;
      }

      expr = e;
      expected = expectedResult[0];
      afterEnd = expectedResult[1];
    }

    // If there are prior expressions in this test function,
    // attach them as setup code
    const setup = priorExprsInFn.length > 0 ? [...priorExprsInFn] : undefined;

    cases.push({ expr, expected, file: relPath, line, setup });

    // Record this expression for potential follow-ups
    priorExprsInFn.push(expr);
    lastInterpretEnd = afterEnd;

    searchPos = afterEnd;
  }

  return cases;
}

/** Escape a string for embedding inside a Wolfram Language string literal */
function escapeForWolfram(s: string): string {
  return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function main() {
  const testFiles = [
    join(ROOT, "tests/interpreter_tests.rs"),
    join(ROOT, "tests/list_tests.rs"),
    join(ROOT, "tests/high_level_functions_tests.rs"),
  ];

  let allCases: TestCase[] = [];
  for (const f of testFiles) {
    allCases = allCases.concat(extractTestCases(f));
  }

  // Known intentional divergences from wolframscript.
  // These are expressions where Woxi deliberately differs from Wolfram.
  const SKIP_EXPRS = new Set([
    // Float precision: Woxi uses Rust f64 formatting which differs
    "9.6 / 3",
    "9.6 / 3 + 3.0 / 3",
    // Block[{x}, x] — Woxi returns Null, Wolfram returns x
    "Block[{x}, x]",
    // Return in Block/Module — Woxi catches Return, Wolfram doesn't
    "Block[{}, Return[42]]",
    "Module[{x = 10}, Return[x + 1]]",
    // Switch no-match — Woxi returns Null, Wolfram returns unevaluated
    "Switch[4, 1, a, 2, b, 3, c]",
    // Minus[5, 2] — Woxi uses U+2212, encoding comparison is unreliable
    "Minus[5, 2]",
  ]);

  let skipped = 0;
  console.log(`Extracted ${allCases.length} test cases`);

  // Build a single Wolfram program that evaluates all expressions
  // and reports mismatches.
  // Clear state before each test to prevent interference.
  const wolframLines: string[] = [];

  // Define a helper that mimics Woxi's output formatting:
  // - InputForm for single-line rendering (fractions, powers, etc.)
  // - Strip quotes (Woxi displays strings without quotes)
  // - Special case FullForm (ToString[FullForm[...]] already works)
  // Use WoxiTest` context so ClearAll["Global`*"] doesn't remove it.
  wolframLines.push('$RecursionLimit = 4096');
  wolframLines.push(
    'WoxiTest`str[FullForm[x_]] := ToString[FullForm[x]]'
  );
  wolframLines.push(
    'WoxiTest`str[x_] := StringReplace[ToString[x, InputForm], "\\"" -> ""]'
  );

  for (let i = 0; i < allCases.length; i++) {
    const { expr, expected, setup } = allCases[i];

    // Skip expressions containing literal newlines (multiline tests)
    // — they break the generated Wolfram script and are tested elsewhere
    if (expr.includes("\n")) {
      skipped++;
      continue;
    }

    // Skip known intentional divergences
    if (SKIP_EXPRS.has(expr)) {
      skipped++;
      continue;
    }

    // Escape for embedding in Wolfram strings
    const exprEscaped = escapeForWolfram(expr);
    const expectedEscaped = escapeForWolfram(expected);

    // Clear all Global symbols before each test
    wolframLines.push('ClearAll["Global`*"]');

    // If this test depends on prior interpret() calls, run setup code first
    if (setup) {
      for (const setupExpr of setup) {
        if (!setupExpr.includes("\n")) {
          wolframLines.push(setupExpr);
        }
      }
    }

    // Each test: evaluate, convert via WoxiTest`str, compare
    const n = i + 1;
    const wExpr = "WoxiTest`str[" + expr + "]";
    const wExpected = '"' + expectedEscaped + '"';
    const wLabel = '"FAIL #' + n + ": " + exprEscaped + '"';
    const wExpectedLabel = '"  Expected: ' + expectedEscaped + '"';
    wolframLines.push(
      "Module[{res$$ = " + wExpr + "}," +
        " If[res$$ =!= " + wExpected + "," +
        " Print[" + wLabel + "];" +
        " Print[" + wExpectedLabel + "];" +
        ' Print["  Got:      " <> res$$]]]'
    );
  }

  // Add a final sentinel so we know the script completed
  wolframLines.push('Print["DONE"]');

  const wolframProgram = wolframLines.join(";\n");

  // Write to a temp file and run wolframscript
  const tmpFile = join(ROOT, "tests/wolframscript/.verify_unit_tests.wls");
  writeFileSync(tmpFile, wolframProgram);

  const tested = allCases.length - skipped;
  console.log(`Running wolframscript on ${tested} test cases (${skipped} skipped)...`);

  let output: string;
  try {
    output = execSync(`wolframscript -file "${tmpFile}"`, {
      encoding: "utf-8",
      timeout: 300_000,
      maxBuffer: 10 * 1024 * 1024,
    });
  } catch (err: any) {
    console.error("wolframscript failed:");
    console.error(err.stderr || err.message);
    process.exit(2);
  } finally {
    try { unlinkSync(tmpFile); } catch {}
  }

  const outputLines = output.trim().split("\n");

  // Check for DONE sentinel
  const lastLine = outputLines[outputLines.length - 1]?.trim();
  if (lastLine !== "DONE") {
    console.error("wolframscript did not complete successfully");
    console.error("Output:", output);
    process.exit(2);
  }

  // Collect failures
  const failures: string[] = [];
  for (const line of outputLines) {
    if (line.startsWith("FAIL") || line.startsWith("  ")) {
      failures.push(line);
    }
  }

  const failCount = failures.filter((l) => l.startsWith("FAIL")).length;
  const passCount = tested - failCount;

  if (failCount === 0) {
    console.log(`All ${tested} test cases match wolframscript.`);
  } else {
    console.error(`\n${passCount}/${tested} passed, ${failCount} differ from wolframscript:\n`);
    for (const line of failures) {
      console.error(line);
    }

    // Also print which test case each failure corresponds to
    console.error("\nFile locations:");
    for (const line of failures) {
      const m = line.match(/^FAIL #(\d+)/);
      if (m) {
        const idx = parseInt(m[1]) - 1;
        const tc = allCases[idx];
        if (tc) {
          console.error(`  ${tc.file}:${tc.line}`);
        }
      }
    }

    process.exit(1);
  }
}

main();
