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

/** Escape a string for embedding inside a Wolfram Language string literal.
 * Non-ASCII characters are escaped as \\:XXXX (Wolfram 4-digit hex escape) */
function escapeForWolfram(s: string): string {
  let result = "";
  for (const ch of s) {
    const code = ch.codePointAt(0)!;
    if (code > 127) {
      result += "\\:" + code.toString(16).padStart(4, "0");
    } else if (ch === "\\") {
      result += "\\\\";
    } else if (ch === '"') {
      result += '\\"';
    } else {
      result += ch;
    }
  }
  return result;
}

/**
 * Split a top-level semicolon-separated expression into statements.
 * Respects brackets [], parens (), braces {}, and strings "...".
 */
function splitTopLevelSemicolons(s: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let inString = false;
  let start = 0;

  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (inString) {
      if (ch === "\\" && i + 1 < s.length) {
        i++; // skip escaped char
      } else if (ch === '"') {
        inString = false;
      }
    } else {
      if (ch === '"') {
        inString = true;
      } else if (ch === "(" || ch === "[" || ch === "{" || ch === "<" && s[i + 1] === "|") {
        depth++;
      } else if (ch === ")" || ch === "]" || ch === "}" || ch === "|" && s[i + 1] === ">") {
        depth--;
      } else if (ch === ";" && depth === 0) {
        // Make sure it's not /; (Condition)
        if (i > 0 && s[i - 1] === "/") continue;
        parts.push(s.substring(start, i).trim());
        start = i + 1;
      }
    }
  }

  const last = s.substring(start).trim();
  if (last.length > 0) {
    parts.push(last);
  }

  return parts;
}

/**
 * Run an expression through woxi eval, wrapping it in
 * ToString[expr, InputForm] to get the canonical comparison format.
 *
 * For expressions containing multiple top-level semicolon-separated
 * statements, we only wrap the last one in ToString[(...), InputForm]
 * so that := definitions (which can't appear inside parens) work correctly.
 * Everything stays in a single woxi eval call to preserve state.
 */
function runWoxi(expr: string): string {
  let fullExpr: string;

  // Check if the expression contains := (function definitions) which
  // can't be wrapped inside ToString[(...), InputForm] parens.
  // In that case, split into setup statements and wrap only the last.
  if (expr.includes(":=")) {
    const stmts = splitTopLevelSemicolons(expr);
    if (stmts.length > 1) {
      const setup = stmts.slice(0, -1);
      const last = stmts[stmts.length - 1];
      fullExpr = setup.join("; ") + "; ToString[(" + last + "), InputForm]";
    } else {
      fullExpr = 'ToString[(' + expr + '), InputForm]';
    }
  } else {
    // No function definitions — wrap the whole expression (preserves trailing ;)
    fullExpr = 'ToString[(' + expr + '), InputForm]';
  }

  try {
    const output = execSync(`woxi eval '${fullExpr.replace(/'/g, "'\\''")}'`, {
      encoding: "utf-8",
      timeout: 10_000,
      stdio: ["pipe", "pipe", "ignore"], // suppress stderr (error messages like Part::partw)
    }).trim();
    // Return the full trimmed output.
    // Multi-line results (e.g. Definition) must be preserved intact.
    return output;
  } catch {
    return "<WOXI_ERROR>";
  }
}

/**
 * Build a wolframscript .wls that evaluates ToString[expr, InputForm]
 * for each test case, comparing against the expected Woxi result.
 * Mismatches are reported via Print.
 */
function buildWolframScript(
  cases: { expr: string; woxiResult: string; idx: number }[]
): string {
  const lines: string[] = [];
  lines.push("$RecursionLimit = 4096");

  for (const { expr, woxiResult, idx } of cases) {
    lines.push('ClearAll["Global`*"]');

    const exprEscaped = escapeForWolfram(expr);
    const expectedEscaped = escapeForWolfram(woxiResult);

    // Only split if expression contains := (function definitions can't be inside parens)
    let wBlock: string;
    if (expr.includes(":=")) {
      const stmts = splitTopLevelSemicolons(expr);
      if (stmts.length > 1) {
        const setup = stmts.slice(0, -1).join("; ");
        const last = stmts[stmts.length - 1];
        wBlock = setup + "; ToString[(" + last + "), InputForm]";
      } else {
        wBlock = "ToString[(" + expr + "), InputForm]";
      }
    } else {
      wBlock = "ToString[(" + expr + "), InputForm]";
    }

    const wExpected = '"' + expectedEscaped + '"';
    const wLabel = '"FAIL #' + (idx + 1) + ": " + exprEscaped + '"';
    lines.push(
      "Module[{res$$ = (" + wBlock + ")}," +
        " If[res$$ =!= " + wExpected + "," +
        " Print[" + wLabel + "];" +
        ' Print["  Woxi:    ' + expectedEscaped + '"];' +
        ' Print["  Wolfram: " <> res$$]]]'
    );
  }

  lines.push('Print["DONE"]');
  return lines.join(";\n");
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

  console.log(`Extracted ${allCases.length} test cases`);

  // Filter out multiline expressions (they break the generated scripts)
  const cases = allCases.filter((c) => !c.expr.includes("\n"));
  const skipped = allCases.length - cases.length;
  const tested = cases.length;

  // Step 1: Run each expression through woxi eval with ToString[_, InputForm]
  // Woxi is fast (~20ms per call), so this takes ~10s for 500 tests.
  console.log(`Running ${tested} test cases through woxi eval (${skipped} skipped)...`);
  const woxiResults: { expr: string; woxiResult: string; idx: number }[] = [];
  for (let i = 0; i < tested; i++) {
    const { expr, setup } = cases[i];
    // For expressions with setup, prepend setup code
    const fullExpr = setup
      ? [...setup.filter((s) => !s.includes("\n")), expr].join("; ")
      : expr;
    const result = runWoxi(fullExpr);
    woxiResults.push({ expr: fullExpr, woxiResult: result, idx: i });
  }

  // Step 2: Build a single wolframscript program that evaluates all
  // expressions and compares against the Woxi results.
  console.log("Running wolframscript...");
  const wolframProgram = buildWolframScript(woxiResults);
  const tmpFile = join(ROOT, "tests/wolframscript/.verify_unit_tests.wls");
  writeFileSync(tmpFile, wolframProgram);

  let output: string;
  try {
    output = execSync(`wolframscript -charset UTF8 -file "${tmpFile}"`, {
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
    console.log(`All ${tested} test cases match between Woxi and wolframscript.`);
  } else {
    console.error(`\n${passCount}/${tested} passed, ${failCount} differ:\n`);
    for (const line of failures) {
      console.error(line);
    }

    // Print file locations
    console.error("\nFile locations:");
    for (const line of failures) {
      const m = line.match(/^FAIL #(\d+)/);
      if (m) {
        const idx = parseInt(m[1]) - 1;
        const tc = cases[idx];
        if (tc) {
          console.error(`  ${tc.file}:${tc.line}`);
        }
      }
    }

    process.exit(1);
  }
}

main();
