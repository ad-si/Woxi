#!/usr/bin/env node
/**
 * Extracts assert_eq!(interpret("EXPR").unwrap(), "EXPECTED") pairs
 * from Rust unit test files and verifies them against wolframscript.
 *
 * Usage: npx tsx tests/wolframscript/verify_unit_tests.ts
 */

import {
  readFileSync,
  writeFileSync,
  unlinkSync,
  readdirSync,
  statSync,
} from "fs";
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
    .replace(/\\u\{([0-9a-fA-F]+)\}/g, (_, hex) =>
      String.fromCodePoint(parseInt(hex, 16))
    )
    .replace(/\\\\/g, "\\");
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
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
    const preceding = content.substring(Math.max(0, idx - 240), idx);
    const isAssertEqForm = /assert_eq!\(\s*$/.test(preceding);
    const letMatch = preceding.match(
      /let\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*[^=]+)?=\s*$/
    );
    const letVar = letMatch?.[1] ?? null;
    const isLetForm = letVar !== null;

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
      const assertPattern = new RegExp(
        "\\s*\\)\\s*\\.unwrap\\(\\)\\s*;\\s*assert_eq!\\(\\s*" +
          escapeRegex(letVar!) +
          "\\s*,\\s*"
      );
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
    } else if (ch === "\n") {
      result += "\\n";
    } else if (ch === "\r") {
      result += "\\r";
    } else if (ch === "\t") {
      result += "\\t";
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
    const output = execSync(`woxi eval --quiet-print '${fullExpr.replace(/'/g, "'\\''")}'`, {
      encoding: "utf-8",
      timeout: 10_000,
      stdio: ["pipe", "pipe", "ignore"], // suppress stderr (error messages like Part::partw)
    });
    // Preserve leading whitespace (important for OutputForm 2D rendering),
    // only strip trailing line breaks from CLI output.
    return output.replace(/[\r\n]+$/, "");
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
    // Wrap the ToString[...] part in Quiet[...] to suppress wolframscript messages
    // (e.g. Prime::intpp) that would otherwise pollute stdout and break the DONE check.
    let wBlock: string;
    if (expr.includes(":=")) {
      const stmts = splitTopLevelSemicolons(expr);
      if (stmts.length > 1) {
        const setup = stmts.slice(0, -1).join("; ");
        const last = stmts[stmts.length - 1];
        wBlock = setup + "; Quiet[ToString[(" + last + "), InputForm]]";
      } else {
        wBlock = "Quiet[ToString[(" + expr + "), InputForm]]";
      }
    } else {
      wBlock = "Quiet[ToString[(" + expr + "), InputForm]]";
    }

    const wExpected = '"' + expectedEscaped + '"';
    const wLabel = '"FAIL #' + (idx + 1) + ": " + exprEscaped + '"';
    // Wrap in CheckAbort so Abort[]/Interrupt[] calls inside test cases
    // don't kill the entire script run.
    // Strip trailing newlines from both sides before comparison,
    // because runWoxi strips trailing newlines from CLI output which
    // removes content newlines too (e.g. MathMLForm output ends with \n).
    lines.push(
      "Module[{res$$ = CheckAbort[(" + wBlock + '), "$Aborted"], rr$$, ee$$},' +
        " If[!StringQ[res$$], res$$ = ToString[res$$, InputForm]];" +
        ' rr$$ = StringReplace[res$$, RegularExpression["[\\\\r\\\\n]+$"] -> ""];' +
        " ee$$ = " + wExpected + ";" +
        " If[rr$$ =!= ee$$," +
        " Print[" + wLabel + "];" +
        ' Print["  Woxi:    ' + expectedEscaped + '"];' +
        ' Print["  Wolfram: " <> rr$$]]]'
    );
  }

  lines.push('Print["DONE"]');
  return lines.join(";\n");
}

function listRustFiles(dir: string): string[] {
  const files: string[] = [];

  for (const entry of readdirSync(dir)) {
    const fullPath = join(dir, entry);
    const st = statSync(fullPath);
    if (st.isDirectory()) {
      // Skip auxiliary/non-unit trees.
      if (entry.startsWith("_")) continue;
      if (entry === "cli") continue;
      if (entry === "notebooks") continue;
      if (entry === "woxi") continue;
      files.push(...listRustFiles(fullPath));
      continue;
    }
    if (entry.endsWith(".rs")) {
      files.push(fullPath);
    }
  }

  return files;
}

/**
 * Run one batch of test cases through wolframscript.
 * Returns the raw output string, or throws on failure.
 */
function runWolframBatch(
  batch: { expr: string; woxiResult: string; idx: number }[],
  timeoutMs = 300_000
): string {
  const wolframProgram = buildWolframScript(batch);
  try {
    return execSync(`wolframscript -charset UTF8 -code ${shellQuoteForExec(wolframProgram)}`, {
      encoding: "utf-8",
      timeout: timeoutMs,
      maxBuffer: 10 * 1024 * 1024,
    });
  } catch (err: any) {
    throw new Error(err.stderr || err.message || "wolframscript batch failed");
  }
}

/**
 * Return true iff a batch completed successfully (DONE sentinel present).
 * Uses a shorter timeout so bisection doesn't take forever.
 */
function batchOk(
  batch: { expr: string; woxiResult: string; idx: number }[]
): boolean {
  try {
    const out = runWolframBatch(batch, 60_000);
    return out.split("\n").some((l) => l.trim() === "DONE");
  } catch {
    return false;
  }
}

/**
 * Binary-search within a failing batch to find the first expression that
 * causes wolframscript to crash, hang, or produce no output.
 * Returns the culprit entry, or null if the batch unexpectedly passes now.
 */
function findFailingExpression(
  batch: { expr: string; woxiResult: string; idx: number }[]
): { expr: string; woxiResult: string; idx: number } | null {
  if (batch.length === 0) return null;
  if (batch.length === 1) {
    return batchOk(batch) ? null : batch[0];
  }
  const mid = Math.floor(batch.length / 2);
  const first = batch.slice(0, mid);
  if (!batchOk(first)) return findFailingExpression(first);
  return findFailingExpression(batch.slice(mid));
}

/** Shell-quote a string for use as a -code argument. */
function shellQuoteForExec(s: string): string {
  return "'" + s.replace(/'/g, "'\\''") + "'";
}

function main() {
  const testFiles = listRustFiles(join(ROOT, "tests"))
    .filter((f) => {
      const content = readFileSync(f, "utf-8");
      return content.includes("#[test]") && content.includes("interpret(");
    })
    .sort();

  let allCases: TestCase[] = [];
  for (const f of testFiles) {
    allCases = allCases.concat(extractTestCases(f));
  }

  console.log(`Extracted ${allCases.length} test cases`);

  // Expressions that produce inherently implementation-specific results and
  // can never match between Woxi and wolframscript:
  //  - Fit[]: floating-point rounding at machine-epsilon level (different QR vs LAPACK)
  //  - SeedRandom[]: returns RNG internal state (ChaCha8 vs ExtendedCA)
  //  - Share[]: returns system-specific memory deduplication byte count
  //  - Names[]: returns implementation-specific set of built-in symbols
  // (Hash with 1 arg uses assert! not assert_eq!, so it's naturally excluded.)
  const IMPL_SPECIFIC_PATTERNS = [
    /\bFit\[/,
    /\bSeedRandom\[/,
    /\bShare\[/,
    /\bNames\[/,
  ];

  // Filter out multiline expressions (they break the generated scripts).
  // Also skip Interrupt[] — it sends a kernel interrupt that crashes wolframscript
  // even inside CheckAbort, so it cannot be tested via batch conformance.
  // Also skip bare Goto[tag] without a matching Label — it fatally aborts the
  // wolframscript session (uncatchable, even by CheckAbort/Catch).
  const cases = allCases.filter(
    (c) =>
      !c.expr.includes("\n") &&
      !c.expr.includes("Interrupt[]") &&
      !/[^\x00-\x7F]/.test(c.expr) && // Non-ASCII chars get garbled by wolframscript encoding
      !(c.expr.match(/^Goto\[/) && !c.expr.includes("Label[")) &&
      !IMPL_SPECIFIC_PATTERNS.some((p) => p.test(c.expr))
  );
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

  // Filter out rendered-object placeholders: Graphics/Image objects render
  // to SVG/pixels internally so their InputForm is implementation-specific
  // (different sampling points, coordinate transforms, etc.) and will never
  // match between Woxi and wolframscript.
  const RENDERED_PLACEHOLDERS = ["-Graphics-", "-Graphics3D-"];
  const beforeFilter = woxiResults.length;
  const filteredResults = woxiResults.filter(
    (r) => !RENDERED_PLACEHOLDERS.includes(r.woxiResult)
      // PDF output differs between generators — skip byte-level comparison
      && !r.woxiResult.startsWith("%PDF-")
      // Box-formatted output (DisplayForm[RowBox[...]]) uses private-use Unicode
      // code points in wolframscript but plain ASCII in Woxi — the visual output
      // is identical but byte-level comparison fails.
      && !r.woxiResult.startsWith("DisplayForm[")
  );
  const renderedSkipped = beforeFilter - filteredResults.length;
  if (renderedSkipped > 0) {
    console.log(`Skipped ${renderedSkipped} rendered-object tests (Graphics/Image placeholders).`);
  }
  const woxiResultsFiltered = filteredResults;

  // Step 2: Run wolframscript in batches to avoid server timeout/buffer limits.
  // Each batch runs independently; we accumulate failures across all batches.
  const BATCH_SIZE = 50;
  const totalBatches = Math.ceil(woxiResultsFiltered.length / BATCH_SIZE);
  console.log(`Running wolframscript in ${totalBatches} batches of up to ${BATCH_SIZE}...`);

  const failures: string[] = [];
  let failCount = 0;

  for (let b = 0; b < totalBatches; b++) {
    const batchStart = b * BATCH_SIZE;
    const batch = woxiResultsFiltered.slice(batchStart, batchStart + BATCH_SIZE);

    let output: string;
    let batchCrashed = false;
    let crashErr = "";
    try {
      output = runWolframBatch(batch);
    } catch (err: any) {
      batchCrashed = true;
      crashErr = err.message || String(err);
      output = "";
    }

    const outputLines = output.trim().split("\n");

    // Check for DONE sentinel — Print["DONE"] returns Null which wolframscript
    // may print as an extra trailing line, so search all lines rather than
    // requiring DONE to be last.
    const doneIdx = outputLines.findIndex((l) => l.trim() === "DONE");
    if (batchCrashed || doneIdx === -1) {
      const batchEnd = batchStart + batch.length - 1;
      const reason = batchCrashed
        ? `crashed: ${crashErr}`
        : output.trim() === ""
          ? "produced no output (crash or timeout)"
          : `did not contain DONE sentinel`;
      console.error(
        `\nBatch ${b + 1}/${totalBatches} (cases ${batchStart + 1}–${batchEnd + 1}) ${reason}.`
      );
      if (!batchCrashed && output.trim()) {
        console.error(`wolframscript output:\n${output}`);
      }
      console.error(`\nBisecting to find the failing expression...`);
      const culprit = findFailingExpression(batch);
      if (culprit) {
        const tc = cases[culprit.idx];
        console.error(`\nFailing expression (case #${culprit.idx + 1}): ${culprit.expr}`);
        console.error(`Woxi result: ${culprit.woxiResult}`);
        if (tc) console.error(`Source: ${tc.file}:${tc.line}`);
      } else {
        console.error(`Bisection could not reproduce the failure (flaky?).`);
      }
      process.exit(2);
    }

    // Collect failures from this batch
    for (const line of outputLines) {
      if (line.startsWith("FAIL") || line.startsWith("  ")) {
        failures.push(line);
        if (line.startsWith("FAIL")) failCount++;
      }
    }

    if (totalBatches > 1) {
      process.stdout.write(`  batch ${b + 1}/${totalBatches} done\r`);
    }
  }

  if (totalBatches > 1) {
    process.stdout.write("\n");
  }

  const testedFiltered = woxiResultsFiltered.length;
  const passCount = testedFiltered - failCount;

  if (failCount === 0) {
    console.log(`All ${testedFiltered} test cases match between Woxi and wolframscript.`);
  } else {
    console.error(`\n${passCount}/${testedFiltered} passed, ${failCount} differ:\n`);
    for (const line of failures) {
      const m = line.match(/^FAIL #(\d+)/);
      if (m) {
        const idx = parseInt(m[1]) - 1;
        const tc = cases[idx];
        if (tc) {
          console.error(`\n${tc.file}:${tc.line}`);
        }
      }
      console.error(line);
    }

    process.exit(1);
  }
}

main();
