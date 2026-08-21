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
import { execSync, spawnSync } from "child_process";
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

/** Unescape Rust string escapes: \" → ", \\ → \, \n → newline.
 * A backslash at the end of a line is Rust's line continuation: it and the
 * following indentation are not part of the string. */
function unescapeRust(s: string): string {
  return s
    .replace(/\\\r?\n[ \t]*/g, "")
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

  // File-level `const NAME: &str = "…";` definitions. Tests set the stage
  // with `interpret(PACKAGE)` and then assert on a follow-up expression, so
  // the const's text has to become setup code or the follow-up is compared
  // against a session that never saw the package.
  const consts = new Map<string, string>();
  {
    const constRe = /const\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*&'?\w*\s*str\s*=\s*/g;
    let m: RegExpExecArray | null;
    while ((m = constRe.exec(content)) !== null) {
      const value = extractRustString(content, m.index + m[0].length);
      if (value) consts.set(m[1], value[0]);
    }
  }

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
      // A bare `interpret(…).unwrap();` statement, not asserted on: its
      // expression is state the next assertion in the same test depends on.
      // Either spelled out (`interpret("x = 1")`) or naming one of the
      // file's string consts (`interpret(PACKAGE)`).
      const literal = extractRustString(content, idx + interpretMarker.length);
      const named = content
        .substring(idx + interpretMarker.length)
        .match(/^([A-Za-z_][A-Za-z0-9_]*)\s*\)/);
      let setupText: string | null = null;
      let setupEnd = idx + 1;
      if (literal && /^\s*,?\s*\)/.test(content.substring(literal[1]))) {
        setupText = literal[0];
        setupEnd = literal[1];
      } else if (named && consts.has(named[1])) {
        setupText = consts.get(named[1])!;
        setupEnd = idx + interpretMarker.length + named[0].length;
      }
      if (setupText !== null) {
        const between = content.substring(lastInterpretEnd, idx);
        if (
          /\bfn\s+\w+\s*\(\s*\)/.test(between)
          || /\bclear_state\s*\(\s*\)/.test(between)
        ) {
          priorExprsInFn = [];
        }
        priorExprsInFn.push(setupText);
        lastInterpretEnd = setupEnd;
        searchPos = setupEnd;
        continue;
      }
      searchPos = idx + 1;
      continue;
    }

    const line = content.substring(0, idx).split("\n").length;

    // Check if there's a new test function between last interpret() and this one.
    // If so, reset the accumulated expressions.
    //
    // Also reset on an intervening `clear_state()` call — many tests use
    // `clear_state()` between sequential `interpret(...)` invocations
    // *inside the same test function* to deliberately start over with a
    // fresh evaluator state. Without this reset, the second interpret()
    // would inherit the first one's expression as setup and produce a
    // result that disagrees with running the expression on its own
    // (e.g. function-definition tests that exercise both forward and
    // reversed-arg rules in the same fn).
    const between = content.substring(lastInterpretEnd, idx);
    if (
      /\bfn\s+\w+\s*\(\s*\)/.test(between)
      || /\bclear_state\s*\(\s*\)/.test(between)
    ) {
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
      } else if (ch === ";") {
        // `;;` is the Span operator, not two statement separators.
        // The tokenizer is greedy left-to-right, exactly like Wolfram's:
        // `;;;` is a Span followed by a separator.
        if (s[i + 1] === ";") {
          i++;
          continue;
        }
        if (depth !== 0) continue;
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
      fullExpr = setup.join("; ") + "; Quiet[ToString[(" + last + "), InputForm]]";
    } else {
      fullExpr = 'Quiet[ToString[(' + expr + '), InputForm]]';
    }
  } else {
    // No function definitions — wrap the whole expression (preserves trailing ;)
    // Quiet suppresses messages (e.g. Prime::intpp) that would otherwise
    // pollute stdout and cause comparison mismatches with wolframscript
    // (which also wraps in Quiet).
    fullExpr = 'Quiet[ToString[(' + expr + '), InputForm]]';
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

  // Numeric-tolerance comparison used for APPROX_MATCH cases (see the note on
  // that set). Both operands are InputForm strings. DateObjects are compared by
  // the AbsoluteTime of their leading date list (so Minute/Instant granularity
  // and the calendar/time-zone tail are ignored); everything else must share
  // the same number-blanked skeleton, then each number is compared within a
  // relative/absolute tolerance.
  // The helpers live in their own context: every case starts with
  // `ClearAll["Global`*"]`, which would take them with it.
  lines.push("WX`DateTol = 90");
  lines.push("WX`RelTol = 0.02");
  lines.push("WX`AbsTol = 0.01");
  lines.push(
    "WX`Nums[s$_] := ToExpression /@ StringCases[s$, NumberString]"
  );
  lines.push(
    'WX`Skeleton[s$_] := StringReplace[s$, NumberString -> "#"]'
  );
  lines.push(
    'WX`DateTime[s$_] := AbsoluteTime[ToExpression /@ StringCases[' +
      'StringTake[s$, First[StringPosition[s$, "{"]][[1]] ;; ' +
      'First[StringPosition[s$, "}"]][[1]]], NumberString]]'
  );
  lines.push(
    "WX`ApproxQ[woxi$_, ws$_] := Module[{a$, b$}," +
      ' If[StringContainsQ[woxi$, "DateObject"] && StringContainsQ[ws$, "DateObject"],' +
      " Return[TrueQ[Quiet[Check[" +
      "Abs[WX`DateTime[woxi$] - WX`DateTime[ws$]] <= WX`DateTol, False]]]]];" +
      " If[WX`Skeleton[woxi$] =!= WX`Skeleton[ws$], Return[False]];" +
      " a$ = WX`Nums[woxi$]; b$ = WX`Nums[ws$];" +
      " If[Length[a$] =!= Length[b$] || Length[a$] == 0, Return[False]];" +
      " TrueQ[And @@ MapThread[" +
      "Abs[#1 - #2] <= Max[WX`AbsTol, WX`RelTol*Max[Abs[#1], Abs[#2]]] &," +
      " {a$, b$}]]]"
  );

  // Cases that unprotect a symbol and then define it (`Unprotect[Red];
  // Red = 42`) change what that name means for the rest of the batch, where
  // Woxi runs each case in a fresh process. Collect every name any case in
  // this batch unprotects, remember its definitions once up front, and put
  // them back before each case. Restoring — rather than clearing — is what
  // keeps `Red` the colour it was born as.
  const unprotected = new Set<string>();
  for (const { expr } of cases) {
    for (const m of expr.matchAll(/\bUnprotect\[([^\]]*)\]/g)) {
      for (const name of m[1].split(",")) {
        const trimmed = name.trim();
        if (/^[A-Za-z$][A-Za-z0-9$]*$/.test(trimmed)) unprotected.add(trimmed);
      }
    }
  }
  const restores: string[] = [];
  for (const name of unprotected) {
    lines.push(
      'WX`Own["' + name + '"] = Quiet[OwnValues[' + name + "]]",
      'WX`Down["' + name + '"] = Quiet[DownValues[' + name + "]]",
      // Whether the name was born Protected. A built-in (`Red`) has to go
      // back to protected; a plain `x` some case protects along the way
      // must not, or the next `Protect[x]` case answers `{}` where a fresh
      // kernel answers `{"x"}`.
      'WX`Prot["' + name + '"] = Quiet[MemberQ[Attributes[' + name +
        "], Protected]]"
    );
    restores.push(
      "Quiet[Unprotect[" + name + "];" +
        " OwnValues[" + name + '] = WX`Own["' + name + '"];' +
        " DownValues[" + name + '] = WX`Down["' + name + '"];' +
        ' If[WX`Prot["' + name + '"], Protect[' + name + "]]]"
    );
  }

  for (const { expr, woxiResult, idx } of cases) {
    // A `Protect[x]` case leaves `x` protected for the rest of the batch,
    // and `ClearAll` refuses to touch a protected symbol — so a later
    // `Protect[x]` would answer `{}` (nothing left to protect) where a
    // fresh kernel answers `{"x"}`. Unprotect first, then clear.
    lines.push('Quiet[Unprotect["Global`*"]]');
    lines.push('ClearAll["Global`*"]');
    // Woxi runs every case in a fresh process, so the session state a case
    // leaves behind must not reach the next one. `ClearAll` above only
    // empties `Global``; the context machinery keeps its own state.
    lines.push("$ContextAliases = <||>");
    lines.push(...restores);

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
    // Approx cases compare within a numeric tolerance; all others by exact
    // string equality.
    const mismatchTest = APPROX_MATCH.has(expr)
      ? "!WX`ApproxQ[ee$$, rr$$]"
      : "rr$$ =!= ee$$";
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
        " If[" + mismatchTest + "," +
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

// Astronomy: ephemeris-precision divergences. Woxi computes Sun/Moon positions
// and phase/eclipse times from truncated Meeus series; Wolfram uses full
// proprietary ephemerides (VSOP87/ELP plus eclipse catalogs). The values agree
// to a few arcseconds / a few tens of seconds of time but differ in the last
// reported digits, which no rewrite of a truncated series can reproduce
// (Sunrise/Sunset additionally use Woxi's deliberate Minute granularity versus
// Wolfram's Instant). Rather than skip these outright, compare them numerically
// with a tolerance (see WX`ApproxQ in buildWolframScript): DateObjects within
// WX`DateTol seconds, other numbers within a relative/absolute tolerance, once
// the non-numeric "skeleton" of the two InputForms matches. A real regression
// (wrong structure, or a value off by more than the tolerance) still fails.
const APPROX_MATCH = new Set([
  "MoonPhase[DateObject[{2024, 1, 20, 12, 0, 0}]]",
  "MoonPhase[{2024, 1, 25}]",
  "NewMoon[DateObject[{2024, 4, 1}]]",
  "FullMoon[DateObject[{2024, 1, 1}]]",
  'MoonPhaseDate[DateObject[{2024, 4, 1}], "FullMoon"]',
  "SunPosition[{52.52, 13.405}, DateObject[{2024, 12, 21, 12, 0, 0}]]",
  "Sunrise[GeoPosition[{52.52, 13.405}], DateObject[{2024, 6, 21}]]",
  "Sunset[GeoPosition[{52.52, 13.405}], DateObject[{2024, 6, 21}]]",
  "Sunrise[GeoPosition[{52.52, 13.405}], {2024, 12, 21}]",
  "Sunset[{0, 0}, {2024, 3, 20}]",
  "SolarEclipse[DateObject[{2024, 4, 1}]]",
  "LunarEclipse[DateObject[{2025, 1, 1}]]",
  // An iterated root differs in the last digit or two: where the two solvers
  // stop is their own business. The root of `J0'` is 3.8317059702075123156…,
  // which Woxi reports as 3.831705970207513 and Wolfram as
  // 3.831705970207511 — one and three units in the last place off the
  // correctly rounded double, in opposite directions.
  "FindRoot[D[BesselJ[0, r], r] == 0, {r, 3}]",
]);

/**
 * stderr of the most recent wolframscript batch. Kept so a failing batch can
 * still report what wolframscript complained about (license flakes, hard
 * kernel errors) even though stderr is no longer forwarded to our terminal.
 */
let lastWolframStderr = "";

/** Keep a stderr dump readable: qhull alone emits ~50 lines per degenerate hull. */
function truncateStderr(stderr: string, maxLines = 40): string {
  const lines = stderr.trimEnd().split("\n");
  return lines.length <= maxLines
    ? lines.join("\n")
    : [
        ...lines.slice(0, maxLines),
        `… (${lines.length - maxLines} more stderr lines omitted)`,
      ].join("\n");
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
  // spawnSync (rather than execSync) so wolframscript's stderr is captured
  // instead of forwarded straight to our own stderr. Wolfram's bundled
  // libraries write diagnostics there that are not test failures — e.g. qhull's
  // multi-page "Initial simplex is flat" report for degenerate ConvexHullMesh /
  // DelaunayMesh inputs — and they would otherwise bury the progress output.
  // The captured text is still reported whenever a batch actually fails.
  const res = spawnSync(
    `wolframscript -charset UTF8 -code ${shellQuoteForExec(wolframProgram)}`,
    {
      shell: true,
      encoding: "utf-8",
      timeout: timeoutMs,
      maxBuffer: 10 * 1024 * 1024,
      killSignal: "SIGKILL", // SIGTERM is ignored by wolframscript during computation
    }
  );
  lastWolframStderr = res.stderr ?? "";
  if (res.error) {
    const err: any = new Error(res.error.message || "wolframscript batch failed");
    // Distinguish "took too long" from "died". A timeout means the batch was
    // merely too slow and can be retried in smaller pieces; anything else is a
    // crash or a cold-start flake (see runBatchResilient).
    err.timedOut =
      (res.error as any).code === "ETIMEDOUT" ||
      /ETIMEDOUT/.test(res.error.message ?? "");
    throw err;
  }
  if (res.signal || res.status !== 0) {
    throw new Error(
      lastWolframStderr.trim() ||
        `wolframscript exited with ${res.signal ?? res.status}`
    );
  }
  return res.stdout ?? "";
}

type BatchCase = { expr: string; woxiResult: string; idx: number };

/**
 * How long a (sub-)batch may take. Some single expressions are legitimately
 * slow in wolframscript — `SurfaceArea[SphericalShell[]]` alone takes ~30 s —
 * so even a one-case batch gets minutes, while a full batch gets proportionally
 * more. Anything above the ceiling is a hang, not slowness.
 */
function batchTimeoutMs(caseCount: number): number {
  // Flat override, mainly to exercise the split/hang handling below on demand:
  //   WX_BATCH_TIMEOUT_MS=1000 WX_ONLY=… node tests/wolframscript/verify_unit_tests.ts
  const override = Number(process.env.WX_BATCH_TIMEOUT_MS);
  if (Number.isFinite(override) && override > 0) return override;
  return Math.min(600_000, Math.max(240_000, 12_000 * caseCount));
}

/** True iff a batch run reached its `Print["DONE"]` sentinel. */
function hasDoneSentinel(output: string): boolean {
  return output
    .trim()
    .split("\n")
    .some((l) => l.trim() === "DONE");
}

/** A single expression that wolframscript cannot finish on its own. */
class HangingCaseError extends Error {
  entry: BatchCase;
  timeoutMs: number;
  constructor(entry: BatchCase, timeoutMs: number) {
    super(`wolframscript hangs on: ${entry.expr}`);
    this.entry = entry;
    this.timeoutMs = timeoutMs;
  }
}

/**
 * A timed-out batch is killed with SIGKILL, which can leave its WolframKernel
 * child running and burning CPU — every following batch would then be slower
 * than the one that already timed out, cascading into more false timeouts.
 *
 * Only kernels reparented to init (i.e. whose wolframscript is gone) are
 * killed; a blanket `pkill -f WolframKernel` would also take out a kernel a
 * concurrently running wolframscript still owns, which surfaces as a spurious
 * "installation is not activated" flake on the next batch.
 */
function reapOrphanedKernels(): void {
  const found = spawnSync("pgrep", ["-P", "1", "-f", "WolframKernel"], {
    encoding: "utf-8",
  });
  const pids = (found.stdout ?? "").split("\n").filter((l) => /^\d+$/.test(l.trim()));
  if (pids.length === 0) return;
  console.error(`Reaping ${pids.length} orphaned WolframKernel process(es).`);
  spawnSync("kill", ["-9", ...pids], { stdio: "ignore" });
}

/** `cases #12–#37 (26)`, using the same numbers as the failure reports. */
function rangeLabel(batch: BatchCase[]): string {
  const first = batch[0].idx + 1;
  const last = batch[batch.length - 1].idx + 1;
  return batch.length === 1
    ? `case #${first}`
    : `cases #${first}–#${last} (${batch.length})`;
}

const FLAKE_RETRIES = 3;

/**
 * Run a batch through wolframscript, collecting its output lines.
 *
 * Failures are handled by kind rather than by bisecting for a "culprit":
 * bisection with a short timeout blames whichever expression happens to be
 * slowest, even when the batch merely exceeded its budget by being slow all
 * over. Instead a timed-out batch is split in half and both halves rerun with
 * their own (still generous) budget, so slow-but-finite work simply completes.
 * Only a single case that cannot finish alone is a genuine hang, and only that
 * aborts the run. Non-timeout failures are wolframscript cold-start flakes
 * (transient "not activated" / SIGKILL / empty output) and are retried.
 */
function runBatchResilient(
  batch: BatchCase[],
  failures: string[],
  label = rangeLabel(batch)
): string[] {
  const timeoutMs = batchTimeoutMs(batch.length);
  for (let flakeAttempt = 0; ; flakeAttempt++) {
    let output = "";
    let crashErr = "";
    let timedOut = false;
    try {
      output = runWolframBatch(batch, timeoutMs);
    } catch (err: any) {
      crashErr = err.message || String(err);
      timedOut = err.timedOut === true;
      output = "";
    }
    if (!crashErr && hasDoneSentinel(output)) {
      return output.trim().split("\n");
    }

    const reason = crashErr
      ? timedOut
        ? `timed out after ${timeoutMs / 1000}s`
        : `crashed: ${crashErr}`
      : output.trim() === ""
        ? "produced no output"
        : "did not contain DONE sentinel";
    console.error(`\n${label} ${reason}.`);
    if (!crashErr && output.trim()) {
      console.error(`wolframscript output:\n${output}`);
    }
    // stderr is captured rather than forwarded (see runWolframBatch), so echo
    // it here — it carries the license/activation flake messages and kernel
    // errors that explain a batch without a DONE sentinel.
    if (
      lastWolframStderr.trim() &&
      !crashErr.includes(lastWolframStderr.trim())
    ) {
      console.error(`wolframscript stderr:\n${truncateStderr(lastWolframStderr)}`);
    }

    if (timedOut) {
      reapOrphanedKernels();
      if (batch.length === 1) throw new HangingCaseError(batch[0], timeoutMs);
      break;
    }
    if (flakeAttempt < FLAKE_RETRIES) {
      console.error(
        `Retrying (attempt ${flakeAttempt + 2}/${FLAKE_RETRIES + 1})…`
      );
      continue;
    }
    // Persistent non-timeout failure. Splitting isolates it; a lone case that
    // still refuses to run is recorded rather than aborting the whole run, so
    // genuine mismatches in later batches still surface.
    if (batch.length === 1) {
      failures.push(
        `FLAKY case #${batch[0].idx + 1} never produced DONE: ${batch[0].expr}`
      );
      return [];
    }
    break;
  }

  const mid = Math.floor(batch.length / 2);
  const first = batch.slice(0, mid);
  const second = batch.slice(mid);
  console.error(
    `Splitting into ${rangeLabel(first)} and ${rangeLabel(second)} and rerunning…`
  );
  return [
    ...runBatchResilient(first, failures),
    ...runBatchResilient(second, failures),
  ];
}

/** Shell-quote a string for use as a -code argument. */
function shellQuoteForExec(s: string): string {
  return "'" + s.replace(/'/g, "'\\''") + "'";
}

function main() {
  // A whole run takes hours, so `WX_ONLY` narrows it to the test files whose
  // path contains any of the given substrings — enough to re-verify one fix:
  //   WX_ONLY=interpreter_tests/calculus.rs,interpreter_tests/algebra.rs \
  //     npx tsx tests/wolframscript/verify_unit_tests.ts
  const only = (process.env.WX_ONLY ?? "").split(",").filter(Boolean);
  const testFiles = listRustFiles(join(ROOT, "tests"))
    .filter((f) => only.length === 0 || only.some((o) => f.includes(o)))
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
    /\bDotProduct\[/,         // VectorAnalysis package — not loaded by default in Wolfram
    /\bCrossProduct\[/,       // VectorAnalysis package — not loaded by default in Wolfram
    /\bScalarTripleProduct\[/, // VectorAnalysis package — not loaded by default in Wolfram
    /\bCoordinatesToCartesian\[/, // VectorAnalysis package — not loaded by default in Wolfram
    /\bCoordinatesFromCartesian\[/, // VectorAnalysis package — not loaded by default in Wolfram
    /^Coordinates\[/,         // VectorAnalysis package — not loaded by default in Wolfram
    /\bSetCoordinates\[/,     // VectorAnalysis package — not loaded by default in Wolfram
    /\bWriteString\[\s*"stdout"/, // WriteString to stdout pollutes the verify harness's stdout capture
    /\bWriteString\[\s*"stderr"/, // Same as above
    /\bFindFile\[/,           // Path lookups depend on Mathematica install location

    /\bStack\[/,        // Returns internal evaluation stack (different call frames per implementation)
    /\bRasterize\[/,
    /\bN\[Erf\[/,    // Arbitrary-precision Erf differs in low-order digits (different algorithm)
    /\bN\[Erfc\[/,   // Same as Erf (Erfc = 1 - Erf)
    /\bAdjacencyMatrix\[/,  // Woxi returns dense list, Wolfram returns SparseArray
    /\bAdjacencyGraph\[/,   // Different internal Graph representation (edge list vs SparseArray)
    /\bGraphEmbedding\[/,   // Different layout algorithms produce different coordinates
    /\bBezierFunction\[/,   // Different internal representation of BezierFunction objects
    /\bN\[BesselJZero\[/,   // Last-ULP floating-point differences in root finding
    /\bBodePlot\[/,         // Complex Graphics output, implementation-specific rendering
    /\bDefaultButton\[/,    // Complex UI rendering (Button with Dynamic)
    /\bParallelSubmit\[/,   // Returns EvaluationObject with internal state
    /\bTimelinePlot\[/,     // Complex Graphics output, implementation-specific rendering
    /\bAngularGauge\[/,     // Complex Graphics output, implementation-specific rendering
    /\bSmoothDensityHistogram\[/, // Complex Graphics output, implementation-specific rendering
    /\bServiceConnect\[/,   // Network-dependent Failure result
    /\bNetGraph\[/,         // Neural network internals differ between implementations
    /\bStreams\[/, // Woxi returns hardcoded stdout/stderr, Wolfram includes temp file streams
    /\bConnectedComponents\[/, // Vertex ordering within components is implementation-specific
    /\bStarGraph\[/,         // Internal Graph representation differs (edge list vs SparseArray)
    /\bCompleteGraph\[/,     // Internal Graph representation differs (edge list vs SparseArray)
    /\bCrossMatrix\[/,       // Woxi returns dense list, Wolfram returns SparseArray
    /\bSymmetrize\[/,        // Woxi returns dense list, Wolfram returns SymmetrizedArray
    /\bTensorWedge\[/,      // Woxi returns dense list, Wolfram returns SymmetrizedArray
    /\bVertexAdd\[/,        // Returns Graph object (edge list vs SparseArray representation)
    /\bIndexGraph\[/,       // Returns Graph object (edge list vs SparseArray representation)
    /\bConnectedGraphComponents\[/, // Returns Graph objects (edge list vs SparseArray representation)
    /\bFindSpanningTree\[/, // Wolfram uses SparseArray internal Graph representation
    /\bTransitiveReductionGraph\[/, // Returns Graph object (edge list vs SparseArray representation); CLI display Graph[<3>, <2>] matches
    /\bTransitiveClosureGraph\[/, // Returns Graph object (edge list vs SparseArray representation); CLI display matches
    /\bMersennePrimeExponent\[/, // Woxi uses a lookup table; Wolfram computes primality of 2^p-1 which hangs for large indices
    /\bStationaryDistribution\[/, // Complex computation, Woxi keeps as inert wrapper
    /\bDatedUnit\[/,        // Version-specific evaluation behavior
    /\bVoronoiMesh\[/,      // Different bounding box and vertex coordinates
    /\bEntityStores\[/,     // EntityStores accumulates across the batch session; ClearAll doesn't reset the global registry
    /\bEntityUnregister\[/, // Depends on EntityStores isolation (prior registrations persist in wolframscript batch)
    /\bQuantity/,           // Wolfram's unit interpretation uses an online entity framework
                            // (Interpreter["Unit"]) that requires internet and produces
                            // flaky results in batch mode. All Quantity, UnitConvert,
                            // CompatibleUnitQ, QuantityMagnitude, QuantityUnit expressions
                            // are covered by Woxi's own 167 unit tests instead.
    /\bInput\[/,            // Interactive: in `wolframscript -code` batch mode Input[]
                            // blocks on stdin and prevents the DONE sentinel from being
                            // reached. Woxi's script-mode EndOfFile behavior is covered
                            // by the input_function unit tests in tests/interpreter_tests/io.rs.
    /\bInputString\[/,      // Same as Input[] — blocks on stdin in batch mode.
    /\bCantorMesh\[/,       // Wolfram returns an opaque MeshRegion[...] whose Part / Head
                            // round-trip via InputForm with the original head intact;
                            // Woxi exposes the underlying {vertices, polygons} data.
    /\bGridGraph\[/,        // Internal Graph representation differs (edge list vs SparseArray)
    /\bImageConvolve\[/,    // Last-ULP floating-point differences between filter algorithms
    /\bImageCorrelate\[/,   // Shares ImageConvolve's filter, so the same last-ULP differences
    /\bColorBalance\[/,     // Woxi implements the documented "LMSScaling" (von Kries/Bradford)
                            // adaptation; Wolfram's reference white and matrix variant are
                            // internal, so the channel values agree in shape, not bit-for-bit.
    /\bHistogramTransform\[/, // Woxi maps the empirical CDF directly; Wolfram bins integer
                            // images into an internal fixed-level histogram first, so the
                            // equalized values differ by the binning.
    /\bDominantColors\[/,   // Implementation-specific color algorithm: Woxi uses a GrayLevel
                            // ramp on single-channel inputs, Wolfram hashes labels into RGB.
    /\bCenteredInterval\[/, // Internal representation differs: Woxi stores {value, radius}
                            // pairs, Wolfram stores an arbitrary-precision ball encoding
                            // (e.g. {{5, 0, 536870912, -29}, 63}).
    /\bCloudExport\[/,      // Cloud-dependent: Wolfram uploads to wolframcloud.com and
                            // returns a CloudObject URL; Woxi keeps the call symbolic.
    /\bPlay\[/,            // Wolfram compiles Play[] into a Sound[SampledSoundFunction[...]]
                            // with a CompiledFunction body; Woxi wraps the inert Play[] in a
                            // Sound object that renders as -Sound-, so the printed forms differ.

    // ── Whole subsystems whose canonical form is an implementation-specific
    // internal representation that can never match byte-for-byte ──────────────

    // Audio objects: Wolfram stores samples as a quantized NumericArray[...,
    // "Real32"] and prints the full option list (Appearance, AudioOutputDevice,
    // SampleRate, SoundVolume); Woxi keeps the plain {samples} + SampleRate form
    // (and matches Wolfram byte-for-byte on WAV export). Measurement/statistic
    // divergences (channel wrapping {0.5} vs 0.5, int-vs-real, last-ULP filter
    // coefficients) all stem from this same internal-representation choice.
    /\bAudio\[/,
    /\bAudioCapture\[/,     // Wolfram keeps AudioCapture[] symbolic (needs a device);
                            // Woxi returns $Failed with no capture device.
    /\bWebAudioSearch\[/,   // Network-dependent; Wolfram returns Failure[...], Woxi $Failed.

    // AssessmentFunction / QuestionObject: Wolfram's AssessmentResultObject and
    // AssessmentFunction embed a live Timestamp -> DateObject[<now>] plus a rich
    // settings association, so the printed form is non-deterministic and can
    // never match; Woxi keeps a simplified deterministic form.
    /\bAssessmentFunction\[/,
    /\bQuestionObject\[/,

    // Molecule objects: Wolfram stores atoms as bare element strings ("O") with a
    // trailing bond-stereo slot ({}), and renders MolecularFormula as a
    // Row[{Subscript[...]}] box; Woxi wraps atoms in Atom["O"], adds implicit
    // hydrogens, and returns a plain "C8H10N4O2" string. Internal representation.
    /\bMolecule/,

    // Wavelet subsystem: Wolfram renders filter coefficients as machine reals
    // (0.5) where Woxi keeps exact rationals (1/2), wraps DiscreteWaveletData
    // values one level deeper ({{...}} vs {...}), carries default arguments
    // (SymletWavelet[4], CDFWavelet["9/7"], MexicanHatWavelet[1]) and diverges by
    // last-ULP in the transforms. Value-correct, representation-divergent.
    /Wavelet/,

    // TimeSeries: Wolfram's TimeSeries stores an internal temporal-path object
    // whose Length is the number of data points and whose values are coerced to
    // reals; Normal renders dates as DateObject[..., "Day"] granularity. Woxi's
    // literal {{t, v}, ...} model diverges on Length, Mean type, and date form.
    /\bTimeSeries\[/,

    // DateInterval: Wolfram appends two trailing calendar/timezone slots
    // (Automatic, Automatic or "UT", Automatic) that Woxi's 4-argument canonical
    // form omits. See project_dateinterval_canonical_form.
    /\bDateInterval\[/,

    // WikidataData: network-dependent knowledge-base lookups; Wolfram returns
    // arbitrary-precision quantities / DateObject granularity that Woxi's cached
    // data rounds differently.
    /\bWikidataData\[/,

    // FindShortestCurve: geodesic tie-breaking (which semicircle) and int-vs-real
    // coordinate preservation are implementation-specific; the numeric Annulus
    // solver is unimplemented.
    /\bFindShortestCurve\[/,

    // Legacy packages Woxi implements directly under their qualified names
    // (it has no package system for `Needs` to load into), so wolframscript
    // leaves them unevaluated unless the package happens to be loaded — the
    // same situation as the VectorAnalysis entries above.
    /\bCombinatorica`/,
    /\bPolyhedronOperations`/,

    // Ephemeris positions: Woxi computes them from its own bundled VSOP/ELP
    // truncations, wolframscript from its full series, so the two agree to
    // ~1e-4 degrees rather than exactly.
    /\b(SunPosition|MoonPosition|MoonPhaseDate|SiderealTime)\[/,

    // ShortTimeFourier: Woxi partitions with a different default offset and
    // names the partition properties differently ("WindowSize"/"Offset" vs
    // wolframscript's "PartitionSize"/"PartitionOffset"), so neither the
    // frame count nor the property vocabulary lines up.
    /\bShortTimeFourier\[/,

    // GeoRegionValuePlot: same family as the Head[GeoGraphics] entry — Woxi
    // returns bare Graphics with the colour scale drawn into the picture,
    // wolframscript a Legended[GeoGraphics[…], Placed[BarLegend[…], …]].
    /\bGeoRegionValuePlot\[/,
  ];

  // Specific expressions where Woxi is more accurate than Wolfram.
  // NSolve cubic: Woxi gives exact integer roots (1.) via symbolic solving,
  // while Wolfram's companion-matrix eigenvalues introduce machine-epsilon
  // artifacts (1.0000000000000002).
  const EXACT_EXPR_SKIP = new Set([
    // Attributes[ParallelDo] is non-deterministic in wolframscript: ParallelDo
    // autoloads lazily. On a cold kernel it is a stub with {Protected,
    // ReadProtected}; once the Parallel subsystem initializes (e.g. after any
    // ParallelDo actually runs) the real definition installs {HoldAll,
    // Protected}. There is no single stable reference to conform to, so the
    // fuzzer must not chase it (this expression was flip-flopped twice before).
    // Woxi returns the cold-kernel value, matching the unit test.
    "Attributes[ParallelDo]",
    // Same lazy-autoload non-determinism as Attributes[ParallelDo].
    "Attributes[ParallelSelect]",
    "Attributes[ParallelCases]",
    // The integer in InputStream[String, n] is wolframscript's per-session
    // stream counter: a cold kernel hands out 4, 5, ... and every stream any
    // earlier expression in the batch opened shifts it further. There is no
    // stable reference value; Woxi numbers its own streams from 1.
    'ReadString[StringToStream["abc"], 2]',
    "NSolve[x^3 - 3*x^2 + 2*x == 0, x]",
    // wolframscript leaks its internal System`HarmonicNumberDump`MQHN symbol
    // for a symbolic exponent (a WL bug); Woxi stays unevaluated instead.
    "HyperHarmonicNumber[2, 3, s]",
    // Last-ULP floating-point differences (different summation algorithms at machine epsilon):
    "HypergeometricPFQ[{1, 2}, {3}, 0.5]",
    "HypergeometricPFQ[{1}, {2}, 1.0]",
    "N[HypergeometricPFQ[{1/2}, {3/2}, -1]]",
    "RiemannR[10.]",
    "N[RiemannR[1000000]]",
    // Polynomial factoring: Woxi returns expanded form, Wolfram returns factored
    "FindSequenceFunction[{1, 3, 6, 10, 15}, n]",
    // PiecewiseExpand[Clip]: equivalent but differently ordered Piecewise cases
    "PiecewiseExpand[Clip[x, {0, 10}]]",
    // Woxi evaluates exactly (1.5), Wolfram has floating-point rounding (1.4999999999999998)
    "PDF[BetaDistribution[2, 3], 0.5]",
    // Simplify[trig]: canonical Times factor ordering difference when two factors
    // have the same sort key ("Cos"). Woxi outputs ((1+3*Cos[2*x])*Cos[x]^2)/2,
    // Wolfram outputs (Cos[x]^2*(1+3*Cos[2*x]))/2. Mathematically equivalent.
    "Simplify[2*Cos[x]^4 - Cos[x]^2*Sin[x]^2]",
    // Algebraic form differences: mathematically equivalent but different simplification level
    // Term ordering in Times: E*(-1+E) vs (-1+E)*E
    "Variance[LogNormalDistribution[0, 1]]",
    // Exponent form: k/2-1 vs -1+k/2 (canonical Plus ordering)
    "PDF[ChiSquareDistribution[k], x]",
    // Division vs negative exponent: (a*k^a)/x^(1+a) vs a*k^a*x^(-1-a)
    "PDF[ParetoDistribution[k, a], x]",
    // Nested fraction simplification: 2/3/(3*E^(1/9)) vs 2/(9*E^(1/9))
    "PDF[WeibullDistribution[2, 3], 1]",
    // Exponent form: (a-1) vs (-1+a) (canonical Plus ordering)
    "PDF[WeibullDistribution[a, b], x]",
    // Canonical Plus ordering: (1 - x) vs (-1 + x)
    // Fraction expansion form difference
    "GeneratingFunction[f[n + 1], n, x]",
    // 1/Pi vs Pi^(-1) (canonical Power form)
    "PDF[CauchyDistribution[0, 1], 0]",
    // Term ordering in Times: s^2*(2 - Pi/2) vs (2 - Pi/2)*s^2
    "Variance[RayleighDistribution[s]]",
    // FailureDistribution read-once CDF: value matches, but the two complement
    // factors sort differently — Woxi (1 - E^(-3*t))*(1 - E^(-5*t)), Wolfram
    // (1 - E^(-5*t))*(1 - E^(-3*t)). Same canonical Times factor-ordering
    // divergence as the entries above.
    "CDF[FailureDistribution[(x || y) && (x || z), {{x, ExponentialDistribution[2]}, {y, ExponentialDistribution[3]}, {z, ExponentialDistribution[5]}}], t]",
    // Complex polynomial algebra (not yet implemented)
    "CoefficientRules[x, y]",
    "PolynomialReduce[x, y]",
    "PolynomialGCD[x, y]",
    // Complex transform functions (not yet implemented)
    "ZTransform[x, y, z]",
    "InverseZTransform[x, y, z]",
    "FourierCoefficient[x, y, z]",
    // Complex optimization (not yet implemented)
    "MinValue[x, y]",
    "ArgMax[x, y]",
    // FindArgMin: Woxi gives exact -1.5, Wolfram introduces FP noise -1.5000000000000004
    "FindArgMin[x^2 + 3*x + 2, x]",
    // Higher-order Derivative[n][pure-fn]: Woxi simplifies the nested Times
    // produced by repeated differentiation (6*#1 & vs Wolfram's 3*(2*#1) &),
    // but both represent the same pure function.
    "Derivative[2][#^3&]",
    "Derivative[3][#^3&]",
    // Last-ULP floating-point differences: Woxi is closer to the true value
    // (verified against 25-digit Wolfram precision) but f64 rounds differently
    "AiryAiPrime[1.0]",
    "AiryAiPrime[-1.0]",
    "AiryBiPrime[0.0]",
    "AiryBiPrime[1.0]",
    "AiryBiPrime[-1.0]",
    // Last-ULP floating-point differences in window/filter/prime functions:
    "BlackmanWindow[0.3]",
    "PrimeZetaP[2.0]",
    "N[PrimeZetaP[2]]",
    "PrimeZetaP[3.0]",
    "BandpassFilter[{1, 2, 3, 4, 5}, {0.1, 0.3}]",
    "BandpassFilter[{1, 2, 3, 4, 5}, {0.1, 0.3}, 3]",
    "BandpassFilter[{1, 2, 3, 4, 5}, {0.1, 0.3}, 4]",
    "BandpassFilter[{1, 2, 3, 4, 5, 6, 7, 8}, {0.1, 0.3}]",
    "LowpassFilter[{1, 2, 3, 4, 5}, 0.3]",
    "HighpassFilter[{1, 2, 3, 4, 5}, 0.3]",
    // BandpassFilter symbolic: last-ULP coefficient differences
    "BandpassFilter[{a, b, c}, {0.1, 0.3}]",
    // Times factor ordering: (Cosh+Sinh)*Sin vs Sin*(Cosh+Sinh)
    "ExponentialGeneratingFunction[Sin[n], n, x]",
    // Insphere: algebraic factoring difference. Woxi gives (n+Sqrt[n])^(-1), Wolfram factors to 1/(Sqrt[n]*(1+Sqrt[n]))
    "Insphere[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]",
    "Insphere[Tetrahedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}]]",
    // Variance symbolic: Plus ordering of BesselK terms (different ordering of positive/negative terms)
    "Variance[HyperbolicDistribution[a, b, d, m]]",
    // LogLogistic Variance: canonical Plus/Times ordering. Symbolic form
    // orders the inner Plus terms differently (2 g Csc[...] first vs WL's
    // -(Pi Csc[Pi/g]^2) first); the numeric form places the Pi factor before
    // the Plus (4 Pi (...) vs WL's 4 (...) Pi) because a Constant sorts ahead
    // of a sum by term-priority while an Identifier sorts after it. Both are
    // value-correct; see the variance_symbolic / variance_numeric comments in
    // tests/interpreter_tests/distributions.rs.
    "Variance[LogLogisticDistribution[g, s]]",
    "Variance[LogLogisticDistribution[3, 2]]",
    // FindIntegerNullVector: sign convention is implementation-specific (LLL algorithm produces different signs)
    "FindIntegerNullVector[{2, 6}]",
    // 3D rotation about a symbolic axis: wolframscript returns an unsimplified
    // artifact of its internal Gram-Schmidt rather than a canonical form — the
    // (1,1) entry keeps an un-cancelled x*Conjugate[x]/Abs[x]^2 (i.e. 1) while
    // the (2,2) entry of the same rotation is a plain -1, and a fully symbolic
    // {a, b, c} axis expands to ~8 KB of nested Conjugate[...] terms that are
    // singular on the coordinate axes. There is no stable reference form to
    // conform to, so Woxi leaves the symbolic-axis case unevaluated.
    // See "RotationMatrix / RotationTransform about a symbolic axis" in
    // tests/cli/comparison/mathematica/conformance_gaps.md.
    "RotationTransform[Pi, {0, 0, x}, {1, 0, 0}]",
    // JohnsonDistribution: Plus ordering differences (gamma + delta*f vs delta*f + gamma)
    "PDF[JohnsonDistribution[\"SN\", gamma, delta, mu, sigma], x]",
    "PDF[JohnsonDistribution[\"SU\", gamma, delta, mu, sigma], x]",
    // JohnsonDistribution CDF: Plus ordering in Erfc/Erf argument
    "CDF[JohnsonDistribution[\"SN\", gamma, delta, mu, sigma], x]",
    "CDF[JohnsonDistribution[\"SU\", gamma, delta, mu, sigma], x]",
    // JohnsonDistribution SB numeric PDF: 1/(Sqrt[2*Pi]/4) vs 2*Sqrt[2/Pi] (equivalent, different simplification)
    "PDF[JohnsonDistribution[\"SB\", 0, 1, 0, 1], 1/2]",
    // JohnsonDistribution Mean/Variance: Plus/Times ordering and Sinh expansion differences
    "Mean[JohnsonDistribution[\"SU\", gamma, delta, mu, sigma]]",
    "Mean[JohnsonDistribution[\"SL\", gamma, delta, mu, sigma]]",
    "Variance[JohnsonDistribution[\"SU\", gamma, delta, mu, sigma]]",
    "Variance[JohnsonDistribution[\"SU\", 1, 2, 3, 4]]",
    "Variance[JohnsonDistribution[\"SL\", gamma, delta, mu, sigma]]",
    "Variance[JohnsonDistribution[\"SL\", 0, 1, 0, 1]]",
    // Entity state accumulation: in the batch wolframscript session, EntityStores
    // from prior test cases persist (ClearAll doesn't reset the global registry),
    // so this "unregistered" lookup finds entities from earlier tests.
    "Entity[\"Pet\", \"cat1\"][\"Name\"]",
    // Graph option wrapping differs: Woxi uses bare options, Wolfram wraps in {VertexSize -> {Medium}}
    "Graph[{UndirectedEdge[1, 2]}, VertexSize -> Medium]",
    // Norm[{1.0, 2, 3}]: Wolfram's Norm uses an internal BLAS-style algorithm that
    // produces 3.741657386773941 (1 ULP below correctly-rounded Sqrt[14.]),
    // while Woxi's Sqrt[sum-of-squares] gives the IEEE 754 correct 3.7416573867739413.
    "Norm[{1.0, 2, 3}]",
    // Attributes[Plot3D]: in a fresh wolframscript kernel Plot3D has only
    // {Protected, ReadProtected}; once Plot3D is mentioned (as here), the
    // HoldAll attribute is added automatically. Woxi matches the post-mention
    // state, so the fresh-kernel comparison differs.
    "Attributes[Plot3D]",
    // LegendreP[2, 1, x]: factor ordering in Times differs. Woxi emits
    // -3*Sqrt[1-x^2]*x while Wolfram emits -3*x*Sqrt[1-x^2]. Same value.
    "LegendreP[2, 1, x]",
    // LegendreP[1, 1, 0.5] = -Sqrt[0.75]: Woxi returns the IEEE-correct
    // -0.8660254037844386 (verified against WL's 30-digit value); Wolfram's
    // machine result -0.8660254037844385 is 1 ULP low.
    "LegendreP[1, 1, 0.5]",
    // LaguerreL[5, 2, x]: Woxi returns expanded form, Wolfram returns
    // the factored-over-120 form. Same polynomial.
    "LaguerreL[5, 2, x]",
    // InverseLaplaceTransform of a complex-conjugate pole pair: Woxi
    // returns the real damped oscillation, Wolfram the equivalent sum of
    // complex exponentials its own Simplify settles on. Same function;
    // see conformance_gaps.md.
    "InverseLaplaceTransform[1/(s^3 + 2 s^2 + 5 s), s, t]",
    // Same terms, different canonical Plus order: Wolfram sorts the
    // `Derivative[1][DiracDelta][t]` term last, Woxi first.
    "InverseLaplaceTransform[s^2/(s + 1), s, t]",
    // An `Unevaluated[…]` a pure function's body produced survives into
    // the enclosing expression in both runtimes (`{0, …, 9}` keeps it),
    // but Wolfram only strips a wrapper written *literally* in an
    // argument list, while Woxi's argument-consuming built-ins strip
    // whichever wrapper reaches them. The bare unit test passes; only
    // this harness's `ToString[(…), InputForm]` wrapper — itself such a
    // built-in — sees the difference. See conformance_gaps.md.
    "(Unevaluated[Sequence[#, #^2]]) & [3]",
    // Hold[n_Integer?NonNegative]: PatternTest against a typed pattern;
    // Wolfram renders parens around the typed-pattern, Woxi omits them.
    // Parser/formatter detail, same structure.
    "Hold[n_Integer?NonNegative]",
    // SequenceForm InputForm: Wolfram renders children concatenated without
    // separator, producing a nonstandard InputForm like `"[""x = "56"]"`.
    // Woxi prints the rendered string.
    "SequenceForm[\"[\", \"x = \", 56, \"]\"]",
    // StringForm InputForm: Wolfram preserves the literal backtick escape
    // `\`` as `\`` in InputForm; Woxi double-escapes it.
    "StringForm[\"`` is Global\\`a\", a]",
    // Derivative OutputForm: 2D formatted output with superscript notation
    // differs structurally from Woxi's linear `f^(n)[x]` rendering.
    "ToString[OutputForm[Derivative[3][g][y]]]",
    "ToString[OutputForm[Derivative[4][f][x]]]",
    // ElementData: Woxi returns raw numeric values and plain strings;
    // Wolfram returns Quantity[...] objects and Row[...] for electron
    // configuration. Both forms are valid; implementation-specific output.
    "ElementData[\"He\", \"AbsoluteBoilingPoint\"]",
    "ElementData[\"Carbon\", \"AbsoluteMeltingPoint\"]",
    "ElementData[\"He\", \"ElectroNegativity\"]",
    "ElementData[16, \"ElectronConfigurationString\"]",
    "ElementData[\"Iron\", \"ElectronConfigurationString\"]",
    "ElementData[1, \"ElectronConfigurationString\"]",
    "ElementData[\"He\", \"ElectronConfigurationString\"]",
    "ElementData[\"Tc\", \"SpecificHeat\"]",
    // IonizationEnergies requires Quantity wrapping (same reason as above)
    "ElementData[\"Carbon\", \"IonizationEnergies\"]",
    // Same: Wolfram has Helium IonizationEnergies data wrapped as Quantity[];
    // Woxi only tabulates Hydrogen and Carbon, so Helium correctly returns
    // Missing[NotAvailable]. The unit test exercises that "NotAvailable" path
    // intentionally, so the wolframscript divergence is by design.
    "ElementData[\"Helium\", \"IonizationEnergies\"]",
    // Properties list differs — Woxi exposes the subset it implements,
    // Wolfram exposes its full superset.
    "ElementData[\"Properties\"]",
    // Equivalent[a, False]: Woxi renders as `Not[a]`, Wolfram as prefix `!a`.
    // Semantically identical.
    "Equivalent[a, False]",
    // ParentDirectory: Wolfram only evaluates when the directory actually
    // exists on disk; Woxi does pure string manipulation. Unit tests rely
    // on the string-manipulation form with synthetic paths like /a/b/c.
    "ParentDirectory[\"/a/b/c\"]",
    "ParentDirectory[\"a/b/c\"]",
    // E^(a+I*Pi): Woxi preserves symbolic form (no over-simplification) but
    // Plus ordering differs — Woxi emits `E^(I*Pi + a)`, Wolfram emits
    // `E^(a + I*Pi)`. Semantic fix lives in arithmetic.rs; surface ordering
    // is a broader canonical-Plus issue.
    "E^(a+I Pi)",
    "E^(a+2 I Pi)",
    // ThreeJSymbol valid cases: Woxi only handles degenerate-zero cases;
    // full Racah-formula evaluation (e.g. Sqrt[5/143]) is not implemented.
    "ThreeJSymbol[{2, 0}, {6, 0}, {4, 0}]",
    // Bare Span expressions wrapped in parens (as the verify harness does)
    // fail to parse in Woxi — the Span-sep rules only fire at top level.
    // The direct-interpret unit test passes; the ToString[(expr),InputForm]
    // wrapping used here does not.
    ";; // FullForm",
    "1;;4;;2 // FullForm",
    "2;;-2 // FullForm",
    ";;3 // FullForm",
    // Contexts[]: Wolfram lists hundreds of built-in contexts (Accelerators`,
    // Algebra`, ...) whereas Woxi only exposes System` and Global`. A
    // minimal-context runtime is expected; the list diverges fundamentally.
    "Contexts[]",
    "Contexts[\"Sys*\"]",
    "Contexts[\"*\"]",
    // `?? sym` (Information operator): Wolfram returns an InformationData[...]
    // association with documentation/values metadata; Woxi returns
    // Missing["UnknownSymbol", "name"]. Implementation-specific surface.
    "a + ?? b",
    // Hold[??a + b] — Wolfram parses ?? as a postfix that swallows the
    // entire RHS, producing the bizarre `Information["a", LongForm -> True]
    // *(Plus[b])`. Woxi parses ?? as a unary information query on `a`.
    "Hold[??a + b]",
    // `3.5 I` — Wolfram's REPL prints a pure-imaginary machine real as
    // `0. + 3.5*I` (the inexact-zero Complex form). Woxi shows just
    // `3.5*I`. Both are mathematically the same value.
    "3.5 I",
    // `(I/2)*Pi` parens differ between formatters — Wolfram inserts
    // explicit parens around the rational coefficient, Woxi omits them.
    "ArcCosh[0]",
    "ArcCoth[0]",
    "Log[I]",
    "Exp[I Pi / 3]",
    "Exp[I Pi / 6]",
    "Exp[I Pi / 4]",
    // Pattern formatter difference: Wolfram parenthesises typed
    // patterns inside Plus/Times (`(a_.) + (b_)`); Woxi keeps them bare.
    "a_. + b_",
    "a_. - b_",
    "A[a_. + B[b_.*x_]] -> {a, b, x}",
    "p + Condition[1, 2 > 1]",
    // Same parens-formatting issue, with operands swapped so the Condition is
    // on the left of Plus before canonical reordering.
    "Condition[1, 2 > 1] + p",
    "FullForm[Hold[_Integer?NonNegative]]",
    // Trailing-empty Span position: Wolfram renders the implicit empty
    // slot as blank (`Hold[a; Null; ]`); Woxi prints the explicit `Null`
    // symbol (`Hold[a; Null; Null]`). Same FullForm structure, surface only.
    "FullForm[Hold[a ; ;]]",
    // Apart-on-Equation: Woxi formatter still strips quotes when an
    // Equal node has a single comparison and a string operand — the
    // round-tripping path is fixed but the verify run was generated
    // before the fix; harmless follow-up entry.
    // (Already covered by the InputForm comparison fix; left here as a
    // safety net for any remaining tooling differences.)
    // `Integrate[-Infinity, {x, 0, Infinity}]` — Woxi deliberately
    // evaluates this to -Infinity per the comment in calculus.rs:15-18;
    // Wolfram leaves it unevaluated. Design choice on the Woxi side.
    "Integrate[-Infinity, {x, 0, Infinity}]",
    // `$Version` — Woxi sets $Version to "Woxi <git>"; the
    // StringStartsQ check is inherently identity-sensitive.
    "StringStartsQ[$Version, \"Woxi \"]",
    // `Unprotect[Pi]; Clear[Pi]; Attributes[Pi]` — in a fresh wolframscript
    // kernel this returns `{Constant, ReadProtected}` (Clear doesn't
    // remove built-in attributes). The verify batch saw `{}` only because
    // a prior test polluted Pi's attribute state — not actual divergence.
    "Unprotect[Pi]; Clear[Pi]; Attributes[Pi]",
    // Bare top-level Span FullForm — see "Hold[Out[-1]]" comment block;
    // these forms exercise the parser at top level and don't round-trip
    // identically through the Quiet[ToString[(...), InputForm]] wrapper.
    "ToString[FullForm[1 ;; All]]",
    // ReadList with a fresh StringToStream stream — the InputStream ID is
    // session-specific (Woxi starts from 1 in a fresh kernel; wolframscript
    // accumulates IDs across the batch session). Surface form is otherwise
    // identical and the unit test asserts the Woxi-side ID directly.
    "ReadList[StringToStream[\"a 1 b 2\"], {Word, Number}, -1]",
    // `E^(I Pi/n)` paren formatting — same root cause as the existing
    // `Exp[I Pi/n]` block above (Wolfram wraps `(I/n)` in explicit parens
    // when it appears as a coefficient of Pi; Woxi prints `I/n*Pi`).
    // The bare-Power surface form has the same divergence as the
    // function-call form.
    "E^(I Pi/4)",
    "E^(I Pi/3)",
    "E^(I Pi/6)",
    "Gudermannian[Pi I / 4]",
    // MakeBoxes[OutputForm[expr]] / MakeBoxes[expr // OutputForm] —
    // Wolfram's MakeBoxes unwraps the OutputForm[…] head inside the
    // generated `InterpretationBox` (second argument is the underlying
    // expression, not the form-wrapper). Woxi keeps the wrapper visible.
    // Box-structure is otherwise identical.
    "MakeBoxes[Graphics[{Disk[{0,0}, 1]}]//OutputForm]",
    "MakeBoxes[Graphics3D[{Sphere[{0,0,0}, 1]}]//OutputForm]",
    "MakeBoxes[OutputForm[3.142`3]]",
    "MakeBoxes[OutputForm[3.14`5]]",
    // MakeBoxes[Format[F[x], <form>]] — Wolfram emits the `#1` pure-function
    // slot quoted as `"#1" &` inside the TagBox; Woxi emits it bare as `#1 &`.
    // For TraditionalForm, Wolfram also renders `F[x]` as `F(x)` (parens
    // instead of brackets) — surface rendering of the inner FormBox.
    "MakeBoxes[Format[F[x], StandardForm]]",
    "MakeBoxes[Format[F[x], TraditionalForm]]",
    // Colorize — implementation-specific color algorithm. Wolfram hashes
    // integer labels to RGB triplets (`UnsignedInteger8`, ColorSpace -> "RGB");
    // Woxi maps to a Real64 grayscale ramp. Both produce a valid Image; the
    // unit tests assert against the `-Image-` placeholder only.
    "Colorize[{{1, 2}, {2, 2}, {2, 3}}, ColorFunction -> (Blend[{White, Blue}, #]&)]",
    "Colorize[{{1, 2}, {3, 4}}]",
    // `N[c, p_?(#>10&)] := p; N[c, 11]` — Wolfram's NValues mechanism
    // coerces the rule's return to Real (`11.`) because the outer call is
    // `N[…]`; Woxi returns the bound pattern variable as-is (`11`). The
    // unit test asserts Woxi's integer-passthrough behavior.
    "N[c, p_?(#>10&)] := p; N[c, 11]",
    // Reduce[Exists[…], a] — Woxi proves the inner system has a witness for
    // every real a and returns True; Wolfram preserves the implicit domain
    // marker and returns Element[a, Reals]. Both descriptions are equivalent
    // over the reals; surface-form divergence in the Reduce result.
    "Reduce[Exists[{x, y}, x^2 + a*y^2 <= 1 && x - y >= 1], a]",
    // `Sin[x_] := y` — Sin is Protected on a fresh kernel, so this fails
    // with SetDelayed::write and returns $Failed. Inside the verify batch a
    // prior `Unprotect[Sin]` test leaves Sin unprotected, so wolframscript
    // sets a DownValue and returns Null. The Woxi unit test asserts the
    // fresh-kernel $Failed behavior.
    "Sin[x_] := y",
    // Attributes[Manipulate]: in a fresh wolframscript kernel Manipulate has
    // only {Protected, ReadProtected}; once Manipulate is mentioned, HoldAll
    // is added automatically. Same root cause as Attributes[Plot3D] above.
    "Attributes[Manipulate]",
    // Attributes[FunctionInterpolation]: in a fresh wolframscript kernel it has
    // only {Protected, ReadProtected} (Woxi matches this); once
    // FunctionInterpolation is used the autoloaded definition adds HoldAll.
    // Same batch-pollution root cause as Attributes[Plot3D]/Attributes[Manipulate].
    "Attributes[FunctionInterpolation]",
    // Expectation[x*y, Distributed[{x, y}, BinormalDistribution[r]]] = r;
    // Wolfram computes the covariance/correlation moment directly, Woxi
    // keeps the call symbolic (no joint-distribution evaluator yet).
    "Expectation[x*y, Distributed[{x, y}, BinormalDistribution[1/3]]]",
    // InverseFunction[(a*#1 + b)/(c*#1 + d) &] — Möbius inverse. Surface form
    // differs only in Times factor ordering: Woxi `#1*d` vs Wolfram `d*#1`.
    "InverseFunction[(a*#1 + b)/(c*#1 + d) &]",
    // Multinomial[3, x] — Woxi keeps the Binomial[3+x, x] reduction; Wolfram
    // expands to the polynomial ((1+x)(2+x)(3+x))/6. Mathematically equal.
    "Multinomial[3, x]",
    // Series[x!!, {x, 0, 2}] — Woxi factors the third-order coefficient
    // (`6*(EulerGamma - Log[2])^2 + Pi^2*(1 + Log[64] - 6*Log[Pi])`), Wolfram
    // expands it. Same value, different surface.
    "Series[x!!, {x, 0, 2}]",
    // Series[Pochhammer[x, 1/2], {x, 0, 2}] — Woxi uses the closed form
    // `-Sqrt[Pi]*Log[4]` for the linear coefficient; Wolfram leaves it as
    // `Sqrt[Pi]*(EulerGamma + PolyGamma[0, 1/2])`. These are equal:
    // EulerGamma + PolyGamma[0, 1/2] = -Log[4].
    "Series[Pochhammer[x, 1/2], {x, 0, 2}]",
    // GaussianFilter — last-ULP floating-point difference at the kernel
    // sampling points (Woxi 0.09938048320860668 vs Wolfram 0.09938048320860672).
    "GaussianFilter[{0., 0., 1., 0., 0.}, 1]",
    // Series[BarnesG[x], {x, 0, 2}] — Plus ordering inside the linear
    // coefficient: Woxi `(-1 + Log[2*Pi])/2 + EulerGamma`, Wolfram
    // `EulerGamma + (-1 + Log[2*Pi])/2`.
    "Series[BarnesG[x], {x, 0, 2}]",
    // WeberE[v, 0] = (1 - Cos[Pi*v]) / (Pi*v); Wolfram rewrites this as
    // (Pi*v*Sinc[(Pi*v)/2]^2)/2 via the half-angle identity. Equivalent.
    "WeberE[v, 0]",
    // AngerJ[v, 0] = Sin[Pi*v]/(Pi*v); Wolfram folds this to Sinc[Pi*v].
    "AngerJ[v, 0]",
    // CDF[StudentTDistribution[v], 0] — Woxi uses the symmetry shortcut
    // (the StudentT distribution is symmetric about 0, so the CDF at 0 is
    // exactly 1/2 for any v > 0). Wolfram leaves the BetaRegularized form.
    "CDF[StudentTDistribution[v], 0]",
    // Around[x, Scaled[0.1]] — Wolfram resolves Scaled[0.1] to 0.1*x at
    // evaluation time; Woxi keeps the Scaled[] uncertainty marker symbolic.
    "Around[x, Scaled[0.1]]",
    // Median[ChiDistribution[3]] — algebraic surface difference: Woxi gives
    // Sqrt[2]*Sqrt[InverseGammaRegularized[3/2, 0, 1/2]], Wolfram fuses the
    // square roots to Sqrt[2*InverseGammaRegularized[3/2, 0, 1/2]].
    "Median[ChiDistribution[3]]",
    // Benini PDF/CDF with symbolic α: value-correct form divergence. Wolfram
    // keeps the survival factor as a single exponential
    // E^(-(a*Log[x/s]) - b*Log[x/s]^2); Woxi's core evaluator folds
    // E^(-a*Log[x/s]) back into (x/s)^(-a), yielding the equivalent
    // 1/(E^(b*Log[x/s]^2)*(x/s)^a). Reconciling would require suppressing the
    // general E^(k*Log[y]) -> y^k normalization (a risky core-normalizer change).
    "PDF[BeniniDistribution[a, b, s], x]",
    "CDF[BeniniDistribution[a, b, s], x]",
    // Distribution/geometry closed forms Woxi keeps unevaluated where Wolfram
    // computes a special-function or numerically-factored result. Each is a
    // scoped feature gap documented at the corresponding unit test:
    //  - Coxian PDF with a repeated phase rate needs Erlang-style terms and
    //    Wolfram's Together-factored form (same rabbit hole as the
    //    Hypoexponential {2, 2} case below).
    "PDF[CoxianDistribution[{1/2, 1/3}, {2, 2, 3}], x]",
    //  - Hypoexponential PDF with repeated rates: Wolfram folds the Erlang
    //    terms into a single E^(-max*x)-denominator fraction (e.g.
    //    (12*(1 - E^x + E^x*x))/E^(3*x)); matching that factoring is a
    //    Together/Simplify form-divergence rabbit hole.
    "PDF[HypoexponentialDistribution[{2, 2}], x]",
    //  - TsallisQGaussian CDF for q != 1 needs the incomplete-Beta /
    //    hypergeometric machinery Woxi has not wired for the CDF.
    "CDF[TsallisQGaussianDistribution[0, 2, 3/2], x]",
    //  - Hoyt CDF is expressed via MarcumQ, which Woxi does not implement.
    "CDF[HoytDistribution[q, w], x]",
    //  - FirstPassageTime PDF with a symbolic argument needs the Markov-chain
    //    eigendecomposition closed form (2^(-x) for this 2-state chain).
    "PDF[FirstPassageTimeDistribution[DiscreteMarkovProcess[1, {{1/2, 1/2}, {1/3, 2/3}}], 2], x]",
    //  - DiskSegment perimeter for an *elliptical* disk needs the elliptic
    //    integral EllipticE (6 + 4*EllipticE[-5/4]); the circular cases match.
    "Perimeter[DiskSegment[{0, 0}, {3, 2}, {0, Pi}]]",
    //  - HalfSpace symbolic membership requires Wolfram's Reduce
    //    linear-inequality canonicalization (Element[x | y, Reals] && x <= 2,
    //    with sign-flip / coefficient-division normalization for general n).
    "RegionMember[HalfSpace[{1, 0}, 2], {x, y}]",
    //  - RegionMoment for a Polygon: Wolfram uses numerical quadrature whose
    //    last-ULP noise (e.g. {2,0} -> 0.08333333333333331, off from exact
    //    1/12) cannot be reproduced; Woxi keeps polygon moments exact/inert.
    "RegionMoment[Polygon[{{0, 0}, {1, 0}, {0, 1}}], {1, 1}]",

    // ───────────────────────────────────────────────────────────────────────
    // CAS capabilities Woxi returns unevaluated where Wolfram computes a closed
    // form (parametric/underdetermined solving, nonlinear ODEs, symbolic
    // transforms/sums, Piecewise/DiscreteDelta results, sum-to-product
    // factoring). Woxi's own unit tests assert the unevaluated/symbolic result.
    "Solve[x + y == 3]", // underdetermined linear system
    // 2F1 closed forms Woxi keeps as HypergeometricPFQ: Wolfram reduces
    // 2F1(1/2, 1/2; 3/2; z) -> ArcSin[Sqrt[z]]/Sqrt[z] (a tabulated special
    // identity Woxi does not carry) and 2F1(3, 1; 2; x) -> (2-x)/(2(-1+x)^2)
    // (value-correct via the Euler transform, but the denominator-factoring /
    // power-base sign of the rational form diverges — a Together canonical-form
    // rabbit hole). Woxi's unit tests assert the unevaluated PFQ form.
    "HypergeometricPFQ[{1/2, 1/2}, {3/2}, z]",
    "HypergeometricPFQ[{3, 1}, {2}, x]",
    "GroebnerBasis[{Sin[x]}, {x}]", // non-polynomial generator passthrough
    "PadeApproximant[1/(1 - x), {x, 0, {2, 2}}]", // rank-deficient Padé system
    "TrigFactor[Sin[2 x] + Sin[4 x]]", // sum-to-product factoring
    "DSolve[y'[x] == y[x]^2, y[x], x]", // nonlinear first-order ODE
    "DSolve[y'[x] == Sin[y[x]], y[x], x]",
    "FunctionRange[Gamma[x], x, y]",
    "ZTransform[Sin[n], n, z]",
    "ZTransform[n^2 + n + 1, n, z]",
    "FourierCoefficient[Sin[t], t, n]", // Piecewise result
    "FourierSinCoefficient[Sin[t], t, n]", // DiscreteDelta result
    "Sum[Binomial[n, k], {k, 1, n}]", // symbolic binomial sums
    "Sum[k Binomial[n, k], {k, 0, n}]",
    "Sum[k/k!, {k, 1, Infinity}]",
    "SumConvergence[Sin[n], n]", // n-th-term divergence test
    "HazardFunction[PoissonDistribution[2], x]", // Piecewise result
    "PDF[MultinormalDistribution[{0, 0}, {{2, 1}, {1, 3}}], {x, y}]",
    "EmpiricalDistribution[{x, y}]", // DataDistribution wrapper
    "PDF[MinStableDistribution[2, 3, 0], x]",
    "Covariance[{{a, b}, {c, d}}]", // Conjugate expansion
    "LogLikelihood[ExponentialDistribution[a], {x1, x2}]", // Piecewise result
    "JordanDecomposition[{{1, 1, 0}, {0, 1, 0}, {0, 0, 2}}]",
    "Threshold[{1., -2., 3., 4.}, {\"Firm\", 1, 3}]", // "Firm" shrinkage mode
    // RSolve: non-arithmetic forcing term / non-unit coefficient recurrences;
    // Woxi keeps them symbolic, Wolfram returns the closed-form a[n] -> ...
    "RSolve[a[n] == a[n-1] + n, a[n], n]",
    "RSolve[a[n] == 3 a[n-1] + 1, a[n], n]",
    // DSolve: separable nonlinear first-order ODEs (y' = x y^2, y' = x^2 y^2);
    // kept unevaluated like the existing bare-y^2 skips above.
    "DSolve[y'[x] == x y[x]^2, y[x], x]",
    "DSolve[y'[x] == x^2 y[x]^2, y[x], x]",
    // InverseSurvivalFunction[BetaDistribution[...]]: Wolfram returns a Root
    // object (Head -> Root); Woxi keeps the call symbolic (Head ->
    // InverseSurvivalFunction).
    "Head[InverseSurvivalFunction[BetaDistribution[2, 3], 1/4]]",
    // BooleanFunction index form: Wolfram renders the applied/bare object via
    // its internal "BDD" -> {...} encoding; Woxi keeps the BooleanFunction[i, n]
    // head verbatim.
    "BooleanFunction[7, 2][a, b]",
    "BooleanFunction[7, 2][True]",
    "BooleanFunction[7, 2]",
    // MinMax[Interval[{a, b}]] with symbolic bounds: Wolfram returns the nested
    // {{Interval[...]}, {Interval[...]}} form; Woxi keeps it unevaluated.
    "MinMax[Interval[{a, b}]]",
    // ArrayFilter with a list-valued radius {1}: Woxi leaves it unevaluated;
    // Wolfram applies the filter (the {1} radius is a per-level spec).
    "ArrayFilter[Mean, {1, 2, 3, 4}, {1}]",
    // Gamma[3, 2.0, 5.0]: last-ULP float difference in the incomplete-gamma
    // difference (Woxi ...648, Wolfram ...645).
    "Gamma[3, 2.0, 5.0]",
    // Sum[k 2^k]: divergent, both stay unevaluated; only the Times factor order
    // differs (Woxi k*2^k vs Wolfram 2^k*k) — core canonical-ordering gap.
    "Sum[k 2^k, {k, 1, Infinity}]",

    // Canonical Plus/Times ordering differences (mathematically identical):
    "ZTransform[a^n, n, z]", // -(z/(a - z)) vs z/(-a + z)
    "ZTransform[n^2 a^n, n, z]",
    "Log[E^(a + 3 I)]", // a + 3*I vs 3*I + a (numeric-imaginary term first)
    // Together keeps a complex-conjugate denominator factored
    // ((a - I*x)*(a + I*x)); Woxi multiplies it out to a^2 + x^2.
    "Together[3 (1/(a - I x) + 1/(a + I x))]",
    // Reduce[Cos[x] == 1/2, x]: Woxi wraps each periodic branch in
    // ConditionalExpression; Wolfram factors out Element[C[1], Integers] && (...).
    "Reduce[Cos[x] == 1/2, x]",
    // Cos[x]/Sin[x]: Wolfram canonicalizes to Cot[x]; Woxi keeps the quotient
    // (a separate, form-divergent trig canonicalization Woxi does not perform).
    "Cos[x]/Sin[x]",
    // D[x Sign[x], x]: Plus term ordering — Woxi x*Sign'[x] + Sign[x] vs
    // Wolfram Sign[x] + x*Sign'[x].
    "D[x Sign[x], x]",

    // PermutationProduct prints with the private-use centred-dot infix glyph
    // (U+F3DE); Woxi keeps the call symbolic with the standard head.
    "PermutationProduct[{2, 1, 4}, {1, 3, 2}]",

    // FailureDistribution constructor: wolframscript wraps each normalized
    // event index in an invisible private-use glyph (U+F7D7) in its InputForm
    // output (e.g. "\[F7D7]1 || \[F7D7]2"), so the printed forms can never
    // match byte-for-byte. Woxi emits the plain indices (structurally identical).
    // Same un-matchable rendering artifact as the PermutationProduct glyph above.
    "FailureDistribution[x || y, {{x, ExponentialDistribution[a]}, {y, ExponentialDistribution[b]}}]",

    // Symbol[]-in-Set: Woxi's dynamic_variable_names feature assigns to `xy`
    // (returns 99); Wolfram's Set holds Symbol["xy"] literally so `xy` stays
    // unbound. Intentional Woxi divergence (rosetta_script_fixes). This test is
    // a single interpret() call, so the verify's c.expr is the whole chain.
    "Set[Symbol[\"xy\"], 99]; xy",

    // DeleteCases at level {0} deletes the root → Sequence[]. Wolfram's empty
    // Sequence[] flattens into the verify harness's ToString[(...), InputForm]
    // wrapper, yielding the literal "InputForm"; Woxi returns Null.
    "DeleteCases[{1, a, 2}, _, {0}]",

    // Entity / CountryData: Woxi's bundled entity data differs from Wolfram's
    // online knowledge base (canonical names, population figures, Failure head).
    "Interpreter[\"Country\"][\"USA\"]",
    "Interpreter[\"Country\"][\"Bosnia & Herzegovina\"]",
    "Head[Interpreter[\"Country\"][\"Scotland\"]]",
    "CountryData[\"Qatar\", \"Population\"]",

    // Graph-valued results: Wolfram stores graphs as an internal SparseArray
    // adjacency encoding (and attaches GraphLayout/VertexCoordinates options);
    // Woxi uses an explicit edge list. Same reason as the AdjacencyGraph /
    // StarGraph / CompleteGraph / GridGraph skips above.
    "TransitiveClosureGraph[Graph[{1 -> 2, 2 -> 3}]]",
    "TransitiveClosureGraph[{1 -> 2}]",
    "WeightedAdjacencyGraph[{{Infinity, 2, Infinity}, {2, Infinity, 5}, {Infinity, 5, Infinity}}]",
    "NearestNeighborGraph[{{0, 0}, {1, 0}, {5, 5}, {6, 5}}]",
    "HararyGraph[2, 8]",
    "HararyGraph[4, 9]",
    "HararyGraph[3, 7, PlotLabel -> x]",
    "EdgeConnectivity[CycleGraph[5], 1, 7]",
    "VertexConnectivity[CycleGraph[5], 9, 2]",
    "KCoreComponents[CycleGraph[5], 2, \"Bogus\"]",
    "KCoreComponents[CycleGraph[5], 1.5]",
    "KCoreComponents[CycleGraph[5]]",
    "FindClique[CycleGraph[5], 0]",
    "FindClique[CycleGraph[5], {1, 2, 3}]",
    "Subgraph[CycleGraph[5], {1, 9}]",
    "Subgraph[CycleGraph[5], 3]",
    "AdjacencyList[CycleGraph[5], 9]",
    "EdgeIndex[CycleGraph[5], 1 <-> 3]",

    // Large-real OutputForm: Wolfram renders scientific notation as a 2D
    // `mantissa 10^exp` superscript; Woxi emits the 1D `*^` form.
    "ToString[15000000000.]",
    "ToString[12000000000.]",
    "ToString[2.0*^10]",
    "ToString[123456789012.]",

    // Last-ULP f64 differences at machine precision (different algorithms):
    "SphericalBesselJ[0, {1., 2.}]",
    "StruveH[0, {1., 2.}]",

    // Geodesic last-ULP divergence: geographiclib-rs agrees with WL's own
    // geodesic to ~12 significant figures, but the 15th–17th digit differs
    // because the two libraries use different float implementations. Invisible
    // at map-pixel scale (the only rendering use of these helpers).
    "GeoDistance[{40, -100}, {34, -118}]",
    "GeoDistance[GeoPosition[{40, -100}], GeoPosition[{34, -118}]]",
    "GeoDestination[{40, -100}, {100000, 45}]",
    "GeoLength[GeoPath[{{40, -100}, {34, -118}}]]",
    // GeoBounds: Woxi takes the exact min/max of the input coordinates; WL runs
    // them through its geodesic bounding-box machinery, which injects last-ULP
    // noise (-118.00000000000001 / -99.99999999999999) that can't be reproduced.
    "GeoBounds[{GeoPosition[{40, -100}], GeoPosition[{34, -118}]}]",
    // GeoNearest: Woxi returns the human-readable country name; WL returns its
    // internal canonical entity id (no spaces, "of the" dropped):
    // "UnitedStates", "DemocraticRepublicCongo". Reproducing those exactly needs
    // WL's per-country entity-id table, which is not derivable from the name.
    "GeoNearest[\"Country\", GeoPosition[{40, -100}]]",
    "GeoNearest[\"Country\", GeoPosition[{-2, 23}]]",
    // Head[GeoGraphics[...]]: Woxi renders GeoGraphics to a Graphics object (head
    // Graphics) so it flows through the shared SVG export pipeline; WL keeps the
    // head GeoGraphics. Changing the head would break SVG export.
    "Head[GeoGraphics[Entity[\"Country\", \"France\"]]]",

    // Integrating a Laurent series with a 1/x term needs 1/x -> Log[x], which
    // Woxi leaves unevaluated by design (the series engine has no Log support).
    "Integrate[Series[1/x + x, {x, 0, 3}], x]",
    // Transcendental digamma sum: Sum[1/(9 n^2 - 1)] = (9 - Sqrt[3] Pi)/18.
    // Woxi has no general PolyGamma-difference summation, so it stays symbolic.
    "Sum[1/(9 n^2 - 1), {n, 1, Infinity}]",

    // MakeBoxes of number-display forms: Woxi's box FullForm omits the inner
    // ShowStringCharacters quoting WL stores around the mantissa/exponent string
    // leaves (e.g. "\"1.23456\"" vs "1.23456") and keeps spaces around the "×"
    // operator. The rendered SVG and the computed value are already correct; only
    // the internal box representation differs.
    "MakeBoxes[ScientificForm[12345.6]]",
    "MakeBoxes[EngineeringForm[12345.6]]",
    "MakeBoxes[NumberForm[12345.6]]",
    "MakeBoxes[NumberForm[1234567.8]]",
    "MakeBoxes[NumberForm[3.14159, {6, 2}]]",

    // LyapunovSolve with a NON-diagonal symbolic matrix: WL emits a page-long
    // unsimplified Conjugate quotient from its general symbolic solver. Woxi
    // solves the decoupled diagonal case (which matches WL exactly) and
    // leaves non-diagonal symbolic input unevaluated.
    "LyapunovSolve[{{a, 1}, {0, b}}, {{1, 0}, {0, 1}}]",
    // RotationTransform / RotationMatrix about a symbolic 3D axis {a,b,c}: WL
    // emits a page-long unsimplified Conjugate/Sqrt expression from its generic
    // Householder construction. Woxi has no symbolic-axis rotation builder and
    // leaves it unevaluated; reproducing WL's exact unsimplified form is not
    // feasible.
    "RotationTransform[Pi/2,{a,b,c}]",
    "RotationMatrix[Pi/2,{a,b,c}]",
    // FullSimplify[n!/(n-4)!]: WL's smallest-LeafCount form keeps a Gamma
    // denominator (n!/Gamma[-3+n]) rather than the falling-factorial polynomial.
    // Woxi keeps n!/(-4+n)!; matching WL's Gamma representation choice is a
    // simplification-form rabbit hole.
    "FullSimplify[n! / (n - 4)!]",
    // Integrate[Sqrt[1-x^2],{x,0,2}]: the interval leaves the radical's domain,
    // so WL emits a combined complex closed form ((2*I)*Sqrt[3]+ArcSin[2])/2.
    // Woxi's continuous ArcSin antiderivative is value-correct but stays as a
    // sum of terms (I*Sqrt[3]+ArcSin[2]/2) that doesn't match WL's single
    // fraction; the design deliberately leaves out-of-domain cases unevaluated.
    "Integrate[Sqrt[1 - x^2], {x, 0, 2}]",
    // Sum[1/(2 n - 1), ...]: a divergent harmonic sum both engines echo
    // unevaluated; only the held body's Plus ordering differs (Woxi 2*n - 1 vs
    // WL -1 + 2*n). Woxi's standalone canonical order matches WL, but held
    // (HoldAll) bodies aren't Orderless-sorted — a broad held-expression form
    // difference, not specific to Sum.
    "Sum[1/(2 n - 1), {n, 1, Infinity}]",
    // Residue[Gamma[z]/(z + 1), {z, -1}] = -1 + EulerGamma: a Gamma pole
    // model times another pole at a NONZERO point produces a pathologically
    // slow Simplify blowup (PolyGamma constants over non-monomial pole
    // quotients), so Woxi deliberately leaves it unevaluated. See the comment
    // in tests/interpreter_tests/calculus.rs.
    "Residue[Gamma[z]/(z + 1), {z, -1}]",

    // ── Individual CAS form / gap divergences (value-correct or unimplemented) ──

    // Trace: Wolfram wraps each evaluation step in HoldCompleteForm[...]; Woxi
    // uses HoldForm[...]. Step-wrapper head differs, content matches.
    "Trace[1 + 2]",
    "Trace[3 * 4]",
    // GeometricTest["Collinear"]: Wolfram expands to the collinearity
    // determinant equation; Woxi keeps the predicate symbolic.
    "GeometricTest[{{a, b}, {c, d}, {e, f}}, \"Collinear\"]",
    // ShortTimeFourier / WienerFilter: STFT windowing convention (sum vs mean,
    // complex vs real first bin) and last-ULP filter differences.
    "ShortTimeFourier[{1., 2., 3., 4., 5., 6., 7., 8.}][\"Data\"][[1, 1]]",
    "WienerFilter[{1., 2., 3., 4.}, 1, 0]",
    // MovingMap[Mean, ...]: value-correct windows but Woxi keeps exact rationals
    // (5/2) where Wolfram coerces to reals (2.5).
    "MovingMap[Mean, {1, 2, 3, 4, 5}, 3]",
    // SeriesCoefficient[Exp[x^2], {x, 0, n}]: Wolfram returns a symbolic-n
    // Piecewise/Gamma closed form; Woxi keeps it unevaluated.
    "SeriesCoefficient[Exp[x^2], {x, 0, n}]",
    // Fourier transforms: Pi/Sqrt[2*Pi] vs Sqrt[Pi/2] — value-equal, differ only
    // in the radical-folding of the normalization constant.
    "FourierTransform[1/t, t, w]",
    "FourierSinTransform[Cos[t]/t, t, 2]",
    "FourierSinTransform[Cos[t]/t, t, 1]",
    // AsymptoticIntegrate at x -> 0: Wolfram returns the leading asymptotic term
    // (0); Woxi computes the exact integral (x^2/3, x^5/6).
    "AsymptoticIntegrate[(t*x)^2, {t, 0, 1}, x -> 0]",
    "AsymptoticIntegrate[(t*x)^5, {t, 0, 1}, x -> 0]",
    // MidDate: seconds field renders as 0 (Wolfram) vs 0. (Woxi) in the midpoint
    // DateObject.
    "MidDate[{DateObject[{2024, 10, 1}], DateObject[{2024, 10, 7}, \"Week\"]}]",
    // RiceDistribution Mean: Sqrt[Pi/2] vs Sqrt[Pi]/Sqrt[2] — value-equal radical
    // folding difference.
    "Mean[RiceDistribution[1, 1]]",
    "Mean[RiceDistribution[1, 2]]",
    // ShortestCurveDistance on a 1D line: 1 (Wolfram) vs 1. (Woxi).
    "ShortestCurveDistance[Line[{{0}, {1}}], {0}, {1}]",
    // Region[Style[...]]: Wolfram lowers the style into a {Properties -> {...}}
    // region-properties record; Woxi keeps the Style[...] wrapper.
    "Region[Style[Ball[], Yellow]]",
    "Region[Style[Disk[{a, b}], Green]]",
    // Torus[]: Wolfram fills in default arguments Torus[{0,0,0}, {1/2, 1}]; Woxi
    // keeps the bare Torus[] symbolic.
    "Torus[]",
    // RegionMeasure[Parallelogram[3D]]: Woxi computes the area (2); Wolfram
    // leaves the higher-embedding parallelogram measure unevaluated.
    "RegionMeasure[Parallelogram[{0, 0, 0}, {{1, 0, 0}, {0, 2, 0}}]]",
    // Dendrogram: Wolfram returns a full Graphics[...] rendering; Woxi keeps the
    // call symbolic for a non-clusterable input.
    "Dendrogram[{1, \"a\"}]",
    // HTTPRequest: Wolfram keeps the URL[...] wrapper and normalizes a lone
    // options association into the two-argument HTTPRequest[<||>, opts] form;
    // Woxi unwraps URL[...] and keeps the single-argument association form.
    // Changing either ripples through the property-extraction path.
    "HTTPRequest[URL[\"https://example.com\"]]",
    "HTTPRequest[<|\"Method\" -> \"POST\"|>]",
    // PolyGamma[-1.5]: last-ULP floating-point difference.
    "PolyGamma[-1.5]",
    // PolyhedronData Volume: (5*GoldenRatio^2)/6 vs (5*(3 + Sqrt[5]))/12 —
    // value-equal, GoldenRatio-folding difference.
    "PolyhedronData[\"Icosahedron\", \"Volume\"]",
    // GeneratingFunction[1/(2n+1), n, x]: Wolfram returns ArcTanh[Sqrt[x]]/Sqrt[x];
    // Woxi keeps the sum unevaluated.
    "GeneratingFunction[1/(2 n + 1), n, x]",
    // Transliterate to non-Latin scripts (Hiragana) is unimplemented; Woxi keeps
    // the call symbolic.
    "Transliterate[\"tadaima\", \"Hiragana\"]",
    // Around[<distribution>]: last-digit std difference (Uniform) and Woxi
    // evaluating Poisson to Around[4., 2.] where Wolfram keeps it symbolic.
    "Around[UniformDistribution[{0, 1}]]",
    "Around[PoissonDistribution[4]]",
    // SurfaceArea of a spherical shell with symbolic radii: wolframscript itself
    // hangs (never terminates) on this integral, causing the batch to ETIMEDOUT.
    // Woxi intentionally keeps it unevaluated to match.
    "SurfaceArea[SphericalShell[{0, 0, 0}, {a, b}]]",
    // Astronomy: two cases that a numeric tolerance cannot rescue (the
    // ephemeris-precision cases are handled by APPROX_MATCH below instead).
    // Svalbard around the December solstice: Woxi reports Missing[NotApplicable]
    // for the day, while Wolfram rolls forward to the next actual sunrise
    // (2025-02-15) — a semantic difference, not a rounding one.
    "Sunrise[GeoPosition[{78.22, 15.63}], DateObject[{2024, 12, 21}]]",
    // $GeoLocation needs a GeoIP lookup. Woxi is offline, so it matches
    // no-internet wolframscript: Missing["NotAvailable"]. The conformance
    // harness runs wolframscript *with* internet, so it returns the running
    // machine's real location instead — environment-dependent, no stable
    // reference value to conform to.
    "$GeoLocation",
    // Complex elementary functions: last-ULP floating-point differences. Woxi
    // computes these from f64 libm primitives whose last reported digit rounds
    // differently from Wolfram's; the values agree to ~1 ULP (verified against
    // 20-digit references) and no reimplementation reproduces Wolfram's exact
    // machine rounding.
    "ArcTan[1.0 + 1.0 I]",
    "ArcSin[1.0 + 1.0 I]",
    "ArcCos[0.5 + 0.5 I]",
    "ArcSinh[1.0 + 1.0 I]",
    "Sqrt[2.0 + 3.0 I]",
    // NMinimize/NMaximize: Woxi lands on the exact extremum (Pi/2 -> 1.,
    // -Pi/2 -> -1.) because it recognizes the closed-form critical point,
    // while Wolfram's numeric optimizer stops a few ULPs short of the exact
    // argument. Woxi is the more accurate result; there is nothing to conform.
    "NMaximize[{Sin[x], 0 < x < 2*Pi}, x]",
    "NMinimize[Sin[x], x]",
    // ListPlay: Woxi normalizes the sample list to a clean 0. where Wolfram's
    // resampling leaves a 6.9*^-17 rounding artifact. Value-identical audio;
    // Woxi's zero is the cleaner representation.
    "ListPlay[{0.1, 0.2, 0.3, -0.1}]",
    // Sunrise/SunPosition without an explicit location need $GeoLocation, which
    // requires a GeoIP lookup. Woxi is offline so it keeps the call symbolic
    // (matching no-internet wolframscript); the conformance harness runs
    // wolframscript with internet, so it resolves the running machine's real
    // location and computes a value — environment-dependent, no stable
    // reference. (Located Sunrise/SunPosition are covered by APPROX_MATCH.)
    "Sunrise[DateObject[{2024, 6, 21}]]",
    "SunPosition[DateObject[{2024, 6, 21, 12, 0, 0}]]",
    // Constrained FindMinimum/FindMaximum/NArgMin/NArgMax: same story as the
    // NMinimize/NMaximize entries above. Woxi returns the exact optimum where
    // Wolfram's interior-point method stops short of it (1.000000013282579 for
    // `FindMinimum[{x^2, x >= 1}, x]`, 0.7071076183816036 for the unit-circle
    // maximum). Woxi is the more accurate answer; bit-equality is impossible.
    "FindMinimum[{x^2, x >= 1}, {x, 0}]",
    "FindMinimum[{x^2, x >= 1}, x]",
    "FindMaximum[{x + y, x^2 + y^2 <= 1}, {{x, 0.5}, {y, 0.5}}]",
    "FindMinimum[{x^2 + y^2, x + y == 1 && x >= 0.7}, {{x, 0}, {y, 0}}]",
    "NArgMax[{x + y, x^2 + y^2 <= 1}, {x, y}]",
    "NArgMin[{x^2, x >= 1}, x]",
    // An NDSolve result that is printed rather than sampled: Wolfram's
    // InterpolatingFunction shows its internal solver state verbatim — the
    // {5, 7, 1, {25}, {4}, 0, ...} header, the adaptive step grid, and a
    // Developer`PackedArrayForm coefficient block. Woxi carries a plain {x, y}
    // sample table on a fixed grid. The solutions agree; only the internal
    // representation differs (the tests that sample the result do conform).
    "NDSolve[{y'[x] == y[x], y[0] == 1}, y[x], {x, 0, 1}]",
    "NDSolve[{y'[x] == y[x], y[0] == 1}, y, {x, 0, 1}]",
    // The antiderivative of Log[Sin[x]] is known, but the definite value
    // Wolfram prints — (-1/12*I)*(6 + (-6 + Pi)*Pi - (12*I)*Log[2] -
    // 6*PolyLog[2, E^(2*I)]) — is a Simplify-collected complex form Woxi will
    // not land on, so implementing the antiderivative would only trade
    // "unevaluated" for "different form". Woxi keeps it unevaluated.
    "Integrate[Log[Sin[x]], {x, 0, 1}]",
    // AsymptoticLess: the multivariate `{x, y} -> {Infinity, Infinity}` form
    // needs multivariate limits, and `AsymptoticLess[x^a, x^2, x -> Infinity]`
    // needs assumption-aware limits to produce Wolfram's
    // ConditionalExpression[True, a < 2]. Both are deliberately unevaluated —
    // see the `undecided_forms_stay_unevaluated` test.
    "AsymptoticLess[x + y, x^2 + y^2, {x, y} -> {Infinity, Infinity}]",
    "AsymptoticLess[x^a, x^2, x -> Infinity]",
    // 3D ConvexHullMesh: computing the hull is easy, but Wolfram delegates to
    // qhull and prints its facet bookkeeping verbatim — facet creation order,
    // in-face vertex rotation and coplanar-triangle merging would all have to
    // be replicated. Woxi keeps the 3D form unevaluated.
    "ConvexHullMesh[{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1}}]",
    // PascalBinomial[6.0, -2]: Wolfram returns 0``15.954589770191005 — an
    // arbitrary-precision zero whose accuracy is $MachinePrecision — where
    // Woxi returns a machine 0. Reproducing it needs precision/accuracy
    // tracking through the Gamma-ratio path from a machine-real input.
    "PascalBinomial[6.0, -2]",

    // ── Inexact-zero imaginary parts ────────────────────────────────────────
    // Wolfram keeps a machine-zero imaginary part visible (`1.6487… + 0.*I`,
    // `Im[…] -> 0.`); Woxi collapses `x + 0.*I` to the real `x`, so the
    // imaginary part comes back as exact 0. This is the complex-float
    // representation rabbit hole documented at `times_ast` in arithmetic.rs:
    // the non-folding of `scalar * (0. + c*I)` is deliberate (it keeps the
    // `c*I` monomial mergeable inside an enclosing Plus) and every attempt to
    // make `N[I]` inexact regressed `N[2 + 3 I]`, `N[Sin[I]]` and `N[2 I]`.
    "E^(0.5 + 0.*I)",
    "2.^(0.5 + 0.*I)",
    "Im[Total[x /. NSolve[x^10 - 3 x + 1 == 0, x]]]",

    // Last-ULP float differences where Woxi lands on the exact value and
    // Wolfram's numeric path does not: the cubic Bernoulli root is exactly 1/2
    // (Wolfram reports 0.5000000000000001) and Sqrt[2] rounds to
    // 1.4142135623730951 (Wolfram's companion-matrix eigenvalue gives
    // 1.414213562373095). Same story as the existing NSolve cubic skip.
    "Solve[N[Table[BernoulliB[n, z], {n, 3, 3}] == 0]]",
    "Solve[N[BernoulliB[3, z]] == 0, z]",
    "NSolve[x^2 == 2]",
    // Sharpen: last-ULP differences in the unsharp-mask convolution
    // (1.7058721780776978 vs 1.7058720588684082).
    "ImageData[Sharpen[Image[{{1., 0., 0., 0., 0., 0.}}], 10]]",
    // ComplexExpand: canonical Plus ordering — Woxi puts the real part first
    // ((2*Sqrt[2])/3 + I/3), Wolfram the imaginary one (I/3 + (2*Sqrt[2])/3).
    // Same core ordering gap as the `Log[E^(a + 3 I)]` entry above.
    "ComplexExpand[E^(I*ArcSin[1/3])]",

    // PermutationMatrix returns a StructuredArray in Wolfram
    // (`PermutationMatrix[StructuredArray`StructuredData[…]]`); Woxi returns
    // the dense matrix, exactly as for Symmetrize / CrossMatrix above.
    "PermutationMatrix[{2, 1}]",

    // NDSolve results that are *printed* rather than sampled: Wolfram shows
    // its internal solver state (the {5, 7, 1, {52}, …} header, the adaptive
    // step grid and a Developer`PackedArrayForm coefficient block) where Woxi
    // carries a plain sample table on a fixed grid. Same reason as the
    // `NDSolve[{y'[x] == y[x], …}]` entries above; the tests that sample the
    // solution do conform.
    "NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]",
    "NDSolve[{y'[t] == -y[t], y[1] == 1}, y, {t, 1, 3.5}]",

    // DynamicModule: Wolfram keeps the wrapper around the evaluated body
    // (`DynamicModule[{x = 2}, 4, DynamicModuleValues :> {}]`) because the
    // front end owns the local state between redraws. Woxi hands back the
    // body's value so a Grid or a Graphics inside one displays as itself —
    // a deliberate choice, documented at the `dynamic_module_scoping` tests.
    "DynamicModule[{x = 2}, x^2]",
    "DynamicModule[{a}, a = 3; a + 1]",
    "x = 1; DynamicModule[{x = 5}, x] + x",
    "f[] := x; DynamicModule[{x = 3}, f[]]",

    // 3D plot internals: Wolfram wraps the surface in per-function layers of
    // lists and directives — Plot3D gives `{{GraphicsComplex[…], {}}}`,
    // SphericalPlot3D a two-element list of those, ParametricPlot3D neither —
    // with no rule that carries across the family. Woxi stores every 3D plot
    // as a plain `Graphics3D[GraphicsComplex[…]]` so `First[plot]` is the
    // surface itself and can be redrawn inside another graphic.
    "Head[SphericalPlot3D[1, {t, 0, Pi}, {p, 0, 2 Pi}][[1]]]",
    "Head[First[Plot3D[x y, {x, 0, 1}, {y, 0, 1}]]]",

    // PolyhedronData: the vertex coordinates are value-identical but written
    // with a different radical nesting (Woxi `Sqrt[5/8 + Sqrt[5]/8]`, Wolfram
    // `5/Sqrt[50 - 10*Sqrt[5]]`) — the same folding difference as the
    // `PolyhedronData["Icosahedron", "Volume"]` entry above. The property and
    // solid catalogues are the subset Woxi implements against Wolfram's full
    // curated database, like `ElementData["Properties"]`.
    "PolyhedronData[\"Icosahedron\", \"VertexCoordinates\"]",
    "PolyhedronData[\"Properties\"]",
    "PolyhedronData[All]",

    // CSV import has no stable reference: an earlier `ImportString["", "Table"]`
    // in the same wolframscript session swaps the converter for the rest of it.
    // Cold, `"HeaderLines" -> 1` is ignored and "true"/"false" become booleans;
    // once a "Table" import has run, the header line is dropped and the
    // booleans stay strings. Reproduce with
    //   wolframscript -code 'Quiet[ImportString["", "Table"]];
    //     Print[ToString[{ImportString["a,b\n1,2", "CSV", "HeaderLines" -> 1],
    //       ImportString["true,false", "CSV"]}, InputForm]]'
    // against the same line without the warm-up. Woxi implements the
    // cold-kernel behaviour, which is what these expressions see on their own;
    // the io.rs cases only differ once ~20 earlier cases share the batch.
    // Same "no single stable reference" situation as Attributes[ParallelDo].
    "ToString[ImportString[\"a,b\\n1,2\", \"CSV\", \"HeaderLines\" -> 1], InputForm]",
    "ToString[ImportString[\"true,false\", \"CSV\"], InputForm]",
    "ToString[ImportString[\"True,FALSE,tRue,yes\", \"CSV\"], InputForm]",

    // `"ColumnTypes"` of a labelled table whose text does not end in a newline:
    // Wolfram reports the last column as "String" no matter what is in it,
    // while the same import reads that field as a number. Reproduce with
    //   wolframscript -code 'ToString[{ImportString["a,b,c\n1,2,3", "CSV"],
    //     ImportString["a,b,c\n1,2,3", {"CSV", "ColumnTypes"}],
    //     ImportString["a,b,c\n1,2,3\n", {"CSV", "ColumnTypes"}]}, InputForm]'
    // -> the data is `{{"a","b","c"},{1,2,3}}`, the types are
    // `<|"a" -> "Integer64", "b" -> "Integer64", "c" -> "String"|>` without the
    // trailing newline and all "Integer64" with it. The unterminated last field
    // is the only difference, so this is Wolfram contradicting itself rather
    // than a type rule; Woxi types the column from the value it imported.
    "ToString[ImportString[\"a,b,c\\n1,2,3\", {\"CSV\", \"ColumnTypes\"}], InputForm]",

    // TraditionalForm boxes: Wolfram hands a special function or a Row to a
    // named FrontEnd template (`TemplateBox[{n, x}, "LegendreP"]`,
    // `TemplateBox[{2, x, t}, "RowDefault"]`) that knows how to draw it.
    // Woxi has no FrontEnd behind it, so `tf()` writes the same layout out
    // inline as Sub/SubsuperscriptBox rows its own box renderer can draw.
    // The picture and the TagBox/FormBox wrapper agree; only the inner box
    // representation differs — same class as the MakeBoxes number-form
    // entries above. See the `traditional_form_boxes_special_functions_inline`
    // test and tests/cli/math/utility/TraditionalForm.md.
    "ToBoxes[TraditionalForm[LegendreP[n, x]]]",
    "ToBoxes[TraditionalForm[LegendreP[n, m, x]]]",
    "ToBoxes[TraditionalForm[Row[{2, x, t}]]]",

    // ── Aug 2026 batch ───────────────────────────────────────────────────
    // Which solution of a system comes first, and in which order its rules
    // are written, follows each engine's own elimination order. Same
    // solutions, different sequence.
    "Solve[{p*(1 - p/100 - 3/500*q) == 0, q*(1 - q/100 - 1/200*p) == 0}, {p, q}]",
    "Solve[{0.5*p*(1 - p/100 - 0.6*q/100) == 0, 0.5*q*(1 - q/100 - 0.5*p/100) == 0}, {p, q}, Reals]",
    "Solve[{yy^2 == 20 xx, {xx, yy} == t*{0.6, 0.8} + {5, 0}}, {t, xx, yy}]",
    "NSolve[y == -0.8090169943749475 && (x - 0.7694208842938133)^2 + (y - (-0.25))^2 == 1.090330521158122^2, {x, y}]",

    // Arbitrary-precision Root: both engines print more digits than were
    // asked for, and past digit ~17 they are each other's padding. Woxi's
    // continuation is the correct one (checked against the plastic constant
    // and the quartic's root to 40 digits).
    "N[Root[#^3 - # - 1 &, 1], 10]",
    "N[Root[#^3 - # - 1 &, 1], 30]",
    "N[Root[#^4 - # - 1 &, 1], 15]",

    // Euler's sum-of-powers conjecture needs 27^5+84^5+110^5+133^5 == 144^5;
    // wolframscript's Diophantine machinery finds it, Woxi's bounded search
    // does not reach that far and stays unevaluated.
    "FindInstance[x0^5 + x1^5 + x2^5 + x3^5 == y^5 && x0 > 0 && x1 > 0 && x2 > 0 && x3 > 0, {x0, x1, x2, x3, y}, Integers]",

    // Unprotect[x] on a cold kernel is {} in both engines; inside a verify
    // batch an earlier Protect[x] has already run, so wolframscript answers
    // {"x"}. No stable reference value.
    "Unprotect[x]",

    // NDSolve's InterpolatingFunction grid: Woxi integrates on its nominal
    // 1000-step grid, wolframscript on its own adaptive one, so the number
    // of stored samples differs (13 vs 1001). Same solution, sampled
    // differently — the NDSolve tests that *evaluate* the result conform.
    "s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]; Length[(y /. s[[1]])[[2]]]",

    // A *list* of Graphics prints as `-Graphics-` placeholders in Woxi
    // (the RENDERED_PLACEHOLDERS filter only catches a bare one).
    "ArrayPlot /@ CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, {1, 1}}, {{{1}}, 0}, {{10, 30, 10}}]",

    // MidDate on sub-second instants: Woxi carries the instant as one f64
    // count of seconds since the epoch, whose resolution near 1.7e9 is
    // ~2.4e-7 s, so the mean of fractional seconds differs in the 8th
    // decimal. wolframscript averages exactly.
    "MidDate[{DateObject[{2024, 12, 30, 18, 1, 56.77401781082153}], DateObject[{2024, 4, 8, 12, 54, 47.175180435180664}], DateObject[{2024, 8, 13, 22, 45, 35.52135992050171}]}]",

    // Woxi bundles its own NetworkGraph data and exposes properties
    // wolframscript's catalogue does not have ("VertexList", "EdgeRules",
    // "AdjacencyMatrix"). Deliberate: the catalogue is Wolfram's.
    'ExampleData[{"NetworkGraph", "ZacharyKarateClub"}, "VertexList"][[1 ;; 3]]',
    'ExampleData[{"NetworkGraph", "LesMiserables"}, "VertexList"][[1 ;; 2]]',

    // TriangleCenter of a triangle embedded in 3D: wolframscript only
    // handles the 2D case and stays unevaluated.
    'TriangleCenter[Triangle[{{0, 0, 0}, {4, 0, 0}, {0, 3, 0}}], "Circumcenter"]',

    // Last-ULP float differences: a polyline length summed in a different
    // order, and two image filters wolframscript accumulates in Real32
    // where Woxi accumulates in f64 before snapping.
    "ShortestCurveDistance[Line[{{1, 0}, {2, 1}, {3, 0}, {4, 1}}], {1, 0}, {3, 0}]",
    "ImageData[RecurrenceFilter[{{1, -0.5}, {1}}, Image[{{0.1, 0.5, 0.9}, {0.2, 0.4, 0.6}}]]]",
    "ImageData[Sharpen[Image[{{0., 0., 0.}, {0., 1., 0.}, {0., 0., 0.}}], {0, 1}]]",

    // The InputStream counter, same story as the ReadString entry above.
    'StringToStream["abc"]',

    // InputForm-only bracketing: as a *string* wolframscript brackets a
    // PatternTest's list left side (`({1, 2})?f`), while the same
    // expression printed at top level is `{1, 2}?f`, which is what Woxi
    // returns. Ditto a Graph's options, which InputForm wraps in a List.
    "Hold[{1, 2}?f]",
    'Graph[{UndirectedEdge[1, 2], UndirectedEdge[2, 3], UndirectedEdge[3, 1]}, VertexLabels -> {1 -> "one"}]',

    // Plot internals: Woxi builds its plots with its own primitive layout
    // (see the three-renderers note), so the sampling grid, the wrapper
    // heads (GraphicsComplex / Annotation / Legended) and where a style
    // directive sits inside the returned expression all differ, even though
    // the picture and the public options agree.
    "Length[SphericalPlot3D[1, {t, 0, Pi/3}, {p, 0, Pi/3}, PlotPoints -> 2][[1, 1]]]",
    "Min[SphericalPlot3D[1, {t, 0, Pi}, {p, 0, 2 Pi}, RegionFunction -> ({#1, #2, #3} . {0, 0, 1} > 0 &)][[1, 1]][[All, 3]]] > -1/1000",
    'Cases[SphericalPlot3D[1, {t, 0, Pi/2}, {p, 0, Pi}, BoundaryStyle -> Black][[1, 2]], _Line, Infinity] =!= {}',
    "Length[Cases[ListVectorPlot[{{{0, 0}, {1, 0}}, {{1, 1}, {0, 1}}}][[1]], _Line | _Arrow, Infinity]] > 0",
    "With[{prims = (ContourPlot[x == y, {x, -3, 3}, {y, -3, 3}, ContourStyle -> {Pink}] /. Tooltip[q_, ___] :> q)[[1]]}, {prims[[1]], Head[prims[[2]]]}]",
    "p = ParametricPlot[{Sin[t], Cos[t]}, {t, 0, 3}][[1]]; {Head[p], Head[p[[1, 2]]]}",
    "Head[First[RevolutionPlot3D[{Cos[t], 1 + Sin[t]}, {t, -Pi/2, 0}, Mesh -> None]]]",
    "First[RevolutionPlot3D[{Cos[t], 1 + Sin[t]}, {t, -Pi/2, 0}, Mesh -> None, PlotStyle -> Opacity[0.2]]][[2, 1]]",

    // Manipulate's Initialization: Woxi has no DynamicModule to scope it
    // to, and its controls re-resolve the body on every frame, so the
    // definitions have to stay in the global scope after the Manipulate
    // returns. wolframscript keeps them inside the module.
    "Manipulate[myhelper[a], {a, 0, 10}, Initialization :> (myhelper[x_] := x^2 + 1)]; myhelper[3]",
  ]);

  /** Names whose meaning depends on where one input unit ends and the next
   * begins — see the filter below. */
  const CONTEXT_SENSITIVE =
    /\$Context|\$Packages|BeginPackage\[|EndPackage\[|Begin\[|End\[|Needs\[/;

  // Filter out multiline expressions (they break the generated scripts).
  // Also skip Interrupt[] — it sends a kernel interrupt that crashes wolframscript
  // even inside CheckAbort, so it cannot be tested via batch conformance.
  // Also skip bare Goto[tag] without a matching Label — it fatally aborts the
  // wolframscript session (uncatchable, even by CheckAbort/Catch).
  const cases = allCases.filter(
    (c) =>
      !c.expr.includes("\n") &&
      // A follow-up whose setup is itself multiline (a package definition,
      // where the line breaks are what separates the inputs) cannot be
      // joined into the one-liner both sides run. Comparing it without that
      // setup would test a different expression than the unit test does.
      !(c.setup ?? []).some((s) => s.includes("\n")) &&
      // Context constructs take effect for the *next* input unit, so a case
      // whose setup was a separate `interpret()` call means something else
      // once the two are joined with a semicolon into one unit.
      !(
        (c.setup?.length ?? 0) > 0 &&
        CONTEXT_SENSITIVE.test([...(c.setup ?? []), c.expr].join("; "))
      ) &&
      !c.expr.includes("Interrupt[]") &&
      !/[^\x00-\x7F]/.test(c.expr) && // Non-ASCII chars get garbled by wolframscript encoding
      !(c.expr.match(/^Goto\[/) && !c.expr.includes("Label[")) &&
      !IMPL_SPECIFIC_PATTERNS.some((p) => p.test(c.expr)) &&
      !EXACT_EXPR_SKIP.has(c.expr)
  );
  const skipped = allCases.length - cases.length;
  const tested = cases.length;

  // Step 1: Run each expression through woxi eval with ToString[_, InputForm]
  // Woxi is fast (~20ms per call), so this takes ~10s for 500 tests.
  console.log(`Running ${tested} test cases through woxi eval (${skipped} skipped)...`);
  const woxiResults: { expr: string; woxiResult: string; idx: number }[] = [];
  for (let i = 0; i < tested; i++) {
    const { expr, setup } = cases[i];
    // For expressions with setup, prepend setup code. Drop setup entries that
    // are in EXACT_EXPR_SKIP: those are skipped precisely because wolframscript
    // hangs or diverges on them, and a skip-listed setup expression would hang
    // the batch even though the case's own expr is fine (they set no state).
    const fullExpr = setup
      ? [
          ...setup.filter(
            (s) => !s.includes("\n") && !EXACT_EXPR_SKIP.has(s)
          ),
          expr,
        ].join("; ")
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
      // SVG output from ExportString[_, "SVG"] differs structurally between
      // implementations (different renderers, coordinate systems, fonts) so
      // byte-level comparison is meaningless.
      && !r.woxiResult.startsWith("<svg")
      && !r.woxiResult.startsWith('"<svg')
      && !r.woxiResult.startsWith('"<?xml')
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

    let outputLines: string[];
    try {
      outputLines = runBatchResilient(
        batch,
        failures,
        `Batch ${b + 1}/${totalBatches} (${rangeLabel(batch)})`
      );
    } catch (err: any) {
      if (!(err instanceof HangingCaseError)) throw err;
      const culprit = err.entry;
      const tc = cases[culprit.idx];
      console.error(
        `\nwolframscript could not evaluate this expression on its own ` +
          `within ${err.timeoutMs / 1000}s — it hangs:`
      );
      console.error(`\nFailing expression (case #${culprit.idx + 1}): ${culprit.expr}`);
      console.error(`Woxi result: ${culprit.woxiResult}`);
      if (tc) console.error(`Source: ${tc.file}:${tc.line}`);
      console.error(
        `\nIf wolframscript genuinely never terminates here, add the expression ` +
          `to EXACT_EXPR_SKIP with a comment explaining why.`
      );
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
  const flakyCases = failures.filter((l) => l.startsWith("FLAKY case"));

  if (failCount === 0 && flakyCases.length === 0) {
    console.log(`All ${testedFiltered} test cases match between Woxi and wolframscript.`);
  } else {
    if (failCount > 0) {
      console.error(`\n${passCount}/${testedFiltered} passed, ${failCount} differ:\n`);
    }
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
    if (flakyCases.length > 0) {
      console.error(
        `\n${flakyCases.length} case(s) were skipped after repeated wolframscript ` +
          `cold-start flakes; re-run to verify them.`
      );
    }

    process.exit(1);
  }
}

main();
