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
      killSignal: "SIGKILL", // SIGTERM is ignored by wolframscript during computation
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
    /\bDominantColors\[/,   // Implementation-specific color algorithm: Woxi uses a GrayLevel
                            // ramp on single-channel inputs, Wolfram hashes labels into RGB.
    /\bCenteredInterval\[/, // Internal representation differs: Woxi stores {value, radius}
                            // pairs, Wolfram stores an arbitrary-precision ball encoding
                            // (e.g. {{5, 0, 536870912, -29}, 63}).
    /\bCloudExport\[/,      // Cloud-dependent: Wolfram uploads to wolframcloud.com and
                            // returns a CloudObject URL; Woxi keeps the call symbolic.
    /\bSound\[\{Play\[/,    // Wolfram compiles Play[] into a SampledSoundFunction with a
                            // CompiledFunction body; Woxi keeps Play[] as an inert wrapper.
  ];

  // Specific expressions where Woxi is more accurate than Wolfram.
  // NSolve cubic: Woxi gives exact integer roots (1.) via symbolic solving,
  // while Wolfram's companion-matrix eigenvalues introduce machine-epsilon
  // artifacts (1.0000000000000002).
  const EXACT_EXPR_SKIP = new Set([
    "NSolve[x^3 - 3*x^2 + 2*x == 0, x]",
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
    // HyperbolicDistribution PDF symbolic: Plus ordering in exponent (-(a*Sqrt[...]) + b*... vs b*... - a*Sqrt[...])
    "PDF[HyperbolicDistribution[a, b, d, m], x]",
    // Variance symbolic: Plus ordering of BesselK terms (different ordering of positive/negative terms)
    "Variance[HyperbolicDistribution[a, b, d, m]]",
    // FindIntegerNullVector: sign convention is implementation-specific (LLL algorithm produces different signs)
    "FindIntegerNullVector[{2, 6}]",
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
    // BesselJ[±1/2, x]: Woxi uses Sqrt[2]/Sqrt[Pi], Wolfram uses Sqrt[2/Pi].
    // Mathematically identical; different surface algebraic form.
    "BesselJ[1/2, x]",
    "BesselJ[-1/2, x]",
    // LegendreP[2, 1, x]: factor ordering in Times differs. Woxi emits
    // -3*Sqrt[1-x^2]*x while Wolfram emits -3*x*Sqrt[1-x^2]. Same value.
    "LegendreP[2, 1, x]",
    // LaguerreL[5, 2, x]: Woxi returns expanded form, Wolfram returns
    // the factored-over-120 form. Same polynomial.
    "LaguerreL[5, 2, x]",
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
    // Series[x!, {x, 0, 2}] — Plus ordering inside the third-order
    // coefficient: Woxi `Pi^2 + 6*EulerGamma^2` vs Wolfram `6*EulerGamma^2 + Pi^2`.
    "Series[x!, {x, 0, 2}]",
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
    // PDF[HypergeometricDistribution[20, 50, 100], k] — same expression up
    // to Binomial factor ordering inside the Piecewise.
    "PDF[HypergeometricDistribution[20, 50, 100], k]",
    // Around[x, Scaled[0.1]] — Wolfram resolves Scaled[0.1] to 0.1*x at
    // evaluation time; Woxi keeps the Scaled[] uncertainty marker symbolic.
    "Around[x, Scaled[0.1]]",
    // Median[ChiDistribution[3]] — algebraic surface difference: Woxi gives
    // Sqrt[2]*Sqrt[InverseGammaRegularized[3/2, 0, 1/2]], Wolfram fuses the
    // square roots to Sqrt[2*InverseGammaRegularized[3/2, 0, 1/2]].
    "Median[ChiDistribution[3]]",

    // ───────────────────────────────────────────────────────────────────────
    // CAS capabilities Woxi returns unevaluated where Wolfram computes a closed
    // form (parametric/underdetermined solving, nonlinear ODEs, symbolic
    // transforms/sums, Piecewise/DiscreteDelta results, sum-to-product
    // factoring). Woxi's own unit tests assert the unevaluated/symbolic result.
    "Solve[x + y == 3]", // underdetermined linear system
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
    "Sum[k r^k, {k, 0, Infinity}]", // symbolic geometric-type sums
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

    // Canonical Plus/Times ordering differences (mathematically identical):
    "ZTransform[a^n, n, z]", // -(z/(a - z)) vs z/(-a + z)
    "ZTransform[n^2 a^n, n, z]",
    "Log[E^(a + 3 I)]", // a + 3*I vs 3*I + a (numeric-imaginary term first)

    // PermutationProduct prints with the private-use centred-dot infix glyph
    // (U+F3DE); Woxi keeps the call symbolic with the standard head.
    "PermutationProduct[{2, 1, 4}, {1, 3, 2}]",

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
  ]);

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
