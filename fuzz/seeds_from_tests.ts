#!/usr/bin/env node
/**
 * Write every Wolfram Language snippet the test suite evaluates into a
 * libFuzzer corpus directory, one snippet per file.
 *
 * `tests/scripts/*.wls` alone gives the crash fuzzer a few hundred
 * programs; the unit tests (`interpret("…")` in `tests/**\/*.rs`) and the
 * doc tests (`$ wo '…'` in `tests/cli/**\/*.md`) together hold tens of
 * thousands, and they are where nearly every builtin, operator and syntax
 * form the interpreter supports is exercised. Starting the fuzzer from them
 * means its mutations begin at that coverage instead of having to find it.
 *
 * Unit-test cases reuse the extractor of `verify_unit_tests.ts`, including
 * its setup tracking: a follow-up assertion is written together with the
 * earlier `interpret` calls of the same test function, so it runs against
 * the definitions it depends on.
 *
 * Files are named by content hash, so duplicates collapse.
 *
 * `--sample <n>` writes only a pseudo-random subset of n snippets, chosen
 * by `--sample-seed <s>`. libFuzzer runs every seed once under ASan before
 * it starts mutating, and the whole set takes longer than a nightly
 * fuzzing budget; a different seed each night rotates through all of them
 * instead.
 *
 * Usage: node fuzz/seeds_from_tests.ts [--sample <n>] [--sample-seed <s>] <corpus-dir>
 */

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  extractTestCases,
  joinStatements,
  listRustFiles,
} from "../tests/wolframscript/verify_unit_tests.ts";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/** Snippets of the unit tests, each prefixed with its test's setup code. */
function unitTestSnippets(): string[] {
  const snippets: string[] = [];
  for (const file of listRustFiles(join(ROOT, "tests")).sort()) {
    const content = readFileSync(file, "utf-8");
    if (!content.includes("interpret(")) continue;
    for (const tc of extractTestCases(file)) {
      snippets.push(joinStatements([...(tc.setup ?? []), tc.expr]));
    }
  }
  return snippets;
}

function listMarkdownFiles(dir: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir)) {
    const fullPath = join(dir, entry);
    if (statSync(fullPath).isDirectory()) files.push(...listMarkdownFiles(fullPath));
    else if (entry.endsWith(".md")) files.push(fullPath);
  }
  return files;
}

/**
 * Snippets of the doc tests: the shell-quoted argument of every
 * single-line `$ wo '…'` command. A `'\''` inside it is the shell's way of
 * writing one `'`.
 */
export function docTestSnippets(markdown: string): string[] {
  const snippets: string[] = [];
  for (const line of markdown.split("\n")) {
    const m = line.match(/^\$ wo '(.*)'\s*$/);
    if (m) snippets.push(m[1].replaceAll("'\\''", "'"));
  }
  return snippets;
}

/** Deterministic PRNG (mulberry32), so a sample replays from its seed. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The first `n` entries of a seeded Fisher–Yates shuffle of `xs`. */
export function sample<T>(xs: T[], n: number, seed: number): T[] {
  const ys = [...xs];
  const random = mulberry32(seed);
  const k = Math.min(n, ys.length);
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(random() * (ys.length - i));
    [ys[i], ys[j]] = [ys[j], ys[i]];
  }
  return ys.slice(0, k);
}

function usage(): never {
  console.error(
    "usage: node fuzz/seeds_from_tests.ts [--sample <n>] [--sample-seed <s>] <corpus-dir>"
  );
  process.exit(2);
}

function main() {
  let sampleSize = Infinity;
  let sampleSeed = 0;
  let outDir: string | undefined;
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--sample") sampleSize = Number(args[++i]);
    else if (args[i] === "--sample-seed") sampleSeed = Number(args[++i]);
    else if (outDir === undefined) outDir = args[i];
    else usage();
  }
  if (!outDir || !Number.isFinite(sampleSeed) || !(sampleSize >= 0)) usage();
  mkdirSync(outDir, { recursive: true });

  const unit = unitTestSnippets();
  const doc = listMarkdownFiles(join(ROOT, "tests", "cli"))
    .sort()
    .flatMap((f) => docTestSnippets(readFileSync(f, "utf-8")));

  const distinct = new Map<string, string>();
  const seen = new Set<string>();
  for (const [prefix, snippets] of [["unit", unit], ["doc", doc]] as const) {
    for (const snippet of snippets) {
      if (snippet.trim() === "") continue;
      const hash = createHash("sha1").update(snippet).digest("hex");
      if (seen.has(hash)) continue;
      seen.add(hash);
      distinct.set(`${prefix}-${hash}`, snippet);
    }
  }
  const names = [...distinct.keys()].sort();
  const chosen = sampleSize < names.length ? sample(names, sampleSize, sampleSeed) : names;
  for (const name of chosen) writeFileSync(join(outDir, name), distinct.get(name)!);

  console.log(
    `seeds_from_tests: ${unit.length} unit-test and ${doc.length} doc-test ` +
      `snippets, ${names.length} distinct, ${chosen.length} written to ${outDir}`
  );
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
