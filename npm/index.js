// JavaScript wrapper around the Woxi WebAssembly build (see ../src/wasm.rs).
//
// The generated bindings in pkg/ (built by `make npm-build` in the repo
// root) expose the raw wasm-bindgen functions. This module adds:
//
// - camelCase names and JSON parsing for the structured APIs
// - a Node implementation of the `__woxi_fetch_url` host hook, so
//   `Import["https://..."]` works out of the box
// - panic recovery: a Rust panic leaves the wasm instance unusable, so the
//   wrapper converts it into a JS Error carrying the panic message and
//   transparently loads a fresh instance on the next call

"use strict";

const path = require("node:path");

const PKG_ENTRY = path.join(__dirname, "pkg", "woxi.js");

let lastPanicMessage = null;

// Host hook called by the wasm panic hook right before the trap fires.
if (typeof globalThis.__woxi_report_panic !== "function") {
  globalThis.__woxi_report_panic = (msg) => {
    lastPanicMessage = msg;
  };
}

// Host hook used by `Import["https://..."]`: fetch a URL synchronously and
// return its body base64-encoded. The wasm side is fully synchronous, so we
// shell out to a child Node process that performs the async fetch.
if (typeof globalThis.__woxi_fetch_url !== "function") {
  globalThis.__woxi_fetch_url = (url) => {
    const { execFileSync } = require("node:child_process");
    const script = `
      fetch(process.argv[1])
        .then((res) => {
          if (!res.ok) throw new Error("HTTP status " + res.status);
          return res.arrayBuffer();
        })
        .then((buf) => process.stdout.write(Buffer.from(buf).toString("base64")))
        .catch((err) => { process.stderr.write(String(err.message || err)); process.exit(1); });
    `;
    try {
      return execFileSync(process.execPath, ["-e", script, url], {
        encoding: "utf8",
        maxBuffer: 512 * 1024 * 1024,
      });
    } catch (err) {
      const stderr = err.stderr ? String(err.stderr).trim() : "";
      throw new Error(stderr || `failed to fetch ${url}`);
    }
  };
}

let wasm = null;

function getWasm() {
  if (wasm === null) {
    // Drop stale copies from the require cache so that a reload after a
    // panic instantiates a fresh wasm instance instead of the broken one.
    delete require.cache[require.resolve(PKG_ENTRY)];
    wasm = require(PKG_ENTRY);
  }
  return wasm;
}

// Run a raw wasm call, translating a Rust panic (which surfaces as a
// WebAssembly "unreachable" RuntimeError) into a descriptive JS Error and
// discarding the now-broken instance.
function guarded(fn) {
  lastPanicMessage = null;
  try {
    return fn(getWasm());
  } catch (err) {
    if (lastPanicMessage !== null) {
      wasm = null; // instance is poisoned - reload lazily on the next call
      throw new Error(`Woxi crashed while evaluating: ${lastPanicMessage}`, {
        cause: err,
      });
    }
    throw err;
  }
}

/**
 * Evaluate one or more Wolfram Language statements and return the combined
 * output (Print output followed by the final result) as a string.
 */
function evaluate(code) {
  return guarded((w) => w.evaluate(code));
}

/**
 * Evaluate all top-level statements and return structured output items
 * ({type: "text" | "graphics" | "print" | "warning" | "error" | ...}).
 */
function evaluateAll(code) {
  return JSON.parse(guarded((w) => w.evaluate_all(code)));
}

/** Split code into top-level statements (for progressive evaluation). */
function splitStatements(code) {
  return JSON.parse(guarded((w) => w.split_statements(code)));
}

/** Evaluate a single statement, returning structured output items. */
function evaluateStatement(statement) {
  return JSON.parse(guarded((w) => w.evaluate_statement(statement)));
}

/** SVG graphics captured by the last evaluate() call ("" when none). */
function getGraphics() {
  return guarded((w) => w.get_graphics());
}

/** Base64-encoded audio captured by the last evaluate() call ("" when none). */
function getSound() {
  return guarded((w) => w.get_sound());
}

/** Warnings emitted by the last evaluate() call. */
function getWarnings() {
  const text = guarded((w) => w.get_warnings());
  return text === "" ? [] : text.split("\n");
}

/** Clear all interpreter state (variables and function definitions). */
function clear() {
  guarded((w) => w.clear());
}

/** Toggle dark-mode colors for SVG output. */
function setDarkMode(enabled) {
  guarded((w) => w.set_dark_mode(enabled));
}

/**
 * Register an in-memory file so `Import["name"]` can read it. `data` may be
 * a string, Buffer, Uint8Array, or ArrayBuffer.
 */
function setVirtualFile(name, data) {
  let bytes;
  if (typeof data === "string") {
    bytes = new TextEncoder().encode(data);
  } else if (data instanceof Uint8Array) {
    bytes = data;
  } else if (data instanceof ArrayBuffer) {
    bytes = new Uint8Array(data);
  } else {
    throw new TypeError(
      "setVirtualFile data must be a string, Uint8Array, Buffer, or ArrayBuffer",
    );
  }
  guarded((w) => w.set_virtual_file(name, bytes));
}

/** Remove all files registered with setVirtualFile(). */
function clearVirtualFiles() {
  guarded((w) => w.clear_virtual_files());
}

module.exports = {
  evaluate,
  evaluateAll,
  splitStatements,
  evaluateStatement,
  getGraphics,
  getSound,
  getWarnings,
  clear,
  setDarkMode,
  setVirtualFile,
  clearVirtualFiles,
};
