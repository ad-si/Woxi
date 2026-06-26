let wasm = null
// Bumped on every (re)load so the dynamic glue import specifier is unique.
// Each distinct specifier yields a fresh module namespace with its own `wasm`
// binding — the only way to get a brand-new instance, since `module.default()`
// short-circuits once its cached `wasm` is set.
let loadCount = 0
// Message forwarded by the Rust panic hook for the in-flight evaluation. Read
// after a trap to report a real cause instead of a bare "unreachable".
let lastPanic = null

// The Rust panic hook calls this just before the `unreachable` trap, letting
// us surface the panic message and know the instance must be reinstantiated.
globalThis.__woxi_report_panic = function (msg) {
  lastPanic = msg
}

// Provide __woxi_fetch_url to the WASM module so Import["https://..."] works.
// Called from Rust via wasm_bindgen extern.  Returns base64-encoded bytes.
// Synchronous XHR is fine inside a Web Worker (not on the main thread).
globalThis.__woxi_fetch_url = function (url) {
  const xhr = new XMLHttpRequest()
  xhr.open("GET", url, false) // synchronous
  // Use overrideMimeType to get raw bytes as a Latin-1 string
  xhr.overrideMimeType("text/plain; charset=x-user-defined")
  xhr.send()
  if (xhr.status < 200 || xhr.status >= 300) {
    throw new Error("HTTP " + xhr.status + " " + xhr.statusText)
  }
  const text = xhr.responseText
  // Convert Latin-1 string → binary string → base64
  const bytes = new Uint8Array(text.length)
  for (let i = 0; i < text.length; i++) {
    bytes[i] = text.charCodeAt(i) & 0xff
  }
  // btoa needs a binary string, build it from the Uint8Array
  let binary = ""
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary)
}

async function initWasm() {
  try {
    wasm = await loadWasm()
    postMessage({ type: "init", success: true })
  }
  catch (error) {
    postMessage({
      type: "init",
      success: false,
      message: error.message,
    })
  }
}

// Import the glue with a unique query so a fresh `wasm` instance is created,
// then instantiate it. Used for both the initial load and post-crash recovery.
async function loadWasm() {
  const module = await import("./pkg/woxi.js?reload=" + (loadCount++))
  const wasmUrl = new URL("./pkg/woxi_bg.wasm", self.location.href)
  wasmUrl.searchParams.set("v", Date.now())
  await module.default(wasmUrl)
  return module
}

// Reinstantiate the module after a trap. The previous instance is
// unrecoverable: a trap (e.g. a Rust panic compiled to `unreachable`) leaves
// its globals corrupted, so every later call re-traps. Returns a description
// of what crashed.
async function recoverWasm(fallback) {
  const cause = lastPanic || fallback
  lastPanic = null
  wasm = null
  try {
    wasm = await loadWasm()
  }
  catch (error) {
    return cause + "\n(failed to restart the Woxi kernel: " + error.message + ")"
  }
  return cause
}

self.onmessage = async function (e) {
  const { type, code } = e.data

  if (type === "init") {
    await initWasm()
    return
  }

  if (type === "clear") {
    if (wasm) wasm.clear()
    return
  }

  if (type === "set_theme") {
    if (wasm) wasm.set_dark_mode(e.data.dark)
    return
  }

  if (type === "evaluate") {
    if (!wasm) {
      postMessage({
        type: "result",
        success: false,
        message: "WASM module not loaded",
      })
      return
    }

    try {
      lastPanic = null
      // Stream output one statement at a time so Print/Pause/Print
      // sequences appear progressively rather than batched.
      const stmts = JSON.parse(wasm.split_statements(code))
      for (const stmt of stmts) {
        const partial = wasm.evaluate_statement(stmt)
        postMessage({ type: "partial_result", success: true, result: partial })
      }
      postMessage({ type: "result_done", success: true })
    }
    catch (error) {
      // A thrown exception here is a WASM trap (handled WL errors come back as
      // strings). Reinstantiate so the next cell works, and report the cause.
      const cause = await recoverWasm(error.message)
      postMessage({
        type: "result_done",
        success: false,
        message:
          "Error: the Woxi kernel hit an internal error and was " +
          "automatically restarted. All definitions have been cleared — " +
          "re-run earlier cells to continue.\nCause: " + cause,
        restarted: true,
      })
    }
  }

  // Re-evaluate a Manipulate body with a specific set of variable
  // bindings.  Triggered by the main thread when the user drags a
  // slider or changes a dropdown inside an interactive Manipulate cell.
  if (type === "evaluate_manipulate") {
    if (!wasm) {
      postMessage({
        type: "manipulate_result",
        requestId: e.data.requestId,
        success: false,
        message: "WASM module not loaded",
      })
      return
    }
    try {
      lastPanic = null
      const body = e.data.body
      const bindings = JSON.stringify(e.data.bindings || {})
      const result = wasm.evaluate_manipulate(body, bindings)
      postMessage({
        type: "manipulate_result",
        requestId: e.data.requestId,
        success: true,
        result,
      })
    }
    catch (error) {
      const cause = await recoverWasm(error.message)
      postMessage({
        type: "manipulate_result",
        requestId: e.data.requestId,
        success: false,
        message: "Error: the Woxi kernel restarted after an internal error. " +
          "Cause: " + cause,
        restarted: true,
      })
    }
  }
}
