let wasm = null

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
    const module = await import("./pkg/woxi.js")
    await module.default()
    wasm = module
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
      const result = wasm.evaluate_all(code)
      postMessage({ type: "result", success: true, result })
    }
    catch (error) {
      postMessage({
        type: "result",
        success: false,
        message: "Error: " + error.message,
      })
    }
  }
}
