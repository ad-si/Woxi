let wasm = null

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
      const result = wasm.evaluate(code)
      const warnings = wasm.get_warnings()
      const graphics = wasm.get_graphics()
      const outputSvg = wasm.get_output_svg()
      postMessage({ type: "result", success: true, result, warnings, graphics, outputSvg })
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
