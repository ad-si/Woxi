let worker = null
let ready = false
let initResolve = null
let initReject = null
let evalResolve = null
let evalReject = null
let evalTimeout = null

export function isReady() {
  return ready
}

export function initWoxi() {
  return new Promise((resolve, reject) => {
    initResolve = resolve
    initReject = reject
    worker = new Worker("worker.js", { type: "module" })
    worker.onmessage = onMessage
    worker.onerror = (e) => {
      ready = false
      if (initReject) {
        initReject(new Error(e.message))
        initReject = null
        initResolve = null
      }
    }
    worker.postMessage({ type: "init" })
  })
}

export function evaluateCode(code) {
  if (!ready) return Promise.reject(new Error("WASM not loaded"))
  return new Promise((resolve, reject) => {
    evalResolve = resolve
    evalReject = reject
    evalTimeout = setTimeout(() => {
      evalReject(new Error("Evaluation timed out (20s)"))
      evalResolve = null
      evalReject = null
      evalTimeout = null
    }, 20000)
    worker.postMessage({ type: "evaluate", code })
  })
}

export function clearState() {
  if (worker && ready) {
    worker.postMessage({ type: "clear" })
  }
}

function onMessage(e) {
  const { type, success, message, result, graphics, warnings } = e.data

  if (type === "init") {
    if (success) {
      ready = true
      if (initResolve) initResolve()
    } else {
      ready = false
      if (initReject) initReject(new Error(message))
    }
    initResolve = null
    initReject = null
    return
  }

  if (type === "result") {
    if (evalTimeout) {
      clearTimeout(evalTimeout)
      evalTimeout = null
    }
    if (success) {
      if (evalResolve) evalResolve({ result, graphics, warnings })
    } else {
      if (evalReject) evalReject(new Error(message))
    }
    evalResolve = null
    evalReject = null
  }
}
