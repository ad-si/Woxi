import { initWoxi, evaluateCode, clearState, isReady } from "./woxi.js"
import { sendMessage } from "./api.js"
import {
  getSettings, saveSettings, hasApiKey, hasBothApiKeys, getConversationIndex,
  getConversation, createConversation, deleteConversation,
  appendMessage, getMessages, clearAllData, truncateMessages, updateToolMessage,
} from "./chat.js"
import {
  createMessageElement, createStreamingAssistant, appendStreamingText,
  finalizeStreamingMessage, appendStreamingToolCard, updateToolCard,
  showToast, scrollToBottom, createToolCard,
} from "./render.js"

// --- DOM refs ---
const sidebar = document.getElementById("sidebar")
const sidebarOverlay = document.getElementById("sidebar-overlay")
const sidebarOpenBtn = document.getElementById("sidebar-open-btn")
const sidebarCloseBtn = document.getElementById("sidebar-close-btn")
const newChatBtn = document.getElementById("new-chat-btn")
const conversationList = document.getElementById("conversation-list")
const settingsBtn = document.getElementById("settings-btn")
const settingsModal = document.getElementById("settings-modal")
const settingsCloseBtn = document.getElementById("settings-close-btn")
const saveSettingsBtn = document.getElementById("save-settings-btn")
const clearDataBtn = document.getElementById("clear-data-btn")
const providerBtns = document.querySelectorAll(".provider-btn")
const openaiSettings = document.getElementById("openai-settings")
const anthropicSettings = document.getElementById("anthropic-settings")
const openaiKeyInput = document.getElementById("openai-key")
const anthropicKeyInput = document.getElementById("anthropic-key")
const messagesEl = document.getElementById("messages")
const chatView = document.getElementById("chat-view")
const welcomeEl = document.getElementById("welcome")
const wasmStatus = document.getElementById("wasm-status")
const settingsIntro = document.getElementById("settings-intro")

// Both input/button pairs
const inputEls = document.querySelectorAll(".chat-input")
const sendBtns = document.querySelectorAll(".send-btn")
const sendIcons = document.querySelectorAll(".send-icon")
const stopIcons = document.querySelectorAll(".stop-icon")
const providerToggles = document.querySelectorAll(".provider-toggle")

// --- State ---
let activeConvId = null
let abortController = null
let isSending = false
let selectedProvider = "openai"

const PROVIDER_LABELS = { openai: "GPT 5.2 Codex", anthropic: "Claude Opus 4.6" }
const MAX_TOOL_CALLS_PER_TURN = 10

/** Returns the currently visible input textarea */
function activeInput() {
  return welcomeEl.classList.contains("hidden")
    ? document.getElementById("input-chat")
    : document.getElementById("input-welcome")
}

// --- Init ---
async function init() {
  applyDarkMode()
  loadSettings()
  renderConversationList()

  // Open most recent conversation or show welcome
  const index = getConversationIndex()
  if (index.length > 0) {
    switchConversation(index[0].id)
  } else {
    showWelcome()
  }

  updateInputState()

  // Auto-open settings if no API key configured
  if (!hasApiKey()) {
    openSettings(true)
  } else {
    activeInput().focus()
  }

  // Init WASM
  try {
    await initWoxi()
    wasmStatus.textContent = "WASM Ready"
    wasmStatus.className = "text-xs px-2 py-1 rounded-full bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300"
  } catch (e) {
    wasmStatus.textContent = "WASM Failed"
    wasmStatus.className = "text-xs px-2 py-1 rounded-full bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300"
    showToast("Failed to load WASM: " + e.message)
  }
}

function applyDarkMode() {
  if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
    document.documentElement.classList.add("dark")
  }
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    document.documentElement.classList.toggle("dark", e.matches)
  })
}

// --- Settings ---
function loadSettings() {
  const s = getSettings()
  selectedProvider = s.provider
  openaiKeyInput.value = s.openai_key || ""
  anthropicKeyInput.value = s.anthropic_key || ""
  updateProviderUI()
}

function updateProviderUI() {
  providerBtns.forEach((btn) => {
    const active = btn.dataset.provider === selectedProvider
    btn.classList.toggle("bg-blue-600", active)
    btn.classList.toggle("text-white", active)
    btn.classList.toggle("border-blue-600", active)
  })
  openaiSettings.classList.toggle("hidden", selectedProvider !== "openai")
  anthropicSettings.classList.toggle("hidden", selectedProvider !== "anthropic")
}

function openSettings(showIntro = false) {
  loadSettings()
  settingsIntro.classList.toggle("hidden", !showIntro)
  settingsModal.classList.remove("hidden")
}

function closeSettings() {
  settingsModal.classList.add("hidden")
}

// --- Sidebar ---
function openSidebar() {
  sidebar.classList.remove("-translate-x-full")
  sidebarOverlay.classList.remove("hidden")
}

function closeSidebar() {
  sidebar.classList.add("-translate-x-full")
  sidebarOverlay.classList.add("hidden")
}

function renderConversationList() {
  const index = getConversationIndex()
  conversationList.innerHTML = ""
  for (const c of index) {
    const item = document.createElement("div")
    item.className = `conv-item${c.id === activeConvId ? " active" : ""}`
    item.dataset.id = c.id

    const title = document.createElement("span")
    title.className = "truncate"
    title.textContent = c.title
    item.appendChild(title)

    const delBtn = document.createElement("button")
    delBtn.className = "delete-btn"
    delBtn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>'
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation()
      handleDeleteConversation(c.id)
    })
    item.appendChild(delBtn)

    item.addEventListener("click", () => {
      switchConversation(c.id)
      closeSidebar()
    })
    conversationList.appendChild(item)
  }
}

// --- Conversation ---
function switchConversation(id) {
  activeConvId = id
  renderConversationList()
  renderMessages()

  // Check if conversation has user messages
  const messages = getMessages(id)
  const hasUserMessages = messages.some((m) => m.role === "user")
  if (hasUserMessages) {
    showChatView()
  } else {
    showWelcome()
  }
  updateInputState()
}

function renderMessages() {
  messagesEl.innerHTML = ""
  if (!activeConvId) return

  const messages = getMessages(activeConvId)
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    if (msg.role === "system") continue

    if (msg.role === "tool") {
      continue
    }

    const el = createMessageElement(msg, i)
    if (!el) continue

    if (msg.role === "assistant" && msg.tool_calls) {
      messagesEl.appendChild(el)
      const inner = el.querySelector(".max-w-3xl")
      if (inner) {
        inner.querySelectorAll(".tool-card").forEach((tc) => tc.remove())
        for (const tc of msg.tool_calls) {
          const args = typeof tc.function.arguments === "string"
            ? JSON.parse(tc.function.arguments)
            : tc.function.arguments
          const toolResult = messages.find(
            (m, j) => j > i && m.role === "tool" && m.tool_call_id === tc.id
          )
          inner.appendChild(
            createToolCard(
              tc.id,
              args.code,
              toolResult?.content,
              toolResult?.content?.startsWith("Error:"),
              toolResult?.graphics
            )
          )
        }
      }
      continue
    }

    messagesEl.appendChild(el)
  }
  scrollToBottom()
}

function showWelcome() {
  welcomeEl.classList.remove("hidden")
  chatView.classList.add("hidden")
}

function showChatView() {
  welcomeEl.classList.add("hidden")
  chatView.classList.remove("hidden")
}

function handleNewChat() {
  const conv = createConversation()
  clearState()
  activeConvId = conv.id
  renderConversationList()
  messagesEl.innerHTML = ""
  showWelcome()
  updateInputState()
  closeSidebar()
  activeInput().focus()
}

function handleDeleteConversation(id) {
  const remaining = deleteConversation(id)
  if (id === activeConvId) {
    if (remaining.length > 0) {
      switchConversation(remaining[0].id)
    } else {
      activeConvId = null
      messagesEl.innerHTML = ""
      showWelcome()
    }
  }
  renderConversationList()
}

// --- Input ---
function updateInputState() {
  const canSend = hasApiKey() && !isSending
  inputEls.forEach((el) => { el.disabled = !hasApiKey() })
  sendBtns.forEach((btn) => { btn.disabled = !canSend })

  if (isSending) {
    sendIcons.forEach((el) => el.classList.add("hidden"))
    stopIcons.forEach((el) => el.classList.remove("hidden"))
    sendBtns.forEach((btn) => { btn.disabled = false })
  } else {
    sendIcons.forEach((el) => el.classList.remove("hidden"))
    stopIcons.forEach((el) => el.classList.add("hidden"))
  }

  const showToggle = hasBothApiKeys()
  providerToggles.forEach((btn) => {
    btn.classList.toggle("hidden", !showToggle)
    const label = btn.querySelector(".provider-label")
    if (label) label.textContent = PROVIDER_LABELS[selectedProvider] || selectedProvider
  })
}

async function handleSend() {
  const input = activeInput()
  const text = input.value.trim()
  if (!text || isSending) return

  if (!hasApiKey()) {
    openSettings(true)
    return
  }

  if (!activeConvId) {
    const conv = createConversation()
    clearState()
    activeConvId = conv.id
    renderConversationList()
  }

  // Switch to chat view before appending message
  showChatView()

  // Append user message
  const userMsg = { role: "user", content: text }
  appendMessage(activeConvId, userMsg)
  renderConversationList()
  const msgIndex = getMessages(activeConvId).length - 1
  const userEl = createMessageElement(userMsg, msgIndex)
  messagesEl.appendChild(userEl)
  scrollToBottom()

  // Clear both inputs
  inputEls.forEach((el) => { el.value = "" })

  await runAssistantTurn()
}

async function runAssistantTurn() {
  isSending = true
  updateInputState()
  abortController = new AbortController()

  let toolCallCount = 0

  try {
    let continueLoop = true

    while (continueLoop && toolCallCount < MAX_TOOL_CALLS_PER_TURN) {
      continueLoop = false
      const messages = getMessages(activeConvId)
      const settings = getSettings()
      const apiKey = settings.provider === "openai" ? settings.openai_key : settings.anthropic_key

      const streamEl = createStreamingAssistant()
      messagesEl.appendChild(streamEl)
      scrollToBottom()

      let assistantMsg = null
      let pendingToolCalls = []

      await new Promise((resolve, reject) => {
        sendMessage(messages, {
          provider: settings.provider,
          apiKey,
          signal: abortController.signal,

          onToken(text) {
            appendStreamingText(text)
            scrollToBottom()
          },

          onToolCall(tc) {
            pendingToolCalls.push(tc)
            appendStreamingToolCard(tc.id, tc.arguments.code)
            scrollToBottom()
          },

          onDone(msg) {
            assistantMsg = msg
            resolve()
          },

          onError(err) {
            reject(err)
          },
        })
      })

      if (!assistantMsg) {
        finalizeStreamingMessage()
        break
      }

      const assistantMsgIndex = getMessages(activeConvId).length
      appendMessage(activeConvId, assistantMsg)
      finalizeStreamingMessage(assistantMsgIndex)

      if (pendingToolCalls.length > 0) {
        for (const tc of pendingToolCalls) {
          toolCallCount++
          let result, isError = false, graphics = ""

          let warnings = ""
          if (!isReady()) {
            result = "Error: WASM interpreter not loaded"
            isError = true
          } else {
            try {
              const evalResult = await evaluateCode(tc.arguments.code)
              result = evalResult.result
              graphics = evalResult.graphics || ""
              warnings = evalResult.warnings || ""
              isError = result.startsWith("Error:")
                || warnings.includes("not yet implemented")
            } catch (e) {
              result = "Error: " + e.message
              isError = true
            }
          }

          updateToolCard(tc.id, result, isError, graphics, warnings)
          scrollToBottom()

          const content = warnings ? `${result}\n\nWarning: ${warnings}` : result
          const toolMsg = { role: "tool", tool_call_id: tc.id, content }
          if (graphics) toolMsg.graphics = graphics
          appendMessage(activeConvId, toolMsg)
        }

        continueLoop = true
      }
    }

    if (toolCallCount >= MAX_TOOL_CALLS_PER_TURN) {
      showToast("Reached maximum tool calls per turn", "info")
    }
  } catch (e) {
    if (e.name === "AbortError") {
      finalizeStreamingMessage()
    } else {
      finalizeStreamingMessage()
      showToast(e.message)

      const errorEl = document.createElement("div")
      errorEl.className = "px-4 py-3"
      errorEl.innerHTML = `<div class="max-w-3xl mx-auto text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 rounded-lg px-4 py-2">${escapeHtml(e.message)}</div>`
      messagesEl.appendChild(errorEl)
      scrollToBottom()
    }
  } finally {
    isSending = false
    abortController = null
    updateInputState()
    activeInput().focus()
  }
}

function handleStop() {
  if (abortController) {
    abortController.abort()
  }
}

function escapeHtml(text) {
  const div = document.createElement("div")
  div.textContent = text
  return div.innerHTML
}

// --- Event listeners ---
sidebarOpenBtn.addEventListener("click", openSidebar)
sidebarCloseBtn.addEventListener("click", closeSidebar)
sidebarOverlay.addEventListener("click", closeSidebar)
newChatBtn.addEventListener("click", handleNewChat)
settingsBtn.addEventListener("click", openSettings)
settingsCloseBtn.addEventListener("click", closeSettings)

settingsModal.addEventListener("click", (e) => {
  if (e.target === settingsModal) closeSettings()
})

providerBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    selectedProvider = btn.dataset.provider
    updateProviderUI()
  })
})

saveSettingsBtn.addEventListener("click", () => {
  saveSettings({
    provider: selectedProvider,
    openai_key: openaiKeyInput.value.trim(),
    anthropic_key: anthropicKeyInput.value.trim(),
  })
  closeSettings()
  updateInputState()
  showToast("Settings saved", "success")
})

clearDataBtn.addEventListener("click", () => {
  if (confirm("This will delete all conversations and settings. Continue?")) {
    clearAllData()
    activeConvId = null
    messagesEl.innerHTML = ""
    renderConversationList()
    showWelcome()
    loadSettings()
    updateInputState()
    showToast("All data cleared", "info")
  }
})

providerToggles.forEach((btn) => {
  btn.addEventListener("click", () => {
    const s = getSettings()
    selectedProvider = selectedProvider === "openai" ? "anthropic" : "openai"
    s.provider = selectedProvider
    saveSettings(s)
    updateProviderUI()
    updateInputState()
  })
})

document.querySelectorAll(".input-container").forEach((container) => {
  container.addEventListener("click", (e) => {
    if (e.target.closest("button")) return
    container.querySelector("textarea")?.focus()
  })
})

sendBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    if (isSending) handleStop()
    else handleSend()
  })
})

inputEls.forEach((el) => {
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  })
})

document.querySelectorAll(".example-prompt").forEach((btn) => {
  btn.addEventListener("click", () => {
    const input = activeInput()
    input.value = btn.textContent.trim()
    input.focus()
  })
})

// --- Menu ---
const menuBtn = document.getElementById("menu-btn")
const menuDropdown = document.getElementById("menu-dropdown")
const exportChatBtn = document.getElementById("export-chat-btn")
const exportNotebookBtn = document.getElementById("export-notebook-btn")

menuBtn.addEventListener("click", (e) => {
  e.stopPropagation()
  menuDropdown.classList.toggle("hidden")
})

document.addEventListener("click", () => {
  menuDropdown.classList.add("hidden")
})

menuDropdown.addEventListener("click", (e) => {
  e.stopPropagation()
})

exportChatBtn.addEventListener("click", () => {
  menuDropdown.classList.add("hidden")

  if (!activeConvId) {
    showToast("No conversation to export", "info")
    return
  }

  const conv = getConversation(activeConvId)
  if (!conv) {
    showToast("Conversation not found", "info")
    return
  }

  const blob = new Blob([JSON.stringify(conv, null, 2)], { type: "application/json" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `${exportTimestamp()}-woxi-chat-${conv.title.replace(/[^a-zA-Z0-9]/g, "_").slice(0, 30)}.json`
  a.click()
  URL.revokeObjectURL(url)
})

exportNotebookBtn.addEventListener("click", () => {
  menuDropdown.classList.add("hidden")

  if (!activeConvId) {
    showToast("No conversation to export", "info")
    return
  }

  const conv = getConversation(activeConvId)
  if (!conv) {
    showToast("Conversation not found", "info")
    return
  }

  const cells = []
  const messages = conv.messages
  let execCount = 1

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]

    if (msg.role === "system") continue

    if (msg.role === "user") {
      cells.push({
        id: crypto.randomUUID(),
        cell_type: "markdown",
        metadata: { trusted: true },
        source: splitLines(`**User:** ${msg.content}`),
      })
    } else if (msg.role === "assistant") {
      if (msg.content) {
        cells.push({
          id: crypto.randomUUID(),
          cell_type: "markdown",
          metadata: { trusted: true },
          source: splitLines(`**Assistant:** ${msg.content}`),
        })
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          const args = typeof tc.function.arguments === "string"
            ? JSON.parse(tc.function.arguments)
            : tc.function.arguments
          const toolResult = messages.find(
            (m, j) => j > i && m.role === "tool" && m.tool_call_id === tc.id
          )
          const n = execCount++
          const outputs = []
          if (toolResult?.graphics) {
            outputs.push({
              output_type: "execute_result",
              metadata: {},
              data: {
                "image/svg+xml": toolResult.graphics,
                "text/plain": "-Graphics-",
              },
              execution_count: n,
            })
          } else if (toolResult?.content) {
            outputs.push({
              output_type: "execute_result",
              metadata: {},
              data: { "text/plain": splitLines(toolResult.content) },
              execution_count: n,
            })
          }
          cells.push({
            id: crypto.randomUUID(),
            cell_type: "code",
            metadata: { trusted: true },
            source: splitLines(args.code),
            outputs,
            execution_count: n,
          })
        }
      }
    }
    // Skip tool messages — their content is already attached to code cells above
  }

  const notebook = {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: {
        name: "woxi",
        display_name: "Woxi (Wolfram Language)",
        language: "wolfram",
      },
      language_info: {
        codemirror_mode: { name: "mathematica" },
        file_extension: ".wls",
        mimetype: "application/vnd.wolfram.mathematica",
        name: "wolfram",
        version: "0.1.0",
      },
    },
    cells,
  }

  const blob = new Blob([JSON.stringify(notebook, null, 1)], { type: "application/x-ipynb+json" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `${exportTimestamp()}-woxi-chat-${conv.title.replace(/[^a-zA-Z0-9]/g, "_").slice(0, 30)}.ipynb`
  a.click()
  URL.revokeObjectURL(url)
})

/** Return a timestamp like "2026-02-13t1537" */
function exportTimestamp() {
  const d = new Date()
  const pad = (n) => String(n).padStart(2, "0")
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}t${pad(d.getHours())}${pad(d.getMinutes())}`
}

/** Split text into notebook-style source lines (each ending with \n except the last) */
function splitLines(text) {
  const lines = text.split("\n")
  return lines.map((line, i) => i < lines.length - 1 ? line + "\n" : line)
}

// --- Copy tool code + result ---
messagesEl.addEventListener("click", (e) => {
  const copyBtn = e.target.closest(".copy-result-btn")
  if (!copyBtn) return
  const card = copyBtn.closest(".tool-card")
  if (!card) return
  const code = card.querySelector(".tool-card-code")?.textContent || ""
  const result = card.querySelector(".tool-card-result:not(.warning):not(.graphics)")?.textContent || ""
  const text = result ? `${code}\n\n${result}` : code
  navigator.clipboard.writeText(text)
  showToast("Copied to clipboard", "success")
})

// --- Recalculate tool card ---
messagesEl.addEventListener("click", async (e) => {
  const recalcBtn = e.target.closest(".recalc-btn")
  if (!recalcBtn) return

  const card = recalcBtn.closest(".tool-card")
  if (!card) return
  if (recalcBtn.classList.contains("recalculating")) return

  const codeEl = card.querySelector(".tool-card-code")
  if (!codeEl) return
  const code = codeEl.textContent

  if (!isReady()) {
    showToast("WASM interpreter not loaded")
    return
  }

  recalcBtn.classList.add("recalculating")

  // Remove old result/warning/graphics elements
  card.querySelectorAll(".tool-card-result").forEach((el) => el.remove())
  card.classList.remove("collapsed")

  try {
    const evalResult = await evaluateCode(code)
    const result = evalResult.result
    const graphics = evalResult.graphics || ""
    const warnings = evalResult.warnings || ""
    const isError = result.startsWith("Error:")

    // Update header status icon (the one that's not .chevron and not inside .recalc-btn)
    const header = card.querySelector(".tool-card-header")
    const icons = header.querySelectorAll(":scope > svg:not(.chevron)")
    const statusIcon = icons.length > 0 ? icons[0] : null
    if (statusIcon) {
      statusIcon.outerHTML = isError
        ? '<svg class="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>'
        : '<svg class="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>'
    }

    updateToolCard(card.dataset.toolCallId, result, isError, graphics, warnings)

    // Persist updated result to localStorage
    if (activeConvId) {
      const content = warnings ? `${result}\n\nWarning: ${warnings}` : result
      updateToolMessage(activeConvId, card.dataset.toolCallId, content, graphics)
    }
  } catch (err) {
    updateToolCard(card.dataset.toolCallId, "Error: " + err.message, true, "", "")
  } finally {
    recalcBtn.classList.remove("recalculating")
  }
})

// --- Message actions (copy / edit / retry) ---
messagesEl.addEventListener("click", (e) => {
  const btn = e.target.closest(".msg-action-btn")
  if (!btn) return
  const action = btn.dataset.action

  // Assistant message actions
  if (action === "copy-assistant" || action === "retry-assistant") {
    const row = btn.closest(".assistant-message-row")
    if (!row || !row.dataset.msgIndex) return
    const msgIndex = parseInt(row.dataset.msgIndex, 10)
    if (action === "copy-assistant") handleCopyAssistantMessage(msgIndex)
    else handleRetryAssistantMessage(msgIndex)
    return
  }

  // User message actions
  const row = btn.closest(".user-message-row")
  if (!row || !row.dataset.msgIndex) return
  const msgIndex = parseInt(row.dataset.msgIndex, 10)

  if (action === "copy") handleCopyMessage(msgIndex)
  else if (action === "edit") handleEditMessage(msgIndex)
  else if (action === "retry") handleRetryMessage(msgIndex)
})

function handleCopyMessage(msgIndex) {
  const messages = getMessages(activeConvId)
  const msg = messages[msgIndex]
  if (!msg) return
  navigator.clipboard.writeText(msg.content)
  showToast("Copied to clipboard", "success")
}

function handleEditMessage(msgIndex) {
  if (isSending) return
  const messages = getMessages(activeConvId)
  const msg = messages[msgIndex]
  if (!msg) return

  const row = messagesEl.querySelector(`.user-message-row[data-msg-index="${msgIndex}"]`)
  if (!row) return

  // Already editing
  if (row.querySelector(".edit-area")) return

  const bubble = row.querySelector(".rounded-2xl")
  const actions = row.querySelector(".user-msg-actions")
  bubble.classList.add("hidden")
  actions.classList.add("hidden")

  const editContainer = document.createElement("div")
  editContainer.className = "edit-area flex flex-col items-end gap-2 max-w-[80%]"

  const textarea = document.createElement("textarea")
  textarea.className = "w-full rounded-2xl bg-blue-600 text-white text-sm px-4 py-2.5 resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
  textarea.value = msg.content

  const btnRow = document.createElement("div")
  btnRow.className = "flex gap-2"

  const cancelBtn = document.createElement("button")
  cancelBtn.className = "px-3 py-1 text-xs rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600"
  cancelBtn.textContent = "Cancel"

  const saveBtn = document.createElement("button")
  saveBtn.className = "px-3 py-1 text-xs rounded-lg bg-blue-600 text-white hover:bg-blue-700"
  saveBtn.textContent = "Save & Send"

  btnRow.appendChild(cancelBtn)
  btnRow.appendChild(saveBtn)
  editContainer.appendChild(textarea)
  editContainer.appendChild(btnRow)
  row.appendChild(editContainer)
  textarea.focus()

  cancelBtn.addEventListener("click", () => {
    editContainer.remove()
    bubble.classList.remove("hidden")
    actions.classList.remove("hidden")
  })

  saveBtn.addEventListener("click", () => {
    const newText = textarea.value.trim()
    if (!newText) return

    truncateMessages(activeConvId, msgIndex)
    const userMsg = { role: "user", content: newText }
    appendMessage(activeConvId, userMsg)
    renderMessages()
    scrollToBottom()
    runAssistantTurn()
  })

  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      saveBtn.click()
    }
    if (e.key === "Escape") {
      cancelBtn.click()
    }
  })
}

function handleRetryMessage(msgIndex) {
  if (isSending) return
  const messages = getMessages(activeConvId)
  const msg = messages[msgIndex]
  if (!msg) return

  truncateMessages(activeConvId, msgIndex)

  const userMsg = { role: "user", content: msg.content }
  appendMessage(activeConvId, userMsg)
  renderMessages()
  scrollToBottom()

  runAssistantTurn()
}

function handleCopyAssistantMessage(msgIndex) {
  const messages = getMessages(activeConvId)
  const msg = messages[msgIndex]
  if (!msg || !msg.content) return
  navigator.clipboard.writeText(msg.content)
  showToast("Copied to clipboard", "success")
}

function handleRetryAssistantMessage(msgIndex) {
  if (isSending) return
  truncateMessages(activeConvId, msgIndex)
  renderMessages()
  scrollToBottom()
  runAssistantTurn()
}

// --- Start ---
init()
