const SETTINGS_KEY = "woxi_chat_settings"
const CONVERSATIONS_KEY = "woxi_chat_conversations"
const CONV_PREFIX = "woxi_chat_conv_"

const SYSTEM_PROMPT = `
  You are a helpful assistant with access to a Wolfram Language interpreter called Woxi.
  When the user asks a mathematical, computational, or symbolic question,
    use the evaluate_wolfram tool to compute the answer.
  Keep the reasoning to a minimum and focus on
    providing the Wolfram Language code that will produce the answer.
  Do not repeat the output of the code if you don't have anything important to add to it.
  The interpreter maintains state across calls in this conversation —
    you can define variables and functions and use them in subsequent evaluations.
  The suggested Wolfram Language code must not contain any graphical functions
    (such as Plot, ListPlot, Histogram, Graphics, Show, etc.)
    as the environment does not support displaying them.
  If the code causes any error, try some other code to get the answer
    and don't try exactly the same code again.
  Even if the error message implies that a function was used incorrectly,
    it might actually be the case that the function is actually not defined in the environment.
  If the final result is a symbolic expression,
    always also show the numerical approximation of it using N[] for better readability.
  Use camelCase for function names and variables.
`

export function getSettings() {
  const raw = localStorage.getItem(SETTINGS_KEY)
  if (!raw) return { provider: "openai", openai_key: "", anthropic_key: "" }
  return JSON.parse(raw)
}

export function saveSettings(settings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
}

export function hasApiKey() {
  const s = getSettings()
  if (s.provider === "openai") return !!s.openai_key
  if (s.provider === "anthropic") return !!s.anthropic_key
  return false
}

export function getConversationIndex() {
  const raw = localStorage.getItem(CONVERSATIONS_KEY)
  if (!raw) return []
  return JSON.parse(raw)
}

function saveConversationIndex(index) {
  localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(index))
}

export function getConversation(id) {
  const raw = localStorage.getItem(CONV_PREFIX + id)
  if (!raw) return null
  return JSON.parse(raw)
}

function saveConversation(conv) {
  localStorage.setItem(CONV_PREFIX + conv.id, JSON.stringify(conv))
}

export function createConversation() {
  const id = crypto.randomUUID()
  const conv = {
    id,
    title: "New Chat",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [{
      role: "system",
      content: SYSTEM_PROMPT.replaceAll(/\s+/g, " ").trim(),
    }],
  }
  saveConversation(conv)

  const index = getConversationIndex()
  index.unshift({ id, title: conv.title, updatedAt: conv.updatedAt })
  saveConversationIndex(index)

  return conv
}

export function deleteConversation(id) {
  localStorage.removeItem(CONV_PREFIX + id)
  const index = getConversationIndex().filter((c) => c.id !== id)
  saveConversationIndex(index)
  return index
}

export function appendMessage(convId, message) {
  const conv = getConversation(convId)
  if (!conv) return
  conv.messages.push(message)
  conv.updatedAt = Date.now()

  // Set title from first user message
  if (
    message.role === "user" &&
    conv.messages.filter((m) => m.role === "user").length === 1
  ) {
    conv.title = message.content.slice(0, 40)
    const index = getConversationIndex()
    const entry = index.find((c) => c.id === convId)
    if (entry) {
      entry.title = conv.title
      entry.updatedAt = conv.updatedAt
    }
    saveConversationIndex(index)
  } else {
    const index = getConversationIndex()
    const entry = index.find((c) => c.id === convId)
    if (entry) entry.updatedAt = conv.updatedAt
    saveConversationIndex(index)
  }

  saveConversation(conv)
}

export function getMessages(convId) {
  const conv = getConversation(convId)
  if (!conv) return []
  return conv.messages
}

export function truncateMessages(convId, fromIndex) {
  const conv = getConversation(convId)
  if (!conv) return
  conv.messages = conv.messages.slice(0, fromIndex)
  conv.updatedAt = Date.now()
  saveConversation(conv)
}

export function clearAllData() {
  const index = getConversationIndex()
  for (const c of index) {
    localStorage.removeItem(CONV_PREFIX + c.id)
  }
  localStorage.removeItem(CONVERSATIONS_KEY)
  localStorage.removeItem(SETTINGS_KEY)
}
