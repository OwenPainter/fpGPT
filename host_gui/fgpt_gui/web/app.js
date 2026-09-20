"use strict";

// fpGPT host GUI frontend.
// Talks to the stdlib host bridge over SSE (/api/events) for the streaming
// reply and POST (/api/prompt, /api/reset) for commands.

const el = (id) => document.getElementById(id);
const chat = el("chat");
const promptBox = el("prompt");
const sendBtn = el("send");
const resetBtn = el("reset");
const exportBtn = el("export");

const state = {
  config: null,
  params: null,
  charMin: 32,
  charMax: 126,
  genTokens: 16,
  busy: false,
  current: null, // { node, text }
  transcript: [], // { role, text }
  foodFile: null,
  foodBusy: false,
};

function badge(id, text) {
  el(id).textContent = text;
}

function setConnected(online) {
  el("conn-dot").classList.toggle("online", online);
}

function setState(text, cls) {
  const node = el("state-text");
  node.textContent = text;
  node.className = "state " + (cls || "");
}

function addMessage(role, text, cls) {
  const node = document.createElement("div");
  node.className = "msg " + (cls || role);
  const label = document.createElement("span");
  label.className = "role";
  label.textContent = role;
  node.appendChild(label);
  const body = document.createElement("span");
  body.className = "body";
  body.textContent = text || "";
  node.appendChild(body);
  chat.appendChild(node);
  chat.scrollTop = chat.scrollHeight;
  return { node, body };
}

function appendToken(char) {
  if (!state.current) {
    state.current = addMessage("assistant", "");
  }
  state.current.body.textContent += char;
  state.current.node.classList.add("cursor");
  chat.scrollTop = chat.scrollHeight;
}

function endReply(reason) {
  if (state.current) {
    state.current.node.classList.remove("cursor");
    state.transcript.push({ role: "assistant", text: state.current.body.textContent });
    state.current = null;
  }
  if (reason && reason !== "complete" && reason !== "ok") {
    addMessage("system", "reply ended: " + reason, "system");
  }
}

function handleEvent(event) {
  switch (event.type) {
    case "hello":
      state.config = event.config;
      applyConfig(event.config);
      setConnected(true);
      setState("idle", "idle");
      break;
    case "info":
      if (event.params) {
        state.params = event.params;
        applyParams(event.params);
      }
      break;
    case "status":
      state.busy = !!event.busy;
      sendBtn.disabled = state.busy;
      if (event.state === "error") {
        setState("error", "error");
      } else if (state.busy) {
        setState("generating\u2026", "busy");
      } else {
        setState("idle", "idle");
      }
      break;
    case "token":
      appendToken(event.char);
      break;
    case "reply_end":
      endReply(event.reason);
      break;
    case "error":
      addMessage("system", "error: " + event.message, "system error");
      setState("error", "error");
      break;
    case "ready":
      addMessage("system", "board ready (protocol v" + event.protocol_version + ")", "system");
      break;
    default:
      break;
  }
}

function applyConfig(config) {
  if (config.slm_engine) {
    const sel = el("model-selector");
    if (sel && sel.value !== config.slm_engine) {
      sel.value = config.slm_engine;
    }
    // Show/hide port controls based on engine
    const portControls = el("fpga-port-controls");
    if (portControls) {
      portControls.style.display = config.slm_engine === "fpga" ? "inline" : "none";
    }
  }

  if (config.slm_engine === "fpga" || config.transport === "serial") {
    badge("badge-transport", "FPGA: Serial (" + (config.port || "unknown") + ")");
  } else if (config.transport === "slm") {
    badge("badge-transport", "SLM: " + (config.slm_engine || "MicroGPT") + " (Local CPU)");
  } else {
    badge("badge-transport", "transport: " + config.transport
      + (config.mock_style ? " (" + config.mock_style + ")" : ""));
  }
  badge("badge-mode", "mode: " + config.mode);
  state.genTokens = config.gen_tokens;
  if (config.params) applyParams(config.params);
  applyFoodConfig(config.food);
  if (config.warnings && config.warnings.length) {
    config.warnings.forEach((w) => addMessage("system", "warning: " + w, "system"));
  }
  el("gen-info").textContent = "reply length: " + config.gen_tokens + " chars";
}

function applyFoodConfig(food) {
  state.food = food || { available: false };
  if (state.food.model) {
    el("food-model-name").textContent = state.food.model;
  }
  if (state.food.available) {
    badge("badge-food", "food: " + (state.food.model || "ready"));
    el("food-classify").title = "";
  } else {
    badge("badge-food", "food: unavailable");
    el("food-classify").title = state.food.error || "classifier unavailable";
  }
}

function applyParams(params) {
  state.charMin = params.char_min;
  state.charMax = params.char_max;
  badge("badge-model", "model: d" + params.d_model + " L" + params.num_layers
    + " V" + params.vocab_size + " T" + params.max_seq_len);
  el("char-range").textContent =
    "supported: " + String.fromCharCode(params.char_min) + "\u2013"
    + String.fromCharCode(params.char_max);
}

function countInvalid() {
  const text = promptBox.value;
  let invalid = 0;
  for (const ch of text) {
    const code = ch.codePointAt(0);
    if (ch === "\n") continue;
    if (code < state.charMin || code > state.charMax) invalid++;
  }
  const total = text.replace(/\n/g, "").length;
  el("char-count").textContent = total + " chars"
    + (invalid ? " \u00b7 " + invalid + " unsupported" : "");
  el("char-count").style.color = invalid ? "var(--warn)" : "";
}

async function sendPrompt() {
  const text = promptBox.value;
  if (!text.trim() || state.busy) return;
  addMessage("user", text);
  state.transcript.push({ role: "user", text });
  promptBox.value = "";
  countInvalid();
  sendBtn.disabled = true;
  try {
    const response = await fetch("/api/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    if (!data.ok) {
      addMessage("system", "not sent: " + (data.error || "unknown error"), "system error");
      sendBtn.disabled = false;
    }
  } catch (err) {
    addMessage("system", "request failed: " + err, "system error");
    sendBtn.disabled = false;
  }
}

async function resetSession() {
  state.current = null;
  try {
    await fetch("/api/reset", { method: "POST" });
  } catch (err) {
    // ignore; the stream status will reflect reality
  }
}

function exportTranscript() {
  const lines = state.transcript.map(
    (m) => (m.role === "user" ? "> " : "< ") + m.text
  );
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "fpgpt-transcript.txt";
  a.click();
  URL.revokeObjectURL(url);
}

// ── tabs ──
function setActiveTab(panelId) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.panel === panelId;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === panelId);
  });
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => setActiveTab(tab.dataset.panel));
  });
}

// ── food photo tab ──
function setFoodResult(html, cls) {
  const node = el("food-result");
  node.className = "food-result " + (cls || "");
  node.innerHTML = html;
  node.classList.remove("hidden");
}

function clearFood() {
  if (state.foodFile && el("food-preview").src) {
    URL.revokeObjectURL(el("food-preview").src);
  }
  state.foodFile = null;
  el("food-file").value = "";
  el("food-preview").src = "";
  el("food-preview-wrap").classList.add("hidden");
  el("food-result").classList.add("hidden");
  el("food-classify").disabled = true;
  el("food-clear").disabled = true;
}

function selectFoodFile(file) {
  if (!file) return;
  if (!file.type || !file.type.startsWith("image/")) {
    setFoodResult("Please choose an image file.", "error");
    return;
  }
  state.foodFile = file;
  const url = URL.createObjectURL(file);
  const preview = el("food-preview");
  preview.src = url;
  el("food-preview-wrap").classList.remove("hidden");
  el("food-result").classList.add("hidden");
  el("food-classify").disabled = false;
  el("food-clear").disabled = false;
}

function foodScoreHtml(result) {
  const pct = Math.round(result.probability * 1000) / 10;
  const conf = Math.round(result.confidence * 1000) / 10;
  const icon = result.is_hotdog ? "\ud83c\udf2d " : "\u274c ";
  const timeStr = result.inference_ms ? ` &middot; ${result.inference_ms}ms` : "";
  return `
    <div class="verdict">
      <span class="label">${icon}${result.label}</span>
      <span class="score">${conf}% confident</span>
    </div>
    <div class="bar"><span style="width:${pct}%"></span></div>
    <div class="note">P(hot dog) = ${result.probability.toFixed(4)}${timeStr} &middot; threshold ${result.threshold} &middot; ${result.model}</div>
    <div style="margin-top: 0.8rem; text-align: right;">
      <button type="button" class="secondary" id="ask-food-ai" style="font-size:0.8rem; padding:0.35rem 0.75rem;">
        \ud83d\udcac Ask AI about this result
      </button>
    </div>`;
}

async function classifyFood() {
  if (!state.foodFile || state.foodBusy) return;
  state.foodBusy = true;
  el("food-classify").disabled = true;
  setFoodResult("Analyzing&hellip;", "pending");
  try {
    const response = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": state.foodFile.type || "application/octet-stream" },
      body: state.foodFile,
    });
    const data = await response.json();
    if (!data.ok) {
      setFoodResult("Error: " + (data.error || "classification failed"), "error");
    } else {
      setFoodResult(foodScoreHtml(data), data.is_hotdog ? "hotdog" : "nothotdog");
      const askBtn = el("ask-food-ai");
      if (askBtn) {
        askBtn.addEventListener("click", () => {
          el("tab-chat").click();
          promptBox.value = `I just ran a photo through the SeeFood classifier and it resulted in: ${data.label} (${Math.round(data.confidence * 100)}% confidence). What do you think?`;
          sendPrompt();
        });
      }
    }
  } catch (err) {
    setFoodResult("Request failed: " + err, "error");
  } finally {
    state.foodBusy = false;
    el("food-classify").disabled = !state.foodFile;
  }
}

function setupFood() {
  const dropzone = el("dropzone");
  const input = el("food-file");
  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => selectFoodFile(input.files[0]));
  ["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    selectFoodFile(file);
  });
  el("food-classify").addEventListener("click", classifyFood);
  el("food-clear").addEventListener("click", clearFood);
}

function subscribe() {
  const events = new EventSource("/api/events");
  events.onopen = () => setConnected(true);
  events.onerror = () => setConnected(false);
  events.onmessage = (message) => {
    try {
      handleEvent(JSON.parse(message.data));
    } catch (err) {
      console.error("bad event", message.data, err);
    }
  };
}

function setupModelSelector() {
  const selector = el("model-selector");
  const portControls = el("fpga-port-controls");
  const portSelector = el("fpga-port-selector");
  const connectBtn = el("fpga-connect");
  const refreshBtn = el("fpga-refresh");
  if (!selector) return;

  async function refreshPorts() {
    try {
      const resp = await fetch("/api/ports");
      const data = await resp.json();
      // Clear old options (keep the placeholder)
      while (portSelector.options.length > 1) portSelector.remove(1);
      (data.ports || []).forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        portSelector.appendChild(opt);
      });
      if (data.ports && data.ports.length === 0) {
        addMessage("system", "No COM ports detected. Is the ESP32-S2 connected?", "system");
      }
    } catch (err) {
      addMessage("system", "Could not fetch serial ports: " + err, "system error");
    }
  }

  function showPortControls(show) {
    portControls.style.display = show ? "inline" : "none";
  }

  portSelector.addEventListener("change", () => {
    connectBtn.disabled = !portSelector.value;
  });

  refreshBtn.addEventListener("click", refreshPorts);

  connectBtn.addEventListener("click", async () => {
    const port = portSelector.value;
    if (!port) return;
    connectBtn.disabled = true;
    addMessage("system", "Connecting to FPGA on " + port + "...", "system");
    try {
      const response = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine: "fpga", port }),
      });
      const data = await response.json();
      if (!data.ok) {
        addMessage("system", "FPGA connect failed: " + (data.error || "unknown"), "system error");
        connectBtn.disabled = false;
      } else {
        addMessage("system", "Connected to FPGA on " + port + " at 115200 baud.", "system");
      }
    } catch (err) {
      addMessage("system", "FPGA connect error: " + err, "system error");
      connectBtn.disabled = false;
    }
  });

  selector.addEventListener("change", async () => {
    const engine = selector.value;

    if (engine === "fpga") {
      showPortControls(true);
      refreshPorts();
      return; // don't send settings yet — user must pick a port and click Connect
    }

    showPortControls(false);

    addMessage("system", "Switching engine to " + engine + "...", "system");
    try {
      const response = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine }),
      });
      const data = await response.json();
      if (!data.ok) {
        addMessage("system", "Failed to switch engine: " + (data.error || "unknown"), "system error");
      }
    } catch (err) {
      addMessage("system", "Failed to switch engine: " + err, "system error");
    }
  });
}

function init() {
  addMessage("system",
    "Connected to the host bridge. Type a prompt; the FPGA streams the reply.",
    "system");
  sendBtn.addEventListener("click", sendPrompt);
  resetBtn.addEventListener("click", resetSession);
  exportBtn.addEventListener("click", exportTranscript);
  promptBox.addEventListener("input", countInvalid);
  promptBox.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendPrompt();
    }
  });
  const shutdownBtn = el("btn-shutdown");
  if (shutdownBtn) {
    shutdownBtn.addEventListener("click", async () => {
      if (!confirm("Are you sure you want to stop the local fpGPT server?")) return;
      try {
        await fetch("/api/shutdown", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        });
        alert("Server has been stopped cleanly. You can close this browser tab.");
        setConnected(false);
      } catch (err) {
        alert("Server shutdown requested: " + err);
      }
    });
  }
  setupTabs();
  setupFood();
  setupModelSelector();
  countInvalid();
  subscribe();
}

init();