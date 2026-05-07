from __future__ import annotations

CUSTOMER_SERVICE_UI = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>智能客服工作台</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #147d64;
      --accent-2: #2251a4;
      --danger: #b42318;
      --soft: #eef4ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    .status { display: flex; gap: 8px; align-items: center; color: var(--muted); font-size: 13px; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent); }
    main {
      display: grid;
      grid-template-columns: 260px minmax(360px, 1fr) 320px;
      gap: 1px;
      min-height: calc(100vh - 56px);
      background: var(--line);
    }
    aside, section { background: var(--panel); }
    aside { padding: 18px; }
    label { display: block; margin: 0 0 6px; color: var(--muted); font-size: 12px; font-weight: 650; }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
      color: var(--text);
      background: #fff;
    }
    textarea { min-height: 78px; resize: vertical; }
    .field { margin-bottom: 14px; }
    .quick { display: grid; gap: 8px; margin-top: 18px; }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      min-height: 38px;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; font-weight: 650; }
    button:hover { border-color: var(--accent-2); }
    .chat { display: flex; flex-direction: column; min-width: 0; }
    .messages { flex: 1; overflow: auto; padding: 20px; display: grid; align-content: start; gap: 12px; }
    .message { max-width: 78%; padding: 11px 12px; border-radius: 8px; line-height: 1.5; white-space: pre-wrap; }
    .user { justify-self: end; background: var(--accent-2); color: #fff; }
    .assistant { justify-self: start; background: var(--soft); color: var(--text); }
    .composer { display: grid; grid-template-columns: 1fr 96px; gap: 10px; padding: 14px; border-top: 1px solid var(--line); }
    .side-title { margin: 0 0 12px; font-size: 14px; font-weight: 700; }
    .kv { display: grid; gap: 9px; font-size: 13px; }
    .kv div { display: flex; justify-content: space-between; gap: 12px; padding-bottom: 8px; border-bottom: 1px solid #edf0f5; }
    .kv span:first-child { color: var(--muted); }
    pre {
      min-height: 180px;
      overflow: auto;
      margin: 14px 0 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
      font-size: 12px;
      white-space: pre-wrap;
    }
    .handoff { color: var(--danger); font-weight: 650; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { min-height: auto; }
      .message { max-width: 92%; }
    }
  </style>
</head>
<body>
  <header>
    <h1>智能客服工作台</h1>
    <div class="status"><span class="dot"></span><span>offline_stub</span></div>
  </header>
  <main>
    <aside>
      <div class="field">
        <label for="userId">用户</label>
        <input id="userId" value="user_001" />
      </div>
      <div class="field">
        <label for="sessionId">会话</label>
        <input id="sessionId" value="session_001" />
      </div>
      <div class="quick">
        <button data-message="我的订单 ORD123456 到哪了？">订单状态</button>
        <button data-message="帮我查一下物流 ORD123456">物流查询</button>
        <button data-message="AirBuds Pro 2 还有库存吗？">商品库存</button>
        <button data-message="我要给订单 ORD123456 退款">退款申请</button>
        <button data-message="我喜欢顺丰配送，以后发货优先顺丰">保存偏好</button>
        <button data-message="你记得我的配送偏好吗？">召回记忆</button>
        <button data-message="我要投诉，给我转人工">转人工</button>
        <button id="clearMemory" type="button">清除记忆</button>
      </div>
    </aside>
    <section class="chat">
      <div id="messages" class="messages">
        <div class="message assistant">你好，我是智能客服。</div>
      </div>
      <form id="chatForm" class="composer">
        <textarea id="messageInput" placeholder="输入客服问题"></textarea>
        <button class="primary" type="submit">发送</button>
      </form>
    </section>
    <aside>
      <h2 class="side-title">运行状态</h2>
      <div class="kv">
        <div><span>质量分</span><strong id="quality">-</strong></div>
        <div><span>Token</span><strong id="tokens">-</strong></div>
        <div><span>降级状态</span><strong id="degraded">否</strong></div>
        <div><span>转人工</span><strong id="handoff">否</strong></div>
        <div><span>原因</span><strong id="reason">-</strong></div>
      </div>
      <h2 class="side-title" style="margin-top: 20px;">用户记忆</h2>
      <pre id="memories">[]</pre>
      <h2 class="side-title" style="margin-top: 20px;">会话摘要</h2>
      <pre id="summary"></pre>
      <h2 class="side-title" style="margin-top: 20px;">订单上下文</h2>
      <pre id="context">{}</pre>
    </aside>
  </main>
  <script>
    const form = document.querySelector("#chatForm");
    const messages = document.querySelector("#messages");
    const input = document.querySelector("#messageInput");

    function addMessage(text, cls) {
      const node = document.createElement("div");
      node.className = `message ${cls}`;
      node.textContent = text;
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
    }

    async function sendMessage(text) {
      const message = text.trim();
      if (!message) return;
      addMessage(message, "user");
      input.value = "";
      const response = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          user_id: document.querySelector("#userId").value,
          session_id: document.querySelector("#sessionId").value,
          message,
        }),
      });
      const payload = await response.json();
      addMessage(payload.answer || "请求失败", "assistant");
      document.querySelector("#quality").textContent = payload.quality_score ?? "-";
      document.querySelector("#tokens").textContent = payload.token_used ?? "-";
      document.querySelector("#degraded").textContent = payload.degraded ? "是" : "否";
      document.querySelector("#degraded").className = payload.degraded ? "handoff" : "";
      document.querySelector("#handoff").textContent = payload.needs_human_transfer ? "是" : "否";
      document.querySelector("#handoff").className = payload.needs_human_transfer ? "handoff" : "";
      document.querySelector("#reason").textContent = payload.transfer_reason || payload.degrade_reason || "-";
      document.querySelector("#memories").textContent = JSON.stringify(payload.user_memories || [], null, 2);
      document.querySelector("#summary").textContent = payload.memory_summary || "";
      document.querySelector("#context").textContent = JSON.stringify(payload.order_context || {}, null, 2);
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      sendMessage(input.value);
    });
    document.querySelectorAll("[data-message]").forEach((button) => {
      button.addEventListener("click", () => sendMessage(button.dataset.message));
    });
    document.querySelector("#clearMemory").addEventListener("click", async () => {
      const userId = document.querySelector("#userId").value;
      const response = await fetch(`/users/${encodeURIComponent(userId)}/memories`, {method: "DELETE"});
      const payload = await response.json();
      document.querySelector("#memories").textContent = "[]";
      addMessage(`已清除 ${payload.deleted ?? 0} 条用户记忆`, "assistant");
    });
  </script>
</body>
</html>
"""
