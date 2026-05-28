import { useState, useEffect } from "react";
import "./styles.css";
import { marked } from "marked";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [useLora, setUseLora] = useState(true);

  // Load chat history on page load
  useEffect(() => {
    fetch(`${API_BASE}/history`)
      .then(res => res.json())
      .then(data => setMessages(data))
      .catch(err => console.log("History fetch error:", err));
  }, []);

  // Send message
  const sendMessage = async (msgText) => {
  const msg = msgText || input;

  if (!msg.trim() || isLoading) return;

  setInput("");
  setIsLoading(true);

  // show user message instantly
  setMessages(prev => [
    ...prev,
    { role: "user", content: msg }
  ]);

  // empty bot message for streaming
  let botIndex;

  setMessages(prev => {
    botIndex = prev.length + 1;

    return [
      ...prev,
      { role: "bot", content: "" }
    ];
  });

  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: msg,
        history: history,
        use_lora: useLora
      })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    let fullText = "";

    while (true) {
      const { value, done } = await reader.read();

      if (done) break;

      const chunk = decoder.decode(value);

      const lines = chunk.split("\n");

      for (let line of lines) {
        if (line.startsWith("data: ")) {

          const token = line.replace("data: ", "");

          if (token === "[DONE]") break;

          fullText += token;

          setMessages(prev => {
            const updated = [...prev];

            updated[updated.length - 1] = {
              role: "bot",
              content: fullText
            };

            return updated;
          });
        }
      }
    }

    setHistory(prev => [
      ...prev,
      { role: "user", content: msg },
      { role: "assistant", content: fullText }
    ]);

  } catch (err) {

    setMessages(prev => [
      ...prev,
      {
        role: "bot",
        content: "Error: " + err.message
      }
    ]);

  }

  setIsLoading(false);
};

  const clearChat = () => {
    setMessages([]);
    setHistory([]);
  };

  const topics = [
    "Variables & data types",
    "Lists & tuples",
    "Dictionaries",
    "Functions & scope",
    "OOP & classes",
    "List comprehensions",
    "File handling",
    "Error handling",
    "Decorators",
    "Generators",
    "Async / await",
    "pip & packages",
  ];

  return (
    <div>
      {/* HEADER */}
      <header>
        <div className="logo">
          <div className="logo-icon">&gt;_</div>
          PythonBot
        </div>

        <div className="header-right">
          <div className="mode-toggle">
            <span>LoRA</span>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={useLora}
                onChange={() => setUseLora(!useLora)}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <button className="btn-clear" onClick={clearChat}>
            Clear chat
          </button>
        </div>
      </header>

      {/* MAIN */}
      <main>
        {/* SIDEBAR */}
        <aside className="sidebar">
          <div className="sidebar-label">Quick Topics</div>

          {topics.map((t, i) => (
            <button
              key={i}
              className="topic-btn"
              onClick={() => sendMessage(`Explain ${t} in Python with examples.`)}
            >
              <span className="dot"></span>
              {t}
            </button>
          ))}
        </aside>

        {/* CHAT */}
        <div className="chat-container">
          <div className="messages">

            {messages.length === 0 && (
              <div className="welcome">
                <div className="welcome-icon">&gt;_</div>
                <h1>Hey, I'm PythonBot 🐍</h1>
                <p>Your personal Python tutor.</p>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`msg ${msg.role}`}>
                <div className="avatar">
                  {msg.role === "user" ? "U" : "PB"}
                </div>

                <div className="bubble">
                  {msg.role === "bot" ? (
                    <div
                      dangerouslySetInnerHTML={{
                        __html: marked.parse(msg.content),
                      }}
                    />
                  ) : (
                    msg.content
                  )}
                </div>
              </div>
            ))}

          </div>

          {/* INPUT */}
          <div className="input-area">
            <div className="input-wrap">
              <textarea
                className="input-box"
                placeholder="Ask me anything about Python…"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
              />

              <button
                className="send-btn"
                onClick={() => sendMessage()}
                disabled={isLoading}
              >
                <svg viewBox="0 0 24 24">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                </svg>
              </button>
            </div>

            <div className="input-meta">
              <span>Model ready</span>
              <span>Enter to send</span>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;