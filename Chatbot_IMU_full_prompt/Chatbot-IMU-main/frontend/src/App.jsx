import React, { useEffect, useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useLocation,
} from "react-router-dom";
import Chat from "./components/Chat";
import PromptEditor from "./components/PromptEditor";
import "./App.css";

const labels = {
  main_prompt: "Prompt principale",
  prompt_nazionale: "Normativa nazionale",
  prompt_comunale: "Regolamento comunale",
  prompt_aliquote: "Aliquote",
  prompt_file_rimanenti: "Altri documenti",
  prompt_vocab: "Vocabolario",
};

function App() {
  return (
    <Router>
      <AppLayout />
    </Router>
  );
}

function AppLayout() {
  const [prompts, setPrompts] = useState({});
  const [selectedPrompt, setSelectedPrompt] = useState("");
  const [chats, setChats] = useState([]);
  const [selectedChatId, setSelectedChatId] = useState(null);
  const [isSending, setIsSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const location = useLocation();

  // Carica le chat salvate dal localStorage
  useEffect(() => {
    const savedChats = localStorage.getItem("chats");
    if (savedChats) {
      try {
        setChats(JSON.parse(savedChats));
      } catch (error) {
        console.error("Errore caricamento chat salvate:", error);
      }
    }
  }, []);

  // Salva le chat nel localStorage quando cambiano
  useEffect(() => {
    localStorage.setItem("chats", JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    fetch("http://localhost:8000/prompts")
      .then((res) => res.json())
      .then((data) => {
        setPrompts(data);
        setSelectedPrompt(Object.keys(data)[0] || "");
      })
      .catch((error) => {
        console.error("Errore caricamento prompt:", error);
      });
  }, []);

  const promptKeys = Object.keys(prompts);
  const isPromptPage = location.pathname === "/prompt";

  console.log("Chats:", chats);
  console.log("Selected Chat ID:", selectedChatId);

  return (
    <div className="app-layout">
      <button
        className={`sidebar-toggle sidebar-toggle-fixed ${sidebarOpen ? 'hidden' : 'visible'}`}
        onClick={() => setSidebarOpen(!sidebarOpen)}
        title={sidebarOpen ? "Nascondi sidebar" : "Mostra sidebar"}
      >
        ☰
      </button>
      <aside className={`app-sidebar ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
        <div className="sidebar-header">
          <h2>Menu</h2>
          {sidebarOpen && (
            <button
              className="sidebar-toggle"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title={sidebarOpen ? "Nascondi sidebar" : "Mostra sidebar"}
            >
              ☰
            </button>
          )}
        </div>

        <nav className="sidebar-links">
          <Link
            className={`sidebar-link ${location.pathname === "/" ? "active" : ""}`}
            to="/"
            onClick={(e) => isSending && e.preventDefault()}
            style={{ pointerEvents: isSending ? "none" : "auto", opacity: isSending ? 0.5 : 1 }}
          >
            Chat
          </Link>
          <Link
            className={`sidebar-link ${location.pathname === "/prompt" ? "active" : ""}`}
            to="/prompt"
            onClick={(e) => isSending && e.preventDefault()}
            style={{ pointerEvents: isSending ? "none" : "auto", opacity: isSending ? 0.5 : 1 }}
          >
            Prompt
          </Link>
        </nav>

        <hr className="sidebar-divider" />

        {isPromptPage && promptKeys.length > 0 && (
          <div className="prompt-sidebar">
            <h3>Seleziona prompt</h3>
            <div className="prompt-list">
              {promptKeys.map((key) => (
                <button
                  key={key}
                  type="button"
                  className={`prompt-item ${selectedPrompt === key ? "active" : ""}`}
                  onClick={() => setSelectedPrompt(key)}
                >
                  {labels[key] || key}
                </button>
              ))}
            </div>
          </div>
        )}

        {!isPromptPage && (
          <div className="chat-sidebar">
            <button
              className="new-chat-button"
              onClick={() => {
                setSelectedChatId(null);
              }}
              disabled={isSending}
              style={{ pointerEvents: isSending ? "none" : "auto", opacity: isSending ? 0.5 : 1 }}
            >
            Nuova chat
            </button>
            {chats.length > 0 && (
              <div className="chat-list">
                <h3>Conversazioni</h3>
                <div className="chat-items">
                  {chats.map((chat) => (
                    <button
                      key={chat.id}
                      type="button"
                      className={`chat-item ${selectedChatId === chat.id ? "active" : ""}`}
                      onClick={() => setSelectedChatId(chat.id)}
                      title={`${chat.title}\n${new Date(chat.timestamp).toLocaleDateString()}`}
                      disabled={isSending}
                      style={{ pointerEvents: isSending ? "none" : "auto", opacity: isSending ? 0.7 : 1 }}
                    >
                      <span className="chat-title">{chat.title}</span>
                      <span className="chat-date">
                        {new Date(chat.timestamp).toLocaleDateString()}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </aside>

      <main className="app-main">
        <div className="app-header">
          <h1>Chatbot IMU</h1>
        </div>

        <Routes>
          <Route 
            path="/" 
            element={
              <Chat 
                chats={chats}
                setChats={setChats}
                selectedChatId={selectedChatId}
                setSelectedChatId={setSelectedChatId}
                isSending={isSending}
                setIsSending={setIsSending}
              />
            } 
          />
          <Route
            path="/prompt"
            element={
              <PromptEditor
                prompts={prompts}
                selected={selectedPrompt}
                onSelectedChange={setSelectedPrompt}
                onPromptsUpdate={setPrompts}
              />
            }
          />
        </Routes>
      </main>
    </div>
  );
}

export default App;
