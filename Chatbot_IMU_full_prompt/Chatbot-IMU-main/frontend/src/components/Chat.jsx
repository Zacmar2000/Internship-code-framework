import { v4 as uuidv4 } from "uuid";
import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from 'remark-breaks';
import "./Chat.css";

const Chat = ({ chats, setChats, selectedChatId, setSelectedChatId, isSending, setIsSending }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const chatEndRef = useRef(null);
  const maxInputHeight = 150;

  // Flag per evitare loop infiniti
  const isLoadingChat = useRef(false);
  const sessionId = useRef(null);
  const abortController = useRef(null); // Controller per cancellare richieste

  // Inizializza la chat quando selectedChatId cambia
  useEffect(() => {
    if (selectedChatId && !isLoadingChat.current) {
      isLoadingChat.current = true;
      // Carica una chat precedente
      const chat = chats.find(c => c.id === selectedChatId);
      if (chat) {
        // Solo se la chat ha messaggi, caricali (evita di sovrascrivere chat appena create)
        if (chat.messages.length > 0) {
          setMessages(chat.messages);
        }
        sessionId.current = chat.sessionId;
      }
      // Reset del flag dopo un breve delay per permettere al componente di stabilizzarsi
      setTimeout(() => { isLoadingChat.current = false; }, 100);
    } else if (!selectedChatId) {
      // Nuova chat - resetta tutto
      sessionId.current = uuidv4();
      setMessages([]);
      isLoadingChat.current = false;
    }
  }, [selectedChatId, chats]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const cancelMessage = () => {
    if (abortController.current) {
      abortController.current.abort();
      setIsSending(false);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Salva i messaggi nella cache quando cambiano (solo per chat esistenti)
  useEffect(() => {
    if (messages.length > 0 && selectedChatId) {
      // Aggiorna solo le chat esistenti (non creare nuove chat qui)
      setChats(prev =>
        prev.map(c => c.id === selectedChatId ? { ...c, messages } : c)
      );
    }
  }, [messages, selectedChatId]);

  const sendMessage = async () => {
    const message = input.trim();
    if (!message || isSending) return; // blocca se vuoto o già in invio

    // Se è una nuova chat (selectedChatId è null), crea la chat prima di inviare il messaggio
    let currentChatId = selectedChatId;
    if (!selectedChatId) {
      const newChat = {
        id: uuidv4(),
        sessionId: sessionId.current,
        title: message.substring(0, 50),
        messages: [],
        timestamp: Date.now()
      };
      setChats(prev => [newChat, ...prev]);
      currentChatId = newChat.id;
      setSelectedChatId(newChat.id);
    }

    setIsSending(true); // blocca subito
    setMessages(prev => [...prev, { sender: "user", text: message }]);
    setInput("");

    // Crea un nuovo AbortController per questa richiesta
    abortController.current = new AbortController();

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session: sessionId.current }),
        signal: abortController.current.signal // Passa il signal al fetch
      });

      if (!response.ok) throw new Error("Errore nella richiesta al server");

      const assistantMessage = await response.text();

      // console.log(JSON.stringify(assistantMessage))

      // 1. Rimuovi le virgolette iniziali e finali
      let formattedMessage = assistantMessage;

      // Se la stringa inizia e finisce con virgolette doppie, rimuovile
      if (formattedMessage.startsWith('"') && formattedMessage.endsWith('"')) {
        formattedMessage = formattedMessage.slice(1, -1);
      }

      // 2. Trasforma i \\n in veri newline
      formattedMessage = formattedMessage.replace(/\\n/g, '\n');

      // 3. Rimuovi eventuali spazi o newline all’inizio e alla fine
      formattedMessage = formattedMessage.trim();
      
      setMessages(prev => [...prev, { sender: "assistant", text: formattedMessage }]);

    } catch (err) {
      // Se è un AbortError (utente ha cliccato Annulla), non mostrare errore
      if (err.name === 'AbortError') {
        setMessages(prev => [...prev, { sender: "assistant", text: "Generazione annullata", abortSymbol: true }]);
      } else {
        setMessages(prev => [...prev, { sender: "assistant", text: "Errore: " + err.message }]);
      }
    } finally {
      setIsSending(false); // sblocca
    }
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
    e.target.style.height = "auto";         // reset altezza
    const newHeight = e.target.scrollHeight;
    e.target.style.height = `${newHeight}px`; // adatta a contenuto
    e.target.style.overflowY = newHeight > maxInputHeight ? "auto" : "hidden";
  };

  return (
    <div className="chat-container">
      <div className="chat-log">
        {messages.length === 0 ? (
          <div className="message assistant welcome-message">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
              {"Ciao! 👋\n\nSono il tuo assistente IMU. Sono qui per aiutarti con domande e informazioni. Come posso assisterti oggi?"}
            </ReactMarkdown>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              {msg.sender === "assistant" ? (
                msg.abortSymbol ? (
                  <div className="assistant-abort-message">
                    {msg.text}
                  </div>
                ) : (
                  <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                    {msg.text}
                  </ReactMarkdown>
                )
              ) : (
                msg.text
              )}
            </div>
          ))
        )}
        <div ref={chatEndRef}></div>
      </div>

      <div className="chat-input-wrapper">
        <textarea
          className="chat-input"
          value={input}
          onChange={handleInputChange}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && input.trim() && !isSending) {
              e.preventDefault();
              sendMessage();
            }
          }}
          placeholder="Scrivi un messaggio..."
          rows={1}  // altezza iniziale
        />
        <button
          className={`chat-button ${isSending ? 'sending' : ''}`}
          onClick={isSending ? cancelMessage : sendMessage}
          disabled={!isSending && !input.trim()}
        >
          {isSending ? <span className="stop-icon" /> : "Invia"}
        </button>
      </div>
    </div>
  );
};

export default Chat;