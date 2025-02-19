// frontend (React - App.js)

import React, { useState } from "react";
import axios from "axios";
import "./App.css";

//  Important: The URL now points to *your* backend, not directly to RunPod.
const API_URL = "/api/generate";  //  Or "http://localhost:3001/api/generate" during local development

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { role: "user", content: input };
    setMessages([...messages, userMessage]);
    setInput("");
    setLoading(true);

    try {
      console.log("Sending request...");
      const response = await axios.post(
        API_URL,
        {
          prompt: input,  // Send only the necessary data to your backend
          max_length: 100,   // You *can* still send these if you want to
          num_return_sequences: 1, // control them from the frontend.
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      console.log("API Response:", response.data);

      if (response.data && response.data.generated) { //  Match the response structure from your backend
        setMessages(prevMessages => [
          ...prevMessages,
          { role: "assistant", content: response.data.generated }
        ]);
      } else {
        console.error("Unexpected response format:", response.data);
        //  Consider adding a user-friendly error message here.
        setMessages(prevMessages => [
          ...prevMessages,
          { role: "assistant", content: "Sorry, I couldn't generate a response." }
        ]);
      }

    } catch (error) {
      console.error("Error fetching response:", error);
      //  Provide user feedback on errors.
      let errorMessage = "An error occurred while contacting the server.";
      if (error.response) {
        console.error("Response Data:", error.response.data);
        console.error("Response Status:", error.response.status);
        console.error("Response Headers:", error.response.headers);
        errorMessage = error.response.data.error || errorMessage; // Use backend error message if available
      }
        setMessages(prevMessages => [
            ...prevMessages,
            { role: "assistant", content: errorMessage}
        ])

    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <h1 className="app-title">LyricGenAI</h1>
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
        {loading && <div className="message assistant">Typing...</div>}
      </div>
      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {  //  Add this to handle Enter key
            if (e.key === 'Enter' && !e.shiftKey) { //  Allow Shift+Enter for newlines
              e.preventDefault();  //  Prevent default Enter behavior (form submission)
              sendMessage();
            }
          }}
          placeholder="Type your message..."
        />
        <button onClick={sendMessage} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;