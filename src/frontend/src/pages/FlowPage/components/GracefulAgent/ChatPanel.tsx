import {
  Bot,
  X,
  Lightbulb,
  MousePointerClick,
  Send,
} from "lucide-react";
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import axios from "axios";

// ---- Add these interfaces ----
interface ChatMessage {
  id: string;
  flow_id: string;
  session_id: string;
  sender: string;
  message: string;
  timestamp: string;
}

interface ChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  flowId: string;
  sessionId: string;
}

export function ChatPanel({ isOpen, onClose, flowId, sessionId }: ChatPanelProps) {
  const [mode, setMode] = useState<"ideate" | "edit">("ideate");
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // ---- Fetch chat history on open/flow/session change ----
  useEffect(() => {
    if (isOpen && flowId && sessionId) {
      axios
        .get(`/api/v1/chat-history`, { params: { flow_id: flowId, session_id: sessionId } })
        .then((res) => setMessages(res.data));
    }
  }, [isOpen, flowId, sessionId]);

  // ---- Scroll to bottom on new message ----
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ---- Send message and get Claude response ----
  const handleSendMessage = async () => {
    if (!inputValue.trim() || loading) return;
    setLoading(true);

    // 1. Store user message in backend
    await axios.post("/api/v1/chat-history", {
      flow_id: flowId,
      session_id: sessionId,
      sender: "user",
      message: inputValue,
    });

    // 2. Optimistically update UI
    setMessages((prev) => [
      ...prev,
      {
        id: Math.random().toString(),
        flow_id: flowId,
        session_id: sessionId,
        sender: "user",
        message: inputValue,
        timestamp: new Date().toISOString(),
      },
    ]);
    setInputValue("");

    // 3. Call MCP server for Claude response
    try {
      const mcpResponse = await axios.post(
        process.env.REACT_APP_MCP_SERVER_URL || "http://localhost:5100/tool", // adjust as needed
        {
          name: "get_claude_response_with_history",
          arguments: {
            flow_id: flowId,
            session_id: sessionId,
          },
        }
      );
      // The MCP tool should:
      // - fetch chat history for flow/session
      // - call Claude with full history
      // - store Claude's response in chat history
      // - return the Claude message

      const assistantMsg = mcpResponse.data?.data?.message || mcpResponse.data?.message;
      if (assistantMsg) {
        setMessages((prev) => [
          ...prev,
          {
            id: Math.random().toString(),
            flow_id: flowId,
            session_id: sessionId,
            sender: "assistant",
            message: assistantMsg,
            timestamp: new Date().toISOString(),
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: Math.random().toString(),
          flow_id: flowId,
          session_id: sessionId,
          sender: "assistant",
          message: "Sorry, there was an error getting a response from Claude.",
          timestamp: new Date().toISOString(),
        },
      ]);
    }
    setLoading(false);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
  };

  return (
    <motion.div
      initial={false}
      animate={
        isOpen
          ? {
              width: 320,
              height: "calc(100vh - 140px)",
              opacity: 1,
              borderRadius: 12,
              top: 120,
              right: 16,
            }
          : {
              width: 36,
              height: 36,
              opacity: 0,
              borderRadius: 18,
              top: 52,
              right: 16,
            }
      }
      transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      className="absolute bg-white border border-gray-200 shadow-lg overflow-hidden flex flex-col pointer-events-auto"
      style={{
        transformOrigin: "top right",
        pointerEvents: isOpen ? "auto" : "none",
      }}
    >
      {isOpen && (
        <>
          {/* Header */}
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="bg-red-600 text-white p-1.5 rounded-full">
                  <Bot className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-sm">Graceful</h3>
                  <p className="text-xs text-gray-500">
                    What's up for today?
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="hover:bg-gray-100 h-7 w-7 p-0"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            {/* Mode Toggle */}
            <div className="flex gap-2">
              <Button
                variant={mode === "ideate" ? "default" : "outline"}
                size="sm"
                onClick={() => setMode("ideate")}
                className="flex-1 h-7 text-xs"
              >
                Ideate
              </Button>
              <Button
                variant={mode === "edit" ? "default" : "outline"}
                size="sm"
                onClick={() => setMode("edit")}
                className="flex-1 h-7 text-xs"
              >
                Edit
              </Button>
            </div>
          </div>

          {/* Content Area */}
          <div className="flex-1 overflow-y-auto p-4">
            {/* Chat history */}
            <div className="mb-4">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`mb-2 flex ${
                    msg.sender === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`rounded-lg px-3 py-2 text-xs max-w-[80%] ${
                      msg.sender === "user"
                        ? "bg-blue-100 text-blue-900"
                        : "bg-gray-100 text-gray-900"
                    }`}
                  >
                    {msg.message}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
            {mode === "ideate" ? (
              // Ideate Mode Empty State & Suggestions
              <div className="h-full flex flex-col items-center justify-center text-center px-6">
                <div className="mb-6">
                  <Lightbulb className="w-12 h-12 text-gray-400 mx-auto" />
                </div>
                <div className="space-y-4 w-full">
                  <div
                    className="flex items-start gap-2 cursor-pointer hover:opacity-70 transition-opacity"
                    onClick={() =>
                      handleSuggestionClick(
                        "Create an auto responder to email from students and notify me about a draft to give them"
                      )
                    }
                  >
                    <div className="text-center">
                      <p className="text-xs text-gray-900 mb-1">
                        Suggestion of the Day
                      </p>
                      <p className="text-xs text-gray-600">
                        Create an auto responder to email from students and notify me about a draft to give them
                      </p>
                    </div>
                  </div>
                  <div
                    className="flex items-start gap-2 cursor-pointer hover:opacity-70 transition-opacity px-[30px] py-[0px]"
                    onClick={() =>
                      handleSuggestionClick("View templates that are out there")
                    }
                  >
                    <div className="text-center">
                      <p className="text-xs text-gray-600">
                        View{" "}
                        <span style={{ textDecoration: "underline" }}>
                          Templates
                        </span>{" "}
                        that are out there
                      </p>
                    </div>
                  </div>
                  <div
                    className="flex items-start gap-2 cursor-pointer hover:opacity-70 transition-opacity px-[80px] py-[0px]"
                    onClick={() => handleSuggestionClick("Shuffle it up")}
                  >
                    <div className="text-center">
                      <p className="text-xs text-gray-600">Shuffle it up</p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              // Edit Mode Empty State
              <div className="h-full flex flex-col items-center justify-center text-center px-6">
                <div className="mb-4">
                  <MousePointerClick className="w-12 h-12 text-gray-400 mx-auto" />
                </div>
                <p className="text-sm text-gray-600">
                  Please select the components in the canvas to edit
                </p>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="p-3 border-t border-gray-200">
            <div className="flex gap-2 items-end">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder="Type your message..."
                className="flex-1 text-xs"
                disabled={loading}
              />
              <Button
                onClick={handleSendMessage}
                size="sm"
                className="h-8 w-8 p-0 flex-shrink-0"
                disabled={loading}
              >
                <Send className="w-3.5 h-3.5" />
              </Button>
            </div>
          </div>
        </>
      )}
    </motion.div>
  );
}