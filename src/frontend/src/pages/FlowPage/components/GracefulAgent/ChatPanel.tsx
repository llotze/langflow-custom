import { X, Send, Settings } from "lucide-react";
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { cn } from "@/utils/utils";
import { useDarkStore } from "@/stores/darkStore";
import { mcpApiHelpers, hasUserApiKey, setUserApiKey, clearUserApiKey } from "@/controllers/API/mcp-api";
import gracefulRobotHead from "@/assets/graceful/graceful-robot-head.png";

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
  headerOffset?: number;
  chatWidth?: number;
  setChatWidth?: (width: number) => void;
  minChatWidth?: number;
  maxChatWidth?: number;
  toolbarGap?: number;
  setIsResizing?: (val: boolean) => void;
}

export function ChatPanel({
  isOpen,
  onClose,
  flowId,
  sessionId,
  headerOffset = 0,
  chatWidth = 420,
  setChatWidth,
  minChatWidth = 320,
  maxChatWidth = 720,
  toolbarGap = 12,
  setIsResizing,
}: ChatPanelProps) {
  const isDark = useDarkStore((state) => state.dark);
  const [mode, setMode] = useState<"ideate" | "edit">("edit");
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const messageRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [pendingScrollId, setPendingScrollId] = useState<string | null>(null);
  const [isResizingLocal, setIsResizingLocal] = useState(false);
  const [showApiKeyInput, setShowApiKeyInput] = useState(!hasUserApiKey());
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [apiKeyError, setApiKeyError] = useState("");
  
  // Add ref to store EventSource instance
  const eventSourceRef = useRef<EventSource | null>(null);

  // ---- Fetch chat history on open/flow/session change ----
  useEffect(() => {
    if (isOpen && flowId && sessionId) {
      axios
        .get<ChatMessage[]>(`/api/v1/chat-history/`, { params: { flow_id: flowId, session_id: sessionId } })
        .then((res) => setMessages(res.data))
        .catch((err) => {
          console.error("Error fetching chat history:", err);
          setMessages([]);
        });
    }
  }, [isOpen, flowId, sessionId]);

  // ---- SSE Connection Management ----
  useEffect(() => {
    // Only connect if panel is open and we have a flowId
    if (!isOpen || !flowId) {
      return;
    }

    // Close existing connection if any
    if (eventSourceRef.current) {
      console.log('🧹 Closing previous SSE connection');
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    // Create new SSE connection
    const mcpServerUrl = process.env.VITE_MCP_SERVER_URL;
    const sseUrl = `${mcpServerUrl}/mcp/api/flow-updates/${flowId}`;
    
    console.log(`🔌 Connecting to SSE for flow ${flowId}...`);
    
    const eventSource = new EventSource(sseUrl);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      console.log('✅ SSE connected successfully');
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('📨 SSE update received:', data);
        
        // Handle flow updates here if needed
        if (data.type === 'flow_updated') {
          // You could trigger a flow refresh or show a notification
          console.log('Flow was updated via SSE');
        }
      } catch (err) {
        console.error('Error parsing SSE message:', err);
      }
    };

    eventSource.onerror = (error) => {
      console.error('❌ SSE error:', error);
      
      // EventSource automatically reconnects, so we only log errors
      // Don't close the connection unless it's permanently closed
      if (eventSource.readyState === EventSource.CLOSED) {
        console.log('SSE connection closed permanently');
        eventSourceRef.current = null;
      }
    };

    // Cleanup function: Close connection when component unmounts or flowId changes
    return () => {
      console.log('🧹 Cleaning up SSE connection');
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [isOpen, flowId]); // Only re-run when isOpen or flowId changes

  // ---- Send message and get response ----
  const handleSendMessage = async () => {
    // Check if API key is set
    if (!hasUserApiKey()) {
      setShowApiKeyInput(true);
      setMessages(prev => [...prev, {
        id: Math.random().toString(),
        flow_id: flowId,
        session_id: sessionId,
        sender: "assistant",
        message: "🔑 Please set your Langflow API key first to use Hopper.",
        timestamp: new Date().toISOString()
      }]);
      return;
    }

    if (!inputValue.trim() || loading) return;
    
    // Validate flowId before sending
    if (!flowId || flowId.trim() === "") {
      console.error("Error: flowId is required");
      setMessages((prev) => [
        ...prev,
        {
          id: Math.random().toString(),
          flow_id: flowId || "",
          session_id: sessionId,
          sender: "assistant",
          message: "Error: No flow ID available. Please ensure you're working with a valid flow.",
          timestamp: new Date().toISOString(),
        },
      ]);
      return;
    }

    setLoading(true);

    const messageId =
      (typeof crypto !== "undefined" && "randomUUID" in crypto && crypto.randomUUID()) ||
      `msg_${Date.now()}`;

    const userMessage = inputValue;
    
    // Optimistically update UI with user message
    setMessages((prev) => [
      ...prev,
      {
        id: messageId,
        flow_id: flowId,
        session_id: sessionId,
        sender: "user",
        message: userMessage,
        timestamp: new Date().toISOString(),
      },
    ]);
    setPendingScrollId(messageId);
    setInputValue("");

    // Process message using MCP assistant (no flow run)
    try {
      const assistantRes = await mcpApiHelpers.assistantChat(flowId, sessionId, userMessage);
      const assistantPayload = (assistantRes.data || {}) as { reply?: string; message?: string };
      const assistantText = assistantPayload.reply ?? assistantPayload.message ?? "I couldn't find a response from the assistant.";

      setMessages((prev) => [
        ...prev,
        {
          id: Math.random().toString(),
          flow_id: flowId,
          session_id: sessionId,
          sender: "assistant",
          message: assistantText,
          timestamp: new Date().toISOString(),
        },
      ]);
    } catch (err: any) {
      console.error("Error processing message with MCP:", err);
      setMessages((prev) => [
        ...prev,
        {
          id: Math.random().toString(),
          flow_id: flowId,
          session_id: sessionId,
          sender: "assistant",
          message: `Sorry, there was an error: ${err?.response?.data?.error || err?.message || "Unknown error"}`,
          timestamp: new Date().toISOString(),
        },
      ]);
    }
    setLoading(false);
  };

  // ---- Scroll the submitted message to the top of the view ----
  useEffect(() => {
    if (!pendingScrollId) return;
    const container = scrollContainerRef.current;
    const target = messageRefs.current[pendingScrollId];
    if (container && target) {
      const offset = 12;
      const top = target.offsetTop - container.offsetTop - offset;
      container.scrollTo({ top: Math.max(top, 0), behavior: "smooth" });
    }
    setPendingScrollId(null);
  }, [pendingScrollId]);

  const panelHeight = `calc(100vh - ${headerOffset}px)`;
  const closedTop = headerOffset + 16;
  const dragState = useRef<{ startX: number; startWidth: number } | null>(null);

  const handleSaveApiKey = () => {
    if (!apiKeyInput || apiKeyInput.trim().length === 0) {
      setApiKeyError("Please enter a valid API key");
      return;
    }
    
    setUserApiKey(apiKeyInput.trim());
    setShowApiKeyInput(false);
    setApiKeyError("");
    
    setMessages(prev => [...prev, {
      id: Math.random().toString(),
      flow_id: flowId,
      session_id: sessionId,
      sender: "assistant",
      message: "✅ API key saved! You can now use Hopper to build and modify flows.",
      timestamp: new Date().toISOString()
    }]);
  };

  const handleClearApiKey = () => {
    clearUserApiKey();
    setShowApiKeyInput(true);
    setApiKeyInput("");
    
    setMessages(prev => [...prev, {
      id: Math.random().toString(),
      flow_id: flowId,
      session_id: sessionId,
      sender: "assistant",
      message: "🔑 API key cleared. Please enter a new one to continue using Hopper.",
      timestamp: new Date().toISOString()
    }]);
  };

  return (
    <motion.div
      initial={false}
      animate={
        isOpen
          ? {
              width: chatWidth,
              height: panelHeight,
              opacity: 1,
              borderRadius: 12,
              top: headerOffset,
              right: 0,
              x: 0,
            }
          : {
              width: 36,
              height: 36,
              opacity: 0,
              borderRadius: 18,
              top: closedTop,
              right: 16,
              x: chatWidth + toolbarGap + 16,
            }
      }
      transition={{
        duration: isResizingLocal ? 0 : 0.5,
        ease: [0.4, 0, 0.2, 1],
      }}
      // Invert chat theme when dark mode is on so Hopper matches the app shell
      className={cn(
        "fixed border shadow-lg overflow-hidden flex flex-col pointer-events-auto transition-colors",
        isDark
          ? "bg-zinc-900 border-zinc-800 text-zinc-100"
          : "bg-white border-gray-200 text-gray-900",
      )}
      style={{
        transformOrigin: "top right",
        maxWidth: "100vw",
        height: isOpen ? panelHeight : undefined,
        width: isOpen ? chatWidth : undefined,
        pointerEvents: isOpen ? "auto" : "none",
        zIndex: 30,
      }}
    >
      {isOpen && setChatWidth && (
        <div
          className="absolute left-0 top-0 h-full w-2 cursor-col-resize z-10"
          style={{ transform: "translateX(-1px)" }}
          onPointerDown={(e) => {
            e.preventDefault();
            dragState.current = { startX: e.clientX, startWidth: chatWidth };
            setIsResizing?.(true);
            setIsResizingLocal(true);
            const handleMove = (ev: PointerEvent) => {
              if (!dragState.current) return;
              const delta = dragState.current.startX - ev.clientX;
              const next = Math.min(
                Math.max(dragState.current.startWidth + delta, minChatWidth),
                maxChatWidth,
              );
              setChatWidth(next);
            };
            const handleUp = () => {
              dragState.current = null;
              setIsResizing?.(false);
              setIsResizingLocal(false);
              window.removeEventListener("pointermove", handleMove);
              window.removeEventListener("pointerup", handleUp);
              window.removeEventListener("pointercancel", handleUp);
            };
            window.addEventListener("pointermove", handleMove);
            window.addEventListener("pointerup", handleUp);
            window.addEventListener("pointercancel", handleUp);
          }}
          aria-label="Resize chat width"
          title="Resize chat width"
        />
      )}

      {isOpen && (
        <>
          {/* Header */}
          <div
            className={cn(
              "p-4 border-b",
              isDark ? "border-zinc-800" : "border-gray-200",
            )}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <img
                  src={gracefulRobotHead}
                  alt="Hopper"
                  className="w-12 h-12 object-contain"
                />
                <div>
                  <h3 className="text-sm font-medium">Hopper</h3>
                </div>
              </div>
              
              <div className="flex items-center gap-1">
                {/* Settings Button */}
                {!showApiKeyInput && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowApiKeyInput(true)}
                    className="hover:bg-gray-100 h-7 w-7 p-0"
                    title="API Key Settings"
                  >
                    <Settings className="w-4 h-4" />
                  </Button>
                )}
                
                {/* Close Button */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onClose}
                  className="hover:bg-gray-100 h-7 w-7 p-0"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
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

            {/* API Key Setup UI */}
            {showApiKeyInput && (
              <div
                className={cn(
                  "mt-3 p-3 border rounded-lg",
                  isDark ? "bg-amber-900/20 border-amber-800 text-amber-50" : "bg-yellow-50 border-yellow-200",
                )}
              >
                <div className="flex items-start gap-2">
                  <div className="flex-shrink-0 text-xl">🔑</div>
                  <div className="flex-1">
                    <div className="flex items-start justify-between mb-1">
                      <h4
                        className={cn(
                          "text-xs font-semibold",
                          isDark ? "text-zinc-50" : "text-gray-900",
                        )}
                      >
                        Langflow API Key {hasUserApiKey() ? "(Optional)" : "Required"}
                      </h4>
                  <button
                    onClick={() => setShowApiKeyInput(false)}
                    className={cn(
                      "-mt-1 -mr-1",
                      isDark
                        ? "text-zinc-400 hover:text-zinc-200"
                        : "text-gray-400 hover:text-gray-600",
                    )}
                    title="Close settings"
                  >
                    <X className="w-4 h-4" />
                  </button>
                    </div>
                    
                    <p
                      className={cn(
                        "text-xs mb-2",
                        isDark ? "text-zinc-300" : "text-gray-600",
                      )}
                    >
                      {hasUserApiKey() 
                        ? "Update your API key or clear it below"
                        : "Create an API key in Settings → API Keys"
                      }
                    </p>
                    
                    <div className="flex gap-2 mb-2">
                      <input
                        type="password"
                        value={apiKeyInput}
                        onChange={(e) => setApiKeyInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            handleSaveApiKey();
                          }
                          if (e.key === 'Escape') {
                            setShowApiKeyInput(false);
                          }
                        }}
                        placeholder={hasUserApiKey() ? "Enter new API key..." : "sk-lf-..."}
                        className={cn(
                          "flex-1 px-2 py-1.5 border rounded text-xs focus:outline-none focus:ring-1 focus:ring-blue-500",
                          isDark
                            ? "bg-zinc-800 border-zinc-700 text-zinc-100 placeholder:text-zinc-500"
                            : "border-gray-300",
                        )}
                      />
                      <button
                        onClick={handleSaveApiKey}
                        className="px-3 py-1.5 bg-blue-600 text-white rounded text-xs font-medium hover:bg-blue-700 transition-colors"
                      >
                        Save
                      </button>
                    </div>
                    
                    {apiKeyError && (
                      <p className="text-xs text-red-500 mb-2">{apiKeyError}</p>
                    )}
                    
                    {hasUserApiKey() && (
                      <button
                        onClick={handleClearApiKey}
                        className="text-xs text-red-500 hover:text-red-400 underline"
                      >
                        Clear existing key
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Content Area */}
          <div
            className="flex-1 overflow-y-auto overflow-x-hidden p-4"
            ref={scrollContainerRef}
          >
            {/* Chat history */}
            <div className="space-y-2">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  ref={(el) => {
                    messageRefs.current[msg.id] = el;
                  }}
                  className={`flex ${
                    msg.sender === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={cn(
                      "rounded-lg px-3 py-2 text-xs max-w-[80%]",
                      msg.sender === "user"
                        ? isDark
                          ? "bg-blue-900/60 text-blue-50"
                          : "bg-blue-100 text-blue-900"
                        : isDark
                          ? "bg-zinc-800 text-zinc-100"
                          : "bg-gray-100 text-gray-900",
                    )}
                    style={{ wordBreak: "break-word", whiteSpace: "pre-wrap" }}
                  >
                    {msg.message}
                  </div>
                </div>
              ))}
              
              {/* Loading bubble */}
              {loading && (
                <div className="flex justify-start">
                  <div
                    className={cn(
                      "rounded-lg px-3 py-2 text-xs",
                      isDark ? "bg-zinc-800 text-zinc-100" : "bg-gray-100 text-gray-900",
                    )}
                  >
                    <div className="flex items-center gap-1">
                      <div
                        className={cn(
                          "w-2 h-2 rounded-full animate-bounce",
                          isDark ? "bg-zinc-400" : "bg-gray-400",
                        )}
                        style={{ animationDelay: '0ms' }}
                      ></div>
                      <div
                        className={cn(
                          "w-2 h-2 rounded-full animate-bounce",
                          isDark ? "bg-zinc-400" : "bg-gray-400",
                        )}
                        style={{ animationDelay: '150ms' }}
                      ></div>
                      <div
                        className={cn(
                          "w-2 h-2 rounded-full animate-bounce",
                          isDark ? "bg-zinc-400" : "bg-gray-400",
                        )}
                        style={{ animationDelay: '300ms' }}
                      ></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Input Area */}
          <div
            className={cn(
              "p-3 border-t",
              isDark ? "border-zinc-800" : "border-gray-200",
            )}
          >
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
                className={cn(
                  "flex-1 text-xs",
                  isDark && "bg-zinc-800 border-zinc-700 text-zinc-100 placeholder:text-zinc-500",
                )}
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
