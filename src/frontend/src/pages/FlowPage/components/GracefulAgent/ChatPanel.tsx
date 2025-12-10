import { X, Lightbulb, MousePointerClick, Send } from "lucide-react";
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { mcpApiHelpers } from "@/controllers/API/mcp-api";
import gracefulRobotHead from "@/assets/graceful/graceful-robot-head.png";

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
  const [mode, setMode] = useState<"ideate" | "edit">("ideate");
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const messageRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [pendingScrollId, setPendingScrollId] = useState<string | null>(null);
  const [isResizingLocal, setIsResizingLocal] = useState(false);

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

  // ---- Send message and get response ----
  const handleSendMessage = async () => {
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
      const offset = 12; // small gap from the header
      const top = target.offsetTop - container.offsetTop - offset;
      container.scrollTo({ top: Math.max(top, 0), behavior: "smooth" });
    }
    setPendingScrollId(null);
  }, [pendingScrollId]);

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
  };

  const panelHeight = `calc(100vh - ${headerOffset}px)`;
  const closedTop = headerOffset + 16;
  const dragState = useRef<{ startX: number; startWidth: number } | null>(null);

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
      className="fixed bg-white border border-gray-200 shadow-lg overflow-hidden flex flex-col pointer-events-auto"
      style={{
        transformOrigin: "top right",
        maxWidth: "100vw",
        height: isOpen ? panelHeight : undefined,
        width: isOpen ? chatWidth : undefined,
        pointerEvents: isOpen ? "auto" : "none",
        zIndex: 60,
      }}
    >
      {/* Resize handle on the left edge */}
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
          <div className="p-4 border-b border-gray-200">
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
          <div
            className="flex-1 overflow-y-auto overflow-x-hidden p-4"
            ref={scrollContainerRef}
          >
            {/* Chat history */}
            <div className="mb-4">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  ref={(el) => {
                    messageRefs.current[msg.id] = el;
                  }}
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
                  style={{ wordBreak: "break-word", whiteSpace: "pre-wrap" }}
                  >
                    {msg.message}
                  </div>
                </div>
              ))}
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