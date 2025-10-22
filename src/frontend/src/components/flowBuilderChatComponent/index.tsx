import React, { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Loader2, Send, Sparkles, X, Bot, User } from "lucide-react";
import useAlertStore from "@/stores/alertStore";
import useFlowStore from "@/stores/flowStore";
import useFlowsManagerStore from "@/stores/flowsManagerStore";
import { api } from "@/controllers/API/api";

interface Message {
  id: string;
  type: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  flowData?: any;
}

interface FlowBuilderChatProps {
  isOpen: boolean;
  onClose: () => void;
}

export const FlowBuilderChat: React.FC<FlowBuilderChatProps> = ({
  isOpen,
  onClose,
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      type: "assistant",
      content: "Hi! I'm your Flow Builder AI assistant. Describe the workflow you'd like to create, and I'll generate a complete Langflow for you. For example:\n\n• \"Create a chatbot that answers questions about uploaded documents\"\n• \"Build a workflow that summarizes text and sends it via email\"\n• \"Make an agent that searches the web and formats results\"",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
    const addFlow = useFlowsManagerStore((state) => state.setCurrentFlow);
  const setNodes = useFlowStore((state) => state.setNodes);
  const setEdges = useFlowStore((state) => state.setEdges);
  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (message: Omit<Message, "id" | "timestamp">) => {
    const newMessage: Message = {
      ...message,
      id: Date.now().toString(),
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newMessage]);
  };

  const buildFlow = async (userRequest: string) => {
    setIsLoading(true);
    
    try {
      // Add user message
      addMessage({
        type: "user",
        content: userRequest,
      });      // Call the Flow Builder API
      const response = await api.post("/api/v1/flow_builder/build", {
        query: userRequest,
        flow_name: `Generated Flow - ${new Date().toLocaleDateString()}`,
      });

      const { flow_json, explanation, components_used } = response.data;

      // Add assistant response with explanation
      let assistantMessage = `Great! I've created your flow with the following components:\n\n`;
      assistantMessage += `**Components Used:** ${components_used.join(", ")}\n\n`;
      assistantMessage += `**Explanation:** ${explanation}\n\n`;
      assistantMessage += `Would you like me to load this flow into your workspace?`;

      addMessage({
        type: "assistant",
        content: assistantMessage,
        flowData: flow_json,
      });

    } catch (error: any) {
      console.error("Error building flow:", error);
      addMessage({
        type: "assistant",
        content: `I encountered an error while building your flow: ${
          error.response?.data?.detail || error.message || "Unknown error"
        }. Please try rephrasing your request or contact support if the issue persists.`,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadFlow = (flowData: any) => {
    try {
      if (flowData && flowData.data) {        // Load the flow into the current workspace
        setNodes(flowData.data.nodes || []);
        setEdges(flowData.data.edges || []);
          setSuccessData({
          title: "Flow Loaded Successfully",
        });
        
        // Close the chat after loading
        onClose();
      }    } catch (error) {
      console.error("Error loading flow:", error);      setErrorData({
        title: "Error Loading Flow",
      });
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userRequest = inputValue.trim();
    setInputValue("");
    
    await buildFlow(userRequest);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl h-[600px] flex flex-col bg-background border rounded-lg shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-primary/10">
              <Sparkles className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h2 className="font-semibold text-lg">Flow Builder AI</h2>
              <p className="text-sm text-muted-foreground">
                Describe your workflow, get instant Langflow
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Messages */}
        <div className="flex-1 p-4 overflow-y-auto">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${
                  message.type === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`flex gap-3 max-w-[85%] ${
                    message.type === "user" ? "flex-row-reverse" : "flex-row"
                  }`}
                >
                  <div
                    className={`p-2 rounded-full ${
                      message.type === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    }`}
                  >
                    {message.type === "user" ? (
                      <User className="h-4 w-4" />
                    ) : (
                      <Bot className="h-4 w-4" />
                    )}
                  </div>
                  <div
                    className={`rounded-lg p-3 ${
                      message.type === "user"
                        ? "bg-primary text-primary-foreground ml-auto"
                        : "bg-muted"
                    }`}
                  >
                    <div className="whitespace-pre-wrap text-sm">
                      {message.content}
                    </div>
                    {message.flowData && (
                      <div className="mt-3 pt-3 border-t border-border/50">
                        <Button
                          size="sm"
                          onClick={() => handleLoadFlow(message.flowData)}
                          className="text-xs"
                        >
                          <Sparkles className="h-3 w-3 mr-1" />
                          Load Flow into Workspace
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="p-2 rounded-full bg-muted">
                  <Bot className="h-4 w-4" />
                </div>
                <div className="rounded-lg p-3 bg-muted">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm">Building your flow...</span>
                  </div>
                </div>
              </div>
            )}          </div>
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 border-t">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Describe the workflow you want to create..."
              disabled={isLoading}
              className="flex-1"
            />
            <Button type="submit" disabled={isLoading || !inputValue.trim()}>
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </form>
          <p className="text-xs text-muted-foreground mt-2">
            Press Enter to send, Shift+Enter for new line
          </p>        </div>
      </div>
    </div>
  );
};

export default FlowBuilderChat;
