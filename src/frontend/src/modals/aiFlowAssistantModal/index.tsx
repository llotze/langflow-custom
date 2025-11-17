// AI Flow Assistant Modal - Hopper
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import HopperLogo from "@/assets/graceful/graceful-robot-head.png";
import ForwardedIconComponent from "@/components/common/genericIconComponent";
import ShadTooltip from "@/components/common/shadTooltipComponent";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useCustomNavigate } from "@/customization/hooks/use-custom-navigate";
import { track } from "@/customization/utils/analytics";
import useAddFlow from "@/hooks/flows/use-add-flow";
import useAlertStore from "@/stores/alertStore";
import type { newFlowModalPropsType } from "../../types/components";
import BaseModal from "../baseModal";
import FilePreview from "../IOModal/components/chatView/fileComponent/components/file-preview";
import { HOPPER_SYSTEM_PROMPT, TEMPLATE_CATALOG } from "./templateCatalog";

interface ChatMessage {
  id: string;
  role: "assistant" | "user" | "system";
  content: string;
}

interface TemplateRecommendation {
  id: string;
  name: string;
  description: string;
}

const STORAGE_KEY = "hopper_conversation_history";
const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY || "";
const DEMO_MODE = !OPENAI_API_KEY; // Automatically enable demo mode if no API key

// Mock responses for demo mode
const getMockResponse = (userMessage: string): string => {
  const lowerMessage = userMessage.toLowerCase();
  
  if (lowerMessage.includes("not sure") || lowerMessage.includes("don't know")) {
    return "No problem! Let me help you discover what you can build. What kind of work do you do, or what's a problem you're trying to solve?";
  }
  
  if (lowerMessage.includes("chatbot") || lowerMessage.includes("chat") || lowerMessage.includes("conversation")) {
    return "Perfect! A Memory Chatbot would be ideal for your needs. It remembers conversation history and provides contextual responses. Ready to create it?\n\nRECOMMEND: memory_chatbot";
  }
  
  if (lowerMessage.includes("document") || lowerMessage.includes("pdf") || lowerMessage.includes("qa") || lowerMessage.includes("question")) {
    return "Great choice! A Document Q&A system will let you upload documents and ask questions about them. The AI will search through your documents and provide accurate answers.\n\nRECOMMEND: document_qa";
  }
  
  if (lowerMessage.includes("rag") || lowerMessage.includes("search") || lowerMessage.includes("knowledge")) {
    return "Excellent! A Vector Store RAG system is perfect for semantic search across your knowledge base. It uses embeddings to find relevant information.\n\nRECOMMEND: vector_store_rag";
  }
  
  if (lowerMessage.includes("blog") || lowerMessage.includes("content") || lowerMessage.includes("writing") || lowerMessage.includes("article")) {
    return "Sounds great! A Blog Writer will help you generate high-quality articles and content. Perfect for content marketing and SEO.\n\nRECOMMEND: blog_writer";
  }
  
  if (lowerMessage.includes("agent") || lowerMessage.includes("automation") || lowerMessage.includes("task")) {
    return "Nice! An AI Agent can autonomously complete tasks and use tools. For multi-step workflows, I'd recommend the Sequential Tasks Agent.\n\nRECOMMEND: sequential_tasks_agent, complex_agent";
  }
  
  // Default response
  return "That sounds interesting! Could you tell me a bit more about what you want this system to do? For example, do you need it to answer questions, generate content, or automate tasks?";
};

export default function AIFlowAssistantModal({
  open,
  setOpen,
}: newFlowModalPropsType): JSX.Element {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [recommendedTemplates, setRecommendedTemplates] = useState<
    TemplateRecommendation[]
  >([]);
  const [files, setFiles] = useState<Array<{ file: File; id: string }>>([]);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const addFlow = useAddFlow();
  const navigate = useCustomNavigate();
  const { folderId } = useParams();
  const setErrorData = useAlertStore((state) => state.setErrorData);

  // Always start fresh when modal opens for better UX
  useEffect(() => {
    if (open) {
      initializeConversation();
    }
  }, [open]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, recommendedTemplates, isLoading]);

  const initializeConversation = () => {
    setMessages([
      {
        id: "1",
        role: "assistant",
        content:
          "Hi! I'm Hopper and I'm here to help build your perfect flow. What would you like to build today?",
      },
    ]);
    setShowSuggestions(true);
    setRecommendedTemplates([]);
  };

  const quickReplies = [
    "I'm not sure what I want to build",
    "Custom AI Chatbot",
    "RAG System",
    "Document Q&A",
    "AI Agent",
    "Data Pipeline",
  ];

  const handleQuickReply = async (reply: string) => {
    setShowSuggestions(false);
    
    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: reply,
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setIsLoading(true);

    try {
      // Call OpenAI
      const response = await callOpenAI(updatedMessages);

      // Parse for template recommendations
      const templateIds = parseRecommendations(response);
      const cleanResponse = response.replace(/RECOMMEND:.*$/gm, "").trim();

      // Add assistant response
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: cleanResponse,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Set recommended templates
      if (templateIds.length > 0) {
        const templates = TEMPLATE_CATALOG.filter((t) =>
          templateIds.includes(t.id)
        ).map((t) => ({
          id: t.id,
          name: t.name,
          description: t.description,
        }));
        setRecommendedTemplates(templates);
      }
    } catch (error) {
      console.error("Error calling OpenAI:", error);
      setErrorData({
        title: "Failed to get response",
        list: [
          error instanceof Error
            ? error.message
            : "Please check your OpenAI API key configuration",
        ],
      });

      // Remove the user message on error
      setMessages(updatedMessages.slice(0, -1));
    } finally {
      setIsLoading(false);
    }
  };

  const callOpenAI = async (conversationMessages: ChatMessage[]) => {
    // Demo mode: use mock responses
    if (DEMO_MODE) {
      const lastUserMessage = conversationMessages[conversationMessages.length - 1];
      return new Promise<string>((resolve) => {
        setTimeout(() => {
          resolve(getMockResponse(lastUserMessage.content));
        }, 800); // Simulate API delay
      });
    }

    // Real OpenAI call
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: HOPPER_SYSTEM_PROMPT },
          ...conversationMessages.map((msg) => ({
            role: msg.role,
            content: msg.content,
          })),
        ],
        temperature: 0.7,
        max_tokens: 300,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error?.message || "Failed to get response from OpenAI");
    }

    const data = await response.json();
    return data.choices[0].message.content;
  };

  const parseRecommendations = (content: string): string[] => {
    const recommendLine = content
      .split("\n")
      .find((line) => line.startsWith("RECOMMEND:"));
    if (!recommendLine) return [];

    const templateIds = recommendLine
      .replace("RECOMMEND:", "")
      .trim()
      .split(",")
      .map((id) => id.trim());

    return templateIds;
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    // Hide suggestions after first message
    setShowSuggestions(false);

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue,
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInputValue("");
    setFiles([]); // Clear uploaded files
    setIsLoading(true);

    try {
      // Call OpenAI
      const response = await callOpenAI(updatedMessages);

      // Parse for template recommendations
      const templateIds = parseRecommendations(response);
      const cleanResponse = response.replace(/RECOMMEND:.*$/gm, "").trim();

      // Add assistant response
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: cleanResponse,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Set recommended templates
      if (templateIds.length > 0) {
        const templates = TEMPLATE_CATALOG.filter((t) =>
          templateIds.includes(t.id)
        ).map((t) => ({
          id: t.id,
          name: t.name,
          description: t.description,
        }));
        setRecommendedTemplates(templates);
      }
    } catch (error) {
      console.error("Error calling OpenAI:", error);
      setErrorData({
        title: "Failed to get response",
        list: [
          error instanceof Error
            ? error.message
            : "Please check your OpenAI API key configuration",
        ],
      });

      // Remove the user message on error and restore input
      setMessages(updatedMessages.slice(0, -1));
      setInputValue(userMessage.content);
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetConversation = () => {
    initializeConversation();
    setFiles([]); // Clear uploaded files
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files;
    if (selectedFiles) {
      const newFiles = Array.from(selectedFiles)
        .filter(file => {
          const fileExtension = file.name.split(".").pop()?.toLowerCase();
          return fileExtension && ["jpg", "jpeg", "png", "gif", "webp"].includes(fileExtension);
        })
        .map(file => ({
          file,
          id: `${Date.now()}-${Math.random()}`,
        }));
      setFiles(prev => [...prev, ...newFiles]);
    }
    // Reset input
    if (event.target) {
      event.target.value = "";
    }
  };

  const handleDeleteFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleCreateFlow = (templateId: string) => {
    // TODO: Create flow from template
    // For now, just create a blank flow
    addFlow().then((id) => {
      navigate(`/flow/${id}${folderId ? `/folder/${folderId}` : ""}`);
      track("Flow Created via Hopper", { template: templateId });
      setOpen(false);
    });
  };

  return (
    <BaseModal 
      size="templates" 
      open={open} 
      setOpen={(isOpen) => {
        setOpen(isOpen);
        if (!isOpen) {
          // Reset conversation when closing for fresh start next time
          setTimeout(() => {
            initializeConversation();
            setFiles([]);
            setRecommendedTemplates([]);
          }, 300);
        }
      }}
    >
      <BaseModal.Header
        description={
          DEMO_MODE ? (
            <div className="flex items-center justify-center gap-1 px-2 py-1 rounded-md bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-200 text-xs">
              <ForwardedIconComponent name="Info" className="h-3 w-3" />
              <span>Demo Mode - Add OpenAI API key for full experience</span>
            </div>
          ) : undefined
        }
      >
        <div className="flex w-full items-center justify-between">
          <span className="text-2xl font-semibold">Hopper</span>
          {messages.length > 1 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleResetConversation}
              className="text-xs"
            >
              <ForwardedIconComponent name="RotateCcw" className="h-3 w-3 mr-1" />
              Reset
            </Button>
          )}
        </div>
      </BaseModal.Header>
      <BaseModal.Content overflowHidden>
        <div className="flex h-full flex-col">
          {/* Initial Welcome Screen - Only shown before first message */}
          {messages.length === 1 && !isLoading && (
            <div className="flex-1 flex flex-col items-center justify-center px-6 pb-4">
              <img 
                src={HopperLogo} 
                alt="Hopper" 
                className="h-32 w-auto mb-6"
              />
              <p className="text-center text-foreground text-base mb-8 max-w-md">
                {messages[0].content}
              </p>
              
              {/* Quick Reply Suggestions */}
              {showSuggestions && (
                <div className="flex flex-wrap gap-2 justify-center max-w-2xl">
                  {quickReplies.map((reply, index) => (
                    <Button
                      key={index}
                      onClick={() => handleQuickReply(reply)}
                      variant="outline"
                      size="sm"
                      className="rounded-full"
                    >
                      {reply}
                    </Button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Chat Messages Area - Shown after first interaction */}
          {(messages.length > 1 || isLoading) && (
            <div 
              ref={chatContainerRef}
              className="flex-1 overflow-y-auto px-8 py-6 space-y-6"
            >
              {messages.map((message) => (
                <div key={message.id}>
                  {message.role === "user" ? (
                    <div className="flex justify-end">
                      <div className="max-w-[66%] rounded-2xl px-4 py-3 bg-primary text-primary-foreground">
                        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                      </div>
                    </div>
                  ) : (
                    <div className="w-full">
                      <p className="text-sm text-foreground whitespace-pre-wrap leading-relaxed">
                        {message.content}
                      </p>
                    </div>
                  )}
                </div>
              ))}

              {/* Loading Indicator */}
              {isLoading && (
                <div className="w-full">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce delay-100"></div>
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce delay-200"></div>
                  </div>
                </div>
              )}

              {/* Template Recommendations */}
              {recommendedTemplates.length > 0 && !isLoading && (
                <div className="w-full space-y-2 pt-2">
                  <p className="text-xs font-semibold text-muted-foreground">
                    Recommended templates:
                  </p>
                  {recommendedTemplates.map((template) => (
                    <div
                      key={template.id}
                      className="border rounded-lg p-3 hover:border-primary transition-colors"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1">
                          <h4 className="font-semibold text-sm mb-1">{template.name}</h4>
                          <p className="text-xs text-muted-foreground">
                            {template.description}
                          </p>
                        </div>
                        <Button
                          size="sm"
                          onClick={() => handleCreateFlow(template.id)}
                          className="shrink-0"
                        >
                          <ForwardedIconComponent name="Plus" className="h-3 w-3 mr-1" />
                          Create
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Input Area - Always at bottom */}
          <div className="px-6 py-4">
            <div className="flex w-full flex-col-reverse">
              <div
                data-testid="input-wrapper"
                className="flex w-full flex-col rounded-md border cursor-text border-input p-4 hover:border-muted-foreground focus:border-[1.75px] has-[:focus]:border-primary"
                onClick={() => {
                  if (inputRef.current) {
                    inputRef.current.focus();
                    inputRef.current.setSelectionRange(
                      inputRef.current.value.length,
                      inputRef.current.value.length,
                    );
                  }
                }}
                onMouseDown={(e) => {
                  const target = e.target as HTMLElement;
                  if (target.closest("textarea")) {
                    return;
                  }
                  e.stopPropagation();
                  e.preventDefault();
                }}
              >
                <Textarea
                  data-testid="input-chat-playground"
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey && !isLoading) {
                      event.preventDefault();
                      handleSend();
                    }
                  }}
                  rows={1}
                  ref={inputRef}
                  disabled={isLoading}
                  style={{
                    resize: "none",
                    bottom: `${inputRef?.current?.scrollHeight}px`,
                    maxHeight: "150px",
                    overflow: `${
                      inputRef.current && inputRef.current.scrollHeight > 150
                        ? "auto"
                        : "hidden"
                    }`,
                  }}
                  value={inputValue}
                  onChange={(event) => {
                    setInputValue(event.target.value);
                  }}
                  className={`form-input block w-full border-0 custom-scroll focus:border-ring rounded-none shadow-none focus:ring-0 p-0 sm:text-sm !bg-transparent ${
                    files.length > 0 ? "!rounded-t-none border-t-0" : ""
                  }`}
                  placeholder="Send a message..."
                />

                <div className="flex w-full items-center gap-2 py-2 overflow-auto">
                  {files.map((fileObj, index) => (
                    <FilePreview
                      key={fileObj.id}
                      error={false}
                      file={fileObj.file}
                      loading={false}
                      onDelete={() => handleDeleteFile(index)}
                    />
                  ))}
                </div>
                
                <div className="flex w-full items-end justify-between">
                  <div className={isLoading ? "cursor-not-allowed" : ""}>
                    <ShadTooltip
                      styleClasses="z-50"
                      side="right"
                      content="Attach image (png, jpg, jpeg)"
                    >
                      <div>
                        <input
                          disabled={isLoading}
                          type="file"
                          ref={fileInputRef}
                          style={{ display: "none" }}
                          onChange={handleFileUpload}
                          accept="image/*"
                          multiple
                        />
                        <Button
                          disabled={isLoading}
                          className={`btn-playground-actions ${
                            isLoading
                              ? "cursor-not-allowed"
                              : "text-muted-foreground hover:text-primary"
                          }`}
                          onClick={(e: React.MouseEvent<HTMLButtonElement>) => {
                            e.stopPropagation();
                            fileInputRef.current?.click();
                          }}
                          unstyled
                        >
                          <ForwardedIconComponent className="h-[18px] w-[18px]" name="Image" />
                        </Button>
                      </div>
                    </ShadTooltip>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      className={`form-modal-send-button ${
                        isLoading 
                          ? "bg-muted hover:bg-secondary-hover dark:hover:bg-input text-foreground cursor-pointer" 
                          : inputValue.trim()
                            ? "bg-primary text-primary-foreground hover:bg-primary-hover hover:text-secondary"
                            : "bg-primary text-primary-foreground hover:bg-primary-hover hover:text-secondary"
                      }`}
                      onClick={(e: React.MouseEvent<HTMLButtonElement>) => {
                        e.stopPropagation();
                        handleSend();
                      }}
                      disabled={!inputValue.trim() && files.length === 0}
                      unstyled
                      data-testid="button-send"
                    >
                      <div className="flex h-fit w-fit items-center gap-2 text-sm font-medium">
                        {isLoading ? "Sending..." : "Send"}
                      </div>
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </BaseModal.Content>
    </BaseModal>
  );
}

