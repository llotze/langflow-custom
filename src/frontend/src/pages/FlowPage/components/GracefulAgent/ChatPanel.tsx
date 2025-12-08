import {
Bot,
X,
Lightbulb,
Grid3x3,
Shuffle,
MousePointerClick,
Send,
} from "lucide-react";
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { motion } from "framer-motion";
import { useState } from "react";

interface ChatPanelProps {
isOpen: boolean;
onClose: () => void;
}

export function ChatPanel({ isOpen, onClose }: ChatPanelProps) {
const [mode, setMode] = useState<"ideate" | "edit">("ideate");
const [inputValue, setInputValue] = useState("");

const handleModeChange = (newMode: "ideate" | "edit") => {
setMode(newMode);
};

const handleSendMessage = () => {
if (!inputValue.trim()) return;

// Handle sending message
console.log("Sending:", inputValue);
setInputValue("");
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
            variant={
                mode === "ideate" ? "default" : "outline"
            }
            size="sm"
            onClick={() => handleModeChange("ideate")}
            className="flex-1 h-7 text-xs"
            >
            Ideate
            </Button>
            <Button
            variant={
                mode === "edit" ? "default" : "outline"
            }
            size="sm"
            onClick={() => handleModeChange("edit")}
            className="flex-1 h-7 text-xs"
            >
            Edit
            </Button>
        </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-4">
        {mode === "ideate" ? (
            // Ideate Mode Empty State
            <div className="h-full flex flex-col items-center justify-center text-center px-6">
            <div className="mb-6">
                <Lightbulb className="w-12 h-12 text-gray-400 mx-auto" />
            </div>

            {/* Suggestion Text Items - Centered */}
            <div className="space-y-4 w-full">
                {/* Suggestion of the Day */}
                <div
                className="flex items-start gap-2 cursor-pointer hover:opacity-70 transition-opacity"
                onClick={() =>
                    handleSuggestionClick(
                    "Create an auto responder to email from students and notify me about a draft to give them",
                    )
                }
                >
                <div className="text-center">
                    <p className="text-xs text-gray-900 mb-1">
                    Suggestion of the Day
                    </p>
                    <p className="text-xs text-gray-600">
                    Create an auto responder to email from
                    students and notify me about a draft to
                    give them
                    </p>
                </div>
                </div>

                {/* View Templates */}
                <div
                className="flex items-start gap-2 cursor-pointer hover:opacity-70 transition-opacity px-[30px] py-[0px]"
                onClick={() =>
                    handleSuggestionClick(
                    "View templates that are out there",
                    )
                }
                >
                <div className="text-center">
                    <p className="text-xs text-gray-600">
                    View{" "}
                    <span
                        style={{
                        textDecoration: "underline",
                        }}
                    >
                        Templates
                    </span>{" "}
                    that are out there
                    </p>
                </div>
                </div>

                {/* Shuffle it up */}
                <div
                className="flex items-start gap-2 cursor-pointer hover:opacity-70 transition-opacity px-[80px] py-[0px]"
                onClick={() =>
                    handleSuggestionClick("Shuffle it up")
                }
                >
                <div className="text-center">
                    <p className="text-xs text-gray-600">
                    Shuffle it up
                    </p>
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
                Please select the components in the canvas to
                edit
            </p>
            </div>
        )}
        </div>

        {/* Input Area - Always visible */}
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
            />
            <Button
            onClick={handleSendMessage}
            size="sm"
            className="h-8 w-8 p-0 flex-shrink-0"
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