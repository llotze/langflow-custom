import { Sparkles } from "lucide-react";
import { TypewriterMessage } from "./TypewriterMessage";

export function HopperChat() {
  return (
    <div className="relative w-[383px] h-[600px] bg-white rounded-xl shadow-2xl overflow-hidden flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-[20px] font-semibold text-gray-900 leading-tight">
              AI Agent Builder
            </h2>
            <p className="text-xs text-gray-500">Powered by Claude</p>
          </div>
        </div>
      </div>

      {/* Preview Badge */}
      <div className="px-6 pt-3">
        <div className="inline-flex items-center gap-1.5 bg-gradient-to-r from-purple-500 to-purple-600 text-white px-3 py-1.5 rounded-full text-xs font-medium shadow-sm">
          <Sparkles className="w-3 h-3" />
          Preview: What you could build
        </div>
      </div>

      {/* Chat Content Area */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {/* AI Message */}
        <div className="flex justify-start">
          <div className="max-w-[85%]">
            <div className="flex items-start gap-2">
              <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <span className="text-lg">🤖</span>
              </div>
              <div className="bg-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                <p className="text-gray-900 text-[15px] leading-[22px]">
                  Hi! I'm here to help you build your AI agent. What would you like to create?
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* User Message with Typewriter */}
        <TypewriterMessage />
      </div>

      {/* Disabled Input Footer */}
      <div className="px-4 py-4 border-t border-gray-200 bg-gray-50">
        <div className="flex items-center gap-2">
          <div className="flex-1 bg-gray-200 rounded-lg px-4 py-3 cursor-not-allowed opacity-60">
            <div className="flex items-center gap-2">
              <svg
                className="w-4 h-4 text-gray-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                />
              </svg>
              <span className="text-gray-500 text-sm">Example preview only...</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}



