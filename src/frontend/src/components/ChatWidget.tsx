import { useState, useRef, useEffect } from "react";
import { motion } from "motion/react";
import { LuBotMessageSquare } from "react-icons/lu";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface ChatWidgetProps {
  headerMode?: boolean;
}

export default function ChatWidget({ headerMode = false }: ChatWidgetProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [position, setPosition] = useState({ x: 50, y: 50 });
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef<{ startX: number; startY: number; startPosX: number; startPosY: number } | null>(null);

  // If in header mode, render a simple inline button
  if (headerMode) {
    return (
      <>
        <motion.div
          className="hit-area-hover flex items-center gap-2 rounded-md p-1 text-muted-foreground cursor-pointer"
          animate={{ scale: isHovered ? 1.05 : 1 }}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          onClick={() => setIsChatOpen(true)}
        >
          <motion.div
            animate={{ rotate: isHovered ? [0, -2, 2, -2, 0] : 0 }}
            transition={{ duration: 0.5 }}
          >
            <LuBotMessageSquare className="h-4 w-4" />
          </motion.div>
          <span className="text-xs font-semibold">AI Chat</span>
        </motion.div>

        <Dialog open={isChatOpen} onOpenChange={setIsChatOpen}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Chat with AI Assistant</DialogTitle>
            </DialogHeader>
            <div className="flex flex-col gap-4 py-4">
              <div className="flex flex-col gap-3 h-[300px] overflow-y-auto border rounded-lg p-4 bg-gray-50">
                <div className="flex gap-2">
                  <div className="bg-blue-500 text-white rounded-lg px-4 py-2 max-w-[80%]">
                    Hi there! How can I help you today? 👋
                  </div>
                </div>
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Type your message..."
                  className="flex-1 border rounded-lg px-4 py-2"
                />
                <button className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600">
                  Send
                </button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </>
    );
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startPosX: position.x,
      startPosY: position.y,
    };
    e.preventDefault();
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging && dragRef.current) {
      const deltaX = e.clientX - dragRef.current.startX;
      const deltaY = e.clientY - dragRef.current.startY;
      
      setPosition({
        x: dragRef.current.startPosX + deltaX,
        y: dragRef.current.startPosY + deltaY,
      });
    }
  };

  const handleMouseUp = (e: MouseEvent) => {
    if (isDragging) {
      const deltaX = Math.abs(e.clientX - (dragRef.current?.startX || 0));
      const deltaY = Math.abs(e.clientY - (dragRef.current?.startY || 0));
      
      // Only open chat if it wasn't dragged (small movement threshold)
      if (deltaX < 5 && deltaY < 5) {
        setIsChatOpen(true);
      }
      
      setIsDragging(false);
      dragRef.current = null;
    }
  };

  // Add/remove global mouse event listeners
  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => handleMouseMove(e);
    const handleGlobalMouseUp = (e: MouseEvent) => handleMouseUp(e);

    if (isDragging) {
      window.addEventListener("mousemove", handleGlobalMouseMove);
      window.addEventListener("mouseup", handleGlobalMouseUp);

      return () => {
        window.removeEventListener("mousemove", handleGlobalMouseMove);
        window.removeEventListener("mouseup", handleGlobalMouseUp);
      };
    }
  }, [isDragging]);

  return (
    <>
      <motion.div
        style={{
          position: "fixed",
          left: position.x,
          top: position.y,
          width: "80px",
          height: "80px",
          cursor: isDragging ? "grabbing" : "grab",
          zIndex: 50,
        }}
        animate={{
          scale: isHovered ? 1.15 : 1,
        }}
        transition={{
          type: "spring",
          stiffness: 300,
          damping: 20,
        }}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        onMouseDown={handleMouseDown}
      >
        <div className="relative size-full bg-white rounded-[10px] border-[3px] border-black border-solid flex items-center justify-center">
          <motion.div
            animate={{
              rotate: isHovered ? [0, -5, 5, -5, 0] : 0,
            }}
            transition={{
              duration: 0.5,
            }}
          >
            <LuBotMessageSquare className="w-8 h-8 text-blue-600" />
          </motion.div>
        </div>
      </motion.div>

      {/* Chat Dialog */}
      <Dialog open={isChatOpen} onOpenChange={setIsChatOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Chat with Bot</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-4 py-4">
            <div className="flex flex-col gap-3 h-[300px] overflow-y-auto border rounded-lg p-4 bg-gray-50">
              <div className="flex gap-2">
                <div className="bg-blue-500 text-white rounded-lg px-4 py-2 max-w-[80%]">
                  Hi there! How can I help you today? 👋
                </div>
              </div>
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Type your message..."
                className="flex-1 border rounded-lg px-4 py-2"
              />
              <button className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600">
                Send
              </button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}