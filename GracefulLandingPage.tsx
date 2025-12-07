/**
 * Langflow Landing Page - Standalone Component
 * 
 * This is a complete landing page component showcasing Langflow's AI agent building capabilities.
 * It includes scroll animations, typewriter effects, and interactive diagrams.
 * 
 * Dependencies:
 * - motion/react (formerly Framer Motion)
 * - lucide-react (for icons)
 * - react
 * - Tailwind CSS
 * 
 * Installation:
 * npm install motion lucide-react
 * 
 * Usage:
 * import LangflowLandingPage from './LangflowLandingPage';
 * <LangflowLandingPage />
 */

import { useState, useEffect } from "react";
import { motion } from "motion/react";
import { 
  Search, 
  ArrowUp, 
  User, 
  Sparkles, 
  Upload, 
  ChevronDown, 
  MessageCircle 
} from "lucide-react";

// ============================================================================
// TYPEWRITER SEARCH COMPONENT
// ============================================================================

const prompts = [
  "Build a customer support chatbot that handles refunds and complaints",
  "Create a research assistant that summarizes academic papers",
  "Design a sales agent that qualifies leads and schedules meetings",
  "Build a content moderator that filters inappropriate messages",
  "Create a personal assistant that manages my calendar and emails",
  "Design a coding helper that explains code and suggests improvements",
  "Build a language tutor that teaches conversational Spanish",
  "Create a data analyst that generates insights from CSV files",
  "Design a recipe generator based on ingredients I have",
  "Build a travel planner that suggests itineraries and books trips"
];

function TypewriterSearch() {
  const [currentPromptIndex, setCurrentPromptIndex] = useState(0);
  const [displayedText, setDisplayedText] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);
  const [typingSpeed, setTypingSpeed] = useState(50);

  useEffect(() => {
    const currentPrompt = prompts[currentPromptIndex];

    const handleTyping = () => {
      if (!isDeleting) {
        if (displayedText.length < currentPrompt.length) {
          setDisplayedText(currentPrompt.substring(0, displayedText.length + 1));
          setTypingSpeed(50);
        } else {
          setTimeout(() => setIsDeleting(true), 2000);
        }
      } else {
        if (displayedText.length > 0) {
          setDisplayedText(currentPrompt.substring(0, displayedText.length - 1));
          setTypingSpeed(30);
        } else {
          setIsDeleting(false);
          setCurrentPromptIndex((prevIndex) => (prevIndex + 1) % prompts.length);
        }
      }
    };

    const timer = setTimeout(handleTyping, typingSpeed);
    return () => clearTimeout(timer);
  }, [displayedText, isDeleting, currentPromptIndex, typingSpeed]);

  return (
    <div className="w-full mx-auto">
      <div className="relative flex items-center w-full pl-12 pr-2 py-4 border-2 border-gray-300 rounded-full bg-white transition-colors hover:border-gray-400 shadow-lg hover:shadow-xl">
        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400">
          <Search className="w-6 h-6" />
        </div>
        <div className="flex-1 min-h-[32px] leading-8 text-gray-900" style={{ fontSize: '20px' }}>
          {displayedText}
          <span className="inline-block w-0.5 h-7 bg-blue-500 ml-0.5 animate-pulse align-middle" />
        </div>
        <button 
          className="ml-2 flex items-center justify-center w-10 h-10 bg-black rounded-full hover:bg-gray-800 transition-colors flex-shrink-0"
          aria-label="Search"
        >
          <ArrowUp className="w-4 h-4 text-white" />
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// RAG DIAGRAM COMPONENTS
// ============================================================================

interface RAGCardProps {
  title: string;
  icon: React.ReactNode;
  position: { x: number; y: number };
  width?: number;
  delay?: number;
  children?: React.ReactNode;
}

function RAGCard({ title, icon, position, width = 280, delay = 0, children }: RAGCardProps) {
  return (
    <motion.div
      className="absolute bg-white rounded-xl p-4 shadow-md border border-gray-200"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        width: `${width}px`,
      }}
      initial={{ opacity: 0, scale: 0.95, y: 20 }}
      whileInView={{ opacity: 1, scale: 1, y: 0 }}
      viewport={{ once: false, margin: "-100px" }}
      transition={{ duration: 0.5, delay, ease: "easeOut" }}
    >
      <div className="flex items-center gap-2 mb-3">
        {icon}
        <h3 className="text-gray-800">{title}</h3>
      </div>
      {children}
    </motion.div>
  );
}

interface ConnectionPathProps {
  start: { x: number; y: number };
  end: { x: number; y: number };
  delay?: number;
}

function ConnectionPath({ start, end, delay = 0 }: ConnectionPathProps) {
  const midX = (start.x + end.x) / 2;
  const path = `M ${start.x} ${start.y} C ${midX} ${start.y}, ${midX} ${end.y}, ${end.x} ${end.y}`;

  return (
    <motion.path
      d={path}
      stroke="#3b82f6"
      strokeWidth="2"
      fill="none"
      strokeLinecap="round"
      initial={{ pathLength: 0, opacity: 0 }}
      whileInView={{ pathLength: 1, opacity: 1 }}
      viewport={{ once: false, margin: "-100px" }}
      transition={{ duration: 0.8, delay: delay + 1.4, ease: "easeInOut" }}
    />
  );
}

function OpenAILogo() {
  return (
    <div className="flex items-center gap-2">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" fill="#10A37F"/>
      </svg>
      <span className="text-gray-700 text-sm">GPT-4o</span>
    </div>
  );
}

function RAGDiagram() {
  const padding = 32;
  
  const chatInputCard = { x: 20, y: 20, width: 280, height: 60 };
  const webSearchCard = { x: 20, y: 140, width: 280, height: 200 };
  const fileUploadCard = { x: 20, y: 450, width: 280, height: 120 };
  const agentCard = { x: 380, y: 120, width: 260, height: 280 };
  const chatOutputCard = { x: 710, y: 180, width: 300, height: 60 };
  
  const chatInputRight = { 
    x: padding + chatInputCard.x + chatInputCard.width, 
    y: padding + chatInputCard.y + chatInputCard.height / 2 
  };
  const webSearchRight = { 
    x: padding + webSearchCard.x + webSearchCard.width, 
    y: padding + webSearchCard.y + 60 
  };
  const fileUploadRight = { 
    x: padding + fileUploadCard.x + fileUploadCard.width, 
    y: padding + fileUploadCard.y + 50 
  };
  const agentLeft1 = { 
    x: padding + agentCard.x, 
    y: padding + agentCard.y + 40 
  };
  const agentLeft2 = { 
    x: padding + agentCard.x, 
    y: padding + agentCard.y + 90 
  };
  const agentLeft3 = { 
    x: padding + agentCard.x, 
    y: padding + agentCard.y + 180 
  };
  const agentRight = { 
    x: padding + agentCard.x + agentCard.width, 
    y: padding + agentCard.y + 140 
  };
  const chatOutputLeft = { 
    x: padding + chatOutputCard.x, 
    y: padding + chatOutputCard.y + chatOutputCard.height / 2 
  };
  
  return (
    <div className="relative w-full h-[700px] p-8">
      <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 1 }}>
        <ConnectionPath start={chatInputRight} end={agentLeft1} delay={0.2} />
        <ConnectionPath start={webSearchRight} end={agentLeft2} delay={0.25} />
        <ConnectionPath start={fileUploadRight} end={agentLeft3} delay={0.3} />
        <ConnectionPath start={agentRight} end={chatOutputLeft} delay={0.35} />
        
        {/* Connection Nodes */}
        {[
          { ...chatInputRight, delay: 0.15 },
          { ...webSearchRight, delay: 0.2 },
          { ...fileUploadRight, delay: 0.25 },
          { ...agentLeft1, delay: 0.15 },
          { ...agentLeft2, delay: 0.2 },
          { ...agentLeft3, delay: 0.25 },
          { ...agentRight, delay: 0.3 },
          { ...chatOutputLeft, delay: 0.3 }
        ].map((node, idx) => (
          <motion.circle
            key={idx}
            cx={node.x}
            cy={node.y}
            r="6"
            fill="white"
            stroke="#3b82f6"
            strokeWidth="2"
            initial={{ scale: 0 }}
            whileInView={{ scale: 1 }}
            viewport={{ once: false, margin: "-100px" }}
            transition={{ duration: 0.2, delay: node.delay, ease: "easeOut" }}
          />
        ))}
      </svg>

      <div className="relative" style={{ zIndex: 2 }}>
        <RAGCard
          title="Chat Input"
          icon={<MessageCircle className="w-5 h-5 text-blue-400" />}
          position={{ x: chatInputCard.x, y: chatInputCard.y }}
          delay={0.05}
        />

        <RAGCard
          title="Web Search"
          icon={<Search className="w-5 h-5 text-blue-500" />}
          position={{ x: webSearchCard.x, y: webSearchCard.y }}
          delay={0.1}
        >
          <div className="space-y-3">
            <div>
              <p className="text-gray-500 text-xs mb-1">Query</p>
              <div className="bg-gray-50 rounded-lg px-3 py-2 border border-gray-200">
                <div className="h-4"></div>
              </div>
            </div>
            <div>
              <p className="text-gray-500 text-xs mb-1">URL</p>
              <div className="bg-gray-50 rounded-lg px-3 py-2 border border-gray-200">
                <div className="h-4"></div>
              </div>
            </div>
            <div>
              <p className="text-gray-500 text-xs mb-1">Depth</p>
              <div className="bg-gray-50 rounded-lg px-3 py-2">
                <div className="relative h-1 bg-gray-200 rounded-full">
                  <div className="absolute h-1 bg-blue-500 rounded-full" style={{ width: '50%' }}></div>
                  <div className="absolute w-3 h-3 bg-blue-500 rounded-full -top-1" style={{ left: 'calc(50% - 6px)' }}></div>
                </div>
              </div>
            </div>
          </div>
        </RAGCard>

        <RAGCard
          title="File Upload"
          icon={<Upload className="w-5 h-5 text-blue-500" />}
          position={{ x: fileUploadCard.x, y: fileUploadCard.y }}
          delay={0.15}
        >
          <div className="border-2 border-dashed border-gray-300 rounded-lg px-3 py-4 text-center">
            <Upload className="w-5 h-5 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-500 text-xs">Drop files here</p>
          </div>
        </RAGCard>

        <RAGCard
          title="Agent"
          icon={<Sparkles className="w-5 h-5 text-blue-400" />}
          position={{ x: agentCard.x, y: agentCard.y }}
          width={agentCard.width}
          delay={0.08}
        >
          <div className="space-y-4">
            <div>
              <p className="text-gray-500 text-xs mb-1">Model</p>
              <div className="flex items-center gap-2 bg-gray-50 rounded-lg px-3 py-2 border border-gray-200">
                <OpenAILogo />
                <div className="ml-auto flex items-center gap-1">
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                </div>
              </div>
            </div>
            <div>
              <p className="text-gray-500 text-xs mb-1">API Key</p>
              <div className="bg-gray-50 rounded-lg px-3 py-2 border border-gray-200">
                <p className="text-gray-600 text-xs">••••••••••••••••</p>
              </div>
            </div>
            <div>
              <p className="text-gray-500 text-xs mb-1">Role</p>
              <div className="flex items-center gap-2 bg-gray-50 rounded-lg px-3 py-2">
                <Search className="w-4 h-4 text-gray-600" />
                <p className="text-gray-800 text-sm">Researcher</p>
                <div className="ml-auto flex items-center gap-1">
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                </div>
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <p className="text-gray-700 text-sm">Tools</p>
                </div>
                <p className="text-gray-500 text-xs">3 added</p>
              </div>
            </div>
            <div>
              <p className="text-gray-500 text-xs mb-1">Prompt</p>
              <div className="bg-gray-50 rounded-lg px-3 py-2 border border-gray-200">
                <div className="h-8"></div>
              </div>
            </div>
          </div>
        </RAGCard>

        <RAGCard
          title="Chat Output"
          icon={<User className="w-5 h-5 text-gray-400" />}
          position={{ x: chatOutputCard.x, y: chatOutputCard.y }}
          width={chatOutputCard.width}
          delay={0.2}
        />
      </div>
    </div>
  );
}

// ============================================================================
// TEMPLATE CAROUSEL COMPONENT
// ============================================================================

function TemplateCarousel() {
  const templates = [
    { name: "Customer Support Bot", category: "Support" },
    { name: "Research Assistant", category: "Productivity" },
    { name: "Sales Agent", category: "Sales" },
    { name: "Content Moderator", category: "Moderation" },
    { name: "Personal Assistant", category: "Productivity" },
  ];

  return (
    <motion.div 
      className="w-full max-w-[1100px] mx-auto px-12"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: false, margin: "-100px" }}
      transition={{ duration: 0.8, ease: "easeOut" }}
    >
      <h2 className="text-gray-800 text-2xl mb-4">Popular Templates</h2>
      <div className="flex gap-4 overflow-x-auto pb-4">
        {templates.map((template, idx) => (
          <motion.div
            key={idx}
            className="min-w-[250px] bg-white rounded-xl p-6 shadow-md border border-gray-200 cursor-pointer hover:shadow-lg transition-shadow"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: false, margin: "-100px" }}
            transition={{ duration: 0.5, delay: idx * 0.1, ease: "easeOut" }}
          >
            <div className="w-12 h-12 bg-blue-100 rounded-lg mb-3 flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-blue-500" />
            </div>
            <h3 className="text-gray-800 mb-1">{template.name}</h3>
            <p className="text-gray-500 text-sm">{template.category}</p>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

// ============================================================================
// MAIN LANDING PAGE COMPONENT
// ============================================================================

export default function LangflowLandingPage() {
  return (
    <div className="min-h-screen w-full overflow-x-hidden bg-gradient-to-br from-blue-100 via-blue-50 to-indigo-100 relative">
      {/* Background effects */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(0,0,0,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(0,0,0,0.02)_1px,transparent_1px)] bg-[size:100px_100px]" />
      
      {/* Animated gradient blobs */}
      <div className="absolute blur-3xl filter left-0 opacity-[0.793] rounded-full size-[800px] top-0 bg-gradient-to-br from-blue-500/30 to-transparent animate-pulse" style={{ animationDuration: '8s' }} />
      <div className="absolute blur-3xl filter left-[641px] opacity-[0.563] rounded-full size-[600px] top-[400px] bg-gradient-to-bl from-blue-400/25 to-transparent animate-pulse" style={{ animationDuration: '10s', animationDelay: '2s' }} />
      <div className="absolute blur-3xl filter left-[372.3px] opacity-[0.904] rounded-full size-[700px] top-[1900px] bg-gradient-to-tr from-cyan-400/20 to-transparent animate-pulse" style={{ animationDuration: '12s', animationDelay: '4s' }} />

      {/* Content container */}
      <div className="relative z-10 max-w-7xl mx-auto px-8">
        {/* Hero Section */}
        <section className="pt-24 pb-16 flex flex-col items-center text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: false, margin: "-100px" }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <h1 className="text-black mb-6 leading-tight max-w-2xl">
              All Your Ideas, Imagined
            </h1>
            <p className="text-gray-600 text-xl max-w-2xl mx-auto mb-12">
              Powered by Claude, our Assistant's advanced reasoning capabilities guide you through the entire process of building an AI Agent, making AI development accessible to everyone. Unsure of what you want to create? Our Ideation feature allows you to talk through your needs and brainstorm what you want to create.
            </p>
          </motion.div>

          {/* Typewriter Search */}
          <motion.div
            className="w-full max-w-2xl"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: false, margin: "-100px" }}
            transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
          >
            <TypewriterSearch />
          </motion.div>
        </section>

        {/* Template Carousel Section */}
        <section className="py-16">
          <TemplateCarousel />
        </section>

        {/* RAG Diagram Section */}
        <section className="py-16">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: false, margin: "-100px" }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <RAGDiagram />
            
            <div className="mt-16 text-center max-w-3xl mx-auto">
              <h2 className="text-black mb-6">
                Build and Customize Your AI Agents
              </h2>
              <p className="text-gray-600 text-lg">
                Langflow's intuitive drag-and-drop interface lets you design complex AI workflows without writing a single line of code. Connect components, configure settings, and watch your agent come to life.
              </p>
            </div>
          </motion.div>
        </section>

        {/* Bottom CTA Section */}
        <section className="py-24">
          <motion.div
            className="flex flex-col items-center justify-center gap-16 text-center"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: false, margin: "-100px" }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <h2 className="text-black text-center text-[56px] leading-[64px] tracking-[0.123px] max-w-3xl">
              Explore Designs And Get Inspired
            </h2>
            <button className="bg-black text-white px-12 py-5 rounded-full text-[20px] hover:bg-gray-800 transition-colors">
              Let's begin
            </button>
          </motion.div>
        </section>
      </div>
    </div>
  );
}
