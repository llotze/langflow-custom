import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import {
  Sparkles,
  ChevronDown,
  MessageCircle,
  ArrowRight,
  ChevronRight,
  Rocket,
  Check,
  Loader2,
  Play,
  Lightbulb
} from "lucide-react";
import { HopperChat } from "./components/HopperChat";
import TemplateExamples from "./components/TemplateExamples";
import LogoCarousel from "./components/LogoCarousel";
import GracefulLogo from "@/assets/graceful/graceful-no-ai-made-easy-full-logo.png";


// ============================================================================
// PROCESS BREAKDOWN COMPONENT
// ============================================================================

function ProcessBreakdown() {
  const navigate = useNavigate();
  
  return (
    <section className="py-20 px-8">
      <motion.div
        className="max-w-7xl mx-auto"
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <div className="text-center mb-16">
          <h2 className="text-black text-4xl lg:text-5xl font-bold mb-6">
            From Conversation to Working Agent
          </h2>
          <p className="text-gray-600 text-lg max-w-3xl mx-auto">
            No coding required. Just describe what you want, and our AI builds the entire workflow for you in a fraction of the time.
          </p>
        </div>

        <div className="flex flex-col lg:flex-row items-stretch justify-center gap-4">
          <motion.div
            className="w-full lg:w-[320px] bg-white rounded-xl p-6 border border-gray-200 flex flex-col"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
          >
            <div className="flex items-start gap-4 mb-4">
              <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center flex-shrink-0">
                <MessageCircle className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1">
                <span className="text-blue-500 text-xs font-semibold uppercase tracking-wide">Step 1</span>
                <h3 className="text-black text-xl font-bold mt-1">Describe Your Idea</h3>
              </div>
            </div>
            <p className="text-gray-600 mb-4">
              Tell our AI what you want to build in plain English. No technical knowledge needed.
            </p>
            <div className="bg-gray-100/60 rounded-lg p-3 border border-gray-200 h-[200px] flex flex-col">
              <p className="text-gray-700 text-sm italic mb-4">
                "Create a study assistant that uses the attached history files. It should explain key concepts, highlight main themes, and quiz me as I study."
              </p>
              <div className="space-y-2 flex-1">
                <div className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-blue-500 flex-shrink-0" />
                  <span className="text-gray-700 text-sm">Natural language input</span>
                </div>
                <div className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-blue-500 flex-shrink-0" />
                  <span className="text-gray-700 text-sm">No coding required</span>
                </div>
                <div className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-blue-500 flex-shrink-0" />
                  <span className="text-gray-700 text-sm">AI understands context</span>
                </div>
              </div>
            </div>
          </motion.div>

          <div className="hidden lg:flex items-center justify-center">
            <ChevronRight className="w-8 h-8 text-gray-400" />
          </div>

          <motion.div
            className="w-full lg:w-[320px] bg-white rounded-xl p-6 border border-gray-200 flex flex-col"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
          >
            <div className="flex items-start gap-4 mb-4">
              <div className="w-12 h-12 bg-purple-500 rounded-lg flex items-center justify-center flex-shrink-0">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1">
                <span className="text-purple-500 text-xs font-semibold uppercase tracking-wide">Step 2</span>
                <h3 className="text-black text-lg font-bold mt-1 leading-tight">AI Builds the Workflow</h3>
              </div>
            </div>
            <p className="text-gray-600 mb-4">
              Watch as AI automatically creates the visual workflow with all the components and connections.
            </p>
            <div className="bg-gray-100/60 rounded-lg p-3 border border-gray-200 h-[200px] flex flex-col">
              <div className="flex items-center gap-2 mb-3">
                <Loader2 className="w-4 h-4 text-purple-500 animate-spin flex-shrink-0" style={{ animationDuration: '2s' }} />
                <span className="text-purple-600 text-sm font-medium">Building components...</span>
              </div>
              <motion.div 
                className="space-y-2 flex-1"
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-100px" }}
                variants={{
                  hidden: { opacity: 0 },
                  visible: { opacity: 1 }
                }}
              >
                {[
                  { name: 'Chat Input', delay: 0 },
                  { name: 'Web Search Tool', delay: 0.3 },
                  { name: 'Agent Logic', delay: 0.6 },
                  { name: 'Response Generator', delay: 0.9 },
                  { name: 'Chat Output', delay: 1.2 },
                ].map((component) => (
                  <motion.div
                    key={component.name}
                    className="flex items-center justify-between gap-2"
                    initial={{ opacity: 0, x: -10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true, margin: "-100px" }}
                    transition={{ duration: 0.4, delay: component.delay }}
                  >
                    <div className="flex items-center gap-2 flex-1">
                      <motion.div
                        className="w-2 h-2 bg-purple-500 rounded-full flex-shrink-0"
                        initial={{ scale: 0 }}
                        whileInView={{ scale: 1 }}
                        viewport={{ once: true, margin: "-100px" }}
                        transition={{ duration: 0.3, delay: component.delay + 0.2 }}
                      />
                      <span className="text-purple-700 text-sm">{component.name}</span>
                    </div>
                    <motion.div
                      initial={{ opacity: 0, scale: 0 }}
                      whileInView={{ opacity: 1, scale: 1 }}
                      viewport={{ once: true, margin: "-100px" }}
                      transition={{ duration: 0.2, delay: component.delay + 0.3 }}
                    >
                      <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                    </motion.div>
                  </motion.div>
                ))}
              </motion.div>
            </div>
          </motion.div>

          <div className="hidden lg:flex items-center justify-center">
            <ChevronRight className="w-8 h-8 text-gray-400" />
          </div>

          <motion.div
            className="w-full lg:w-[320px] bg-white rounded-xl p-6 border border-gray-200 flex flex-col"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, delay: 0.3, ease: "easeOut" }}
          >
            <div className="flex items-start gap-4 mb-4">
              <div className="w-12 h-12 bg-green-500 rounded-lg flex items-center justify-center flex-shrink-0">
                <Rocket className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1">
                <span className="text-green-500 text-xs font-semibold uppercase tracking-wide">Step 3</span>
                <h3 className="text-black text-xl font-bold mt-1">Test & Deploy</h3>
              </div>
            </div>
            <p className="text-gray-600 mb-4">
              Click Playground to test your agent instantly. Chat with it, refine, then deploy.
            </p>
            <div className="bg-gray-100/60 rounded-lg p-2 border border-gray-200 h-[200px] flex flex-col overflow-hidden">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 bg-green-500 rounded-full flex-shrink-0"></div>
                <span className="text-green-600 text-sm font-medium">Agent Ready</span>
              </div>
              <button className="w-full mb-2 flex items-center justify-center gap-2 bg-white border border-gray-300 rounded-lg px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50 transition-colors flex-shrink-0">
                <Play className="w-3 h-3" />
                Playground
              </button>
              <div className="border-t border-gray-300 mb-2"></div>
              <div className="flex-1 flex flex-col justify-start gap-2">
                <div className="bg-gray-100 border border-gray-300 rounded-lg p-1.5 self-start max-w-[85%]">
                  <p className="text-gray-700 text-xs">Hello! How can I help?</p>
                </div>
                <div className="bg-blue-100 rounded-lg p-1.5 self-end max-w-[85%]">
                  <p className="text-gray-700 text-xs">What should I study most for the exam?</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </section>
  );
}

// ============================================================================
// IDEATION SHOWCASE COMPONENT
// ============================================================================

function IdeationShowcase() {
  const navigate = useNavigate();
  
  return (
    <section className="py-20 px-8">
      <motion.div
        className="max-w-7xl mx-auto"
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <div className="bg-gradient-to-br from-orange-50 via-yellow-50 to-orange-50 rounded-3xl border-2 border-orange-200 overflow-hidden relative">
          {/* Decorative blur circles */}
          <div className="absolute -right-12 -top-12 w-48 h-48 bg-orange-200/30 rounded-full blur-3xl" />
          <div className="absolute -left-12 -bottom-12 w-48 h-48 bg-yellow-200/30 rounded-full blur-3xl" />
          
          {/* Content wrapper */}
          <div className="relative z-10">
            <div className="grid lg:grid-cols-2 gap-0 items-stretch">
            {/* Left Column */}
            <div className="space-y-5 bg-yellow-50 p-6 md:p-8 lg:p-10">
              <div className="w-16 h-16 bg-gradient-to-br from-orange-500 to-yellow-500 rounded-2xl flex items-center justify-center">
                <Lightbulb className="w-8 h-8 text-white" />
              </div>
            <h2 className="text-black text-3xl sm:text-4xl lg:text-5xl font-bold lg:whitespace-nowrap">
              Not sure what to build?
            </h2>
            <p className="text-gray-700 text-lg">
              Our ideation assistant asks targeted questions about your workflow, identifies time-consuming tasks, and suggests AI agents perfectly suited to your needs.
            </p>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-gray-900">Guided conversation:</span>{" "}
                  <span className="text-gray-700">AI asks the right questions to understand your needs</span>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-gray-900">Smart recommendations:</span>{" "}
                  <span className="text-gray-700">Get personalized agent suggestions based on your workflow</span>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-gray-900">Instant building:</span>{" "}
                  <span className="text-gray-700">Go from conversation to working agent in minutes</span>
                </div>
              </div>
            </div>
            <button
              onClick={() => navigate("/signup")}
              className="bg-gray-800 hover:bg-gray-900 text-white px-6 py-3 rounded-lg font-medium flex items-center gap-2 transition-colors"
            >
              Sign Up to Access
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>

          {/* Right Column - Chat Preview Card (non-interactive) */}
          <div className="bg-white p-6 md:p-8 lg:p-10 flex items-center">
            <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200 w-full">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-orange-500 rounded-lg flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="text-black font-bold text-lg">Ideation Assistant</h3>
                <p className="text-gray-500 text-sm">Available after sign up</p>
              </div>
            </div>
            <div className="border-t border-gray-200 mb-4"></div>
            <div className="space-y-4 mb-6">
              <div className="bg-gray-100 border border-gray-300 rounded-lg p-3 self-start max-w-[85%]">
                <p className="text-gray-700 text-sm">What tasks take up most of your time?</p>
              </div>
              <div className="bg-blue-100 rounded-lg p-3 self-end max-w-[85%] ml-auto">
                <p className="text-gray-700 text-sm">Grading short-answer assignments…</p>
              </div>
              <div className="bg-gray-100 border border-gray-300 rounded-lg p-3 self-start max-w-[85%]">
                <p className="text-gray-700 text-sm">I can help you build an agent that speeds up grading.</p>
              </div>
            </div>
            <button
              onClick={() => navigate("/signup")}
              className="w-full bg-orange-50 border-2 border-orange-500 text-orange-600 px-6 py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors hover:bg-orange-100"
            >
              <Lightbulb className="w-5 h-5 stroke-orange-500 fill-white" />
              Sign up to start chatting
            </button>
            </div>
          </div>
        </div>
          </div>
        </div>
      </motion.div>
    </section>
  );
}

// ============================================================================
// MAIN LANDING PAGE COMPONENT
// ============================================================================

export default function LandingPage() {
  const navigate = useNavigate();
  const [scrollY, setScrollY] = useState(0);
  const [hasAppeared, setHasAppeared] = useState(false);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setHasAppeared(true);
    }, 2000);
    
    const handleScroll = (e: Event) => {
      const target = e.target as HTMLDivElement;
      setScrollY(target.scrollTop);
    };
    
    const container = document.getElementById('landing-page-container');
    if (container) {
      container.addEventListener('scroll', handleScroll, { passive: true });
      // Set initial scroll position
      setScrollY(container.scrollTop);
    }
    
    return () => {
      clearTimeout(timer);
      if (container) {
        container.removeEventListener('scroll', handleScroll);
      }
    };
  }, []);

  const scrollIndicatorOpacity = hasAppeared 
    ? Math.max(0, 1 - scrollY / 20)
    : 0;

  const handleSignIn = () => {
    navigate("/login");
  };

  const handleSignUp = () => {
    navigate("/signup");
  };

  const handleStartBuilding = handleSignUp;

  return (
    <div 
      id="landing-page-container"
      className="h-screen w-full overflow-y-auto overflow-x-hidden bg-gradient-to-br from-blue-100 via-blue-50 to-indigo-100 relative"
    >
      {/* Background effects */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(0,0,0,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(0,0,0,0.02)_1px,transparent_1px)] bg-[size:100px_100px]" />

      {/* Animated gradient blobs */}
      <div className="absolute blur-3xl filter left-0 opacity-[0.793] rounded-full size-[800px] top-0 bg-gradient-to-br from-blue-500/30 to-transparent animate-pulse" style={{ animationDuration: '8s' }} />
      <div className="absolute blur-3xl filter left-[641px] opacity-[0.563] rounded-full size-[600px] top-[400px] bg-gradient-to-bl from-blue-400/25 to-transparent animate-pulse" style={{ animationDuration: '10s', animationDelay: '2s' }} />
      <div className="absolute blur-3xl filter left-[372.3px] opacity-[0.904] rounded-full size-[700px] top-[1900px] bg-gradient-to-tr from-cyan-400/20 to-transparent animate-pulse" style={{ animationDuration: '12s', animationDelay: '4s' }} />

      {/* Header */}
      <header className="relative z-20 w-full py-6">
        <div className="flex items-center justify-between px-8">
          {/* Logo - positioned at far left */}
          <div className="flex items-center">
            <img
              src={GracefulLogo}
              alt="Graceful Logo"
              className="h-16 w-auto object-contain"
            />
          </div>

          {/* Sign In/Sign Up Buttons - positioned at far right */}
          <div className="flex items-center gap-4">
            <button
              onClick={handleSignIn}
              className="px-6 py-2 text-gray-700 hover:text-gray-900 transition-colors font-medium"
            >
              Sign In
            </button>
            <button
              onClick={handleSignUp}
              className="px-6 py-2 bg-black text-white rounded-full hover:bg-gray-800 transition-colors font-medium"
            >
              Sign Up
            </button>
          </div>
        </div>
      </header>

      {/* Content container */}
      <div className="relative z-10 max-w-7xl mx-auto px-8">
        {/* Hero Section */}
        <section className="pt-8 pb-20 flex flex-col lg:flex-row items-center justify-between gap-12">
          {/* Left side - Text content */}
          <motion.div
            className="flex-1 text-left max-w-2xl"
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <h1 className="text-black mb-6 leading-tight text-6xl lg:text-7xl xl:text-8xl font-bold">
              All Your Ideas, <span className="text-[#29ABE2]">Imagined</span>
            </h1>
            <p className="text-gray-600 text-xl max-w-2xl mb-8">
              Create AI agents and workflows through natural conversation. Describe what you need, and our assistant builds it for you — no code, no friction. Go from idea to fully functional AI tools in one seamless flow.
            </p>

            {/* Start Building Now Button */}
            <motion.button
              onClick={handleStartBuilding}
              className="group inline-flex items-center gap-2 px-8 py-4 bg-black text-white rounded-full hover:bg-gray-800 transition-all duration-300 font-semibold text-lg shadow-lg hover:shadow-xl"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Start Building Now
              <ArrowRight className="w-5 h-5 transition-transform duration-300 group-hover:translate-x-1" />
            </motion.button>
          </motion.div>

          {/* Right side - Hopper Chat */}
          <motion.div
            className="flex-shrink-0"
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
          >
            <HopperChat />
          </motion.div>
        </section>

        {/* Scroll Indicator - disappears when user scrolls */}
        <div
          className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-20 transition-opacity duration-300"
          style={{ opacity: scrollIndicatorOpacity, pointerEvents: scrollIndicatorOpacity > 0 ? 'auto' : 'none' }}
        >
          <motion.div
            className="flex flex-col items-center gap-2 text-gray-500"
            animate={{ y: [0, 6, 0] }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            <span className="text-sm font-medium">Scroll to explore</span>
            <ChevronDown className="w-5 h-5" />
          </motion.div>
        </div>

        {/* Process Breakdown Section */}
        <ProcessBreakdown />

        {/* Ideation Showcase Section */}
        <IdeationShowcase />

        {/* See What You Can Build Header Section */}
        <section className="py-20 px-8">
          <motion.div
            className="max-w-7xl mx-auto text-center"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <h2 className="text-black text-4xl lg:text-5xl font-bold mb-6">
              See What You Can Build
            </h2>
            <p className="text-gray-600 text-lg max-w-3xl mx-auto mb-12">
              Explore real-world examples of AI agents built with Graceful. Each one was created in minutes with just a simple description.
            </p>
          </motion.div>
          {/* Template Examples Section - directly under header */}
          <TemplateExamples />
        </section>

        {/* Logo Carousel Section */}
        <LogoCarousel />


      </div>
    </div>
  );
}
