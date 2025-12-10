import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useGetBasicExamplesQuery } from "@/controllers/API/queries/flows/use-get-basic-examples";
import useAddFlow from "@/hooks/flows/use-add-flow";
import { updateIds } from "@/utils/reactflowUtils";
import type { FlowType } from "@/types/flow";
import useAuthStore from "@/stores/authStore";
import { BookOpen, FileSearch, BookOpenCheck, FileVolume } from "lucide-react";
import { FaRobot } from "react-icons/fa";
import TemplateFlowPreview from "./TemplateFlowPreview";
import taAssistantJson from "@/assets/graceful/LandingPage/json-examples/TA-assistant-example.json";
import sentimentAnalysisJson from "@/assets/graceful/LandingPage/json-examples/txt-sentiment-analysis-example.json";
import researchAssistantJson from "@/assets/graceful/LandingPage/json-examples/research-assiatant-example.json";
import lectureSummarizerJson from "@/assets/graceful/LandingPage/json-examples/lecture-summarizer-example.json";

// Simple icon mapping by example name
const EXAMPLE_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  "AI Teaching Assistant": FaRobot,
  "Course Feedback Summarizer": FileSearch,
  "AI Research Assistant": BookOpenCheck,
  "AI Lecture Summarizer": FileVolume,
};

// Icon color classes for template icons
const ICON_COLORS = [
  "bg-blue-500 text-white",
  "bg-green-500 text-white",
  "bg-purple-500 text-white",
  "bg-orange-500 text-white",
  "bg-cyan-500 text-white",
  "bg-pink-500 text-white",
];

// Icon color classes without text-white (for header icons)
const ICON_COLORS_HEADER = [
  "bg-blue-500",
  "bg-green-500",
  "bg-purple-500",
  "bg-orange-500",
  "bg-cyan-500",
  "bg-pink-500",
];

const getTemplateIcon = (template: FlowType) => {
  // Check for exact name match first
  if (template.name && EXAMPLE_ICONS[template.name]) {
    return EXAMPLE_ICONS[template.name];
  }
  
  // Default fallback
  return BookOpen;
};

export default function TemplateExamples() {
  const navigate = useNavigate();
  const addFlow = useAddFlow();
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const { data: templates, isLoading } = useGetBasicExamplesQuery();
  const [selectedTemplates, setSelectedTemplates] = useState<FlowType[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  
  useEffect(() => {
    if (templates && templates.length > 0) {
      // Filter out unwanted templates
      const filteredTemplates = templates.filter(t => 
        t.name !== "Document Q&A" && 
        t.name !== "Custom Component Generator" &&
        t.name !== "Image Sentiment Analysis" &&
        t.name !== "Blog Writer" &&
        t.name !== "Instagram Copy Writer"
      );

      // Get first 4 templates
      const templatesToShow = filteredTemplates.slice(0, 4);
      
      // Update the first template to be "AI Teaching Assistant" with JSON data
      if (templatesToShow.length > 0) {
        templatesToShow[0] = {
          ...templatesToShow[0],
          name: "AI Teaching Assistant",
          description: "This AI Teaching Assistant provides 24/7 support to students by answering questions directly from course materials. Professors can simply upload their syllabus, lectures, and readings, and the assistant will provide accurate, cited answers to help students succeed.",
          data: taAssistantJson.data as any, // Inject JSON data
        };
      }

      // Find and replace "Basic Prompting" template with sentiment analysis
      const basicPromptingIndex = templatesToShow.findIndex(t => 
        t.name?.toLowerCase().includes("basic prompting") || 
        t.name?.toLowerCase().includes("basic prompt")
      );
      
      if (basicPromptingIndex !== -1) {
        templatesToShow[basicPromptingIndex] = {
          ...templatesToShow[basicPromptingIndex],
          name: "Course Feedback Summarizer",
          description: "A professor receives hundreds of student comments every semester and doesn't have time to read them all. This flow automatically analyzes course feedback, groups comments by theme, detects sentiment, and highlights trends. It helps instructors quickly understand what's working, what isn't, and how to improve the student experience.",
          data: sentimentAnalysisJson.data as any, // Inject JSON data
        };
      }

      // Find and replace "Financial Report Parser" template with research assistant
      const financialReportIndex = templatesToShow.findIndex(t => 
        t.name?.toLowerCase().includes("financial report parser") || 
        t.name?.toLowerCase().includes("financial report")
      );
      
      if (financialReportIndex !== -1) {
        templatesToShow[financialReportIndex] = {
          ...templatesToShow[financialReportIndex],
          name: "AI Research Assistant",
          description: "This assistant uses Retrieval-Augmented Generation (RAG) to pull in relevant information from multiple sources and summarize it for the student.\n\nIt highlights important ideas, organizes insights, and makes complex research topics easier to understand when writing papers or preparing projects.",
          data: researchAssistantJson.data as any, // Inject JSON data
        };
      }

      // Find and replace "Hybrid Search RAG" template with lecture summarizer
      const hybridSearchIndex = templatesToShow.findIndex(t => 
        t.name?.toLowerCase().includes("hybrid search rag") || 
        t.name?.toLowerCase().includes("hybrid search") ||
        (t.name?.toLowerCase().includes("rag") && !t.name?.toLowerCase().includes("research"))
      );
      
      if (hybridSearchIndex !== -1) {
        templatesToShow[hybridSearchIndex] = {
          ...templatesToShow[hybridSearchIndex],
          name: "AI Lecture Summarizer",
          description: lectureSummarizerJson.description || "Upload a lecture recording and instantly receive a clear summary with key ideas, explanations, and important concepts. The assistant also generates optional study notes or review questions to make exam preparation easier.",
          data: lectureSummarizerJson.data as any, // Inject JSON data
        };
      }

      setSelectedTemplates(templatesToShow);
      // Auto-select first template
      if (templatesToShow[0]?.id) {
        setSelectedTemplateId(templatesToShow[0].id);
      }
    }
  }, [templates]);
  
  const handleUseTemplate = async (template: FlowType) => {
    // If user is not authenticated, route to signup page
    if (!isAuthenticated) {
      navigate("/signup");
      return;
    }

    // User is authenticated, proceed with creating the flow
    if (template.data) {
      updateIds(template.data);
    }
    const id = await addFlow({ flow: template });
    navigate(`/flow/${id}`);
  };
  
  const handleTemplateClick = (templateId: string) => {
    setSelectedTemplateId(templateId);
  };
  
  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto text-center">
        <div className="text-gray-500">Loading templates...</div>
      </div>
    );
  }

  if (!selectedTemplates.length) {
    return null;
  }

  const selectedTemplate = selectedTemplates.find(t => t.id === selectedTemplateId) || selectedTemplates[0];

  return (
    <motion.div
      className="max-w-7xl mx-auto"
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.8, ease: "easeOut" }}
    >
      {/* New Layout: Sidebar (left) + Content (right) */}
      <div className="flex flex-col lg:flex-row gap-8">
        
        {/* Left Sidebar: Template List */}
        <div className="w-full lg:w-1/5 flex flex-col gap-3">
          {selectedTemplates.map((template, idx) => {
            const IconComponent = getTemplateIcon(template);
            const isSelected = selectedTemplateId === template.id;
            const colorClass = ICON_COLORS[idx % ICON_COLORS.length];
            
            return (
              <motion.button
                key={template.id || idx}
                onClick={() => handleTemplateClick(template.id || String(idx))}
                className={`flex items-center gap-3 p-2 rounded-xl text-left transition-all border ${
                  isSelected 
                    ? "bg-white border-blue-200 shadow-md ring-1 ring-blue-100" 
                    : "bg-white border-transparent hover:bg-gray-50 hover:shadow-sm"
                }`}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-100px" }}
                transition={{ duration: 0.3, delay: idx * 0.05 }}
              >
                <div className={`w-10 h-10 ${colorClass} rounded-lg flex items-center justify-center flex-shrink-0`}>
                  <IconComponent className="w-5 h-5" />
                </div>
                <div className="flex flex-col">
                  <span className={`text-sm font-bold ${isSelected ? "text-gray-900" : "text-gray-700"}`}>
                    {template.name}
                  </span>
                </div>
              </motion.button>
            );
          })}
        </div>

        {/* Right Content: Description + Preview */}
        <div className="w-full lg:w-4/5">
          <AnimatePresence mode="wait">
            {selectedTemplate && (
              <motion.div
                key={selectedTemplate.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
                className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden"
              >
                {/* Header Section inside the card */}
                <div className="p-8 border-b border-gray-100">
                  <div className="flex items-start justify-between gap-6">
                    <div className="flex items-start gap-4">
                      {(() => {
                        const IconComponent = getTemplateIcon(selectedTemplate);
                        const iconIndex = selectedTemplates.findIndex(t => t.id === selectedTemplate.id);
                        const colorClass = ICON_COLORS_HEADER[iconIndex % ICON_COLORS_HEADER.length];
                        
                        return (
                          <div className={`w-12 h-12 ${colorClass} rounded-xl flex items-center justify-center flex-shrink-0`}>
                            <IconComponent className="w-6 h-6 text-white" />
                          </div>
                        );
                      })()}
                      <div>
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">
                          {selectedTemplate.name}
                        </h3>
                        <p className="text-gray-600 text-base leading-relaxed max-w-2xl whitespace-pre-line">
                          {selectedTemplate.description}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => handleUseTemplate(selectedTemplate)}
                      className="flex-shrink-0 bg-blue-600 text-white px-6 py-2.5 rounded-lg font-medium hover:bg-blue-700 transition-colors shadow-sm hover:shadow flex items-center gap-2"
                    >
                      Build This
                    </button>
                  </div>
                </div>

                {/* Preview Area */}
                <div className="p-6 bg-gray-50/50">
                  <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden relative group/preview">
                      <TemplateFlowPreview template={selectedTemplate} />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

      </div>
    </motion.div>
  );
}
