// Template catalog for Hopper AI Assistant
export interface TemplateInfo {
  id: string;
  name: string;
  category: string;
  description: string;
  useCases: string[];
  keywords: string[];
}

export const TEMPLATE_CATALOG: TemplateInfo[] = [
  {
    id: "memory_chatbot",
    name: "Memory Chatbot",
    category: "Chatbots",
    description: "A conversational AI that remembers previous messages in the conversation",
    useCases: ["customer support", "personal assistant", "tutoring", "FAQ bot"],
    keywords: ["chat", "conversation", "memory", "context", "assistant", "support"],
  },
  {
    id: "basic_prompting",
    name: "Basic Prompting",
    category: "Basic",
    description: "Simple prompt-based interaction with an AI model",
    useCases: ["simple queries", "content generation", "text transformation"],
    keywords: ["prompt", "simple", "basic", "text", "generation"],
  },
  {
    id: "vector_store_rag",
    name: "Vector Store RAG",
    category: "RAG",
    description: "Retrieval Augmented Generation using vector database for semantic search",
    useCases: ["document search", "knowledge base", "semantic search", "research"],
    keywords: ["rag", "vector", "search", "documents", "knowledge", "retrieval"],
  },
  {
    id: "document_qa",
    name: "Document Q&A",
    category: "RAG",
    description: "Ask questions about your documents and get accurate answers",
    useCases: ["document analysis", "research", "legal document review", "policy questions"],
    keywords: ["qa", "questions", "answers", "documents", "search", "find information"],
  },
  {
    id: "blog_writer",
    name: "Blog Writer",
    category: "Content Generation",
    description: "Generate high-quality blog posts and articles",
    useCases: ["content marketing", "blog writing", "article generation", "SEO content"],
    keywords: ["blog", "writing", "content", "articles", "marketing", "seo"],
  },
  {
    id: "complex_agent",
    name: "AI Agent",
    category: "Agents",
    description: "Autonomous AI agent that can use tools and make decisions",
    useCases: ["automation", "task completion", "research", "multi-step workflows"],
    keywords: ["agent", "autonomous", "tools", "automation", "workflow", "tasks"],
  },
  {
    id: "sequential_tasks_agent",
    name: "Sequential Tasks Agent",
    category: "Agents",
    description: "AI agent that executes tasks in a specific order",
    useCases: ["workflow automation", "step-by-step processes", "data pipelines"],
    keywords: ["sequential", "workflow", "pipeline", "steps", "order", "process"],
  },
  {
    id: "hierarchical_tasks_agent",
    name: "Hierarchical Tasks Agent",
    category: "Agents",
    description: "AI agent with multiple sub-agents handling different specialized tasks",
    useCases: ["complex workflows", "team coordination", "multi-domain problems"],
    keywords: ["hierarchical", "complex", "team", "coordination", "specialized"],
  },
];

export const getTemplatesByKeywords = (keywords: string[]): TemplateInfo[] => {
  const lowerKeywords = keywords.map((k) => k.toLowerCase());
  return TEMPLATE_CATALOG.filter((template) =>
    template.keywords.some((tk) =>
      lowerKeywords.some((kw) => tk.includes(kw) || kw.includes(tk))
    )
  ).slice(0, 3); // Return top 3 matches
};

// System prompt for Hopper AI Assistant
export const HOPPER_SYSTEM_PROMPT = `You are Hopper, a friendly AI assistant for Graceful AI's Langflow platform. Your goal is to help users discover the perfect flow template for their needs.

AVAILABLE TEMPLATES:
${TEMPLATE_CATALOG.map(
  (t) =>
    `- ${t.name} (${t.category}): ${t.description}\n  Use cases: ${t.useCases.join(", ")}`
).join("\n")}

YOUR APPROACH:
1. If the user describes a clear goal or use case, recommend 1-2 best matching templates
2. If the user is unsure ("I'm not sure what to build"), ask them about:
   - Their role, industry, or professional context
   - A problem they're trying to solve
   - Type of work they do or data they work with
3. Keep it conversational and friendly - maximum 2-3 questions before suggesting templates
4. When recommending templates, explain what they'll be able to DO (not technical details)
5. Don't use the word "template" - say things like "build a system that..." or "create a flow that..."

IMPORTANT RULES:
- Be concise (2-3 sentences max per response)
- Ask ONE question at a time
- When suggesting templates, respond with: "RECOMMEND: [template_id]" on a new line at the end
- You can recommend multiple templates: "RECOMMEND: template1, template2"
- Focus on user benefits, not technical implementation

CONVERSATION STYLE:
- Warm and encouraging
- Use simple language
- Get to recommendations quickly (don't over-ask)

Example responses:
"That sounds perfect for a Document Q&A system! You'll be able to upload your policy documents and instantly get answers to questions. Would you like to create this? RECOMMEND: document_qa"

"Great! A Memory Chatbot would be ideal for customer support. Your bot will remember the conversation context and provide personalized help. Ready to build it? RECOMMEND: memory_chatbot"`;

