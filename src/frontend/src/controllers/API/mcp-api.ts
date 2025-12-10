import axios from "axios";

// Use Vite env variable (must be prefixed with VITE_)
const MCP_SERVER_URL = import.meta.env.VITE_MCP_SERVER_URL || "http://localhost:3001";

// Create a separate axios instance for MCP server (no auth needed)
export const mcpApi = axios.create({
  baseURL: MCP_SERVER_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// MCP API helper functions
export const mcpApiHelpers = {
  // Health check
  healthCheck: () => mcpApi.get("/health"),

  // Template operations
  searchTemplates: (keyword?: string, tags?: string, page?: number, pageSize?: number) =>
    mcpApi.get("/mcp/api/search-templates", { 
      params: { keyword, tags, page, pageSize } 
    }),

  getTemplate: (templateId: string) =>
    mcpApi.get(`/mcp/api/get-template/${templateId}`),

  createFlowFromTemplate: (templateId: string, name?: string, description?: string) =>
    mcpApi.post(`/mcp/api/create-flow-from-template/${templateId}`, { name, description }),

  // Flow operations
  tweakFlow: (flowId: string, operations: any[], validateAfter?: boolean, continueOnError?: boolean) =>
    mcpApi.post(`/mcp/api/tweak-flow/${flowId}`, { 
      operations, 
      validateAfter, 
      continueOnError 
    }),

  runFlow: (
    flowId: string,
    input: {
      input_value: string;
      session_id?: string;
      input_type?: string;
      output_type?: string;
      tweaks?: Record<string, any>;
    }
  ) => mcpApi.post(`/mcp/api/run-flow/${flowId}`, { input }),

  assistantChat: (flow_id: string, session_id: string, message: string) =>
    mcpApi.post("/mcp/api/assistant", { flow_id, session_id, message }),

  getFlowDetails: (flowId: string) =>
    mcpApi.get(`/mcp/api/flow-details/${flowId}`),

  // Component discovery
  searchComponents: (keyword: string) =>
    mcpApi.get("/mcp/api/search", { params: { keyword } }),

  getComponentDetails: (componentName: string) =>
    mcpApi.get(`/mcp/api/components/${componentName}`),

  getComponentEssentials: (componentName: string) =>
    mcpApi.get(`/mcp/api/component-essentials/${componentName}`),

  searchComponentProperties: (componentName: string, query: string) =>
    mcpApi.get(`/mcp/api/search-component-properties/${componentName}`, { 
      params: { query } 
    }),

  // Flow building
  buildFlow: (name: string, description: string, nodes: any[], connections: any[]) =>
    mcpApi.post("/mcp/api/build-flow", { name, description, nodes, connections }),

  createTestFlow: () =>
    mcpApi.post("/mcp/api/test-flow"),
};

