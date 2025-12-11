import axios from "axios";

const MCP_SERVER_URL = import.meta.env.VITE_MCP_SERVER_URL || "http://localhost:3001";

/**
 * Retrieves user's API key from localStorage.
 */
function getUserApiKey(): string | null {
  return localStorage.getItem('langflow_mcp_api_key');
}

/**
 * Stores user's API key in localStorage.
 */
export function setUserApiKey(apiKey: string): void {
  localStorage.setItem('langflow_mcp_api_key', apiKey);
}

/**
 * Removes user's API key from localStorage.
 */
export function clearUserApiKey(): void {
  localStorage.removeItem('langflow_mcp_api_key');
}

/**
 * Checks if user has a valid API key stored.
 */
export function hasUserApiKey(): boolean {
  const key = getUserApiKey();
  return !!key && key.trim().length > 0;
}

/**
 * Axios instance for MCP server communication.
 */
export const mcpApi = axios.create({
  baseURL: MCP_SERVER_URL,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true,
});

/**
 * Adds user's API key to all outgoing requests.
 */
mcpApi.interceptors.request.use(
  (config) => {
    const apiKey = getUserApiKey();
    
    if (apiKey) {
      config.headers['x-api-key'] = apiKey;
    }
    
    return config;
  },
  (error) => Promise.reject(error)
);

/**
 * Handles authentication errors from the server.
 */
mcpApi.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 || error.response?.status === 403) {
      console.error('MCP authentication failed - invalid or missing API key');
    }
    return Promise.reject(error);
  }
);

/**
 * Helper functions for MCP API operations.
 */
export const mcpApiHelpers = {
  healthCheck: () => mcpApi.get("/health"),

  searchTemplates: (keyword?: string, tags?: string, page?: number, pageSize?: number) =>
    mcpApi.get("/mcp/api/search-templates", { 
      params: { keyword, tags, page, pageSize } 
    }),

  getTemplate: (templateId: string) =>
    mcpApi.get(`/mcp/api/get-template/${templateId}`),

  createFlowFromTemplate: (templateId: string, name?: string, description?: string) =>
    mcpApi.post(`/mcp/api/create-flow-from-template/${templateId}`, { name, description }),

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

  buildFlow: (name: string, description: string, nodes: any[], connections: any[]) =>
    mcpApi.post("/mcp/api/build-flow", { name, description, nodes, connections }),

  createTestFlow: () =>
    mcpApi.post("/mcp/api/test-flow"),
};

