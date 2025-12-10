import { useEffect, useRef } from 'react';

interface FlowUpdateEvent {
  type: 'flow_updated' | 'connected';
  flowId: string;
  nodes?: any[];
  edges?: any[];
  operationsApplied?: number;
  timestamp?: string;
}

interface UseFlowSSEOptions {
  flowId: string | undefined;
  enabled: boolean;
  onUpdate: (data: { nodes: any[]; edges: any[] }) => void;
}

/**
 * React hook for subscribing to real-time flow updates via SSE.
 * 
 * Automatically connects when enabled, reconnects on disconnect,
 * and cleans up on unmount.
 */
export function useFlowSSE({ flowId, enabled, onUpdate }: UseFlowSSEOptions) {
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!enabled || !flowId) {
      // Clean up if disabled or no flowId
      if (eventSourceRef.current) {
        console.log('Closing SSE connection (disabled or no flowId)');
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      return;
    }

    const connectSSE = () => {
      console.log(`Connecting to SSE for flow ${flowId}...`);
      
      const mcpServerUrl = import.meta.env.VITE_MCP_SERVER_URL || 'http://localhost:3001';
      const eventSource = new EventSource(`${mcpServerUrl}/mcp/api/flow-updates/${flowId}`);

      eventSource.onopen = () => {
        console.log(`SSE connected to flow ${flowId}`);
      };

      eventSource.onmessage = (event) => {
        try {
          const data: FlowUpdateEvent = JSON.parse(event.data);
          console.log('SSE event received:', data);

          if (data.type === 'flow_updated' && data.nodes && data.edges) {
            console.log(`Applying ${data.operationsApplied} operations to canvas`);
            onUpdate({ nodes: data.nodes, edges: data.edges });
          }
        } catch (err) {
          console.error('Failed to parse SSE event:', err);
        }
      };

      eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        eventSource.close();
        eventSourceRef.current = null;

        // Attempt reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting SSE reconnect...');
          connectSSE();
        }, 3000);
      };

      eventSourceRef.current = eventSource;
    };

    connectSSE();

    // Cleanup on unmount or dependency change
    return () => {
      console.log('Cleaning up SSE connection');
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, [flowId, enabled, onUpdate]);

  return {
    connected: eventSourceRef.current?.readyState === EventSource.OPEN,
  };
}