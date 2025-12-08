import { Background, ReactFlow, useReactFlow, Panel } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo, useState, useRef } from "react";
import { motion, AnimatePresence, useInView } from "framer-motion";
import { MousePointer, ZoomIn, ZoomOut, Maximize, X } from "lucide-react";
import GenericNode from "@/CustomNodes/GenericNode";
import NoteNode from "@/CustomNodes/NoteNode";
import { PreviewEdge } from "./PreviewEdge";
import { scapedJSONStringfy } from "@/utils/reactflowUtils";
import type { FlowType } from "@/types/flow";
import type { AllNodeType, EdgeType } from "@/types/flow";

const nodeTypes = {
  genericNode: GenericNode,
  noteNode: NoteNode,
};

interface TemplateFlowPreviewProps {
  template: FlowType;
}

function AutoFitView() {
  const { fitView } = useReactFlow();
  
  useEffect(() => {
    // Fit view after a short delay to ensure nodes are rendered
    const timer = setTimeout(() => {
      fitView({ padding: 0.1, duration: 0 });
    }, 100);
    
    return () => clearTimeout(timer);
  }, [fitView]);
  
  return null;
}

function PreviewControls() {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  
  return (
    <Panel position="bottom-left" className="bg-white rounded-lg border border-gray-200 shadow-sm p-1 flex flex-col gap-1 mb-2 ml-2">
      <button 
        onClick={() => zoomIn()} 
        className="p-1.5 hover:bg-gray-50 rounded-md text-gray-600 hover:text-gray-900 transition-colors" 
        title="Zoom In"
      >
        <ZoomIn className="w-4 h-4" />
      </button>
      <button 
        onClick={() => zoomOut()} 
        className="p-1.5 hover:bg-gray-50 rounded-md text-gray-600 hover:text-gray-900 transition-colors" 
        title="Zoom Out"
      >
        <ZoomOut className="w-4 h-4" />
      </button>
      <button 
        onClick={() => fitView({ duration: 200 })} 
        className="p-1.5 hover:bg-gray-50 rounded-md text-gray-600 hover:text-gray-900 transition-colors" 
        title="Fit View"
      >
        <Maximize className="w-4 h-4" />
      </button>
    </Panel>
  );
}

export default function TemplateFlowPreview({ template }: TemplateFlowPreviewProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [hasInteracted, setHasInteracted] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef(null);
  const isInView = useInView(containerRef, { once: true, amount: 0.5 });

  useEffect(() => {
    // Start timer to show tooltip only when in view
    if (isInView && !hasInteracted) {
      timerRef.current = setTimeout(() => {
        setShowTooltip(true);
      }, 2000);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isInView, hasInteracted]);

  const handleInteraction = (event?: any) => {
    // Only hide if triggered by actual user event (not programmatic fitView)
    if (event && !hasInteracted) {
      setHasInteracted(true);
      setShowTooltip(false);
    }
  };

  const nodes = useMemo(() => {
    const templateNodes = (template.data?.nodes || []) as AllNodeType[];
    // Ensure all nodes have the correct type field for ReactFlow
    return templateNodes
      .map(node => {
        const nodeType = node.type as string | undefined;
        const dataType = node.data && 'type' in node.data ? (node.data as { type?: string }).type : undefined;
        const type = nodeType || dataType;
        // Keep noteNodes as is, convert others to genericNode
        if (type === "NoteNode" || type === "noteNode") {
          return {
            ...node,
            type: "noteNode",
            data: {
              ...node.data,
              id: node.id,
            }
          };
        }
        return {
          ...node,
          type: "genericNode", // Force genericNode to use the real component
          data: {
            ...node.data,
            id: node.id, // Ensure ID is passed to data for internal logic
          }
        };
      });
  }, [template.data?.nodes]);
  
  const edges = useMemo(() => {
    const templateEdges = (template.data?.edges || []) as EdgeType[];
    // Ensure edges have proper structure for ReactFlow
    return templateEdges.map(edge => {
      // Ensure handle IDs are properly formatted strings if they are missing
      let sourceHandle = edge.sourceHandle;
      let targetHandle = edge.targetHandle;
      
      if (!sourceHandle && edge.data?.sourceHandle) {
        sourceHandle = scapedJSONStringfy(edge.data.sourceHandle);
      }
      
      if (!targetHandle && edge.data?.targetHandle) {
        targetHandle = scapedJSONStringfy(edge.data.targetHandle);
      }

      return {
        ...edge,
        id: edge.id || `edge-${edge.source}-${edge.target}`,
        type: edge.type || "default",
        sourceHandle,
        targetHandle,
        style: { stroke: '#555555', strokeWidth: 2 },
        data: {
          ...edge.data,
        }
      };
    });
  }, [template.data?.edges]);
  
  if (!nodes.length) {
    return (
      <div className="w-full h-[400px] bg-gray-50 rounded-lg border border-gray-200 flex items-center justify-center">
        <p className="text-gray-500">No flow preview available</p>
      </div>
    );
  }
  
  return (
    <div 
      ref={containerRef}
      className="w-full h-[400px] bg-gray-50 rounded-lg border border-gray-200 overflow-hidden relative"
      onWheel={(e) => {
        // Allow page scrolling when zoomOnScroll is disabled
        // Find the nearest scrollable ancestor or use window
        let scrollableParent: HTMLElement | null = null;
        let parent = e.currentTarget.parentElement;
        
        while (parent && parent !== document.body) {
          const style = window.getComputedStyle(parent);
          const overflowY = style.overflowY || style.overflow;
          if (overflowY === 'auto' || overflowY === 'scroll') {
            scrollableParent = parent;
            break;
          }
          parent = parent.parentElement;
        }
        
        if (scrollableParent) {
          scrollableParent.scrollTop += e.deltaY;
        } else {
          window.scrollBy(0, e.deltaY);
        }
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2, duration: 0 }}
        minZoom={0.1}
        maxZoom={1}
        zoomOnScroll={false}
        zoomOnPinch={true}
        panOnDrag={true}
        onMoveStart={handleInteraction}
        onPaneClick={handleInteraction}
        onNodeClick={handleInteraction}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: true }}
        className="bg-gray-50"
      >
        <Background size={2} gap={20} className="" />
        <AutoFitView />
        <PreviewControls />
      </ReactFlow>
      
      {/* Interactive Badge & Tooltip Container */}
      <div className="absolute top-3 right-3 flex flex-col items-end gap-2 z-10">
        {/* Animated Interactive Badge */}
        <motion.div 
          className="bg-blue-500 text-white px-2 py-1 rounded-md text-xs font-medium shadow-sm flex items-center gap-1"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <motion.div
            animate={{ opacity: [1, 0.5, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <MousePointer className="w-3 h-3" />
          </motion.div>
          <span>Interactive</span>
        </motion.div>

        {/* Tooltip */}
        <AnimatePresence>
          {showTooltip && (
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="relative bg-yellow-50 border border-yellow-200 rounded-xl p-4 shadow-lg max-w-[320px]"
            >
              {/* Triangle Pointer */}
              <div className="absolute -top-1.5 right-6 w-3 h-3 bg-yellow-50 border-t border-l border-yellow-200 rotate-45 transform" />
              
              <div className="flex items-start gap-3 relative z-10">
                <div className="flex-1 space-y-1">
                  <h4 className="text-sm font-bold text-gray-900 flex items-center gap-1">
                    <span>💡</span>
                    <div className="flex">
                      {"Double-click to explore".split("").map((char, index) => (
                        <motion.span
                          key={index}
                          className="inline-block"
                          initial={{ y: 0 }}
                          animate={{ y: [0, -3, 0] }}
                          transition={{
                            duration: 0.4,
                            repeat: Infinity,
                            repeatDelay: 2,
                            delay: index * 0.03 + 0.5, // Initial delay + stagger
                            ease: "easeInOut"
                          }}
                        >
                          {char === " " ? "\u00A0" : char}
                        </motion.span>
                      ))}
                    </div>
                  </h4>
                  <p className="text-xs text-gray-600 leading-relaxed">
                    Read the yellow README notes for flow details. This is an example so changes can't be saved. 😊
                  </p>
                </div>
                <button 
                  onClick={(e) => {
                    e.stopPropagation();
                    setHasInteracted(true);
                    setShowTooltip(false);
                  }}
                  className="text-gray-400 hover:text-gray-600 transition-colors -mt-1 -mr-1"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

