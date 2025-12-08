import { Background, Panel, useReactFlow } from "@xyflow/react";
import { memo, useState, useEffect } from "react";
import { ZoomIn, ZoomOut, Maximize } from "lucide-react";
import ForwardedIconComponent from "@/components/common/genericIconComponent";
import ShadTooltip from "@/components/common/shadTooltipComponent";
import CanvasControlButton from "@/components/core/canvasControlsComponent/CanvasControlButton";
import LogCanvasControls from "@/components/core/logCanvasControlsComponent";
import { SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import { ENABLE_NEW_SIDEBAR } from "@/customization/feature-flags";
import { cn } from "@/utils/utils";
import { useSearchContext } from "../flowSidebarComponent";
import { NAV_ITEMS } from "../flowSidebarComponent/components/sidebarSegmentedNav";

export const MemoizedBackground = memo(() => (
  <Background size={2} gap={20} className="" />
));

export const MemoizedLogCanvasControls = memo(() => <LogCanvasControls />);

export const MemoizedFlowZoomControls = memo(() => {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const [isAddNoteActive, setIsAddNoteActive] = useState(false);

  useEffect(() => {
    const handleEndAddNote = () => setIsAddNoteActive(false);
    window.addEventListener("lf:end-add-note", handleEndAddNote);
    return () => window.removeEventListener("lf:end-add-note", handleEndAddNote);
  }, []);

  const handleAddNote = () => {
    window.dispatchEvent(new Event("lf:start-add-note"));
    setIsAddNoteActive(true);
  };

  return (
    <Panel
      position="top-left"
      className="bg-white rounded-lg border border-gray-200 shadow-sm p-1 flex flex-col gap-1 !m-2"
    >
      <ShadTooltip content="Zoom In" side="right">
        <button 
          onClick={() => zoomIn()} 
          className={cn(
            "p-1.5 rounded-md transition-all duration-200",
            "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
          )}
        >
          <ZoomIn className="w-4 h-4" />
        </button>
      </ShadTooltip>
      <ShadTooltip content="Zoom Out" side="right">
        <button 
          onClick={() => zoomOut()} 
          className={cn(
            "p-1.5 rounded-md transition-all duration-200",
            "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
          )}
        >
          <ZoomOut className="w-4 h-4" />
        </button>
      </ShadTooltip>
      <ShadTooltip content="Fit View" side="right">
        <button 
          onClick={() => fitView({ duration: 200 })} 
          className={cn(
            "p-1.5 rounded-md transition-all duration-200",
            "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
          )}
        >
          <Maximize className="w-4 h-4" />
        </button>
      </ShadTooltip>
      <ShadTooltip content="Add Sticky Notes" side="right">
        <button
          onClick={handleAddNote}
          className={cn(
            "p-1.5 rounded-md transition-all duration-200",
            isAddNoteActive
              ? "bg-accent text-accent-foreground"
              : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
          )}
        >
          <ForwardedIconComponent name="sticky-note" className="w-4 h-4" />
        </button>
      </ShadTooltip>
    </Panel>
  );
});

export const MemoizedSidebarTrigger = memo(() => {
  const { open, toggleSidebar, setActiveSection } = useSidebar();
  const { focusSearch } = useSearchContext();
  if (ENABLE_NEW_SIDEBAR) {
    return (
      <Panel
        className={cn(
          "react-flow__controls !top-auto !m-2 flex gap-1.5 rounded-md border border-secondary-hover bg-background p-0.5 text-primary shadow transition-all duration-300 [&>button]:border-0 [&>button]:bg-background hover:[&>button]:bg-accent",
          "pointer-events-auto opacity-100 group-data-[open=true]/sidebar-wrapper:pointer-events-none group-data-[open=true]/sidebar-wrapper:-translate-x-full group-data-[open=true]/sidebar-wrapper:opacity-0",
        )}
        position="top-left"
      >
        {NAV_ITEMS.map((item) => (
          <CanvasControlButton
            data-testid={`sidebar-trigger-${item.id}`}
            iconName={item.icon}
            iconClasses={item.id === "mcp" ? "h-8 w-8" : ""}
            tooltipText={item.tooltip}
            onClick={() => {
              setActiveSection(item.id);
              if (!open) {
                toggleSidebar();
              }
              if (item.id === "search") {
                // Add a small delay to ensure the sidebar is open and input is rendered
                setTimeout(() => focusSearch(), 100);
              }
            }}
            testId={item.id}
          />
        ))}
      </Panel>
    );
  }

  return (
    <Panel
      className={cn(
        "react-flow__controls !top-auto !m-2 flex gap-1.5 rounded-md border border-secondary-hover bg-background p-1.5 text-primary shadow transition-all duration-300 [&>button]:border-0 [&>button]:bg-background hover:[&>button]:bg-accent",
        "pointer-events-auto opacity-100 group-data-[open=true]/sidebar-wrapper:pointer-events-none group-data-[open=true]/sidebar-wrapper:-translate-x-full group-data-[open=true]/sidebar-wrapper:opacity-0",
      )}
      position="top-left"
    >
      <SidebarTrigger className="h-fit w-fit px-3 py-1.5">
        <ForwardedIconComponent name="PanelRightClose" className="h-4 w-4" />
        <span className="text-foreground">Components</span>
      </SidebarTrigger>
    </Panel>
  );
});
