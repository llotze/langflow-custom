import { Panel } from "@xyflow/react";
import { memo, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useHotkeys } from "react-hotkeys-hook";
import { track } from "@/customization/utils/analytics";
import useFlowStore from "../../../stores/flowStore";
import { useShortcutsStore } from "../../../stores/shortcuts";
import { cn, isThereModal } from "../../../utils/utils";
import FlowToolbarOptions from "./components/flow-toolbar-options";

type FlowToolbarProps = {
  isChatOpen?: boolean;
  chatWidth?: number;
  toolbarGap?: number;
  isResizing?: boolean;
};

const FlowToolbar = memo(function FlowToolbar({
  isChatOpen,
  chatWidth,
  toolbarGap = 12,
  isResizing = false,
}: FlowToolbarProps): JSX.Element {
  const preventDefault = true;
  const [open, setOpen] = useState<boolean>(false);
  const [openApiModal, setOpenApiModal] = useState<boolean>(false);
  const [openShareModal, setOpenShareModal] = useState<boolean>(false);
  const handleAPIWShortcut = (e: KeyboardEvent) => {
    if (isThereModal() && !openApiModal) return;
    setOpenApiModal((oldOpen) => !oldOpen);
  };

  const handleChatWShortcut = (e: KeyboardEvent) => {
    if (isThereModal() && !open) return;
    if (useFlowStore.getState().hasIO) {
      setOpen((oldState) => !oldState);
    }
  };

  const handleShareWShortcut = (e: KeyboardEvent) => {
    if (isThereModal() && !openShareModal) return;
    setOpenShareModal((oldState) => !oldState);
  };

  const openPlayground = useShortcutsStore((state) => state.openPlayground);
  const api = useShortcutsStore((state) => state.api);
  const flow = useShortcutsStore((state) => state.flow);

  useHotkeys(openPlayground, handleChatWShortcut, { preventDefault });
  useHotkeys(api, handleAPIWShortcut, { preventDefault });
  useHotkeys(flow, handleShareWShortcut, { preventDefault });

  useEffect(() => {
    if (open) {
      track("Playground Button Clicked");
    }
  }, [open]);

  return (
    <>
      <Panel className="!top-auto !m-2" position="top-right">
        <motion.div
          animate={{
            x:
              isChatOpen && chatWidth
                ? -1 * (chatWidth + toolbarGap)
                : 0,
          }}
          transition={{
            duration: isResizing ? 0 : 0.5,
            ease: [0.4, 0, 0.2, 1],
          }}
        >
          <div
            className={cn(
              "hover:shadow-round-btn-shadow flex h-11 items-center justify-center gap-7 rounded-md border bg-background px-1.5 shadow transition-all",
            )}
          >
            <FlowToolbarOptions
              open={open}
              setOpen={setOpen}
              openApiModal={openApiModal}
              setOpenApiModal={setOpenApiModal}
            />
          </div>
        </motion.div>
      </Panel>
    </>
  );
});

export default FlowToolbar;
