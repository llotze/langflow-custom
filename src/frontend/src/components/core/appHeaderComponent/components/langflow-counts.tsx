import { FaGithub } from "react-icons/fa";

import ShadTooltip from "@/components/common/shadTooltipComponent";
import { GITHUB_URL } from "@/constants/constants";
import FlowBuilderChat from "@/components/flowBuilderChatComponent";
import useFlowBuilderChat from "@/hooks/flows/use-flow-builder-chat";

export const LangflowCounts = () => {
  const { isOpen, closeChat } = useFlowBuilderChat();

  return (
    <>
      <FlowBuilderChat isOpen={isOpen} onClose={closeChat} />
      <div className="flex items-center">
        <ShadTooltip
          content="Go to GitHub repo"
          side="bottom"
          styleClasses="z-10"
        >
          <div
            onClick={() => window.open(GITHUB_URL, "_blank")}
            className="hit-area-hover flex items-center rounded-md p-1 text-muted-foreground"
          >
            <FaGithub className="h-4 w-4" />
          </div>
        </ShadTooltip>
      </div>
    </>
  );
};

export default LangflowCounts;
