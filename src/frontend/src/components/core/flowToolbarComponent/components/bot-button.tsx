import ShadTooltip from "@/components/common/shadTooltipComponent";
import BotIcon from "@/assets/CLOSEDBOT.png";

const BotButton = () => {
  return (
    <div className="relative flex items-center">
      <ShadTooltip content="AI Assistant Bot">
        <div
          className="playground-btn-flow-toolbar hover:bg-accent cursor-pointer"
          onClick={() => {
            // Functionality to be added later
            console.log("Bot button clicked");
          }}
        >
          <div className="h-4 w-4 flex items-center justify-center">
            <img
              src={BotIcon}
              alt="Bot"
              className="h-4 w-4 transition-all"
              style={{ transform: "scale(2.0)" }}
            />
          </div>
          <span className="hidden md:block"></span>
        </div>
      </ShadTooltip>
      {/* Animated, elevated speech bubble below with upward arrow and bounce */}
      <div
        className="absolute left-1/2 top-full mt-3 -translate-x-1/4 -ml-8 z-20 flex flex-col items-center animate-bounce-slow pointer-events-none select-none"
      >
        {/* Arrow below bubble, pointing up */}
        <svg width="32" height="18" viewBox="0 0 32 18" fill="none" className="-mt-1">
          <polygon points="16,0 28,16 4,16" fill="#f9fafb" className="dark:fill-zinc-800" />
          <polygon points="16,0 28,16 4,16" fill="none" stroke="#cbd5e1" strokeWidth="1" />
        </svg>
        {/* Bubble */}
        <div
          className="bg-white dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700 rounded-2xl px-4 py-2 text-xs shadow-xl text-gray-900 dark:text-gray-100 font-medium"
          style={{

            boxShadow: "0 4px 24px 0 rgba(0,0,0,0.10), 0 1.5px 6px 0 rgba(0,0,0,0.08)",
          }}
        >
          Use AI to help build!
        </div>
        <style>
          {`
            @keyframes bounce-slow {
              0%, 100% { transform: translateY(0); }
              20% { transform: translateY(-8px);}
              40% { transform: translateY(0);}
              60% { transform: translateY(-4px);}
              80% { transform: translateY(0);}
            }
            .animate-bounce-slow {
              animation: bounce-slow 2.2s infinite;
            }
          `}
        </style>
      </div>
    </div>
  );
};

export default BotButton;