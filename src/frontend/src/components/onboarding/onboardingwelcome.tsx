import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import ForwardedIconComponent from "@/components/common/genericIconComponent";

type OnboardingWelcomeProps = {
  onGetStarted: () => void;
};

export function OnboardingWelcome({ onGetStarted }: OnboardingWelcomeProps) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[500px] px-6">
      {/* Welcome heading */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-6 text-center"
      >
        <h1 className="text-4xl md:text-5xl font-semibold text-primary mb-2">
          Welcome to LangFlow
        </h1>
        <h2 className="text-4xl md:text-5xl font-semibold text-primary">
          with GracefulAI
        </h2>
      </motion.div>

      {/* Sub-heading */}
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="mb-12 text-xl text-foreground text-center max-w-md"
      >
        Begin Building AI solutions Now
      </motion.p>

      {/* Get Started button */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Button
          variant="default"
          size="lg"
          onClick={onGetStarted}
          className="px-8 py-6 text-base font-semibold gap-3 bg-foreground hover:bg-foreground/90 text-background"
        >
          Get Started
          <ForwardedIconComponent
            name="ArrowRight"
            className="h-5 w-5"
          />
        </Button>
      </motion.div>
    </div>
  );
}