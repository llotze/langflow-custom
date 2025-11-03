import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

type OnboardingStepProps = {
  question: string;
  options: string[];
  onSelect: (option: string) => void;
};

export function OnboardingStep({
  question,
  options,
  onSelect,
}: OnboardingStepProps) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[500px] px-6">
      {/* Question */}
      <h1 className="mb-10 text-center w-full max-w-md text-2xl font-semibold text-primary">
        {question}
      </h1>

      {/* Options */}
      <div className="w-full max-w-md flex flex-col gap-3">
        {options.map((option, index) => (
          <motion.div
            key={option}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <Button
              variant="outline"
              size="lg"
              onClick={() => onSelect(option)}
              className="w-full justify-center py-6 text-base hover:bg-primary hover:text-primary-foreground transition-all duration-200"
            >
              {option}
            </Button>
          </motion.div>
        ))}
      </div>
    </div>
  );
}