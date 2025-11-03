import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { OnboardingStep } from './onboardingstep';
import { OnboardingWelcome } from './onboardingwelcome';
import { cn } from "@/utils/utils";

type StepData = {
  question: string;
  options: string[];
};

const steps: StepData[] = [
  {
    question: 'What best describes you?',
    options: [
      'Student or Researcher',
      'Instructor or Teacher',
      'Developer or Engineer',
      'Entrepreneur or Business Owner',
      "Hobbyist / Curious Explorer",
      'Other',
    ],
  },
  {
    question: 'How would you rate your technical or coding experience?',
    options: ["None (I don't code)", 'Beginner', 'Intermediate', 'Advanced / Professional'],
  },
  {
    question: 'What brings you to Graceful (Langflow) today?',
    options: [
      "I have an idea or workflow I want to build",
      "I want to explore what AI can do",
      "I want to learn how AI tools work",
      "I want to automate or simplify a process",
      "Other",
    ],
  },
  {
    question: "Would you like help brainstorming what you can build with Graceful (Langflow)?",
    options: [
      "Yes — I'd love some help exploring ideas",
      "No — I already know what I want to build",
    ],
  }
];

type OnboardingFlowProps = {
  onComplete: (selections: string[]) => void;
};

export function OnboardingFlow({ onComplete }: OnboardingFlowProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [selections, setSelections] = useState<string[]>([]);
  const [showWelcome, setShowWelcome] = useState(false);

  const handleSelection = (option: string) => {
    const newSelections = [...selections, option];

    if (currentStep < steps.length - 1) {
      setSelections(newSelections);
      setTimeout(() => {
        setCurrentStep(currentStep + 1);
      }, 300);
    } else {
      // Last question answered - show welcome screen
      setSelections(newSelections);
      setTimeout(() => {
        setShowWelcome(true);
      }, 300);
    }
  };

  const handleGetStarted = () => {
    setTimeout(() => {
      // Complete onboarding with all selections
      onComplete(selections);
    }, 100);
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Progress bars - static, outside animated content */}
      {!showWelcome && (
        <div className="mb-8 flex gap-2 justify-center">
          {Array.from({ length: steps.length }).map((_, index) => (
            <div
              key={index}
              className={cn(
                "h-2 w-8 rounded-full transition-colors duration-300",
                index < currentStep + 1
                  ? "bg-primary"
                  : "bg-muted"
              )}
            />
          ))}
        </div>
      )}

      {/* Animated content - question/options or welcome screen */}
      <AnimatePresence mode="wait">
        {showWelcome ? (
          <motion.div
            key="welcome"
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.4, ease: 'easeInOut' }}
          >
            <OnboardingWelcome
              onGetStarted={handleGetStarted}
            />
          </motion.div>
        ) : (
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.4, ease: 'easeInOut' }}
          >
            <OnboardingStep
              question={steps[currentStep].question}
              options={steps[currentStep].options}
              onSelect={handleSelection}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}