import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { OnboardingFlow } from './onboardingflow';
import { cn } from "@/utils/utils";

type OnboardingWrapperProps = {
  onComplete: (selections: string[]) => void;
};

export function OnboardingWrapper({ onComplete }: OnboardingWrapperProps) {
  const [isExiting, setIsExiting] = useState(false);

  const handleComplete = (selections: string[]) => {
    // Start fade-out animation
    setIsExiting(true);
    
    // Wait for fade-out to complete before calling onComplete
    setTimeout(() => {
      onComplete(selections);
    }, 250); // Match the fade-out duration
  };

  return (
    <AnimatePresence>
      {!isExiting && (
        <motion.div
          key="onboarding-wrapper"
          initial={{ opacity: 1 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25, ease: 'easeInOut' }}
          className={cn(
            "flex h-screen w-screen items-center justify-center bg-background",
            "fixed left-0 top-0 z-[999]"
          )}
        >
          <OnboardingFlow onComplete={handleComplete} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}