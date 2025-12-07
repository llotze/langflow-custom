import { useContext, useState } from "react";
import { Outlet } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { OnboardingWrapper } from "@/components/onboarding/onboardingwrapper";
import { useCustomPostAuth } from "@/customization/hooks/use-custom-post-auth";
import { AuthContext } from "@/contexts/authContext";
import { useUpdateUser } from "@/controllers/API/queries/auth";
import { useGetUserData } from "@/controllers/API/queries/auth";
import useAuthStore from "@/stores/authStore";
import { LoadingPage } from "@/pages/LoadingPage";

export function AppAuthenticatedPage() {
  useCustomPostAuth();
  const { userData, getUser } = useContext(AuthContext);
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const [isOnboardingComplete, setIsOnboardingComplete] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const { mutate: updateUser } = useUpdateUser();
  const { mutate: mutateLoggedUser } = useGetUserData();

  // Only check for onboarding if:
  // 1. User is authenticated
  // 2. User data exists (user is logged in with valid session)
  const needsOnboarding = 
    isAuthenticated &&
    userData && 
    (userData.optins?.onboarding_completed === false || 
     userData.optins?.onboarding_completed === undefined);

  const handleOnboardingComplete = (selections: string[]) => {
    if (!userData?.id) {
      console.error("Cannot save onboarding: user data not available");
      return;
    }
    const onboardingPreferences = {
      question_1_role: selections[0] || "",
      question_2_experience: selections[1] || "",
      question_3_goal: selections[2] || "",
      question_4_wants_help: selections[3] || "",
    };

    const currentOptins = userData.optins ?? {};
    const updatedOptins = {
      ...currentOptins,
      onboarding_completed: true,
      onboarding_preferences: onboardingPreferences,
    };

    updateUser(
      {
        user_id: userData.id,
        user: {
          optins: updatedOptins,
        },
      },
      {
        onSuccess: () => {
          mutateLoggedUser(
            {},
            {
              onSuccess: () => {
                setIsOnboardingComplete(true);
                // Wait a moment for onboarding fade-out, then show dashboard with fade-in
                setTimeout(() => {
                  setShowDashboard(true);
                  getUser();
                }, 250); // Slightly after onboarding fade completes
              },
            },
          );
        },
        onError: (error) => {
          console.error("Failed to save onboarding preferences:", error);
        },
      },
    );
  };

  if (isAuthenticated && !userData) {
    return <LoadingPage overlay />;
  }

  // Show onboarding if needed
  if (needsOnboarding && !isOnboardingComplete) {
    return <OnboardingWrapper onComplete={handleOnboardingComplete} />;
  }

  // Show dashboard with fade-in animation
  return (
    <AnimatePresence mode="wait">
      {showDashboard && (
        <motion.div
          key="dashboard"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25, ease: 'easeInOut' }}
        >
          <Outlet />
        </motion.div>
      )}
      {!showDashboard && !needsOnboarding && (
        <motion.div
          key="dashboard-immediate"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.25, ease: 'easeInOut' }}
        >
          <Outlet />
        </motion.div>
      )}
    </AnimatePresence>
  );
}