import { useContext } from "react";
import { Outlet } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { useCustomPostAuth } from "@/customization/hooks/use-custom-post-auth";
import { AuthContext } from "@/contexts/authContext";
import useAuthStore from "@/stores/authStore";
import { LoadingPage } from "@/pages/LoadingPage";

export function AppAuthenticatedPage() {
  useCustomPostAuth();
  const { userData } = useContext(AuthContext);
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  if (isAuthenticated && !userData) {
    return <LoadingPage overlay />;
  }

  // Onboarding flow removed: all authenticated users go straight to the main app.

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key="dashboard"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.25, ease: "easeInOut" }}
      >
        <Outlet />
      </motion.div>
    </AnimatePresence>
  );
}