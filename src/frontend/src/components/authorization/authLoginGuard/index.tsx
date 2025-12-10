import { type ReactNode } from "react";
import { CustomNavigate } from "@/customization/components/custom-navigate";
import useAuthStore from "@/stores/authStore";

export const ProtectedLoginRoute = ({ children }: { children: ReactNode }) => {
  const autoLogin = useAuthStore((state) => state.autoLogin);
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  if (autoLogin === true || isAuthenticated) {
    const urlParams = new URLSearchParams(window.location.search);
    const redirectPath = urlParams.get("redirect");

    if (redirectPath) {
      return <CustomNavigate to={redirectPath} replace />;
    }
    return <CustomNavigate to="/flows" replace />;
  }

  return children;
};
