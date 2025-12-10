export function useCustomApiHeaders() {
  // Attempt to resolve API key from multiple sources
  // Priority: window override -> localStorage -> Vite env
  const win = typeof window !== "undefined" ? (window as any) : undefined;
  const fromWindow = win?.__LANGFLOW_API_KEY__ as string | undefined;
  const fromStorage = win?.localStorage?.getItem("LANGFLOW_API_KEY") as string | null;
  // Vite exposes env at build time
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore - import.meta.env is provided by Vite
  const fromEnv = import.meta?.env?.VITE_LANGFLOW_API_KEY as string | undefined;

  const apiKey = fromWindow || fromStorage || fromEnv;

  const customHeaders: Record<string, string> = {};
  if (apiKey) {
    customHeaders["x-api-key"] = apiKey;
  }

  return customHeaders;
}
