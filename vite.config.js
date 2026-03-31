import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages project sites live at https://<user>.github.io/<repo>/.
// CI sets BASE_PATH (e.g. /Diffusions-LLM-DLLM-/); local dev omits it → "/".
const base = process.env.BASE_PATH || "/";

export default defineConfig({
  base: base.endsWith("/") ? base : `${base}/`,
  plugins: [react()],
});
