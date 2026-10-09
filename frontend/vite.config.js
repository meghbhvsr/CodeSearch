import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In dev (`npm run dev`), Vite serves the UI on :5173 and forwards /api to the Python
// backend on :8000. In production, `npm run build` writes dist/, which FastAPI serves itself.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
