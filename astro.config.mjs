import { defineConfig } from "astro/config";

// Locally the site lives at "/". On GitHub Pages (https://<user>.github.io/<repo>/) the deploy
// workflow sets SITE and BASE_PATH, so every link and image gets the "/<repo>" prefix.
export default defineConfig({
  site: process.env.SITE || undefined,
  base: process.env.BASE_PATH || "/",
  // Local-only during development: listen on this machine only.
  server: { host: "127.0.0.1", port: 4321 },
  devToolbar: { enabled: false },
});
