// Prefix a site path with the deploy base ("/" locally, "/<repo>" on GitHub Pages).
const base = import.meta.env.BASE_URL.replace(/\/$/, "");
export const url = (p: string) => (p.startsWith("/") ? base + p : p);
