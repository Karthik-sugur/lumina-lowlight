// Build the static site into dist/ and inject the backend URL from the environment.
// Usage: LUMINA_API_BASE=https://<space>.hf.space node build.mjs   (Vercel runs this automatically)
// The API base is a public URL, never a secret; no tokens are used anywhere in the frontend.
import { cpSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";

const api = (process.env.LUMINA_API_BASE || "https://darshuuuu-lumina-lowlight.hf.space").trim().replace(/\/+$/, "");
if (!/^https?:\/\/[^\s"'<>\\]+$/.test(api)) throw new Error(`Invalid LUMINA_API_BASE: ${api}`);

mkdirSync("dist", { recursive: true });   // empty it rather than delete it (a dev server may be serving it)
for (const f of readdirSync("dist")) rmSync(`dist/${f}`, { recursive: true, force: true });
for (const f of ["index.html", "ui.js", "lumina.js", "benchmarks.json", "samples"]) cpSync(f, `dist/${f}`, { recursive: true });
writeFileSync("dist/config.js", `window.LUMINA_CONFIG = ${JSON.stringify({ apiBase: api })};\n`);
console.log(`built dist/ (apiBase = ${api})`);
