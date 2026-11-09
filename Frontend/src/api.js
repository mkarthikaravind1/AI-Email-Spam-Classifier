import axios from "axios";

// Backend URL: set VITE_API_URL in Frontend/.env (see .env.example) to change it
export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
export const MAX_CHARS = 5000; // must match the backend's limit

export const api = axios.create({ baseURL: API_URL });

// Turns an axios error into a message a user can act on
export function errorMessage(error) {
  if (!error.response) {
    return `Can't reach the backend at ${API_URL}. Is it running?`;
  }
  const detail = error.response.data?.detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors (422): [{ msg: "Value error, ..." }, ...]
    return detail.map((d) => d.msg.replace(/^Value error, /, "")).join("; ");
  }
  return detail || `The backend returned an error (HTTP ${error.response.status}).`;
}

// One fixed colour per model, used in every card, chart and legend
// (categorical slots 1-5 of the validated reference palette, in order)
export const MODELS = [
  { key: "regex", name: "Regex Rules", detail: "Weighted patterns", color: "#2a78d6" },
  { key: "ml", name: "Statistical ML", detail: "Char n-gram Logistic Regression", color: "#eb6834" },
  { key: "distilbert", name: "DistilBERT", detail: "Fine-tuned transformer", color: "#1baf7a" },
  { key: "llm", name: "Zero-shot NLI", detail: "BART-large-MNLI", color: "#eda100", slow: true },
];
export const ENSEMBLE_COLOR = "#e87ba4";
export const MODEL_BY_KEY = Object.fromEntries(MODELS.map((m) => [m.key, m]));

export const pct = (value, digits = 1) =>
  value === null || value === undefined ? "-" : `${Number(value).toFixed(digits)}%`;
