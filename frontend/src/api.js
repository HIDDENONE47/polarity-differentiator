const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export function getHealth() {
  return request("/health");
}

export function getStats() {
  return request("/stats");
}

export function runQuery({ query, top_k = 8, entity_filter = null, field_filter = null }) {
  return request("/query", {
    method: "POST",
    body: JSON.stringify({ query, top_k, entity_filter, field_filter }),
  });
}
