async function request(path, options = {}) {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    // FastAPI sends {"detail": "..."} or, for validation errors, {"detail": [{msg}, ...]}
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return body;
}

export const listRepos = () => request("/repos");

export const addRepo = (url, refresh = false) =>
  request("/repos", { method: "POST", body: JSON.stringify({ url, refresh }) });

export const search = ({ repo, query, k, tests }) =>
  request("/search", { method: "POST", body: JSON.stringify({ repo, query, k, tests }) });
