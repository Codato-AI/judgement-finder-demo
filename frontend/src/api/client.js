const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export const api = {
  getStats: () => request('/stats'),
  getCases: (skip = 0, limit = 20) => request(`/cases?skip=${skip}&limit=${limit}`),
  getCase: (id) => request(`/cases/${id}`),
  viewDocument: (id) => request(`/cases/${id}/document/view`),
  downloadDocumentUrl: (id) => `${BASE}/cases/${id}/document`,

  bulkIngest: (file) => {
    const form = new FormData()
    form.append('file', file)
    return fetch(`${BASE}/ingest/bulk`, { method: 'POST', body: form }).then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || 'Ingest failed')
      }
      return res.json()
    })
  },

  ingestPdf: (file) => {
    const form = new FormData()
    form.append('file', file)
    return fetch(`${BASE}/ingest/pdf`, { method: 'POST', body: form }).then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || 'PDF ingest failed')
      }
      return res.json()
    })
  },

  finderSearch: (query, top_k = 5) =>
    request('/finder/search', {
      method: 'POST',
      body: JSON.stringify({ query, top_k }),
    }),

  testModels: (case_id, model_ids = null) =>
    request('/models/test', {
      method: 'POST',
      body: JSON.stringify({ case_id, model_ids }),
    }),

  getModelResults: (case_id = null) =>
    request(`/models/results${case_id ? `?case_id=${case_id}` : ''}`),

  getAvailableModels: () => request('/models/available'),
}
