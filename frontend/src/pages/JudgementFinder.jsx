import { useState } from 'react'
import { Search, Scale, FileText, Loader2, AlertCircle } from 'lucide-react'
import { api } from '../api/client'

function SimilarityBar({ score }) {
  const pct = Math.round(score * 100)
  const color = pct >= 80 ? 'bg-emerald-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-slate-500'
  return (
    <div className="flex items-center gap-2">
      <div className="score-bar-track flex-1">
        <div className={`score-bar-fill ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-slate-300 w-10 text-right">{pct}%</span>
    </div>
  )
}

const EXAMPLES = [
  "Fundamental right to speech was curtailed by government order without due process",
  "Land acquisition without fair compensation challenged under Article 300A",
  "Bail denied in NDPS case despite prolonged undertrial detention",
  "Termination of public servant without following natural justice principles",
]

export default function JudgementFinder() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expandedId, setExpandedId] = useState(null)
  const [yearFrom, setYearFrom] = useState('')
  const [yearTo, setYearTo] = useState('')
  const [actsCited, setActsCited] = useState('')

  const search = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const filters = {
        year_from: yearFrom ? Number(yearFrom) : null,
        year_to: yearTo ? Number(yearTo) : null,
        acts_cited: actsCited.trim() || null,
      }
      const data = await api.finderSearch(query, 5, filters)
      setResults(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => e.key === 'Enter' && !e.shiftKey && search()

  return (
    <div className="p-8 max-w-4xl">
      <h1 className="text-2xl font-bold text-white mb-1 flex items-center gap-2">
        <Search size={22} className="text-saffron-400" /> Judgement Finder
      </h1>
      <p className="text-slate-400 mb-6 text-sm">
        Describe your case or paste a snippet. The AI finds the most similar Supreme Court precedents using vector similarity.
      </p>

      {/* Search box */}
      <div className="card mb-4">
        <textarea
          className="input resize-none h-28 mb-3"
          placeholder="Paste your case snippet or describe the legal situation…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKey}
        />
        <div className="flex items-center justify-between">
          <div className="flex gap-2 flex-wrap">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                className="text-xs text-navy-400 hover:text-navy-200 bg-navy-900 hover:bg-navy-800 px-2 py-1 rounded transition-colors"
                onClick={() => setQuery(ex)}
              >
                {ex.slice(0, 42)}…
              </button>
            ))}
          </div>
          <button className="btn-primary flex items-center gap-2 ml-4 shrink-0" onClick={search} disabled={loading || !query.trim()}>
            {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
            {loading ? 'Searching…' : 'Find Cases'}
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="card mb-4 flex flex-wrap items-end gap-4">
        <div>
          <label className="block text-xs text-slate-400 mb-1">Court</label>
          <select className="input" disabled value="Supreme Court of India">
            <option>Supreme Court of India</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-slate-400 mb-1">Year from</label>
          <input
            type="number"
            className="input w-28"
            placeholder="1950"
            value={yearFrom}
            onChange={(e) => setYearFrom(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-xs text-slate-400 mb-1">Year to</label>
          <input
            type="number"
            className="input w-28"
            placeholder="2025"
            value={yearTo}
            onChange={(e) => setYearTo(e.target.value)}
          />
        </div>
        <div className="flex-1 min-w-48">
          <label className="block text-xs text-slate-400 mb-1">Acts cited contains</label>
          <input
            type="text"
            className="input"
            placeholder="e.g. Article 21"
            value={actsCited}
            onChange={(e) => setActsCited(e.target.value)}
          />
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 bg-red-950 border border-red-800 rounded-lg flex items-center gap-2 text-red-400 text-sm mb-4">
          <AlertCircle size={15} /> {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div>
          <p className="text-slate-400 text-sm mb-3">
            {results.results.length > 0
              ? `Found ${results.results.length} similar judgments`
              : results.message || 'No results found'}
          </p>
          <div className="space-y-3">
            {results.results.map((r, i) => (
              <div key={r.case_id} className="card">
                <div className="flex items-start justify-between gap-4 mb-2">
                  <div className="flex items-center gap-3">
                    <span className="text-saffron-400 font-bold text-lg">#{i + 1}</span>
                    <div>
                      <span className="font-mono text-xs text-slate-400">{r.ks_id}</span>
                      <p className="text-white font-medium text-sm">{r.subject || 'No subject'}</p>
                      <p className="text-slate-500 text-xs">{r.date || '—'}</p>
                    </div>
                  </div>
                  <div className="shrink-0 w-36">
                    <p className="text-xs text-slate-400 mb-1">Similarity</p>
                    <SimilarityBar score={r.similarity_score} />
                  </div>
                </div>

                {r.acts_cited && (
                  <p className="text-xs text-slate-400 mb-2">
                    <span className="text-slate-500">Acts: </span>{r.acts_cited}
                  </p>
                )}

                <p className="text-slate-300 text-sm leading-relaxed">
                  {r.summary || 'No summary available for this case.'}
                </p>

                {expandedId === r.case_id && (
                  <p className="text-slate-400 text-xs leading-relaxed mt-2 italic border-l-2 border-slate-700 pl-2">
                    {r.snippet}
                  </p>
                )}

                <div className="flex items-center gap-3 mt-3">
                  <button
                    className="text-xs text-navy-400 hover:text-navy-200 transition-colors"
                    onClick={() => setExpandedId(expandedId === r.case_id ? null : r.case_id)}
                  >
                    {expandedId === r.case_id ? 'Hide matching passage' : 'Show matching passage'}
                  </button>
                  {r.has_document && (
                    <a
                      href={api.downloadDocumentUrl(r.case_id)}
                      download
                      className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
                    >
                      <FileText size={12} /> Download Summary
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
