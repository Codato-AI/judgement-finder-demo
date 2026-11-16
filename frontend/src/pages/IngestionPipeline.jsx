import { useState, useRef } from 'react'
import { Upload, FileText, Download, ChevronRight, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { api } from '../api/client'
import ReactMarkdown from 'react-markdown'

function StatusBadge({ status }) {
  const map = {
    ready: ['bg-slate-700 text-slate-300', 'Ready'],
    processing: ['bg-yellow-900 text-yellow-300', 'Processing…'],
    done: ['bg-emerald-900 text-emerald-300', 'Done'],
    error: ['bg-red-900 text-red-300', 'Error'],
  }
  const [cls, label] = map[status] || map.ready
  return <span className={`badge ${cls}`}>{label}</span>
}

export default function IngestionPipeline() {
  const [file, setFile] = useState(null)
  const [ingestStatus, setIngestStatus] = useState('ready')
  const [ingestResult, setIngestResult] = useState(null)
  const [cases, setCases] = useState([])
  const [selectedCase, setSelectedCase] = useState(null)
  const [docContent, setDocContent] = useState(null)
  const [loadingDoc, setLoadingDoc] = useState(false)
  const inputRef = useRef()

  const handleFile = (e) => setFile(e.target.files[0] || null)

  const handleIngest = async () => {
    if (!file) return
    setIngestStatus('processing')
    setIngestResult(null)
    try {
      const result = file.name.toLowerCase().endsWith('.pdf')
        ? await api.ingestPdf(file)
        : await api.bulkIngest(file)
      setIngestResult(result)
      setIngestStatus('done')
      loadCases()
    } catch (e) {
      setIngestResult({ error: e.message })
      setIngestStatus('error')
    }
  }

  const loadCases = async () => {
    try {
      const data = await api.getCases(0, 50)
      setCases(data)
    } catch (e) {
      console.error(e)
    }
  }

  const viewDocument = async (id) => {
    setSelectedCase(id)
    setLoadingDoc(true)
    setDocContent(null)
    try {
      const data = await api.viewDocument(id)
      setDocContent(data.content)
    } catch (e) {
      setDocContent(`Error: ${e.message}`)
    } finally {
      setLoadingDoc(false)
    }
  }

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold text-white mb-1">Ingestion Pipeline</h1>
      <p className="text-slate-400 mb-6 text-sm">Upload SC judgment CSV → PII scrubbing → document generation → vector indexing</p>

      {/* Upload */}
      <div className="card mb-6">
        <h2 className="font-semibold text-white mb-3 flex items-center gap-2"><Upload size={17} /> Upload Dataset</h2>
        <div
          className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center cursor-pointer hover:border-navy-500 transition-colors"
          onClick={() => inputRef.current?.click()}
        >
          <input ref={inputRef} type="file" accept=".csv,.pdf" className="hidden" onChange={handleFile} />
          {file ? (
            <p className="text-emerald-400 font-medium">{file.name} ({(file.size / 1024 / 1024).toFixed(1)} MB)</p>
          ) : (
            <>
              <Upload size={32} className="mx-auto text-slate-600 mb-2" />
              <p className="text-slate-400">Click to select a CSV or PDF file</p>
              <p className="text-slate-600 text-xs mt-1">SC judgments dataset (CSV) or a single case document (PDF)</p>
            </>
          )}
        </div>
        <div className="flex items-center gap-3 mt-4">
          <button
            className="btn-primary flex items-center gap-2"
            onClick={handleIngest}
            disabled={!file || ingestStatus === 'processing'}
          >
            {ingestStatus === 'processing' ? <Loader2 size={16} className="animate-spin" /> : <ChevronRight size={16} />}
            {ingestStatus === 'processing' ? 'Processing…' : 'Run Pipeline'}
          </button>
          <StatusBadge status={ingestStatus} />
        </div>

        {ingestResult && !ingestResult.error && (
          <div className="mt-4 p-4 bg-emerald-950 border border-emerald-800 rounded-lg">
            <div className="flex items-center gap-2 text-emerald-400 font-semibold mb-2">
              <CheckCircle size={16} /> Pipeline complete
            </div>
            {ingestResult.ingested !== undefined ? (
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div><p className="text-slate-400">Ingested</p><p className="text-white font-bold">{ingestResult.ingested}</p></div>
                <div><p className="text-slate-400">Test Set</p><p className="text-white font-bold">{ingestResult.test_set_size}</p></div>
                <div><p className="text-slate-400">Skipped</p><p className="text-white font-bold">{ingestResult.skipped}</p></div>
              </div>
            ) : (
              <div className="text-sm">
                <p className="text-white font-medium">{ingestResult.message}</p>
                {ingestResult.case_id && <p className="text-slate-400 mt-1">Case ID: {ingestResult.case_id}</p>}
              </div>
            )}
          </div>
        )}
        {ingestResult?.error && (
          <div className="mt-4 p-4 bg-red-950 border border-red-800 rounded-lg flex items-center gap-2 text-red-400 text-sm">
            <AlertCircle size={15} /> {ingestResult.error}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Case list */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-white">Ingested Cases</h2>
            <button className="btn-secondary text-xs py-1 px-3" onClick={loadCases}>Refresh</button>
          </div>
          {cases.length === 0 ? (
            <p className="text-slate-500 text-sm">No cases yet. Run the pipeline first.</p>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
              {cases.map((c) => (
                <div
                  key={c.id}
                  className={`p-3 rounded-lg cursor-pointer transition-colors border ${
                    selectedCase === c.id
                      ? 'border-navy-500 bg-navy-900'
                      : 'border-slate-800 hover:border-slate-700 bg-slate-800'
                  }`}
                  onClick={() => c.has_document && viewDocument(c.id)}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-slate-400">{c.ks_id}</span>
                    <div className="flex gap-1">
                      {c.has_document && <span className="badge bg-emerald-900 text-emerald-400">doc</span>}
                      {c.has_embedding && <span className="badge bg-violet-900 text-violet-400">indexed</span>}
                    </div>
                  </div>
                  <p className="text-white text-sm mt-1 truncate">{c.subject || 'No subject'}</p>
                  <p className="text-slate-500 text-xs">{c.date || '—'}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Document viewer */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-white flex items-center gap-2"><FileText size={16} /> Document Preview</h2>
            {selectedCase && (
              <a
                href={api.downloadDocumentUrl(selectedCase)}
                download
                className="btn-secondary text-xs py-1 px-3 flex items-center gap-1"
              >
                <Download size={13} /> Download
              </a>
            )}
          </div>
          {loadingDoc && <div className="flex items-center gap-2 text-slate-400"><Loader2 size={16} className="animate-spin" /> Loading document…</div>}
          {docContent && !loadingDoc && (
            <div className="prose prose-invert prose-sm max-w-none max-h-96 overflow-y-auto">
              <ReactMarkdown>{docContent}</ReactMarkdown>
            </div>
          )}
          {!docContent && !loadingDoc && (
            <p className="text-slate-500 text-sm">Select a case with a generated document to preview it here.</p>
          )}
        </div>
      </div>
    </div>
  )
}
