import { useEffect, useRef, useState } from "react";
import { getHealth, getStats, runQuery } from "./api";
import "./App.css";

const SAMPLE_QUERIES = [
  "Which family offices focus on technology or venture investments?",
  "List principals with a verified direct email address",
  "Which entities made a fund commitment recently?",
];

function StatusDot({ ok }) {
  return <span className={`status-dot ${ok ? "status-dot--ok" : "status-dot--off"}`} />;
}

function SourceLedger({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="ledger">
      <div className="ledger__label">grounded on {sources.length} record field{sources.length === 1 ? "" : "s"}</div>
      <div className="ledger__strip">
        {sources.map((s, i) => (
          <a
            key={i}
            className="ledger__tick"
            style={{ opacity: 0.45 + s.similarity * 0.55 }}
            href={s.source_url || undefined}
            target="_blank"
            rel="noreferrer"
            title={`${s.entity_name} - ${s.field_label} (similarity ${s.similarity})`}
          />
        ))}
      </div>
      <ul className="ledger__list">
        {sources.map((s, i) => (
          <li key={i} className="ledger__item">
            <span className="ledger__entity">{s.entity_name}</span>
            <span className="ledger__field">{s.field_label}</span>
            <span className="ledger__sim">{s.similarity.toFixed(2)}</span>
            {s.source_url ? (
              <a className="ledger__source" href={s.source_url} target="_blank" rel="noreferrer">
                source (link)
              </a>
            ) : (
              <span className="ledger__source ledger__source--none">no url on file</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [query, setQuery] = useState("");
  const [entityFilter, setEntityFilter] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const inputRef = useRef(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "unreachable" }));
    getStats().then(setStats).catch(() => setStats(null));
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    const q = query.trim();
    if (!q || loading) return;
    setLoading(true);
    setError(null);
    try {
      const data = await runQuery({
        query: q,
        entity_filter: entityFilter.trim() || null,
      });
      setResult(data);
      setHistory((h) => [{ query: q, grounded: data.grounded }, ...h].slice(0, 6));
    } catch (err) {
      setError(err.message || "The query could not be answered.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function useSample(s) {
    setQuery(s);
    inputRef.current?.focus();
  }

  const indexOk = health?.index_loaded;

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <span className="brand__mark">PIQ</span>
          <div className="brand__text">
            <span className="brand__title">Family Office Micro-RAG</span>
            <span className="brand__subtitle">grounded retrieval, not general knowledge</span>
          </div>
        </div>
        <div className="topbar__stats">
          <div className="stat">
            <StatusDot ok={indexOk} />
            <span>{health ? (indexOk ? "index loaded" : "no index") : "checking..."}</span>
          </div>
          {stats && (
            <>
              <div className="stat stat--mono">{stats.unique_entities} entities</div>
              <div className="stat stat--mono">{stats.total_chunks} chunks</div>
            </>
          )}
        </div>
      </header>

      <main className="console">
        <form className="query-box" onSubmit={handleSubmit}>
          <span className="query-box__prompt">&gt;</span>
          <input
            ref={inputRef}
            className="query-box__input"
            placeholder="Ask the dataset a question..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <input
            className="query-box__filter"
            placeholder="entity filter (optional)"
            value={entityFilter}
            onChange={(e) => setEntityFilter(e.target.value)}
          />
          <button className="query-box__submit" type="submit" disabled={loading || !query.trim()}>
            {loading ? "..." : "Run"}
          </button>
        </form>

        <div className="samples">
          {SAMPLE_QUERIES.map((s) => (
            <button key={s} type="button" className="samples__chip" onClick={() => useSample(s)}>
              {s}
            </button>
          ))}
        </div>

        {error && (
          <div className="panel panel--error">
            <div className="panel__label">query failed</div>
            <p>{error}</p>
          </div>
        )}

        {result && !error && (
          <div className={`panel ${result.grounded ? "panel--grounded" : "panel--ungrounded"}`}>
            <div className="panel__label">
              {result.grounded ? "answer, grounded in the dataset" : "no grounded match"}
            </div>
            <p className="panel__answer">{result.answer}</p>
            <SourceLedger sources={result.sources} />
          </div>
        )}

        {!result && !error && (
          <div className="panel panel--empty">
            <div className="panel__label">nothing run yet</div>
            <p>Ask a question above, or pick a sample. Every answer below will show exactly which record fields it was grounded on. If nothing in the dataset clears the similarity bar, you get an honest "no match," not a guess.</p>
          </div>
        )}

        {history.length > 0 && (
          <div className="history">
            <div className="history__label">recent</div>
            {history.map((h, i) => (
              <div key={i} className="history__item">
                <StatusDot ok={h.grounded} />
                <span>{h.query}</span>
              </div>
            ))}
          </div>
        )}
      </main>

      <footer className="footer">
        <span>PolarityIQ / Falcon Scaling - Stage 1 Differentiator Assessment</span>
      </footer>
    </div>
  );
}