import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

type Score = { label: string; score: number };
type Prediction = { predicted: string[]; scores: Score[]; score_kind: string; threshold: number };
type Run = { run_id: string; experiment: string; dataset: string; has_model: boolean };

function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState('');
  const [text, setText] = useState('I was nervous at first, but now I am excited and grateful for the opportunity.');
  const [result, setResult] = useState<Prediction | null>(null);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch('/api/runs').then(r => {
      if (!r.ok) throw new Error('API unavailable');
      return r.json();
    }).then(data => {
      const usable = (data.runs as Run[]).filter(run => run.has_model);
      setRuns(usable);
      setRunId(usable[0]?.run_id ?? '');
    }).catch(() => setStatus('尚未连接 API。请先启动后端服务。'));
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true); setStatus(''); setResult(null);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, run_id: runId || undefined }),
      });
      const json = await response.json();
      if (!response.ok) throw new Error(json.detail || 'Prediction failed');
      setResult(json.result as Prediction);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Unknown error');
    } finally { setBusy(false); }
  }

  return <main className="layout">
    <aside className="side">
      <div className="brand"><span className="brand-mark">◌</span> nlp-text-classification</div>
      <p className="side-label">RESEARCH WORKSPACE</p>
      <div className="nav active">01 / Emotion playground</div>
      <div className="nav">02 / Experiment runs</div>
      <p className="side-foot">GoEmotions · 28 labels<br />Multi-label classification</p>
    </aside>
    <section className="content">
      <div className="eyebrow">NATURAL LANGUAGE PROCESSING / 001</div>
      <h1>Explore emotions<br /><em>within language.</em></h1>
      <p className="intro">输入一段英文文本，查看训练模型识别出的情绪标签与分数。当前版本支持 E1 SVM；其他模型将在后续实验中接入。</p>
      <div className="workarea">
        <form onSubmit={submit}>
          <div className="toolbar"><label htmlFor="text">INPUT TEXT</label><span>English · up to 6,000 characters</span></div>
          <textarea id="text" maxLength={6000} value={text} onChange={event => setText(event.target.value)} required />
          <div className="controls">
            <select aria-label="选择模型" value={runId} onChange={event => setRunId(event.target.value)}>
              {runs.length ? runs.map(run => <option key={run.run_id} value={run.run_id}>{run.experiment} / {run.run_id}</option>) : <option value="">尚无训练结果</option>}
            </select>
            <button disabled={busy || !runId || !text.trim()}>{busy ? 'Evaluating...' : 'Analyze text ↗'}</button>
          </div>
        </form>
        <div className="results">
          <div className="toolbar"><span>MODEL OUTPUT</span><span>{result ? result.score_kind.toUpperCase() : 'AWAITING INFERENCE'}</span></div>
          {status && <p className="error">{status}</p>}
          {!result && !status && <div className="empty"><div className="empty-symbol">◎</div>训练 E1 后即可在此查看真实分类结果。</div>}
          {result && <>
            <p className="hint">Selected: {result.predicted.length ? result.predicted.join(' · ') : 'No label above threshold'}</p>
            <div className="scores">{result.scores.slice(0, 10).map(row => {
              const width = Math.max(0, Math.min(100, 50 + row.score * 25));
              return <div className="score-row" key={row.label}>
                <div className="score-top"><span>{row.label}</span><strong>{row.score.toFixed(3)}</strong></div>
                <div className="track"><div style={{ width: `${width}%` }} /></div>
              </div>;
            })}</div>
            <p className="caption">SVM decision margins (zero-centered visual scale), not probabilities. Decision threshold: {result.threshold.toFixed(3)}.</p>
          </>}
        </div>
      </div>
      <footer>Research prototype · Text-level emotion labeling is not a psychological diagnosis.</footer>
    </section>
  </main>;
}

createRoot(document.getElementById('root')!).render(<App />);
