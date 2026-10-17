import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Bell, Brain, Check, Clock3, Play, RotateCcw, ShieldCheck, Sparkles, UserRound, Zap } from "lucide-react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import "./styles.css";

const API = "http://localhost:8000/api";
const defaults = {
  sleep_hours: 7, sleep_quality_1_5: 3, pre_fatigue_1_5: 3, pre_stress_1_5: 3,
  pre_focus_1_5: 4, task_familiarity_1_5: 4, expected_difficulty_1_5: 3,
  baseline_typing_wpm: 65, baseline_error_rate: 0.03, baseline_pause_seconds: 0.2,
  interruption_urgency_1_5: 2, interruption_relevance_1_5: 3,
  interruption_cognitive_demand_1_5: 3, emotional_disturbance_1_5: 3,
  task_type: "Programming", environment: "Multiple activities", deadline_pressure: "Low",
  interruption_type: "Phone call", interruption_expectedness: "Unexpected",
  condition: "Phone call",
};
const categoryContext = {
  "Phone call": [3, 3, 3], Message: [2, 3, 2], WhatsApp: [2, 3, 2],
  Instagram: [3, 1, 3], YouTube: [3, 1, 4], Email: [2, 2, 2], "Other app": [3, 2, 3],
};

function App() {
  const [user, setUser] = useState(null);
  const [name, setName] = useState("");
  const [form, setForm] = useState(defaults);
  const [baseline, setBaseline] = useState({ wpm: 65, error_rate: 0.03, accuracy: 0.97, task_type: "Programming" });
  const [category, setCategory] = useState("Phone call");
  const [event, setEvent] = useState(null);
  const [history, setHistory] = useState([]);
  const [seconds, setSeconds] = useState(0);
  const [running, setRunning] = useState(false);
  const [outcome, setOutcome] = useState({ actual_duration_seconds: "", returned_to_task: true, post_wpm: "", post_error_rate: "" });
  const [tab, setTab] = useState("overview");
  const [notifications, setNotifications] = useState(true);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = async (id = user?.user_id) => {
    if (!id) return;
    const response = await fetch(`${API}/users/${id}/history`);
    if (response.ok) setHistory(await response.json());
  };
  useEffect(() => { if (user) refresh(); }, [user]);
  useEffect(() => {
    if (!running || seconds <= 0) return undefined;
    const timer = window.setInterval(() => setSeconds((value) => {
      if (value <= 1) {
        window.clearInterval(timer); setRunning(false);
        if (notifications && "Notification" in window && Notification.permission === "granted")
          new Notification("Recovery time is complete", { body: "Your recovery time is done. You can start working again." });
        return 0;
      }
      return value - 1;
    }), 1000);
    return () => window.clearInterval(timer);
  }, [running, seconds, notifications]);

  const chart = useMemo(() => history.filter((item) => item.post_wpm).slice().reverse().map((item, i) => ({
    name: `Event ${i + 1}`, wpm: item.post_wpm, error: item.post_error_rate ? item.post_error_rate * 100 : 0,
  })), [history]);
  const set = (key, value) => setForm((current) => ({ ...current, [key]: value }));
  const format = (value) => `${Math.floor(value / 60).toString().padStart(2, "0")}:${(value % 60).toString().padStart(2, "0")}`;

  async function createProfile(e) {
    e.preventDefault(); setBusy(true); setError("");
    try {
      const response = await fetch(`${API}/users`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ display_name: name }) });
      if (!response.ok) throw new Error("Could not create profile.");
      setUser(await response.json());
    } catch (reason) { setError(reason.message); } finally { setBusy(false); }
  }
  async function saveBaseline(e) {
    e.preventDefault(); setBusy(true); setError("");
    try {
      const response = await fetch(`${API}/users/${user.user_id}/baseline`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(baseline) });
      if (!response.ok) throw new Error("Could not save baseline.");
      setForm((current) => ({ ...current, baseline_typing_wpm: baseline.wpm, baseline_error_rate: baseline.error_rate }));
      setTab("overview");
    } catch (reason) { setError(reason.message); } finally { setBusy(false); }
  }
  async function simulate() {
    setBusy(true); setError("");
    try {
      const response = await fetch(`${API}/events/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, user_id: user.user_id, source: "simulator", category }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed.");
      setEvent(data); setSeconds(Math.round(data.duration_seconds)); setRunning(true);
      if (notifications && "Notification" in window && Notification.permission === "default") await Notification.requestPermission();
      await refresh();
    } catch (reason) { setError(reason.message); } finally { setBusy(false); }
  }
  async function complete() {
    if (!event) return;
    setBusy(true);
    try {
      const response = await fetch(`${API}/events/${event.event_id}/complete`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...outcome, actual_duration_seconds: Number(outcome.actual_duration_seconds), post_wpm: outcome.post_wpm ? Number(outcome.post_wpm) : null, post_error_rate: outcome.post_error_rate ? Number(outcome.post_error_rate) : null }) });
      if (!response.ok) throw new Error("Could not record outcome.");
      setEvent(null); setRunning(false); setSeconds(0); setOutcome({ actual_duration_seconds: "", returned_to_task: true, post_wpm: "", post_error_rate: "" }); await refresh();
    } catch (reason) { setError(reason.message); } finally { setBusy(false); }
  }
  async function deleteData() {
    if (!window.confirm("Delete this profile and all stored events?")) return;
    await fetch(`${API}/users/${user.user_id}/data`, { method: "DELETE" }); setUser(null); setName(""); setHistory([]); setEvent(null);
  }
  function chooseCategory(value) {
    const [urgency, relevance, demand] = categoryContext[value];
    setCategory(value); setForm((current) => ({ ...current, interruption_type: value, condition: value, interruption_urgency_1_5: urgency, interruption_relevance_1_5: relevance, interruption_cognitive_demand_1_5: demand }));
  }

  if (!user) return <div className="onboarding"><div className="onboard-card"><div className="brand"><span className="brand-mark"><Sparkles size={18} /></span><span>focusline</span></div><p className="eyebrow">PERSONALIZED INTERRUPTION COMPANION</p><h1>Learn your interruptions.<br /><em>Protect your focus.</em></h1><p>Create a local profile to store your baseline and measured interruption outcomes. Nothing is sent outside this app.</p><form onSubmit={createProfile}><label>Your name<input required value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Alex" /></label><button className="primary-button" disabled={busy}>Create private profile <span>→</span></button></form><div className="privacy-line"><ShieldCheck size={15} /> Simulator only · browser notification access is optional</div>{error && <div className="error-banner">{error}</div>}</div></div>;

  return <div className="app-shell">
    <aside className="sidebar"><div className="brand"><span className="brand-mark"><Sparkles size={18} /></span><span>focusline</span></div><div className="sidebar-copy"><p className="eyebrow">PERSONALIZED COMPANION</p><h1>Protect your<br /><em>deep work.</em></h1><p>Learned from your own interruption history, one measured event at a time.</p></div><div className="sidebar-note"><Brain size={18} /><span>Simulated events are clearly marked. Real notification access needs a separate extension or mobile companion.</span></div><div className="sidebar-footer">USER · {user.display_name.toUpperCase()} <span>v2.0</span></div></aside>
    <main className="main-content"><header className="topbar"><div><p className="eyebrow">FOCUS SESSION · RESEARCH MODE</p><h2>Hello, {user.display_name}.</h2></div><div className="top-actions"><button className="icon-button" title="Toggle notifications" onClick={() => setNotifications(!notifications)}><Bell size={18} className={notifications ? "active-icon" : ""} /></button><div className="avatar"><UserRound size={16} /></div></div></header>
      {error && <div className="error-banner">{error}</div>}
      <nav className="app-nav">{[["overview","Overview"],["simulate","Test event"],["baseline","Typing baseline"],["history","History & outcomes"]].map(([key, label]) => <button key={key} className={tab === key ? "active" : ""} onClick={() => setTab(key)}>{label}</button>)}</nav>
      {tab === "overview" && <><section className="hero-grid"><div className="panel"><div className="panel-heading"><div><p className="eyebrow">LIVE INTERRUPTION PANEL</p><h3>{event ? "Prediction ready" : "No active interruption"}</h3></div><span className="status-pill"><span className="status-dot" /> {event ? event.prediction_kind.replaceAll("_", " ") : "Ready"}</span></div>{event ? <div className="active-event"><div className="event-meta"><strong>{category}</strong><span>Simulator event · {event.historical_sample_count} matching observations</span></div><div className="big-duration">{format(seconds)}<small>{running ? "remaining" : "predicted duration"}</small></div><div className="timer-actions"><button className="timer-button" onClick={() => setRunning(!running)}><Play size={16} fill="currentColor" /> {running ? "Pause timer" : "Start predicted timer"}</button><button className="secondary-button" onClick={() => { setSeconds(Math.round(event.duration_seconds)); setRunning(false); }}><RotateCcw size={16} /></button></div><div className="prediction-mini"><span>Expected speed <b>{event.predicted_wpm.toFixed(1)} WPM</b></span><span>Expected errors <b>{(event.predicted_error_rate * 100).toFixed(2)}%</b></span></div></div> : <div className="empty-state"><Clock3 size={28} /><p>Use <b>Test event</b> to simulate a notification and get a personalized estimate.</p></div>}</div><div className="panel countdown-panel"><div className="panel-heading"><div><p className="eyebrow">PERSONALIZATION STATUS</p><h3>Your learned profile</h3></div><Zap size={18} className="muted-icon" /></div><div className="profile-stat"><strong>{history.length}</strong><span>events recorded</span></div><div className="profile-stat"><strong>{history.filter((item) => item.actual_duration).length}</strong><span>measured outcomes</span></div><div className="research-note"><Sparkles size={16} /><span>After two measured events in a category, its history influences future predictions.</span></div></div></section><section className="metric-grid"><div className="metric-card"><div className="metric-icon indigo"><ActivityIcon /></div><p>Baseline typing speed</p><strong>{baseline.wpm} <small>WPM</small></strong><span className="metric-foot"><Check size={13} /> Your current profile</span></div><div className="metric-card"><div className="metric-icon coral"><Zap size={18} /></div><p>Baseline error rate</p><strong>{(baseline.error_rate * 100).toFixed(2)} <small>%</small></strong><span className="metric-foot"><Check size={13} /> Update as observations grow</span></div><div className="metric-card"><div className="metric-icon mint"><ShieldCheck size={18} /></div><p>Data boundary</p><strong>Local <small>only</small></strong><span className="metric-foot"><Check size={13} /> No message contents stored</span></div></section></>}
      {tab === "simulate" && <section className="panel form-panel"><div className="panel-heading"><div><p className="eyebrow">TESTING MODE · SIMULATOR</p><h3>Simulate an incoming notification</h3></div><span className="status-pill">Clearly labelled</span></div><p className="helper">A normal website cannot read notifications from WhatsApp, Android, Windows, or other apps. This event is a safe manual simulator; it stores only the category, not message contents. The timer starts automatically when you trigger the prediction.</p><div className="form-grid"><label>Application / category<select value={category} onChange={(e) => chooseCategory(e.target.value)}>{Object.keys(categoryContext).map((item) => <option key={item}>{item}</option>)}</select></label><label>Task type<select value={form.task_type} onChange={(e) => set("task_type", e.target.value)}>{["Programming","Writing"].map((item) => <option key={item}>{item}</option>)}</select></label><label>Focus right now<select value={form.pre_focus_1_5} onChange={(e) => set("pre_focus_1_5", Number(e.target.value))}>{[1,2,3,4,5].map((item) => <option key={item} value={item}>{item} / 5</option>)}</select></label><label>Task difficulty<select value={form.expected_difficulty_1_5} onChange={(e) => set("expected_difficulty_1_5", Number(e.target.value))}>{[1,2,3,4,5].map((item) => <option key={item} value={item}>{item} / 5</option>)}</select></label></div><button className="primary-button" onClick={simulate} disabled={busy}>Predict and start timer <span>→</span></button></section>}
      {tab === "baseline" && <section className="panel form-panel"><div className="panel-heading"><div><p className="eyebrow">STEP 1 · ESTABLISH YOUR BASELINE</p><h3>Typing assessment</h3></div><span className="status-pill"><UserRound size={13} /> {user.display_name}</span></div><p className="helper">Enter a short, consent-based typing assessment result. In a production study, this would be collected by the typing test rather than typed manually.</p><form onSubmit={saveBaseline}><div className="form-grid"><label>Typing speed<div className="input-with-unit"><input type="number" required value={baseline.wpm} onChange={(e) => setBaseline({ ...baseline, wpm: Number(e.target.value) })} /><span>WPM</span></div></label><label>Error rate<div className="input-with-unit"><input type="number" step="0.001" required value={baseline.error_rate} onChange={(e) => setBaseline({ ...baseline, error_rate: Number(e.target.value) })} /><span>rate</span></div></label><label>Accuracy<div className="input-with-unit"><input type="number" step="0.001" required value={baseline.accuracy} onChange={(e) => setBaseline({ ...baseline, accuracy: Number(e.target.value) })} /><span>rate</span></div></label><label>Task type<select value={baseline.task_type} onChange={(e) => setBaseline({ ...baseline, task_type: e.target.value })}>{["Programming","Writing"].map((item) => <option key={item}>{item}</option>)}</select></label></div><button className="primary-button" disabled={busy}>Save baseline <span>→</span></button></form></section>}
      {tab === "history" && <section className="panel"><div className="panel-heading"><div><p className="eyebrow">MEASURED OUTCOMES</p><h3>Prediction history</h3></div><button className="danger-button" onClick={deleteData}>Delete my data</button></div><p className="helper">Predicted duration becomes a measurable outcome only after you record what happened. This is the data used for personalization.</p>{chart.length > 1 && <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chart}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8eaf2" /><XAxis dataKey="name" axisLine={false} tickLine={false} /><YAxis axisLine={false} tickLine={false} /><Tooltip /><Area type="monotone" dataKey="wpm" stroke="#686af3" fill="#686af322" /></AreaChart></ResponsiveContainer></div>}<div className="history-list">{history.length ? history.map((item) => <div className="history-row" key={item.event_id}><div><strong>{item.category}</strong><span>{new Date(item.created_at).toLocaleString()} · {item.prediction_kind.replaceAll("_", " ")}</span></div><div><b>{Math.round(item.predicted_duration)}s predicted</b><span>{item.actual_duration ? `${Math.round(item.actual_duration)}s actual` : "Outcome pending"}</span></div></div>) : <div className="empty-state">No events yet. Run the simulator to start learning.</div>}</div></section>}
      {event && tab !== "simulate" && <section className="panel outcome-panel"><div><p className="eyebrow">AFTER THE EVENT</p><h3>Record what actually happened</h3><p className="helper">This is collected after return, not used as an input to the original prediction.</p></div><div className="form-grid"><label>Actual time away<div className="input-with-unit"><input type="number" min="1" value={outcome.actual_duration_seconds} onChange={(e) => setOutcome({ ...outcome, actual_duration_seconds: e.target.value })} placeholder="seconds" /><span>sec</span></div></label><label>Post-event WPM<div className="input-with-unit"><input type="number" min="1" value={outcome.post_wpm} onChange={(e) => setOutcome({ ...outcome, post_wpm: e.target.value })} placeholder="optional" /><span>WPM</span></div></label><label>Post-event error rate<div className="input-with-unit"><input type="number" step="0.001" value={outcome.post_error_rate} onChange={(e) => setOutcome({ ...outcome, post_error_rate: e.target.value })} placeholder="optional" /><span>rate</span></div></label></div><button className="primary-button" onClick={complete} disabled={busy || !outcome.actual_duration_seconds}>Save measured outcome <span>→</span></button></section>}
      <footer className="app-footer">Personalized research prototype <span>·</span> Simulator, not live notification access <span>·</span> No medical advice</footer>
    </main>
  </div>;
}
function ActivityIcon() { return <span style={{ fontSize: 17 }}>⌁</span>; }
createRoot(document.getElementById("root")).render(<App />);
