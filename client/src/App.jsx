import { useEffect, useRef, useState } from "react";
import "./App.css";

const BANNERS = {
  FOUND_WITH_ACTIVITY: [
    "success",
    "Found with activity - the object disappeared near a person.",
  ],
  FOUND_NO_ACTIVITY: [
    "warning",
    "Found, but no clear activity - showing last-seen position.",
  ],
  NOT_DETECTED: [
    "error",
    "Not confidently detected - send the full feed for manual review.",
  ],
};

export default function App() {
  const [file, setFile] = useState(null);
  const [object, setObject] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(0); // 0 = show all matches, otherwise the rank to isolate
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const submit = async () => {
    if (!file || !object.trim())
      return setError("Choose a video and enter an object name.");
    setError("");
    setResult(null);
    setLoading(true);
    const body = new FormData();
    body.append("video", file);
    body.append("object", object);
    try {
      const res = await fetch("/api/detect", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setSelected(0); // show all matches for every new result
      setResult(data);
    } catch (e) {
      setError(e.message || "Something went wrong");
    }
    setLoading(false);
  };

  // draw only the selected candidate's box, synced with video playback
  useEffect(() => {
    if (!result) return;
    const byFrame = {};
    result.frames.forEach((f) => (byFrame[f.f] = f.boxes));
    let raf;
    const draw = () => {
      const v = videoRef.current,
        c = canvasRef.current;
      if (v && c) {
        c.width = v.videoWidth || result.width;
        c.height = v.videoHeight || result.height;
        const ctx = c.getContext("2d");
        const frame =
          Math.floor((v.currentTime * result.fps) / result.stride) *
          result.stride;
        (byFrame[frame] || [])
          .filter((b) => selected === 0 || b.id === selected)
          .forEach((b) => {
            ctx.strokeStyle = ctx.fillStyle = "#22c55e";
            ctx.lineWidth = Math.max(3, c.width / 300);
            ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
            ctx.font = `${Math.max(16, c.width / 50)}px sans-serif`;
            ctx.fillText(
              `${result.object} #${b.id}`,
              b.x1,
              Math.max(18, b.y1 - 6),
            );
          });
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [result, selected]);

  const chosen = result?.candidates?.find((c) => c.rank === (selected || 1));
  const lastSeen = chosen ? chosen.last_seen_sec : result?.last_seen_sec;
  const outcome = chosen ? chosen.outcome : result?.outcome;

  const jumpToLastSeen = () => {
    videoRef.current.currentTime = lastSeen;
    videoRef.current.pause();
  };

  const banner = result && BANNERS[outcome];

  return (
    <div className="page">
      <h1>CCTV Lost-Item Tracker</h1>
      <p className="sub">Agent 3 - Object Detection & Tracking</p>

      <div className="card controls">
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <input
          type="text"
          placeholder="Describe the object, e.g. pink and blue bag"
          value={object}
          onChange={(e) => setObject(e.target.value)}
        />
        <button onClick={submit} disabled={loading}>
          {loading
            ? "Analysing... (this can take a few minutes)"
            : "Detect & Track"}
        </button>
      </div>

      {error && <div className="banner error">{error}</div>}

      {result && (
        <div className="card">
          <div className={`banner ${banner[0]}`}>
            {banner[1]}
            {lastSeen != null && <> Last seen at {lastSeen}s.</>}
          </div>

          {result.understood && (
            <p className="sub">Understood as: {result.understood}</p>
          )}

          <div className="player">
            <video
              ref={videoRef}
              src={result.videoUrl}
              controls
              autoPlay
              muted
            />
            <canvas ref={canvasRef} />
          </div>

          {lastSeen != null && (
            <button className="secondary" onClick={jumpToLastSeen}>
              Jump to last seen
            </button>
          )}

          {result.candidates?.length > 0 && (
            <>
              <p className="sub" style={{ marginTop: 16 }}>
                {result.candidates.length} match
                {result.candidates.length > 1 ? "es" : ""} found, best first -
                click one to isolate it
              </p>
              <div
                style={{
                  display: "flex",
                  gap: 12,
                  marginTop: 8,
                  flexWrap: "wrap",
                }}
              >
                <button
                  onClick={() => setSelected(0)}
                  style={{
                    padding: 8,
                    borderRadius: 8,
                    cursor: "pointer",
                    background: "transparent",
                    color: "inherit",
                    border:
                      selected === 0
                        ? "2px solid #22c55e"
                        : "2px solid #334155",
                  }}
                >
                  Show all
                </button>
                {result.candidates.map((c) => (
                  <button
                    key={c.rank}
                    onClick={() => setSelected(c.rank)}
                    style={{
                      padding: 8,
                      borderRadius: 8,
                      cursor: "pointer",
                      background: "transparent",
                      color: "inherit",
                      border:
                        selected === c.rank
                          ? "2px solid #22c55e"
                          : "2px solid #334155",
                    }}
                  >
                    {c.thumb && (
                      <img
                        src={c.thumb}
                        alt=""
                        style={{ height: 70, display: "block" }}
                      />
                    )}
                    <div>
                      #{c.rank} · {Math.round(c.score * 100)}%
                    </div>
                    <div style={{ fontSize: 12 }}>
                      {c.first_seen_sec}s – {c.last_seen_sec}s
                    </div>
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
