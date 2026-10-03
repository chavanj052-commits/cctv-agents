import express from "express";
import cors from "cors";
import multer from "multer";
import mongoose from "mongoose";
import path from "path";
import fs from "fs";
import { Agent, setGlobalDispatcher } from "undici";

setGlobalDispatcher(new Agent({ headersTimeout: 0, bodyTimeout: 0 })); // agent can take minutes

const AGENT_URL = process.env.AGENT_URL || "http://127.0.0.1:5001";
const MONGO_URI = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/cctv_agents";

const app = express();
app.use(cors());
app.use(express.json());

fs.mkdirSync("uploads", { recursive: true });
const upload = multer({
  storage: multer.diskStorage({
    destination: "uploads",
    filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname)),
  }),
});
app.use("/uploads", express.static("uploads")); // serves videos (supports seeking)

const Case = mongoose.model(
  "Case",
  new mongoose.Schema(
    { video: String, object: String, outcome: String, lastSeenSec: Number },
    { timestamps: true }
  )
);
mongoose
  .connect(MONGO_URI, { serverSelectionTimeoutMS: 3000 })
  .then(() => console.log("MongoDB connected"))
  .catch(() => console.log("MongoDB not running - cases won't be saved (app still works)"));

app.post("/api/detect", upload.single("video"), async (req, res) => {
  try {
    const object = (req.body.object || "").trim();
    if (!req.file || !object) return res.status(400).json({ error: "video and object are required" });

    const r = await fetch(`${AGENT_URL}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_path: path.resolve(req.file.path), object }),
    });
    const result = await r.json();
    if (!r.ok) throw new Error(result.error || "Agent error");

    if (mongoose.connection.readyState === 1) {
      await Case.create({
        video: req.file.filename, object,
        outcome: result.outcome, lastSeenSec: result.last_seen_sec,
      });
    }
    res.json({ ...result, videoUrl: `/uploads/${req.file.filename}` });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/cases", async (req, res) => {
  if (mongoose.connection.readyState !== 1) return res.json([]);
  res.json(await Case.find().sort({ createdAt: -1 }).limit(20));
});

app.listen(5000, () => console.log("Server on http://localhost:5000"));
