# CCTV Lost-Item Tracker (MERN + Python agent)
- agent/   Flask API running Agent 3 (YOLO-World detection + tracking)  -> port 5001
- server/  Express + MongoDB (uploads, calls the agent, stores cases)    -> port 5000
- client/  React (Vite) UI with video + box overlay                      -> port 5173
Start order: agent, server, client (3 terminals).
