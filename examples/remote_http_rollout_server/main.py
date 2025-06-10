diff --git a//dev/null b/examples/remote_http_rollout_server/main.py
index 0000000000000000000000000000000000000000..722614d1aadada3a42f8bf1331fbdfd69dc85972 100644
--- a//dev/null
+++ b/examples/remote_http_rollout_server/main.py
@@ -0,0 +1,59 @@
+import os
+import uuid
+from typing import Dict, List
+
+from fastapi import FastAPI, HTTPException
+
+from reward_kit.agent.remote_http_rollout_client import (
+    EndEpisodeRequest,
+    StartEpisodeResponse,
+    StepRequest,
+    StepResponse,
+)
+
+app = FastAPI()
+
+# This server simulates a minimal MCP rollout backend. It keeps a text file per
+# episode and simply returns the updated file contents. The calling pipeline is
+# responsible for invoking any LLMs.
+
+EPISODES: Dict[str, str] = {}
+
+
+@app.post("/start_episode", response_model=StartEpisodeResponse)
+async def start_episode() -> StartEpisodeResponse:
+    episode_id = str(uuid.uuid4())
+    file_path = f"/tmp/episode_{episode_id}.txt"
+    with open(file_path, "w") as f:
+        f.write("")
+    EPISODES[episode_id] = file_path
+    return StartEpisodeResponse(
+        episode_id=episode_id, observation={"file_contents": ""}
+    )
+
+
+@app.post("/step", response_model=StepResponse)
+async def step(req: StepRequest) -> StepResponse:
+    file_path = EPISODES.get(req.episode_id)
+    if not file_path:
+        raise HTTPException(status_code=404, detail="Unknown episode")
+    append_text = req.action.get("append", "")
+    with open(file_path, "a") as f:
+        f.write(append_text)
+    with open(file_path, "r") as f:
+        content = f.read()
+
+    return StepResponse(
+        observation={"file_contents": content}, is_done="DONE" in content
+    )
+
+
+@app.post("/end_episode")
+async def end_episode(req: EndEpisodeRequest) -> dict:
+    file_path = EPISODES.pop(req.episode_id, None)
+    if not file_path:
+        raise HTTPException(status_code=404, detail="Unknown episode")
+    with open(file_path, "r") as f:
+        content = f.read()
+    os.remove(file_path)
+    return {"status": "ended", "final_content": content, "episode_id": req.episode_id}
