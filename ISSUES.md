# Goal
- makes the type more correct

I made the following type changes


```
(.venv) (base) bchen@dev-modeling:~/home/reward-kit(main)$ git diff reward_kit/models.py
diff --git a/reward_kit/models.py b/reward_kit/models.py
index 46aa9e2..b838d5c 100644
--- a/reward_kit/models.py
+++ b/reward_kit/models.py
@@ -2,7 +2,7 @@ from typing import Dict, List, Optional, Any, Union, Callable, Literal
 from dataclasses import dataclass, field
 from dataclasses_json import dataclass_json
 import json
-from pydantic import BaseModel, Field, RootModel
+from pydantic import BaseModel, Field
 
 # Import OpenAI message types
 from openai.types.chat import ChatCompletionMessageParam
@@ -21,16 +21,18 @@ class Message(BaseModel):
 class MetricResult(BaseModel):
     """Result of a single metric evaluation."""
 
-    success: bool
+    success: Optional[bool] = None
     score: float = Field(..., ge=0.0, le=1.0)
     reason: str
 
 
-# Use RootModel for pydantic v2 compatibility
-class EvaluateResult(RootModel):
+class EvaluateResult(BaseModel):
     """The complete result of an evaluator with multiple metrics."""
-
-    root: Dict[str, MetricResult]
+    
+    error: Optional[str] = None
+    score: float = Field(..., ge=0.0, le=1.0)
+    reason: Optional[str] = None
+    metrics: Dict[str, MetricResult]
 
 
 # Original dataclass-based models for backwards compatibility
 ```

 please help me fix all the downstream typing issues
