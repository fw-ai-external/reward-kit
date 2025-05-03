# Goal
- makes the type more correct

I made the following type changes

```
(.venv) (base) bchen@dev-modeling:~/home/reward-kit(main)$ git diff
diff --git a/reward_kit/typed_interface.py b/reward_kit/typed_interface.py
index c9f781d..d24e06b 100644
--- a/reward_kit/typed_interface.py
+++ b/reward_kit/typed_interface.py
@@ -105,21 +105,7 @@ def reward_function(func: EvaluateFunction) -> DictEvaluateFunction:
         # Handle the updated EvaluateResult model structure
         if isinstance(result_model, EvaluateResult):
             # Build a response including all the metrics
-            result_dict = {}
-            
-            # Add each metric to the result dictionary
-            for key, metric in result_model.metrics.items():
-                result_dict[key] = {
-                    "success": metric.success,
-                    "score": metric.score,
-                    "reason": metric.reason,
-                }
-            
-            # If there's an error, add it to the result
-            if result_model.error:
-                result_dict["error"] = {"error": result_model.error}
-            
-            return result_dict
+            return result_model.model_dump()
         else:
             return _res_adapter.dump_python(result_model, mode="json")
 
diff --git a/setup.py b/setup.py
index 68283db..cf4dba9 100644
--- a/setup.py
+++ b/setup.py
@@ -2,7 +2,7 @@ from setuptools import setup, find_packages
 
 setup(
```

please help me fix all the downstream typing issues, and check if my test covers this, if not please help me add a test
