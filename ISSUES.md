# Goal
Replica the evaluation preview and deploy logic in firectl in the SDK

## Evaluation preview

So if you want to preview the code (needs main.py that implements evaluate()), you can build the latest firectl-admin and do this

```
firectl-admin preview eval --metric-folder <metric_name>=/path/to/python-folder --sample-file /path/to/file.jsonl
```

## Evaluation creation
after preview, you can then create with higher confidence, let me grab the command
```
firectl-admin create eval --metric-folder metric=/path/to/folder/ <eval_id>
```
Or if it's one folder generating multiple metrics, taht's called --multi-metrics
```
firectl-admin create eval --multi-metrics --folder /path/to/folder/ <eval_id>
```
