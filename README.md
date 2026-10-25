# PredictScale

ML-based predictive cloud resource manager prototype.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If `python` is not recognized on Windows, try `py`.

## Features
- Random Forest workload forecast with a 5-minute horizon
- Chronological 80/20 split and MAE/RMSE
- Predictive vs reactive allocation simulation
- Utilization, simulated latency, relative cost and energy proxy
- Underprediction/overprediction discussion and safety limits
- Optional read-only CloudWatch CPU metric query for an existing EC2 instance
- CSV export of strategy comparison

## CloudWatch requirements
- An existing EC2 instance in the selected AWS region that has published CPU metrics
- AWS credentials configured using an approved method
- Permission for `cloudwatch:GetMetricData`

The CloudWatch integration is read-only. It does not create, start, stop, resize, or autoscale AWS resources.


