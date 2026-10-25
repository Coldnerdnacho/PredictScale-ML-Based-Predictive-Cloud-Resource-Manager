
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import boto3
from datetime import datetime, timedelta, timezone
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="PredictScale", page_icon="☁️", layout="wide")

st.title("☁️ PredictScale")
st.subheader("ML-Based Predictive Cloud Resource Manager")
st.caption(
    "Prototype: ML workload forecasting, predictive resource allocation, "
    "and a CloudWatch-style synthetic metrics feed with optional read-only AWS integration."
)

with st.sidebar:
    st.header("Simulation controls")
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)
    points = st.slider("Workload history (minutes)", 300, 1500, 600, 100)
    capacity = st.slider("Capacity per worker (requests/min)", 50, 250, 100, 10)
    target_cpu = st.slider("Target utilization (%)", 40, 80, 65, 5) / 100
    max_workers = st.slider("Maximum workers", 2, 12, 8)
    spike = st.checkbox("Include sudden workload spike", value=True)

data_source = st.radio(
    "Workload data source",
    ["Synthetic CloudWatch feed", "AWS CloudWatch", "External demo (reference only)"],
    horizontal=True,
    help=(
        "Synthetic CloudWatch feed generates CloudWatch-style metrics locally. "
        "AWS CloudWatch queries your account. The external demo is not connected to this app."
    )
)

# ---------- Synthetic workload and ML model ----------
rng = np.random.default_rng(int(seed))
t = np.arange(int(points))

demand = (
    100
    + 35 * np.sin(2 * np.pi * t / 120)
    + 18 * np.sin(2 * np.pi * t / 35)
    + 0.025 * t
    + rng.normal(0, 9, len(t))
)
demand = np.maximum(20, demand)

if spike:
    start = int(len(t) * 0.78)
    end = min(len(t), start + 18)

    if end > start:
        demand[start:end] += np.linspace(100, 220, end - start)

    end2 = min(len(t), end + 15)
    if end2 > end:
        demand[end:end2] += np.linspace(180, 20, end2 - end)

df = pd.DataFrame({"minute": t, "demand": demand})
df["lag_1"] = df["demand"].shift(1)
df["lag_2"] = df["demand"].shift(2)
df["rolling_5"] = df["demand"].shift(1).rolling(5).mean()
df["rolling_15"] = df["demand"].shift(1).rolling(15).mean()
df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
df["target"] = df["demand"].shift(-5)

features = [
    "lag_1",
    "lag_2",
    "rolling_5",
    "rolling_15",
    "minute_sin",
    "minute_cos",
]

data = df.dropna().reset_index(drop=True)
split = int(len(data) * 0.8)
train = data.iloc[:split]
test = data.iloc[split:]

model = RandomForestRegressor(
    n_estimators=80,
    max_depth=10,
    min_samples_leaf=2,
    random_state=int(seed),
    n_jobs=-1,
)
model.fit(train[features], train["target"])
predicted = np.maximum(0, model.predict(test[features]))

actual_future = test["target"].to_numpy()
mae = mean_absolute_error(actual_future, predicted)
rmse = np.sqrt(mean_squared_error(actual_future, predicted))

# ---------- Resource allocation simulation ----------
actual = test["demand"].to_numpy()
cap = float(capacity)
target_util = float(target_cpu)

reactive_workers = np.clip(
    np.ceil(actual / (cap * target_util)), 1, int(max_workers)
)
predictive_workers = np.clip(
    np.ceil(predicted / (cap * target_util)), 1, int(max_workers)
)


def evaluate(workers, actual_demand):
    available_capacity = workers * cap
    load_ratio = actual_demand / np.maximum(available_capacity, 1)
    utilization = np.minimum(load_ratio, 1.0)

    # Illustrative latency model, not measured production latency.
    latency = (
        40
        + 0.12 * actual_demand
        + 180 * np.maximum(load_ratio - 0.8, 0) ** 2
        + 250 * np.maximum(load_ratio - 1, 0)
    )

    return {
        "avg_util": 100 * np.mean(utilization),
        "avg_latency": np.mean(latency),
        "worker_minutes": int(np.sum(workers)),
        "energy_units": float(
            np.sum(workers * (0.25 + 0.75 * utilization))
        ),
        "overload_pct": 100 * np.mean(actual_demand > available_capacity),
        "workers": workers,
    }


reactive = evaluate(reactive_workers, actual)
predictive = evaluate(predictive_workers, actual)

# ---------- Source mode information ----------
if data_source == "Synthetic CloudWatch feed":
    st.info(
        "Synthetic CloudWatch feed selected. PredictScale generates a local, "
        "CloudWatch-style stream of CPU, network, and disk metrics. The simulated "
        "workload drives the ML forecast. No external website or AWS account is "
        "sending data. This is a local end-to-end demonstration."
    )

elif data_source == "External demo (reference only)":
    st.warning(
        "This mode does not fetch data into PredictScale. It only links to an "
        "external reference dashboard. Its data may be synthetic."
    )
    st.markdown(
        "[Open the public EC2 metrics dashboard](https://aws-cpu-utilization-metrics-six.vercel.app)"
    )
    st.caption(
        "This external dashboard is not connected to PredictScale."
    )

else:
    st.info(
        "AWS CloudWatch mode retrieves CPU metrics from your existing EC2 instance. "
        "It does not replace the synthetic training dataset or train the model "
        "on AWS metrics."
    )

# ---------- CloudWatch-style synthetic metrics feed ----------
if data_source == "Synthetic CloudWatch feed":
    st.markdown("### CloudWatch-style metrics feed")
    st.caption(
        "These sample metrics are generated locally by PredictScale. "
        "They resemble EC2 CloudWatch metrics but are not fetched from AWS."
    )

    feed_rng = np.random.default_rng(int(seed) + 100)

    metric_cpu = np.clip(
        28
        + 18 * np.sin(2 * np.pi * t / 90)
        + (demand / max(float(np.max(demand)), 1)) * 38
        + feed_rng.normal(0, 4, len(t)),
        2,
        99,
    )

    metric_network_in = np.maximum(
        0, demand * 0.7 + feed_rng.normal(0, 12, len(t))
    )
    metric_network_out = np.maximum(
        0, demand * 0.52 + feed_rng.normal(0, 10, len(t))
    )
    metric_disk_read = np.maximum(
        0, demand * 0.08 + feed_rng.normal(0, 2, len(t))
    )

    metric_df = pd.DataFrame({
        "Simulated minute": t,
        "CPUUtilization (%)": metric_cpu,
        "NetworkIn (proxy)": metric_network_in,
        "NetworkOut (proxy)": metric_network_out,
        "DiskReadOps (proxy)": metric_disk_read,
    })

    f1, f2, f3 = st.columns(3)
    f1.metric("Latest CPUUtilization", f"{metric_cpu[-1]:.1f}%")
    f2.metric("Latest NetworkIn (proxy)", f"{metric_network_in[-1]:.1f}")
    f3.metric("Latest NetworkOut (proxy)", f"{metric_network_out[-1]:.1f}")

    metric_fig = go.Figure()

    metric_fig.add_trace(go.Scatter(
        x=metric_df["Simulated minute"],
        y=metric_df["CPUUtilization (%)"],
        mode="lines",
        name="CPUUtilization (%)",
    ))

    metric_fig.add_trace(go.Scatter(
        x=metric_df["Simulated minute"],
        y=metric_df["NetworkIn (proxy)"],
        mode="lines",
        name="NetworkIn (proxy)",
        yaxis="y2",
    ))

    metric_fig.update_layout(
        xaxis_title="Simulated minute",
        yaxis=dict(title="CPU utilization (%)"),
        yaxis2=dict(
            title="Network activity (proxy)",
            overlaying="y",
            side="right",
        ),
        height=320,
    )

    st.plotly_chart(metric_fig, use_container_width=True)

    with st.expander("View synthetic metrics records"):
        st.dataframe(
            metric_df.tail(30),
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download synthetic CloudWatch-style metrics (CSV)",
        data=metric_df.to_csv(index=False).encode("utf-8"),
        file_name="predictscale_synthetic_cloudwatch_metrics.csv",
        mime="text/csv",
    )

# ---------- Key metrics ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast MAE", f"{mae:.1f} req/min")
c2.metric("Forecast RMSE", f"{rmse:.1f} req/min")
c3.metric("Predictive latency*", f"{predictive['avg_latency']:.1f} ms")
c4.metric("Worker-minutes*", f"{predictive['worker_minutes']}")

st.caption(
    "*Latency and worker-minutes are simulation outputs, not actual AWS billing "
    "or production measurements."
)

# ---------- Workload prediction chart ----------
st.markdown("### 1. Workload prediction")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=test["minute"],
    y=actual_future,
    mode="lines",
    name="Actual demand (5 min ahead)",
))
fig.add_trace(go.Scatter(
    x=test["minute"],
    y=predicted,
    mode="lines",
    name="ML forecast",
))
fig.update_layout(
    title="Actual vs Predicted Workload (5-minute horizon)",
    xaxis_title="Simulated minute",
    yaxis_title="Requests per minute",
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Allocation chart ----------
st.markdown("### 2. Resource allocation")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=test["minute"],
    y=reactive["workers"],
    mode="lines",
    name="Reactive workers",
))
fig2.add_trace(go.Scatter(
    x=test["minute"],
    y=predictive["workers"],
    mode="lines",
    name="Predictive workers",
))
fig2.update_layout(
    xaxis_title="Simulated minute",
    yaxis_title="Workers allocated",
    height=300,
    yaxis=dict(dtick=1),
)
st.plotly_chart(fig2, use_container_width=True)

# ---------- Decision explanation ----------
st.markdown("### 3. Resource allocation decision")

latest_prediction = float(predicted[-1])
recommended = int(np.clip(
    np.ceil(latest_prediction / (cap * target_util)),
    1,
    int(max_workers),
))
planned_capacity = recommended * cap
estimated_util = 100 * latest_prediction / max(planned_capacity, 1)

d1, d2, d3, d4 = st.columns(4)
d1.metric("Latest 5-min forecast", f"{latest_prediction:.1f} req/min")
d2.metric("Recommended workers", str(recommended))
d3.metric("Planned capacity", f"{planned_capacity:.0f} req/min")
d4.metric("Forecast load / capacity", f"{estimated_util:.1f}%")

st.code(
    f"workers = ceil(predicted_demand / "
    f"(capacity_per_worker × target_utilization))\n"
    f"workers = ceil({latest_prediction:.1f} / "
    f"({cap:.0f} × {target_util:.2f})) = "
    f"{int(np.ceil(latest_prediction / (cap * target_util)))}\n"
    f"Applied safety bounds: minimum 1, maximum {int(max_workers)} workers",
    language="text",
)

if recommended >= int(max_workers):
    st.warning("Recommendation reached the configured maximum worker limit.")
else:
    st.success("Recommendation is within the configured worker limit.")

# ---------- Strategy comparison ----------
st.markdown("### 4. Strategy comparison")

comparison = pd.DataFrame({
    "Metric": [
        "Avg utilization (%)",
        "Avg latency (ms)",
        "Worker-minutes",
        "Energy units",
        "Overloaded time (%)",
    ],
    "Reactive": [
        reactive["avg_util"],
        reactive["avg_latency"],
        reactive["worker_minutes"],
        reactive["energy_units"],
        reactive["overload_pct"],
    ],
    "Predictive": [
        predictive["avg_util"],
        predictive["avg_latency"],
        predictive["worker_minutes"],
        predictive["energy_units"],
        predictive["overload_pct"],
    ],
})

st.dataframe(
    comparison.style.format({
        "Reactive": "{:.2f}",
        "Predictive": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
    height=230,
)

st.download_button(
    "Download strategy comparison (CSV)",
    data=comparison.to_csv(index=False).encode("utf-8"),
    file_name="predictscale_comparison.csv",
    mime="text/csv",
    key="comparison_csv",
)

# ---------- Prediction errors ----------
st.markdown("### 5. What if the prediction is wrong?")

under = np.maximum(actual - predicted, 0)
over = np.maximum(predicted - actual, 0)

m1, m2, m3 = st.columns(3)
m1.metric("Average underprediction", f"{under.mean():.1f} req/min")
m2.metric("Average overprediction", f"{over.mean():.1f} req/min")
m3.metric("Fallback policy", "Reactive scale-up")

st.info(
    "Underprediction may cause higher latency if demand exceeds capacity. "
    "Overprediction can waste capacity and cost. Mitigations include minimum/maximum "
    "capacity bounds, a safety buffer, reactive scale-up when observed load crosses "
    "a threshold, and a cooldown before scale-down."
)

# ---------- Read-only AWS CloudWatch integration ----------
st.markdown("### 6. AWS CloudWatch metrics (read-only)")
st.write(
    "Retrieve CPUUtilization for an existing EC2 instance. This does not create, "
    "start, stop, or resize any AWS resources."
)

with st.expander("Connect to CloudWatch"):
    aws_region = st.text_input(
        "AWS region", value="ap-south-1", key="cw_region"
    )
    instance_id = st.text_input(
        "Existing EC2 instance ID",
        placeholder="i-0123456789abcdef0",
        key="cw_instance",
    )
    hours = st.selectbox("History window", [1, 3, 6], index=0)

    if st.button("Fetch real CPU metrics", key="fetch_cloudwatch"):
        if not instance_id.startswith("i-") or len(instance_id) < 10:
            st.error(
                "Enter a valid-looking EC2 instance ID, "
                "such as i-0123456789abcdef0."
            )
        else:
            try:
                cloudwatch = boto3.client(
                    "cloudwatch", region_name=aws_region
                )
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=int(hours))

                response = cloudwatch.get_metric_data(
                    StartTime=start_time,
                    EndTime=end_time,
                    ScanBy="TimestampAscending",
                    MetricDataQueries=[
                        {
                            "Id": "cpu",
                            "Label": "EC2 CPU Utilization",
                            "MetricStat": {
                                "Metric": {
                                    "Namespace": "AWS/EC2",
                                    "MetricName": "CPUUtilization",
                                    "Dimensions": [
                                        {
                                            "Name": "InstanceId",
                                            "Value": instance_id,
                                        }
                                    ],
                                },
                                "Period": 300,
                                "Stat": "Average",
                            },
                            "ReturnData": True,
                        }
                    ],
                )

                series = response["MetricDataResults"][0]

                if not series["Values"]:
                    st.warning(
                        "No datapoints returned. Verify the instance ID and region, "
                        "check IAM permission cloudwatch:GetMetricData, and confirm "
                        "that the instance has reported metrics during this window."
                    )
                else:
                    cloud_df = pd.DataFrame({
                        "Timestamp": series["Timestamps"],
                        "CPU utilization (%)": series["Values"],
                    }).sort_values("Timestamp")

                    st.success(
                        f"Retrieved {len(cloud_df)} real CloudWatch datapoints "
                        f"for {instance_id} in {aws_region}."
                    )
                    st.metric(
                        "Latest CPU utilization",
                        f"{cloud_df['CPU utilization (%)'].iloc[-1]:.2f}%",
                    )
                    st.line_chart(
                        cloud_df.set_index("Timestamp")["CPU utilization (%)"]
                    )
                    st.dataframe(
                        cloud_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "These are real AWS CPU metrics if the query succeeded. "
                        "They are not request-rate data and are not used to train "
                        "the synthetic workload model above."
                    )

            except NoCredentialsError:
                st.error(
                    "No AWS credentials found. Do not paste access keys into this app. "
                    "Configure an approved AWS credential method or continue in "
                    "Synthetic CloudWatch feed mode."
                )
            except (BotoCoreError, ClientError) as exc:
                st.error(
                    "CloudWatch request failed. Check region, instance ID and IAM "
                    f"permissions. Details: {exc}"
                )
            except Exception as exc:
                st.error(f"Unexpected error while retrieving metrics: {exc}")

# ---------- Methodology and limitations ----------
st.markdown("### Methodology and limitations")
st.write(
    "Input features: previous 1- and 2-minute demand, 5- and 15-minute rolling "
    "averages, and cyclical time features. Model: Random Forest Regressor. "
    "Prediction horizon: 5 minutes. Evaluation: chronological 80/20 split using "
    "MAE and RMSE. The workload is synthetic; latency, energy and cost are proxies. "
    "The CloudWatch panel is read-only and optional. This prototype does not "
    "provision or autoscale real AWS resources."
)

st.download_button(
    "Download strategy comparison (CSV)",
    data=comparison.to_csv(index=False).encode("utf-8"),
    file_name="predictscale_comparison.csv",
    mime="text/csv",
)