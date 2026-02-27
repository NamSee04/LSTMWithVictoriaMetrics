# LSTM Anomaly Detection for VictoriaMetrics

This project is a standalone Python service that performs anomaly detection on time-series metrics stored in VictoriaMetrics. It uses Long Short-Term Memory (LSTM) neural networks to learn the normal behavior of your metrics and detect anomalies when actual values deviate significantly from predictions.

## How It Works (The Flow)

The service operates in a continuous loop, orchestrated by a built-in scheduler. The workflow consists of two main cycles: **Fit (Training)** and **Infer (Prediction)**.

```text
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ VictoriaMetrics │       │   LSTM Service  │       │ VictoriaMetrics │
│      (TSDB)     │       │                 │       │      (TSDB)     │
│                 │   1   │                 │   3   │                 │
│  /query_range   │ ────► │  Read Metrics   │ ────► │ /import/prom... │
└─────────────────┘       └────────┬────────┘       └─────────────────┘
                                   │
                               2   │
                                   ▼
                          ┌─────────────────┐
                          │    LSTM Model   │
                          │ - Fit (Train)   │
                          │ - Infer (Score) │
                          └─────────────────┘
```

1. **Read Data (`vm_reader.py`)**: The service periodically queries VictoriaMetrics using MetricsQL/PromQL through the HTTP API (`/api/v1/query_range`). It fetches historical data for training or recent data for inference based on queries defined in `config.yaml`.
2. **Process Data (`lstm_model.py`)**: 
   - **Fit Cycle**: The model trains on historical data to learn the pattern and computes a baseline error distribution (mean and standard deviation).
   - **Infer Cycle**: The trained model predicts the expected value (`yhat`) for the current time step. It compares the prediction against the actual value to calculate an `anomaly_score`. A score > 1.0 indicates an anomaly.
3. **Write Results (`vm_writer.py`)**: The generated metrics (`anomaly_score`, `yhat`, `yhat_lower`, `yhat_upper`) are formatted into Prometheus text format and pushed back into VictoriaMetrics via the HTTP API (`/api/v1/import/prometheus`).

## Components

- `src/main.py`: The entry point that initializes all components and starts the scheduler.
- `src/scheduler.py`: Orchestrates when to run the Fit cycle (e.g., every 1 hour) and Infer cycle (e.g., every 1 minute).
- `src/lstm_model.py`: The PyTorch LSTM neural network. It trains a separate model instance for each unique metric labelset to ensure isolated context learning.
- `src/vm_reader.py`: Handles HTTP queries to VictoriaMetrics.
- `src/vm_writer.py`: Handles HTTP writes back to VictoriaMetrics.
- `config.yaml`: The central configuration file where you define queries, model hyperparameters, and scheduling intervals.
- `alerts/anomaly_rules.yml`: Example `vmalert` rules to trigger alerts based on the generated anomaly scores.

## Installation & Setup

1. **Install Dependencies**:
   This service requires Python 3.9+ and CPU-only PyTorch.
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure**:
   Edit `config.yaml` to specify your VictoriaMetrics URL and the PromQL/MetricsQL queries for the metrics you want to monitor.

3. **Run**:
   ```bash
   python src/main.py --config config.yaml
   ```

## Integration with VictoriaMetrics Ecosystem

Once the service is running, it will continuously write new time series back to VictoriaMetrics:
- `lstm_anomaly_anomaly_score{for="query_alias", ...}`
- `lstm_anomaly_yhat{for="query_alias", ...}` 

You can then use **vmalert** (using the provided `alerts/anomaly_rules.yml`) to trigger notifications when `lstm_anomaly_anomaly_score` exceeds 1.0, and visualize the actual vs predicted values in **Grafana**.
