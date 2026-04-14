# AI-Powered-Insider-Threat-Detection
This project presents an advanced insider threat detection system that combines continual learning, anomaly detection, explainable AI, and graph-based analytics to identify suspicious user behavior in real time.
By leveraging state-of-the-art machine learning techniques including **Isolation Forests** and **Autoencoders**, alongside transparent **SHAP (SHapley Additive exPlanations)** explainability, this tool aims to spot lateral movement, data exfiltration, and compromised credentials before significant damage occurs.

## 🌟 Key Features

1. **Dashboard Overview**: Highly visual command center summarizing active anomalies, risk scores, and system health metrics.
2. **Interactive Relationship Graph**: NetworkX and PyVis powered knowledge graph tracing the ties between internal users, machines, and detected threats.
3. **Custom CSV Data Ingestion**: Bypass the native simulator and upload your own enterprise log files to be seamlessly analyzed by the anomaly detection pipeline.
4. **Live Threat Intelligence**: Connects anomalous alerts to specific behavior profiles (e.g., "The Data Hoarder" or "Disgruntled Leaker").
5. **XAI Explainability**: Understand *why* the AI flagged an event using SHAP values calculated natively on the backend.

---

## Setup & Installation

### Prerequisites
- **Python 3.9+**
- macOS/Linux/Windows Terminal

### 1. Start the Backend

The backend is built with FastAPI and runs the machine learning pipelines.

1. Open a terminal and navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```bash
   python run.py
   ```
   > The backend will start on `http://127.0.0.1:8000`. Leave this terminal running.

### 2. Start the Frontend

The frontend is a lightweight, Vanilla JS, premium-styled web app.

1. Open a **new** terminal window and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Start a simple Python HTTP server:
   ```bash
   python -m http.server 5050
   ```
3. Open your web browser and navigate to:
   **[http://localhost:5050](http://localhost:5050)**
 
4. dataset used: https://www.kaggle.com/datasets/mrajaxnp/cert-insider-threat-detection-research/data
---

## 🛠️ Usage Guide

* **Simulate Events**: Head to the Dashboard and click the big blue "Run Simulation" button. This will generate thousands of mock events, pass them through the ML models, and visualize the threats.
* **Upload Custom Data**: Click "Upload Data" from the sidebar navigation. You can securely format your logs to match the required CSV headers and feed them uniquely into the pipeline engine.
* **View the Graph**: Navigate to the "Relationships" page and filter by "Anomalies" to see visual linkages between risk events.
* **Configure Models**: Open the "Settings" page to disable specific AI engines, adjust the global critical risk threshold, or set integration URLs.

---

## 📦 Tech Stack

- **Frontend**: HTML5, Vanilla JavaScript, CSS3
- **Backend API**: FastAPI, Uvicorn, Python-Multipart
- **Machine Learning Engine**: TensorFlow (Autoencoders), Scikit-Learn (Isolation Forest)
- **Explainable AI (XAI)**: SHAP
- **Graphing & Networks**: NetworkX, PyVis
- **Environment**: Python 3.9+

## 
