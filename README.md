# COGNIS

### Self-Healing AI Framework for Monitoring, Diagnosis, and Autonomous Recovery of Machine Learning Models

## Overview

COGNIS is a lightweight, model-agnostic self-healing AI framework designed to monitor machine learning models, detect performance degradation, diagnose the underlying cause, apply corrective actions, validate repairs, and explain its decisions in natural language.

Unlike traditional machine learning deployments that rely on manual monitoring and retraining, COGNIS continuously evaluates model health and attempts to recover from common failure modes automatically.

The framework treats the machine learning model as a black box and focuses on reliability, observability, and autonomous maintenance.

---

## Key Features

### Model-Agnostic Architecture

COGNIS can work with any classification model that supports standard prediction interfaces.

### Continuous Health Monitoring

Tracks multiple performance and behavioral signals including:

* Accuracy
* Log Loss
* Confidence Distribution
* Prediction Entropy
* Confidence Drift
* Class Distribution Changes
* Calibration Error Indicators

### Root Cause Diagnosis

Uses a rule-based diagnosis engine to identify likely causes of degradation.

Currently supported diagnoses:

* Concept Drift
* Class Imbalance
* Label Noise
* Calibration Error

### Autonomous Self-Healing

Applies targeted recovery strategies based on the diagnosed issue.

Examples include:

* Fine-tuning / retraining
* Class reweighting
* Noise-aware weighting
* Temperature scaling

### Safety Validation and Rollback

Every repair is validated before promotion.

If performance does not improve:

* The candidate repair is rejected
* The model is rolled back to its previous stable state

### Natural Language Explanations

COGNIS generates human-readable reports describing:

* What was detected
* Why it occurred
* What repair was applied
* Whether the repair succeeded

LLM support is optional and includes fallback explanations when external APIs are unavailable.

### Interactive Dashboard

Includes a Streamlit-based interface for:

* Model upload
* Dataset upload
* Diagnosis execution
* Conversational-style explanation output

---

## System Architecture

```text
                 ┌───────────────────┐
                 │ Uploaded ML Model │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Model Interface   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Health Monitor    │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Diagnosis Engine  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Self-Healing      │
                 │ Engine            │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Validator         │
                 └─────────┬─────────┘
                           │
                 ┌─────────▼─────────┐
                 │ Explainer (LLM)   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Dashboard UI      │
                 └───────────────────┘
```

---

## Project Structure

```text
COGNIS/
│
├── core/
│   ├── cognis.py
│   ├── model_interface.py
│   ├── monitoring.py
│   ├── diagnosis.py
│   ├── fixer.py
│   ├── validator.py
│   └── explainer.py
│
├── utils/
│   ├── config.py
│   ├── metrics.py
│   └── drift.py
│
├── dashboard/
│   └── app.py
│
├── experiments/
│   └── tests
│
└── README.md
```

---

## Monitoring Signals

COGNIS evaluates several independent monitoring signals:

| Signal                      | Purpose                                 |
| --------------------------- | --------------------------------------- |
| Accuracy Drop               | Detects performance degradation         |
| Entropy Shift               | Detects uncertainty changes             |
| Confidence Drift            | Detects confidence distribution changes |
| Class Imbalance             | Detects class distribution skew         |
| Label Noise Indicator       | Detects high-confidence mistakes        |
| Calibration Error Indicator | Detects confidence-accuracy mismatch    |

---

## Diagnosis Rules

The diagnosis engine combines monitoring signals to identify likely causes.

| Diagnosed Issue   | Primary Signals                                  |
| ----------------- | ------------------------------------------------ |
| Concept Drift     | Accuracy Drop + Entropy Shift + Confidence Drift |
| Class Imbalance   | Class Distribution Shift                         |
| Label Noise       | High Confidence Incorrect Predictions            |
| Calibration Error | Confidence vs Accuracy Gap                       |

---

## Healing Strategies

| Issue             | Strategy              |
| ----------------- | --------------------- |
| Concept Drift     | Fine-Tuning           |
| Class Imbalance   | Class Reweighting     |
| Label Noise       | Noise-Aware Weighting |
| Calibration Error | Temperature Scaling   |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-repository/cognis.git
cd cognis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

---

## Workflow

1. Upload a trained model.
2. Upload evaluation data.
3. Optionally upload baseline data.
4. Run diagnosis.
5. COGNIS:

   * Evaluates model health
   * Detects degradation
   * Diagnoses root cause
   * Applies repair
   * Validates repair
   * Generates explanation
6. Review results through the dashboard.

---

## Current Status

Implemented:

* Model Interface Layer
* Health Monitoring System
* Signal-Based Detection
* Diagnosis Engine
* Self-Healing Engine
* Validation & Rollback
* LLM Explanation Layer
* Streamlit Dashboard

Planned:

* Experience Memory
* Healing Timeline Visualization
* Historical Recovery Analytics
* Adaptive Repair Recommendation System

---

## Design Philosophy

COGNIS is intentionally lightweight.

The goal is not to build a large autonomous AI agent, but to create a practical and explainable framework capable of improving the reliability of deployed machine learning systems.

The framework prioritizes:

* Transparency
* Safety
* Explainability
* Modularity
* Low Resource Consumption

---

## License

MIT License
