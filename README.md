# 🎬 Film Stream Forecasting API

[![Python 3.11+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![uv](https://img.shields.io/badge/package%20manager-uv-de5fe9.svg)](https://docs.astral.sh/uv/)

A production-ready **FastAPI** service that generates 24-hour stream-start capacity forecasts for video-on-demand (VOD) film assets at 15-minute granularity. 

Designed for downstream infrastructure autoscaling engines, load balancing, and real-time streaming capacity planning.

---

## ✨ Features

- **High-Resolution Forecasting:** Generates 96 continuous 15-minute interval predictions over a rolling 24-hour horizon.
- **Confidence Bounds:** Calculates 95% confidence bounds (`lower` and `upper`) alongside expected stream counts using Poisson/Gaussian distribution noise modeling.
- **Batch & Catalog Queries:** Predict for a single film ID, a batch of film IDs, or the entire catalog (`"all"`).
- **FastAPI Core:** Leverages asynchronous Python, Pydantic v2 data validation, and NumPy vectorization.
- **Numpydoc Documented:** Fully typed and documented using strict NumPy-style docstring standards.
- **Kubernetes Ready:** Includes `/healthz/live` and `/healthz/ready` probes for automated deployment orchestration.

---

## 🏗️ Architecture & Model Schema

The forecast model combines daily harmonic curves, weekly seasonality multipliers, and target-specific baseline scaling:

$$y_t = \max\left(0, \text{Base} \cdot \left[1 + 0.4 \sin\left(\frac{2\pi(h-6)}{24}\right)\right] \cdot \left[1 + 0.15 \sin\left(\frac{2\pi h}{168}\right)\right] + \epsilon\right)$$

Where:
- $\text{Base}$ is the film-specific baseline volume parameter ($50 + \text{seed} \bmod 200$).
- $h$ is the forecast time horizon in hours.
- $\epsilon \sim \mathcal{N}(0, 3)$ represents stochastic load variance.
- Confidence intervals are estimated using $\sigma = 1.5 \sqrt{\hat{y} + 1}$.

---

## 🚀 Quickstart

### 1. Installation & Environment Setup

Ensure you have [`uv`](https://docs.astral.sh/uv/) installed. Clone the repository and sync the project dependencies:

```bash
git clone [https://github.com/your-username/film-stream-forecaster.git](https://github.com/your-username/film-stream-forecaster.git)
cd film-stream-forecaster

# Sync all production and dev dependencies (creates .venv automatically)
uv sync --all-groups

```

### 2. Running the Application

Start the FastAPI application using Uvicorn via `uv`:

```bash
uv run uvicorn app.main:app --reload

```

Upon startup, the server log banner will display local interactive documentation URLs:

```text
============================================================
 🚀 FILM STREAM FORECASTING API STARTED
 📖 Interactive Swagger UI Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
 📑 ReDoc UI Docs:               [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
 🟢 Health Check:                [http://127.0.0.1:8000/healthz/live](http://127.0.0.1:8000/healthz/live)
============================================================

```

### 3. Running Tests

Run the complete asynchronous test suite using `pytest`:

```bash
# Run all tests with verbose output
uv run pytest -v

# Run tests with code coverage report
uv run pytest --cov=app

```

### 4. Code Formatting & Validation

Ensure the codebase conforms to `black` formatting standards:

```bash
# Check formatting
uv run black --check app tests

# Apply auto-formatting
uv run black app tests

```
---
