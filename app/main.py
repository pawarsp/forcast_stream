from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import numpy as np
from pydantic import BaseModel, Field, field_validator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan manager for startup and shutdown hooks.

    Parameters
    ----------
    app : FastAPI
        The current FastAPI application instance.

    Yields
    ------
    None
        Yields execution back to the FastAPI web server runtime.
    """
    print("\n" + "=" * 60)
    print(" 🚀 FILM STREAM FORECASTING API STARTED")
    print(" 📖 Interactive Swagger UI Docs: http://127.0.0.1:8000/docs")
    print(" 📑 ReDoc UI Docs:               http://127.0.0.1:8000/redoc")
    print(" 🟢 Health Check:                http://127.0.0.1:8000/healthz/live")
    print("=" * 60 + "\n")
    yield


app = FastAPI(
    title="Film Stream Forecasting API",
    version="0.1.0",
    description="24h forecast of stream-start counts per film, 15-minute resolution.",
    lifespan=lifespan,
)

INTERVAL_MINUTES = 15
HORIZON_INTERVALS = 24 * 60 // INTERVAL_MINUTES  # 96
FILMS = [f"film_{i:04d}" for i in range(1, 1001)]


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    """Redirect root traffic directly to interactive Swagger UI documentation.

    Returns
    -------
    RedirectResponse
        HTTP 307 temporary redirect pointing to `/docs`.
    """
    return RedirectResponse(url="/docs")


class ForecastRequest(BaseModel):
    """Request schema for stream capacity prediction query.

    Attributes
    ----------
    film_ids : Union[str, List[str]]
        A single film identifier string (e.g. 'film_0010') or a list of film identifiers.
    forecast_start : datetime
        UTC timestamp marking the beginning of the 24-hour prediction window.
    include_bounds : bool
        Whether to calculate lower and upper prediction confidence bounds.
    """

    film_ids: Union[str, List[str]] = Field(
        default="film_0010",
        description="A single film_id string ('film_0001'), or a list of film_ids.",
    )
    forecast_start: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    include_bounds: bool = True

    @field_validator("film_ids", mode="before")
    @classmethod
    def normalize_film_ids(
        cls, v: Optional[Union[str, List[str]]]
    ) -> Union[str, List[str]]:
        """Normalize input film_ids payload into standard string or list representation.

        Parameters
        ----------
        v : Optional[Union[str, List[str]]]
            Raw payload value passed into the `film_ids` field.

        Returns
        -------
        Union[str, List[str]]
            Normalized list of string identifiers or string keyword.
        """
        if v is None or v == "":
            return "film_0001"
        if isinstance(v, str) and v.strip().lower() != "all":
            return [v.strip()]
        return v


class ForecastPoint(BaseModel):
    """Data point representing stream prognosis for a single 15-minute interval.

    Attributes
    ----------
    interval_start : datetime
        UTC start timestamp of the 15-minute bucket.
    predicted_streams : int
        Expected total stream-start count (floored integer).
    lower : Optional[int]
        Lower 95% confidence interval bound (floored integer).
    upper : Optional[int]
        Upper 95% confidence interval bound (floored integer).
    """

    interval_start: datetime
    predicted_streams: int
    lower: Optional[int] = None
    upper: Optional[int] = None


class FilmForecast(BaseModel):
    """Container holding metadata and interval predictions for a single film asset.

    Attributes
    ----------
    film_id : str
        Unique identifier of the forecasted film.
    forecast_start : datetime
        UTC timestamp marking start of the prediction horizon.
    forecast_end : datetime
        UTC timestamp marking end of the prediction horizon.
    interval_minutes : int
        Duration of each interval in minutes.
    model_version : str
        Registry version identifier of the active ML model.
    feature_version : str
        Feature store view/version used during inference.
    feature_watermark : datetime
        Timestamp of the latest feature ingested into the online store.
    data_freshness_seconds : int
        Age of features in seconds at generation time.
    forecast_source : Literal["ml_model", "baseline", "stale_cache"]
        Pipeline source path generating the inference data.
    points : List[ForecastPoint]
        Sequence of interval-level prediction objects across the horizon.
    """

    film_id: str
    forecast_start: datetime
    forecast_end: datetime
    interval_minutes: int
    model_version: str
    feature_version: str
    feature_watermark: datetime
    data_freshness_seconds: int
    forecast_source: Literal["ml_model", "baseline", "stale_cache"]
    points: List[ForecastPoint]


class ForecastResponse(BaseModel):
    """Top-level API response envelope containing film stream predictions.

    Attributes
    ----------
    generated_at : datetime
        UTC timestamp when the response payload was rendered.
    interval_minutes : int
        Granularity of predictions in minutes.
    horizon_intervals : int
        Total number of steps in the prediction horizon.
    forecasts : List[FilmForecast]
        List of per-film forecast data objects.
    """

    generated_at: datetime
    interval_minutes: int
    horizon_intervals: int
    forecasts: List[FilmForecast]


def _dummy_forecast(film_id: str, start: datetime, n: int) -> List[ForecastPoint]:
    """Generate a deterministic pseudo-forecast based on daily/weekly seasonality.

    Applies daily harmonic variations, weekly cycles, static film base offsets,
    and Gaussian noise to compute projected stream starts and 95% confidence bounds.

    Parameters
    ----------
    film_id : str
        Target film identifier used to seed the random generator.
    start : datetime
        UTC timestamp for the first interval in the forecast series.
    n : int
        Total number of consecutive intervals to generate (e.g., 96 for 24h).

    Returns
    -------
    List[ForecastPoint]
        A sequence of interval forecasts with integer stream counts and bounds.
    """
    seed = abs(hash(film_id)) % (2**32)
    rng = np.random.default_rng(seed)
    base = 50 + (seed % 200)
    hours = np.arange(n) * (INTERVAL_MINUTES / 60.0)
    daily = 1.0 + 0.4 * np.sin(2 * np.pi * (hours - 6) / 24.0)
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * hours / (24 * 7))
    noise = rng.normal(0, 3, size=n)

    # Clip at zero and calculate confidence interval bounds
    yhat = np.clip(base * daily * weekly + noise, 0, None)
    sigma = 1.5 * np.sqrt(yhat + 1)

    # Apply floor and convert to integer stream counts
    yhat_floor = np.floor(yhat).astype(int)
    lower_floor = np.floor(np.maximum(0.0, yhat - 1.96 * sigma)).astype(int)
    upper_floor = np.floor(yhat + 1.96 * sigma).astype(int)

    return [
        ForecastPoint(
            interval_start=start + timedelta(minutes=INTERVAL_MINUTES * i),
            predicted_streams=int(yhat_floor[i]),
            lower=int(lower_floor[i]),
            upper=int(upper_floor[i]),
        )
        for i in range(n)
    ]


@app.post(
    "/v1/predictions/capacity",
    response_model=ForecastResponse,
    tags=["Forecasting"],
    operation_id="predict_capacity_endpoint",
    summary="Fetch 24h capacity prognosis",
)
def predict_capacity(req: Optional[ForecastRequest] = None) -> ForecastResponse:
    """Predict 24-hour stream-start capacity for requested film assets.

    Accepts single film identifiers or lists of identifiers, returning 15-minute
    resolution time series data suitable for downstream autoscaling engines.

    Parameters
    ----------
    req : Optional[ForecastRequest], default=None
        Request payload containing requested `film_ids` and start timestamp.
        Defaults to `ForecastRequest()` if missing.

    Returns
    -------
    ForecastResponse
        Validated 24-hour capacity predictions across all target films.

    Raises
    ------
    HTTPException
        If any requested film identifier does not exist in the active catalog (422).
    """
    if req is None:
        req = ForecastRequest()

    if isinstance(req.film_ids, list):
        target_films = req.film_ids
    elif isinstance(req.film_ids, str) and req.film_ids.lower() == "all":
        target_films = FILMS
    else:
        target_films = ["film_0001"]

    # Validate film_ids existence
    unknown = [f for f in target_films if f not in FILMS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown film_ids: {unknown[:5]}")

    now = datetime.now(timezone.utc)
    forecasts = [
        FilmForecast(
            film_id=fid,
            forecast_start=req.forecast_start,
            forecast_end=req.forecast_start
            + timedelta(minutes=INTERVAL_MINUTES * HORIZON_INTERVALS),
            interval_minutes=INTERVAL_MINUTES,
            model_version="dummy-global-regressor-v0",
            feature_version="features-v1",
            feature_watermark=now - timedelta(minutes=2),
            data_freshness_seconds=120,
            forecast_source="ml_model",
            points=_dummy_forecast(fid, req.forecast_start, HORIZON_INTERVALS),
        )
        for fid in target_films
    ]

    return ForecastResponse(
        generated_at=now,
        interval_minutes=INTERVAL_MINUTES,
        horizon_intervals=HORIZON_INTERVALS,
        forecasts=forecasts,
    )


@app.get("/healthz/live")
def live() -> dict:
    """Check application liveness for Kubernetes liveness probes.

    Returns
    -------
    dict
        Status payload indicating API runtime viability.
    """
    return {"status": "ok"}


@app.get("/healthz/ready")
def ready() -> dict:
    """Check application readiness and backend dependency connectivity.

    Returns
    -------
    dict
        Status payload with health indicators for Redis, Postgres, and Model stores.
    """
    # In production: check Redis, Postgres, model artifact availability.
    return {
        "status": "ok",
        "dependencies": {"redis": "ok", "postgres": "ok", "model": "ok"},
    }
