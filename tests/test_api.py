"""Integration test suite for the Film Stream Forecasting API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.anyio
async def test_forecast_shape() -> None:
    """Test the structure and volume of capacity forecast predictions.

    Verifies that posting a valid list of film IDs returns an HTTP 200 response
    with the expected horizon length (96 intervals), correct count of film
    forecasts, and appropriate metadata sources.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the API response status is not 200, or if response payload structure
        does not match the 24-hour / 96-interval horizon expectations.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        r = await ac.post(
            "/v1/predictions/capacity",
            json={"film_ids": ["film_0001", "film_0042"]},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["horizon_intervals"] == 96
    assert len(body["forecasts"]) == 2
    for f in body["forecasts"]:
        assert len(f["points"]) == 96
        assert f["forecast_source"] == "ml_model"


@pytest.mark.anyio
async def test_unknown_film_rejected() -> None:
    """Test handling of invalid or non-existent film identifiers.

    Verifies that requests containing unknown film IDs are rejected with an
    HTTP 422 Unprocessable Entity status code.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the response status code is not 422.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        r = await ac.post("/v1/predictions/capacity", json={"film_ids": ["nope"]})
    assert r.status_code == 422


@pytest.mark.anyio
async def test_health_check() -> None:
    """Test the application liveness health probe endpoint.

    Verifies that GET requests to `/healthz/live` return HTTP 200 with a status
    payload of `{"status": "ok"}`.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the status code is not 200 or the returned JSON payload is incorrect.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        r = await ac.get("/healthz/live")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
