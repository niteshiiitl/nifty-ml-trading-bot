"""Optional Upstox OAuth helper (standalone; NOT a runtime dependency).

The live agent reads ``UPSTOX_ACCESS_TOKEN`` directly from the environment and
never imports this module. This helper is a convenience for the operator to
exchange a manual authorization code (obtained by completing the Upstox login
URL in a browser) for an access token, which they then set as
``UPSTOX_ACCESS_TOKEN``.

Usage (manual, operator-run):

    1. Open the Upstox login URL in a browser and complete login.
    2. Copy the ``code`` query param from the redirect URL.
    3. Run:

        python -c "from src.auth import exchange_auth_code; \
import os; print(exchange_auth_code('<CODE>', os.environ.get('UPSTOX_API_KEY'), \
os.environ.get('UPSTOX_API_SECRET')))"

    4. Set the printed token as UPSTOX_ACCESS_TOKEN.

There is no automated OAuth refresh in scope; the token expires daily (~03:30
IST) and the operator re-runs this flow when the system alerts on 401.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from src.config import UPSTOX_REST_BASE, redact

logger = logging.getLogger(__name__)

# Upstox OAuth2 token endpoint (authorization-code grant).
TOKEN_PATH = "/v3/login/oauth2/token"


def exchange_auth_code(
    auth_code: str,
    api_key: Optional[str],
    api_secret: Optional[str],
) -> str:
    """Exchange a manual Upstox auth code for an access token.

    Args:
        auth_code: The ``code`` from the Upstox login redirect URL.
        api_key: ``UPSTOX_API_KEY`` (client id). If None, read from the env
            inside this function call is NOT performed — the caller passes it.
        api_secret: ``UPSTOX_API_SECRET`` (client secret).

    Returns:
        The access token string. Print it and set it as ``UPSTOX_ACCESS_TOKEN``.

    Raises:
        ValueError: if ``auth_code``/``api_key``/``api_secret`` are missing.
        RuntimeError: if the Upstox token endpoint returns a non-success response.
    """
    if not auth_code:
        raise ValueError("auth_code is required (obtain it from the Upstox login redirect).")
    if not api_key or not api_secret:
        raise ValueError(
            "UPSTOX_API_KEY and UPSTOX_API_SECRET are required to exchange an "
            "auth code. Set them in the environment (see .env.example)."
        )

    url = f"{UPSTOX_REST_BASE}{TOKEN_PATH}"
    payload = {
        "code": auth_code,
        "client_id": api_key,
        "client_secret": api_secret,
        "redirect_uri": "https://127.0.0.1:5000/",
        "grant_type": "authorization_code",
    }
    logger.info("Exchanging auth code for access token (api_key=%s)...", redact(api_key, 4))
    try:
        resp = requests.post(url, data=payload, timeout=20)
    except requests.RequestException as exc:  # network/transport error
        raise RuntimeError(f"Network error contacting Upstox token endpoint: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"Upstox token endpoint returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    try:
        data = resp.json()
    except ValueError as exc:  # not JSON
        raise RuntimeError(f"Upstox token endpoint returned non-JSON: {exc}") from exc

    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"Upstox token response missing access_token: {data}")
    logger.info("Access token obtained (redacted): %s", redact(token))
    return token
