"""Currency conversion via exchangerate-api (keyed) or open.er-api.com (keyless)."""

from __future__ import annotations

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from ..cache import cached
from ..config import get_settings
from ._common import error, http_client, safe_call, to_json


@cached("fx_rates", ttl_hours=12)
def _rates(base: str) -> dict:
    key = get_settings().exchangerate_api_key
    url = (
        f"https://v6.exchangerate-api.com/v6/{key}/latest/{base}"
        if key
        else f"https://open.er-api.com/v6/latest/{base}"
    )
    with http_client() as c:
        r = c.get(url)
        r.raise_for_status()
    data = r.json()
    rates = data.get("conversion_rates") or data.get("rates")
    if data.get("result") != "success" or not rates:
        return error(f"Rate lookup failed for {base}: {data.get('error-type', 'unknown')}")
    return rates


def convert(amount: float, from_currency: str, to_currency: str) -> dict:
    src, dst = from_currency.upper(), to_currency.upper()
    if src == dst:
        return {"amount": amount, "currency": dst, "rate": 1.0}
    rates = safe_call(_rates, src)
    if "error" in rates:
        return rates
    if dst not in rates:
        return error(f"Unknown currency: {dst}")
    rate = rates[dst]
    return {"amount": round(amount * rate, 2), "currency": dst, "rate": rate}


class CurrencyInput(BaseModel):
    amount: float = Field(..., description="Amount to convert")
    from_currency: str = Field(..., description="ISO code, e.g. 'EUR'")
    to_currency: str = Field(..., description="ISO code, e.g. 'USD'")


class CurrencyTool(BaseTool):
    name: str = "convert_currency"
    description: str = "Convert an amount between currencies using live rates. Returns JSON."
    args_schema: type[BaseModel] = CurrencyInput

    def _run(self, amount: float, from_currency: str, to_currency: str) -> str:
        return to_json(convert(amount, from_currency, to_currency))
