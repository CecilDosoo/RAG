from __future__ import annotations

import json
import urllib.parse
import urllib.request


def _wmo_condition_label(code: int) -> str | None:
    """Short phrase for Open-Meteo / WMO weather_code (helps when precip mm is zero but it's still raining)."""
    if code in (51, 53, 55):
        return "drizzle (WMO code)"
    if code in (56, 57, 66, 67):
        return "freezing rain or ice pellets (WMO code)"
    if code in (61, 63, 65, 80, 81, 82):
        return "rain or showers (WMO code)"
    if code in (71, 73, 75, 77, 85, 86):
        return "snow (WMO code)"
    if code in (95, 96, 99):
        return "thunderstorm (WMO code)"
    return None


def _weather_for_city(city: str) -> str:
    """Return one line about current conditions, or an error-friendly string."""
    city = city.strip()
    if not city:
        return ""

    try:
        q = urllib.parse.quote(city)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1"
        with urllib.request.urlopen(geo_url, timeout=12) as r:
            geo = json.loads(r.read().decode())
        results = geo.get("results") or []
        if not results:
            return f"Weather: no match for '{city}'."

        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        label = results[0].get("name", city)

        wx_url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,precipitation,weather_code"
        )
        with urllib.request.urlopen(wx_url, timeout=12) as r:
            wx = json.loads(r.read().decode())
        cur = wx.get("current") or {}
        t = cur.get("temperature_2m")
        h = cur.get("relative_humidity_2m")
        p = cur.get("precipitation")
        code = cur.get("weather_code")
        parts = [f"near {label}"]
        if t is not None:
            parts.append(f"~{t}°C")
        if h is not None:
            parts.append(f"humidity ~{h}%")
        if code is not None:
            wc = _wmo_condition_label(int(code))
            if wc:
                parts.append(wc)
        if p is not None and float(p) > 0:
            parts.append(f"precip (recent window) {p} mm")
        return "Weather now: " + ", ".join(parts) + "."
    except Exception as exc:
        return f"Weather: unavailable ({exc})."


def build_situation(city: str, indoor_outdoor: str, surface: str) -> str:
    """Combine optional weather + setting + surface into one block for the model / retrieval."""
    lines = []
    if city and city.strip() and city != "(not set)":
        w = _weather_for_city(city)
        if w:
            lines.append(w)

    io = (indoor_outdoor or "").strip()
    if io and io != "(not set)":
        lines.append(f"Setting: {io}")

    su = (surface or "").strip()
    if su and su != "(not set)":
        lines.append(f"Surface: {su}")

    return "\n".join(lines)
