import json
import os
import config

# GUARDRAIL IN CODE: only these keys can ever be saved, whatever the AI tries to do
ALLOWED_KEYS = {
    "usual_budget": "Usual maximum price per share in rupees (a number)",
    "preferred_sectors": "Sectors the user is interested in (short text)",
    "risk_comfort": "low, medium or high",
    "notes": "Another short, lasting preference about how answers should look",
}


def load_profile():
    if not os.path.exists(config.PROFILE_FILE):
        return {}
    try:
        with open(config.PROFILE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write(profile):
    with open(config.PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def save_preference(key, value):
    """TOOL: saves one preference after checking it."""
    key = str(key).strip()
    if key not in ALLOWED_KEYS:
        return {"error": f"Cannot save '{key}'. Allowed keys: {list(ALLOWED_KEYS)}"}

    value = str(value).strip()[:200]
    if not value:
        return {"error": "Empty value."}

    if key == "usual_budget":
        try:
            value = float(value.replace(",", "").replace("₹", "").replace("Rs", "").strip())
        except ValueError:
            return {"error": "usual_budget must be a number."}
        if value <= 0:
            return {"error": "usual_budget must be above zero."}
    elif key == "risk_comfort":
        value = value.lower()
        if value not in ("low", "medium", "high"):
            return {"error": "risk_comfort must be low, medium or high."}

    profile = load_profile()
    profile[key] = value
    _write(profile)
    return {"saved": {key: value}}


def forget_all():
    if os.path.exists(config.PROFILE_FILE):
        os.remove(config.PROFILE_FILE)


def profile_text():
    """Readable version of the saved preferences, or an empty string."""
    profile = load_profile()
    return "\n".join(f"- {k}: {v}" for k, v in profile.items() if k in ALLOWED_KEYS)


if __name__ == "__main__":
    print(save_preference("usual_budget", "₹150"))
    print(save_preference("risk_comfort", "very risky"))     # should be rejected
    print(save_preference("bank_password", "abc123"))        # should be rejected
    print(profile_text())
    forget_all()
    print("After forget_all:", repr(profile_text()))