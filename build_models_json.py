#!/usr/bin/env python3
import os, json, urllib.request

GROQ_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

def http_get(url, headers=None, timeout=12):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")

def groq_models():
    if not GROQ_KEY:
        return []
    data = http_get("https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type":"application/json"})
    js = json.loads(data)
    ids = [m.get("id","") for m in js.get("data", []) if m.get("id")]
    return sorted(set(ids))

def gemini_models():
    if not GEMINI_KEY:
        return []
    data = http_get(f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_KEY}")
    js = json.loads(data)
    out = []
    for m in js.get("models", []):
        name = m.get("name","")               # e.g. models/gemini-2.0-flash
        methods = set(m.get("supportedGenerationMethods", []))
        if "generateContent" in methods and name.startswith("models/"):
            out.append(name.split("/",1)[1])  # drop 'models/'
    return sorted(set(out))

GROQ_DENY = {"llama3-8b-8192"}
GEMINI_DENY = {"gemini-1.5-flash"}

GROQ_FALLBACK = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
GEMINI_FALLBACK = ["gemini-2.0-flash", "gemini-2.0-pro"]

def rank_prefer(candidates, deny, fallback):
    live = [m for m in candidates if m not in deny]
    return live or fallback

def main():
    gm = groq_models()
    mm = gemini_models()
    groq_prefer = rank_prefer(gm, GROQ_DENY, GROQ_FALLBACK)
    gemini_prefer = rank_prefer(mm, GEMINI_DENY, GEMINI_FALLBACK)

    payload = {
        "version": 1,
        "providers": {
            "groq": {
                "prefer": groq_prefer,
                "deny": sorted(GROQ_DENY),
                "default_temperature": 0.1
            },
            "gemini": {
                "prefer": gemini_prefer,
                "deny": sorted(GEMINI_DENY),
                "default_temperature": 0.3
            }
        },
        "refresh_seconds": 86400
    }
    with open("models.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("models.json written, counts: groq=%d gemini=%d" % (len(groq_prefer), len(gemini_prefer)))

if __name__ == "__main__":
    main()
