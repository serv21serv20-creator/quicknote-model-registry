#!/usr/bin/env python3
import os, json, urllib.request, urllib.error, sys

GROQ_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

class HttpError(RuntimeError):
    pass

def http_get(url, headers=None, timeout=12):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code = getattr(r, "status", 200)
            body = r.read().decode("utf-8")
            if code >= 400:
                raise HttpError(f"HTTP {code} for {url}")
            return body
    except urllib.error.HTTPError as e:
        raise HttpError(f"HTTP {e.code} for {url}") from e
    except urllib.error.URLError as e:
        raise HttpError(f"URL error for {url}: {e}") from e

def groq_models():
    if not GROQ_KEY:
        print("WARN: GROQ_API_KEY missing; skipping live Groq models.")
        return []
    data = http_get("https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type":"application/json"})
    js = json.loads(data)
    ids = [m.get("id","") for m in js.get("data", []) if m.get("id")]
    return sorted(set(ids))

def gemini_models():
    if not GEMINI_KEY:
        print("WARN: GEMINI_API_KEY missing; skipping live Gemini models.")
        return []
    # نستخدم v1beta هنا لقائمة النماذج (قد تختلف نقاط generateContent لاحقًا داخل التطبيق)
    data = http_get(f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_KEY}")
    js = json.loads(data)
    out = []
    for m in js.get("models", []):
        name = m.get("name","")               # e.g. models/gemini-2.0-flash
        methods = set(m.get("supportedGenerationMethods", []))
        if "generateContent" in methods and name.startswith("models/"):
            out.append(name.split("/",1)[1])  # drop 'models/'
    return sorted(set(out))

# قوائم منع/بدائل
GROQ_DENY = {"llama3-8b-8192"}
GEMINI_DENY = {"gemini-1.5-flash"}  # مثال: لو واجهنا مشاكل معه

GROQ_FALLBACK = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
GEMINI_FALLBACK = ["gemini-2.0-flash", "gemini-2.0-pro"]

def rank_prefer(candidates, deny, fallback):
    live = [m for m in candidates if m not in deny]
    return live or fallback

def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def main():
    try:
        gm = []
        mm = []
        try:
            gm = groq_models()
            print(f"INFO: Groq live models: {len(gm)}")
        except Exception as e:
            print(f"ERROR: groq_models failed: {e}")

        try:
            mm = gemini_models()
            print(f"INFO: Gemini live models: {len(mm)}")
        except Exception as e:
            print(f"ERROR: gemini_models failed: {e}")

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
        write_json("models.json", payload)
        print("models.json written, counts: groq=%d gemini=%d" % (len(groq_prefer), len(gemini_prefer)))
        return 0
    except Exception as e:
        # Fallback نهائي حتى لو حدث خطأ غير متوقّع
        print(f"FATAL: unexpected error: {e}")
        fallback = {
            "version": "fallback",
            "providers": {
                "groq": {
                    "prefer": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
                    "deny": [],
                    "default_temperature": 0.1
                },
                "gemini": {
                    "prefer": ["gemini-1.5-pro", "gemini-1.5-flash"],
                    "deny": [],
                    "default_temperature": 0.3
                }
            },
            "refresh_seconds": 86400
        }
        write_json("models.json", fallback)
        print("models.json written with safe fallback.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
