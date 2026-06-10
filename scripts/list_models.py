import os
import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"c:\Users\ASUS\Desktop\Projects\Kapruka-Nelum Agent\.env")
gemini_key = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}"
try:
    r = httpx.get(url, timeout=10.0)
    print("HTTP Status:", r.status_code)
    if r.status_code == 200:
        models = r.json().get("models", [])
        for m in models:
            print(f"- {m['name']} ({m.get('displayName')})")
    else:
        print("Error:", r.text[:500])
except Exception as e:
    print("Exception:", e)
