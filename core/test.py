import os
import requests


def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ API key not found")
        return

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }

    prompt = "Explain AI in one sentence."

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)

        print("Status Code:", response.status_code)
        print("Raw Response:", response.text)

        if response.status_code != 200:
            print("❌ API call failed")
            return

        data = response.json()

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            print("\n✅ LLM OUTPUT:\n", text)
        except Exception:
            print("❌ Failed to parse response")

    except Exception as e:
        print("❌ Exception occurred:", e)


if __name__ == "__main__":
    test_gemini()