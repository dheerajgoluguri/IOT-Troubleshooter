"""
IoT Troubleshooter — Flask backend
Groq Chat Completions API + RAG over a local IoT knowledge base
"""

import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ── IoT Knowledge Base ───────────────────────────────────────────────────────
# Each entry: {"topic": str, "content": str}
KNOWLEDGE_BASE = [
    {
        "topic": "Wi-Fi connection failure",
        "content": (
            "Many IoT devices fail to connect to Wi-Fi because they only support 2.4 GHz networks, "
            "not 5 GHz. Check your router settings and ensure a 2.4 GHz SSID is available. "
            "Move the device closer to the router to improve signal strength. "
            "Restart both the router and the IoT device. Re-enter the Wi-Fi credentials carefully. "
            "Ensure the password contains no special characters that the device app might not handle."
        ),
    },
    {
        "topic": "Incorrect Wi-Fi password / changed password",
        "content": (
            "If you recently changed your Wi-Fi password or router, your IoT device still holds the "
            "old credentials. You must re-pair the device: open the manufacturer app, remove the device, "
            "then add it again using the new Wi-Fi password. Some devices require a factory reset before "
            "re-pairing. Confirm your phone is connected to the same 2.4 GHz network you want the device on."
        ),
    },
    {
        "topic": "2.4 GHz vs 5 GHz compatibility",
        "content": (
            "Most budget IoT devices — smart bulbs, plugs, sensors — only support the 2.4 GHz band. "
            "5 GHz routers have shorter range but higher speed. If your router broadcasts a single SSID "
            "for both bands (band steering), the device may land on 5 GHz and fail. "
            "Solution: split your SSIDs so 2.4 GHz and 5 GHz have different names, then connect the "
            "IoT device to the 2.4 GHz SSID explicitly."
        ),
    },
    {
        "topic": "Device offline / not responding",
        "content": (
            "When an IoT device shows as offline in the app, first confirm it has power. "
            "Check whether the Wi-Fi network is working from your phone. "
            "Restart the device by unplugging it for 10 seconds and plugging it back in. "
            "Restart the router. If the device comes online briefly then drops, the signal may be weak — "
            "move the device closer to the router or add a Wi-Fi extender. "
            "Also check for app or firmware updates."
        ),
    },
    {
        "topic": "Device not detected by app",
        "content": (
            "If the app cannot find the device during setup, ensure Bluetooth and Location permissions "
            "are enabled on your phone (many apps use BLE for initial discovery). "
            "Confirm the device is in pairing/setup mode — usually indicated by a flashing LED. "
            "Check that your phone is on 2.4 GHz Wi-Fi, not a VPN, and not on a guest network. "
            "Try uninstalling and reinstalling the manufacturer app."
        ),
    },
    {
        "topic": "Bluetooth pairing problems",
        "content": (
            "Bluetooth pairing failures are common when the device is too far away (keep within 1 metre "
            "during pairing), when another device is already paired to it, or when the device is not in "
            "pairing mode. To resolve: power cycle the IoT device, ensure it is in pairing mode "
            "(hold the button until the LED flashes rapidly), forget the device from your phone's "
            "Bluetooth settings, then scan again. Disable any Bluetooth audio devices nearby that might "
            "interfere."
        ),
    },
    {
        "topic": "Pairing failure",
        "content": (
            "Pairing failures during initial setup often occur due to: wrong network band (use 2.4 GHz), "
            "phone connected to VPN (disable it), device not in setup mode, or firewall blocking the "
            "discovery broadcast. Steps: ensure the device LED is in setup/pairing mode, disable VPN, "
            "connect your phone to 2.4 GHz, open the app and follow the add-device wizard step by step. "
            "If pairing still fails, perform a factory reset on the device and retry."
        ),
    },
    {
        "topic": "Firmware update failure",
        "content": (
            "Firmware updates can fail if the device loses power or Wi-Fi during the update, or if the "
            "app times out. Do not unplug the device while the update LED is blinking. "
            "Ensure the device is close to the router for a stable connection. "
            "If the update failed and the device is bricked, try a factory reset: hold the reset button "
            "for 10 seconds. Then re-add the device and let the firmware update complete before moving it."
        ),
    },
    {
        "topic": "Frequent disconnections",
        "content": (
            "Frequent disconnections are usually caused by weak Wi-Fi signal, IP address conflicts, or "
            "router DHCP lease expiry. Fixes: assign a static IP or DHCP reservation to the device in "
            "the router admin panel, reduce the number of devices on the 2.4 GHz channel, use a Wi-Fi "
            "extender or mesh node, and update both the device firmware and the router firmware. "
            "Also check if the router has a 'client isolation' or 'AP isolation' setting enabled — "
            "disable it for IoT devices."
        ),
    },
    {
        "topic": "Factory reset problems",
        "content": (
            "If a factory reset is not working, check the manufacturer's exact reset procedure — usually "
            "holding the reset/pair button for 5–10 seconds until the LED blinks a specific pattern. "
            "Some devices require the button to be held while powering on. "
            "If the device is completely unresponsive, try power cycling 5 times in quick succession "
            "(some devices enter reset mode this way). Check the manufacturer's support page for "
            "model-specific reset instructions."
        ),
    },
    {
        "topic": "Smart camera not connecting",
        "content": (
            "Smart cameras commonly fail to connect after a Wi-Fi password change. "
            "You must reset the camera (hold reset for 10 seconds), open the camera app, "
            "remove the old device entry, and re-add the camera using the current Wi-Fi credentials. "
            "Ensure your phone is on 2.4 GHz during setup. Grant the app Camera, Microphone, and "
            "Local Network permissions. Some cameras also require QR code scanning during setup — "
            "hold the code steady and ensure adequate lighting."
        ),
    },
    {
        "topic": "Smart plug offline",
        "content": (
            "A smart plug that goes offline is often caused by a power outage that the plug did not "
            "recover from, or a Wi-Fi credential change. Unplug and re-plug the smart plug. "
            "If it does not reconnect automatically within 60 seconds, check the app. "
            "If still offline, perform a factory reset (hold the button for 5 seconds until the "
            "LED blinks) and re-add it in the app. Assign a DHCP reservation in your router so the "
            "plug always gets the same IP address."
        ),
    },
    {
        "topic": "Smart speaker not responding",
        "content": (
            "If a smart speaker is not responding to voice commands, check: microphone mute button "
            "is not active, the device has a stable internet connection, the wake-word detection "
            "sensitivity is not set too low in the app. Restart the speaker. Check if the companion "
            "app shows the speaker as online. Re-link your account if the speaker shows as unauthorised. "
            "For persistent issues, deregister and re-register the device from the app."
        ),
    },
    {
        "topic": "IoT sensor not reporting data",
        "content": (
            "If an IoT sensor (temperature, motion, door) stops reporting data, first check the battery "
            "level in the app. Replace batteries if below 20%. Ensure the sensor is within Zigbee, "
            "Z-Wave, or Wi-Fi range of the hub. Check the hub is online. Re-pair the sensor if it "
            "has been offline for more than 24 hours. For Zigbee sensors, adding a Zigbee repeater "
            "between the sensor and hub can improve reliability."
        ),
    },
]

# ── Build TF-IDF index at startup ────────────────────────────────────────────
_kb_texts = [f"{e['topic']}. {e['content']}" for e in KNOWLEDGE_BASE]
_vectorizer = TfidfVectorizer(stop_words="english")
_kb_matrix  = _vectorizer.fit_transform(_kb_texts)
logger.info("Knowledge base indexed: %d documents", len(KNOWLEDGE_BASE))


def retrieve_context(query: str, top_k: int = 2) -> str:
    """Return the top-k most relevant KB entries as a single context string."""
    try:
        q_vec   = _vectorizer.transform([query])
        scores  = cosine_similarity(q_vec, _kb_matrix).flatten()
        indices = np.argsort(scores)[::-1][:top_k]
        chunks  = []
        for i in indices:
            if scores[i] > 0.0:
                chunks.append(f"[{KNOWLEDGE_BASE[i]['topic']}]\n{KNOWLEDGE_BASE[i]['content']}")
        return "\n\n".join(chunks) if chunks else ""
    except Exception as exc:
        logger.warning("RAG retrieval error: %s", exc)
        return ""


# ── Groq client (lazy initialisation) ────────────────────────────────────────
_groq_client = None

def get_groq_client():
    """Return a cached Groq client instance, or raise if credentials missing."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment.")

    from groq import Groq
    _groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client ready (model: %s)", GROQ_MODEL)
    return _groq_client


def build_messages(user_message: str, context: str, history: list) -> list:
    """Assemble the OpenAI-compatible messages list for the Groq Chat Completions API."""
    system_content = (
        "You are an expert IoT device troubleshooting assistant. "
        "When the user describes an IoT problem, respond ONLY using this exact structure:\n\n"
        "Problem:\n<restate the user's problem in one sentence>\n\n"
        "Likely Cause:\n<explain the most probable root cause in 1-2 sentences>\n\n"
        "Troubleshooting Steps:\n"
        "1. <step>\n2. <step>\n3. <step>\n4. <step>\n\n"
        "If unresolved:\n<suggest contacting support or additional escalation>\n\n"
        "If the user has not provided enough information, ask up to two diagnostic questions "
        "instead of guessing. Keep answers clear and beginner-friendly."
    )

    if context:
        system_content += f"\n\nRelevant knowledge base context:\n{context}"

    messages = [{"role": "system", "content": system_content}]

    for turn in history[-6:]:  # keep last 3 exchanges (6 messages)
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/style.css")
def stylesheet():
    return send_from_directory(".", "style.css")


@app.route("/health")
def health():
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    status = "ok" if not missing else "degraded"
    return jsonify({"status": status, "missing_env": missing})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    user_message = (data.get("message") or "").strip()
    history      = data.get("history") or []

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # 1. RAG retrieval
    context = retrieve_context(user_message)

    # 2. Build messages list
    messages = build_messages(user_message, context, history)

    # 3. Call Groq
    try:
        client      = get_groq_client()
        completion  = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=700,
            temperature=0.3,
        )
        answer = completion.choices[0].message.content.strip()
    except ValueError as exc:
        logger.error("Config error: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.error("Groq error: %s", exc)
        return jsonify({"error": "The AI model could not be reached. Please check your GROQ_API_KEY and try again."}), 503

    return jsonify({"reply": answer})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting IoT Troubleshooter on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
