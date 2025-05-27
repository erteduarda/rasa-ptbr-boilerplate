import os
import subprocess
import tempfile
from io import BytesIO

from fastapi import FastAPI, Request, HTTPException
from telegram import Bot, Update
import whisper
import httpx
from gtts import gTTS

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
RASA_REST_URL    = os.getenv("RASA_REST_URL") 
WHISPER_MODEL    = os.getenv("WHISPER_MODEL", "small")

if not TELEGRAM_TOKEN or not RASA_REST_URL:
    raise RuntimeError("É preciso definir TELEGRAM_TOKEN e RASA_URL no ambiente")

bot   = Bot(token=TELEGRAM_TOKEN)
model = whisper.load_model(WHISPER_MODEL)
app   = FastAPI()

@app.post("/webhooks/telegram/webhook")
async def receive_update(request: Request):
    payload = await request.json()
    update  = Update.de_json(payload, bot)
    chat_id = update.effective_chat.id

    is_voice = bool(update.message and update.message.voice)
    if is_voice:
        file_id = update.message.voice.file_id
        with tempfile.NamedTemporaryFile(suffix=".oga", delete=False) as ogg:
            file = await bot.get_file(file_id)
            await file.download_to_drive(ogg.name)
            wav_path = ogg.name + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", ogg.name, wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        result = model.transcribe(wav_path)
        user_text = result.get("text", "").strip()
    elif update.message and update.message.text:
        user_text = update.message.text.strip()
    else:
        raise HTTPException(400, "Mensagem sem texto nem voz")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            RASA_REST_URL,
            json={"sender": str(chat_id), "message": user_text},
            timeout=10.0
        )
        resp.raise_for_status()
        responses = resp.json()

    for msg in responses:
        text = msg.get("text")
        if not text:
            continue
        if is_voice:
            clean_text = " ".join(text.splitlines())
            tts = gTTS(clean_text, lang="pt")
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            await bot.send_audio(chat_id=chat_id, audio=mp3_fp)
        else:
            await bot.send_message(chat_id=chat_id, text=text)

    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
