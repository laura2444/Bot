import logging
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import os
load_dotenv()  # carga .env en variables de entorno (NO subir .env al repo)


# ---- CONFIGURACIÓN ----
logging.basicConfig(level=logging.INFO)

# Carga de modelos locales
vectorizer = joblib.load("modelos_guardados/tfidf_vectorizer.pkl")
nb_model = joblib.load("modelos_guardados/naive_bayes_model.pkl")
tokenizer = AutoTokenizer.from_pretrained("modelos_guardados/distilbert_model")
llm_model = AutoModelForSequenceClassification.from_pretrained("modelos_guardados/distilbert_model")


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN not found in environment. Set it before running the bot.")


# ---- FUNCIONES DEL BOT ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I'm your assistant.\n\n"
        "Send me a message and I’ll tell you if it’s **SPAM or not**, using:\n"
        "1️⃣ TF-IDF + Naive Bayes Model\n"
        "2️⃣ LLM Model (DistilBERT fine-tuned)\n\n"
        "💡 Note: I was trained on **English data**, so please send texts in English."
    )

async def analizar_texto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    texto = update.message.text.strip().lower()

    # --- Respuestas básicas ---
    saludos = [ "good morning", "good afternoon", "good evening"]
    despedidas = ["bye", "goodbye", "see you", "take care"]
    agradecimientos = ["thanks", "thank you", "thx"]

    if any(palabra in texto for palabra in saludos):
        await update.message.reply_text(" Hello there! We are going to classify those texts!")
        return
    elif any(palabra in texto for palabra in despedidas):
        await update.message.reply_text(" Bye! Have a great day!")
        return
    elif any(palabra in texto for palabra in agradecimientos):
        await update.message.reply_text(" You’re welcome!")
        return

    # --- TF-IDF + NB ---
    X_tfidf = vectorizer.transform([texto])
    pred_nb = nb_model.predict(X_tfidf)[0]
    etiqueta_nb = "SPAM" if pred_nb == 1 else "NO SPAM"

    # --- LLM ---
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = llm_model(**inputs)
    pred_llm = torch.argmax(outputs.logits, dim=1).item()
    etiqueta_llm = "SPAM" if pred_llm == 1 else "NO SPAM"

    respuesta = (
        f"📩 *Text:* {texto}\n\n"
        f"🔹 TF-IDF + Naive Bayes → {etiqueta_nb}\n"
        f"🔸 DistilBERT → {etiqueta_llm}"
    )

    await update.message.reply_text(respuesta)

# ---- MAIN ----
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analizar_texto))

    print("🤖 Bot en ejecución...")
    app.run_polling()

if __name__ == "__main__":
    main()
