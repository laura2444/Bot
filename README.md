#  Bot Clasificador de Spam (TF-IDF + LLM)

Este proyecto implementa un **bot de Telegram** que clasifica mensajes de texto como **SPAM** o **NO SPAM** utilizando dos enfoques distintos de **Procesamiento de Lenguaje Natural (PLN)**:

1️. Modelo tradicional con TF-IDF + Naive Bayes**  
2️. Modelo avanzado basado en LLM (DistilBERT de Hugging Face)**  

Ambos modelos fueron **entrenados y guardados previamente en Google Colab** y se integran en este bot para comparar sus resultados de manera práctica.

<img width="2142" height="1616" alt="image" src="https://github.com/user-attachments/assets/f814730b-40c7-42bb-827e-46d2c58d6ac8" />

---

## Descripción del Proyecto

El objetivo de este bot es demostrar la diferencia entre un modelo tradicional de PLN y un modelo basado en un **Lenguaje de Gran Escala (LLM)**.  

Cuando el usuario envía un mensaje en inglés, el bot analiza el texto y responde con el resultado de ambos modelos, mostrando si el mensaje es considerado **SPAM o NO SPAM** por cada uno.

---

##  Estructura del Proyecto

├── modelos_guardados/ # Carpeta con los modelos entrenados en Colab
│ ├── tfidf_vectorizer.pkl # Vectorizador TF-IDF
│ ├── naive_bayes_model.pkl # Modelo Naive Bayes
│ └── distilbert_model/ # Modelo preentrenado DistilBERT (LLM)
├── bot.py # Código principal del bot de Telegram
├── .env # Archivo con el token del bot (NO subir a GitHub)
├── requirements.txt # Dependencias necesarias
└── README.md # Documentación del proyecto

## Entorno virtual

En Windows:

python -m venv venv
venv\Scripts\activate

## Instalar dependencias
pip install -r requirements.txt

## Configurar el Token de Telegram

Crea un archivo llamado .env en la raíz del proyecto con el siguiente contenido:

TELEGRAM_TOKEN=tu_token_aquí

##Ejecución del Bot

python bot.py
