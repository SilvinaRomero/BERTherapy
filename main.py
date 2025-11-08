from bots.Therapist import Therapist
from bots.Patient import Patient
import os
import time
from deep_translator import GoogleTranslator

# Obtener la ruta del directorio del proyecto (BERTherapy)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def _sanitize_text(text):
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    if isinstance(text, str):
        return text.encode("utf-8", "surrogatepass").decode("utf-8", errors="ignore")
    return str(text)


def translate(text, source, target):
    """Traduce el texto para que sea en español"""
    safe_text = _sanitize_text(text)
    translator = GoogleTranslator(source=source, target=target)
    translated = translator.translate(safe_text)
    return _sanitize_text(translated)

# Emociones del dataset
emotions = [
    "joy",
    "sadness", 
    "anger",
    "fear",
    "surprise",
    "love"
]

# Sentimientos del dataset
sentiments = [
    "hopeful",
    "sad",
    "annoyed", 
    "prepared",
    "apprehensive",
    "unknown"
]


# En el dataset no usamos problemas específicos, solo emociones y sentimientos
def cleanup_used_responses():
    """Limpia los archivos de respuestas usadas al terminar la conversación
       Se crean en Person.py al crear los clusters de respuestas.
    """
    files_to_clean = [
        os.path.join(PROJECT_ROOT, "used_reponses_therapist.txt"),
        os.path.join(PROJECT_ROOT, "used_reponses_patient.txt")
    ]
    
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            os.remove(file_path)
            # print(f"✅ Archivo limpiado: {os.path.basename(file_path)}")

def run_conversation():
    # inicializar los bots
    therapist = Therapist()
    therapist.initParameters() # crear clusters de respuestas de terapeuta entre otros parámetros

    patient = Patient()
    patient.initParameters() # crear clusters de respuestas de paciente entre otros parámetros

    # Selección de emoción
    print("\nSelecciona la emoción del paciente:")
    for i, em in enumerate(emotions):
        print(f"{i}: {translate(em, 'en', 'es')}")
    choice_emotion = input(
        f"Selecciona un número (0-{len(emotions)-1}) para la emoción: "
    ).strip()
    emotion = (
        emotions[0]
        if not choice_emotion
        or not choice_emotion.isdigit()
        or int(choice_emotion) < 0
        or int(choice_emotion) >= len(emotions)
        else emotions[int(choice_emotion)]
    )

    # Selección de sentimiento
    print(f"\nSelecciona el sentimiento del paciente:")
    for i, sent in enumerate(sentiments):
        print(f"{i}: {translate(sent, 'en', 'es')}")
    choice_sentiment = input(
        f"Selecciona un número (0-{len(sentiments)-1}) para el sentimiento: "
    ).strip()
    sentiment = (
        sentiments[0]
        if not choice_sentiment
        or not choice_sentiment.isdigit()
        or int(choice_sentiment) < 0
        or int(choice_sentiment) >= len(sentiments)
        else sentiments[int(choice_sentiment)]
    )

    # Solicitar primer input del terapeuta
    first_input = input("\nTERAPEUTA: ").strip()
    while not first_input:
        print("El mensaje del terapeuta no puede estar vacío.")
        first_input = input("TERAPEUTA: ").strip()

    # Inicializar context con el primer mensaje del terapeuta
    context = f"[SYS]: {translate(first_input, 'es', 'en')}"

    # con el primer contexto, la emocion, el sentimiento y el primer input del terapeuta, el paciente responde
    resp = patient.respond(
        emotion, sentiment, context, first_input
    )
    print("\nUSER: " + translate(resp, 'en', 'es'))
    context += f" [SEP] [USR]: {resp}" # [SEP] para separar los contextos: emoción [SEP] sentimiento [SEP] paciente [SEP] terapeuta = contexto
    turno = 1  # Empieza con el terapeuta
    # una vez dado el contexto, la emotion y el sentiment, y obtenido el primer input del paciente, ronda pregunta/respuesta
    while turno <= 10:
        if turno % 2 != 0:  # Turno impar: Terapeuta
            resp_t = therapist.respond(emotion, sentiment, context, resp)  # Terapeuta responde
            context += f" [SEP] [SYS]: {resp_t}"
            print(f"\nTERAPEUTA: {translate(resp_t, 'en', 'es')}")
        else:  # Turno par: Paciente
            resp = patient.respond(emotion, sentiment, context, resp_t)  # Paciente responde
            context += f" [SEP] [USR]: {resp}"
            print("\nUSER: " + translate(resp, 'en', 'es'))

        turno += 1
        time.sleep(1)
    print("\n")
    
    # Limpiar archivos de respuestas usadas al terminar
    cleanup_used_responses()


if __name__ == "__main__":
    run_conversation()
