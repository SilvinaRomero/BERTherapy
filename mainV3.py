from bots.TherapistV3 import Therapist
from bots.PatientV3 import Patient

# Emociones del dataset v3
emotions = [
    "joy",
    "sadness", 
    "anger",
    "fear",
    "surprise",
    "love"
]

# Sentiments del dataset v3
sentiments = [
    "hopeful",
    "sad",
    "annoyed", 
    "prepared",
    "apprehensive",
    "unknown"
]


# En el dataset v3 no usamos problemas específicos, solo emotion y sentiment


def run_conversation():
    therapist = Therapist()
    therapist.initParameters()

    patient = Patient()
    patient.initParameters()

    # Selección de emoción
    print("\nSelecciona la emoción del paciente:")
    for i, em in enumerate(emotions):
        print(f"{i}: {em}")
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

    # Selección de sentiment
    print(f"\nSelecciona el sentiment del paciente:")
    for i, sent in enumerate(sentiments):
        print(f"{i}: {sent}")
    choice_sentiment = input(
        f"Selecciona un número (0-{len(sentiments)-1}) para el sentiment: "
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
    context = f"[SYS]: {first_input}"
    resp = patient.respond(
        emotion, sentiment, context, first_input
    )  # Pasar emotion, sentiment, context, input
    print("\nUSER: " + resp)
    context += f" [SEP] [USR]: {resp}"
    turno = 1  # Empieza con el terapeuta (turno impar)
    ## una vez dado el contexto, la emotion y el sentiment, ronda pregunta/respuesta
    while turno <= 10:
        if turno % 2 != 0:  # Turno impar: Terapeuta
            # print("contexto para el terapeuta")
            # print(context)
            resp_t = therapist.respond(emotion, sentiment, context, resp)  # Terapeuta responde
            context += f" [SEP] [SYS]: {resp_t}"
            print(f"\nTERAPEUTA: {resp_t}")
        else:  # Turno par: Paciente
            # print("contexto para el usuario")
            # print(context)
            resp = patient.respond(emotion, sentiment, context, resp_t)  # Paciente responde
            context += f" [SEP] [USR]: {resp}"
            print("\nUSER: " + resp)

        turno += 1


if __name__ == "__main__":
    run_conversation()
