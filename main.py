from bots.Therapist import Therapist
from bots.Patient import Patient

emotions = [
    "anxiety",
    "depression",
    "sadness",
    "fear",
    "disgust",
    "anger",
    "shame",
    "guilt",
    "jealousy",
    "nervousness",
    "pain",
]


# Definir problemas por defecto según emoción
problem_options = {
    "anxiety": [
        "academic pressure",
        "alcohol abuse",
        "appearance anxiety",
        "breakup with partner",
        "conflict with parents",
        "job crisis",
        "ongoing depression",
        "procrastination",
        "problems with friends",
        "sleep problems",
    ],
    "depression": [
        "academic pressure",
        "alcohol abuse",
        "appearance anxiety",
        "breakup with partner",
        "job crisis",
        "ongoing depression",
        "problems with friends",
        "procrastination",
        "school bullying",
        "sleep problems",
    ],
    "sadness": [
        "academic pressure",
        "alcohol abuse",
        "appearance anxiety",
        "breakup with partner",
        "conflict with parents",
        "issues with children",
        "job crisis",
        "ongoing depression",
        "problems with friends",
        "sleep problems",
    ],
    "fear": [
        "academic pressure",
        "breakup with partner",
        "issues with children",
        "issues with parents",
        "job crisis",
        "ongoing depression",
        "problems with friends",
        "sleep problems",
    ],
    "disgust": [
        "appearance anxiety",
        "breakup with partner",
        "issues with children",
        "job crisis",
        "ongoing depression",
        "procrastination",
        "problems with friends",
    ],
    "anger": [
        "academic pressure",
        "breakup with partner",
        "conflict with parents",
        "issues with children",
        "job crisis",
        "ongoing depression",
        "problems with friends",
        "sleep problems",
    ],
    "shame": [
        "academic pressure",
        "alcohol abuse",
        "breakup with partner",
        "job crisis",
        "ongoing depression",
        "procrastination",
        "problems with friends",
    ],
    "guilt": ["academic pressure", "breakup with partner"],
    "jealousy": ["problems with friends"],
    "nervousness": ["academic pressure", "conflict with parents"],
    "pain": ["breakup with partner"],
}


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

    # Selección de problema
    print(f"\nOpciones de problema para {emotion}:")
    for i, prob in enumerate(problem_options[emotion]):
        print(f"{i}: {prob}")
    choice_problem = input(
        f"Selecciona un número (0-{len(problem_options[emotion])-1}) para el problema: "
    ).strip()
    problem = (
        problem_options[emotion][0]
        if not choice_problem
        or not choice_problem.isdigit()
        or int(choice_problem) < 0
        or int(choice_problem) >= len(problem_options[emotion])
        else problem_options[emotion][int(choice_problem)]
    )

    # Solicitar primer input del terapeuta
    first_input = input("\nTERAPEUTA: ").strip()
    while not first_input:
        print("El mensaje del terapeuta no puede estar vacío.")
        first_input = input("TERAPEUTA: ").strip()

    # Inicializar context con el primer mensaje del terapeuta
    context = f"[SEP] [SYS]: {first_input} "
    resp = patient.respond(
        problem, emotion, context, first_input
    )  # Pasar problem, emotion, context, input
    print("\nUSER: " + resp)
    context += f"[SEP] [USR]: {resp} "
    turno = 1  # Empieza con el terapeuta (turno impar)
    ## una vez dado el contexto, el problema y la emocion, ronda pregunta/respuesta
    while turno <= 10:
        if turno % 2 != 0:  # Turno impar: Terapeuta
            # print("contexto para el terapeuta")
            # print(context)
            resp_t = therapist.respond(problem, emotion,context, resp)  # Terapeuta responde
            context += f"[SEP] [SYS]: {resp_t} "
            print(f"\nTERAPEUTA: {resp_t}")
        else:  # Turno par: Paciente
            # print("contexto para el usuario")
            # print(context)
            resp = patient.respond(problem, emotion,context, resp_t)  # Paciente responde
            context += f"[SEP] [USR]: {resp} "
            print("\nUSER: " + resp)

        turno += 1


if __name__ == "__main__":
    run_conversation()
