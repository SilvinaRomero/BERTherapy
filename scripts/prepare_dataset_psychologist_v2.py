from datasets import load_dataset
import os
import pandas as pd
import json
import sys
import random
import re

# Cargar el dataset directamente de HuggingFace
ds = load_dataset("thu-coai/esconv")
data_train = ds["train"]

# Rutas de guardado
PROCESSED_CSV_PATH_THERAPIST = "data/processed/bertherapy_dataset_therapist.csv"
PROCESSED_CSV_PATH_PATIENT = "data/processed/bertherapy_dataset_patient.csv"
RESPONSES_TXT_PATH_THERAPIST = "data/processed/response_candidates_therapist.txt"
RESPONSES_TXT_PATH_PATIENT = "data/processed/response_candidates_patient.txt"

rows_therapist = []  # para guardar los input-response-label terapeuta
rows_patient = []  # para guardar los input-response-label paciente
all_responses_therapist = set()  # para los pools terapeuta
all_responses_patient = set()  # para los pools paciente

# funcion que recibe la lista, la trata como dataframe, para invertir las respuesta y devolver el último response
# del terapeuta cuyo label era 1 (input para el usuario)
def getLastResponse(data):
    df = pd.DataFrame(data)
    filtrado = df[df["label"] == 1]["response"]
    if not filtrado.empty:
        return filtrado.iloc[-1]
    return ""

# eliminar con regex la última aparicion del texto en el contexto (temp_context)
def _remove_last_re(haystack: str, needle: str) -> str:
    pattern = re.compile(re.escape(needle) + r"(?!.*" + re.escape(needle) + r")", re.S)
    return pattern.sub("", haystack, count=1)

count_experiencies = len(data_train)
print(f"TOTAL EXPERIENCIAS: {count_experiencies}")

for i in range(count_experiencies):
    print(f"Batch: {i}\n")
    # encodear el text
    parsed_train = json.loads(data_train[i]["text"])
    count_dialogs = len(parsed_train["dialog"])
    # si el diálogo es demasiado corto o demasiado largo (demaciado contexto puede dar error), saltamos
    if count_dialogs < 2 or count_dialogs > 40:
        continue

    print(f"TOTAL DIALOGOS: {count_dialogs}")
    print("_______________________________")
    context = ""  # para guardar el contexto del dialogos completo por cada situacion
    is_first_time_for_user = True # controla que la primera vez en cada experiencia, el usuario recibe un input vacio

    for j in range(count_dialogs - 1):
        current = parsed_train["dialog"][j]
        next_msg = parsed_train["dialog"][j + 1]
        context += f"[{current['speaker'].upper()}]: {current['text']} " # en cada iteración se guarda el texto actual en el contexto
        if (
            parsed_train["dialog"][j]["speaker"] == "usr"
            and parsed_train["dialog"][j + 1]["speaker"] == "sys"
        ):
            ## si es usario-> terapeuta: label = 1
            patient_response = parsed_train["dialog"][j]["text"]
            therapist_response = parsed_train["dialog"][j + 1]["text"]

            print("Paciente:", patient_response)
            print("Terapeuta:", therapist_response)

            ## contexto temporal para no incluir su propio response en el contexto

            temp_context_patient = _remove_last_re(
                context, f"[{current['speaker'].upper()}]: {current['text']} "
            )
            #### aquí hay que buscar el ultimo input del terapeuta label 1 que está guardado en rows_therapist
            if is_first_time_for_user:
                last_therapist_response = ""
                is_first_time_for_user = False
            else:
                last_therapist_response = getLastResponse(rows_therapist)

            rows_patient.append(
                {
                    "context": temp_context_patient,
                    "input": last_therapist_response,
                    "response": patient_response,
                    "label": 1,
                }
            )
            all_responses_patient.add(patient_response)
            temp_context_therapist = _remove_last_re(
                context, f"[{next_msg['speaker'].upper()}]: {next_msg['text']} "
            )
            rows_therapist.append(
                {
                    "context": temp_context_therapist,
                    "input": patient_response,
                    "response": therapist_response,
                    "label": 1,
                }
            )
            all_responses_therapist.add(therapist_response)
            last_therapist_response = "" # resetear la última respuesta en cada bucle
        elif (
            parsed_train["dialog"][j]["speaker"] == "usr"
            and parsed_train["dialog"][j + 1]["speaker"] == "usr"
        ) or (
            parsed_train["dialog"][j]["speaker"] == "sys"
            and parsed_train["dialog"][j + 1]["speaker"] == "sys"
        ):
            # para otros casos, guardamos responses erroneos
            anyone_response = parsed_train["dialog"][j]["text"]
            another_response = parsed_train["dialog"][j + 1]["text"]
            rows_therapist.append(
                {
                    "context": context,
                    "input": anyone_response,
                    "response": another_response,
                    "label": 0,
                }
            )
            rows_patient.append(
                {
                    "context": context,
                    "input": another_response,
                    "response": anyone_response,
                    "label": 0,
                }
            )
            all_responses_therapist.add(anyone_response)
            all_responses_patient.add(another_response)

    print("_________________________________________________________________")


# Guardar los CSV
# terapeuta
os.makedirs(os.path.dirname(PROCESSED_CSV_PATH_THERAPIST), exist_ok=True)
df = pd.DataFrame(rows_therapist)
df.to_csv(PROCESSED_CSV_PATH_THERAPIST, index=False, encoding="utf-8")
# paciente
os.makedirs(os.path.dirname(PROCESSED_CSV_PATH_PATIENT), exist_ok=True)
df = pd.DataFrame(rows_patient)
df.to_csv(PROCESSED_CSV_PATH_PATIENT, index=False, encoding="utf-8")

# Guardar los archivo de respuestas únicas
# terapeuta
with open(RESPONSES_TXT_PATH_THERAPIST, "w", encoding="utf-8") as f:
    for resp in sorted(all_responses_therapist):
        f.write(resp.replace("\n", " ") + "\n")
# paciente
with open(RESPONSES_TXT_PATH_PATIENT, "w", encoding="utf-8") as f:
    for resp in sorted(all_responses_patient):
        f.write(resp.replace("\n", " ") + "\n")

print(f"Dataset terapeuta listo en: {PROCESSED_CSV_PATH_THERAPIST}")
print(f"Dataset paciente listo en: {PROCESSED_CSV_PATH_PATIENT}")
print(f"Lista de respuestas terapueta guardada en: {RESPONSES_TXT_PATH_THERAPIST}")
print(f"Lista de respuestas paciente guardada en: {RESPONSES_TXT_PATH_PATIENT}")
