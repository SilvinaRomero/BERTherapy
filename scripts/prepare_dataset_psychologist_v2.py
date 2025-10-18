from datasets import load_dataset
import os
import pandas as pd
import json
import sys
import random

# Cargar el dataset directamente de HuggingFace
ds = load_dataset("thu-coai/esconv")
# print(ds["train"][0]["text"])
# print("_____________________")
# print(ds["train"][0]['text'].keys())
# sys.exit()
data_train = ds["train"]
data_test = ds["test"]

# Rutas de guardado
PROCESSED_CSV_PATH_THERAPIST = "data/processed/bertherapy_dataset_therapist.csv"
PROCESSED_CSV_PATH_PATIENT = "data/processed/bertherapy_dataset_patient.csv"
RESPONSES_TXT_PATH_THERAPIST = "data/processed/response_candidates_therapist.txt"
RESPONSES_TXT_PATH_PATIENT = "data/processed/response_candidates_patient.txt"

rows_therapist = []  # para guardar los input-response-label terapeuta
rows_patient = []  # para guardar los input-response-label paciente
all_responses_therapist = set()  # para los pools terapeuta
all_responses_patient = set()  # para los pools paciente

count_experiencies = len(data_train)
print(f"TOTAL EXPERIENCIAS: {count_experiencies}")
# for i in range(count_experiencies):
for i in range(3):
    parsed_train = json.loads(data_train[i]["text"])
    count_dialogs = len(parsed_train["dialog"])
    # si el diálogo es demasiado corto o demasiado largo (demaciado contexto puede dar error), saltamos
    if count_dialogs < 2 or count_dialogs > 40:
        continue

    print(f"TOTAL DIALOGOS: {count_dialogs}")
    print("_______________________________")
    # break
    context = ""  # para guardar el contexto del dialogos completo por cada situacion

    for j in range(count_dialogs - 1):
        current = parsed_train["dialog"][j]
        next_msg = parsed_train["dialog"][j + 1]

        if (
            parsed_train["dialog"][j]["speaker"] == "usr"
            and parsed_train["dialog"][j + 1]["speaker"] == "sys"
        ):
            ## si es usario-> terapeuta
            patient_response = parsed_train["dialog"][j]["text"]
            therapist_response = parsed_train["dialog"][j + 1]["text"]
            print("Paciente:", patient_response)
            print("Terapeuta:", therapist_response)
            print("-----")

            ## cuando encuentra par usuario->terapeuta, guardamos el registro y el contexto del dialogo para cada uno.
            rows_therapist.append({
                "context": context.strip(),
                "input": patient_response,
                "response": therapist_response,
                "label": 1
            })
            all_responses_therapist.add(therapist_response)

            #### aquí hay que buscar el ultimo input del terapeuta label 1 que está guardado en el csv
            if len(rows_therapist) > 1 and rows_therapist[-2]["label"] == 1:
                last_therapist_response = rows_therapist[-2]["response"]
                rows_patient.append({
                    "context": context.strip(),
                    "input": last_therapist_response,
                    "response": patient_response,
                    "label": 1
                })
                all_responses_patient.add(patient_response)

        elif (
            (parsed_train["dialog"][j]["speaker"] == "usr"
             and parsed_train["dialog"][j + 1]["speaker"] == "usr")
            or
            (parsed_train["dialog"][j]["speaker"] == "sys"
             and parsed_train["dialog"][j + 1]["speaker"] == "sys")
        ):
            # para otros casos, guardamos responses erroneos
            anyone_response = parsed_train["dialog"][j]["text"]
            another_response = parsed_train["dialog"][j + 1]["text"]
            rows_therapist.append({
                "context": context.strip(),
                "input": anyone_response,
                "response": another_response,
                "label": 0
            })
            rows_patient.append({
                "context": context.strip(),
                "input": another_response,
                "response": anyone_response,
                "label": 0
            })
            all_responses_therapist.add(anyone_response)
            all_responses_patient.add(another_response)

        # 🔹 después de procesar, agregas el turno actual al contexto
        context += f"[{current['speaker'].upper()}]: {current['text']} "

    print('_________________________________________________________________')

    break

## añadir mas datos label 0
all_good_responses_therapist = [
    r["response"] for r in rows_therapist if r["label"] == 1
]
all_good_responses_patient = [r["response"] for r in rows_patient if r["label"] == 1]

# Contar positivos y negativos actuales para balancear el dataset
pos_therapist = [r for r in rows_therapist if r["label"] == 1]
neg_therapist = [r for r in rows_therapist if r["label"] == 0]
pos_patient = [r for r in rows_patient if r["label"] == 1]
neg_patient = [r for r in rows_patient if r["label"] == 0]

missing_neg_therapist = max(0, len(pos_therapist) - len(neg_therapist))
missing_neg_patient = max(0, len(pos_patient) - len(neg_patient))

print(
    f"Therapist: {len(pos_therapist)} positivos, {len(neg_therapist)} negativos → faltan {missing_neg_therapist}"
)
print(
    f"Patient: {len(pos_patient)} positivos, {len(neg_patient)} negativos → faltan {missing_neg_patient}"
)

for _ in range(missing_neg_therapist):
    row = random.choice(pos_therapist)
    fake_resp = random.choice(all_good_responses_therapist)
    while fake_resp == row["response"]:
        fake_resp = random.choice(all_good_responses_therapist)
    rows_therapist.append(
        {
            "context": row["context"],
            "input": row["input"],
            "response": fake_resp,
            "label": 0,
        }
    )

for _ in range(missing_neg_patient):
    row = random.choice(pos_patient)
    fake_resp = random.choice(all_good_responses_patient)
    while fake_resp == row["response"]:
        fake_resp = random.choice(all_good_responses_patient)
    rows_patient.append(
        {
            "context": row["context"],
            "input": row["input"],
            "response": fake_resp,
            "label": 0,
        }
    )
    print("✅ Negativos añadidos hasta equilibrar datasets.")

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
