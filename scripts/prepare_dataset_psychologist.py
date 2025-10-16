from datasets import load_dataset
import os
import pandas as pd

# Carga el dataset directamente de HuggingFace
ds = load_dataset("jkhedri/psychology-dataset")
data = ds["train"]

# Rutas de guardado
PROCESSED_CSV_PATH = "data/processed/bertherapy_dataset.csv"
RESPONSES_TXT_PATH = "data/processed/response_candidates.txt"

rows = []
all_responses = set()

n = len(data["question"])

for i in range(n):
    q = data["question"][i]

    # Respuesta buena (label 1)
    rows.append({"question": q, "response": data["response_j"][i], "label": 1})
    all_responses.add(data["response_j"][i])

    # Respuesta tóxica (label 0)
    rows.append({"question": q, "response": data["response_k"][i], "label": 0})
    all_responses.add(data["response_k"][i])

    # Respuesta buena de otra pregunta (distractor, label 0)
    j = (i + 1) % n
    if j == i:
        j = (i + 2) % n
    rows.append({"question": q, "response": data["response_j"][j], "label": 0})
    all_responses.add(data["response_j"][j])

# Guarda el CSV
os.makedirs(os.path.dirname(PROCESSED_CSV_PATH), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(PROCESSED_CSV_PATH, index=False, encoding="utf-8")

# Guarda el archivo de respuestas únicas
with open(RESPONSES_TXT_PATH, "w", encoding="utf-8") as f:
    for resp in sorted(all_responses):
        f.write(resp.replace("\n", " ") + "\n")

print(f"Dataset listo en: {PROCESSED_CSV_PATH}")
print(f"Lista de respuestas guardada en: {RESPONSES_TXT_PATH}")
