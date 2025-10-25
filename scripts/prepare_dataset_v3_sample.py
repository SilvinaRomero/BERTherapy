from datasets import load_dataset
import os
import pandas as pd
import json
import re
import sys
import random

# Cargar el dataset directamente de HuggingFace
ds = load_dataset("ego02/mental-health-chatbot-training")
data_train = ds["train"]

# Rutas de guardado
PROCESSED_CSV_PATH_THERAPIST = "data/processed/bertherapy_dataset_dialogues_therapist_v3_sample.csv"
PROCESSED_CSV_PATH_PATIENT = "data/processed/bertherapy_dataset_dialogues_patient_v3_sample.csv"
RESPONSES_TXT_PATH_THERAPIST = "data/processed/response_candidates_dialogues_therapist_v3_sample.txt"
RESPONSES_TXT_PATH_PATIENT = "data/processed/response_candidates_dialogues_patient_v3_sample.txt"

rows_therapist = []  # para guardar los input-response-label del terapeuta
rows_patient = []  # para guardar los input-response-label del paciente
all_responses_therapist = set()  # para el pool de respuestas del terapeuta
all_responses_patient = set()  # para el pool de respuestas del paciente

def extract_emotion_sentiment(text):
    """Extrae emotion y sentiment del texto [INST]"""
    emotion_match = re.search(r'\[emotion: (\w+)\]', text)
    sentiment_match = re.search(r'\[sentiment: (\w+)\]', text)
    
    emotion = emotion_match.group(1) if emotion_match else "unknown"
    sentiment = sentiment_match.group(1) if sentiment_match else "unknown"
    
    return emotion, sentiment

def parse_conversation(text):
    """Parsea una conversación completa del nuevo formato"""
    # Dividir por </s> para obtener turnos individuales
    turns = text.split('</s>')
    conversation = []
    
    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue
            
        # Buscar patrones [INST] ... [/INST] respuesta
        inst_match = re.search(r'<s>\[INST\](.*?)\[/INST\](.*?)(?=<s>|$)', turn, re.DOTALL)
        if inst_match:
            inst_content = inst_match.group(1).strip()
            response = inst_match.group(2).strip()
            
            # Extraer emotion y sentiment
            emotion, sentiment = extract_emotion_sentiment(inst_content)
            
            # Limpiar el contenido del usuario (quitar etiquetas)
            user_content = re.sub(r'\[emotion: \w+\]', '', inst_content)
            user_content = re.sub(r'\[sentiment: \w+\]', '', user_content)
            user_content = user_content.strip()
            
            conversation.append({
                'user': user_content,
                'therapist': response,
                'emotion': emotion,
                'sentiment': sentiment
            })
    
    return conversation

def generate_negative_samples(conversation, emotion, sentiment, role='therapist'):
    """Genera muestras negativas mezclando respuestas de diferentes conversaciones"""
    negative_samples = []
    
    if role == 'therapist':
        # Para el terapeuta: mezclar respuestas de diferentes conversaciones
        if len(conversation) > 1 and all_responses_therapist:
            for i in range(len(conversation) - 1):
                current = conversation[i]
                
                # Crear contexto hasta el turno actual
                context = ""
                for j in range(i + 1):
                    if j == 0:
                        context += f"[SYS]: {conversation[j]['therapist']}"
                    else:
                        context += f" [SEP] [USR]: {conversation[j]['user']} [SEP] [SYS]: {conversation[j]['therapist']}"
                
                # Respuesta incorrecta: usar respuesta de otra conversación
                wrong_response = random.choice(list(all_responses_therapist))
                
                negative_samples.append({
                    'context': context,
                    'input': current['user'],
                    'response': wrong_response,
                    'label': 0,
                    'emotion': emotion,
                    'sentiment': sentiment
                })
    
    elif role == 'patient':
        # Para el paciente: mezclar respuestas de diferentes conversaciones
        if len(conversation) > 1 and all_responses_patient:
            # Generar negativos para el primer turno del paciente
            if len(conversation) >= 1:
                wrong_response = random.choice(list(all_responses_patient))
                negative_samples.append({
                    'context': "",
                    'input': conversation[0]['therapist'],
                    'response': wrong_response,
                    'label': 0,
                    'emotion': emotion,
                    'sentiment': sentiment
                })
            
            # Generar negativos para turnos siguientes del paciente
            context = ""
            for i in range(len(conversation) - 1):
                if i == 0:
                    context = f"[SYS]: {conversation[i]['therapist']}"
                else:
                    context += f" [SEP] [USR]: {conversation[i]['user']} [SEP] [SYS]: {conversation[i]['therapist']}"
                
                # Respuesta incorrecta: usar respuesta de otra conversación
                wrong_response = random.choice(list(all_responses_patient))
                
                negative_samples.append({
                    'context': context,
                    'input': conversation[i]['therapist'],
                    'response': wrong_response,
                    'label': 0,
                    'emotion': emotion,
                    'sentiment': sentiment
                })
    
    return negative_samples

count_experiences = len(data_train)
print(f"TOTAL EXPERIENCIAS: {count_experiences}")

# Procesar solo una muestra pequeña para probar (primeras 100 experiencias)
SAMPLE_SIZE = 100
print(f"Procesando muestra de {SAMPLE_SIZE} experiencias...")

for i in range(min(SAMPLE_SIZE, count_experiences)):
    if i % 10 == 0:
        print(f"Procesando experiencia {i}/{SAMPLE_SIZE}")
    
    text = data_train[i]['text']
    conversation = parse_conversation(text)
    
    if len(conversation) < 2:  # Saltar conversaciones muy cortas
        continue
    
    # Procesar cada turno de la conversación
    context = ""
    is_first_turn = True
    
    for j, turn in enumerate(conversation):
        user_msg = turn['user']
        therapist_msg = turn['therapist']
        emotion = turn['emotion']
        sentiment = turn['sentiment']
        
        if is_first_turn:
            # Primer turno: contexto vacío, input del terapeuta
            context = f"[SYS]: {therapist_msg}"
            
            # Datos del paciente (respuesta al primer mensaje del terapeuta)
            rows_patient.append({
                'context': "",
                'input': therapist_msg,
                'response': user_msg,
                'label': 1,
                'emotion': emotion,
                'sentiment': sentiment
            })
            all_responses_patient.add(user_msg)
            is_first_turn = False
        else:
            # Turnos siguientes
            context += f" [SEP] [USR]: {user_msg}"
            
            # Datos del terapeuta
            rows_therapist.append({
                'context': context,
                'input': user_msg,
                'response': therapist_msg,
                'label': 1,
                'emotion': emotion,
                'sentiment': sentiment
            })
            all_responses_therapist.add(therapist_msg)
            
            # Actualizar contexto para el siguiente turno del paciente
            context += f" [SEP] [SYS]: {therapist_msg}"
            
            # Datos del paciente (si hay siguiente turno)
            if j < len(conversation) - 1:
                next_turn = conversation[j + 1]
                rows_patient.append({
                    'context': context,
                    'input': therapist_msg,
                    'response': next_turn['user'],
                    'label': 1,
                    'emotion': emotion,
                    'sentiment': sentiment
                })
                all_responses_patient.add(next_turn['user'])
    
    # Generar muestras negativas para balancear el dataset
    if len(conversation) > 1:
        emotion = conversation[0]['emotion']
        sentiment = conversation[0]['sentiment']
        
        # Generar muestras negativas para terapeuta
        negative_therapist = generate_negative_samples(conversation, emotion, sentiment, 'therapist')
        rows_therapist.extend(negative_therapist)
        
        # Generar muestras negativas para paciente
        negative_patient = generate_negative_samples(conversation, emotion, sentiment, 'patient')
        rows_patient.extend(negative_patient)

print(f"\nProcesamiento completado!")
print(f"Total muestras terapeuta: {len(rows_therapist)}")
print(f"Total muestras paciente: {len(rows_patient)}")

# Guardar los CSV
os.makedirs(os.path.dirname(PROCESSED_CSV_PATH_THERAPIST), exist_ok=True)
df_therapist = pd.DataFrame(rows_therapist)
df_therapist.to_csv(PROCESSED_CSV_PATH_THERAPIST, index=False, encoding="utf-8")

os.makedirs(os.path.dirname(PROCESSED_CSV_PATH_PATIENT), exist_ok=True)
df_patient = pd.DataFrame(rows_patient)
df_patient.to_csv(PROCESSED_CSV_PATH_PATIENT, index=False, encoding="utf-8")

# Guardar los archivos de respuestas únicas
with open(RESPONSES_TXT_PATH_THERAPIST, "w", encoding="utf-8") as f:
    for resp in sorted(all_responses_therapist):
        f.write(resp.replace("\n", " ") + "\n")

with open(RESPONSES_TXT_PATH_PATIENT, "w", encoding="utf-8") as f:
    for resp in sorted(all_responses_patient):
        f.write(resp.replace("\n", " ") + "\n")

print(f"\nDataset terapeuta guardado en: {PROCESSED_CSV_PATH_THERAPIST}")
print(f"Dataset paciente guardado en: {PROCESSED_CSV_PATH_PATIENT}")
print(f"Respuestas terapeuta guardadas en: {RESPONSES_TXT_PATH_THERAPIST}")
print(f"Respuestas paciente guardadas en: {RESPONSES_TXT_PATH_PATIENT}")

# Mostrar estadísticas
print(f"\n=== ESTADÍSTICAS ===")
print(f"Terapeuta - Total: {len(df_therapist)}, Positivos: {len(df_therapist[df_therapist['label']==1])}, Negativos: {len(df_therapist[df_therapist['label']==0])}")
print(f"Paciente - Total: {len(df_patient)}, Positivos: {len(df_patient[df_patient['label']==1])}, Negativos: {len(df_patient[df_patient['label']==0])}")

# Mostrar distribución de emociones
print(f"\n=== DISTRIBUCIÓN DE EMOCIONES ===")
print("Terapeuta:")
print(df_therapist['emotion'].value_counts())
print("\nPaciente:")
print(df_patient['emotion'].value_counts())

# Mostrar algunas muestras
print(f"\n=== MUESTRAS DEL DATASET ===")
print("Terapeuta (primeras 3 filas):")
print(df_therapist.head(3))
print("\nPaciente (primeras 3 filas):")
print(df_patient.head(3))
