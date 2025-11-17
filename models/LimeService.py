from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk
import os
import json
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# cargar config de terapeuta y paciente para obtener la version del modelo
with open(os.path.join(PROJECT_ROOT, "models", "config", "therapist.json"), "r") as file:
    config_therapist = json.load(file)
with open(os.path.join(PROJECT_ROOT, "models", "config", "patient.json"), "r") as file:
    config_patient = json.load(file)
# obtener la ruta del modelo de terapeuta y paciente
model_patient = os.path.join(PROJECT_ROOT, f"models/bert_patient_v{config_patient['version']}")
model_therapist = os.path.join(PROJECT_ROOT, f"models/bert_therapist_v{config_therapist['version']}")

# obtener la ruta de los datasets de terapeuta y paciente
test_dataset_patient = os.path.join(PROJECT_ROOT, f"models/bert_patient_v{config_patient['version']}_test_data")
test_dataset_therapist = os.path.join(PROJECT_ROOT, f"models/bert_therapist_v{config_therapist['version']}_test_data")

# crear el servicio de lime
class LimeService:
    def __init__(self,model_path, test_dataset_path, model_type):
        self.explainer = LimeTextExplainer(class_names=['neg', 'pos'])
        self.n_samples = 100
        self.model_type = model_type

        # cargar el dataset de test
        self.test_dataset = load_from_disk(test_dataset_path)

        ## modelos ya entrenados y cargados por la version de la configuracion
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Modo evaluación


        os.makedirs(os.path.join(PROJECT_ROOT, "lime", self.model_type), exist_ok=True)

    def predict_proba(self, texts):
        """Predecir probabilidades"""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        
        return probs.numpy()

    def get_examples(self):
        """Obtiene 1 falso positivo, 1 falso negativo, 1 correcto donde 1 == 1 y uno correcto donde 0 == 0"""
        batch_size = 32
        total = len(self.test_dataset)
        cases = {
            'false_positive': None,
            'false_negative': None,
            'correct_negative': None,
            'correct_positive': None
        }
        # buscamos por batch para no saturar la memoria
        for start in range(0, total, batch_size):
            batch_end = min(start + batch_size, total)
            batch = self.test_dataset[start:batch_end]

            texts = [
                f"{emotion} [SEP] {sentiment} [SEP] {context} [SEP] {input_text} [SEP] {response}"
                for emotion, sentiment, context, input_text, response in zip(
                    batch['emotion'],
                    batch['sentiment'],
                    batch['context'],
                    batch['input'],
                    batch['response']
                )
            ]

            probs = self.predict_proba(texts)
            print("PROBS: ", probs)
            predictions = np.argmax(probs, axis=1)
            print("PREDICTIONS: ", predictions)

            for idx, pred in enumerate(predictions):
                label = int(batch['label'][idx])
                formatted_text = texts[idx]

                # predicho = 1 real= 0 -> falso positivo
                if cases['false_positive'] is None and label == 0 and pred == 1:
                    cases['false_positive'] = {
                        'label': label,
                        'predicted': int(pred),
                        'text_formatted': formatted_text
                    }
                # predicho = 0 real= 1 -> falso negativo
                elif cases['false_negative'] is None and label == 1 and pred == 0:
                    cases['false_negative'] = {
                        'label': label,
                        'predicted': int(pred),
                        'text_formatted': formatted_text
                    }
                # predicho = 0 real= 0 -> correcto negativo
                elif cases['correct_negative'] is None and label == pred and pred == 0:
                    cases['correct_negative'] = {
                        'label': label,
                        'predicted': int(pred),
                        'text_formatted': formatted_text
                    }
                # predicho = 1 real= 1 -> correcto positivo
                elif cases['correct_positive'] is None and label == pred and pred == 1:
                    cases['correct_positive'] = {
                        'label': label,
                        'predicted': int(pred),
                        'text_formatted': formatted_text
                    }

            if all(value is not None for value in cases.values()):
                break

        return cases

    def explain_examples(self):
        """Genera explicaciones LIME para los 4 casos"""
        examples = self.get_examples()
        
        for case_name, row in examples.items():
            if row is None:
                print(f"⚠️  No hay {case_name} en el test set")
                continue
            
            print(f"\n{'='*50}")
            print(f"📌 {case_name.upper()}")
            print(f"\n🔍 Real: {row['label']} | Predicho: {row['predicted']}")
            print(f"\n🔍 Texto: {row['text_formatted'][:100]}...")
            
            # Generar explicación LIME en formato html
            exp = self.explainer.explain_instance(
                row['text_formatted'],
                self.predict_proba,
                num_features=10,
                num_samples=self.n_samples
            )
            
            # Guardar
            filename = f"{case_name}.html"
            save_path = os.path.join(PROJECT_ROOT, "lime", self.model_type, filename)
            exp.save_to_file(save_path) # formato html
            print(f"✅ Guardado: {save_path}")

# Crear una instancia por modelo
service_patient = LimeService(
    model_path=model_patient,
    test_dataset_path=test_dataset_patient,
    model_type='patient'
)
service_patient.explain_examples()

service_therapist = LimeService(
    model_path=model_therapist,
    test_dataset_path=test_dataset_therapist,
    model_type='therapist'
)
service_therapist.explain_examples()

