from bots.Person import Person
import os
import json

# Obtener la ruta del directorio del proyecto (BERTherapy)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Obtener la configuración del terapeuta (guardada con optuna)
with open(os.path.join(PROJECT_ROOT, "models", "config", "therapist.json"), "r") as file:
    config_therapist = json.load(file)


class Therapist(Person):
    def __init__(self):
        # ruta del modelo, selección del pool de respuestas y número de clusters
        self.dir_pool = os.path.join(PROJECT_ROOT, "data", "processed", "response_candidates_therapist_full.txt")
        self.dir_model = os.path.join(PROJECT_ROOT, "models", f"bert_therapist_v{config_therapist['version']}")
        super().__init__(self.dir_model, self.dir_pool, max_pool_size=15000, n_clusters=100)
    
    def respond(self, emotion, sentiment, context, input_text):
        response = self.get_best_response(emotion, sentiment, context, input_text)
        return response