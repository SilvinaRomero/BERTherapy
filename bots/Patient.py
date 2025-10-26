from bots.Person import Person

class Patient(Person):
    def __init__(self):
        self.dir_pool = "/home/silvina/proyectos/BERTherapy/data/processed/response_candidates_patient_full.txt"
        self.dir_model = "/home/silvina/proyectos/BERTherapy/models/bert_patient_v1.0"  # Usar modelo existente por ahora
        super().__init__(self.dir_model, self.dir_pool, max_pool_size=10000, n_clusters=100)

    def respond(self, emotion, sentiment, context, input_text):
        response = self.get_best_response(emotion, sentiment, context, input_text)
        return response
