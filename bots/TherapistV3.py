from bots.PersonV3 import Person

class Therapist(Person):
    def __init__(self):
        self.dir_pool = "/home/silvina/proyectos/BERTherapy/data/processed/response_candidates_full_therapist_v3.txt"
        self.dir_model = "/home/silvina/proyectos/BERTherapy/models/bert_psychologist_vV3_FULL_full"
        super().__init__(self.dir_model, self.dir_pool, max_pool_size=15000, n_clusters=100)
    
    def respond(self, emotion, sentiment, context, input_text):
        response = self.get_best_response(emotion, sentiment, context, input_text)
        return response