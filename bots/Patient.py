from bots.Person import Person


class Patient(Person):
    def __init__(self):
        self.dir_pool = "/home/silvina/proyectos/BERTherapy/data/processed/response_candidates_patient_improved.txt"
        self.dir_model = (
            "/home/silvina/proyectos/BERTherapy/models/bert_patient_vH_improved"
        )
        super().__init__(self.dir_model, self.dir_pool,max_pool_size=5000, n_clusters=20)

    def respond(self, problem, emotion, context, input_text):
        return self.get_best_response(problem, emotion, context, input_text)
