from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans # para agrupar los contextos y reducir el pool
import torch
import numpy as np
import random

class Person:
    def __init__(self, dir_model, dir_pool, max_pool_size=5000, n_clusters=100): # ~ pool de 50 respuestas
        self.dir_model = dir_model
        self.dir_pool= dir_pool
        self.max_pool_size = max_pool_size
        self.n_clusters = n_clusters
        self.responses = self.setPool()
        self.kmeans = None
        self.model = None
        self.response_clusters = {}

    def initParameters(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.dir_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        self.setPool()
        self._cluster_responses()

    # "data/processed/response_candidates.txt"
    def setPool(self):
        with open(self.dir_pool) as f:
            responses = [line.strip() for line in f if line.strip()]
        if len(responses) > self.max_pool_size: # muestra de pool
            responses = random.sample(responses, self.max_pool_size)
        self.responses= responses
    def tokenize_batch(self, text_a_batch, response_batch):
        """
        Tokeniza un lote de texto y respuestas, preparando las entradas para el modelo.
        """
        encodings = self.tokenizer(
            text_a_batch,
            response_batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.model.device) for k, v in encodings.items()}
        if 'labels' in encodings:
            del encodings['labels']
        return encodings

    def predict_scores(self, encodings):
        """
        Realiza la predicción y devuelve los scores de probabilidad.
        """
        with torch.no_grad():
            logits = self.model(**encodings).logits
            probs = torch.softmax(logits, dim=1)
            return probs[:, 1].cpu().tolist()
    
    def batchify(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    def _process_batch(self, context, input_text, candidate_responses, batch_size=16):
        """
        Procesa un lote de candidatos y devuelve sus scores.
        """
        scores = []
        for response_batch in self.batchify(candidate_responses, batch_size):
            batch_size = len(response_batch)
            text_a_batch = [context + " [SEP] " + input_text] * batch_size
            encodings = self.tokenize_batch(text_a_batch, response_batch)
            scores.extend(self.predict_scores(encodings))
        return scores
    

    ## aqui en el context ya se incluye la emocion y el problema
    def find_best_response(self, context, input_text, candidate_responses, max_iterations=1):
        """
        Encuentra la mejor respuesta refinando la selección si hay demasiados empates.
        Usa un enfoque top-k para optimizar.
        """
        # Obtener scores iniciales
        scores = self._process_batch(context, input_text, candidate_responses)
        if not scores:
            return "No puedo responder a eso ahora", 0.0

        # Convertir a numpy array para optimización
        scores_np = np.array(scores)
        indices = np.arange(len(scores))

        iteration = 0
        while iteration < max_iterations:
            # Encontrar el máximo score y los índices correspondientes
            max_score = np.max(scores_np)
            best_indices = indices[scores_np == max_score]

            if len(best_indices) <= 10:  # Si ya tenemos <= 10, salir
                break
            elif len(best_indices) > 100:  # Si hay más de 100, reducir a top-k
                k = min(100, len(best_indices))  # Límite inicial de 100
                top_k_indices = indices[np.argpartition(scores_np, -k)[-k:]]
                candidate_subset = [candidate_responses[i] for i in top_k_indices]
                scores_np = np.array(self._process_batch(context, input_text, candidate_subset))
                indices = np.arange(len(candidate_subset))
            else:
                break  # Si está entre 10 y 100, no necesitamos más iteraciones

            iteration += 1

        # Seleccionar la mejor de las restantes
        if len(best_indices) > 10:
            top_10_indices = np.argpartition(scores_np, -10)[-10:]
            best_idx = np.random.choice(top_10_indices)
            best_response = candidate_subset[best_idx]
        else:
            best_idx = np.random.choice(best_indices) if len(best_indices) > 1 else best_indices[0]
            best_response = candidate_responses[best_idx]

        best_score = scores_np[best_idx] if len(best_indices) > 10 else max_score

        return best_response, float(best_score)
    
    def get_best_response(self, problem, emotion, context, input_text):
        """Encuentra la mejor respuesta usando clustering"""
        
        # 1. Obtener embedding del contexto + input con problem y emotion
        query_text = f"{problem} [SEP] {emotion} [SEP] {context} [SEP] {input_text}"
        query_embedding = self._get_embeddings([query_text])
        
        # 2. Predecir el cluster más cercano
        closest_cluster = self.kmeans.predict(query_embedding)[0]
        
        # 3. Obtener candidatos solo de ese cluster
        candidate_indices = self.response_clusters[closest_cluster]
        candidate_responses = [self.responses[i] for i in candidate_indices]
        
        # 4. Buscar la mejor respuesta SOLO en ese cluster (mucho más rápido)
        best_response, best_score = self.find_best_response(context, input_text, candidate_responses)
        
        return best_response
    
    def _get_embeddings(self, texts, batch_size=32):
        """Obtiene embeddings de una lista de textos"""
        all_embeddings = []
        
        for text_batch in self.batchify(texts, batch_size):
            encodings = self.tokenizer(
                text_batch,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            encodings = {k: v.to(self.model.device) for k, v in encodings.items()}
            
            with torch.no_grad():
                # Usa el [CLS] token como embedding
                outputs = self.model.bert(**encodings)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)
        
        return np.vstack(all_embeddings)
    
    def _cluster_responses(self):
        """Agrupa las respuestas en clusters (se hace 1 vez al inicio)"""
        # print(f"Clustering {len(self.responses)} responses into {self.n_clusters} groups...")
        
        # Obtener embeddings de todas las respuestas
        response_embeddings = self._get_embeddings(self.responses)
        
        # KMeans clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=30)
        cluster_labels = self.kmeans.fit_predict(response_embeddings)
        
        # Organizar respuestas por cluster
        for idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in self.response_clusters:
                self.response_clusters[cluster_id] = []
            self.response_clusters[cluster_id].append(idx)
        # Personalizar el mensaje según la clase
        class_name = self.__class__.__name__
        print(f"Clustering complete for {class_name}! Average responses per cluster: {len(self.responses) / self.n_clusters:.1f}")