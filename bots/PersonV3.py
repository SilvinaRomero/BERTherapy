from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans # para agrupar los contextos y reducir el pool
import torch
import numpy as np
import random
import os

class Person:
    def __init__(self, dir_model, dir_pool, max_pool_size=10000, n_clusters=100): # ~ pool de 50 respuestas
        self.dir_model = dir_model
        self.dir_pool= dir_pool
        self.max_pool_size = max_pool_size
        self.n_clusters = n_clusters
        self.responses = self.setPool()
        self.kmeans = None
        self.model = None
        self.response_clusters = {}
        # Archivos para gestionar respuestas usadas
        self.used_responses_file = self._get_used_responses_file()
        self.used_responses = self._load_used_responses()

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
    
    def _get_used_responses_file(self):
        """Determina el archivo de respuestas usadas según el tipo de persona"""
        class_name = self.__class__.__name__.lower()
        if class_name == "therapist":
            return "/home/silvina/proyectos/BERTherapy/used_reponses_therapist.txt"
        elif class_name == "patient":
            return "/home/silvina/proyectos/BERTherapy/used_reponses_patient.txt"
        else:
            return "/home/silvina/proyectos/BERTherapy/used_responses_generic.txt"
    
    def _load_used_responses(self):
        """Carga las respuestas ya usadas desde el archivo"""
        if os.path.exists(self.used_responses_file):
            with open(self.used_responses_file, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        return set()
    
    def _save_used_response(self, response, cluster_responses=None):
        """Guarda una respuesta usada y sus similares del cluster en el archivo"""
        responses_to_save = [response]
        
        # Si se proporcionan las respuestas del cluster, calcular similitudes solo de ese cluster
        if cluster_responses:
            similar_responses = self._find_similar_in_cluster(response, cluster_responses)
            responses_to_save.extend(similar_responses)
        
        # Guardar todas las respuestas (original + similares)
        with open(self.used_responses_file, 'a', encoding='utf-8') as f:
            for resp in responses_to_save:
                if resp not in self.used_responses:  # Evitar duplicados
                    self.used_responses.add(resp)
                    f.write(resp + '\n')
    
    def _find_similar_in_cluster(self, selected_response, cluster_responses, threshold=0.6):
        """Encuentra respuestas similares solo dentro del cluster actual"""
        if not cluster_responses or len(cluster_responses) <= 1:
            return []
        
        # Verificar que la respuesta seleccionada esté en el cluster
        if selected_response not in cluster_responses:
            print(f"⚠️  Respuesta seleccionada no encontrada en cluster para {self.__class__.__name__}")
            return []
        
        # Calcular embeddings solo del cluster actual
        cluster_embeddings = self._get_embeddings(cluster_responses)
        selected_idx = cluster_responses.index(selected_response)
        
        similar_responses = []
        for i, response in enumerate(cluster_responses):
            if i != selected_idx:  # No comparar consigo misma
                # Calcular similitud coseno
                dot_product = np.dot(cluster_embeddings[selected_idx], cluster_embeddings[i])
                norm_selected = np.linalg.norm(cluster_embeddings[selected_idx])
                norm_current = np.linalg.norm(cluster_embeddings[i])
                similarity = dot_product / (norm_selected * norm_current)
                
                # Si es muy similar, agregarla
                if similarity > threshold:
                    similar_responses.append(response)
        
        return similar_responses
    
    def _filter_used_responses(self, candidate_responses):
        """Filtra las respuestas ya usadas de la lista de candidatos"""
        return [resp for resp in candidate_responses if resp not in self.used_responses]
    
    def _reset_used_responses(self):
        """Reinicia el archivo de respuestas usadas cuando se agotan todas las respuestas"""
        self.used_responses.clear()
        if os.path.exists(self.used_responses_file):
            os.remove(self.used_responses_file)
        print(f"✅ Archivo de respuestas usadas reiniciado para {self.__class__.__name__}")
    
    
    
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
    
    def get_best_response(self, emotion, sentiment, context, input_text):
        """Encuentra la mejor respuesta usando clustering"""
        
        # 1. Obtener embedding del contexto + input con emotion y sentiment
        query_text = f"{emotion} [SEP] {sentiment} [SEP] {context} [SEP] {input_text}"
        query_embedding = self._get_embeddings([query_text])
        
        # 2. Predecir el cluster más cercano
        closest_cluster = self.kmeans.predict(query_embedding)[0]
        
        # 3. Obtener candidatos solo de ese cluster
        candidate_indices = self.response_clusters[closest_cluster]
        candidate_responses = [self.responses[i] for i in candidate_indices]
        
        # 4. Filtrar respuestas ya usadas (incluye similares pre-calculadas)
        available_responses = self._filter_used_responses(candidate_responses)
        
        # 5. Si no hay respuestas disponibles en este cluster, buscar en otros clusters
        if not available_responses:
            # Buscar en todos los clusters si no hay respuestas disponibles
            all_candidate_responses = self.responses
            available_responses = self._filter_used_responses(all_candidate_responses)
            
            # Si aún no hay respuestas disponibles, reiniciar el archivo de respuestas usadas
            if not available_responses:
                print(f"⚠️  Todas las respuestas han sido usadas para {self.__class__.__name__}. Reiniciando...")
                self._reset_used_responses()
                available_responses = candidate_responses  # Usar las respuestas del cluster original
        
        # 6. Buscar la mejor respuesta entre las disponibles
        best_response, best_score = self.find_best_response(context, input_text, available_responses)
        
        # Guardar la respuesta usada junto con las respuestas del cluster para calcular similitudes
        # Usar available_responses si best_response está ahí, sino usar candidate_responses
        cluster_for_similarity = available_responses if best_response in available_responses else candidate_responses
        self._save_used_response(best_response, cluster_for_similarity)
        
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