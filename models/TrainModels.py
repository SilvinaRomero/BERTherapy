from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import os
import random


class TrainModels:
    def __init__(
        self,
        dir_dataset,
        output_dir_images,
        output_dir_model,
        check_dir_model,
        num_train_epochs,
        batch_size,
        learning_rate,
        freezeLayer,
        early,
        fill_nan
    ):
        self.dir_dataset = dir_dataset
        self.output_dir_images = output_dir_images
        self.output_dir_model = output_dir_model
        self.check_dir_model = check_dir_model
        self.epochs = num_train_epochs
        self.batch = batch_size
        self.learning_rate = learning_rate
        self.freezeLayer = freezeLayer
        self.early = early
        self.fill_nan = fill_nan
        # Definir id2label y label2id aquí mismo
        self.label2id = {"pos": 1, "neg": 0}
        self.id2label = {1: "pos", 0: "neg"}

        # crear carpeta de salida de graficos
        os.makedirs(self.output_dir_images, exist_ok=True)

        self.data = None
        self.train_dataset = None
        self.test_dataset = None
        self.tokenizer = None
        self.collator = None
        self.model = None
        self.trainer = None
        self.args = None
        self.train_tok = None
        self.test_tok = None
        

    def run_all(self):
        if self.fill_nan:
            self.set_data() # iniciar una conversacion
        self.split_data()
        self.tokenize_data()
        self.set_collator()
        self.setModel()
        self.freeze_layers()
        self.setArgs()
        self.trainer_()
        self.save_show_metrics()
        self.exportModel()

    def setModel(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def freeze_layers(self):
        if self.freezeLayer == 0:
            return
        # congelar capas
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.model.bert.encoder.layer[:self.freezeLayer]:
            for param in layer.parameters():
                param.requires_grad = False

    def setArgs(self):
        use_cuda = torch.cuda.is_available()
        self.args = TrainingArguments(
            output_dir=self.check_dir_model,  # Carpeta para outputs/checkpoints
            overwrite_output_dir=True,  # Sobrescribe el output si existe
            num_train_epochs=self.epochs,  # Nº de épocas de entrenamiento
            per_device_train_batch_size=self.batch,  # Tamaño de batch en train
            per_device_eval_batch_size=self.batch,  # Tamaño de batch en validación
            eval_strategy="epoch",  # Evalúa al final de cada época
            save_strategy="epoch",  # Guarda checkpoint al final de cada época
            logging_steps=50,  # Registra logs cada 50 pasos
            learning_rate=self.learning_rate,  # Learning rate inicial
            weight_decay=0.01,  # L2 weight decay (regularización)
            fp16=use_cuda,  # Usa float16 si hay GPU compatible
            load_best_model_at_end=True,  # Carga mejor modelo (eval_loss más bajo)
            metric_for_best_model="eval_loss",  # Métrica para mejor modelo
            greater_is_better=False,  # eval_loss, así que menor es mejor
            seed=42,  # Semilla
            report_to="none",  # No reporta a Wandb u otros
            save_total_limit=3,  # Solo guarda los 3 mejores checkpoints
            save_safetensors=True,  # Guarda checkpoints en formato seguro
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean().item()
        return {"accuracy": float(acc)}

    def trainer_(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_tok,
            eval_dataset=self.test_tok,
            tokenizer=self.tokenizer,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.early)],
        )
        self.trainer.train()

    def exportModel(self):
        self.model.save_pretrained(self.output_dir_model)
        self.tokenizer.save_pretrained(self.output_dir_model)
        print(f"Modelo y tokenizer guardados en: {self.output_dir_model}")

    def save_show_metrics(self):
        self.plot_metrics(self.trainer, self.output_dir_images)
        self.plot_confusion_matrix(self.trainer, self.test_tok, self.output_dir_images)

        print("\n==> Métricas en test:", self.trainer.evaluate())

    
    def getDF(self):
        ## cargar dataset
        df = pd.read_csv(self.dir_dataset)
        df = df.drop_duplicates() #eliminar duplicados de los dos datasets
        if self.fill_nan:
                # Función local para obtener mensaje inicial
                def get_initial_message(row):
                    emotion = row["emotion"]
                    sentiment = row["sentiment"]
                    
                    # Verificar que emotion no sea None ni vacío
                    if not emotion or pd.isna(emotion):
                        raise ValueError("Emotion must be non-empty and non-null")
                    
                    # Obtener mensajes base por emoción
                    messages = self.data.get(emotion.lower(), self.data.get("default"))
                    
                    # Reemplazar [SENTIMENT] con el valor de sentiment si existe
                    if sentiment and not pd.isna(sentiment):
                        formatted_messages = [msg.replace("[SENTIMENT]", sentiment) for msg in messages]
                    else:
                        formatted_messages = messages
                    
                    # Verificar que haya mensajes
                    if not formatted_messages:
                        raise KeyError(f"No messages found for emotion {emotion}")
                    
                    return random.choice(formatted_messages)

                # Rellenar contexto e input vacíos con un mensaje inicial basado en emoción y sentimiento
                df["context"] = df.apply(lambda row: get_initial_message(row) if pd.isna(row["context"]) or row["context"] == "" else row["context"], axis=1)
                df["input"] = df.apply(lambda row: get_initial_message(row) if pd.isna(row["input"]) or row["input"] == "" else row["input"], axis=1)

        print("\nForma del dataset:")
        print(df.shape)
        print("\nValores nulos:")
        print(df.isnull().sum())
        print("\nDuplicados:")
        print(df.duplicated().sum())
        print("\nPrimeras filas:")
        print(df.head())
        return df

    def split_data(self):
        df = self.getDF()
        # df = df.iloc[0:100000] # solo 100k registros para entrenamiento.
        X = df[["context", "input", "response","emotion","sentiment"]]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("\nTamaño train:")
        print(X_train.shape, X_test.shape)
        print("\nTamaño test:")
        print(y_train.value_counts(), y_test.value_counts())
        print("*" * 80)

        # convertir a datasets
        self.train_dataset = Dataset.from_pandas(pd.concat([X_train, y_train], axis=1))
        self.test_dataset = Dataset.from_pandas(pd.concat([X_test, y_test], axis=1))

    def tokenize(self, batch):
        # Usar emotion y sentiment en lugar de problem
        text_a = [f"{e} [SEP] {s} [SEP] {c} [SEP] {i}" for e, s, c, i in zip(batch["emotion"], batch["sentiment"], batch["context"], batch["input"])]
        return self.tokenizer(
            text_a,
            batch["response"],
            truncation=True,
            padding=True,  # diferentes longitudes de texto
            max_length=128,
        )

    def tokenize_data(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.train_tok = self.train_dataset.map(self.tokenize, batched=True)
        self.test_tok = self.test_dataset.map(self.tokenize, batched=True)
        print("\ntrain-test:")
        print(self.train_tok)
        print(self.test_tok)
        print("\nMuestra de tokenización:")
        print(self.train_tok[0])

    def set_collator(self):
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)


    def plot_confusion_matrix(self, trainer, test_dataset, output_dir_images):
        """
        Genera y guarda la matriz de confusión en el test set.
        """
        preds = trainer.predict(test_dataset)
        y_true = preds.label_ids
        y_pred = np.argmax(preds.predictions, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["neg", "pos"])
        disp.plot(cmap="Blues")
        plt.title("Matriz de confusión (test set)")
        plt.savefig(os.path.join(output_dir_images, "confusion_matrix.jpg"))
        plt.close()

    def plot_metrics(self, trainer, output_dir_images):
        """
        Grafica de la evolución de loss y accuracy durante el entrenamiento.
        """
        logs = trainer.state.log_history

        # Separar métricas de entrenamiento y validación
        train_loss = []
        train_epochs = []
        eval_loss = []
        eval_acc = []
        eval_epochs = []

        for log in logs:
            # Loss de entrenamiento (al final de cada época)
            if "loss" in log and "epoch" in log and "eval_loss" not in log:
                train_loss.append(log["loss"])
                train_epochs.append(log["epoch"])

            # Métricas de validación
            if "eval_loss" in log:
                eval_loss.append(log["eval_loss"])
                eval_epochs.append(log["epoch"])

            if "eval_accuracy" in log:
                eval_acc.append(log["eval_accuracy"])

        # 📉 Gráfico de pérdida
        plt.figure(figsize=(8, 6))
        plt.plot(train_epochs, train_loss, label="Entrenamiento", marker="o")
        plt.plot(eval_epochs, eval_loss, label="Validación", marker="o")
        plt.title("Evolución de la pérdida")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir_images, "loss_curve.jpg"))
        plt.close()

        # 📈 Gráfico de accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(eval_epochs, eval_acc, label="Validación", color="orange", marker="o")
        plt.title("Evolución del accuracy")
        plt.xlabel("Época")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir_images, "accuracy_curve.jpg"))
        plt.close()

    def test_model_responses(self, emotion, sentiment, context, input_text, candidate_responses):
        """
        Evalúa múltiples respuestas candidatas y devuelve sus scores.
        """
        n = len(candidate_responses)
        responses = candidate_responses
        text_a = [f"{emotion} [SEP] {sentiment} [SEP] {context} [SEP] {input_text}" for _ in range(n)]

        encodings = self.tokenizer(
            text_a,
            responses,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.model.device) for k, v in encodings.items()}
        if 'labels' in encodings:
            del encodings['labels']
        
        with torch.no_grad():
            logits = self.model(**encodings).logits
        
        probs = torch.softmax(logits, dim=1)
        scores = probs[:, 1].tolist()
        print(f"Scores: {scores}, Candidates: {len(candidate_responses)}")
        return list(zip(candidate_responses, scores))


    def run_mini_test(self, mini_tests):
        """
        Ejecuta el mini test con los diálogos proporcionados.
        
        Args:
            mini_tests: Lista de diccionarios con 'context', 'input' y 'candidates'
        """
        if not mini_tests:
            print("\nNo se proporcionaron mini tests.")
            return
        
        print("\n" + "="*80)
        print("EJECUTANDO MINI TEST")
        print("="*80)
        
        for i, test in enumerate(mini_tests):
            print(f"\n=== Turno {i+1} ===")
            results = self.test_model_responses(
                test["emotion"], 
                test["sentiment"], 
                test["context"], 
                test["input"], 
                test["candidates"]
            )
            results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
            for resp, score in results_sorted:
                print(f"Score: {score:.3f}  -->  {resp}")
        
        print("\nMini test completado.")
    
    def set_data(self):
        self.data = {
            "anxiety": [
                "[SYS]: Hi, it seems you're feeling anxious. How can I support you today?",
                "[SYS]: Hello, anxiety can be challenging. What's on your mind?",
                "[SYS]: Hey, it looks like something's troubling you. How can I help with [SENTIMENT]?"
            ],
            "depression": [
                "[SYS]: Hi, it sounds like you might be feeling down. I'm here to listen, how are you?",
                "[SYS]: Hello, depression can be heavy. What would you like to talk about?",
                "[SYS]: Hey, it's okay to feel low. How can I assist with [SENTIMENT]?"
            ],
            "sadness": [
                "[SYS]: Hi, it seems you're feeling sad. How can I support you today?",
                "[SYS]: Hello, sadness can be tough. What's on your mind?",
                "[SYS]: Hey, I'm here for you. How can I help with [SENTIMENT]?"
            ],
            "fear": [
                "[SYS]: Hi, it seems you're feeling fearful. How can I help you through this?",
                "[SYS]: Hello, fear can be overwhelming. What's worrying you?",
                "[SYS]: Hey, let's face this together. How can I assist with [SENTIMENT]?"
            ],
            "anger": [
                "[SYS]: Hi, it seems you're feeling angry. How can I help you process this?",
                "[SYS]: Hello, anger can be intense. What's triggering you with [SENTIMENT]?",
                "[SYS]: Hey, let's work through this. How can I assist with [SENTIMENT]?"
            ],
            "joy": [
                "[SYS]: Hi, it sounds like you're feeling joyful. How can I support you today?",
                "[SYS]: Hello, joy is wonderful. What's bringing you happiness?",
                "[SYS]: Hey, it's great to see you feeling good. How can I help with [SENTIMENT]?"
            ],
            "surprise": [
                "[SYS]: Hi, it seems you're feeling surprised. How can I support you today?",
                "[SYS]: Hello, surprises can be unexpected. What's on your mind?",
                "[SYS]: Hey, I'm here for you. How can I help with [SENTIMENT]?"
            ],
            "love": [
                "[SYS]: Hi, it sounds like you're feeling loving. How can I support you today?",
                "[SYS]: Hello, love is beautiful. What would you like to talk about?",
                "[SYS]: Hey, I'm here for you. How can I help with [SENTIMENT]?"
            ],
            "default": [
                "[SYS]: Hi, I'm here to help. How can I support you today?",
                "[SYS]: Hello, it sounds like something's up. What's on your mind with [SENTIMENT]?"
            ]
        }

