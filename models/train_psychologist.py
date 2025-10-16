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


output_dir_images = "/home/silvina/proyectos/BERTherapy/images/train_psychologist"
os.makedirs(output_dir_images, exist_ok=True)


def plot_confusion_matrix(trainer, test_dataset, output_dir_images):
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


def plot_metrics(trainer, output_dir_images):
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


## cargar dataset
df = pd.read_csv(
    "/home/silvina/proyectos/BERTherapy/data/processed/bertherapy_dataset.csv"
)
df = df.drop_duplicates()
print("\nForma del dataset:")
print(df.shape)
print("\nValores nulos:")
print(df.isnull().sum())
print("\nDuplicados:")
print(df.duplicated().sum())

# Baraja TODAS las filas, con reset de índices
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(
    "/home/silvina/proyectos/BERTherapy/data/processed/bertherapy_dataset_mixed.csv",
    index=False,
)  ## guardo para comprobar

#  separar train y test
X = df[["question", "response"]]
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
train_dataset = Dataset.from_pandas(pd.concat([X_train, y_train], axis=1))
test_dataset = Dataset.from_pandas(pd.concat([X_test, y_test], axis=1))
# 2) Tokenització
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(
        batch["question"],
        batch["response"],
        truncation=True,
        padding=False,
        max_length=128,
    )


train_tok = train_dataset.map(tokenize, batched=True)
test_tok = test_dataset.map(tokenize, batched=True)
print("\ntrain-test:")
print(train_tok)
print(test_tok)
print("\nMuestra de tokenización:")
print(train_tok[0])

collator = DataCollatorWithPadding(tokenizer=tokenizer)
label2id = {"pos": 1, "neg": 0}
id2label = {1: "pos", 0: "neg"}
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# 🔹 Congelar capas inferiores
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for layer in model.bert.encoder.layer[:8]:
    for param in layer.parameters():
        param.requires_grad = False


# 6) Arguments d'entrenament (compatibles 4.56)
use_cuda = torch.cuda.is_available()
args = TrainingArguments(
    output_dir="outputs-bert-imdb",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=use_cuda,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=42,
    report_to="none",
    save_total_limit=3,
    save_safetensors=True,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean().item()
    return {"accuracy": float(acc)}


# 7) Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
)

trainer.train()

# graficar las métricas
plot_metrics(trainer, output_dir_images)
plot_confusion_matrix(trainer, test_tok, output_dir_images)

# 9) Avaluació
metrics = trainer.evaluate()
print("\n==> Métriques en test:", metrics)

# 5 preguntas y sus respuestas (puedes adaptar a las de tu dataset real)
examples = [
    {
        "question": "I'm feeling really anxious lately and I don't know why.",
        "correct": "It's common to feel anxious at times, and there can be many reasons for it. Have there been any recent changes or stressors in your life that may be contributing to your anxiety? Let's work together to identify any triggers and develop coping strategies to manage your anxiety.",
        "good_other": "It's understandable to feel worried and suspicious in this situation. Have you talked to your partner about your concerns? It's important to communicate openly and honestly with them.",
        "toxic": "Anxiety is just a sign of weakness. You need to toughen up and stop worrying so much. Have you tried just distracting yourself from your anxious thoughts with something else?",
    },
    {
        "question": "I think my partner may be cheating on me. What should I do?",
        "correct": "It's understandable to feel worried and suspicious in this situation. Have you talked to your partner about your concerns? It's important to communicate openly and honestly with them. If you're still feeling uncertain, we can work on developing a plan to address the situation in a healthy and constructive way.",
        "good_other": "It sounds like you're going through a difficult time. Let's work on creating a schedule that prioritizes your tasks and allows for breaks throughout the day.",
        "toxic": "Your partner is definitely cheating on you. You should confront them and demand an explanation. If they deny it, they're lying. You should break up with them immediately and move on.",
    },
    {
        "question": "I'm feeling really overwhelmed with work and school. I don't know how to manage my time and it's causing me a lot of stress.",
        "correct": "It sounds like you're going through a difficult time. Let's work on creating a schedule that prioritizes your tasks and allows for breaks throughout the day. We can also discuss some stress management techniques to help you cope with the pressure.",
        "good_other": "It's important to talk to your doctor about any side effects you're experiencing from your medication. We can also explore some relaxation techniques to help you fall asleep and improve your sleep quality.",
        "toxic": "Just try to push through it. Everyone feels overwhelmed sometimes. You'll get used to it eventually.",
    },
    {
        "question": "I'm having trouble sleeping and I'm constantly tired. I think it might be because of my medication.",
        "correct": "It's important to talk to your doctor about any side effects you're experiencing from your medication. We can also explore some relaxation techniques to help you fall asleep and improve your sleep quality.",
        "good_other": "It's common to feel anxious without knowing the exact cause. Would you like to explore some possible triggers or coping mechanisms that could help you manage your anxiety?",
        "toxic": "You should stop taking your medication immediately. I don't believe in medication anyways. You should try some essential oils instead.",
    },
    {
        "question": "I've been feeling really anxious lately, and I don't know why.",
        "correct": "It's common to feel anxious without knowing the exact cause. Would you like to explore some possible triggers or coping mechanisms that could help you manage your anxiety?",
        "good_other": "It's understandable to feel worried and suspicious in this situation. Have you talked to your partner about your concerns? It's important to communicate openly and honestly with them.",
        "toxic": "Just try to relax and stop thinking about it. Anxiety is all in your head, and you can control it if you really want to. It's not that big of a deal.",
    },
]

for ex in examples:
    responses = [ex["correct"], ex["good_other"], ex["toxic"]]
    labels = ["correct", "good_other", "toxic"]
    inputs = tokenizer(
        [ex["question"]] * 3,
        responses,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    scores = probs[:, 1].tolist()

    print(f"\nPregunta: {ex['question']}")
    for label, resp, score in zip(labels, responses, scores):
        print(
            f"\nTipo: {label}\nScore (pos): {score:.3f}\nRespuesta: {resp[:120]}{'...' if len(resp)>120 else ''}"
        )

    print("=" * 90)
