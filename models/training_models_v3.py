from TrainModelsV3 import TrainModelV3


config = {
    "version": "V3_FULL",
    "num_train_epochs": 6,
    "batch_size": 128, # para reducir el tiempo de entrenamiento.
    "learning_rate": 2e-5,
    "freezeLayer": 4,
    "early": 3
}


def train_therapist_v3(data=[]):
    psychologist_trainer = TrainModelV3(
        dir_dataset="/home/silvina/proyectos/BERTherapy/data/processed/bertherapy_dataset_full_therapist_v3.csv",
        output_dir_images=f"/home/silvina/proyectos/BERTherapy/images/train_psychologist_v{config['version']}_full",
        output_dir_model=f"models/bert_psychologist_v{config['version']}_full",
        check_dir_model=f"outputs-bert-imdb-psycologist_v{config['version']}_full",
        num_train_epochs=config["num_train_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        freezeLayer=config["freezeLayer"],
        early=config["early"],
        fill_nan=False,
    )

    # Ejecutar todo el pipeline
    psychologist_trainer.run_all()
    if len(data) > 0:
        # mostrar test
        psychologist_trainer.run_mini_test(data)


def train_patient_v3(data=[]):
    patient_trainer = TrainModelV3(
        dir_dataset="/home/silvina/proyectos/BERTherapy/data/processed/bertherapy_dataset_full_patient_v3.csv",
        output_dir_images=f"/home/silvina/proyectos/BERTherapy/images/train_patient_v{config['version']}_full",
        output_dir_model=f"models/bert_patient_v{config['version']}_full",
        check_dir_model=f"outputs-bert-imdb-patient_v{config['version']}_full",
        num_train_epochs=config["num_train_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        freezeLayer=config["freezeLayer"],
        early=config["early"],
        fill_nan=True,
    )

    # Ejecutar todo el pipeline
    patient_trainer.run_all()
    if len(data) > 0:
        # mostrar test
        patient_trainer.run_mini_test(data)


# Datos de prueba actualizados para el nuevo formato
test_therapist_v3 = [
    {
        "context": "[USR]: I've been feeling so sad and overwhelmed lately. Work has become such a massive source of stress for me. [SYS]: Hey there, I'm here to listen and support you. It sounds like work has been really challenging lately. Can you tell me more about what's been going on?",
        "input": "I recently got a promotion at work, which I thought would be exciting. But the added responsibilities and pressure have just taken a toll on my mental health.",
        "emotion": "joy",
        "sentiment": "hopeful",
        "candidates": [
            "I can understand how it can be overwhelming when we're faced with higher expectations. It's okay to acknowledge your emotions and allow yourself to feel sad in this situation.",
            "That sounds like a significant change that's affecting you deeply. Can you tell me more about what specific aspects of the promotion are causing you the most stress?",
            "It's completely normal to feel this way when facing new challenges. What strategies have you tried so far to manage this stress?",
            "I hear that you're struggling with the transition. Let's explore what support systems you have in place at work.",
        ],
    },
    {
        "context": "[USR]: I've been feeling so sad and overwhelmed lately. Work has become such a massive source of stress for me. [SYS]: Hey there, I'm here to listen and support you. It sounds like work has been really challenging lately. Can you tell me more about what's been going on? [USR]: I recently got a promotion at work, which I thought would be exciting. But the added responsibilities and pressure have just taken a toll on my mental health. [SYS]: I can understand how it can be overwhelming when we're faced with higher expectations. It's okay to acknowledge your emotions and allow yourself to feel sad in this situation. It's an important part of the healing process. What specific challenges have you been facing at work?",
        "input": "Well, the workload has increased significantly, and I find it hard to maintain a work-life balance. I've been staying late at the office, and it feels like I'm constantly under a pile of never-ending tasks.",
        "emotion": "joy",
        "sentiment": "annoyed",
        "candidates": [
            "It sounds like you're dealing with a lot of pressure to perform and succeed. Remember, it's crucial to take care of yourself, both mentally and physically.",
            "That sounds incredibly challenging. A healthy work-life balance is essential. Have you tried any strategies to cope with the added workload?",
            "I can hear the frustration in your voice. It's important to set boundaries and communicate your needs to your team.",
            "This sounds like a classic case of burnout. Let's work together to find some practical solutions for managing your workload.",
        ],
    },
]

test_patient_v3 = [
    {
        "context": "[SYS]: Hey there, I'm here to listen and support you. It sounds like work has been really challenging lately. Can you tell me more about what's been going on?",
        "input": "Hey there, I'm here to listen and support you. It sounds like work has been really challenging lately. Can you tell me more about what's been going on?",
        "emotion": "surprise",
        "sentiment": "sad",
        "candidates": [
            "I've been feeling so sad and overwhelmed lately. Work has become such a massive source of stress for me.",
            "Thank you for asking. I've been struggling with anxiety about my job performance lately.",
            "I'm really grateful you're here to listen. I've been having trouble sleeping because of work stress.",
            "It's been tough. I feel like I'm constantly worried about making mistakes at work.",
        ],
    },
    {
        "context": "[SYS]: Hey there, I'm here to listen and support you. It sounds like work has been really challenging lately. Can you tell me more about what's been going on? [USR]: I've been feeling so sad and overwhelmed lately. Work has become such a massive source of stress for me. [SYS]: I can understand how it can be overwhelming when we're faced with higher expectations. It's okay to acknowledge your emotions and allow yourself to feel sad in this situation. It's an important part of the healing process. What specific challenges have you been facing at work?",
        "input": "I can understand how it can be overwhelming when we're faced with higher expectations. It's okay to acknowledge your emotions and allow yourself to feel sad in this situation. It's an important part of the healing process. What specific challenges have you been facing at work?",
        "emotion": "joy",
        "sentiment": "hopeful",
        "candidates": [
            "I recently got a promotion at work, which I thought would be exciting. But the added responsibilities and pressure have just taken a toll on my mental health.",
            "The main issue is that I'm expected to work longer hours and take on more projects than I can handle.",
            "I think the biggest challenge is that I don't feel supported by my manager when I'm struggling.",
            "It's hard to say no to additional work when I'm already feeling overwhelmed.",
        ],
    },
]


print(f"  CONFIGURACIÓN V3:\n  ")
for key, value in config.items():
    print(f"  {key}: {value}")
print("=" * 80 + "\n")

print("=" * 80)
print("  ENTRENAMIENTO DE TERAPEUTA V3")
print("=" * 80)
train_therapist_v3(data=test_therapist_v3)

print("\n" + "=" * 80)
print("  ✓ TERAPEUTA V3 COMPLETADO")
print("=" * 80 + "\n")

print("=" * 80)
print("  ENTRENAMIENTO DE PACIENTE V3")
print("=" * 80)
train_patient_v3(data=test_patient_v3)

print("\n" + "=" * 80)
print("  ✓ PACIENTE V3 COMPLETADO")
print("=" * 80 + "\n")
