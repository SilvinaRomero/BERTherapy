import optuna
import json
import os
from TrainModels import TrainModels

# Obtener la ruta del directorio del proyecto (BERTherapy)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def optimize(n_trials=10, type="therapist"):
    def objective(trial):
        # # --- Sugerir hiperparámetros ---
        config = {
            "version": f"{trial.number}",
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 8),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "freezeLayer": trial.suggest_int("freezeLayer", 0, 6),
            "early": trial.suggest_int("early", 2, 4),
        }

        print(f"\nTRIAL {trial.number} | Config: {config}")

        # --- Instanciar y entrenar ---
        trainer = TrainModels(
            dir_dataset=os.path.join(PROJECT_ROOT, f"data/processed/bertherapy_dataset_{type}_full.csv"),
            output_dir_images=os.path.join(PROJECT_ROOT, f"images/train_{type}_v{config['version']}"),
            output_dir_model=os.path.join(PROJECT_ROOT, f"models/bert_{type}_v{config['version']}"),
            check_dir_model=os.path.join(PROJECT_ROOT, f"outputs-bert-imdb-{type}_v{config['version']}"),
            output_dir_tensorboard=os.path.join(PROJECT_ROOT, f"tensorboard/{type}/train_{type}_v{config['version']}"),
            num_train_epochs=config["num_train_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            freezeLayer=config["freezeLayer"],
            early=config["early"],
            fill_nan=(type == "patient"),  # patient needs fill_nan=True
        )

        # Ejecutar entrenamiento completo
        trainer.run_all()

        # --- Obtener métrica de validación ---
        metrics = trainer.trainer.evaluate()
        val_loss = metrics.get("eval_loss", float('inf'))

        print(f"TRIAL {type.upper()}: {trial.number} → eval_loss = {val_loss:.4f}")

        # Opcional: podar trials malos
        trial.report(val_loss, step=config["num_train_epochs"])
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_loss

    # --- Ejecutar estudio ---
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "="*60)
    print(f"MEJOR CONFIGURACIÓN ({type.upper()})")
    print("="*60)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  Mejor eval_loss: {study.best_value:.4f}")
    print("="*60)

    return study

types = [
    "therapist", 
    "patient",
]
for type_val in types:
    print("\n" + "="*80)
    print(f"  OPTIMIZACIÓN Y ENTRENAMIENTO DE {type_val.upper()}")
    print("="*80 + "\n")
    
    study = optimize(n_trials=15, type=type_val)
    best_params = study.best_params
    
    print("\n" + "="*80)
    print(f"  MEJOR CONFIGURACIÓN ENCONTRADA ({type_val.upper()})")
    print("="*80)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  Mejor eval_loss: {study.best_value:.4f}")
    print("="*80 + "\n")
    
    config_to_save = {
        "version": f"{study.best_trial.number}",
        "num_train_epochs": best_params["num_train_epochs"],
        "batch_size": best_params["batch_size"],
        "learning_rate": best_params["learning_rate"],
        "freezeLayer": best_params["freezeLayer"],
        "early": best_params["early"],
    }

    # sobreescribir los mejores hiperparámetros en un archivo json (therapist.json o patient.json)
    config_path = os.path.join(PROJECT_ROOT, f"models/config/{type_val}.json")
    with open(config_path, "w") as file:
        json.dump(config_to_save, file, indent=4)
    print(f"✓ Configuración guardada en {config_path}\n")
    print(f"✓ Mejor modelo ya guardado en: models/bert_{type_val}_v{config_to_save['version']}\n")

    
    