def optimize(n_trials=10,type):
    def objective(trial):
        # --- Sugerir hiperparámetros ---
        config = {
            "version": f"{trial.number}",
            "num_train_epochs": trial.suggest_int("num_train_epochs", 6, 12),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "freezeLayer": trial.suggest_int("freezeLayer", 0, 6),
            "early": trial.suggest_int("early", 2, 5),
        }

        print(f"\nTRIAL {trial.number} | Config: {config}")

        # --- Instanciar y entrenar ---
        trainer = TrainModels(
            dir_dataset=f"/home/silvina/proyectos/BERTherapy/data/processed/bertherapy_dataset_{type}_full.csv",
            output_dir_images=f"/home/silvina/proyectos/BERTherapy/images/train_{type}_v{config['version']}",
            output_dir_model=f"models/bert_{type}_v{config['version']}",
            check_dir_model=f"outputs-bert-imdb-{type}_v{config['version']}",
            output_dir_tensorboard=f"/home/silvina/proyectos/BERTherapy/tensorboard/{type}/train_{type}_v{config['version']}",
            num_train_epochs=config["num_train_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            freezeLayer=config["freezeLayer"],
            early=config["early"],
            fill_nan=False,
        )

        # Ejecutar entrenamiento completo
        trainer.run_all()

        # --- Obtener métrica de validación ---
        metrics = trainer.trainer.evaluate()
        val_acc = metrics.get("eval_accuracy", 0.0)

        print(f"TRIAL {type.upper()}: {trial.number} → eval_accuracy = {val_acc:.4f}")

        # Opcional: podar trials malos
        trial.report(val_acc, step=config["num_train_epochs"])
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_acc

    # --- Ejecutar estudio ---
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "="*60)
    print(f"MEJOR CONFIGURACIÓN ({type.upper()})")
    print("="*60)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  Mejor eval_accuracy: {study.best_value:.4f}")
    print("="*60)

    return study

types = ["therapist", "patient"]
for type in types:
   study = optimize(n_trials=10, type=type)
   best_params = study.best_params
   print(f"MEJOR CONFIGURACIÓN ({type.upper()}): {best_params}")
   print(f"MEJOR EVAL_ACCURACY ({type.upper()}): {study.best_value:.4f}")
   print("="*60)
   config_to_save = {
    "version": f"{study.best_trial.number}",
    "num_train_epochs": best_params["num_train_epochs"],
    "batch_size": best_params["batch_size"],
    "learning_rate": best_params["learning_rate"],
    "freezeLayer": best_params["freezeLayer"],
    "early": best_params["early"],
    }

   # guardar los mejores hiperparámetros en un archivo json
   with open(f"config/best_{type}.json", "w") as file:
      json.dump(config_to_save, file, indent=4)

    # como ya tenemos los mejores hiperparámetros, entrenamos el modelo con los mejores hiperparámetros
    trainer = TrainModels(
        dir_dataset=f"/home/silvina/proyectos/BERTherapy/data/processed/bertherapy_dataset_{type}_full.csv",
        output_dir_images=f"/home/silvina/proyectos/BERTherapy/images/train_{type}_v{config_to_save['version']}",
        output_dir_model=f"models/bert_{type}_v{config_to_save['version']}",
        check_dir_model=f"outputs-bert-imdb-{type}_v{config_to_save['version']}",
        output_dir_tensorboard=f"/home/silvina/proyectos/BERTherapy/tensorboard/{type}/train_{type}_v{config_to_save['version']}",
        num_train_epochs=config_to_save["num_train_epochs"],
        batch_size=config_to_save["batch_size"],
        learning_rate=config_to_save["learning_rate"],
        freezeLayer=config_to_save["freezeLayer"],
        early=config_to_save["early"],
        fill_nan=False,
    )
    trainer.run_all()
    ## recuperar accuracy
    accuracy = trainer.trainer.evaluate()
    print(f"Accuracy del {type.upper()}: {accuracy['eval_accuracy']:.4f }")