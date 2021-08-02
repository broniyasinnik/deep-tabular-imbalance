import optuna
from experiment_runner import ExperimentRunner


def objective_meta(trial):
    runner = ExperimentRunner(exper_dir, trial=trial)
    runner.config.hparams.lr_model = trial.suggest_loguniform("lr_model", 1e-3, 1e-1)
    runner.config.hparams.lr_z = trial.suggest_loguniform("lr_z", 1e1, 1e4)
    runner.config.hparams.lr_meta = trial.suggest_loguniform("lr_meta", 1e-3, 1e-2)
    runner.config.hparams.lr_kde = trial.suggest_loguniform("lr_kde", 1e-3, 1e-1)
    runner.run_meta_experiment()
    score = trial.best_score
    return score


def objective_base(trial):
    runner = ExperimentRunner(exper_dir, trial=trial)
    runner.config.model.hiddens = [int(trial.suggest_loguniform("hiddens", 32, 128))]
    runner.config.hparams.lr_model = trial.suggest_loguniform("lr_model", 1e-3, 1e-1)
    runner.run_baseline_experiment()
    score = trial.best_score
    return score


def run_study(obj_func):
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=1, n_warmup_steps=0, interval_steps=1
        ),
    )
    study.optimize(objective_base, n_trials=20, timeout=300)
    print(study.best_value, study.best_params)


if __name__ == "__main__":
    exper_dir = './Adult/ir100'
    run_study(objective_base)
