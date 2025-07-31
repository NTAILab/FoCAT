import mlflow
import numpy as np
import torch
import os
import shutil
from torch import nn
from ticl.priors import MLPPrior

def make_training_callback(
    save_every, 
    model_string, 
    base_path, 
    report, 
    config, 
    use_mlflow, 
    checkpoint_dir, 
    classification, 
    validate
):
    from ticl.model_builder import save_model
    config = config.copy()

    def save_callback(model, optimizer, scheduler, epoch, task_stats=None):
        if not hasattr(model, 'last_saved_epoch'):
            model.last_saved_epoch = 0
        log_file = f'{base_path}/log/{model_string}.log'
        if epoch == "start":
            print(f"Starting training of model {model_string}")
            epoch = 0

        if epoch != "on_exit" and epoch != 0:
            try:
                os.makedirs(f"{base_path}/log", exist_ok=True)
                with open(log_file, 'a') as f:
                    f.write(f'Epoch {epoch} loss {model.losses[-1]} train_std {task_stats}'
                            f'learning_rate {model.learning_rates[-1]}\n')
            except Exception as e:
                print(f'Failed to write to log file {log_file}: {e}')
            wallclock_ticker = max(1, int(model.wallclock_times[-1]//(60 * 5)))
            if use_mlflow:
                mlflow.log_metric(key="wallclock_time", value=model.wallclock_times[-1], step=epoch)
                mlflow.log_metric(key="loss", value=model.losses[-1], step=epoch)
                mlflow.log_metric(key="learning_rate", value=model.learning_rates[-1], step=epoch)
                mlflow.log_metric(key="wallclock_ticker", value=wallclock_ticker, step=epoch)
                mlflow.log_metric(key="epoch", value=epoch, step=epoch)
                mlflow.log_metric(key="train_std", value=task_stats, step=epoch)
            # if wandb.run is not None:
            #     wandb.log({"loss": model.losses[-1], "learning_rate": model.learning_rates[-1], "wallclock_time": model.wallclock_times[-1],
            #                "wallclock_ticker": wallclock_ticker, "epoch": epoch})
            if report is not None:
                # synetune callback
                report(epoch=epoch, loss=model.losses[-1], wallclock_time=wallclock_ticker)  # every 5 minutes

        if epoch != "on_exit" and validate:
            # inference_time = None
            # if on_cuda:
            #     gpu_start_time = torch.cuda.Event(enable_timing=True)
            #     gpu_end_time = torch.cuda.Event(enable_timing=True)
            # if validate:
            #     inference_start = time.time()
            #     if on_cuda:
            #         gpu_start_time.record()
                
            validation_score = validate_model(model, config)
                
            #     inference_end = time.time()
            #     if on_cuda:
            #         gpu_end_time.record()
            #         torch.cuda.synchronize()
                
            #     inference_time = inference_end - inference_start
            #     if on_cuda:
            #         gpu_inference_time = gpu_start_time.elapsed_time(gpu_end_time)
            #     else:
            #         gpu_inference_time = 0

            print(f"Validation score: {validation_score}")

            if use_mlflow:
                mlflow.log_metric(key="val_score", value=validation_score, step=epoch)
        
        if (epoch == "on_exit") or epoch % save_every == 0:
            if checkpoint_dir is not None:
                if epoch == "on_exit":
                    return
                file_name = f'{base_path}/checkpoint.mothernet'
            else:
                file_name = f'{base_path}/models_diff/{model_string}_epoch_{epoch}.cpkt'
            try:
                os.makedirs(f"{base_path}/models_diff", exist_ok=True)
                disk_usage = shutil.disk_usage(f"{base_path}/models_diff")
                if disk_usage.free < 1024 * 1024 * 1024 * 2:
                    print("Not saving model, not enough disk space")
                    print("DISK FULLLLLLL")
                    return
                with open(log_file, 'a') as f:
                    f.write(f'Saving model to {file_name}\n')
                print(f'Saving model to {file_name}')
                config['epoch_in_training'] = epoch
                config['learning_rates'] = model.learning_rates
                config['losses'] = model.losses
                config['wallclock_times'] = model.wallclock_times

                save_model(model, optimizer, scheduler, base_path, file_name, config)
            
            except Exception as e:
                print("WRITING TO MODEL FILE FAILED")
                print(e)

            # on_cuda = next(model.parameters()).is_cuda

                            
                #     if wandb.run is not None:
                #         val_metrics = {
                #             "val_score": validation_score, 
                #             "epoch": epoch, 
                #             'inference_time': inference_time,
                #             'gpu_inference_time': gpu_inference_time,
                #         }
                        
                #         for dataset, score in per_dataset_score.items():
                #             val_metrics[f"val_score_{dataset}"] = score
                #         wandb.log(val_metrics)
                
                # remove checkpoints that are worse than current
                if epoch - save_every > 0:
                    this_loss = model.losses[-1]
                    for i in range(epoch // save_every):
                        loss = model.losses[i * save_every - 1]  # -1 because we start at epoch 1
                        old_file_name = f'{base_path}/models_diff/{model_string}_epoch_{i * save_every}.cpkt'
                        if os.path.exists(old_file_name):
                            if loss > this_loss:
                                try:
                                    print(f"Removing old model file {old_file_name}")
                                    os.remove(old_file_name)
                                except Exception as e:
                                    print(f"Failed to remove old model file {old_file_name}: {e}")
                            else:
                                print(f"Not removing old model file {old_file_name} because loss is too high ({loss} < {this_loss})")
                if validate: 
                    return 0.0#inference_time

    return save_callback


def synetune_handle_checkpoint(args):
    # handle syne-tune restarts
    checkpoint_dir = args.st_checkpoint_dir
    base_path = args.base_path
    warm_start_from = args.warm_start_from
    continue_run = args.continue_run
    report = None
    if checkpoint_dir is not None:
        from syne_tune import Reporter
        report = Reporter()
        base_path = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.mothernet"
        if checkpoint_path.exists():
            continue_run = True
            warm_start_from = checkpoint_path
    return base_path, continue_run, warm_start_from, report

def normalize_data(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1
    return (x - mean) / std

def validate_model(model, config):
    mlp_prior_config = {
        "pre_sample_causes": True,
        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
        'prior_mlp_scale_weights_sqrt': True,
        'random_feature_rotation': True,
        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True, 'lower_bound': 2},
        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 'lower_bound': 4},
        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
        # This mustn't be too high since activations get too large otherwise
        "init_std": {'distribution': 'log_uniform', 'min': 1e-2, 'max': 12},
        "noise_std": {'distribution': 'log_uniform', 'min': 1e-4, 'max': .1},
        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                        'lower_bound': 2},
        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        # "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
        "prior_mlp_activations": {'distribution': 'meta_choice', 'choice_values': [
            torch.nn.Tanh, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.Sigmoid
        ]},
        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
                        'add_uninformative_features': False}
    train_n = 256
    test_n = 1024
    f_n = 64
    mlp_prior = MLPPrior(mlp_prior_config)
    X, y_0, y_1 = mlp_prior.get_batch(1, train_n + test_n, f_n, 'cpu')
    X = normalize_data(X[:, 0, :].numpy())
    y_0, y_1 = y_0.ravel().numpy(), y_1.ravel().numpy()
    X, X_test = X[:train_n], X[train_n:]
    diff_test = y_1[train_n:] - y_0[train_n:]
    y_0, y_1 = y_0[:train_n], y_1[:train_n]
    c = np.random.binomial(1, 0.3, train_n)
    y = np.take_along_axis(np.stack((y_0, y_1), axis=-1), c[:, None], -1)[:, 0]

    model.fit(X, y, c)
    cate_test = model.predict(X_test)
    return np.mean(np.abs(cate_test - diff_test))