import os, pdb
import subprocess as sp

import torch#, wandb
from torch import nn
import numpy as np

import ticl.models.encoders as encoders
from ticl.dataloader import get_dataloader
from ticl.train import train
from ticl.model_configs import get_model_default_config
from ticl.models.mothernet import MotherNet
from ticl.models.mothernet_reg import MotherNetRegression
from ticl.config_utils import nested_dict

try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)

class GaussHistLoss:
    def __init__(self, device: torch.device, 
                 total_num_bins: int, 
                 num_bins_to_sum: int):
        self.num_bins = total_num_bins
        self.num_bins_to_sum = num_bins_to_sum
        self.bell_idx = total_num_bins // 2
        sigma = num_bins_to_sum / 6
        self.idx = (torch.arange(total_num_bins, device=device) + self.bell_idx)[None, None, :]
        coef = 1 / (np.sqrt(2 * np.pi) * sigma)
        x = torch.arange(total_num_bins, device=device)[None, None, :]
        self.exp_vec = coef * torch.exp(-(x - self.bell_idx) ** 2 / (2 * sigma ** 2))
        
    def __call__(self, outputs, targets):
        logits, steps = outputs
        target_idx = torch.searchsorted(steps, targets[..., None], side='left').clamp_(max=steps.shape[-1] - 1)
        shift = self.idx - target_idx
        zero_mask = (shift >= self.num_bins) | (shift < 0)
        shift.clamp_(min=0, max=self.num_bins - 1)
        coeffs = torch.take_along_dim(self.exp_vec, shift, dim=-1)
        coeffs[zero_mask] = 0
        loss = torch.log_softmax(logits, dim=-1) * coeffs
        return torch.sum(-loss, dim=-1)

def get_criterion(max_num_classes, device, 
                  total_num_bins=0, 
                  num_bins_to_sum=None):
    if max_num_classes == 0:
        if num_bins_to_sum is None or total_num_bins == 0 or num_bins_to_sum is None:
            loss = nn.MSELoss(reduction='none')
        else:
            loss = GaussHistLoss(device, total_num_bins, num_bins_to_sum)
    elif max_num_classes == 2:
        loss = nn.BCEWthLogitsLoss(reduction='none')
    elif max_num_classes > 2:
        loss = nn.CrossEntropyLoss(reduction='none')
    else:
        raise ValueError(f"Invalid number of classes: {max_num_classes}")
    return loss


def save_model(model, optimizer, scheduler, path, filename, config_sample):
    optimizer_dict = optimizer.state_dict() if optimizer is not None else None

    import cloudpickle
    torch.save((model.state_dict(), optimizer_dict, scheduler, config_sample), os.path.join(path, filename), pickle_module=cloudpickle)


def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode('ascii')
    return memory_free_info


# @cache
def load_model(path, device, verbose=False, f_regressor=False):
    states = torch.load(path, map_location='cpu', weights_only=False)
    model_state = states[0]
    config_sample = states[-1]
    config_sample['device'] = device
    if 'y_encoder' not in config_sample and 'onehot' in str(path):
        # workaround for the single model that was saved without y_encoder
        # that happens to be my reference model.
        config_sample['y_encoder'] = 'one_hot'
    _, model, *_ = get_model(config_sample, device=device, should_train=False,
                             verbose=verbose, f_regressor=f_regressor)
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model_state.pop("criterion.weight", None)

    decoder_summary_weights = ["query", "output_layer.q_proj_weight", "output_layer.in_proj_weight", "output_layer.k_proj_weight", "output_layer.v_proj_weight",
                               "output_layer.in_proj_bias", "output_layer.out_proj.weight", "output_layer.out_proj.bias"]
    for weights in decoder_summary_weights:
        full_name = "decoder." + weights
        if full_name in model_state:
            model_state['decoder.summary_layer.' + weights] = model_state.pop(full_name)

    if "encoder.weight" in model_state and "model_type" in config_sample and config_sample['model_type'] == "additive":
        model_state['encoder.1.weight'] = model_state.pop("encoder.weight")
        model_state['encoder.1.bias'] = model_state.pop("encoder.bias")

    for module_name in list(model_state.keys()):
        if module_name.startswith("ssm"):
            # for backwards compatibility
            model_state["linear_attention" + module_name[3:]] = model_state.pop(module_name)

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model, config_sample


def get_y_encoder(config, attention_type):
    if config[attention_type]['y_encoder'] == 'one_hot':
        y_encoder = encoders.OneHotAndLinear(config['prior']['classification']['max_num_classes'], emsize=config[attention_type]['emsize'])
    elif config[attention_type]['y_encoder'] == 'linear':
        y_encoder = encoders.Linear(1, emsize=config[attention_type]['emsize'])
    elif config[attention_type]['y_encoder'] in ['none', 'None', None]:
        y_encoder = None
    else:
        raise ValueError(f"Unknown y_encoder: {config[attention_type]['y_encoder']}")
    return y_encoder


def old_config_to_new(old_config, new_config):
    # this is not for restarting learning, only inference, so it doesn't convert orchestration parameters
    # it takes a flat config and converts it to a nested one based on the structure of new_config
    old_config['learning_rate'] = old_config.pop('lr')
    if "bptt" in old_config:
        old_config['n_samples'] = old_config.pop('bptt')
    old_config.update(old_config.pop("differentiable_hyperparameters", {}))
    if "y_encoder" not in old_config:
        old_config['y_encoder'] = 'linear'
    if "decoder_em_size" in old_config:
        old_config['decoder_embed_dim'] = old_config.pop('decoder_em_size')
    if "model_maker" in old_config:
        old_config['model_type'] = old_config.pop('model_maker')
    if "em_size" in old_config:
        old_config['emsize'] = old_config.pop('em_size')
    if "aggregate_gradients" in old_config:
        old_config['aggregate_k_gradients'] = old_config.pop('aggregate_gradients')
    if "model_type" not in old_config:
        old_config['model_type'] = 'tabpfn'
    if "num_predicted_hidden_layers" in old_config:
        old_config['predicted_hidden_layers'] = old_config.pop('num_predicted_hidden_layers')
    if "boolean_p_uninformative" in old_config:
        old_config['p_uninformative'] = old_config.pop('boolean_p_uninformative')
    if "boolean_max_fraction_uninformative" in old_config:
        old_config['max_fraction_uninformative'] = old_config.pop('boolean_max_fraction_uninformative')
    if old_config.pop("special_token", False):
        old_config['decoder_type'] = 'special_token'

    if old_config.pop("prenorm", False):
        print("prenorm is not supported anymore")
    if not old_config.pop("output_attention", True):
        raise NotImplementedError("output_attention=False is not supported anymore")
    if old_config.pop("decoder_two_hidden_layers", False):
        old_config['decoder_hidden_layers'] = 2
    else:
        old_config['decoder_hidden_layers'] = 1

    ignored_configs = ['seq_len_used', 'verbose', 'noise_type', 'normalize_to_ranking', 'normalize_by_used_features', 'num_categorical_features_sampler_a',
                       'differentiable', 'flexible', 'bptt_extra_samples', 'dynamic_batch_size', 'new_mlp_per_example', 'batch_size_per_gp_sample',
                       'normalize_ignore_label_too', 'differentiable_hps_as_style', 'rotate_normalized_labels', 'canonical_y_encoder',
                       'total_available_time_in_s', 'normalize_with_sqrt', 'done_part_in_training', 'mix_activations', 'save_every', 'create_new_run',
                       'perceiver_large_dataset', 'no_double_embedding', 'losses', 'wallclock_times', 'learning_rates', 'experiment', 'base_path',
                       'num_gpus', 'device', 'epoch_in_training', 'hid_factor', 'warm_start_from', 'continue_old_config', 'use_cpu', 'st_checkpoint_dir',
                       'use_mlflow', 'load_file', 'continue_run', 'load_strict', 'restart_scheduler', 'extra_fast_test', 'stop_after_epochs', 'shared_embedding',
                       'n_samples_used', 'double_embedding', 'learing_rate', 'gpu_id', 'agg_gradients', 'boolean_prior', 'seed_everything', 'model-type',
                       'num_features_used', 'max_eval_pos', 'nan_prob_unknown_reason_reason_prior', 'nan_prob_unknown_reason', 'no_mlflow']
    if old_config['model_type'] == 'tabpfn':
        # we used to store mothernet parameters in tabpfn models, but we no longer allow that
        ignored_configs.extend(['decoder_embed_dim', 'decoder_hidden_size', 'predicted_hidden_layer_size',
                                'predicted_hidden_layers', 'weight_embedding_rank', 'decoder_hidden_layers'])
    if old_config['model_type'] in ['mothernet', 'additive']:
        ignored_configs.extend(['num_latents'])
    for k in ignored_configs:
        old_config.pop(k, None)
    translated_config = nested_dict()
    for k, v in new_config.items():
        if k in old_config:
            translated_config[k] = old_config.pop(k)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        if k3 in old_config:
                            translated_config[k][k2][k3] = old_config.pop(k3)
                elif k2 in old_config:
                    translated_config[k][k2] = old_config.pop(k2)
    if len(old_config):
        raise ValueError(f"Unknown parameters: {old_config.keys()}")
    return translated_config


def get_model(
    config, 
    device, 
    should_train=True, 
    verbose=False, 
    model_state=None, 
    optimizer_state=None,
    scheduler=None, 
    epoch_callback=None, 
    load_model_strict=True,
    f_regressor=False
):
    passed_config = config.copy()

    # backwards compatibility for model names
    if 'model_type' not in passed_config:
        if 'model_maker' in passed_config:
            passed_config['model_type'] = passed_config.pop('model_maker')
        else:
            passed_config['model_type'] = 'tabpfn'

    if passed_config['model_type'] == 'mlp':
        passed_config['model_type'] = 'mothernet'
    config = get_model_default_config(passed_config['model_type'])

    if 'optimizer' not in passed_config:
        passed_config = old_config_to_new(passed_config, config)
    config.update(passed_config)
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config['verbose'] = verbose_prior
    
    criterion = get_criterion(config['prior']['classification']['max_num_classes'],
                              device=config['device'],
                              total_num_bins=config['mothernet'].get('bins_output_n'),
                              num_bins_to_sum=config['mothernet'].get('bins_to_sum_n'),
                              )


    if 'transformer' in config:
        attention_type = 'transformer'
    elif 'linear_attention' in config:
        attention_type = 'linear_attention'
        if 'ssm' in passed_config:
            passed_config['linear_attention'] = passed_config.pop('ssm')
    else:
        raise ValueError(f"Unknown attention type")
    
    # backwards compatibility for cases where absence of parameter doesn't correspond to current default
    if 'n_samples' not in passed_config['prior']:
        config['prior']['n_samples'] = config['bptt']
    if 'y_encoder' not in passed_config[attention_type]:
        config[attention_type]['y_encoder'] = 'linear'

    if 'mothernet' in config:
        if 'decoder_activation' not in passed_config['mothernet']:
            config['mothernet']['decoder_activation'] = 'relu'
        if 'decoder_type' not in passed_config['mothernet']:
            config['mothernet']['decoder_type'] = 'output_attention'
        if (passed_config['mothernet'].get('weight_embedding_rank', None) is not None
                and 'low_rank_weights' not in passed_config['mothernet']):
            config['mothernet']['low_rank_weights'] = True

    y_encoder = get_y_encoder(config, attention_type)

    if config['prior']['classification']['max_num_classes'] > 2:
        n_out = config['prior']['classification']['max_num_classes']
    else:
        n_out = 1

    model_type = config['model_type']
    n_features = config['prior']['num_features']

    if model_type == "mothernet":
        if f_regressor:
            model = MotherNetRegression(
                n_out=n_out, y_encoder_layer=y_encoder,
                n_features=n_features, **config['transformer'],
                **config['mothernet']
            )
        else:
            model = MotherNet(
                n_out=n_out, y_encoder_layer=y_encoder,
                n_features=n_features, **config['transformer'],
                **config['mothernet'])
    elif model_type == 'perceiver':
        # model = TabPerceiver(n_out=n_out, y_encoder_layer=y_encoder, n_features=n_features,
        #                     **config['transformer'], **config['mothernet'], **config['perceiver'])
        raise NotImplementedError()
    elif model_type == "additive":
        model = MotherNetAdditive(
            n_out=n_out, n_features=n_features,
            y_encoder_layer=y_encoder, **config['transformer'], **config['mothernet'], **config['additive'])
    elif model_type == "tabpfn":
        model = TabPFN(n_out=n_out, n_features=n_features, y_encoder_layer=y_encoder, **config['transformer'])
    elif model_type == "batabpfn":
        # FIXME hack
        config['transformer']['nhead'] = 4
        model = BiAttentionTabPFN(
            n_out=n_out, y_encoder_layer=y_encoder, **config['transformer'], **config['biattention'])
    elif model_type == "baam":
        # FIXME hack
        config['transformer']['nhead'] = 4
        model = GAMformer(
            n_out=n_out, n_features=config['prior']['num_features'],
            y_encoder_layer=y_encoder, **config['transformer'], **config['mothernet'], **config['additive'])
    elif model_type in ['tabflex', 'ssm_tabpfn']:
        from ticl.models.tabflex import TabFlex
        config['linear_attention'].pop('causal_mask', None)
        model = TabFlex(
            n_out=n_out, 
            n_features=n_features, 
            y_encoder_layer=y_encoder, 
            **config['linear_attention']
        )
    elif model_type == 'la_mothernet':
        from ticl.models.la_mothernet import SSMMotherNet
        model = SSMMotherNet(
            n_out=n_out, 
            y_encoder_layer=y_encoder, 
            n_features=n_features, 
            **config['linear_attention'], 
            **config['mothernet']
        )
    else:
        raise ValueError(f"Unknown model type {model_type}.")

    if model_state is not None:
        if not load_model_strict:
            for k, v in model.state_dict().items():
                if k in model_state and model_state[k].shape != v.shape:
                    model_state.pop(k)
        model.load_state_dict(model_state, strict=load_model_strict)

    if verbose:
        model_size = sum(p.numel() for p in model.parameters())
        # if wandb.run: wandb.log({"model_size": model_size})
        if 'linear_attention' in model_type:
            print(f"Using a SSM with {model_size/1000/1000:.{2}f} M parameters")
        else:
            print(f"Using a Transformer with {model_size/1000/1000:.{2}f} M parameters")

    if 'losses' in config:
        # for continuing training
        model.losses = config['losses']
        model.learning_rates = config['learning_rates']
        model.wallclock_times = config.get('wallclock_times', [])

    print(device)

    if should_train:
        model_attr = None if 'linear_attention' not in model_type else config['linear_attention']['model']
        dl = get_dataloader(
            prior_config=config['prior'], 
            dataloader_config=config['dataloader'], 
            device='cpu',
            model = model_attr,
        )
        model = train(dl, model, criterion=criterion, optimizer_state=optimizer_state, scheduler=scheduler,
                      epoch_callback=epoch_callback, verbose=verbose_train, device=device,
                      experiment_name=config['orchestration']['experiment'],
                      progress_bar=config['orchestration']['progress_bar'], **config['optimizer'])
    else:
        model = None, model, None, None

    return model
