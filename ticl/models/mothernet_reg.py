import torch
from torch import nn
from ticl.models.layer import TransformerEncoderLayer, TransformerEncoderSimple
from ticl.models.decoders import MLPModelDecoder
from ticl.utils import SeqBN, get_init_method
from ticl.models.encoders import Linear
from ticl.models.encoders import OneHotAndLinear
from ticl.models.decoders import MLPModelDecoder
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from datetime import datetime

class MotherNetRegression(torch.nn.Module):
    def __init__(self, *, n_out, emsize, nhead, nhid_factor, nlayers, n_features, dropout=0.0, y_encoder_layer=None,
                 input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False,
                 all_layers_same_init=False, efficient_eval_masking=True, decoder_type="output_attention", predicted_hidden_layer_size=None,
                 cls_token_n=0, decoder_embed_dim=2048, classification_task=True,
                 decoder_hidden_layers=1, decoder_hidden_size=None, predicted_hidden_layers=1, weight_embedding_rank=None, y_encoder=None,
                 low_rank_weights=False, tabpfn_zero_weights=True, decoder_activation="relu", predicted_activation="relu",
                 bins_output_n=0, bins_to_sum_n=0, low_rank_last=False):
        super().__init__()
        self.classification_task = classification_task
        # decoder activation = "relu" is legacy behavior
        nhid = emsize * nhid_factor
        # mothernet has batch_first=False, unlike all the other models.
        def encoder_layer_creator(): return TransformerEncoderLayer(emsize, nhead, nhid, dropout, activation=activation,
                                                                    pre_norm=pre_norm, recompute_attn=recompute_attn, batch_first=False)
        self.transformer_encoder = TransformerEncoderSimple(encoder_layer_creator, nlayers)
        
        backbone_size = sum(p.numel() for p in self.transformer_encoder.parameters())
        # if wandb.run: wandb.log({"backbone_size": backbone_size})
        print("Number of parameters in backbone: ", backbone_size)

        self.decoder_activation = decoder_activation
        self.emsize = emsize
        self.encoder = Linear(n_features, emsize, replace_nan_by_zero=True)
        self.y_encoder = y_encoder_layer
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.n_out = bins_output_n if bins_output_n > 0 else 1
        self.nhid = nhid
        self.decoder_type = decoder_type
        decoder_hidden_size = decoder_hidden_size or nhid
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.predicted_activation = predicted_activation
        self.bins_output_n = bins_output_n
        self.low_rank_last = low_rank_last

        self.decoder = MLPModelDecoder(emsize=emsize, hidden_size=decoder_hidden_size, n_out=self.n_out, decoder_type=self.decoder_type,
                                       predicted_hidden_layer_size=predicted_hidden_layer_size, embed_dim=decoder_embed_dim,
                                       decoder_hidden_layers=decoder_hidden_layers, nhead=nhead, predicted_hidden_layers=predicted_hidden_layers,
                                       weight_embedding_rank=weight_embedding_rank, low_rank_weights=low_rank_weights, decoder_activation=decoder_activation,
                                       in_size=n_features, cls_token_n=cls_token_n, low_rank_last=low_rank_last)
        if decoder_type in ["special_token", "cls_token"]:
            self.token_embedding = nn.Parameter(torch.randn(cls_token_n, 1, emsize))

        self.init_weights()

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if self.tabpfn_zero_weights:
            for layer in self.transformer_encoder.layers:
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)
                attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

    def inner_forward(self, train_x):
        return self.transformer_encoder(train_x)
    
    def calc_nn_weights(self, x_train: torch.Tensor, 
                        y_train: torch.Tensor):
        x_enc = self.encoder(x_train)
        y_enc = self.y_encoder(y_train[..., None])
        enc_train = x_enc + y_enc
        
        if self.decoder_type in ["cls_token", "class_tokens"]:
            cls_tokens_rep = self.token_embedding.repeat(1, enc_train.shape[1], 1)
            enc_train = torch.cat((cls_tokens_rep, enc_train), dim=0)

        output = self.inner_forward(enc_train)
        return self.decoder(output, y_train)
    
    def nn_forward(self, x_test: torch.Tensor, layers):
        (b1, w1), *layers = layers
        x_test_nona = torch.nan_to_num(x_test, nan=0)
        h = (x_test_nona.unsqueeze(-1) * w1.unsqueeze(0)).sum(2)

        if self.decoder.weight_embedding_rank is not None and len(layers):
            h = torch.matmul(h, self.decoder.shared_weights[0])
        h = h + b1

        for i, (b, w) in enumerate(layers):
            if self.predicted_activation == "relu":
                h = torch.relu(h)
            elif self.predicted_activation == "gelu":
                h = torch.nn.functional.gelu(h)
            else:
                raise ValueError(f"Unsupported predicted activation: {self.predicted_activation}")
            if i < len(layers) - 1:
                h = (h.unsqueeze(-1) * w.unsqueeze(0)).sum(2)
                if self.decoder.weight_embedding_rank is not None: 
                    # last layer has no shared weights
                    h = torch.matmul(h, self.decoder.shared_weights[i + 1])
            else:
                if self.low_rank_last:
                    h = torch.matmul(h, self.decoder.shared_weights[i + 1])
                    h = (h.unsqueeze(-1) * w.unsqueeze(0)).sum(2)
                else:
                    h = (h.unsqueeze(-1) * w.unsqueeze(0)).sum(2)
            h = h + b

        if h.isnan().all():
            print("NAN")
            raise ValueError("NAN")
        return h
    
    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x, y, c)'

        _, x, y = src
        y_train = y[:single_eval_pos]
        layers = self.calc_nn_weights(x[:single_eval_pos],
                                      y_train)
        y_pred = self.nn_forward(x[single_eval_pos:], layers)
        if torch.all(torch.isnan(y_pred)):
            with open('nan_log.txt', 'a') as file:
                dt = datetime.today()
                file.write(f'-------[{dt.strftime("%H_%M_%S|%d_%m_%Y")}]-------\n')
                nan_tf_count = 0
                for weights in self.transformer_encoder.parameters():
                    nan_tf_count += torch.count_nonzero(torch.isnan(weights))
                msg = 'NaN in TF: ' + str(nan_tf_count)
                file.write(msg)
                msg = 'NaN after TF: ' + 'YES' if torch.any(torch.isnan(layers[0][1])) else 'NO' + '\n'
                file.write(msg)
                msg = f'y values (pivot {single_eval_pos}): {y}\n\nmean train: {y[:single_eval_pos].mean(0)}, std train: {y[:single_eval_pos].std(0)}\n'
                file.write(msg)
        if self.bins_output_n > 0:
            y_min = y_train.amin(dim=0, keepdim=True)
            y_max = y_train.amax(dim=0, keepdim=True)
            y_diff = (y_max - y_min) / self.bins_output_n
            idx = torch.arange(self.bins_output_n, device=y_pred.device)[None, None, :].repeat(y_pred.shape[0], 1, 1)
            y_ruler = idx * y_diff[..., None] + y_min[..., None]
            y_pred = (y_pred, y_ruler)
        return y_pred
        
    def get_device(self):
        return self.decoder.shared_weights[0].device

    def pad_zeros(self, x: np.ndarray):
        feats_n = self.encoder.in_features
        if x.shape[-1] > feats_n:
            raise RuntimeError(f"Maximum number of features is {feats_n}")
        elif x.shape[-1] == feats_n:
            return x
        shape = x.shape[:-1] + (feats_n - x.shape[-1],)
        return np.concatenate((x, np.zeros(shape)), axis=-1)
    
    @torch.inference_mode()    
    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2 or X.ndim == 3
        X = self.pad_zeros(X)
        tr_func = lambda arg: torch.from_numpy(arg[0]).to(device=self.get_device(), dtype=arg[1])
        df_dt = torch.get_default_dtype()
        X_t, y_t = map(tr_func, ((X, df_dt), (y, df_dt)))
        if X.ndim == 2:
            X_t = X_t[:, None, :]
            y_t = y_t[:, None]
        # print(f"Fitting model for {X_t.shape[1]} task(s) with {X_t.shape[0]} points")
        y_mean = torch.mean(y_t, dim=0, keepdim=True)
        y_std = torch.std(y_t, dim=0, keepdim=True)
        y_std[y_std < 1e-6] = 1
        self.y_std = y_std
        self.y_mean = y_mean
        y_t = (y_t - y_mean) / y_std
        # print(f"Fitting model for {X_t.shape[1]} task(s) with {X_t.shape[0]} points")
        self.pred_layers = self.calc_nn_weights(X_t, y_t)
        if self.bins_output_n > 0:
            y_min = y_t.amin(dim=0, keepdim=True)
            y_max = y_t.amax(dim=0, keepdim=True)
            y_diff = (y_max - y_min) / self.bins_output_n
            idx = torch.arange(self.bins_output_n, device=self.get_device())[None, :]
            self.y_ruler = idx * y_diff + y_min
        return self
    
    @torch.inference_mode()
    def predict(self, X: np.ndarray, batch_size: int = 1024):
        X = self.pad_zeros(X)
        layers = getattr(self, 'pred_layers', None)
        if layers is None:
            raise RuntimeError("Fit model first")
        if X.ndim == 2:
            X = X[:, None, :]
        X_t = torch.tensor(X).to(device=self.get_device(), dtype=torch.get_default_dtype())
        ds = TensorDataset(X_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        pred_idx_list = []
        for x_b, in dl:
            cur_pred = self.nn_forward(x_b, layers)
            if self.bins_output_n > 0:
                pred_idx_list.append(
                    torch.argmax(cur_pred, dim=-1)
                )
            else:
                pred_idx_list.append(cur_pred)
        pred_idx = torch.cat(pred_idx_list, dim=0)
        if self.bins_output_n > 0:
            y = torch.take_along_dim(self.y_ruler, pred_idx.T, dim=-1).T
        else:
            y = pred_idx
        y = y * self.y_std + self.y_mean
        pred = y.cpu().numpy()
        return pred.squeeze()

class MNRegAdapter:
    def __init__(self, mn_model: MotherNetRegression):
        self.mn_model = mn_model
    
    def get_params(self):
        return {'mn_model': self.mn_model}
    
    @torch.inference_mode()    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MNRegAdapter':
        assert X.ndim == 2 or X.ndim == 3
        X = self.mn_model.pad_zeros(X)
        tr_func = lambda arg: torch.from_numpy(arg[0]).to(device=self.mn_model.get_device(), dtype=arg[1])
        df_dt = torch.get_default_dtype()
        X_t, y_t = map(tr_func, ((X, df_dt), (y, df_dt)))
        if X.ndim == 2:
            X_t = X_t[:, None, :]
            y_t = y_t[:, None]
        y_mean = torch.mean(y_t, dim=0, keepdim=True)
        y_std = torch.std(y_t, dim=0, keepdim=True)
        y_std[y_std < 1e-6] = 1
        self.y_std = y_std
        self.y_mean = y_mean
        y_t = (y_t - y_mean) / y_std
        # print(f"Fitting model for {X_t.shape[1]} task(s) with {X_t.shape[0]} points")
        self.pred_layers = self.mn_model.calc_nn_weights(X_t, y_t)
        if self.mn_model.bins_output_n > 0:
            y_min = y_t.amin(dim=0, keepdim=True)
            y_max = y_t.amax(dim=0, keepdim=True)
            y_diff = (y_max - y_min) / self.mn_model.bins_output_n
            idx = torch.arange(self.mn_model.bins_output_n, device=self.mn_model.get_device())[None, :]
            self.y_ruler = idx * y_diff + y_min
        return self
    
    @torch.inference_mode()
    def predict(self, X: np.ndarray, batch_size: int = 1024):
        # X_std = np.std(X, axis=0, keepdims=True)
        # X_std[X_std < 1e-6] = 1
        # X = (X - np.mean(X, axis=0, keepdims=True)) / X_std
        X = self.mn_model.pad_zeros(X)
        layers = getattr(self, 'pred_layers', None)
        if layers is None:
            raise RuntimeError("Fit model first")
        if X.ndim == 2:
            X = X[:, None, :]
        X_t = torch.tensor(X).to(device=self.mn_model.get_device(), dtype=torch.get_default_dtype())
        ds = TensorDataset(X_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        pred_idx_list = []
        for x_b, in dl:
            cur_pred = self.mn_model.nn_forward(x_b, layers)
            if self.mn_model.bins_output_n > 0:
                pred_idx_list.append(
                    torch.argmax(cur_pred, dim=-1)
                )
            else:
                pred_idx_list.append(cur_pred)
        pred_idx = torch.cat(pred_idx_list, dim=0)
        if self.mn_model.bins_output_n > 0:
            y = torch.take_along_dim(self.y_ruler, pred_idx.T, dim=-1).T
        else:
            y = pred_idx
        y = y * self.y_std + self.y_mean
        pred = y.cpu().numpy()
        return pred.squeeze()
