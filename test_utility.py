import torch
import numpy as np
from abc import ABC, abstractmethod
from functools import cache
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from collections import defaultdict
from ticl.priors.mlp import MLPPrior
from sklearn.datasets import make_regression
from tnw_cate_funcs import SpiralFunc, PowFunc, StepFunc, LogFunc
import matplotlib.pyplot as plt
import json
import sys
from sklearn.preprocessing import QuantileTransformer

def normalize_data(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1
    return (x - mean) / std

def normalize_cate(y_0: np.ndarray,
                   y_1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sigma = np.std(np.concatenate((y_0, y_1), axis=0), axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1
    mean_0 = np.mean(y_0, axis=0, keepdims=True)
    mean_1 = np.mean(y_1, axis=0, keepdims=True)
    return (y_0 - mean_0) / sigma, (y_1 - mean_1) / sigma 

class DataGenerator(ABC):
    def __init__(self, train_n: int, test_n: int, 
                 treatment_part: float, feat_n: int):
        self.train_n = train_n
        self.test_n = test_n
        self.treatment_part = treatment_part
        self.feat_n = feat_n
        self.w = None
        self.X = None
        self.y_0 = None
        self.y_1 = None

    def gen_default_w(self) -> None:
        return np.random.binomial(1, self.treatment_part, self.train_n)
    
    @abstractmethod
    def gen_all_data(self) -> None:
        pass

    def get_train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.X is None:
            self.gen_all_data()
            assert self.X.shape[0] == self.train_n + self.test_n
        X = self.X[:self.train_n]
        w = self.w #[:self.train_n]
        y_0 = self.y_0[:self.train_n]
        y_1 = self.y_1[:self.train_n]
        y = np.take_along_axis(np.stack((y_0, y_1), axis=-1), w[:, None], -1)[:, 0]
        return X, y, w

    def get_test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.X is None:
            self.gen_all_data()
            assert self.X.shape[0] == self.train_n + self.test_n
        X = self.X[self.train_n:]
        diff = self.y_1[self.train_n:] - self.y_0[self.train_n:]
        return X, diff

class MLPGenerator(DataGenerator):
    config = {"pre_sample_causes": True,
        "sampling": 'normal',  # hp.choice('sampling', ['mixed', 'normal']), # uniform
        'prior_mlp_scale_weights_sqrt': True,
        'random_feature_rotation': True,
        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True, 'lower_bound': 2},
        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 'lower_bound': 4},
        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
        # This mustn't be too high since activations get too large otherwise
        "init_std": {'distribution': 'log_uniform', 'min': 1e-2, 'max': 12},
        "noise_std": 0.0,
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
        'add_uninformative_features': False
    }

    def __init__(self, train_n: int, test_n: int,
                 treatment_part: float, feat_n: int):
        super().__init__(train_n, test_n, treatment_part, feat_n)
        self.mlp_prior = MLPPrior(self.config)

    def gen_all_data(self):
        self.w = self.gen_default_w()
        X, y_0, y_1 = self.mlp_prior.get_batch(1, self.train_n + self.test_n,
                                          self.feat_n, 'cpu')
        X = normalize_data(X[:, 0, :].numpy())
        y_0, y_1 = normalize_cate(y_0.ravel().numpy(), y_1.ravel().numpy())
        self.X = X
        self.y_0 = y_0
        self.y_1 = y_1

class MLPNoiseGenerator(MLPGenerator):
    def __init__(self, train_n: int, test_n: int,
                 treatment_part: float, feat_n: int):
        self.config = self.config.copy()
        self.config['noise_std'] = {'distribution': 'log_uniform', 'min': 1e-2, 'max': .3}
        super().__init__(train_n, test_n, treatment_part, feat_n)

class LinearGenerator(DataGenerator):
    def __init__(self, train_n: int, test_n: int, treatment_part: float,
                 feat_n: int, informative_n: int):
        super().__init__(train_n, test_n, treatment_part, feat_n)
        self.informative_n = informative_n

    def gen_all_data(self):
        self.w = self.gen_default_w()
        X, y_0, coeffs_0 = make_regression(n_samples=self.train_n + self.test_n,
                                           n_features=self.feat_n,
                                           n_informative=self.informative_n,
                                           coef=True)
        idx = np.argwhere(~np.isclose(coeffs_0, 0)).ravel()
        rng = np.random.default_rng()
        idx_to_change = rng.choice(idx, idx.shape[0] // 5, False, shuffle=True)
        coeffs_1 = coeffs_0.copy()
        coeffs_1[idx_to_change] = 0
        y_1 = np.sum(X * coeffs_1[None, :], axis=-1)
        y_0 += np.random.normal(0, 0.1 * np.std(y_0), y_0.shape)
        y_1 += np.random.normal(0, 0.1 * np.std(y_1), y_1.shape)
        self.X = normalize_data(X)
        # self.X = X
        self.y_0, self.y_1 = normalize_cate(y_0.ravel(), y_1.ravel())
        # y = np.take_along_axis(np.stack((y_0[:self.train_n],
        #                                  y_1[:self.train_n]), axis=-1), self.w[:, None], -1)[:, 0]
        # mean = y.mean()
        # std = y.std()
        # if std < 1e-8:
        #     std = 1
        # self.y_0 = (y_0 - mean) / std
        # self.y_1 = (y_1 - mean) / std

class QuadraticGenerator(DataGenerator):
    def __init__(self, train_n: int, test_n: int, treatment_part: float,
                 feat_n: int):
        super().__init__(train_n, test_n, treatment_part, feat_n)

    def gen_all_data(self):
        A = np.random.normal(0, 1, (self.feat_n, self.feat_n))
        A = A @ A.T
        x1 = np.random.normal(-3, 1, ((self.train_n + self.test_n) // 2, self.feat_n))
        x2 = np.random.normal(3, 1, (self.train_n + self.test_n - x1.shape[0], self.feat_n))
        x = np.concatenate((x1, x2), axis=0)
        rng = np.random.default_rng()
        rng.shuffle(x, axis=0)
        y_0 = np.sum(np.sum(x[..., None] * A[None, ...], axis=1) * x, axis=-1)
        c = np.random.uniform(-100, 100, self.feat_n)
        c_num = min(16, self.feat_n)
        idx = np.random.choice(self.feat_n, c_num, replace=False)
        mask = np.ones(self.feat_n, dtype=bool)
        mask[idx] = False
        c[mask] = 0
        y_1 = np.clip(y_0 + np.sum(x * c[None, :], axis=-1), a_min=0, a_max=None)
        self.X = normalize_data(x)
        self.y_0, self.y_1 = normalize_cate(y_0.ravel(), y_1.ravel())
        # self.y_0 += np.random.normal(0, 0.05, self.train_n + self.test_n)
        # self.y_1 += np.random.normal(0, 0.05, self.train_n + self.test_n)
        self.w = self.gen_default_w()


class FriedmanGenerator(DataGenerator):
    def __init__(self, train_n: int, test_n: int, treatment_part: float,
                 feat_n: int, noise_std: int):
        assert feat_n >= 4
        super().__init__(train_n, test_n, treatment_part, feat_n)
        self.noise_std = noise_std

    def gen_all_data(self):
        self.w = self.gen_default_w()
        n = self.train_n + self.test_n
        X = np.random.uniform(0, 1, (n, self.feat_n))
        X[:, 0] *= 100
        X[:, 1] *= 520 * np.pi
        X[:, 1] += 40 * np.pi
        X[:, 3] *= 10
        X[:, 3] += 1
        y_0 = (X[:, 0] ** 2 + (X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5 + self.noise_std * np.random.normal(0, 1, n)
        y_1 = y_0.copy()
        effect_mask = (((X[:, 0] < 80) & (X[:, 0] > 20)) | (X[:, 1] < 80 * np.pi)) & (X[:, 3] > 4)
        y_1[effect_mask] += 500
        y_1 += self.noise_std * np.random.normal(0, 1, n)
        # y_1 = np.arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]) + self.noise_std * np.random.normal(0, 1, n)
        X = np.tile(X, (1, 16))
        self.X = normalize_data(X)
        self.y_0, self.y_1 = normalize_cate(y_0.ravel(), y_1.ravel())
        

class SingleParameterGenerator(DataGenerator, ABC):
    def __init__(self, train_n: int, test_n: int,
                 treatment_part: float, feat_n: int,
                 t_bounds):
        super().__init__(train_n, test_n, treatment_part, feat_n)
        self.t_bounds = t_bounds

    @abstractmethod
    def get_trt_func(self):
        pass

    @abstractmethod
    def get_cnt_func(self):
        pass
        
    def gen_all_data(self):
        self.w = self.gen_default_w()
        trt_func = self.get_trt_func()
        cnt_func = self.get_cnt_func()
        # t = np.random.uniform(*self.t_bounds, self.train_n + self.test_n)
        t = torch.distributions.continuous_bernoulli.ContinuousBernoulli(0.9).sample((self.train_n + self.test_n,))
        t = t.numpy() * (self.t_bounds[1] - self.t_bounds[0]) + self.t_bounds[0]
        X = cnt_func.calc_x(t)
        X = np.tile(cnt_func.calc_x(t), (1, 3))
        y_0 = cnt_func.calc_y(t)
        y_1 = trt_func.calc_y(t)
        y = np.take_along_axis(np.stack((y_0[:self.train_n],
                                         y_1[:self.train_n]), axis=-1), self.w[:, None], -1)[:, 0]
        mean = y.mean()
        std = y.std()
        if std < 1e-8:
            std = 1
        self.y_0 = (y_0 - mean) / std
        self.y_1 = (y_1 - mean) / std
        self.X = normalize_data(X)
        # self.y_0, self.y_1 = normalize_cate(y_0, y_1)

class SpiralGenerator(SingleParameterGenerator):
    def __init__(self, train_n: int, test_n: int,
                 treatment_part: float, feat_n: int):
        super().__init__(train_n, test_n, treatment_part, feat_n, (0, 10))
        self.trt_coef_bnd = (8, 10)
        self.cnt_coef_bnd = (1, 4)

    def get_trt_func(self):
        return SpiralFunc(self.trt_coef_bnd, self.feat_n)

    def get_cnt_func(self):
        return SpiralFunc(self.cnt_coef_bnd, self.feat_n)

class PowGenerator(SingleParameterGenerator):
    def __init__(self, train_n: int, test_n: int,
                 treatment_part: float, feat_n: int):
        super().__init__(train_n, test_n, treatment_part, feat_n, (0, 5))

    def get_trt_func(self):
        return PowFunc(self.feat_n, np.random.uniform(2, 4), np.random.uniform(1, 2), 2.5)

    def get_cnt_func(self):
        return PowFunc(self.feat_n, np.random.uniform(1, 2), np.random.uniform(0.25, 1), 2.5)

class LogGenerator(SingleParameterGenerator):
    def __init__(self, train_n: int, test_n: int,
                 treatment_part: float, feat_n: int):
        super().__init__(train_n, test_n, treatment_part, feat_n, (0.5, 5))
        
    def get_trt_func(self):
        return LogFunc(self.feat_n, (-4, -1), (1, 4), False)

    def get_cnt_func(self):
        return LogFunc(self.feat_n, (-4, -1), (1, 4), True)

class StepGenerator(DataGenerator):
    def __init__(self, train_n: int, test_n: int,
                 treatment_part: float, feat_n: int):
        super().__init__(train_n, test_n, treatment_part, feat_n)
        self.x_bnd = (-1, 1)
        self.func = StepFunc(self.feat_n)

    def gen_all_data(self):
        self.w = self.gen_default_w()
        X = np.random.uniform(*self.x_bnd, (self.train_n + self.test_n, self.feat_n))
        y_0 = self.func.get_control_y(X)
        y_1 = self.func.get_treat_y(X)
        self.X = normalize_data(X)
        # y = np.take_along_axis(np.stack((y_0[:self.train_n],
        #                                  y_1[:self.train_n]), axis=-1), self.w[:, None], -1)[:, 0]
        # mean = y.mean()
        # std = y.std()
        # if std < 1e-8:
        #     std = 1
        # self.y_0 = (y_0 - mean) / std
        # self.y_1 = (y_1 - mean) / std
        # self.X = normalize_data(X)
        self.y_0, self.y_1 = normalize_cate(y_0.ravel(), y_1.ravel())

@cache
def get_ihdp_ds():
    data = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_2.csv",
                           header=None)
    W = data.iloc[:, 0].to_numpy().astype(np.int64)
    Y0 = data.iloc[:, 3].to_numpy().astype(np.float64).ravel()
    Y1 = data.iloc[:, 4].to_numpy().astype(np.float64).ravel()
    X = data.iloc[:, 5:].to_numpy().astype(np.float64)
    X = normalize_data(X)
    Y0, Y1 = normalize_cate(Y0, Y1)
    return X, Y0, Y1, W

class IHDPGenerator:
    def __init__(self, train_part: float,
                 treatment_part: float):
        assert 0 < train_part < 1 
        self.train_part = train_part
        self.treatment_part = treatment_part
        self.X = None

    def gen_all_data(self) -> None:
        X, y_0, y_1, w = get_ihdp_ds()
        self.train_n = int(X.shape[0] * self.train_part)
        self.test_n = X.shape[0] - self.train_n
        self.treatment_n = round(self.treatment_part * w.shape[0])
        rng = np.random.default_rng()
        idx = np.arange(X.shape[0])
        rng.shuffle(idx)
        self.X, self.y_0, self.y_1, self.w = X[idx], y_0[idx], y_1[idx], w[idx]
        idx_treated = np.argwhere(w == 1).ravel()[:self.treatment_n]
        w = np.zeros(self.X.shape[0], dtype=np.int64)
        w[idx_treated] = 1
        

    def get_train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.X is None:
            self.gen_all_data()
        X = self.X[:self.train_n]
        w = self.w[:self.train_n]
        y_0 = self.y_0[:self.train_n]
        y_1 = self.y_1[:self.train_n]
        y = np.take_along_axis(np.stack((y_0, y_1), axis=-1), w[:, None], -1)[:, 0]
        return X, y, w

    def get_test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.X is None:
            self.gen_all_data()
        X = self.X[self.train_n:]
        diff = self.y_1[self.train_n:] - self.y_0[self.train_n:]
        return X, diff

class DGFactory():
    class FakeList():
        def __init__(self, value):
            self.value = value

        def __getitem__(self, i):
            return self.value
    
    def __init__(self, generator_cls: DataGenerator,
                 inst_n: int, **kw):
        for o in kw.values():
            if type(o) is list and len(o) != inst_n:
                raise RuntimeError("Pass either list with length of inst_n or one value")
        self.inst_n = inst_n
        self.iter = 0
        self.dg_cls = generator_cls
        self.kw = {}
        for k, v in kw.items():
            self.kw[k] = v if type(v) is list else self.FakeList(v)

    def get_generator(self) -> DataGenerator:
        cur_kw = {k: v[self.iter] for k, v in self.kw.items()}
        dg = self.dg_cls(**cur_kw)
        self.iter += 1
        return dg

class CVLearner(ABC):
    class GSDummy:
        def __init__(self, base_cls, param_grid):
            self.base_cls = base_cls
            self.param_grid = param_grid
            self.model = None

        def fit(self, X, y):
            self.model = self.base_cls(**self.param_grid).fit(X, y)
            return self

        @property
        def best_estimator_(self):
            return self.model

        def get_params(self):
            return self.param_grid
    
    def __init__(self, base_cls, fold_n: int,
                 param_grid, use_cv: bool = True, 
                 random_state=None):
        self.fold_n = fold_n
        self.base_cls = base_cls
        self.random_state = random_state
        self.param_grid = param_grid.copy()
        self.best_params = None
        self.use_cv = use_cv

    def get_grid_search(self, **kw):
        if self.use_cv:
            return GridSearchCV(
                estimator=self.base_cls(),
                param_grid=self.param_grid,
                scoring='neg_mean_squared_error',
                n_jobs=8,
                cv=KFold(self.fold_n, random_state=self.random_state, shuffle=True),
                **kw
            )
        else:
            return self.GSDummy(base_cls=self.base_cls, param_grid=self.param_grid)

    @abstractmethod
    def fit(self, X, y, w):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def set_params(self, params) -> None:
        if params is not None:
            self.best_params = params.copy()

    def get_params(self) -> Dict | None:
        return self.best_params

class TLearner(CVLearner):
    def fit(self, X, y, w):
        w_0, w_1 = w == 0, w == 1
        X_0, y_0 = X[w_0, ...], y[w_0]
        X_1, y_1 = X[w_1, ...], y[w_1]
        if self.get_params() is None:
            gs = self.get_grid_search()
            self.model_0 = gs.fit(X_0, y_0).best_estimator_
        else:
            self.model_0 = self.base_cls(**self.get_params()).fit(X_0, y_0)
        self.model_1 = self.base_cls(**self.model_0.get_params()).fit(X_1, y_1)
        self.set_params(self.model_0.get_params())
        return self

    def predict(self, X):
        return self.model_1.predict(X) - self.model_0.predict(X)

class SLearner(CVLearner):
    def fit(self, X, y, w):
        w_0, w_1 = w == 0, w == 1
        X_0, y_0 = X[w_0, ...], y[w_0]
        X_all = np.concatenate((X, w[:, None]), axis=-1)
        if self.get_params() is None:
            gs = self.get_grid_search(refit=False)
            gs.fit(X_0, y_0)
            self.model = self.base_cls(**gs.best_params_).fit(X_all, y)
        else:
            self.model = self.base_cls(**self.get_params()).fit(X_all, y)
        self.set_params(self.model.get_params())
        return self

    def predict(self, X):
        X_0 = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=-1)
        X_1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=-1)
        return self.model.predict(X_1) - self.model.predict(X_0)

class XLearner(CVLearner):
    def fit(self, X, y, w):
        self.g = w.sum() / w.shape[0]
        w_0, w_1 = w == 0, w == 1
        X_0, y_0 = X[w_0, ...], y[w_0]
        X_1, y_1 = X[w_1, ...], y[w_1]
        if self.get_params() is None:
            gs = self.get_grid_search()
            mu_0 = gs.fit(X_0, y_0)
        else:
            mu_0 = self.base_cls(**self.get_params()).fit(X_0, y_0)
        mu_1 = self.base_cls(**mu_0.get_params()).fit(X_1, y_1)
        d_0 = mu_1.predict(X_0) - y_0
        d_1 = y_1 - mu_0.predict(X_1)
        self.tau_0 = self.base_cls(**mu_0.get_params()).fit(X_0, d_0)
        self.tau_1 = self.base_cls(**mu_0.get_params()).fit(X_1, d_1)
        self.set_params(mu_0.get_params())
        return self

    def predict(self, X):
        return self.g * self.tau_0.predict(X) + (1 - self.g) * self.tau_1.predict(X)

LEARNERS = {
    'T-Learner': TLearner,
    'S-Learner': SLearner,
    'X-Learner': XLearner,
}

def get_model_name(model: str, learner: str):
    return f"{learner} {model}"

def comparison_experiment(data_gen_factory: DGFactory,
                          mn_models: Dict[str, Any],
                          cmp_cls_list: Dict[str, Any],
                          gs_params: Dict[str, Dict[str, List]],
                          gs_flags: Dict[str, bool],
                          exp_iters: int,
                          fold_n: int,
                          random_state: int | None = None,
                          tqdm_desc: str='') -> Dict[str, List[float]]:
    result = defaultdict(list)
    for i in range(1, exp_iters + 1):
        prog_bar = tqdm(None, tqdm_desc + f'Iter {i}', unit='model', ascii=True,
                        total=len(mn_models) + len(cmp_cls_list) * len(LEARNERS))
        data_generator = data_gen_factory.get_generator()
        X_train, y_train, w_train = data_generator.get_train()
        X_test, cate_test = data_generator.get_test()
        for model_name, model in mn_models.items():
            prog_bar.set_postfix_str(f"Training {model_name}")
            model.fit(X_train, y_train, w_train)
            prog_bar.set_postfix_str(f"Testing {model_name}")
            cate_pred = model.predict(X_test)
            result[model_name].append(mean_squared_error(cate_test, cate_pred))
            prog_bar.update()
        for cls_name, base_cls in cmp_cls_list.items():
            best_params = None
            for learner_name, learner in LEARNERS.items():
                model_name = get_model_name(cls_name, learner_name) 
                prog_bar.set_postfix_str(f"Training {model_name}")
                model = learner(base_cls=base_cls, fold_n=fold_n,
                                param_grid=gs_params[cls_name], 
                                use_cv=gs_flags.get(cls_name, True),
                                random_state=random_state)
                model.set_params(best_params)
                model.fit(X_train, y_train, w_train)
                if best_params is None:
                    best_params = model.get_params()
                prog_bar.set_postfix_str(f"Testing {model_name}")
                cate_pred = model.predict(X_test)
                result[model_name].append(mean_squared_error(cate_test, cate_pred))
                prog_bar.update()
        prog_bar.close()
    return result

def count_stats(data: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    res = defaultdict(list)
    for d in data:
        for k, v in d.items():
            v_non_outliers = np.asarray(v)
            v_non_outliers[v_non_outliers > 10] = np.nan
            res[k].append(np.nanmean(v_non_outliers))
        for k in res:
            res[k] = sorted(res[k], key=lambda x: -x)
    return res

def make_line_style(models_style: Dict[str, Dict]):
    if models_style is None:
        models_style = {}
    learners_style = {
        'T-Learner': {'markersize': 8.5, 'markerfacecolor': 'white', 'marker': '^'},
        'S-Learner': {'markersize': 6.5, 'markerfacecolor': 'white', 'marker': 'o'},
        'X-Learner': {'markersize': 8.5, 'markerfacecolor': 'white', 'marker': 'X'},
    }
    mn_style = {'markersize': 10, 'markerfacecolor': 'white', 'marker': '*'}
    result_style = {}
    for key, val in models_style.items():
        result_style[key] = mn_style.copy()
        result_style[key].update(val)
        for lr in LEARNERS.keys():
            model_name = get_model_name(key, lr)
            result_style[model_name] = learners_style[lr].copy()
            result_style[model_name].update(val)
    return result_style

def draw_dep(metric_name: str, x_name: str,
             x_list: List, 
             metrics_list: List[Dict[str, List[float]]],
             groups: Dict[str, List[str]],
             models_style: Dict[str, Dict]):
    mosaic = list(groups.keys())
    common_kw = {'xlabel': x_name}
    subplots_kw = {}
    for i, key in enumerate(mosaic):
        subplots_kw[key] = common_kw.copy()
        if i == 0:
            subplots_kw[key]['ylabel'] = metric_name
        subplots_kw[key]['title'] = key
    fig, axes = plt.subplot_mosaic(
        mosaic=[mosaic],
        sharex=True, sharey=True,
        per_subplot_kw=subplots_kw,
        figsize=(len(mosaic) * 4, 4),
    )
    stats = count_stats(metrics_list)
    fin_line_style = make_line_style(models_style)
    for group_key, group_list in groups.items():
        ax = axes[group_key]
        for model, val in stats.items():
            if model in group_list:
                ax.plot(x_list, val, label=model, **fin_line_style[model])
        ax.grid()
        ax.legend()
    return fig, ax

def load_metrics(path_json: str):
    with open(path_json, 'r') as file:
        d = json.load(file)
        return d
