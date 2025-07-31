import torch


class BagPrior:
    def __init__(self, base_priors, prior_weights, verbose=False):
        self.base_priors = base_priors
        # let's make sure we get consistent sorting of the base priors by name
        self.prior_names = sorted(base_priors.keys())
        self.prior_weights = prior_weights
        self.verbose = verbose

    def get_batch(
        self, 
        *, 
        batch_size, 
        n_samples, 
        num_features, 
        device, 
        epoch=None, 
        single_eval_pos=None
    ):
        args = {
            'device': device, 
            'n_samples': n_samples, 
            'num_features': num_features,
            'batch_size': batch_size, 
            'epoch': epoch, 
            'single_eval_pos': single_eval_pos
        }

        weights = torch.tensor([self.prior_weights[prior_name] for prior_name in self.prior_names], dtype=torch.float)
        weights = weights / torch.sum(weights)
        batch_assignments = torch.multinomial(weights, 1, replacement=True).numpy()

        x, y_0, y_1, info = self.base_priors[self.prior_names[int(batch_assignments[0])]].get_batch(**args)
        return x.detach(), y_0.detach(), y_1.detach(), info
