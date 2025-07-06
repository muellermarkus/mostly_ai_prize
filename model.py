import torch
import torch.nn as nn
import torch.nn.functional as F


class NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings.

    From: https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py

    In other words, each feature embedding is transformed by its own dedicated linear layer.
    """

    def __init__(
        self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming uniform, replicating from nn.Linear
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]
        x = x.transpose(0, 1) # (n_features, batch_size, dim)
        # use broadcasting, it automatically does self.weight.unsqueeze(1)
        x = x @ self.weight # (n, B, in_dim) x (n, in_dim, out_dim)
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x
    
    def forward_single(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Forward pass for a single feature."""
        assert x.shape[-1] == self.weight.shape[-2]
        x = x @ self.weight[i] # (B, in_dim) x (in_dim, out_dim)
        if self.bias is not None:
            x = x + self.bias[i]
        return x


class ARGN(nn.Module):
    def __init__(self, 
                 order_idx, # order of features to be generated
                 n_classes, # list with number of classes for each feature
                 n_cond_classes=None, # list with number of classes for each conditioning feature
                 emb_dim=8, 
                 proportions=None, 
                 set_mask_zero=False, 
                 cond_dim=128
                 ):
        super().__init__()
        
        self.num_features = len(n_classes)
        self.embedding = nn.Embedding(sum(n_classes), emb_dim)
        cat_offset = torch.tensor(n_classes).cumsum(dim=-1)[:-1]
        cat_offset = torch.cat(
            (torch.zeros((1,), dtype=torch.long), cat_offset)
        )
        self.register_buffer("cat_offset", cat_offset)
        
        self.cat_bias = nn.Parameter(torch.zeros((self.num_features, self.num_features * emb_dim)))
        
        if n_cond_classes is not None:
            self.cond_embedding = nn.Embedding(sum(n_cond_classes), emb_dim)
            cond_offset = torch.tensor(n_cond_classes).cumsum(dim=-1)[:-1]
            cond_offset = torch.cat(
                (torch.zeros((1,), dtype=torch.long), cond_offset)
            )
            self.register_buffer("cond_offset", cond_offset)
            
            self.cond_map = nn.Sequential(
                nn.Linear(len(n_cond_classes) * emb_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
                nn.SiLU(),
            )
        
        in_dim = self.num_features * emb_dim + cond_dim * int(n_cond_classes is not None)
        self.processor = nn.Sequential(
            NLinear(self.num_features, in_dim, 2*emb_dim),
            nn.SiLU(),
            NLinear(self.num_features, 2*emb_dim, 1*emb_dim),
            nn.SiLU(),
        )
        
        self.predictors = nn.ModuleList()
        for i, n_class in enumerate(n_classes):
            self.predictors.append(nn.Linear(1*emb_dim, n_class))
            nn.init.zeros_(self.predictors[-1].weight)
            if proportions is None or n_cond_classes is not None:
                nn.init.zeros_(self.predictors[-1].bias)
            else:
                self.predictors[-1].bias = nn.Parameter(proportions[i].log())
        self.proportions = proportions
        
        self.order_idx = order_idx.unsqueeze(0)
        mask = self.generate_mask(order_idx).unsqueeze(-1) # (B, n_features, n_features, 1)
        if set_mask_zero:
            mask = torch.zeros_like(mask)
        self.register_buffer("mask", mask)
              
    @property
    def device(self):
        return next(self.parameters()).device
    
    def derive_embeddings(self, x, cond=None):
        x_emb = self.embedding(x + self.cat_offset)
        x_emb = x_emb.unsqueeze(1) * self.mask # (B, 1, n_features, emb_dim) x (1, n_features, n_features, 1)
        x_emb = x_emb.flatten(2) + self.cat_bias
        
        if cond is not None:
            cond_emb = self.cond_embedding(cond + self.cond_offset) # (B, n_features_cond, emb_dim)
            cond_emb = cond_emb.flatten(1)
            cond_emb = self.cond_map(cond_emb) # (B, cond_dim)
            x_emb = torch.concatenate((x_emb, cond_emb.unsqueeze(1).expand(-1, self.num_features, -1)), dim=-1)
            
        return x_emb

    def forward(self, x, cond=None, return_stuff=False):
        x_emb = self.derive_embeddings(x, cond=cond)
        x = self.processor(x_emb)
        logits = [predictor(x[:,i]) for i, predictor in enumerate(self.predictors)]
        
        if not return_stuff:
            return logits
        else:
            return logits, x_emb, x
        
    def loss_fn(self, x, cond):
        logits = self.forward(x, cond)
        
        losses = torch.stack(
            [
                F.cross_entropy(logits[i], x[:, i], reduction="none")
                for i in range(self.num_features)
            ],
            dim=1,
        )

        return losses
    
    def get_accuracy(self, x, cond):
        logits = self.forward(x, cond)
        accs = []
        for i in range(self.num_features):
            acc = (logits[i].argmax(dim=-1) == x[:, i]).float().mean()
            accs.append(acc)
        accs = torch.stack(accs)
        return accs
    
    def generate_mask(self, order_idx):
        """
        order_idx is a tensor of shape (n_features) with entries being features indices corresponding to which feature is generated first, second, etc.
        
        mask = 1 if we condition on this information
        2nd dim: features to be generated
        3rd dim: = 1 for each feature to be generated, what features are condtioned on
        
        Example
        order_idx[0] = [2,1,0]
        first, m[0] = [[1., 1., 0.], # conditions on first two generated features
                    [1., 0., 0.], # conditions on first generated feature
                    [0., 0., 0.]] # conditions on none
        have to bring this into the correct order to form conditioning mask:
        second, m[0] = [[0., 1., 1.], # mask for feature 1, such that feature 2 and 3 are condtioned on
                    [0., 0., 1.], # mask for feature 2, such that feature 3 is conditioned on
                    [0., 0., 0.]] # mask for feature 3, such that no feature is conditioned on
        """
        
        assert order_idx.shape[0] == self.num_features
        idx = order_idx.argsort()
        ones = torch.ones(order_idx.shape[0], order_idx.shape[0], dtype=torch.int32)
        mask = torch.tril(ones, diagonal=-1)
        mask = mask[idx,:][:,idx].unsqueeze(0)
        
        return mask
    
    @torch.inference_mode()
    def generate(self, num_samples, cond=None, batch_size=4096, get_logits=False):
        
        n_batches, remainder = divmod(num_samples, batch_size)
        sample_sizes = (
            n_batches * [batch_size] + [remainder]
            if remainder != 0
            else n_batches * [batch_size]
        )
        
        if cond is not None:
            assert cond.shape[0] == num_samples
            cond_parts = torch.split(cond, sample_sizes, dim=0)
        
        x_list = []
        logits_list = []
        for i, n_samples in enumerate(tqdm(sample_sizes)):
            
            x = torch.zeros(n_samples, self.num_features, dtype=torch.long, device=self.device)
            sample_logits = []
            for feat_idx in range(self.num_features):

                cond_part = cond_parts[i].to(self.device) if cond is not None else None
                logits = self.forward(x, cond=cond_part)[self.order_idx[0, feat_idx]]

                # convert to probabilities and sample
                p_i = F.softmax(logits, dim=-1)
                x[:, self.order_idx[0, feat_idx]] = torch.multinomial(p_i, 1, replacement=True).squeeze()
                sample_logits.append(logits)
            logits_list.append(torch.column_stack(sample_logits).cpu())
            x_list.append(x.cpu())
            
        if get_logits:
            return torch.row_stack(x_list), torch.row_stack(logits_list)
                
        return torch.row_stack(x_list)
    
    
    
    

def save_model(model, to_disk=False):
    if not to_disk:
        state_dict = model.state_dict()
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        serialized_bytes = buffer.getvalue() # uncompressed pickle
        state_dict_gzip = gzip.compress(serialized_bytes)
        return state_dict_gzip
    else:
        state_dict = model._orig_mod.state_dict()         
        torch.save(state_dict, os.path.join('results/final_argn/model.pt'))

def load_state(best_model_state):
    decompressed = gzip.decompress(best_model_state)
    buffer = io.BytesIO(decompressed)
    loaded_state = torch.load(buffer)
    return loaded_state