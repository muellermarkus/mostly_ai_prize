import os

import numpy as np
import pandas as pd
import polars as pl
import torch
from dython.nominal import associations
from mostlyai import qa
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_preprocess import DataPreprocessor
from data_utils import DataEditor, count_to_get_probs
from model import ARGN, load_state, save_model
from utils import set_seeds

#######################################
# DATA PREPROCESSING
#######################################

set_seeds(42, cuda_deterministic=True)

# Path to the training data file
file_path = 'data/flat-training.csv.gz'  
dp = DataPreprocessor(file_path, num_quantiles=100, train_batch_size=4096, val_batch_size=4096, cont_encoder=1, train_prop=0.9, condition_on_binary=False, cat_min_frequency=10)
train_loader, val_loader = dp.preprocess()


#######################################
# DECISION WHICH FEATURES TO CONDITION ON
# essentially about defining p(x_0) in factorization p(x_0) p(x_1 | x_0)
# here p(x_0) may be multivariate!
#######################################

col_names = list(dp.cat_cols) + [f"num_enc_{i}" for i in range(train_loader.X_num_enc.shape[1])]
x = torch.column_stack((train_loader.X_cat, train_loader.X_num_enc))

# retrieve features with most strong associations to other features
res = associations(pd.DataFrame(x), nominal_columns='all', compute_only=True)
strong_res = res['corr'] > 0.25
test = strong_res.sum(axis=0).sort_values(ascending=False)
select_idx = test[test >= 15].index.to_numpy()

n_classes = dp.X_cat_n_classes + dp.X_num_n_classes
select_n_classes = torch.tensor(n_classes)[select_idx]
select_labels = [col_names[i] for i in select_idx]

# clean selected features: we do not want to condition on features with too many classes
# this would hurt privacy a lot
cleaned_select_idx = []
cleaned_n_classes = []
cleaned_labels = []

for i, idx in enumerate(select_idx):
    if select_n_classes[i] > 100:
        print(f"Feature {select_labels[i]} has {select_n_classes[i]} classes, skipping")
    else:
        cleaned_select_idx.append(idx.item())
        cleaned_n_classes.append(select_n_classes[i].item())
        cleaned_labels.append(select_labels[i])
        
# sort features by number of classes 
# to increase likelihood of more features making it into conditioning set without hurting privacy
new_order = torch.tensor(cleaned_n_classes).argsort()
cleaned_select_idx = torch.tensor(cleaned_select_idx)[new_order].tolist()
       
# load clean data
def sample_from_probs(count_data, count_probs, n_samples=1000):
    idx = np.random.choice(len(count_probs), size=n_samples, replace=True, p=count_probs)
    return count_data[idx]

df_prob = pl.DataFrame(torch.column_stack((train_loader.X_cat, train_loader.X_num_enc)).numpy(),
                       schema=col_names)

aux_idx = []
for v in cleaned_select_idx:
    aux_idx.append(v)
    aux_labels = [col_names[i] for i in aux_idx]
    count_data, _ = count_to_get_probs(df_prob, aux_labels)
    if count_data.height > 50000: #FIXME: make this relative to total number of samples! 
        aux_idx = aux_idx[:-1]  # remove last feature
        break
cleaned_select_idx = aux_idx

# TODO: add all bivariate features, they have close to 100% accuracy anyways
# bin_cols = [lab for i, lab in enumerate(dp.cat_cols) if torch.tensor(dp.X_cat_n_classes)[i] == 2]
# cleaned_select_idx += [i for i, col in enumerate(col_names) if col in bin_cols]

# update selected labels
select_labels = [col_names[i] for i in cleaned_select_idx]
select_n_classes = torch.tensor(n_classes)[cleaned_select_idx].tolist()
    
# get groups and probabilities to sample from, this defines p(x_0)
count_data, count_probs = count_to_get_probs(df_prob, select_labels)


#######################################
# TRAINING
#######################################

epochs = 1000
lr = 1e-3/3
device = 'cuda'
torch.set_float32_matmul_precision('high') # use TensorFloat32

# need to adjust data loaders from data preprocessor for conditioning features
data_editor = DataEditor(select_labels) 
train_loader, val_loader = data_editor.adjust_loaders(train_loader, val_loader, dp)

# generate from small to large cardinality
n_classes = data_editor.cat_n_classes + data_editor.num_n_classes
order_idx = torch.tensor(n_classes).argsort()

argn = ARGN(order_idx, n_classes, select_n_classes, emb_dim=16)
argn = torch.compile(argn).to(device)
n_params = sum(p.numel() for p in argn.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {n_params}")

optimizer = torch.optim.AdamW(argn.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr = 0.01*lr)


patience = 0
best_loss = float('inf')
writer = SummaryWriter(os.path.join('results', 'tb'))
epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs")
for e in epoch_bar:
    n_inputs = 0
    n_val_inputs = 0
    acc_train_loss = 0.0
    acc_val_loss = 0.0
    
    for batch in train_loader:
        mask, X_bin, X_cat, X_num_enc, X_num, add_cond = [x.to(device) if x is not None else None for x in batch]
        B = len(X_cat)
        n_inputs += B
        x = torch.column_stack((X_cat, X_num_enc))
        optimizer.zero_grad(set_to_none=True)
        loss = argn.loss_fn(x, cond=add_cond).mean()
        loss.backward()
        optimizer.step()
        acc_train_loss += loss * B
        
    with torch.inference_mode():
        
        argn.eval()
        for batch in val_loader:
            mask, X_bin, X_cat, X_num_enc, X_num, add_cond = [x.to(device) if x is not None else None for x in batch]
            B = len(X_cat)
            x = torch.column_stack((X_cat, X_num_enc))            
            loss = argn.loss_fn(x, cond=add_cond).mean()
            acc_val_loss += loss * B
            n_val_inputs += B
        argn.train()
        
    train_loss = acc_train_loss / n_inputs
    val_loss = acc_val_loss / n_val_inputs
    scheduler.step(val_loss)
    
    if val_loss < best_loss:
        best_loss = val_loss
        patience = 0
        best_model_state = save_model(argn)
    else:
        patience += 1
        if patience > 20:
            print(f"Early stopping at epoch {e} with best loss {best_loss:.4f}")
            break
    
    # metric logging
    train_dict = {'train loss': train_loss, 'val loss': val_loss}
    epoch_bar.set_postfix({"train loss": f"{train_loss:.4f}"})
    for metric_name, metric_value in train_dict.items():
        writer.add_scalar('losses/{}'.format(metric_name), metric_value, global_step=e)

# load best model from checkpoint
argn.load_state_dict(load_state(best_model_state))
argn.eval()


#######################################
# SAMPLING
#######################################

df_trn = dp.load_train_data().to_pandas()
df_val = dp.load_val_data().to_pandas()
best_acc = 0.0

for i in range(20):
    set_seeds(i * 100, cuda_deterministic=True)
    cond_gen = sample_from_probs(count_data, count_probs, n_samples=100_000).to_torch().long()
    gen_samples = argn.generate(100_000, cond=cond_gen, batch_size=5000)
    X_num_enc_gen = gen_samples[:, len(data_editor.new_X_cat_cols):]
    X_cat_gen = gen_samples[:, :len(data_editor.new_X_cat_cols)]
    assert X_num_enc_gen.shape[1] == len(data_editor.new_X_num_enc_cols)
    
    X_cat_gen, X_num_enc_gen = data_editor.adjust_output(X_cat_gen, X_num_enc_gen, cond_gen)
    df_gen = dp.postprocess(None, X_cat_gen, X_num_enc_gen, sample_uniformly=True)
    report_path, metrics = qa.report(
        syn_tgt_data=df_gen.to_pandas(),
        trn_tgt_data=df_trn,
        hol_tgt_data=df_val,
        report_path= f'results/qa_report_{i}.html'
    )
    
    if metrics.accuracy.overall > best_acc:
        best_acc = metrics.accuracy.overall
        print(f"New best accuracy: {best_acc:.6f} at iteration {i}")
        df_gen.write_csv('results/submission.csv')
