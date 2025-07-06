
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler

from data_utils import (
    BinaryGenerator,
    CatEncoder,
    ContEncoder,
    ContEncoder2,
    DigitEncoder,
    FastTensorDataLoader,
    determine_column_types,
    determine_digit_encoding,
)
from utils import set_seeds


class DataPreprocessor:
    """
    Does not encode data based on validation and training set, only on training set!
    """
    def __init__(self, file_path, seed=42, num_quantiles=10, encode_min_max=True, train_prop=0.9,
                 train_batch_size=2048, val_batch_size=2048, cat_min_frequency=5, condition_on_binary=True,
                 encode_cat_jointly=False, cont_encoder=1, digit_prec_threshold=3, max_unique_vals=200):
        self.file_path = file_path
        self.seed = seed
        self.train_prop = train_prop
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_quantiles = num_quantiles
        self.encode_min_max = encode_min_max
        self.cat_min_frequency = cat_min_frequency
        self.condition_on_binary = condition_on_binary
        self.encode_cat_jointly = encode_cat_jointly
        self.cont_encoder = cont_encoder
        self.digit_prec_threshold = digit_prec_threshold
        self.max_unique_vals = max_unique_vals

    def preprocess(self):
        set_seeds(self.seed)

        # nan_vals = ['NA', 'NULL', 'null', 'nan', 'NaN', 'N/A', 'n/a', '', ' ', 'None', 'none', '?']
        data = pl.read_csv(self.file_path, infer_schema_length=100000)
        self.orig_cols = data.columns
        self.orig_schema = data.schema

        # split data into train and validation set
        idx = np.arange(data.height)
        np.random.shuffle(idx)
        
        n_train_obs = int(self.train_prop * data.height)
        self.train_idx = idx[:n_train_obs]
        self.val_idx = idx[n_train_obs:]
        
        train_data = data[self.train_idx,:]
        val_data = data[self.val_idx,:]

        self.binary_cols, self.cat_cols, num_cols, self.col_to_round_digits = determine_column_types(data, MAX_UNIQUE_VALS=self.max_unique_vals)
        self.digit_cols = determine_digit_encoding(train_data.select(num_cols), digit_prec_treshold=self.digit_prec_threshold)
        
        # update num_cols
        self.num_cols = [c.item() for c in num_cols if c not in self.digit_cols]
     
        # encode digits
        if len(self.digit_cols) > 0:
            self.digit_enc = DigitEncoder(data.select(self.digit_cols))
            X_digit_train = self.digit_enc.encode(train_data.select(self.digit_cols))
            X_digit_val = self.digit_enc.encode(val_data.select(self.digit_cols))
        
        if not self.condition_on_binary:
            self.cat_cols = np.append(self.cat_cols, self.binary_cols)
            self.binary_cols = []

        self.cat_enc = CatEncoder(data.select(self.cat_cols), min_frequency=self.cat_min_frequency)
        X_cat_train = self.cat_enc.transform(train_data.select(self.cat_cols))
        X_cat_val = self.cat_enc.transform(val_data.select(self.cat_cols))
        X_cat_train = torch.tensor(X_cat_train).long()
        X_cat_val = torch.tensor(X_cat_val).long() if val_data.height > 0 else None
        self.X_cat_n_classes = [self.cat_enc.idx_to_stats[i]['n_classes'] for i in range(len(self.cat_cols))]
        
        # encode numerical columns into bins and other categories
        if self.cont_encoder == 1:
            self.num_bin_enc = ContEncoder(data.select(self.num_cols), num_quantiles=self.num_quantiles, encode_min_max=self.encode_min_max)
        elif self.cont_encoder == 2:
            self.num_bin_enc = ContEncoder2(data.select(self.num_cols), num_quantiles=self.num_quantiles, encode_min_max=self.encode_min_max)
            
        X_num_enc_train, mask_train = self.num_bin_enc.encode(train_data.select(self.num_cols))
        X_num_enc_val, mask_val = self.num_bin_enc.encode(val_data.select(self.num_cols)) if val_data.height > 0 else (None, None)
        
        # combine with digit data
        self.x_num_enc_orig_num_features = X_num_enc_train.shape[1]
        if len(self.digit_cols) > 0:
            X_num_enc_train  = np.column_stack((X_num_enc_train, X_digit_train)) 
            X_num_enc_val = np.column_stack((X_num_enc_val, X_digit_val)) if (val_data.height > 0) else None

        X_num_enc_train = torch.tensor(X_num_enc_train).long()
        X_num_enc_val = torch.tensor(X_num_enc_val).long() if val_data.height > 0 else None
        mask_train = torch.tensor(mask_train).bool()
        mask_val = torch.tensor(mask_val).bool() if val_data.height > 0 else None
        self.X_num_n_classes = [len(cats) for cats in self.num_bin_enc.ord_enc.categories_] + self.digit_enc.n_classes
        
        # check number of observations that fall into each bin (cross-table)
        # df_aux = pl.DataFrame(X_num_enc_train)
        # df_aux = df_aux.group_by(pl.all()).len()
        # test = df_aux.filter(pl.col('len') == 1)
        # test.height/df_aux.height # percent have only one observation in bin, 99% already with 10 quantiles...

        # encode binary columns and initialize generator
        if len(self.binary_cols) > 0:
            self.bin_generator = BinaryGenerator(data, self.binary_cols)
            self.bin_enc = OrdinalEncoder()
            self.bin_enc.fit(data.select(self.binary_cols).to_numpy())
            X_bin_train = torch.tensor(self.bin_enc.transform(train_data.select(self.binary_cols).to_numpy())).long()
            X_bin_val = torch.tensor(self.bin_enc.transform(val_data.select(self.binary_cols).to_numpy())).long() if val_data.height > 0 else None
            self.X_bin_n_classes = [2] * len(self.binary_cols)
        else:
            X_bin_train = None
            X_bin_val = None
            self.X_bin_n_classes = None
        
        # encode numerical columns for high-resolution model
        self.num_quant_enc = QuantileTransformer(
                output_distribution = 'normal',
                n_quantiles = max(min(n_train_obs // 30, 1000), 10),
                subsample = int(1e9), random_state=self.seed,
                )
        self.num_quant_enc.fit(data.select(self.num_cols).to_numpy())
        X_num_train = self.num_quant_enc.transform(train_data.select(self.num_cols).to_numpy())
        X_num_val = self.num_quant_enc.transform(val_data.select(self.num_cols).to_numpy()) if val_data.height > 0 else None
        
        # standardize just to be save, in case there are holes in the distribution
        self.num_std_scaler = StandardScaler()
        X_num_train = self.num_std_scaler.fit_transform(X_num_train)
        X_num_val = self.num_std_scaler.transform(X_num_val) if val_data.height > 0 else None

        # ensure correct types
        X_num_train = torch.tensor(X_num_train).float()
        X_num_val = torch.tensor(X_num_val).float() if val_data.height > 0 else None
        
        
        # some tests on privacy protection for continuous features

        # for i in range(X_num_train.shape[1]):
        #     col = pl.Series(X_num_train[:,i])
        #     min_v = col.min()
        #     max_v = col.max()
        #     n_min_vals = col.drop_nulls().sort(descending=False, nulls_last=True).head(10)
        #     n_max_vals = col.drop_nulls().sort(descending=True, nulls_last=True).head(10)
        #     reduced_min = n_min_vals[8]
        #     reduced_max = n_max_vals[8]
        #     if reduced_min != min_v:
        #         print(f'{i}: adjust min for column', col.name, 'from', min_v, 'to', reduced_min)
        #     if reduced_max != max_v:
        #         print(f'{i}: adjust max for column', col.name, 'from', max_v, 'to', reduced_max)
        

        # derive group-specific means
        means = []
        for i in range(len(self.num_cols)):
            df_aux = pl.DataFrame((X_num_train[:,i].numpy(), X_num_enc_train[:,i].numpy()), schema=['num', 'num_enc'])
            df_aux = df_aux.group_by('num_enc').agg(pl.col('num').mean().alias('mean'))
            df_aux = df_aux.sort('num_enc', descending=False)
            m = df_aux.get_column('mean').to_torch()
            
            # pad them to same dimension to simplify later model setup
            m = F.pad(m, (0, max(self.X_num_n_classes) - m.shape[0]), value=0.0, mode='constant')
            means.append(m)
        self.X_num_group_means = torch.stack(means, dim=0)

 
        train_loader = FastTensorDataLoader(
            X_bin=X_bin_train,
            X_cat=X_cat_train,
            X_num_enc=X_num_enc_train,
            X_num=X_num_train,
            mask=mask_train,
            batch_size=min(self.train_batch_size, n_train_obs),
            shuffle=True,
            drop_last=True,
        )
        
        if val_data.height > 0:
            val_loader = FastTensorDataLoader(
                X_bin=X_bin_val,
                X_cat=X_cat_val,
                X_num_enc=X_num_enc_val,
                X_num=X_num_val,
                mask=mask_val,
                batch_size=min(self.val_batch_size, val_data.height),
                shuffle=False,
                drop_last=False,
            )
        else:
            val_loader = None

        return train_loader, val_loader
    
    def load_train_data(self):
        data = pl.read_csv(self.file_path, infer_schema_length=100000)
        return data[self.train_idx,:]
    
    def load_val_data(self):
        data = pl.read_csv(self.file_path, infer_schema_length=100000)
        return data[self.val_idx,:]
        
    def generate_mask(self, X_num_enc_gen):
        
        """
        Generate mask for generated numerical binned data.
        This indicates which values are to be inferred by the high-resolution model.
        """
        _, mask = self.num_bin_enc.decode(X_num_enc_gen)
        return mask
    
    def generate_miss_mask(self, X_num_enc_gen):
        return self.num_bin_enc.get_miss_mask(X_num_enc_gen)

    def postprocess(self, X_bin_gen, X_cat_gen, X_num_enc_gen, X_num_gen=None, sample_uniformly=True):
        """
        Postprocess synthetic data samples.
        Returns full synthetic data frame with same structure as original data.
        """
        
        if self.condition_on_binary:
            X_bin_gen = self.bin_enc.inverse_transform(X_bin_gen)
        X_cat_gen = self.cat_enc.inverse_transform(X_cat_gen)
        
        # assign missing values
        # for i, missing_category in self.idx_to_cat_missing_categories.items():
        #     X_cat_gen[:, i] = np.where(X_cat_gen[:, i] == missing_category, np.nan, X_cat_gen[:, i])
        
        num_cols = self.num_cols
        if X_num_gen is not None:
            X_num_gen = self.num_std_scaler.inverse_transform(X_num_gen)
            X_num_gen = self.num_quant_enc.inverse_transform(X_num_gen)
            
        # masks indicate which values to infer from highres model
        # requires generating binned and categorized numerical data
        if X_num_enc_gen is not None:
            
            if len(self.digit_cols) > 0:
                X_digit_enc_gen = X_num_enc_gen[:, self.x_num_enc_orig_num_features:]
                X_num_enc_gen = X_num_enc_gen[:, :self.x_num_enc_orig_num_features]
                
                # decode digit data
                X_digit_gen = self.digit_enc.decode(X_digit_enc_gen)
                

            if self.cont_encoder == 1:
                if sample_uniformly:
                    X_num_gen = self.num_bin_enc.uniform_sample(X_num_enc_gen)
                else:
                    X_num_dec_gen, masks = self.num_bin_enc.decode(X_num_enc_gen)
                    # overwrite numerical values by num_enc values where necessary
                    # once nan is involved (missing data), X_num_gen is supposed to be nan anyways
                    X_num_gen = masks * X_num_gen + (~masks) * X_num_dec_gen
                    
            elif self.cont_encoder == 2:
                X_num_gen, masks = self.num_bin_enc.decode(X_num_enc_gen)
                
                
            if len(self.digit_cols) > 0:
                # combine digit data with numerical data
                X_num_gen = np.column_stack((X_num_gen, X_digit_gen))
                num_cols = self.num_cols + self.digit_cols
                
        X_num_gen = pd.DataFrame(X_num_gen, columns=num_cols)
                
        # rounding numerical values
        for col_name, decimals in self.col_to_round_digits.items():
            X_num_gen[col_name] = np.round(X_num_gen[col_name], decimals)
            
        # combine and reorder columns, make sure types are correct
        if self.condition_on_binary:
            X_columns = np.concatenate((self.binary_cols, self.cat_cols, num_cols))
            X = pl.from_pandas(pd.DataFrame(np.column_stack((X_bin_gen, X_cat_gen, X_num_gen))))
        else:
            X_columns = np.concatenate((self.cat_cols, num_cols))
            X = pl.from_pandas(pd.DataFrame(np.column_stack((X_cat_gen, X_num_gen))))
            
        X.columns = X_columns
        
        # bring in correct order and apply original schema
        X = X.select(self.orig_cols)
        df_gen = pl.DataFrame(X, schema=self.orig_schema)

        return df_gen
    