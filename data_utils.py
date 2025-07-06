import math

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import pyarrow as pa
import torch
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder


def determine_column_types(data, MAX_UNIQUE_VALS=100):
    """
    Split columns into binary, categorical, and numeric features.
    If unique vals <= MAX_UNIQUE_VALS treat columns as categorical.
    idx_to_round_digits is a dictionary mapping column index to the number of digits to round the num_col feature values to.
    """
    
    unique_counts = data.select([pl.col(col).n_unique().alias(col) for col in data.columns])

    binary_cols = np.array(data.columns)[np.array((unique_counts[0] == 2).rows()[0])]

    # determine string columns and treat as categorical
    str_cols = np.array(
        data
        .select(~cs.by_name(binary_cols))
        .select(pl.col(pl.String))
        .columns
    )

    # select numeric_discrete columns and treat as categorical
    disc_cols = (
        unique_counts
        .select(~cs.by_name(binary_cols))
        .select(~cs.by_name(str_cols))
        .unpivot()
        .filter(pl.col("value") <= MAX_UNIQUE_VALS)
        .get_column("variable")
        .to_numpy()
    )

    # treat strings and discrete columns as categorical
    cat_cols = np.concatenate((str_cols, disc_cols))

    # determine numeric features that are later binned and possibly rounded
    num_cols = np.array(
        data
        .select(~cs.by_name(binary_cols))
        .select(~cs.by_name(cat_cols))
        .columns
    )

    col_to_round_digits = {}
    for i, d in enumerate(data.select(num_cols).iter_columns()):
        if d.dtype == pl.Int64:
            col_to_round_digits[d.name] = 0
        else:
            col_to_round_digits[d.name] = d.filter(~d.is_null()).to_pandas().map(lambda x: str(x).split('.')[1]).apply(len).max().item()


    assert len(cat_cols) + len(binary_cols) + len(num_cols) == data.width

    return binary_cols, cat_cols, num_cols, col_to_round_digits



class ContEncoder:
    """
    Encoder for continuous features, which encodes them into categories corresponding to quantiles, min/max values, inflated values, and missings. These categories are then endoded using an ordinal encoder and are the inputs to the low-resolution model.
    Also outputs mask which is = 1 for values that need to be inferred by a high-resolution model. Mask = 0 indicates inflated values, and missings that are determined by the low-resolution model.
    # TODO: treat min/max values different from quantiles such that mask = 0 becomes possible? or alternative just generate mask based on matching min/max values
    """
    
    def __init__(self, train_set, num_quantiles=10, encode_min_max=False):

        self.n_features = train_set.shape[1]
        self.encode_min_max = encode_min_max
        
        # encode min/max values as separate categories
        self.idx_to_min_max = self.determine_min_max(train_set)
        
        # determine inflated values to encode separately
        self.idx_to_inflated = self.determine_inflated(train_set)
        
        # check if columns have missings to encode separately
        self.has_missing = self.determine_missing(train_set)
        
        # determine bins for remaining values
        self.idx_to_quantiles = self.determine_quantiles(train_set, num_quantiles=num_quantiles)
        
        # fit ordinal encoder on training set
        self.ord_enc = OrdinalEncoder()
        self.ord_enc.fit(self.partial_encode(train_set))
        
        
    def determine_min_max(self, x: pl.DataFrame, min_max_threshold=20):
        
        idx_to_min_max = {}
        min_vals = x.min()
        max_vals = x.max()
        
        for i in range(x.shape[1]):
            idx_dict = {}
            n_min_val = x[:,i].filter(x[:,i] == min_vals[:,i]).shape[0]
            n_max_val = x[:,i].filter(x[:,i] == max_vals[:,i]).shape[0]
            idx_dict['min'] = min_vals[:,i].item()
            idx_dict['max'] = max_vals[:,i].item()
            idx_dict['encode_min'] = n_min_val > min_max_threshold
            idx_dict['encode_max'] = n_max_val > min_max_threshold
            idx_to_min_max[i] = idx_dict
            
        return idx_to_min_max
    
        
    def determine_inflated(self, x: pl.DataFrame, INFLATED_THRESHOLD=0.05):
        # maps feature index to inflated value
        # as long as threshold > 0.5, we have a single inflated value
        idx_to_inflated = {}
        for i in range(self.n_features):
            d = x[:,i].drop_nulls()
            
            if self.encode_min_max:
                d = d.filter(d != self.idx_to_min_max[i]['min']) if self.idx_to_min_max[i]['encode_min'] else d
                d = d.filter(d != self.idx_to_min_max[i]['max']) if self.idx_to_min_max[i]['encode_max'] else d

            vals, counts = np.unique(d, return_counts=True)
            props = counts / sum(counts) # normalize by remaining values (non-missing, non-min, non-max)
            inflated_vals = vals[props >= INFLATED_THRESHOLD]
            
            # define also as inflated value if highest proportion is at least twice the second highest proportion
            highest_two = np.sort(props)[-2:]
            if highest_two[1] > 2 * highest_two[0]:
                inflated_vals = np.concat((inflated_vals, vals[props == highest_two[1]]))
            if (len(inflated_vals) > 0):
                idx_to_inflated[i] = np.unique(inflated_vals)
            else:
                idx_to_inflated[i] = None
        return idx_to_inflated
    

    def determine_missing(self, x):
        has_missing = {i: (x.null_count()[col] > 0).item() for i, col in enumerate(x.columns)}
        return has_missing
        
        
    def determine_quantiles(self, x, num_quantiles=10):
        idx_to_quantiles = {}
        for i in range(x.shape[1]):
            d = x[:,i].drop_nulls()
                        
            # filter out min/max values if needed
            if self.encode_min_max:
                d = d.filter(d != self.idx_to_min_max[i]['min']) if self.idx_to_min_max[i]['encode_min'] else d
                d = d.filter(d != self.idx_to_min_max[i]['max']) if self.idx_to_min_max[i]['encode_max'] else d

            # filter out inflated values
            if self.idx_to_inflated[i] is not None:
                for inflated_val in self.idx_to_inflated[i]:
                    d = d.filter(d != inflated_val)

            qs = np.linspace(0, 1, num_quantiles+1)
                
            # determine quantiles for later grouping
            qs = np.nanquantile(d, q=qs, method='closest_observation')
            
            # keep only unique quantiles
            qs = np.unique(qs)
            
            # add min bin and min bin
            if self.encode_min_max:
                qs = np.insert(qs, 0, self.idx_to_min_max[i]['min']) if self.idx_to_min_max[i]['encode_min'] else qs
                qs = np.append(qs, self.idx_to_min_max[i]['max']) if self.idx_to_min_max[i]['encode_max'] else qs

            # add -Inf and Inf to the quantiles
            qs = qs.astype(np.float32)
            qs = np.insert(qs, 0, -np.inf)
            qs = np.append(qs, np.inf)

            idx_to_quantiles[i] = qs
            
        return idx_to_quantiles
    
    def partial_encode(self, x):
        return np.column_stack([self.encode_(x, i)[0] for i in range(x.shape[1])])
        
    def encode(self, x):
        x_enc = []
        masks = []
        for i in range(x.shape[1]):
            x_enc_i, mask_i = self.encode_(x, i)
            x_enc.append(x_enc_i)
            masks.append(mask_i)
        x_enc = self.ord_enc.transform(np.column_stack(x_enc))
        masks = np.column_stack(masks)
        return x_enc, masks
    
    def decode(self, x_enc):
        x_dec_aux = self.ord_enc.inverse_transform(x_enc)
        x_dec = []
        masks = []
        for i in range(x_enc.shape[1]):
            x_dec_i, mask_i = self.decode_(x_dec_aux, i)
            x_dec.append(x_dec_i)
            masks.append(mask_i)
        x_dec = np.column_stack(x_dec)
        masks = np.column_stack(masks)
        return x_dec, masks
    
    def get_miss_mask(self, x_enc):
        x_dec_aux = self.ord_enc.inverse_transform(x_enc)
        return (x_dec_aux == 'MISSING')

    def encode_(self, x, i):
        
        # order of encoding:
        # QUANTILES
        # MISSING = missing (if exists)
        # INFLATED = inflated val (if exists)
        
        d = x[:,i].to_numpy()
        
        # create mask, which values to infer using high resolution model
        mask = np.ones_like(d, dtype=bool)
        
        if np.isnan(d).any():
            assert self.has_missing[i], f"Column {i} has NaNs but is not marked as having missing values!"
        
        # clamp to min, max values seen in training set
        d = np.clip(d, self.idx_to_min_max[i]['min'], self.idx_to_min_max[i]['max'])
        
        right_quantiles = self.idx_to_quantiles[i][1:]
        x_enc = np.digitize(d, right_quantiles, right=True).astype(str)
        
        if self.has_missing[i]:
            x_enc[np.isnan(d)] = 'MISSING'
            mask[np.isnan(d)] = False
        
        if self.idx_to_inflated[i] is not None:
            for inflated_val in self.idx_to_inflated[i]:
                x_enc[d == inflated_val] = 'INFLATED_' + str(inflated_val)
                mask[d == inflated_val] = False

        return x_enc, mask

    def decode_(self, x_dec, i, return_quantiles=False):
        
        out = np.zeros_like(x_dec[:,i], dtype=np.float32)
        mask = np.ones_like(x_dec[:,i], dtype=bool)

        # decode missing category, mask = True if to be inferred by high resolution model
        if self.has_missing[i]:
            # TODO: also map unknown values to NAN from encoder? probably not needed, since we never decode OOD data
            out[x_dec[:,i] == 'MISSING'] = np.nan
            mask[x_dec[:,i] == 'MISSING'] = False
            
        # decode inflated category
        if self.idx_to_inflated[i] is not None:
            for inflated_val in self.idx_to_inflated[i]:
                out[x_dec[:,i] == 'INFLATED_' + str(inflated_val)] = inflated_val
                mask[x_dec[:,i] == 'INFLATED_' + str(inflated_val)] = False
            
        if return_quantiles:
            # decode quantiles (remaining categories)
            aux = np.ones_like(x_dec[:,i], dtype=np.int64)
            aux[mask] = x_dec[:,i][mask].astype(int)
            
            lower_bounds = self.idx_to_quantiles[i][:-1][aux * mask]
            upper_bounds = self.idx_to_quantiles[i][1:][aux * mask]
            
            # deal with min, max values
            lower_bounds[lower_bounds == -np.inf] = min(self.idx_to_quantiles[i][self.idx_to_quantiles[i] != -np.inf])
            upper_bounds[upper_bounds == np.inf] = max(self.idx_to_quantiles[i][self.idx_to_quantiles[i] != np.inf])
            
            return out, mask, lower_bounds, upper_bounds
        
        return out, mask
    
    
    def uniform_sample(self, x_enc):
        """
        Sample uniformly in quantile groups.
        """
        
        x_dec_aux = self.ord_enc.inverse_transform(x_enc)
        x_num_gen = []
        for i in range(x_enc.shape[1]):
            x_dec_i, mask_i, left_q_i, right_q_i = self.decode_(x_dec_aux, i, return_quantiles=True)
            
            # sample uniformly in quantile groups and scale by bin boundaries
            x_aux_gen = (right_q_i - left_q_i) * np.random.rand(x_enc.shape[0]) + left_q_i
            x_num_gen.append(x_aux_gen * mask_i + x_dec_i * ~mask_i)
        x_num_gen = np.column_stack(x_num_gen)
        
        return x_num_gen
    
    
    
    
    



class ContEncoder2:
    """
    Alternative.
    """
    
    def __init__(self, train_set, num_quantiles=10, encode_min_max=True):

        self.n_features = train_set.shape[1]
        self.encode_min_max = encode_min_max
        
        # encode min/max values as separate categories
        self.idx_to_min_max = self.determine_min_max(train_set)
        
        # determine inflated values to encode separately
        self.idx_to_inflated = self.determine_inflated(train_set)
        
        # check if columns have missings to encode separately
        self.has_missing = self.determine_missing(train_set)
        
        # determine bins for remaining values
        # self.idx_to_quantiles = self.determine_quantiles(train_set, num_quantiles=num_quantiles)
        
        self.bin_encs = [KBinsDiscretizer(n_bins=num_quantiles, encode='ordinal', strategy='quantile') for _ in range(self.n_features)]
    
        for i, bin_enc in enumerate(self.bin_encs):
            d = np.clip(train_set[:, i].drop_nulls().to_numpy(), self.idx_to_min_max[i]['min'], self.idx_to_min_max[i]['max'])
            
            # remove inflated values
            if self.idx_to_inflated[i] is not None:
                for val in self.idx_to_inflated[i]:
                    d = d[d != val]
            
            # remove min/max values
            d = d[d != self.idx_to_min_max[i]['min']] if self.idx_to_min_max[i]['encode_min'] else d
            d = d[d != self.idx_to_min_max[i]['max']] if self.idx_to_min_max[i]['encode_max'] else d
            
            bin_enc.fit(d.reshape(-1, 1)) 
            
            
        # fit ordinal encoder on training set
        self.ord_enc = OrdinalEncoder()
        self.ord_enc.fit(self.partial_encode(train_set))
        
        
    def determine_min_max(self, x: pl.DataFrame, min_max_threshold=20):
        
        idx_to_min_max = {}
        min_vals = x.min()
        max_vals = x.max()
        
        for i in range(x.shape[1]):
            idx_dict = {}
            n_min_val = x[:,i].filter(x[:,i] == min_vals[:,i]).shape[0]
            n_max_val = x[:,i].filter(x[:,i] == max_vals[:,i]).shape[0]
            idx_dict['min'] = min_vals[:,i].item()
            idx_dict['max'] = max_vals[:,i].item()
            idx_dict['encode_min'] = n_min_val > min_max_threshold
            idx_dict['encode_max'] = n_max_val > min_max_threshold
            idx_to_min_max[i] = idx_dict
            
        return idx_to_min_max
    
        
    def determine_inflated(self, x: pl.DataFrame, INFLATED_THRESHOLD=0.05):
        # maps feature index to inflated value
        # as long as threshold > 0.5, we have a single inflated value
        idx_to_inflated = {}
        for i in range(self.n_features):
            d = x[:,i].drop_nulls()
            
            if self.encode_min_max:
                d = d.filter(d != self.idx_to_min_max[i]['min']) if self.idx_to_min_max[i]['encode_min'] else d
                d = d.filter(d != self.idx_to_min_max[i]['max']) if self.idx_to_min_max[i]['encode_max'] else d

            vals, counts = np.unique(d, return_counts=True)
            props = counts / sum(counts) # normalize by remaining values (non-missing, non-min, non-max)
            inflated_vals = vals[props >= INFLATED_THRESHOLD]
            
            # define also as inflated value if highest proportion is at least twice the second highest proportion
            highest_two = np.sort(props)[-2:]
            if highest_two[1] > 2 * highest_two[0]:
                inflated_vals = np.concat((inflated_vals, vals[props == highest_two[1]]))
            if (len(inflated_vals) > 0):
                idx_to_inflated[i] = np.unique(inflated_vals)
            else:
                idx_to_inflated[i] = None
        return idx_to_inflated
    

    def determine_missing(self, x):
        has_missing = {i: (x.null_count()[col] > 0).item() for i, col in enumerate(x.columns)}
        return has_missing
        
        
    def determine_quantiles(self, x, num_quantiles=10):
        idx_to_quantiles = {}
        for i in range(x.shape[1]):
            d = x[:,i].drop_nulls()
                        
            # filter out min/max values if needed
            if self.encode_min_max:
                d = d.filter(d != self.idx_to_min_max[i]['min']) if self.idx_to_min_max[i]['encode_min'] else d
                d = d.filter(d != self.idx_to_min_max[i]['max']) if self.idx_to_min_max[i]['encode_max'] else d

            # filter out inflated values
            if self.idx_to_inflated[i] is not None:
                for inflated_val in self.idx_to_inflated[i]:
                    d = d.filter(d != inflated_val)

            qs = np.linspace(0, 1, num_quantiles+1)
                
            # determine quantiles for later grouping
            qs = np.nanquantile(d, q=qs, method='closest_observation')
            
            # keep only unique quantiles
            qs = np.unique(qs)
            
            # add min bin and min bin
            if self.encode_min_max:
                qs = np.insert(qs, 0, self.idx_to_min_max[i]['min']) if self.idx_to_min_max[i]['encode_min'] else qs
                qs = np.append(qs, self.idx_to_min_max[i]['max']) if self.idx_to_min_max[i]['encode_max'] else qs

            # add -Inf and Inf to the quantiles
            qs = qs.astype(np.float32)
            qs = np.insert(qs, 0, -np.inf)
            qs = np.append(qs, np.inf)

            idx_to_quantiles[i] = qs
            
        return idx_to_quantiles
    
    def partial_encode(self, x):
        return np.column_stack([self.encode_(x, i)[0] for i in range(x.shape[1])])
        
    def encode(self, x):
        x_enc = []
        masks = []
        for i in range(x.shape[1]):
            x_enc_i, mask_i = self.encode_(x, i)
            x_enc.append(x_enc_i)
            masks.append(mask_i)
        x_enc = self.ord_enc.transform(np.column_stack(x_enc)).astype(np.int32)
        masks = np.column_stack(masks)
        return x_enc, masks
    
    def decode(self, x_enc):
        x_dec_aux = self.ord_enc.inverse_transform(x_enc)
        x_dec = []
        masks = []
        for i in range(x_enc.shape[1]):
            x_dec_i, mask_i = self.decode_(x_dec_aux, i)
            x_dec.append(x_dec_i)
            masks.append(mask_i)
        x_dec = np.column_stack(x_dec)
        masks = np.column_stack(masks)
        return x_dec, masks
    
    def get_miss_mask(self, x_enc):
        x_dec_aux = self.ord_enc.inverse_transform(x_enc)
        return (x_dec_aux == 'MISSING')

    def encode_(self, x, i):
        
        # order of encoding:
        # QUANTILES
        # MISSING = missing (if exists)
        # INFLATED = inflated val (if exists)
        
        d = x[:,i].to_numpy()
        
        # create mask, which values to infer using high resolution model
        mask = np.ones_like(d, dtype=bool)
        
        if np.isnan(d).any():
            assert self.has_missing[i], f"Column {i} has NaNs but is not marked as having missing values!"
        
        # clamp to min, max values seen in training set
        d = np.clip(d, self.idx_to_min_max[i]['min'], self.idx_to_min_max[i]['max'])
        
        # right_quantiles = self.idx_to_quantiles[i][1:]
        # x_enc = np.digitize(d, right_quantiles, right=True).astype(str)
        x_enc = self.bin_encs[i].transform(np.nan_to_num(d, copy=True).reshape(-1, 1)).flatten().astype(int).astype(str)

        if self.has_missing[i]:
            x_enc[np.isnan(d)] = 'MISSING'
            mask[np.isnan(d)] = False
        
        if self.idx_to_inflated[i] is not None:
            for inflated_val in self.idx_to_inflated[i]:
                x_enc[d == inflated_val] = 'INFLATED_' + str(inflated_val)
                mask[d == inflated_val] = False
                
        if self.idx_to_min_max[i]['encode_min']:
            x_enc[d == self.idx_to_min_max[i]['min']] = 'MIN'
            mask[d == self.idx_to_min_max[i]['min']] = False
            
        if self.idx_to_min_max[i]['encode_max']:
            x_enc[d == self.idx_to_min_max[i]['max']] = 'MAX'
            mask[d == self.idx_to_min_max[i]['max']] = False

        return x_enc, mask
    

    def decode_(self, x_dec, i, return_quantiles=False):
        
        out = np.zeros_like(x_dec[:,i], dtype=np.float32)
        mask = np.ones_like(x_dec[:,i], dtype=bool)

        # decode missing category, mask = True if to be inferred by high resolution model
        if self.has_missing[i]:
            # TODO: also map unknown values to NAN from encoder? probably not needed, since we never decode OOD data
            out[x_dec[:,i] == 'MISSING'] = np.nan
            mask[x_dec[:,i] == 'MISSING'] = False
            
        # decode inflated category
        if self.idx_to_inflated[i] is not None:
            for inflated_val in self.idx_to_inflated[i]:
                out[x_dec[:,i] == 'INFLATED_' + str(inflated_val)] = inflated_val
                mask[x_dec[:,i] == 'INFLATED_' + str(inflated_val)] = False
                
                
        # decode min/max 
        if self.idx_to_min_max[i]['encode_min']:
            out[x_dec[:,i] == 'MIN'] = self.idx_to_min_max[i]['min']
            mask[x_dec[:,i] == 'MIN'] = False
            
        if self.idx_to_min_max[i]['encode_max']:
            out[x_dec[:,i] == 'MAX'] = self.idx_to_min_max[i]['max']
            mask[x_dec[:,i] == 'MAX'] = False
        
            
        # if return_quantiles:
        #     # decode quantiles (remaining categories)
        #     aux = np.ones_like(x_dec[:,i], dtype=np.int64)
        #     aux[mask] = x_dec[:,i][mask].astype(int)
            
        #     lower_bounds = self.idx_to_quantiles[i][:-1][aux * mask]
        #     upper_bounds = self.idx_to_quantiles[i][1:][aux * mask]
            
        #     # deal with min, max values
        #     lower_bounds[lower_bounds == -np.inf] = min(self.idx_to_quantiles[i][self.idx_to_quantiles[i] != -np.inf])
        #     upper_bounds[upper_bounds == np.inf] = max(self.idx_to_quantiles[i][self.idx_to_quantiles[i] != np.inf])
            
            # return out, mask, lower_bounds, upper_bounds
            
        x_dec[:,i][x_dec[:,i] == 'MISSING'] = 0
        x_dec[:,i][np.strings.startswith(x_dec[:,i], 'INFLATED_')] = 0
        x_dec[:,i][x_dec[:,i] == 'MIN'] = 0
        x_dec[:,i][x_dec[:,i] == 'MAX'] = 0
        
        out = out * (1-mask) + mask * self.bin_encs[i].inverse_transform(x_dec[:,i].astype(int).reshape(-1, 1)).flatten()
        
        
        return out, mask
    
    
    def uniform_sample(self, x_enc):
        """
        Sample uniformly in quantile groups.
        """
        
        x_dec_aux = self.ord_enc.inverse_transform(x_enc)
        x_num_gen = []
        for i in range(x_enc.shape[1]):
            x_dec_i, mask_i, left_q_i, right_q_i = self.decode_(x_dec_aux, i, return_quantiles=True)
            
            # sample uniformly in quantile groups and scale by bin boundaries
            x_aux_gen = (right_q_i - left_q_i) * np.random.rand(x_enc.shape[0]) + left_q_i
            x_num_gen.append(x_aux_gen * mask_i + x_dec_i * ~mask_i)
        x_num_gen = np.column_stack(x_num_gen)
        
        return x_num_gen



NUMERIC_DIGIT_MAX_DECIMAL = 18
NUMERIC_DIGIT_MIN_DECIMAL = -8



def _type_safe_numeric_series(numeric_array: np.ndarray | list, pd_dtype: str) -> pd.Series:
    # make a safe conversion using numpy's astype as an intermediary
    # and then pandas type to match values to pd_dtype
    np_dtype = int if pd_dtype == "Int64" else float
    i_min = np.iinfo(int).min
    i_max = np.iinfo(int).max

    def _clip_int(vals):
        # clip the array values to keep within representable boundaries in signed int representation
        return [min(max(int(v), i_min), i_max) for v in vals]

    if isinstance(numeric_array, list):
        if np_dtype is int:
            numeric_array = _clip_int(numeric_array)
        numeric_array = np.array(numeric_array)

    elif np_dtype is int:
        try:
            numeric_array.astype(int, casting="safe")
        except TypeError:
            # if it cannot be casted safely (e.g. without integer overflow)
            numeric_array = np.array(_clip_int(numeric_array))

    return pd.Series(np.array([v for v in numeric_array]).astype(np_dtype), dtype=pd_dtype)


def split_sub_columns_digit(
    values: pd.Series,
    max_decimal=NUMERIC_DIGIT_MAX_DECIMAL,
    min_decimal=NUMERIC_DIGIT_MIN_DECIMAL,
) -> pd.DataFrame:

    columns = [f"E{i}" for i in np.arange(max_decimal, min_decimal - 1, -1)]
    if values.isna().all():
        # handle special case when all values are nan
        df = pd.DataFrame({c: [0] * len(values) for c in columns})
    else:
        # convert to float64 as `np.format_float_positional` doesn't support Float64
        values = values.astype("float64")
        # rely on `np.format_float_positional` to determine string representation of absolute values
        values_str = (
            values.abs()
            .apply(lambda x: np.format_float_positional(x, unique=True, pad_left=50, pad_right=20, precision=20))
            # convert to string[pyarrow] for faster processing
            .astype("string[pyarrow]")
            # replace nan with pd.NA for faster processing
            .replace("nan", pd.NA)
        )
        values_str = values_str.str.replace(" ", "0")
        values_str = values_str.str.replace(".", "", n=1, regex=False)
        values_str = values_str.str[(49 - max_decimal) : (49 - min_decimal + 1)]
        df = values_str.str.split("", n=max_decimal - min_decimal + 2, expand=True)
        df = df.drop(columns=[0, max_decimal - min_decimal + 2])
        df = df.fillna("0")
        df.columns = columns
    df.insert(0, "nan", values.isna())
    df.insert(1, "neg", (~values.isna()) & (values < 0))
    df = df.astype("int")
    return df

def is_boolean_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_boolean(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_bool_dtype(x)
    
    
def is_integer_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_integer(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_integer_dtype(x)


def is_float_dtype(x: pd.Series) -> bool:
    if isinstance(x.dtype, pd.ArrowDtype):
        return pa.types.is_floating(x.dtype.pyarrow_dtype)
    else:
        return pd.api.types.is_float_dtype(x)




def safe_convert_numeric(values: pd.Series, nullable_dtypes: bool = False) -> pd.Series:
    if is_boolean_dtype(values):
        # convert booleans to integer -> True=1, False=0
        values = values.astype("Int8")
    elif not is_integer_dtype(values) and not is_float_dtype(values):
        # convert other non-numerics to string, and extract valid numeric sub-string
        valid_num = r"(-?[0-9]*[\.]?[0-9]+(?:[eE][+\-]?\d+)?)"
        values = values.astype(str).str.extract(valid_num, expand=False)
    values = pd.to_numeric(values, errors="coerce")
    if nullable_dtypes:
        values = values.convert_dtypes()
    return values



def _encode_numeric_digit(values: pd.Series, stats: dict, _: pd.Series | None = None) -> pd.DataFrame:
    values = safe_convert_numeric(values)
    # try to convert to int, if possible
    dtype = "Int64" if stats["min_decimal"] == 0 else "Float64"
    if dtype == "Int64":
        values = values.round()
    try:
        values = values.astype(dtype)
    except TypeError:
        if dtype == "Int64":  # if couldn't safely convert to int, stick to float
            dtype = "Float64"
            values = values.astype(dtype)
    # reset index, as `values.mask` can throw errors for misaligned indices
    values.reset_index(drop=True, inplace=True)
    # replace extreme values with min/max
    if stats["min"] is not None:
        reduced_min = _type_safe_numeric_series([stats["min"]], dtype).iloc[0]
        values.loc[values < reduced_min] = reduced_min
    if stats["max"] is not None:
        reduced_max = _type_safe_numeric_series([stats["max"]], dtype).iloc[0]
        values.loc[values > reduced_max] = reduced_max
    # split to sub_columns
    df = split_sub_columns_digit(values, stats["max_decimal"], stats["min_decimal"])
    is_not_nan = df["nan"] == 0
    # normalize values to `[0, max_digit-min_digit]`
    for d in np.arange(stats["max_decimal"], stats["min_decimal"] - 1, -1):
        key = f"E{d}"
        # subtract minimum value
        df[key] = df[key].where(~is_not_nan, df[key] - stats["min_digits"][key])
        # ensure that any value is mapped onto valid value range
        df[key] = np.minimum(df[key], stats["max_digits"][key] - stats["min_digits"][key])
        df[key] = np.maximum(df[key], 0)

    # ensure that encoded digits are mapped onto valid value range
    for d in np.arange(stats["max_decimal"], stats["min_decimal"] - 1, -1):
        df[f"E{d}"] = np.minimum(df[f"E{d}"], stats["max_digits"][f"E{d}"])
    if not stats["has_nan"]:
        df.drop("nan", inplace=True, axis=1)
    if not stats["has_neg"] or len(np.unique(df["neg"])) == 1:
        df.drop("neg", inplace=True, axis=1)
    return df



def _decode_numeric_digit(df_encoded: pd.DataFrame, stats: dict) -> pd.Series:
    max_decimal = stats["max_decimal"]
    min_decimal = stats["min_decimal"]
    # sum up all digits positions
    values = [
        (df_encoded[f"E{d}"] + stats["min_digits"][f"E{d}"]).to_numpy("uint64") * 10 ** int(d)
        for d in np.arange(max_decimal, min_decimal - 1, -1)
    ]
    values = sum(values)
    # convert to float if necessary
    dtype = "Float64" if min_decimal < 0 else "Int64"
    values = _type_safe_numeric_series(values, dtype)
    if "nan" in df_encoded.columns:
        values[df_encoded["nan"] == 1] = pd.NA
    if "neg" in df_encoded.columns:
        values[df_encoded["neg"] == 1] = -1 * values[df_encoded["neg"] == 1]
    if stats['has_neg'] and 'neg' not in df_encoded.columns:
        values = -1 * values
    # replace extreme values with min/max
    if stats["min"] is not None and stats["max"] is not None:
        is_too_low = values.notna() & (values < stats["min"])
        is_too_high = values.notna() & (values > stats["max"])
        values.loc[is_too_low] = _type_safe_numeric_series(np.ones(sum(is_too_low)) * stats["min"], dtype).values
        values.loc[is_too_high] = _type_safe_numeric_series(np.ones(sum(is_too_high)) * stats["max"], dtype).values
    elif "nan" in df_encoded.columns:
        # set all values to NaN if no valid values were present
        values[df_encoded["nan"] == 0] = pd.NA
    # round to min_decimal precision
    values = np.round(values, -min_decimal)
    return values


def determine_digit_encoding(train_set, digit_prec_treshold=3):
    
    digit_cols = []
    for i in range(train_set.shape[1]):
        d = pd.Series(train_set[:, i].to_numpy())
        df_split = split_sub_columns_digit(d, NUMERIC_DIGIT_MAX_DECIMAL, NUMERIC_DIGIT_MIN_DECIMAL)
        is_not_nan = df_split["nan"] == 0

        # extract min/max digit for each position to determine valid value range for digit encoding
        if any(is_not_nan):
            max_digits = {k: int(df_split[k][is_not_nan].max()) for k in df_split if k.startswith("E")}
        else:
            max_digits = {k: 0 for k in df_split if k.startswith("E")}
    
        non_zero_prec = [k for k in max_digits.keys() if max_digits[k] > 0 and k.startswith("E")]

        if len(non_zero_prec) <= digit_prec_treshold:
            digit_cols.append(train_set[:,i].name)
            
    return digit_cols
    

class DigitEncoder:
    """
    Digit encoding.
    """
    
    def __init__(self, train_set):

        self.n_features = train_set.shape[1]
        self.n_classes = []

        # encode min/max values as separate categories
        self.idx_to_min_max = self.determine_min_max(train_set)

        self.feat_stats = {}
        for i in range(self.n_features):
            d = pd.Series(train_set[:, i].to_numpy())
            df_split = split_sub_columns_digit(d, NUMERIC_DIGIT_MAX_DECIMAL, NUMERIC_DIGIT_MIN_DECIMAL)

            is_not_nan = df_split["nan"] == 0
            has_nan = sum(df_split["nan"]) > 0
            has_neg = sum(df_split["neg"]) > 0

            # extract min/max digit for each position to determine valid value range for digit encoding
            if any(is_not_nan):
                min_digits = {k: int(df_split[k][is_not_nan].min()) for k in df_split if k.startswith("E")}
                max_digits = {k: int(df_split[k][is_not_nan].max()) for k in df_split if k.startswith("E")}
            else:
                min_digits = {k: 0 for k in df_split if k.startswith("E")}
                max_digits = {k: 0 for k in df_split if k.startswith("E")}
        
            non_zero_prec = [k for k in max_digits.keys() if max_digits[k] > 0 and k.startswith("E")]
            min_decimal = min([int(k[1:]) for k in non_zero_prec]) if len(non_zero_prec) > 0 else 0
        
            decimal_cap = [d[1:] for d in max_digits.keys()][0]
            decimal_cap = int(decimal_cap) if decimal_cap.isnumeric() else NUMERIC_DIGIT_MAX_DECIMAL
            
            stats = {'has_nan': has_nan,
                    'has_neg': has_neg,
                    'min_digits': min_digits,
                    'max_digits': max_digits,
                    'min': self.idx_to_min_max[i]['min'],
                    'max': self.idx_to_min_max[i]['max'],}
            
            reduced_min = self.idx_to_min_max[i]['min']
            reduced_max = self.idx_to_min_max[i]['max']
            max_abs = np.max(np.abs(np.array([reduced_min, reduced_max])))
            max_decimal = int(np.floor(np.log10(max_abs))) if max_abs >= 10 else 0
            max_decimal = min(max(min_decimal, max_decimal), decimal_cap)
            stats['min_decimal'] = min_decimal
            stats['max_decimal'] = max_decimal
            
            df_split = _encode_numeric_digit(d, stats)
            stats['n_classes'] = []
            for j in range(df_split.shape[1]):
                n_unique_vals = len(np.unique(df_split.iloc[:, j]))
                stats['n_classes'].append(n_unique_vals)
                self.n_classes.append(n_unique_vals)
            
            self.feat_stats[i] = stats
            
            
            
            
            
        
    def determine_min_max(self, x: pl.DataFrame, min_max_threshold=20):
        
        idx_to_min_max = {}
        min_vals = x.min()
        max_vals = x.max()
        
        for i in range(x.shape[1]):
            idx_dict = {}
            n_min_val = x[:,i].filter(x[:,i] == min_vals[:,i]).shape[0]
            n_max_val = x[:,i].filter(x[:,i] == max_vals[:,i]).shape[0]
            idx_dict['min'] = min_vals[:,i].item()
            idx_dict['max'] = max_vals[:,i].item()
            idx_dict['encode_min'] = n_min_val > min_max_threshold
            idx_dict['encode_max'] = n_max_val > min_max_threshold
            idx_to_min_max[i] = idx_dict
            
        return idx_to_min_max
    
    
    def encode(self, x):
        x_digit_enc = []

        for i in range(x.shape[1]):
            x_enc_i = self.encode_digits(x, i)
            self.feat_stats[i]['col_names'] = x_enc_i.columns.tolist()
            x_digit_enc.append(x_enc_i.to_numpy())
        
        return np.column_stack(x_digit_enc)
    
    
    def decode(self, x_enc):
        
        x_dec = []
        start_idx = 0
        for i in range(self.n_features):
            x_dec_aux = x_enc[:, start_idx:start_idx + self.feat_stats[i]['n_cats']]
            start_idx += self.feat_stats[i]['n_cats']
            df = pd.DataFrame(x_dec_aux, columns=self.feat_stats[i]['col_names'])
            x_dec_i = _decode_numeric_digit(df, self.feat_stats[i])
            x_dec.append(x_dec_i.to_numpy())
    
        return np.column_stack(x_dec)
    
    
    def encode_digits(self, x, i):
        d = pd.Series(x[:,i].to_numpy())
        stats = self.feat_stats[i]
        df_split = _encode_numeric_digit(d, stats)
        self.feat_stats[i]['n_cats'] = df_split.shape[1] # how many subcolumns per feature
        return df_split
    






class CatEncoder:
                
    def __init__(self, train_data, min_frequency=8):
        self.min_frequency = min_frequency
        self.num_features = train_data.shape[1]
        self.fit(train_data)
        
    def fit(self, train_data):
        
        self.idx_to_stats = {}
        for i in range(self.num_features):
            stats = {}
            d = train_data[:,i]
            stats['orig_dtype'] = d.dtype
            d = d.cast(pl.String)
            if d.has_nulls():
                stats['has_missing'] = True
                d = d.fill_null('MISSING')
            else:
                stats['has_missing'] = False
            vals, cnt = np.unique(d, return_counts=True)

            if (cnt < self.min_frequency).any():
                stats['has_rare'] = True
                stats['rare_cats'] = vals[cnt < self.min_frequency]
                stats['rare_counts'] = cnt[cnt < self.min_frequency]
                d = d.replace(stats['rare_cats'], 'UNKNOWN')
            else:
                stats['has_rare'] = False
            
            vals, cnt = np.unique(d, return_counts=True)
            stats['values'] = vals
            stats['count'] = cnt
            stats['n_classes'] = len(vals)
            self.idx_to_stats[i] = stats
            
    
    def transform(self, X):
        
        X_enc = np.zeros((X.shape[0], self.num_features), dtype=np.int32)
        for i in range(self.num_features):
            d = X[:,i].cast(pl.String)

            if d.has_nulls():
                assert self.idx_to_stats[i]['has_missing'], f"Column {i} has missing values but was not fitted with missing values."
                d = d.fill_null('MISSING')
                
            if self.idx_to_stats[i]['has_rare']:
                d = d.replace(self.idx_to_stats[i]['rare_cats'], 'UNKNOWN')
                
            vals = np.unique(d)
            new_vals = [v for v in vals if v not in self.idx_to_stats[i]['values']]
            # print(f"Col {i} New values not in categories: {new_vals}")
            
            if len(new_vals) > 0:
                d = d.replace(new_vals, 'UNKNOWN')
            
            d = d.cast(pl.Enum(self.idx_to_stats[i]['values']))
            X_enc[:, i] = d.to_physical().to_numpy()
                
        return X_enc
    
    
    def inverse_transform(self, X_enc):
        cols = []
        for i in range(self.num_features):
            
            d = pl.Series(X_enc[:,i])
            d = d.cast(pl.Enum(self.idx_to_stats[i]['values'])).to_pandas()
            
            if self.idx_to_stats[i]['has_rare']:
                # sample category from categories
                n_unknown = (d == 'UNKNOWN').sum()
                p_choices = self.idx_to_stats[i]['rare_counts'] /  self.idx_to_stats[i]['rare_counts'] .sum()
                col_choices = self.idx_to_stats[i]['rare_cats']
                # renormalize probabilities
                draws = np.random.choice(col_choices, size=n_unknown, 
                                            replace=True, p=p_choices)
                d = d.cat.add_categories(np.unique(draws))
                d[d == 'UNKNOWN'] = draws
                
            if self.idx_to_stats[i]['has_missing']:
                d[d == 'MISSING'] = np.nan
                
            d = pl.Series(d, dtype=self.idx_to_stats[i]['orig_dtype'])
            cols.append(d)
        return pl.DataFrame(cols).to_numpy()
    

class BinaryGenerator():
    def __init__(self, data, binary_cols):
        
        joint_counts = (
            data
            .select(binary_cols)
            .group_by(binary_cols)  
            .agg(pl.len().alias("count"))
        )
        joint_probs = joint_counts.with_columns(
            prob=pl.col("count") / data.height,
        )

        assert math.isclose(joint_probs.select(pl.col('prob')).sum().item(), 1)
        
        self.probs = joint_probs.select('prob').to_numpy().squeeze()
        self.data = joint_counts.select(binary_cols)
        self.binary_cols = binary_cols
            
    def sample(self, n_samples, seed):
        # sample from the joint distribution of binary columns
        np.random.seed(seed)
        idx = np.random.choice(len(self.probs), size=n_samples, 
                               replace=True, p=self.probs)
        return self.data[idx]
   

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Adapted from: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, X_bin, X_cat, X_num_enc, X_num, mask, batch_size=32, shuffle=False, drop_last=False):
        """
        Initialize a FastTensorDataLoader.

        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        
        self.dataset_len = X_cat.shape[0] if X_cat is not None else X_num.shape[0]
        assert all(t.shape[0] == self.dataset_len for t in (X_bin, X_cat, X_num) if t is not None)
        
        self.mask = mask
        self.X_bin = X_bin
        self.X_cat = X_cat
        self.X_num_enc = X_num_enc
        self.X_num = X_num
        
        self.batch_size = batch_size
        self.shuffle = shuffle

        if drop_last:
            self.dataset_len = (self.dataset_len // self.batch_size) * self.batch_size
            
        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = {}
            batch['mask'] = torch.index_select(self.mask, 0, indices) if self.mask is not None else None
            batch['X_bin'] = torch.index_select(self.X_bin, 0, indices) if self.X_bin is not None else None
            batch['X_cat'] = torch.index_select(self.X_cat, 0, indices) if self.X_cat is not None else None
            batch['X_num_enc'] = torch.index_select(self.X_num_enc, 0, indices) if self.X_num_enc is not None else None
            batch['X_num'] = torch.index_select(self.X_num, 0, indices) if self.X_num is not None else None
            
        else:
            batch = {}
            batch['mask'] = self.mask[self.i:self.i+self.batch_size] if self.mask is not None else None
            batch['X_bin'] = self.X_bin[self.i:self.i+self.batch_size] if self.X_bin is not None else None
            batch['X_cat'] = self.X_cat[self.i:self.i+self.batch_size] if self.X_cat is not None else None
            batch['X_num_enc'] = self.X_num_enc[self.i:self.i+self.batch_size] if self.X_num_enc is not None else None
            batch['X_num'] = self.X_num[self.i:self.i+self.batch_size] if self.X_num is not None else None
            
        self.i += self.batch_size
    
        batch = tuple(batch.values())
        return batch

    def __len__(self):
        return self.n_batches
     
    

class FastTensorDataLoaderWithConditioning:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Adapted from: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, X_bin, X_cat, X_num_enc, X_num, mask, add_cond, batch_size=32, shuffle=False, drop_last=False):
        """
        Initialize a FastTensorDataLoader.

        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        
        self.dataset_len = X_cat.shape[0] if X_cat is not None else X_num.shape[0]
        assert all(t.shape[0] == self.dataset_len for t in (X_bin, X_cat, X_num) if t is not None)
        
        self.mask = mask
        self.X_bin = X_bin
        self.X_cat = X_cat
        self.X_num_enc = X_num_enc
        self.X_num = X_num
        self.add_cond = add_cond
        
        self.batch_size = batch_size
        self.shuffle = shuffle

        if drop_last:
            self.dataset_len = (self.dataset_len // self.batch_size) * self.batch_size
            
        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = {}
            batch['mask'] = torch.index_select(self.mask, 0, indices) if self.mask is not None else None
            batch['X_bin'] = torch.index_select(self.X_bin, 0, indices) if self.X_bin is not None else None
            batch['X_cat'] = torch.index_select(self.X_cat, 0, indices) if self.X_cat is not None else None
            batch['X_num_enc'] = torch.index_select(self.X_num_enc, 0, indices) if self.X_num_enc is not None else None
            batch['X_num'] = torch.index_select(self.X_num, 0, indices) if self.X_num is not None else None
            batch['add_cond'] = torch.index_select(self.add_cond, 0, indices) if self.add_cond is not None else None
            
        else:
            batch = {}
            batch['mask'] = self.mask[self.i:self.i+self.batch_size] if self.mask is not None else None
            batch['X_bin'] = self.X_bin[self.i:self.i+self.batch_size] if self.X_bin is not None else None
            batch['X_cat'] = self.X_cat[self.i:self.i+self.batch_size] if self.X_cat is not None else None
            batch['X_num_enc'] = self.X_num_enc[self.i:self.i+self.batch_size] if self.X_num_enc is not None else None
            batch['X_num'] = self.X_num[self.i:self.i+self.batch_size] if self.X_num is not None else None
            batch['add_cond'] = self.add_cond[self.i:self.i+self.batch_size] if self.add_cond is not None else None
            
        self.i += self.batch_size
    
        batch = tuple(batch.values())
        return batch

    def __len__(self):
        return self.n_batches
    
    
def count_to_get_probs(data, feat_names):
    joint_counts = (
        data
        .select(feat_names)
        .group_by(feat_names)  
        .agg(pl.len().alias("count"))
    )
    joint_probs = joint_counts.with_columns(
        prob=pl.col("count") / data.height,
    )

    assert math.isclose(joint_probs.select(pl.col('prob')).sum().item(), 1)
    return joint_counts.select(feat_names), joint_probs.select('prob').to_numpy().squeeze()



class DataEditor():
    def __init__(self, select_labels):
        self.select_labels = select_labels
    
    def adjust_loaders(self, train_loader, val_loader, data_preprocessor):
        
        dp = data_preprocessor
        cat_n_classes = dp.X_cat_n_classes
        num_n_classes = dp.X_num_n_classes
        
        num_enc_names = [f"num_enc_{i}" for i in range(train_loader.X_num_enc.shape[1])]
        self.orig_col_names_cat = list(dp.cat_cols)
        self.orig_col_names_num_enc = num_enc_names
        orig_col_names = list(dp.cat_cols) + num_enc_names
        df_trn = pl.DataFrame(torch.column_stack((train_loader.X_cat,
                                                   train_loader.X_num_enc)).numpy(), 
                               schema=orig_col_names)
        
        df_val = pl.DataFrame(torch.column_stack((val_loader.X_cat, 
                                                  val_loader.X_num_enc)).numpy(),
                            schema=orig_col_names)
        
        add_cond_train = df_trn.select(self.select_labels).to_torch().long()
        add_cond_val = df_val.select(self.select_labels).to_torch().long()
        
        # remove features we condition on from training data
        # new_train_df = df_trn.select(pl.exclude(self.select_labels))
        
        keep_idx = [i for i, col in enumerate(dp.cat_cols) if col not in self.select_labels]
        self.cat_n_classes = [cat_n_classes[i] for i in keep_idx]
        self.new_X_cat_cols = [col for col in dp.cat_cols if col not in self.select_labels]
        new_X_cat_train = df_trn.select(pl.col(self.new_X_cat_cols)).to_torch().long()
        
        # new_val_df = df_val.select(pl.exclude(self.select_labels))
        new_X_cat_val = df_val.select(pl.col(self.new_X_cat_cols)).to_torch().long()
        
        keep_idx = [i for i, col in enumerate(num_enc_names) if col not in self.select_labels]
        self.num_n_classes = [num_n_classes[i] for i in keep_idx]
        self.new_X_num_enc_cols = [col for col in num_enc_names if col not in self.select_labels]
        new_X_num_enc_train = df_trn.select(pl.col(self.new_X_num_enc_cols)).to_torch().long()
        new_X_num_enc_val = df_val.select(pl.col(self.new_X_num_enc_cols)).to_torch().long()
        
        train_loader = FastTensorDataLoaderWithConditioning(train_loader.X_bin,
                                            new_X_cat_train,
                                            new_X_num_enc_train,
                                            None,
                                            None,
                                            add_cond_train,
                                            batch_size=train_loader.batch_size,
                                            shuffle=True,
                                            drop_last=True)

        val_loader = FastTensorDataLoaderWithConditioning(val_loader.X_bin,
                                        new_X_cat_val,
                                        new_X_num_enc_val,
                                        None,
                                        None,
                                        add_cond_val,
                                        batch_size=val_loader.batch_size)

        return train_loader, val_loader
    
    
    def adjust_output(self, X_cat_gen, X_num_enc_gen, cond_gen):
        
        X_cat_gen = pl.DataFrame(X_cat_gen.numpy(), schema=self.new_X_cat_cols)
        X_num_enc_gen = pl.DataFrame(X_num_enc_gen.numpy(), schema=self.new_X_num_enc_cols)
        cond_gen = pl.DataFrame(cond_gen.numpy(), schema=self.select_labels)
        
        num_enc_cols = [lab for lab in self.select_labels if lab.startswith('num_enc_')]
        cat_cols = [lab for lab in self.select_labels if lab not in num_enc_cols]
        
        X_cond_cat = cond_gen.select(pl.col(cat_cols))
        X_cond_num_enc = cond_gen.select(pl.col(num_enc_cols))
        
        X_cat_gen = pl.concat([X_cat_gen, X_cond_cat], how='horizontal')
        X_num_enc_gen = pl.concat([X_num_enc_gen, X_cond_num_enc], how='horizontal')
        
        X_cat_gen = X_cat_gen.select(self.orig_col_names_cat).to_torch().long()
        X_num_enc_gen = X_num_enc_gen.select(self.orig_col_names_num_enc).to_torch().long()

        return X_cat_gen, X_num_enc_gen


