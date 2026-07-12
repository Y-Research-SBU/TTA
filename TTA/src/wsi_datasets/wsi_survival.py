from __future__ import print_function, division
import os
from os.path import join as j_
import torch
import numpy as np
import pandas as pd
import pdb
import pickle
import sys

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import h5py
from .dataset_utils import apply_sampling
sys.path.append('../')
from utils.pandas_helper_funcs import df_sdir, series_diff

import warnings
warnings.filterwarnings('ignore')

def _series_intersection(s1, s2):
    r"""
    Return insersection of two sets

    Args:
        - s1 : set
        - s2 : set

    Returns:
        - pd.Series

    """
    return pd.Series(list(set(s1) & set(s2)))


class WSISurvivalDataset(Dataset):
    """WSI Survival Dataset."""

    def __init__(self,
                 df,
                 data_source,
                 target_transform=None,
                 sample_col='case_id',
                 slide_col='slide_id',
                 survival_time_col='os_survival_days',
                 censorship_col='os_censorship',
                 n_label_bins=4,
                 label_bins=None,
                 bag_size=0,
                 include_surv_t0=True,
                 lazy_init=False,
                 sample_list=None,
                 **kwargs):
        """
        Args:
        """
        # Optional feature-dimension filter (e.g., resnet50 -> 768)
        self.feat_dim_filter = kwargs.get('feat_dim_filter', None)

        self.init_df(df, data_source, target_transform, sample_col, slide_col,
                     survival_time_col, censorship_col, n_label_bins, 
                     label_bins, bag_size, include_surv_t0)

        self.validate_survival_dataset()

        if not lazy_init:
            self.construct_df(sample_list)

    def __len__(self):
        return len(self.idx2sample_df)

    def init_df(self, df, data_source, target_transform, sample_col, slide_col,
                survival_time_col, censorship_col, n_label_bins, 
                label_bins, bag_size, include_surv_t0):

        self.data_source = []
        for src in data_source:
            assert os.path.basename(src) in ['feats_h5', 'feats_pt']
            self.use_h5 = True if os.path.basename(src) == 'feats_h5' else False
            self.data_source.append(src)

        # Do not guess filter by directory name. Keep whatever was passed in via kwargs.

        self.data_df = df
        assert 'Unnamed: 0' not in df.columns
        self.sample_col = sample_col
        self.slide_col = slide_col
        self.target_col = survival_time_col
        self.survival_time_col = survival_time_col
        self.censorship_col = censorship_col
        self.include_surv_t0 = include_surv_t0

        is_nan_censorship = self.data_df[self.censorship_col].isna()
        if sum(is_nan_censorship) > 0:
            print('# of NaNs in Censorship col, dropping:', sum(is_nan_censorship))
            self.data_df = self.data_df[~is_nan_censorship]

        is_nan_survival = self.data_df[self.survival_time_col].isna()
        if sum(is_nan_survival) > 0:
            print('# of NaNs in Survival time col, dropping:', sum(is_nan_survival))
            self.data_df = self.data_df[~is_nan_survival]

        if (self.data_df[self.survival_time_col] < 0).sum() > 0 and (not self.include_surv_t0):
            self.data_df = self.data_df[self.data_df[self.survival_time_col] > 0]

        censorship_vals = self.data_df[self.censorship_col].value_counts().index
        if set(censorship_vals) != set([0,1]):
            print('Censorship values must be binary integers, found:', censorship_vals)
            sys.exit()

        self.target_transform = target_transform
        self.n_label_bins = n_label_bins
        self.label_bins = None
        self.bag_size = bag_size
        # Expected histology feature dimension (set on first load, used for mismatch logging)
        self._expected_feat_dim = None

    def set_feat_paths_in_df(self):
        """
        Sets the feature path (for each slide id) in self.data_df. At the same time, checks that all slides 
        specified in the split (or slides for the cases specified in the split) exist within data source.
        """
        self.feats_df = pd.concat([df_sdir(feats_dir, cols=['fpath', 'fname', self.slide_col]) for feats_dir in self.data_source]).drop(['fname'], axis=1).reset_index(drop=True)

        # Case-insensitive matching only: compare on lowercase views
        _split_lower = self.data_df[self.slide_col].astype(str).str.lower()
        _feats_lower = self.feats_df[self.slide_col].astype(str).str.lower()
        missing_feats_in_split = series_diff(_split_lower, _feats_lower)

        # If missing features exist, log and SKIP those slides instead of exiting
        if len(missing_feats_in_split) > 0:
            print(f"[warn] Missing Features in Split (skipping these slides):\n{missing_feats_in_split}")
            miss_mask = _split_lower.isin(set(missing_feats_in_split.astype(str)))
            missing_cases = self.data_df.loc[miss_mask, self.sample_col].astype(str).unique().tolist()
            print(f"[warn] Affected case_id(s): {missing_cases}")
            # Drop the rows with missing slides and keep indices in sync
            keep_mask = ~_split_lower.isin(set(missing_feats_in_split.astype(str)))
            self.data_df = self.data_df[keep_mask].reset_index(drop=True)
            if hasattr(self, 'idx2sample_df') and isinstance(self.idx2sample_df, pd.DataFrame) and ('sample_id' in self.idx2sample_df.columns):
                keep_samples = set(self.data_df[self.sample_col].astype(str).unique())
                self.idx2sample_df = self.idx2sample_df[self.idx2sample_df['sample_id'].astype(str).isin(keep_samples)].reset_index(drop=True)

        ### Assertion to make sure that all slide ids to feature paths have a one-to-one mapping (no duplicated features).
        # Build lowercase join keys to merge fpath back while keeping original slide_id
        self.data_df['_slide_lower'] = self.data_df[self.slide_col].astype(str).str.lower()
        self.feats_df['_slide_lower'] = self.feats_df[self.slide_col].astype(str).str.lower()
        self.data_df = self.data_df.merge(self.feats_df[['_slide_lower', 'fpath']], how='left', on='_slide_lower', validate='1:1')
        self.data_df = self.data_df.drop(columns=['_slide_lower'])
        assert self.feats_df['_slide_lower'].duplicated().sum() == 0

        # Drop slides that do not have any valid feature after prefilter/merge
        if 'fpath' in self.data_df.columns:
            missing_mask = self.data_df['fpath'].isna()
            if missing_mask.any():
                n_drop = int(missing_mask.sum())
                print(f"[feature_dim_prefilter] drop_slides_without_valid_feats={n_drop}")
                self.data_df = self.data_df[~missing_mask].reset_index(drop=True)

        self.data_df = self.data_df[list(self.data_df.columns[-1:]) + list(self.data_df.columns[:-1])]

    def validate_survival_dataset(self):
        """Validate that the survival dataset is valid."""
        # check that each case_id has only one survival value
        num_unique_surv_times = self.data_df.groupby(self.sample_col)[self.survival_time_col].unique().apply(len)
        assert (num_unique_surv_times == 1).all()

        # check that all survival values are numeric
        assert not pd.to_numeric(self.data_df[self.survival_time_col], errors='coerce').isna().any()

        # check that all survival values are positive
        assert (self.data_df[self.survival_time_col] >= 0).all()
        if not self.include_surv_t0:
            assert (self.data_df[self.survival_time_col] > 0).all()

        # check that all censorship values are binary integers
        assert self.data_df[self.censorship_col].isin([0, 1]).all()

    def construct_df(self, sample_list=None, label_bins=None):
        """
        Additional preprocessing for organizing survival dataset
        """
        if sample_list is None:
            self.idx2sample_df = pd.DataFrame({'sample_id': self.data_df[self.sample_col].astype(str).unique()})
        else:
            self.idx2sample_df = pd.DataFrame({'sample_id': sample_list})

        self.set_feat_paths_in_df()
        # If a strict feature-dimension is required, drop rows whose fpath does not meet it
        if 'fpath' in self.data_df.columns and getattr(self, 'feat_dim_filter', None) is not None:
            ok_mask = []
            for fp in self.data_df['fpath'].tolist():
                if self.use_h5:
                    with h5py.File(fp, 'r') as f:
                        feat_dim = int(f['features'].shape[-1])
                else:
                    arr = torch.load(fp)
                    if len(arr.shape) > 2:
                        arr = np.squeeze(arr, axis=0)
                    feat_dim = int(arr.shape[-1])
                ok = (feat_dim == int(self.feat_dim_filter))
                ok_mask.append(ok)
            ok_mask = pd.Series(ok_mask, index=self.data_df.index)
            dropped_rows = int((~ok_mask).sum())
            if dropped_rows > 0:
                print(f"[feature_dim_filter_rows] drop_rows={dropped_rows} keep_rows={int(ok_mask.sum())} required_dim={self.feat_dim_filter}")
            self.data_df = self.data_df[ok_mask].reset_index(drop=True)
            # sync sample index list to only those with valid fpaths
            valid_samples = set(self.data_df[self.sample_col].astype(str).unique())
            self.idx2sample_df = self.idx2sample_df[self.idx2sample_df['sample_id'].astype(str).isin(valid_samples)].reset_index(drop=True)
        # Sync idx2sample_df with slides that still have valid features (non-null fpath)
        if 'fpath' in self.data_df.columns:
            valid_samples = set(self.data_df[self.sample_col].astype(str).unique())
            self.idx2sample_df = self.idx2sample_df[self.idx2sample_df['sample_id'].astype(str).isin(valid_samples)].reset_index(drop=True)
        self.data_df.index = self.data_df[self.sample_col].astype(str)
        self.data_df.index.name = 'sample_id'
        self.X = None
        self.y = None
        
        if 'disc_label' in self.data_df.columns:
            self.data_df = self.data_df.drop('disc_label', axis=1)
        
        if self.n_label_bins > 0:
            disc_labels, label_bins = compute_discretization(df=self.data_df,
                                                             survival_time_col=self.survival_time_col,
                                                             censorship_col=self.censorship_col,
                                                             n_label_bins=self.n_label_bins,
                                                             label_bins=label_bins)
            self.data_df = self.data_df.join(disc_labels)
            self.label_bins = label_bins
            self.target_col = disc_labels.name
            assert self.data_df.index.nunique() == self.idx2sample_df.index.nunique()

        self.survival_time_labels = []
        self.censorship_labels = []
        self.disc_labels = []
        for idx in self.idx2sample_df.index:
            survival_time, censorship, disc_label = self.get_labels(idx)
            self.survival_time_labels.append(survival_time)
            self.censorship_labels.append(censorship)
            self.disc_labels.append(disc_label)

        self.survival_time_labels = torch.tensor(self.survival_time_labels)
        self.censorship_labels = torch.tensor(self.censorship_labels)
        self.disc_labels = torch.tensor(self.disc_labels)


    def get_sample_id(self, idx):
        return self.idx2sample_df.loc[idx]['sample_id']

    def get_feat_paths(self, idx):
        feat_paths = self.data_df.loc[self.get_sample_id(idx), 'fpath']
        if isinstance(feat_paths, str):
            feat_paths = [feat_paths]
        return feat_paths

    def get_labels(self, idx):
        labels = self.data_df.loc[self.get_sample_id(idx), [self.survival_time_col, self.censorship_col, self.target_col]]
        if isinstance(labels, pd.Series):
            labels = list(labels)
        elif isinstance(labels, pd.DataFrame):
            labels = list(labels.iloc[0])
        return labels

    def __getitem__from_emb__(self, idx):
        out = {'img': self.X[idx],
            'coords': [],
            'survival_time': torch.tensor([self.survival_time_labels[idx]]),
            'censorship': torch.tensor([self.censorship_labels[idx]]),
            'label': torch.tensor([self.disc_labels[idx]])}
        return out

    def __getitem__(self, idx):
        if self.X is not None:
            return self.__getitem__from_emb__(idx)
        
        survival_time, censorship, label = self.get_labels(idx)
        # Read features (and coordinates, Optional) from pt/h5 file
        all_features = []
        all_coords = []

        feat_paths = self.get_feat_paths(idx)
        for feat_path in feat_paths:
            if self.use_h5:
                with h5py.File(feat_path, 'r') as f:
                    features = f['features'][:]
                    # Optional coordinates for visualization if present in H5
                    if 'coords' in f:
                        coords_arr = f['coords'][:]
                        if coords_arr is not None and len(coords_arr) > 0:
                            # ensure 2-D
                            coords_arr = np.asarray(coords_arr)
                            all_coords.append(coords_arr)
            else:
                features = torch.load(feat_path)

            if len(features.shape) > 2:
                assert features.shape[0] == 1, f'{features.shape} is not compatible! It has to be (1, numOffeats, feat_dim) or (numOffeats, feat_dim)'
                features = np.squeeze(features, axis=0)

            # Dimension checking / optional strict filter
            feat_dim = int(features.shape[-1])
            if self.feat_dim_filter is not None:
                if feat_dim != int(self.feat_dim_filter):
                    print(f"[feature_dim_skip] path={feat_path} dim={feat_dim} required={self.feat_dim_filter}")
                    continue
            else:
                if self._expected_feat_dim is None:
                    self._expected_feat_dim = feat_dim
                elif feat_dim != self._expected_feat_dim:
                    slide_val = self.data_df.loc[self.get_sample_id(idx), self.slide_col]
                    if isinstance(slide_val, pd.Series) or isinstance(slide_val, pd.DataFrame):
                        slide_val = slide_val.iloc[0]
                    print(f"[feature_dim_mismatch] slide_id={slide_val} path={feat_path} shape={features.shape} expected_dim={self._expected_feat_dim}")

            all_features.append(features)
        # Concatenate coords from multiple files if any
        if len(all_coords) > 0:
            all_coords = np.concatenate(all_coords, axis=0)
        if len(all_features) == 0:
            slide_val = self.get_sample_id(idx)
            raise ValueError(f"No valid feature arrays for slide_id={slide_val} after filtering (required_dim={self.feat_dim_filter}). Paths={feat_paths}")
        all_features = torch.from_numpy(np.concatenate(all_features, axis=0))

        # apply sampling if needed, return attention mask if sampling is applied else None
        all_features, all_coords, attn_mask = apply_sampling(self.bag_size, all_features, all_coords)

        out = {'img': all_features,
            'survival_time': torch.Tensor([survival_time]),
            'censorship': torch.Tensor([censorship]),
            'label': torch.Tensor([label])}
        # Only attach coords if available (for downstream visualization); no impact on training
        if isinstance(all_coords, np.ndarray) and all_coords.size > 0:
            out['coords'] = all_coords

        if attn_mask is not None:
            out['attn_mask'] = attn_mask

        return out
    
    def get_label_bins(self):
        return self.label_bins


def compute_discretization(df, survival_time_col='os_survival_days', censorship_col='os_censorship', n_label_bins=4, label_bins=None):
    df = df[~df['case_id'].duplicated()] # make sure that we compute discretization on unique cases

    if label_bins is not None:
        assert len(label_bins) == n_label_bins + 1
        q_bins = label_bins
    else:
        uncensored_df = df[df[censorship_col] == 0]
        disc_labels, q_bins = pd.qcut(uncensored_df[survival_time_col], q=n_label_bins, retbins=True, labels=False)
        q_bins[-1] = 1e6  # set rightmost edge to be infinite
        q_bins[0] = -1e-6  # set leftmost edge to be 0

    disc_labels, q_bins = pd.cut(df[survival_time_col], bins=q_bins,
                                retbins=True, labels=False,
                                include_lowest=True)
    assert isinstance(disc_labels, pd.Series) and (disc_labels.index.name == df.index.name)
    disc_labels.name = 'disc_label'
    return disc_labels, q_bins

#
# Dataset for Omics
#

class WSIOmicsSurvivalDataset(WSISurvivalDataset):
    """
    WSI Survival Dataset, combined with omics data.
    """
    def __init__(self,
                 df_histo,
                 df_gene,
                 data_source,
                 target_transform=None,
                 sample_col='case_id',
                 slide_col='slide_id',
                 survival_time_col='os_survival_days',
                 censorship_col='os_censorship',
                 n_label_bins=4,
                 label_bins=None,
                 bag_size=0,
                 include_surv_t0=True,
                 prepare_emb=False,
                 omics_dir=None,
                 omics_modality='pathway',
                 type_of_path='hallmarks',
                 **kwargs):

        super().__init__(df_histo, data_source, target_transform, sample_col, slide_col,
                       survival_time_col, censorship_col, n_label_bins, label_bins, bag_size, include_surv_t0,
                       lazy_init=True, **kwargs)

        # Get the intersection of histo and gene df
        self.omics_data = df_gene[~df_gene['case_id'].duplicated()]

        # print("=== DEBUG INFO ===")
        # print("omics_data case_id type:", type(self.omics_data['case_id'].iloc[0]))
        # print("omics_data case_id example:", self.omics_data['case_id'].head().tolist())
        # print("data_df case_id type:", type(self.data_df['case_id'].iloc[0]))  
        # print("data_df case_id example:", self.data_df['case_id'].head().tolist())
        # print("================")

        sample_list = np.intersect1d(np.unique(self.omics_data['case_id'].values), np.unique(self.data_df['case_id'].values))
        sample_list = sorted(sample_list)

        # Get histo df of intersection
        self.construct_df(sample_list)
        # Get gene df of intersection
        self.omics_data = self.omics_data[self.omics_data['case_id'].isin(sample_list)].sort_values(by=['case_id'])
        self.omics_data = self.omics_data.set_index('case_id')

        self.omics_dir = omics_dir
        self.omics_modality = omics_modality

        if self.omics_modality == 'functional':
            self._setup_func_genes()
        elif self.omics_modality == 'pathway':
            assert type_of_path in ['hallmarks'], f"Pathway needs to be hallmark!"
            self._setup_pathways(type_of_path)
        else:
            raise NotImplementedError(f"No implemented for {self.omics_modality}!")

    def get_scaler(self):
        # Get scaler to normalize omics data
        # Only fit on numeric columns (exclude case_id, sample, etc.)
        numeric_cols = self.omics_data.select_dtypes(include=[np.number]).columns
        non_numeric_cols = self.omics_data.select_dtypes(exclude=[np.number]).columns
        
        print(f"Numeric columns for scaling: {len(numeric_cols)}")
        print(f"Non-numeric columns (excluded): {list(non_numeric_cols)}")
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in omics data for scaling!")
        
        scaler = StandardScaler().fit(self.omics_data[numeric_cols])
        
        # Store column info for later use
        self._numeric_cols = numeric_cols
        self._non_numeric_cols = non_numeric_cols

        return scaler

    def apply_scaler(self, scaler):
        # Apply scaler to normalize omics data
        # Only transform numeric columns, preserve non-numeric columns
        
        # Use stored column info from get_scaler(), or detect again if not available
        if hasattr(self, '_numeric_cols') and hasattr(self, '_non_numeric_cols'):
            numeric_cols = self._numeric_cols
            non_numeric_cols = self._non_numeric_cols
        else:
            numeric_cols = self.omics_data.select_dtypes(include=[np.number]).columns
            non_numeric_cols = self.omics_data.select_dtypes(exclude=[np.number]).columns
        
        # Transform only numeric columns
        scaled_data = scaler.transform(self.omics_data[numeric_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=self.omics_data.index)
        
        # Preserve non-numeric columns
        for col in non_numeric_cols:
            scaled_df[col] = self.omics_data[col].values
        
        # Reorder columns to match original order (approximately)
        if 'case_id' in scaled_df.columns:
            case_col = scaled_df['case_id']
            scaled_df = scaled_df.drop('case_id', axis=1)
            scaled_df.insert(0, 'case_id', case_col)
        
        self.omics_data = scaled_df
        
        # Set case_id as index if it's not already
        if 'case_id' in self.omics_data.columns:
            self.omics_data = self.omics_data.set_index('case_id')

    def _setup_func_genes(self, signature_path='./data_csvs/rna/metadata'):
        r"""
        Process the signatures for the 6 functional groups required to run MCAT baseline

        Args:
            - self

        Returns:
            - None

        """
        self.signatures = pd.read_csv(os.path.join(signature_path, 'signatures.csv'))
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.omics_data.columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]

    def _setup_pathways(self, type_of_path='hallmark', signature_path='./data_csvs/rna/metadata'):
        r"""
        Process the signatures for the 331 pathways required to run SurvPath baseline. Also provides functinoality to run SurvPath with
        MCAT functional families (use the commented out line of code to load signatures)

        Args:
            - self

        Returns:
            - None

        """
        # running with hallmarks, reactome, and combined signatures
        self.signatures = pd.read_csv(os.path.join(signature_path, "{}_signatures.csv".format(type_of_path)))

        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.omics_data.columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]

    def __getitem__(self, idx):
        case_id = self.get_sample_id(idx)
        
        # Get survival information
        survival_time, censorship, label = self.get_labels(idx)
        
        # For gene-only modality, create dummy WSI features
        if getattr(self, 'modality_type', None) == 'gene':
            # Create dummy WSI features for compatibility
            dummy_img = torch.zeros(1, 1024)  # same as input_dim: 1024
            
            out = {
                'img': dummy_img,
                'survival_time': torch.Tensor([survival_time]),
                'censorship': torch.Tensor([censorship]),
                'label': torch.Tensor([label])
            }
        else:
            # Load WSI features for multimodal cases
            out = super().__getitem__(idx)

        # Add omics data
        omics_list = []
        if self.omics_modality == 'functional':
            for oidx in range(6):
                omics_list.append(self.omics_data.loc[case_id, self.omic_names[oidx]])

        elif self.omics_modality == 'pathway':
            for i in range(len(self.omic_names)):
                omics_list.append(torch.tensor(self.omics_data.loc[case_id, self.omic_names[i]]))

        else:
            raise NotImplementedError(f"Not Implemented for {self.omics_modality}")

        out['omics'] = omics_list

        return out