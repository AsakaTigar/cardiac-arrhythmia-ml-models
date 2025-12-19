# ============== 🚑 PyTorch 2.6+ Compatibility Patch ==============
# Fix for pytorch_tabular compatibility with PyTorch 2.6+
# PyTorch 2.6+ changed weights_only default to True, breaking omegaconf loading
import torch
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **kwargs.setdefault('weights_only', False) or kwargs)
# =================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, roc_curve
)
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibrationDisplay, calibration_curve
import shap
import joblib
import warnings
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing TabNet and PyTorch Tabular
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except ImportError as e:
    HAS_TABNET = False
    print(f"TabNet not available: {e}")

try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import CategoryEmbeddingModelConfig, GatedAdditiveTreeEnsembleConfig, NodeConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
    HAS_PYTORCH_TABULAR = True
except ImportError:
    HAS_PYTORCH_TABULAR = False

# Config
warnings.filterwarnings('ignore')
Path("outputs").mkdir(exist_ok=True)
Path("outputs/shap").mkdir(exist_ok=True)
ENABLE_DCA = False
ENABLE_CALIBRATION = False

# Plot settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Reading Data
def read_xlsx(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_excel(p, na_values=['X', 'x'])
    # Clean columns
    df.columns = [str(c).strip().replace("\u3000", " ") for c in df.columns]
    return df

try:
    train_df_raw = read_xlsx('./旧的/train_小标签_lrz.xlsx')
    val_df_raw = read_xlsx('./旧的/val_小标签_lrz.xlsx')
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Common Preprocessing
DELIM_PATTERN = r"[^a-zA-Z0-9_\u4e00-\u9fa5]+"

def parse_multilabels(series):
    labels_list = []
    keep_mask = []
    for val in series:
        if isinstance(val, str):
            parts = [s.strip() for s in re.split(DELIM_PATTERN, val) if s.strip()]
            labels_list.append(parts if parts else [])
            keep_mask.append(len(parts) > 0)
        else:
            labels_list.append([])
            keep_mask.append(False)
    return labels_list, np.array(keep_mask, dtype=bool)

def get_base_features(df):
    # Mapping
    train_column_mapping = {
        '性别 （0=男，1=女）': '性别',
        '束支传导阻滞(1=yes, 2=no)': '束支阻滞',
        '心动过缓（1=有，0=无）': '心动过缓',
        '体重(kg)': '体重kg',
        '身高(cm)': '身高cm',
        '腰围(cm)': '腰围',
        '第一次诊断AF(y)': '第一次诊断出AF(y)',
        '心衰=1，no=0': '心衰',
        '高血压=1，no=0': '高血压',
        '糖尿病=1，no=0': '糖尿病',
        '中风史=2，无=0': '中风',
        '周围血管病vascular disease=1，no=0': '周围血管病',
        '冠心病CAD': '冠心病',
        'Cardiomyopathy(无=0，扩心病DCM=1, 肥心病HCM=2, others=3)': '心肌病',
        'COPD慢性阻塞性肺疾病(yes=2, no=0)': '慢性阻塞性肺疾病',
        '睡眠呼吸综合征': '阻塞性睡眠呼吸暂停综合征OSAs',
        'LA（mm）前后径': 'LA',
        'LVDd（mm）': 'LVDd',
        'EF（%）': 'EF',
        'WBC(109)': 'WBC(X109)',
        'UA(umol|L)': 'UA(umol/L)',
        '术前INR（手术前一天）': 'INR'
    }
    df = df.rename(columns=train_column_mapping)
    return df

train_df_renamed = get_base_features(train_df_raw.copy())
val_df_renamed = get_base_features(val_df_raw.copy())

# Label Processors
class LabelProcessor:
    def __init__(self, mode):
        self.mode = mode
        self.rng = np.random.default_rng(42)

    def process(self, labels_list):
        if self.mode == 'big':
            return self._process_big(labels_list)
        else:
            return self._process_small(labels_list)

    def _process_big(self, labels_list):
        # 1. Expand SVT
        expanded_list = []
        for labels in labels_list:
            new_labels = []
            for lab in labels:
                if lab == "SVT":
                    r = self.rng.random()
                    if r < 0.6: new_labels.append("AVNRT")
                    elif r < 0.9: new_labels.append("AVRT")
                    else: new_labels.append("AT")
                else:
                    new_labels.append(lab)
            expanded_list.append(new_labels)
        
        # 2. Map to Big
        small_to_big = {
            # PACs已移除 
            "AT": "Atrial Arrhythmias",
            "AF": "Atrial Arrhythmias",
            "AFL": "Atrial Arrhythmias",
            "PVCs": "Ventricular Arrhythmias",
            "VT": "Ventricular Arrhythmias",
            "VF": "Ventricular Arrhythmias",
            "AVNRT": "SVT",
            "AVRT": "SVT"
        }
        mapped_list = []
        for labels in expanded_list:
            mapped = set()
            for lab in labels:
                if lab in small_to_big:
                    mapped.add(small_to_big[lab])
                else:
                    mapped.add(lab)
            mapped_list.append(list(mapped))
        
        # 3. Filter Ignore (包含PACs - 样本量太少)
        labels_to_ignore = {'AVVRT', 'PA', 'PAC', 'PACs', 'VF', '加速性室性自主心律', '室扑', '房室结双径路', '持续性', '迷走神经亢进'}
        final_list = []
        for labels in mapped_list:
            final_list.append([l for l in labels if l not in labels_to_ignore])
        return final_list

    def _process_small(self, labels_list):
        # Merge AF/AFL -> AF/AFL
        # Filter Ignore (drop VT/VF/PACs per requirement)
        labels_to_ignore = {'VT', 'VF', 'PACs', 'PAC', 'AVVRT', 'PA',
                            '加速性室性自主心律', '室扑', '房室结双径路', '持续性', '迷走神经亢进',
                            'SVT_AVNRT', 'SVT_AVRT'}
        final_list = []
        for labels in labels_list:
            new_labels = []
            for l in labels:
                if l in ['AF', 'AFL']:
                    new_labels.append('AF/AFL')
                elif l not in labels_to_ignore:
                    new_labels.append(l)
            final_list.append(list(set(new_labels))) # unique mapping
        return final_list

    def process(self, labels_list):
        if self.mode == 'big':
            return self._process_big(labels_list)
        else:
            return self._process_small(labels_list)

# Prepare Data Function
def prepare_data(mode):
    # Parse Labels
    train_labels_list, train_keep = parse_multilabels(train_df_renamed['诊断'])
    val_labels_list, val_keep = parse_multilabels(val_df_renamed['诊断'])
    
    # Filter rows
    train_df = train_df_renamed.loc[train_keep].reset_index(drop=True)
    val_df = val_df_renamed.loc[val_keep].reset_index(drop=True)
    
    processor_mode = 'small' if mode.startswith('small') else mode
    processor = LabelProcessor(processor_mode)
    train_labels_proc = processor.process([train_labels_list[i] for i in range(len(train_labels_list)) if train_keep[i]])
    val_labels_proc = processor.process([val_labels_list[i] for i in range(len(val_labels_list)) if val_keep[i]])

    # --- Mode Specific Filtering (Row Dropping) ---
    # This needs to happen before MLB fit_transform to ensure correct classes
    
    exclude_labels_for_mode = []
    if mode == 'small_no_AT':
        exclude_labels_for_mode = ['AT']
    elif mode == 'small_refined':
        # User request: "Exclude AT and VT/VF".
        # VT/VF are now kept by _process_small, so we can filter them here.
        exclude_labels_for_mode = ['AT', 'VT', 'VF', 'PVCs'] # PVCs also often grouped with VT/VF
            
    if exclude_labels_for_mode:
        keep_indices_train = []
        for i, labs in enumerate(train_labels_proc):
            if not any(l in exclude_labels_for_mode for l in labs):
                keep_indices_train.append(i)
                
        keep_indices_val = []
        for i, labs in enumerate(val_labels_proc):
            if not any(l in exclude_labels_for_mode for l in labs):
                keep_indices_val.append(i)
                
        train_df = train_df.iloc[keep_indices_train].reset_index(drop=True)
        val_df = val_df.iloc[keep_indices_val].reset_index(drop=True)
        train_labels_proc = [train_labels_proc[i] for i in keep_indices_train]
        val_labels_proc = [val_labels_proc[i] for i in keep_indices_val]
        
    # After filtering rows, ensure no empty label lists remain
    train_nonempty = [len(l) > 0 for l in train_labels_proc]
    val_nonempty = [len(l) > 0 for l in val_labels_proc]

    train_df = train_df.loc[train_nonempty].reset_index(drop=True)
    val_df = val_df.loc[val_nonempty].reset_index(drop=True)
    train_labels_proc = [l for l, keep in zip(train_labels_proc, train_nonempty) if keep]
    val_labels_proc = [l for l, keep in zip(val_labels_proc, val_nonempty) if keep]

    # MLB
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(train_labels_proc)
    Y_val = mlb.transform(val_labels_proc)
    
    # Features
    target_column = '诊断'
    # Base common features
    common_features = sorted(list(set(train_df.columns) & set(val_df.columns)))
    if target_column in common_features: common_features.remove(target_column)
    
    # Exclude Leakage Features (Strict for SHAP and Model)
    # Merging logic from both scripts
    features_to_ignore = {
        'SVT', 'AF', 'PVCs', 'AFL', 'VT', 'VF', 'SVT_AVNRT', 'AVNRT', 'AVRT',
        'AF.1', '第一次诊断出AF(y)', '诊断', '心律失常', 'Conclusion', 'conclusion', 'ECG', 'ecg'
    }
    # Add patterns check
    final_features = []
    for f in common_features:
        if f in features_to_ignore: continue
        # Simple name check
        if any(x in f for x in ['诊断', '分型', '亚型']): continue
        # Check against LEAK_FEATURES_EXACT logic form small label script
        norm_f = f.strip()
        if norm_f in features_to_ignore: continue
        final_features.append(f)
        
    X_train_raw = train_df[final_features]
    X_val_raw = val_df[final_features]
    
    # Numeric + Impute
    for c in final_features:
        X_train_raw[c] = pd.to_numeric(X_train_raw[c], errors='coerce')
        X_val_raw[c] = pd.to_numeric(X_val_raw[c], errors='coerce')
        
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=final_features)
    X_val = pd.DataFrame(imputer.transform(X_val_raw), columns=final_features)
    
    return X_train, Y_train, X_val, Y_val, mlb, final_features

def plot_dca(y_true, y_prob, model_name, class_name, save_dir):
    # DCA Logic:
    # Net Benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
    # where pt is probability threshold.
    
    thresholds = np.linspace(0.01, 0.99, 99)
    net_benefits = []
    
    n = len(y_true)
    
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
        
    # Treat All
    # TP = All Positives, FP = All Negatives
    tp_all = np.sum(y_true == 1)
    fp_all = np.sum(y_true == 0)
    net_benefits_all = [(tp_all / n) - (fp_all / n) * (pt / (1 - pt)) for pt in thresholds]
    
    # Treat None
    net_benefits_none = np.zeros(len(thresholds))
    
    plt.figure()
    plt.plot(thresholds, net_benefits, color='red', label=model_name)
    plt.plot(thresholds, net_benefits_all, color='black', linestyle=':', label='Treat All')
    plt.plot(thresholds, net_benefits_none, color='black', linestyle='--', label='Treat None')
    
    plt.ylim(min(min(net_benefits), -0.05), max(max(net_benefits_all) + 0.05, 0.1))
    plt.xlim(0, 1)
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title(f'DCA: {class_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    safe_class = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(class_name))
    plt.savefig(save_dir / f"dca_{model_name}_{safe_class}.png")
    plt.close()

def plot_calibration_curve_func(y_true, y_prob, model_name, class_name, save_dir):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve: {class_name}')
    plt.legend()
    safe_class = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(class_name))
    plt.savefig(save_dir / f"calibration_{model_name}_{safe_class}.png")
    plt.close()


from sklearn.base import BaseEstimator, ClassifierMixin

# Wrapper for PyTorch Tabular to adapt to Sklearn OVR
class PyTorchTabularOVRWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='FT-Transformer', epochs=100, batch_size=256, 
                 layers=None, num_trees=None, learning_rate=None):
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = layers  # For FT-Transformer: e.g., "256-128-64"
        self.num_trees = num_trees  # For NODE: e.g., 1024
        self.learning_rate = learning_rate  # Custom learning rate
        self.model = None
        self.classes_ = [0, 1]
        self._temp_dir = Path("outputs/temp_tabular")
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = Path("outputs/checkpoints")
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, X, y):
        # Clean previous model
        if self.model is not None:
             self.model = None

        if HAS_PYTORCH_TABULAR:
            print(f"  [PyTorchTabularOVRWrapper] Training {self.model_type} (optimized config)...")
            print(f"  Config: {self.epochs} epochs, batch_size={self.batch_size}")
            
            # ============ DEBUG: Input Data Analysis ============
            print(f"  [DEBUG] Input X shape: {X.shape if hasattr(X, 'shape') else len(X)}")
            print(f"  [DEBUG] Input y shape: {y.shape if hasattr(y, 'shape') else len(y)}")
            if hasattr(y, '__iter__'):
                unique_vals = np.unique(y)
                print(f"  [DEBUG] Unique y values: {unique_vals}")
                y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
                print(f"  [DEBUG] y value counts: {y_series.value_counts().to_dict()}")
            # ====================================================
            
            # ============ 关键修复:添加验证集划分 ============
            from sklearn.model_selection import train_test_split
            
            # Prepare Data - Convert to DataFrame if needed
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
            if isinstance(y, np.ndarray):
                y = pd.Series(y, name='target')
            
            # 划分训练集和验证集 (80/20)
            # 关键!pytorch_tabular需要验证集才能使用early_stopping
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")
            
            # ============ DEBUG: Train/Val Distribution ============
            print(f"  [DEBUG] Train y distribution: {y_train.value_counts().to_dict()}")
            print(f"  [DEBUG] Val y distribution: {y_val.value_counts().to_dict()}")
            # =======================================================
            
            # 创建训练和验证DataFrame
            train_df = pd.concat([X_train.reset_index(drop=True), 
                                 y_train.reset_index(drop=True)], axis=1)
            val_df = pd.concat([X_val.reset_index(drop=True), 
                               y_val.reset_index(drop=True)], axis=1)
            # ================================================
            
            data_config = DataConfig(
                target=['target'],
                continuous_cols=list(X_train.columns),  # 使用X_train的列名
                categorical_cols=[]
            )
            
            # 优化后的Trainer配置
            trainer_config = TrainerConfig(
                max_epochs=self.epochs,  # 100
                batch_size=self.batch_size,  # 256
                accelerator='auto',
                devices=1,
                early_stopping='valid_loss',  # 现在有验证集了!
                early_stopping_patience=20,
                checkpoints='valid_loss',
                checkpoints_path=str(self._checkpoint_dir),
                load_best=True,
                progress_bar='none',
                trainer_kwargs={
                    'enable_model_summary': False,
                    'enable_progress_bar': False
                }
            )
            optimizer_config = OptimizerConfig()
            
            # 针对性优化模型配置
            if self.model_type == 'FT-Transformer':
                print(f"  [DEBUG] Configuring FT-Transformer for classification")
                model_config = CategoryEmbeddingModelConfig(
                    task="classification",  # \u6062\u590d: pytorch_tabular\u4e0d\u652f\u6301binary
                    learning_rate=1e-4,
                    layers="128-64",
                    activation="ReLU",
                    dropout=0.1,
                    use_batch_norm=True
                )
            elif self.model_type == 'NODE':
                print(f"  [DEBUG] Configuring NODE for classification")
                model_config = NodeConfig(
                    task="classification",  # \u6062\u590d: pytorch_tabular\u4e0d\u652f\u6301binary
                    num_layers=2,
                    num_trees=512,
                    depth=4,
                    learning_rate=5e-4,
                    choice_function="entmax15",
                    bin_function="entmoid15",
                    additional_tree_output_dim=2
                )
            else:
                model_config = CategoryEmbeddingModelConfig(task="classification")

            self.model = TabularModel(
                data_config=data_config,
                model_config=model_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config,
                verbose=False
            )
            
            try:
                # ============ 关键:传入验证集! ============
                self.model.fit(train=train_df, validation=val_df)
                # =========================================
                print(f"  ✅ {self.model_type} training completed successfully!")
            except Exception as e:
                print(f"  ⚠️ {self.model_type} training encountered error: {e}")
                # 继续,不要让单个模型失败影响整体
                
        return self

    def predict_proba(self, X):
        if not HAS_PYTORCH_TABULAR or self.model is None:
            print(f"  [WARNING] Model not available, returning zeros")
            return np.zeros((len(X), 2))
            
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
        
        # Predict
        res = self.model.predict(X)
        probs = np.zeros((len(X), 2))
        
        # ============ DEBUG: Prediction Output Analysis ============
        print(f"  [DEBUG] Prediction result columns: {res.columns.tolist()}")
        print(f"  [DEBUG] Prediction result shape: {res.shape}")
        print(f"  [DEBUG] First 3 rows of predictions:")
        print(res.head(3))
        # ===========================================================
        
        # Check output structure with enhanced handling
        if 'target_1_probability' in res.columns and 'target_0_probability' in res.columns:
            # ✅ CORRECT: Use the actual probability columns directly
            print(f"  [DEBUG] Using 'target_0_probability' and 'target_1_probability' columns")
            probs[:, 0] = res['target_0_probability'].values
            probs[:, 1] = res['target_1_probability'].values
        elif 'prediction' in res.columns:
            # Direct prediction column
            print(f"  [DEBUG] Using 'prediction' column")
            predictions = res['prediction'].values
            # For binary, prediction might be 0/1, convert to probabilities
            probs[:, 1] = predictions
            probs[:, 0] = 1 - probs[:, 1]
        elif '1_probability' in res.columns:
            print(f"  [DEBUG] Using '1_probability' column")
            probs[:, 1] = res['1_probability'].values
            probs[:, 0] = 1 - probs[:, 1]
        elif 'class_1_probability' in res.columns:
            print(f"  [DEBUG] Using 'class_1_probability' column")
            probs[:, 1] = res['class_1_probability'].values
            probs[:, 0] = res['class_0_probability'].values
        else:
            # Fallback: search for any probability column
            prob_cols = [c for c in res.columns if 'prob' in c.lower()]
            print(f"  [WARNING] Unknown output format!")
            print(f"  [WARNING] Available probability columns: {prob_cols}")
            if prob_cols:
                print(f"  [WARNING] Using fallback column: {prob_cols[0]}")
                probs[:, 1] = res[prob_cols[0]].values
                probs[:, 0] = 1 - probs[:, 1]
            else:
                print(f"  [ERROR] No probability columns found! Returning zeros.")
                
        print(f"  [DEBUG] Final probs sample (first 3): {probs[:3]}")
        return probs
        
    def predict(self, X):
         proba = self.predict_proba(X)
         return (proba[:, 1] >= 0.5).astype(int)

# Wrapper for TabNet to accept DataFrames
class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, verbose=0, seed=42):
        self.verbose = verbose
        self.seed = seed
        self.model = None
        self.classes_ = [0, 1]
    
    def fit(self, X, y):
        # Re-init model to ensure clean state on re-fit (cloning does this, but safely)
        if self.model is None:
             self.model = TabNetClassifier(device_name='cuda', verbose=self.verbose, seed=self.seed)

        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        self.model.fit(X, y)
        return self
        
    def predict_proba(self, X):
        if self.model is None:
             return np.zeros((len(X), 2))
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict_proba(X)
        
    def predict(self, X):
        if self.model is None:
             return np.zeros(len(X))
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict(X)

# Models
def get_model(name):
    def base_xgb():
        return xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='logloss',
            n_estimators=300, max_depth=6, learning_rate=0.05,
            n_jobs=-1, random_state=42,
            device='cuda' # Enable GPU for XGBoost
        )
    def base_lgbm():
        return lgb.LGBMClassifier(
            objective='binary', n_estimators=300, num_leaves=31,
            learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1
        )
    def base_rf():
        return RandomForestClassifier(
            n_estimators=300, max_depth=10, n_jobs=-1, random_state=42
        )
    def base_lr():
        return LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)

    if name == 'XGBoost':
        return base_xgb()
    elif name == 'LightGBM':
        return base_lgbm()
    elif name == 'RandomForest':
        return base_rf()
    elif name == 'LogisticRegression':
        return base_lr()
    elif name == 'MLP':
         # Using larger MLP for "Deep Learning" proxy if TabNet fails
        return MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)
    elif name == 'GaussianNB':
        return GaussianNB()
    elif name == 'SVM':
        return SVC(probability=True, kernel='rbf', random_state=42)
    elif name == 'KNN':
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif name == 'TabNet':
        if HAS_TABNET:
            return TabNetWrapper(verbose=0, seed=42)
        else:
            return None
    elif name == 'FT-Transformer':
        if HAS_PYTORCH_TABULAR:
            return PyTorchTabularOVRWrapper(model_type='FT-Transformer')
        return None
    elif name == 'NODE':
        if HAS_PYTORCH_TABULAR:
            return PyTorchTabularOVRWrapper(model_type='NODE')
        return None
    elif name == 'Voting':
        estimators = [
            ('xgb', base_xgb()),
            ('lgbm', base_lgbm()),
            ('rf', base_rf()),
            ('lr', base_lr())
        ]
        return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    elif name == 'Stacking':
        estimators = [
            ('xgb', base_xgb()),
            ('lgbm', base_lgbm()),
            ('rf', base_rf()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
        ]
        final_est = base_lr()
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_est,
            stack_method='predict_proba',
            n_jobs=-1,
            verbose=2
        )
    return None

# Main Execution
if __name__ == "__main__":
    results = []
    results_csv_path = "outputs/all_models_performance.csv"
    existing_pairs = set()
    
    if Path(results_csv_path).exists():
        try:
            existing_df = pd.read_csv(results_csv_path)
            results = existing_df.to_dict('records')
            for r in results:
                existing_pairs.add((r['Label_Mode'], r['Model']))
            print(f"Loaded {len(results)} existing results. Resuming...")
        except Exception as e:
            print(f"Could not load existing results: {e}. Starting fresh.")

    MODELS = [
        'XGBoost', 'LightGBM', 'RandomForest', 'LogisticRegression', 
        'MLP', 'GaussianNB', 'SVM', 'KNN', 'TabNet', 'FT-Transformer', 'NODE',
        'Voting', 'Stacking'
    ]
    MODES = ['big', 'small', 'small_no_AT', 'small_refined']
    
    best_model_for_shap = None
    X_val_shap = None
    
    print("Starting Comprehensive Analysis...")
    
    for mode in MODES:
        print(f"\nProcessing Mode: {mode.upper()} LABELS")
        
        # Check if all models done for this mode to save data loading time?
        # No, keep it simple.
        
        X_train, Y_train, X_val, Y_val, mlb, feature_names = prepare_data(mode)
        print(f"Features: {len(feature_names)}, Classes: {len(mlb.classes_)} ({mlb.classes_})")
        
        for model_name in MODELS:
            if (mode, model_name) in existing_pairs:
                print(f"  Skipping {model_name} (Already done)")
                continue
                
            print(f"  Training {model_name}...")
            try:
                base_clf = get_model(model_name)
                if base_clf is None:
                    print(f"  Model {model_name} not available (None returned).")
                    continue
                    
                clf = OneVsRestClassifier(base_clf, verbose=10)
                clf.fit(X_train, Y_train)
                
                # Predict
                Y_prob = clf.predict_proba(X_val)
                Y_pred = (Y_prob >= 0.5).astype(int)
                
                # Metrics
                micro_f1 = f1_score(Y_val, Y_pred, average='micro')
                macro_f1 = f1_score(Y_val, Y_pred, average='macro')
                try:
                    micro_auc = roc_auc_score(Y_val, Y_prob, average='micro')
                except:
                    micro_auc = None
                
                # Per Class Metrics
                if ENABLE_DCA:
                    save_dca_dir = Path(f"outputs/dca_curves/{mode}")
                    save_dca_dir.mkdir(parents=True, exist_ok=True)
                if ENABLE_CALIBRATION:
                    save_cal_dir = Path(f"outputs/calibration_curves/{mode}")
                    save_cal_dir.mkdir(parents=True, exist_ok=True)
        
                for i, class_name in enumerate(mlb.classes_):
                    y_true_i = Y_val[:, i]
                    y_prob_i = Y_prob[:, i]
                    y_pred_i = Y_pred[:, i]
                    
                    # AUC
                    try:
                        auc = roc_auc_score(y_true_i, y_prob_i) if len(np.unique(y_true_i)) > 1 else None
                    except:
                        auc = None
                    
                    f1 = f1_score(y_true_i, y_pred_i)
                    rec = recall_score(y_true_i, y_pred_i)
                    prec = precision_score(y_true_i, y_pred_i, zero_division=0)
                    
                    results.append({
                        'Label_Mode': mode,
                        'Model': model_name,
                        'Class': class_name,
                        'AUC': auc,
                        'F1': f1,
                        'Recall': rec,
                        'Precision': prec,
                        'Micro_F1': micro_f1,
                        'Macro_F1': macro_f1
                    })
        
                    # Generate Plots (DCA & Calibration) optionally
                    if len(np.unique(y_true_i)) > 1:
                        if ENABLE_DCA:
                            plot_dca(y_true_i, y_prob_i, model_name, class_name, save_dca_dir)
                        if ENABLE_CALIBRATION:
                            plot_calibration_curve_func(y_true_i, y_prob_i, model_name, class_name, save_cal_dir)
                
                # Save model for SHAP (XGBoost + Small Label)
                if mode == 'small' and model_name == 'XGBoost':
                    best_model_for_shap = clf
                    X_val_shap = X_val
                    
                # Intermediate Save
                pd.DataFrame(results).to_csv(results_csv_path, index=False)
                
            except Exception as e:
                print(f"  Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Final Save
    
    # Save Results
    results_df = pd.DataFrame(results)
    results_csv_path = "outputs/all_models_performance.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved Performance Table to {results_csv_path}")
    
    # Print Pivot Table for User
    print("\n=== Performance Summary (AUC per Class) ===")
    pivot_auc = results_df.pivot_table(index=['Label_Mode', 'Class'], columns='Model', values='AUC')
    print(pivot_auc.to_string())
    
    print("\n=== Performance Summary (F1 per Class) ===")
    pivot_f1 = results_df.pivot_table(index=['Label_Mode', 'Class'], columns='Model', values='F1')
    print(pivot_f1.to_string())
    
    # SHAP Analysis
    print("\nRunning SHAP Analysis on XGBoost (Small Labels)...")
    if best_model_for_shap and X_val_shap is not None:
        # Use TreeExplainer
        # Since it's OneVsRest, we have multiple estimators.
        # We aggregate their shap values? Or typically for OVR we explain the prediction of the class of interest.
        # The existing script aggregated them: sum(shap_values) / n_estimators. This essentially explains the "average positive contribution"?
        # Actually, for multiclass, expected SHAP output is list of matrices.
        # OVR Classifier has `estimators_`.
        
        # We will compute SHAP for EACH class and average? Or just follow the existing script logic?
        # Existing script logic:
        # shap_sum += sv
        # return shap_sum / len(estimators)
        # This implies calculating importance across ALL classes combined.
        
        shap_dir = Path("outputs/shap")
        X_sample = X_val_shap.sample(min(300, len(X_val_shap)), random_state=42)
        
        shap_sum = np.zeros(X_sample.shape, dtype=float)
        dmat = xgb.DMatrix(X_sample, feature_names=list(X_sample.columns))
        
        # Use XGBoost native SHAP contributions to avoid base_score parsing issues
        for est in best_model_for_shap.estimators_:
            booster = est.get_booster()
            contribs = booster.predict(dmat, pred_contribs=True)
            contribs = np.array(contribs)
            shap_sum += contribs[:, :-1]  # drop bias term
            
        avg_shap_values = shap_sum / len(best_model_for_shap.estimators_)
        
        # Summary Plot
        plt.figure()
        shap.summary_plot(avg_shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / "shap_summary_xgb_small_combined.png", dpi=300)
        plt.close()
        
        # Top 10 features
        mean_shap = np.mean(np.abs(avg_shap_values), axis=0)
        top_indices = np.argsort(mean_shap)[::-1][:10]
        top_features = X_sample.columns[top_indices]
        
        print("\nTop 10 SHAP Features:")
        print(top_features.tolist())
        
        # Dependence Plots for Top 5
        for f in top_features[:5]:
            plt.figure()
            shap.dependence_plot(f, avg_shap_values, X_sample, show=False)
            safe_name = re.sub(r'[\\/*?:"<>|]', "", str(f))
            plt.tight_layout()
            plt.savefig(shap_dir / f"shap_dependence_{safe_name}.png", dpi=300)
            plt.close()
            
        print(f"SHAP plots saved to {shap_dir}")
    
    else:
        print("Skipping SHAP (Model not ready).")
